# TODO === TODO === TODO === TODO === TODO
# remove vestigial phase jitter measurement
# remove (or revise?) delay, radar range resolution is incredibly low to make an accurate guess
# find multipath *properly* and revise hard-coded singular paths

import re
import numpy as np
import scipy.signal as signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import ft8

sender = ft8.FT8Send() # weakmon util

from channel_estimator import estimate_channel

def isolate_signal_iq(iq_waveform, center_hz, sample_rate=12000, bandwidth=90.0):
    """Isolates FT8 channel band and returns true baseband (centered at 0 Hz)."""
    n = len(iq_waveform)
    t = np.arange(n) / sample_rate

    # Down-mix to baseband (0 Hz) and lowpass filter
    shifted_down = iq_waveform * np.exp(-1j * 2 * np.pi * center_hz * t)

    nyquist = sample_rate / 2.0
    cutoff = (bandwidth / 2.0) / nyquist
    b, a = signal.butter(4, cutoff, btype="low")
    
    # Return directly at baseband; do NOT remix back to high carrier frequency
    return signal.filtfilt(b, a, shifted_down)


def extract_tap_metrics(channel_samples, sample_rate):
    """
    Extracts Watterson channel parameters from de-modulated channel response c(t).
    """
    n = len(channel_samples)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    t = np.arange(n) / sample_rate

    # Filter out residual FT8 keying modulation (6.25 Hz symbols)
    # Watterson ionospheric dynamics occur at low sub-Hz to low-Hz frequencies
    b_lp, a_lp = signal.butter(2, 5.0 / (sample_rate / 2.0), btype="low")
    c_filtered = signal.filtfilt(b_lp, a_lp, channel_samples)

    # 1. Spectral Analysis for Doppler Shift & Spread
    window = signal.windows.hann(n)
    fft_vals = np.fft.fftshift(np.fft.fft(c_filtered * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))

    # Mask to baseband Doppler window (-15 Hz to +15 Hz)
    mask = (freqs >= -15.0) & (freqs <= 15.0)
    psd = np.abs(fft_vals[mask]) ** 2
    f_axis = freqs[mask]

    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Center of mass -> Mean Doppler shift
    doppler_shift = np.sum(psd * f_axis) / psd_sum

    # RMS Width -> Doppler Spread
    doppler_spread = np.sqrt(np.sum(psd * (f_axis - doppler_shift) ** 2) / psd_sum)

    # Normalized total path power
    path_power = psd_sum / (n ** 2)

    # 2. Phase Dynamics (Phase Rate & Bounded Jitter)
    unwrapped_phase = np.unwrap(np.angle(c_filtered))

    # Linear fit: phase = slope * t + intercept
    slope, intercept = np.polyfit(t, unwrapped_phase, 1)
    
    # Calculate phase residual and wrap to [-pi, pi] for accurate jitter
    phase_residuals = unwrapped_phase - (slope * t + intercept)
    wrapped_residuals = np.angle(np.exp(1j * phase_residuals))
    phase_jitter = np.std(wrapped_residuals)

    return (
        float(doppler_shift),
        float(doppler_spread),
        float(path_power),
        float(slope),
        float(phase_jitter),
    )


def analyze(waveform, msg, sample_rate=12000, iq_waveform=None):
    """
    Analyzes FT8 reception and parameterizes ionospheric propagation channel.
    """
    hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
    msg["hz"] = hz_val

    ft8_signal_span_hz = 6.25 * 7  # 43.75 Hz
    filter_center_hz = hz_val + ft8_signal_span_hz / 2.0

    # 1. Isolate target signal to baseband
    if iq_waveform is not None:
        if len(iq_waveform) != len(waveform):
            n = min(len(iq_waveform), len(waveform))
            iq_waveform = iq_waveform[:n]
        rx_iq = isolate_signal_iq(
            iq_waveform, center_hz=filter_center_hz, sample_rate=sample_rate, bandwidth=90.0
        )
    else:
        nyquist = sample_rate / 2.0
        low = (filter_center_hz - 45.0) / nyquist
        high = (filter_center_hz + 45.0) / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        clean_audio = signal.filtfilt(b, a, waveform)
        
        # Mix audio down to baseband
        t = np.arange(len(clean_audio)) / sample_rate
        rx_iq = signal.hilbert(clean_audio) * np.exp(-1j * 2 * np.pi * filter_center_hz * t)

    # 2. Synthesize baseband reference signal
    i3 = 3 if re.match(r"^([A-Z0-9]+) ([A-Z0-9]+) (5[2-9]9) ([A-Z]+)$", msg["message"]) else 1
    bits77 = sender.pack(msg["message"], i3)
    audio_ref = sender.tones(bits77, hz_val, sample_rate)
    
    # Downmix synthesized reference to baseband matching filter_center_hz
    t_ref = np.arange(len(audio_ref)) / sample_rate
    ref_iq = signal.hilbert(audio_ref) * np.exp(-1j * 2 * np.pi * filter_center_hz * t_ref)

    min_len = min(len(rx_iq), len(ref_iq))
    rx_iq = rx_iq[:min_len]
    ref_iq = ref_iq[:min_len]
    
    dt = msg.get("dt", 0.0) or 0.0
    
    est = estimate_channel(rx_iq, ref_iq, sample_rate, dt)
    return {"message": msg, "channel": est["channel"],
            "quality": est["quality"], "z": est["z"]}
    # yay
