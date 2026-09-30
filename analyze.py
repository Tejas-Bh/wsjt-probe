# note: AI was used in developing this file. specifically, for the extract_tap_metrics() function and comment additions
# however, the system architecture and signal processing flow were created by Tejas-Bh.
# if you have any questions, let me know!

import re
import numpy as np
import scipy.signal as signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import ft8

sender = ft8.FT8Send() # weakmon util


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

    # 3. Time alignment offset using dt
    dt = msg.get("dt", 0.0) or 0.0
    initial_shift = int(dt * sample_rate)
    if initial_shift > 0:
        rx_iq = np.pad(rx_iq[initial_shift:], (0, initial_shift))
    elif initial_shift < 0:
        ref_iq = np.pad(ref_iq[-initial_shift:], (0, -initial_shift))

    # 4. Doppler pilot estimation
    pilot_channel_est = rx_iq * np.conj(ref_iq)
    pilot_window = signal.windows.hann(len(pilot_channel_est))
    pilot_X = np.fft.fftshift(np.fft.fft(pilot_channel_est * pilot_window))
    pilot_freqs = np.fft.fftshift(np.fft.fftfreq(len(pilot_channel_est), d=1.0 / sample_rate))
    pilot_mask = (pilot_freqs >= -50.0) & (pilot_freqs <= 50.0)
    pilot_doppler_hz = float(pilot_freqs[pilot_mask][np.argmax(np.abs(pilot_X[pilot_mask]) ** 2)])

    # 5. Cross-correlation fine delay search
    costas_samples = min(int(7 * 0.16 * sample_rate), min_len)
    t_short = np.arange(costas_samples) / sample_rate
    rx_short = rx_iq[:costas_samples] * np.exp(-1j * 2 * np.pi * pilot_doppler_hz * t_short)
    ref_short = ref_iq[:costas_samples]

    corr = signal.correlate(rx_short, ref_short, mode="full")
    lags = signal.correlation_lags(len(rx_short), len(ref_short))

    # Search window relative to time alignment baseline (0 to +25ms)
    search_window_ms = 25.0
    search_window = int((search_window_ms / 1000.0) * sample_rate)
    center_idx = np.argmin(np.abs(lags - 0))
    
    # Restrict lag search to non-negative delays
    start_idx = center_idx
    end_idx = min(len(corr), center_idx + search_window)

    corr_slice = np.abs(corr[start_idx:end_idx])
    lag_slice = lags[start_idx:end_idx]

    max_val = np.max(corr_slice) if len(corr_slice) > 0 else 1.0
    peaks, _ = signal.find_peaks(
        corr_slice,
        height=max_val * 0.4,
        prominence=max_val * 0.2,
        distance=int(0.002 * sample_rate)
    )

    if len(peaks) == 0:
        peaks = [np.argmax(corr_slice)] if len(corr_slice) > 0 else [0]

    # Calculate initial metrics to identify peak power baseline
    raw_taps = []
    for p in peaks:
        tap_lag = lag_slice[p]
        tap_delay_ms = max(0.0, (tap_lag / sample_rate) * 1000.0)

        if tap_lag > 0:
            rx_tap = rx_iq[tap_lag:]
            ref_tap = ref_iq[: len(rx_tap)]
        else:
            rx_tap = rx_iq
            ref_tap = ref_iq

        channel_response = rx_tap * np.conj(ref_tap)
        d_shift, d_spread, power, p_rate, p_jitter = extract_tap_metrics(
            channel_response, sample_rate
        )

        raw_taps.append({
            "delay_ms": float(tap_delay_ms),
            "doppler_shift_hz": d_shift,
            "doppler_spread_hz": d_spread,
            "power": power,
            "phase_rate_rad_s": p_rate,
            "phase_jitter_rad": p_jitter,
        })

    max_power = max([t["power"] for t in raw_taps]) if raw_taps else 1e-6

    watterson_taps = []
    for tap in raw_taps:
        rel_db = 10 * np.log10(max(tap["power"] / max_power, 1e-6))
        # Ensure relative power cannot exceed 0 dB
        rel_db = min(0.0, float(rel_db))

        watterson_taps.append({
            "delay_ms": tap["delay_ms"],
            "doppler_shift_hz": tap["doppler_shift_hz"],
            "doppler_spread_hz": tap["doppler_spread_hz"],
            "relative_power_db": rel_db,
            "phase_rate_rad_s": tap["phase_rate_rad_s"],
            "phase_jitter_rad": tap["phase_jitter_rad"],
        })

    return {
        "message": msg,
        "num_paths": len(watterson_taps),
        "watterson_taps": watterson_taps,
    }
    # yay
