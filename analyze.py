# TODO === TODO === TODO === TODO === TODO
# remove vestigial phase jitter measurement
# remove (or revise?) delay, radar range resolution is incredibly low to make an accurate guess
# find multipath *properly* and revise hard-coded singular paths

# analyze.py
import numpy as np
import scipy.signal as signal
from channel_estimator import estimate_channel

def isolate_signal_iq(iq_waveform, center_hz, sample_rate=12000, bandwidth=90.0):
    """Isolates channel band and down-mixes to true baseband (0 Hz)."""
    n = len(iq_waveform)
    t = np.arange(n) / sample_rate
    shifted_down = iq_waveform * np.exp(-1j * 2 * np.pi * center_hz * t)

    nyquist = sample_rate / 2.0
    cutoff = (bandwidth / 2.0) / nyquist
    b, a = signal.butter(4, cutoff, btype="low")
    return signal.filtfilt(b, a, shifted_down)


def analyze(waveform, msg, frontend, sample_rate=12000, iq_waveform=None):
    """
    Generic channel processor. Works for any mode supported by a ModeFrontend.
    """
    # 1. Frontend synthesizes ideal reference and provides target center frequency
    ref_iq, filter_center_hz = frontend.synthesize_reference_iq(msg, sample_rate)

    # 2. Isolate received signal to baseband
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

        t = np.arange(len(clean_audio)) / sample_rate
        rx_iq = signal.hilbert(clean_audio) * np.exp(-1j * 2 * np.pi * filter_center_hz * t)

    # 3. Align lengths
    min_len = min(len(rx_iq), len(ref_iq))
    rx_iq = rx_iq[:min_len]
    ref_iq = ref_iq[:min_len]

    dt = msg.get("dt", 0.0) or 0.0

    # 4. Perform ML Channel Estimation using the frontend's symbol duration T
    est = estimate_channel(
        rx_iq, 
        ref_iq, 
        sample_rate, 
        dt, 
        symbol_s=frontend.symbol_duration
    )

    hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]

    clean_msg = {
        "message": str(msg.get("message", "")),
        "snr": float(msg["snr"]) if msg.get("snr") is not None else None,
        "hz": float(hz_val) if hz_val is not None else None,
        "dt": float(dt),
        "timestamp_utc": msg.get("timestamp_utc")
    }
    
    return {
        "message": clean_msg,
        "mode": frontend.mode_name,
        "channel": est["channel"],
        "quality": est["quality"],
        "z": est["z"]
    }
