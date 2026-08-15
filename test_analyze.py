#!/usr/bin/env python3
"""
Verification script for analyze.py using synthetic FT8 signals.
Generates an FT8 frame, applies controlled channel impairments (delay, Doppler, fading),
and tests parameter extraction.
"""

import numpy as np
import scipy.signal as signal
from analyze import analyze, isolate_signal_iq
import sys
from pathlib import Path

# Ensure paths match workspace layout
sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import ft8


def generate_synthetic_ft8_signal(
    message="K1ABC W1AW FN42",
    freq_hz=1000.0,
    sample_rate=12000,
    delay_sec=0.25,
    doppler_shift_hz=1.5,
    doppler_spread_hz=0.5,
    snr_db=20.0,
):
    """
    Generates a synthetic complex I/Q FT8 signal with specified channel impairments.
    """
    sender = ft8.FT8Send()

    # 1. Generate clean reference audio
    bits77 = sender.pack(message, 1)
    clean_audio = sender.tones(bits77, freq_hz, sample_rate)

    # 2. Convert to Analytic Signal (Complex I/Q) via Hilbert Transform
    iq_clean = signal.hilbert(clean_audio)
    n_samples = len(iq_clean)
    t = np.arange(n_samples) / sample_rate

    # 3. Apply Time Delay (Pad/Shift)
    delay_samples = int(round(delay_sec * sample_rate))
    if delay_samples > 0:
        iq_delayed = np.pad(iq_clean, (delay_samples, 0))[:n_samples]
    else:
        iq_delayed = iq_clean

    # 4. Apply Doppler Shift (Carrier Phase Trajectory)
    doppler_carrier = np.exp(1j * 2 * np.pi * doppler_shift_hz * t)

    # 5. Apply Rayleigh Fading (Doppler Spread)
    if doppler_spread_hz > 0.05:
        raw_noise = (
            np.random.normal(size=n_samples) + 1j * np.random.normal(size=n_samples)
        ) / np.sqrt(2)
        cutoff = min(doppler_spread_hz / (sample_rate / 2.0), 0.99)
        b_filt, a_filt = signal.butter(2, cutoff, btype="low")
        fading_envelope = signal.filtfilt(b_filt, a_filt, raw_noise)
        fading_envelope /= np.std(fading_envelope) + 1e-12
    else:
        fading_envelope = 1.0

    iq_impaired = iq_delayed * doppler_carrier * fading_envelope

    # 6. Add Gaussian Noise for target SNR
    sig_power = np.mean(np.abs(iq_impaired) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = (
        np.random.normal(scale=np.sqrt(noise_power / 2), size=n_samples)
        + 1j * np.random.normal(scale=np.sqrt(noise_power / 2), size=n_samples)
    )

    rx_iq = iq_impaired + noise
    rx_audio = np.real(rx_iq)

    return rx_audio, rx_iq


def main():
    sample_rate = 12000
    test_message = "K1ABC W1AW FN42"
    target_hz = 1000.0
    known_delay_sec = 0.28
    known_doppler_shift = 2.25  # Hz
    known_doppler_spread = 0.8  # Hz

    print("==================================================")
    print(" Generating Synthetic Channel State")
    print("==================================================")
    print(f" Message:        '{test_message}'")
    print(f" Target Frequency:{target_hz} Hz")
    print(f" Injected Delay:  {known_delay_sec * 1000:.1f} ms")
    print(f" Injected Shift:  {known_doppler_shift:.2f} Hz")
    print(f" Injected Spread: {known_doppler_spread:.2f} Hz")
    print("--------------------------------------------------")

    # Generate synthetic signal
    waveform, iq_waveform = generate_synthetic_ft8_signal(
        message=test_message,
        freq_hz=target_hz,
        sample_rate=sample_rate,
        delay_sec=known_delay_sec,
        doppler_shift_hz=known_doppler_shift,
        doppler_spread_hz=known_doppler_spread,
        snr_db=15.0,
    )

    # Format decoder dictionary matching weakmon output format
    msg_dict = {
        "message": test_message,
        "snr": 15.0,
        "hz": target_hz,
        "dt": known_delay_sec,
        "timestamp_utc": 1700000000.0,
    }

    print("\nRunning analyze()...\n")
    results = analyze(
        waveform=waveform,
        msg=msg_dict,
        sample_rate=sample_rate,
        iq_waveform=iq_waveform,
    )

    print("==================================================")
    print(" Parameter Extraction Results")
    print("==================================================")
    print(f" Paths Detected: {results['num_paths']}")

    for i, tap in enumerate(results["watterson_taps"]):
        print(f"\n --- Tap #{i+1} ---")
        print(
            f" Delay (ms):       {tap['delay_ms']:.2f} ms  (Expected ~{known_delay_sec*1000:.1f} ms)"
        )
        print(
            f" Doppler Shift:    {tap['doppler_shift_hz']:.2f} Hz  (Expected ~{known_doppler_shift:.2f} Hz)"
        )
        print(
            f" Doppler Spread:   {tap['doppler_spread_hz']:.2f} Hz  (Expected ~{known_doppler_spread:.2f} Hz)"
        )
        print(f" Relative Power:   {tap['relative_power_db']:.2f} dB")
        print(f" Phase Rate:       {tap['phase_rate_rad_s']:.2f} rad/s")
        print(f" Phase Jitter:     {tap['phase_jitter_rad']:.2f} rad")

    print("\nVerification completed successfully!")


if __name__ == "__main__":
    main()
