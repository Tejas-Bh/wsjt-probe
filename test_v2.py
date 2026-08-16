import re
import numpy as np
import scipy.signal as signal
from pathlib import Path
import sys

# Align path imports with analyze.py
sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import ft8
from analyze import analyze

sender = ft8.FT8Send()


def generate_synthetic_ft8(
    msg_str,
    hz=1000.0,
    dt=0.3,
    sample_rate=12000,
    paths=None,
    snr_db=0.0
):
    """
    Generates a full FT8 audio/IQ signal using the weakmon ft8 encoder,
    applying channel propagation paths and noise.
    """
    if paths is None:
        paths = [{"delay_ms": 10.0, "doppler_hz": 0.2, "power_linear": 1.0}]

    # 1. Encode message using weakmon sender (matches analyze.py logic)
    i3 = 3 if re.match(r"^([A-Z0-9]+) ([A-Z0-9]+) (5[2-9]9) ([A-Z]+)$", msg_str) else 1
    bits77 = sender.pack(msg_str, i3)
    audio_ref = sender.tones(bits77, hz, sample_rate)
    
    # Convert real audio to complex analytical signal
    base_iq = signal.hilbert(audio_ref)
    num_samples = len(base_iq)
    t = np.arange(num_samples) / sample_rate

    # 2. Apply multipath channel parameters
    rx_iq = np.zeros(num_samples, dtype=complex)
    for path in paths:
        delay_samples = int((path["delay_ms"] / 1000.0) * sample_rate)
        doppler_hz = path["doppler_hz"]
        power = path["power_linear"]

        # Shift in time and apply Doppler frequency modulation
        shifted = np.pad(base_iq, (delay_samples, 0))[:num_samples]
        doppler_mod = np.exp(1j * 2 * np.pi * doppler_hz * t)
        rx_iq += np.sqrt(power) * shifted * doppler_mod

    # 3. Apply time alignment offset (dt)
    dt_samples = int(dt * sample_rate)
    if dt_samples > 0:
        rx_iq = np.pad(rx_iq, (dt_samples, 0))[:num_samples]
    elif dt_samples < 0:
        rx_iq = rx_iq[-dt_samples:]
        rx_iq = np.pad(rx_iq, (0, -dt_samples))

    # 4. Add complex AWGN matching target SNR
    sig_power = np.mean(np.abs(rx_iq) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = (np.random.normal(0, np.sqrt(noise_power / 2), num_samples) +
             1j * np.random.normal(0, np.sqrt(noise_power / 2), num_samples))
    
    return rx_iq + noise


def run_tests():
    sample_rate = 12000

    test_cases = [
        {
            "name": "Standard Single Path (Low Doppler)",
            "msg": {"message": "CQ K5EK FM06", "hz": 667.1875, "dt": -0.03},
            "snr_db": 0.0,
            "paths": [
                {"delay_ms": 12.0, "doppler_hz": 0.15, "power_linear": 1.0}
            ],
        },
        {
            "name": "Multipath Channel (Two Taps)",
            "msg": {"message": "PT2MAF V31DL R-16", "hz": 1110.9375, "dt": 0.35},
            "snr_db": -5.0,
            "paths": [
                {"delay_ms": 0.0, "doppler_hz": -0.10, "power_linear": 1.0},
                {"delay_ms": 8.5, "doppler_hz": 0.40, "power_linear": 0.35}  # ~ -4.5 dB tap
            ],
        },
        {
            "name": "Low SNR Conditions (-15 dB)",
            "msg": {"message": "WC9I N0VB EL98", "hz": 2050.0, "dt": 0.325},
            "snr_db": -15.0,
            "paths": [
                {"delay_ms": 15.0, "doppler_hz": -0.30, "power_linear": 1.0}
            ],
        },
    ]

    for test in test_cases:
        print(f"=== Running: {test['name']} ===")
        
        # Generate full FT8 signal using weakmon utilities
        iq_wave = generate_synthetic_ft8(
            msg_str=test["msg"]["message"],
            hz=test["msg"]["hz"],
            dt=test["msg"]["dt"],
            sample_rate=sample_rate,
            paths=test["paths"],
            snr_db=test["snr_db"]
        )
        audio_wave = np.real(iq_wave)

        # Execute analysis
        res = analyze(
            waveform=audio_wave,
            msg=test["msg"],
            sample_rate=sample_rate,
            iq_waveform=iq_wave
        )

        # Output results
        print(f"Message: {res['message']['message']}")
        print(f"Detected Paths: {res['num_paths']}")
        for idx, tap in enumerate(res["watterson_taps"]):
            print(f" Tap {idx + 1}:")
            print(f"  - Delay:          {tap['delay_ms']:.2f} ms")
            print(f"  - Doppler Shift:  {tap['doppler_shift_hz']:.3f} Hz")
            print(f"  - Doppler Spread: {tap['doppler_spread_hz']:.3f} Hz")
            print(f"  - Rel Power:      {tap['relative_power_db']:.2f} dB")
            print(f"  - Phase Jitter:   {tap['phase_jitter_rad']:.3f} rad")
        print()


if __name__ == "__main__":
    run_tests()
