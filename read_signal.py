from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "watterson"))

from gen_ft8 import synthetic_ft8, sample_rate as fs
from watterson import WattersonPath, WattersonChannel
import numpy as np

# true_delay = 180
# true_doppler = 2.4

paths = [
    WattersonPath(delay_s=1e-3, doppler_spread_hz=0.0, doppler_shift_hz=0.0, gain=1 / np.sqrt(2)),
    WattersonPath(delay_s=0.015, doppler_spread_hz=0.0, doppler_shift_hz=0.0, gain=1 / np.sqrt(2)),
    # WattersonPath(delay_s=0.0, doppler_spread_hz=1.0, doppler_shift_hz=0.0, gain=1 / np.sqrt(2)),
    # WattersonPath(delay_s=1e-3, doppler_spread_hz=1.0, doppler_shift_hz=0.0, gain=1 / np.sqrt(2)),
]
channel = WattersonChannel(paths, fs=fs, seed=1234)
waveform = channel.apply(synthetic_ft8)

# in_power = np.mean(np.abs(synthetic_ft8) ** 2)
# out_power = np.mean(np.abs(iq_out) ** 2)
# print(f"input power:  {in_power:.4f}")
# print(f"output power: {out_power:.4f}")
# print(f"output envelope: min={np.abs(iq_out).min():.4f}, "
#         f"max={np.abs(iq_out).max():.4f}, "
#         f"mean={np.abs(iq_out).mean():.4f}")

if False:
    from gen_ft8 import synthetic_ft8, sample_rate

    # generates a *dummy* waveform with added ionospheric effects
    import numpy as np

    true_delay = 180 # in samples
    true_doppler = 2.4 # in Hz

    # Create the time vector matching the amount of samples on synthetic_ft8
    t = np.arange(0, len(synthetic_ft8)) / sample_rate # arange() is like range() but for np arrays

    # A. Apply time delay (shift array forward)
    waveform = np.zeros_like(synthetic_ft8)
    waveform[true_delay:] = synthetic_ft8[:-true_delay]

    # B. Apply doppler frequency shift
    doppler_shift_vector = np.exp(1j * 2 * np.pi * true_doppler * t) # this is where t is used, also exp() used to make the wave w a frequency via e base
    waveform = waveform * doppler_shift_vector

    # C. Add atmospheric static noise (additive complex white noise)
    # TODO after I ensure that the CAF works!!!!