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