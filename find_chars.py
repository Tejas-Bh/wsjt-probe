import sys
from pathlib import Path

# Go up one directory level to the parent folder
sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from gen_ft8 import synthetic_ft8, sample_rate
from read_signal import waveform#, true_delay, true_doppler
from processor import calculate_caf, extract_radar_peaks
import numpy as np
from scipy.signal import find_peaks

# Run the CAF on the same waves to find points of ambiguity (pain points), as suggested by K8GU
caf_result, doppler_result = calculate_caf(waveform, synthetic_ft8)

# Finds the peaks (samples with highest amplitude) for the CAF result for doppler shift and time delay
detected_delay, detected_doppler, peak_row = extract_radar_peaks(caf_result, doppler_result, max_delay_samples=700)

noise_floor = np.mean(peak_row) 
threshold = noise_floor + 4 * np.std(peak_row)

# 2. Find only the true target peaks
peaks_found, properties = find_peaks(
    peak_row, 
    height=threshold,      # Ignores noise spikes below this power level
    distance=10             # Prevents splitting a single wide target into multiple peaks
)

print("\n================ TEST RESULTS ================")
print(f"Calculated Path Delay: {detected_delay} samples")
print(f"Calculated Doppler Shift: {detected_doppler:.2f} Hz")
print(f"Doppler shifts found: {len(peaks_found)}")
print(f"Doppler shifts frequencies: {[doppler_result[i] for i in peaks_found]}")
print("==============================================")

input("Press enter to continue...")

# Visualize the ambiguity function
from visualize_ambiguity import plot_ambiguity_surface

# plot_ambiguity_surface(caf_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=10, true_delay_samples=true_delay, true_doppler_hz=true_doppler)
# plot_ambiguity_surface(caf_result, doppler_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=10, true_delay_samples=true_delay, true_doppler_hz=true_doppler)
plot_ambiguity_surface(caf_result, doppler_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=10)