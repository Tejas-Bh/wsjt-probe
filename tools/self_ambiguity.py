import sys
from pathlib import Path

# Go up one directory level to the parent folder
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to the top of the search path
sys.path.insert(0, str(parent_dir))

from gen_ft8 import synthetic_ft8, sample_rate
# from read_signal import waveform
from processor import calculate_caf, extract_radar_peaks

# Run the CAF on the same waves to find points of ambiguity (pain points), as suggested by K8GU
caf_result, doppler_result = calculate_caf(synthetic_ft8, synthetic_ft8)

# Finds the peaks (samples with highest amplitude) for the CAF result for doppler shift and time delay
detected_delay, detected_doppler = extract_radar_peaks(caf_result, doppler_result, max_delay_samples=700)

print("\n================ TEST RESULTS ================")
print(f"Calculated Path Delay: {detected_delay} samples")
print(f"Calculated Doppler Shift: {detected_doppler:.2f} Hz")
print("==============================================")

# Visualize the ambiguity function
from visualize_ambiguity import plot_ambiguity_surface

# plot_ambiguity_surface(caf_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=10, true_delay_samples=true_delay, true_doppler_hz=true_doppler)
# plot_ambiguity_surface(caf_result, doppler_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=10, true_delay_samples=true_delay, true_doppler_hz=true_doppler)
plot_ambiguity_surface(caf_result, doppler_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=10)