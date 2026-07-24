import matplotlib.pyplot as plt
import numpy as np

def plot_ambiguity_surface(caf_matrix, doppler_axis, sample_rate, max_delay_samples, doppler_zoom_hz=10, true_delay_samples=None, true_doppler_hz=None):
    # Note: this function was debugged with the help of AI.
    """
    Plots the decoupled 2D Cross-Ambiguity Function surface as a heat map.
    Properly maps Doppler to the X-axis and Time Delay to the Y-axis.
    """
    # 1. Slice the target Doppler window from the Doppler axis
    doppler_mask = (doppler_axis >= -doppler_zoom_hz) & (doppler_axis <= doppler_zoom_hz)
    zoomed_doppler = doppler_axis[doppler_mask]
    
    # 2. Slice the matrix: Rows (Doppler mask), Columns (up to max_delay_samples)
    zoomed_caf = caf_matrix[doppler_mask, :max_delay_samples]
    
    # 3. Define the physical Y-axis coordinates (Time Delays in microseconds)
    time_delays_us = (np.arange(max_delay_samples) / sample_rate) * 1e6
    
    plt.figure(figsize=(10, 6))
    
    # CRITICAL FIX: Transpose zoomed_caf using '.T' 
    # This moves Time Delays to the rows (Y-axis) and Doppler to the columns (X-axis)
    heatmap = plt.imshow(
        zoomed_caf.T, 
        extent=[zoomed_doppler[0], zoomed_doppler[-1], time_delays_us[0], time_delays_us[-1]],
        cmap='viridis', 
        aspect='auto',
        origin='lower'  # Puts 0 Hz and 0 µs at the bottom-left corner
    )
    
    # Overlay the red verification 'X'
    if true_delay_samples is not None and true_doppler_hz is not None:
        true_delay_us = (true_delay_samples / sample_rate) * 1e6
        plt.scatter(true_doppler_hz, true_delay_us, color='red', marker='x', s=150, 
                    linewidths=2.5, zorder=5, label=f'True: ({true_doppler_hz} Hz, {true_delay_us:.1f} µs)')
        plt.legend(loc='upper right')
    
    plt.title("Cross-Ambiguity Reference (CQ KJ5OAE EM10)", fontsize=14, pad=15)
    plt.xlabel("Doppler Frequency Shift (Hz)", fontsize=11)
    plt.ylabel("Time Delay (microseconds / µs)", fontsize=11)
    
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Correlation Energy Magnitude", fontsize=11)
    
    plt.grid(color='white', linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()
