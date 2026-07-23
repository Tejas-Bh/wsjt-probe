import numpy as np
import matplotlib.pyplot as plt

def plot_ambiguity_surface(caf_magnitude, sample_rate, max_delay_samples, doppler_zoom_hz=10, true_delay_samples=None, true_doppler_hz=None):
    """
    Plots the 2D Cross-Ambiguity Function surface as a heat map.
    
    doppler_zoom_hz: Limits the X-axis view to +/- this value (e.g., +/- 10 Hz) 
                     so you can see tiny shifts clearly.
    """
    signal_length = caf_magnitude.shape[1]
    
    # 1. Calculate the exact physical values for our axes
    # Y-Axis: Convert sample delays to time delays (microseconds)
    time_delays_us = (np.arange(max_delay_samples) / sample_rate) * 1e6
    
    # X-Axis: Get the exact Doppler frequency array matching our columns
    doppler_freqs = np.fft.fftfreq(signal_length, 1 / sample_rate)
    doppler_freqs = np.fft.fftshift(doppler_freqs)
    
    # 2. Find the index limits to zoom into our target Doppler window
    doppler_mask = (doppler_freqs >= -doppler_zoom_hz) & (doppler_freqs <= doppler_zoom_hz)
    zoomed_doppler = doppler_freqs[doppler_mask]
    zoomed_caf = caf_magnitude[:, doppler_mask]
    
    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    
    # imshow draws the 2D matrix. 'extent' maps the matrix rows/columns to real numbers.
    # aspect='auto' prevents the plot from stretching awkwardly.
    heatmap = plt.imshow(
        zoomed_caf, 
        extent=[zoomed_doppler[0], zoomed_doppler[-1], time_delays_us[-1], time_delays_us[0]],
        cmap='viridis', 
        aspect='auto'
    )
    
    # Flip the Y-axis so 0 delay (shortest path) is at the bottom, just like a standard graph
    plt.gca().invert_yaxis()

    if true_delay_samples is not None and true_doppler_hz is not None:
        # Convert the injected sample delay into microseconds to match the Y-axis scale
        true_delay_us = (true_delay_samples / sample_rate) * 1e6
        
        # Plot a bright red 'X' at the exact injection coordinates
        plt.scatter(true_doppler_hz, true_delay_us, color='red', marker='x', s=150, 
                    linewidths=2.5, zorder=5, label=f'({true_doppler_hz}Hz, {true_delay_us:.1f}µs)')
        plt.legend(loc='upper right')
    
    # 4. Add styling, labels, and color-bar scale
    plt.title("Cross-Ambiguity Reference (CQ KJ5OAE EM10)", fontsize=14, pad=15)
    plt.xlabel("Doppler Frequency Shift (Hz)", fontsize=11)
    plt.ylabel("Time Delay (microseconds / µs)", fontsize=11)
    
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Correlation Energy Magnitude", fontsize=11)
    
    plt.grid(color='white', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()