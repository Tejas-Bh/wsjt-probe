import numpy as np

def calculate_caf(received_signal, reference_signal, sample_rate=12000.0, span_hz=50.0, bs_hz=0.10):
    # Note: this function was debugged with the help of AI.
    """
    Computes a decoupled Cross-Ambiguity Function (CAF).
    Rows = Doppler Frequency Shifts (Hz)
    Columns = Time Delays (Samples)
    """
    signal_length = len(reference_signal)
    t = np.arange(signal_length) / sample_rate
    
    # 1. Build a clean, precise Doppler frequency search axis matching your specs
    doppler_axis = np.arange(-span_hz, span_hz + bs_hz, bs_hz)
    
    # Rows: Doppler bins, Columns: Time delay options (full signal length correlation)
    caf_matrix = np.zeros((len(doppler_axis), signal_length))
    
    # Compute the frequency domain version of your template once
    REF_FRQ = np.fft.fft(reference_signal)
    
    print(f"Evaluating {len(doppler_axis)} Doppler slices...")
    
    # Loop over Doppler offsets to peel back the phase errors
    for idx, f_doppler in enumerate(doppler_axis):
        # 2. De-rotate / wipe off the Doppler phase from the received signal
        doppler_correction = np.exp(-1j * 2 * np.pi * f_doppler * t)
        corrected_signal = received_signal * doppler_correction
        
        # 3. Use FFT cross-correlation to calculate ALL time delays instantly
        CORR_FRQ = np.fft.fft(corrected_signal) * np.conj(REF_FRQ)
        time_domain_correlation = np.fft.ifft(CORR_FRQ)
        
        # Save the absolute magnitude profile for this frequency layer
        caf_matrix[idx, :] = np.abs(time_domain_correlation)
        
    return caf_matrix, doppler_axis


def extract_radar_peaks(caf_magnitude, doppler_axis, max_delay_samples=500):
    """
    Extracts the clean, decoupled peak metrics from the inverted grid.
    """
    # Restrict our search grid to your max_delay_samples window to avoid noise wraps
    search_grid = caf_magnitude[:, :max_delay_samples]
    
    # Find the row (doppler) and column (delay) coordinates of the peak absolute value
    doppler_idx, delay_samples = np.unravel_index(np.argmax(search_grid), search_grid.shape)
    
    detected_doppler = doppler_axis[doppler_idx]
    
    return delay_samples, detected_doppler