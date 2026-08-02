from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))


import ft8
import weakutil
import scipy.signal as signal


from processor import calculate_caf, extract_radar_peaks
import numpy as np

sender = ft8.FT8Send()

def analyze(waveform, msg):

    """
    Part one: reconstructing the I/Q signal
    TODO: Implement delta t!!!!!!
    """
    waveform = signal.hilbert(waveform)

    bits77 = sender.pack(msg["message"])

    sample_rate = 12000
    audio = sender.tones(bits77, msg["hz"](), sample_rate)

    synthetic_ft8 = signal.hilbert(audio)

    """
    Part two: use the caf to calculate the basic values
    - Doppler shift (!)
    - Path delay (????)
    - Path shifts (!)
    """
    caf_result, doppler_result = calculate_caf(waveform, sythetic_ft8)

    detected_delay, detected_doppler, peak_row = extract_radar_peaks(caf_result, doppler_result, max_delay_samples=700)

    noise_floor = np.mean(peak_row)
    threshold = noise_floor + 4 * np.std(peak_row)

    peaks_found, properties = signal.find_peaks(
            peak_row,
            height=threshold,
            distance=10
            )
    
    
    print(f"Finished analyzing signal @ {msg['hz']()}")
    return {
            "message": msg,
            "path_delay_samples": detected_delay,
            "doppler_shift_hz": detected_doppler,
            "path_shifts": len(peaks_found),
            "path_shifts_freqs": [doppler_result[i] for i in peaks_found]
            }
