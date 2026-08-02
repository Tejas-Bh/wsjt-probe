from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import ft8
import re
import weakutil
import scipy.signal as signal

from processor import calculate_caf, extract_radar_peaks
import numpy as np

sender = ft8.FT8Send()

def isolate_signal_time_domain(waveform, center_hz, sample_rate=12000, bandwidth=70.0):
    """
    Isolates a single FT8 signal in the frequency domain using STFT,
    clears all other signals, and returns a clean time-domain waveform via ISTFT.
    Perfect for preparing clean inputs for passive radar CAF calculations.
    """
    # 1. Match the FT8 symbol duration (~0.16 seconds)
    nperseg = int(sample_rate * 0.16)
    noverlap = int(nperseg * 0.75)  # High overlap preserves phase perfectly
    
    # 2. Go to frequency domain
    frequencies, times, Zxx = signal.stft(
        waveform, 
        fs=sample_rate, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    
    # 3. Create a strict mask for our target signal window
    half_bw = bandwidth / 2.0
    outside_mask = (frequencies < (center_hz - half_bw)) | (frequencies > (center_hz + half_bw))
    
    # 4. CRITICAL: Zero out everything else (destroys interference & noise floor)
    Zxx_isolated = Zxx.copy()
    Zxx_isolated[outside_mask, :] = 0.0
    
    # 5. Return to the time domain cleanly
    _, isolated_waveform = signal.istft(
        Zxx_isolated, 
        fs=sample_rate, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    
    # Trim or pad to match the original waveform size exactly if needed
    if len(isolated_waveform) > len(waveform):
        isolated_waveform = isolated_waveform[:len(waveform)]
    elif len(isolated_waveform) < len(waveform):
        isolated_waveform = np.pad(isolated_waveform, (0, len(waveform) - len(isolated_waveform)))
        
    return isolated_waveform




def analyze(waveform, msg, sample_rate=12000):
    """
    Part one: reconstructing the I/Q reference signal and aligning dt
    """

    hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
    msg["hz"] = hz_val

    clean_waveform = isolate_signal_time_domain(waveform, center_hz=hz_val, sample_rate=sample_rate)

    waveform_iq = signal.hilbert(clean_waveform)

    # Pack message into 77-bit FT8 payload
    if not re.match(r'^([A-Z0-9]+) ([A-Z0-9]+) (5[2-9]9) ([A-Z]+)$', msg["message"]):
        i3 = 1
    else:
        i3 = 3

    bits77 = sender.pack(msg["message"], i3)

    audio = sender.tones(bits77, hz_val, sample_rate)
    synthetic_iq = signal.hilbert(audio)

    dt = msg.get("dt", 0.0)
    if dt is not None and dt != 0:
        sample_offset = int(dt * sample_rate)
        if sample_offset > 0:
            # Signal arrived late: pad synthetic signal start with zeros
            synthetic_iq = np.pad(synthetic_iq, (sample_offset, 0), mode='constant')[:len(waveform_iq)]
        elif sample_offset < 0:
            # Signal arrived early: shift synthetic signal forward
            shift = abs(sample_offset)
            synthetic_iq = np.pad(synthetic_iq[shift:], (0, shift), mode='constant')[:len(waveform_iq)]

    # Ensure equal length for cross-ambiguity matrix calculation
    min_len = min(len(waveform_iq), len(synthetic_iq))
    waveform_iq = waveform_iq[:min_len]
    synthetic_iq = synthetic_iq[:min_len]

    """
    Part two: use the CAF to calculate radar metrics
    """
    caf_result, doppler_result = calculate_caf(waveform_iq, synthetic_iq)

    detected_delay, detected_doppler, peak_row = extract_radar_peaks(
        caf_result, doppler_result, max_delay_samples=700
    )

    noise_floor = np.mean(peak_row)
    threshold = noise_floor + 4 * np.std(peak_row)

    peaks_found, properties = signal.find_peaks(
        peak_row,
        height=threshold,
        distance=10
    )
    
    print(f"Finished analyzing signal @ {hz_val} Hz")
    
    return {
        "message": msg,
        "path_delay_samples": detected_delay,
        "doppler_shift_hz": detected_doppler,
        "path_shifts": len(peaks_found),
        "path_shifts_freqs": [doppler_result[i] for i in peaks_found]
    }
