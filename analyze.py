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

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

sender = ft8.FT8Send()

def isolate_signal_time_domain(waveform, center_hz, sample_rate=12000, bandwidth=70.0):
    """
    Isolates a single FT8 signal in the frequency domain using Butterworth bandpass filter,
    """
    
    nyquist = sample_rate / 2.0
    low = (center_hz - (bandwidth / 2.0)) / nyquist
    high = (center_hz + (bandwidth / 2.0)) / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    # Use filtfilt for minimal phase distortion during filtering
    isolated_waveform = signal.filtfilt(b, a, waveform)
    
    # Trim or pad to match the original waveform size exactly if needed
    if len(isolated_waveform) > len(waveform):
        isolated_waveform = isolated_waveform[:len(waveform)]
    elif len(isolated_waveform) < len(waveform):
        isolated_waveform = np.pad(isolated_waveform, (0, len(waveform) - len(isolated_waveform)))
        
    return isolated_waveform


def isolate_signal_iq(iq_waveform, center_hz, sample_rate=12000, bandwidth=90.0):
    """
    Same job as isolate_signal_time_domain(), but for genuine complex I/Q
    instead of real audio: isolate a single FT8 signal's band directly
    out of the true I/Q, without ever going through a real-valued
    intermediate. This is what lets analyze() skip the
    bandpass-real-audio -> Hilbert-reconstruct-analytic-signal round trip
    when true I/Q is available (see the iq_waveform branch in analyze()).

    Because scipy doesn't have a ready-made complex bandpass, this does
    the standard trick: mix the desired band down to 0 Hz, run a real
    lowpass (butter+filtfilt, same filter design as the real-audio path,
    just lowpass instead of bandpass), then mix back up. filtfilt handles
    complex input fine (verified directly: 67 dB rejection of a synthetic
    400 Hz-away decoy tone in testing). Net effect is the same shape of
    output as isolate_signal_time_domain()+hilbert() used to produce --
    a complex signal still "at" center_hz, not shifted to baseband -- so
    everything downstream (dt alignment, calculate_caf()) is unchanged.
    """
    n = len(iq_waveform)
    t = np.arange(n) / sample_rate

    shifted_down = iq_waveform * np.exp(-1j * 2 * np.pi * center_hz * t)

    nyquist = sample_rate / 2.0
    cutoff = (bandwidth / 2.0) / nyquist
    b, a = signal.butter(4, cutoff, btype='low')
    filtered_baseband = signal.filtfilt(b, a, shifted_down)

    return filtered_baseband * np.exp(1j * 2 * np.pi * center_hz * t)


def compute_phase_stability(rx_iq, ref_iq):
    """
    Extracts the instantaneous phase added by the channel and measures its stability.
    """
    # Time-domain deconvolution removes the FT8 phase changes
    channel_samples = rx_iq * np.conj(ref_iq)
    
    # Extract the unwrapped phase trajectory
    instantaneous_phase = np.unwrap(np.angle(channel_samples))
    
    # Measure phase stability (Standard Deviation of the phase fluctuations)
    # High values indicate severe fading, scattering, or rapid layer movement
    phase_variance_rad = np.std(instantaneous_phase)
    
    # Calculate phase velocity (mean slope), which maps directly to the rate of TEC change
    time_steps = np.arange(len(instantaneous_phase))
    slope, _ = np.polyfit(time_steps, instantaneous_phase, 1)
    
    return float(phase_variance_rad), float(slope)

def compute_doppler_spread(rx_iq, ref_iq, sample_rate):
    """
    Measures the frequency broadening (Doppler Spread) of the channel profile.
    """
    # Multiply received signal by the complex conjugate of the reference to de-modulate it
    # This leaves behind only the complex channel variations (fading, phase shifts)
    channel_phase_evolution = rx_iq * np.conj(ref_iq)
    
    # Compute the Power Spectral Density (PSD) of the isolated channel variations
    fft_vals = np.fft.fftshift(np.fft.fft(channel_phase_evolution))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(channel_phase_evolution), d=1/sample_rate))
    psd = np.abs(fft_vals) ** 2
    
    # Isolate the immediate area around 0 Hz center to avoid capturing out-of-band noise
    center_mask = (freqs > -5.0) & (freqs < 5.0)
    clean_psd = psd[center_mask]
    clean_freqs = freqs[center_mask]
    
    # Calculate the RMS frequency spread
    mean_freq = np.sum(clean_psd * clean_freqs) / np.sum(clean_psd)
    doppler_spread_hz = np.sqrt(np.sum(clean_psd * (clean_freqs - mean_freq)**2) / np.sum(clean_psd))
    
    return float(doppler_spread_hz)



def analyze(waveform, msg, sample_rate=12000, iq_waveform=None):
    """
    Part one: reconstructing the I/Q reference signal and aligning dt

    iq_waveform: optional genuine complex I/Q covering the same slot as
    `waveform`, same sample_rate, same length (e.g. from
    KiwiSDR.read_iq() in kiwisdr.py). When given, this is used directly
    for waveform_iq instead of bandpass-filtering the real `waveform` and
    reconstructing an approximate analytic signal via hilbert() -- the
    real thing instead of a Hilbert approximation of it. `waveform`
    itself is still required, since it's what the FT8 decoder decoded
    (msg's hz/dt/message came from it) and is used as the fallback when
    iq_waveform isn't available, e.g. plain sound-card sources.
    """

    hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
    msg["hz"] = hz_val

    # hz_val (as reported by weakmon's decoder, and as passed to
    # FT8Send.tones() below) is the frequency of *tone 0*, not the center
    # of the signal. FT8 is 8-FSK at 6.25 Hz spacing (weakutil.fsk()), so
    # the real signal occupies [hz_val, hz_val + 7*6.25] = a 43.75 Hz
    # span above hz_val, not centered on it. A +/-35 Hz filter centered on
    # hz_val therefore clips the top of that span (tone 7 sits at
    # hz_val+43.75, outside the passband entirely; tone 6 at hz_val+37.5
    # is right on the rolloff edge) -- a real, if secondary, source of
    # signal-dependent SNR loss and extra CAF phase distortion on top of
    # the costas_samples bug above. Centering on the middle of the tone
    # span and widening the passband a bit fixes it.
    ft8_tone_spacing_hz = 6.25
    ft8_num_tones = 8
    ft8_signal_span_hz = ft8_tone_spacing_hz * (ft8_num_tones - 1)  # 43.75 Hz
    filter_center_hz = hz_val + ft8_signal_span_hz / 2.0

    if iq_waveform is not None:
        # True I/Q path: no bandpass-then-hilbert round trip. iq2usb()
        # (the phasing-method SSB demod used upstream to produce
        # `waveform` for the decoder) doesn't shift frequencies, so
        # hz_val is on the same Hz scale in iq_waveform as it is in
        # `waveform` -- no rescaling needed here.
        if len(iq_waveform) != len(waveform):
            n = min(len(iq_waveform), len(waveform))
            iq_waveform = iq_waveform[:n]
        waveform_iq = isolate_signal_iq(
            iq_waveform, center_hz=filter_center_hz, sample_rate=sample_rate, bandwidth=90.0
        )
    else:
        clean_waveform = isolate_signal_time_domain(
            waveform, center_hz=filter_center_hz, sample_rate=sample_rate, bandwidth=90.0
        )
        waveform_iq = signal.hilbert(clean_waveform)

    # Pack message into 77-bit FT8 payload
    if not re.match(r'^([A-Z0-9]+) ([A-Z0-9]+) (5[2-9]9) ([A-Z]+)$', msg["message"]):
        i3 = 1
    else:
        i3 = 3

    bits77 = sender.pack(msg["message"], i3)

    audio = sender.tones(bits77, hz_val, sample_rate)
    synthetic_iq = signal.hilbert(audio)

    # Force equal length first to protect dimensions
    min_len = min(len(waveform_iq), len(synthetic_iq))
    waveform_iq = waveform_iq[:min_len]
    synthetic_iq = synthetic_iq[:min_len]

    dt = msg.get("dt", 0.0) or 0.0

    # Initialize shift to 0 in case dt == 0
    sample_offset = 0
    shift = 0

    if dt != 0.0:
        sample_offset = int(dt * sample_rate)
        
        # dt is positive: Shift waveform_iq EARLIER (chop front, pad back)
        if sample_offset > 0:
            shift = sample_offset
            waveform_iq = np.pad(waveform_iq[shift:], (0, shift), mode='constant')
            
        # dt is negative: Shift synthetic_iq EARLIER (chop front, pad back)
        elif sample_offset < 0:
            shift = abs(sample_offset)
            synthetic_iq = np.pad(synthetic_iq[shift:], (0, shift), mode='constant')

    if False:
        # --- RESTORED VISUALIZATION BLOCK (Variables safely defined) ---
        plt.figure(figsize=(12, 4))
        sample_axis = np.arange(min_len)
        
        # Isolate the real components of the aligned analytical signals
        waveform_baseband = np.real(waveform_iq)
        synthetic_baseband = np.real(synthetic_iq)
        
        plt.plot(sample_axis, np.abs(waveform_baseband), label="received (baseband magnitude)", linewidth=1.0)
        plt.plot(sample_axis, np.abs(synthetic_baseband), label="synthetic (baseband magnitude)", linewidth=1.0, alpha=0.8)
        plt.xlabel("Sample")
        plt.ylabel("Envelope Amplitude")
        plt.title(f"Baseband Envelopes (dt={dt:.6f}s, applied_offset={sample_offset} samples)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("caf_debug_real_components.png", dpi=150)
        plt.close()

        input("Press Enter to continue to CAF analysis...==================================================================================================")

    """
    Part two: use the CAF to calculate radar metrics
    """

    nominal_costas_samples = int((7 * 0.16) * sample_rate)
    costas_samples = nominal_costas_samples + sample_offset

    # Guard against a real bug: when dt is negative enough (beyond about
    # -1.12s at 12000 Hz), costas_samples goes negative. numpy/Python do
    # NOT raise on waveform_iq[:costas_samples] when costas_samples < 0 --
    # a negative stop index is silently reinterpreted as "count back from
    # the end", so calculate_caf() would get handed a window tens of
    # thousands of samples long and badly misaligned, instead of the
    # intended ~13k-sample Costas-only window. That would show up as
    # exactly the kind of wildly-wrong, seemingly-random Doppler you're
    # seeing -- and only "sometimes", because it only bites messages
    # whose decoder-reported dt happens to be very negative. weakmon's
    # own decoder (pass0_tminus/pass0_tplus in ft8.py) searches dt out to
    # roughly -2.1s, so this is a reachable case for real off-air decodes,
    # not just a theoretical one -- and it's very unlikely to ever trigger
    # on synthetic/Watterson test signals where you control dt yourself,
    # which would explain why this looked fine in testing.
    min_usable_samples = int(3 * 0.16 * sample_rate)  # ~3 Costas symbols, bare minimum
    if costas_samples < min_usable_samples:
        raise ValueError(
            "dt=%.3fs leaves only %d samples (%.3fs) for the CAF window "
            "(need >= %d) -- skipping rather than silently mis-slicing"
            % (dt, costas_samples, costas_samples / float(sample_rate), min_usable_samples)
        )

    phase_slope = compute_phase_stability(waveform_iq, synthetic_iq)

    # 1. Extract true Doppler from your phase slope calculation
    actual_slope = phase_slope[1]
    true_doppler = (actual_slope * sample_rate) / (2 * np.pi)
    
    # 2. De-drift the receiver array before running the CAF
    t_vector = np.arange(len(waveform_iq)) / sample_rate
    correction_vector = np.exp(-1j * 2 * np.pi * true_doppler * t_vector)
    corrected_waveform_iq = waveform_iq * correction_vector
    
    # 3. Feed the corrected waveform into your CAF windows
    caf_result, doppler_result = calculate_caf(
        corrected_waveform_iq[:costas_samples], 
        synthetic_iq[:costas_samples]
    )

    # caf_result, doppler_result = calculate_caf(waveform_iq[:costas_samples], synthetic_iq[:costas_samples])

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

    if False:
        from visualize_ambiguity import plot_ambiguity_surface
        plot_ambiguity_surface(caf_result, doppler_result, sample_rate, max_delay_samples=700, doppler_zoom_hz=25)

    # """
    # Part 3: analyzing phase data
    # """
    #
    # quotient = waveform_iq / synthetic_iq
    #
    # phase = np.angle(quotient) # phase in radians
    #
    # if True: # PLOT THE PHASE
    #     plt.plot(phase)
    #     plt.title("Phase Plot")
    #     plt.xlabel("Index")
    #     plt.ylabel("Phase (radians)")
    #     plt.show()



    
    return {
        "message": msg,
        "path_delay_samples": detected_delay,
        "caf_doppler_shift_hz": detected_doppler,
        "true_doppler_shift_hz": (actual_slope * sample_rate) / (2 * np.pi),
        "path_shifts": len(peaks_found),
        "path_shifts_freqs": [doppler_result[i] for i in peaks_found],
        "doppler_spread": compute_doppler_spread(waveform_iq, synthetic_iq, sample_rate),
        "phase_stability": phase_slope
    }
