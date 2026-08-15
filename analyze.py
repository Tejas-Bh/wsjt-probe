import re
import numpy as np
import scipy.signal as signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import ft8

sender = ft8.FT8Send()


def isolate_signal_iq(iq_waveform, center_hz, sample_rate=12000, bandwidth=90.0):
    """Isolates FT8 channel band directly out of complex I/Q."""
    n = len(iq_waveform)
    t = np.arange(n) / sample_rate

    # Mix down to baseband, lowpass filter, mix back up
    shifted_down = iq_waveform * np.exp(-1j * 2 * np.pi * center_hz * t)

    nyquist = sample_rate / 2.0
    cutoff = (bandwidth / 2.0) / nyquist
    b, a = signal.butter(4, cutoff, btype="low")
    filtered_baseband = signal.filtfilt(b, a, shifted_down)

    return filtered_baseband * np.exp(1j * 2 * np.pi * center_hz * t)


def extract_tap_metrics(channel_samples, sample_rate):
    """
    Extracts Watterson channel parameters (Doppler Shift, Doppler Spread,
    Phase Rate, Phase Jitter) from a de-modulated channel response c(t).
    """
    n = len(channel_samples)
    t = np.arange(n) / sample_rate

    # 1. Spectral Analysis for Doppler Shift & Spread
    window = signal.windows.hann(n)
    fft_vals = np.fft.fftshift(np.fft.fft(channel_samples * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))

    # Isolate spectral region around baseband (-15 Hz to +15 Hz)
    mask = (freqs >= -15.0) & (freqs <= 15.0)
    psd = np.abs(fft_vals[mask]) ** 2
    f_axis = freqs[mask]

    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Center of mass -> Mean Doppler shift
    doppler_shift = np.sum(psd * f_axis) / psd_sum

    # RMS Width -> Doppler Spread
    doppler_spread = np.sqrt(np.sum(psd * (f_axis - doppler_shift) ** 2) / psd_sum)

    # Total path power
    path_power = psd_sum

    # 2. Phase Dynamics (Phase Velocity & Jitter)
    unwrapped_phase = np.unwrap(np.angle(channel_samples))

    # Linear fit: phase = slope * t + intercept
    slope, _ = np.polyfit(t, unwrapped_phase, 1)
    phase_jitter = np.std(unwrapped_phase - (slope * t))

    return (
        float(doppler_shift),
        float(doppler_spread),
        float(path_power),
        float(slope),
        float(phase_jitter),
    )


def analyze(waveform, msg, sample_rate=12000, iq_waveform=None):
    """
    Analyzes an FT8 reception and parameterizes the ionospheric propagation channel
    into a compact Watterson Channel Model tap profile.
    """
    hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
    msg["hz"] = hz_val

    ft8_signal_span_hz = 6.25 * 7  # 43.75 Hz
    filter_center_hz = hz_val + ft8_signal_span_hz / 2.0

    # 1. Isolate target signal
    if iq_waveform is not None:
        if len(iq_waveform) != len(waveform):
            n = min(len(iq_waveform), len(waveform))
            iq_waveform = iq_waveform[:n]
        rx_iq = isolate_signal_iq(
            iq_waveform, center_hz=filter_center_hz, sample_rate=sample_rate, bandwidth=90.0
        )
    else:
        nyquist = sample_rate / 2.0
        low = (filter_center_hz - 45.0) / nyquist
        high = (filter_center_hz + 45.0) / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        clean_audio = signal.filtfilt(b, a, waveform)
        rx_iq = signal.hilbert(clean_audio)

    # 2. Synthesize clean reference signal
    i3 = 3 if re.match(r"^([A-Z0-9]+) ([A-Z0-9]+) (5[2-9]9) ([A-Z]+)$", msg["message"]) else 1
    bits77 = sender.pack(msg["message"], i3)
    audio_ref = sender.tones(bits77, hz_val, sample_rate)
    ref_iq = signal.hilbert(audio_ref)

    min_len = min(len(rx_iq), len(ref_iq))
    rx_iq = rx_iq[:min_len]
    ref_iq = ref_iq[:min_len]

    # 3. Coarse dt alignment -- THE MISSING STEP.
    #
    # dt was being computed (below) and used only to bias where the
    # correlation *search window* sat -- rx_iq itself was never actually
    # shifted. rx_iq (from isolate_signal_iq on the full captured
    # buffer) has its real signal starting wherever dt says it does --
    # thousands of samples in -- while ref_iq starts its own tone
    # sequence at index 0. The min_len slice above just chops the *tail*
    # off rx_iq; it doesn't fix that the two arrays' real content starts
    # at genuinely different absolute positions. That mismatch is what
    # tap_delay_ms was actually reporting (confirmed directly: a 300-
    # sample/25ms true extra delay came back as "594.58 ms", almost
    # exactly the size of dt itself).
    #
    # This shift-and-pad is the same pattern validated earlier in this
    # project: chop the front off whichever signal is "ahead" so both
    # arrays' real content starts at the same index.
    dt = msg.get("dt", 0.0) or 0.0
    initial_shift = int(dt * sample_rate)
    if initial_shift > 0:
        rx_iq = np.pad(rx_iq[initial_shift:], (0, initial_shift))
    elif initial_shift < 0:
        ref_iq = np.pad(ref_iq[-initial_shift:], (0, -initial_shift))

    # 4. Coarse Doppler pilot estimate, to compensate the short window
    # below before delay-finding.
    #
    # The delay-vs-Doppler bias just above is systematic, not noise
    # (confirmed directly: ~-34 samples at 0.4 Hz, ~-176 samples at
    # 1.5 Hz, consistent regardless of the true delay) -- exactly the
    # decorrelation-smearing signature, so it's correctable rather than
    # something to just average away. rx_iq/ref_iq are already
    # dt-aligned above, so residual delay here is small (a fine
    # multipath offset, not weakmon's whole dt search range) -- safe
    # for a single coherent FFT read of the full-length channel
    # estimate to find a coarse Doppler, unlike trying this before any
    # delay alignment at all (that failure mode: an unresolved
    # multi-hundred-sample delay leaves the FT8 modulation not fully
    # cancelled, creating a periodic artifact at the 6.25 Hz symbol
    # rate and its harmonics that can outweigh a genuine sub-Hz
    # Doppler peak -- tested and confirmed separately; not an issue
    # here because dt has already done the coarse alignment).
    pilot_channel_est = rx_iq * np.conj(ref_iq)
    pilot_window = signal.windows.hann(len(pilot_channel_est))
    pilot_X = np.fft.fftshift(np.fft.fft(pilot_channel_est * pilot_window))
    pilot_freqs = np.fft.fftshift(np.fft.fftfreq(len(pilot_channel_est), d=1.0 / sample_rate))
    pilot_mask = (pilot_freqs >= -50.0) & (pilot_freqs <= 50.0)
    pilot_doppler_hz = float(pilot_freqs[pilot_mask][np.argmax(np.abs(pilot_X[pilot_mask]) ** 2)])

    # 5. Fine delay search via cross-correlation -- SHORT window only,
    # now Doppler-compensated using the pilot estimate.
    costas_samples = min(int(7 * 0.16 * sample_rate), min_len)
    t_short = np.arange(costas_samples) / sample_rate
    rx_short = rx_iq[:costas_samples] * np.exp(-1j * 2 * np.pi * pilot_doppler_hz * t_short)
    ref_short = ref_iq[:costas_samples]

    corr = signal.correlate(rx_short, ref_short, mode="full")
    lags = signal.correlation_lags(len(rx_short), len(ref_short))

    # Search window now centered on 0 (rx_iq/ref_iq are already
    # dt-aligned), not on dt itself.
    search_window_ms = 25.0
    search_window = int((search_window_ms / 1000.0) * sample_rate)
    center_idx = np.argmin(np.abs(lags - 0))
    start_idx = max(0, center_idx - search_window)
    end_idx = min(len(corr), center_idx + search_window)

    corr_slice = np.abs(corr[start_idx:end_idx])
    lag_slice = lags[start_idx:end_idx]

    # Find multipath peaks
    max_val = np.max(corr_slice)
    peaks, _ = signal.find_peaks(
        corr_slice,
        height=max_val * 0.4,
        prominence=max_val * 0.2,
        distance=int(0.002 * sample_rate) # must be 2 milliseconds apart
    )

    if len(peaks) == 0:
        peaks = [np.argmax(corr_slice)]

    # Primary peak magnitude for relative power calculation
    max_peak_power = corr_slice[peaks[0]] ** 2

    watterson_taps = []
    for p in peaks:
        tap_lag = lag_slice[p]
        tap_delay_ms = (tap_lag / sample_rate) * 1000.0

        # Shift reference to align with this tap -- full-length signals
        # now (not the short window used just above for delay-finding),
        # so extract_tap_metrics gets full Doppler frequency resolution.
        if tap_lag > 0:
            rx_tap = rx_iq[tap_lag:]
            ref_tap = ref_iq[: len(rx_tap)]
        elif tap_lag < 0:
            ref_tap = ref_iq[-tap_lag:]
            rx_tap = rx_iq[: len(ref_tap)]
        else:
            rx_tap = rx_iq
            ref_tap = ref_iq

        # De-modulate channel response c(t) = rx * conj(ref)
        channel_response = rx_tap * np.conj(ref_tap)

        # Calculate Watterson parameters
        d_shift, d_spread, power, p_rate, p_jitter = extract_tap_metrics(
            channel_response, sample_rate
        )

        rel_power_db = 10 * np.log10(max(power / max_peak_power, 1e-6))

        watterson_taps.append(
            {
                "relative_delay_ms": float(tap_delay_ms),
                "doppler_shift_hz": d_shift,
                "doppler_rms_spread_hz": d_spread,
                "relative_power_db": float(rel_power_db),
                "phase_rate_rad_s": p_rate,
                "phase_jitter_rad": p_jitter,
            }
        )

    return {
        "message": msg,
        "num_paths": len(watterson_taps),
        "watterson_taps": watterson_taps,
    }
