"""
align_diag.py -- is rx * conj(ref) actually phase-coherent?

Call diagnose() inside analyze() right after the two arrays are truncated to
min_len and BEFORE the dt shift is applied (so the sweep can be compared to dt):

    from align_diag import diagnose
    ...
    rx_iq = rx_iq[:min_len]; ref_iq = ref_iq[:min_len]
    diag = diagnose(rx_iq, ref_iq, sample_rate, dt_s, extract_tap_metrics)
    # add diag to the returned dict, e.g. result["diag"] = diag

Convention: shift s > 0 means the received signal arrives s samples LATER than
the reference (compare rx[s:] with ref[:N-s]). Compare best_shift_s with the
decoder's dt: they should agree to within a few tens of ms. A constant
difference (0.5 s?) means a dt-convention problem.

What to look at in the output:
  peak_over_floor    sweep peak / median of the sweep far from the peak. In
                     synthetic tests, noise alone gave 1.05-1.08, while real
                     aligned signals gave 1.13 (-3 dB in-band SNR) to 1.27
                     (+4 dB). Treat >1.10 as "something is aligned", and
                     confirm with the null comparison below.
  sweep_z            (peak-floor)/robust-sigma. NOT calibrated: noise-only
                     data gave ~7, so do not threshold on it.
  best_shift_s       where the peak is. Compare with dt.
  pilot_hz/_prom_db  frequency offset of the product at that shift (receiver
                     clock error + real Doppler). A Kiwi clock error of ~1 ppm
                     is ~7 Hz at 7 MHz. extract_tap_metrics() low-passes at
                     5 Hz, so an offset like that is filtered away.
  as_is / derotated  extract_tap_metrics() on the product without and with the
                     pilot offset removed.
  null               same chain on a deliberately wrong (time-rolled)
                     reference. Real results must differ clearly from this.
"""

import numpy as np

SYMBOL_S = 0.16


def _pair(rx, ref, s):
    if s >= 0:
        a = rx[s:]
        b = ref[: len(a)]
    else:
        b = ref[-s:]
        a = rx[: len(b)]
    m = min(len(a), len(b))
    return a[:m], b[:m]


def _block_metric(a, b, blk):
    """Sum over short blocks of |coherent block correlation|, normalised to <=1.
    Short blocks (default 20 ms) tolerate frequency offsets up to ~25 Hz."""
    k = len(a) // blk
    if k < 20:
        return 0.0
    n = k * blk
    p = (a[:n] * np.conj(b[:n])).reshape(k, blk).sum(axis=1)
    den = np.sqrt(np.sum(np.abs(a[:n]) ** 2) * np.sum(np.abs(b[:n]) ** 2))
    return float(np.sum(np.abs(p)) / den) if den > 0 else 0.0


def alignment_sweep(rx, ref, fs=12000, span_s=1.0, step_s=0.005, blk_s=0.02):
    blk = int(round(blk_s * fs))
    step = max(1, int(round(step_s * fs)))
    span = int(round(span_s * fs))
    shifts = np.arange(-span, span + 1, step)
    m = np.array([_block_metric(*_pair(rx, ref, int(s)), blk) for s in shifts])
    i = int(np.argmax(m))

    fine = np.arange(shifts[i] - step, shifts[i] + step + 1)
    mf = np.array([_block_metric(*_pair(rx, ref, int(s)), blk) for s in fine])
    j = int(np.argmax(mf))
    best, peak = int(fine[j]), float(mf[j])

    off = np.abs(shifts - best) > int(0.5 * fs)       # far from the peak = floor
    floor = m[off] if np.any(off) else m
    med = float(np.median(floor))
    mad = float(np.median(np.abs(floor - med))) * 1.4826 + 1e-12
    return {
        "best_shift_s": best / fs,
        "best_shift_samples": best,
        "peak_metric": peak,
        "floor_metric": med,
        "peak_over_floor": peak / (med + 1e-12),
        "sweep_z": (peak - med) / mad,
    }


def _pilot(prod, fs, fmax=50.0):
    n = len(prod)
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    X = np.fft.fftshift(np.fft.fft(prod * np.hanning(n), nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    msk = np.abs(f) <= fmax
    P = np.abs(X[msk]) ** 2
    k = int(np.argmax(P))
    return float(f[msk][k]), float(10 * np.log10(P[k] / (np.median(P) + 1e-30)))


def _metrics(prod, fs, f0, metrics_fn):
    t = np.arange(len(prod)) / fs
    d = metrics_fn(prod * np.exp(-2j * np.pi * f0 * t), fs)
    # extract_tap_metrics returns (shift, spread, power, phase_rate, jitter)
    return {
        "doppler_shift_hz": float(d[0]) + f0,
        "doppler_spread_hz": float(d[1]),
        "power": float(d[2]),
        "phase_jitter_rad": float(d[4]),
    }


def diagnose(rx_iq, ref_iq, fs, dt_s, metrics_fn, null_roll_s=3.3):
    sw = alignment_sweep(rx_iq, ref_iq, fs)
    s = sw["best_shift_samples"]
    a, b = _pair(rx_iq, ref_iq, s)
    prod = a * np.conj(b)

    f0, prom_db = _pilot(prod, fs)
    out = dict(sw)
    out["dt_reported_s"] = float(dt_s or 0.0)
    out["sweep_minus_dt_s"] = sw["best_shift_s"] - float(dt_s or 0.0)
    out["pilot_hz"] = f0
    out["pilot_prom_db"] = prom_db
    out["as_is"] = _metrics(prod, fs, 0.0, metrics_fn)       # what analyze() does today
    out["derotated"] = _metrics(prod, fs, f0, metrics_fn)    # pilot offset removed first

    bn = np.roll(b, int(null_roll_s * fs))                   # wrong reference
    pn = a * np.conj(bn)
    f0n, _ = _pilot(pn, fs)                                  # same procedure, fair null
    out["null"] = _metrics(pn, fs, f0n, metrics_fn)
    return out
