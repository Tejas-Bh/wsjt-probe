"""
channel_estimator.py -- symbol-rate channel estimate + ML Doppler fit for FT8.

Replaces steps 3-5 and the tap loop of analyze(). Call it after rx_iq / ref_iq
are built and truncated to min_len:

    from channel_estimator import estimate_channel
    est = estimate_channel(rx_iq, ref_iq, sample_rate, dt)
    return {"message": msg, "channel": est["channel"],
            "quality": est["quality"], "z": est["z"]}

Method
------
1. Alignment. weakmon's dt is relative to the nominal 0.5 s start, so the first
   guess is dt + 0.5 s; a short +/-40 ms search then fixes the residual.
2. Pilot frequency (clock error + mean Doppler) is removed from rx*conj(ref).
3. Matched integration over each 0.16 s symbol -> z[k], one complex channel
   sample per symbol (6.25 Hz sampling, +/-3.125 Hz unambiguous Doppler). This
   replaces the 5 Hz low-pass, which only attenuated the 6.25 Hz keying a little.
4. Maximum-likelihood fit of  z ~ CN(0, Sigma),  Sigma[m] = P*R_c(m) + N*delta:
   Gaussian Doppler spectrum (Watterson model) with mean f_d and std sigma_f,
   plus a free white-noise term N. Because N is fitted, noise no longer adds to
   the spread (the old moment estimator was floor-limited). The 0.16 s
   integration window is included in the model.

Definitions
-----------
doppler_spread_hz        sigma_f, the standard deviation of the Doppler power
                         spectrum (same quantity the old RMS-width estimator
                         targeted). The ITU/Watterson "spread" is 2*sigma_f.
doppler_shift_hz         pilot + fitted mean Doppler (raw: still contains
                         receiver clock error; subtract your wwv_calibrate offset).
spread_lo_hz/spread_hi_hz  profile-likelihood interval (compute_ci=True, ~0.5 s extra per
                         message). Empirical coverage in simulation was 78-85%.
                         lo == 0 means "not resolved: spread could be ~0".
quality.llr              log-likelihood gain of 'signal present' over 'noise only'.
quality.snr_sym_db       fitted channel power / noise power per symbol.
quality.ok               llr > LLR_MIN and enough symbols; gate your results on it.

Limits (from simulation): spread is recovered to ~10% for sigma_f 0.1-1.0 Hz at
in-band SNR of 0 to +4 dB. Near sigma_f >= 1.5 Hz the 6.25 Hz symbol sampling
aliases the spectrum, the channel looks like noise, and quality.ok goes False.
Delay is NOT estimated (FT8's ~50 Hz bandwidth cannot resolve it).
"""

import numpy as np
from scipy.linalg import toeplitz, solve_triangular
from scipy.optimize import minimize

SYMBOL_S = 0.16
LLR_MIN = 12.0          # noise-only runs (60 trials, full pipeline) gave LLR <= 9.5, p99 = 8.7

# 8-point Gauss-Legendre on [-1,0] and [0,1] for the triangular symbol window
_gx, _gw = np.polynomial.legendre.leggauss(8)
_V = np.concatenate([(_gx - 1) / 2, (_gx + 1) / 2])
_WV = np.concatenate([_gw / 2 * (1 - np.abs((_gx - 1) / 2)),
                      _gw / 2 * (1 - np.abs((_gx + 1) / 2))])


# ---------------------------------------------------------------- alignment
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
    k = len(a) // blk
    if k < 20:
        return 0.0
    n = k * blk
    p = (a[:n] * np.conj(b[:n])).reshape(k, blk).sum(axis=1)
    den = np.sqrt(np.sum(np.abs(a[:n]) ** 2) * np.sum(np.abs(b[:n]) ** 2))
    return float(np.sum(np.abs(p)) / den) if den > 0 else 0.0


def refine_alignment(rx, ref, fs, shift0, span_s=0.04, coarse_s=0.002, blk_s=0.02):
    blk = int(round(blk_s * fs))
    coarse = max(1, int(round(coarse_s * fs)))
    span = int(round(span_s * fs))
    cand = np.arange(shift0 - span, shift0 + span + 1, coarse)
    m = [_block_metric(*_pair(rx, ref, int(s)), blk) for s in cand]
    s1 = int(cand[int(np.argmax(m))])
    fine = np.arange(s1 - coarse, s1 + coarse + 1)
    mf = [_block_metric(*_pair(rx, ref, int(s)), blk) for s in fine]
    return int(fine[int(np.argmax(mf))])


def _pilot(prod, fs, fmax=50.0):
    n = len(prod)
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    X = np.fft.fftshift(np.fft.fft(prod * np.hanning(n), nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    msk = np.abs(f) <= fmax
    return float(f[msk][int(np.argmax(np.abs(X[msk]) ** 2))])


# ----------------------------------------------------------- symbol series
def symbol_series(a, b, fs, f0):
    """z[k] = sum(rx*conj(ref)) / sum(|ref|^2) over each symbol; f0 removed."""
    t = np.arange(len(a)) / fs
    prod = a * np.conj(b) * np.exp(-2j * np.pi * f0 * t)
    en = np.abs(b) ** 2
    thr = 0.1 * np.max(np.abs(b))
    on = np.nonzero(np.abs(b) > thr)[0]
    n0 = int(on[0]) if len(on) else 0
    L = int(round(SYMBOL_S * fs))
    K = min(79, (len(prod) - n0) // L)
    if K <= 0:
        return np.zeros(0, dtype=complex)
    idx = n0 + np.arange(K * L)
    num = prod[idx].reshape(K, L).sum(axis=1)
    den = en[idx].reshape(K, L).sum(axis=1)
    keep = den > 0.5 * np.median(den)
    return (num[keep] / den[keep]).astype(complex)


# --------------------------------------------------------------- ML fit
def _acov(K, T, sigma, fd, P, N):
    tau = (np.arange(K)[:, None] + _V[None, :]) * T
    a = 2 * np.pi ** 2 * sigma ** 2
    r = P * (_WV[None, :] * np.exp(-a * tau ** 2 + 2j * np.pi * fd * tau)).sum(axis=1)
    r[0] += N
    return r


def _nll_x(x, z, T):
    lp, ln, ls, fd = x
    return _nll(z, T, np.exp(ls), fd, np.exp(lp), np.exp(ln))


def _nll(z, T, sigma, fd, P, N):
    r = _acov(len(z), T, sigma, fd, P, N)
    S = toeplitz(r, np.conj(r))
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return 1e10
    y = solve_triangular(L, z, lower=True)
    return float(2 * np.sum(np.log(np.diag(L).real)) + np.sum(np.abs(y) ** 2))


SIG_MIN, SIG_MAX, FD_MAX = 0.02, 6.0, 3.1
CI_DELTA = 1.5   # empirically calibrated: delta-NLL 0.5 gave only ~52% coverage in simulation


def fit_doppler(z, T=SYMBOL_S, compute_ci=False):
    K = len(z)
    if K < 20:
        return None
    s = np.sqrt(np.mean(np.abs(z) ** 2))
    if s == 0:
        return None
    zn = z / s
    r1 = np.mean(zn[1:] * np.conj(zn[:-1]))
    fd0 = float(np.clip(np.angle(r1) / (2 * np.pi * T), -FD_MAX, FD_MAX))
    bounds = [(-12, 6), (-12, 6), (np.log(SIG_MIN), np.log(SIG_MAX)), (-FD_MAX, FD_MAX)]

    best = None
    for sig0, p0 in ((0.1, 0.6), (0.4, 0.6), (1.5, 0.4)):
        x0 = [np.log(p0), np.log(1 - p0 + 1e-3), np.log(sig0), fd0]
        res = minimize(_nll_x, x0, args=(zn, T), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    lp, ln, ls, fd = best.x
    nll0 = K * np.log(np.mean(np.abs(zn) ** 2)) + K       # noise-only model
    out = {
        "sigma_hz": float(np.exp(ls)),
        "fd_hz": float(fd),
        "power": float(np.exp(lp) * s ** 2),
        "noise": float(np.exp(ln) * s ** 2),
        "snr_sym_db": float(10 * np.log10(np.exp(lp - ln))),
        "llr": float(nll0 - best.fun),
        "n_symbols": int(K),
        "spread_lo_hz": None,
        "spread_hi_hz": None,
    }
    if compute_ci:
        sig_hat = out["sigma_hz"]
        grid = np.unique(np.concatenate([np.geomspace(SIG_MIN, SIG_MAX, 20), [sig_hat]]))
        prof = []
        for sg in grid:
            f = lambda y, sg=sg: _nll(zn, T, sg, y[2], np.exp(y[0]), np.exp(y[1]))
            r = minimize(f, [lp, ln, fd], method="L-BFGS-B",
                         bounds=[bounds[0], bounds[1], bounds[3]])
            prof.append(r.fun)
        prof = np.array(prof)
        prof = prof - min(prof.min(), best.fun)
        lg = np.log(grid)
        i0 = int(np.argmin(np.abs(grid - sig_hat)))

        def crossing(idx_range):
            prev = i0
            for i in idx_range:
                if prof[i] > CI_DELTA:
                    w = (CI_DELTA - prof[prev]) / (prof[i] - prof[prev] + 1e-12)
                    return float(np.exp(lg[prev] + w * (lg[i] - lg[prev])))
                prev = i
            return None

        lo = crossing(range(i0 - 1, -1, -1))
        hi = crossing(range(i0 + 1, len(grid)))
        out["spread_lo_hz"] = lo if lo is not None else 0.0     # 0 = not resolved
        out["spread_hi_hz"] = hi if hi is not None else float(SIG_MAX)
    return out


# --------------------------------------------------------------- top level
def _z_record(z):
    """Compact, reprocessable record of the symbol-rate channel series (79 complex values).
    Normalised to unit RMS so the JSON stays small; z_rms restores the scale."""
    rms = float(np.sqrt(np.mean(np.abs(z) ** 2))) if len(z) else 0.0
    if rms == 0.0:
        return {"rms": 0.0, "ri": []}
    zn = z / rms
    ri = np.round(np.column_stack([zn.real, zn.imag]).ravel(), 4)
    return {"rms": rms, "ri": [float(v) for v in ri]}


def estimate_channel(rx_iq, ref_iq, fs, dt, compute_ci=False):
    n = min(len(rx_iq), len(ref_iq))
    rx_iq, ref_iq = rx_iq[:n], ref_iq[:n]
    shift0 = int(round((dt + 0.5) * fs)) if dt is not None else int(0.5 * fs)
    shift = refine_alignment(rx_iq, ref_iq, fs, shift0)
    a, b = _pair(rx_iq, ref_iq, shift)
    f0 = _pilot(a * np.conj(b), fs)
    z = symbol_series(a, b, fs, f0)
    fit = fit_doppler(z, SYMBOL_S, compute_ci=compute_ci)
    align_ms = (shift - shift0) / fs * 1e3

    if fit is None:
        return {
            "channel": {"doppler_shift_hz": None, "doppler_spread_hz": None,
                        "spread_lo_hz": None, "spread_hi_hz": None, "path_power": None},
            "quality": {"ok": False, "llr": 0.0, "snr_sym_db": None, "n_symbols": int(len(z)),
                        "pilot_hz": f0, "align_offset_ms": align_ms},
            "z": _z_record(z),
        }

    channel = {
        "doppler_shift_hz": float(f0 + fit["fd_hz"]),   # raw: still contains receiver clock error
        "doppler_spread_hz": fit["sigma_hz"],
        "spread_lo_hz": fit["spread_lo_hz"],
        "spread_hi_hz": fit["spread_hi_hz"],
        "path_power": fit["power"],
    }
    quality = {
        "ok": bool(fit["llr"] > LLR_MIN and fit["n_symbols"] >= 60),
        "llr": fit["llr"],
        "snr_sym_db": fit["snr_sym_db"],
        "n_symbols": fit["n_symbols"],
        "pilot_hz": f0,                 # removed before the fit; fitted shift = pilot + fd
        "align_offset_ms": align_ms,
    }
    return {"channel": channel, "quality": quality, "z": _z_record(z)}
