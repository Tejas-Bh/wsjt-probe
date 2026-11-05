"""
channel_estimator.py -- symbol-rate channel estimate + ML Doppler fit (mode-independent).

Usage (FT8 is the default mode; pass spec=modes.WSPR for WSPR):

    from channel_estimator import estimate_channel
    est = estimate_channel(rx_iq, ref_iq, sample_rate, dt)            # FT8
    est = estimate_channel(rx_iq, ref_iq, sample_rate, dt, spec=WSPR) # WSPR
    return {"message": msg, "channel": est["channel"],
            "quality": est["quality"], "z": est["z"]}

Method (all mode constants come from modes.ModeSpec)
------
1. Alignment: first guess (decoder dt + spec.dt_offset_s), then a search over +/-align_span_s
   (coarse on a decimated copy, then refined at full rate). Never trust the decoder dt blindly.
2. Pilot (receiver clock + transmitter offset) is estimated and removed; for modes with
   chirp_max_hz_per_s > 0 (WSPR) a linear frequency drift is estimated and removed too, and
   reported as quality["drift_hz_per_min"]. Removed drift mixes transmitter and ionosphere.
3. Matched integration over each symbol -> z[k] (one complex channel sample per symbol).
4. Maximum-likelihood fit of z ~ CN(0, Sigma), Sigma[m] = P*R_c(m) + N*delta: Gaussian Doppler
   spectrum (mean f_d, std sigma_f) + free white-noise term; the symbol integration window is
   part of the model, so noise does not inflate the spread.

Definitions
-----------
doppler_spread_hz   sigma_f, std of the Doppler power spectrum (ITU "spread" = 2*sigma_f).
doppler_shift_hz    pilot + fitted mean Doppler (raw: still contains receiver clock error).
quality.ok          llr > spec.llr_min and enough symbols.
quality.llr         log-likelihood gain of 'signal present' over 'noise only'.

Validation against WSJT-X's ft8sim (one realisation per fdop, SNR -10): alignment within +/-5 ms
(so FT8 dt + 0.5 s is confirmed against jt9's own convention) and sigma_f ~= fdop / 4 (fdop 0.3
-> 0.078 Hz, 0.6 -> 0.152 Hz; the 0.05 and 0.1 files sit at the estimator floor, as they should).
So ft8sim's nominal "spread" is about 4*sigma_f, NOT the 2*sigma_f of the ITU convention.
Run validate_ft8sim.py for statistics over many files.

Limits: spreads above ~Nyquist/2 (FT8 ~1.5 Hz, WSPR ~0.4 Hz) alias and read as noise.
Delay is not estimated (FT8 ~50 Hz and WSPR ~6 Hz bandwidth cannot resolve HF multipath delay).
"""

import numpy as np
import scipy.signal as sps
from scipy.linalg import toeplitz, solve_triangular
from scipy.optimize import minimize

from modes import FT8, ModeSpec  # noqa: F401  (ModeSpec re-exported for convenience)

# FT8 defaults kept as module-level names for older scripts (multipath_fit.py imports them)
SYMBOL_S = FT8.symbol_s
LLR_MIN = FT8.llr_min
SIG_MIN, SIG_MAX, FD_MAX = FT8.sig_min_hz, FT8.sig_max_hz, FT8.fd_max_hz
CHIRP_LLR_GAIN = 3.0   # extra log-likelihood required to accept the drift-removed fit (1 extra parameter)
CI_DELTA = 1.5   # empirically calibrated: delta-NLL 0.5 gave only ~52% coverage in simulation

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
    """Slow reference implementation (full-rate search)."""
    blk = int(round(blk_s * fs))
    coarse = max(1, int(round(coarse_s * fs)))
    span = int(round(span_s * fs))
    cand = np.arange(shift0 - span, shift0 + span + 1, coarse)
    m = [_block_metric(*_pair(rx, ref, int(s)), blk) for s in cand]
    s1 = int(cand[int(np.argmax(m))])
    fine = np.arange(s1 - coarse, s1 + coarse + 1)
    mf = [_block_metric(*_pair(rx, ref, int(s)), blk) for s in fine]
    return int(fine[int(np.argmax(mf))])


def refine_alignment_fast(rx, ref, fs, shift0, spec=FT8):
    """Coarse search on a decimated copy, then short full-rate refinements.
    Returns (shift, rx_decimated, ref_decimated)."""
    D = spec.decim
    rxd = sps.resample_poly(rx, 1, D)
    refd = sps.resample_poly(ref, 1, D)
    fsd = fs / D
    blk_d = max(2, int(round(spec.align_block_s * fsd)))
    c0 = int(round(shift0 / D))
    span_d = int(round(spec.align_span_s * fsd))
    cand = np.arange(c0 - span_d, c0 + span_d + 1)
    m = [_block_metric(*_pair(rxd, refd, int(c)), blk_d) for c in cand]
    s1 = int(cand[int(np.argmax(m))]) * D

    blk = int(round(spec.align_block_s * fs))
    step = max(1, D // 5)
    stages = [(step, D)]
    s = step
    while s > 1:
        ns = max(1, s // 4)
        stages.append((ns, 3 if s <= 4 else s))
        s = ns
    for step, half in stages:
        cand = np.arange(s1 - half, s1 + half + 1, step)
        mf = [_block_metric(*_pair(rx, ref, int(c)), blk) for c in cand]
        s1 = int(cand[int(np.argmax(mf))])
    return s1, rxd, refd


def _pilot(prod, fs, fmax=50.0):
    n = len(prod)
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    X = np.fft.fftshift(np.fft.fft(prod * np.hanning(n), nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    msk = np.abs(f) <= fmax
    return float(f[msk][int(np.argmax(np.abs(X[msk]) ** 2))])


def _dechirp(prod, fs, fdot_max, fmax):
    """Search (f0, fdot) maximising the dechirped spectral peak. phase = 2*pi*f0*t + pi*fdot*t^2."""
    n = len(prod)
    t = np.arange(n) / fs
    tobs = n / fs
    step = 3.0 / tobs ** 2
    nfft = 1 << (int(np.ceil(np.log2(n))) + 3)
    fr = np.fft.fftfreq(nfft, 1.0 / fs)
    msk = np.abs(fr) <= fmax
    w = np.hanning(n)
    best = (-1.0, 0.0, 0.0)
    for fd in np.arange(-fdot_max, fdot_max + step, step):
        X = np.abs(np.fft.fft(prod * np.exp(-1j * np.pi * fd * t ** 2) * w, nfft))
        Xm = np.where(msk, X, 0.0)
        k = int(np.argmax(Xm))
        if Xm[k] > best[0]:
            best = (Xm[k], float(fr[k]), float(fd))
    return best[1], best[2]


# ----------------------------------------------------------- symbol series
def symbol_series(a, b, fs, f0, fdot=0.0, spec=FT8):
    """z[k] = sum(rx*conj(ref)) / sum(|ref|^2) over each symbol; pilot (and drift) removed."""
    t = np.arange(len(a)) / fs
    prod = a * np.conj(b) * np.exp(-1j * (2 * np.pi * f0 * t + np.pi * fdot * t ** 2))
    en = np.abs(b) ** 2
    thr = 0.1 * np.max(np.abs(b))
    on = np.nonzero(np.abs(b) > thr)[0]
    n0 = int(on[0]) if len(on) else 0
    L = int(round(spec.symbol_s * fs))
    K = min(spec.n_symbols, (len(prod) - n0) // L)
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


def fit_doppler(z, T=SYMBOL_S, compute_ci=False, fd_max=FD_MAX, sig_min=SIG_MIN, sig_max=SIG_MAX,
                min_symbols=20):
    K = len(z)
    if K < min_symbols:
        return None
    s = np.sqrt(np.mean(np.abs(z) ** 2))
    if s == 0:
        return None
    zn = z / s
    r1 = np.mean(zn[1:] * np.conj(zn[:-1]))
    fd0 = float(np.clip(np.angle(r1) / (2 * np.pi * T), -fd_max, fd_max))
    bounds = [(-12, 6), (-12, 6), (np.log(sig_min), np.log(sig_max)), (-fd_max, fd_max)]

    best = None
    mid = float(np.sqrt(sig_min * sig_max))
    for sig0, p0 in ((max(sig_min * 5, 0.1 * mid), 0.6), (mid * 0.5, 0.6), (mid * 2.0, 0.4)) \
            if (sig_min, sig_max) != (SIG_MIN, SIG_MAX) else ((0.1, 0.6), (0.4, 0.6), (1.5, 0.4)):
        sig0 = float(np.clip(sig0, sig_min * 1.01, sig_max * 0.99))
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
        grid = np.unique(np.concatenate([np.geomspace(sig_min, sig_max, 20), [sig_hat]]))
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
        out["spread_hi_hz"] = hi if hi is not None else float(sig_max)
    return out


# --------------------------------------------------------------- top level
def _z_record(z):
    """Compact, reprocessable record of the symbol-rate channel series.
    Normalised to unit RMS so the JSON stays small; rms restores the scale."""
    rms = float(np.sqrt(np.mean(np.abs(z) ** 2))) if len(z) else 0.0
    if rms == 0.0:
        return {"rms": 0.0, "ri": []}
    zn = z / rms
    ri = np.round(np.column_stack([zn.real, zn.imag]).ravel(), 4)
    return {"rms": rms, "ri": [float(v) for v in ri]}


def estimate_channel(rx_iq, ref_iq, fs, dt, compute_ci=False, fast=True, spec=FT8):
    n = min(len(rx_iq), len(ref_iq))
    rx_iq, ref_iq = rx_iq[:n], ref_iq[:n]
    off = spec.dt_offset_s
    shift0 = int(round(((dt or 0.0) + off) * fs)) if dt is not None else int(round(off * fs))

    fdot = 0.0
    f0_chirp = None
    if fast:
        shift, rxd, refd = refine_alignment_fast(rx_iq, ref_iq, fs, shift0, spec)
        ad, bd = _pair(rxd, refd, int(round(shift / spec.decim)))
        f0 = _pilot(ad * np.conj(bd), fs / spec.decim, spec.pilot_max_hz)
        if spec.chirp_max_hz_per_s > 0:
            f0_chirp, fdot_chirp = _dechirp(ad * np.conj(bd), fs / spec.decim,
                                            spec.chirp_max_hz_per_s, spec.pilot_max_hz)
    else:
        shift = refine_alignment(rx_iq, ref_iq, fs, shift0, spec.align_span_s,
                                 blk_s=spec.align_block_s)
        a_, b_ = _pair(rx_iq, ref_iq, shift)
        f0 = _pilot(a_ * np.conj(b_), fs, spec.pilot_max_hz)
    a, b = _pair(rx_iq, ref_iq, shift)

    def _fit(z_):
        return fit_doppler(z_, spec.symbol_s, compute_ci=compute_ci, fd_max=spec.fd_max_hz,
                           sig_min=spec.sig_min_hz, sig_max=spec.sig_max_hz)

    z = symbol_series(a, b, fs, f0, 0.0, spec)
    fit = _fit(z)
    drift_used = False
    if f0_chirp is not None:
        # keep the drift removal only if it clearly improves the likelihood: a real transmitter
        # drift breaks the no-drift fit, whereas on a drift-free signal the dechirp would just
        # absorb part of the genuine Doppler fluctuation and bias the spread low.
        z1 = symbol_series(a, b, fs, f0_chirp, fdot_chirp, spec)
        fit1 = _fit(z1)
        if fit1 is not None and (fit is None or fit1["llr"] > fit["llr"] + CHIRP_LLR_GAIN):
            z, fit, f0, fdot, drift_used = z1, fit1, f0_chirp, fdot_chirp, True
    align_ms = (shift - shift0) / fs * 1e3

    base_q = {"mode": spec.name, "pilot_hz": f0, "drift_hz_per_min": fdot * 60.0, "drift_removed": drift_used,
              "align_offset_ms": align_ms}
    if fit is None:
        return {
            "channel": {"doppler_shift_hz": None, "doppler_spread_hz": None,
                        "spread_lo_hz": None, "spread_hi_hz": None, "path_power": None},
            "quality": dict(base_q, ok=False, llr=0.0, snr_sym_db=None, n_symbols=int(len(z))),
            "z": _z_record(z),
        }

    channel = {
        "doppler_shift_hz": float(f0 + fit["fd_hz"]),   # raw: still contains receiver clock error
        "doppler_spread_hz": fit["sigma_hz"],
        "spread_lo_hz": fit["spread_lo_hz"],
        "spread_hi_hz": fit["spread_hi_hz"],
        "path_power": fit["power"],
    }
    quality = dict(base_q,
                   ok=bool(fit["llr"] > spec.llr_min and fit["n_symbols"] >= spec.min_symbols),
                   llr=fit["llr"], snr_sym_db=fit["snr_sym_db"], n_symbols=fit["n_symbols"])
    return {"channel": channel, "quality": quality, "z": _z_record(z)}
