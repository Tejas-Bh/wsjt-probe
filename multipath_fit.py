"""
multipath_fit.py -- offline Doppler-domain multipath test on the stored z series.

FT8 occupies ~50 Hz, so delay-domain multipath (typical HF differential delays of
0.5-5 ms vs ~20 ms resolution) cannot be resolved. What CAN be resolved is a
difference in Doppler: modes/hops with different path-length rates show up as
separate lines in the 12.6 s symbol-rate spectrum (resolution ~0.08 Hz).

This fits one vs two Gaussian-spectrum components to z[k] (same likelihood as
channel_estimator.py, plus a free white-noise term) and reports

    d_nll   NLL(1 component) - NLL(2 components)   (larger = stronger evidence for two)
    comps   the two (power fraction, shift_hz, spread_hz) triples if d_nll > threshold

Use it on the JSON written by channel_estimator (record["z"]):

    from multipath_fit import z_from_record, two_component_test
    z = z_from_record(rec["z"])
    res = two_component_test(z)

Threshold: see D_NLL_MIN (calibrated on single-component simulations; re-check it
against your own data -- real spectra are not exactly Gaussian, which inflates d_nll).
Runtime is ~1 s per message, so run it offline and only on quality.ok messages.
"""

import numpy as np
from scipy.linalg import toeplitz, solve_triangular
from scipy.optimize import minimize

from channel_estimator import _V, _WV, SYMBOL_S, SIG_MIN, FD_MAX, fit_doppler

D_NLL_MIN = 10.0    # single-component simulations (75 trials) gave d_nll <= 8.6 (p99 8.5)


def z_from_record(zrec):
    ri = np.asarray(zrec["ri"], dtype=float).reshape(-1, 2)
    return (ri[:, 0] + 1j * ri[:, 1]) * zrec.get("rms", 1.0)


def _acov_multi(K, T, comps, N):
    tau = (np.arange(K)[:, None] + _V[None, :]) * T
    r = np.zeros(K, dtype=complex)
    for P, fd, sg in comps:
        a = 2 * np.pi ** 2 * sg ** 2
        r += P * (_WV[None, :] * np.exp(-a * tau ** 2 + 2j * np.pi * fd * tau)).sum(axis=1)
    r[0] += N
    return r


def _nll_multi(x, z, T, nc):
    lN = x[0]
    comps = []
    for i in range(nc):
        lp, ls, fd = x[1 + 3 * i: 4 + 3 * i]
        comps.append((np.exp(lp), fd, np.exp(ls)))
    r = _acov_multi(len(z), T, comps, np.exp(lN))
    S = toeplitz(r, np.conj(r))
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return 1e10
    y = solve_triangular(L, z, lower=True)
    return float(2 * np.sum(np.log(np.diag(L).real)) + np.sum(np.abs(y) ** 2))


def two_component_test(z, T=SYMBOL_S, threshold=None):
    thr = D_NLL_MIN if threshold is None else threshold
    K = len(z)
    if K < 40:
        return None
    one = fit_doppler(z, T)
    if one is None:
        return None
    s = np.sqrt(np.mean(np.abs(z) ** 2))
    zn = z / s
    P1, N1 = one["power"] / s ** 2, one["noise"] / s ** 2
    sg1, f1 = one["sigma_hz"], one["fd_hz"]
    bounds = [(-12, 6)] + [(-12, 6), (np.log(SIG_MIN), np.log(6.0)), (-FD_MAX, FD_MAX)] * 2

    # 1-component NLL in the same parameterisation (fit_doppler already minimised it)
    x1 = [np.log(N1), np.log(P1), np.log(sg1), f1]
    nll1 = _nll_multi(x1, zn, T, 1)

    best = None
    for df in (0.15, 0.4, 0.9, 1.6):
        for sgn in (-1, 1):
            f2 = float(np.clip(f1 + sgn * df, -FD_MAX, FD_MAX))
            for sg2 in (max(sg1, 0.05), 0.08):
                x0 = [np.log(N1),
                      np.log(0.6 * P1), np.log(max(sg1 * 0.7, 0.03)), f1,
                      np.log(0.4 * P1), np.log(sg2), f2]
                r = minimize(_nll_multi, x0, args=(zn, T, 2), method="L-BFGS-B", bounds=bounds,
                             options={"maxiter": 200})
                if best is None or r.fun < best.fun:
                    best = r
    nll1 = min(nll1, nll1)
    d_nll = float(nll1 - best.fun)
    x = best.x
    comps = sorted(
        [(float(np.exp(x[1])), float(x[3]), float(np.exp(x[2]))),
         (float(np.exp(x[4])), float(x[6]), float(np.exp(x[5])))], key=lambda c: c[1])
    tot = comps[0][0] + comps[1][0]
    out = {
        "d_nll": d_nll,
        "two_paths": bool(d_nll > thr),
        "n_symbols": K,
        "comps": [{"power_frac": c[0] / tot, "shift_hz": c[1], "spread_hz": c[2]} for c in comps],
        "one": {"shift_hz": f1, "spread_hz": sg1},
    }
    return out
