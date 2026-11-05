#!/usr/bin/env python3
"""
validate_ft8sim.py -- compare the Doppler-spread estimator with ft8sim's nominal `fdop`.

Generate known-answer files (SNR -10, 1 ms delay; many files per fdop for statistics):

    mkdir f03 && cd f03 && ft8sim "K1ABC W9XYZ EN37" 1500.0 0.0 0.3 1.0 20 -10 && cd ..
    (same for 0.05, 0.1, 0.6 ...)

Then:

    python validate_ft8sim.py "K1ABC W9XYZ EN37" 0.3:"f03/*.wav" 0.6:"f06/*.wav" ...

Output per fdop: the median estimated sigma_f and k = fdop / sigma_f, the factor that converts
ft8sim's nominal spread into our sigma_f. On the first five files uploaded (one realisation per
fdop) k came out ~4: ft8sim's fdop looks like a 4*sigma full width (sigma_f = fdop / 4).
Spreads near the estimator floor (sigma_f < ~0.03 Hz) are reported as 'floor' and excluded.
"""

import glob
import sys
import wave

import numpy as np
import scipy.signal as sps

import channel_estimator as ce
import wsjtx_io as w
from modes import FT8

FS = 12000


def load(path):
    r = wave.open(path)
    return np.frombuffer(r.readframes(r.getnframes()), dtype="<i2").astype(float)


def to_baseband(x, f0=1500.0, bw=90.0):
    c = f0 + 3.5 * 6.25
    nyq = FS / 2
    b, a = sps.butter(4, [(c - bw / 2) / nyq, (c + bw / 2) / nyq], btype="band")
    y = sps.filtfilt(b, a, x)
    t = np.arange(len(y)) / FS
    return sps.hilbert(y) * np.exp(-2j * np.pi * c * t)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    msg = sys.argv[1]
    sym = w.ft8sim_symbols(msg)
    if sym is None:
        sys.exit("could not get channel symbols from ft8sim for %r" % msg)
    ref = w.reference_baseband(sym, FT8, FS, 15 * FS)
    print("%8s %4s | %8s %8s  %8s | %6s %6s" % ("fdop", "n", "med_sig", "mean_sig", "k=fdop/sig", "floor", "!ok"))
    for arg in sys.argv[2:]:
        fd_s, pat = arg.split(":", 1)
        fd = float(fd_s)
        sig, floor, bad = [], 0, 0
        for f in sorted(glob.glob(pat)):
            e = ce.estimate_channel(to_baseband(load(f)), ref, FS, -0.0, spec=FT8)
            if not e["quality"]["ok"]:
                bad += 1
                continue
            s = e["channel"]["doppler_spread_hz"]
            if s < 1.5 * FT8.sig_min_hz:
                floor += 1
            else:
                sig.append(s)
        if sig:
            sig = np.array(sig)
            print("%8.3f %4d | %8.4f %8.4f  %8.2f | %6d %6d" % (fd, len(sig), np.median(sig), sig.mean(),
                                                                 fd / np.median(sig), floor, bad))
        else:
            print("%8.3f %4d | (all at floor or not ok)         | %6d %6d" % (fd, 0, floor, bad))


if __name__ == "__main__":
    main()
