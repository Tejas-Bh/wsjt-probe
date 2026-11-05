"""
modes.py -- everything that differs between digital modes, in one place.

The channel estimator (channel_estimator.py), the aggregation (aggregate.py) and the capture
loop are mode-independent; they take a ModeSpec. Add a mode by adding a spec here.

UNVERIFIED values are marked. dt_offset_s in particular must be confirmed empirically with the
alignment sweep (align_diag.py) -- for FT8 it turned out to be 0.5 s (weakmon's dt is relative
to the nominal start), not 0. For WSPR the true constant is uncertain (the transmission starts
1 s into the even minute; the decoder's dt reference may differ), so the WSPR spec centres a
+/-1 s search on 1.5 s and the estimator reports quality["align_offset_ms"] = where it really
was. Read that number from the first real captures and put the constant here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    name: str
    slot_s: float              # capture cycle length
    symbol_s: float            # symbol period (matched-integration window)
    n_symbols: int
    n_tones: int
    tone_spacing_hz: float
    dt_offset_s: float         # first guess: shift = (decoder_dt + dt_offset_s) seconds
    align_span_s: float        # +/- search span around that guess
    align_block_s: float       # coherent block length of the coarse alignment metric
    decim: int                 # decimation for coarse alignment / pilot search (12 kHz input)
    fd_max_hz: float           # fitted mean Doppler bound (just under symbol-rate Nyquist)
    sig_min_hz: float
    sig_max_hz: float          # beyond ~Nyquist/2 the spectrum aliases and cannot be measured
    pilot_max_hz: float        # search range for the pilot (receiver + transmitter offset)
    chirp_max_hz_per_s: float  # 0 = no drift (chirp) removal; >0 = search +/- this rate
    llr_min: float             # 'signal present' threshold (calibrate per mode from noise-only runs)
    min_symbols: int           # need at least this many usable symbols

    @property
    def nyquist_hz(self):
        return 0.5 / self.symbol_s


FT8 = ModeSpec(
    name="ft8", slot_s=15.0, symbol_s=0.16, n_symbols=79, n_tones=8, tone_spacing_hz=6.25,
    dt_offset_s=0.5, align_span_s=0.04, align_block_s=0.02, decim=20,
    fd_max_hz=3.1, sig_min_hz=0.02, sig_max_hz=6.0, pilot_max_hz=50.0,
    chirp_max_hz_per_s=0.0, llr_min=12.0, min_symbols=60,
)

# WSPR-2: 162 symbols of 8192/12000 s, 4-FSK, 1.4648 Hz spacing, 110.6 s long.
# dt_offset_s = 1.0 is VERIFIED on a wsprsim file (signal starts at exactly 1.000 s and wsprd
# prints dt = -0.0). llr_min = 15: noise-only runs (24 trials, full pipeline including the
# alignment search) gave LLR <= 11.0 (p90 9.0); re-check on real data.
WSPR = ModeSpec(
    name="wspr", slot_s=120.0, symbol_s=8192 / 12000.0, n_symbols=162, n_tones=4,
    tone_spacing_hz=12000.0 / 8192.0,
    dt_offset_s=1.0, align_span_s=0.5, align_block_s=0.1, decim=120,
    fd_max_hz=0.72, sig_min_hz=0.005, sig_max_hz=1.0, pilot_max_hz=3.0,
    chirp_max_hz_per_s=0.067, llr_min=15.0, min_symbols=140,
)

MODES = {"ft8": FT8, "wspr": WSPR}


def get_mode(name):
    try:
        return MODES[name.lower()]
    except KeyError:
        raise ValueError("unknown mode %r (known: %s)" % (name, ", ".join(MODES)))
