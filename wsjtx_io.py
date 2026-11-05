"""
wsjtx_io.py -- glue between our IQ/audio buffers and the WSJT-X 2.7 command-line tools.

Formats below were taken from real output of wsprsim / wsprd / ft8sim / jt9 on the author's
install, and from the contents of a wsprsim-written test.c2:

  .c2 file      14-byte name | int32 ntrmin (=2 for WSPR-2) | float64 dial MHz |
                45000 complex64 samples at 375 Hz (120 s). Baseband zero = dial + 1500 Hz.
                *** Spectrum is mirrored relative to normal USB complex baseband: ***
                symbol s sits at -(s-1.5)*1.4648 Hz in the c2 file (verified: perfect
                correlation only with the conjugated reference). So c2 = conj(standard
                baseband); iq_to_c2() does this, and read_c2() undoes it.
  WSPR timing   wsprsim places the signal at exactly 1.000 s (110.592 s long) and wsprd prints
                dt = -0.0 for it  =>  decoder dt is relative to a 1.0 s nominal start
                (modes.WSPR.dt_offset_s = 1.0).
  wsprd line    [name | yymmdd hhmm]  snr  dt  freq_MHz  drift  message...
  jt9 -8 line   hhmmss  snr  dt  freq_Hz  ~  message   (trailing a1/a2/? flags possible)
  ft8code/wsprcode  print the channel symbols directly (preferred: no waveform synthesis)
  wsprsim -c    prints "Channel symbols:" then 162 digits 0-3
  ft8sim        prints "Channel symbols:" then a 79-digit string 0-7 (+ writes wav files)

Reference tones are built in STANDARD orientation (symbol s -> +(s-centre)*spacing Hz) to
match our own IQ, which is not mirrored.
"""

import os
import re
import struct
import subprocess
import tempfile
import wave

import numpy as np
import scipy.signal as sps
from scipy.special import erf

C2_RATE = 375.0
C2_N = 45000


# ------------------------------------------------------------------ c2 files
def read_c2(path):
    """Return (dial_mhz, z) with z in STANDARD orientation (already conjugated), 375 Hz."""
    b = open(path, "rb").read()
    ntrmin = struct.unpack("<i", b[14:18])[0]
    dial = struct.unpack("<d", b[18:26])[0]
    x = np.frombuffer(b[26:26 + 8 * C2_N], dtype="<f4").reshape(-1, 2)
    z = (x[:, 0] + 1j * x[:, 1]).astype(np.complex128)
    return dial, np.conj(z), ntrmin


def iq_to_c2(iq, dial_khz, fs=12000.0, path=None, name="slot.c2", center_audio_hz=1500.0):
    """12 kHz complex IQ centred on the dial (standard orientation, t=0 = even-minute start)
    -> c2 baseband array (mirrored, 375 Hz, 45000 samples). Writes the file if path is given."""
    t = np.arange(len(iq)) / fs
    base = iq * np.exp(-2j * np.pi * center_audio_hz * t)
    q = int(round(fs / C2_RATE))
    d = sps.resample_poly(base, 1, q, window=("kaiser", 8.0))
    d = np.conj(d)                                  # c2 convention is mirrored
    out = np.zeros(C2_N, dtype=np.complex64)
    m = min(C2_N, len(d))
    out[:m] = d[:m]
    if path:
        with open(path, "wb") as f:
            f.write(name.encode()[:14].ljust(14, b"\0"))
            f.write(struct.pack("<i", 2))
            f.write(struct.pack("<d", dial_khz / 1000.0))
            f.write(np.column_stack([out.real, out.imag]).astype("<f4").tobytes())
    return out


def write_wav(path, audio, fs=12000):
    x = np.asarray(audio, dtype=float)
    pk = np.max(np.abs(x)) or 1.0
    pcm = np.int16(np.clip(x / pk * 0.5 * 32767, -32768, 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------- output parsers
def parse_wsprd_line(line):
    t = line.split()
    if len(t) < 6:
        return None
    i = 2 if (re.fullmatch(r"\d{6}", t[0]) and re.fullmatch(r"\d{4}", t[1])) else 1
    try:
        snr = int(float(t[i])); dt = float(t[i + 1]); fmhz = float(t[i + 2]); drift = int(float(t[i + 3]))
    except (ValueError, IndexError):
        return None
    msg = " ".join(t[i + 4:])
    return {"message": msg, "snr": snr, "dt": dt, "freq_mhz": fmhz, "drift": drift}


def parse_jt9_line(line):
    t = line.split()
    if len(t) < 6 or not re.fullmatch(r"\d{4,6}", t[0]):
        return None
    try:
        snr = int(float(t[1])); dt = float(t[2]); hz = float(t[3])
    except ValueError:
        return None
    rest = t[4:]
    if rest and rest[0] in ("~", "@", "#"):
        rest = rest[1:]
    while rest and (re.fullmatch(r"a\d", rest[-1]) or rest[-1] == "?"):
        rest = rest[:-1]
    return {"message": " ".join(rest), "snr": snr, "dt": dt, "hz": hz}


# ------------------------------------------------------------------ wrappers
def run_wsprd(c2_path, wsprd="wsprd", timeout=120, extra_args=()):
    r = subprocess.run([wsprd, *extra_args, os.path.basename(c2_path)], capture_output=True, text=True,
                       timeout=timeout, cwd=os.path.dirname(os.path.abspath(c2_path)))
    return [d for d in (parse_wsprd_line(l) for l in r.stdout.splitlines()) if d]


def run_jt9_ft8(wav_path, jt9="jt9", depth=3, timeout=60):
    r = subprocess.run([jt9, "-8", "-d", str(depth), os.path.basename(wav_path)], capture_output=True,
                       text=True, timeout=timeout, cwd=os.path.dirname(os.path.abspath(wav_path)))
    return [d for d in (parse_jt9_line(l) for l in r.stdout.splitlines()) if d]


def wsprsim_symbols(message, wsprsim="wsprsim"):
    r = subprocess.run([wsprsim, "-c", message], capture_output=True, text=True, timeout=30)
    toks = r.stdout.split("Channel symbols:")[-1].split()
    s = [int(x) for x in toks if x.isdigit()]
    return np.array(s[:162]) if len(s) >= 162 else None


def parse_ft8code(text):
    """79 channel symbols from `ft8code "msg"` output ('Channel symbols (79 tones):' + a line of
    digit groups: Sync | Data | Sync | Data | Sync)."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if "Channel symbols" in l:
            for nxt in lines[i + 1:i + 4]:
                d = re.sub(r"\s+", "", nxt)
                if re.fullmatch(r"[0-7]{79}", d):
                    return np.array([int(c) for c in d])
    return None


def parse_wsprcode(text):
    """162 channel symbols from `wsprcode "msg"` output (the block after 'Channel symbols:')."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith("Channel symbols"):
            digits = []
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    if digits:
                        break
                    continue
                if not re.fullmatch(r"[\s0-3]+", nxt):
                    break
                digits += nxt.split()
            if len(digits) == 162:
                return np.array([int(c) for c in digits])
    return None


def ft8code_symbols(message, ft8code="ft8code"):
    r = subprocess.run([ft8code, message], capture_output=True, text=True, timeout=30)
    return parse_ft8code(r.stdout)


def wsprcode_symbols(message, wsprcode="wsprcode"):
    r = subprocess.run([wsprcode, message], capture_output=True, text=True, timeout=30)
    return parse_wsprcode(r.stdout)


def ft8sim_symbols(message, ft8sim="ft8sim"):
    with tempfile.TemporaryDirectory() as d:       # ft8sim writes wav files into the cwd
        r = subprocess.run([ft8sim, message, "1500.0", "0.0", "0.0", "0.0", "1", "-10"],
                           capture_output=True, text=True, timeout=30, cwd=d)
    lines = r.stdout.splitlines()
    for i, l in enumerate(lines):
        if "Channel symbols" in l:
            for nxt in lines[i + 1:i + 3]:
                s = nxt.strip()
                if re.fullmatch(r"[0-7]{79}", s):
                    return np.array([int(c) for c in s])
    return None


# ------------------------------------------------------------------ references
def _gfsk_pulse(x, bt=2.0):
    c = np.pi * np.sqrt(2.0 / np.log(2.0))
    return 0.5 * (erf(c * bt * (x + 0.5)) - erf(c * bt * (x - 0.5)))


def reference_baseband(symbols, spec, fs=12000.0, n_total=None, gfsk=None):
    """Complex baseband reference (standard orientation), centred on the tone set, starting at
    sample 0 and zero-padded to n_total. FT8 uses WSJT-X's Gaussian (BT=2) frequency shaping;
    WSPR is plain continuous-phase FSK."""
    sym = np.asarray(symbols, dtype=float)
    L = int(round(spec.symbol_s * fs))
    centre = (spec.n_tones - 1) / 2.0
    if gfsk is None:
        gfsk = spec.name == "ft8"
    if gfsk:
        n = len(sym) * L
        imp = np.zeros(n + 3 * L)
        imp[(np.arange(len(sym)) * L + L // 2)] = sym - centre
        x = (np.arange(-int(1.5 * L), int(1.5 * L) + 1)) / L
        freq = sps.fftconvolve(imp, _gfsk_pulse(x))[int(1.5 * L):int(1.5 * L) + n] * spec.tone_spacing_hz
    else:
        freq = np.repeat((sym - centre) * spec.tone_spacing_hz, L)
    ref = np.exp(2j * np.pi * np.cumsum(freq) / fs)
    if n_total is not None:
        out = np.zeros(n_total, dtype=complex)
        out[:min(n_total, len(ref))] = ref[:n_total]
        return out
    return ref


def isolate_iq(iq, center_hz, bw_hz, fs=12000.0):
    """Mix a signal at center_hz to 0 Hz and low-pass to +/-bw/2 (zero phase)."""
    t = np.arange(len(iq)) / fs
    x = iq * np.exp(-2j * np.pi * center_hz * t)
    b, a = sps.butter(4, (bw_hz / 2.0) / (fs / 2.0), btype="low")
    return sps.filtfilt(b, a, x)
