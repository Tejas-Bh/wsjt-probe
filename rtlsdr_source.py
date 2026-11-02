#
# RTL-SDR IQ source for weakmon / wsjt-probe -- sibling of kiwisdr.py with the
# same interface (read(), read_iq(), flush()), so ft8_receiver.py and
# main.py do not change.
#
# UNTESTED AGAINST REAL HARDWARE (written without an RTL-SDR attached). The
# decimation chain was checked offline with synthetic tones; the pyrtlsdr
# calls follow its documented API. Send me the traceback if something differs.
#
# Register in weakaudio.new():
#     if desc[0] == "rtlsdr":
#         import rtlsdr_source
#         return rtlsdr_source.RTLSDRSource(desc[1], rate)
#
# NOTE: this file must NOT be called rtlsdr.py (it would shadow pyrtlsdr).
#
# desc format:  [ "rtlsdr", "index:dial_khz:mode[:gain[:lo_offset_khz]]" ]
#   mode = "direct"  HF direct-sampling (RTL-SDR Blog V3 HF input, or a
#                    dongle with the Q-branch mod). R820T tuner is bypassed.
#   mode = "tuner"   normal tuner path; use with an upconverter, e.g.
#                    lo_offset_khz = 125000 for a Ham It Up (RF = dial + LO).
#   gain = "auto" or dB (tuner mode only; ignored in direct mode)
# examples:
#   ["rtlsdr", "0:14074.0:direct"]
#   ["rtlsdr", "0:14074.0:tuner:30:125000"]
#
# Frequency convention matches kiwisdr.py: the complex buffer is centred on
# the dial frequency, and a tone the decoder reports at "hz" sits at +hz.
#
# Receiver clock: an RTL-SDR crystal is off by tens of ppm (TCXO models ~1
# ppm). At 14 MHz, 1 ppm = 14 Hz. Absolute Doppler SHIFT is therefore not
# trustworthy without wwv_calibrate.py-style correction, and short-term
# drift during a 15 s slot shows up as extra apparent Doppler spread.
#

import sys
import time
import queue
import threading

import numpy
import scipy.signal as signal
import weakutil

try:
    from rtlsdr import RtlSdr  # pip install pyrtlsdr (needs librtlsdr)
except ImportError:
    RtlSdr = None


class _Decimator(object):
    """Stateful FIR low-pass + decimate-by-q that is continuous across blocks."""

    def __init__(self, q, numtaps, cutoff_hz, fs_in):
        self.q = q
        self.taps = signal.firwin(numtaps, cutoff_hz, window=("kaiser", 8.0), fs=fs_in)
        self.zi = numpy.zeros(numtaps - 1, dtype=numpy.complex128)
        self.off = 0  # index of first kept sample in the next block

    def process(self, x):
        y, self.zi = signal.lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.off::self.q]
        self.off = (self.off - len(x)) % self.q
        return out


class RTLSDRSource(object):
    RAW_RATE = 240000       # valid RTL2832 rate (225001-300000 band), /20 = 12 kHz
    BLOCK = 65536           # samples per async callback (multiple of 512)
    SCALE = 32768.0         # pyrtlsdr gives +/-1 floats; match Kiwi's int16-ish scale

    def __init__(self, desc1, rate):
        if RtlSdr is None:
            sys.stderr.write("rtlsdr_source: pyrtlsdr not installed (pip install pyrtlsdr)\n")
            sys.exit(1)

        parts = desc1.split(":")
        if len(parts) < 3 or len(parts) > 5:
            sys.stderr.write(
                'rtlsdr: descriptor must be "index:dial_khz:mode[:gain[:lo_offset_khz]]", got "%s"\n' % desc1
            )
            sys.exit(1)

        self.rate = int(rate) if rate is not None else 12000
        if self.rate != 12000:
            sys.stderr.write("rtlsdr_source: only a 12000 Hz output rate is implemented\n")
            sys.exit(1)

        self.dev_index = int(parts[0])
        self.freq_khz = float(parts[1])
        self.mode = parts[2].lower()
        gain = parts[3] if len(parts) > 3 else "auto"
        lo_khz = float(parts[4]) if len(parts) > 4 else 0.0
        if self.mode not in ("direct", "tuner"):
            sys.stderr.write('rtlsdr: mode must be "direct" or "tuner"\n')
            sys.exit(1)

        self.sdr = RtlSdr(device_index=self.dev_index)
        self.sdr.sample_rate = self.RAW_RATE
        if abs(self.sdr.sample_rate - self.RAW_RATE) > 1:
            sys.stderr.write("rtlsdr: device gave sample rate %s, wanted %s\n"
                             % (self.sdr.sample_rate, self.RAW_RATE))

        if self.mode == "direct":
            self.sdr.set_direct_sampling(2)   # 2 = Q branch (RTL-SDR Blog V3 HF port)
            rf_hz = self.freq_khz * 1e3
        else:
            self.sdr.gain = "auto" if gain == "auto" else float(gain)
            rf_hz = (self.freq_khz + lo_khz) * 1e3
        self.sdr.center_freq = rf_hz

        # 240k -> 48k (/5) -> 12k (/4). Passband +/-5 kHz (same as Kiwi IQ),
        # stopbands placed so anything that aliases lands outside +/-5 kHz.
        self._d1 = _Decimator(5, 63, 24000.0, self.RAW_RATE)
        self._d2 = _Decimator(4, 127, 6000.0, self.RAW_RATE / 5.0)

        self.cardlock = threading.Lock()
        self.bufbuf = []        # real USB audio, for read()
        self.bufbuf_iq = []     # complex baseband, for read_iq()
        self.raw_count = 0
        self.t_start = time.time()
        self._raw_q = queue.Queue()

        self._rx = threading.Thread(target=self._rx_loop)
        self._rx.daemon = True
        self._rx.start()
        self._drain = threading.Thread(target=self._drain_loop)
        self._drain.daemon = True
        self._drain.start()

    # ---- threads ----
    def _rx_loop(self):
        def cb(samples, ctx):
            self.raw_count += len(samples)
            self._raw_q.put(numpy.array(samples, dtype=numpy.complex64))
        self.sdr.read_samples_async(cb, self.BLOCK)

    def _drain_loop(self):
        while True:
            try:
                raw = self._raw_q.get(timeout=1.0)
            except queue.Empty:
                continue
            y = self._d2.process(self._d1.process(raw))
            if len(y) == 0:
                continue
            iq = (y * self.SCALE).astype(numpy.complex64)
            usb = weakutil.iq2usb(iq)   # same demod the Kiwi path uses
            with self.cardlock:
                self.bufbuf.append(usb)
                self.bufbuf_iq.append(iq)

    # ---- weakaudio-style interface ----
    def flush(self):
        with self.cardlock:
            self.bufbuf = []
            self.bufbuf_iq = []
        while True:
            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                break

    def read(self):
        with self.cardlock:
            bufs, self.bufbuf = self.bufbuf, []
        if len(bufs) == 0:
            return [numpy.array([]), time.time()]
        return [numpy.concatenate(bufs), time.time()]

    def read_iq(self):
        with self.cardlock:
            bufs, self.bufbuf_iq = self.bufbuf_iq, []
        if len(bufs) == 0:
            return [numpy.array([], dtype=numpy.complex64), time.time()]
        return [numpy.concatenate(bufs), time.time()]

    def stats(self):
        """Raw samples received vs. wall clock. Ratio well below 1.0 means USB drops."""
        el = time.time() - self.t_start
        return {"raw_samples": self.raw_count, "elapsed_s": el,
                "effective_rate": self.raw_count / el if el > 0 else 0.0}

    def close(self):
        try:
            self.sdr.cancel_read_async()
        finally:
            self.sdr.close()


if __name__ == "__main__":
    # Smoke test: python rtlsdr_source.py 0:14074.0:direct [seconds]
    desc = sys.argv[1] if len(sys.argv) > 1 else "0:14074.0:direct"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    src = RTLSDRSource(desc, 12000)
    time.sleep(2.0)
    src.flush()
    chunks = []
    t0 = time.time()
    while time.time() - t0 < secs:
        b, _ = src.read_iq()
        if len(b):
            chunks.append(b)
        time.sleep(0.05)
    x = numpy.concatenate(chunks)
    print("got %d samples (%.1f s at 12 kHz), stats: %s" % (len(x), len(x) / 12000.0, src.stats()))
    f, p = signal.welch(x, fs=12000, nperseg=4096, return_onesided=False)
    f, p = numpy.fft.fftshift(f), numpy.fft.fftshift(p)
    floor = numpy.median(p)
    print("median floor = 0 dB; band power every 500 Hz (dB re floor):")
    for lo in range(-5000, 5000, 500):
        m = (f >= lo) & (f < lo + 500)
        print("%6d..%6d Hz  %6.1f" % (lo, lo + 500, 10 * numpy.log10(p[m].mean() / floor)))
    print("FT8 traffic should show up as raised power in 0..3000 Hz.")
    src.close()
