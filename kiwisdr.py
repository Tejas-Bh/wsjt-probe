#
# KiwiSDR IQ source for weakmon, following the same shape as
# sdrip.py / sdriq.py / eb200.py (the RFSpace/AirSpy-style backends
# already used by weakaudio.py).
#
# This talks to a KiwiSDR over its websocket API (via kiwiclient,
# https://github.com/jks-prv/kiwiclient) and pulls down raw I/Q,
# tagged with the Kiwi's GPS timestamp when it has one.
#
# NOTE ON PROVENANCE: I (Claude) do not have network access in the
# sandbox I wrote this in, so this still hasn't run against a live
# KiwiSDR from my end -- one real bug already surfaced this way (a
# missing _writer_message() override, fixed below; see the comment on
# it for what happened and why). If you hit another AttributeError on
# self._options.<something>, kiwiclient's version on your machine
# references an option this file doesn't set -- just add it to
# KiwiOptions with a sane default and it'll keep going. Send me the
# traceback either way and I'll fix it against what actually happened
# rather than guessing again.
#
# Expected desc format from weakaudio.new(desc, rate):
#   desc = [ "kiwi", "host:port:dial_khz" ]
# e.g. ["kiwi", "192.168.1.50:8073:14074.0"]
# dial_khz is the *dial* (tuned) frequency in kHz -- same number
# you'd type into the Kiwi's web UI, chosen the same way you'd choose
# it for a sound-card FT8 setup (e.g. 1 kHz below the FT8 calling
# frequency so the traffic lands at 300-2900 Hz of "audio").
#

import sys
import time
import threading
import numpy
import weakutil

sys.path.insert(0, "kiwiclient")  # adjust to wherever you check out jks-prv/kiwiclient
from kiwi import KiwiSDRStream, KiwiWorker


class KiwiOptions(object):
    """
    Minimal stand-in for the optparse.Values object that kiwirecorder.py's
    CLI normally builds. Every attribute here was found by grepping
    kiwiclient's kiwi/client.py and kiwi/worker.py for self._options.X --
    NOT copied from kiwirecorder's --help (which I didn't have access to
    in full). It deliberately skips options that only matter for
    kiwirecorder's own squelch/scan/wav-writing/resampling machinery,
    since this class avoids that code path entirely (see KiwiIQStream
    below).
    """
    def __init__(self, host, port, freq_khz, user="weakmon-ft8", password=""):
        self.server_host = host
        self.server_port = port
        self.frequency = freq_khz     # dial freq, kHz
        self.user = user
        self.password = password
        self.tlimit_password = ""
        self.wideband = False
        self.ws_timestamp = int(time.time())
        self.socket_timeout = 10
        self.freq_pbc = False
        self.freq_offset = 0
        self.nolocal = False
        self.admin = False
        self.idx = 0
        self.wf_cal = None
        self.ADC_OV = False
        self.S_meter = -1
        self.sound = True
        self.tstamp = False
        self.sdt = 0
        self.stats = False
        self.camp_allow_1ch = False
        self.netcat = False
        self.resample = 0
        self.rev_bin = False
        self.tlimit = None
        # KiwiWorker-specific
        self.rigctl_enabled = False
        self.connect_retries = 0      # 0 == retry forever
        self.busy_retries = 5
        self.connect_timeout = 5
        self.busy_timeout = 15
        self.is_kiwi_tdoa = False
        self.no_api = False


class KiwiIQStream(KiwiSDRStream):
    """
    Thin KiwiSDRStream subclass. Deliberately skips kiwirecorder.py's
    KiwiSoundRecorder / _init_common / _setup_rx_snd_params -- those pull
    in squelch, scan-yaml, resampler-selection and wav-file writing that
    this project doesn't need, and each one references more self._options
    attributes. This does only what's needed to pull raw IQ:
      - ask for mode='iq' with the full +/-5kHz passband
      - hand every block of samples (plus its GPS tag, if any) to a queue
    """
    def __init__(self, options, out_queue):
        super(KiwiIQStream, self).__init__()
        self._options = options
        self._type = "SND"
        self._freq = options.frequency
        self._freq_offset = options.freq_offset
        self._out_queue = out_queue
        self.sample_rate = None  # filled in once the Kiwi tells us

    def _setup_rx_params(self):
        self.set_name(self._options.user)
        # +/-5000 Hz is the Kiwi's full IQ passband (its "iq" default,
        # see _default_passbands in kiwiclient's kiwi/client.py).
        self.set_mod("iq", -5000, 5000, self._freq)
        self.set_agc(on=True)

    def _on_sample_rate_change(self):
        self.sample_rate = self._sample_rate

    def _process_iq_samples(self, seq, samples, rssi, gps, fmt):
        # samples: numpy complex64 array, real/imag on the same int16-ish
        # scale as ordinary audio (see kiwiclient's _process_aud()).
        # gps: dict with last_gps_solution/gpssec/gpsnsec, or None.
        # last_gps_solution == 0 means "has a recent fix"; 254/255 mean
        # "no recent GPS solution" (see GNSSPerformance in kiwirecorder.py).
        self._out_queue.put((seq, numpy.array(samples, dtype=numpy.complex64), rssi, gps))

    def _writer_message(self):
        # THE ACTUAL BUG from your traceback: KiwiSDRStream's base
        # _writer_message() is just `pass` (implicitly returns None). It's
        # meant to be overridden by whatever recorder subclass has
        # outgoing data to send -- I never overrode it, so the writer
        # thread's `msg, oob, closed = self._writer_message()` in
        # kiwi/client.py's run() tried to unpack None and blew up. That's
        # a bug in what I gave you, not in kiwiclient.
        #
        # We don't need to send anything here: keepalives are already
        # sent automatically by the *reader* thread (KiwiSDRStream.
        # _process_message() calls _set_keepalive() once a second
        # whenever SND packets arrive, which is the entire time we're
        # receiving IQ). So (None, None, False) -- no msg, no oob,
        # not closed -- is the correct no-op reply.
        #
        # The sleep is required, not optional: kiwi/worker.py's writer
        # loop is `while self._do_run(): self._recorder.run()` with zero
        # pacing of its own. Returning instantly here would spin that
        # thread at ~100% of one CPU core forever. 0.2s keeps it
        # responsive to shutdown without burning CPU.
        time.sleep(0.2)
        return (None, None, False)


class KiwiSDR(object):
    """
    weakaudio-compatible IQ source: give it .read() and it hands back
    [ real_audio_samples, unix_time_of_last_sample ], exactly like
    weakaudio.SDRIP / EB200 / SDRplay. Register it in weakaudio.new()
    the same way those are registered, e.g.:

        if desc[0] == "kiwi":
            return KiwiSDR(desc[1], rate)

    Internally: connect over the Kiwi's websocket, pull raw IQ,
    demodulate to real single-sideband "audio" with weakutil.iq2usb()
    (the same phasing-method demod weakaudio.SDRIP already uses for
    the SDR-IP's IQ mode), then resample to the requested `rate`.
    """
    def __init__(self, desc1, rate):
        parts = desc1.split(":")
        if len(parts) != 3:
            sys.stderr.write(
                'weakaudio: kiwi descriptor must be "host:port:dial_khz", got "%s"\n' % desc1
            )
            sys.exit(1)
        host, port, freq_khz = parts[0], int(parts[1]), float(parts[2])

        self.rate = rate if rate is not None else 12000
        self.host = host
        self.port = port
        self.freq_khz = freq_khz

        self._options = KiwiOptions(host, port, freq_khz)
        self._sample_queue = __import__("queue").Queue()

        self._recorder = KiwiIQStream(self._options, self._sample_queue)

        self.resampler = None      # for the real-audio (USB) path, read()
        self.resampler_iq = None   # for the raw-IQ path, read_iq()

        # Two independent buffers fed by the same _drain_loop(), so
        # read() and read_iq() can each be drained on their own schedule
        # without one starving the other -- see capture_utc_slot_async()
        # in ft8_receiver.py, which now calls both every iteration.
        self.bufbuf = []
        self.bufbuf_iq = []
        self.cardcount = 0
        self.cardcount_iq = 0
        self.starttime = None
        self.starttime_iq = None
        self.cardlock = threading.Lock()

        # last GPS tag we've seen, exposed for callers that want it
        # (e.g. to sanity-check timing); NOT currently consumed by
        # ft8_receiver.py's slot-boundary logic -- see the note in the
        # writeup this shipped with about that being a real limitation.
        self.last_gps = None

        self._run_event = threading.Event()
        self._run_event.set()

        # THE ACTUAL BUG from your last run: KiwiWorker.__init__() does
        #   self._recorder._reader = self._reader
        # -- it stamps the role flag onto the *recorder object*, not the
        # thread. I originally pointed both the reader and writer
        # KiwiWorker at the same KiwiIQStream instance. The writer's
        # __init__ runs second and overwrites that shared instance's
        # _reader to False, so by the time either thread calls
        # recorder.run(), KiwiSDRStream.run() sees _reader=False and
        # takes the writer branch every time -- including for the
        # "reader" thread, which is exactly why nothing was ever being
        # read off the socket after auth. Fix: two separate recorder
        # instances. The reader one owns the real connection, auth, and
        # all IQ-processing state. The writer one is a mostly-idle shell
        # (see _writer_message() above) that only needs a valid _stream
        # to exist on -- it gets the real, shared connection object via
        # the same q_stream hand-off kiwiclient already does internally
        # (worker.py: writer does `self._recorder._stream = q_stream.get()`
        # after the reader connects), so there's still only one actual
        # websocket connection underneath; only the Python-side book-
        # keeping is now correctly separated between two objects instead
        # of racing on one.
        self._writer_recorder = KiwiIQStream(self._options, self._sample_queue)

        self._reader_thread = KiwiWorker(
            args=(self._recorder, self._options, True, False, self._run_event, None)
        )
        self._writer_thread = KiwiWorker(
            args=(self._writer_recorder, self._options, False, False, self._run_event, None)
        )
        self._reader_thread.daemon = True
        self._writer_thread.daemon = True
        self._reader_thread.start()
        self._writer_thread.start()

        self._drain_thread = threading.Thread(target=self._drain_loop)
        self._drain_thread.daemon = True
        self._drain_thread.start()


    def flush(self):
        """Flushes all accumulated/backlogged audio in queues and buffers."""
        self.cardlock.acquire()
        self.bufbuf = []
        self.bufbuf_iq = []
        self.cardcount = 0
        self.cardcount_iq = 0
        self.starttime = None
        self.starttime_iq = None
        self.cardlock.release()

        # Drain the underlying thread queue
        while not self._sample_queue.empty():
            try:
                self._sample_queue.get_nowait()
            except Exception:
                break

    def _drain_loop(self):
        # move samples from the KiwiIQStream's queue into our own
        # buffers -- one feeding read() (the SDRIP-style real-audio
        # interface), one feeding read_iq() (raw complex passthrough).
        # Both get every sample block; each is only cleared by its own
        # read*() call, so calling one doesn't starve the other.
        while True:
            try:
                seq, samples, rssi, gps = self._sample_queue.get(timeout=1.0)
            except Exception:
                continue

            if gps is not None:
                self.last_gps = gps

            self.cardlock.acquire()
            self.bufbuf.append(samples)
            self.cardcount += len(samples)
            if self.starttime is None:
                self.starttime = time.time() - len(samples) / float(self._effective_rate())

            self.bufbuf_iq.append(samples)
            self.cardcount_iq += len(samples)
            if self.starttime_iq is None:
                self.starttime_iq = time.time() - len(samples) / float(self._effective_rate())
            self.cardlock.release()

    def _effective_rate(self):
        return self._recorder.sample_rate or 12000.0

    def _int_rate(self, hz):
        # weakutil.Resampler slices buffers with `self.from_rate // 2`,
        # which returns a float (and then fails on `buf[i:i+float]`) if
        # from_rate isn't already an int. Kiwi reports sample_rate as a
        # float (client.py: self._sample_rate = float(value)), so this
        # cast is required, not cosmetic -- confirmed by testing
        # Resampler(12000.001, 12000) directly, which raises TypeError.
        return int(round(hz))

    # returns [ buf, tm ], tm = unix seconds of the last sample --
    # same contract as weakaudio.SDRIP.read(). buf is real single-
    # sideband "audio", for the existing FT8 decode path.
    def read(self):
        buf1, tm = self.raw_read()
        buf2 = self.postprocess(buf1)
        return [buf2, tm]

    def raw_read(self):
        self.cardlock.acquire()
        bufbuf = self.bufbuf
        cardcount = self.cardcount
        self.bufbuf = []
        self.cardlock.release()

        if self.starttime is not None:
            buf_time = self.starttime + cardcount / float(self._effective_rate())
        else:
            buf_time = time.time()

        if len(bufbuf) == 0:
            return [numpy.array([], dtype=numpy.complex64), buf_time]

        buf1 = numpy.concatenate(bufbuf)
        return [buf1, buf_time]


    def postprocess(self, buf1):
        if len(buf1) == 0:
            return numpy.array([])

        # complex IQ -> real USB "audio", via weakmon's own phasing-method
        # demod (the same function weakaudio.SDRIP uses for its IQ mode).
        usb = weakutil.iq2usb(buf1)

        src_rate = self._int_rate(self._effective_rate())
        dst_rate = self._int_rate(self.rate)
        if src_rate == dst_rate:
            return usb

        if self.resampler is None or self.resampler.from_rate != src_rate:
            self.resampler = weakutil.Resampler(src_rate, dst_rate)
        return self.resampler.resample(usb)

    # ---- raw-IQ passthrough, for analyze()'s true-I/Q path ----
    #
    # returns [ complex_buf, tm ], same tm contract as read(). complex_buf
    # is genuine I/Q, resampled to `rate` but NOT demodulated -- no
    # iq2usb(), no Hilbert reconstruction later. Frequencies in this
    # buffer are relative to the Kiwi's tuned (dial) frequency, on the
    # exact same Hz scale as the decoder's reported "hz" for a message
    # decoded from read()'s audio: iq2usb() (the phasing method) doesn't
    # shift frequencies, it just removes the negative-frequency image, so
    # a tone the decoder calls "1521 Hz" sits at +1521 Hz in this complex
    # buffer too. That's what makes it safe to reuse hz_val directly in
    # analyze.py's isolate_signal_iq() without any extra bookkeeping.
    def read_iq(self):
        buf1, tm = self.raw_read_iq()
        buf2 = self.postprocess_iq(buf1)
        return [buf2, tm]

    def raw_read_iq(self):
        self.cardlock.acquire()
        bufbuf = self.bufbuf_iq
        cardcount = self.cardcount_iq
        
        # Reset state here too
        self.bufbuf_iq = []
        self.cardcount_iq = 0
        self.starttime_iq = None
        self.cardlock.release()

        if len(bufbuf) == 0:
            return [numpy.array([], dtype=numpy.complex64), time.time()]

        buf1 = numpy.concatenate(bufbuf)
        buf_time = time.time()
        return [buf1, buf_time]

    def postprocess_iq(self, buf1):
        if len(buf1) == 0:
            return numpy.array([], dtype=numpy.complex64)

        src_rate = self._int_rate(self._effective_rate())
        dst_rate = self._int_rate(self.rate)
        if src_rate == dst_rate:
            return buf1

        # weakutil.Resampler's default path (scipy.signal.resample, FFT-
        # based) handles complex input fine -- tested directly. Its
        # "interp" fallback (numpy.interp) does NOT support complex data;
        # if you've set weakutil.which_resampler = "interp" for speed
        # elsewhere, this will break and needs separate real/imag calls.
        if self.resampler_iq is None or self.resampler_iq.from_rate != src_rate:
            self.resampler_iq = weakutil.Resampler(src_rate, dst_rate)
        return self.resampler_iq.resample(buf1)

# del KiwiSDR().bufbuf
# del KiwiSDR().bufbuf_iq
