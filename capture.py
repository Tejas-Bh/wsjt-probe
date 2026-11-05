"""
capture.py -- UTC-aligned slot capture for any mode (slot length from the mode spec).

Same job as ft8_receiver.FT8BlockReceiver, minus the built-in weakmon decode: decoding now
happens in the analysis workers (slot_pipeline.process_slot) so the capture loop only records.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))
import weakaudio  # noqa: E402


class SlotReceiver:
    def __init__(self, card_desc, sample_rate=12000, slot_s=15.0, stall_timeout=30.0):
        self.sample_rate = sample_rate
        self.slot_s = float(slot_s)
        self.stall_timeout = stall_timeout
        self.audio_stream = weakaudio.new(card_desc, sample_rate)
        self.has_iq = hasattr(self.audio_stream, "read_iq")

    def start_check(self, seconds=10.0):
        """Read the source for a few seconds and report what arrives (catches a dead connection
        immediately instead of after a whole slot). Returns once data has flowed for 'seconds'."""
        t0 = time.time()
        n_a = n_i = 0
        while time.time() - t0 < seconds:
            b, _ = self.audio_stream.read()
            n_a += len(b)
            if self.has_iq:
                ib, _ = self.audio_stream.read_iq()
                n_i += len(ib)
            time.sleep(0.01)
        el = time.time() - t0
        return {"audio_samples_per_s": n_a / el, "iq_samples_per_s": n_i / el, "has_iq": self.has_iq}

    def _wait_until_next_slot(self):
        now = time.time()
        wait = self.slot_s - (now % self.slot_s)
        if wait < 0.05:
            wait += self.slot_s
        target = now + wait
        print("[%s] waiting %.0f s for the next %.0f s slot boundary (%s UTC)" % (
            time.strftime("%H:%M:%S", time.gmtime(now)), wait, self.slot_s,
            time.strftime("%H:%M:%S", time.gmtime(target))), flush=True)
        if hasattr(self.audio_stream, "flush"):
            self.audio_stream.flush()
        while time.time() < target:          # keep draining so buffers do not overflow
            self.audio_stream.read()
            if self.has_iq:
                self.audio_stream.read_iq()
            time.sleep(0.001)
        return math.floor(target / self.slot_s) * self.slot_s

    def capture_slot(self):
        """Block until the next slot boundary, record one slot. Returns (audio, iq, slot_start_utc);
        iq is None if the source has no raw IQ stream."""
        slot_start = self._wait_until_next_slot()
        target = int(self.sample_rate * self.slot_s)
        a_list, i_list, n_a = [], [], 0
        last_progress = time.time()
        last_report = time.time()
        print("[%s] recording slot %s (%.0f s)" % (
            time.strftime("%H:%M:%S", time.gmtime()), time.strftime("%H:%M:%S", time.gmtime(slot_start)),
            self.slot_s), flush=True)
        while n_a < target:
            if self.slot_s > 20 and time.time() - last_report > 30:
                print("    %d/%d samples (%.0f%%)" % (n_a, target, 100.0 * n_a / target), flush=True)
                last_report = time.time()
            buf, _ = self.audio_stream.read()
            if len(buf):
                a_list.append(buf)
                n_a += len(buf)
                last_progress = time.time()
            if self.has_iq:
                ib, _ = self.audio_stream.read_iq()
                if len(ib):
                    i_list.append(ib)
                    last_progress = time.time()
            if time.time() - last_progress > self.stall_timeout:
                raise RuntimeError("no samples for %.0f s (%d/%d collected): source stalled"
                                   % (self.stall_timeout, n_a, target))
        audio = np.concatenate(a_list)[:target]
        iq = None
        if self.has_iq:
            iq = np.concatenate(i_list)[:target] if i_list else np.zeros(0, dtype=np.complex64)
            if len(iq) < target:
                iq = np.pad(iq, (0, target - len(iq)))
        return audio, iq, slot_start
