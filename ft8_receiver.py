#!/usr/bin/env python3
"""
UTC-Synchronized FT8 Receive-only Module for weakmon.
Captures audio/IQ waveforms synchronized to UTC 15-second boundaries
and decodes FT8 messages in separate processes to prevent audio buffer overflows.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))

import time
import math
import concurrent.futures
import numpy as np
import ft8
import weakaudio


# Standalone worker function for ProcessPoolExecutor
def decode_waveform_worker(waveform, sample_rate, sample_time, verbose=False):
    """
    Top-level function that runs in a separate OS process to prevent
    GIL / CPU contention with the audio capture stream.
    """
    decoder = ft8.FT8()
    decoder.verbose = verbose
    decoder.cardrate = sample_rate
    decoder.rcardrate = sample_rate

    decoder.process(waveform, sample_time)

    raw_decodes = decoder.get_msgs()
    messages = []

    for dec in raw_decodes:
        msg_info = {
            "message": getattr(dec, "msg", str(dec)),
            "snr": getattr(dec, "snr", None),
            "hz": getattr(dec, "hz", None),
            "dt": getattr(dec, "dt", None),
            "timestamp_utc": sample_time
        }
        messages.append(msg_info)

    return messages


class FT8BlockReceiver:
    """
    Interfaces with weakmon's weakaudio to record UTC-aligned 15-second FT8 windows.
    """
    def __init__(self, card_desc, sample_rate=12000, max_workers=2):
        self.card_desc = card_desc
        self.sample_rate = sample_rate
        self.audio_stream = weakaudio.new(self.card_desc, self.sample_rate)
        # ProcessPoolExecutor isolates decoder CPU execution from audio streaming
        # self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1, 
            max_tasks_per_child=1  # Recycles the worker process after every decode
        )

    def wait_until_next_slot(self, slot_seconds=15.0):
        now = time.time()
        rem = now % slot_seconds
        wait_time = slot_seconds - rem

        if wait_time < 0.05:
            wait_time += slot_seconds

        target_utc_start = now + wait_time

        # Flush any stale audio accumulated while main.py was running analyze()
        if hasattr(self.audio_stream, "flush"):
            self.audio_stream.flush()

        has_iq = hasattr(self.audio_stream, "read_iq")

        while time.time() < target_utc_start:
            self.audio_stream.read()
            if has_iq:
                self.audio_stream.read_iq()
            time.sleep(0.001)

        slot_start_time = math.floor(target_utc_start / slot_seconds) * slot_seconds
        return slot_start_time

    def capture_utc_slot_async(self, slot_seconds=15.0, stall_timeout=30.0):
        slot_start_utc = self.wait_until_next_slot(slot_seconds)

        target_samples = int(self.sample_rate * slot_seconds)
        samples_list = []
        collected_samples = 0

        has_iq = hasattr(self.audio_stream, "read_iq")
        iq_samples_list = []
        iq_collected_samples = 0

        # Read audio stream as fast as possible without sleep inside loop.
        # When the source has a parallel raw-IQ stream (kiwisdr.py), pull
        # both every iteration -- they're two independently-drained
        # buffers fed from the same underlying samples (see
        # KiwiSDR._drain_loop() in kiwisdr.py), so this keeps them
        # covering the same time span without one blocking the other.
        #
        # This loop used to spin here forever, silently, if the source
        # connects but never actually delivers samples (auth issue,
        # server never sends the expected setup messages, proxy quirk,
        # etc.) -- collected_samples just never moves and there's no
        # sign anything is wrong. stall_timeout turns that into a loud,
        # specific failure instead: main.py already wraps this call in
        # try/except, so this surfaces as a clear one-line error rather
        # than a program that looks frozen.
        loop_start = time.time()
        last_progress_time = loop_start
        while collected_samples < target_samples:
            buf, tm = self.audio_stream.read()
            if len(buf) > 0:
                samples_list.append(buf)
                collected_samples += len(buf)
                last_progress_time = time.time()

            if has_iq:
                iq_buf, iq_tm = self.audio_stream.read_iq()
                if len(iq_buf) > 0:
                    iq_samples_list.append(iq_buf)
                    iq_collected_samples += len(iq_buf)
                    last_progress_time = time.time()

            if time.time() - last_progress_time > stall_timeout:
                if collected_samples == 0:
                    raise RuntimeError(
                        "No audio samples received in %.0fs (zero total) -- "
                        "the source never delivered any data. For a KiwiSDR, "
                        "this usually means the connection never actually "
                        "started streaming (auth/setup issue, or the server "
                        "never sent 'sample_rate' so _setup_rx_params() was "
                        "never called to send SET mod=iq). Turn on Python "
                        "logging (logging.basicConfig(level=logging.DEBUG)) "
                        "before running this and check what kiwiclient logs "
                        "during connect." % stall_timeout
                    )
                else:
                    raise RuntimeError(
                        "Audio stream stalled: %d/%d samples collected, then "
                        "no progress for %.0fs. The connection was working "
                        "and then stopped delivering data mid-slot."
                        % (collected_samples, target_samples, stall_timeout)
                    )

        if len(samples_list) > 0:
            waveform = np.concatenate(samples_list)[:target_samples]
        else:
            waveform = np.array([], dtype=np.float32)

        iq_waveform = None
        if has_iq:
            if len(iq_samples_list) > 0:
                iq_waveform = np.concatenate(iq_samples_list)[:target_samples]
            else:
                iq_waveform = np.array([], dtype=np.complex64)
            # capture_utc_slot_async() returns as soon as the real-audio
            # side hits target_samples; the IQ side, being drained on the
            # same loop, should generally land within a block or two of
            # the same length. Pad/trim defensively so analyze()'s
            # len(iq_waveform) != len(waveform) check is a rare no-op
            # rather than the normal case.
            if len(iq_waveform) < target_samples:
                iq_waveform = np.pad(iq_waveform, (0, target_samples - len(iq_waveform)))
            else:
                iq_waveform = iq_waveform[:target_samples]

        # Submit decode task to isolated worker process. Only the real
        # audio goes to the decoder -- iq_waveform is for analyze()'s
        # true-I/Q path in main.py, after a message has been decoded.
        future = self.executor.submit(
            decode_waveform_worker, waveform, self.sample_rate, slot_start_utc
        )

        return waveform, iq_waveform, future, slot_start_utc
