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
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def wait_until_next_slot(self, slot_seconds=15.0):
        now = time.time()
        rem = now % slot_seconds
        wait_time = slot_seconds - rem

        if wait_time < 0.05:
            wait_time += slot_seconds

        target_utc_start = now + wait_time

        # Drain audio continuously while waiting for boundary
        while time.time() < target_utc_start:
            self.audio_stream.read()
            time.sleep(0.001)

        slot_start_time = math.floor(target_utc_start / slot_seconds) * slot_seconds
        return slot_start_time

    def capture_utc_slot_async(self, slot_seconds=15.0):
        slot_start_utc = self.wait_until_next_slot(slot_seconds)

        target_samples = int(self.sample_rate * slot_seconds)
        samples_list = []
        collected_samples = 0

        # Read audio stream as fast as possible without sleep inside loop
        while collected_samples < target_samples:
            buf, tm = self.audio_stream.read()
            if len(buf) > 0:
                samples_list.append(buf)
                collected_samples += len(buf)

        if len(samples_list) > 0:
            waveform = np.concatenate(samples_list)[:target_samples]
        else:
            waveform = np.array([], dtype=np.float32)

        # Submit decode task to isolated worker process
        future = self.executor.submit(
            decode_waveform_worker, waveform, self.sample_rate, slot_start_utc
        )

        return waveform, future, slot_start_utc