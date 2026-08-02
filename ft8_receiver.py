#!/usr/bin/env python3
"""
UTC-Synchronized FT8 Receive-only Module for weakmon.
Captures audio/IQ waveforms synchronized to UTC 15-second boundaries
and decodes FT8 messages into memory.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))


import time
import math
import numpy as np
import ft8
import weakaudio


def wait_for_next_utc_slot(slot_seconds=15.0):
    """
    Blocks until the start of the next UTC time slot (00, 15, 30, 45 seconds past minute).
    Returns the UTC timestamp of the start of the slot.
    """
    now = time.time()
    # Calculate seconds remaining until the next 15-second boundary
    rem = now % slot_seconds
    wait_time = slot_seconds - rem
    
    # Avoid zero sleep if called right on the boundary
    if wait_time < 0.05:
        wait_time += slot_seconds

    time.sleep(wait_time)
    
    # Target starting timestamp aligned to the slot
    slot_start_time = math.floor(time.time() / slot_seconds) * slot_seconds
    return slot_start_time


class FT8Decoder:
    """
    Decodes FT8 messages directly from an in-memory PCM or I/Q audio waveform array.
    """
    def __init__(self, sample_rate=12000, verbose=False):
        self.sample_rate = sample_rate
        self.verbose = verbose

    def decode_waveform(self, waveform, sample_time=None):
        """
        Decodes a numpy array representing the audio/IQ waveform for a 15-second FT8 slot.
        """
        if sample_time is None:
            sample_time = time.time()

        # Instantiate FT8 decoder instance from ft8.py
        decoder = ft8.FT8()
        decoder.verbose = self.verbose

        # Set expected sample rate attributes for direct array processing
        decoder.cardrate = self.sample_rate
        decoder.rcardrate = self.sample_rate

        # Process waveform using ft8 engine
        decoder.process(waveform, sample_time)

        # Retrieve decoded messages
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
    Interfaces with weakmon's weakaudio to record a UTC-aligned 15-second FT8 window.
    """
    def __init__(self, card_desc, sample_rate=12000):
        """
        :param card_desc: Card identifier list for weakaudio.new(), 
                          e.g., ["0", "0"] for audio card 0 ch 0, or ["sdrip", "192.168.1.100"]
        :param sample_rate: Audio sampling rate (12000 Hz for FT8)
        """
        self.card_desc = card_desc
        self.sample_rate = sample_rate
        self.decoder = FT8Decoder(sample_rate=sample_rate)

    def capture_utc_slot(self, slot_seconds=15.0):
        """
        Waits for the next UTC boundary (:00, :15, :30, :45), records for 15 seconds,
        and decodes the waveform.

        :return: Tuple (waveform, messages, slot_start_utc)
                 - waveform: numpy ndarray of recorded raw samples
                 - messages: list of decoded message dicts
                 - slot_start_utc: UTC UNIX epoch timestamp of slot start
        """
        # 1. Wait until top of next UTC slot
        slot_start_utc = wait_for_next_utc_slot(slot_seconds)

        # 2. Open weakmon audio stream
        audio_stream = weakaudio.new(self.card_desc, self.sample_rate)
        
        samples_list = []
        end_time = slot_start_utc + slot_seconds

        # 3. Read stream until end of 15-second window
        while time.time() < end_time:
            buf, tm = audio_stream.read()
            if len(buf) > 0:
                samples_list.append(buf)
            else:
                time.sleep(0.02)

        # Close stream resources if applicable
        if hasattr(audio_stream, "close"):
            audio_stream.close()

        # Concatenate waveform buffers
        if len(samples_list) > 0:
            waveform = np.concatenate(samples_list)
        else:
            waveform = np.array([], dtype=np.float32)

        # 4. Decode in-memory waveform
        messages = self.decoder.decode_waveform(waveform, sample_time=slot_start_utc)

        return waveform, messages, slot_start_utc
