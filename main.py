#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

import time
import os
import json
import numpy as np  # Imported to handle NumPy type conversions
from ft8_receiver import FT8BlockReceiver
from analyze import analyze

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to automatically convert NumPy types to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from concurrent.futures import ProcessPoolExecutor
from slot_worker import analyze_slot
MAX_MSGS, MAX_BACKLOG = 8, 3


def main():
    # card_desc = ["4", "0"]  # local sound card: card 4, channel 0
    #
    # dial_khz = 14074.0, used as given. One thing worth watching on the
    # Kiwi's waterfall the first time you run this: "iq" mode is a true
    # zero-IF tap centered exactly on dial_khz, and zero-IF receivers can
    # have a DC/LO-leakage spike sitting right at 0 Hz baseband. If your
    # FT8 traffic (audio ~200-2900 Hz per weakmon's real_min_hz/max_hz)
    # is landing close to that, you'd see it as a birdie in the CAF too.
    # I haven't seen your waterfall so I can't tell you whether this
    # actually bites on a Kiwi in practice -- if it does, retuning a
    # little (e.g. dial_khz = 14073.0, everything else unchanged since
    # hz_val is relative to dial freq either way) is the standard fix.
    card_desc = ["kiwi", "marco5g.zapto.org:8073:7074.0"] # GERMANY, WORKS WELL @ 7-9PM GMT
    # card_desc = ["rtlsdr", "0:28074.0:direct"]
    # card_desc = ["kiwi", "azwhtmtns.proxy.kiwisdr.com:8073:14074.0"]
    # card_desc = ["kiwi", "22477.proxy.kiwisdr.com:8073:14074.0"]
    receiver = FT8BlockReceiver(card_desc=card_desc, sample_rate=12000)
    pool = ProcessPoolExecutor(max_workers=2)   # moved here
    pending = []

    while True:
        try:
            print("Waiting for next UTC slot boundary (:00, :15, :30, :45)...")

            waveform, iq_waveform, future, slot_utc = receiver.capture_utc_slot_async()

            print("Captured waveform")
            # print(f"[DEBUG] Waveform shape: {waveform.shape}, Non-zero samples: {np.count_nonzero(waveform)}, RMS: {np.sqrt(np.mean(waveform**2)):.6f}")

            # Retrieve decoded messages from background process
            messages = future.result()

            pending = [f for f in pending if not f.done()]
            if len(pending) >= MAX_BACKLOG:
                print("analysis backlog, skipping slot"); continue
            f = pool.submit(analyze_slot, waveform, iq_waveform, messages,
                receiver.sample_rate, slot_utc, MAX_MSGS)
            f.add_done_callback(lambda fut: print("[analysis]", fut.exception() or fut.result()))
            pending.append(f)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing slot: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        os._exit(0)
    except Exception as e:
        print(f"There was an error: {e}")
        os._exit(1)

