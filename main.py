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

    while True:
        try:
            print("Waiting for next UTC slot boundary (:00, :15, :30, :45)...")

            waveform, iq_waveform, future, slot_utc = receiver.capture_utc_slot_async()

            print("Captured waveform")
            # print(f"[DEBUG] Waveform shape: {waveform.shape}, Non-zero samples: {np.count_nonzero(waveform)}, RMS: {np.sqrt(np.mean(waveform**2)):.6f}")

            # Retrieve decoded messages from background process
            messages = future.result()

            print("Retrieved messages")

            utc_str = time.strftime("%H:%M:%S UTC", time.gmtime(slot_utc))
            file_timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(slot_utc))

            print(f"\n--- Slot {utc_str} ---")
            print(f"Waveform size: {len(waveform)} samples | Type: {waveform.dtype}")
            print(f"Messages decoded: {len(messages)}")

            slot_data_log = []

            for msg in messages:
                text = msg["message"]
                snr = msg["snr"]
                hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
                dt = msg["dt"]
                print(f"[{utc_str}] SNR: {snr:.2f} dB | DT: {dt}s | Freq: {hz_val} Hz | Message: {text}")

                msg_entry = {}

                try:
                    radar_data = analyze(
                        waveform, msg, sample_rate=receiver.sample_rate, iq_waveform=iq_waveform
                    )
                    # print(f"  └─ Delay: {radar_data['path_delay_samples']} samples | Doppler: {radar_data['true_doppler_shift_hz']:.2f} Hz | Multipaths: {radar_data['path_shifts']}")
                    print(radar_data)
                    msg_entry["radar_analysis"] = radar_data
                except Exception as eval_err:
                    # Clean up the output stream and catch the exception string cleanly
                    err_msg = str(eval_err).strip() if str(eval_err) else "Unknown parsing error"
                    print(f"  └─ Analysis error: {err_msg}")
                    msg_entry["radar_analysis"] = {"error": err_msg}

                slot_data_log.append(msg_entry)

            # Save data using our custom encoder to fix the int64 crash
            if slot_data_log:
                json_filename = f"ft8_radar_{file_timestamp}.json"
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(slot_data_log, f, cls=NumpyEncoder, indent=4)
                print(f"\n[Saved] Slot data successfully written to {json_filename}")

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

