#!/usr/bin/env python3
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
    card_desc = ["4", "0"]
    receiver = FT8BlockReceiver(card_desc=card_desc, sample_rate=12000)

    print("Waiting for next UTC slot boundary (:00, :15, :30, :45)...")

    if True:
        try:
            waveform, future, slot_utc = receiver.capture_utc_slot_async()

            # Retrieve decoded messages from background process
            messages = future.result()

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
                    radar_data = analyze(waveform, msg)
                    print(f"  └─ Delay: {radar_data['path_delay_samples']} samples | Doppler: {radar_data['doppler_shift_hz']:.2f} Hz | Multipaths: {radar_data['path_shifts']}")
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

