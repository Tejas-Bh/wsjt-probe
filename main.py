#!/usr/bin/env python3
import time
import os
import json
from ft8_receiver import FT8BlockReceiver
from analyze import analyze

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

            # List to aggregate all decoded message objects for this slot
            slot_data_log = []

            for msg in messages:
                text = msg["message"]
                snr = msg["snr"]
                hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
                dt = msg["dt"]
                print(f"[{utc_str}] SNR: {snr} dB | DT: {dt}s | Freq: {hz_val} Hz | Message: {text}")

                # Build dictionary for the individual message data
                msg_entry = {}

                try:
                    radar_data = analyze(waveform, msg)
                    print(f"  └─ Delay: {radar_data['path_delay_samples']} samples | Doppler: {radar_data['doppler_shift_hz']} Hz | Multipaths: {radar_data['path_shifts']}")
                    
                    # Store radar results directly into the message dictionary
                    msg_entry["radar_analysis"] = radar_data
                except Exception as eval_err:
                    print(f"  └─ Analysis error: {eval_err}")
                    msg_entry["radar_analysis"] = {"error": str(eval_err)}

                slot_data_log.append(msg_entry)

            # Save the gathered data to a timestamped JSON file if messages exist
            if slot_data_log:
                json_filename = f"ft8_radar_{file_timestamp}.json"
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(slot_data_log, f, indent=4)
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

