#!/usr/bin/env python3
import time
import os
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

            print(f"\n--- Slot {utc_str} ---")
            print(f"Waveform size: {len(waveform)} samples | Type: {waveform.dtype}")
            print(f"Messages decoded: {len(messages)}")

            for msg in messages:
                text = msg["message"]
                snr = msg["snr"]
                hz_val = msg["hz"]() if callable(msg["hz"]) else msg["hz"]
                dt = msg["dt"]
                print(f"[{utc_str}] SNR: {snr} dB | DT: {dt}s | Freq: {hz_val} Hz | Message: {text}")

                try:
                    radar_data = analyze(waveform, msg)
                    print(f"  └─ Delay: {radar_data['path_delay_samples']} samples | Doppler: {radar_data['doppler_shift_hz']} Hz | Multipaths: {radar_data['path_shifts']}")
                except Exception as eval_err:
                    print(f"  └─ Analysis error: {eval_err}")

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
