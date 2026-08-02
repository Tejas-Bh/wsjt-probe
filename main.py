#!/usr/bin/env python3
import time
from ft8_receiver import FT8BlockReceiver

def main():
    # Example card configuration: soundcard 0, channel 0
    card_desc = ["4", "0"]
    receiver = FT8BlockReceiver(card_desc=card_desc, sample_rate=12000)

    print("Waiting for next UTC slot boundary (:00, :15, :30, :45)...")

    # Capture aligned 15-second slot
    waveform, messages, slot_utc = receiver.capture_utc_slot()

    # Format timestamp for printing
    utc_str = time.strftime("%H:%M:%S UTC", time.gmtime(slot_utc))

    print(f"\n--- Slot {utc_str} ---")
    print(f"Waveform array size in memory: {len(waveform)} samples")
    print(f"Messages decoded: {len(messages)}")

    # Store and access in variables as needed
    for msg in messages:
        text = msg["message"]
        snr = msg["snr"]
        hz = msg["hz"]()
        dt = msg["dt"]
        print(f"[{utc_str}] SNR: {snr} dB | DT: {dt}s | Freq: {hz} Hz | Message: {text}")

if __name__ == "__main__":
    try:
        if True: main()
    except KeyboardInterrupt:
        print("goodbye!")
        exit(0)
    except Exception as e:
        print("There was an error: ")
        print(e)
        exit(1)
