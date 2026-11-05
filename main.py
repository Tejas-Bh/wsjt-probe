#!/usr/bin/env python3
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

import os
from concurrent.futures import ProcessPoolExecutor
from ft8_receiver import FT8BlockReceiver
from slot_worker import analyze_slot
from config_loader import load_config

def main():
    cfg = load_config("config.json")
    print(f"Loaded configuration for Mode: {cfg['mode']}")
    print(f"Source: {cfg['card_desc']}")

    receiver = FT8BlockReceiver(
        card_desc=cfg["card_desc"], 
        sample_rate=cfg["sample_rate"]
    )
    
    max_backlog = int(cfg.get("max_backlog", 3))
    max_workers = int(cfg.get("max_workers", 2))
    
    pool = ProcessPoolExecutor(max_workers=max_workers)
    pending = []

    while True:
        try:
            print(f"Waiting for next UTC slot boundary ({cfg['mode']})...")

            waveform, iq_waveform, future, slot_utc = receiver.capture_utc_slot_async()

            print("Captured waveform")

            # Retrieve decoded messages from background worker
            messages = future.result()

            # Filter completed background tasks
            pending = [f for f in pending if not f.done()]
            
            # Ensure pending length is checked against an integer
            if len(pending) >= max_backlog:
                print("Analysis backlog limit reached, skipping slot")
                continue

            f = pool.submit(
                analyze_slot, 
                waveform, 
                iq_waveform, 
                messages,
                receiver.sample_rate, 
                slot_utc,
                mode_name=cfg["mode"],
                max_msgs=cfg.get("max_msgs_per_slot", 8),
                out_dir=cfg.get("output_dir", ".")
            )
            
            # Print result or exception cleanly without returning exceptions into comparisons
            def _log_result(fut):
                err = fut.exception()
                if err is not None:
                    print(f"[analysis error] {err}")
                else:
                    print(f"[analysis] {fut.result()}")

            f.add_done_callback(_log_result)
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
