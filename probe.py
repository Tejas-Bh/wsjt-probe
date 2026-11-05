#!/usr/bin/env python3
"""
probe.py -- wsjt-probe entry point (FT8 or WSPR, chosen in the config file).

    python probe.py config.toml [--debug]

With --debug the receiver library's connection log is printed (use it if nothing arrives).

The loop only records slots; decoding and analysis run in worker processes so a slow slot never
makes the next one be missed. See config.example.toml for every option.
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from modes import get_mode
from probe_config import build_card_desc, load_config
from slot_pipeline import process_slot


def main(path, debug=False):
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(path)
    spec = get_mode(cfg["channel"]["mode"])
    an = cfg["analysis"]
    from capture import SlotReceiver                     # imports weakaudio (weakmon checkout)
    print("connecting to the receiver ...", flush=True)
    rx = SlotReceiver(build_card_desc(cfg), 12000, spec.slot_s)
    chk = rx.start_check(10.0)
    print("source check (10 s): audio %.0f samples/s, IQ %.0f samples/s (expect ~12000 each)" % (
        chk["audio_samples_per_s"], chk["iq_samples_per_s"]), flush=True)
    if chk["audio_samples_per_s"] < 1000:
        print("WARNING: the receiver is not delivering samples. Re-run with --debug and check the "
              "connection (host:port, a free Kiwi slot, firewall).", flush=True)
    if not chk["has_iq"]:
        print("WARNING: this source has no raw IQ stream; the pipeline needs one (Kiwi 'iq' mode or the "
              "RTL-SDR source).", flush=True)
    pool = ProcessPoolExecutor(max_workers=an["workers"],
                               mp_context=multiprocessing.get_context("spawn"))
    pending = []
    print("wsjt-probe: mode=%s dial=%.1f kHz decoder=%s slot=%.0f s workers=%d out=%s" % (
        spec.name, cfg["channel"]["dial_khz"], cfg["channel"]["decoder"], spec.slot_s, an["workers"],
        an["out_dir"]))
    while True:
        try:
            audio, iq, slot_utc = rx.capture_slot()
            pending = [f for f in pending if not f.done()]
            if len(pending) >= an["max_backlog"]:
                print("analysis backlog (%d slots pending): skipping this slot" % len(pending))
                continue
            f = pool.submit(process_slot, audio, iq, slot_utc, cfg)
            f.add_done_callback(lambda fut: print("[analysis]", fut.exception() or fut.result()))
            pending.append(f)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("error capturing slot: %r" % (e,))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--debug"]
    if len(args) != 1:
        sys.exit(__doc__)
    try:
        main(args[0], debug="--debug" in sys.argv)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        os._exit(0)
