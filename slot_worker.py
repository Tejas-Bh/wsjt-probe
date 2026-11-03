"""
slot_worker.py -- run the per-message analysis OFF the capture loop.

Why: main.py currently analyses every decoded message before it goes back to wait for the
next UTC slot. At ~4 s per message and ~20 messages per slot that is ~80 s of analysis per
15 s slot, so most slots are never captured. Here the capture loop only decodes and
submits; analysis runs in worker processes.

main.py changes (see the chat reply for the full diff):

    from concurrent.futures import ProcessPoolExecutor
    from slot_worker import analyze_slot
    pool = ProcessPoolExecutor(max_workers=2)
    pending = []
    ...
    messages = future.result()
    pending = [f for f in pending if not f.done()]
    if len(pending) >= MAX_BACKLOG:
        print("analysis backlog, skipping slot"); continue
    f = pool.submit(analyze_slot, waveform, iq_waveform, messages,
                    receiver.sample_rate, slot_utc, MAX_MSGS)
    f.add_done_callback(lambda fut: print("[analysis]", fut.exception() or fut.result()))
    pending.append(f)

Sampling: if a slot has more than max_msgs messages, a RANDOM subset is analysed (seeded
by the slot time), not the strongest ones -- picking the strongest would bias the
path-distance statistics toward nearby stations.
"""

import json
import os
import random
import time

import numpy as np

from analyze import analyze


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_slot(waveform, iq_waveform, messages, sample_rate, slot_utc,
                 max_msgs=None, out_dir="."):
    t0 = time.time()
    total = len(messages)
    if max_msgs and total > max_msgs:
        messages = random.Random(int(slot_utc)).sample(list(messages), max_msgs)

    log = []
    for msg in messages:
        text = msg.get("message")
        entry = {"slot_total_decoded": total}
        try:
            entry["radar_analysis"] = analyze(
                waveform, msg, sample_rate=sample_rate, iq_waveform=iq_waveform)
        except Exception as e:
            entry["radar_analysis"] = {"error": repr(e), "message": text}
        log.append(entry)

    fname = None
    if log:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(slot_utc))
        fname = os.path.join(out_dir, "ft8_radar_%s.json" % stamp)
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(log, f, cls=NumpyEncoder, indent=1)

    return {"slot": time.strftime("%H:%M:%S", time.gmtime(slot_utc)), "decoded": total,
            "analyzed": len(log), "seconds": round(time.time() - t0, 1), "file": fname}
