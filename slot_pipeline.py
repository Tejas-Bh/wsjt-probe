"""
slot_pipeline.py -- decode one captured slot and analyse every (sampled) message. Mode-independent:
everything mode-specific comes from modes.ModeSpec; the decoders are the WSJT-X tools (or weakmon
for FT8). Runs in a worker process, so it takes and returns plain data.

Per slot:
  1. decode   FT8: jt9 -8 on a WAV (or weakmon).  WSPR: wsprd on a .c2 built from the IQ.
  2. sample   at most max_messages_per_slot, chosen at random (unbiased distance statistics).
  3. per message: channel symbols from ft8code / wsprcode  ->  baseband reference;
                  isolate the signal in the IQ; channel_estimator.estimate_channel(spec=...).
  4. write <out_dir>/<mode>_radar_<YYYYmmdd_HHMMSS>.json  (same layout aggregate.py reads).
"""

import json
import os
import random
import shutil
import tempfile
import time

import numpy as np

import channel_estimator as ce
import wsjtx_io as w
from modes import get_mode

FS = 12000


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _decode(audio, iq, slot_utc, spec, cfg, tmp):
    ch, wj = cfg["channel"], cfg["wsjtx"]
    if spec.name == "ft8":
        if ch["decoder"] == "weakmon":
            from ft8_receiver import decode_waveform_worker      # needs the weakmon checkout
            out = []
            for m in decode_waveform_worker(audio, FS, slot_utc):
                hz = m["hz"]() if callable(m["hz"]) else m["hz"]
                out.append({"message": m["message"], "snr": m["snr"], "dt": m["dt"], "hz": float(hz)})
            return out
        wav = os.path.join(tmp, time.strftime("%y%m%d_%H%M%S", time.gmtime(slot_utc)) + ".wav")
        w.write_wav(wav, audio, FS)
        return [{"message": d["message"], "snr": d["snr"], "dt": d["dt"], "hz": d["hz"]}
                for d in w.run_jt9_ft8(wav, wj["jt9"], wj["jt9_depth"])]
    # WSPR
    c2 = os.path.join(tmp, time.strftime("%y%m%d_%H%M", time.gmtime(slot_utc)) + ".c2")
    w.iq_to_c2(iq, ch["dial_khz"], FS, path=c2)
    out = []
    for d in w.run_wsprd(c2, wj["wsprd"], extra_args=tuple(wj["wsprd_args"])):
        center_hz = d["freq_mhz"] * 1e6 - ch["dial_khz"] * 1e3      # audio frequency of the tone-set centre
        out.append({"message": d["message"], "snr": d["snr"], "dt": d["dt"], "hz": center_hz,
                    "drift_reported": d["drift"]})
    return out


_SYM_CACHE = {}


def _symbols(spec, message, cfg):
    key = (spec.name, message)
    if key not in _SYM_CACHE:
        wj = cfg["wsjtx"]
        _SYM_CACHE[key] = (w.ft8code_symbols(message, wj["ft8code"]) if spec.name == "ft8"
                           else w.wsprcode_symbols(message, wj["wsprcode"]))
        if len(_SYM_CACHE) > 5000:
            _SYM_CACHE.clear()
    return _SYM_CACHE[key]


def process_slot(audio, iq, slot_utc, cfg):
    t0 = time.time()
    spec = get_mode(cfg["channel"]["mode"])
    an = cfg["analysis"]
    tmp = tempfile.mkdtemp(prefix="probe_")
    try:
        if iq is None or len(iq) == 0:
            raise RuntimeError("this pipeline needs the raw IQ stream (read_iq) from the source")
        n_total = int(round(spec.slot_s * FS))
        iq = np.asarray(iq)
        if len(iq) < n_total:
            iq = np.pad(iq, (0, n_total - len(iq)))
        decoded = _decode(audio, iq, slot_utc, spec, cfg, tmp)
        total = len(decoded)
        if an["max_messages_per_slot"] and total > an["max_messages_per_slot"]:
            decoded = random.Random(int(slot_utc)).sample(decoded, an["max_messages_per_slot"])

        log = []
        for msg in decoded:
            entry = {"mode": spec.name, "slot_total_decoded": total}
            try:
                sym = _symbols(spec, msg["message"], cfg)
                if sym is None:
                    raise ValueError("no channel symbols for this message (ft8code/wsprcode could not pack it)")
                if spec.name == "ft8":
                    centre = msg["hz"] + (spec.n_tones - 1) / 2.0 * spec.tone_spacing_hz
                    bw = 90.0
                else:
                    centre, bw = msg["hz"], 30.0
                rx = w.isolate_iq(iq, centre, bw, FS)
                ref = w.reference_baseband(sym, spec, FS, len(iq))
                est = ce.estimate_channel(rx, ref, FS, msg["dt"], compute_ci=an["compute_ci"], spec=spec)
                entry["radar_analysis"] = {"message": msg, "channel": est["channel"],
                                           "quality": est["quality"], "z": est["z"]}
            except Exception as e:                       # keep the text so failures can be classified
                entry["radar_analysis"] = {"error": repr(e), "message": msg.get("message")}
            log.append(entry)

        fname = None
        if log:
            os.makedirs(an["out_dir"], exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(slot_utc))
            fname = os.path.join(an["out_dir"], "%s_radar_%s.json" % (spec.name, stamp))
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(log, f, cls=NumpyEncoder, indent=1)
        return {"mode": spec.name, "slot": time.strftime("%H:%M:%S", time.gmtime(slot_utc)),
                "decoded": total, "analyzed": len(log), "seconds": round(time.time() - t0, 1),
                "file": fname}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
