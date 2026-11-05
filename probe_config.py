"""probe_config.py -- load and validate the TOML configuration (see config.example.toml)."""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:                                   # pip install tomli on older Pythons
    import tomli as tomllib

from modes import MODES

DEFAULTS = {
    "receiver": {"type": "kiwi", "address": "", "rtl_index": 0, "rtl_mode": "direct",
                 "rtl_gain": "auto", "rtl_lo_offset_khz": 0.0},
    "channel": {"mode": "ft8", "dial_khz": 7074.0, "decoder": "wsjtx"},
    "wsjtx": {"jt9": "jt9", "wsprd": "wsprd", "ft8code": "ft8code", "wsprcode": "wsprcode",
              "ft8sim": "ft8sim", "wsprsim": "wsprsim",     # ft8sim/wsprsim: legacy, no longer used
              "jt9_depth": 3, "wsprd_args": []},
    "analysis": {"workers": 2, "max_messages_per_slot": 8, "max_backlog": 3, "compute_ci": False,
                 "out_dir": "."},
}


def load_config(path):
    with open(path, "rb") as f:
        user = tomllib.load(f)
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    for sec, vals in user.items():
        if sec not in cfg:
            raise ValueError("unknown config section [%s]" % sec)
        for k, v in vals.items():
            if k not in cfg[sec]:
                raise ValueError("unknown key %s in [%s]" % (k, sec))
            cfg[sec][k] = v
    ch = cfg["channel"]
    if ch["mode"].lower() not in MODES:
        raise ValueError("channel.mode must be one of %s" % ", ".join(MODES))
    ch["mode"] = ch["mode"].lower()
    if ch["decoder"] not in ("wsjtx", "weakmon"):
        raise ValueError("channel.decoder must be 'wsjtx' or 'weakmon'")
    if ch["decoder"] == "weakmon" and ch["mode"] != "ft8":
        raise ValueError("the weakmon decoder only supports ft8")
    if cfg["receiver"]["type"] not in ("kiwi", "rtlsdr"):
        raise ValueError("receiver.type must be 'kiwi' or 'rtlsdr'")
    if cfg["receiver"]["type"] == "kiwi" and ":" not in cfg["receiver"]["address"]:
        raise ValueError("receiver.address must be host:port for a Kiwi")
    return cfg


def build_card_desc(cfg):
    """weakaudio-style descriptor for the configured receiver."""
    r, ch = cfg["receiver"], cfg["channel"]
    if r["type"] == "kiwi":
        return ["kiwi", "%s:%s" % (r["address"], ch["dial_khz"])]
    desc = "%d:%s:%s" % (r["rtl_index"], ch["dial_khz"], r["rtl_mode"])
    if r["rtl_mode"] == "tuner":
        desc += ":%s:%s" % (r["rtl_gain"], r["rtl_lo_offset_khz"])
    return ["rtlsdr", desc]
