#!/usr/bin/env python3
"""
aggregate.py -- quality-gated, per-station aggregation of wsjt-probe JSON files.

    python aggregate.py DIR_OR_GLOB [--rx-grid JO62] [--bin 30min] [--out run1]

Reads ft8_radar_YYYYMMDD_HHMMSS.json files (new channel_estimator format:
{"radar_analysis": {"message":..., "watterson_taps":[...], "quality":{...}}}) and writes
    <out>_messages.csv   one row per message (incl. ok/llr/snr_sym, callsign, grid, distance)
    <out>_slots.csv      one row per slot, ok-only medians (replaces the old mean-based CSV)
    <out>_report.txt     counts, error texts, binned trends

What it does differently from the old aggregation:
  * only messages with quality.ok == True feed the statistics (medians, not means);
  * spread is also reported relative to each station's own median ("station-demeaned"),
    which removes the transmitter-stability / station-mix confound from the trend;
  * path distance (needs --rx-grid) gives a second day/night observable that needs no
    channel estimation;
  * analysis errors are counted by text. Note: main.py currently drops the message text
    for failed analyses, so see the one-line patch in the chat reply to keep it.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

GRID_RE = re.compile(r"^[A-R]{2}[0-9]{2}([A-X]{2})?$", re.I)
NOT_CALL = {"RR73", "RRR", "73", "CQ", "DE", "QRZ"}
FNAME_RE = re.compile(r"(ft8|wspr)_radar_(\d{8})_(\d{6})")


def is_call(tok):
    t = tok.upper()
    if t in NOT_CALL or GRID_RE.match(t) or re.match(r"^[+-]?\d+$", t) or t.startswith("R-") or t.startswith("R+"):
        return False
    return 3 <= len(t) <= 12 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)


def parse_wspr(text):
    """WSPR: 'CALL GRID PWR' | 'PFX/CALL PWR' | '<PFX/CALL> GRID6 PWR'. Returns (call, grid, dBm)."""
    toks = text.replace("<", "").replace(">", "").split()
    if not toks:
        return None, None, None
    call = toks[0].upper()
    grid = toks[1].upper() if len(toks) >= 3 and GRID_RE.match(toks[1]) else None
    p = toks[-1].lstrip("+-")
    power = int(toks[-1]) if p.isdigit() and len(toks) >= 2 else None
    return call, grid, power


def parse_message(text):
    """Return (sender_call, sender_grid). FT8: 'TO FROM payload'; CQ: 'CQ [mod] FROM grid'."""
    toks = text.replace("<", "").replace(">", "").split()
    if not toks:
        return None, None
    grid = toks[-1].upper() if (GRID_RE.match(toks[-1]) and toks[-1].upper() != "RR73") else None
    call = None
    if toks[0].upper() == "CQ":
        for t in toks[1:]:
            if is_call(t):
                call = t.upper()
                break
    elif len(toks) >= 2 and is_call(toks[1]):
        call = toks[1].upper()
    return call, grid


def grid_to_latlon(g):
    g = g.upper()
    lon = (ord(g[0]) - 65) * 20 - 180 + int(g[2]) * 2
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3])
    if len(g) >= 6:
        lon += (ord(g[4]) - 65) * (2 / 24) + (1 / 24)
        lat += (ord(g[5]) - 65) * (1 / 24) + (1 / 48)
    else:
        lon += 1.0
        lat += 0.5
    return lat, lon


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def collect_files(arg):
    if os.path.isdir(arg):
        files = glob.glob(os.path.join(arg, "*_radar_*.json"))
    else:
        files = glob.glob(arg)
    return sorted(files)


def load(files, rx_ll):
    rows, errors = [], Counter()
    n_err_total = 0
    for fp in files:
        m = FNAME_RE.search(os.path.basename(fp))
        if not m:
            continue
        fmode = m.group(1)
        slot = pd.to_datetime(m.group(2) + m.group(3), format="%Y%m%d%H%M%S", utc=True)
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print("skip %s: %s" % (fp, e), file=sys.stderr)
            continue
        for entry in data:
            r = entry.get("radar_analysis", entry)
            if "error" in r:
                n_err_total += 1
                errors[str(r["error"])[:120]] += 1
                rows.append({"slot": slot, "error": True, "mode": fmode, "error_text": str(r["error"])[:120],
                             "message": r.get("message")})
                continue
            msg = r.get("message", {}) or {}
            tap = r.get("channel") or (r.get("watterson_taps") or [{}])[0]
            q = r.get("quality")
            mode_ = (q or {}).get("mode") or entry.get("mode") or fmode
            power_dbm = None
            if mode_ == "wspr":
                call, grid, power_dbm = parse_wspr(msg.get("message", "") or "")
            else:
                call, grid = parse_message(msg.get("message", "") or "")
            dist = None
            if rx_ll is not None and grid:
                try:
                    dist = haversine_km(rx_ll, grid_to_latlon(grid))
                except Exception:
                    dist = None
            rows.append({
                "slot": slot, "error": False, "mode": mode_, "power_dbm": power_dbm,
                "message": msg.get("message"), "call": call, "grid": grid, "dist_km": dist,
                "snr_db": msg.get("snr"), "audio_hz": msg.get("hz"), "dt": msg.get("dt"),
                "has_quality": q is not None,
                "ok": bool(q["ok"]) if q is not None else None,
                "llr": q.get("llr") if q else None,
                "snr_sym_db": q.get("snr_sym_db") if q else None,
                "pilot_hz": q.get("pilot_hz") if q else None,
                "align_offset_ms": q.get("align_offset_ms") if q else None,
                "drift_hz_per_min": q.get("drift_hz_per_min") if q else None,
                "drift_removed": q.get("drift_removed") if q else None,
                "spread_hz": tap.get("doppler_spread_hz"),
                "spread_lo_hz": tap.get("spread_lo_hz"),
                "spread_hi_hz": tap.get("spread_hi_hz"),
                "shift_hz": tap.get("doppler_shift_hz"),
            })
    return pd.DataFrame(rows), errors, n_err_total


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def station_report(good, call, binspec):
    g = good[good["call"] == call].sort_values("slot")
    print("station %s: %d ok messages in %d slots" % (call, len(g), g["slot"].nunique()))
    if len(g) < 5:
        print("too few messages for a series")
        return
    t = (g["slot"] - g["slot"].min()).dt.total_seconds() / 3600.0
    print("spread: median %.3f Hz, IQR %.3f, range %.3f-%.3f" % (
        g["spread_hz"].median(), np.subtract(*np.percentile(g["spread_hz"], [75, 25])),
        g["spread_hz"].min(), g["spread_hz"].max()))
    if g["dist_km"].notna().any():
        print("distance: %.0f km (grid %s)" % (g["dist_km"].median(), g["grid"].dropna().iloc[0]))
    try:
        from scipy.stats import spearmanr, theilslopes
        r = spearmanr(t, g["spread_hz"])
        sl, ic, lo, hi = theilslopes(g["spread_hz"], t)
        print("trend: Spearman rho=%.2f p=%.3g (n=%d);  Theil-Sen slope %+.3f Hz/h [%+.3f, %+.3f]" % (
            r.statistic, r.pvalue, len(g), sl, lo, hi))
        r2 = spearmanr(g["snr_db"], g["spread_hz"])
        print("spread vs reported SNR within this station: rho=%.2f" % r2.statistic)
    except Exception:
        pass
    g = g.assign(bin=g["slot"].dt.floor(binspec))
    print("%-6s %4s %9s %9s %8s" % ("bin", "n", "med_spr", "mean_spr", "snr"))
    for b, x in g.groupby("bin"):
        print("%-6s %4d %9.3f %9.3f %8.1f" % (b.strftime("%H:%M"), len(x), x["spread_hz"].median(),
                                              x["spread_hz"].mean(), x["snr_db"].median()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--rx-grid", default=None, help="receiver Maidenhead grid, e.g. JO62 (enables distance)")
    ap.add_argument("--bin", default=None, help="time bin (default 30min for ft8, 60min for wspr)")
    ap.add_argument("--mode", default=None, choices=["ft8", "wspr"], help="analyse one mode (required if the data holds both)")
    ap.add_argument("--out", default="wsjt_probe")
    ap.add_argument("--min-station-msgs", type=int, default=3)
    ap.add_argument("--station", default=None, help="print the within-station spread series for CALL and exit")
    ap.add_argument("--far-km", type=float, default=1000.0, help="threshold for the 'far' fraction")
    a = ap.parse_args()

    rx_ll = grid_to_latlon(a.rx_grid) if a.rx_grid else None
    files = collect_files(a.input)
    if not files:
        sys.exit("no ft8_radar_*.json or wspr_radar_*.json files found")
    df, errors, n_err = load(files, rx_ll)
    present = sorted(df["mode"].dropna().unique()) if "mode" in df else []
    if a.mode:
        df = df[df["mode"] == a.mode]
    elif len(present) > 1:
        sys.exit("the data holds several modes (%s): pass --mode" % ", ".join(present))
    mode_name = a.mode or (present[0] if present else "ft8")
    if a.bin is None:
        a.bin = "60min" if mode_name == "wspr" else "30min"
    n_err = int(df["error"].sum()) if len(df) else 0
    msgs = df[~df["error"]].copy()
    errdf = df[df["error"]].copy()
    out = []
    out.append("wsjt-probe aggregation report")
    out.append("=" * 60)
    out.append("mode: %s   time bin: %s" % (mode_name, a.bin))
    out.append("files: %d   messages: %d   analysis errors: %d (%.1f%%)" % (
        len(files), len(df), n_err, 100.0 * n_err / max(len(df), 1)))
    for k, v in errors.most_common(8):
        out.append("   error x%d: %s" % (v, k or "(empty)"))

    if len(errdf) and "message" in errdf and errdf["message"].notna().any():
        em = errdf["message"].dropna().astype(str)
        cats = {
            "hashed call <...>": em.str.contains("<", regex=False),
            "compound call (has /)": em.str.contains("/", regex=False),
            "CQ with modifier": em.str.match(r"^CQ [A-Z0-9]{2,4} ", case=False) & ~em.str.contains("<|/", regex=True),
            "<=2 tokens / free text": em.str.split().str.len() <= 2,
            "RR73/73/RRR": em.str.contains(r"\b(?:RR73|RRR|73)\b", regex=True),
        }
        out.append("")
        out.append("Failed-analysis message categories (%d messages with text; a message can be in several):" % len(em))
        for k, v in cats.items():
            out.append("   %-26s %4d  (%.0f%%)" % (k, v.sum(), 100.0 * v.mean()))
        out.append("   examples (random 15):")
        for t in em.sample(min(15, len(em)), random_state=0):
            out.append("      " + t)
        errdf[["slot", "error_text", "message"]].to_csv(a.out + "_errors.csv", index=False)

    if len(msgs) == 0:
        print("\n".join(out))
        return
    if not msgs["has_quality"].all():
        out.append("WARNING: %d messages have no quality block (old format?); they are excluded "
                   "from gated statistics." % (~msgs["has_quality"]).sum())
    good = msgs[msgs["ok"] == True].copy()  # noqa: E712
    out.append("quality.ok: %d of %d messages (%.0f%%)" % (len(good), len(msgs), 100.0 * len(good) / max(len(msgs), 1)))
    if len(good) == 0:
        print("\n".join(out))
        msgs.to_csv(a.out + "_messages.csv", index=False)
        return

    if a.station:
        station_report(good, a.station.upper(), a.bin)
        return

    # station-demeaned spread
    cnt = good.groupby("call")["spread_hz"].transform("count")
    med = good.groupby("call")["spread_hz"].transform("median")
    good["spread_rel"] = np.where((good["call"].notna()) & (cnt >= a.min_station_msgs),
                                  good["spread_hz"] - med, np.nan)

    good["bin"] = good["slot"].dt.floor(a.bin)
    out.append("")
    out.append("Binned trends (quality.ok messages only; CI = 95% bootstrap of the bin mean)")
    hdr = "%-6s %5s | %-8s %-8s | %-22s | %-9s %s" % (
        "bin", "n_ok", "med_spr", "mean_spr", "station-demeaned (n)", "med_snr", "med_dist_km")
    out.append(hdr)
    dist_rows = []
    for b, g in good.groupby("bin"):
        rel = g["spread_rel"].dropna()
        lo, hi = boot_ci(rel.values)
        rel_s = "%+.3f [%+.3f,%+.3f] (%d)" % (rel.mean(), lo, hi, len(rel)) if len(rel) >= 3 else "n/a (%d)" % len(rel)
        gd = g.dropna(subset=["call", "dist_km"])
        # each station counts once per bin (its median distance), so a persistent caller
        # cannot dominate the distance statistics
        per = gd.groupby("call")["dist_km"].median()
        if len(per) >= 3:
            dist_rows.append((b.strftime("%H:%M"), len(per), np.percentile(per, 10), per.median(),
                              np.percentile(per, 90), 100.0 * (per > a.far_km).mean()))
        dist = "%.0f" % per.median() if len(per) else "-"
        out.append("%-6s %5d | %-8.3f %-8.3f | %-22s | %-9.1f %s" % (
            b.strftime("%H:%M"), len(g), g["spread_hz"].median(), g["spread_hz"].mean(), rel_s,
            g["snr_db"].median(), dist))

    if dist_rows:
        out.append("")
        out.append("Path distance per bin, one value per station per bin (needs --rx-grid)")
        out.append("%-6s %9s | %8s %8s %8s | %s" % ("bin", "stations", "p10_km", "p50_km", "p90_km", "frac>%.0fkm" % a.far_km))
        for r_ in dist_rows:
            out.append("%-6s %9d | %8.0f %8.0f %8.0f | %5.1f%%" % r_)

    spr_rows = []
    for b, g in good.groupby("bin"):
        per = g.dropna(subset=["call"]).groupby("call")["spread_hz"].median()
        if len(per) >= 5:
            spr_rows.append((b.strftime("%H:%M"), len(per), np.percentile(per, 10), per.median(),
                             np.percentile(per, 90)))
    if spr_rows:
        out.append("")
        out.append("Spread per bin, one value per station (station median in the bin).")
        out.append("p10 is the lower envelope: the quietest stations bound the ionospheric contribution.")
        out.append("%-6s %9s | %7s %7s %7s" % ("bin", "stations", "p10_Hz", "p50_Hz", "p90_Hz"))
        for r_ in spr_rows:
            out.append("%-6s %9d | %7.3f %7.3f %7.3f" % r_)

    if mode_name == "wspr":
        out.append("")
        out.append("WSPR per bin: drift removed (|Hz/min|, transmitter + path combined) and reported power")
        out.append("%-6s %5s | %-12s %-10s | %s" % ("bin", "n_ok", "drift_removed", "med|drift|", "med_power_dBm"))
        for b, g in good.groupby("bin"):
            dr = g["drift_removed"].fillna(False).astype(bool)
            md = g.loc[dr, "drift_hz_per_min"].abs().median() if dr.any() else float("nan")
            pw = g["power_dbm"].median() if g["power_dbm"].notna().any() else float("nan")
            out.append("%-6s %5d | %-12s %-10.2f | %.0f" % (b.strftime("%H:%M"), len(g), "%.0f%%" % (100 * dr.mean()), md, pw))

    try:
        from scipy.stats import spearmanr
        t = (good["slot"] - good["slot"].min()).dt.total_seconds() / 3600.0
        r = spearmanr(t, good["spread_hz"])
        out.append("")
        out.append("Spearman spread vs time (raw): rho=%.2f p=%.2g  (messages are not independent; "
                   "treat p as optimistic)" % (r.statistic, r.pvalue))
        gr = good.dropna(subset=["spread_rel"])
        if len(gr) > 10:
            r2 = spearmanr((gr["slot"] - good["slot"].min()).dt.total_seconds() / 3600.0, gr["spread_rel"])
            out.append("Spearman station-demeaned spread vs time: rho=%.2f p=%.2g (n=%d)" % (r2.statistic, r2.pvalue, len(gr)))
        r3 = spearmanr(good["snr_db"], good["spread_hz"])
        out.append("Spearman spread vs reported SNR: rho=%.2f  (should be ~0 if the estimator is floor-free)" % r3.statistic)
    except Exception:
        pass

    top = good.dropna(subset=["call"]).groupby("call").agg(
        n=("spread_hz", "size"), slots=("slot", "nunique"), med_spread=("spread_hz", "median"),
        iqr=("spread_hz", lambda x: np.subtract(*np.percentile(x, [75, 25]))),
        med_dist=("dist_km", "median")).sort_values("n", ascending=False).head(12)
    out.append("")
    out.append("Most-seen stations (ok messages): per-station spread should be stable if the "
               "measurement is real")
    out.append(top.round(3).to_string())

    # per-slot (ok-only medians)
    slots = good.groupby("slot").agg(
        n_ok=("spread_hz", "size"), spread_med=("spread_hz", "median"),
        shift_med=("shift_hz", "median"), snr_med=("snr_db", "median"),
        dist_med=("dist_km", "median"), llr_med=("llr", "median")).reset_index()
    tot = df.groupby("slot").agg(n_records=("error", "size"), n_errors=("error", "sum")).reset_index()
    slots = tot.merge(slots, on="slot", how="left")
    slots.to_csv(a.out + "_slots.csv", index=False)
    msgs.drop(columns=["error"]).to_csv(a.out + "_messages.csv", index=False)
    with open(a.out + "_report.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwrote %s_messages.csv, %s_slots.csv, %s_report.txt" % (a.out, a.out, a.out))


if __name__ == "__main__":
    main()
