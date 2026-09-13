"""The FLAGS the operator raised ("this is not working"), with the room as it was - for fixing later.

    python tools/dj/dj_flags.py                 # every flag, newest last, one block each
    python tools/dj/dj_flags.py --kind dead     # only one kind (see lib/dj/director.py FLAG_KINDS)
    python tools/dj/dj_flags.py --last 5        # the last five
    python tools/dj/dj_flags.py --session 670b43
    python tools/dj/dj_flags.py --json          # raw rows
    python tools/dj/dj_flags.py --summary       # counts per kind and per session

Each block: when, the kind and the note, the engine and mode, what played (every song on a deck with its
position, lanes and gains), the plan and the armed moment, the bus level, and the last moves - enough to
re-render the combination offline from the stems and hear what the operator heard.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _hms(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(t)))


def _block(f):
    out = [f"== {_hms(f['t'])}  sid {f.get('sid')}  {'✓' if f.get('good') else '⚑'} [{f.get('kind')}] {f.get('label')}" + (f"  - {f['text']}" if f.get("text") else "")]
    out.append(f"   {f.get('engine') or '?'} / {f.get('mode')}  theme {f.get('theme')}  pool {f.get('pool')}  bus rms {f.get('rms')}  bar {f.get('bar')}")
    d = f.get("dials") or {}
    if d:
        out.append("   dials: " + ", ".join(f"{k} {v}" for k, v in d.items()))
    if f.get("engine") == "conductor":
        out.append(f"   clock {f.get('master_bpm') or 0:.1f} bpm key {f.get('key_centre')}  bar_n {f.get('bar_n')}  phrase {f.get('phrase_bars')}/{f.get('change_bars')}  calm {f.get('calm_phrases')}"
                   f"  moves/10min {f.get('moves_10min')}")
        out.append("   lanes: " + ", ".join(f"{ln}: {t or '-'}" for ln, t in (f.get("lanes") or {}).items()))
        for dk, s in (f.get("songs") or {}).items():
            out.append(f"   deck {dk}: {s.get('title')} - {s.get('artist')}  {s.get('bpm')} bpm {s.get('camelot')} shift {s.get('shift')}  t={s.get('time_s')}  "
                       f"{s.get('section')}  lanes {s.get('lanes')}  gains {s.get('gains')}  eq_low {s.get('eq_low')}  {'playing' if s.get('playing') else 'stopped'}"
                       + (" leaving" if s.get("leaving") else "") + (" was_bed" if s.get("was_bed") else ""))
        p = f.get("plan")
        if p:
            out.append(f"   plan: {p.get('words')} ({p.get('in_bars')} bars) - {p.get('why')}" + (f"  (moved {p['moved']}x)" if p.get("moved") else ""))
        if f.get("wait_why"):
            out.append(f"   waiting: {f['wait_why']}")
        mo = f.get("moment")
        if mo:
            out.append(f"   moment: {mo.get('label')} {'running' if mo.get('started') else 'armed'} ({mo.get('who')}), the one in {mo.get('in_bars')} bars on {mo.get('what')}")
        for m in f.get("moves") or []:
            out.append(f"   move {m.get('hms')} {m.get('kind')}{'/' + m['lane'] if m.get('lane') else ''}: {m.get('text')}")
    elif f.get("engine") == "autodj":
        cur = f.get("current") or {}
        out.append(f"   playing: {cur.get('title')} - {cur.get('artist')} at {cur.get('pos_s')} s of {cur.get('duration_s')}  state {f.get('state')}")
        nx = f.get("next") or {}
        if nx:
            out.append(f"   next: {nx.get('title')}  plan {(f.get('plan') or {}).get('style')}  blend in {f.get('blend_in_s')} s")
        for h in f.get("history") or []:
            out.append(f"   seam: {h}")
    lv = f.get("last_verdict")
    if lv:
        out.append(f"   last verdict: {'GOOD' if lv.get('up') else 'BAD'} {lv.get('what')}")
    do = f.get("last_dropout")
    if do:
        out.append(f"   last dropout: {do}")
    for e in f.get("events") or []:
        out.append(f"   event {e}")
    if f.get("snapshot_error"):
        out.append(f"   (snapshot error: {f['snapshot_error']})")
    return "\n".join(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the flag glyphs on a cp1252 console
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--good", action="store_true", help="only the good flags (the cases to protect)")
    ap.add_argument("--bad", action="store_true", help="only the problems")
    ap.add_argument("--session")
    ap.add_argument("--last", type=int)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    from lib.dj.session import read_flags
    flags = read_flags()
    if a.kind:
        flags = [f for f in flags if f.get("kind") == a.kind]
    if a.good:
        flags = [f for f in flags if f.get("good")]
    if a.bad:
        flags = [f for f in flags if not f.get("good")]
    if a.session:
        flags = [f for f in flags if str(f.get("sid", "")).startswith(a.session)]
    if a.last:
        flags = flags[-a.last:]
    if not flags:
        print("no flags" + (" of that kind" if a.kind else "") + " - raise one with FLAG in the Director tab (or SET on the nanoKONTROL2)")
        return 0
    if a.summary:
        good = [f for f in flags if f.get("good")]
        print(f"{len(flags)} flags: {len(flags) - len(good)} problems, {len(good)} kept as good")
        for k, n in Counter(f.get("kind") for f in flags).most_common():
            print(f"  {n:3d}  {'✓' if k in {g.get('kind') for g in good} else '⚑'} {k}: {next((f.get('label') for f in flags if f.get('kind') == k), '')}")
        print("per session:")
        for s, n in Counter(f.get("sid") for f in flags).most_common():
            first = min(f["t"] for f in flags if f.get("sid") == s)
            print(f"  {n:3d}  {s}  ({_hms(first)})")
        return 0
    if a.json:
        for f in flags:
            print(json.dumps(f, ensure_ascii=False))
        return 0
    for f in flags:
        print(_block(f))
        print()
    print(f"{len(flags)} flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
