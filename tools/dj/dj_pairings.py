"""PAIRINGS: for every record in the stem library, the other songs whose parts sit on it - computed once,
offline, from the stems (docs/DJ_PAIRINGS.md).

  python tools/dj/dj_pairings.py --envelopes --music D:/Devel/music [--workers 3]
      the per-stem 2 Hz envelopes of every stem track -> .stems/<id>/env.npz (resumable; the same envelopes
      the conductor reads live, lib/dj/stemload.stem_levels)
  python tools/dj/dj_pairings.py --pairs --music D:/Devel/music
      the per-song measures (sung stretches, instrumental grooves, the bass from the landmark) and the two
      ranked lists per record (VOCAL OVER, BASS UNDER) -> .stems/pairings.json
  python tools/dj/dj_pairings.py --show "Grandma Dance" --music D:/Devel/music
      a record's lists in words
  python tools/dj/dj_pairings.py --all --music D:/Devel/music        (envelopes, then pairs)
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("DJ_NO_PLAY_LOG", "1")

STEMS = ("drums", "bass", "other", "vocals")
SING_ON, SING_OFF = 0.12, 0.05          # the conductor's lines (lib/dj/remix.py)
CARRY_ENV = 0.35
MIN_SUNG_BARS = 8
MIN_GROOVE_BARS = 16
MIN_BASS_BARS = 16
KEY_FIT_MIN = 0.68                      # ENTRY_FIT: what an arriving tonal stem must fit
SHIFTS = (0, 1, -1, 2, -2)
SHIFT_COST = 0.04                       # per semitone
VOCAL_RATE = (0.94, 1.06)               # a vocal shifted further than 6 % sounds wrong
BASS_RATE = (0.90, 1.10)                # the tempo wall
PLAY_MIN_S = 210.0
RUNWAY_S = 90.0
TOP_N = 12
VERSION = 1


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def env_path(music, tid):
    return os.path.join(music, ".stems", str(tid), "env.npz")


# -- the envelope pass ------------------------------------------------------------------------------------------
def _envelope_one(args):
    music, tid, body_s, sections = args
    out = env_path(music, tid)
    if os.path.exists(out):
        return tid, "have"
    try:
        os.environ["DJ_DECODE_INPROC"] = "1"
        from lib.dj.stems import load_stems
        from lib.dj.stemload import stem_levels
        stems = load_stems(music, tid)
        if not stems:
            return tid, "no stems"
        lv = stem_levels(stems, body_s, sections)
        env = lv.get("_env") or {}
        np.savez_compressed(out, hop_s=0.5, **{k: np.asarray(v, dtype=np.float16) for k, v in env.items()},
                            rms=np.asarray([float(lv.get(k, 0.0)) for k in STEMS], dtype=np.float32))
        return tid, "ok"
    except Exception as e:  # noqa: BLE001
        return tid, f"{type(e).__name__}: {e}"


def envelopes(music, workers):
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    db = LibraryDB(music)
    lib = [t for t in B.load_library(db) if getattr(t, "has_stems", False)]
    todo = [t for t in lib if not os.path.exists(env_path(music, t.id))]
    print(f"envelopes: {len(lib)} stem tracks, {len(lib) - len(todo)} done, {len(todo)} to do, {workers} workers")
    jobs = [(music, t.id, _body_start(t, None), [dict(start_s=float(s["start_s"]), end_s=float(s["end_s"])) for s in (t.sections or [])]) for t in todo]
    t0 = time.time()
    done, bad = 0, []
    if workers <= 1:
        it = map(_envelope_one, jobs)
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(workers)
        it = pool.imap_unordered(_envelope_one, jobs)
    for tid, st in it:
        done += 1
        if st not in ("ok", "have"):
            bad.append((tid, st))
        if done % 25 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min){'  problems: ' + str(len(bad)) if bad else ''}", flush=True)
    for tid, st in bad[:10]:
        print("  problem", tid, st)
    print(f"envelopes done in {(time.time() - t0) / 60:.1f} min")


# -- the per-song measures ---------------------------------------------------------------------------------------
def _bar_s(t):
    return 4 * 60.0 / max(float(t.bpm or 120.0), 1e-6)


def _downbeat(t, x):
    try:
        return float(t.nearest_downbeat(x))
    except Exception:
        return float(x)


def _carries(env, lane_names, t0, span_s, thr=CARRY_ENV):
    seen, good = False, True
    for ln in lane_names:
        e = env.get(ln)
        if e is None or len(e) < 4:
            continue
        i0, i1 = max(0, int(t0 * 2)), min(len(e), int((t0 + span_s) * 2) + 1)
        if i1 - i0 < 2:
            continue
        seen = True
        seg = np.asarray(e[i0:i1], dtype=np.float32)
        if float(np.percentile(seg, 25)) < thr:
            good = False
            continue
        if len(seg) >= 4 and float(np.convolve(seg, np.ones(4, dtype=np.float32) / 4.0, mode="valid").min()) < 0.5 * thr:
            good = False
    return good if seen else None


def _body_start(t, env):
    secs = t.sections or []
    cands = [s for s in secs if s.get("kind") == "groove" and s.get("bass_share", 0.3) >= 0.28]
    cands += [s for s in secs if s.get("kind") not in ("intro", "outro") and s not in cands]
    first = None
    for s in cands:
        t0 = _downbeat(t, float(s["start_s"]))
        if first is None:
            first = t0
        if env is None or _carries(env, ("drums", "bass"), t0, 16 * _bar_s(t)) is not False:
            return t0
    return first if first is not None else 0.0


def _landmark(t, env):
    body = _body_start(t, env)
    secs = t.sections or []
    lm = body
    for i in range(1, len(secs)):
        if secs[i - 1].get("kind") == "build" and secs[i].get("kind") == "groove" and body <= float(secs[i]["start_s"]) <= body + 90.0 \
                and float(t.duration_s) - float(secs[i]["start_s"]) >= RUNWAY_S:
            lm = _downbeat(t, float(secs[i]["start_s"]))
            break
    return lm


def _sung_stretches(t, env):
    """[(t0, t1)] where the vocal sings (60 % of a two-bar window over the silent line and a peak over the
    singing line), gaps under two bars merged, eight bars or more."""
    vox = env.get("vocals")
    if vox is None or len(vox) < 8:
        return []
    bar = _bar_s(t)
    arr = np.asarray(vox, dtype=np.float32)
    w = max(2, int(2 * bar * 2))
    on = np.zeros(len(arr), dtype=bool)
    for i in range(0, len(arr) - w + 1):
        seg = arr[i:i + w]
        if float(np.mean(seg >= SING_OFF)) >= 0.6 and float(seg.max()) >= SING_ON:
            on[i:i + w] = True
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            out.append([i / 2.0, j / 2.0])
            i = j
        else:
            i += 1
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] <= 2 * bar:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_SUNG_BARS * bar]


def _instrumental_grooves(t, env):
    """[(t0, t1)] stretches of sixteen bars or more: a groove by the map, drums and bass carrying, the vocal
    silent (peak under the singing line) - where another vocal can sit."""
    secs = t.sections or []
    bar = _bar_s(t)
    vox = env.get("vocals")
    out = []
    for s in secs:
        if s.get("kind") not in ("groove", "intro"):
            continue
        a, b = _downbeat(t, float(s["start_s"])), float(s["end_s"])
        # walk in four-bar steps, keeping the longest sub-stretch that is instrumental and carrying
        step = 4 * bar
        cur = None
        x = a
        while x + 4 * bar <= b + 0.01:
            i0, i1 = int(x * 2), int((x + step) * 2) + 1
            # quiet = not SINGING there: less than a third of the half-seconds over the silent line (a lone ad-lib
            # or bleed spike at 0.14 split a 47-second instrumental groove into pieces too short to use)
            vseg = np.asarray(vox[i0:i1], dtype=np.float32) if (vox is not None and len(vox) >= 4 and i1 > i0) else None
            quiet = vseg is None or (float(np.mean(vseg >= SING_OFF)) < 0.3 and float(np.percentile(vseg, 90)) < 0.25)
            carry = _carries(env, ("drums", "bass"), x, step) is not False
            if quiet and carry:
                cur = [x, x + step] if cur is None else [cur[0], x + step]
            else:
                if cur is not None and cur[1] - cur[0] >= MIN_GROOVE_BARS * bar:
                    out.append(tuple(cur))
                cur = None
            x += step
        if cur is not None and cur[1] - cur[0] >= MIN_GROOVE_BARS * bar:
            out.append(tuple(cur))
    return out


def measure_song(t, env):
    bar = _bar_s(t)
    lm = _landmark(t, env)
    bass_ok = _carries(env, ("bass",), lm, MIN_BASS_BARS * bar) is not False and _carries(env, ("drums",), lm, MIN_BASS_BARS * bar) is not False
    hook = None
    if getattr(t, "hook", None):
        hook = [float(x) for x in (t.hook.get("starts") or [t.hook.get("start_s")]) if x is not None]
    sung = _sung_stretches(t, env)
    return {"id": t.id, "title": t.title, "bpm": float(t.bpm or 0.0), "camelot": t.camelot, "duration_s": float(t.duration_s),
            "bar_s": bar, "landmark": lm, "body": _body_start(t, env), "bass_from_landmark": bool(bass_ok),
            "runway_after_landmark": float(t.duration_s) - lm,
            "sung": sung, "hook_starts": hook or [],
            "grooves": _instrumental_grooves(t, env)}


# -- the pairings ------------------------------------------------------------------------------------------------
def _key_fit(k_r, k_s):
    from lib.dj.brain import camelot_compat, _shift_camelot
    if not k_r or not k_s:
        return 0.6, 0
    best = (camelot_compat(k_r, k_s), 0)
    for s in SHIFTS[1:]:
        c = camelot_compat(k_r, _shift_camelot(k_s, s)) - SHIFT_COST * abs(s)
        if c > best[0] + 1e-9:
            best = (c, s)
    return best


def pairs(music):
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    db = LibraryDB(music)
    lib = [t for t in B.load_library(db) if getattr(t, "has_stems", False)]
    meas = {}
    t0 = time.time()
    for t in lib:
        p = env_path(music, t.id)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        env = {k: z[k] for k in STEMS if k in z.files}
        meas[t.id] = measure_song(t, env)
    print(f"measured {len(meas)} of {len(lib)} stem tracks in {(time.time() - t0):.0f} s")
    out = {"version": VERSION, "built": time.strftime("%Y-%m-%d %H:%M"), "records": {}}
    ms = list(meas.values())
    for r in ms:
        if not r["bpm"]:
            continue
        vo, bu = [], []
        for s in ms:
            if s["id"] == r["id"] or not s["bpm"]:
                continue
            rate = r["bpm"] / s["bpm"]                 # S played at the record's tempo
            fit, shift = _key_fit(r["camelot"], s["camelot"])
            if fit < KEY_FIT_MIN:
                continue
            # VOCAL OVER: S's sung stretch on one of R's instrumental grooves
            if VOCAL_RATE[0] <= rate <= VOCAL_RATE[1] and s["sung"] and r["grooves"]:
                tempo = 1.0 - abs(1.0 - rate) / 0.06 * 0.3
                best = None
                for a, b in s["sung"]:
                    bars = (b - a) / max(s["bar_s"], 1e-6)
                    hook = any(a - 0.5 * s["bar_s"] <= h <= b for h in s["hook_starts"])
                    for ga, gb in r["grooves"]:
                        room = (gb - ga) / max(r["bar_s"], 1e-6)
                        if room < MIN_SUNG_BARS:
                            continue
                        length = min(1.0, min(bars, room) / 16.0)
                        score = fit * tempo * (0.6 + 0.4 * length) * (1.2 if hook else 1.0)
                        if best is None or score > best["score"]:
                            best = {"id": s["id"], "title": s["title"], "score": round(score, 3), "fit": round(fit, 2), "shift": shift, "rate": round(rate, 3),
                                    "sung": [round(a, 1), round(b, 1)], "hook": hook, "groove": [round(ga, 1), round(gb, 1)]}
                if best is not None:
                    vo.append(best)
            # BASS UNDER: S's bass and melody from its landmark under R's drums, S the record after
            if BASS_RATE[0] <= rate <= BASS_RATE[1] and s["bass_from_landmark"] and s["runway_after_landmark"] / max(rate, 1e-6) >= PLAY_MIN_S + RUNWAY_S:
                tempo = 1.0 - abs(1.0 - rate) / 0.10 * 0.3
                score = fit * tempo
                bu.append({"id": s["id"], "title": s["title"], "score": round(score, 3), "fit": round(fit, 2), "shift": shift, "rate": round(rate, 3),
                           "landmark": round(s["landmark"], 1), "runway_s": round(s["runway_after_landmark"], 0)})
        vo.sort(key=lambda x: -x["score"])
        bu.sort(key=lambda x: -x["score"])
        out["records"][str(r["id"])] = {"title": r["title"], "grooves": [[round(a, 1), round(b, 1)] for a, b in r["grooves"]],
                                        "sung": [[round(a, 1), round(b, 1)] for a, b in r["sung"]], "landmark": round(r["landmark"], 1),
                                        "vocal_over": vo[:TOP_N], "bass_under": bu[:TOP_N],
                                        "n_vocal_over": len(vo), "n_bass_under": len(bu)}
    path = os.path.join(music, ".stems", "pairings.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=0)
    n = len(out["records"])
    with_vo = sum(1 for v in out["records"].values() if v["vocal_over"])
    with_bu = sum(1 for v in out["records"].values() if v["bass_under"])
    no_groove = sum(1 for v in out["records"].values() if not v["grooves"])
    print(f"pairings: {n} records; {with_vo} have a vocal that sits over them, {with_bu} have a bassline that swaps under them; "
          f"{no_groove} have no instrumental groove of 16 bars; median candidates per record: vocal {np.median([v['n_vocal_over'] for v in out['records'].values()]):.0f}, "
          f"bass {np.median([v['n_bass_under'] for v in out['records'].values()]):.0f} -> {path} in {(time.time() - t0):.0f} s")


def show(music, query):
    path = os.path.join(music, ".stems", "pairings.json")
    data = json.load(open(path, encoding="utf-8"))
    hits = [(k, v) for k, v in data["records"].items() if query.lower() in v["title"].lower()]
    for k, v in hits[:3]:
        print(f"== {v['title']} (id {k}): landmark {v['landmark']} s; instrumental grooves {v['grooves']}; sings {v['sung']}")
        print("   VOCAL OVER it:")
        for x in v["vocal_over"][:8]:
            print(f"     {x['score']:.2f}  {x['title'][:40]:40s} fit {x['fit']} shift {x['shift']:+d} rate {x['rate']} sings {x['sung']}{' (its hook)' if x['hook'] else ''} over {x['groove']}")
        print("   BASS UNDER it (then the record):")
        for x in v["bass_under"][:8]:
            print(f"     {x['score']:.2f}  {x['title'][:40]:40s} fit {x['fit']} shift {x['shift']:+d} rate {x['rate']} from {x['landmark']} s ({x['runway_s']:.0f} s after)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    if "--envelopes" in sys.argv or "--all" in sys.argv:
        envelopes(music, arg("--workers", 3, int))
    if "--pairs" in sys.argv or "--all" in sys.argv:
        pairs(music)
    if "--show" in sys.argv:
        show(music, arg("--show", ""))
