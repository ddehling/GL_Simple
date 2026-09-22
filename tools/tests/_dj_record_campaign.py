"""THE RECORD CAMPAIGN: many unattended layered nights rendered headless, measured with the instruments the
ear exam validated (docs/DJ_VERIFICATION.md), and compared - the record policy against the arrangement
policy it replaced, on identical runs, and against the autoDJ's campaign baseline where the denominators
allow.

One run = the conductor left alone for `--seconds` of music at ~2x real time (decodes are real), no
gestures. Captured: the mix, every deck's post-EQ mono, every deck's position trace, the lane map per
block, the move log, the seam log. Measured:

  per SEAM (record: first step -> the harmony swap; arrangement: a song's arrival -> its bed change)
    kick flam        seamverify.measured_kick_alignment on the two decks' low-band attacks (EAR-VALIDATED;
                     bar 45 ms; declines on diffuse material)
    dead air         min / median 0.5 s RMS over the seam +- 8 s (bar 0.15)
    level lurch      the largest 0.5 s step inside the seam vs the pair's own steps outside (bar max(6, solo+1.5))
    double bass      low band (< 130 Hz) peak during the overlap vs the solo p95 (bar +3.5 dB)
    harmony split    bars with the melody and the bass on different songs (must be 0)
    two drums        bars with two decks' drums both open past the one-bar cross (must be 0)
    level step       RMS of the four bars after the swap vs the four before the seam (dB)
    section          A's section kinds through the seam (build / breakdown are wrong there)
  per RUN
    minutes per record, lane moves per ten minutes, songs that came and went without becoming the record,
    the longest stretch of one song alone, dead bars over the whole run, clipping, the beat lock per deck,
    the vocal lane on silent sections

Usage (from the repo root, plain python):
  python tools/tests/_dj_record_campaign.py --worker --policy record --seed 3 --seconds 480 --music D:/Devel/music
  python tools/tests/_dj_record_campaign.py --campaign --policies record,arrangement --seeds 1-8 --seconds 480 --parallel 2
  python tools/tests/_dj_record_campaign.py --report
Results: logs/record_campaign/<policy>_s<seed>.json; the report prints the tables with 95 % intervals.
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402  (idle priority for this process and its helpers: it may run beside the live planner)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # (the report's arrows crashed a cp1252 console, 2026-09-21)
except Exception:
    pass
os.environ.setdefault("DJ_NO_PLAY_LOG", "1")
os.environ.setdefault("DJ_NO_MIDI", "1")
from lib.dj.submix import RATE  # noqa: E402

OUT_DIR = os.path.join("logs", "record_campaign")
BLOCK = 4410
LANES = ("drums", "bass", "other", "vocals")
AUTODJ_BASELINE = {"lurch": 0.142, "dead": 0.073, "flam45": 0.306, "double_bass": 0.022}   # docs/DJ_VERIFICATION.md, 4,667 seams


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def wilson(k, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


# ----------------------------------------------------------------------------------------------------------------
def run_one(policy, seed, seconds, music, play_min, theme):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from lib.audio_engine import AudioEngine
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    from lib.dj.remix import RemixConductor
    from lib.dj.submix import _odd_segment
    from lib.dj import seamverify as SV

    db = LibraryDB(music)
    lib = [t for t in B.load_library(db) if not t.excluded]
    rc = RemixConductor(db, music, lib, theme=theme, seed=seed)
    # the Director's layered-two settings
    rc.set_change_bars(8)
    rc.set_blend(0.3)
    rc.set_vocal_freedom(0.4)
    rc.set_auto(1.0)
    rc.fx_level = 0.0
    if policy == "record":
        rc.policy = "record"
        rc.set_play_min(play_min)
        rc.set_seam("auto", 1.0)
        rc.set_event_rate(arg("--events", 1, int))       # one EVENT per record from the pairing lists (MOMENTS some)
    else:
        rc.policy = "arrangement"
        rc.bass_moments = True
        rc.strip_before_bed = False
        rc.calm_phrases = 2
    ok = rc.start(threaded=False)
    if not ok:
        return {"error": rc.last_error}
    engine = AudioEngine()
    engine.attach_track("remix", rc.submix)
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    sub = rc.submix

    audio, deck_pcm, pos_trace, lane_hist, tel_hist = [], {d: [] for d in sub.decks}, {d: [] for d in sub.decks}, [], []
    errs = {}
    deck_track = {d: [] for d in sub.decks}          # (clock, track_id) whenever it changes

    def _wrap(orig, nm):
        def f(n):
            blk = orig(n)
            deck_pcm[nm].append((sub.clock, blk.mean(axis=1).astype(np.float32)))
            return blk
        return f
    for nm, d in sub.decks.items():
        d.read = _wrap(d.read, nm)

    bpm_hist = []                                     # (clock_s, master_bpm) once a second: the clock's travel (docs/DJ_THE_SET.md rule 3)

    def pump(n_blocks):
        for _ in range(n_blocks):
            audio.append(np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2).copy())
            rc.step()
            if rc.master is not None and len(audio) % 10 == 0:
                bpm_hist.append((sub.clock / RATE, float(rc.master_bpm)))
            tel = sub.telemetry or {}
            decks = tel.get("decks") or {}
            for nm, d in sub.decks.items():
                if d.playing:
                    pos_trace[nm].append((sub.clock, d.source_time_s()))
                    tid = d.track_id
                    if not deck_track[nm] or deck_track[nm][-1][1] != tid:
                        deck_track[nm].append((sub.clock, tid))
            sync = tel.get("sync") or {}
            views = sync.get("slaves") or ({sync["slave"]: sync} if sync else {})
            for sl, v in views.items():
                m, s = decks.get(v["master"]) or {}, decks.get(sl) or {}
                if not (m.get("playing") and s.get("playing")):
                    continue
                if _odd_segment(sub.decks.get(sl)) or _odd_segment(sub.decks.get(v["master"])):
                    continue
                e = (float(s["beat_phase"]) - float(m["beat_phase"]) - float(v.get("bias_beats") or 0.0) + 0.5) % 1.0 - 0.5
                errs.setdefault(sl, []).append(abs(e))
            lane_hist.append((sub.clock, tuple(rc.lanes[s] for s in LANES),
                              tuple((rc.songs[rc.lanes[s]].track.id if rc.lanes[s] in rc.songs else None) for s in LANES)))
            if len(audio) % 10 == 0:
                tel_hist.append((sub.clock, {d: {"g": float(t.get("gain") or 0.0), "st": {k: float(v) for k, v in (t.get("stem_gains") or {}).items()},
                                               "play": bool(t.get("playing")), "t": float(t.get("time_s") or 0.0), "tid": t.get("track_id")}
                                             for d, t in decks.items()}))

    t0 = time.time()
    while rc.master is None and time.time() - t0 < 90:
        pump(1)
        time.sleep(0.02)
    if rc.master is None:
        return {"error": "the first song never opened", "last_error": rc.last_error}
    opened_clock = sub.clock
    total_blocks = int(seconds * RATE) // BLOCK
    t_start, done = time.time(), 0
    while done < total_blocks:
        pump(4)
        done += 4
        lag = done * BLOCK / RATE / 2.0 - (time.time() - t_start)
        if lag > 0:
            time.sleep(min(lag, 0.2))
    rc.stop(fade_s=0.5)
    pump(10)

    # ---- assemble ---------------------------------------------------------------------------------------------
    mix = np.concatenate(audio, axis=0)
    c0 = sub.clock - len(mix)
    mono = mix.mean(axis=1).astype(np.float64)

    def deck_array(nm):
        arr = np.zeros(len(mono), dtype=np.float32)
        for clk, blk in deck_pcm[nm]:
            i0 = int(clk - c0)
            if 0 <= i0 and i0 + len(blk) <= len(arr):
                arr[i0:i0 + len(blk)] = blk
        return arr
    darr = {nm: deck_array(nm) for nm in sub.decks}
    deck_pcm.clear()
    bar_s = rc._bar_s(rc.songs[rc.master].track) if rc.master in rc.songs else 2.0
    bar_c = int(bar_s * RATE)

    def clock_of_move(m):
        return int(round((m.get("clock_s") or 0.0) * RATE))

    def title_of(tid):
        t = rc._track(tid) if tid is not None else None
        return t.title if t else "?"

    def deck_of_track_at(tid, clk):
        for nm, rows in deck_track.items():
            cur = None
            for c, t in rows:
                if c <= clk:
                    cur = t
                else:
                    break
            if cur == tid:
                return nm
        return None

    def rms_windows(x, w):
        n = len(x) // w
        return np.sqrt(np.mean(x[:n * w].reshape(-1, w) ** 2, axis=1)) if n else np.zeros(0)

    from scipy.signal import butter, sosfilt
    sos_low = butter(4, 130.0, fs=RATE, output="sos")

    # ---- the seams -------------------------------------------------------------------------------------------------
    ml = rc.move_log
    seams = []
    if policy == "record":
        starts = [m for m in ml if m["kind"] == "voice_in"]
        ends = [m for m in ml if m["kind"] == "bed_to" and ("the bed passes to" in m["text"] or "the record is its" in m["text"])]
    else:
        starts = [m for m in ml if m["kind"] == "voice_in"]
        ends = [m for m in ml if m["kind"] == "bed_to"]
    came_and_went = 0
    for st in starts:
        b_tid = st.get("to_id")
        end = next((e for e in ends if e.get("to_id") == b_tid and e["t"] >= st["t"]), None)
        if end is None:
            came_and_went += 1
            continue
        if policy == "arrangement":
            # the LAST bed_to for this song within two minutes closes the change (the follow)
            later = [e for e in ends if e.get("to_id") == b_tid and st["t"] <= e["t"] <= end["t"] + 120]
            end = later[-1] if later else end
        a_tid = st.get("from_id")
        blend_c, swap_c = clock_of_move(st) + bar_c, clock_of_move(end) + bar_c
        if swap_c <= blend_c or swap_c - c0 + 8 * RATE > len(mono):
            continue
        A, Bd = deck_of_track_at(a_tid, blend_c), deck_of_track_at(b_tid, swap_c)
        a_track, b_track = rc._track(a_tid) if a_tid else None, rc._track(b_tid) if b_tid else None
        rec = {"a": title_of(a_tid), "b": title_of(b_tid), "blend_s": round((blend_c - c0) / RATE, 1), "swap_s": round((swap_c - c0) / RATE, 1),
               "len_bars": round((swap_c - blend_c) / bar_c, 1), "text": st["text"][:90]}
        w0, w1 = max(0, blend_c - c0 - 8 * RATE), min(len(mono), swap_c - c0 + 8 * RATE)
        seg = mono[w0:w1]
        w = RATE // 2
        r = rms_windows(seg, w)
        if len(r) > 4:
            act = r[1:]
            rec["rms_min_ratio"] = round(float(act.min() / max(np.median(act), 1e-9)), 3)
            steps = 20 * np.abs(np.diff(np.log10(np.maximum(act, 1e-6))))
            i0 = max(0, int((blend_c - c0 - w0) / w) - 1)
            i1 = min(len(steps), int((swap_c - c0 - w0) / w) + 1)
            inside, outside = steps[i0:i1], np.concatenate([steps[:i0], steps[i1:]])
            rec["lurch_db"] = round(float(inside.max()) if len(inside) else 0.0, 1)
            rec["lurch_solo_db"] = round(float(outside.max()) if len(outside) else 0.0, 1)
            # the anatomy: the 0.5 s levels (dBFS) from 8 s before the seam to 8 s after the swap, and where the
            # three biggest steps inside the seam sit, in bars from the seam's first bar (negative = before the swap)
            rec["rms_db"] = [round(float(20 * np.log10(v + 1e-9)), 1) for v in act]
            rec["t0_rel_s"] = round((w0 - (blend_c - c0)) / RATE + 0.5, 1)
            top = sorted(range(i0, i1), key=lambda k: -steps[k])[:3] if i1 > i0 else []
            rec["worst_steps"] = [(round(((k + 1.5) * w + w0 - (blend_c - c0)) / bar_c, 1), round(float(steps[k]), 1),
                                   round(float(20 * np.log10(act[k] + 1e-9)), 1), round(float(20 * np.log10(act[k + 1] + 1e-9)), 1)) for k in top]
            rec["swap_bar"] = round((swap_c - blend_c) / bar_c, 1)
        low = sosfilt(sos_low, seg)
        lr = rms_windows(low, w)
        b0, s0 = int((blend_c - c0 - w0) / w), int((swap_c - c0 - w0) / w)
        pre, mid, post = lr[1:max(b0, 2)], lr[b0:max(s0, b0 + 1)], lr[s0:]
        if len(pre) and len(mid) and len(post):
            solo = max(np.percentile(pre, 95), np.percentile(post, 95))
            rec["bass_bump_db"] = round(float(20 * np.log10(max(mid.max(), 1e-9) / max(solo, 1e-9))), 1)
        # level step: four bars after the swap vs four before the seam
        pre4 = mono[max(0, blend_c - c0 - 4 * bar_c):blend_c - c0]
        post4 = mono[swap_c - c0:min(len(mono), swap_c - c0 + 4 * bar_c)]
        if len(pre4) > RATE and len(post4) > RATE:
            rec["level_step_db"] = round(float(20 * np.log10((np.sqrt(np.mean(post4 ** 2)) + 1e-9) / (np.sqrt(np.mean(pre4 ** 2)) + 1e-9))), 1)
        # harmony split and two drums, from the lane map and the gains
        split = sum(1 for clk, lanes, tids in lane_hist if blend_c - bar_c <= clk <= swap_c + 2 * bar_c and tids[1] is not None and tids[2] is not None and tids[1] != tids[2])
        rec["harmony_split_bars"] = round(split * BLOCK / bar_c, 2)
        two_dr = 0
        for clk, pic in tel_hist:
            if blend_c + bar_c <= clk <= swap_c + 2 * bar_c:
                open_dr = [d for d, p in pic.items() if p["play"] and p["g"] > 0.3 and p["st"].get("drums", 0.0) > 0.5]
                if len(open_dr) >= 2:
                    two_dr += 1
        rec["two_drums_s"] = round(two_dr * 10 * BLOCK / RATE, 1)
        # A's sections through the seam
        if a_track is not None and A is not None:
            kinds = []
            for clk, tsrc in pos_trace.get(A, []):
                if blend_c <= clk <= swap_c:
                    kinds.append((a_track.section_at(tsrc) or {}).get("kind"))
            if kinds:
                rec["a_bad_frac"] = round(sum(1 for k in kinds if k in ("build", "breakdown")) / len(kinds), 2)
                rec["a_kinds"] = sorted(set(k for k in kinds if k))
        # the kick flam, on the two decks' own low-band attacks
        if A is not None and Bd is not None and a_track is not None and b_track is not None:
            try:
                marks = {"blend_s": (blend_c - c0) / RATE, "swap_s": (swap_c - c0) / RATE,
                         "pos": {"a": [((c - c0) / RATE, s) for c, s in pos_trace[A] if blend_c - 2 * RATE <= c <= swap_c + 2 * RATE],
                                 "b": [((c - c0) / RATE, s) for c, s in pos_trace[Bd] if blend_c - 2 * RATE <= c <= swap_c + 2 * RATE]}}
                rec["kick"] = SV.measured_kick_alignment({"a": darr[A], "b": darr[Bd]}, marks, a_track, b_track)
            except Exception as e:  # noqa: BLE001
                rec["kick_err"] = f"{type(e).__name__}: {e}"
        seams.append(rec)

    # ---- the run -----------------------------------------------------------------------------------------------
    mono_r = mono[max(0, opened_clock - c0):]
    nb = bar_c
    rms_b = [20 * np.log10(np.sqrt(np.mean(mono_r[i:i + nb] ** 2)) + 1e-9) for i in range(nb, len(mono_r) - nb, nb)]
    changes = [m for m in ml if (m["kind"] == "bed_to" and ("the bed passes to" in m["text"] or "the record is its" in m["text"] or " follow" in m["text"]))
               or m["kind"] in ("drop",)]
    # minutes per record: the intervals between the bars a new song became the record (its drums), from the open
    rec_times = [opened_clock / RATE] + [m["clock_s"] for m in changes if m.get("clock_s") is not None]
    gaps = [b - a for a, b in zip(rec_times, rec_times[1:]) if b > a]
    lane_moves = [m for m in ml if m["kind"] in ("enter", "cross", "cross_new", "rest", "return", "voice_in", "bed_to", "voice_out", "evict", "seam")]
    # the longest stretch of one song alone (blocks with one entered song and every lane on it)
    longest, cur = 0, 0
    for clk, lanes, tids in lane_hist:
        held = {t for t in tids if t is not None}
        if len(held) == 1:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    lock = {}
    for sl, rows in errs.items():
        if len(rows) >= 20:
            lock[sl] = {"med": round(float(np.median(rows)), 3), "p95": round(float(np.percentile(rows, 95)), 3), "n": len(rows)}
    dead_where = []
    sil = [s["clock_s"] for s in getattr(rc, "silence_log", []) if s.get("clock_s") is not None]
    own_silence = 0
    for i, r_db in enumerate(rms_b):
        if r_db < -40.0:
            t_dead = (max(0, opened_clock - c0) + (i + 1) * nb) / RATE
            if any(abs(t_dead - c) < 2.0 * nb / RATE for c in sil):
                own_silence += 1                      # the record's own silent breakdown (silence_log), counted apart
                continue
            near = [m["text"][:70] for m in ml if m.get("clock_s") is not None and abs(m["clock_s"] - t_dead) < 12.0]
            dead_where.append({"t": round(t_dead), "db": round(r_db, 1), "near": near[-3:]})
    events = [m["text"][:110] for m in ml if m["kind"] in ("event_plan", "event")]
    acap = [m["text"][:110] for m in ml if m["kind"] == "moment" and m["text"].startswith("ACAPELLA")]
    travels = [m["text"][:110] for m in ml if m["kind"] == "clock" and "the clock travels" in m["text"]]
    bpms = [b for _t, b in bpm_hist]
    clock = {"travels": len(travels), "opener_bpm": round(bpms[0], 1) if bpms else None, "min_bpm": round(min(bpms), 1) if bpms else None,
             "max_bpm": round(max(bpms), 1) if bpms else None, "end_bpm": round(bpms[-1], 1) if bpms else None, "words": travels[:6]}
    out = {"policy": policy, "seed": seed, "seconds": seconds, "play_min": play_min, "theme": theme, "clock": clock,
           "events": events, "acapella_steps": len(acap), "event_log": list(getattr(rc, "event_log", [])), "dead_where": dead_where[:8],
           "rehearsals": [{k: v for k, v in x.items() if k in ("ok", "why", "order", "len")} for x in getattr(rc, "rehearsal_log", [])],
           "records": len(gaps) + 1, "min_per_record": round(float(np.mean(gaps)) / 60.0, 2) if gaps else round(seconds / 60.0, 2),
           "shortest_record_s": round(float(min(gaps)), 0) if gaps else None,
           "lane_moves_per_10min": round(10.0 * len(lane_moves) / max(seconds / 60.0, 1e-6), 1),
           "came_and_went": came_and_went, "longest_alone_s": round(longest * BLOCK / RATE, 0),
           "dead_bars": int(sum(1 for r in rms_b if r < -40.0)) - own_silence, "own_silence_bars": own_silence,
           "quiet_bars": int(sum(1 for r in rms_b if r < -30.0)), "bars": len(rms_b),
           "median_bar_db": round(float(np.median(rms_b)), 1) if rms_b else None,
           "peak": round(float(np.abs(mix).max()), 3), "clipped": int((np.abs(mix) > 0.999).sum()),
           "lock": lock, "seams": seams, "seam_log": list(getattr(rc, "seam_log", [])),
           "plan_moves": sum(1 for m in ml if m["kind"] == "plan"), "moves": [m["text"][:100] for m in ml][-60:], "error": rc.last_error}
    return out


# ----------------------------------------------------------------------------------------------------------------
def report(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            rows.append(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    rows = [r for r in rows if not r.get("error") or r.get("seams") is not None]
    if not rows:
        print("no results in", out_dir)
        return
    pols = sorted({r["policy"] for r in rows})

    def fmt_rate(k, n):
        p, lo, hi = wilson(k, n)
        return f"{100 * p:5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}] n={n}"
    print(f"== {len(rows)} runs: " + ", ".join(f"{p} x{sum(1 for r in rows if r['policy'] == p)}" for p in pols))
    print("\n-- per run (mean over runs; a DJ: 4-5 min per record, one seam each) --")
    hdr = f"{'policy':12s} {'min/record':>10s} {'moves/10min':>11s} {'came&went':>9s} {'alone max s':>11s} {'dead bars':>9s} {'quiet bars':>10s} {'clip':>5s} {'lock p95':>8s}"
    print(hdr)
    for p in pols:
        rs = [r for r in rows if r["policy"] == p]
        mpr = np.mean([r["min_per_record"] for r in rs])
        mv = np.mean([r["lane_moves_per_10min"] for r in rs])
        cw = sum(r["came_and_went"] for r in rs)
        alone = np.mean([r["longest_alone_s"] for r in rs])
        dead = sum(r["dead_bars"] for r in rs)
        quiet = sum(r["quiet_bars"] for r in rs)
        bars = sum(r["bars"] for r in rs)
        clip = sum(r["clipped"] for r in rs)
        p95s = [v["p95"] for r in rs for v in (r.get("lock") or {}).values()]
        print(f"{p:12s} {mpr:10.2f} {mv:11.1f} {cw:9d} {alone:11.0f} {dead:5d}/{bars:<4d} {quiet:5d}/{bars:<4d} {clip:5d} {max(p95s) if p95s else 0:8.3f}")
    print("\n-- per seam (defect rates with 95 % Wilson intervals; the autoDJ campaign baseline from docs/DJ_VERIFICATION.md) --")
    print(f"{'policy':12s} {'seams':>5s} {'beat lock p95 (beats)':>22s} {'dead air<0.15':>26s} {'lurch':>26s} {'double bass>3.5':>26s} {'harmony split':>14s} {'two drums':>10s} {'A build/brk':>11s}")
    for p in pols:
        ss = [s for r in rows if r["policy"] == p for s in r["seams"]]
        n = len(ss)
        dead = sum(1 for s in ss if s.get("rms_min_ratio") is not None and s["rms_min_ratio"] < 0.15)
        lurch = sum(1 for s in ss if s.get("lurch_db") is not None and s["lurch_db"] > max(6.0, (s.get("lurch_solo_db") or 0.0) + 1.5))
        dbass = sum(1 for s in ss if s.get("bass_bump_db") is not None and s["bass_bump_db"] > 3.5)
        split = sum(1 for s in ss if (s.get("harmony_split_bars") or 0) > 1.0)
        twod = sum(1 for s in ss if (s.get("two_drums_s") or 0) > 1.5 * 2.0)
        abad = sum(1 for s in ss if (s.get("a_bad_frac") or 0) > 0.25)
        p95s = [v["p95"] for r in rows if r["policy"] == p for v in (r.get("lock") or {}).values()]
        lock_txt = f"median {np.median(p95s):.3f} max {max(p95s):.3f}" if p95s else "-"
        print(f"{p:12s} {n:5d} {lock_txt:>22s} {fmt_rate(dead, n):>26s} {fmt_rate(lurch, n):>26s} {fmt_rate(dbass, n):>26s} {split:>10d}/{n:<3d} {twod:>6d}/{n:<3d} {abad:>7d}/{n:<3d}")
    b = AUTODJ_BASELINE
    print(f"{'autoDJ 08-18':12s} {4667:5d} {'(kick flam 30.6 %)':>22s} {100 * b['dead']:25.1f}% {100 * b['lurch']:25.1f}% {100 * b['double_bass']:25.1f}%   (one song at a time: no lanes)")
    # the kick-flam instrument is EAR-VALIDATED on full-mix blends (docs/DJ_VERIFICATION.md) and NOT on stem seams:
    # after a drum swap the old deck's low band is its bassline, not a kick, and it declines on most - indicative only
    for p in pols:
        ss = [s for r in rows if r["policy"] == p for s in r["seams"]]
        meas = [s for s in ss if s.get("kick")]
        flam = sum(1 for s in meas if s["kick"]["flam_med_ms"] > 45.0)
        print(f"{'':12s} ({p}: kick flam > 45 ms on {flam} of {len(meas)} measurable of {len(ss)} - the instrument is not validated for stem seams; the beat lock above is the sync number)")
    print("\n-- seam shape --")
    for p in pols:
        ss = [s for r in rows if r["policy"] == p for s in r["seams"]]
        if not ss:
            continue
        lens = [s["len_bars"] for s in ss]
        steps = [s["level_step_db"] for s in ss if s.get("level_step_db") is not None]
        flams = [s["kick"]["flam_med_ms"] for s in ss if s.get("kick")]
        print(f"{p:12s} length bars median {np.median(lens):.0f} (min {min(lens):.0f}, max {max(lens):.0f}); level step after vs before: median {np.median(steps) if steps else 0:+.1f} dB, "
              f"worst {max(np.abs(steps)) if steps else 0:.1f} dB; kick flam median {np.median(flams) if flams else float('nan'):.0f} ms over {len(flams)} measurable of {len(ss)}")
        if p == "record":
            sl = [x for r in rows if r["policy"] == p for x in r.get("seam_log", [])]
            if sl:
                errs = [abs(x["err_bars"]) for x in sl if x.get("on_lm") and x.get("moved", 0) == 0]
                offs = [-x["err_bars"] for x in sl if not x.get("on_lm")]
                orders = {o: sum(1 for x in sl if x["order"] == o) for o in ("drums", "bass", "vocal", "cut")}
                print(f"{'':12s} orders {orders}; landmark on the completion bar: {len(errs)} placed there, median error {np.median(errs) if errs else float('nan'):.1f} bars, "
                      f"within a bar {sum(1 for e in errs if e <= 1.0)}/{len(errs)}; the other {len(offs)}: the drop {np.median(offs) if offs else 0:+.0f} bars from the completion (median)")
            whys = {}
            for r in rows:
                if r["policy"] != p:
                    continue
                for m in r["moves"]:
                    if m.startswith("the seam:") and " - " in m:
                        w = m.split(" - ", 1)[1].split(" (")[0]
                        w = "into A's section" if w.startswith("into ") else ("before A runs out" if w.startswith("before ") else w)
                        whys[w] = whys.get(w, 0) + 1
            print(f"{'':12s} where the seams were placed: {whys}")
    print("\n-- the events (docs/DJ_PAIRINGS.md) --")
    for p in pols:
        rs = [r for r in rows if r["policy"] == p]
        planned = sum(1 for r in rs for e in r.get("events", []) if e.startswith("event in"))
        armed = sum(1 for r in rs for e in r.get("events", []) if e.startswith("the event:"))
        refused = [e for r in rs for e in r.get("events", []) if e.startswith("the event refused")]
        acap = sum(r.get("acapella_steps", 0) for r in rs)
        recs = sum(r["records"] for r in rs)
        print(f"{p:12s} records {recs}, events planned {planned}, armed {armed}, refused {len(refused)}, acapella steps {acap}"
              + (f"; e.g. refused: {refused[0][:80]}" if refused else ""))
        for e in [e for r in rs for e in r.get("events", []) if e.startswith("event in")][:4]:
            print(f"{'':12s}   {e[:110]}")
    print("\n-- the clock (docs/DJ_THE_SET.md rule 3: it follows each record to its own tempo, half a percent a bar) --")
    for p in pols:
        rs = [r for r in rows if r["policy"] == p and r.get("clock")]
        if not rs:
            continue
        spans = [(r["clock"]["max_bpm"] or 0) - (r["clock"]["min_bpm"] or 0) for r in rs]
        print(f"{p:12s} travels {sum(r['clock']['travels'] for r in rs)} over {len(rs)} runs; span per run median {np.median(spans):.1f} bpm, max {max(spans):.1f}; "
              + "; ".join(f"seed {r['seed']}: {r['clock']['opener_bpm']} → {r['clock']['end_bpm']}" for r in rs[:8]))
    print("\n-- the rehearsals (lib/dj/rehearse.py: the seam rendered from the decks' stems two phrases ahead) --")
    for p in pols:
        rs = [r for r in rows if r["policy"] == p]
        rh = [x for r in rs for x in r.get("rehearsals", [])]
        clean = sum(1 for x in rh if x.get("ok") is True)
        failed = [x for x in rh if x.get("ok") is False]
        skipped = sum(1 for x in rh if x.get("ok") is None)
        print(f"{p:12s} rehearsed {len(rh)}: clean {clean}, failed (moved / reshaped) {len(failed)}, not rendered {skipped}")
        for x in failed[:5]:
            print(f"{'':12s}   {x.get('order')} {x.get('len')} bars: {str(x.get('why'))[:100]}")
        for x in [x for x in rh if x.get("ok") is None][:2]:
            print(f"{'':12s}   not rendered: {str(x.get('why'))[:100]}")
    print("\n-- dead bars (under -40 dBFS), where --")
    for p in pols:
        for r in [r for r in rows if r["policy"] == p]:
            for dw in r.get("dead_where") or []:
                print(f"  {p} seed {r['seed']}: {dw['t']} s ({dw['db']} dB) near: " + " | ".join(dw["near"]))
    print("\n-- the worst seams (for the ear) --")
    for p in pols:
        ss = [(r["seed"], s) for r in rows if r["policy"] == p for s in r["seams"]]
        bad = [(sd, s) for sd, s in ss if (s.get("rms_min_ratio") or 1) < 0.15 or (s.get("lurch_db") or 0) > max(6.0, (s.get("lurch_solo_db") or 0) + 1.5)
               or (s.get("kick") or {}).get("flam_med_ms", 0) > 45 or (s.get("bass_bump_db") or 0) > 3.5 or (s.get("harmony_split_bars") or 0) > 1.0]
        for sd, s in bad[:6]:
            k = s.get("kick") or {}
            print(f"  {p} seed {sd}: {s['a'][:22]} -> {s['b'][:22]} @{s['blend_s']:.0f}s {s['len_bars']:.0f} bars | dead {s.get('rms_min_ratio')} lurch {s.get('lurch_db')}/{s.get('lurch_solo_db')} "
                  f"bass {s.get('bass_bump_db')} flam {k.get('flam_med_ms')} split {s.get('harmony_split_bars')} A {s.get('a_kinds')}")


# ----------------------------------------------------------------------------------------------------------------
def campaign(policies, seeds, seconds, parallel, music, play_min, theme, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(p, s) for p in policies for s in seeds]
    running = []
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < parallel:
            p, s = jobs.pop(0)
            out = os.path.join(out_dir, f"{p}_s{s}.json")
            if os.path.exists(out):
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--policy", p, "--seed", str(s), "--seconds", str(seconds),
                   "--music", music, "--play-min", str(play_min), "--theme", theme, "--out", out]
            log = open(out[:-5] + ".log", "w", encoding="utf-8")
            running.append((p, s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[{time.time() - t0:6.0f}s] started {p} seed {s}", flush=True)
        for r in list(running):
            if r[2].poll() is not None:
                r[3].close()
                print(f"[{time.time() - t0:6.0f}s] done {r[0]} seed {r[1]} (exit {r[2].returncode})", flush=True)
                running.remove(r)
        time.sleep(2.0)
    print(f"campaign done in {(time.time() - t0) / 60:.1f} min")
    report(out_dir)


if __name__ == "__main__":
    music = arg("--music", "D:/Devel/music")
    out_dir = arg("--out-dir", OUT_DIR)
    if "--worker" in sys.argv:
        res = run_one(arg("--policy", "record"), arg("--seed", 1, int), arg("--seconds", 480.0, float), music,
                      arg("--play-min", 210.0, float), arg("--theme", "groove"))
        out = arg("--out", os.path.join(out_dir, f"{arg('--policy', 'record')}_s{arg('--seed', 1, int)}.json"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w", encoding="utf-8"), indent=1, default=float)
        print(json.dumps({k: v for k, v in res.items() if k not in ("seams", "moves", "seam_log")}, default=float))
        for s in res.get("seams") or []:
            print(" seam", json.dumps(s, default=float)[:300])
    elif "--report" in sys.argv:
        report(out_dir)
    elif "--nightly" in sys.argv:
        # the same eight seeds every night into a dated folder, one summary line appended to logs/record_campaign_nightly.log:
        # a regression shows the next morning (schedule: python tools/tests/_dj_record_campaign.py --nightly --music D:/Devel/music)
        import contextlib
        import io
        day = time.strftime("%Y%m%d")
        out_dir = os.path.join("logs", "record_campaign_nightly", day)
        campaign(["record"], list(range(1, 9)), 720.0, arg("--parallel", 3, int), music, arg("--play-min", 210.0, float), arg("--theme", "groove"), out_dir)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(out_dir)
        txt = buf.getvalue()
        line = next((ln for ln in txt.splitlines() if ln.startswith("record ") and "%" in ln), "")
        run = next((ln for ln in txt.splitlines() if ln.startswith("record ") and "%" not in ln), "")
        with open(os.path.join("logs", "record_campaign_nightly.log"), "a", encoding="utf-8") as f:
            f.write(f"{day}  RUN {run.strip()}\n{day}  SEAM {line.strip()}\n")
        print(txt)
    else:
        pols = arg("--policies", "record,arrangement").split(",")
        sd = arg("--seeds", "1-8")
        if "-" in sd:
            a, b = sd.split("-")
            seeds = list(range(int(a), int(b) + 1))
        else:
            seeds = [int(x) for x in sd.split(",")]
        campaign(pols, seeds, arg("--seconds", 480.0, float), arg("--parallel", 2, int), music, arg("--play-min", 210.0, float), arg("--theme", "groove"), out_dir)
