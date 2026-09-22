"""THE PAIR SMOKE (docs/DJ_MORE_THAN_A_MIXER.md 1): the pairs path driven end to end, headless - the Director starts
layered, the best pair whose record is not the record playing is taken (PLAY THIS PAIR), and the run is pumped
until that record plays and the pair's vocal comes over its groove. Three rounds of fixing the path from the
user's logs after the fact (2026-09-20) is what this replaces: it fails on the first thing that does not happen.

    python tools/tests/_dj_pair_smoke.py --music D:/Devel/music [--seed 1] [--seconds 900]

No audio device: the mixer is pulled by hand, paced at 1.5 x real time (the decodes are real seconds).
"""
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402  (idle priority for this process and its helpers: it may run beside the live planner)
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410
PACE = 1.5


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    music = arg("--music", "D:/Devel/music")
    seed = arg("--seed", 1, int)
    seconds = arg("--seconds", 900.0, float)
    D.PLAY_MIN_S["short"] = 60.0
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    d.set_dial("layers", "two")
    d.set_dial("pace", "short")
    d.set_dial("moments", "rare")                     # the DJ proposes nothing: only the pair
    if not d.start(threaded=False):
        print(f"FAIL: did not start: {d.last_error}")
        return 1
    rc = d.rc
    rc.rng = random.Random(seed)
    # the rehearsal's audio (docs/DJ_THE_BLEND.md "hear it before it plays"): every rehearsed seam is rendered to a wav
    # in the run's own folder - checked at the end (a file per rehearsed seam, not empty)
    import tempfile
    wav_dir = os.path.join(tempfile.gettempdir(), "dj_pair_smoke_seams")
    rc.seam_wav_dir = wav_dir
    for f in (os.listdir(wav_dir) if os.path.isdir(wav_dir) else []):
        try:
            os.remove(os.path.join(wav_dir, f))
        except OSError:
            pass
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    audio = []
    pace = {"t0": None, "n0": 0}

    def pump(n):
        for _ in range(n):
            audio.append(np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2).copy())
            d.step()
            if rc.master is not None:
                if pace["t0"] is None:
                    pace["t0"], pace["n0"] = time.time(), len(audio)
                lag = (len(audio) - pace["n0"]) * BLOCK / RATE / PACE - (time.time() - pace["t0"])
                if lag > 0.05:
                    time.sleep(min(lag, 0.2))

    def bar_blocks():
        return max(1, int(rc._bar_clock() / BLOCK)) if rc.master else 20

    t_wait = time.time()
    while time.time() - t_wait < 120.0:
        pump(1)
        if rc.master is not None and (rc._tel_deck(rc.master) or {}).get("playing"):
            break
    if rc.master is None:
        print("FAIL: no first song")
        return 1
    pump(2 * bar_blocks())
    rec0 = rc.songs[rc.record_deck].track
    on_record = "--on-record" in sys.argv                # the pair ON the record playing (its groove still ahead)
    if on_record and not any(p["kind"] == "vocal" and p["r"].id == rec0.id for p in d.pairs(n=400, fit_clock=True)):
        # the opener has no pair on it: the best pair's record is put UP NEXT (not the pair), and the pair is taken
        # once that record PLAYS - the on-record path proper
        cands = [p for p in d.pairs(n=400, fit_clock=True) if p["kind"] == "vocal" and p["r"].id != rec0.id and not p["played"]
                 and float(p["r_start"]) > 120.0]     # (a groove far enough in that it is still ahead when the record plays)
        if not cands:
            print("FAIL: no vocal pair whose record could be queued for the on-record case")
            return 1
        ok_q, msg_q = d.queue(cands[0]["r"].id)
        print(f"on-record case: {cands[0]['r'].title[:30]} UP NEXT ({msg_q}); the pair is taken once it plays")
        t_lim = len(audio) + int(600.0 * RATE / BLOCK)
        while len(audio) < t_lim and not (rc.record_deck in rc.songs and rc.songs[rc.record_deck].track.id == cands[0]["r"].id and rc.songs[rc.record_deck].entered):
            pump(bar_blocks())
        pump(2 * bar_blocks())
        rec0 = rc.songs[rc.record_deck].track
        if rec0.id != cands[0]["r"].id:
            print(f"FAIL: {cands[0]['r'].title[:30]} never became the record")
            return 1
    pairs = [p for p in d.pairs(n=400, fit_clock=True) if p["kind"] == "vocal" and (p["r"].id == rec0.id) == on_record
             and (not p["played"] or (on_record and p["s"].id not in d._played_ids()))]   # (the record playing counts as played)
    # --record / --part: a NAMED pair (title prefixes) - the one that failed on the night, replayed here
    want_r, want_s = arg("--record", "", str).lower(), arg("--part", "", str).lower()
    if want_r or want_s:
        pairs = [p for p in pairs if (not want_r or p["r"].title.lower().startswith(want_r)) and (not want_s or p["s"].title.lower().startswith(want_s))]
    if not pairs:
        print("FAIL: no vocal pair inside the wall" + (f" matching --record '{want_r}' --part '{want_s}' (outside the wall of {rec0.bpm:.0f}, or not offered)" if (want_r or want_s) else ""))
        return 1
    p = pairs[0]
    print(f"record now: {rec0.title[:30]} ({rec0.bpm:.0f})")
    print(f"pair: {p['words']}")
    msg = d.drop_pair(p)
    print(f"PLAY THIS PAIR -> {msg}")
    t0 = len(audio)
    seen = {"record": ("now" if on_record else None), "event_plan": None, "armed": None, "vocal_in": None, "fail": None}
    n_moves = len(rc.move_log)
    t_end = t0 + int(seconds * RATE / BLOCK)
    seen.update({"mark": None, "clip": None, "row": None})
    while len(audio) < t_end and seen["vocal_in"] is None and seen["fail"] is None:
        pump(bar_blocks())
        # THE PICTURE AND THE PLAN must carry the pair before its event is placed (user 2026-09-20: "the timeline STILL
        # doesn't show the plan"): a dashed vocals mark "YOUR PAIR", the other song's dashed clip on the voice slot once it
        # is on a deck, and a PAIR row in THE PLAN
        try:
            clk_now = f"{rc._clock() / RATE:.0f}s"
            if seen["mark"] is None and any(str(m.get("text", "")).startswith("YOUR PAIR") or str(m.get("text", "")).startswith("EVENT:") for m in d.plan_marks):
                seen["mark"] = clk_now
            slot = (getattr(d, "_ghost_slots", {}) or {}).get("voice", (None, []))
            if seen["clip"] is None and slot[1] and any(c.id in slot[1] and c.track_id == p["s"].id for c in d.tl.clips):
                seen["clip"] = clk_now
            if seen["row"] is None and any(r.get("kind") in ("PAIR", "EVENT") for r in (d.status().get("plan_rows") or [])):
                seen["row"] = clk_now
            # the NEXT ghost on the timeline is the seam's own next record, never another staged song (2026-09-21)
            sb_ = rc.songs.get(rc.seam["b"]) if rc.seam is not None else None
            if sb_ is not None and rc.seam.get("state") == "planned" and sb_.staged_at is not None and not sb_.entered and seen["fail"] is None:
                nslot = (getattr(d, "_ghost_slots", {}) or {}).get("next", (None, []))
                ghost_tids = {c.track_id for c in d.tl.clips if c.id in nslot[1]}
                if ghost_tids and sb_.track.id not in ghost_tids:
                    seen["fail"] = f"the timeline's next ghost is {sorted(ghost_tids)} but the seam's next record is {sb_.track.id} ({sb_.track.title[:24]})"
        except Exception as e:  # noqa: BLE001
            seen["fail"] = f"picture check: {type(e).__name__}: {e}"
        for m in rc.move_log[n_moves:]:
            txt = m["text"]
            clk = f"{m.get('clock_s', 0):.0f}s"
            if m["kind"] in ("bed_to", "open", "moment", "order") or "your pair" in txt or "event" in txt or "the seam" in txt or "ACAPELLA" in txt or "IDEA" in txt:
                print(f"  {clk:>6} {m['kind']:10s} {txt[:120]}")
            if m["kind"] == "bed_to" and p["r"].title[:20] in txt and seen["record"] is None:
                seen["record"] = clk
                d_rec = rc.deck_of(p["r"].id) or rc.record_deck
                t_rec = float((rc._tel_deck(d_rec) or {}).get("time_s") or 0.0)
                print(f"         -> {p['r'].title[:24]} is the record at its {int(t_rec) // 60}:{int(t_rec) % 60:02d} (the pair's groove {int(p['r_start']) // 60}:{int(p['r_start']) % 60:02d}-{int(p['r_end'] or 0) // 60}:{int(p['r_end'] or 0) % 60:02d}); decks: "
                      + ", ".join(f"{k}={s.track.title[:12]}{'(staged)' if not s.entered else ''}" for k, s in rc.songs.items()))
            if txt.startswith("event in") and "your pair" in txt:
                seen["event_plan"] = clk
            if txt.startswith("the event:") and "ACAPELLA" in txt:
                seen["armed"] = clk
            if txt.startswith("ACAPELLA") and ("vocal" in txt.lower()) and m["kind"] == "moment":
                seen["vocal_in"] = clk
            if "your pair could not be placed" in txt or "your pair missed" in txt or "the event is dropped" in txt or "the event refused" in txt \
                    or "your pair is REFUSED" in txt or "your pair FAILED" in txt:
                seen["fail"] = txt[:140]
        n_moves = len(rc.move_log)
        if d.last_error or rc.last_error:
            seen["fail"] = f"error: {d.last_error or rc.last_error}"
    # THE VOCAL IS YOURS (2026-09-21, "the vocals from another song last a pathetic amount of time"): after it comes in it
    # holds the lane until its song has sung its last - measured here for up to six phrases: the hold must still stand,
    # or have ended only because the song sang its last / ran out
    held = {"bars": 0, "end": None}
    if seen["vocal_in"] is not None and seen["fail"] is None:
        d_s = rc.deck_of(p["s"].id)
        n_moves = len(rc.move_log)
        for _k in range(48):
            pump(bar_blocks())
            if getattr(rc, "_vocal_hold", None) != d_s or d_s is None or d_s not in rc.songs:
                for m in rc.move_log[n_moves:]:
                    if "sung its last" in m["text"] or "has run out" in m["text"] or "took" in m["text"]:
                        held["end"] = m["text"][:100]
                held["end"] = held["end"] or "the hold ended (no word why)"
                break
            held["bars"] += 1
        stretch_bars = int((float(p.get("s_end") or 0) - float(p.get("s_start") or 0)) / max(rc._bar_s(p["s"].track if hasattr(p["s"], "track") else p["s"]), 1e-6)) if p.get("s_end") else 48
        rests = [m["text"][:80] for m in rc.move_log[n_moves:] if m["kind"] == "rest" and p["s"].title[:20] in m["text"] and "sung its last" not in m["text"]]
        d_s2 = rc.deck_of(p["s"].id)
        g_v = float(((rc._tel_deck(d_s2) or {}).get("stem_gains") or {}).get("vocals") or 0.0) if d_s2 else 0.0
        print(f"the vocal held its lane {held['bars']} bars" + (f", then: {held['end']}" if held["end"] else " and still holds (the run ended)")
              + f"; its lane gain {g_v:.2f}; rests in its pauses: {len(rests)}")
        if rests:
            seen["fail"] = f"your vocal was rested in its pauses ({len(rests)}x): {rests[0]}"
        if held["end"] is not None and held["bars"] < min(24, stretch_bars) and "sung its last" not in held["end"] and "run out" not in held["end"]:
            seen["fail"] = f"your vocal was let go after {held['bars']} bars: {held['end']}"
    d.stop(0.1)
    mono = np.concatenate(audio[t0:]).mean(axis=1) if len(audio) > t0 + 10 else np.zeros(1)
    nb = int(bar_blocks() * BLOCK)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(0, len(mono) - nb, nb)]
    sil = [s["clock_s"] for s in getattr(rc, "silence_log", []) if s.get("clock_s") is not None]
    dead = [i for i, r in enumerate(rms) if r < -45.0 and not any(abs((t0 * BLOCK + (i + 1) * nb) / RATE - c) < 2.0 * nb / RATE for c in sil)]
    print(f"\nrecord played at {seen['record']}, the pair planned at {seen['event_plan']}, armed at {seen['armed']}, the vocal in at {seen['vocal_in']}; dead bars {len(dead)} of {len(rms)}")
    wavs = sorted(f for f in (os.listdir(wav_dir) if os.path.isdir(wav_dir) else []) if f.endswith(".wav"))
    rehearsed = sum(1 for m in rc.move_log if "rehearsed" in m["text"])
    sizes = [os.path.getsize(os.path.join(wav_dir, f)) for f in wavs]
    print(f"the seams' audio: {len(wavs)} wav(s) for {len(rc.rehearsal_log)} rehearsals ({', '.join(f'{s // 1024} KB' for s in sizes[:6])})")
    if rc.rehearsal_log and not wavs and seen["fail"] is None:
        seen["fail"] = f"{len(rc.rehearsal_log)} seams rehearsed but no seam wav was written ({rc.last_error})"
    if any(s < 200 * 1024 for s in sizes) and seen["fail"] is None:
        seen["fail"] = "a seam wav is under 200 KB: an empty render"
    print(f"the picture: PAIR row in THE PLAN from {seen['row']}, the vocals mark from {seen['mark']}, the other song's clip from {seen['clip']}")
    if seen["fail"] is None and seen["vocal_in"] is not None and (seen["row"] is None or seen["mark"] is None or seen["clip"] is None):
        seen["fail"] = "the pair played but the plan / the picture never showed it (row %s, mark %s, clip %s)" % (seen["row"], seen["mark"], seen["clip"])
    if seen["fail"]:
        print(f"FAIL: {seen['fail']}")
        return 1
    if seen["vocal_in"] is None:
        print("FAIL: the pair's vocal never came in" + ("" if seen["record"] else " (its record never played)"))
        return 1
    if dead:
        # where the room went silent, with the moves around it and the decks' state (as the control gate prints it)
        for i in dead[:3]:
            t_dead = (t0 * BLOCK + (i + 1) * nb) / RATE
            print(f"  DEAD at {t_dead:.0f}s ({rms[i]:.0f} dB)")
            for m in rc.move_log:
                if m.get("clock_s") is not None and -6.0 < t_dead - m["clock_s"] < 30.0:
                    print(f"          {m['clock_s']:.0f}s {m['kind']:10s} {m['text'][:110]}")
        print(f"FAIL: {len(dead)} dead bars")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
