"""THE CONTROL GATE - "a control system needs to respond to controls" (2026-09-16).

The contract: every control the Director offers does ONE of two things within a bar and a bit - the thing
you asked, or one line in the story row saying why, with a reason from the short allowed list (the song is
not on a deck, still decoding, and the few honest ones). No silent failure, no third outcome, no exception
swallowed into last_error.

The gate drives the Director headless (no audio device: the mixer is pulled by hand, as the remix gate does)
in layered mode and presses random controls against random songs in whatever state the machine is in -
staged, playing, looping, mid-seam, mid-moment, held, a song with a late hook, a song near its end - many
songs, many seeds. It fails on the first control that neither acted nor spoke, spoke a reason outside the
list, or raised.

    python tools/tests/_dj_control_gate.py --music D:/Devel/music [--seeds 1-3] [--presses 30] [--seconds 420]
"""
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410
PACE = 1.5                                 # rendered seconds per real second (decodes are real seconds)
LANES = ("drums", "bass", "other", "vocals")
MOMENTS = ("slam", "break", "sing", "takeover", "bridge")
# the reasons a control may give instead of acting - anything else is a refusal the orders rewrite missed
ALLOWED = ("not on a deck", "no such song", "still decoding", "still staging", "no clock", "press START", "nothing armed",
           "a seam is running", "in flight", "neither drums nor bass", "looping its last bars", "no second song",
           "never two singers", "is running out", "outside the tempo wall", "no free deck", "no bed", "needs songs layered",
           "not layered", "layer first", "only TAKEOVER and BRIDGE", "already", "nothing to", "no drop or hook",
           "would take nothing away", "too quiet", "the vocal you brought", "is the one playing", "nothing to take away",
           "vocal is not on the lane", "is not singing here", "no hook of")


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def seeds_of(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


class Press:
    def __init__(self, name, target, verify, settle_bars=1.5, wait_pending_s=45.0):
        self.name, self.target, self.verify, self.settle_bars, self.wait_pending_s = name, target, verify, settle_bars, wait_pending_s


def run(music, seed, presses, seconds, verbose):
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    D.PLAY_MIN_S["short"] = 60.0                      # a short minimum play: many seams in a few rendered minutes
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    d.set_dial("layers", "two")
    d.set_dial("pace", "short")
    d.set_dial("moments", "some")
    ok = d.start(threaded=False)
    if not ok:
        return {"seed": seed, "fatal": f"did not start: {d.last_error}", "rows": []}
    rc = d.rc
    rc.rng = random.Random(seed)                      # the conductor's dice by the seed too
    rng = random.Random(1000 + seed)
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    audio = []
    errors = []
    seen_err = {"d": None, "rc": None}

    pace = {"t0": None, "n0": 0}
    tel_hist = []                                     # (block index, master, {deck: (gain, stem gains, playing, time_s, title, held)}) once a second

    def pump(n):
        # paced at PACE x real time once the first song plays: decodes and stagings are real seconds, and a run
        # rendered at five times real time starved the decks (four songs decoding, the record ran out into silence)
        for _ in range(n):
            audio.append(np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2).copy())
            d.step()
            if len(audio) % 10 == 0:
                decks = (rc.submix.telemetry or {}).get("decks") or {}
                tel_hist.append((len(audio), rc.master, {dk: (round(float(t.get("gain") or 0.0), 2), {k: round(float(v), 2) for k, v in (t.get("stem_gains") or {}).items()},
                                                          bool(t.get("playing")), round(float(t.get("time_s") or 0.0), 1),
                                                          (rc.songs[dk].track.title[:14] + ("(leaving)" if rc.songs[dk].leaving else ("(staged)" if not rc.songs[dk].entered else "")) if dk in rc.songs else "-"),
                                                          (rc.songs[dk].held(rc.lanes) if dk in rc.songs else [])) for dk, t in decks.items()}))
            if rc.master is not None:
                if pace["t0"] is None:
                    pace["t0"], pace["n0"] = time.time(), len(audio)
                lag = (len(audio) - pace["n0"]) * BLOCK / RATE / PACE - (time.time() - pace["t0"])
                if lag > 0.05:
                    time.sleep(min(lag, 0.2))
            for key, obj in (("d", d), ("rc", rc)):
                e = getattr(obj, "last_error", None)
                if e and e != seen_err[key]:
                    seen_err[key] = e
                    errors.append((len(audio) * BLOCK / RATE, key, e))

    def bar_blocks():
        return max(1, int(rc._bar_clock() / BLOCK)) if rc.master else 20

    # the first song must open (a decode is real seconds: wait on the clock, not on rendered blocks)
    t_wait = time.time()
    while time.time() - t_wait < 120.0:
        pump(1)
        if rc.master is not None and (rc._tel_deck(rc.master) or {}).get("playing"):
            break
        if len(audio) % 50 == 0:
            time.sleep(0.05)
    if rc.master is None:
        d.stop(0.1)
        return {"seed": seed, "fatal": "no first song", "rows": []}
    pump(4 * bar_blocks())
    opened_at = len(audio)

    stems = [t for t in d.library if getattr(t, "has_stems", False) and t.bpm]
    late_hook = [t for t in stems if getattr(t, "hook", None) and float(t.duration_s) - float(t.hook["start_s"]) < 90.0]

    def any_song():
        pool = late_hook if (late_hook and rng.random() < 0.3) else stems
        return rng.choice(pool)

    def on_deck_song():
        cands = [s.track for s in rc.songs.values() if not s.leaving and s.staged_at is not None]
        return rng.choice(cands) if cands else None

    def deck_of(tid):
        return rc.deck_of(int(tid))

    def state_words():
        sm = rc.seam
        mo = rc.moment_status()
        rec = rc.songs.get(rc.record_deck)
        return ", ".join(x for x in [
            f"record {rec.track.title[:16]}" if rec else "no record",
            None if rc.master else "NO MASTER",
            (f"seam {sm['state']} ({sm['order']} in {sm['bar'] - rc.bar_n} bars)" if sm else "no seam"),
            (f"moment {mo['label']} {'running' if mo['started'] else 'armed'}" if mo else None),
            "held" if rc.hold else None,
            "record looping" if rec is not None and rec.loop is not None else None,
            f"{len(rc.songs)} on decks"] if x)

    def make_press():
        kind = rng.choice(["stem_in", "stem_in", "stem_in", "vocal_late", "queue", "next", "next", "hold", "drop", "break", "moment", "moment", "moment_with",
                           "stem_out", "save_recall", "dial"])
        if kind in ("stem_in", "vocal_late"):
            if kind == "vocal_late" and late_hook:
                t, lane = rng.choice(late_hook), "vocals"      # tonight's case: a vocal from a song whose hook is too near its end
            else:
                t = on_deck_song() if rng.random() < 0.4 else any_song()
                if t is None:
                    t = any_song()
                lane = rng.choice(LANES)

            def verify(_before):
                dk = deck_of(t.id)
                return dk is not None and rc.lanes.get(lane) == dk
            return Press(f"stem_in {lane} of {t.title[:22]}", lambda: d.stem_in(t.id, lane), verify)
        if kind == "queue":
            t = any_song()

            def verify(_before):
                return int(t.id) in d.up_next or deck_of(t.id) is not None
            return Press(f"UP NEXT {t.title[:22]}", lambda: d.queue(t.id), verify, settle_bars=0.5)
        if kind == "next":
            before_seams = len(rc.seam_log)

            def verify(before):
                return len(rc.moves) > before["moves"] and any("NEXT" in m for _, m in rc.moves[before["moves"]:]) or len(rc.seam_log) > before_seams
            return Press("NEXT", d.next, verify)
        if kind == "hold":
            return Press("HOLD", d.hold, lambda _b: rc.hold or d._hold_until is not None, settle_bars=0.5)
        if kind == "drop":
            def verify(_before):
                holders = {x for x in rc.lanes.values() if x is not None}
                return len(holders) == 1
            return Press("DROP", d.drop, verify)
        if kind == "break":
            def verify(before):
                return any("BREAK" in m for _, m in rc.moves[before["moves"]:])
            return Press("BREAK", d.break_, verify)
        if kind == "moment":
            mk = rng.choice(MOMENTS)

            def verify(before):
                mo = rc.moment_status()
                return (mo is not None and mo["label"].lower() == mk) or any(mk.upper() in m for _, m in rc.moves[before["moves"]:])
            return Press(f"arm {mk.upper()}", lambda: d.arm_moment(mk), verify, settle_bars=0.5)
        if kind == "moment_with":
            mk = rng.choice(("takeover", "bridge"))
            t = on_deck_song() if rng.random() < 0.5 else any_song()
            if t is None:
                t = any_song()

            def verify(before):
                mo = rc.moment_status()
                return (mo is not None and mo["label"].lower() == mk) or any(mk.upper() in m for _, m in rc.moves[before["moves"]:])
            return Press(f"{mk.upper()} with {t.title[:22]}", lambda: d.moment_with(t.id, mk), verify, settle_bars=0.5)
        if kind == "stem_out":
            lane = rng.choice(LANES)
            bed = rc.lanes.get("drums")

            def verify(_before):
                return rc.lanes.get(lane) in (bed, None, rc.record_deck, rc.lanes.get("drums"))
            return Press(f"stem_out {lane}", lambda: d.stem_out(lane), verify)
        if kind == "save_recall":
            def do():
                snap = rc.save_snapshot()
                return "saved" if snap else "nothing to save"

            def verify(_before):
                return bool(rc.snapshots)
            return Press("SAVE", do, verify, settle_bars=0.2)
        from lib.dj.director import DIALS
        name = rng.choice(["pace", "seams", "vocals", "moments", "mixing", "energy"])
        v = rng.choice(list(DIALS[name]))

        def verify(_before):
            return d.dials.get(name) == v
        return Press(f"dial {name} = {v}", lambda: d.set_dial(name, v), verify, settle_bars=0.2)

    rows = []
    t_end = opened_at + int(seconds * RATE / BLOCK)
    n = 0
    gap_blocks = max(bar_blocks(), int(seconds * RATE / BLOCK / max(presses, 1)) - 2 * bar_blocks())   # spread over the run: seams and events happen between presses
    while n < presses and len(audio) < t_end:
        pump(rng.randint(max(1, gap_blocks // 2), gap_blocks + bar_blocks()))
        p = make_press()
        n += 1
        before = {"moves": len(rc.moves), "events": len(d.events), "state": state_words(), "t": len(audio) * BLOCK / RATE, "err": len(errors)}
        try:
            ret = p.target()
        except Exception as e:  # noqa: BLE001
            rows.append({"press": p.name, "state": before["state"], "outcome": "EXCEPTION", "detail": f"{type(e).__name__}: {e}", "t": before["t"]})
            continue
        pump(int(p.settle_bars * bar_blocks()) + 1)
        # a control that stages first ("staging, in as soon as it is ready") is given its decode time
        t0 = time.time()
        while (getattr(d, "_pending_stem", None) is not None or getattr(d, "_pending_moment", None) is not None) and time.time() - t0 < p.wait_pending_s:
            pump(bar_blocks())
        if getattr(d, "_pending_stem", None) is None and getattr(d, "_pending_moment", None) is None and not p.verify(before):
            pump(int(1.5 * bar_blocks()))
        acted = p.verify(before)
        said = [e[1] for e in d.events[before["events"]:]]
        refusals = [s for s in said if "refused" in s.lower() or "dropped" in s.lower() or any(a in s for a in ALLOWED)]
        new_err = errors[before["err"]:]
        if new_err:
            outcome, detail = "EXCEPTION", "; ".join(f"{k}: {e}" for _t, k, e in new_err)
        elif acted:
            outcome, detail = "acted", (str(ret)[:80] if ret else "")
        elif refusals:
            reason = refusals[-1]
            outcome = "said" if any(a in reason for a in ALLOWED) else "WRONG REASON"
            detail = reason[:120]
        elif said:
            outcome, detail = "SPOKE, DID NOT ACT", " | ".join(s[:60] for s in said[-2:])
        else:
            outcome, detail = "SILENT", str(ret)[:80] if ret else ""
        rows.append({"press": p.name, "state": before["state"], "outcome": outcome, "detail": detail, "t": round(before["t"])})
        if verbose or outcome not in ("acted", "said"):
            print(f"  seed {seed} {before['t']:5.0f}s  {outcome:18s} {p.name:44s} [{before['state']}] {detail[:90]}", flush=True)
        if outcome not in ("acted", "said") or ("NO MASTER" in state_words() and "NO MASTER" not in before["state"]):
            # the conductor's last moves and the room, for the diagnosis
            for hms, m in rc.moves[max(0, before["moves"] - 8):]:
                print(f"          {hms} {m[:150]}")
            print(f"          room now: master {rc.master} record {rc.record_deck} lanes {rc.lanes} songs " +
                  ", ".join(f"{dk}={s.track.title[:14]}{'(leaving)' if s.leaving else ''}{'(staged)' if s.staged_at is not None and not s.entered else ''}" for dk, s in rc.songs.items()), flush=True)
    # the room stayed alive
    mono = np.concatenate(audio[opened_at:]).mean(axis=1) if len(audio) > opened_at + 10 else np.zeros(1)
    nb = int(bar_blocks() * BLOCK)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(0, len(mono) - nb, nb)]
    sil = [s["clock_s"] for s in getattr(rc, "silence_log", []) if s.get("clock_s") is not None]
    dead = [i for i, r in enumerate(rms) if r < -45.0 and not any(abs((opened_at * BLOCK + (i + 1) * nb) / RATE - c) < 2.0 * nb / RATE for c in sil)]
    if dead:
        # where the room went silent, with the moves around it (the log's clock_s is the submix clock in seconds)
        shown = -1e9
        for i in dead:
            t_dead = (opened_at * BLOCK + (i + 1) * nb) / RATE
            if t_dead - shown < 8.0:
                continue
            shown = t_dead
            near = [f"{m['clock_s']:.0f}s {m['text'][:90]}" for m in rc.move_log if m.get("clock_s") is not None and -25.0 < t_dead - m["clock_s"] < 6.0]
            print(f"  seed {seed} DEAD at {t_dead:5.0f}s ({rms[i]:.0f} dB)", flush=True)
            for x in near[-12:]:
                print(f"          {x}")
            b0 = int(t_dead * RATE) // BLOCK
            for k, master, pic in [r for r in tel_hist if b0 - 30 <= r[0] <= b0 + 10][::3]:
                print(f"          decks at {k * BLOCK / RATE:.0f} s (master {master}): " + "; ".join(
                    f"{dk}={p[4]} g{p[0]} {p[1]} {'▶' if p[2] else '■'} t{p[3]} held {p[5]}" for dk, p in pic.items()), flush=True)
    d.stop(0.1)
    return {"seed": seed, "rows": rows, "dead_bars": len(dead), "bars": len(rms), "errors": [(round(t), k, e) for t, k, e in errors]}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    seeds = seeds_of(arg("--seeds", "1-3"))
    presses = arg("--presses", 30, int)
    seconds = arg("--seconds", 420.0, float)
    verbose = "--verbose" in sys.argv
    results = []
    t0 = time.time()
    for s in seeds:
        print(f"== seed {s}", flush=True)
        results.append(run(music, s, presses, seconds, verbose))
    rows = [r for res in results for r in res["rows"]]
    by = {}
    for r in rows:
        k = r["press"].split(" ")[0] if not r["press"].startswith(("arm", "TAKEOVER", "BRIDGE", "dial", "stem_in", "stem_out", "UP")) else r["press"].split(" ")[0] + (" " + r["press"].split(" ")[1] if r["press"].startswith(("arm", "dial")) else "")
        by.setdefault(k, {}).setdefault(r["outcome"], 0)
        by[k][r["outcome"]] += 1
    print(f"\n== the control gate: {len(rows)} presses over {len(results)} seeds in {(time.time() - t0) / 60:.1f} min")
    print(f"{'control':22s} {'acted':>6s} {'said':>6s} {'SILENT':>7s} {'WRONG':>6s} {'SPOKE':>6s} {'EXC':>5s}")
    for k in sorted(by):
        o = by[k]
        print(f"{k:22s} {o.get('acted', 0):6d} {o.get('said', 0):6d} {o.get('SILENT', 0):7d} {o.get('WRONG REASON', 0):6d} {o.get('SPOKE, DID NOT ACT', 0):6d} {o.get('EXCEPTION', 0):5d}")
    bad = [r for r in rows if r["outcome"] not in ("acted", "said")]
    said = {}
    for r in rows:
        if r["outcome"] == "said":
            said[r["detail"][:70]] = said.get(r["detail"][:70], 0) + 1
    if said:
        print("\n-- what it said instead of acting (allowed reasons) --")
        for k, v in sorted(said.items(), key=lambda x: -x[1])[:12]:
            print(f"  {v:3d}  {k}")
    if bad:
        print("\n-- the contract broken --")
        for r in bad[:40]:
            print(f"  seed? {r['t']:5.0f}s  {r['outcome']:18s} {r['press']:44s} [{r['state']}] {r['detail'][:110]}")
    fatal = [res for res in results if res.get("fatal")]
    dead = sum(res.get("dead_bars", 0) for res in results)
    errs = [e for res in results for e in res.get("errors", [])]
    for f in fatal:
        print(f"  FATAL seed {f['seed']}: {f['fatal']}")
    if errs:
        print("\n-- errors swallowed into last_error --")
        for t, k, e in errs[:20]:
            print(f"  {t:5.0f}s {k}: {e}")
    print(f"\ndead bars (under -45 dBFS, the record's own silence excluded): {dead} of {sum(res.get('bars', 0) for res in results)}")
    ok = not bad and not fatal and dead == 0 and not errs
    print("\nALL OK" if ok else f"\nFAIL: {len(bad)} broken presses, {len(fatal)} fatal runs, {dead} dead bars, {len(errs)} errors")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
