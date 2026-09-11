"""Remix mode gate: the conductor runs headless (NO audio device - the mixer is pulled by hand) for a
few rendered minutes with a fast change rate, and the checks are the ones the mode lives by:

  * more than one song is live, lanes cross between songs, songs enter and leave lane by lane
  * every slave deck locks to the clock: grid-phase error median < 0.05 beat, p95 < 0.12
  * the harmonic guard holds at every block: tonal lanes held by different songs are never below
    CLASH_BELOW Camelot compatibility (after each song's key shift)
  * no clipping, no dead air once the first song has opened
  * controls: HOLD freezes the lane map; NEXT forces a move on the next bar

    python tools/tests/_dj_remix_test.py --music D:/Devel/music [--seconds 240] [--seed 7] [--wav out.wav]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.dj.submix import RATE  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    from lib.audio_engine import AudioEngine
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    from lib.dj.remix import RemixConductor, CLASH_BELOW, TONAL, _compat

    music = arg("--music", "D:/Devel/music")
    seconds = arg("--seconds", 240.0, float)
    seed = arg("--seed", 7, int)
    db = LibraryDB(music)
    lib = [t for t in B.load_library(db) if not t.excluded]
    rc = RemixConductor(db, music, lib, theme="groove", seed=seed)
    print(f"== remix: {len(rc.library)} stem-bearing tracks of {len(lib)}")
    rc.set_change_bars(4)
    rc.set_blend(0.8)
    rc.set_vocal_freedom(0.6)
    ok = rc.start(threaded=False)
    check(ok, "conductor started" + (f" ({rc.last_error})" if not ok else ""))
    if not ok:
        return 1

    engine = AudioEngine()
    engine.attach_track("remix", rc.submix)
    gen = engine._mixer()
    next(gen)
    gen.send(256)

    audio, errs, guard_viol, live_hist, lane_hist = [], {}, 0, [], []
    block = 4410

    def pump(n_blocks):
        nonlocal guard_viol
        for _ in range(n_blocks):
            audio.append(np.frombuffer(gen.send(block), dtype=np.float32).reshape(-1, 2).copy())
            rc.step()
            tel = rc.submix.telemetry or {}
            sync = tel.get("sync") or {}
            views = sync.get("slaves") or ({sync["slave"]: sync} if sync else {})
            decks = tel.get("decks") or {}
            for sl, v in views.items():
                m, s = decks.get(v["master"]) or {}, decks.get(sl) or {}
                if not (m.get("playing") and s.get("playing")):
                    continue
                e = (float(s["beat_phase"]) - float(m["beat_phase"]) - float(v.get("bias_beats") or 0.0) + 0.5) % 1.0 - 0.5
                errs.setdefault(sl, []).append((len(audio), abs(e)))
            # the harmonic guard, as heard: tonal lanes actually open (gain > 0.5) on different songs
            open_tonal = {}
            for d, song in rc.songs.items():
                g = (decks.get(d) or {}).get("stem_gains") or {}
                for lane in TONAL:
                    if float(g.get(lane, 0.0)) > 0.5:
                        open_tonal[lane] = song
            songs = list({id(s): s for s in open_tonal.values()}.values())
            for i in range(len(songs)):
                for j in range(i + 1, len(songs)):
                    a, b = songs[i], songs[j]
                    if a.track.camelot and b.track.camelot and _compat(a.key(), b.key()) < CLASH_BELOW - 1e-9:
                        guard_viol += 1
            live_hist.append(len([s for s in rc.songs.values() if s.entered and not s.leaving]))
            lane_hist.append(tuple(rc.lanes[s] for s in ("drums", "bass", "other", "vocals")))

    # wait (real time) for the first decode, rendering silence meanwhile
    import time
    t0 = time.time()
    while rc.master is None and time.time() - t0 < 60:
        pump(1)
        time.sleep(0.02)
    check(rc.master is not None, f"first song opened after {time.time() - t0:.1f} s: {rc.status()['songs'].get(rc.master, {}).get('title')}")
    opened_at = len(audio)
    total_blocks = int(seconds * RATE) // block
    # decodes run on real threads (5-10 s each): pace the render near 2x real time so picks arrive the
    # way they do live, with a phrase or two of the first song, not a third of the render
    def paced(n_blocks):
        t_start, done = time.time(), 0
        while done < n_blocks:
            pump(4)
            done += 4
            lag = done * block / RATE / 2.0 - (time.time() - t_start)
            if lag > 0:
                time.sleep(min(lag, 0.2))
    paced(total_blocks * 2 // 3)
    # HOLD: the lane map must not change for a phrase
    rc.set_hold(True)
    bars = rc._bar_s(rc.songs[rc.master].track)
    n_hold = int(2 * rc.change_bars * bars * RATE) // block
    h0 = len(lane_hist)
    pump(n_hold)
    held_maps = set(lane_hist[h0 + int(bars * RATE) // block:])       # after the bar that was already scheduled
    check(len(held_maps) <= 2, f"HOLD: lane map changed {len(held_maps) - 1} times in {2 * rc.change_bars} bars (runway moves allowed)")
    rc.set_hold(False)
    # NEXT: a move within the next bar and a bit
    before = len(rc.moves)
    rc.next_move()
    pump(int(1.5 * bars * RATE) // block)
    check(len(rc.moves) > before, f"NEXT forced a move: {rc.moves[-1][1] if rc.moves else '-'}")
    # the performance buttons: LOOP 8 holds every song, BREAK rests all lanes but one and restores them,
    # DROP puts every lane on the newest song; GOOD / BAD store verdicts that tilt the weights
    n_songs = len([s for s in rc.songs.values() if not s.leaving])
    rc.loop(8)
    pump(int(1.5 * bars * RATE) // block)
    looped = [d for d, s in rc.songs.items() if (rc._tel_deck(d).get("loop") is not None) and not s.leaving]
    check(rc.user_loop_bars == 8 and len(looped) >= min(2, n_songs), f"LOOP 8: {len(looped)} of {n_songs} songs looping")
    rc.loop(8)                                     # again = release
    pump(int(1.5 * bars * RATE) // block)
    check(rc.user_loop_bars is None, "LOOP 8 again released the loops")
    held_before = dict(rc.lanes)
    ok_b = rc.break_(bars=2)
    pump(int(1.0 * bars * RATE) // block)
    resting = [ln for ln, d in rc.lanes.items() if d is None]
    check(ok_b and len(resting) >= 2, f"BREAK: {len(resting)} lanes resting, {rc.move_log[-1]['lane']} alone")
    pump(int(2.5 * bars * RATE) // block)
    back = sum(1 for ln, d in rc.lanes.items() if d is not None and d == held_before.get(ln))
    check(rc._break is None and back >= 2, f"BREAK over: {back} lanes back where they were")
    rc.db.add_seam_feedback = lambda *a, **k: None      # the gate never writes verdicts into the library
    n_v0 = rc.n_verdicts
    w0 = rc._w("break")
    rc.rate_last(False)
    check(rc.n_verdicts == n_v0 + 1 and rc._w("break") < w0, f"BAD on the break stored: weight {w0:.2f} -> {rc._w('break'):.2f}")
    rc.drop()
    pump(int(1.5 * bars * RATE) // block)
    holders = {d for d in rc.lanes.values() if d is not None}
    check(len(holders) == 1, f"DROP: every lane on one song ({rc.move_log[-1]['text'][:60]})")
    rc.rate_last(True)
    check(rc._w("drop") > 0.5, f"GOOD on the drop stored: weight {rc._w('drop'):.2f}")
    paced(max(0, opened_at + total_blocks - len(audio)))
    rc.stop(fade_s=0.5)
    pump(10)

    # -- verdicts --------------------------------------------------------------------------------
    st = rc.status()
    crosses = [m for _, m in rc.moves if "crosses" in m or "enters through" in m or "rests" in m or "returns" in m]
    check(max(live_hist or [0]) >= 2, f"songs live at once: max {max(live_hist or [0])}")
    check(len(crosses) >= 6, f"lane moves: {len(crosses)} ({len(rc.moves)} moves in all)")
    frac_multi = float(np.mean([1.0 if n >= 2 else 0.0 for n in live_hist])) if live_hist else 0.0
    check(frac_multi > 0.4, f"time with 2+ songs heard: {100 * frac_multi:.0f}%")
    entered = [m for _, m in rc.moves if "enters through" in m]
    left = [m for _, m in rc.moves if m.endswith("leaves")]
    check(len(entered) >= 1, f"{len(entered)} songs entered lane by lane, {len(left)} left")
    check(guard_viol == 0, f"harmonic guard violated in {guard_viol} blocks")
    blk_per_bar = bars * RATE / block
    for sl, rows in errs.items():
        if not rows:
            continue
        first = rows[0][0]
        settled = [e for (k, e) in rows if k >= first + 2 * blk_per_bar]
        if len(settled) < 10:
            continue
        med, p95 = float(np.median(settled)), float(np.percentile(settled, 95))
        check(med < 0.05 and p95 < 0.12, f"deck {sl} lock: median {med:.3f} beat, p95 {p95:.3f} (n {len(settled)})")
    x = np.concatenate(audio, axis=0)
    peak = float(np.abs(x).max())
    check(peak < 0.99, f"peak {peak:.3f}")
    mono = x[opened_at * block:-10 * block].mean(axis=1) if len(x) > (opened_at + 12) * block else x.mean(axis=1)
    nb = int(bars * RATE)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(nb, len(mono) - nb, nb)]
    dead = sum(1 for r in rms if r < -40.0)
    check(dead == 0, f"dead bars: {dead} of {len(rms)} (median bar level {np.median(rms):.1f} dBFS)")
    if "--wav" in sys.argv:
        import soundfile as sf
        out = arg("--wav", "")
        sf.write(out, x, RATE, subtype="PCM_16")
        print(f"  --   wrote {out} ({len(x) / RATE:.0f} s)")
    print("  -- moves:")
    for t, m in rc.moves:
        print(f"      {t}  {m}")
    if rc.last_error:
        print(f"  -- last error: {rc.last_error}")
    db.close()
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAIL: " + "; ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
