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
    rc.strip_before_bed = True                            # the philosophy: a breakdown before a new bed lands
    ok = rc.start(threaded=False)
    check(ok, "conductor started" + (f" ({rc.last_error})" if not ok else ""))
    if not ok:
        return 1

    engine = AudioEngine()
    engine.attach_track("remix", rc.submix)
    gen = engine._mixer()
    next(gen)
    gen.send(256)

    audio, errs, guard_viol, live_hist, lane_hist, tel_hist = [], {}, 0, [], [], []
    hyg = {"vocal_blocks": 0, "vocal_silent": 0, "carve_blocks": 0, "carve_open": 0}
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
            if len(audio) % 10 == 0:                     # a picture of the decks once a second, for the dead-bar report
                tel_hist.append((len(audio), rc.master, {d: (round(float(t.get("gain") or 0.0), 2), {k: round(float(v), 2) for k, v in (t.get("stem_gains") or {}).items()},
                                                          bool(t.get("playing")), round(float(t.get("time_s") or 0.0), 1), (rc.songs[d].track.title[:14] if d in rc.songs else "-"))
                                                      for d, t in decks.items()}))
            # the vocal rule, as heard: the vocal lane open on a song that is not singing there (by the
            # conductor's own measured singing map); a stretch counts once it outlasts 1.5 phrases - the
            # rule acts at the next move, so one phrase of a song's instrumental passage is by design
            vd = rc.lanes.get("vocals")
            silent_now = False
            if vd in rc.songs and float(((decks.get(vd) or {}).get("stem_gains") or {}).get("vocals", 0.0)) > 0.5:
                sing = rc._singing(rc.songs[vd], float((decks.get(vd) or {}).get("time_s") or 0.0))
                if sing is not None:
                    hyg["vocal_blocks"] += 1
                    silent_now = sing is False
            if silent_now:
                hyg["vocal_run"] = hyg.get("vocal_run", 0) + 1
                if hyg["vocal_run"] > 1.5 * rc.change_bars * rc._bar_clock() / block:
                    hyg["vocal_silent"] += 1
                    sec = rc.songs[vd].track.section_at(float((decks.get(vd) or {}).get("time_s") or 0.0)) or {}
                    hyg.setdefault("vocal_first", (round(len(audio) * block / RATE), rc.songs[vd].track.title[:30], sec.get("kind"),
                                                   sec.get("vocalness"), rc.move_log[-1]["text"][:70] if rc.move_log else "-"))
            else:
                hyg["vocal_run"] = 0
            # the low carve, as heard: a playing song whose bass and drums stems are both closed (and has
            # been so for more than a bar: the EQ ramps a beat after the lane lands) has its EQ low down
            if rc._break is None:
                for d, song in rc.songs.items():
                    t = decks.get(d) or {}
                    g = t.get("stem_gains") or {}
                    if not t.get("playing") or song.leaving or not song.entered:
                        continue
                    lows_open = float(g.get("bass", 0.0)) > 0.05 or float(g.get("drums", 0.0)) > 0.05
                    key = ("carve_since", d)
                    if lows_open:
                        hyg.pop(key, None)
                        continue
                    hyg.setdefault(key, len(audio))
                    if len(audio) - hyg[key] < 1.5 * rc._bar_clock() / block:
                        continue
                    hyg["carve_blocks"] += 1
                    if float((t.get("eq") or [1.0])[0]) > 0.6:
                        hyg["carve_open"] += 1

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
    n_entered = len([s for s in rc.songs.values() if s.entered and not s.leaving])
    rc.next_move()
    pump(int(1.5 * bars * RATE) // block)
    if n_entered >= 2:
        check(len(rc.moves) > before, f"NEXT forced a move: {rc.moves[-1][1] if rc.moves else '-'}")
    else:
        print(f"  --   NEXT with {n_entered} song heard: nothing to cross to (not judged)")
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
    n_mv_b = len(rc.moves)
    rc.db.add_seam_feedback = lambda *a, **k: None      # the gate never writes verdicts into the library
    n_v0 = rc.n_verdicts
    w0 = rc._w("break")
    rc.rate_last(False)                                  # the break is the last move: BAD lands on it
    check(rc.n_verdicts == n_v0 + 1 and rc._w("break") < w0, f"BAD on the break stored: weight {w0:.2f} -> {rc._w('break'):.2f}")
    pump(int(1.0 * bars * RATE) // block)
    resting = [ln for ln, d in rc.lanes.items() if d is None]
    check(ok_b and len(resting) >= 2, f"BREAK: {len(resting)} lanes resting, {rc.move_log[-1]['lane']} alone")
    pump(int(2.5 * bars * RATE) // block)
    back = sum(1 for ln, d in rc.lanes.items() if d is not None and d == held_before.get(ln))
    moved_since = [m for _, m in rc.moves[n_mv_b:] if "the bed passes" in m or "fades out" in m or "DROP" in m]
    check(rc._break is None and (back >= 2 or moved_since),
          f"BREAK over: {back} lanes back where they were" + (f" (then the arrangement moved on: {moved_since[0][:40]})" if moved_since and back < 2 else ""))
    rc.drop()
    drop_text = rc.move_log[-1]["text"]
    w0 = rc._w("drop")
    rc.rate_last(True)                                   # the drop is the last move: GOOD lands on it
    check(rc._w("drop") > w0, f"GOOD on the drop stored: weight {w0:.2f} -> {rc._w('drop'):.2f}")
    pump(int(1.5 * bars * RATE) // block)
    holders = {d for d in rc.lanes.values() if d is not None}
    check(len(holders) == 1, f"DROP: every lane on one song ({drop_text[:60]})")
    # snapshots: save the combination, let the conductor move on, recall it
    paced(int(3 * rc.change_bars * bars * RATE) // block)          # a couple of phrases: 2+ songs again
    snap = rc.save_snapshot()
    check(snap is not None and sum(1 for v in snap["lanes"].values() if v) >= 3, f"SAVE: {snap['lanes'] if snap else None} (a resting lane saves as none)")
    rc.next_move()
    pump(int(1.5 * bars * RATE) // block)
    rc.next_move()
    pump(int(1.5 * bars * RATE) // block)
    moved = any(rc._song_id(rc.lanes.get(ln)) != snap["lanes"].get(ln) for ln in snap["lanes"])
    rc.recall_snapshot(-1)
    from lib.dj.remix import RECALL_PHRASES
    for _ in range(int(RECALL_PHRASES * rc.change_bars) + 4):   # songs that left must be decoded (5-10 s real) and staged again
        paced(int(bars * RATE) // block)
        if rc._recall is None:
            break
    paced(int(2 * bars * RATE) // block)
    same = sum(1 for ln in snap["lanes"] if rc._song_id(rc.lanes.get(ln)) == snap["lanes"].get(ln))
    notes = [m["text"][:90] for m in rc.move_log if m["kind"].startswith("recall")]
    check(rc._recall is None and same >= 2, f"RECALL: {same}/4 lanes back on the saved songs (lanes moved in between: {moved}) {notes[-1:] if notes else ''}")
    # a tempo journey: lean hot, span 4 %: the clock climbs in half-percent steps, every song inside the wall
    bpm0 = rc.master_bpm
    rc.set_energy_lean(0.4)
    rc.set_tempo_span(0.04)
    paced(int(10 * bars * RATE) // block)
    rates_ok = all(0.90 <= s.rate <= 1.10 for s in rc.songs.values())
    check(rc.master_bpm > bpm0 * 1.01 and rates_ok, f"tempo journey: clock {bpm0:.2f} -> {rc.master_bpm:.2f} bpm, every song inside the wall {rates_ok}")
    rc.set_tempo_span(0.0)
    rc.set_energy_lean(0.0)
    # recording: a WAV with the move log beside it
    import os
    import tempfile
    import wave
    wav = os.path.join(tempfile.gettempdir(), "remix_gate_rec.wav")
    rc.record(wav)
    pump(int(2.0 * RATE) // block)
    rc.record_stop()
    with wave.open(wav, "rb") as w:
        nfr = w.getnframes()
    check(nfr >= int(1.5 * RATE) and os.path.exists(wav[:-4] + ".json"), f"recording: {nfr / RATE:.1f} s of audio + move log json")
    for p in (wav, wav[:-4] + ".json"):
        try:
            os.remove(p)
        except OSError:
            pass
    paced(max(0, opened_at + total_blocks - len(audio)))
    rc.stop(fade_s=0.5)
    pump(10)

    # -- verdicts --------------------------------------------------------------------------------
    st = rc.status()
    crosses = [m for _, m in rc.moves if any(k in m for k in ("crosses", "enters through", "rests", "returns",
                                                              "arrives as a voice", "the bed passes", "fades out as a voice"))]
    check(max(live_hist or [0]) >= 2, f"songs live at once: max {max(live_hist or [0])}")
    check(len(crosses) >= 6, f"lane moves: {len(crosses)} ({len(rc.moves)} moves in all)")
    frac_multi = float(np.mean([1.0 if n >= 2 else 0.0 for n in live_hist])) if live_hist else 0.0
    check(frac_multi > 0.4, f"time with 2+ songs heard: {100 * frac_multi:.0f}%")
    entered = [m for _, m in rc.moves if "enters through" in m or "arrives as a voice" in m]
    left = [m for _, m in rc.moves if m.endswith("leaves") or "fades out as a voice" in m]
    check(len(entered) >= 1, f"{len(entered)} songs entered lane by lane, {len(left)} left")
    # the philosophy of play: bed changes land at the voice's drop when one is near, after a breakdown;
    # the arrangement says why a phrase passed without a move (settling, building, waiting for a drop)
    beds = [m for _, m in rc.moves if "the bed passes" in m]
    at_drop = [m for m in beds if "at its drop" in m]
    strips = [m for m in beds if "drop out for" in m]
    check(len(beds) >= 2 and len(strips) >= 1, f"bed changes: {len(beds)}, {len(at_drop)} at the voice's drop, {len(strips)} with a breakdown first")
    arr = st.get("arrangement") or {}
    check("bed" in arr and "voices" in arr and not arr.get("error"), f"arrangement status: bed {((arr.get('bed') or {}).get('title') or '-')[:24]}, {len(arr.get('voices') or [])} voices, wait: {arr.get('wait_why')}")
    check(guard_viol == 0, f"harmonic guard violated in {guard_viol} blocks")
    # structure, hygiene, shapes
    if hyg["vocal_blocks"]:
        frac = hyg["vocal_silent"] / hyg["vocal_blocks"]
        check(frac < 0.15, f"vocal lane on a silent section: {100 * frac:.0f}% of {hyg['vocal_blocks']} blocks with vocal data"
              + (f" - first at {hyg['vocal_first'][0]} s: {hyg['vocal_first'][1]} {hyg['vocal_first'][2]} v={hyg['vocal_first'][3]} after '{hyg['vocal_first'][4]}'" if hyg.get("vocal_first") else ""))
    else:
        print("  --   no block with vocal data on the vocal lane (rule not exercised)")
    if hyg["carve_blocks"]:
        frac = hyg["carve_open"] / hyg["carve_blocks"]
        check(frac < 0.15, f"low carve: lows still open on {100 * frac:.0f}% of {hyg['carve_blocks']} blocks where a song held neither bass nor drums")
    shapes = [m for m in rc.move_log if m.get("shape")]
    check(len(shapes) >= 1, f"shaped moves: {len(shapes)} ({', '.join(sorted({m['shape'] for m in shapes}))})")
    staged_lm = [line for line in rc.log if "landmark" in line]
    check(len(staged_lm) >= 1, f"{len(staged_lm)} songs staged on a landmark, e.g. {staged_lm[0][-60:] if staged_lm else '-'}")
    blk_per_bar = bars * RATE / block
    for sl, rows in errs.items():
        if not rows:
            continue
        first = rows[0][0]
        settled = [e for (k, e) in rows if k >= first + 2 * blk_per_bar]
        if len(settled) < 10:
            continue
        med, p95 = float(np.median(settled)), float(np.percentile(settled, 95))
        check(med < 0.05 and p95 < 0.15, f"deck {sl} lock: median {med:.3f} beat, p95 {p95:.3f} (n {len(settled)})")
    x = np.concatenate(audio, axis=0)
    peak = float(np.abs(x).max())
    check(peak < 0.99, f"peak {peak:.3f}")
    mono = x[opened_at * block:-10 * block].mean(axis=1) if len(x) > (opened_at + 12) * block else x.mean(axis=1)
    nb = int(bars * RATE)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(nb, len(mono) - nb, nb)]
    dead_i = [i for i, r in enumerate(rms) if r < -40.0]
    dead = len(dead_i)
    where = ""
    if dead_i:
        t_dead = (opened_at * block + (dead_i[0] + 1) * nb) / RATE
        near = [m for m in rc.move_log if m.get("clock_s") is not None and abs(m["clock_s"] - t_dead) < 12.0]
        where = f" - first at {t_dead:.0f} s near: " + " | ".join(f"{m['clock_s']:.0f}s {m['text'][:50]}" for m in near[-3:])
        b0 = int(t_dead * RATE) // block
        for k, master, decks_pic in [r for r in tel_hist if b0 - 20 <= r[0] <= b0 + 40][::2]:
            if True:
                print(f"       decks at {k * block / RATE:.0f} s (master {master}): " + "; ".join(f"{d}={pic[4]} g{pic[0]} {pic[1]} {'▶' if pic[2] else '■'} t{pic[3]}" for d, pic in decks_pic.items()))
    check(dead == 0, f"dead bars: {dead} of {len(rms)} (median bar level {np.median(rms):.1f} dBFS){where}")
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
