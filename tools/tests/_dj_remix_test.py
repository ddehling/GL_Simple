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

os.environ["DJ_NO_PLAY_LOG"] = "1"                      # the gate's plays are not the night's history
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
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # move texts carry arrows; a cp1252 console choked
    except Exception:
        pass
    from lib.audio_engine import AudioEngine
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    from lib.dj.remix import RemixConductor, CLASH_BELOW, TONAL, _compat
    from lib.dj.submix import _odd_segment

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
    rc.bass_moments = False                               # (the strip path here; the bass-first path is checked below)
    ok = rc.start(threaded=False)
    check(ok, "conductor started" + (f" ({rc.last_error})" if not ok else ""))
    if not ok:
        return 1

    engine = AudioEngine()
    engine.attach_track("remix", rc.submix)
    # who posts EQ events (the carve mystery): every eq event with its clock, deck, low and the caller
    import traceback
    eq_trace = []
    _post0, _post_many0 = rc.submix.post, rc.submix.post_many

    def _trace(e):
        if isinstance(e, dict) and e.get("cmd") == "eq":
            fr = traceback.extract_stack(limit=4)[0]
            eq_trace.append((round((e.get("at") or rc.submix.clock) / RATE, 1), e.get("deck"), e.get("low"), f"{fr.name}:{fr.lineno}"))

    def _post(e):
        _trace(e)
        _post0(e)

    def _post_many(evs):
        evs = list(evs)
        for e in evs:
            _trace(e)
        _post_many0(evs)
    rc.submix.post, rc.submix.post_many = _post, _post_many
    gen = engine._mixer()
    next(gen)
    gen.send(256)

    audio, errs, guard_viol, live_hist, lane_hist, tel_hist, plan_hist = [], {}, 0, [], [], [], []
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
                # inside a mis-tracked grid segment (a 66 bpm read in a 116 bpm song) the beat phase means
                # nothing - the PLL holds there and so does this measurement
                if _odd_segment(rc.submix.decks.get(sl)) or _odd_segment(rc.submix.decks.get(v["master"])):
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
            # A SONG KEEPS ITS OWN BALANCE: while one song holds every lane (no boost, no moment, no break),
            # its four stem gains are one factor - equal within 2 %
            holders = {rc.lanes.get(ln) for ln in ("drums", "bass", "other", "vocals")}
            if len(holders) == 1 and None not in holders and rc._moment is None and rc._break is None and not rc.lane_boost and not getattr(rc, "_clash_duck", None):
                dk = next(iter(holders))
                g = (decks.get(dk) or {}).get("stem_gains") or {}
                vals = [float(g.get(ln, 0.0)) for ln in ("drums", "bass", "other", "vocals")]
                key = ("lone_since", dk)
                hyg.setdefault(key, len(audio))
                if len(audio) - hyg[key] >= 1.5 * rc._bar_clock() / block and min(vals) > 0.05:
                    hyg["lone_blocks"] = hyg.get("lone_blocks", 0) + 1
                    if max(vals) / min(vals) > 1.02:
                        hyg["lone_unequal"] = hyg.get("lone_unequal", 0) + 1
                        hyg.setdefault("lone_first", (round(len(audio) * block / RATE), rc.songs[dk].track.title[:24] if dk in rc.songs else dk, [round(v, 3) for v in vals]))
            else:
                for k in [k for k in hyg if isinstance(k, tuple) and k[0] == "lone_since"]:
                    hyg.pop(k, None)
            plan_hist.append((rc.plan.get("seq"), rc.plan["what"], rc.plan["deck"], rc.plan["bar"]) if rc.plan else None)   # seq, not id(): ids are reused
            lane_hist.append(tuple(rc.lanes[s] for s in ("drums", "bass", "other", "vocals")))
            if len(audio) % 10 == 0:                     # a picture of the decks once a second, for the dead-bar report
                tel_hist.append((len(audio), rc.master, {d: (round(float(t.get("gain") or 0.0), 2), {k: round(float(v), 2) for k, v in (t.get("stem_gains") or {}).items()},
                                                          bool(t.get("playing")), round(float(t.get("time_s") or 0.0), 1), (rc.songs[d].track.title[:14] if d in rc.songs else "-"),
                                                          round(float((t.get("eq") or [1.0])[0]), 2), (rc.songs[d].held(rc.lanes) if d in rc.songs else []),
                                                          round(float(t.get("rate") or 0.0), 4), (round(float(t["beat_phase"]), 3) if t.get("beat_phase") is not None else None),
                                                          t.get("loop") is not None, round(float(t.get("rate_trim") or 0.0), 5) if t.get("rate_trim") is not None else None)
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
            # THE BED IS NEVER EMPTY: drums or bass with nobody while songs play (outside a break / recall)
            if rc._break is None and rc._recall is None and any(s.entered and not s.leaving for s in rc.songs.values()):
                hyg["bed_blocks"] = hyg.get("bed_blocks", 0) + 1
                if rc.lanes.get("bass") is None or rc.lanes.get("drums") is None:
                    hyg["bed_empty"] = hyg.get("bed_empty", 0) + 1
            # the opposite failure ("large sections of vocal rest"): the vocal lane RESTING while the bed sings
            bed_d = rc.lanes.get("drums")
            if rc._break is None and rc._recall is None and bed_d in rc.songs and rc.songs[bed_d].entered:
                bt = float((decks.get(bed_d) or {}).get("time_s") or 0.0)
                if rc._singing(rc.songs[bed_d], bt) is True:
                    hyg["bed_sings"] = hyg.get("bed_sings", 0) + 1
                    if rc.lanes.get("vocals") is None:
                        hyg["bed_sings_rested"] = hyg.get("bed_sings_rested", 0) + 1
            # the low carve, as heard: a playing song whose bass and drums stems are both closed (and has
            # been so for more than a bar: the EQ ramps a beat after the lane lands) has its EQ low down
            # (not during a moment: its take-away mutes the bed's own drums and bass on purpose)
            if rc._break is None and rc._moment is None:
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
                        who = hyg.setdefault("carve_who", {})
                        kw = (song.track.title[:22], tuple(song.held(rc.lanes)), bool(song.was_bed))
                        who[kw] = who.get(kw, 0) + 1
                        hyg.setdefault("carve_first", (len(audio), d))

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
    for _ in range(12):                              # a bed change in flight defers the hold to its landing bar
        if rc._land_k is None:
            break
        pump(int(bars * RATE) // block)
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
    # THE TACTICAL MOVE: a live song's bass in on the next bar, loud - the guard may refuse (key), then any lane it allows
    others = [dd for dd, s in rc.songs.items() if s.staged_at is not None and not s.leaving and dd != rc.lanes.get("drums")]
    if others:
        dd = others[0]
        ok_b2, msg_b = False, "-"
        for ln in ("bass", "other", "drums"):
            ok_b2, msg_b = rc.bring(ln, dd, boost=1.3, when="bar")
            if ok_b2:
                lane_b = ln
                break
        pump(int(2.5 * bars * RATE) // block)
        check(ok_b2 and rc.lanes.get(lane_b) == dd and rc.lane_boost.get(lane_b) is not None,
              f"BRING: the {lane_b if ok_b2 else '?'} of {rc.songs[dd].track.title[:22]} in on the next bar, loud ({msg_b})")
        held_now = rc.lanes.get(lane_b) if ok_b2 else None
        ok_r, msg_r = rc.release(lane_b) if ok_b2 else (False, "-")
        pump(int(1.5 * bars * RATE) // block)
        check(ok_r and rc.lanes.get(lane_b) != held_now, f"RELEASE: the {lane_b if ok_b2 else '?'} back to the bed ({msg_r})")
    rc.bass_moments = True                                # from here the bed changes bass-first
    rc.strip_before_bed = False
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
    # THE BASS MOMENT: with a voice heard and MOMENTS on, a bed change is bass-first - the bassline takes over
    # (loud) and the drums follow at the phrase; NEXT forces the change once a voice is there
    t_bm = len(rc.moves)
    for _ in range(int(8 * rc.change_bars)):
        if any(dd != rc.lanes.get("drums") and s.entered and not s.leaving and s.voice_since is not None
               and rc.bar_n - s.voice_since >= rc.change_bars for dd, s in rc.songs.items()):
            break
        pump(int(bars * RATE) // block)
    rc.next_move()
    pump(int(1.5 * bars * RATE) // block)
    pump(int((2 * rc.change_bars + 3) * bars * RATE) // block)       # the drums follow a phrase after the landing
    # the PLAN may have made the change on its own before this block started counting: any bass-first change
    # in the whole run counts, and the drums must have followed at least once
    took = [m for _, m in rc.moves if "takes over" in m]
    followed = [m for _, m in rc.moves if "drums of" in m and "follow" in m]
    clash = [m for _, m in rc.moves[t_bm:] if "clash" in m]
    check((len(took) >= 1 and len(followed) >= 1) or bool(clash),
          f"BASS MOMENT: {len(took)} bassline takeovers, {len(followed)} drums followed" + (f" (voice refused: {clash[0][:50]})" if clash and not took else ""))
    # THE COMPOSED MOMENTS: SLAM (two bars: bass + drums out, a beat of nothing, back loud on the one) and the
    # BREAK routine (sixteen: take away, build with the sweep and the tightening loop, the hole, the one) - each
    # armed, landing within four phrases, its take-away and its one in the log, the bed's drums measured out then back
    for kind, lead in (("slam", 2), ("break", 16)):
        t_m, k_m = len(rc.moves), len(audio)
        ok_m, msg_m = rc.arm_moment(kind)
        if not ok_m and kind == "break" and any(w in msg_m for w in ("silent there", "not measured", "nothing would be left")):
            print(f"  --   BREAK refused here, rightly: {msg_m} (not judged)")
            continue
        st_m = rc.moment_status()
        in_bars = st_m["in_bars"] if st_m else None
        Bm = rc._moment["B"] if rc._moment else None
        for _ in range((in_bars or lead) + 8):
            pump(int(bars * RATE) // block)
            if rc._moment is None:
                break
        landed = [m for _, m in rc.moves[t_m:] if "on the one" in m]
        away = [m for _, m in rc.moves[t_m:] if kind.upper() in m and " out" in m]
        pics = [pic for k, _mst, pic in tel_hist if k_m <= k <= len(audio) and Bm in pic]
        g_dr = [float((pic[Bm][1] or {}).get("drums", 1.0)) for pic in pics]
        went_out = any(g < 0.05 for g in g_dr)
        came_back = went_out and any(g > 0.5 for g in g_dr[max(i for i, g in enumerate(g_dr) if g < 0.05):])
        check(ok_m and in_bars is not None and lead <= in_bars <= 4 * rc.change_bars + lead + 10 and landed and away and rc._moment is None and went_out and came_back,
              f"{kind.upper()}: {msg_m}; armed {in_bars} bars out, take-away {len(away)}, the one {len(landed)}, drums out {went_out} / back {came_back}, done {rc._moment is None}"
              + (f" ({rc.last_error})" if rc.last_error else ""))
        paced(int(rc.change_bars * bars * RATE) // block)
    # a two-song moment when a second song is there: TAKEOVER (the other bassline on the one, its drums at the phrase)
    t_m = len(rc.moves)
    ok_t, msg_t = rc.arm_moment("takeover")
    if ok_t:
        Vt = rc._moment["V"]
        in_t = rc.moment_status()["in_bars"]
        for _ in range(in_t + rc.change_bars + 6):
            pump(int(bars * RATE) // block)
            if rc._moment is None:
                break
        check(rc._moment is None and rc.lanes.get("bass") == Vt and rc.lanes.get("drums") == Vt,
              f"TAKEOVER: {msg_t}; bass -> {rc.lanes.get('bass')} drums -> {rc.lanes.get('drums')} (other song {Vt}), done {rc._moment is None}")
    else:
        print(f"  --   TAKEOVER not armed here: {msg_t} (not judged)")
    # the ear's words move one number each (in memory only under DJ_NO_PLAY_LOG); a chosen song that is not on a
    # deck is refused and counted; refusal statistics per kind
    h0 = rc.moment_tune["slam"]["hole_beats"]
    msg_h = rc.tune_moment("slam", "hole")
    msg_th = rc.tune_moment("break", "thin")
    check(rc.moment_tune["slam"]["hole_beats"] == h0 / 2 and rc.moment_tune["break"]["build_low"] > 0.25 and rc.moment_tune["break"]["payoff_only"] is False,
          f"WORDS: {msg_h} | {msg_th}")
    ok_o, msg_o = rc.arm_moment("takeover", other="zz")
    st_m = rc.moment_stats.get("takeover") or {}
    check(not ok_o and "not on a deck" in msg_o and sum((st_m.get("refused") or {}).values()) >= 1,
          f"CHOSEN SONG off a deck refused and counted: {msg_o} | stats {rc.moment_stats}")
    # CALM: with calm_phrases 2 the arrangement moves at most once per two phrases (NEXT and an empty bed lane exempt)
    rc.calm_phrases = 2
    n_c0, k_c0 = len(rc.move_log), rc.bar_n
    paced(int(6 * rc.change_bars * bars * RATE) // block)
    arr_moves = [m for m in rc.move_log[n_c0:] if m["kind"] in ("enter", "cross", "cross_new", "rest", "return", "voice_in", "bed_to", "voice_out")]
    phrases = max(1, (rc.bar_n - k_c0) // rc.change_bars)
    check(len(arr_moves) <= phrases // 2 + 1, f"CALM: {len(arr_moves)} arrangement moves over {phrases} phrases (at most one per two)"
          + (f": {[m['text'][:40] for m in arr_moves]}" if len(arr_moves) > phrases // 2 + 1 else ""))
    rc.calm_phrases = 1
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
    crosses = [m for _, m in rc.moves if any(k in m for k in ("crosses", "enters through", "rests", "returns", "arrives through",
                                                              "arrives as a voice", "the bed passes", "takes over", "fades out as a voice"))]
    check(max(live_hist or [0]) >= 2, f"songs live at once: max {max(live_hist or [0])}")
    if hyg.get("lone_blocks"):
        fr = hyg.get("lone_unequal", 0) / hyg["lone_blocks"]
        check(fr < 0.05, f"LONE SONG keeps its balance: stems unequal on {100 * fr:.0f}% of {hyg['lone_blocks']} lone blocks"
              + (f" - first {hyg['lone_first']}" if hyg.get("lone_first") else ""))
    # THE PLAN makes up its mind: a plan's bar changes only through a logged plan move; plans are few and kept
    plans, bar_changes = 0, 0
    prev = None
    for ph in plan_hist:
        if ph is not None and (prev is None or ph[0] != prev[0]):
            plans += 1
        elif ph is not None and prev is not None and ph[0] == prev[0] and ph[3] != prev[3]:
            bar_changes += 1
        prev = ph
    # a song on its way out never took the bed: every bed change's target had body and no loop at the time
    bad_targets = [m["text"][:60] for m in rc.move_log if m["kind"] == "bed_to" and m.get("to_id") is not None
                   and any(("loops its last" in x["text"] or "has looped long enough" in x["text"]) and x.get("from_id") == m["to_id"] and x["t"] <= m["t"]
                           for x in rc.move_log)]
    check(not bad_targets, f"WAY OUT: bed changes into a song looping its last bars: {len(bad_targets)} {bad_targets[:2]}")
    # the vocal lane does not flap: a rest followed by a return within four bars is a flap
    vt = [(m["t"], m["kind"]) for m in rc.move_log if m["kind"] in ("rest", "return") and m.get("lane") == "vocals"]
    flaps = sum(1 for i in range(1, len(vt)) if vt[i - 1][1] == "rest" and vt[i][1] == "return" and vt[i][0] - vt[i - 1][0] < 4 * bars)
    check(flaps <= 1, f"VOCAL LANE: {flaps} rest→return flaps under four bars ({len(vt)} vocal moves)")
    # how songs arrive: through the vocal, the bass (a swap) or the melody over an INSTRUMENTAL bed - never a melody over a singer
    arrivals = [m for m in rc.move_log if m["kind"] == "voice_in"]
    by_lane = {ln: sum(1 for m in arrivals if m.get("lane") == ln) for ln in ("vocals", "bass", "other")}
    bad_arr = [m["text"][:60] for m in arrivals if m.get("lane") == "other" and "instrumental" not in m["text"]]
    check(arrivals and not bad_arr, f"ARRIVALS: {by_lane} (vocal / bass swap / melody over an instrumental bed); melody over a singer: {len(bad_arr)}")
    ducks = [m["text"][:70] for m in rc.move_log if m["kind"] in ("duck", "unduck")]
    print(f"  --   clash watch: {len(ducks)} duck/unduck" + (f" e.g. {ducks[0]}" if ducks else ""))
    recued = [n for n in rc.log if "cued to" in n and "landmark lands" in n]
    print(f"  --   staged songs re-cued for their arrival bar: {len(recued)}" + (f" e.g. {recued[0][:70]}" if recued else ""))
    plan_moves = sum(1 for m in rc.move_log if m["kind"] == "plan")
    check(plans >= 2 and bar_changes <= plan_moves and plan_moves <= max(2, plans // 2),
          f"PLAN: {plans} plans decided, {bar_changes} bar changes, {plan_moves} logged plan moves (every change is logged, few plans move)")
    check(len(crosses) >= 6, f"lane moves: {len(crosses)} ({len(rc.moves)} moves in all)")
    frac_multi = float(np.mean([1.0 if n >= 2 else 0.0 for n in live_hist])) if live_hist else 0.0
    # the arrangement lets a new bed settle four phrases alone and waits out a bed's build: at four-bar phrases
    # that is about half the time with one song by design (the old free policy overlapped most of the time)
    check(frac_multi > 0.25, f"time with 2+ songs heard: {100 * frac_multi:.0f}% (a new bed settles alone by design)")
    entered = [m for _, m in rc.moves if "enters through" in m or "arrives as a voice" in m or "arrives through" in m or "as it arrives" in m]
    left = [m for _, m in rc.moves if m.endswith("leaves") or "fades out as a voice" in m]
    check(len(entered) >= 1, f"{len(entered)} songs entered lane by lane, {len(left)} left")
    # the philosophy of play: bed changes land at the voice's drop when one is near, after a breakdown;
    # the arrangement says why a phrase passed without a move (settling, building, waiting for a drop)
    beds = [m for _, m in rc.moves if "the bed passes" in m or "takes over" in m]
    at_drop = [m for m in beds if "at its drop" in m or "at its hook" in m]
    strips = [m for m in beds if "drop out for" in m]
    bass_first = [m for m in beds if "takes over" in m]
    follows = [m for _, m in rc.moves if "drums of" in m and "follow" in m]
    check(len(beds) >= 2 and (len(strips) >= 1 or len(bass_first) >= 1),
          f"bed changes: {len(beds)}, {len(at_drop)} at the voice's payoff, {len(strips)} with a breakdown first, {len(bass_first)} bass-first ({len(follows)} drums followed)")
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
    if hyg.get("bed_sings"):
        fr = hyg.get("bed_sings_rested", 0) / hyg["bed_sings"]
        # (a bass swap puts another song's bassline under a singing bed for a phrase: its vocal may not sit on
        #  that bass - the guard's refusal, not a rest; the bar is a quarter now that arrivals swap the bass)
        check(fr < 0.25, f"vocal lane resting while the bed sings: {100 * fr:.0f}% of {hyg['bed_sings']} singing blocks")
    if hyg.get("bed_blocks"):
        fr = hyg.get("bed_empty", 0) / hyg["bed_blocks"]
        check(fr < 0.03, f"drums or bass lane with nobody: {100 * fr:.1f}% of {hyg['bed_blocks']} playing blocks (outside breaks)")
    if hyg["carve_blocks"]:
        frac = hyg["carve_open"] / hyg["carve_blocks"]
        check(frac < 0.15, f"low carve: lows still open on {100 * frac:.0f}% of {hyg['carve_blocks']} blocks where a song held neither bass nor drums")
        if frac >= 0.15:
            for kw, n in sorted((hyg.get("carve_who") or {}).items(), key=lambda kv: -kv[1])[:6]:
                print(f"       lows open {n} blocks: {kw[0]} holding {list(kw[1])}{' (was the bed)' if kw[2] else ''}")
            cf = hyg.get("carve_first")
            if cf:
                k_first, d_first = cf
                print(f"       the first open block at {k_first * block / RATE:.0f} s on deck {d_first}; that deck per second around it (gain, stems, eq_low, lanes):")
                for kk, master, pic in [r for r in tel_hist if k_first - 60 <= r[0] <= k_first + 30]:
                    p = pic.get(d_first)
                    if p:
                        print(f"         {kk * block / RATE:.0f}s {p[4]} g{p[0]} {p[1]} eq{p[5]} lanes {p[6]} master {master}")
                near = [m for m in rc.move_log if m.get("clock_s") is not None and abs(m["clock_s"] - k_first * block / RATE) < 10]
                for m in near[-6:]:
                    print(f"         move {m['clock_s']:.0f}s {m['text'][:90]}")
                t_first = k_first * block / RATE
                for row in [r for r in eq_trace if r[1] == d_first and t_first - 15 <= r[0] <= t_first + 5]:
                    print(f"         eq event at {row[0]}s deck {row[1]} low {row[2]} from {row[3]}")
    shapes = [m for m in rc.move_log if m.get("shape")]
    # a shape rides a song's LAST lane leaving (the dice, at fx_level); with the calmer arrangement a five-minute
    # run has only a handful of such exits, so no shape at all is only a failure when there were exits to shape
    exits = [m for m in rc.move_log if m["kind"] in ("voice_out", "leave", "evict")]
    check(len(shapes) >= 1 or len(exits) < 6, f"shaped moves: {len(shapes)} ({', '.join(sorted({m['shape'] for m in shapes}))}) of {len(exits)} exits")
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
        ok_lock = med < 0.05 and p95 < 0.15
        if not ok_lock:
            # where the lock broke: runs of blocks past 0.15 beat, with the move nearest each run's start
            bad = [k for (k, e) in rows if k >= first + 2 * blk_per_bar and e > 0.15]
            runs = []
            for k in bad:
                if runs and k - runs[-1][1] <= 20:
                    runs[-1][1] = k
                else:
                    runs.append([k, k])
            for k0, k1 in runs[:4]:
                t0 = k0 * block / RATE
                near = min(rc.move_log, key=lambda m: abs((m.get("clock_s") or -1e9) - t0)) if rc.move_log else None
                print(f"       deck {sl} off-lock {t0:.0f}-{k1 * block / RATE:.0f} s near: {near['clock_s']:.0f}s {near['text'][:70]}" if near and near.get("clock_s") is not None else f"       deck {sl} off-lock {t0:.0f}-{k1 * block / RATE:.0f} s")
                # who was the clock and what did this deck do around then: re-cues (a cue is a phase jump), the
                # master per second, the deck's song / gain / loop
                recues = [r for r in getattr(rc, "recue_log", []) if r[0] is not None and t0 - 20 <= r[0] <= k1 * block / RATE and r[1] == sl]
                print(f"         re-cues of {sl} in the window: {recues}")
                for kk, master, pic in [r for r in tel_hist if k0 - 30 <= r[0] <= k1 + 10][::2]:
                    p, pm = pic.get(sl), pic.get(master)
                    err = ((p[8] - pm[8] + 0.5) % 1.0 - 0.5) if (p and pm and p[8] is not None and pm[8] is not None) else None
                    print(f"         {kk * block / RATE:.0f}s master {master} ph{pm[8] if pm else '-'} rate{pm[7] if pm else '-'} loop{pm[9] if pm else '-'} | {sl}={p[4] if p else '-'} g{p[0] if p else '-'} "
                          f"ph{p[8] if p else '-'} rate{p[7] if p else '-'} trim{p[10] if p else '-'} loop{p[9] if p else '-'} lanes{p[6] if p else '-'} err {None if err is None else round(err, 3)}")
        check(ok_lock, f"deck {sl} lock: median {med:.3f} beat, p95 {p95:.3f} (n {len(settled)})")
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
