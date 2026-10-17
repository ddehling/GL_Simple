"""The mashup smoke (docs/DJ_MASHUP.md): LAYERS mashup, left alone for a rendered stretch, then the pads.

    python tools/tests/_dj_mashup_smoke.py --music D:/Devel/music [--seed 1] [--seconds 600] [--moments some]

Checks: three songs come in; scenes change on their bars (a scene every mash_bars); every scene has at least two
lanes sounding; a song rotates; a pad press moves the part on the next bar; no dead bars; no error.
"""
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410
PACE = 1.5


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    music = arg("--music", "D:/Devel/music")
    seed = arg("--seed", 1, int)
    seconds = arg("--seconds", 600.0, float)
    moments = arg("--moments", "some")
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    start_mash = "--start-mashup" in sys.argv          # LAYERS mashup chosen before START
    d.set_dial("layers", "mashup" if start_mash else "two")
    d.set_dial("moments", moments)
    if not d.start(threaded=False):
        print(f"FAIL: did not start: {d.last_error}")
        return 1
    rc = d.rc
    rc.rng = random.Random(seed)
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
    pump(4 * bar_blocks())
    if not start_mash:
        d.set_dial("layers", "mashup")
    print(f"LAYERS mashup at bar {rc.bar_n}: policy {rc.policy}, a scene every {rc.mash_bars} bars")
    t0 = len(audio)
    n_moves = 0
    t_end = t0 + int(seconds * RATE / BLOCK)
    scene_lanes = []
    pads_done, pad_results = 0, []
    entered_titles = set()
    while len(audio) < t_end:
        pump(bar_blocks())
        for m in rc.move_log[n_moves:]:
            if m["kind"] in ("mashup", "mashup_plan"):
                print(f"  {m.get('clock_s', 0):6.0f}s {m['kind']:11s} {m['text'][:130]}")
            if m["kind"] == "mashup" and m["text"].startswith("SCENE"):
                pump(2 * bar_blocks())                 # (the scene's lanes have landed)
                lanes = {ln: d_ for ln, d_ in rc.lanes.items() if d_ is not None and d_ in rc.songs}
                scene_lanes.append(lanes)
                for d_ in lanes.values():
                    entered_titles.add(rc.songs[d_].track.title)
        n_moves = len(rc.move_log)
        ms = rc.mash_status()
        # the pads: twice, once the three songs are ready, half way through
        if ms and pads_done < 2 and len(audio) > t0 + (0.45 + 0.1 * pads_done) * (t_end - t0) and all(s["ready"] for s in ms["slots"]):
            slot = (pads_done + 1) % 3
            lane = "vocals" if pads_done == 0 else "drums"
            want_deck = ms["slots"][slot]["deck"]
            before = rc.lanes.get(lane)
            msg = d.mash_pad(slot, lane)
            pump(3 * bar_blocks())
            after = rc.lanes.get(lane)
            ok = (after == want_deck) if before != want_deck else (after is None)
            pad_results.append((slot, lane, ok, msg))
            print(f"  PAD slot {slot + 1} {lane}: {msg} -> {'acted' if ok else 'DID NOT ACT'}")
            pads_done += 1
        if d.last_error or rc.last_error:
            print(f"  error: {d.last_error or rc.last_error}")
            break
    back_ok = None
    if "--back" in sys.argv:
        # LAYERS back to two: the record policy takes over from the song holding the drums; the room keeps playing
        d.set_dial("layers", "two")
        pump(24 * bar_blocks())
        rec = rc.songs.get(rc.record_deck)
        back_ok = rc.policy == "record" and rc._mash is None and rec is not None and rc.lanes.get("drums") is not None
        print(f"  BACK to two: policy {rc.policy}, record {rec.track.title[:24] if rec else None}, lanes {rc.lanes} -> {'ok' if back_ok else 'NOT OK'}")
    d.stop(0.1)
    mono = np.concatenate(audio[t0:]).mean(axis=1) if len(audio) > t0 + 10 else np.zeros(1)
    nb = int(bar_blocks() * BLOCK)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(0, len(mono) - nb, nb)]
    dead = [i for i, r in enumerate(rms) if r < -45.0]
    ms = rc.mash_status() or {}
    thin = [l for l in scene_lanes if len(l) < 2]
    expect = int(seconds / (rc.mash_bars * rc._bar_clock() / RATE)) // 2
    print(f"\nscenes {len(scene_lanes)} (expected at least {expect}), thin scenes {len(thin)}, songs heard {len(entered_titles)}, "
          f"rotations {ms.get('rotations')}, pads {[(s + 1, ln, ok) for s, ln, ok, _ in pad_results]}, dead bars {len(dead)} of {len(rms)}")
    fails = []
    if len(scene_lanes) < max(3, expect):
        fails.append(f"only {len(scene_lanes)} scenes")
    if thin:
        fails.append(f"{len(thin)} scenes with fewer than two lanes")
    if len(entered_titles) < 3:
        fails.append(f"only {len(entered_titles)} songs heard")
    if seconds >= 500 and not ms.get("rotations"):
        fails.append("no rotation")
    if any(not ok for _s, _l, ok, _m in pad_results) or len(pad_results) < 2:
        fails.append(f"pads: {pad_results}")
    if dead:
        fails.append(f"{len(dead)} dead bars")
    if back_ok is False:
        fails.append("LAYERS back to two did not hand the room to the record policy")
    if d.last_error or rc.last_error:
        fails.append(f"error: {d.last_error or rc.last_error}")
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
