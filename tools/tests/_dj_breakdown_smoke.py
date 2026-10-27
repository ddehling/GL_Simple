"""THE BREAKDOWN SWAP smoke (docs/DJ_MIXING.md 5 C): a pair the library offers - a record with an 8+ bar breakdown past
its minimum play, a next record with a build into its landmark - is played headless by the RecordPlayer; the plan must
be the breakdown swap, the next record must be audible (bass closed) under the breakdown, its bass must open and the
record go out within a beat of the planned drop. Writes the rendered seam to --wav for the ear.

    python tools/tests/_dj_breakdown_smoke.py --music D:/Devel/music [--play-min 40] [--wav out.wav] [--pair A_ID,B_ID]

Exit 0 = OK, 1 = FAIL, 2 = no pair found.
"""
import math
import os
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.tests._low_priority import PRIORITY  # noqa: E402,F401
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def write_wav(path, pcm_i16):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.ascontiguousarray(pcm_i16).tobytes())


def find_pair(p, play_min):
    from lib.dj.recordplay import BREAK_MIN_BARS, MAX_TEMPO_GAP
    bs_ok = []
    for t in p.library:
        if p.fits_as_record(t) is not None:
            continue
        bar = p.bar_s(t, 0.0)
        for s in (t.sections or []):
            if s.get("kind") != "breakdown":
                continue
            n = (float(s["end_s"]) - float(s["start_s"])) / bar
            # past the minimum play from the first downbeat, late enough that the next record is decoded by then
            # (the first decode of a run takes tens of seconds: the helper warms), with room after it
            if n >= BREAK_MIN_BARS and float(s["start_s"]) >= max(play_min + 8 * bar, 150.0) and float(t.duration_s) - float(s["end_s"]) >= 40.0:
                bs_ok.append((t, s))
                break
    # the next record: a build into its landmark; the shortest files first (they decode fastest); and NEVER TWO SINGERS
    # (2026-10-26): its intro and build must be vocal-quiet up to its landmark, by the same envelope the player measures
    from lib.dj.recordplay import vocal_envelope, mix_level, RecordPlayer
    from lib.dj.features import decode_file_stereo
    builds = sorted((t for t in p.library if p.fits_as_record(t) is None and p.landmark_has_build(t)), key=lambda t: float(t.duration_s))
    quiet_b = {}

    def b_quiet(b):
        if b.id not in quiet_b:
            try:
                samples = decode_file_stereo(p.db.abs(b.path))
                env = vocal_envelope(p.music_root, b.id, mix_rms=mix_level(samples))
                del samples
                from lib.dj.recordplay import Rec
                r = Rec("x", b)
                r.vox = env
                lm = p.landmark(b)
                quiet_b[b.id] = RecordPlayer._vox_quiet(r, max(0.0, lm - 24 * p.bar_s(b, lm)), lm) is not False
            except Exception:
                quiet_b[b.id] = False
        return quiet_b[b.id]
    tried = 0
    for a, s in bs_ok:
        for b in builds:
            if b.id == a.id:
                continue
            ratio = float(a.bpm) / float(b.bpm)
            if abs(ratio - 1.0) <= 0.5 * MAX_TEMPO_GAP:
                tried += 1
                if tried > 40:
                    return None
                if b_quiet(b):
                    return a, b, s
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    play_min = arg("--play-min", 40.0, float)
    wav = arg("--wav", None)
    pair = arg("--pair", None)
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    from lib.dj.recordplay import RecordPlayer
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    p = RecordPlayer(d.db, d.music_root, d.library, theme="groove")
    p.play_min_s = play_min
    if pair:
        ia, ib = (int(x) for x in pair.split(","))
        a = next(t for t in p.library if t.id == ia)
        b = next(t for t in p.library if t.id == ib)
        s = next(x for x in a.sections if x.get("kind") == "breakdown")
    else:
        found = find_pair(p, play_min)
        if found is None:
            print("no pair in the library offers a breakdown swap at this minimum play")
            return 2
        a, b, s = found
    bar_a = p.bar_s(a, 0.0)
    print(f"A: {a.title} ({a.bpm:.1f} bpm) breakdown {s['start_s']:.0f}-{s['end_s']:.0f} s ({(s['end_s'] - s['start_s']) / bar_a:.0f} bars)")
    print(f"B: {b.title} ({b.bpm:.1f} bpm) landmark {p.landmark(b):.0f} s, build before it: {p.landmark_has_build(b)}")
    p.pick_hook = lambda: b
    engine.attach_track("dj_record", p.submix)
    if not p.start(first_track=a, threaded=False):
        print("did not start:", p.last_error)
        return 1
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    pcm, tel = [], []
    plan = None
    t0 = time.time()
    nb = 0
    while True:
        x = np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2)
        nb += 1
        if wav:
            pcm.append((np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16))
        p.step()
        decks = (p.submix.telemetry or {}).get("decks") or {}
        tel.append((p.submix.clock, {dk: (float(t.get("gain") or 0.0), bool(t.get("playing")), float((t.get("eq") or [1.0])[0])) for dk, t in decks.items()}))
        if p.cut is not None and plan is None:
            plan = dict(p.cut)
            print(f"plan after {nb * BLOCK / RATE:.0f} s rendered: kind={plan['kind']} size={plan.get('size')} start bar {plan['bar']} swap bar {plan['swap_bar']} "
                  f"cue_b {plan['cue_b']:.1f} s ratio {plan['ratio']:.4f}")
            for m in p.moves[-3:]:
                print("  ", m["text"][:220])
        if plan is not None and p.submix.clock > plan["stop_clock"] + int(10 * RATE):
            break
        if p.record is None:
            if time.time() - t0 > 180:
                print("the first record did not start in 180 s:", p.last_error)
                break
            time.sleep(0.02)
            continue
        if time.time() - t0 > 900:
            print("timeout")
            break
        if p.last_error:
            print("error:", p.last_error)
            return 1
    ok = True
    if plan is None:
        print("FAIL: no plan was made; the player said:")
        for m in p.moves[-8:]:
            print("  ", m["text"][:200])
        print("  wait_why:", p.wait_why)
        return 1
    if plan["kind"] != "breakdown":
        print(f"FAIL: the plan was a {plan['kind']}, not the breakdown swap")
        ok = False
    a_dk, b_dk = plan["a"], plan["b"]
    T_K, T_D = plan["clock"], plan["swap_clock"]
    beat = int(60.0 / float(a.bpm) * RATE)

    def at(clock):
        return min(tel, key=lambda r: abs(r[0] - clock))[1]

    # under the breakdown: both audible, B's low closed
    mid = at((T_K + T_D) // 2 + 2 * beat)
    ga, pa, la = mid.get(a_dk, (0, False, 1))
    gb, pb, lb = mid.get(b_dk, (0, False, 1))
    print(f"mid-breakdown: A gain {ga:.2f} low {la:.2f} playing {pa}; B gain {gb:.2f} low {lb:.2f} playing {pb}")
    if not (pa and pb and ga > 0.5 and gb > 0.5 and lb < 0.1 and la > 0.9):
        print("FAIL: under the breakdown the two records should both be audible with B's bass closed")
        ok = False
    after = at(T_D + 2 * beat)
    ga2, pa2, la2 = after.get(a_dk, (0, False, 1))
    gb2, pb2, lb2 = after.get(b_dk, (0, False, 1))
    print(f"two beats after the drop: A gain {ga2:.2f}; B gain {gb2:.2f} low {lb2:.2f}")
    if not (gb2 > 0.95 and lb2 > 0.95 and ga2 < 0.05):
        print("FAIL: at the drop B's bass should be open and A gone within a beat")
        ok = False
    if wav and pcm:
        write_wav(wav, np.concatenate(pcm))
        print(f"wrote {wav} (the next record enters at {T_K / RATE:.1f} s, the drop at {T_D / RATE:.1f} s)")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
