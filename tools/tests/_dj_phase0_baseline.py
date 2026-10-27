"""PHASE 0 of docs/DJ_RESCOPE.md - the bar: what the CURRENT Director sounds like, as clips for the ear.

Drives the Director headless (no audio device; the mixer is pulled by hand as the control gate does) in
layered mode with its defaults (pace normal, moments some), rotating the MIXING dial record by record so
the baseline holds every seam kind, and pressing SLAM / BREAK / SING now and then. The whole bus is kept;
afterwards every transition and moment the conductor logged becomes one clip (20 s before, 25 s after the
last move of the episode) in logs/phase0_baseline/, with index.md saying what each clip is. The rehearsal's
own offline renders of the same seams go to logs/phase0_baseline/rehearsal/ for the Phase 2 comparison.

    python tools/tests/_dj_phase0_baseline.py --music D:/Devel/music [--minutes 28] [--seed 3] [--pace 2.5]

Put three transitions from real DJ mixes you like in the same folder: they are the bar the clips are graded against.
"""
import bisect
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.tests._low_priority import PRIORITY  # noqa: E402,F401  (idle priority: it may run beside the planner)
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410                                       # 0.1 s
EPISODE_KINDS = ("bed_to", "seam", "moment", "moment_in", "event", "drop", "break", "break_auto")
MIX_CYCLE = ("auto", "cut", "blend", "morph")
PRESSES = {2.5: "slam", 6.0: "break", 10.0: "sing", 14.0: "slam", 18.0: "break", 22.0: "sing", 26.0: "slam"}   # rendered minutes -> moment


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def write_wav(path, pcm_i16):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.ascontiguousarray(pcm_i16).tobytes())


def main():
    music = arg("--music", "D:/Devel/music")
    minutes = arg("--minutes", 28.0, float)
    seed = arg("--seed", 3, int)
    pace = arg("--pace", 2.5, float)
    out = os.path.join(ROOT, "logs", "phase0_baseline")
    os.makedirs(os.path.join(out, "rehearsal"), exist_ok=True)

    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    d.set_dial("layers", "two")
    d.set_dial("pace", "normal")
    d.set_dial("moments", "some")
    d.set_dial("mixing", "auto")
    if not d.start(threaded=False):
        print("did not start:", d.last_error)
        return 2
    rc = d.rc
    rc.rng = random.Random(seed)
    rc.seam_wav_dir = os.path.join(out, "rehearsal")
    gen = engine._mixer()
    next(gen)
    gen.send(256)

    blocks = []                                    # int16 (BLOCK, 2) per 0.1 s of the bus
    t_of_block = []                                # wall time when each block was rendered
    notes = []                                     # (rendered_s, text) what this script did
    n_blocks = int(minutes * 60.0 * RATE / BLOCK)
    t0 = time.time()
    t_play0 = None
    n_play0 = 0
    seq_seen = d._song_seq
    mix_i = 0
    pressed = set()
    err_seen = {"d": None, "rc": None}
    print(f"library {len(d.library)} · rendering {minutes:.0f} min at {pace}x once the first song plays", flush=True)
    i = 0
    while i < n_blocks:
        x = np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2)
        blocks.append((np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16))
        t_of_block.append(time.time())
        d.step()
        playing = rc.master is not None and (rc._tel_deck(rc.master) or {}).get("playing")
        if t_play0 is None:
            if playing:
                t_play0, n_play0 = time.time(), i
                notes.append((i * BLOCK / RATE, f"first song playing ({time.time() - t0:.1f} s after START)"))
                print(notes[-1][1], flush=True)
            elif time.time() - t0 > 180.0:
                print("no first song in 180 s:", d.last_error, rc.last_error)
                return 2
            else:
                time.sleep(0.02)
        else:
            rendered_min = (i - n_play0) * BLOCK / RATE / 60.0
            # the MIXING dial turns once per record so every seam kind is in the baseline
            if d._song_seq != seq_seen:
                seq_seen = d._song_seq
                mix_i += 1
                d.set_dial("mixing", MIX_CYCLE[mix_i % len(MIX_CYCLE)])
                notes.append((i * BLOCK / RATE, f"new record (#{seq_seen}) - MIXING {MIX_CYCLE[mix_i % len(MIX_CYCLE)]}"))
            for at_min, kind in PRESSES.items():
                if at_min not in pressed and rendered_min >= at_min:
                    pressed.add(at_min)
                    try:
                        ok, msg = d.arm_moment(kind)
                    except Exception as e:  # noqa: BLE001
                        ok, msg = False, f"{type(e).__name__}: {e}"
                    notes.append((i * BLOCK / RATE, f"pressed {kind.upper()}: {'armed' if ok else 'refused'} - {msg}"))
                    print(f"[{rendered_min:5.1f} min] {notes[-1][1]}", flush=True)
            lag = (i - n_play0) * BLOCK / RATE / pace - (time.time() - t_play0)
            if lag > 0.05:
                time.sleep(min(lag, 0.2))
            if i % 600 == 0:
                print(f"[{rendered_min:5.1f} min] {len(rc.move_log)} moves · {len(d.events)} events · err {d.last_error} / {rc.last_error}", flush=True)
        for key, obj in (("d", d), ("rc", rc)):
            e = getattr(obj, "last_error", None)
            if e and e != err_seen[key]:
                err_seen[key] = e
                notes.append((i * BLOCK / RATE, f"ERROR ({key}): {e}"))
        i += 1
    d.stop(0.1)
    print("rendered; cutting the clips", flush=True)

    # episodes: the conductor's own moves, grouped while they fall within 45 s of each other
    def block_at(t_wall):
        k = bisect.bisect_left(t_of_block, t_wall)
        return max(0, min(len(blocks) - 1, k))
    marks = [(block_at(m["t"]), m["kind"], m["text"]) for m in rc.move_log if m["kind"] in EPISODE_KINDS]
    marks.sort()
    episodes = []
    for b, kind, text in marks:
        if episodes and b - episodes[-1]["last"] <= int(45.0 * RATE / BLOCK):
            episodes[-1]["last"] = b
            episodes[-1]["moves"].append((b, kind, text))
        else:
            episodes.append({"first": b, "last": b, "moves": [(b, kind, text)]})
    pre, post = int(20.0 * RATE / BLOCK), int(25.0 * RATE / BLOCK)
    lines = [f"# Phase 0 baseline - the current Director, {time.strftime('%Y-%m-%d %H:%M')}",
             f"seed {seed} · {minutes:.0f} rendered minutes · layers two · pace normal · moments some · MIXING rotated per record",
             f"engine in the working tree at commit {os.popen('git -C ' + ROOT + ' rev-parse --short HEAD').read().strip()} (uncommitted audio-path changes present if any)", "",
             "Grade each clip against the three real-mix transitions you put in this folder. Clip = 20 s before the first move, 25 s after the last.", ""]
    n = 0
    for ep in episodes:
        a, z = max(0, ep["first"] - pre), min(len(blocks), ep["last"] + post)
        if z - a < int(10.0 * RATE / BLOCK):
            continue
        n += 1
        kinds = sorted({k for _b, k, _t in ep["moves"]})
        name = f"clip_{n:02d}_{'-'.join(kinds)[:30]}.wav"
        write_wav(os.path.join(out, name), np.concatenate(blocks[a:z]))
        lines.append(f"## {name}  ({a * BLOCK / RATE / 60:.1f} to {z * BLOCK / RATE / 60:.1f} min of the render)")
        for b, kind, text in ep["moves"]:
            lines.append(f"- {(b - a) * BLOCK / RATE:5.1f} s  [{kind}] {text}")
        lines.append("")
    lines += ["## What the script did", ""] + [f"- {t / 60:5.1f} min  {txt}" for t, txt in notes] + ["", "## The Director's story row", ""]
    lines += [f"- {hms}  {txt}" for hms, txt in d.events]
    with open(os.path.join(out, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_wav(os.path.join(out, "full_render.wav"), np.concatenate(blocks))
    print(f"{n} clips + full_render.wav + index.md in {out}")
    return 0


if __name__ == "__main__":
    code = main()
    os._exit(code or 0)
