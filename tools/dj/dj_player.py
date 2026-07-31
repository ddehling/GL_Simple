"""Standalone DJ player - test the autonomous DJ without the show app.

Modes:
    # Full autonomous set, live on this machine's speakers:
    python tools/dj/dj_player.py --live [--theme groove] [--minutes 30]

    # Render an autonomous set to WAV (faster than realtime, no device):
    python tools/dj/dj_player.py --wav out.wav --minutes 12 [--theme peak_heavy]

    # Audition ONE planned transition between two library tracks (renders
    # ~40s before + ~30s after the seam - the quality-assurance loop):
    python tools/dj/dj_player.py --audition "trackA.mp3" "trackB.mp3" [--wav seam.wav]

    # Single file through a deck at a rate (stretch ear-check):
    python tools/dj/dj_player.py --file song.mp3 --rate 1.06 [--wav out.wav]

Defaults: library at <repo_parent>/music (override with --dir), theme
'groove', seed random per run unless --seed.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from lib.dj import resolve_music_dir

RATE = 44100


def _write_wav(path, mix):
    from scipy.io import wavfile
    wavfile.write(path, RATE, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    print(f"wrote {path} ({len(mix) / RATE:.0f}s)")


def _render_offline(engine, dj, seconds, on_block=None):
    """Hand-pump the mixer generator; step the DJ brain between blocks."""
    gen = engine._mixer()
    next(gen)
    out = []
    block = 4410
    n_blocks = int(seconds * RATE) // block
    for i in range(n_blocks):
        buf = gen.send(block)
        out.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, 2))
        dj.step()
        if on_block:
            on_block(i * block / RATE)
    return np.concatenate(out, axis=0)


def _status_line(dj):
    s = dj.status()
    cur = s["current"] or {}
    line = (f"[{s['state']:8s}] {cur.get('title', '-')[:36]:36s} "
            f"{(cur.get('pos_s') or 0):6.1f}s  bpm {cur.get('bpm') or 0:6.1f} "
            f"arc {s['arc_phase']:.2f}/{s['arc_heat']:.2f}")
    if s["next"]:
        line += f" | next: {s['next']['title'][:24]} ({s['style'] or '?'}"
        if s["blend_in_s"] is not None:
            line += f" in {s['blend_in_s']:.0f}s"
        line += ")"
    return line


def cmd_autonomous(args):
    from lib.audio_engine import AudioEngine
    from lib.dj.system import DJSystem
    root = resolve_music_dir(args.dir)
    engine = AudioEngine()
    live = not args.wav
    dj = DJSystem(root, engine=engine, theme=args.theme,
                  night_hours=args.night_hours, seed=args.seed,
                  threaded=live)
    if not dj.start():
        return 1
    if args.setlist:
        dj.load_setlist(args.setlist)
        if not live:
            dj.step()                    # consume the request now
    minutes = args.minutes or (30.0 if live else 10.0)
    if live:
        engine.start()
        print("Playing live - Ctrl+C to stop.\n")
        try:
            t0 = time.time()
            while time.time() - t0 < minutes * 60:
                print("\r" + _status_line(dj)[:118], end="", flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        print("\nStopping...")
        dj.stop()
        time.sleep(2.5)
        engine.stop()
    else:
        print(f"Rendering {minutes:.0f} minutes offline...")
        last = [0.0]

        def progress(t):
            if t - last[0] >= 30.0:
                last[0] = t
                print(f"  {t/60:5.1f} min | " + _status_line(dj)[:100])
        mix = _render_offline(engine, dj, minutes * 60, progress)
        dj.stop()
        _write_wav(args.wav, mix)
    return 0


def cmd_audition(args):
    """Render just the seam between two specific tracks."""
    from lib.dj.db import LibraryDB
    from lib.dj.brain import Brain, load_library
    from lib.dj.themes import get_theme
    from lib.dj.audition import render_seam

    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    lib = load_library(db)

    def find(needle):
        needle = needle.lower()
        hits = [t for t in lib if needle in t.path.lower()
                or needle in t.title.lower()]
        if not hits:
            print(f"no library track matches '{needle}'")
            sys.exit(1)
        return hits[0]

    a, b = find(args.audition[0]), find(args.audition[1])
    theme = get_theme(args.theme)
    brain = Brain(lib, theme, seed=args.seed)
    score, meta = brain.score(a, b, arc_target=0.6, out_bpm=a.bpm)
    if meta is None:
        rate, eff = brain.rate_for(a.bpm, b)
        if rate is None:
            print(f"tempo-incompatible: {a.bpm:.1f} vs {b.bpm:.1f} "
                  f"(stretch outside {brain.stretch_min}-{brain.stretch_max})")
            return 1
        meta = {"rate": rate, "eff_bpm": eff, "pair": None}
    plan = brain.plan_transition(a, b, meta)
    print(f"A: {a.title}  ({a.bpm:.1f} bpm, {a.camelot})")
    print(f"B: {b.title}  ({b.bpm:.1f} bpm, {b.camelot}, rate {plan['rate']:.3f})")
    print(f"plan: {plan['style']} | A exits {plan['out_s']:.1f}s | "
          f"B enters {plan['in_s']:.1f}s | pair score {plan['pair_score']}")

    # Shared renderer (lib/dj/audition.py) - same automation the planner's
    # audition button plays, dynamic pre-roll included (the old fixed 12s
    # pre-roll squeezed every long blend).
    mix = render_seam(db, a, b, plan, status=lambda s: print("  " + s))
    wav = args.wav or os.path.join("logs", "dj_audition.wav")
    os.makedirs(os.path.dirname(os.path.abspath(wav)), exist_ok=True)
    _write_wav(wav, mix)
    if args.play:
        _play_blocking(mix)
    return 0


def cmd_file(args):
    from lib.dj.deck import Deck
    from lib.dj.features import decode_file_stereo, analyze_file
    path = args.file
    if not os.path.isfile(path):
        path = os.path.join(resolve_music_dir(args.dir), args.file)
    print(f"Analyzing {os.path.basename(path)} ...")
    r = analyze_file(path, deep=False)
    print(f"  bpm {r['bpm']:.1f} (conf {r['bpm_conf']:.2f}), key {r['camelot']}")
    deck = Deck("solo")
    deck.load(decode_file_stereo(path), grid=r["beat_grid"],
              gain_db=r["loudness_gain_db"])
    deck.gain = 1.0
    deck.set_rate(args.rate)
    deck.cue(args.start)
    deck.start()
    seconds = args.minutes * 60 if args.minutes else 30.0
    n = int(seconds * RATE)
    chunks, done = [], 0
    while done < n and not deck.finished:
        chunks.append(deck.read(4410))
        done += 4410
    mix = np.concatenate(chunks, axis=0)
    if args.wav:
        _write_wav(args.wav, mix)
    else:
        print(f"playing {len(mix)/RATE:.0f}s at rate {args.rate} "
              f"(-> {r['bpm'] * args.rate:.1f} bpm), Ctrl+C to stop")
        _play_blocking(mix)
    return 0


def _play_blocking(mix):
    import miniaudio
    pos = [0]

    def gen():
        required = yield b""
        while pos[0] < len(mix):
            chunk = mix[pos[0]:pos[0] + required]
            pos[0] += required
            required = yield chunk.astype(np.float32).tobytes()
    g = gen()
    next(g)
    dev = miniaudio.PlaybackDevice(
        output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2,
        sample_rate=RATE)
    dev.start(g)
    try:
        while pos[0] < len(mix):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    dev.close()


def main():
    ap = argparse.ArgumentParser(description="Standalone autonomous DJ")
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--theme", default="groove")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--wav", default="", help="render to this WAV instead")
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--night-hours", type=float, default=6.0)
    ap.add_argument("--setlist", default="", help="follow a saved setlist")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--audition", nargs=2, metavar=("A", "B"),
                    help="render one planned transition between two tracks")
    ap.add_argument("--play", action="store_true",
                    help="also play the audition render")
    ap.add_argument("--file", default="", help="single-file deck test")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0)
    args = ap.parse_args()
    if args.audition:
        return cmd_audition(args)
    if args.file:
        return cmd_file(args)
    if not (args.live or args.wav):
        print("pick a mode: --live, --wav out.wav, --audition A B, or --file X")
        return 1
    return cmd_autonomous(args)


if __name__ == "__main__":
    sys.exit(main())
