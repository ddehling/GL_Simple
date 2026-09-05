"""Standalone generative player - Phase 0 of docs/GENERATIVE_MUSIC_PLAN.md.
Composes notes and renders them WITHOUT the show app.

    # Render 3 minutes to WAV (numpy/numba analog rack, faster than realtime):
    python tools/gen/gen_player.py --wav out.wav --minutes 3 --style groove --bpm 124 --key 8A --seed 1

    # Same, with SoundFont instruments on the keys and pad slots:
    python tools/gen/gen_player.py --wav out.wav --fluid-slots keys,pad [--soundfont path.sf2]

    # The same notes through SuperCollider, non-realtime (needs scsynth + supriya):
    python tools/gen/gen_player.py --wav out_sc.wav --backend sc-nrt

    # Live on this machine's speakers (numpy rack via miniaudio, or sc-live):
    python tools/gen/gen_player.py --live --minutes 10 [--backend sc-live]

    # Print the composer's phrase log (section / energy / chords / motif op):
    python tools/gen/gen_player.py --wav out.wav --log

    # Render a Strudel pattern file through the same rack (node + `npm install` in tools/gen/strudel):
    python tools/gen/gen_player.py --wav out.wav --strudel media/patterns/example.js

Styles: groove | downtempo | ambient.  Key: Camelot (8A) or name (Am, F#m, C).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                       # noqa: E402
from lib.gen.composer import Composer          # noqa: E402

BLOCK = 2048


def _arc(minutes):
    """A simple set arc: rise to a peak two-thirds in, then wind down."""
    def f(bar):
        bpm_guess = 120.0
        prog = (bar * 4 * 60.0 / bpm_guess) / max(1.0, minutes * 60.0)
        if prog < 0.66:
            return 0.35 + 0.65 * (prog / 0.66)
        return max(0.2, 1.0 - (prog - 0.66) / 0.34 * 0.8)
    return f


def compose(args):
    c = Composer(args.style, bpm=args.bpm, key=args.key, seed=args.seed, arc_fn=_arc(args.minutes))
    if args.strudel:
        from lib.gen.composer.strudel import StrudelBridge, StrudelSource
        bridge = StrudelBridge()
        bridge.start()
        src = StrudelSource(bridge, c.style["slots"].keys())
        src.load(open(args.strudel, encoding="utf-8").read())
        c.pattern_source = src
        print(f"Strudel pattern from {args.strudel}")
    total = int(args.minutes * 60 * RATE)
    phrases = list(c.phrases_until(total))
    if args.log:
        for bar, sec, e, ch, op in c.log:
            print(f"bar {bar:4d}  {sec:7s}  energy {e:.2f}  {' '.join(ch):16s}  lead:{op}")
        print(f"key {c.key}  bpm {c.bpm:.1f}  phrases {len(phrases)}  notes {sum(len(p.events) for p in phrases)}")
    return c, phrases, total


def build_rack(c, args):
    from lib.gen.synth import SynthRack
    fluid, slots = None, ()
    if args.fluid_slots:
        from lib.gen.synth.fluid import FluidVoice
        fluid = FluidVoice(args.soundfont)
        slots = tuple(s for s in args.fluid_slots.split(",") if s in c.style["slots"])
        print(f"FluidSynth on {slots} with {fluid.path}")
    return SynthRack(c.style, c.bpm, fluid=fluid, fluid_slots=slots, seed=args.seed)


def render_numpy(c, phrases, total, args):
    rack = build_rack(c, args)
    for p in phrases:
        rack.schedule(p.events)
    t0 = time.time()
    out = []
    while rack.clock < total:
        out.append(rack.render(BLOCK))
    mix = np.concatenate(out)[:total]
    dt = time.time() - t0
    print(f"rendered {total / RATE:.0f}s in {dt:.1f}s ({total / RATE / dt:.1f}x realtime), "
          f"peak {rack.stats['peak']:.3f}, notes {rack.stats['notes']}")
    return mix


def write_wav(path, mix):
    try:
        import soundfile as sf
        sf.write(path, np.clip(mix, -1, 1), RATE, subtype="PCM_16")
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(path, RATE, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    print(f"wrote {path}")


def live_numpy(c, phrases, total, args):
    import miniaudio
    rack = build_rack(c, args)
    for p in phrases:
        rack.schedule(p.events)

    def gen():
        n = yield b""
        while rack.clock < total:
            buf = rack.render(n)
            n = yield buf.astype(np.float32).tobytes()

    g = gen(); next(g)
    dev = miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2, sample_rate=RATE)
    dev.start(g)
    print("playing... ctrl-c to stop")
    try:
        while rack.clock < total:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    dev.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="output WAV path")
    ap.add_argument("--live", action="store_true", help="play on the default device")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--style", default="groove")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--key", default="8A")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "sc-nrt", "sc-live"])
    ap.add_argument("--fluid-slots", default="", help="comma list of slots rendered by FluidSynth, e.g. keys,pad")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--strudel", metavar="FILE.js", help="render this Strudel pattern (one cycle = one bar) instead of the rule composer")
    args = ap.parse_args()
    if not args.wav and not args.live:
        ap.error("need --wav PATH or --live")
    c, phrases, total = compose(args)
    events = [e for p in phrases for e in p.events]
    if args.backend == "numpy":
        if args.live:
            live_numpy(c, phrases, total, args)
        else:
            write_wav(args.wav, render_numpy(c, phrases, total, args))
    elif args.backend == "sc-nrt":
        from lib.gen.backends.sc import render_nrt
        t0 = time.time()
        path = render_nrt(events, total / RATE, args.wav, c.style)
        print(f"scsynth NRT rendered {total / RATE:.0f}s in {time.time() - t0:.1f}s -> {path}")
    else:
        from lib.gen.backends.sc import SCLive
        live = SCLive(c.style, c.bpm)
        try:
            live.schedule(events)
            print("scsynth playing... ctrl-c to stop")
            while live.now_sample() < total:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            live.quit()


if __name__ == "__main__":
    main()
