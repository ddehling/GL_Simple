"""Listening loop: render short excerpts of every style x section (and,
optionally, a full set per style) to WAV, with the mix numbers that
matter next to each file, so a sound-design pass can be A/B'd against
the previous one by ear AND by number.

    python tools/gen/listen.py                       # 30 s per style x section -> logs/listen/<stamp>/
    python tools/gen/listen.py --seconds 45 --styles groove
    python tools/gen/listen.py --full 4              # + a 4-minute set per style
    python tools/gen/listen.py --out /tmp/pass7      # name the folder (compare with a previous one)
    python tools/gen/listen.py --compare logs/listen/2026-09-06_10-00

Each folder gets a stats.json and a README.txt table: RMS, peak, crest,
L/R correlation, octave-band tilt, per-slot solo RMS. --compare prints
the deltas against another folder's stats.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.composer.styles import STYLES                 # noqa: E402
from lib.gen.synth import SynthRack                        # noqa: E402

BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]


def _db(x):
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) + 1e-9)


def stats(x, solos=None):
    from scipy.signal import welch
    rms = _db(x)
    pk = float(np.abs(x).max())
    f, P = welch(x.mean(axis=1), fs=RATE, nperseg=8192)
    bands = {f"{lo}-{hi}": round(float(10 * np.log10(P[(f >= lo) & (f < hi)].sum() + 1e-12)), 1) for lo, hi in BANDS}
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if pk > 0 else 1.0
    out = {"rms_db": round(rms, 1), "peak": round(pk, 3), "crest_db": round(20 * np.log10(pk / (10 ** (rms / 20) + 1e-9) + 1e-9), 1),
           "lr_corr": round(corr, 3), "bands_db": bands}
    if solos:
        out["slots_db"] = {k: round(v, 1) for k, v in solos.items()}
    return out


def render(style, section, seconds, seed, bpm=None, solo_slots=False):
    """Render `seconds` of `style` held in `section` (None = the natural
    form from the top). Returns (audio, slot solo dBFS or None, composer)."""
    def compose():
        c = Composer(style, bpm=bpm, key="8A", seed=seed)
        if section is not None:
            c.form.section = section
            c.form.bars_left = 10 ** 6
            c.form.hold = True
        return c

    def run(mute=()):
        c = compose()
        c.muted = set(mute)
        rack = SynthRack(c.style, c.bpm, seed=seed)
        rack.warm_up()
        for p in c.phrases_until(int(seconds * RATE)):
            rack.schedule(p.events)
        blocks = []
        while rack.clock < seconds * RATE:
            blocks.append(rack.render(2048))
        return np.concatenate(blocks)[: int(seconds * RATE)], c

    x, c = run()
    solos = None
    if solo_slots:
        solos = {}
        slots = sorted(c.style["slots"])
        for s in slots:
            y, _ = run(mute=[t for t in slots if t != s])
            solos[s] = _db(y)
    return x, solos, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--sections", default="", help="comma list; default = every section of the style")
    ap.add_argument("--full", type=float, default=0.0, help="also render a full set of N minutes per style")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--solo", action="store_true", help="also measure per-slot solo levels (slow: one render per slot)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="a previous listen folder to diff against")
    args = ap.parse_args()
    import soundfile as sf
    stamp = time.strftime("%Y-%m-%d_%H-%M")
    out_dir = args.out or os.path.join("logs", "listen", stamp)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    lines = [f"listen {stamp}  seed={args.seed}  {args.seconds:.0f} s per excerpt", ""]
    lines.append(f"{'file':34s} {'rms':>6s} {'peak':>5s} {'crest':>5s} {'L/R':>5s}  bands(dB) 20..16k")
    for style in [s for s in args.styles.split(",") if s]:
        sections = [s for s in args.sections.split(",") if s and s in STYLES[style]["sections"]] or list(STYLES[style]["sections"])
        jobs = [(sec, args.seconds) for sec in sections]
        if args.full > 0:
            jobs.append((None, args.full * 60.0))
        for sec, seconds in jobs:
            name = f"{style}_{sec or 'set'}"
            t0 = time.time()
            x, solos, c = render(style, sec, seconds, args.seed, solo_slots=args.solo and sec is not None)
            path = os.path.join(out_dir, name + ".wav")
            sf.write(path, np.clip(x, -1, 1), RATE, subtype="PCM_16")
            st = stats(x, solos)
            st["seconds"] = seconds
            st["render_x_realtime"] = round(seconds / max(time.time() - t0, 1e-3), 1)
            st["sections"] = [s for _, s, _ in c.form.history] if sec is None else [sec]
            results[name] = st
            bands = " ".join(f"{v:5.0f}" for v in st["bands_db"].values())
            lines.append(f"{name:34s} {st['rms_db']:6.1f} {st['peak']:5.2f} {st['crest_db']:5.1f} {st['lr_corr']:5.2f}  {bands}")
            if solos:
                lines.append("    " + "  ".join(f"{k}:{v:.0f}" for k, v in st["slots_db"].items()))
            print(lines[-1] if not solos else lines[-2] + "\n" + lines[-1], flush=True)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    if args.compare:
        try:
            with open(os.path.join(args.compare, "stats.json"), encoding="utf-8") as fh:
                prev = json.load(fh)
            lines += ["", f"vs {args.compare}:"]
            for name, st in results.items():
                if name not in prev:
                    continue
                q = prev[name]
                d_b = {k: round(st["bands_db"][k] - q["bands_db"].get(k, 0.0), 1) for k in st["bands_db"]}
                lines.append(f"{name:34s} rms {st['rms_db'] - q['rms_db']:+5.1f}  crest {st['crest_db'] - q['crest_db']:+5.1f}  "
                             f"L/R {st['lr_corr'] - q['lr_corr']:+5.2f}  bands " + " ".join(f"{v:+4.0f}" for v in d_b.values()))
        except Exception as e:  # noqa: BLE001
            lines.append(f"(compare failed: {e})")
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-(len(results) + 3):]) if args.compare else f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
