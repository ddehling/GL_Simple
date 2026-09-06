"""Build the melody model from public-domain scores (music21's corpus):
what real melodies do - how often they step vs leap, how leaps resolve,
which two-bar rhythms recur, how phrases rise and fall and cadence.

    python tools/gen/melody_corpus.py                 # -> lib/gen/composer/data/melody_model.json
    python tools/gen/melody_corpus.py --max 400       # quicker: fewer scores
    python tools/gen/melody_corpus.py --collections ryansMammoth,essenFolksong,oneills1850,bach

Per score: the top part (or the soprano of a chorale), key by analysis,
notes -> (scale-degree index with octave, onset in 16th steps). Only
duple-metre scores whose onsets quantise to a 16th grid are used
(fiddle tunes in 6/8 are skipped). Two-bar windows (32 steps) become
rhythm cells; the pitch line becomes an order-2 model over degree
intervals conditioned on metric strength; phrase-final notes give the
cadence table; window contours give the shape table.

The result is small (tens of kB), committed, and read by
lib/gen/composer/melody_model.py. Rebuild only when you change what is
extracted."""
import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT = os.path.join("lib", "gen", "composer", "data", "melody_model.json")
STEPS_PER_BAR = 16
WINDOW = 2 * STEPS_PER_BAR


def _degree(pitch, key):
    """Scale-degree index (0 = tonic, +7 per octave) of a music21 pitch in
    key, or None when chromatic."""
    from music21 import scale as m21scale
    sc = key.getScale()
    try:
        deg = sc.getScaleDegreeFromPitch(pitch, comparisonAttribute="pitchClass")
    except Exception:
        deg = None
    if deg is None:
        return None
    octave = pitch.octave if pitch.octave is not None else 4
    tonic_oct = key.tonic.octave if key.tonic.octave is not None else 4
    # index relative to the tonic in the same octave as the note, then carry
    rel = (deg - 1)
    ref = key.tonic.pitchClass
    below = (pitch.pitchClass - ref) % 12
    # octave number of the note relative to the tonic just below it
    tonic_below_midi = pitch.midi - below
    n_oct = (tonic_below_midi - (12 * (tonic_oct + 1) + ref)) // 12
    return rel + 7 * n_oct


def extract(score, key):
    """[(step_abs, degree_index, strength)] for the top line, quantised
    to 16ths; None if the score is not duple or does not quantise."""
    from music21 import meter
    parts = score.parts if hasattr(score, "parts") and len(score.parts) else [score]
    part = parts[0]
    ts = part.recurse().getElementsByClass(meter.TimeSignature)
    ts = ts[0] if ts else None
    if ts is None or ts.numerator not in (2, 4) or ts.denominator not in (2, 4):
        return None
    beat_len = 4.0 / ts.denominator            # quarter notes per beat
    bar_len = ts.numerator * beat_len
    out = []
    flat = part.flatten().notes
    for n in flat:
        if n.isChord:
            p = max(n.pitches, key=lambda x: x.midi)
        else:
            p = n.pitch
        off = float(n.offset)
        step = off / bar_len * STEPS_PER_BAR
        if abs(step - round(step)) > 0.05:
            return None                          # triplets / odd grid: skip the piece
        deg = _degree(p, key)
        if deg is None:
            continue
        s = int(round(step))
        strength = 2 if s % 4 == 0 else (1 if s % 2 == 0 else 0)
        out.append((s, deg, strength))
    return out if len(out) >= 16 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=1200)
    ap.add_argument("--collections", default="ryansMammoth,essenFolksong,oneills1850,bach")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    from music21 import corpus, converter  # noqa: F401
    wanted = set(args.collections.split(","))
    paths = [p for p in corpus.getPaths() if os.path.basename(os.path.dirname(str(p))) in wanted]
    paths = paths[: args.max]
    print(f"{len(paths)} candidate scores from {sorted(wanted)}")
    intervals = collections.Counter()          # (prev_iv, strength) -> next_iv counts
    first_iv = collections.Counter()
    rhythms = collections.Counter()
    contours = collections.Counter()
    cadences = collections.Counter()
    leap_resolution = collections.Counter()    # after |iv|>=3: next iv sign relative
    iv_hist = collections.Counter()
    starts = collections.Counter()
    used = 0
    for i, path in enumerate(paths):
        try:
            sc = corpus.parse(path)
            key = sc.analyze("key")
            notes = extract(sc, key)
        except Exception:
            notes = None
        if not notes:
            continue
        used += 1
        degs = [d for _, d, _ in notes]
        ivs = [b - a for a, b in zip(degs, degs[1:])]
        ivs = [max(-7, min(7, v)) for v in ivs]
        for k in range(len(ivs)):
            iv_hist[ivs[k]] += 1
            strength = notes[k + 1][2]
            prev = ivs[k - 1] if k > 0 else 0
            intervals[(prev, strength, ivs[k])] += 1
            if k > 0 and abs(ivs[k - 1]) >= 3:
                leap_resolution[("back" if ivs[k] * ivs[k - 1] < 0 else "on") if ivs[k] != 0 else "same"] += 1
        if ivs:
            first_iv[ivs[0]] += 1
        starts[degs[0] % 7] += 1
        # two-bar windows
        last_step = notes[-1][0]
        for w0 in range(0, last_step + 1, WINDOW):
            win = [(s - w0, d) for s, d, _ in notes if w0 <= s < w0 + WINDOW]
            if len(win) < 3:
                continue
            steps = tuple(sorted({s for s, _ in win}))
            rhythms[steps] += 1
            ds = [d for _, d in win]
            span = max(ds) - min(ds)
            if span == 0:
                shape = "flat"
            else:
                third = max(1, len(ds) // 3)
                a, b, c = sum(ds[:third]) / third, sum(ds[third:2 * third]) / max(1, len(ds[third:2 * third])), sum(ds[-third:]) / third
                if b > a and b > c:
                    shape = "arch"
                elif b < a and b < c:
                    shape = "valley"
                elif c > a:
                    shape = "rise"
                else:
                    shape = "fall"
            contours[shape] += 1
        # phrase ends: the last note of every 8-bar span and the piece
        for w0 in range(0, last_step + 1, 8 * STEPS_PER_BAR):
            seg = [d for s, d, _ in notes if w0 <= s < w0 + 8 * STEPS_PER_BAR]
            if seg:
                cadences[seg[-1] % 7] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} scanned, {used} used")
    tot_iv = sum(iv_hist.values()) or 1
    model = {
        "source": sorted(wanted), "scores_used": used,
        "interval_hist": {str(k): round(v / tot_iv, 5) for k, v in sorted(iv_hist.items())},
        "intervals": [[p, s, n, c] for (p, s, n), c in intervals.items() if c >= 2],
        "first_interval": {str(k): v for k, v in first_iv.items()},
        "rhythms": [[list(k), c] for k, c in rhythms.most_common(400)],
        "contours": dict(contours),
        "cadence_degrees": {str(k): v for k, v in cadences.items()},
        "start_degrees": {str(k): v for k, v in starts.items()},
        "leap_resolution": dict(leap_resolution),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, separators=(",", ":"))
    step_share = sum(v for k, v in iv_hist.items() if abs(k) <= 1) / tot_iv
    print(f"wrote {args.out} ({os.path.getsize(args.out) // 1024} kB): {used} scores, {tot_iv} intervals, "
          f"steps+repeats {step_share:.0%}, leap resolution {dict(leap_resolution)}, contours {dict(contours)}, "
          f"cadence {cadences.most_common(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
