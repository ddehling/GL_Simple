"""Loop material for the layer deck: find it, slice it, tempo-fit it.

The layer is a percussion bed ridden UNDER the playing track (deck C, see
DJSystem._do_layer). Two rules shape everything here, and both come from
outside this file:

  DRUMS ONLY. Real DJ tools are percussion because anything pitched
  clashes in key with whatever it is laid over. There is no key matching
  in this module and there should not be - the material must be
  key-neutral by construction.

  IT MUST NOT BE THE SONG THAT IS PLAYING. Four operator "moments" were
  retired in a day because "the payoff was still the same song"
  (system._do_moment). Looping the live track's own drums under itself is
  that same trap, so `candidates()` refuses the playing track.

Two sources, deliberately both:

  library  a scored loop point (db.loops_for, self-similarity x
           repetitiveness) sliced out of a track's DRUMS stem. Needs no
           new analysis and no new files, but only ~42% of the library
           has stems rendered, and demucs drums carry some bleed.
  file     media/loops/*.wav - the real DJ-tool path. BPM comes from the
           filename (`whomp_124.wav`), which is the universal convention
           for loop packs, so this needs no analysis pass at all.

Having both matters for judging the feature: if stem-derived loops sound
muddy, that must not be mistaken for "layering doesn't work".

Nothing here touches the mixer or the plan - it returns buffers.
"""
import os
import re

import numpy as np

RATE = 44100
# Deck.set_rate clips to [0.90, 1.10]; stay inside it with a margin so a
# loop is never silently played at the wrong tempo.
MAX_STRETCH = 0.08
# A bed wants a whole number of bars. 4 is the shortest that reads as a
# pattern rather than a stutter; 8 is the DJ-tool default.
GOOD_BEATS = (16, 8, 32)
# Every prepared loop leaves here at the same ENERGY, so system.LAYER_GAIN
# is one decision rather than a per-loop lottery (see prepare()). A ~0.12
# RMS bed at gain 0.35 sits roughly 13 dB under a mastered club track.
TARGET_RMS = 0.12
PEAK_CEIL = 0.9         # ...but never let a spiky loop eat the headroom
# How much of the material FOLLOWING a library slice is blended back over
# its head so the wrap continues rather than truncates (see prepare()).
# Long enough to carry a snare/hat decay, short enough not to smear the
# downbeat that lands right after it. Curated files skip this - a DJ tool
# is already built to loop, and it has no "following" material to borrow.
XFADE_S = 0.030
_BPM_IN_NAME = re.compile(r"(?:^|[_\-. ])(\d{2,3})(?:bpm)?(?:[_\-. ]|$)",
                          re.IGNORECASE)


def loops_dir(music_root):
    """Where curated DJ tools live. Beside the library, not in the repo -
    it is user material, like the music itself."""
    return os.path.join(os.path.dirname(os.path.abspath(music_root)),
                        "media", "loops")


def _bpm_from_name(path):
    """`whomp_124.wav` -> 124.0. Returns None when the name says nothing;
    the caller decides whether to guess or skip."""
    stem = os.path.splitext(os.path.basename(path))[0]
    best = None
    for m in _BPM_IN_NAME.finditer(stem):
        v = float(m.group(1))
        if 60.0 <= v <= 200.0:
            best = v            # last plausible number wins ("kit3_124")
    return best


def _fit_rate(loop_bpm, target_bpm):
    """Playback rate that puts `loop_bpm` at `target_bpm`, also trying the
    half/double reads a loop pack routinely mislabels. None = out of reach."""
    if not loop_bpm or not target_bpm:
        return None
    best = None
    for mult in (1.0, 2.0, 0.5):
        r = target_bpm / (loop_bpm * mult)
        if abs(np.log(r)) <= abs(np.log(1.0 + MAX_STRETCH)):
            if best is None or abs(np.log(r)) < abs(np.log(best)):
                best = r
    return best


# ---------------------------------------------------------------- sources
def _file_candidates(music_root, target_bpm):
    d = loops_dir(music_root)
    out = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for nm in names:
        if not nm.lower().endswith((".wav", ".flac", ".aiff", ".aif")):
            continue
        path = os.path.join(d, nm)
        bpm = _bpm_from_name(path)
        if bpm is None:
            continue            # unlabelled: skip rather than guess wrong
        rate = _fit_rate(bpm, target_bpm)
        if rate is None:
            continue
        out.append({"kind": "file", "path": path, "bpm": bpm,
                    "rate": rate, "label": os.path.splitext(nm)[0],
                    "score": 1.0})
    return out


def _library_candidates(db, library, target_bpm, exclude_ids):
    out = []
    for t in library:
        if t.id in exclude_ids or not getattr(t, "has_stems", False):
            continue
        rate = _fit_rate(t.bpm, target_bpm)
        if rate is None:
            continue
        try:
            loops = db.loops_for(t.id)
        except Exception:
            continue
        for lp in loops[:2]:            # the two best per track is plenty
            beats = int(lp.get("beats") or 0)
            if beats not in GOOD_BEATS:
                continue
            out.append({"kind": "library", "track": t, "bpm": t.bpm,
                        "rate": rate, "beats": beats,
                        "start_s": float(lp["start_s"]),
                        # the timestamp disambiguates two same-length
                        # loops lifted from different parts of one track
                        "label": (f"{t.title[:28]} · {beats}b @"
                                  f"{int(lp['start_s']) // 60}:"
                                  f"{int(lp['start_s']) % 60:02d}"),
                        "score": float(lp.get("score") or 0.0)})
    return out


def candidates(db, library, target_bpm, exclude_ids=(), music_root=None,
               limit=12):
    """Loop options playable at `target_bpm` inside the deck's rate wall.

    `exclude_ids` MUST carry the currently-playing track (see the module
    docstring): a bed made of the song it rides under adds nothing."""
    ex = set(exclude_ids or ())
    out = []
    if music_root:
        out += _file_candidates(music_root, target_bpm)
    if db is not None and library:
        out += _library_candidates(db, library, target_bpm, ex)
    # Least stretch first, then loop quality - a bed that had to be pulled
    # 7% is audibly wrong before its pattern even matters.
    out.sort(key=lambda c: (abs(np.log(c["rate"])) * 4.0 - c["score"]))
    return out[:limit]


# ---------------------------------------------------------------- loading
def _load_file(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if sr != RATE:
        n = int(round(len(x) * RATE / float(sr)))
        idx = np.linspace(0, len(x) - 1, n)
        x = np.stack([np.interp(idx, np.arange(len(x)), x[:, ch])
                      for ch in range(2)], axis=1).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def prepare(cand, db=None, music_root=None):
    """Turn a candidate into something deck C can mount.

    Returns {"samples", "loop_s", "rate", "label", "bpm"} or None. The
    buffer IS the loop: it starts at frame 0 and `loop_s` is its length,
    so the deck loops [0, loop_s] and the caller needs no offsets."""
    try:
        if cand["kind"] == "file":
            x = _load_file(cand["path"])
        else:
            from lib.dj.stems import load_stems
            t = cand["track"]
            stems = load_stems(music_root, t.id)
            drums = (stems or {}).get("drums")
            if drums is None or not len(drums):
                return None
            # THE GRID'S OWN PERIOD, not 60/bpm. The stored bpm is rounded
            # and the grid can carry a different period per segment; a
            # length that is off by even a few ms per beat puts the wrap
            # off the beat, which reads as a glitch however clean the
            # crossfade is.
            per = 60.0 / max(t.bpm, 1e-6)
            for g in (t.grid or []):
                if g["start_s"] <= cand["start_s"] <= g["end_s"] \
                        and g.get("period_s"):
                    per = float(g["period_s"])
                    break
            a = int(round(cand["start_s"] * RATE))
            b = a + int(round(cand["beats"] * per * RATE))
            if a < 0 or b > len(drums):
                return None
            # Blend the material that NATURALLY FOLLOWED the slice over
            # its own head, so the wrap continues whatever was decaying
            # instead of truncating it. Standard practice for cutting a
            # loop out of continuous audio, and free here because the
            # following audio is right there in the stem.
            #
            # HONEST SCOPE: this was added chasing an operator-reported
            # click and it does NOT explain that click. Measured on the
            # deck's real output, the wrap carries a max sample step of
            # 0.00065 against a 99.9th-percentile step of 0.065 elsewhere
            # in the same loop - 100x below the loop's own transients,
            # identical with and without this crossfade. The wrap is not
            # the discontinuity. Remaining suspects are artefacts inside
            # the demucs drum stem (which would recur once per loop and
            # read the same way) and slices that are simply not musically
            # seamless. Curated media/loops/ files skip this entirely:
            # a DJ tool is built to loop and has no "following" audio.
            xf = min(int(XFADE_S * RATE), (b - a) // 8)
            tail = None
            if xf > 32 and b + xf <= len(drums):
                tail = np.asarray(drums[b:b + xf], dtype=np.float32)
            x = np.ascontiguousarray(drums[a:b], dtype=np.float32)
            if tail is not None:
                f = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
                x[:xf] = x[:xf] * np.sqrt(f) + tail * np.sqrt(1.0 - f)
    except Exception:
        return None
    if x is None or len(x) < RATE // 8:
        return None
    peak = float(np.abs(x).max())
    if peak < 0.02:
        return None                      # that slice has no drums in it
    # NORMALISE BY RMS, NOT PEAK. A bed's perceived level tracks its
    # energy, and drum loops are spiky: peak-normalising a sparse conga
    # pattern and a dense break to the same 0.7 leaves them ~15 dB apart
    # to the ear, so LAYER_GAIN would mean something different for every
    # loop. Measured on a real stem loop under a real track, peak-normalised
    # at gain 0.35: RMS lift x1.003 - inaudible - while the peak rose 42%.
    # A ceiling still bounds the transients so the master has headroom.
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    if rms < 1e-4:
        return None
    x = x * (TARGET_RMS / rms)
    over = float(np.abs(x).max())
    if over > PEAK_CEIL:
        x = x * (PEAK_CEIL / over)
    x = x.astype(np.float32)
    # NO EDGE FADES. An earlier version faded 4 ms to silence at both ends
    # "to avoid a click" and guaranteed one instead: Deck's wrap blend
    # (LOOP_XFADE, 128 frames = 2.9 ms) crossfades the loop START against
    # the material just BEFORE the loop end, so fading both edges to zero
    # left it blending silence with silence - a ~4 ms amplitude notch on
    # every single repeat, which is exactly the click the operator heard.
    # The wrap is the deck's job; the launch is covered by the gain fade
    # in _do_layer, which ramps from 0 over ~2 bars.
    return {"samples": x, "loop_s": len(x) / RATE, "rate": cand["rate"],
            "label": cand["label"], "bpm": cand["bpm"],
            "kind": cand["kind"]}
