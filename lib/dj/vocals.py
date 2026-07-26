"""Optional ML vocal detection - the scanner's second pass.

WHY A MODEL: five classical discriminators were prototyped against
labeled tracks from the real library (mid-band share x chroma clarity,
off-grid modulation spectra, center-channel stereo placement, HPSS +
upper-harmonic vibrato ridges + cepstral peak prominence, and
repetition-based foreground separation). None separated sung vocals
from expressive melodic synths in organic house - the synths glide,
wobble and vary exactly like voices. The only reliable sensor is a
source-separation model's vocal stem: we run Demucs (htdemucs) over
sampled windows and measure the vocal stem's energy share of the mix.

DEPENDENCIES ARE OPTIONAL: torch + demucs live in
requirements-dj-vocals.txt, NEVER in requirements.txt (Pi installs must
stay light). Without them `available()` is False, sections keep
vocalness 0.0 and no vocal tags are issued - an honest "unknown", not a
guess from features proven not to work.

SCALE: sections.vocalness stores min(vocal_energy_share * 2.5, 1.0), so
0.5 == a clear lead vocal (20% of mix energy). Existing consumers
(brain clash gates, waveform underline at ~0.5) keep their thresholds.
Tracks get axes["vocal"] = fraction of duration with vocalness > 0.5
and axes["vocal_src"] = "demucs" so tag logic knows real data exists.
"""
import json

import numpy as np

RATE = 44100
WIN_S = 8.0                 # analysis window fed to the model
HOP_S = 8.0                 # contiguous coverage (was 24 s / ~33%): the
                            # fine curve axes["vc"] now resolves individual
                            # vocal phrases, which seam planning uses to
                            # place swaps in the GAPS between vocal lines
                            # and to gate stem styles. Cost only rises on
                            # tracks that pass the 6-window screen - pure
                            # instrumentals still short-circuit.
BATCH = 1                   # windows per model call. KEEP AT 1: batching
                            # multiplies htdemucs' transformer memory per
                            # segment - BATCH=4 hard-crashed the process
                            # (no Python traceback) partway into the
                            # library on the first full pass.
SHARE_SCALE = 2.5           # energy share -> stored vocalness
QUIET_RMS = 1e-3            # ~-60 dBFS: below this a window is silence -
                            # per-window std normalization would amplify
                            # noise/reverb tails and the model's output
                            # reads as full-scale "vocals" (outro sections
                            # scored 1.0 before this floor)


def available():
    try:
        import torch                          # noqa: F401
        import demucs.pretrained              # noqa: F401
        return True
    except Exception:
        return False


class VocalAnalyzer:
    """Loads htdemucs once; scores tracks. Construct only if available()."""

    def __init__(self):
        import torch
        from demucs.pretrained import get_model
        self._torch = torch
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = get_model('htdemucs').to(self._device)
        self._model.eval()
        self._vidx = self._model.sources.index('vocals')
        if self._device == 'cuda':
            print("[DJ vocals] using CUDA")

    def share_curve(self, samples, starts=None):
        """Vocal-stem energy share sampled across the track.

        samples: mono float32 @44100. Returns (times, shares) at window
        centers; a tail window is added so the outro is covered. Pass
        explicit window `starts` (frames) to control coverage."""
        torch = self._torch
        from demucs.apply import apply_model
        win = int(WIN_S * RATE)
        hop = int(HOP_S * RATE)
        if starts is None:
            starts = list(range(0, max(len(samples) - win, 1), hop))
            last_end = (starts[-1] + win) if starts else 0
            if len(samples) - last_end > hop * 0.5 and len(samples) > win:
                starts.append(len(samples) - win)
        times, shares = [], []
        for i0 in range(0, len(starts), BATCH):
            batch = starts[i0:i0 + BATCH]
            kept, rms = [], []
            for a in batch:
                r = float(np.sqrt(np.mean(samples[a:a + win] ** 2)))
                times.append((a + win / 2) / RATE)
                if r < QUIET_RMS:
                    shares.append(0.0)
                else:
                    kept.append(a)
                    shares.append(None)          # filled below
                    rms.append(r)
            if not kept:
                continue
            segs = np.stack([samples[a:a + win] for a in kept])
            audio = torch.from_numpy(segs).unsqueeze(1).repeat(1, 2, 1)
            std = audio.std(dim=(1, 2), keepdim=True) + 1e-8
            audio = audio / std
            with torch.no_grad():
                out = apply_model(self._model, audio, device=self._device,
                                  shifts=0, split=True, overlap=0.1,
                                  progress=False).cpu()
            it = iter(range(len(kept)))
            for k, s in enumerate(shares):
                if s is None:
                    j = next(it)
                    voc = out[j, self._vidx].mean(dim=0).numpy()
                    mix = audio[j].mean(dim=0).numpy()
                    v = float(np.mean(voc ** 2))
                    u = float(np.mean(mix ** 2)) + 1e-12
                    shares[k] = v / u
        return np.array(times), np.array(shares, dtype=np.float64)

    def share_curve_full(self, samples):
        """Contiguous (times, shares) from ONE full-track separation.

        The per-window path pays ~45 apply_model calls per track (BATCH=1
        is a hard constraint - see the header - so windows can't batch
        their way out of the per-call overhead). One call over the whole
        track does the same work through demucs' own internal chunking,
        measurably faster on GPU. Falls back to the windowed path for
        very long tracks (the full stem tensor is 4x2xN float32 in RAM -
        a 12-min cap keeps it under ~1 GB)."""
        if len(samples) / RATE > 720.0:
            return self.share_curve(samples)
        torch = self._torch
        from demucs.apply import apply_model
        audio = torch.from_numpy(samples).unsqueeze(0) \
            .repeat(2, 1).unsqueeze(0)               # (1, 2, n)
        std = float(audio.std()) + 1e-8
        with torch.no_grad():
            out = apply_model(self._model, audio / std,
                              device=self._device, shifts=0, split=True,
                              overlap=0.1, progress=False)[0].cpu()
        voc = out[self._vidx].mean(dim=0).numpy()    # normalized units
        del out
        mix_n = samples / std                        # same units as voc
        win = int(WIN_S * RATE)
        hop = int(HOP_S * RATE)
        starts = list(range(0, max(len(samples) - win, 1), hop))
        last_end = (starts[-1] + win) if starts else 0
        if len(samples) - last_end > hop * 0.5 and len(samples) > win:
            starts.append(len(samples) - win)
        times, shares = [], []
        for a in starts:
            r = float(np.sqrt(np.mean(samples[a:a + win] ** 2)))
            times.append((a + win / 2) / RATE)
            if r < QUIET_RMS:
                shares.append(0.0)
                continue
            u = float(np.mean(mix_n[a:a + win] ** 2)) + 1e-12
            v = float(np.mean(voc[a:a + win] ** 2))
            shares.append(v / u)
        return np.array(times), np.array(shares, dtype=np.float64)

    def update_track(self, db, track_id, samples):
        """Write real vocalness into this track's sections + axes.

        Returns the track's vocal duration fraction (axes['vocal']).

        Path order, cheapest first:
        1. PRE-RENDERED STEMS on disk (dj_stems.py already separated this
           track once) -> derive the curve from the vocals stem, no model
           call at all.
        2. 6-window screen; every window essentially vocal-free (most
           club tracks!) -> instrumental, no full pass.
        3. share_curve_full: ONE whole-track separation."""
        win = int(WIN_S * RATE)
        got = curve_from_stems(db, track_id, samples)
        if got is not None:
            times, shares = got
        elif len(samples) > 3 * win:
            fr = np.linspace(0.08, 0.9, 6)
            screen = [int(f * (len(samples) - win)) for f in fr]
            _, s_shares = self.share_curve(samples, starts=screen)
            if len(s_shares) and float(s_shares.max()) < 0.04:
                times = np.array([(a + win / 2) / RATE for a in screen])
                shares = np.zeros(len(screen))
            else:
                times, shares = self.share_curve_full(samples)
        else:
            times, shares = self.share_curve_full(samples)
        if not len(times):
            return None
        vocalness = np.minimum(shares * SHARE_SCALE, 1.0)
        sections = db.sections_for(track_id)
        tot_dur = 0.0
        voc_dur = 0.0
        for s in sections:
            inside = (times >= s["start_s"]) & (times < s["end_s"])
            if inside.any():
                v = float(vocalness[inside].mean())
            else:                    # short section between windows
                mid = 0.5 * (s["start_s"] + s["end_s"])
                v = float(np.interp(mid, times, vocalness))
            db.conn.execute(
                "UPDATE sections SET vocalness = ? WHERE id = ?",
                (round(v, 3), s["id"]))
            dur = max(s["end_s"] - s["start_s"], 0.1)
            tot_dur += dur
            if v > 0.5:
                voc_dur += dur
        # Duration fraction straight from the curve as well - sections can
        # average a short hook away; take the larger (more honest) figure.
        curve_frac = float((vocalness > 0.5).mean())
        frac = round(max(voc_dur / max(tot_dur, 1e-9), curve_frac), 3)
        row = db.conn.execute("SELECT axes FROM tracks WHERE id = ?",
                              (track_id,)).fetchone()
        axes = json.loads(row["axes"]) if row and row["axes"] else {}
        axes["vocal"] = frac
        axes["vocal_src"] = "demucs"
        # Fine curve for PHRASE-LEVEL mixing decisions (time the bass swap
        # into the gap between vocal lines, not just into a low-vocal
        # section). vc_hop records the resolution so a refine pass can
        # find tracks measured at the old coarse 24 s hop.
        axes["vc"] = [[round(float(t), 1), round(float(v), 2)]
                      for t, v in zip(times, vocalness)]
        axes["vc_hop"] = HOP_S
        db.conn.execute("UPDATE tracks SET axes = ? WHERE id = ?",
                        (json.dumps(axes), track_id))
        db.conn.commit()
        return frac


def curve_from_stems(db, track_id, samples):
    """(times, shares) derived from PRE-RENDERED stems on disk - the
    separation already happened once (tools/dj/dj_stems.py), so the vocal
    share is a pure numpy ratio, no torch/model/GPU involved. None when
    stems are absent or unreadable (callers fall through to the model).
    The stems' lossy encode adds a ~2000-sample codec delay vs the mix -
    irrelevant at 8-second windows."""
    try:
        from lib.dj.stems import stem_paths
        paths = stem_paths(db.music_root, track_id)
        if not paths:
            return None
        from lib.dj.features import decode_file
        voc = decode_file(paths["vocals"])
    except Exception:
        return None
    win = int(WIN_S * RATE)
    hop = int(HOP_S * RATE)
    n = min(len(voc), len(samples))
    if n < win:
        return None
    starts = list(range(0, n - win + 1, hop))
    if not starts:
        return None
    if n - (starts[-1] + win) > hop * 0.5:
        starts.append(n - win)
    times, shares = [], []
    for a in starts:
        r = float(np.sqrt(np.mean(samples[a:a + win] ** 2)))
        times.append((a + win / 2) / RATE)
        if r < QUIET_RMS:
            shares.append(0.0)
        else:
            v = float(np.mean(voc[a:a + win] ** 2))
            shares.append(v / (r * r))
    return np.array(times), np.array(shares, dtype=np.float64)


def vocal_tags(axes, user_tags=()):
    """Tag decisions from a track's axes. Only fires when a model actually
    measured the track (vocal_src set); user tags override the model."""
    if not axes or not axes.get("vocal_src"):
        return []
    user = set(user_tags or ())
    if "instrumental" in user or "no-vocals" in user:
        return ["instrumental"]
    if "vocal-heavy" in user or "vocals" in user:
        return []                    # user's own tag already says it
    v = axes.get("vocal", 0.0)
    if v > 0.25:
        return ["vocal-heavy"]
    if v >= 0.08:
        return ["vocals"]
    if v < 0.05:
        return ["instrumental"]
    return []
