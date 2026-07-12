"""Optional ML mood/valence pass - Music2Emo descriptors for the library.

WHY A MODEL: danceability/valence/mood were previously DERIVED heuristically
(lib/dj/character.py) because Essentia (the toolkit that produced Spotify's
descriptors) has no Windows build. Music2Emo (github.com/AMAAI-Lab/
Music2Emotion, Apache-2.0) is a PyTorch model that runs on the same
torch+CUDA stack the demucs vocal pass already uses. It reads MERT embeddings
+ a chord/key branch through a multi-task head and outputs, per track:
predicted MOOD tags (56-class Jamendo set) + VALENCE and AROUSAL (1..9).
character.py PREFERS these when present and falls back to the heuristic when
they are absent - an honest "no model, best guess" rather than a hard dep.

DEPENDENCY IS OPTIONAL AND EXTERNAL: the model code + weights live in a
clone of Music2Emotion (a few hundred MB of checkpoints - too large to vendor
into the repo). find_model_dir() locates it via $DJ_MOOD_MODEL_DIR or a
sibling `../Music2Emotion`; without it available() is False and nothing runs.

WHY A SUBPROCESS (tools/dj_mood.py, not an in-process QThread like enrich):
Music2Emo's predict() uses paths relative to its own repo root, so we must
chdir into it, and it pulls a heavy transformers/MERT stack into VRAM. Both
are isolated (and killable) in a dedicated process. MoodExtractor therefore
applies its runtime patches at CONSTRUCTION only - importing this module and
calling available() stays cheap and side-effect-free.

NORMALIZATION: raw valence/arousal are 1..9; we store (x-1)/8 -> 0..1 so they
sit on the same scale as every other character axis. load_library then
percentile-ranks valence across the library for spread, exactly like the
heuristic path.
"""
import os

_SIBLING = "Music2Emotion"


def find_model_dir():
    """Locate the Music2Emotion clone. $DJ_MOOD_MODEL_DIR wins; otherwise a
    sibling of the GL_Simple repo root. Returns an abspath or None."""
    env = os.environ.get("DJ_MOOD_MODEL_DIR")
    cands = []
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.dirname(os.path.dirname(  # lib/dj/ -> repo
        os.path.abspath(__file__))))
    cands.append(os.path.join(os.path.dirname(here), _SIBLING))  # ../sibling
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "music2emo.py")) and \
                os.path.isdir(os.path.join(c, "saved_models")):
            return os.path.abspath(c)
    return None


def available():
    """True if the model clone is present and torch imports. Cheap: does NOT
    import music2emo or load any weights, and applies no global patches."""
    if find_model_dir() is None:
        return False
    try:
        import torch                              # noqa: F401
        return True
    except Exception:
        return False


def _norm9(x):
    """1..9 -> 0..1, clamped."""
    try:
        return max(0.0, min(1.0, (float(x) - 1.0) / 8.0))
    except (TypeError, ValueError):
        return None


class MoodExtractor:
    """Loads Music2Emo once, scores tracks. Construct only in a dedicated
    process/thread: __init__ chdir's into the model repo and monkeypatches
    torch/torchaudio globally (safe there, not in the GUI process)."""

    def __init__(self, model_dir=None):
        import sys
        import types
        self.model_dir = os.path.abspath(model_dir or find_model_dir())
        if not self.model_dir or not os.path.isfile(
                os.path.join(self.model_dir, "music2emo.py")):
            raise RuntimeError("Music2Emotion model dir not found")

        # music2emo.predict() resolves ./inference/... relative to cwd.
        os.chdir(self.model_dir)
        if self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)

        # gradio is imported at module top for the demo UI only; stub it so we
        # don't pull that heavy dep for headless inference.
        sys.modules.setdefault("gradio", types.ModuleType("gradio"))

        import numpy as np
        import torch
        import torchaudio
        import librosa

        # torchaudio 2.11 routes load() through torchcodec (an ffmpeg backend
        # we don't ship); librosa already decodes mp3/flac here.
        def _load(path, *a, **k):
            y, sr = librosa.load(path, sr=None, mono=False)
            if y.ndim == 1:
                y = y[None, :]
            return torch.from_numpy(np.ascontiguousarray(y)).float(), sr
        torchaudio.load = _load

        # torch>=2.6 defaults weights_only=True; the repo's own checkpoints
        # (trusted, shipped in the clone) pickle numpy scalars.
        _orig = torch.load

        def _tload(*a, **k):
            k.setdefault("weights_only", False)
            return _orig(*a, **k)
        torch.load = _tload

        from music2emo import Music2emo
        self._m = Music2emo()           # loads MERT + BTC + Jamendo head

    def analyze(self, abs_path):
        """Return {valence, arousal, moods, valence9, arousal9} with
        valence/arousal normalized 0..1, or None on any failure."""
        p = os.path.abspath(abs_path).replace("\\", "/")
        try:
            out = self._m.predict(p)
        except Exception:
            return None
        v9, a9 = out.get("valence"), out.get("arousal")
        return {
            "valence": _norm9(v9),
            "arousal": _norm9(a9),
            "moods": list(out.get("predicted_moods") or []),
            "valence9": round(float(v9), 2) if v9 is not None else None,
            "arousal9": round(float(a9), 2) if a9 is not None else None,
        }
