"""Optional ML structure segmentation via allin1 (All-In-One Music
Structure Analyzer): intro/verse/chorus/bridge/inst/solo/break/outro
segment boundaries, stored as DB v12 tracks.structure.

WHY: the internal SSM sectionizer finds BOUNDARIES and coarse kinds
(intro/outro/breakdown/build/groove) from novelty, but it cannot name a
chorus or tell a bridge from a verse - so seam planning could still enter
a track in the middle of its hook. allin1 was trained on labeled EDM/pop
structure and yields functional labels the brain folds into mix-in/out
fit (never entering mid-chorus by accident, exiting on real outros).

DEPENDENCIES ARE OPTIONAL: torch + allin1 live in
requirements-dj-structure.txt, NEVER in requirements.txt. Without them
available() is False and ml_segments stays empty - selection falls back
to the internal sections exactly as before.

NOTE: allin1 runs its own internal demucs demix + madmom features; a few
seconds per track on GPU, much slower on CPU. Its NATTEN dependency has
no official Windows wheels for every torch build - see the requirements
file header for install notes.
"""

LABELS = ("start", "intro", "verse", "chorus", "bridge", "inst",
          "solo", "break", "outro", "end")


def available():
    try:
        import torch                      # noqa: F401
        import allin1                     # noqa: F401
        return True
    except Exception:
        return False


class StructureAnalyzer:
    """Loads allin1 lazily; analyze() one file at a time so a crashed
    track costs one track. Construct only if available()."""

    def __init__(self):
        import torch
        import allin1
        self._allin1 = allin1
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            print("[DJ structure] using CUDA")

    def analyze(self, abs_path):
        """Returns the tracks.structure blob:
        {segments: [[start_s, end_s, label], ...], source, bpm}."""
        r = self._allin1.analyze(abs_path, device=self._device,
                                 keep_byproducts=False)
        segments = [[round(float(s.start), 2), round(float(s.end), 2),
                     str(s.label)]
                    for s in (r.segments or [])]
        return {
            "segments": segments,
            "source": "allin1",
            "bpm": float(r.bpm) if getattr(r, "bpm", None) else None,
        }
