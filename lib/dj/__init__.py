"""Autonomous DJ subsystem.

Submodules (import them directly; this package init stays dependency-light
because multiprocessing scan workers import lib.dj.features on every spawn):

    lib.dj.db        - sqlite library database (tracks/sections/loops/setlists)
    lib.dj.features  - offline per-track analysis (tempo/key/structure)
    lib.dj.scan      - incremental library scanner
    lib.dj.stretch   - WSOLA time-stretcher            (Phase B)
    lib.dj.eq        - 3-band LR4 EQ                   (Phase B)
    lib.dj.deck      - playback deck                   (Phase B)
    lib.dj.submix    - the single audio-engine track   (Phase B)
    lib.dj.brain     - selection + transition planner  (Phase C)
    lib.dj.themes    - show themes / arcs              (Phase C)
    lib.dj.setlist   - preplanned sets                 (Phase E)
"""
import os


def default_music_dir():
    """The music library lives PARALLEL to the repo directory so every
    install finds it at a known relative location: <repo_parent>/music."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(os.path.dirname(repo_root), "music")


def resolve_music_dir(configured=""):
    """Config override wins when non-empty; else the parallel default."""
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return default_music_dir()


_rb_available = None


def stretch_engine_name():
    """Selected time-stretch engine (env DJ_STRETCH_ENGINE, or config.yaml
    dj.stretch_engine which Stories_OGL maps onto the env at boot):
    'rubberband' / 'rb' - keylock via Rubber Band R3 (DEFAULT 2026-07-22,
              user-picked by ear over varispeed; optional dep pylibrb,
              wheels for Windows/Linux x86_64/macOS). Constant pitch,
              warble-free, enables the ±1-semitone key rescue. Resolves
              to 'vari' when the module is missing so the BRAIN's
              planning semantics (transpose rescue, dual-deck bend split)
              always match what the decks actually run - a Pi without
              the wheel just plays turntable-style.
    'vari'  - varispeed/turntable mode: pitch rides tempo like a pitch
              fader; zero time-stretch artifacts. The brain splits each
              tempo match across BOTH decks so each song shifts half as far.
    'wsola' - keylock via WSOLA (constant pitch, slight warble under stretch)
    'pv'    - keylock via phase vocoder (less warble, weaker seams)
    Env read dynamically (not cached) so tests can flip engines
    per-process; only the pylibrb availability probe is cached."""
    name = os.environ.get("DJ_STRETCH_ENGINE", "rubberband").lower()
    if name in ("rubberband", "rb"):
        global _rb_available
        if _rb_available is None:
            try:
                import pylibrb                            # noqa: F401
                _rb_available = True
            except Exception:
                _rb_available = False
        return "rubberband" if _rb_available else "vari"
    return name
