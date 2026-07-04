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
