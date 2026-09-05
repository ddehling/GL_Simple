"""Generative note-level music: a composer that decides notes a few bars
ahead, a sample-accurate scheduler, and a synth rack that renders them at
44100 Hz stereo float32 - the AudioEngine track protocol.

Plan and rationale: docs/GENERATIVE_MUSIC_PLAN.md. Keep this package
dependency-light on import (numba / fluidsynth / supriya are imported
lazily by the backends that need them), same discipline as lib/dj.
"""
RATE = 44100
