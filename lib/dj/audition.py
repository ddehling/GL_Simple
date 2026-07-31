"""Offline seam renderer: decode two tracks, run the REAL transition
automation (brain.build_events over a DJSubmix) and return the audio.

One implementation shared by the planner's audition button and
dj_player --audition - the two had forked, and only the planner copy had
the dynamic pre-roll fix, so the CLI auditioned every long blend squeezed
and slammy.
"""
import numpy as np

RATE = 44100


def render_seam(db, a, b, plan, status=None):
    """Render the seam a -> b for a compiled `plan` dict. Returns a float32
    stereo ndarray. `status(text)` optional progress callback. Raises on
    decode/automation failure - callers present the error their own way."""
    from lib.audio_engine import AudioEngine
    from lib.dj.brain import Brain
    from lib.dj.features import decode_file_stereo
    from lib.dj.submix import DJSubmix
    from lib.dj.themes import get_theme

    plan = dict(plan)
    if status:
        status("decoding...")
    sa = decode_file_stereo(db.abs(a.path))
    sb = decode_file_stereo(db.abs(b.path))
    engine = AudioEngine()
    sub = DJSubmix()
    engine.attach_track("dj", sub)
    # Pre-roll must cover the WHOLE planned blend: at 12s fixed, a 32-64
    # beat blend's start clamped to "now" and the geometry compressed -
    # every long seam auditioned squeezed and slammy (the live night
    # never does this; it arms far ahead).
    pre = max(12.0, plan.get("beats", 32) * 60.0
              / max(a.bpm, 60.0) + 6.0)
    cue_a = a.nearest_downbeat(max(0.0, plan["out_s"] - pre))
    sub.post_many([
        {"cmd": "load", "deck": "a", "samples": sa, "grid": a.grid,
         "gain_db": a.gain_db, "cue_s": cue_a},
        {"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01},
        {"cmd": "start", "deck": "a"},
        {"cmd": "load", "deck": "b", "samples": sb, "grid": b.grid,
         "gain_db": b.gain_db, "cue_s": plan["in_s"]},
    ])
    # Prime telemetry with one tiny silent read so build_events sees deck A.
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    brain = Brain([], get_theme("groove"))
    events, swap_at, blend_at = brain.build_events(
        plan, sub.telemetry, "a", "b", a, b)
    sub.post_many(events)
    if status:
        status("rendering seam...")
    total = pre + (swap_at - blend_at) / RATE + 25.0
    out = [np.frombuffer(gen.send(4410), dtype=np.float32).reshape(-1, 2)
           for _ in range(int(total * RATE) // 4410)]
    return np.concatenate(out, axis=0)
