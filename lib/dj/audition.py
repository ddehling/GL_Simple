"""Offline seam renderer: decode two tracks, run the REAL transition
automation (brain.build_events over a DJSubmix) and return the audio.

One implementation shared by the planner's audition button and
dj_player --audition - the two had forked, and only the planner copy had
the dynamic pre-roll fix, so the CLI auditioned every long blend squeezed
and slammy.
"""
import numpy as np

RATE = 44100


def render_seam(db, a, b, plan, status=None, info=None):
    """Render the seam a -> b for a compiled `plan` dict. Returns a float32
    stereo ndarray. `status(text)` optional progress callback. Raises on
    decode/automation failure - callers present the error their own way.

    Pass a dict as `info` to receive the EXACT automation behind the audio
    (the planner's seam scope draws from it, so the picture can't drift
    from the render): {"events", "plan" (with no_return_at stamped),
    "t0_clock" (submix clock of render second 0), "cue_a", "blend_at",
    "swap_at"} - clocks are submix samples; subtract t0_clock for render
    time."""
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
         "track_id": a.id, "gain_db": a.gain_db, "cue_s": cue_a},
        {"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01},
        {"cmd": "start", "deck": "a"},
        {"cmd": "load", "deck": "b", "samples": sb, "grid": b.grid,
         "track_id": b.id, "gain_db": b.gain_db, "cue_s": plan["in_s"]},
    ])
    # Attach stems when rendered so the STEM styles (and the vocal duck)
    # audition truthfully - without this, stem_gains no-op against a
    # stem-less deck and the audition plays full mixes.
    from lib.dj.stems import load_stems
    for deck, t, arr in (("a", a, sa), ("b", b, sb)):
        if getattr(t, "has_stems", False):
            st = load_stems(db.music_root, t.id, expected_len=len(arr))
            if st:
                sub.post({"cmd": "attach_stems", "deck": deck, "stems": st})
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
    if info is not None:
        # sub.clock is where the collected audio begins (the priming read
        # above already advanced it) - the anchor that turns event clocks
        # into render seconds.
        info.update({"events": events, "plan": plan, "t0_clock": sub.clock,
                     "cue_a": cue_a, "blend_at": blend_at,
                     "swap_at": swap_at})
    total = pre + (swap_at - blend_at) / RATE + 25.0
    out = [np.frombuffer(gen.send(4410), dtype=np.float32).reshape(-1, 2)
           for _ in range(int(total * RATE) // 4410)]
    return np.concatenate(out, axis=0)


def render_layer(db, track, prep, at_s, bars, gain, lead_s=6.0, tail_s=4.0,
                 with_layer=True, status=None):
    """Render `track` from just before `at_s` with a LOOP LAYER riding on
    deck C - the offline twin of DJSystem._do_layer, for auditioning.

    `prep` is a lib.dj.looplayer.prepare() result. `with_layer=False`
    renders the identical span with deck C silent, so the two can be
    A/B'd: same track, same window, same everything but the bed.

    Returns a float32 stereo ndarray. The event shape here MUST track
    _do_layer's (mount now, transport on the downbeat, fade in/out over
    LAYER_FADE_BARS) or the audition stops predicting the live sound.
    """
    from lib.audio_engine import AudioEngine
    from lib.dj.features import decode_file_stereo
    from lib.dj.submix import DJSubmix
    from lib.dj.system import LAYER_FADE_BARS

    period = max(float(getattr(track, "period_s", 0.5) or 0.5), 0.15)
    start = max(track.nearest_downbeat(max(at_s - lead_s, 0.0)), 0.0)
    if status:
        status("decoding...")
    samples = decode_file_stereo(db.abs(track.path))
    engine = AudioEngine()
    sub = DJSubmix()
    engine.attach_track("dj", sub)
    sub.post_many([
        {"cmd": "load", "deck": "a", "samples": samples,
         "track_id": track.id, "grid": track.grid,
         "gain_db": getattr(track, "gain_db", 0.0), "cue_s": start},
        {"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01},
        {"cmd": "start", "deck": "a"},
    ])
    gen = engine._mixer()
    next(gen)
    gen.send(256)                       # prime, as render_seam does
    t0 = sub.clock

    span = bars * 4 * period            # deck A runs at rate 1.0 here
    fade = min(LAYER_FADE_BARS * 4 * period, span * 0.4)
    at = t0 + int(max(at_s - start, 0.0) * RATE)
    end = at + int(span * RATE)
    if with_layer:
        sub.post_many([
            {"cmd": "load", "deck": "c", "samples": prep["samples"],
             "cue_s": 0.0, "gain_db": 0.0, "grid": []},
            {"cmd": "loop", "deck": "c", "start_s": 0.0,
             "end_s": prep["loop_s"]},
            {"cmd": "rate", "deck": "c", "value": prep["rate"]},
            {"cmd": "gain", "deck": "c", "value": 0.0, "ramp_s": 0.001},
        ])
        sub.post_many([
            {"at": at, "cmd": "start", "deck": "c"},
            {"at": at, "cmd": "gain", "deck": "c", "value": gain,
             "ramp_s": fade},
            {"at": end - int(fade * RATE), "cmd": "gain", "deck": "c",
             "value": 0.0, "ramp_s": fade},
            {"at": end, "cmd": "stop", "deck": "c"},
        ])
    if status:
        status("rendering layer..." if with_layer else "rendering dry...")
    total = (at_s - start) + span + tail_s
    out = [np.frombuffer(gen.send(4410), dtype=np.float32).reshape(-1, 2)
           for _ in range(int(total * RATE) // 4410)]
    return np.concatenate(out, axis=0)
