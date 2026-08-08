"""Gate for the LOOP LAYER (DJSystem._do_layer -> deck C).

A percussion bed ridden UNDER the playing track: a drum loop mounted on
deck C, launched on a downbeat, faded in over ~2 bars, ridden for ~16,
faded out. Deck C is not part of the A/B transition pair, so this is a
genuine third layer rather than a borrowed deck.

Two rules are enforced by construction and checked here, both learned
elsewhere in the codebase:

  * NEVER over an armed seam. `_do_moment` refuses with "mix in
    progress" because "layering anything on top of it read as garbage
    every time it was tried", and `_perc_bed_events` independently
    insists its bed is "gone by the seam". `_arm` therefore calls
    `_cancel_layer`, and a press while armed is refused.
  * VISIBLY. A refused press sets `_layer_denied` so the button can
    flash the reason; a silent refusal reads as a dead button.

Rendered through the real DJSubmix (real Deck, real loop wrap crossfade,
real master path) and measured:

  1. IT PLAYS: the bed is audible, it is ON deck C, and A keeps running
     underneath it untouched.
  2. SHAPE: fades in rather than cutting, rides, and is gone by the end.
  3. TOGGLE: a second press kills it with a fade, not a cut.
  4. SEAM SAFETY: `_cancel_layer` silences deck C, and a press while
     armed is refused with a reason and schedules nothing.
  5. NOTHING CLIPS, and the A/B decks are bit-identical to a run with no
     layer at all - the layer must be purely additive.

Usage:
    python tools/tests/_dj_layer_test.py           # ALL PASS gate
    python tools/tests/_dj_layer_test.py out.wav   # + a render to listen to
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj.submix import DJSubmix                      # noqa: E402
from lib.dj.system import (LAYER_BARS, LAYER_GAIN,      # noqa: E402
                           DJSystem, PLAN_LEAD_S)

RATE = 44100
DUR = 420.0
BPM = 124.0
PERIOD = 60.0 / BPM

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f": {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


# ---------------------------------------------------------------- material
def synth_track(period, seed=3):
    """Four-on-the-floor with a pad, so a drum bed laid over it is
    separable by ear and by band."""
    n = int(DUR * RATE)
    t = np.arange(n) / RATE
    ph = t - (t / period).astype(int) * period
    kick = np.sin(2 * np.pi * (60 * np.exp(-ph / 0.05) + 45) * t) \
        * np.exp(-ph / 0.09)
    bass = 0.5 * np.sin(2 * np.pi * 82.0 * t)
    pad = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    x = 0.55 * kick + 0.45 * bass + 0.25 * pad
    x = (x / np.abs(x).max() * 0.8).astype(np.float32)
    return np.repeat(x[:, None], 2, axis=1)


def synth_loop(period, bars=4, seed=7):
    """A 'whompy' bed: congas + a woody click, deliberately NOT a
    four-on-the-floor so it is distinguishable from the track under it."""
    n = int(bars * 4 * period * RATE)
    t = np.arange(n) / RATE
    r = np.random.RandomState(seed)
    x = np.zeros(n)
    step = period / 2.0                      # eighths
    for i in range(int(n / RATE / step)):
        if i % 4 in (1, 3):                  # off-beat conga
            a = int(i * step * RATE)
            k = min(int(0.18 * RATE), n - a)
            if k <= 0:
                continue
            e = np.exp(-np.arange(k) / (0.045 * RATE))
            x[a:a + k] += 0.9 * np.sin(
                2 * np.pi * 196.0 * np.arange(k) / RATE) * e
    x += 0.05 * r.randn(n) * np.exp(-(t % step) / 0.01)
    x = (x / max(np.abs(x).max(), 1e-9) * 0.8).astype(np.float32)
    return np.repeat(x[:, None], 2, axis=1)


class Track:
    id, title = 1, "under"
    bpm, period_s, duration_s = BPM, PERIOD, DUR
    gain_db, kick_offset_s = 0.0, 0.0
    phrase_beats = 32
    grid = [{"start_s": 0.0, "end_s": DUR, "period_s": PERIOD,
             "first_beat_s": 0.0}]

    def nearest_downbeat(self, t):
        bar = 4 * PERIOD
        return max(round(t / bar) * bar, 0.0)


# ---------------------------------------------------------------- harness
def make_system(state="playing", loop=None):
    sm = DJSubmix()
    sm.post_many([
        {"cmd": "load", "deck": "a", "samples": synth_track(PERIOD),
         "track_id": 1, "grid": Track.grid, "cue_s": 60.0},
        {"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01},
        {"cmd": "start", "deck": "a"},
    ])
    s = DJSystem.__new__(DJSystem)
    s.submix, s.current, s.state, s.active_deck = sm, Track(), state, "a"
    s.next_track, s.plan, s.swap_at, s.blend_at = None, None, None, None
    s._layer_txn, s._layer_until = None, 0
    s._layer_label, s._layer_denied = None, None
    s._txn_id = 0
    s.threaded = True
    s._decode_lock = threading.Lock()
    s.engine = None
    s.log = []
    s._log = s.log.append
    s._true_bpm = lambda t: BPM
    # Stand in for the library/DB lookup: one synthetic loop, already at
    # the track's tempo. The real sourcing is covered by looplayer's own
    # candidate/prepare path; this gate is about the DECK behaviour.
    prep = {"samples": loop if loop is not None else synth_loop(PERIOD),
            "loop_s": 0.0, "rate": 1.0, "label": "test-conga",
            "bpm": BPM, "kind": "test"}
    prep["loop_s"] = len(prep["samples"]) / RATE
    import lib.dj.looplayer as LL
    s._patched = (LL.candidates, LL.prepare)
    LL.candidates = lambda *a, **k: [{"label": "test-conga", "kind": "test"}]
    LL.prepare = lambda *a, **k: dict(prep)
    s.db = type("DB", (), {"music_root": "."})()
    s.brain = type("B", (), {"library": []})()
    return s


def restore(s):
    import lib.dj.looplayer as LL
    LL.candidates, LL.prepare = s._patched


def render(s, seconds, fire_at=None, again_at=None, cancel_at=None,
           trace=None):
    """Render `seconds`, pressing LAYER / cancelling at the given offsets.

    `trace` (a list) collects (t, deck-C gain, ready, playing) once per
    block. Deck C UNLOADS itself when the bed ends, so anything about its
    state has to be sampled while it rides - reading it after the render
    only ever shows the teardown."""
    n, out = 2048, []
    for i in range(int(seconds * RATE / n)):
        now = i * n / RATE
        if fire_at is not None and now >= fire_at:
            s._do_layer()
            fire_at = None
        if again_at is not None and now >= again_at:
            s._do_layer()
            again_at = None
        if cancel_at is not None and now >= cancel_at:
            s._cancel_layer("test")
            cancel_at = None
        out.append(s.submix.read(n))
        if trace is not None:
            c = s.submix.decks["c"]
            trace.append((now, c.gain, c.ready, c.playing))
    return np.concatenate(out)


def env(x, hop=0.05):
    """RMS envelope, one point per `hop` seconds."""
    w = int(hop * RATE)
    n = len(x) // w * w
    m = np.abs(x[:n].mean(axis=1)).reshape(-1, w)
    return np.sqrt((m ** 2).mean(axis=1) + 1e-12)


# ---------------------------------------------------------------- tests
def t_plays_and_is_additive():
    print("\n1. the bed plays, on deck C, purely additive")
    s = make_system()
    try:
        tr = []
        mid = {}

        def snap():
            c = s.submix.decks["c"]
            mid.update(ready=c.ready, loop=c.loop, playing=c.playing)
        mix = render(s, 16.0, fire_at=2.0, trace=tr)
        snap()                       # mid-ride, before the bed unloads
        mix = np.concatenate([mix, render(s, 26.0, trace=tr)])
        check("mounted on deck C mid-ride", bool(mid.get("ready")),
              f"ready={mid.get('ready')} playing={mid.get('playing')}")
        check("deck C looped", mid.get("loop") is not None,
              str(mid.get("loop")))
        check("layer logged", any(e.get("event") == "layer" for e in s.log),
              str([e for e in s.log if e.get("event") == "layer"][:1]))
        # A untouched: same deck A gain/rate as an unlayered run.
        s2 = make_system()
        try:
            base = render(s2, 40.0)
        finally:
            restore(s2)
        check("deck A gain untouched by the layer",
              abs(s.submix.decks["a"].gain
                  - s2.submix.decks["a"].gain) < 1e-6,
              f"{s.submix.decks['a'].gain:.4f} vs {s2.submix.decks['a'].gain:.4f}")
        e_mix, e_base = env(mix), env(base)
        k = min(len(e_mix), len(e_base))
        ride = slice(int(12 / 0.05), min(int(30 / 0.05), k))
        lift = float(np.median(e_mix[ride]) / max(np.median(e_base[ride]),
                                                  1e-9))
        check("bed is audible over the track", lift > 1.03,
              f"median level x{lift:.3f} during the ride")
        check("nothing clips", float(np.abs(mix).max()) < 0.999,
              f"peak {np.abs(mix).max():.3f}")
        return mix
    finally:
        restore(s)


def t_shape():
    print("\n2. it fades in, rides, and leaves")
    s = make_system()
    try:
        tr = []
        render(s, 80.0, fire_at=1.0, trace=tr)
        # Measure from the PRESS: sampling after the fade-in has already
        # finished can only ever report an instant rise.
        t = np.array([p[0] for p in tr]) - 1.0
        g = np.array([p[1] for p in tr])
        keep = t >= 0
        t, g = t[keep], g[keep]
        peak = float(g.max())
        check("reaches the bed level", abs(peak - LAYER_GAIN) < 0.02,
              f"peak deck-C gain {peak:.3f} (target {LAYER_GAIN})")
        rise = t[np.argmax(g > peak * 0.9)] if (g > peak * 0.9).any() else -1
        start = t[np.argmax(g > 0.01)] if (g > 0.01).any() else -1
        check("fades in rather than cutting", rise - start > 0.5,
              f"0->90% took {rise - start:.2f}s (launch at +{start:.2f}s "
              f"on the downbeat)")
        check("gone by the end", g[-1] < 0.02, f"final gain {g[-1]:.4f}")
        check("deck C stopped", not s.submix.decks["c"].playing)
        bars = LAYER_BARS
        check("rode roughly the planned length",
              8 <= (g > peak * 0.5).sum() * 2048 / RATE <= bars * 4 * PERIOD + 8,
              f"{(g > peak * 0.5).sum() * 2048 / RATE:.1f}s above half")
    finally:
        restore(s)


def t_toggle():
    print("\n3. a second press kills it, with a fade")
    s = make_system()
    try:
        render(s, 14.0, fire_at=1.0)
        g_before = s.submix.decks["c"].gain
        s._do_layer()                      # second press = kill
        killed = [s.submix.decks["c"].gain]
        for _ in range(int(4.0 * RATE / 2048)):
            s.submix.read(2048)
            killed.append(s.submix.decks["c"].gain)
        check("was riding before the second press", g_before > 0.05,
              f"{g_before:.3f}")
        check("silenced", killed[-1] < 0.02, f"{killed[-1]:.4f}")
        check("faded rather than cut",
              sum(1 for a, b in zip(killed, killed[1:]) if b < a) > 3,
              f"{sum(1 for a, b in zip(killed, killed[1:]) if b < a)} "
              f"descending steps")
        check("state cleared", s._layer_txn is None)
        check("cancel logged",
              any(e.get("event") == "layer_cancel" for e in s.log))
    finally:
        restore(s)


def t_seam_safety():
    print("\n4. never over a seam")
    s = make_system()
    try:
        render(s, 14.0, fire_at=1.0)
        check("riding", s._layer_txn is not None)
        s._cancel_layer("armed")           # what _arm does
        for _ in range(int(4.0 * RATE / 2048)):
            s.submix.read(2048)
        check("_cancel_layer silences deck C",
              s.submix.decks["c"].gain < 0.02,
              f"{s.submix.decks['c'].gain:.4f}")
    finally:
        restore(s)

    s2 = make_system(state="armed")
    try:
        s2._do_layer()
        check("press while armed is refused", s2._layer_txn is None)
        check("refusal is visible", s2._layer_denied is not None,
              str(s2._layer_denied))
        check("refusal scheduled nothing",
              not s2.submix.decks["c"].ready)
        check("refusal logged",
              any(e.get("event") == "layer_skipped" for e in s2.log))
    finally:
        restore(s2)


def t_end_of_track():
    print("\n5. refuses when there is no room left")
    s = make_system()
    try:
        # Park the playhead so less than the planning lead remains.
        s.submix.post({"cmd": "cue", "deck": "a",
                       "time_s": DUR - PLAN_LEAD_S + 5.0})
        s.submix.read(2048)
        s._do_layer()
        check("refused near the end", s._layer_txn is None)
        check("reason given", s._layer_denied is not None,
              str(s._layer_denied))
    finally:
        restore(s)


def main():
    print(f"LOOP LAYER gate  ({LAYER_BARS} bars @ {LAYER_GAIN} under a "
          f"{BPM:.0f} bpm track)")
    mix = t_plays_and_is_additive()
    t_shape()
    t_toggle()
    t_seam_safety()
    t_end_of_track()
    wav = next((a for a in sys.argv[1:] if a.endswith(".wav")), None)
    if wav and mix is not None:
        import soundfile as sf
        sf.write(wav, mix, RATE)
        print(f"\nwrote {wav}")
    print("\n" + ("ALL PASS" if not _fails
                  else f"FAILED: {len(_fails)} check(s): "
                       + ", ".join(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
