"""Gate for THE operator MOMENT (DJSystem._do_moment -> nextdrop).

Four flavors shipped on 2026-07-29; the operator's verdict the next day
was final: 'only next is good', 'drop is awful', 'stall is worthless',
'spinback is basically another next'. Every same-track gesture was cut.
What remains is one button that either double-drops the set forward into
the NEXT track's real drop, or refuses with the reason on the button:

  build on the LIVE track (HP sweep + trim push + snare roll +
  LOOP-ROLL: the track beat-repeats 1 -> 1/2 -> 1/4 beat) -> 1-beat
  hole (the dying deck cuts, the incoming deck pre-rolls silently) ->
  landing: THE NEXT TRACK'S DROP, cold, full gain, impact.

It arms as a real transition (swap_at = the landing, _finish_swap flips
the decks) and recalls via _do_abort. Rendered through the real DJSubmix
(real Deck, real SweepFilter, real master limiter) and measured:

  1. FULL GESTURE: bass swept out of the build, loop-roll stutter
     present, real hole, handover ON the landing, deck B riding the
     incoming drop, slam over the hole, nothing clips.
  2. RECALL: a second press mid-build aborts - original deck restored
     and still playing, incoming deck killed, countdown cleared.
  3. REFUSALS, each visible (_moment_denied) with nothing scheduled:
     no next queued / next has no drop / deck faded out / seam armed.
  4. SCHEDULE across 90-174 bpm and off-unity rates: the build spans
     the whole 8-24-beat wait (no dead air after the press).
  5. LEGACY FLAVORS: old panels may still send drop/spinback/stall -
     every one of them is a nextdrop now.

Usage:
    python tools/tests/_dj_moment_test.py           # ALL PASS gate
    python tools/tests/_dj_moment_test.py out.wav   # + render to listen to
"""
import os
import sys
import threading

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj.submix import DJSubmix, RATE          # noqa: E402
from lib.dj.system import DJSystem                # noqa: E402
import lib.dj.fx as fxmod                         # noqa: E402

BPM = 120.0
# Long enough that the incoming drop clears the runway rule
# (PLAN_LEAD_S + 25 + ride ~ 115s of track needed after the landing).
DUR = 260.0
DROP_S = 120.0        # the synthetic track's one real drop (sections below)
CLEAN = {"gain": 1.0, "eq": [1.0, 1.0, 1.0], "filter": "off", "echo": False,
         "rate": 1.0}
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def synth_track(period):
    """Bass-dominant four-on-the-floor: the sweep's job is to take the low
    end away, so the material has to have a low end to take."""
    n = int(DUR * RATE)
    t = np.arange(n) / RATE
    ph = t - (t / period).astype(int) * period      # time since last beat
    kick = np.sin(2 * np.pi * (60 * np.exp(-ph / 0.05) + 45) * t) \
        * np.exp(-ph / 0.09)
    bass = 0.5 * np.sin(2 * np.pi * 82.0 * t) * (
        0.6 + 0.4 * np.sin(2 * np.pi * t / (8 * period)))
    hat = (np.random.RandomState(3).randn(n) * 0.12
           * np.exp(-((ph - period / 2) % period) / 0.012))
    pad = 0.25 * np.sin(2 * np.pi * 440.0 * t) \
        + 0.2 * np.sin(2 * np.pi * 660.0 * t)
    x = 0.55 * kick + 0.45 * bass + hat + 0.25 * pad
    # ANTHEM section from DROP_S on: what the landing arrives ON has to
    # be audibly hotter than what died, or 'landing on the real drop' is
    # unmeasurable here and inaudible in the demo renders.
    hot = (t >= DROP_S).astype(np.float64)
    saw = sum(np.sin(2 * np.pi * 164.8 * k * t + k) / k for k in range(1, 6))
    stab = 0.5 + 0.5 * np.sin(2 * np.pi * t / (2 * period) - np.pi / 2)
    eighth = np.exp(-(t % (period / 2)) / 0.05)
    x = x + hot * (0.4 * saw * stab + 0.35 * np.sin(2 * np.pi * 82.0 * t)
                   * eighth)
    x = (x / np.abs(x).max() * 0.85).astype(np.float32)
    return np.stack([x, x], axis=1)


class Track:
    """The slice of TrackInfo the gesture actually reads."""
    id, title = 1, "synthetic"
    period_s, phrase_beats, duration_s = 60.0 / BPM, 32, DUR
    sections = [
        {"start_s": 0.0, "end_s": DROP_S - 10.0, "energy": 0.35,
         "kind": "groove"},
        {"start_s": DROP_S - 10.0, "end_s": DROP_S, "energy": 0.30,
         "kind": "break"},
        {"start_s": DROP_S, "end_s": DUR, "energy": 0.92, "kind": "groove"},
    ]

    def nearest_phrase(self, t):
        span = self.phrase_beats * self.period_s
        return round(t / span) * span

    def nearest_downbeat(self, t):
        return round(t / self.period_s) * self.period_s

    def section_at(self, t):
        for s in self.sections:
            if s["start_s"] <= t < s["end_s"]:
                return s
        return self.sections[-1]


class Track2(Track):
    """The queued next track: same grid/sections shape, its own id.
    Its analyzed drop is at DROP_S, where synth_track's anthem is."""
    id, title = 2, "incoming"
    grid = [{"start_s": 0.0, "end_s": DUR, "period_s": Track.period_s,
             "first_beat_s": 0.0}]
    gain_db, kick_offset_s = 0.0, 0.0


def make_system(state="playing"):
    sm = DJSubmix()
    sm.post({"cmd": "load", "deck": "a", "samples": synth_track(
        Track.period_s), "track_id": 1,
        "grid": [{"start_s": 0.0, "end_s": DUR, "period_s": Track.period_s,
                  "first_beat_s": 0.0}], "cue_s": 40.0})
    sm.post({"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01})
    sm.post({"cmd": "start", "deck": "a"})
    s = DJSystem.__new__(DJSystem)          # no DB / library / thread needed
    s.submix, s.current, s.state, s.active_deck = sm, Track(), state, "a"
    s._moment_clock, s._moment_txn = 0, None
    s._moment_gain, s._moment_stamped = 1.0, 0
    s._moment_flavor, s._moment_hole = None, (0, 0)
    s._moment_rate, s._moment_denied = 1.0, None
    s.next_track, s.plan, s.swap_at, s.blend_at = None, None, None, None
    s.threaded = True
    s._decode_lock = threading.Lock()
    s._decoded = {}
    s._predecode = lambda t: None
    s._txn_id, s._recovery_txn = 0, None
    s._started_clock, s._exit_played, s._seam_metrics = 0, 0.0, None
    s.engine = None
    s.log = []
    s._log = s.log.append
    return s


def nextdrop_system(state="playing"):
    """make_system + a decoded queued next and a minimal handover flip
    (the real _finish_swap needs the DB/brain stack; the gesture's
    contract is that it ARMS a transition whose swap lands on the drop -
    full bookkeeping has its own coverage via the transition machinery)."""
    s = make_system(state)
    s.next_track = Track2()
    s._decoded = {2: synth_track(Track.period_s)}

    def fin():
        s.current, s.next_track = s.next_track, None
        s.active_deck = "b"
        s.state, s.plan = "playing", None
        s.swap_at = s.blend_at = None
    s._finish_swap = fin
    return s


def render(s, seconds, fire_at=None, again_at=None, flavor="nextdrop"):
    """Render `seconds`, pressing MOMENT at the given offsets and doing
    the tick loop's armed-state duty (fire _finish_swap on swap_at)."""
    n, out = 2048, []
    for i in range(int(seconds * RATE / n)):
        now = i * n / RATE
        if fire_at is not None and now >= fire_at:
            s._do_moment(flavor)
            fire_at = None
        if again_at is not None and now >= again_at:
            s._do_moment(flavor)
            again_at = None
        if s.state == "armed" and s.swap_at is not None \
                and s.submix.clock >= s.swap_at:
            s._finish_swap()
        out.append(s.submix.read(n))
    return np.concatenate(out)


def deck_state(s, deck="a"):
    d = s.submix.decks[deck]
    return {"gain": round(d.gain, 3),
            "eq": [round(float(g), 3) for g in d.eq.gains],
            "filter": d.filter.mode,
            "echo": bool(d.echo.active),
            "rate": round(float(d.rate), 3)}


def deck_time(s, deck="a"):
    return float(((s.submix.telemetry or {}).get("decks", {})
                  .get(deck) or {}).get("time_s") or -1.0)


def peak(a, t0, t1):
    return float(np.abs(a[int(t0 * RATE):int(t1 * RATE)]).max())


def rms(a, t0, t1):
    w = a[int(t0 * RATE):int(t1 * RATE)]
    return float(np.sqrt(np.mean(w ** 2)))


def periodicity(a, t0, t1, lag_s):
    """Normalized autocorrelation at one lag - the loop-roll makes the
    audio periodic at the loop length, which nothing else does."""
    w = a[int(t0 * RATE):int(t1 * RATE), 0].astype(np.float64)
    k = int(lag_s * RATE)
    x, y = w[:-k], w[k:]
    d = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


def low_rms(a, t0, t1, hz=200.0):
    """RMS below `hz` - measures the sweep itself, independent of how
    much mid/high content the material happens to carry."""
    w = a[int(t0 * RATE):int(t1 * RATE), 0]
    sos = butter(4, hz, "low", fs=RATE, output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, w) ** 2)))


def denied_why(s):
    return s._moment_denied[1] if s._moment_denied else None


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n1. the full gesture")
    s = make_system()          # music-only twin below measures the build
    s = nextdrop_system()
    audio = render(s, 30.0, fire_at=4.5)
    got = [e for e in s.log if e.get("event") == "moment"]
    hit = s._moment_clock / RATE
    nrm = rms(audio, 1.0, 4.0)
    check("scheduled onto the next track's drop",
          bool(got) and got[0]["flavor"] == "nextdrop"
          and got[0]["payoff"] == "next" and got[0]["to_s"] == DROP_S,
          f"{got}")
    check("the build spans the whole wait (no dead air)",
          bool(got) and abs(got[0]["build_s"] - got[0]["in_s"]) <= 0.06,
          f"build {got[0]['build_s']}s vs wait {got[0]['in_s']}s"
          if got else "no moment")
    # Music-only twin (one-shots silenced): the build must be in the
    # TRACK - bass swept out, loop-roll stutter, real hole.
    real_pk = fxmod.at_peak
    fxmod.at_peak = lambda buf, *a, **k: np.zeros_like(buf)
    try:
        ms = nextdrop_system()
        music = render(ms, 30.0, fire_at=4.5)
    finally:
        fxmod.at_peak = real_pk
    mh = ms._moment_clock / RATE
    check("the build sweeps the bass out of the dying deck",
          low_rms(music, mh - 2.5, mh - 0.6)
          < 0.3 * low_rms(music, 1.0, 4.0),
          f"<200 Hz RMS {low_rms(music, mh - 2.5, mh - 0.6):.4f} vs "
          f"{low_rms(music, 1.0, 4.0):.4f} normal")
    p_roll = periodicity(music, mh - 0.95, mh - 0.55, Track.period_s / 4)
    p_ctl = periodicity(music, mh - 7.0, mh - 6.6, Track.period_s / 4)
    check("the loop-roll stutters the track itself",
          p_roll > 0.5 and p_roll > p_ctl + 0.2,
          f"autocorr {p_roll:.2f} in the roll vs {p_ctl:.2f} unlooped")
    check("the hole is a real hole (music-only)",
          rms(music, mh - 0.45, mh - 0.05) < 0.2 * rms(music, 1.0, 4.0),
          f"RMS {rms(music, mh - 0.45, mh - 0.05):.3f} vs "
          f"{rms(music, 1.0, 4.0):.3f} normal")
    check("the handover happened ON the landing",
          s.state == "playing" and s.active_deck == "b"
          and s.current is not None and s.current.id == 2,
          f"state={s.state} deck={s.active_deck}")
    check("deck B is riding the incoming drop",
          s.submix.decks["b"].playing
          and abs(deck_time(s, "b") - (DROP_S + (30.0 - hit))) < 0.6,
          f"playhead {deck_time(s, 'b'):.1f}s, expected "
          f"{DROP_S + (30.0 - hit):.1f}s")
    check("the old deck is stopped",
          not s.submix.decks["a"].playing, "deck a still playing")
    check("the landing slams over the hole",
          rms(audio, hit, hit + 0.8) > 1.6 * rms(audio, hit - 0.45,
                                                 hit - 0.05),
          f"RMS {rms(audio, hit, hit + 0.8):.3f} vs "
          f"{rms(audio, hit - 0.45, hit - 0.05):.3f} in the hole")
    check("nothing clips", float(np.abs(audio).max()) <= 0.9851,
          f"peak {float(np.abs(audio).max()):.3f} (ceiling 0.985)")

    print("\n2. second press recalls it (abort path)")
    s2 = nextdrop_system()
    a2 = render(s2, 20.0, fire_at=4.0, again_at=5.5)
    check("recall went through abort",
          any(e.get("event") == "abort" and e.get("via") == "moment_recall"
              for e in s2.log), f"{[e.get('event') for e in s2.log]}")
    check("still on the original deck and track",
          s2.state == "playing" and s2.active_deck == "a"
          and s2.current.id == 1 and s2.next_track is not None,
          f"state={s2.state} deck={s2.active_deck}")
    check("the deck is restored and playing",
          deck_state(s2) == CLEAN and s2.submix.decks["a"].playing,
          f"{deck_state(s2)}")
    check("the incoming deck never took over",
          not s2.submix.decks["b"].playing, "deck b playing")
    check("the countdown clears", s2._moment_clock == 0,
          f"moment_clock {s2._moment_clock}")
    check("the audio is unbroken",
          peak(a2, 16, 20) >= 0.85 * peak(a2, 1, 3),
          f"tail peak {peak(a2, 16, 20):.3f} vs {peak(a2, 1, 3):.3f}")

    print("\n3. refusals are visible and schedule nothing")
    s3 = make_system()                       # no next queued at all
    render(s3, 4.0, fire_at=2.0)
    check("no next queued -> refused",
          denied_why(s3) == "no next queued" and s3._moment_clock == 0,
          f"denied={s3._moment_denied}, log={s3.log}")
    s4 = nextdrop_system()
    s4.next_track.sections = [{"start_s": 0.0, "end_s": DUR, "energy": 0.5,
                               "kind": "groove"}]
    render(s4, 4.0, fire_at=2.0)
    check("next has no drop -> refused",
          denied_why(s4) == "next has no drop" and s4._moment_clock == 0,
          f"denied={s4._moment_denied}")
    s5 = nextdrop_system()
    s5.submix.post({"cmd": "gain", "deck": "a", "value": 0.0, "ramp_s": 0.01})
    render(s5, 4.0, fire_at=2.0)
    check("deck faded out -> refused",
          denied_why(s5) == "deck not up" and s5._moment_clock == 0,
          f"denied={s5._moment_denied}")
    s6 = nextdrop_system("armed")
    render(s6, 4.0, fire_at=2.0)
    check("seam armed -> refused",
          denied_why(s6) == "mix in progress" and s6._moment_clock == 0,
          f"denied={s6._moment_denied}")

    print("\n4. schedule across tempos")
    for bpm, rate in [(90.0, 1.0), (120.0, 1.0), (128.0, 1.03), (174.0, 1.0)]:
        Track.period_s = 60.0 / bpm
        Track2.grid = [{"start_s": 0.0, "end_s": DUR,
                        "period_s": Track.period_s, "first_beat_s": 0.0}]
        st = nextdrop_system()
        st.submix.post({"cmd": "rate", "deck": "a", "value": rate})
        render(st, 6.0, fire_at=3.0)
        got = [e for e in st.log if e.get("event") == "moment"]
        beat = Track.period_s / rate
        ok = bool(got) and abs(got[0]["build_s"] - got[0]["in_s"]) <= 0.06 \
            and 8 * beat - 0.15 <= got[0]["in_s"] <= 24 * beat + 0.5
        check(f"{bpm:.0f} bpm at rate {rate}", ok,
              (f"build {got[0]['build_s']}s ({got[0]['beats']} beats), "
               f"lands in {got[0]['in_s']}s") if got else f"skipped: {st.log}")
    Track.period_s = 60.0 / BPM
    Track2.grid = [{"start_s": 0.0, "end_s": DUR, "period_s": Track.period_s,
                    "first_beat_s": 0.0}]

    print("\n5. legacy flavors all map to the one gesture")
    for legacy in ("drop", "spinback", "stall"):
        sl = nextdrop_system()
        render(sl, 6.0, fire_at=3.0, flavor=legacy)
        got = [e for e in sl.log if e.get("event") == "moment"]
        check(f"'{legacy}' press fires a nextdrop",
              bool(got) and got[0]["flavor"] == "nextdrop", f"{got}")

    if wav:
        try:
            import soundfile as sf
            sf.write(wav, audio, RATE)
            print(f"\nwrote {wav} - the landing is at {hit:.1f}s")
        except Exception as e:
            print(f"\nwav export failed: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
