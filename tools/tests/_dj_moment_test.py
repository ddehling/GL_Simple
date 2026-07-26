"""Gate for the operator MOMENT gesture (DJSystem._do_moment).

The moment is the one control that deliberately CUTS the live deck - a
high-pass sweep, then a beat of near-silence, then everything back on the
downbeat with an impact. Every failure mode of that is a public one, so
this renders the real DJSubmix (real Deck, real SweepFilter, real master
limiter) and measures the audio:

  1. FULL GESTURE: the music holes out, the sweep drains the bass, the
     landing is a step up in level, and the deck is left EXACTLY as it was
     found (gain, EQ, filter). Nothing clips.
  2. RECALL: a second press cancels a gesture in flight and restores the
     deck - the audio must not be left holed.
  3. ARMED: with a transition armed the seam script owns the deck, so the
     moment must be layer-only and touch nothing.
  4. PREEMPTION: _cancel_moment mid-gesture (what _arm, _do_seek and the
     watchdog handoff call) restores the deck.
  5. SCHEDULE: across 90-174 bpm and off-unity rates the build stays a
     musical length and the landing stays within one phrase.
  6. DECK DOWN: a deck that is faded out is skipped, not shaped.

Usage:
    python tools/tests/_dj_moment_test.py           # ALL PASS gate
    python tools/tests/_dj_moment_test.py out.wav   # + render to listen to
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj.submix import DJSubmix, RATE          # noqa: E402
from lib.dj.system import DJSystem                # noqa: E402
import lib.dj.fx as fxmod                         # noqa: E402

BPM = 120.0
DUR = 180.0
CLEAN = {"gain": 1.0, "eq": [1.0, 1.0, 1.0], "filter": "off"}
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
    x = (x / np.abs(x).max() * 0.85).astype(np.float32)
    return np.stack([x, x], axis=1)


class Track:
    """The slice of TrackInfo _do_moment actually reads."""
    id, title = 1, "synthetic"
    period_s, phrase_beats, duration_s = 60.0 / BPM, 32, DUR

    def nearest_phrase(self, t):
        span = self.phrase_beats * self.period_s
        return round(t / span) * span


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
    s.engine = None
    s.log = []
    s._log = s.log.append
    return s


def render(s, seconds, fire_at=None, again_at=None, cancel_at=None):
    """Render `seconds` of audio, pressing MOMENT at the given offsets."""
    n, out = 2048, []
    for i in range(int(seconds * RATE / n)):
        now = i * n / RATE
        if fire_at is not None and now >= fire_at:
            s._do_moment()
            fire_at = None
        if again_at is not None and now >= again_at:
            s._do_moment()
            again_at = None
        if cancel_at is not None and now >= cancel_at:
            s._cancel_moment("armed")       # what _arm() does to take over
            cancel_at = None
        out.append(s.submix.read(n))
    return np.concatenate(out)


def deck_state(s):
    d = s.submix.decks["a"]
    return {"gain": round(d.gain, 3),
            "eq": [round(float(g), 3) for g in d.eq.gains],
            "filter": d.filter.mode}


def peak(a, t0, t1):
    return float(np.abs(a[int(t0 * RATE):int(t1 * RATE)]).max())


def rms(a, t0, t1):
    w = a[int(t0 * RATE):int(t1 * RATE)]
    return float(np.sqrt(np.mean(w ** 2)))


def low_rms(a, t0, t1, hz=200.0):
    """RMS below `hz` - measures the sweep itself, independent of how much
    mid/high content the material happens to carry."""
    w = a[int(t0 * RATE):int(t1 * RATE), 0]
    sos = butter(4, hz, "low", fs=RATE, output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, w) ** 2)))


def render_music_only(state="playing"):
    """Same render with the one-shots silenced: the hole has to be in the
    TRACK, not merely covered by the riser sitting on top of it."""
    real = fxmod.at_peak
    fxmod.at_peak = lambda buf, pk: np.zeros_like(buf)
    try:
        s = make_system(state)
        return s, render(s, 40.0, fire_at=4.0)
    finally:
        fxmod.at_peak = real


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n1. full gesture")
    s = make_system("playing")
    audio = render(s, 40.0, fire_at=4.0)
    hit = s._moment_clock / RATE
    ms, music = render_music_only()
    mh = ms._moment_clock / RATE
    normal = peak(music, mh - 14, mh - 10)
    check("the music holes out before the landing",
          peak(music, mh - 0.35, mh - 0.08) < 0.25 * normal,
          f"peak {peak(music, mh - 0.35, mh - 0.08):.3f} vs {normal:.3f} "
          f"normal")
    check("the build sweeps the bass out",
          low_rms(music, mh - 2.5, mh - 0.6)
          < 0.25 * low_rms(music, mh - 14, mh - 10),
          f"<200 Hz RMS {low_rms(music, mh - 2.5, mh - 0.6):.4f} vs "
          f"{low_rms(music, mh - 14, mh - 10):.4f} normal")
    check("the landing is a step up in level",
          rms(audio, hit, hit + 0.4) > 1.6 * rms(audio, hit - 0.45,
                                                 hit - 0.05),
          f"RMS {rms(audio, hit, hit + 0.4):.3f} after vs "
          f"{rms(audio, hit - 0.45, hit - 0.05):.3f} through the hole")
    check("the track is all the way back afterwards",
          peak(music, mh + 5, mh + 9) >= 0.85 * normal,
          f"peak {peak(music, mh + 5, mh + 9):.3f} vs {normal:.3f}")
    check("the deck is left exactly as it was found",
          deck_state(s) == CLEAN, f"{deck_state(s)}")
    check("nothing clips", float(np.abs(audio).max()) <= 0.9851,
          f"peak {float(np.abs(audio).max()):.3f} (ceiling 0.985)")

    print("\n2. second press recalls it")
    s2 = make_system("playing")
    a2 = render(s2, 30.0, fire_at=4.0, again_at=8.0)
    check("logged the recall",
          any(e.get("event") == "moment_cancel" for e in s2.log),
          f"{[e.get('event') for e in s2.log]}")
    check("the deck is restored", deck_state(s2) == CLEAN,
          f"{deck_state(s2)}")
    check("nothing is left holed", peak(a2, 26, 30) >= 0.85 * peak(a2, 0, 2),
          f"tail peak {peak(a2, 26, 30):.3f} vs {peak(a2, 0, 2):.3f}")
    check("the countdown clears", s2._moment_clock == 0,
          f"moment_clock {s2._moment_clock}")

    print("\n3. armed: layer only")
    s3 = make_system("armed")
    _ms3, m3 = render_music_only("armed")
    h3 = _ms3._moment_clock / RATE
    render(s3, 30.0, fire_at=4.0)
    check("the deck is untouched", deck_state(s3) == CLEAN,
          f"{deck_state(s3)}")
    # RMS over the last two beats, not peak over a 0.2 s slice: a short
    # window lands wherever the kick isn't and reads low on a track that
    # is playing perfectly normally.
    check("no hole is cut in the music",
          rms(m3, h3 - 1.0, h3 - 0.02) > 0.7 * rms(m3, h3 - 14, h3 - 10),
          f"RMS {rms(m3, h3 - 1.0, h3 - 0.02):.3f} vs "
          f"{rms(m3, h3 - 14, h3 - 10):.3f} normal")
    check("it is flagged layer-only",
          any(e.get("layer_only") for e in s3.log), f"{s3.log}")

    print("\n4. preempted mid-gesture (arm / seek / watchdog)")
    s4 = make_system("playing")
    a4 = render(s4, 20.0, fire_at=4.0, cancel_at=8.0)
    check("the deck is restored", deck_state(s4) == CLEAN,
          f"{deck_state(s4)}")
    check("the audio is unbroken",
          peak(a4, 16, 20) >= 0.85 * peak(a4, 0, 2),
          f"tail peak {peak(a4, 16, 20):.3f} vs {peak(a4, 0, 2):.3f}")

    print("\n5. schedule across tempos")
    for bpm, rate in [(90.0, 1.0), (120.0, 1.0), (128.0, 1.03), (174.0, 1.0)]:
        Track.period_s = 60.0 / bpm
        st = make_system("playing")
        st.submix.post({"cmd": "rate", "deck": "a", "value": rate})
        render(st, 6.0, fire_at=3.0)
        got = [e for e in st.log if e.get("event") == "moment"]
        beat = Track.period_s / rate
        ok = bool(got) and 3.0 <= got[0]["build_s"] <= 10.0 \
            and got[0]["build_s"] + 1.0 <= got[0]["in_s"] \
            <= got[0]["build_s"] + 16 * beat
        check(f"{bpm:.0f} bpm at rate {rate}", ok,
              (f"build {got[0]['build_s']}s ({got[0]['beats']} beats), "
               f"lands in {got[0]['in_s']}s") if got else f"skipped: {st.log}")
    Track.period_s = 60.0 / BPM

    print("\n6. deck faded out")
    s6 = make_system("playing")
    s6.submix.post({"cmd": "gain", "deck": "a", "value": 0.0, "ramp_s": 0.01})
    render(s6, 6.0, fire_at=3.0)
    check("skipped with a reason",
          any(e.get("event") == "moment_skipped" for e in s6.log),
          f"{s6.log}")
    check("nothing was scheduled", s6._moment_clock == 0,
          f"moment_clock {s6._moment_clock}")

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
