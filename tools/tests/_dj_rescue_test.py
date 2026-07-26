"""Gate for the DJ rescue paths: abortable transitions + continuity watchdog.

Three scenarios through the REAL offline pipeline (scan -> DJSystem ->
hand-pumped engine):

  1. ABORT: an armed transition is recalled before its point of no return -
     queued events vanish, the incoming deck dies, the outgoing deck's
     gain/EQ restore, the same track keeps playing.
  2. SKIP-WHILE-ARMED: a skip during the armed window recalls the
     transition and replans an urgent exit (the old behavior was a no-op).
  3. WATCHDOG: with selection forced to return nothing, the continuity
     watchdog emergency-picks and hands off before the track runs out -
     the render must never go silent.

Usage:
    python tools/tests/_dj_rescue_test.py           # ALL PASS gate
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RATE = 44100
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def synth_structured(bpm, seconds, seed, bass_hz, pad_hz):
    """Groove with intro/outro so the analyzer finds real mix points
    (same fleet as _dj_brain_test)."""
    beat = 60.0 / bpm
    n = int(seconds * RATE)
    x = np.zeros(n)
    rng = np.random.RandomState(seed)
    intro_e, outro_s = 12.0, seconds - 14.0
    t, k = 0.0, 0
    while t < outro_s:
        if t >= intro_e:
            i0 = int(t * RATE)
            d = int(0.10 * RATE)
            if i0 + d <= n:
                tl = np.arange(d) / RATE
                amp = 1.0 if k % 4 == 0 else 0.75
                x[i0:i0 + d] += amp * np.sin(2 * np.pi * bass_hz * tl) \
                    * np.exp(-tl / 0.045)
        t += beat
        k += 1
    t = 0.0
    while t < seconds - 2.0:
        i0 = int(t * RATE)
        d = int(0.03 * RATE)
        if i0 + d <= n:
            b = rng.randn(d) * np.exp(-np.arange(d) / (0.008 * RATE))
            amp = 0.05 if (t < intro_e or t > outro_s) else 0.10
            x[i0:i0 + d] += amp * np.diff(np.concatenate([[0.0], b]))
        t += beat / 2
    tt = np.arange(n) / RATE
    pad = 0.05 * np.sin(2 * np.pi * pad_hz * tt) \
        + 0.03 * np.sin(2 * np.pi * pad_hz * 1.5 * tt)
    pad[int(outro_s * RATE):] *= np.linspace(1.0, 0.3, n - int(outro_s * RATE))
    x += pad
    return (x / np.max(np.abs(x)) * 0.8).astype(np.float32)


def cancel_unit_test():
    """Submix-level: a cancel recalls exactly its transaction's queued
    events and nothing else."""
    print("Cancel primitive (submix unit):\n")
    from lib.dj.submix import DJSubmix
    sub = DJSubmix()
    far = 10 * RATE
    sub.post({"at": far, "cmd": "gain", "deck": "a", "value": 0.5,
              "txn": 7})
    sub.post({"at": far, "cmd": "stop", "deck": "a", "txn": 7})
    sub.post({"at": far, "cmd": "gain", "deck": "b", "value": 0.9,
              "txn": 8})
    sub.read(1024)                       # queue -> _auto
    n_before = len(sub._auto)
    sub.post({"cmd": "cancel", "txn": 7})
    sub.read(1024)
    left = [e.get("txn") for e in sub._auto]
    check("cancel recalls its txn only", n_before == 3 and left == [8],
          f"3 queued -> {len(sub._auto)} left, txns {left}")


def main():
    cancel_unit_test()

    from scipy.io import wavfile
    from lib.dj.scan import scan_library
    from lib.dj.system import DJSystem
    from lib.dj.themes import Theme
    from lib.audio_engine import AudioEngine

    print("\nRescue scenarios (scan -> DJSystem -> hand-pumped engine):\n")
    tmp = tempfile.mkdtemp(prefix="gl_dj_rescue_")
    specs = [(122.0, 5, 54.0, 220.0), (126.0, 6, 58.0, 262.0),
             (124.0, 7, 50.0, 294.0)]
    for i, (bpm, seed, bass, pad) in enumerate(specs):
        wavfile.write(os.path.join(tmp, f"song_{i}_{int(bpm)}.wav"), RATE,
                      (synth_structured(bpm, 150.0, seed, bass, pad)
                       * 32767).astype(np.int16))
    s = scan_library(tmp, workers=1, vocals_pass=False)
    check("scan clean", s["scanned"] == 3 and s["errors"] == 0,
          f"scanned={s['scanned']} errors={s['errors']}")

    engine = AudioEngine()
    dj = DJSystem(tmp, engine=engine, theme="groove", seed=99,
                  threaded=False, log_dir=tmp)
    ok = dj.start()
    check("dj starts", ok, f"library loaded: {ok}")
    dj.brain.theme = Theme("rescue", bpm_range=(110.0, 135.0),
                           energy_base=0.55, energy_span=0.25,
                           mood_weights={"groove": 1.0, "peak": 0.5},
                           min_play_s=40.0, max_play_s=65.0)

    gen = engine._mixer()
    next(gen)
    block = 4410
    rendered = []

    def pump(seconds):
        for _ in range(int(seconds * RATE) // block):
            buf = gen.send(block)
            rendered.append(
                np.frombuffer(buf, dtype=np.float32).reshape(-1, 2))
            dj.step()

    def pump_until(cond, timeout_s, why, trace=False):
        t = 0.0
        while t < timeout_s:
            pump(0.5)
            t += 0.5
            if cond():
                return True
            if trace and (t % 10.0) < 0.25:
                print(f"    t+{t:.0f} state={dj.state} pos={dj._pos_s()} "
                      f"cur={dj.current.title if dj.current else None} "
                      f"next={dj.next_track.title if dj.next_track else None}"
                      f" err='{dj.last_error}'")
        print(f"    (timeout waiting for {why})")
        return False

    # ---- Scenario 1: abort an armed transition -----------------------------
    got_armed = pump_until(
        lambda: dj.state == "armed" and dj.status()["abortable"],
        180.0, "an abortable armed transition")
    check("transition arms abortable", got_armed,
          f"state={dj.state} style={(dj.plan or {}).get('style')}")
    aborted = False
    txn = dj._txn_id
    cur_before = dj.current.id if dj.current else None
    if got_armed:
        dj.abort_transition()
        dj.step()
        aborted = dj.state == "playing" and dj.plan is None
    check("abort recalls the transition", aborted,
          f"state={dj.state} after abort")
    pump(3.0)                    # let the recovery ramps land
    leftover = [e for e in dj.submix._auto if e.get("txn") == txn]
    check("cancelled events flushed", not leftover,
          f"{len(leftover)} events of txn {txn} still queued")
    tel = dj.submix.telemetry["decks"]
    other = "b" if dj.active_deck == "a" else "a"
    a_ok = (tel[dj.active_deck]["playing"]
            and tel[dj.active_deck]["gain"] > 0.9
            and all(abs(g - 1.0) < 0.05 for g in tel[dj.active_deck]["eq"]))
    check("outgoing deck restored", a_ok,
          f"gain={tel[dj.active_deck]['gain']} eq={tel[dj.active_deck]['eq']}")
    check("incoming deck silenced", not tel[other]["playing"],
          f"deck {other} playing={tel[other]['playing']}")
    check("same track keeps playing",
          dj.current is not None and dj.current.id == cur_before,
          f"current unchanged: {dj.current.title if dj.current else None}")

    # ---- Scenario 2: skip during the armed window ---------------------------
    got_armed2 = pump_until(
        lambda: dj.state == "armed" and dj.status()["abortable"],
        180.0, "a second armed transition")
    skipped = False
    if got_armed2:
        cur_id = dj.current.id
        dj.request_skip()
        dj.step()
        # The skip must recall the old plan and re-arm an urgent exit fast.
        skipped = pump_until(lambda: dj.current is not None
                             and dj.current.id != cur_id, 60.0,
                             "the skip handover")
    check("skip while armed replans", got_armed2 and skipped,
          f"armed={got_armed2}, handover={skipped}")
    # The handover must be REAL audio, not bookkeeping: the new deck has to
    # advance (a leftover abort-recovery stop once killed it silently).
    p0 = dj._pos_s()
    pump(3.0)
    p1 = dj._pos_s()
    check("skip handover actually plays",
          p0 is not None and p1 is not None and p1 - p0 > 2.0,
          f"pos advanced {0.0 if None in (p0, p1) else p1 - p0:.1f}s in 3s")

    # ---- Scenario 3: watchdog rescues a dry selection ------------------------
    # Selection is forced dry BEFORE waiting out the current handover, so
    # the patch lands cleanly on a playing/unarmed machine. The watchdog
    # must inject an emergency pick; the normal planner then still gets
    # first shot at arming a real seam with it.
    dj._pick_next = lambda out_bpm: (None, None)
    cur_id = dj.current.id if dj.current else None
    pump_until(lambda: dj.current is not None and dj.current.id != cur_id
               and dj.state == "playing", 150.0, "the track to run down")
    check("selection is dry", dj.next_track is None and dj.state == "playing",
          f"state={dj.state} next={dj.next_track}")
    dj.seek(dj.current.duration_s - 35.0)
    dj.step()
    cur_id = dj.current.id
    n0 = len(rendered)
    rescued = pump_until(lambda: dj.current is not None
                         and dj.current.id != cur_id, 60.0,
                         "the watchdog rescue")
    check("watchdog rescues a dry selection", rescued,
          f"current={'switched (via ' + dj._history[-1].get('via', '?') + ')' if rescued else 'stuck'} "
          f"last_error='{dj.last_error}'")
    tail = np.concatenate(rendered[n0:], axis=0).mean(axis=1)
    w = RATE // 2
    rms = np.sqrt(np.mean(
        tail[:len(tail) // w * w].reshape(-1, w) ** 2, axis=1))
    check("no dead air through the rescue",
          float(rms.min()) > 0.1 * float(np.median(rms)),
          f"min 0.5s-RMS {rms.min():.3f} vs median {np.median(rms):.3f}")

    # ---- Scenario 4: watchdog survives a dead planner -----------------------
    # The realistic nightmare: _maybe_plan never arms anything (the planner
    # thread spins on a bug). The watchdog alone must keep music playing
    # via the clock-domain emergency fade.
    del dj._pick_next                      # restore real selection
    dj._maybe_plan = lambda: None          # planner is dead
    pump_until(lambda: dj.state == "playing", 30.0, "playing state")
    dj.seek(dj.current.duration_s - 35.0)
    dj.step()
    cur_id = dj.current.id
    n0 = len(rendered)
    rescued2 = pump_until(lambda: dj.current is not None
                          and dj.current.id != cur_id, 60.0,
                          "the emergency handoff")
    check("watchdog survives a dead planner", rescued2,
          f"current={'switched' if rescued2 else 'stuck'}")
    check("handoff was the emergency fade",
          any(h.get("via") == "emergency_fade" for h in dj._history),
          f"history vias: {[h.get('via') for h in dj._history[-3:]]}")
    tail = np.concatenate(rendered[n0:], axis=0).mean(axis=1)
    rms = np.sqrt(np.mean(
        tail[:len(tail) // w * w].reshape(-1, w) ** 2, axis=1))
    check("no dead air through the dead-planner rescue",
          float(rms.min()) > 0.1 * float(np.median(rms)),
          f"min 0.5s-RMS {rms.min():.3f} vs median {np.median(rms):.3f}")

    dj.stop()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
