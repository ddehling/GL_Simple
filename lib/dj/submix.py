"""DJSubmix: the DJ's entire mix as ONE audio-engine track.

Mounted with AudioEngine.attach_track(); the mixer treats it like any other
track (rides the limiter, master volume, monitor tap -> visuals). Internally
it owns two decks, a sample-accurate automation lane, and the beat-sync PLL.

Threading contract: read() runs ONLY on the audio callback thread. Everything
else talks to it through a lock-free SimpleQueue (same idiom as AudioEngine's
own command queue). Telemetry flows back as an immutable dict snapshot
published by reference assignment (atomic in CPython).

Automation events are dicts with an `at` sample timestamp on the submix's
own sample clock, evaluated in 256-frame sub-blocks:
    {"at", "cmd": "start"|"stop"|"cue"|"loop"|"clear_loop"|"seek"
                |"gain"|"eq"|"rate"|"sync"|"end_sync"|"load", "deck", ...}
so a transition scripted 2 bars ahead lands on the exact downbeat sample
regardless of the device's callback size.
"""
from queue import SimpleQueue

import numpy as np

from lib.dj.deck import Deck

RATE = 44100
SUB_BLOCK = 256
PLL_GAIN = 1.2                   # rate trim per beat of phase error
PLL_MAX_TRIM = 0.012             # +/- 1.2% - real house grids drift this much
PLL_DEADBAND = 0.004             # beats
RESNAP_ERR = 0.08                # re-snap if this far off while still fading in
RESNAP_GAIN = 0.50               # ...below half gain (glitch masked by A)


class DJSubmix:
    # Track protocol flags for AudioEngine._mixer:
    is_narrative = False
    is_ambient = False
    is_soundpool = False

    def __init__(self):
        self.decks = {"a": Deck("a"), "b": Deck("b")}
        self._q = SimpleQueue()
        self.clock = 0           # output frames rendered since attach
        self.mix_gain = 1.0
        self._fade = None        # (per_frame_delta, target)
        self.done = False
        self._auto = []          # automation events sorted by 'at'
        self._sync = None        # {"slave": name, "master": name}
        self.telemetry = {}      # replaced wholesale each read()

    # -- control-thread API ----------------------------------------------------
    def post(self, event):
        """Queue one automation/control event (dict; 'at' defaults to now)."""
        self._q.put(event)

    def post_many(self, events):
        for e in events:
            self._q.put(e)

    def fade_out(self, duration=1.0):
        """AudioEngine stop_all contract: ramp to silence then report done."""
        self._q.put({"cmd": "_fade_out", "duration": max(duration, 0.05)})

    # -- audio-callback side -----------------------------------------------------
    def _apply(self, e):
        cmd = e.get("cmd")
        if cmd == "_fade_out":
            self._fade = (1.0 / (e["duration"] * RATE), 0.0)
            return
        deck = self.decks.get(e.get("deck", ""))
        if cmd == "load":
            # Samples were decoded on another thread; this just mounts them.
            deck.load(e["samples"], e.get("track_id"), e.get("grid"),
                      e.get("gain_db", 0.0))
            if "cue_s" in e:
                deck.cue(e["cue_s"])
        elif cmd == "start":
            deck.start()
        elif cmd == "stop":
            deck.stop()
        elif cmd == "unload":
            deck.unload()
        elif cmd == "cue":
            deck.cue(e["time_s"])
        elif cmd == "loop":
            deck.set_loop(e["start_s"], e["end_s"])
        elif cmd == "clear_loop":
            deck.clear_loop()
        elif cmd == "release_loop":
            deck.release_loop()
        elif cmd == "gain":
            deck.set_gain(e["value"], e.get("ramp_s", 0.05))
        elif cmd == "eq":
            deck.eq.set_gains(e.get("low"), e.get("mid"), e.get("high"),
                              e.get("ramp_s", 0.05))
        elif cmd == "rate":
            deck.set_rate(e["value"], e.get("ramp_s", 0.0))
        elif cmd == "sync":
            self._sync = {"slave": e["slave"], "master": e["master"]}
            # DJ sync SNAP. Align the incoming deck to the master (inaudible
            # at gain ~0), then the PLL holds. Grid-phase alignment leaves
            # ~30ms of kick flam because each track's grid sits differently
            # vs its real kicks; so we ALSO cross-correlate the two decks'
            # actual low-band (kick) audio and slide the slave onto the
            # master's transients - the audio is the ground truth.
            master = self.decks.get(e["master"])
            slave = self.decks.get(e["slave"])
            if master and slave and master.playing and slave.playing \
                    and master.grid and slave.grid:
                slave.phase_snap(master.beat_phase())     # coarse: grid
                self._onset_align(master, slave)          # fine: real kicks
        elif cmd == "end_sync":
            self._sync = None
            for d in self.decks.values():
                d.rate_trim = 0.0

    def _onset_align(self, master, slave):
        """Slide the slave within +/- half a beat so its actual kicks sit on
        the master's, via low-band envelope cross-correlation."""
        period = master.beat_period_s() or 0.5
        env_fps = 200
        try:
            em = master.kick_env(2.0, env_fps)
            es = slave.kick_env(2.0, env_fps)
        except Exception:
            return
        if len(em) < 8 or len(es) < 8 or em.max() <= 0 or es.max() <= 0:
            return
        em = em - em.mean()
        es = es - es.mean()
        xc = np.correlate(em, es, mode="full")
        # Only search +/- half a beat so we align beat phase, not skip beats.
        half = int(0.5 * period * env_fps)
        mid = len(es) - 1
        lo, hi = mid - half, mid + half + 1
        seg = xc[max(lo, 0):min(hi, len(xc))]
        if not len(seg):
            return
        lag = (max(lo, 0) + int(np.argmax(seg))) - mid   # env frames
        dt = lag / env_fps                               # seconds: slave += dt
        # Only FINE-TUNE: the grid snap already got us within a beat; a small
        # correction removes flam, but a large nudge here risks aligning to
        # the wrong kick on tracks with different patterns, so cap it tight.
        peak = float(seg.max())
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        if abs(dt) <= 0.05 and peak > 2.5 * rms:
            slave.nudge_seconds(dt)

    def _run_pll(self):
        """Trim the slave deck's rate toward the master's beat phase."""
        cfg = self._sync
        if not cfg:
            return
        master = self.decks.get(cfg["master"])
        slave = self.decks.get(cfg["slave"])
        if not (master and slave and master.playing and slave.playing
                and master.grid and slave.grid):
            return
        err = slave.beat_phase() - master.beat_phase()
        err = (err + 0.5) % 1.0 - 0.5            # wrap to [-0.5, 0.5)
        # If drift outruns the PLL while the slave is still fading in, snap
        # again (inaudible at low gain) rather than let it trainwreck.
        if abs(err) > RESNAP_ERR and slave.gain < RESNAP_GAIN:
            slave.phase_snap(master.beat_phase())
            slave.rate_trim = 0.0
            return
        if abs(err) < PLL_DEADBAND:
            slave.rate_trim *= 0.9               # relax toward 0
            return
        # Positive error = slave ahead -> slow it down.
        trim = float(np.clip(-PLL_GAIN * err, -PLL_MAX_TRIM, PLL_MAX_TRIM))
        # One-pole smoothing keeps the correction from breathing audibly.
        slave.rate_trim = 0.7 * slave.rate_trim + 0.3 * trim

    def read(self, n):
        """Called by AudioEngine._mixer on the audio thread."""
        try:
            while True:
                e = self._q.get_nowait()
                e.setdefault("at", self.clock)
                if e.get("cmd") == "_fade_out" or e["at"] <= self.clock:
                    self._apply(e)
                else:
                    self._auto.append(e)
        except Exception:
            pass
        if self._auto:
            self._auto.sort(key=lambda e: e["at"])

        out = np.zeros((n, 2), dtype=np.float32)
        pos = 0
        while pos < n:
            m = min(SUB_BLOCK, n - pos)
            while self._auto and self._auto[0]["at"] <= self.clock:
                self._apply(self._auto.pop(0))
            self._run_pll()
            for d in self.decks.values():
                if d.playing:
                    out[pos:pos + m] += d.read(m)
            if self._fade is not None:
                step, target = self._fade
                g0 = self.mix_gain
                g1 = max(target, g0 - step * m)
                out[pos:pos + m] *= np.linspace(g0, g1, m)[:, None]
                self.mix_gain = g1
                if g1 <= target:
                    self.done = True
            elif self.mix_gain != 1.0:
                out[pos:pos + m] *= self.mix_gain
            self.clock += m
            pos += m

        self.telemetry = self._snapshot()
        return out

    def _snapshot(self):
        decks = {}
        for name, d in self.decks.items():
            decks[name] = {
                "ready": d.ready, "playing": d.playing,
                "finished": d.finished, "track_id": d.track_id,
                "time_s": round(d.source_time_s(), 3) if d.ready else 0.0,
                "beat_phase": round(d.beat_phase(), 4) if d.ready else 0.0,
                "rate": round(d.effective_rate(), 5),
                "gain": round(d.gain, 4),
                "eq": [round(float(g), 3) for g in d.eq.gains],
                "loop": d.loop,
            }
        return {"clock": self.clock, "clock_s": round(self.clock / RATE, 3),
                "sync": dict(self._sync) if self._sync else None,
                "mix_gain": round(self.mix_gain, 4), "decks": decks,
                "pending_events": len(self._auto)}
