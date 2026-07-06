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
# Loop: position error e, rate trim u, de/dt = -u + bias. With u = Kp*e +
# Ki*int(e): e'' = -Kp e' - Ki e, damping zeta = Kp / (2 sqrt(Ki)). These
# values give zeta ~0.8 (no ringing) with Kp = beat/TC ~0.25 (tau ~4s,
# comfortably slower than the ~0.7s measurement delay).
PLL_TC = 2.0                     # seconds to slew a measured error away
PLL_KI = 0.012                   # integral gain (learns the tempo bias)
PLL_MAX_TRIM = 0.012             # +/- 1.2% - real house grids drift this much
PLL_DEADBAND = 0.004             # beats
RESNAP_ERR = 0.08                # re-snap if this far off while still fading in
RESNAP_GAIN = 0.12               # ...only while near-silent (a reseek is a
                                 # hard jump; never do it once audible)


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
        self._apll_clock = 0     # last audio-phase measurement clock
        self._apll_err = None    # latest audio-phase error (beats, + = late)
        self._apll_i = 0.0       # integral term (learned tempo bias)
        self._fx = []            # active one-shots: [buffer, pos, gain]
        self._duck = None        # {"on", "depth"} sidechain of the slave
        self.record_q = None     # set to a queue to tap the mix (recording)
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
        if cmd == "fx_play":
            # Pre-rendered one-shot (riser/impact) - the third mini-layer.
            self._fx.append([np.asarray(e["samples"], dtype=np.float32),
                             0, float(e.get("gain", 1.0))])
            return
        if cmd == "duck":
            self._duck = ({"depth": float(e.get("depth", 0.22))}
                          if e.get("on") else None)
            return
        deck = self.decks.get(e.get("deck", ""))
        if cmd == "load":
            # Samples were decoded on another thread; this just mounts them.
            deck.load(e["samples"], e.get("track_id"), e.get("grid"),
                      e.get("gain_db", 0.0), e.get("kick_offset_s", 0.0),
                      e.get("pitch_st", 0.0))
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
        elif cmd == "filter":
            deck.filter.set(mode=e.get("mode"),
                            cutoff_hz=e.get("cutoff_hz"),
                            ramp_s=e.get("ramp_s", 0.0), q=e.get("q"))
        elif cmd == "echo":
            deck.echo.set(active=e.get("active"),
                          delay_s=e.get("delay_s"),
                          feedback=e.get("feedback"), wet=e.get("wet"))
        elif cmd == "brake":
            deck.brake(e.get("duration_s", 1.5))
        elif cmd == "jump":
            deck.jump_cut(e["time_s"])
        elif cmd == "sync":
            self._sync = {"slave": e["slave"], "master": e["master"]}
            sl = self.decks.get(e["slave"])
            if sl is not None:
                sl.stretch.no_bypass = True      # no mode-flap phase kicks
            # DJ sync SNAP: align the incoming deck at gain ~0 (inaudible),
            # then the transient PLL holds it through the blend.
            master = self.decks.get(e["master"])
            slave = self.decks.get(e["slave"])
            if master and slave and master.playing and slave.playing \
                    and master.grid and slave.grid:
                # Grid phase alignment: the grids are onset-locked (kick+hat
                # transients), accurate to ~25ms - the right anchor for this
                # genre. (Low-band audio alignment was tried and locks onto
                # OFFBEAT bass stabs instead of kicks - worse, not better.)
                slave.phase_snap(master.beat_phase())
        elif cmd == "end_sync":
            self._sync = None
            self._apll_err = None
            self._apll_i = 0.0
            for d in self.decks.values():
                d.rate_trim = 0.0
                d.stretch.no_bypass = False
                d.stretch.phase_trim = 0.0

    def _audio_phase_err(self, master, slave, beat_s):
        """Beat-phase error measured from the two decks' ACTUAL output
        TRANSIENTS (positive difference of the amplitude rings - kick/hat
        attacks), cross-correlated over the last ~1.5s within +/- a QUARTER
        beat: a refinement around the grid alignment, never a re-anchor
        (wider windows lock onto offbeat bass patterns in this genre).
        Positive = slave's transients land LATE. None when unconfident."""
        from lib.dj.deck import ENV_FPS
        n = int(1.5 * ENV_FPS)
        em = np.maximum(np.diff(master.out_env[-n:].astype(np.float64)), 0.0)
        es = np.maximum(np.diff(slave.out_env[-n:].astype(np.float64)), 0.0)
        if em.max() <= 1e-5 or es.max() <= 1e-5:
            return None
        em -= em.mean()
        es -= es.mean()
        xc = np.correlate(em, es, mode="full")
        mid = len(es) - 1
        half = max(int(0.25 * beat_s * ENV_FPS), 2)
        seg = xc[mid - half:mid + half + 1]
        if len(seg) < 3:
            return None
        k = int(np.argmax(seg))
        peak = float(seg[k])
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-12
        if peak <= 0 or peak < 1.8 * rms:        # no confident beat pattern
            return None
        lag = float(k)
        if 0 < k < len(seg) - 1:                 # parabolic sub-bin (~1ms)
            y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-12:
                lag += 0.5 * (y0 - y2) / den
        # xc[mid+lag] = sum em[i+lag]*es[i]: positive lag means es must
        # shift LATER to match em, i.e. the slave's kicks are EARLY.
        return -((lag - half) / ENV_FPS) / beat_s   # beats; + = slave late

    def _run_pll(self):
        """Continuously trim the slave deck's rate to keep its ACTUAL kicks
        on the master's - audio phase first (the grids sit differently vs
        each track's real kicks, so grid phase alone leaves ~30ms flam),
        grid phase as the fallback."""
        cfg = self._sync
        if not cfg:
            return
        master = self.decks.get(cfg["master"])
        slave = self.decks.get(cfg["slave"])
        if not (master and slave and master.playing and slave.playing
                and master.grid and slave.grid):
            return
        beat_s = master.beat_period_s() or 0.5
        grid_err = slave.beat_phase() - master.beat_phase()
        grid_err = (grid_err + 0.5) % 1.0 - 0.5
        # If drift outruns the PLL while the slave is still fading in, snap
        # again (inaudible at low gain) rather than let it trainwreck.
        if abs(grid_err) > RESNAP_ERR and slave.gain < RESNAP_GAIN:
            slave.phase_snap(master.beat_phase())
            slave.rate_trim = 0.0
            self._apll_err = None
            self._apll_i = 0.0
            return
        # Audio-phase measurement 4x/second; PI control on it. The plant is
        # a delayed position measurement driven by a rate: the INTEGRAL term
        # learns the true tempo-ratio bias (why grids drift at all) so the
        # proportional term only handles small residuals - no limit cycle.
        if self.clock - self._apll_clock >= RATE // 4:
            self._apll_clock = self.clock
            a = self._audio_phase_err(master, slave, beat_s)
            if a is not None:
                hist = getattr(self, "_apll_hist", [])
                hist = (hist + [a])[-3:]
                self._apll_hist = hist
                prev = self._apll_err if self._apll_err is not None else a
                self._apll_err = 0.6 * prev + 0.4 * a
                self._apll_i = float(np.clip(
                    self._apll_i + PLL_KI * self._apll_err * beat_s,
                    -0.008, 0.008))
                # PER-BEAT MICRO-FOLLOWING: absorb the measured PHASE error
                # directly in the stretcher (click-free WSOLA cursor bias).
                # TRUST GATES (both hard-won): (1) the correlation ring
                # spans ~1.5s, so after a nudge the next measurements still
                # contain pre-nudge history - nudge, then hold until the
                # ring refreshes, or it oscillates. (2) On organic material
                # the envelope xcorr often measures RHYTHM-PATTERN offset
                # (A's shakers vs B's congas), not kick flam - chasing that
                # made lag WORSE. Real flam is PERSISTENT: act only when 3
                # consecutive raw measurements agree in sign and spread.
                stable = (len(hist) == 3
                          and max(hist) - min(hist) < 0.07
                          and (all(h > 0 for h in hist)
                               or all(h < 0 for h in hist)))
                blended = (0.7 * (-((slave.beat_phase()
                                     - master.beat_phase() + 0.5) % 1.0
                                    - 0.5))
                           + 0.3 * self._apll_err)
                if (slave.gain >= RESNAP_GAIN and stable
                        and abs(blended) * beat_s > 0.012
                        and self.clock - getattr(self, "_nudge_clock", 0)
                        >= int(1.6 * RATE)):
                    nudge = float(np.clip(blended * beat_s * 0.7,
                                          -0.035, 0.035))
                    slave.stretch.phase_trim += nudge * RATE
                    self._nudge_clock = self.clock
                    self._apll_err = None    # stale until the ring refreshes
                    self._apll_hist = []
            else:
                self._apll_err = None
        if self._apll_err is not None and abs(self._apll_err) > 0.05 \
                and slave.gain < RESNAP_GAIN:
            # Still quiet? A large audio error is corrected by an inaudible
            # JUMP (the PI slews ~12ms/s - a blend would end first).
            slave.nudge_seconds(self._apll_err * beat_s)
            self._apll_err = None
            self._apll_i = 0.0
            slave.rate_trim = 0.0
            return
        # GRID IS PRIMARY. The grids are onset-locked to ~25ms; the audio
        # xcorr on organic material often measures rhythm-PATTERN offset
        # (shakers vs congas), not kick flam - when it was allowed to lead,
        # it dragged decks 30-80 ms OFF grid (measured). It now only
        # REFINES, 30% weight, and only while measurements are stable.
        hist = getattr(self, "_apll_hist", [])
        audio_ok = (self._apll_err is not None and len(hist) == 3
                    and max(hist) - min(hist) < 0.07
                    and (all(h > 0 for h in hist)
                         or all(h < 0 for h in hist)))
        err_s = -grid_err * beat_s
        if audio_ok:
            err_s = 0.7 * err_s + 0.3 * (self._apll_err * beat_s)
        if abs(err_s) < PLL_DEADBAND * beat_s and not audio_ok:
            slave.rate_trim *= 0.9
            return
        want = self._apll_i + err_s / PLL_TC
        trim = float(np.clip(want, -PLL_MAX_TRIM, PLL_MAX_TRIM))
        # One-pole smoothing keeps the correction from breathing audibly.
        slave.rate_trim = 0.8 * slave.rate_trim + 0.2 * trim

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
            slave = self._sync["slave"] if self._sync else None
            master = self.decks.get(self._sync["master"]) if self._sync \
                else None
            for name, d in self.decks.items():
                if not (d.playing or d.echo.ringing):
                    continue
                blk = d.read(m)
                # SIDECHAIN DUCK: pull the incoming deck down a few dB on
                # each of the master's kicks - overlapping tracks sound
                # mixed, not stacked. Recovery ~90 ms, kick-shaped.
                if (self._duck and name == slave and master is not None
                        and master.playing and d.playing):
                    beat_s = master.beat_period_s() or 0.5
                    ph0 = master.beat_phase()
                    ph = (ph0 + np.arange(m) / RATE / beat_s) % 1.0
                    env = 1.0 - self._duck["depth"] * np.exp(
                        -ph * beat_s / 0.09)
                    blk = blk * env[:, None].astype(np.float32)
                out[pos:pos + m] += blk
            for fx in self._fx:
                buf, fpos, g = fx
                k = min(m, len(buf) - fpos)
                if k > 0:
                    out[pos:pos + k] += buf[fpos:fpos + k] * g
                    fx[1] += k
            self._fx = [f for f in self._fx if f[1] < len(f[0])]
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

        # Soft peak guard: two hot tracks + an impact one-shot can stack
        # past full scale. Per-SAMPLE 3:1 soft knee above 0.92 - no
        # frame-level normalization (that reads as pumping; see the fan
        # limiter lesson), just gentle rounding of the rare peaks.
        peaks = np.abs(out)
        hot = peaks > 0.92
        if np.any(hot):
            out[hot] = np.sign(out[hot]) * (0.92 + (peaks[hot] - 0.92) / 3.0)
            # ...and a TRUE ceiling: the knee alone let a double-drop +
            # impact stack reach 1.02 over a full simulated night.
            np.clip(out, -0.985, 0.985, out=out)
        if self.record_q is not None:
            try:
                self.record_q.put_nowait(out.copy())
            except Exception:
                pass
        snap = self._snapshot()
        snap["peak"] = float(np.abs(out).max()) if len(out) else 0.0
        self.telemetry = snap
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
                "braking": d._brake is not None,
            }
        return {"clock": self.clock, "clock_s": round(self.clock / RATE, 3),
                "sync": dict(self._sync) if self._sync else None,
                "mix_gain": round(self.mix_gain, 4), "decks": decks,
                "pending_events": len(self._auto)}
