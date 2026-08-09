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

Events may carry a "txn" tag; {"cmd": "cancel", "txn": id} recalls every
NOT-YET-FIRED event of that transaction - the abort path for an armed
transition (already-fired events are the caller's to unwind).
"""
import os
from queue import SimpleQueue

import numpy as np

from lib.dj.deck import Deck

RATE = 44100
SUB_BLOCK = 256
TEL_HIST_FRAMES = RATE * 2       # 2s of telemetry history for the visuals
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
        # A and B are the transition pair; C is the LOOP LAYER - a
        # percussion bed ridden under whatever A/B are doing (see
        # DJSystem._do_layer). It is never chosen as active_deck or as a
        # sync slave, so every "b" if active == "a" else "a" swap in
        # system.py is blind to it; read() and _snapshot() iterate the
        # dict, so it mixes and reports with no other change.
        self.decks = {"a": Deck("a"), "b": Deck("b"), "c": Deck("c")}
        self._q = SimpleQueue()
        self.clock = 0           # output frames rendered since attach
        self.mix_gain = 1.0
        self._fade = None        # (per_frame_delta, target)
        self._mgain_ramp = None  # (target, per_frame_delta) master-bus ramp
        self.done = False
        self._auto = []          # automation events sorted by 'at'
        self._sync = None        # {"slave": name, "master": name}
        self._sync_bias = 0.0    # kick-alignment bias (beats) the PLL holds
        self._apll_clock = 0     # last audio-phase measurement clock
        self._apll_err = None    # latest audio-phase error (beats, + = late)
        self._apll_i = 0.0       # integral term (learned tempo bias)
        self._fx = []            # active one-shots: [buffer, pos, gain]
        self._duck = None        # {"on", "depth"} sidechain of the slave
        self.record_q = None     # set to a queue to tap the mix (recording)
        self._stat_resnaps = 0   # per-sync seam-quality counters (telemetry)
        self._stat_nudges = 0
        self._stat_cals = 0      # run-in tempo calibrations applied
        self._cal_last = None    # clock of the last pre-audible resnap
        self._cal_applied = 0.0  # cumulative base-rate correction (frac)
        self.telemetry = {}      # replaced wholesale each read()
        # (clock, snapshot) for telemetry_delayed. A LIST that read() replaces
        # wholesale, never mutates in place: read() runs on the audio render-
        # ahead thread while the main thread iterates this in
        # telemetry_delayed(), and an in-place deque append/popleft under that
        # iteration raises "deque mutated during iteration".
        self._tel_hist = []

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
        if cmd == "mix_gain":
            # MASTER-BUS gain, the one gesture that works while a seam
            # script owns both decks (see DJSystem._moment_cut). Separate
            # from _fade, which is the shutdown ramp and sets `done` -
            # and subordinate to it: once a shutdown fade is running,
            # nothing may push the master back up.
            if self._fade is not None:
                return
            tgt = float(np.clip(e.get("value", 1.0), 0.0, 1.0))
            ramp = max(float(e.get("ramp_s", 0.0)), 0.0)
            self._mgain_ramp = None if ramp <= 0.0 else \
                (tgt, abs(tgt - self.mix_gain) / (ramp * RATE))
            if ramp <= 0.0:
                self.mix_gain = tgt
            return
        if cmd == "duck":
            self._duck = ({"depth": float(e.get("depth", 0.22))}
                          if e.get("on") else None)
            return
        if cmd == "cancel":
            # Recall a transaction: drop every queued event carrying this
            # txn tag. Fired events are gone - the caller posts its own
            # recovery ramp (see DJSystem._do_abort).
            txn = e.get("txn")
            self._auto = [a for a in self._auto if a.get("txn") != txn]
            return
        deck = self.decks.get(e.get("deck", ""))
        if cmd == "load":
            # Samples were decoded on another thread; this just mounts them.
            deck.load(e["samples"], e.get("track_id"), e.get("grid"),
                      e.get("gain_db", 0.0), e.get("kick_offset_s", 0.0),
                      e.get("pitch_st", 0.0), stems=e.get("stems"))
            # Phase offsets (beatpower --phase) are measured against the
            # DB grid; a live tempo-FIXED grid has its own phase, so the
            # kick bias must not stack a stale offset onto it.
            deck.grid_is_db = e.get("grid_is_db", True)
            if "cue_s" in e:
                deck.cue(e["cue_s"])
        elif cmd == "attach_stems":
            # Late stem mount for an ALREADY-PLAYING deck (the outgoing
            # side of acapella_out): inert until a stem gain diverges.
            deck.attach_stems(e.get("stems") or {})
        elif cmd == "stem_gains":
            deck.set_stem_gains(e.get("gains") or {}, e.get("ramp_s", 0.05))
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
            # curve: "linear" (default) or "power" for the constant-power
            # crossfade law - see Deck.set_gain.
            deck.set_gain(e["value"], e.get("ramp_s", 0.05),
                          e.get("curve", "linear"))
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
            self._sync = {"slave": e["slave"], "master": e["master"],
                          "audio": e.get("audio_pll", True)}
            self._stat_resnaps = 0
            self._stat_nudges = 0
            self._stat_cals = 0
            self._cal_last = None
            self._cal_applied = 0.0
            sl = self.decks.get(e["slave"])
            if sl is not None:
                sl.stretch.no_bypass = True      # no mode-flap phase kicks
            # DJ sync SNAP: align the incoming deck at gain ~0 (inaudible),
            # then the transient PLL holds it through the blend.
            master = self.decks.get(e["master"])
            slave = self.decks.get(e["slave"])
            self._sync_bias = 0.0
            if master and slave and master.playing and slave.playing \
                    and master.grid and slave.grid:
                # Kick-aligned snap: grid phase plus the measured
                # music-vs-grid bias (see _sync_bias_beats - the stored
                # grid phase misses the audible kicks by ~48ms median, so
                # plain grid alignment is lattice alignment, not music).
                # The brain ships the bias in the event, computed from
                # the SAME profile buckets that shifted the seam anchors
                # (two independent lookups disagreed - 151ms error);
                # _sync_bias_beats is only the fallback for sync events
                # that carry none. Fixed once per sync session; the PLL
                # holds the same target.
                b = e.get("bias_beats")
                self._sync_bias = (float(b) if b is not None else
                                   self._sync_bias_beats(master, slave))
                slave.phase_snap(master.beat_phase() + self._sync_bias)
        elif cmd == "end_sync":
            self._sync = None
            self._sync_bias = 0.0
            self._apll_err = None
            self._apll_i = 0.0
            for d in self.decks.values():
                d.rate_trim = 0.0
                d.stretch.no_bypass = False
                d.stretch.phase_trim = 0.0

    @staticmethod
    def _sync_bias_beats(master, slave):
        """Offset the slave's grid target so the MUSIC lands together.

        The stored grids are periodically right (high confidence) but
        their PHASE misses the audible kicks by ~48ms median in seam
        regions (measured 2026-08-04: rendered low-band attacks vs
        trace-projected beats; 17/20 deck-regions >25ms, signs differing
        per track and side). Grid-to-grid sync therefore aligns LATTICES
        while the ear hears flam - gates showed 5ms 'lock' over audible
        double beats. beatpower --phase measures each track's
        attack-peak-vs-grid offset per region (in/mid/out) from the raw
        audio with the same instrument that exposed the defect; the
        DIFFERENCE is the bias that puts kicks in register. Master plays
        its exit, slave its intro, so regions are out/in.

        History: a kick_offset_s bias was tried 2026-07-13 and REVERTED -
        that scalar was a folded whole-track energy profile dominated by
        BASS PLACEMENT (median 0.35-beat lies), not kick-vs-grid skew.
        The per-region attack-peak median is a different instrument:
        bounded +/-90ms search around each beat, IQR-gated, region-aware.
        DJ_KICK_ALIGN=0 disables for A/B listening."""
        if os.environ.get("DJ_KICK_ALIGN", "1") in ("0", "off"):
            return 0.0
        from lib.dj import beatpower as _bp
        if master.track_id is None or slave.track_id is None:
            return 0.0
        if not (getattr(master, "grid_is_db", True)
                and getattr(slave, "grid_is_db", True)):
            return 0.0       # offsets don't apply to a live-fixed grid
        # By POSITION, not role label: the lab and knob sweep force seams
        # mid-track, where the offset measured at the primary mix points
        # (often 100s away) does not apply - phase is a local property.
        off_m = _bp.phase_offset(master.track_id, region="out",
                                 at_s=master.source_time_s())
        off_s = _bp.phase_offset(slave.track_id, region="in",
                                 at_s=slave.source_time_s())
        if off_m is None or off_s is None:
            return 0.0
        pm = master.beat_period_s() or 0.5
        ps = slave.beat_period_s() or 0.5
        # Sign: phase_snap(master_phase + bias) parks the slave's GRID
        # bias beats EARLY of the master's. Slave kicks land off_s after
        # slave grid, master kicks off_m after master grid; kicks meet
        # when bias = off_s - off_m (slave-minus-master, in beats).
        return float(np.clip(off_s / ps - off_m / pm, -0.25, 0.25))

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
        # +/- a TENTH beat (2026-08-04): the quarter-beat window left
        # room to lock a consistently-late bass stab and feed the
        # integral a lie - measured as a -0.4% rate deficit and 5 ms/s
        # beat drift on material that passed every content screen. At a
        # tenth of a beat the correlator can only refine true kick
        # alignment; anything further off is a different instrument, not
        # a phase error.
        half = max(int(0.10 * beat_s * ENV_FPS), 2)
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
        # The PLL's target carries the kick-alignment bias (see
        # _sync_bias_beats, computed once at sync start): zero error =
        # the KICKS are in register, not the grids.
        bias = self._sync_bias
        grid_err = slave.beat_phase() - master.beat_phase() - bias
        grid_err = (grid_err + 0.5) % 1.0 - 0.5
        # If drift outruns the PLL while the slave is still fading in, snap
        # again (inaudible at low gain) rather than let it trainwreck.
        if abs(grid_err) > RESNAP_ERR and slave.gain < RESNAP_GAIN:
            # RUN-IN TEMPO CALIBRATION: each resnap IS a drift measurement
            # - grid_err beats accumulated since the last snap. A stored
            # bpm wrong beyond the trim cap (the measured half-beat flam
            # class: 17-38 resnaps, then audible double beats) shows up
            # here as a consistent drift RATE; fold it into the deck's
            # BASE rate while still inaudible, where a rate step is free.
            # By the time the deck is audible the tempo ratio is right and
            # the PLL only holds phase.
            now = self.clock
            last = self._cal_last
            if last is not None:
                dt = (now - last) / RATE
                # CLEAN EVIDENCE ONLY (2026-08-04). Measured on trusted-
                # grid pairs: the planned tempo ratios are RIGHT, yet
                # decks rendered 0.2-0.7% slow with 3-6 ms/s beat drift -
                # this block was rewriting base rate from short-dt snap
                # errors that were PHASE artifacts (kick-bias snap, ramp
                # settling, the silent-settle window), not drift. A real
                # tempo error needs time to express itself: demand >=3s
                # between snaps and cap the step at 0.5% - the half-beat
                # flam class this was built for is now excluded from
                # blends entirely by the grid-confidence wall, so gentle
                # correction suffices.
                if 3.0 <= dt <= 12.0:
                    corr = float(np.clip(-grid_err * beat_s / dt,
                                         -0.005, 0.005))
                    if abs(corr) > 0.0015 \
                            and abs(self._cal_applied + corr) <= 0.01:
                        slave.rate *= (1.0 + corr)
                        self._cal_applied += corr
                        self._stat_cals += 1
            self._cal_last = now
            slave.phase_snap(master.beat_phase() + bias)
            slave.rate_trim = 0.0
            self._apll_err = None
            self._apll_i = 0.0
            self._stat_resnaps += 1
            return
        # Audio-phase measurement 4x/second; PI control on it. The plant is
        # a delayed position measurement driven by a rate: the INTEGRAL term
        # learns the true tempo-ratio bias (why grids drift at all) so the
        # proportional term only handles small residuals - no limit cycle.
        if self.clock - self._apll_clock >= RATE // 4 \
                and cfg.get("audio", True):
            # The brain turns the audio path OFF for weak-kick pairs
            # (kick_agreement < 0.5): there the xcorr measures pattern
            # offset, passes its own stability gates (patterns persist),
            # and drags the deck off the kick-true grid+bias target.
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
                    -0.003, 0.003))
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
                blended = 0.7 * (-grid_err) + 0.3 * self._apll_err
                if (slave.gain >= RESNAP_GAIN and stable
                        and abs(blended) * beat_s > 0.012
                        and self.clock - getattr(self, "_nudge_clock", 0)
                        >= int(1.6 * RATE)):
                    nudge = float(np.clip(blended * beat_s * 0.7,
                                          -0.035, 0.035))
                    slave.stretch.phase_trim += nudge * RATE
                    self._nudge_clock = self.clock
                    self._stat_nudges += 1
                    self._apll_err = None    # stale until the ring refreshes
                    self._apll_hist = []
            else:
                self._apll_err = None
        _hist = getattr(self, "_apll_hist", [])
        _stable3 = (len(_hist) == 3 and max(_hist) - min(_hist) < 0.07
                    and (all(h > 0 for h in _hist)
                         or all(h < 0 for h in _hist)))
        if self._apll_err is not None and abs(self._apll_err) > 0.05 \
                and slave.gain < RESNAP_GAIN and _stable3:
            # Still quiet? A large audio error is corrected by an inaudible
            # JUMP (the PI slews ~12ms/s - a blend would end first). Same
            # stability bar as the audible nudge (2026-08-04): a SINGLE
            # xcorr reading on swung/organic material measures rhythm-
            # PATTERN offset, not kick flam, and one bad reading here was
            # jumping the deck ~75ms OFF the kick-true target during the
            # run-in (measured 83ms kick error on an 80bpm swing pair).
            slave.nudge_seconds(self._apll_err * beat_s)
            self._apll_err = None
            self._apll_hist = []
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
            elif self._mgain_ramp is not None:
                tgt, step = self._mgain_ramp
                g0 = self.mix_gain
                g1 = g0 + np.clip(tgt - g0, -step * m, step * m)
                out[pos:pos + m] *= np.linspace(g0, g1, m)[:, None]
                self.mix_gain = float(g1)
                if abs(g1 - tgt) < 1e-6:
                    self._mgain_ramp = None
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
        # Block RMS for the seam self-assessment (dead-air / hole detection).
        snap["rms"] = float(np.sqrt(np.mean(out ** 2))) if len(out) else 0.0
        self.telemetry = snap
        # Keep a short history so the VISUALS can read the state that is
        # being HEARD rather than the one being rendered. read() runs on the
        # render-ahead thread (lib/audio_engine.py), which works ahead of
        # the speakers by the ring depth; without this the club would beat-
        # flash and stage transition moves a beat early. Control/planning
        # code keeps using `telemetry` (render time) - events are scheduled
        # on the render clock and must stay there.
        hist = self._tel_hist + [(self.clock, snap)]
        drop = 0
        while len(hist) - drop > 2 and \
                self.clock - hist[drop][0] > TEL_HIST_FRAMES:
            drop += 1
        # Rebind, never mutate: readers on the main thread hold the old list
        # and keep iterating it safely. ~2s of history is a few dozen entries,
        # so the per-block copy is noise next to the mix itself.
        self._tel_hist = hist[drop:] if drop else hist
        return out

    def telemetry_delayed(self, frames):
        """Telemetry as of `frames` before the render head - i.e. what the
        room is hearing now. Falls back to the newest snapshot when the
        history does not reach that far back (startup, or no delay)."""
        hist = self._tel_hist       # grab once; read() rebinds, never mutates
        if frames <= 0 or not hist:
            return self.telemetry
        target = self.clock - frames
        # NEAREST snapshot, not the last one at-or-before `target`: history
        # is one entry per render block (~46ms), so "last before" is late by
        # up to a full block every time. Nearest turns that systematic lag
        # into a +/- half-block error centred on zero.
        best, best_err = None, None
        for clk, snap in hist:
            err = abs(clk - target)
            if best_err is None or err < best_err:
                best, best_err = snap, err
            elif clk > target:
                break                        # history is ordered; past it
        return best if best is not None else self.telemetry

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
                # DSP state the EQ numbers can't show - a stuck sweep
                # filter or echo is invisible without these (a "no bass"
                # night was undiagnosable from telemetry, 2026-07-13).
                "filter": getattr(d.filter, "mode", "off") or "off",
                "echo": bool(d.echo.active),
                "loop": d.loop,
                "braking": d._brake is not None,
            }
        sync = dict(self._sync) if self._sync else None
        if sync is not None:
            # Kick-alignment bias the PLL is holding: consumers measuring
            # seam lock from raw deck phases must subtract this, or an
            # intentionally offset grid reads as flam.
            sync["bias_beats"] = round(getattr(self, "_sync_bias", 0.0), 4)
        return {"clock": self.clock, "clock_s": round(self.clock / RATE, 3),
                "sync": sync,
                "sync_stats": {"resnaps": self._stat_resnaps,
                               "nudges": self._stat_nudges,
                               "cals": self._stat_cals,
                               "cal_applied": round(self._cal_applied, 4)},
                "mix_gain": round(self.mix_gain, 4), "decks": decks,
                "pending_events": len(self._auto)}
