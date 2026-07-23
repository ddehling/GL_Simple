"""DJSystem: the autonomous DJ's conductor.

Owns the library DB, the Brain, and the DJSubmix; runs the play state
machine on its own planner thread (live) or via explicit step() calls
(offline rendering / tests). All musical scheduling is keyed to the
submix's SAMPLE CLOCK, never wall time, which is exactly why offline
rendering through the hand-pumped engine behaves identically to a live
show.

State machine:
    IDLE -> PLAYING -> (next chosen + decoded) ARMED -> swap -> PLAYING ...

Public control surface (thread-safe, called from UI/main threads):
    set_theme(name)     request_skip()      set_autopilot(bool)
    set_energy_nudge(x) request_next(id)    status() -> dict
    outstate_keys() -> dj_arc_phase / dj_arc_heat / dj_next_drop_eta ...
"""
import json
import os
import threading
import time

import numpy as np

from lib.dj.brain import GLIDE_PER_S, Brain, load_library, TrackInfo
from lib.dj.db import LibraryDB
from lib.dj.rhythm import seam_chips
from lib.dj.submix import DJSubmix
from lib.dj.themes import BUILTIN_THEMES, get_theme

RATE = 44100
PLAN_LEAD_S = 60.0               # start choosing next this early. Must
                                 # exceed the LONGEST blend (96 beats at
                                 # 90 bpm = 64s worst case, ~46s typical):
                                 # build_events clamps the blend start to
                                 # 'now' (now_guard), so arming later than
                                 # the blend span SILENTLY SHORTENS it.
MIN_LEAD_S = 8.0                 # never arm closer than this to the seam
SET_CYCLE_S = 90 * 60.0          # non-all-night themes loop their arc here
WATCHDOG_S = 20.0                # continuity watchdog wakes this close to
                                 # the end of the current track unarmed
WD_LOOP_S = 15.0                 # ...and buys time with a safety loop here


def _intro_start(track):
    """Where a track should START playing from: the first beat of its intro
    (or just its first downbeat), never a deep mix-in point."""
    for s in track.sections:
        if s["kind"] == "intro":
            return max(s["start_s"], 0.0)
    return track.grid[0]["first_beat_s"] if track.grid else 0.0


class DJSystem:
    def __init__(self, music_root, engine=None, theme="groove",
                 night_hours=6.0, autopilot=True, seed=None,
                 stretch_max=1.08, log_dir=None, threaded=True,
                 record=False):
        self.music_root = music_root
        self.engine = engine
        self.night_hours = night_hours
        self.autopilot = autopilot
        self.threaded = threaded
        self._theme_name = theme
        self._seed = seed
        self._stretch_max = stretch_max
        repo = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        self.log_dir = log_dir or os.path.join(repo, "logs")

        self.submix = DJSubmix()
        self.state = "idle"
        self.active_deck = "a"
        self.current = None          # TrackInfo
        self.next_track = None
        self.plan = None
        self.swap_at = None          # submix clock
        self.blend_at = None
        self._started_clock = 0
        self._set_start_clock = 0
        self._history_id = None
        self._energy_nudge = 0.0
        self._urgent_exit = False        # skip/mix_now: exit ASAP, not "best"
        self._tag_vocab = []             # [(tag, count)] for flavor chips
        self._genre_tags = set()         # which vocab tags are genres
        self._arc_waypoints = []         # [(progress 0..1, energy 0..1)]
        self._horizon = []               # provisional next-N briefs
        self._horizon_key = None         # staleness stamp
        self._history = []               # tonight's tracklist briefs
        self._last_style = None          # last completed transition style
        self._played_energy_ema = None   # arc feedback: what actually played
        self._exit_played = 300.0    # drawn per track from theme min/max play
        self._next_meta = None
        self._setlist_name = None
        self._setlist_mode = "order"
        self._setlist_queue = []     # upcoming entry dicts (plan-following)
        self.setlist_names = []      # for the web picker, refreshed by step()
        self._setlists_checked = 0.0
        self._txn_id = 0             # transition transaction tag (abort)
        self._recovery_txn = None    # abort-recovery events still queued
        self._wd_loop = False        # continuity watchdog's safety loop
        self._seam_metrics = None    # collected while a transition runs
        self._last_seam = None       # last seam's measured quality (web)
        self._flam_pairs = set()     # (a_id, b_id) seams that flam-bailed:
                                     # retries become deliberate fades
        self._pulse = (0.0, 0.0)     # momentary energy tap: (value, t_set)
        self._decoded = {}           # track_id -> stereo samples (RAM cache)
        self._decoded_order = []
        self._decoding = set()
        self._decoded_stems = {}     # track_id -> {stem: float16 array}
        self._grid_fix = {}          # track_id -> verified-tempo correction
        self._decode_lock = threading.Lock()
        self._pending = []           # queued control requests
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self.db = None
        self.brain = None
        self.last_error = ""
        self._record = bool(record)
        self._rec_thread = None
        self.record_path = None

    # -- lifecycle -------------------------------------------------------------
    def start(self):
        """Open the library, attach the submix, begin conducting."""
        self.db = LibraryDB(self.music_root)     # planner-thread connection
        # 'do not use' tracks (DB v11) are never grabbed by the autoDJ.
        lib = [t for t in load_library(self.db) if not t.excluded]
        if not lib:
            self.last_error = "library is empty - run tools/dj_scan.py"
            print(f"[DJ] {self.last_error}")
            return False
        self.brain = Brain(lib, get_theme(self._theme_name), seed=self._seed,
                           stretch_max=self._stretch_max)
        self._by_id = {t.id: t for t in lib}
        # LIVE-ENERGY SCALE: drive x curve is compressed on a real
        # library (this one: raw spans ~0.22..0.8) - published as-is the
        # club's response was limp (user-heard). Map the library's OWN
        # p05..p95 of instantaneous energy to 0..1 so a chill breakdown
        # and a peak drop actually span the range. Uses DRIVE (rhythm/mood,
        # no loudness term), not energy_proxy: playback is loudness-
        # compensated, so mastering level must not dim the visuals.
        vals = []
        for t in lib:
            pr = t.drive()
            for c in (t.row.get("energy_curve") or [])[::4]:
                vals.append(pr * float(c))
        vals.sort()
        if len(vals) >= 100:
            self._energy_lo = vals[int(0.05 * (len(vals) - 1))]
            self._energy_hi = max(vals[int(0.95 * (len(vals) - 1))],
                                  self._energy_lo + 0.05)
        else:
            self._energy_lo, self._energy_hi = 0.0, 1.0
        # Remember what recently played ACROSS restarts - with a cold
        # recency memory every night opened with the same tracks in the
        # same order.
        try:
            by_id = {t.id: t for t in lib}
            for row in self.db.recent_plays(hours=12.0):
                t = by_id.get(row["track_id"])
                if t is not None:
                    self.brain.note_played(t, when=row["started_at"])
        except Exception as e:
            print(f"[DJ] recent-plays seed skipped: {e}")
        try:
            n = self.brain.load_pair_memory(self.db)
            if n:
                print(f"[DJ] pair memory: {n} remembered seams")
        except Exception as e:
            print(f"[DJ] pair memory skipped: {e}")
        self._refresh_setlist_names()
        self._refresh_tags()
        if self.engine is not None:
            self.engine.attach_track("dj_submix", self.submix)
        self._running = True
        self._set_start_clock = self.submix.clock
        if self._record:
            self._start_recording()
        if self.threaded:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="dj-brain")
            self._thread.start()
        print(f"[DJ] started: {len(lib)} playable tracks, "
              f"theme={self._theme_name}")
        return True

    def _refresh_tags(self):
        """Re-pull USER tags from the DB (the planner writes them while
        the show runs) and rebuild the flavor-chip vocabulary. User tags
        ALWAYS appear regardless of count - they are deliberate operator
        vocabulary, not a popularity contest; auto tags fill the rest."""
        if self.brain is None:
            return
        try:
            per_track = {}
            for r in self.db.conn.execute("SELECT track_id, tag FROM tags"):
                per_track.setdefault(r["track_id"], []).append(r["tag"])
            # A loaded setlist narrows the night to ITS songs - the
            # steering chips should speak that subset's vocabulary, not
            # the whole library's (tags absent from the list are dead
            # controls). Counts recount inside the list; falls back to
            # the full library when nothing is loaded.
            scope = None
            if self.brain.pool_ids is not None:
                scope = set(self.brain.pool_ids)
            elif self._setlist_queue:
                scope = {e["track_id"] for e in self._setlist_queue}
            if scope is not None and self.current is not None:
                scope.add(self.current.id)
            user_counts = {}
            auto_counts = {}
            genre_counts = {}
            for t in self.brain.library:
                t.user_tags = per_track.get(t.id, [])
                if scope is not None and t.id not in scope:
                    continue
                for tag in t.user_tags:
                    user_counts[tag] = user_counts.get(tag, 0) + 1
                for tag in t.auto_tags:
                    auto_counts[tag] = auto_counts.get(tag, 0) + 1
                # GENRE chips: MusicBrainz genres + embedded file genre. Both
                # already fold into all_tags, so selecting one drives the hard
                # filter / soft lean immediately.
                for g in (getattr(t, "genres", None) or []):
                    gl = str(g).lower()
                    genre_counts[gl] = genre_counts.get(gl, 0) + 1
                for part in (getattr(t, "file_genre", "") or "").replace(
                        "/", ",").replace(";", ",").split(","):
                    part = part.strip().lower()
                    if part:
                        genre_counts[part] = genre_counts.get(part, 0) + 1
            vocab = [(tag, n, True) for tag, n in
                     sorted(user_counts.items(), key=lambda kv: -kv[1])]
            genre_top = sorted(genre_counts.items(),
                               key=lambda kv: -kv[1])[:24]
            self._genre_tags = {g for g, _ in genre_top
                                if g not in user_counts}
            vocab += [(g, n, False) for g, n in genre_top
                      if g in self._genre_tags]
            vocab += [(tag, n, False) for tag, n in
                      sorted(auto_counts.items(), key=lambda kv: -kv[1])
                      if tag not in user_counts and tag not in self._genre_tags]
            self._tag_vocab = vocab[:80]     # everything, sane ceiling
        except Exception as e:
            print(f"[DJ] tag refresh skipped: {e}")

    def _start_recording(self):
        """Tap the submix into a timestamped WAV - every night becomes
        review material (pair with tools/_dj_quality_test metrics)."""
        import queue
        import wave
        os.makedirs(self.log_dir, exist_ok=True)
        self.record_path = os.path.join(
            self.log_dir,
            time.strftime("dj_night_%Y%m%d_%H%M.wav"))
        q = queue.Queue(maxsize=400)
        self.submix.record_q = q

        def _writer():
            import numpy as _np
            w = wave.open(self.record_path, "wb")
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            try:
                while self._running or not q.empty():
                    try:
                        blk = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    w.writeframes((_np.clip(blk, -1, 1)
                                   * 32767).astype(_np.int16).tobytes())
            finally:
                w.close()
        self._rec_thread = threading.Thread(target=_writer, daemon=True,
                                            name="dj-recorder")
        self._rec_thread.start()
        print(f"[DJ] recording night to {self.record_path}")

    def stop(self, fade_s=2.0):
        self._running = False
        self.submix.record_q = None
        self.submix.fade_out(fade_s)
        if self._history_id and self.db:
            try:
                self.db.log_play_end(self._history_id)
            except Exception:
                pass
        self._log({"event": "stop"})

    @property
    def active(self):
        return self._running

    def _run(self):
        while self._running:
            try:
                self.step()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self._error_t = time.time()
                import traceback
                traceback.print_exc()
                try:
                    with open(os.path.join(self.log_dir, "dj_error.log"),
                              "a") as f:
                        f.write(time.strftime("%Y-%m-%d %H:%M:%S ")
                                + traceback.format_exc() + chr(10))
                except OSError:
                    pass
            time.sleep(0.4)

    # -- control surface ---------------------------------------------------------
    def set_theme(self, name):
        with self._lock:
            self._pending.append(("theme", name))

    def request_skip(self):
        with self._lock:
            self._pending.append(("skip", None))

    def abort_transition(self):
        """TRAINWRECK RESCUE: recall an armed transition (only before its
        point of no return) and restore the playing deck - no skip, the
        current track just keeps going."""
        with self._lock:
            self._pending.append(("abort", None))

    def request_next(self, track_id):
        with self._lock:
            self._pending.append(("next_id", track_id))

    def set_autopilot(self, on):
        self.autopilot = bool(on)

    def set_energy_nudge(self, x):
        x = float(max(-0.4, min(0.4, x)))
        changed = abs(x - self._energy_nudge) > 0.049
        self._energy_nudge = x
        if changed:
            self._queue_steer_replan()

    def set_energy_pulse(self, x):
        """Momentary PUSH/COOL tap: +-x on the arc target from NOW,
        decaying linearly to zero over 15 min. Unlike the nudge slider
        (which stays where you left it all night), a tap is a reaction -
        it should wear off on its own."""
        self._pulse = (float(max(-0.4, min(0.4, x))), time.time())
        self._queue_steer_replan()

    def _queue_steer_replan(self):
        """Energy steering must be FELT: the next pick is locked in
        minutes before the seam (early decode), so without this a
        nudge/tap/arc-edit only steers the pick AFTER next - the controls
        read as dead. Queue a soft replan so the very next transition
        obeys the new target."""
        with self._lock:
            self._pending.append(("steer_replan", None))

    def energy_pulse(self):
        """The tap's current (decayed) contribution."""
        val, t0 = self._pulse
        if not val:
            return 0.0
        left = 1.0 - (time.time() - t0) / 900.0
        return val * left if left > 0.0 else 0.0

    def set_arc_waypoints(self, pts):
        """Steer the night's arc live: [(progress 0..1, energy 0..1)]
        interpolated over the theme's curve; empty list = back to theme."""
        with self._lock:
            self._pending.append(("arc", list(pts or [])))

    def hold(self):
        """Push the planned exit one phrase later (crowd's loving it)."""
        with self._lock:
            self._pending.append(("hold", None))

    def reroll_next(self):
        """Veto the provisional next track and pick a different one."""
        with self._lock:
            self._pending.append(("reroll", None))

    def seam_feedback(self, up):
        """Thumbs on the last transition - tonight's style weights learn."""
        with self._lock:
            self._pending.append(("seam_fb", bool(up)))

    def moment(self):
        """OPERATOR MOMENT: build a riser into the next phrase boundary
        and land an impact ON it - a crowd moment on demand, phrase-tight.
        The visuals pre-arm through the published ETA and hear the impact
        as a drop."""
        with self._lock:
            self._pending.append(("moment", None))

    def set_flavor(self, flavor):
        """Live music-type steering: {'prefer_tags': {tag: w}, 'avoid_tags':
        {tag: w}, 'axis_targets': {axis: 0..1}} merged over the theme."""
        with self._lock:
            self._pending.append(("flavor", dict(flavor or {})))

    def load_setlist(self, name, mode="order"):
        """Load a setlist. mode='order': play the entries top to bottom.
        mode='pool': the list is the POOL - the brain steers the order
        live (arc / flavor / nudge apply), each track plays once."""
        with self._lock:
            self._pending.append(("setlist", (name, mode)))

    def seek(self, pos_s):
        """Jump the current track to an absolute position (testing)."""
        with self._lock:
            self._pending.append(("seek", float(pos_s)))

    def seek_relative(self, delta_s):
        with self._lock:
            self._pending.append(("seek", ("rel", float(delta_s))))

    def to_exit(self):
        """Jump to ~20s before this track's exit so the next transition
        happens right away - the fast way to audition a mix."""
        with self._lock:
            self._pending.append(("seek", ("exit", None)))

    def mix_now(self):
        """Arm and run the planned transition immediately (test the mix)."""
        with self._lock:
            self._pending.append(("mix_now", None))

    def bpm_target(self):
        """The night's PLANNED BPM journey: the tempo rides the same shape
        as the energy arc across the theme's range - sets FEEL like they go
        somewhere instead of hovering."""
        theme = self.brain.theme if self.brain else get_theme(self._theme_name)
        lo, hi = theme.bpm_range
        return lo + (hi - lo) * theme.arc_target(self.arc_progress())

    def _note_energy(self, track):
        e = track.energy_proxy()
        self._played_energy_ema = e if self._played_energy_ema is None \
            else 0.6 * self._played_energy_ema + 0.4 * e

    # -- arc / outstate -----------------------------------------------------------
    def arc_progress(self):
        elapsed = (self.submix.clock - self._set_start_clock) / RATE
        theme = self.brain.theme if self.brain else get_theme(self._theme_name)
        if theme.arc == "all_night":
            return min(1.0, elapsed / max(self.night_hours * 3600.0, 60.0))
        return (elapsed % SET_CYCLE_S) / SET_CYCLE_S

    def _arc_base(self, progress):
        """Theme arc, overridden by live waypoints when the operator has
        drawn their own curve."""
        theme = self.brain.theme if self.brain else get_theme(self._theme_name)
        if self._arc_waypoints:
            xs = [p for p, _ in self._arc_waypoints]
            ys = [e for _, e in self._arc_waypoints]
            if progress <= xs[0]:
                return ys[0]
            if progress >= xs[-1]:
                return ys[-1]
            for i in range(len(xs) - 1):
                if xs[i] <= progress <= xs[i + 1]:
                    f = (progress - xs[i]) / max(xs[i + 1] - xs[i], 1e-6)
                    return ys[i] + f * (ys[i + 1] - ys[i])
        return theme.arc_target(progress)

    def arc_target(self):
        target = self._arc_base(self.arc_progress()) + self._energy_nudge \
            + self.energy_pulse()
        # ARC FEEDBACK: if what actually PLAYED has been running hotter or
        # cooler than the theme's arc (library gaps, anchor picks), lean
        # the next choice the other way instead of undershooting all night.
        if self._played_energy_ema is not None:
            target += max(-0.15, min(0.15,
                                     0.6 * (target - self._played_energy_ema)))
        return max(0.0, min(1.0, target))

    def live_energy(self):
        """Ground-truth energy of what's playing RIGHT NOW, 0..1.

        The DSP estimate downstream (audio_signals) is built on AGC bands
        that hover near a constant during any steady music, so the club
        read every track as 'medium'. The DJ doesn't have to guess: each
        track's cross-track DRIVE (rhythm/mood rank - not energy_proxy;
        playback is loudness-compensated so mastering level must not dim
        the room) shaped by its own 2 Hz energy curve at the playhead,
        gain-weighted across live decks, IS the floor's energy -
        breakdowns dip, drops slam, chill is chill. Returns None when
        nothing useful is playing (caller falls back to the measured
        signal).
        """
        tel = self.submix.telemetry or {}
        by_id = getattr(self, "_by_id", None) or {}
        num = den = 0.0
        for d in (tel.get("decks") or {}).values():
            g = float(d.get("gain") or 0.0)
            if not d.get("playing") or g <= 0.02:
                continue
            t = by_id.get(d.get("track_id"))
            if t is None:
                continue
            shape = 1.0
            curve = t.row.get("energy_curve") or []
            if curve:
                i = float(d.get("time_s") or 0.0) * 2.0
                i0 = max(0, min(int(i), len(curve) - 1))
                i1 = min(i0 + 1, len(curve) - 1)
                f = i - i0
                shape = float(curve[i0]) * (1.0 - f) + float(curve[i1]) * f
            num += g * t.drive() * shape
            den += g
        if den <= 0.0:
            return None
        lo = getattr(self, "_energy_lo", 0.0)
        hi = getattr(self, "_energy_hi", 1.0)
        # Floor: percentile expansion sent soft stretches to 0.0 and the
        # room went DARK (user). Playing music is never less than a low
        # simmer.
        return max(0.12, min(1.0, (num / den - lo) / max(hi - lo, 1e-6)))

    def live_beat(self):
        """Ground-truth beat state of the audible deck, computed from the
        stored grid at the live playhead - sample-tight where the DSP
        detector on the mix is laggy/quantized. Cheap enough to call per
        render frame. None when nothing usable is playing."""
        tel = self.submix.telemetry or {}
        d = (tel.get("decks") or {}).get(self.active_deck)
        t = self.current
        if not d or t is None or not d.get("playing") or not d.get("ready"):
            return None
        # TRUST GATE: the analyzer emits SOME grid for any audio -
        # autocorrelation on a beatless Zimmer drone found 111 bpm at
        # conf 0.04, and publishing it as ground truth made the club
        # pulse on beats that don't exist (user-heard on 'Mesa').
        # Same bar the mixing brain uses for beat-matching.
        if (t.bpm_conf or 0.0) < 0.5:
            return None
        pos = float(d.get("time_s") or 0.0)
        rate = max(float(d.get("rate") or 1.0), 1e-6)
        gseg = None
        for g in (t.grid or []):
            if pos >= float(g.get("start_s", 0.0)):
                gseg = g
        per = float((gseg or {}).get("period_s") or t.period_s or 0.0)
        if per <= 0.0:
            return None
        fb = float((gseg or {}).get("first_beat_s") or 0.0)
        idx = (pos - fb) / per
        off = int(t.row.get("downbeat_offset") or 0)
        pb = int(t.phrase_beats or 32)
        sec = t.section_at(pos) or {}
        # DRIVE: how much rhythm the playing SECTION actually has - the
        # grid keeps ticking through a breakdown, but the room must not
        # pulse on resting kicks. (Section density ~6-9 in grooves on
        # this analyzer's scale, ~0 in breakdowns.)
        drive = max(0.0, min(1.0, float(
            sec.get("rhythm_density") or 0.0) / 4.0))
        return {"bpm": 60.0 * rate / per,
                "phase": float(d.get("beat_phase") or 0.0),
                "bar_phase": ((idx - off) % 4.0) / 4.0,
                "phrase_phase": ((idx - off) % pb) / float(pb),
                "bass_share": float(sec.get("bass_share") or 0.33),
                "drive": drive}

    def outstate_keys(self):
        """Published into outstate each tick - the visuals' coupling."""
        eta = None
        if self.blend_at is not None and self.submix.clock < self.blend_at:
            eta = (self.blend_at - self.submix.clock) / RATE
        # TRANSITION CHOREOGRAPHY: the visuals know the future. blend_eta
        # counts down to the overlap starting; swap_eta to the decisive
        # bass/melody handover - the club director stages moves on these
        # exact beats (no human DJ + VJ pair can do this).
        swap_eta = None
        if self.state == "armed" and self.swap_at is not None \
                and self.submix.clock < self.swap_at:
            swap_eta = (self.swap_at - self.submix.clock) / RATE
        # A pending operator MOMENT pre-arms the visuals the same way an
        # approaching seam does.
        m_clk = getattr(self, "_moment_clock", 0)
        if m_clk > self.submix.clock:
            m_eta = (m_clk - self.submix.clock) / RATE
            eta = m_eta if eta is None else min(eta, m_eta)
        # GROUND-TRUTH MUSICAL DROPS: the DB knows every drop section of
        # every track. The DSP drop detector needs a QUIET episode to arm
        # (by design, so fades can't fake drops) - a relentless hard set
        # never gives it one, so the club barely slammed all night
        # (user-heard). Publish the next drop's ETA for visual pre-arm
        # and stamp the moment the playhead crosses one.
        drop_eta = None
        pos = self._pos_s()
        if self.current is not None and pos is not None:
            cid, moments = getattr(self, "_drops_cache", (None, []))
            if cid != self.current.id:
                from lib.dj.features import drop_moments
                moments = drop_moments(self.current.sections)
                self._drops_cache = (self.current.id, moments)
            prev_id, prev_pos = getattr(self, "_drop_scan_prev", (None, None))
            for st in moments:
                d = st - pos
                if 0.0 < d < 20.0 and (drop_eta is None or d < drop_eta):
                    drop_eta = d
                if (prev_id == self.current.id and prev_pos is not None
                        and prev_pos < st <= pos
                        and pos - prev_pos < 2.0):     # not a seek jump
                    self._dj_drop_wall = time.time()
            self._drop_scan_prev = (self.current.id, pos)
        ndrop = eta
        if drop_eta is not None:
            ndrop = drop_eta if ndrop is None else min(ndrop, drop_eta)
        return {"dj_active": self._running,
                "dj_arc_phase": self.arc_progress(),
                "dj_arc_heat": self.arc_target(),
                "dj_energy": self.live_energy(),
                "dj_drop_t": getattr(self, "_dj_drop_wall", None),
                "dj_next_drop_eta": ndrop,
                "dj_blend_eta": eta,
                "dj_swap_eta": swap_eta,
                "dj_style": self.plan["style"] if self.plan else None}

    def status(self):
        tel = self.submix.telemetry or {}
        cur = self.current
        nxt = self.next_track
        countdown = None
        if self.blend_at is not None:
            countdown = max(0.0, (self.blend_at - self.submix.clock) / RATE)
        return {
            "state": self.state, "theme": self._theme_name,
            "themes": sorted(BUILTIN_THEMES),
            "flavor": dict(self.brain.flavor) if self.brain else {},
            "require_tags": (sorted(self.brain.require_tags)
                             if self.brain else []),
            "eligible_pool": (self.brain.eligible_pool_size()
                              if self.brain else 0),
            "reachable_now": (getattr(self.brain, "last_scored_n", None)
                              if self.brain else None),
            "tags": self._tag_vocab,
            "genre_tags": sorted(self._genre_tags),
            "horizon": list(self._horizon),
            "history": self._history[-40:],
            "arc_waypoints": list(self._arc_waypoints),
            "arc_cycle_s": (self.night_hours * 3600.0
                            if (self.brain and
                                self.brain.theme.arc == "all_night")
                            else SET_CYCLE_S),
            "arc_curve": [round(max(0.0, min(1.0, self._arc_base(i / 24.0))),
                          3) for i in range(25)],
            "track_map": self._track_map(),
            "next_map": self._track_map(
                self.next_track,
                entry=self.plan["in_s"] if self.plan else None)
            if self.next_track is not None else None,
            "level": round(float(tel.get("peak", 0.0)), 3),
            "moment_eta": round((self._moment_clock - self.submix.clock)
                                / RATE, 1)
            if getattr(self, "_moment_clock", 0) > self.submix.clock
            else None,
            "autopilot": self.autopilot,
            "arc_phase": round(self.arc_progress(), 4),
            "arc_heat": round(self.arc_target(), 3),
            "energy_nudge": self._energy_nudge,
            "energy_pulse": round(self.energy_pulse(), 2),
            "current": self._track_brief(cur),
            "next": self._track_brief(nxt),
            "style": self.plan["style"] if self.plan else None,
            # Word-first groove chips for the armed seam ("kick clash",
            # "swung vs straight", "half-time"...) - the operator's WHY
            # before it happens, and an informed ABORT MIX.
            "seam_chips": (seam_chips(self.plan,
                                      {"rhythm": self.plan.get("rhythm"),
                                       "mult": (self.plan.get("rhythm")
                                                or {}).get("mult")})
                           if self.plan else []),
            "last_seam": self._last_seam,
            "abortable": (self.state == "armed" and self.plan is not None
                          and self.plan.get("no_return_at") is not None
                          and self.submix.clock
                          < self.plan["no_return_at"] - int(0.5 * RATE)),
            "blend_in_s": round(countdown, 1) if countdown is not None else None,
            "setlist": self._setlist_name,
            "setlist_remaining": (len(self.brain.pool_ids)
                                  if self.brain is not None
                                  and self.brain.pool_ids is not None
                                  else len(self._setlist_queue)),
            "setlist_mode": self._setlist_mode,
            "setlists": list(self.setlist_names),
            "night_hours": self.night_hours,
            "decks": self._deck_brief(tel),
            "deck_telemetry": tel,
            # A transient step error shouldn't scare the banner forever.
            "error": self.last_error
            if time.time() - getattr(self, "_error_t", 0) < 60.0 else "",
        }

    def _deck_brief(self, tel):
        """Compact per-deck view for the web page (what each deck plays)."""
        cur_id = self.current.id if self.current else None
        nxt_id = self.next_track.id if self.next_track else None
        names, bpms = {}, {}
        if self.current:
            names[cur_id] = self.current.title
            bpms[cur_id] = self.current.bpm
        if self.next_track:
            names[nxt_id] = self.next_track.title
            bpms[nxt_id] = self.next_track.bpm
        sync = tel.get("sync") or {}
        out = []
        for name, d in (tel.get("decks") or {}).items():
            if not d.get("playing"):
                continue
            rate = d.get("rate", 1.0)
            nat = bpms.get(d.get("track_id"))
            out.append({
                "deck": name.upper(),
                "track_id": d.get("track_id"),
                "time_s": d.get("time_s"),
                "title": names.get(d.get("track_id"), "?"),
                "gain": round(d.get("gain", 0.0), 2),
                # THE BEAT-MATCH EVIDENCE: what tempo this deck actually
                # plays at right now, and how far its rate is bent to
                # match the other deck (0.0% = riding natural).
                "bpm": round(nat * rate, 1) if nat else None,
                "rate_pct": round((rate - 1.0) * 100.0, 2),
                "synced": name == sync.get("slave"),
                "eq": d.get("eq"),
                # Live DSP state - a stuck sweep filter/echo is otherwise
                # invisible on the panel (undiagnosable "no bass" night).
                "filter": d.get("filter", "off"),
                "echo": bool(d.get("echo")),
                "beat_phase": d.get("beat_phase"),
            })
        return out

    def _track_map(self, t=None, entry=None):
        """Compact track geography for the web context strips."""
        t = t if t is not None else self.current
        if t is None:
            return None
        secs = [[round(x["start_s"], 1), round(x["end_s"], 1), x["kind"],
                 round(x.get("vocalness") or 0.0, 2)]
                for x in t.sections][:40]
        curve = t.row.get("energy_curve") or []
        if curve:
            idx = [int(i * (len(curve) - 1) / 23) for i in range(24)]
            curve = [round(float(curve[i]), 2) for i in idx]
        return {"duration": round(t.duration_s, 1), "sections": secs,
                "energy": curve,
                "entry_s": round(entry, 1) if entry is not None else None,
                "exit_s": round(self.plan["out_s"], 1)
                if (self.plan and t is self.current) else None}

    def _track_brief(self, t):
        if t is None:
            return None
        pos = None
        tel = self.submix.telemetry or {}
        if self.current is t and tel:
            d = tel.get("decks", {}).get(self.active_deck)
            if d:
                pos = d["time_s"]
        return {"id": t.id, "title": t.title, "artist": t.artist,
                "bpm": t.bpm, "camelot": t.camelot,
                "duration_s": t.duration_s, "pos_s": pos}

    # -- state machine --------------------------------------------------------------
    def step(self):
        if not self._running or self.brain is None:
            return
        requests = None
        with self._lock:
            requests, self._pending = self._pending, []
        for kind, val in requests:
            if kind == "theme":
                self.brain.set_theme(get_theme(val))
                self._theme_name = val
                self._log({"event": "theme", "theme": val})
                if self.state == "playing":
                    self.next_track = None       # soft replan
                    self.plan = None
            elif kind == "flavor":
                # Live music-type steering: soft tag leans + axis pulls, PLUS a
                # HARD tag filter (require_tags) - only tracks carrying a
                # required tag may play. Soft replan so the very next pick
                # obeys; a changed hard filter also drops the planned horizon
                # (it may hold now-ineligible picks).
                old_req = set(self.brain.require_tags)
                self.brain.set_flavor(val)
                self.brain.set_require_tags(
                    val.get("require_tags") if isinstance(val, dict) else None)
                self._log({"event": "flavor", "flavor": val})
                if self.state == "playing":
                    self.next_track = None
                    self.plan = None
                    if self.brain.require_tags != old_req:
                        self._horizon = []
            elif kind == "skip" and self.state in ("playing", "armed"):
                self._do_skip()
            elif kind == "abort" and self.state == "armed":
                self._do_abort(via="abort")
            elif kind == "next_id":
                t = next((x for x in self.brain.library if x.id == val), None)
                if t is not None and self.state == "playing":
                    self.next_track = t
                    self._next_requested = True
                    self.plan = None
                    self._log({"event": "pick_next", "track": t.title})
            elif kind == "setlist":
                from lib.dj.setlist import get_setlist
                name, mode = val if isinstance(val, tuple) else (val, "order")
                sl = get_setlist(self.db, name=name) if name else None
                if name and sl is None:
                    self.last_error = f"setlist '{name}' not found"
                else:
                    self._setlist_name = name or None
                    self._setlist_mode = mode
                    self.brain.pool_ids = None
                    self._setlist_queue = []
                    if sl and mode == "pool":
                        # THE LIST AS A POOL: brain picks the order, all
                        # steering applies, nothing outside the list plays.
                        self.brain.pool_ids = {
                            e["track_id"] for e in sl["entries"]}
                        if self.current is not None:
                            self.brain.pool_ids.discard(self.current.id)
                        self._horizon = []          # replan inside the pool
                    elif sl:
                        self._setlist_queue = list(sl["entries"])
                    self._log({"event": "setlist",
                               "name": self._setlist_name, "mode": mode,
                               "tracks": len(sl["entries"]) if sl else 0})
                    self._refresh_tags()
                    if self.state == "playing":
                        self.next_track = None      # replan from the list
                        self.plan = None
            elif kind == "arc":
                self._arc_waypoints = [(max(0.0, min(1.0, float(p))),
                                        max(0.0, min(1.0, float(e))))
                                       for p, e in val][:8]
                self._arc_waypoints.sort()
                self._log({"event": "arc", "waypoints": self._arc_waypoints})
                self._horizon_key = None
                # A drawn curve is steering too - next pick must obey it.
                if self.state == "playing" \
                        and not getattr(self, "_next_requested", False):
                    self.next_track = None
                    self.plan = None
            elif kind == "steer_replan" and self.state == "playing":
                # Nudge/tap changed the energy target: drop the locked-in
                # next pick (unless the user chose it by hand) so the very
                # next transition follows the new target. Armed transitions
                # are left alone - too late to re-aim those.
                if not getattr(self, "_next_requested", False):
                    self.next_track = None
                    self.plan = None
                    self._horizon = []
                    self._horizon_key = None
                    self._log({"event": "steer_replan"})
            elif kind == "hold" and self.state == "playing"                     and self.current is not None:
                bump = (self.current.phrase_beats or 32)                     * self.current.period_s
                self._exit_played += bump
                self.plan = None
                self._log({"event": "hold", "plus_s": round(bump, 1)})
            elif kind == "reroll" and self.state == "playing":
                if self.next_track is not None:
                    self.brain.veto_ids.add(self.next_track.id)
                    self._log({"event": "reroll",
                               "vetoed": self.next_track.title})
                self.next_track = None
                self.plan = None
                self._horizon = self._horizon[1:]   # vetoed item leaves
                self._horizon_key = None
            elif kind == "moment" and self.state in ("playing", "armed")                     and self.current is not None:
                pos = self._pos_s()
                tel_d = (self.submix.telemetry or {}).get("decks", {})
                rate = (tel_d.get(self.active_deck) or {}).get("rate", 1.0)
                if pos is not None:
                    t_hit = self.current.nearest_phrase(
                        pos + 4 * self.current.period_s)
                    while t_hit < pos + 2.5:
                        t_hit += (self.current.phrase_beats or 32)                             * self.current.period_s
                    eta = (t_hit - pos) / max(rate, 1e-6)
                    hit = self.submix.clock + int(eta * RATE)
                    from lib.dj import fx as _fx
                    rise = min(eta - 0.1, 8.0)
                    if rise >= 1.0:
                        self.submix.post({"at": hit - int(rise * RATE),
                                          "cmd": "fx_play",
                                          "samples": _fx.make_riser(
                                              rise, gain=0.16)})
                    self.submix.post({"at": hit, "cmd": "fx_play",
                                      "samples": _fx.make_impact(gain=0.26)})
                    self._moment_clock = hit
                    self._log({"event": "moment", "in_s": round(eta, 1)})
            elif kind == "seam_fb":
                if self._last_style:
                    self.brain.seam_feedback(self._last_style, val)
                    pair = getattr(self, "_last_pair", None)
                    if pair and pair[0] is not None:
                        try:
                            self.db.add_seam_feedback(pair[0], pair[1],
                                                      self._last_style, val)
                        except Exception as e:
                            print(f"[DJ] seam fb store failed: {e}")
                    self._log({"event": "seam_fb", "up": val,
                               "style": self._last_style})
            elif kind == "seek":
                self._do_seek(val)
            elif kind == "mix_now":
                if self.state == "playing":
                    self._do_skip()                 # force the transition now

        if time.time() - self._setlists_checked > 10.0:
            self._refresh_setlist_names()
            self._refresh_tags()

        if self.state == "idle":
            self._start_first()
        elif self.state == "playing":
            self._maybe_plan()
            self._maybe_horizon()
        elif self.state == "armed":
            self._collect_seam_metrics()
            if self.swap_at is not None and self.submix.clock >= self.swap_at:
                self._finish_swap()
        if self.state == "playing":
            self._watchdog(self._pos_s())

    def _pick_next(self, out_bpm):
        """The live pick FOLLOWS the displayed queue: take the horizon's
        front if it still scores, so what the operator sees is what
        plays. Fresh selection only when the queue is empty/invalid."""
        if self._setlist_queue:
            t, meta = self._pop_setlist_next(out_bpm)
            if t is not None:
                return t, meta
        if self._horizon and self.brain is not None:
            t0 = next((t for t in self.brain.library
                       if t.id == self._horizon[0]["id"]), None)
            # A stale queue front must never resurrect a track the night
            # just played - freshness is DISTINCT-SONG based, same model as
            # the brain's hard no-repeat window.
            ck0 = self.brain.ckey.get(t0.id, t0.id) if t0 is not None else None
            ds0 = (self.brain._distinct_since_map().get(ck0)
                   if ck0 is not None else None)
            fresh = t0 is not None and (ds0 is None
                                        or ds0 >= self.brain.norepeat_n)
            if fresh and t0.id != self.current.id                     and t0.id not in self.brain.veto_ids:
                s, meta = self.brain.score(
                    self.current, t0, self.arc_target(), out_bpm,
                    bpm_target=self.bpm_target())
                if s > 0 and meta is not None:
                    return t0, meta
        return self.brain.choose_next(
            self.current, self.arc_target(), out_bpm,
            bpm_target=self.bpm_target())

    def _maybe_horizon(self):
        """Provisional next-N for the trajectory display; recomputed only
        when its inputs changed (it runs real selection, not free)."""
        if self.current is None or self.brain is None:
            return
        steer = (self._theme_name,
                 json.dumps(self.brain.flavor, sort_keys=True),
                 tuple(sorted(self.brain.require_tags)),   # genre/tag HARD filter
                 tuple(self._arc_waypoints), round(self._energy_nudge, 2))
        steered = steer != self._horizon_key
        if not steered and len(self._horizon) >= 3                 and (self.next_track is None
                     or self._horizon[0]["id"] == self.next_track.id):
            return
        if steered:
            self._horizon = []           # steering changed: replan the lot
        self._horizon_key = steer
        if self._setlist_queue:
            # A loaded setlist IS the plan - show it, don't free-plan.
            by_id = {t.id: t for t in self.brain.library}
            items = []
            if self.next_track is not None:
                items.append({
                    "id": self.next_track.id, "title": self.next_track.title,
                    "artist": self.next_track.artist,
                    "bpm": self.next_track.bpm,
                    "energy": self.next_track.energy_proxy(),
                    "tags": self.next_track.all_tags[:4],
                    "why": "setlist next"})
            for e in self._setlist_queue[:3]:
                t = by_id.get(e["track_id"])
                if t is None:
                    continue
                items.append({
                    "id": t.id, "title": t.title, "artist": t.artist,
                    "bpm": t.bpm, "energy": t.energy_proxy(),
                    "tags": t.all_tags[:4],
                    "why": "setlist " + ("anchor" if e.get("pin_type")
                                         == "anchor" else "pick")})
            self._annotate_horizon(items[:3])
            self._horizon = items[:3]
            return
        prog0 = self.arc_progress()
        step = 300.0 / (SET_CYCLE_S if (self.brain.theme.arc != "all_night")
                        else max(self.night_hours * 3600.0, 60.0))

        def arc_at(i):
            return max(0.0, min(1.0, self._arc_base(
                min(prog0 + i * step, 1.0)) + self._energy_nudge))
        try:
            by_id = {t.id: t for t in self.brain.library}
            ck = self.brain.ckey
            kept, seen = [], {ck.get(self.current.id, self.current.id)}
            for h in self._horizon:
                k = ck.get(h["id"], h["id"])
                if h["id"] in by_id and k not in seen:
                    seen.add(k)
                    kept.append(h)
            kept = kept[:3]
            need = 3 - len(kept)
            if need > 0:
                tail = by_id[kept[-1]["id"]] if kept else self.current
                pre = [by_id[h["id"]] for h in kept]
                kept += self.brain.plan_horizon(
                    tail, arc_at, tail.bpm, n=need, preplayed=pre)
            if self.brain.pool_ids is not None and len(kept) < 3:
                # From an off-pool (or tempo-remote) current track the
                # planner may reach nothing - but the pool WILL play,
                # via the dipped-fade fallback. Show it.
                have = {h["id"] for h in kept}
                arc = self.arc_target()
                rest = sorted((t for t in self.brain.library
                               if t.id in self.brain.pool_ids
                               and t.id not in have),
                              key=lambda t: abs(t.energy_proxy() - arc))
                for t in rest[:3 - len(kept)]:
                    kept.append({"id": t.id, "title": t.title,
                                 "artist": t.artist, "bpm": t.bpm,
                                 "energy": t.energy_proxy(),
                                 "tags": t.all_tags[:4],
                                 "why": "setlist pool (fade in)"})
            self._horizon = kept
            self._annotate_horizon(self._horizon)
            if self.next_track is not None and self._horizon:
                # the committed pick leads the horizon
                self._horizon[0] = {"id": self.next_track.id,
                                    "title": self.next_track.title,
                                    "artist": self.next_track.artist,
                                    "bpm": round(self.next_track.bpm, 1),
                                    "energy": round(
                                        self.next_track.energy_proxy(), 2),
                                    "tags": self.next_track.all_tags[:4],
                                    "why": self.brain.explain_pick(
                                        self.current, self.next_track,
                                        self._next_meta)}
                # The same song must not appear twice in the triplet: the
                # committed pick may ALSO be sitting at slot 1/2 from an
                # earlier preview (reroll / fresh pick paths) - evict it.
                nk = self.brain.ckey.get(self.next_track.id,
                                         self.next_track.id)
                self._horizon = [self._horizon[0]] + [
                    h for h in self._horizon[1:]
                    if self.brain.ckey.get(h["id"], h["id"]) != nk]
        except Exception as e:
            print(f"[DJ] horizon skipped: {e}")

    def _start_first(self):
        first = None
        while self._setlist_queue and first is None:
            entry = self._setlist_queue.pop(0)
            first = next((x for x in self.brain.library
                          if x.id == entry["track_id"]), None)
            if first is not None:
                self._play_hint_s = entry.get("target_play_s")
        if first is None:
            first = self.brain.choose_first(self.arc_target())
        if first is None:
            self.last_error = "no track fits the theme"
            return
        samples = self._decode(first)
        if samples is None:
            self.brain.library.remove(first)
            return
        # Start the FIRST track from its beginning (first downbeat), not a
        # deep mix-in point - otherwise the set opens mid-track, skipping a
        # minute of music.
        cue = first.nearest_downbeat(_intro_start(first))
        self.submix.post_many([
            {"cmd": "load", "deck": self.active_deck, "samples": samples,
             "track_id": first.id, "grid": first.grid,
             "gain_db": first.gain_db,
             "kick_offset_s": first.kick_offset_s, "cue_s": cue},
            {"cmd": "gain", "deck": self.active_deck, "value": 1.0,
             "ramp_s": 1.5},
            {"cmd": "start", "deck": self.active_deck},
        ])
        self.current = first
        self.state = "playing"
        self._started_clock = self.submix.clock
        self._draw_exit()
        self.brain.note_played(first)
        self._note_pool_played(first)
        self._note_energy(first)
        self._history.append({"t": time.strftime("%H:%M"),
                              "title": first.title, "artist": first.artist,
                              "via": "start"})
        self._history_id = self.db.log_play_start(first.id, theme=self._theme_name)
        self._log({"event": "play", "track": first.title,
                   "artist": first.artist, "bpm": first.bpm,
                   "camelot": first.camelot})

    def _pos_s(self):
        tel = self.submix.telemetry or {}
        d = tel.get("decks", {}).get(self.active_deck)
        return d["time_s"] if d and d["ready"] else None

    def _maybe_plan(self):
        if self._wd_loop:
            # The continuity watchdog's safety loop is wrapping the deck's
            # cursor - build_events' source->clock mapping is invalid there.
            # The watchdog owns the endgame (clock-domain handoff).
            return
        if not self.autopilot and self.next_track is None:
            return
        pos = self._pos_s()
        if pos is None or self.current is None:
            return
        played = (self.submix.clock - self._started_clock) / RATE
        deadline = self.current.duration_s - 25.0
        # The tempo the incoming track must MATCH is what's actually
        # PLAYING, not the track's natural bpm: after a short play (skip,
        # mix_now, tight setlist) the deck is still mid-glide from the
        # last transition, and matching against natural bpm leaves a
        # 1-3% tempo error the PLL can't absorb - beats audibly slip
        # through the whole next blend.
        tel = (self.submix.telemetry or {}).get("decks", {})
        live_rate = (tel.get(self.active_deck) or {}).get("rate", 1.0)
        # Verified tempo when we have one: the stored bpm of the PLAYING
        # track may be the wrong side of a flam pair too.
        out_bpm = self._true_bpm(self.current) * live_rate

        # EARLY: choose the next track and kick off its background decode as
        # soon as we're settled into the current one, so the decode (a ~0.5s
        # CPU burst that can starve the audio callback) finishes MINUTES
        # before the switch, never near it. The pick is locked in early -
        # exactly what a real DJ does.
        if self.next_track is None and (self.autopilot or self._setlist_queue) \
                and played > 20.0:
            if self._setlist_queue:
                self.next_track, self._next_meta = \
                    self._pop_setlist_next(out_bpm)
            if self.next_track is None:
                cand, meta = self._pick_next(out_bpm)
                if cand is not None:
                    self.next_track, self._next_meta = cand, meta
            if self.next_track is not None:
                self._predecode(self.next_track)

        # _exit_played was drawn from [min_play, max_play] when this track
        # took over; ARM once we're a lead-time away from it (or from the
        # track simply running dry).
        if played < self._exit_played - PLAN_LEAD_S \
                and pos < deadline - PLAN_LEAD_S:
            return
        if self.next_track is None and self._setlist_queue:
            self.next_track, self._next_meta = self._pop_setlist_next(out_bpm)
        if self.next_track is None:
            cand, meta = self._pick_next(out_bpm)
            if cand is None and self.brain.pool_ids is not None:
                # No pool track can FOLLOW this one tempo-wise - same
                # answer as ordered mode: the operator's list outranks
                # beat-matching, take the dipped fade to the closest-fit
                # remaining pool track instead of leaving the list.
                rest = [t for t in self.brain.library
                        if t.id in self.brain.pool_ids]
                if rest:
                    arc = self.arc_target()
                    cand = min(rest, key=lambda t: abs(
                        t.energy_proxy() - arc))
                    meta = {"rate": 1.0, "eff_bpm": cand.bpm,
                            "pair": None, "tempo_clash": True}
                    print(f"[DJ] pool tempo clash - fading to "
                          f"{cand.title[:30]}")
                else:
                    self.brain.pool_ids = None
                    self._setlist_name = None
                    cand, meta = self._pick_next(out_bpm)
            if cand is None:
                self.last_error = "no compatible next track"
                return
            self.next_track = cand
            self._next_meta = meta
        else:
            _, self._next_meta = self.brain.score(
                self.current, self.next_track, self.arc_target(), out_bpm)
            if self._next_meta is None:
                # Requested/injected next that the scorer rejects: same
                # answer as an ordered setlist entry - read the tempo
                # honestly so an unreachable stretch becomes a deliberate
                # fade, never a fake blend of two sliding grids.
                rate, eff = self.brain.rate_for(out_bpm, self.next_track)
                self._next_meta = {"rate": rate or 1.0,
                                   "eff_bpm": eff or self.next_track.bpm,
                                   "pair": None}
                if rate is None:
                    self._next_meta["tempo_clash"] = True
        # Use the RAM-cached samples; if the background decode isn't done
        # yet, wait (we still have PLAN_LEAD_S of runway) rather than decode
        # inline and risk starving the audio callback.
        samples = self._decoded_samples(self.next_track)
        if samples is None:
            if self.next_track.id in self._decoding:
                return                       # decoding - retry next step()
            self.brain.library.remove(self.next_track)
            self.next_track = None
            return
        # A seam that already flam-bailed can't hold a lock (one side's
        # grid is untrustworthy despite its confidence) - retry it as a
        # DELIBERATE fade, the same honest answer given to tempo clashes.
        if (self.current.id, self.next_track.id) in self._flam_pairs:
            self._next_meta = dict(self._next_meta or {})
            self._next_meta["tempo_clash"] = True
        after = pos + max(self._exit_played - played, MIN_LEAD_S)
        plan = self.brain.plan_transition(self.current, self.next_track,
                                          self._next_meta,
                                          after_s=min(after, deadline),
                                          arc=self.arc_target())
        # VERIFIED TEMPO: the plan's rate was computed from the STORED
        # bpm; when predecode measured the incoming track's true tempo,
        # rescale so the deck plays at the tempo that actually matches
        # (split-aware: under the varispeed meet-in-the-middle both
        # sides carry sqrt of the correction).
        fix = self._grid_fix.get(self.next_track.id)
        if fix:
            f = self.next_track.bpm / fix["bpm"]
            if abs(f - 1.0) > 0.003:
                import math as _m
                if plan.get("a_rate") not in (1.0, None):
                    plan["rate"] *= _m.sqrt(f)
                    plan["a_rate"] /= _m.sqrt(f)
                else:
                    plan["rate"] *= f
        if plan["out_s"] <= pos + MIN_LEAD_S:
            plan["out_s"] = self.current.nearest_downbeat(pos + MIN_LEAD_S
                                                          + 2 * self.current.period_s)
        # URGENT EXIT (skip / mix_now): the pair scorer is free to pick a
        # beautiful boundary MINUTES away - correct for autopilot, absurd
        # for a button named "mix now" (measured: blend_in 419s). Force
        # the exit onto the next phrase boundary within ~30 s.
        if self._urgent_exit and plan["out_s"] > pos + 30.0:
            t_exit = self.current.nearest_phrase(pos + 12.0)
            if not (pos + MIN_LEAD_S <= t_exit <= pos + 30.0):
                t_exit = self.current.nearest_downbeat(pos + 12.0)
            plan["out_s"] = min(max(t_exit, pos + MIN_LEAD_S + 1.0),
                                max(self.current.duration_s - 10.0,
                                    pos + MIN_LEAD_S + 1.0))
        incoming = "b" if self.active_deck == "a" else "a"
        if self._recovery_txn is not None:
            # A pending abort-recovery still holds a delayed stop for this
            # deck - recall it before the new transition takes over.
            self.submix.post({"cmd": "cancel", "txn": self._recovery_txn})
            self._recovery_txn = None
        # STEM STYLES: verify the decoded stems actually made it (disk may
        # have changed since library load, decode may have failed) - if
        # not, downgrade to the classic geometry BEFORE building events so
        # the stem_gains commands are never posted against a stem-less
        # deck (which would no-op and leave the full mix riding).
        with self._decode_lock:
            stems_b = self._decoded_stems.get(self.next_track.id)
            stems_a = self._decoded_stems.get(self.current.id) \
                if self.current is not None else None
        need_a = plan["style"] in ("stem_drum_swap", "acapella_out")
        need_b = plan["style"] == "stem_drum_swap"
        if need_a and stems_a is None:
            # Cache may have evicted A's stems (loaded two tracks ago) -
            # they're on disk, re-decode inline (a few seconds on the
            # planner thread, well before the blend arms), aligned to the
            # ACTIVE deck's buffer so attach_stems maps 1:1.
            try:
                d = self.submix.decks.get(self.active_deck)
                n = len(d.samples) if (d is not None
                                       and d.samples is not None) else 0
                if n and getattr(self.current, "has_stems", False):
                    from lib.dj.stems import load_stems
                    stems_a = load_stems(self.db.music_root,
                                         self.current.id, expected_len=n)
            except Exception as e:
                print(f"[DJ] stem re-decode failed: {e}")
                stems_a = None
        if (need_b and stems_b is None) or (need_a and stems_a is None):
            self._log({"event": "stem_downgrade", "style": plan["style"]})
            plan["style"] = "bass_swap"
            plan.pop("tail_beats", None)
        elif need_a and stems_a is not None:
            self.submix.post({"cmd": "attach_stems",
                              "deck": self.active_deck, "stems": stems_a})
        self.submix.post({"cmd": "load", "deck": incoming, "samples": samples,
                          "track_id": self.next_track.id,
                          # The corrected grid IS the sync reference - a
                          # wrong-tempo grid is what the flam seams chased.
                          "grid": (fix["grid"] if fix
                                   else self.next_track.grid),
                          "gain_db": self.next_track.gain_db,
                          "kick_offset_s": self.next_track.kick_offset_s,
                          "pitch_st": plan.get("pitch_st", 0),
                          "cue_s": plan["in_s"],
                          "stems": stems_b})
        events, swap_at, blend_at = self.brain.build_events(
            plan, self.submix.telemetry, self.active_deck, incoming,
            self.current, self.next_track)
        events += self._perc_bed_events(plan, blend_at, swap_at)
        # Tag the whole script as one transaction so _do_abort can recall
        # every not-yet-fired event with a single cancel.
        self._txn_id += 1
        for e in events:
            e["txn"] = self._txn_id
        self.submix.post_many(events)
        self.plan = plan
        self.swap_at = swap_at
        self.blend_at = blend_at
        self.state = "armed"
        rt = plan.get("rhythm") or {}
        self._seam_metrics = {"style": plan["style"], "max_err": 0.0,
                              "err_n": 0, "hole_s": 0.0, "low_since": None,
                              "urgent": self._urgent_exit,
                              # Predicted groove terms ride along so the
                              # self-assessment can log prediction vs
                              # measurement (the calibration loop).
                              "predicted": {k: rt.get(k) for k in
                                            ("score", "kick_agreement",
                                             "swing_delta", "flam_ms",
                                             "conf")} if rt else None}
        self._urgent_exit = False
        self._log({"event": "armed", "style": plan["style"],
                   "next": self.next_track.title,
                   "rate": round(plan["rate"], 4),
                   "out_s": round(plan["out_s"], 2),
                   "in_s": round(plan["in_s"], 2),
                   "pair_score": plan["pair_score"],
                   "blend_in_s": round((blend_at - self.submix.clock) / RATE, 1)})

    def _note_pool_played(self, track):
        pool = self.brain.pool_ids
        if pool is None or track is None:
            return
        pool.discard(track.id)
        if not pool:
            self.brain.pool_ids = None
            self._setlist_name = None
            self._setlist_mode = "order"
            self._log({"event": "setlist_pool_done"})
            print("[DJ] setlist pool complete - free play resumes")

    def _finish_swap(self):
        if self._history_id:
            self.db.log_play_end(self._history_id)
        old = self.current
        self.current = self.next_track
        self.next_track = None
        self._next_requested = False
        self.active_deck = "b" if self.active_deck == "a" else "a"
        self._started_clock = self.submix.clock
        self._draw_exit()
        self.brain.note_played(self.current)
        self._note_pool_played(self.current)
        self._note_energy(self.current)
        self._history_id = self.db.log_play_start(
            self.current.id, transition_style=self.plan["style"],
            theme=self._theme_name)
        self._last_style = self.plan["style"]
        self._last_pair = (old.id if old else None, self.current.id)
        self._wd_loop = False
        self._assess_seam(old)
        # QUEUE CONTINUITY: the played track leaves the front of the
        # horizon; the rest advances and gets topped up - never a
        # wholesale rebuild on swap (read as 'the plan reset').
        if self._horizon and self._horizon[0]["id"] == self.current.id:
            self._horizon = self._horizon[1:]
        else:
            self._horizon = []
        self._history.append({"t": time.strftime("%H:%M"),
                              "title": self.current.title,
                              "artist": self.current.artist,
                              "via": self.plan["style"]})
        self._history = self._history[-60:]
        self._log({"event": "play", "track": self.current.title,
                   "artist": self.current.artist, "bpm": self.current.bpm,
                   "camelot": self.current.camelot,
                   "via": self.plan["style"], "after": old.title if old else None})
        self.plan = None
        self.swap_at = None
        self.blend_at = None
        # Verified-tempo fixes: keep only the now-playing track's (the
        # next track re-verifies at its own predecode).
        self._grid_fix = {k: v for k, v in self._grid_fix.items()
                          if k == self.current.id}
        self.state = "playing"

    def _do_seek(self, val):
        """Jump the current track (playing state only). val is an absolute
        position, ('rel', delta), or ('exit', None)."""
        if self.state != "playing" or self.current is None:
            return
        pos = self._pos_s() or 0.0
        dur = self.current.duration_s
        if isinstance(val, tuple):
            if val[0] == "rel":
                target = pos + val[1]
            elif val[0] == "exit":
                # Jump near the end so the exit gate (pos >= deadline-lead)
                # arms the transition right away - fast mix audition.
                target = dur - 55.0
                target = max(target, pos + 4.0)
            else:
                return
        else:
            target = float(val)
        target = max(0.0, min(target, dur - 10.0))
        target = self.current.nearest_downbeat(target)
        self.submix.post({"cmd": "cue", "deck": self.active_deck,
                          "time_s": target})
        # NOTE: do NOT touch _started_clock - `played` tracks OUTPUT time the
        # track has been up, not its source position. Seeking forward must
        # not make the system think the track is 'done' and fire a mix (the
        # bug that left every jump stuck in the armed state).
        self._log({"event": "seek", "to_s": round(target, 1)})

    def _do_abort(self, via="abort"):
        """Recall an ARMED transition: cancel its not-yet-fired events and
        restore the outgoing deck. Only possible BEFORE the plan's point of
        no return (the bass swap / cut / drop) - past that, finishing the
        mix sounds better than any rescue. Returns True if recalled."""
        if self.state != "armed" or self.plan is None:
            return False
        clk = self.submix.clock
        nra = self.plan.get("no_return_at")
        if nra is None or clk >= nra - int(0.5 * RATE):
            return False
        incoming = "b" if self.active_deck == "a" else "a"
        tel = (self.submix.telemetry or {}).get("decks", {})
        a_rate = float((tel.get(self.active_deck) or {}).get("rate") or 1.0)
        # Cancel first (FIFO: it reaches the submix before the recovery).
        # Posted separately so the recovery-txn tagging below can't clobber
        # its target txn (it once did - the blend script survived its own
        # abort and its stale events fired minutes later).
        self.submix.post({"cmd": "cancel", "txn": self._txn_id})
        # Then unwind whatever already fired: kill the incoming deck, and
        # restate everything a style may have already shaped on the
        # outgoing one (EQ carve, filter, echo, loop, duck, dual-bend).
        ev = [{"cmd": "end_sync"},
              {"cmd": "duck", "on": False},
              {"cmd": "gain", "deck": incoming, "value": 0.0, "ramp_s": 0.4},
              {"at": clk + int(0.6 * RATE), "cmd": "stop", "deck": incoming},
              {"cmd": "release_loop", "deck": self.active_deck},
              {"cmd": "filter", "deck": self.active_deck, "mode": "off"},
              {"cmd": "echo", "deck": self.active_deck, "active": False},
              {"cmd": "eq", "deck": self.active_deck, "low": 1.0,
               "mid": 1.0, "high": 1.0, "ramp_s": 1.2},
              {"cmd": "gain", "deck": self.active_deck, "value": 1.0,
               "ramp_s": 1.2}]
        if abs(a_rate - 1.0) > 1e-3:     # dual-bend ramp already under way
            ev.append({"cmd": "rate", "deck": self.active_deck, "value": 1.0,
                       "ramp_s": abs(a_rate - 1.0) / GLIDE_PER_S})
        # The recovery is its own transaction: its DELAYED incoming-deck
        # stop must be recallable, or a skip's immediate urgent re-arm (new
        # blend can start at +0.3s) gets its deck killed at +0.6s by the
        # leftover stop (measured: the room went silent while the
        # bookkeeping 'handover' completed).
        self._txn_id += 1
        for e in ev:
            e["txn"] = self._txn_id
        self._recovery_txn = self._txn_id
        self.submix.post_many(ev)
        style = self.plan["style"]
        self.plan = None
        self.swap_at = None
        self.blend_at = None
        self.state = "playing"
        self._seam_metrics = None
        # Don't instantly re-arm the same seam: push the drawn exit past
        # the planning lead (an exactly-PLAN_LEAD_S push re-arms the very
        # same step) so the operator (or the skip flow) decides what's next.
        played = (clk - self._started_clock) / RATE
        self._exit_played = max(self._exit_played, played + PLAN_LEAD_S + 30.0)
        self._log({"event": "abort", "via": via, "style": style})
        return True

    def _do_skip(self):
        """Exit at the earliest musical opportunity."""
        if self.state == "armed":
            # A transition in flight can be recalled up to its decisive
            # moment - then the skip replans an urgent exit. Past the point
            # of no return, letting it finish IS the fastest skip.
            if not self._do_abort(via="skip"):
                return
        if self._history_id:
            self.db.log_play_end(self._history_id, skipped=True)
            self._history_id = None
        if self.current is not None:
            self.brain.note_skipped(self.current)
        self._urgent_exit = True
        self._log({"event": "skip", "track":
                   self.current.title if self.current else None})
        # Force planning NOW with an exit a few bars out. An explicitly
        # REQUESTED next survives the skip - 'play this next' + 'mix now'
        # must not throw the request away.
        if not getattr(self, "_next_requested", False):
            self.next_track = None
        self.plan = None
        self._exit_played = (self.submix.clock - self._started_clock) / RATE
        self._maybe_plan()

    def _watchdog(self, pos):
        """CONTINUITY WATCHDOG: the music must never simply run out. If the
        current track is nearly over and no transition armed (persistent
        'no compatible next', a decode that never landed, autopilot off with
        an empty queue), escalate: force a pick (ignoring tempo gates if it
        comes to that), buy time with a safety loop over the last phrase,
        and hand off with a clock-domain fade the moment samples are ready."""
        if self.state != "playing" or self.current is None or pos is None:
            return
        tel = (self.submix.telemetry or {}).get("decks", {})
        d = tel.get(self.active_deck) or {}
        rate = max(float(d.get("rate") or 1.0), 1e-6)
        remaining = (self.current.duration_s - pos) / rate
        if self._wd_loop and remaining > 45.0:
            # Operator seeked back out of the endgame: the cursor is behind
            # the loop window again (a bare clear is mapping-safe there);
            # hand planning back to the brain.
            self.submix.post({"cmd": "clear_loop", "deck": self.active_deck})
            self._wd_loop = False
            self._log({"event": "watchdog_release"})
            return
        if remaining > WATCHDOG_S:
            return
        # 1) Make sure SOMETHING is queued. The emergency pick ignores the
        # tempo gates entirely - continuity outranks polish, the seam will
        # be a fade. Autopilot-off is respected: no pick is forced, the
        # safety loop below holds the floor until the operator acts.
        if self.next_track is None and self.autopilot:
            out_bpm = self.current.bpm * rate
            cand, meta = self._pick_next(out_bpm)
            if cand is None:
                cand, meta = self.brain.emergency_pick(
                    self.current, self.arc_target())
                if cand is not None:
                    self._log({"event": "watchdog_pick", "track": cand.title})
            if cand is not None:
                self.next_track, self._next_meta = cand, meta
                self._predecode(self.next_track)
        # 2) Between WATCHDOG_S and WD_LOOP_S the normal planner gets first
        # shot at the injected pick (a real musical seam beats a rescue
        # fade). Below WD_LOOP_S with samples in RAM, hand off NOW - and
        # never via build_events: with the safety loop wrapping the cursor
        # its source->clock mapping is invalid, so the handoff is scheduled
        # purely in clock time.
        if remaining >= WD_LOOP_S and not self._wd_loop:
            return
        samples = None
        if self.next_track is not None:
            with self._decode_lock:
                samples = self._decoded.get(self.next_track.id)
            if samples is None and not self.threaded:
                samples = self._decoded_samples(self.next_track)
        if samples is not None:
            self._emergency_handoff(samples, pos)
            return
        # 3) Buy time: loop the last phrase until the decode lands (or the
        # operator queues something). Idempotent; released by the handoff.
        if not self._wd_loop and remaining < WD_LOOP_S:
            per = max(self.current.period_s, 0.1)
            le = self.current.nearest_downbeat(self.current.duration_s - 3.0)
            if le <= pos + 1.0:
                le = min(self.current.duration_s - 0.5, pos + 4 * per)
            ls = max(0.0, le - 16 * per)
            self.submix.post({"cmd": "loop", "deck": self.active_deck,
                              "start_s": ls, "end_s": le})
            self._wd_loop = True
            self._log({"event": "watchdog_loop", "track": self.current.title,
                       "start_s": round(ls, 1), "end_s": round(le, 1),
                       "autopilot": self.autopilot})
            print(f"[DJ] watchdog: looping '{self.current.title[:30]}' outro"
                  f" to keep the floor alive")

    def _emergency_handoff(self, samples, pos):
        """Watchdog handoff: a 6s clock-domain dipped fade to the queued
        next track. Deliberately styleless - it exists so the room never
        goes silent, not to win seam-of-the-night."""
        nxt = self.next_track
        incoming = "b" if self.active_deck == "a" else "a"
        if self._recovery_txn is not None:
            self.submix.post({"cmd": "cancel", "txn": self._recovery_txn})
            self._recovery_txn = None
        clk = self.submix.clock
        in_s = nxt.nearest_downbeat(_intro_start(nxt))
        end = clk + int(6.0 * RATE)
        ev = [
            {"at": clk, "cmd": "load", "deck": incoming, "samples": samples,
             "track_id": nxt.id, "grid": nxt.grid, "gain_db": nxt.gain_db,
             "kick_offset_s": nxt.kick_offset_s, "cue_s": in_s},
            {"at": clk, "cmd": "gain", "deck": incoming, "value": 0.0,
             "ramp_s": 0.01},
            {"at": clk, "cmd": "start", "deck": incoming},
            {"at": clk, "cmd": "gain", "deck": incoming, "value": 1.0,
             "ramp_s": 4.0},
            {"at": clk, "cmd": "gain", "deck": self.active_deck,
             "value": 0.0, "ramp_s": 5.0},
            {"at": end, "cmd": "stop", "deck": self.active_deck},
            {"at": end, "cmd": "clear_loop", "deck": self.active_deck},
        ]
        self._txn_id += 1
        for e in ev:
            e["txn"] = self._txn_id
        self.submix.post_many(ev)
        self.plan = {"style": "emergency_fade", "rate": 1.0,
                     "out_s": pos, "in_s": in_s, "beats": 0,
                     "pair_score": 0.0, "cand_id": nxt.id,
                     "no_return_at": clk}
        self.swap_at = end
        self.blend_at = clk
        self.state = "armed"
        self._seam_metrics = None        # nothing to judge on a rescue fade
        self._wd_loop = False
        self._log({"event": "watchdog_handoff", "next": nxt.title})
        print(f"[DJ] watchdog: emergency fade into '{nxt.title[:30]}'")

    def _collect_seam_metrics(self):
        """Sampled each step while a transition runs: worst audible grid
        flam between the synced decks and any level hole in the overlap -
        the raw material for the seam self-assessment."""
        m = self._seam_metrics
        if m is None:
            return
        tel = self.submix.telemetry or {}
        decks = tel.get("decks") or {}
        sync = tel.get("sync")
        clk = tel.get("clock", 0)
        if sync:
            sl = decks.get(sync.get("slave"))
            ms = decks.get(sync.get("master"))
            # SETTLING GRACE: the first seconds after the blend starts are
            # the PLL converging from its snap - especially urgent exits
            # (mix_now arms with ~0.3s run-in, no quiet settling window).
            # Counting convergence as flam made the bailout abort EVERY
            # mix_now (measured: 0.22-0.47 beats within 3s, four aborts in
            # a row on the same pair). Judge only settled lock.
            settled = (self.blend_at is not None
                       and clk >= self.blend_at + int(6.0 * RATE))
            if (settled and sl and ms
                    and sl.get("playing") and ms.get("playing")
                    and (sl.get("gain") or 0) > 0.25
                    and (ms.get("gain") or 0) > 0.25):
                # Judge against the PLL's actual target: sync holds the
                # slave's grid OFFSET by the kick-alignment bias, so raw
                # phase difference minus that bias is the real flam.
                err = ((float(sl.get("beat_phase") or 0.0)
                        - float(ms.get("beat_phase") or 0.0)
                        - float(sync.get("bias_beats") or 0.0)
                        + 0.5) % 1.0) - 0.5
                if abs(err) > m["max_err"]:
                    m["max_err"] = abs(err)
                if abs(err) > 0.12:      # flam must PERSIST, not spike once
                    m["err_n"] += 1
                # LIVE FLAM BAILOUT: persistent large error with both decks
                # audible means a stored tempo is wrong beyond the PLL's
                # trim authority - the drift only grows (measured seams ran
                # to half-beat OPPOSITION for the rest of the blend, 17-38
                # resnaps, user-heard double beats). Clean seams never
                # sustain: p90 max_err 0.094 and spikes don't repeat, so
                # err_n>=3 with a current error near a fifth of a beat is
                # unambiguous. Bail once, live, instead of logging it after.
                if (not m.get("bailed") and m["err_n"] >= 3
                        and abs(err) >= 0.18):
                    m["bailed"] = True
                    self._flam_bailout(abs(err))
        rms = tel.get("rms")
        if (rms is not None and self.blend_at is not None
                and clk >= self.blend_at):
            if rms < 0.02:
                if m["low_since"] is None:
                    m["low_since"] = clk
                m["hole_s"] = max(m["hole_s"],
                                  (clk - m["low_since"]) / RATE)
            else:
                m["low_since"] = None
        st = tel.get("sync_stats") or {}
        m["resnaps"] = st.get("resnaps", 0)
        m["nudges"] = st.get("nudges", 0)
        m["cals"] = st.get("cals", 0)
        m["cal_applied"] = st.get("cal_applied", 0.0)

    def _flam_bailout(self, err):
        """The PLL is losing this blend. Before the commit point, recall
        the whole transition (the rescue layer restores A and replans);
        past it, FINISH FAST - collapse the outgoing deck over ~4 beats so
        the opposition kicks stop, instead of riding the double beat for
        the rest of the overlap. The seam self-assessment still records
        the flam, so pair memory demotes this pairing either way. The pair
        also goes on the session's flam list: a retry of the SAME seam
        must be a deliberate fade, not the identical doomed beat-match
        (measured: mix_now retried the same plan and abort-looped)."""
        clk = self.submix.clock
        if self.current is not None and self.next_track is not None:
            self._flam_pairs.add((self.current.id, self.next_track.id))
        if self._do_abort(via="flam"):
            self._log({"event": "flam_bailout", "mode": "abort",
                       "err_beats": round(err, 3)})
            return
        beat = self.current.period_s if self.current is not None else 0.5
        incoming = "b" if self.active_deck == "a" else "a"
        self.submix.post_many([
            {"at": clk, "cmd": "gain", "deck": self.active_deck,
             "value": 0.0, "ramp_s": 4 * beat},
            {"at": clk, "cmd": "eq", "deck": incoming, "low": 1.0,
             "mid": 1.0, "high": 1.0, "ramp_s": 2 * beat},
            {"at": clk + int(4 * beat * RATE), "cmd": "end_sync"},
        ])
        self._log({"event": "flam_bailout", "mode": "finish_fast",
                   "err_beats": round(err, 3)})

    def _assess_seam(self, old):
        """SEAM SELF-ASSESSMENT: judge the transition that just finished
        from its own measurements and remember measured train-wrecks as a
        gentle auto thumbs-down in pair memory - the DJ gets better every
        night without anyone touching a button."""
        m, self._seam_metrics = self._seam_metrics, None
        if m is None or old is None or self.current is None:
            return
        style = m.get("style")
        beat_matched = style not in (None, "long_fade", "emergency_fade")
        # 0.12 beats sustained while both decks are audible is a clearly
        # audible flam (~55ms at 128 bpm) - well past the PLL's deadband
        # and beyond what a healthy lock ever shows.
        flam = beat_matched and m["max_err"] > 0.12 and m.get("err_n", 0) >= 2
        hole = m.get("hole_s", 0.0) > 1.5
        verdict = "flam" if flam else ("hole" if hole else "clean")
        self._last_seam = {"style": style, "verdict": verdict,
                           "max_err_beats": round(m["max_err"], 3),
                           "hole_s": round(m.get("hole_s", 0.0), 2),
                           "resnaps": m.get("resnaps", 0),
                           "b": self.current.title}
        # Prediction vs measurement: over nights this log answers "does
        # kick_agreement<0.35 actually predict measured flams?" - the data
        # that will eventually tune the composite weights.
        self._log({"event": "seam_quality", **self._last_seam,
                   "a": old.title, "urgent": m.get("urgent", False),
                   "predicted_rhythm": m.get("predicted")})
        if verdict == "clean":
            return
        if m.get("urgent"):
            # A skip/mix-now seam exits from wherever the track happened to
            # be - a hole or rough lock there indicts the button press, not
            # the pairing. Measure and log, but never charge pair memory.
            return
        try:
            self.db.add_seam_feedback(old.id, self.current.id, style,
                                      up=False, source="auto")
        except Exception as e:
            print(f"[DJ] auto seam feedback store failed: {e}")
        key = (old.id, self.current.id)
        cur = self.brain.pair_memory.get(key, 1.0)
        self.brain.pair_memory[key] = max(0.4, cur * 0.85)
        print(f"[DJ] seam self-assessment: {verdict} on "
              f"'{old.title[:24]}' -> '{self.current.title[:24]}' ({style})")

    def _annotate_horizon(self, items):
        """PROJECTED TIMELINE: when each queued track will actually start
        (seconds from now) and how long it should run, so the night chart
        can place them time-true instead of decoratively."""
        pos = self._pos_s()
        played = (self.submix.clock - self._started_clock) / RATE
        if self.plan and pos is not None:
            t0 = max(self.plan["out_s"] - pos, 0.0)
        else:
            t0 = max(self._exit_played - played, 30.0)
        per_play = min(max(self.brain.theme.min_play_s, 150.0), 300.0)
        for h in items:
            h["eta_s"] = round(t0, 1)
            h["play_s"] = round(per_play, 1)
            t0 += per_play

    def _pop_setlist_next(self, out_bpm):
        """Next track from the loaded setlist. Anchors are HARD (played even
        if the seam needs a long_fade); a tempo-impossible SUGGESTION is
        dropped and the live brain substitutes."""
        while self._setlist_queue:
            entry = self._setlist_queue.pop(0)
            t = next((x for x in self.brain.library
                      if x.id == entry["track_id"]), None)
            if t is None:
                continue
            if self.current is not None and t.id == self.current.id:
                continue
            _, meta = self.brain.score(self.current, t,
                                       self.arc_target(), out_bpm)
            if meta is None:
                # The operator ordered THIS track. A tempo clash is not a
                # reason to play something else - it's a reason to fade
                # (the dipped handoff makes any pair playable). The old
                # behavior substituted from the whole library, which on an
                # eclectic setlist cascaded into a mostly-substituted
                # night (user: 'what's the point of it?').
                rate, eff = self.brain.rate_for(out_bpm, t)
                meta = {"rate": rate or 1.0, "eff_bpm": eff or t.bpm,
                        "pair": None}
                if rate is None:
                    meta["tempo_clash"] = True   # plan -> long_fade
            self._play_hint_s = entry.get("target_play_s")
            self._log({"event": "setlist_next", "track": t.title,
                       "pin": entry.get("pin_type", "suggestion"),
                       "remaining": len(self._setlist_queue)})
            return t, meta
        self._setlist_name = None
        self._setlist_mode = "order"
        return None, None

    # -- helpers ---------------------------------------------------------------
    def _refresh_setlist_names(self):
        self._setlists_checked = time.time()
        try:
            from lib.dj.setlist import list_setlists
            self.setlist_names = [s["name"] for s in list_setlists(self.db)]
        except Exception:
            pass

    def _draw_exit(self):
        # An ordered-setlist entry may carry the autofill timing solver's
        # play hint - honoring it is what makes timed anchors land live,
        # not just in the compiled preview. Consumed once per track.
        hint = getattr(self, "_play_hint_s", None)
        self._play_hint_s = None
        if hint:
            self._exit_played = max(40.0, float(hint))
            return
        theme = self.brain.theme
        span = max(theme.max_play_s - theme.min_play_s, 1.0)
        # ARC-COUPLED ROTATION: at peak heat tracks rotate fast (real DJs
        # ride 2-3 minutes per record and keep hitting); in the valleys
        # they breathe long. heat=0 -> [0.25..1.0] of the span, heat=1 ->
        # [0..0.45] - pacing itself follows the night's shape.
        heat = self.arc_target()
        frac = min(1.0, self.brain.rng.random() * (1.0 - 0.55 * heat)
                   + 0.25 * (1.0 - heat))
        self._exit_played = theme.min_play_s + frac * span

    def _predecode(self, track):
        """Start decoding `track` on a background daemon thread so the whole
        file is in RAM before we need it - the synchronous decode used to
        run on the planner thread at plan time and starve the audio callback
        (a 1-3s CPU burst on a 5-min mp3 = an underrun/GAP at the switch).

        Only meaningful live (threaded): offline/tests decode inline in
        _decoded_samples for determinism (no audio callback to protect)."""
        if track is None or not self.threaded:
            return
        with self._decode_lock:
            if track.id in self._decoded or track.id in self._decoding:
                return
            self._decoding.add(track.id)

        def work():
            try:
                s = self._decode_gil_friendly(self.db.abs(track.path))
            except Exception as e:
                s = None
                self.last_error = f"decode failed: {track.path}: {e}"
                print(f"[DJ] {self.last_error}")
            stems = self._decode_stems(track, s)
            if s is not None and track.id not in self._grid_fix:
                self._verify_tempo(track, s)
            with self._decode_lock:
                self._decoding.discard(track.id)
                if s is not None:
                    self._decoded[track.id] = s
                    if stems is not None:
                        self._decoded_stems[track.id] = stems
                    # Keep only a couple of tracks cached (~106MB each).
                    while len(self._decoded_order) >= 3:
                        old = self._decoded_order.pop(0)
                        if old != track.id:
                            self._decoded.pop(old, None)
                            self._decoded_stems.pop(old, None)
                    self._decoded_order.append(track.id)
        threading.Thread(target=work, daemon=True).start()

    def _perc_bed_events(self, plan, blend_at, swap_at):
        """PERCUSSION BED under a long_fade: the fade exists because the
        pair can't beat-match, and its dip is where the floor's energy
        leaks out. When the OUTGOING track has stems, tile 8 beats of its
        own drums (downbeat-aligned, its own tempo - A is still the only
        groove in the room) across the handoff, faded out before B's
        groove needs the space. One pre-baked buffer through the fx bus -
        no new mixer machinery."""
        if plan["style"] != "long_fade" or self.current is None:
            return []
        d = self.submix.decks.get(self.active_deck)
        stems = getattr(d, "stems", None) or \
            self._decoded_stems.get(self.current.id)
        if not stems or "drums" not in stems:
            return []
        cur = self.current
        per = cur.period_s
        ls = cur.nearest_downbeat(plan["out_s"] - 24 * per)
        a = int(ls * RATE)
        b = a + int(8 * per * RATE)
        drums = stems["drums"]
        if a < 0 or b > len(drums):
            return []
        loop = drums[a:b].astype(np.float32)
        if float(np.abs(loop).max()) < 0.05:
            return []                       # that stretch has no drums
        span_s = min((swap_at - blend_at) / RATE + 2.0, 30.0)
        if span_s < 6.0:
            return []
        reps = int(np.ceil(span_s / max(8 * per, 0.5)))
        bed = np.tile(loop, (reps, 1))[:int(span_s * RATE)]
        n = len(bed)
        fi = int(min(2.0, span_s * 0.2) * RATE)
        fo = int(min(5.0, span_s * 0.4) * RATE)
        env = np.ones(n, dtype=np.float32)
        env[:fi] = np.linspace(0.0, 1.0, fi, dtype=np.float32)
        env[n - fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        bed *= env[:, None] * 0.45 * getattr(d, "loudness_gain", 1.0)
        self._log({"event": "perc_bed", "span_s": round(span_s, 1)})
        return [{"at": blend_at, "cmd": "fx_play", "samples": bed,
                 "gain": 1.0}]

    def _verify_tempo(self, track, samples):
        """TRUST BUT VERIFY the stored tempo against the ACTUAL audio.

        The measured flam class (seams drifting to half-beat opposition,
        17-38 resnaps) is a stored BPM wrong beyond the PLL's +/-1.2%
        trim authority - the deck's grid becomes a WRONG sync reference
        and the blend chases it. The full samples are in RAM at predecode
        time, so re-measure the beat period on an 80s groove window and,
        when the library disagrees, hand the deck a corrected grid and
        rescale the planned rate. Local tempo on a groove window is a far
        easier measurement than the scanner's whole-track problem (no
        intros, no tempo ramps), so a confident local read outranks the
        stored value. Runs on the decode background thread - zero cost to
        the audio path."""
        try:
            if samples is None or track.bpm <= 0:
                return
            if self._urgent_exit:
                # An urgent transition is waiting on this decode - a CPU
                # burst here starves the audio callback (audible dropout
                # right at the seam; user-heard). Skip: the run-in
                # calibration + flam bailout cover this one seam, and the
                # track gets verified on its next normal load.
                return
            dur = (len(samples) if samples.ndim == 1
                   else samples.shape[0]) / RATE
            if dur < 60.0:
                return
            span = min(64.0, dur * 0.5)
            t0 = max(0.0, dur * 0.5 - span * 0.5)
            a0, b0 = int(t0 * RATE), int((t0 + span) * RATE)
            mono = (samples[a0:b0].mean(axis=1) if samples.ndim == 2
                    else samples[a0:b0]).astype(np.float32)
            from lib.dj.features import (frame_track, _onset_channels,
                                         estimate_beat_grid, HOP, CHUNK)
            # GIL-FRIENDLY framing: one 80s frame_track call holds the
            # GIL in ~190ms batches (measured 1.15s total) - enough to
            # underrun the Python audio callback. Chunk into 4s pieces
            # (~60ms holds) with real sleeps between, each chunk padded
            # 1s and trimmed so the causal band smoothing is warmed up -
            # the concatenated frames are IDENTICAL to the single call
            # (no boundary onsets to bias the tempo vote).
            F, PAD = 160, 40                 # frames per chunk / warm-up
            n_total = max(0, (len(mono) - CHUNK) // HOP + 1)
            fb, fc = [], []
            k = 0
            while k < n_total:
                nf = min(F, n_total - k)
                pre = min(PAD, k)
                a = (k - pre) * HOP
                b = a + (pre + nf - 1) * HOP + CHUNK
                if b > len(mono):
                    break
                bb, cc = frame_track(np.ascontiguousarray(mono[a:b]))
                fb.append(bb[pre:])
                fc.append(cc[pre:])
                k += nf
                time.sleep(0.015)            # air for the audio callback
            if not fb:
                return
            bands = np.concatenate(fb)
            onset_broad, _ob, onset_perc, _nov = _onset_channels(bands)
            time.sleep(0.02)
            grid, bpm, conf, _beats = estimate_beat_grid(
                onset_broad + 0.5 * onset_perc)
            if not grid or conf < 0.5 or bpm <= 0:
                return
            # Fold the measurement into the stored value's octave (a
            # half/double-time read is a READ, not a tempo error).
            best = min((bpm * m for m in (1.0, 2.0, 0.5)),
                       key=lambda b: abs(np.log(b / track.bpm)))
            dev = best / track.bpm - 1.0
            if abs(dev) < 0.006:
                return                      # stored value is fine
            if abs(dev) > 0.06:
                # A disagreement this large is more likely an estimator
                # dispute (meter/half-time ambiguity) than a 6% tempo
                # error - don't "correct" onto a maybe-wrong value.
                self._log({"event": "tempo_dispute", "track": track.title,
                           "stored": round(track.bpm, 2),
                           "measured": round(best, 2)})
                return
            g0 = max(grid, key=lambda g: g.get("score", 0.0))
            period = 60.0 / best
            fb = (t0 + g0["first_beat_s"]) % period
            self._grid_fix[track.id] = {
                "bpm": best,
                "grid": [{"start_s": 0.0, "end_s": dur, "period_s": period,
                          "first_beat_s": fb, "bpm": best, "score": 1.0}],
            }
            self._log({"event": "tempo_fix", "track": track.title,
                       "stored": round(track.bpm, 2),
                       "measured": round(best, 2),
                       "dev_pct": round(dev * 100.0, 2),
                       "conf": round(conf, 2)})
            print(f"[DJ] tempo fix: {track.title[:30]} stored "
                  f"{track.bpm:.2f} -> measured {best:.2f} "
                  f"({dev * 100:+.2f}%)")
        except Exception as e:
            print(f"[DJ] tempo verify failed for {track.path}: {e}")

    def _true_bpm(self, track):
        """The track's bpm with any verified-tempo fix applied."""
        if track is None:
            return 0.0
        fix = self._grid_fix.get(track.id)
        return fix["bpm"] if fix else track.bpm

    def _decode_stems(self, track, samples):
        """Decode a track's pre-rendered stems (float16, ~2x the mix's RAM
        for all four) when they exist on disk. None on any failure - the
        stem styles then downgrade to classic geometry at plan time."""
        if samples is None or not getattr(track, "has_stems", False):
            return None
        try:
            from lib.dj.stems import load_stems
            return load_stems(self.db.music_root, track.id,
                              expected_len=len(samples))
        except Exception as e:
            print(f"[DJ] stem decode failed for {track.path}: {e}")
            return None

    def _decode_gil_friendly(self, path):
        """Decode to stereo float32, converting in chunks that RELEASE the
        GIL between them (time.sleep(0)) so the miniaudio audio callback
        never blocks on one big ~260MB memcpy while a track is playing.
        Falls back to the plain decoder if miniaudio can't read the file."""
        try:
            import miniaudio
            dec = miniaudio.decode_file(
                path, output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=2, sample_rate=RATE)  # C decode releases the GIL
            src = memoryview(dec.samples)
            n = len(dec.samples)
            out = np.empty(n, dtype=np.float32)
            step = 1 << 20                       # ~1M samples (~12ms) per copy
            for i in range(0, n, step):
                out[i:i + step] = np.frombuffer(
                    src[i:i + step], dtype=np.float32)
                time.sleep(0)                    # yield to the audio callback
            return out.reshape(-1, 2)
        except Exception:
            from lib.dj.features import decode_file_stereo
            return decode_file_stereo(path)

    def _decoded_samples(self, track):
        """Cached samples if ready, else None (kick off a decode). Offline
        (not threaded) decodes inline so tests are deterministic."""
        with self._decode_lock:
            s = self._decoded.get(track.id)
        if s is not None:
            return s
        if not self.threaded:
            s = self._decode(track)
            if s is not None:
                stems = self._decode_stems(track, s)
                if track.id not in self._grid_fix:
                    self._verify_tempo(track, s)
                with self._decode_lock:
                    self._decoded[track.id] = s
                    if stems is not None:
                        self._decoded_stems[track.id] = stems
                    self._decoded_order.append(track.id)
            return s
        self._predecode(track)
        return None

    def _decode(self, track):
        try:
            from lib.dj.features import decode_file_stereo
            return decode_file_stereo(self.db.abs(track.path))
        except Exception as e:
            self.last_error = f"decode failed: {track.path}: {e}"
            print(f"[DJ] {self.last_error}")
            return None

    def _log(self, payload):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            payload = {"t": round(time.time(), 2),
                       "clock_s": round(self.submix.clock / RATE, 2), **payload}
            p = os.path.join(self.log_dir,
                             f"dj_{time.strftime('%Y%m%d')}.jsonl")
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except OSError:
            pass
