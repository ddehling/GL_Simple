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
import sys
import threading
import time

import numpy as np

from lib.dj.brain import GLIDE_PER_S, Brain, load_library, TrackInfo
from lib.dj.db import LibraryDB
from lib.dj.rhythm import seam_chips
from lib.dj.submix import DJSubmix
from lib.dj.themes import BUILTIN_THEMES, PICKER_THEMES, get_theme

RATE = 44100
PLAN_LEAD_S = 60.0               # start choosing next this early. Must
                                 # exceed the LONGEST blend (96 beats at
                                 # 90 bpm = 64s worst case, ~46s typical):
                                 # build_events clamps the blend start to
                                 # 'now' (now_guard), so arming later than
                                 # the blend span SILENTLY SHORTENS it.
MIN_LEAD_S = 8.0                 # never arm closer than this to the seam
HORIZON_N = 6                    # provisional queue depth (display only -
                                 # slot 0 is the sole committed pick)
SET_CYCLE_S = 90 * 60.0          # non-all-night themes loop their arc here
WATCHDOG_S = 20.0                # continuity watchdog wakes this close to
                                 # the end of the current track unarmed
WD_LOOP_S = 15.0                 # ...and buys time with a safety loop here


def _persona_menu():
    """[(name, tagline)] for the web picker - the real characters only
    (auto/off are picker-level modes, and neutral IS off)."""
    from lib.dj.persona import PERSONAS
    return [(p.name, p.tagline) for p in PERSONAS.values()
            if p.name != "neutral"]


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
                 record=False, persona="auto"):
        self.music_root = music_root
        self.engine = engine
        self.night_hours = night_hours
        self.autopilot = autopilot
        self.threaded = threaded
        self._theme_name = theme
        # Persona MODE: "auto" (date-seeded nightly rotation), a persona
        # name, or "off"/None (neutral). Resolved onto brain.persona at
        # start() and live via set_persona().
        self._persona_mode = persona or "auto"
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
        self._horizon_dry = None         # (key, len) when the planner
                                         # couldn't extend - don't retry
                                         # every tick at 0.5s a burn
        self._history = []               # tonight's tracklist briefs
        self._last_style = None          # last completed transition style
        self._played_energy_ema = None   # arc feedback: what actually played
        self._exit_played = 300.0    # drawn per track from theme min/max play
        self._next_meta = None
        self._setlist_name = None
        self._setlist_mode = "order"
        self._setlist_queue = []     # upcoming entry dicts (plan-following)
        self._setlist_total_s = None  # compiled length -> the set's arc clock
        self._setlist_start_clock = None
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
        self._moment_clock = 0       # submix clock the operator MOMENT lands
        self._moment_txn = None      # its txn tag (second press = recall)
        self._moment_gain = 1.0      # deck gain to restore on the landing
        self._moment_stamped = 0     # landing already flashed to the visuals
        self._moment_flavor = None   # active gesture: drop/spinback/stall
        self._moment_hole = (0, 0)   # clock span of the breath-hold (visuals)
        self._moment_rate = 1.0      # deck rate to restore on recall
        self._moment_denied = None   # (flavor, why, wall_t): last refusal,
                                     # surfaced on the panel button - a
                                     # silent refusal reads as a dead button
        self._decoded = {}           # track_id -> stereo samples (RAM cache)
        self._decoded_order = []
        self._decoding = set()
        self._decoded_stems = {}     # track_id -> {stem: float16 array}
        self._grid_fix = {}          # track_id -> verified-tempo correction
        self._tempo_pool = None      # worker process for _verify_tempo
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
            self.last_error = "library is empty - run tools/dj/dj_scan.py"
            print(f"[DJ] {self.last_error}")
            return False
        self.brain = Brain(lib, get_theme(self._theme_name), seed=self._seed,
                           stretch_max=self._stretch_max)
        self.brain.persona = self._resolve_persona(self._persona_mode)
        if self.brain.persona.name != "neutral":
            print(f"[DJ] tonight: {self.brain.persona.name} - "
                  f"{self.brain.persona.tagline}")
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
        # LOOSE-GRID TAIL: the ceiling on tonight's repertoire, stated up
        # front. A track under bpm_conf 0.5 can never be beat-matched, so
        # every pair touching one is a fade - the single biggest driver of
        # the fade share, and previously invisible until the night sounded
        # flat. Only mentioned when it is actually shaping the night.
        try:
            lib = self.brain.library
            loose = sum(1 for t in lib if (t.bpm_conf or 0.0) < 0.5)
            if lib and loose > 0.12 * len(lib):
                f = loose / len(lib)
                print(f"[DJ] {loose}/{len(lib)} tracks have loose grids "
                      f"(<0.50) - ~{100*(1-(1-f)**2):.0f}% of pairs can only "
                      "fade. Run: python tools/dj/dj_scan.py --refine-grids")
        except Exception:
            pass
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
            for t in self.brain.library:
                t.user_tags = per_track.get(t.id, [])
            # COUNT FROM all_tags - the exact set the hard filter
            # (_tag_ok) matches against - so a chip's (n) IS the
            # rotation size a single lit chip produces. The old code
            # counted user/auto/genre from their raw sources: genre
            # chips double-counted tracks tagged by BOTH MusicBrainz
            # and the file genre, and any tag reachable through a
            # second source (user tag on some tracks, genre on others)
            # showed a count smaller than the pool it selects
            # (user-reported: 'not the correct number of songs in
            # rotation', 2026-07-31).
            user_names, genre_names, auto_names = set(), set(), set()
            counts = {}
            for t in self.brain.library:
                if scope is not None and t.id not in scope:
                    continue
                user_names.update(t.user_tags)
                auto_names.update(t.auto_tags)
                for g in (getattr(t, "genres", None) or []):
                    genre_names.add(str(g).lower())
                for part in (getattr(t, "file_genre", "") or "").replace(
                        "/", ",").replace(";", ",").split(","):
                    part = part.strip().lower()
                    if part:
                        genre_names.add(part)
                for tag in set(t.all_tags):
                    counts[tag] = counts.get(tag, 0) + 1
            vocab = [(tag, counts.get(tag, 0), True) for tag in
                     sorted(user_names, key=lambda k: -counts.get(k, 0))]
            genre_top = sorted(
                ((g, counts.get(g, 0)) for g in genre_names
                 if g not in user_names),
                key=lambda kv: -kv[1])[:24]
            self._genre_tags = {g for g, _ in genre_top}
            vocab += [(g, n, False) for g, n in genre_top]
            vocab += [(tag, counts.get(tag, 0), False) for tag in
                      sorted(auto_names, key=lambda k: -counts.get(k, 0))
                      if tag not in user_names
                      and tag not in self._genre_tags]
            self._tag_vocab = vocab[:80]     # everything, sane ceiling
        except Exception as e:
            print(f"[DJ] tag refresh skipped: {e}")

    def _start_recording(self):
        """Tap the submix into a timestamped WAV - every night becomes
        review material (pair with tools/tests/_dj_quality_test metrics)."""
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
        if self._tempo_pool is not None:
            try:
                self._tempo_pool.stdin.close()      # EOF -> worker exits
                self._tempo_pool.wait(timeout=2.0)
            except Exception:
                try:
                    self._tempo_pool.kill()
                except Exception:
                    pass
            self._tempo_pool = None

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
    @staticmethod
    def _resolve_persona(mode):
        """Persona for the given MODE: "auto" = date-seeded rotation that
        avoids yesterday's base pick (stateless - yesterday is recomputed,
        not stored), "off"/None/"" = neutral, else the named persona
        (unknown names fall back to neutral rather than crash a start)."""
        from lib.dj.persona import PERSONAS, for_night
        if mode == "auto":
            import datetime as _dt
            y = for_night(_dt.date.today() - _dt.timedelta(days=1))
            return for_night(avoid=y.name)
        if mode in (None, "", "off"):
            return PERSONAS["neutral"]
        return PERSONAS.get(mode, PERSONAS["neutral"])

    def set_persona(self, mode):
        with self._lock:
            self._pending.append(("persona", mode))

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

    def moment(self, flavor="drop"):
        """OPERATOR MOMENT: an engineered crowd moment on demand, in one
        of four flavors (see _do_moment): 'drop' builds 8-24 beats and
        lands ON the track's real drop, 'spinback' kills the platter and
        slams into it cold, 'stall' catches the beat in a decaying echo
        and slams back, 'nextdrop' runs the build on the live track and
        lands on the INCOMING track's drop (double-drops the set). A
        second press while one is building recalls it. The visuals
        pre-arm through the published ETA and the landing is stamped as
        a hard drop."""
        with self._lock:
            self._pending.append(("moment", str(flavor or "drop")))

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
        # A loaded setlist runs its OWN arc clock over the compiled set
        # length, so a 90-minute planned set traverses its whole arc in 90
        # minutes regardless of the configured night length. The compiled
        # total is an estimate (live play lengths are drawn stochastically,
        # +-10-20%) - the clamp parks the arc at 1.0 if the set runs long.
        if self._setlist_name and self._setlist_total_s \
                and self._setlist_start_clock is not None:
            elapsed = (self.submix.clock - self._setlist_start_clock) / RATE
            return min(1.0, elapsed / max(self._setlist_total_s, 60.0))
        elapsed = (self.submix.clock - self._set_start_clock) / RATE
        theme = self.brain.theme if self.brain else get_theme(self._theme_name)
        if theme.arc == "all_night":
            return min(1.0, elapsed / max(self.night_hours * 3600.0, 60.0))
        return (elapsed % SET_CYCLE_S) / SET_CYCLE_S

    def _arc_slot(self):
        """(phase, cycle_i) of NOW - where a history entry sits on the
        night canvas. cycle_i counts SET_CYCLE_S wraps so the panel can
        show only the current window; all_night themes never wrap."""
        elapsed = (self.submix.clock - self._set_start_clock) / RATE
        theme = self.brain.theme if self.brain else get_theme(self._theme_name)
        if theme.arc == "all_night":
            return self.arc_progress(), 0
        return self.arc_progress(), int(elapsed // SET_CYCLE_S)

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

    def _render_lead(self):
        """Frames the renderer is ahead of the speakers (render-ahead ring).
        0 when there is no engine or no ring (offline renders, tests)."""
        eng = self.engine
        if eng is None:
            return 0
        try:
            return eng.render_lead_frames()
        except AttributeError:
            return 0

    def _vis_tel(self):
        """Telemetry as HEARD - for anything driving visuals."""
        return self.submix.telemetry_delayed(self._render_lead()) or {}

    def _vis_clock(self):
        """Submix clock as HEARD - for countdowns the visuals stage on."""
        return self.submix.clock - self._render_lead()

    def _vis_pos_s(self):
        """Active deck playhead as HEARD (cf. _pos_s, which is render-time
        and is what the planner/automation must keep using)."""
        d = (self._vis_tel().get("decks") or {}).get(self.active_deck)
        return d["time_s"] if d and d.get("ready") else None

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
        tel = self._vis_tel()
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
        tel = self._vis_tel()
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
        # HEARD-TIME countdowns. submix.clock is the RENDER head, which the
        # render-ahead ring puts ~400ms in front of the speakers; staging a
        # visual move on it fires the move most of a beat early. Everything
        # the club director keys off must count down to when the audio is
        # actually heard.
        vclock = self._vis_clock()
        if self.blend_at is not None and vclock < self.blend_at:
            eta = (self.blend_at - vclock) / RATE
        # TRANSITION CHOREOGRAPHY: the visuals know the future. blend_eta
        # counts down to the overlap starting; swap_eta to the decisive
        # bass/melody handover - the club director stages moves on these
        # exact beats (no human DJ + VJ pair can do this).
        swap_eta = None
        if self.state == "armed" and self.swap_at is not None \
                and vclock < self.swap_at:
            swap_eta = (self.swap_at - vclock) / RATE
        # A pending operator MOMENT pre-arms the visuals the same way an
        # approaching seam does - plus its own keys so the room can be
        # CHOREOGRAPHED: dj_moment_hole is True through the breath-hold
        # (the hole / the dying platter / the echo stall), where the
        # renderer pins the build at max and suppresses beat pulses the
        # ear can't hear (see Stories_OGL).
        m_clk = self._moment_clock
        moment_eta = None
        moment_hole = False
        if m_clk > vclock:
            m_eta = (m_clk - vclock) / RATE
            eta = m_eta if eta is None else min(eta, m_eta)
            moment_eta = m_eta
            h0, h1 = self._moment_hole
            moment_hole = bool(h1) and h0 <= vclock < h1
        elif m_clk and m_clk > self._moment_stamped:
            # The landing is an ENGINEERED drop - stamp it directly rather
            # than hoping the DSP detector arms on the hole we just cut,
            # and stamp it HARD: the renderer slams longer for an operator
            # moment than for a passing musical drop.
            self._dj_drop_wall = time.time()
            self._dj_drop_hard = True
            self._moment_stamped = m_clk
        # GROUND-TRUTH MUSICAL DROPS: the DB knows every drop section of
        # every track. The DSP drop detector needs a QUIET episode to arm
        # (by design, so fades can't fake drops) - a relentless hard set
        # never gives it one, so the club barely slammed all night
        # (user-heard). Publish the next drop's ETA for visual pre-arm
        # and stamp the moment the playhead crosses one.
        drop_eta = None
        pos = self._vis_pos_s()
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
                        and pos - prev_pos < 2.0       # not a seek jump
                        # A MOMENT lands one beat shy of the drop it jumped
                        # to, so the playhead crosses the section boundary
                        # right after the engineered stamp - re-stamping it
                        # as 'musical' here downgraded the hard slam to the
                        # 0.35s flash (caught by _dj_moment_vis_test).
                        and not (m_clk and m_clk <= vclock <= m_clk + RATE)):
                    self._dj_drop_wall = time.time()
                    self._dj_drop_hard = False      # musical, not engineered
            self._drop_scan_prev = (self.current.id, pos)
        ndrop = eta
        if drop_eta is not None:
            ndrop = drop_eta if ndrop is None else min(ndrop, drop_eta)
        return {"dj_active": self._running,
                "dj_arc_phase": self.arc_progress(),
                "dj_arc_heat": self.arc_target(),
                "dj_energy": self.live_energy(),
                "dj_drop_t": getattr(self, "_dj_drop_wall", None),
                "dj_drop_hard": bool(getattr(self, "_dj_drop_hard", False)),
                "dj_next_drop_eta": ndrop,
                "dj_moment_eta": moment_eta,
                "dj_moment_hole": moment_hole,
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
            # Curated picker; a live off-list theme (config/planner) still
            # shows its own lit button rather than vanishing.
            "themes": PICKER_THEMES + (
                [self._theme_name]
                if self._theme_name not in PICKER_THEMES
                and self._theme_name in BUILTIN_THEMES else []),
            "persona": (self.brain.persona.name if self.brain else None),
            "persona_mode": self._persona_mode,
            "persona_tagline": (self.brain.persona.tagline
                                if self.brain else ""),
            "personas": _persona_menu(),
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
            if self._moment_clock > self.submix.clock
            else None,
            # Last refused press, shown on its button for a few seconds
            # ("no drop in this track") - a silent refusal is a dead
            # button to the operator.
            "moment_denied": ({"flavor": self._moment_denied[0],
                               "why": self._moment_denied[1]}
                              if self._moment_denied is not None
                              and time.time() - self._moment_denied[2] < 4.0
                              else None),
            "moment_flavor": (self._moment_flavor
                              if self._moment_clock > self.submix.clock
                              else None),
            "autopilot": self.autopilot,
            # Which SET_CYCLE_S window we're in: history entries carry
            # their own "cyc" so the night canvas shows exactly the
            # tracks of the CURRENT window (0m..now).
            "arc_cycle_i": self._arc_slot()[1],
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
            elif kind == "persona":
                self._persona_mode = str(val or "auto")
                self.brain.persona = self._resolve_persona(self._persona_mode)
                self._log({"event": "persona",
                           "mode": self._persona_mode,
                           "persona": self.brain.persona.name})
                # Selection leans changed - the next pick should reflect
                # the new character. Armed transitions are left alone.
                if self.state == "playing":
                    self.next_track = None
                    self.plan = None
                    self._horizon = []
                    self._horizon_key = None
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
                    self._setlist_total_s = None
                    self._setlist_start_clock = None
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
                    if sl:
                        # The set was PLANNED against a theme, a length and
                        # (maybe) a drawn arc - apply all three so the night
                        # plays the set as designed. Each is only a starting
                        # point: the operator's later theme/arc actions
                        # override normally.
                        if sl.get("theme") \
                                and sl["theme"] != self._theme_name:
                            try:
                                self.brain.set_theme(get_theme(sl["theme"]))
                                self._theme_name = sl["theme"]
                                self._log({"event": "theme",
                                           "theme": sl["theme"],
                                           "via": "setlist"})
                            except Exception:
                                pass
                        if sl.get("total_s"):
                            self._setlist_total_s = float(sl["total_s"])
                            self._setlist_start_clock = self.submix.clock
                        arc = sl.get("arc_json")
                        if arc and not self._arc_waypoints:
                            try:
                                pts = json.loads(arc) \
                                    if isinstance(arc, str) else arc
                                self._arc_waypoints = sorted(
                                    (max(0.0, min(1.0, float(p))),
                                     max(0.0, min(1.0, float(e))))
                                    for p, e in pts)[:8]
                            except (ValueError, TypeError):
                                pass
                    self._log({"event": "setlist",
                               "name": self._setlist_name, "mode": mode,
                               "theme": sl.get("theme") if sl else None,
                               "total_s": self._setlist_total_s,
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
                self._do_moment(val or "drop")
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
            elif kind == "tempo_writeback":
                # Queued by _verify_tempo on the decode thread (LibraryDB
                # is one-instance-per-thread); the write happens HERE so
                # the correction persists - the planner's next compile and
                # every future night start from the measured tempo instead
                # of re-discovering the same disagreement.
                try:
                    self.db.set_verified_tempo(
                        val["track_id"], val["bpm"], val["grid"],
                        val["conf"])
                    self._log({"event": "tempo_writeback", **{
                        k: val[k] for k in
                        ("track_id", "bpm", "conf", "dev_pct")}})
                    # The DB grid just changed: beat-power score and
                    # phase offsets were measured against the OLD grid.
                    # Drop the stale record so the next --phase/--bands
                    # pass re-measures instead of biasing kicks with a
                    # lie.
                    try:
                        from lib.dj import beatpower as _bp
                        import json as _json
                        with open(_bp.path(), encoding="utf-8") as f:
                            _doc = _json.load(f)
                        if _doc.get("scores", {}).pop(
                                str(val["track_id"]), None) is not None:
                            with open(_bp.path(), "w",
                                      encoding="utf-8") as f:
                                _json.dump(_doc, f)
                    except (OSError, ValueError):
                        pass
                except Exception as e:
                    print(f"[DJ] tempo write-back failed: {e}")

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

        # AUDIO-STARVATION ATTRIBUTION: the engine counts callbacks its
        # ring couldn't fill (audible skips), but only to the console -
        # nothing correlated them with WHAT was happening. Poll every ~5s
        # and log the delta with context, so the night log can answer
        # "do stem seams (or stem decodes) cause skips" with data.
        now_t = time.time()
        if self.engine is not None \
                and now_t - getattr(self, "_cb_check_t", 0.0) > 5.0:
            self._cb_check_t = now_t
            try:
                stats = self.engine.callback_stats()
            except AttributeError:
                stats = None
            if stats:
                d = stats.get("starved", 0) - getattr(self, "_cb_seen", 0)
                if d > 0:
                    self._cb_seen = stats["starved"]
                    style = (self.plan or {}).get("style") \
                        or self._last_style
                    self._log({
                        "event": "audio_starved", "n": d,
                        "state": self.state, "style": style,
                        "stem_style": style in (
                            "stem_drum_swap", "acapella_out", "acapella_in",
                            "stem_bass_swap", "drum_bridge", "melody_carry")
                        or bool((self.plan or {}).get("duck_vocal_a")),
                        "decoding": bool(getattr(self, "_decoding", None)),
                        "min_depth_ms": stats.get("min_depth_ms")})

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
        when its inputs changed (it runs real selection, not free).

        HORIZON_N deep (was 3). Only slot 0 is ever COMMITTED; on an
        actively-steered night the tail is direction, not promise (log
        measurement 2026-07-30: steering invalidates the plan about once
        per song), so the panel ghosts slots 3+."""
        if self.current is None or self.brain is None:
            return
        steer = (self._theme_name,
                 json.dumps(self.brain.flavor, sort_keys=True),
                 tuple(sorted(self.brain.require_tags)),   # genre/tag HARD filter
                 tuple(self._arc_waypoints), round(self._energy_nudge, 2))
        steered = steer != self._horizon_key
        front_ok = (self.next_track is None
                    or (bool(self._horizon)
                        and self._horizon[0]["id"] == self.next_track.id))
        if not steered and len(self._horizon) >= HORIZON_N and front_ok:
            return
        if not steered and front_ok \
                and self._horizon_dry == (steer, len(self._horizon)):
            return          # planner is dry at this depth; don't re-burn
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
            for e in self._setlist_queue[:HORIZON_N]:
                t = by_id.get(e["track_id"])
                if t is None:
                    continue
                items.append({
                    "id": t.id, "title": t.title, "artist": t.artist,
                    "bpm": t.bpm, "energy": t.energy_proxy(),
                    "tags": t.all_tags[:4],
                    "why": "setlist " + ("anchor" if e.get("pin_type")
                                         == "anchor" else "pick")})
            self._annotate_horizon(items[:HORIZON_N])
            self._horizon = items[:HORIZON_N]
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
            kept = kept[:HORIZON_N]
            filled = len(kept)
            if filled < HORIZON_N:
                # ONE slot per step() tick, not all six in one burst: a
                # steering change used to trigger a ~3s pure-Python
                # replanning burst on the DJ thread - a GIL competitor
                # to the audio producer, right when the operator is
                # touching the controls (perf audit 2026-07-31). The
                # queue now converges over a few ticks instead.
                tail = by_id[kept[-1]["id"]] if kept else self.current
                pre = [by_id[h["id"]] for h in kept]
                kept += self.brain.plan_horizon(
                    tail, arc_at, tail.bpm, n=1, preplayed=pre)
            if self.brain.pool_ids is not None and len(kept) < HORIZON_N:
                # From an off-pool (or tempo-remote) current track the
                # planner may reach nothing - but the pool WILL play,
                # via the dipped-fade fallback. Show it.
                have = {h["id"] for h in kept}
                arc = self.arc_target()
                rest = sorted((t for t in self.brain.library
                               if t.id in self.brain.pool_ids
                               and t.id not in have),
                              key=lambda t: abs(t.energy_proxy() - arc))
                for t in rest[:HORIZON_N - len(kept)]:
                    kept.append({"id": t.id, "title": t.title,
                                 "artist": t.artist, "bpm": t.bpm,
                                 "energy": t.energy_proxy(),
                                 "tags": t.all_tags[:4],
                                 "why": "setlist pool (fade in)"})
            # Dry marker: the planner couldn't extend past this depth
            # (dead end / tiny pool) - don't re-burn a selection pass
            # every 0.4s tick until something actually changes.
            self._horizon_dry = ((steer, len(kept))
                                 if len(kept) == filled else None)
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
                self._next_style_hint = None    # no seam into the opener
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
        ph, ci = self._arc_slot()
        self._history.append({"t": time.strftime("%H:%M"),
                              "title": first.title, "artist": first.artist,
                              "via": "start", "phase": round(ph, 4),
                              "cyc": ci,
                              "energy": round(first.energy_proxy(), 3)})
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
                    self._setlist_total_s = None
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
        # Honor the planner's per-seam style pin (setlist_entries.
        # style_override) - only when the hint still matches the committed
        # next track (reroll/substitution invalidates it by construction).
        hint = getattr(self, "_next_style_hint", None)
        force_style = hint[1] if hint and hint[0] == self.next_track.id \
            else None
        plan = self.brain.plan_transition(self.current, self.next_track,
                                          self._next_meta,
                                          after_s=min(after, deadline),
                                          arc=self.arc_target(),
                                          force_style=force_style)
        if force_style:
            # (hint is NOT consumed here: a hold/steer replan of the same
            # pinned pair keeps the pin; the next queue pop replaces it.)
            pin = (plan.get("diag") or {}).get("style_pin") or {}
            self._log({"event": "style_pin", "want": force_style,
                       "honored": bool(pin.get("honored")),
                       "why_not": pin.get("why_not"),
                       "style": plan.get("style")})
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
        st = plan["style"]
        need_a = st in ("stem_drum_swap", "acapella_out", "melody_carry",
                        "drum_bridge", "stem_bass_swap")
        need_b = st in ("stem_drum_swap", "drum_bridge", "acapella_in",
                        "stem_bass_swap")
        duck_a = bool(plan.get("duck_vocal_a"))
        if duck_a and not need_a:
            need_a = True                # the duck needs A's vocal stem
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
        if duck_a and stems_a is None and st not in (
                "stem_drum_swap", "acapella_out", "melody_carry",
                "drum_bridge", "stem_bass_swap"):
            # Duck-only failure: keep the style, drop the duck (rare -
            # stems were on disk at plan time; one blend risks the
            # vocal overlap rather than tearing up the geometry).
            plan["duck_vocal_a"] = False
            duck_a = False
            need_a = False
            self._log({"event": "stem_downgrade", "style": st,
                       "what": "vocal_duck"})
        if (need_b and stems_b is None) or (need_a and stems_a is None):
            self._log({"event": "stem_downgrade", "style": plan["style"]})
            plan["style"] = "bass_swap"
            plan.pop("tail_beats", None)
            plan["duck_vocal_a"] = False
        elif need_a and stems_a is not None:
            self.submix.post({"cmd": "attach_stems",
                              "deck": self.active_deck, "stems": stems_a})
        self.submix.post({"cmd": "load", "deck": incoming, "samples": samples,
                          "track_id": self.next_track.id,
                          # The corrected grid IS the sync reference - a
                          # wrong-tempo grid is what the flam seams chased.
                          "grid": (fix["grid"] if fix
                                   else self.next_track.grid),
                          "grid_is_db": not fix,
                          "gain_db": self.next_track.gain_db,
                          "kick_offset_s": self.next_track.kick_offset_s,
                          "pitch_st": plan.get("pitch_st", 0),
                          "cue_s": plan["in_s"],
                          "stems": stems_b})
        # Kick-true anchors read beatpower phase offsets, which are
        # measured against the DB grid - flag any side playing a
        # live-FIXED grid so build_events leaves that side alone.
        plan["grid_fixed"] = {
            "a": self.current.id in self._grid_fix,
            "b": self.next_track.id in self._grid_fix}
        events, swap_at, blend_at = self.brain.build_events(
            plan, self.submix.telemetry, self.active_deck, incoming,
            self.current, self.next_track)
        events += self._perc_bed_events(plan, blend_at, swap_at)
        # The seam takes ownership of this deck now: any MOMENT still in
        # flight has to be recalled first, or its restore (or worse, its
        # hole) fires in the middle of the blend.
        self._cancel_moment("armed")
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
        # THE CALIBRATION JOIN: the selection term breakdown and the style
        # gate record ride into the log next to the seam this plan produced,
        # so tools/dj/dj_review.py can put every tuned constant beside what the
        # seam actually MEASURED. Without these the log recorded outcomes
        # with no inputs, and 560 real seams taught nothing.
        diag = plan.get("diag") or {}
        self._log({"event": "armed", "style": plan["style"],
                   "next": self.next_track.title,
                   "rate": round(plan["rate"], 4),
                   "out_s": round(plan["out_s"], 2),
                   "in_s": round(plan["in_s"], 2),
                   "pair_score": plan["pair_score"],
                   "blend_in_s": round((blend_at - self.submix.clock) / RATE, 1),
                   "terms": {k: round(float(v), 4) for k, v in
                             ((self._next_meta or {}).get("terms")
                              or {}).items()},
                   "gated": diag.get("gated") or {},
                   "menu": diag.get("menu") or {},
                   "fade_reason": diag.get("fade_reason"),
                   "arc": round(self.arc_progress(), 3),
                   "theme": self.brain.theme.name,
                   "persona": self.brain.persona.name})

    def _note_pool_played(self, track):
        pool = self.brain.pool_ids
        if pool is None or track is None:
            return
        pool.discard(track.id)
        if not pool:
            self.brain.pool_ids = None
            self._setlist_name = None
            self._setlist_mode = "order"
            self._setlist_total_s = None      # night arc clock resumes
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
        ph, ci = self._arc_slot()
        self._history.append({"t": time.strftime("%H:%M"),
                              "title": self.current.title,
                              "artist": self.current.artist,
                              "via": self.plan["style"],
                              "phase": round(ph, 4), "cyc": ci,
                              "energy": round(
                                  self.current.energy_proxy(), 3)})
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
        # The landing was computed from the OLD playhead - after a jump it
        # points at a bar that no longer exists.
        self._cancel_moment("seek")
        self.submix.post({"cmd": "cue", "deck": self.active_deck,
                          "time_s": target})
        # NOTE: do NOT touch _started_clock - `played` tracks OUTPUT time the
        # track has been up, not its source position. Seeking forward must
        # not make the system think the track is 'done' and fire a mix (the
        # bug that left every jump stuck in the armed state).
        self._log({"event": "seek", "to_s": round(target, 1)})

    def _do_moment(self, flavor="nextdrop"):
        """OPERATOR MOMENT: there is exactly ONE - the set double-drops
        forward into the NEXT track's real drop (_moment_nextdrop).

        Four flavors shipped on 2026-07-29 and the operator's verdict
        the next day was final: 'only next is good', 'drop is awful',
        'stall is worthless', 'spinback is basically another next'.
        Every same-track gesture - build-and-resume, build-and-jump,
        echo stall, spinback dive - failed three consecutive rebuilds
        across three different sound designs, because the payoff was
        still the same song. The one gesture that ever landed is the
        one that changes the music. So: one button, one gesture, and
        anything it can't deliver it refuses OUT LOUD (_moment_skip
        flashes the reason on the button).

        `flavor` is accepted (old panels / queued events may still send
        drop/spinback/stall) and ignored - every press is a nextdrop.

        Recall contract: the gesture arms as a real transition, so a
        second press (or ABORT MIX) routes to _do_abort, which kills
        the incoming deck and restores the outgoing one wholesale. The
        pending landing pre-arms the visuals through outstate
        (dj_moment_eta / the dj_next_drop_eta ramp) and the landing is
        stamped as a HARD drop (see outstate_keys)."""
        if self._moment_clock > self.submix.clock:
            # Second press = take it back (the pending moment is an
            # armed transition; abort is its recall path).
            if self.state == "armed":
                self._do_abort(via="moment_recall")
            return
        if self.state == "armed":
            # A seam script owns both decks until the handover; layering
            # anything on top of it read as garbage every time it was
            # tried. Refuse, visibly.
            self._moment_skip("nextdrop", "mix in progress")
            return
        pos = self._pos_s()
        tel_d = (self.submix.telemetry or {}).get("decks", {})
        dk = tel_d.get(self.active_deck) or {}
        if pos is None or not dk.get("playing"):
            return
        rate = max(float(dk.get("rate") or 1.0), 1e-6)
        g0 = float(dk.get("gain", 1.0))
        if g0 <= 0.05:
            # Mid-fade or already ducked to nothing: there is no music
            # here to build out of.
            self._moment_skip("nextdrop", "deck not up")
            return
        period = max(float(self.current.period_s or 0.5), 0.15)
        beat = period / rate                     # OUTPUT seconds per beat
        self._moment_nextdrop(pos, rate, g0, period, beat,
                              self.active_deck)

    def _moment_skip(self, flavor, why):
        """Refuse a moment press LOUDLY: log it and stamp the denial so
        the panel can flash the reason on the button. A refusal the
        operator can't see is indistinguishable from a dead button."""
        self._moment_denied = (flavor, why, time.time())
        self._log({"event": "moment_skipped", "flavor": flavor, "why": why})

    def _snap_beat(self, t, period):
        """Snap a source time to the track's BEAT grid (nearest_downbeat
        gives bar lines; add whole beats from there)."""
        db = self.current.nearest_downbeat(t)
        return db + round((t - db) / max(period, 1e-6)) * period

    def _moment_landing(self, pos, rate, beat, period, flavor="drop"):
        """Shared drop/nextdrop schedule. Land on the 16-BEAT grid (the
        phrase grid and its midpoint), not on phrases alone: a 32-beat
        phrase is 16s at 120 bpm, so phrase-only landings made the
        operator wait up to a third of a minute for a button that is
        supposed to feel like a reflex. The shortest run-up that still
        reads as a build is 2 bars; the landing is the first grid point
        at least that far out, so the wait is 8-24 beats and the build
        spans ALL of it. Returns (t_hit, eta, hit) or None (logged)."""
        phrase = (self.current.phrase_beats or 32) * period    # source secs
        grid = min(phrase, 16 * period)
        min_build = 8.0 * beat
        anchor = self.current.nearest_phrase(pos + (min_build + 2.0 * beat)
                                             * rate)
        need = pos + (min_build - 0.05) * rate
        cands = [anchor + i * grid for i in range(-2, 9)]
        cands = [c for c in cands
                 if c >= need and c <= self.current.duration_s - 2.0]
        if not cands:
            self._moment_skip(flavor, "track ending")
            return None
        t_hit = min(cands)
        eta = (t_hit - pos) / rate
        return t_hit, eta, self.submix.clock + int(eta * RATE)

    def _next_drop_target(self):
        """Shared by nextdrop and spinback: (next_track, samples, drop_s)
        when the queued next is decoded and has a playable drop under
        the runway rule (entering the incoming track AT its drop with
        less than gate+ride left would arm the NEXT transition seconds
        after the slam); target None otherwise. EARLIEST playable drop:
        enter the track at its drop and keep the rest of it to ride."""
        nxt = self.next_track
        samples = None
        if nxt is not None:
            with self._decode_lock:
                samples = self._decoded.get(nxt.id)
            if samples is None and not getattr(self, "threaded", True):
                samples = self._decoded_samples(nxt)
        target = None
        if nxt is not None and samples is not None:
            pn = max(float(nxt.period_s or 0.5), 0.15)
            try:
                from lib.dj.features import drop_moments
                cands = [nxt.nearest_downbeat(d)
                         for d in drop_moments(nxt.sections or [])]
            except Exception:
                cands = []
            ok = [d for d in cands if 4.0 * pn < d
                  < nxt.duration_s - (PLAN_LEAD_S + 25.0
                                      + max(16.0 * pn, 30.0))]
            target = min(ok) if ok else None
        return nxt, samples, target

    def _moment_nextdrop(self, pos, rate, g0, period, beat, deck):
        """THE moment: DOUBLE-DROP THE SET. A build on the live track
        (sweep + trim push + roll + loop-roll + hole), landing on the
        INCOMING track's real drop, entered cold at full gain - the set
        jumps forward a whole track on one button, straight into its
        hottest bar. The only crowd gesture that survived four review
        rounds; every same-track flavor was cut ('only next is good').

        It arms as a real TRANSITION (plan style 'moment_nextdrop',
        state 'armed', swap_at = the landing), so _finish_swap does all
        the handover bookkeeping - history, brain notes, deck flip -
        and the recall is _do_abort (a second press routes there; the
        ABORT MIX button works too), which kills the incoming deck and
        restores the outgoing one wholesale. When the next isn't
        queued/decoded or has no playable drop it REFUSES with the
        reason on the button (same-track fallbacks are gone - they were
        the gestures the operator kept calling garbage)."""
        nxt, samples, target = self._next_drop_target()
        if target is None:
            if nxt is not None and samples is None:
                self._predecode(nxt)     # be ready for the next press
            self._moment_skip("nextdrop",
                              "no next queued" if nxt is None else
                              "next still decoding" if samples is None
                              else "next has no drop")
            return
        landing = self._moment_landing(pos, rate, beat, period, "nextdrop")
        if landing is None:
            return
        t_hit, eta, hit = landing
        build = eta
        build_beats = int(round(build / beat))
        hole = hit - int(beat * RATE)
        from lib.dj import fx as _fx
        # No synth riser (the whoosh verdict); the dying deck's own
        # sweep + trim + loop-roll carry the build, plus the snare roll.
        roll_beats = 8 if build_beats >= 12 else 4
        roll = {"at": hit - int(roll_beats * beat * RATE), "cmd": "fx_play",
                "samples": _fx.at_peak(_fx.make_roll(beat, roll_beats), 0.5)}
        impact = {"at": hit, "cmd": "fx_play",
                  "samples": _fx.at_peak(_fx.make_impact(), 0.85)}
        dur = build - 5.0 * beat
        anchor = self._snap_beat(pos + dur * rate, period)
        ev = [
            # -- the build on the DYING deck. Sweep starts at 30 Hz,
            # under the sub (a 45 Hz/q1.1 start measurably shoved a
            # bass-heavy master into the limiter); two filter events
            # because one set() call can't sweep (see SweepFilter.set).
            {"at": hit - int(build * RATE), "cmd": "filter", "deck": deck,
             "mode": "hp", "cutoff_hz": 30.0, "ramp_s": 0.0, "q": 1.0},
            {"at": hit - int(build * RATE), "cmd": "filter", "deck": deck,
             "cutoff_hz": 600.0, "ramp_s": build},
            {"at": hit - int(build * RATE), "cmd": "gain", "deck": deck,
             "value": min(1.3 * g0, 1.35), "ramp_s": 0.9 * build},
            roll,
            {"at": hit - int(5 * beat * RATE), "cmd": "loop", "deck": deck,
             "start_s": anchor, "end_s": anchor + period},
            {"at": hit - int(3 * beat * RATE), "cmd": "loop", "deck": deck,
             "start_s": anchor, "end_s": anchor + period / 2.0},
            {"at": hit - int(2 * beat * RATE), "cmd": "loop", "deck": deck,
             "start_s": anchor, "end_s": anchor + period / 4.0},
            {"at": hole, "cmd": "clear_loop", "deck": deck},
            {"at": hole, "cmd": "gain", "deck": deck, "value": 0.0,
             "ramp_s": 0.06},
            # -- the incoming deck: mount silent under the hole and
            # pre-roll the beat before ITS drop (cue one HOLE-length
            # back: it advances `beat` output seconds at rate 1.0).
            {"at": hole, "cmd": "load", "deck": self._other(deck),
             "samples": samples, "track_id": nxt.id, "grid": nxt.grid,
             "gain_db": nxt.gain_db, "kick_offset_s": nxt.kick_offset_s,
             "cue_s": target - beat},
            {"at": hole, "cmd": "gain", "deck": self._other(deck),
             "value": 0.0, "ramp_s": 0.01},
            {"at": hole, "cmd": "start", "deck": self._other(deck)},
            # -- the landing: THE NEXT TRACK'S DROP, cold ----------------
            {"at": hit, "cmd": "gain", "deck": self._other(deck),
             "value": 1.0, "ramp_s": 0.05},
            impact,
            {"at": hit + int(0.25 * RATE), "cmd": "stop", "deck": deck},
            # Insurance: the incoming deck must never stay muted.
            {"at": hit + int(beat * RATE), "cmd": "gain",
             "deck": self._other(deck), "value": 1.0, "ramp_s": 0.05},
        ]
        # Transition txn, not a moment txn: _do_abort recalls by _txn_id
        # and its recovery already restores gain/EQ/filter/rate/loop on
        # the outgoing deck and kills the incoming one.
        self._txn_id += 1
        for e in ev:
            e["txn"] = self._txn_id
        self.submix.post_many(ev)
        self.plan = {"style": "moment_nextdrop", "rate": 1.0,
                     "out_s": t_hit, "in_s": target, "beats": 0,
                     "pair_score": 0.0, "cand_id": nxt.id,
                     "no_return_at": hole}
        self.swap_at = hit               # _finish_swap flips ON the drop
        self.blend_at = self.submix.clock
        self.state = "armed"
        self._seam_metrics = None        # a cut, nothing to judge
        self._moment_clock = hit         # countdown + hole/hard-drop visuals
        self._moment_gain = g0
        self._moment_rate = rate
        self._moment_flavor = "nextdrop"
        self._moment_hole = (hole, hit)
        self._moment_txn = None          # recall is _do_abort, not txn-cancel
        self._log({"event": "moment", "flavor": "nextdrop",
                   "in_s": round(eta, 1), "build_s": round(build, 1),
                   "beats": build_beats, "payoff": "next",
                   "to_s": round(target, 1), "next": nxt.title})

    def _other(self, deck):
        return "b" if deck == "a" else "a"

    def _cancel_moment(self, why="cancel"):
        """Recall a txn-tagged MOMENT and put the deck back. The one
        live gesture (nextdrop) arms as a transition and recalls via
        _do_abort instead, so today this is a no-op guard - but the
        callers (_arm, _do_seek, the watchdog handoff) keep calling it
        so any future txn-tagged gesture is recalled by whatever takes
        over the live deck, never left half-fired."""
        if not self._moment_txn:
            return
        if self._moment_clock and \
                self.submix.clock > self._moment_clock + RATE:
            # Already landed and restored itself - nothing left to
            # recall, and re-posting the stored gain would stomp
            # whatever owns the deck now.
            self._moment_txn = None
            self._moment_flavor = None
            return
        deck = self.active_deck
        self.submix.post({"cmd": "cancel", "txn": self._moment_txn})
        # Queue order is preserved, so these land after the cancel and
        # restate whatever already fired (no-op if nothing had).
        self.submix.post_many([
            {"cmd": "filter", "deck": deck, "mode": "off"},
            {"cmd": "echo", "deck": deck, "active": False},
            {"cmd": "release_loop", "deck": deck},
            {"cmd": "gain", "deck": deck, "value": self._moment_gain,
             "ramp_s": 0.08},
            {"cmd": "rate", "deck": deck, "value": self._moment_rate,
             "ramp_s": 0.15},
        ])
        self._log({"event": "moment_cancel", "why": why,
                   "flavor": self._moment_flavor})
        self._moment_txn = None
        self._moment_clock = 0
        self._moment_flavor = None
        self._moment_hole = (0, 0)

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
        if self._moment_flavor == "nextdrop":
            # The recalled transition WAS the pending moment: clear its
            # countdown/hole state or the visuals keep pre-arming a
            # landing that will never come.
            self._moment_clock = 0
            self._moment_flavor = None
            self._moment_hole = (0, 0)
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
        self._cancel_moment("handoff")   # its hole would land mid-rescue
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
            # Planner style pin, keyed on the track id so it self-
            # invalidates if the pick is rerolled/substituted before arming.
            self._next_style_hint = (t.id, entry.get("style_override")) \
                if entry.get("style_override") else None
            self._log({"event": "setlist_next", "track": t.title,
                       "pin": entry.get("pin_type", "suggestion"),
                       "style_pin": entry.get("style_override"),
                       "remaining": len(self._setlist_queue)})
            return t, meta
        self._setlist_name = None
        self._setlist_mode = "order"
        self._setlist_total_s = None          # night arc clock resumes
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
        # PERSONA pacing: monk lets records breathe ~1.3x, showman rotates
        # ~0.8x. Neutral is exactly the legacy draw.
        self._exit_played = (theme.min_play_s + frac * span) \
            * getattr(self.brain.persona, "play_len_x", 1.0)

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

    def _verify_in_worker(self, mono):
        """Run the beat-grid measurement in the lib.dj.tempo_worker child.

        One long-lived subprocess, started on first use and kept for the
        night (a cold interpreter would otherwise land on every seam).
        Blocking here is free and is the whole point: this runs on the
        decode background thread, and waiting on a pipe RELEASES the GIL
        so the audio callback keeps its deadline.

        Deliberately NOT multiprocessing - see lib/dj/tempo_worker.py for
        why (spawn/forkserver both re-import the parent's __main__, i.e.
        all of Stories_OGL, inside the helper).

        Any failure falls back to computing in-process: a rare dropout
        beats losing the tempo check that keeps seams from flamming.
        """
        import pickle
        import struct
        hdr = struct.Struct("<Q")
        try:
            proc = self._tempo_pool
            if proc is None or proc.poll() is not None:
                import subprocess
                repo = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                proc = subprocess.Popen(
                    [sys.executable, "-m", "lib.dj.tempo_worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    cwd=repo)
                self._tempo_pool = proc
            blob = pickle.dumps(np.ascontiguousarray(mono),
                                protocol=pickle.HIGHEST_PROTOCOL)
            proc.stdin.write(hdr.pack(len(blob)))
            proc.stdin.write(blob)
            proc.stdin.flush()
            head = proc.stdout.read(hdr.size)
            if not head or len(head) < hdr.size:
                raise RuntimeError("tempo worker closed the pipe")
            (n,) = hdr.unpack(head)
            payload = b""
            while len(payload) < n:
                part = proc.stdout.read(n - len(payload))
                if not part:
                    raise RuntimeError("tempo worker truncated its reply")
                payload += part
            res = pickle.loads(payload)
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise RuntimeError(res[1])
            return res
        except Exception as e:
            print(f"[DJ] tempo verify worker unavailable ({type(e).__name__}"
                  f": {e}); measuring in-process")
            try:
                if self._tempo_pool is not None:
                    self._tempo_pool.kill()
            except Exception:
                pass
            self._tempo_pool = None
            try:
                from lib.dj.features import verify_tempo_window
                return verify_tempo_window(mono)
            except Exception:
                return None

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
        stored value. Dispatched from the decode background thread to a
        WORKER PROCESS (see _verify_in_worker) - it is ~1s of solid CPU
        landing seconds before a seam, and in-process it held the GIL long
        enough to starve the audio callback and drop the transition."""
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
            # OFF THE GIL ENTIRELY. This measurement is ~1s of solid CPU,
            # and it lands in the seconds before a seam (predecode time).
            # In-process - even chunked with sleeps - it starved the audio
            # callback: measured 123-323ms of callback block with this
            # function overlapping, i.e. an audible dropout right at the
            # transition. A worker PROCESS has its own GIL, so the parent
            # pays only the pickle of the window.
            res = self._verify_in_worker(mono)
            if res is None:
                return
            grid, bpm, conf = res
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
            # PERSIST the correction (queued: this runs on the decode
            # thread, the DB belongs to the planner thread - same pattern
            # as the auto seam-feedback writes). Confident reads only; the
            # 0.6% dev floor above already prevents churn on re-measures
            # of an accepted correction.
            if conf >= 0.6:
                with self._lock:
                    self._pending.append(("tempo_writeback", {
                        "track_id": track.id, "bpm": round(best, 3),
                        "grid": self._grid_fix[track.id]["grid"],
                        "conf": round(conf, 2),
                        "dev_pct": round(dev * 100.0, 2)}))
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
                # sleep(0) only offers to yield - CPython can hand the GIL
                # straight back to this thread, so the audio callback never
                # gets in. A real (tiny) sleep forces the release. ~25
                # chunks on a 5-min track = ~12ms added to a background
                # decode that has ~20s of lead.
                time.sleep(0.0005)               # yield to the audio callback
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
