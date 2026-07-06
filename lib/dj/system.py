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

from lib.dj.brain import Brain, load_library, TrackInfo
from lib.dj.db import LibraryDB
from lib.dj.submix import DJSubmix
from lib.dj.themes import BUILTIN_THEMES, get_theme

RATE = 44100
PLAN_LEAD_S = 45.0               # start choosing next this early
MIN_LEAD_S = 8.0                 # never arm closer than this to the seam
SET_CYCLE_S = 90 * 60.0          # non-all-night themes loop their arc here


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
        self._arc_waypoints = []         # [(progress 0..1, energy 0..1)]
        self._horizon = []               # provisional next-N briefs
        self._horizon_key = None         # staleness stamp
        self._history = []               # tonight's tracklist briefs
        self._last_style = None          # last completed transition style
        self._played_energy_ema = None   # arc feedback: what actually played
        self._exit_played = 300.0    # drawn per track from theme min/max play
        self._next_meta = None
        self._setlist_name = None
        self._setlist_queue = []     # upcoming entry dicts (plan-following)
        self.setlist_names = []      # for the web picker, refreshed by step()
        self._setlists_checked = 0.0
        self._decoded = {}           # track_id -> stereo samples (RAM cache)
        self._decoded_order = []
        self._decoding = set()
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
        lib = load_library(self.db)
        if not lib:
            self.last_error = "library is empty - run tools/dj_scan.py"
            print(f"[DJ] {self.last_error}")
            return False
        self.brain = Brain(lib, get_theme(self._theme_name), seed=self._seed,
                           stretch_max=self._stretch_max)
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
            user_counts = {}
            auto_counts = {}
            for t in self.brain.library:
                t.user_tags = per_track.get(t.id, [])
                for tag in t.user_tags:
                    user_counts[tag] = user_counts.get(tag, 0) + 1
                for tag in t.auto_tags:
                    auto_counts[tag] = auto_counts.get(tag, 0) + 1
            vocab = [(tag, n, True) for tag, n in
                     sorted(user_counts.items(), key=lambda kv: -kv[1])]
            vocab += [(tag, n, False) for tag, n in
                      sorted(auto_counts.items(), key=lambda kv: -kv[1])
                      if tag not in user_counts]
            self._tag_vocab = vocab[:64]     # everything, sane ceiling
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

    def request_next(self, track_id):
        with self._lock:
            self._pending.append(("next_id", track_id))

    def set_autopilot(self, on):
        self.autopilot = bool(on)

    def set_energy_nudge(self, x):
        self._energy_nudge = float(max(-0.4, min(0.4, x)))

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

    def load_setlist(self, name):
        """Follow a preplanned set: anchors are hard, suggestions soft."""
        with self._lock:
            self._pending.append(("setlist", name))

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
        target = self._arc_base(self.arc_progress()) + self._energy_nudge
        # ARC FEEDBACK: if what actually PLAYED has been running hotter or
        # cooler than the theme's arc (library gaps, anchor picks), lean
        # the next choice the other way instead of undershooting all night.
        if self._played_energy_ema is not None:
            target += max(-0.15, min(0.15,
                                     0.6 * (target - self._played_energy_ema)))
        return max(0.0, min(1.0, target))

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
        return {"dj_active": self._running,
                "dj_arc_phase": self.arc_progress(),
                "dj_arc_heat": self.arc_target(),
                "dj_next_drop_eta": eta,
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
            "tags": self._tag_vocab,
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
            "current": self._track_brief(cur),
            "next": self._track_brief(nxt),
            "style": self.plan["style"] if self.plan else None,
            "blend_in_s": round(countdown, 1) if countdown is not None else None,
            "setlist": self._setlist_name,
            "setlist_remaining": len(self._setlist_queue),
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
                self.brain.theme = get_theme(val)
                self._theme_name = val
                self._log({"event": "theme", "theme": val})
                if self.state == "playing":
                    self.next_track = None       # soft replan
                    self.plan = None
            elif kind == "flavor":
                # Live music-type steering: tag leans + axis pulls merged
                # over the theme; soft replan so the very next pick obeys.
                self.brain.set_flavor(val)
                self._log({"event": "flavor", "flavor": val})
                if self.state == "playing":
                    self.next_track = None
                    self.plan = None
            elif kind == "skip" and self.state in ("playing", "armed"):
                self._do_skip()
            elif kind == "next_id":
                t = next((x for x in self.brain.library if x.id == val), None)
                if t is not None and self.state == "playing":
                    self.next_track = t
                    self.plan = None
                    self._log({"event": "pick_next", "track": t.title})
            elif kind == "setlist":
                from lib.dj.setlist import get_setlist
                sl = get_setlist(self.db, name=val) if val else None
                if val and sl is None:
                    self.last_error = f"setlist '{val}' not found"
                else:
                    self._setlist_name = val or None
                    self._setlist_queue = list(sl["entries"]) if sl else []
                    self._log({"event": "setlist",
                               "name": self._setlist_name,
                               "tracks": len(self._setlist_queue)})
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
            self._maybe_arrange()
            self._maybe_plan()
            self._maybe_horizon()
        elif self.state == "armed":
            if self.swap_at is not None and self.submix.clock >= self.swap_at:
                self._finish_swap()

    def _pick_next(self, out_bpm):
        """The live pick FOLLOWS the displayed queue: take the horizon's
        front if it still scores, so what the operator sees is what
        plays. Fresh selection only when the queue is empty/invalid."""
        if self._horizon and self.brain is not None:
            t0 = next((t for t in self.brain.library
                       if t.id == self._horizon[0]["id"]), None)
            if t0 is not None and t0.id != self.current.id                     and t0.id not in self.brain.veto_ids:
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
                 tuple(self._arc_waypoints), round(self._energy_nudge, 2))
        steered = steer != self._horizon_key
        if not steered and len(self._horizon) >= 3                 and (self.next_track is None
                     or self._horizon[0]["id"] == self.next_track.id):
            return
        if steered:
            self._horizon = []           # steering changed: replan the lot
        self._horizon_key = steer
        prog0 = self.arc_progress()
        step = 300.0 / (SET_CYCLE_S if (self.brain.theme.arc != "all_night")
                        else max(self.night_hours * 3600.0, 60.0))

        def arc_at(i):
            return max(0.0, min(1.0, self._arc_base(
                min(prog0 + i * step, 1.0)) + self._energy_nudge))
        try:
            by_id = {t.id: t for t in self.brain.library}
            kept = [h for h in self._horizon if h["id"] in by_id][:3]
            need = 3 - len(kept)
            if need > 0:
                tail = by_id[kept[-1]["id"]] if kept else self.current
                pre = [by_id[h["id"]] for h in kept]
                kept += self.brain.plan_horizon(
                    tail, arc_at, tail.bpm, n=need, preplayed=pre)
            self._horizon = kept
            # PROJECTED TIMELINE: when each queued track will actually
            # start (seconds from now) and how long it should run, so the
            # night chart can place them time-true instead of decoratively.
            pos = self._pos_s()
            played = (self.submix.clock - self._started_clock) / RATE
            if self.plan and pos is not None:
                t0 = max(self.plan["out_s"] - pos, 0.0)
            else:
                t0 = max(self._exit_played - played, 30.0)
            per_play = min(max(self.brain.theme.min_play_s, 150.0), 300.0)
            for h in self._horizon:
                h["eta_s"] = round(t0, 1)
                h["play_s"] = round(per_play, 1)
                t0 += per_play
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
        except Exception as e:
            print(f"[DJ] horizon skipped: {e}")

    def _maybe_arrange(self):
        """LIVE MICRO-ARRANGEMENT: when the current track sits in a long,
        highly repetitive groove with plenty of runway left, jump forward
        exactly one phrase (a 32-beat cut preserves beat/bar/phrase phase,
        so it's rhythmically seamless) - the radio-edit move a DJ does with
        hot cues. At most once per track, never while a transition is
        armed, never near the planned exit."""
        cur = self.current
        if cur is None or self.next_track is not None:
            return
        if getattr(self, "_arranged_track", None) == cur.id:
            return
        if cur.phrase_beats <= 0 or cur.phrase_conf < 0.1:
            return
        tel = self.submix.telemetry
        deck = (tel or {}).get("decks", {}).get(self.active_deck)
        if not deck or not deck.get("playing"):
            return
        pos = deck["time_s"]
        played = (self.submix.clock - self._started_clock) / RATE
        span = cur.phrase_beats * cur.period_s
        # Runway: the jump must not swallow the planned exit region.
        if played < 75.0 or pos + 2 * span > cur.duration_s - 60.0:
            return
        sec = cur.section_at(pos)
        if (sec and sec.get("kind") == "groove"
                and sec.get("repetitiveness", 0.0) > 0.8
                and sec["end_s"] - sec["start_s"] > 75.0
                and pos - sec["start_s"] > 30.0
                and sec["end_s"] - pos > span + 15.0):
            self.submix.post({"cmd": "jump", "deck": self.active_deck,
                              "time_s": pos + span})
            self._arranged_track = cur.id
            self._log({"event": "phrase_jump", "track": cur.title,
                       "from_s": round(pos, 1), "beats": cur.phrase_beats})

    def _start_first(self):
        first = None
        while self._setlist_queue and first is None:
            entry = self._setlist_queue.pop(0)
            first = next((x for x in self.brain.library
                          if x.id == entry["track_id"]), None)
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
        out_bpm = self.current.bpm * live_rate

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
            if cand is None:
                self.last_error = "no compatible next track"
                return
            self.next_track = cand
            self._next_meta = meta
        else:
            _, self._next_meta = self.brain.score(
                self.current, self.next_track, self.arc_target(), out_bpm)
            if self._next_meta is None:
                self._next_meta = {"rate": 1.0, "eff_bpm": self.next_track.bpm,
                                   "pair": None}
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
        after = pos + max(self._exit_played - played, MIN_LEAD_S)
        plan = self.brain.plan_transition(self.current, self.next_track,
                                          self._next_meta,
                                          after_s=min(after, deadline))
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
        self.submix.post({"cmd": "load", "deck": incoming, "samples": samples,
                          "track_id": self.next_track.id,
                          "grid": self.next_track.grid,
                          "gain_db": self.next_track.gain_db,
                          "kick_offset_s": self.next_track.kick_offset_s,
                          "pitch_st": plan.get("pitch_st", 0),
                          "cue_s": plan["in_s"]})
        events, swap_at, blend_at = self.brain.build_events(
            plan, self.submix.telemetry, self.active_deck, incoming,
            self.current, self.next_track)
        self.submix.post_many(events)
        self.plan = plan
        self.swap_at = swap_at
        self.blend_at = blend_at
        self.state = "armed"
        self._urgent_exit = False
        self._log({"event": "armed", "style": plan["style"],
                   "next": self.next_track.title,
                   "rate": round(plan["rate"], 4),
                   "out_s": round(plan["out_s"], 2),
                   "in_s": round(plan["in_s"], 2),
                   "pair_score": plan["pair_score"],
                   "blend_in_s": round((blend_at - self.submix.clock) / RATE, 1)})

    def _finish_swap(self):
        if self._history_id:
            self.db.log_play_end(self._history_id)
        old = self.current
        self.current = self.next_track
        self.next_track = None
        self.active_deck = "b" if self.active_deck == "a" else "a"
        self._started_clock = self.submix.clock
        self._draw_exit()
        self.brain.note_played(self.current)
        self._note_energy(self.current)
        self._history_id = self.db.log_play_start(
            self.current.id, transition_style=self.plan["style"],
            theme=self._theme_name)
        self._last_style = self.plan["style"]
        self._last_pair = (old.id if old else None, self.current.id)
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

    def _do_skip(self):
        """Exit at the earliest musical opportunity."""
        if self.state == "armed":
            return                        # transition already rolling
        if self._history_id:
            self.db.log_play_end(self._history_id, skipped=True)
            self._history_id = None
        if self.current is not None:
            self.brain.note_skipped(self.current)
        self._urgent_exit = True
        self._log({"event": "skip", "track":
                   self.current.title if self.current else None})
        # Force planning NOW with an exit a few bars out.
        self.next_track = None
        self.plan = None
        self._exit_played = (self.submix.clock - self._started_clock) / RATE
        self._maybe_plan()

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
                rate, eff = self.brain.rate_for(out_bpm, t)
                meta = {"rate": rate or 1.0, "eff_bpm": eff or t.bpm,
                        "pair": None}
                if rate is None and entry.get("pin_type") != "anchor":
                    self._log({"event": "setlist_swap", "dropped": t.title,
                               "why": "tempo clash - brain substitutes"})
                    return None, None       # brain picks instead
            self._log({"event": "setlist_next", "track": t.title,
                       "pin": entry.get("pin_type", "suggestion"),
                       "remaining": len(self._setlist_queue)})
            return t, meta
        self._setlist_name = None
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
        theme = self.brain.theme
        span = max(theme.max_play_s - theme.min_play_s, 1.0)
        self._exit_played = theme.min_play_s + self.brain.rng.random() * span

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
            with self._decode_lock:
                self._decoding.discard(track.id)
                if s is not None:
                    self._decoded[track.id] = s
                    # Keep only a couple of tracks cached (~106MB each).
                    while len(self._decoded_order) >= 3:
                        old = self._decoded_order.pop(0)
                        if old != track.id:
                            self._decoded.pop(old, None)
                    self._decoded_order.append(track.id)
        threading.Thread(target=work, daemon=True).start()

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
                with self._decode_lock:
                    self._decoded[track.id] = s
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
