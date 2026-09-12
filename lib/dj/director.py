"""The Director: high-level behaviour over the two engines that work.

The autoDJ (lib/dj/system.py: one song at a time, seams through the gates) stays exactly as it is. The
stem conductor (lib/dj/remix.py: parts of two or three songs together) is the other engine. The Director
owns one of them at a time on the show's AudioEngine and translates SIX DIALS into what they do:

    songs    a theme, a playlist as the pool, and an optional UP NEXT list (honoured when it can be)
    mixing   auto | blend | cut | morph         the seam family (pins through the gates) / the crossfade
    layers   one | two | three                  the autoDJ, or the conductor with two / three songs live
    energy   cool | hold | amp                  the arc lean (and, amped, a tempo journey when layered)
    pace     short | normal | long              how long records play / how often the lanes move
    loops    off | some | lots                  the loop styles' odds / loop moves

plus four moments - NEXT, HOLD, DROP, BREAK - and a verdict on what just happened. The operator never
touches songs' parts, timing or levels. Every dial takes effect at the next musical opportunity.

Switching layers between one and two/three HANDS THE PLAYING SONG ACROSS: the other engine opens on the
same song at the same song time on a bar of the song, and the two master buses crossfade over a beat, so
nothing stops. The Director ticks both engines itself (neither runs its own thread here), and nothing in
this file opens a device: the caller attaches the submixes it is told about.
"""
import threading
import time

from lib.dj.submix import RATE

DIALS = {
    "mixing": ("auto", "blend", "cut", "morph"),
    "layers": ("one", "two", "three"),
    "energy": ("cool", "hold", "amp"),
    "pace": ("short", "normal", "long"),
    "loops": ("off", "some", "lots"),
}
DEFAULTS = {"mixing": "auto", "layers": "one", "energy": "hold", "pace": "normal", "loops": "off"}
MIX_PIN = {"auto": None, "blend": "long_blend", "cut": "cut_at_drop", "morph": "stem_morph"}
MIX_CROSS = {"auto": 2.0, "blend": 4.0, "cut": 0.5, "morph": 2.0}
ENERGY_LEAN = {"cool": -0.25, "hold": 0.0, "amp": 0.25}
PACE_X = {"short": 0.6, "normal": 1.0, "long": 1.5}
PACE_BARS = {"short": 4, "normal": 8, "long": 16}
LOOP_LEVEL = {"off": 0, "some": 1, "lots": 2}
LAYER_BLEND = {"two": 0.3, "three": 0.85}
HANDOVER_BEATS = 1.0


class Director:
    def __init__(self, music_root, engine=None, theme="groove", attach=None):
        """`attach(key, submix)` mounts a submix on the audio engine (the tab passes engine.attach_track;
        the gate passes its own). Library and DB are opened here once and shared with both engines."""
        from lib.dj.db import LibraryDB
        from lib.dj import brain as B
        self.music_root = music_root
        self.engine = engine
        self._attach = attach or (engine.attach_track if engine is not None else (lambda k, s: None))
        self.db = LibraryDB(music_root)
        self.library = [t for t in B.load_library(self.db) if not t.excluded]
        self.theme = theme
        self.dials = dict(DEFAULTS)
        self.pool_name = None
        self.up_next = []                # track ids, in order
        self.system = None               # the autoDJ, when layers = one
        self.rc = None                   # the conductor, when layers = two / three
        self.mode = None                 # "one" | "layered"
        self._switch = None              # a handover in progress
        self._hold_until = None
        self._fed = None                 # (current id, queued id) fed to the autoDJ
        self._loop_k = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self.events = []                 # (hms, text) the Director's own log
        self.last_error = None
        # THE PICTURE: what played on each lane, when, and what is planned ahead - kept as a Timeline
        # (lib/dj/timeline.py) so the same canvas draws it: real clips behind the playhead, ghosts ahead
        from lib.dj.timeline import Timeline
        self.tl = Timeline("director")
        self._bar = 0.0                  # the Director's bar clock across engines (display)
        self._bar_wall = None
        self._lane_snap = {}
        self._cur_song = None
        self._ghosts = []                # clip ids of the current plan ahead

    # -- lifecycle ---------------------------------------------------------------------------------------
    def start(self, threaded=True):
        want = "one" if self.dials["layers"] == "one" else "layered"
        try:
            if want == "one":
                self.system = self._new_system()
            else:
                self.rc = self._new_conductor()
                self.rc.start(threaded=False)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return False
        self.mode = want
        self.running = True
        self._event(f"started: {self.describe_dials()}")
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True, name="director")
            self._thread.start()
        return True

    def stop(self, fade_s=1.5):
        self.running = False
        if self.system is not None:
            try:
                self.system.stop(fade_s=fade_s)
            except Exception:
                pass
            self.system = None
        if self.rc is not None:
            try:
                self.rc.stop(fade_s=fade_s)
            except Exception:
                pass
            self.rc = None

    def _run(self):
        while self.running:
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                import traceback
                traceback.print_exc()
            time.sleep(0.2)

    # -- engines -------------------------------------------------------------------------------------------
    def _new_system(self, opener=None):
        """The autoDJ as it runs live (its own thread, background decodes). For a handover it is built
        unthreaded so the Director can fire its first step - the opener - on the chosen bar, then its
        thread is spawned and it runs exactly as live from there (see _spawn)."""
        from lib.dj.system import DJSystem
        sysm = DJSystem(self.music_root, engine=None, theme=self.theme, threaded=opener is None)
        if opener is not None:
            sysm.set_opener(*opener)
        sysm._loop_bias = LOOP_LEVEL[self.dials["loops"]]
        self._apply_to_system(sysm)
        if not sysm.start():
            raise RuntimeError(sysm.last_error or "autoDJ failed to start")
        self._attach("dj_submix", sysm.submix)
        self._apply_to_system(sysm)
        return sysm

    @staticmethod
    def _spawn(sysm):
        """Give an unthreaded autoDJ its live thread after its first step."""
        if sysm.threaded:
            return
        sysm.threaded = True
        sysm._thread = threading.Thread(target=sysm._run, daemon=True, name="dj-brain")
        sysm._thread.start()

    def _new_conductor(self):
        from lib.dj.remix import RemixConductor
        rc = RemixConductor(self.db, self.music_root, self.library, theme=self.theme)
        self._attach("dj_remix", rc.submix)
        self._apply_to_conductor(rc)
        return rc

    def _apply_to_system(self, sysm):
        d = self.dials
        sysm.set_mix_style(MIX_PIN[d["mixing"]])
        sysm.set_energy_nudge(ENERGY_LEAN[d["energy"]])
        sysm.set_pace(PACE_X[d["pace"]])
        sysm.set_loop_bias(LOOP_LEVEL[d["loops"]])
        if self.pool_name:
            sysm.load_setlist(self.pool_name, mode="pool")

    def _apply_to_conductor(self, rc):
        d = self.dials
        rc.set_auto(1.0)
        rc.set_cross_beats(MIX_CROSS[d["mixing"]])
        rc.set_blend(LAYER_BLEND.get(d["layers"], 0.3))
        rc.set_energy_lean(ENERGY_LEAN[d["energy"]])
        rc.set_tempo_span(0.03 if d["energy"] == "amp" else 0.0)
        rc.set_change_bars(PACE_BARS[d["pace"]])
        rc.set_vocal_freedom(0.6 if d["layers"] == "three" else 0.4)
        if self.pool_name:
            from lib.dj.setlist import get_setlist
            sl = get_setlist(self.db, name=self.pool_name)
            rc.set_pool([e["track_id"] for e in (sl or {}).get("entries", [])])
            rc.pool_name = self.pool_name

    # -- the dials ------------------------------------------------------------------------------------------
    def set_dial(self, name, value):
        if name not in DIALS or value not in DIALS[name]:
            return False
        if self.dials.get(name) == value:
            return True
        self.dials[name] = value
        self._event(f"{name}: {value}")
        if self.system is not None:
            self._apply_to_system(self.system)
        if self.rc is not None:
            self._apply_to_conductor(self.rc)
        return True

    def set_theme(self, name):
        self.theme = name
        if self.system is not None:
            self.system.set_theme(name)
        if self.rc is not None:
            self.rc.set_theme(name)
        self._event(f"theme: {name}")

    def set_pool(self, name):
        self.pool_name = name or None
        if self.system is not None:
            self.system.load_setlist(name or "", mode="pool")
        if self.rc is not None:
            if name:
                self._apply_to_conductor(self.rc)
            else:
                self.rc.set_pool(None)
        self._event(f"songs: {name or 'whole library'}")

    def queue(self, track_id):
        tid = int(track_id)
        if tid not in self.up_next:
            self.up_next.append(tid)
            self._event(f"up next: {self.title(tid)}")

    def unqueue(self, track_id):
        self.up_next = [t for t in self.up_next if t != int(track_id)]

    def clear_queue(self):
        self.up_next = []

    def describe_dials(self):
        d = self.dials
        return (f"{d['layers']} song{'s' if d['layers'] != 'one' else ''} · {d['mixing']} mixing · energy {d['energy']} · "
                f"{d['pace']} plays · loops {d['loops']}")

    # -- the moments -----------------------------------------------------------------------------------------
    def next(self):
        if self.system is not None:
            self.system.request_skip()
        if self.rc is not None:
            self.rc.next_move()
        self._event("NEXT")

    def hold(self):
        if self.system is not None:
            self.system.request_hold()
        if self.rc is not None:
            self.rc.set_hold(True)
            self._hold_until = self.rc.bar_n + self.rc.change_bars
        self._event("HOLD")

    def drop(self):
        if self.system is not None:
            self.system.moment("drop")
        if self.rc is not None:
            self.rc.drop()
        self._event("DROP")

    def break_(self):
        if self.rc is not None:
            self.rc.break_()
            self._event("BREAK")
            return True
        return False

    def rate(self, up):
        if self.system is not None:
            self.system.seam_feedback(bool(up))
        if self.rc is not None:
            self.rc.rate_last(bool(up))
        self._event("GOOD" if up else "BAD")

    # -- the tick ---------------------------------------------------------------------------------------------
    def step(self):
        if not self.running:
            return
        if self.system is not None and not self.system.threaded:
            self.system.step()
        if self.rc is not None:
            self.rc.step()
            if self._hold_until is not None and self.rc.bar_n >= self._hold_until:
                self.rc.set_hold(False)
                self._hold_until = None
            self._loops_layered()
        self._feed_up_next()
        self._handover()
        try:
            self._record()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"picture: {type(e).__name__}: {e}"

    def _feed_up_next(self):
        """The UP NEXT list: the head goes to the autoDJ as its next track (once per song), or is staged
        for the conductor when a deck is free; popped when it plays."""
        if not self.up_next:
            return
        head = self.up_next[0]
        if self.system is not None and self.system.current is not None:
            cur = self.system.current.id
            if cur == head:
                self.up_next.pop(0)
                self._fed = None
                return
            if self._fed != (cur, head):
                self.system.request_next(head)
                self._fed = (cur, head)
        elif self.rc is not None and self.rc.master is not None:
            d = self.rc.deck_of(head)
            if d is not None and self.rc.songs[d].entered:
                self.up_next.pop(0)
                return
            if d is None:
                ok, msg = self.rc.stage_track(head)
                if not ok and msg not in ("no free deck - eject one first", "already on a deck"):
                    self._event(f"up next {self.title(head)}: {msg}")
                    self.up_next.pop(0)

    def _loops_layered(self):
        """The loops dial in layered mode: now and then the master holds a phrase of four bars."""
        lvl = LOOP_LEVEL[self.dials["loops"]]
        rc = self.rc
        if lvl == 0 or rc.master is None or rc.user_loop_bars:
            return
        if self._loop_k is None or rc.bar_n - self._loop_k >= rc.change_bars * (4 if lvl == 1 else 2):
            self._loop_k = rc.bar_n
            if rc.rng.random() < (0.35 if lvl == 1 else 0.7):
                rc.song_loop(rc.master, 4)
                self._event("loop: the clock holds four bars")
                threading.Timer(4 * 4 * rc._beat_s(), lambda: rc.song_loop(rc.master, None) if rc.master else None).start()

    # -- the handover between engines --------------------------------------------------------------------------
    def _bar_clock_of(self, sub, track, deck):
        """(clock, song_time) of the next bar of `track` on `deck` of submix `sub`, at least 0.6 s ahead."""
        import math
        tel = sub.telemetry or {}
        d = (tel.get("decks") or {}).get(deck) or {}
        now = int(tel.get("clock", sub.clock))
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        seg = next((s for s in track.grid or [] if s["start_s"] <= time_s <= s["end_s"]), (track.grid or [None])[0])
        if seg is None:
            return now + int(0.6 * RATE), time_s + 0.6 * rate
        bar = 4 * seg["period_s"]
        first_down = seg["first_beat_s"] + track.downbeat_offset * seg["period_s"]
        t_min = time_s + 0.6 * rate
        k = math.ceil((t_min - first_down) / bar - 1e-6)
        t_next = first_down + k * bar
        return now + int((t_next - time_s) / rate * RATE), t_next

    def _handover(self):
        want = "one" if self.dials["layers"] == "one" else "layered"
        if self._switch is None:
            if want == self.mode:
                return
            if want == "layered" and self.system is not None and self.system.current is not None \
                    and self.system.state == "playing":
                rc = self._new_conductor()
                rc.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": 0.0})
                rc.start(first_track=self.system.current, threaded=False, cue_s=0.0, lanes=set(("drums", "bass", "other", "vocals")), hold_open=True)
                self._switch = {"to": "layered", "rc": rc, "stage": "decoding", "t": time.time()}
                self._event(f"layers: handing {self.system.current.title} to the conductor")
            elif want == "one" and self.rc is not None and self.rc.master is not None:
                master = self.rc.songs[self.rc.master].track
                self._switch = {"to": "one", "track": master, "stage": "decoding", "t": time.time()}

                def work():
                    try:
                        from lib.dj.features import decode_file_stereo
                        self._switch["samples"] = decode_file_stereo(self.db.abs(master.path))
                        self._switch["stage"] = "ready"
                    except Exception as e:  # noqa: BLE001
                        self._switch["error"] = str(e)
                threading.Thread(target=work, daemon=True).start()
                self._event(f"layers: handing {master.title} back to the autoDJ")
            return
        sw = self._switch
        if sw.get("error") or time.time() - sw["t"] > 60:
            self._event(f"handover abandoned: {sw.get('error') or 'timed out'}")
            self._switch = None
            return
        if sw["to"] == "layered":
            rc = sw["rc"]
            if sw["stage"] == "decoding":
                rc.step()
                if not rc.decoded():
                    return
                sysm = self.system
                T_sys, song_t = self._bar_clock_of(sysm.submix, sysm.current, sysm.active_deck)
                T_rc = rc.submix.clock + (T_sys - sysm.submix.clock)
                beat = 60.0 / max(sysm.current.bpm, 60.0)
                rc.open_pending(T_rc, song_t)
                sysm.submix.post({"at": T_sys, "cmd": "mix_gain", "value": 0.0, "ramp_s": HANDOVER_BEATS * beat})
                rc.submix.post({"at": T_rc, "cmd": "mix_gain", "value": 1.0, "ramp_s": HANDOVER_BEATS * beat})
                sw["stage"], sw["done_at"] = "crossing", T_sys + int((HANDOVER_BEATS * beat + 0.3) * RATE)
                return
            rc.step()
            if self.system.submix.clock >= sw["done_at"]:
                old = self.system
                self.system = None
                self.rc = rc
                self.mode = "layered"
                self._switch = None
                try:
                    old.stop(fade_s=0.05)
                except Exception:
                    pass
                self._event("layers: the conductor has the room")
        else:
            if sw["stage"] != "ready":
                return
            rc = self.rc
            # the conductor collapses to its clock's song first (every lane to the master, a cut)
            if sw.get("collapsed") is None:
                for ln in ("drums", "bass", "other", "vocals"):
                    if rc.lanes.get(ln) != rc.master:
                        rc.assign(ln, rc.master, force=True)
                sw["collapsed"] = rc.bar_n
                return
            if rc.bar_n < sw["collapsed"] + 2:
                return
            if sw.get("sys") is None:
                T_rc, song_t = self._bar_clock_of(rc.submix, sw["track"], rc.master)
                sysm = self._new_system(opener=(sw["track"], song_t, sw["samples"]))
                sysm.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": 0.0})
                sw["sys"], sw["T_rc"], sw["song_t"] = sysm, T_rc, song_t
                return
            sysm = sw["sys"]
            lead = int(0.05 * RATE)
            if rc.submix.clock < sw["T_rc"] - lead:
                return
            beat = 60.0 / max(sw["track"].bpm, 60.0)
            sysm.step()                                    # the opener starts now, at song time song_t
            self._spawn(sysm)                              # and from here the autoDJ runs as it does live
            sysm.submix.post({"cmd": "mix_gain", "value": 1.0, "ramp_s": HANDOVER_BEATS * beat})
            rc.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": HANDOVER_BEATS * beat})
            old = rc
            self.rc = None
            self.system = sysm
            self.mode = "one"
            self._switch = None
            threading.Timer(HANDOVER_BEATS * beat + 0.4, lambda: old.stop(fade_s=0.05)).start()
            self._event("layers: the autoDJ has the room")

    # -- the picture: the run as a timeline ----------------------------------------------------------------------------
    def _bpm(self):
        if self.system is not None and self.system.current is not None:
            return float(self.system.current.bpm or 120.0)
        if self.rc is not None and self.rc.master_bpm:
            return float(self.rc.master_bpm)
        return 120.0

    def bar(self):
        """The Director's bar clock: advances with wall time at the playing tempo, across engines."""
        now = time.time()
        if self._bar_wall is not None and self.running:
            self._bar += (now - self._bar_wall) * self._bpm() / 240.0
        self._bar_wall = now
        return self._bar

    def _bar_len(self, tid):
        t = next((x for x in self.library if x.id == tid), None)
        return 4 * t.period_s if t is not None else 2.0

    def _record(self):
        """Write what is heard onto the Director's timeline (real clips) and the plan ahead (ghosts)."""
        b = int(self.bar())
        LANES = ("drums", "bass", "other", "vocals")
        for cid in self._ghosts:
            c = next((x for x in self.tl.clips if x.id == cid), None)
            if c is not None:
                self.tl.clips.remove(c)
        self._ghosts = []
        if self.system is not None:
            sysm = self.system
            cur = sysm.current
            if cur is None:
                return
            tel = (sysm.submix.telemetry or {}).get("decks", {}).get(sysm.active_deck) or {}
            pos = float(tel.get("time_s") or 0.0)
            if self._cur_song != cur.id:
                self._cur_song = cur.id
                self.tl.add(cur.id, list(LANES), b, max(0.0, pos - (self.bar() - b) * self._bar_len(cur.id)))
                self._lane_snap = {ln: cur.id for ln in LANES}
            st = None
            nxt = sysm.next_track
            if nxt is not None:
                try:
                    st = sysm.status()
                except Exception:
                    st = None
                eta = (st or {}).get("blend_in_s")
                plan = (st or {}).get("plan") or {}
                at = b + (int(eta / self._bar_len(cur.id)) if eta is not None else 24)
                for c in self.tl.add(nxt.id, list(LANES), max(b + 1, at), float(plan.get("in_s") or 0.0), ghost=True):
                    self._ghosts.append(c.id)
        elif self.rc is not None:
            rc = self.rc
            if rc.master is None:
                return
            for ln in LANES:
                d = rc.lanes.get(ln)
                s = rc.songs.get(d) if d else None
                tid = s.track.id if s is not None else None
                if self._lane_snap.get(ln, "unset") != tid:
                    self._lane_snap[ln] = tid
                    if tid is None:
                        from lib.dj.timeline import Clip
                        prev = self.tl.active(ln, b - 1)
                        self.tl.clips.append(Clip(prev.track_id if prev else 0, ln, b, 0.0, end_bar=None, rest=True))
                    else:
                        pos = float(rc._tel_deck(d).get("time_s") or 0.0)
                        self.tl.add(tid, [ln], b, pos)
            # the plan ahead: staged songs enter on the next move
            left = max(0, rc.change_bars - 1 - rc.phrase_bars)
            for d, s in rc.songs.items():
                if s.staged_at is not None and not s.entered and not s.leaving:
                    lane = rc._lane_to_give(d, rc._next_bar_clock()) or "drums"
                    for c in self.tl.add(s.track.id, [lane], b + 1 + left, rc._song_time_at(d, rc._next_bar_clock()) + left * self._bar_len(s.track.id), ghost=True):
                        self._ghosts.append(c.id)

    # -- what the operator sees ---------------------------------------------------------------------------------------
    def title(self, tid):
        t = next((x for x in self.library if x.id == tid), None)
        return t.title if t else f"#{tid}"

    def status(self):
        out = {"mode": self.mode, "dials": dict(self.dials), "theme": self.theme, "pool": self.pool_name,
               "up_next": [{"id": t, "title": self.title(t)} for t in self.up_next],
               "switching": None if self._switch is None else self._switch.get("to"),
               "events": list(self.events[-12:]), "error": self.last_error, "now": None, "next": None,
               "intent": "", "songs": [], "recent": []}
        try:
            if self.system is not None:
                st = self.system.status()
                cur, nxt, plan = st.get("current") or {}, st.get("next") or {}, st.get("plan") or {}
                if cur:
                    out["now"] = {"title": cur.get("title"), "artist": cur.get("artist"), "pos_s": cur.get("pos_s"),
                                  "duration_s": cur.get("duration_s"), "bpm": cur.get("bpm"), "camelot": cur.get("camelot")}
                if nxt:
                    out["next"] = {"title": nxt.get("title"), "artist": nxt.get("artist"), "eta_s": st.get("blend_in_s"),
                                   "how": plan.get("style"), "beats": plan.get("beats")}
                bits = [f"one song at a time", f"{self.dials['mixing']} mixing" if self.dials["mixing"] != "auto" else "mixing as the dice fall",
                        f"energy {self.dials['energy']}", f"{self.dials['pace']} plays"]
                if plan:
                    eta = st.get("blend_in_s")
                    bits.append(f"next seam: {plan.get('style')} " + (f"in {eta:.0f} s" if eta is not None else "when the exit comes"))
                out["intent"] = ", ".join(bits)
                out["recent"] = [f"{time.strftime('%H:%M:%S', time.localtime(e.get('t', 0)))}  {e.get('event')}  "
                                 + ", ".join(f"{k} {v}" for k, v in e.items() if k not in ("t", "clock_s", "event") and not isinstance(v, (dict, list)))[:120]
                                 for e in (st.get("recent_events") or [])[-8:]]
                out["state"] = st.get("state")
                out["error"] = out["error"] or self.system.last_error
            elif self.rc is not None:
                st = self.rc.status()
                songs = st.get("songs") or {}
                lanes = st.get("lanes") or {}
                m = songs.get(st.get("master")) or {}
                if m:
                    out["now"] = {"title": m.get("title"), "artist": m.get("artist"), "pos_s": m.get("time_s"),
                                  "duration_s": m.get("duration_s"), "bpm": st.get("master_bpm"), "camelot": st.get("key_centre")}
                out["songs"] = [{"title": s.get("title"), "lanes": s.get("lanes"), "entered": s.get("entered")} for s in songs.values()]
                staged = [s for s in songs.values() if not s.get("entered")]
                if staged:
                    out["next"] = {"title": staged[0]["title"], "artist": staged[0].get("artist"),
                                   "eta_s": max(0, (st.get("change_bars", 8) - 1 - st.get("phrase_bars", 0))) * 4 * 60.0 / max(st.get("master_bpm") or 120, 1),
                                   "how": "in through one lane"}
                live = [s for s in songs.values() if s.get("entered")]
                bits = [f"{len(live)} song{'s' if len(live) != 1 else ''} layered", f"a move every {st.get('change_bars')} bars",
                        "vocal lane free" if self.rc.vocal_freedom > 0.5 else "vocals cautious", f"energy {self.dials['energy']}"]
                left = st.get("change_bars", 8) - 1 - st.get("phrase_bars", 0)
                bits.append(f"next move in {max(0, left)} bars" + (" (held)" if st.get("hold") else ""))
                out["intent"] = ", ".join(bits)
                out["lanes"] = {ln: (songs.get(d) or {}).get("title") if d else None for ln, d in lanes.items()}
                out["recent"] = [f"{t}  {m}" for t, m in (st.get("moves") or [])[-8:]]
                out["error"] = out["error"] or st.get("error")
        except Exception as e:  # noqa: BLE001
            out["error"] = f"status: {type(e).__name__}: {e}"
        return out

    def _event(self, msg):
        self.events.append((time.strftime("%H:%M:%S"), msg))
        if len(self.events) > 200:
            del self.events[:-200]
