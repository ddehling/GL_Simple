"""THE RECORD PLAYS: layered mode as a DJ plays (docs/DJ_THE_RECORD_PLAYS.md, the user's yes 2026-09-13).

The unit of play is the RECORD: one song, all four stems, its own balance, playing whole for at least
`play_min_s` and never past its OUT point. One SEAM per record, decided once when the next record is
staged: the bar of A (its OUT, at a section boundary where its drums still carry through the seam), the
bar of B (cued so its landmark lands on the bar the seam completes), the seam's length and order. The seam
is the stem morph over phrases - drums first by default (B's drums under A's harmony, A darkening over the
last four bars, the harmony swap on the one: HARMONY TRAVELS TOGETHER), bass first (the harmony swap under
A's drums, A's drums follow: the TAKEOVER shape), vocal first (an acapella lead-in), or the breakdown and a
cut. Stems are events, not the way every song enters (the five moments, your right-click).

A mixin for RemixConductor: everything here is the conductor's own state and helpers.
"""
import math

from lib.dj.submix import RATE

PLAY_MIN_S = 210.0                     # a record plays at least this long (the club default; PACE bends it)
SEAM_BARS = {"cut": 4, "morph": 16, "blend": 32}
DARKEN_HIGH = 0.35                     # the departing harmony's high shelf over the bars before the swap (a fade to darkness)
DARKEN_BARS = 4
VOCAL_LEAD_BARS = 16                   # an acapella lead-in: B's vocal this long before its drums
ORDER_WORDS = {"drums": "morph, drums first", "bass": "bass swap", "vocal": "acapella lead-in", "cut": "breakdown and cut"}


class RecordPlay:
    # -- state -----------------------------------------------------------------------------------------------------
    def _record_init(self):
        self.record_deck = None                 # the deck of the current record
        self.seam = None                   # the one seam: planned, running, or None
        self.play_min_s = PLAY_MIN_S
        self.seam_kind = "auto"            # cut | morph | blend | auto
        self.seam_scale = 1.0              # the SEAMS dial
        self.seam_log = []                 # every seam that completed: where B landed against its landmark
        self._seam_seq = 0
        self._seam_last_order = None

    def set_play_min(self, seconds):
        self.play_min_s = float(max(30.0, seconds))

    def set_seam(self, kind, scale=1.0):
        self.seam_kind = kind if kind in ("cut", "morph", "blend", "auto") else "auto"
        self.seam_scale = float(max(0.25, min(4.0, scale)))

    def _play_min_bars(self):
        return int(math.ceil(self.play_min_s / max(self._bar_clock() / RATE, 1e-6)))

    def _record_set(self, deck, why=""):
        s = self.songs.get(deck)
        if s is None:
            return
        self.record_deck = deck
        if getattr(s, "play_from_k", None) is None:
            s.play_from_k = self.bar_n
        s.bed_since = s.play_from_k
        s.was_bed, s.voice_since = False, None
        if self.seam is not None and self.seam["state"] != "running":
            self._seam_drop(why or "the record changed")

    def _seam_drop(self, why):
        sm = self.seam
        self.seam, self.plan = None, None
        if sm is not None:
            self._cue_later.pop(sm["b"], None)
            self._note(f"seam dropped: {why}")

    # -- the step, once a bar ----------------------------------------------------------------------------------------
    def _record_step(self):
        bar = self.bar_n
        drums = self.lanes.get("drums")
        sm = self.seam
        running = sm is not None and sm["state"] == "running"
        # the record is the song holding the drums (a DROP, a RECALL, a TAKEOVER, your grid may have moved it)
        if not running and drums is not None and drums in self.songs and drums != self.record_deck:
            if self.record_deck in self.songs:
                self.songs[self.record_deck].play_from_k = None
            self.songs[drums].play_from_k = bar
            self._record_set(drums, "another song holds the drums now")
        if self.record_deck not in self.songs or self.songs[self.record_deck].leaving:
            cand = drums if drums in self.songs else self.master
            if cand in self.songs and not self.songs[cand].leaving:
                self.songs[cand].play_from_k = bar
                self._record_set(cand)
            else:
                return
        sm = self.seam
        if sm is not None and sm["state"] == "running":
            self._seam_fire()
            return
        if sm is not None:
            sb = self.songs.get(sm["b"])
            if sm["a"] != self.record_deck or sb is None or sb.leaving or sb.evicted:
                self._seam_drop("the room changed")
                sm = None
        if sm is None:
            b = self._next_record_deck()
            if b is not None:
                self._plan_seam(self.record_deck, b)
            return
        # the plan mirrors the seam (a moment's end clears the plan; the seam stands)
        if self.plan is None or self.plan.get("what") != "seam" or self.plan.get("bar") != sm["bar"]:
            self._seam_plan_mirror()
        if bar >= sm["bar"] - 1:
            if self.hold or self._moment is not None or self._break is not None or self._recall is not None:
                self._seam_move(self._phrase_bar_after(bar + 1), "held" if self.hold else "a moment has the room")
                return
            sm["state"] = "running"
            self._seam_fire()

    def _next_record_deck(self):
        """The next record: a song staged and waiting (your UP NEXT first), else a song you brought a part of
        in (it is heard; the seam morphs into it)."""
        cands = [(d, s) for d, s in self.songs.items() if d != self.record_deck and not s.leaving and not s.evicted
                 and s.staged_at is not None and s.env is not None and not s.was_bed]
        if not cands:
            return None
        cands.sort(key=lambda ds: (not getattr(ds[1], "user_pick", False), ds[1].entered, ds[1].staged_at or 0))
        return cands[0][0]

    def _phrase_bar_after(self, bar):
        """The first phrase boundary of the record at or after `bar` (counted from where its play began)."""
        s = self.songs.get(self.record_deck)
        k0 = (s.play_from_k if s is not None and s.play_from_k is not None else 0)
        n = self.change_bars
        return bar + ((n - (bar - k0) % n) % n)

    # -- planning the seam -------------------------------------------------------------------------------------------
    def _seam_len(self):
        kind = self.seam_kind
        if kind == "auto":
            kind = self.rng.choice(("morph", "blend")) if self.rng.random() < 0.5 else "morph"
        if kind == "cut":
            return SEAM_BARS["cut"], kind
        L = int(round(SEAM_BARS[kind] * self.seam_scale / 4.0)) * 4
        return max(8, L), kind

    def _seam_b_start(self, sb, L):
        """B's song time at the seam's first bar: its landmark minus the seam, when its drums carry from
        there; else the first place its drums carry for the seam (its IN point) - the landmark comes a little
        after the seam then. Returns (t_start, landmark, lands_on_landmark)."""
        t = sb.track
        lm = self._landmark(t, sb.env)
        bar_b = self._bar_s(t, lm)
        t0 = lm - L * bar_b
        if t0 >= 0.0 and self._stem_carries(sb.env, t0, 0.5 * L * bar_b, lanes=("drums",)) is not False:
            return t0, lm, True, L
        # its drums do not carry right before its landmark: the LATEST place before it where they carry for the
        # seam - the landmark comes soon after the seam then (searched from its earliest groove, a record was
        # cued to its intro with its drop 96 bars after the seam - the campaign smoke)
        step = 4 * bar_b
        tt = t0 - step
        while tt >= 0.0:
            if self._stem_carries(sb.env, tt, L * bar_b, lanes=("drums",)) is not False and self._track_steady(t, tt, L, bad=("build", "breakdown")) is not False:
                return tt, lm, False, L
            tt -= step
        t_in = self._steady_from(t, sb.env, 0.0, L, strict=False, lanes=("drums",))
        if t_in is None:
            t_in = max(0.0, t0)
        # its drums carry from later than a full seam before its landmark: a SHORTER seam that still completes
        # on the landmark (eight bars at least), else the landmark falls inside the seam
        room = int(math.floor((lm - t_in) / max(bar_b, 1e-6) / 4.0)) * 4
        if room >= 8 and room < L:
            return lm - room * bar_b, lm, True, room
        return t_in, lm, False, L

    def _seam_order(self, a, b, L, kind, tb):
        """drums first by default; bass first for variety (auto); vocal first when B sings from two phrases
        before its drums and A is instrumental there (auto); the cut when MIXING says cut."""
        if kind == "cut":
            return "cut"
        if self.seam_kind in ("morph", "blend"):
            return "drums"
        sb, sa = self.songs[b], self.songs[a]
        vocal_ok = False
        try:
            bar_b = self._bar_s(sb.track, tb)
            t_v = tb - VOCAL_LEAD_BARS * bar_b
            vox = (sb.env or {}).get("vocals")
            if t_v >= 0.0 and vox is not None and len(vox) > 4:
                import numpy as np
                i0, i1 = int(t_v * 2), int((t_v + 0.5 * VOCAL_LEAD_BARS * bar_b) * 2)
                seg = np.asarray(vox[i0:i1], dtype=np.float32)
                from lib.dj.remix import SING_OFF, SING_ON
                vocal_ok = len(seg) > 4 and float(np.mean(seg >= SING_OFF)) >= 0.6 and float(seg.max()) >= SING_ON
            if vocal_ok:
                # A must not sing under it (never two singers)
                vocal_ok = self._singing(sa, self._song_time_at(a, self._next_bar_clock())) is not True and self.vocal_freedom > 0.1
        except Exception:
            vocal_ok = False
        if vocal_ok and self._seam_last_order != "vocal":
            return "vocal"
        return "bass" if self._seam_last_order == "drums" and self.rng.random() < 0.5 else "drums"

    def _seam_a_bar(self, a, L, order):
        """A's OUT bar: at or after its minimum play, at a section boundary of A where its drums still carry
        through the seam and no build or breakdown of A falls inside it; else a phrase boundary that does;
        else any bar that does; else before it runs out. Returns (bar, why)."""
        sa = self.songs[a]
        t = sa.track
        bar = self.bar_n
        at = self._next_bar_clock()
        t_now = self._song_time_at(a, at)
        bar_a = self._bar_s(t, t_now)
        lead = L + (VOCAL_LEAD_BARS if order == "vocal" else 0)
        k0 = sa.play_from_k if sa.play_from_k is not None else bar
        earliest = max(bar + 1 + max(2, self.change_bars), k0 + self._play_min_bars())
        if order == "vocal":
            earliest = max(earliest, bar + 1 + VOCAL_LEAD_BARS + 2)
        if sa.loop is not None:
            return max(bar + 3, earliest if earliest - bar < 2 * self.change_bars else bar + 3), f"{t.title[:22]} is looping its last bars"
        from lib.dj.remix import RUNWAY_S
        end_a = float(t.duration_s) - RUNWAY_S - 2.0 * bar_a      # the runway loop must not fire inside the seam
        latest = bar + 1 + int(math.floor((end_a - t_now) / max(bar_a, 1e-6))) - L
        if latest < earliest:
            return max(bar + 3, latest), f"before {t.title[:22]} runs out"
        pre = VOCAL_LEAD_BARS if order == "vocal" else 0             # the acapella lead-in is part of the seam too

        def ok(K):
            tK = t_now + (K - pre - bar - 1) * bar_a
            return (self._steady_at(t, sa.env, tK, L + pre, strict=False, lanes=("drums",)) is not False
                    and self._track_steady(t, tK, L + pre, bad=("build", "breakdown")) is not False)
        # its mix-out point: the first section boundary that works (an outro's start, the groove after a breakdown)
        for sec in t.sections or []:
            ts = t.nearest_downbeat(float(sec["start_s"]))
            if ts <= t_now + 0.5 * bar_a:
                continue
            K = bar + 1 + int(round((ts - t_now) / max(bar_a, 1e-6)))
            if K < earliest:
                continue
            if K > latest:
                break
            if ok(K):
                return K, f"at {t.title[:22]}'s {sec.get('kind', 'section')}"
        for K in range(earliest, latest + 1):
            if (K - k0) % self.change_bars == 0 and ok(K):
                return K, "at the phrase"
        for K in range(earliest, latest + 1):
            if ok(K):
                return K, "where its drums carry"
        return latest, f"before {t.title[:22]} runs out"

    def _plan_seam(self, a, b):
        sa, sb = self.songs[a], self.songs[b]
        bar = self.bar_n
        L, kind = self._seam_len()
        tb, lm, on_lm, L = self._seam_b_start(sb, L)
        order = self._seam_order(a, b, L, kind, tb)
        K, why = self._seam_a_bar(a, L, order)
        if order == "vocal":
            K_first = K - VOCAL_LEAD_BARS                    # the vocal comes VOCAL_LEAD_BARS before the drums
        else:
            K_first = K
        self._seam_seq += 1
        self.seam = {"seq": self._seam_seq, "a": a, "b": b, "bar": int(K_first), "drums_bar": int(K), "len": int(L), "order": order, "kind": kind,
                     "tb": float(tb), "lm": float(lm), "on_lm": bool(on_lm), "state": "planned", "fired": set(), "moved": 0,
                     "why": why, "rehearsed": None, "steps": self._seam_steps(order, L)}
        self._seam_cue_b()
        self._seam_plan_mirror()
        sb.ready_k = int(K_first)
        words = self._seam_words()
        after = self._bars_after_seam()
        self._move("seam_plan", None, self._song_id(a), sb.track.id, f"the seam: {words} - {why}"
                   + ("" if on_lm else (f" (its drop {after:d} bars after the seam)" if after > 0 else " (its drop falls inside the seam)")))
        try:
            from lib.dj.session import log_row
            log_row({"event": "plan", "what": "seam", "track_id": sb.track.id, "bar": int(K_first), "in_bars": int(K_first) - bar, "why": why[:120], "order": order, "len": int(L)})
        except Exception:
            pass

    def _bars_after_seam(self):
        sm = self.seam
        if sm is None:
            return 0
        sb = self.songs.get(sm["b"])
        if sb is None:
            return 0
        bar_b = self._bar_s(sb.track, sm["lm"])
        return int(round((sm["lm"] - (sm["tb"] + sm["len"] * bar_b)) / max(bar_b, 1e-6)))

    def _seam_cue_b(self):
        """B is cued so its seam-start time lands on the seam's first drum bar (deferred when it has fewer bars
        before it than the wait). A song already sounding (you brought a part of it in) is never jumped."""
        sm = self.seam
        sb = self.songs.get(sm["b"])
        if sb is None or sb.entered or sb.leaving:
            return
        K = sm["drums_bar"] if sm["order"] != "vocal" else sm["bar"]
        t_at_K = sm["tb"] if sm["order"] != "vocal" else sm["tb"] - VOCAL_LEAD_BARS * self._bar_s(sb.track, sm["tb"])
        if t_at_K < 0.0:
            t_at_K, K = sm["tb"], sm["drums_bar"]
        n = K - (self.bar_n + 1)
        if n < 2:
            return
        at = self._next_bar_clock()
        cue, later = self._cue_plan(sm["b"], t_at_K, n, at)
        self._cue_later.pop(sm["b"], None)
        if later is not None:
            self._cue_later[sm["b"]] = later
            self.recue_log = (getattr(self, "recue_log", []) + [(self._tel().get("clock_s"), sm["b"], round(later[1], 1), K, "deferred")])[-40:]
            return
        now_t = self._song_time_at(sm["b"], at)
        if abs(now_t - cue) < 0.5 * self._bar_s(sb.track, cue):
            return
        self._post_cue(sm["b"], cue, at)
        self.recue_log = (getattr(self, "recue_log", []) + [(self._tel().get("clock_s"), sm["b"], round(cue, 1), K)])[-40:]

    def _seam_plan_mirror(self):
        sm = self.seam
        if sm is None:
            self.plan = None
            return
        self._plan_seq = getattr(self, "_plan_seq", 0) + 1
        self.plan = {"what": "seam", "deck": sm["b"], "bar": sm["bar"], "why": sm["why"], "payoff": None, "set_k": self.bar_n,
                     "moved": sm["moved"], "seq": sm["seq"], "sing": sm["order"] == "vocal"}

    def _seam_words(self):
        sm = self.seam
        if sm is None:
            return "nothing"
        sb = self.songs.get(sm["b"])
        title = sb.track.title[:24] if sb else "?"
        return f"{title} in at bar {sm['bar']}: a {sm['len']}-bar {ORDER_WORDS.get(sm['order'], sm['order'])}"

    def _seam_move(self, K_first, why):
        sm = self.seam
        if sm is None:
            return
        K_first = int(max(K_first, self.bar_n + 2))
        shift = K_first - sm["bar"]
        if shift == 0:
            return
        sm["bar"], sm["drums_bar"], sm["moved"] = K_first, sm["drums_bar"] + shift, sm["moved"] + 1
        self._seam_cue_b()
        self._seam_plan_mirror()
        sb = self.songs.get(sm["b"])
        if sb is not None:
            sb.ready_k = K_first
        self._move("plan", None, None, self._song_id(sm["b"]), f"the seam moved: {self._seam_words()} - {why}")

    def _record_next(self):
        """NEXT: the seam at the next phrase (the same order); nothing when no next record is staged or the
        seam is already running."""
        sm = self.seam
        if sm is None:
            self.wait_why = "NEXT: no next record is staged yet"
            return None
        if sm["state"] == "running":
            self.wait_why = "NEXT: the seam is already running"
            return None
        self._seam_move(self._phrase_bar_after(self.bar_n + 3), "NEXT")   # (three bars: the re-cue needs two to lock)
        return None

    # -- the seam's steps ----------------------------------------------------------------------------------------------
    def _seam_steps(self, order, L):
        """[(offset in bars from the seam's first bar, method name)] and the marks for the picture."""
        if order == "cut":
            steps = [(0, "_st_strip"), (4, "_st_cut"), (5, "_st_done")]
            marks = [(0, 4, ["drums", "bass"], "breakdown: drums and bass out"), (4, None, ["drums", "bass", "other"], "the cut: the new record whole")]
        elif order == "bass":
            steps = [(0, "_st_darken"), (DARKEN_BARS, "_st_harmony"), (L, "_st_drums_in"), (L + 1, "_st_done")]
            marks = [(0, DARKEN_BARS, ["other"], "the harmony darkens"), (DARKEN_BARS, None, ["bass", "other"], "bass swap: the new bassline and melody"),
                     (L, None, ["drums"], "its drums follow: the record is its")]
        elif order == "vocal":
            steps = [(0, "_st_vocal_in"), (VOCAL_LEAD_BARS, "_st_drums_in"), (VOCAL_LEAD_BARS + L - DARKEN_BARS, "_st_darken"),
                     (VOCAL_LEAD_BARS + L, "_st_harmony"), (VOCAL_LEAD_BARS + L + 1, "_st_done")]
            marks = [(0, None, ["vocals"], "acapella lead-in: the new vocal"), (VOCAL_LEAD_BARS, None, ["drums"], "its drums under the old harmony"),
                     (VOCAL_LEAD_BARS + L - DARKEN_BARS, VOCAL_LEAD_BARS + L, ["other"], "the harmony darkens"),
                     (VOCAL_LEAD_BARS + L, None, ["bass", "other"], "the harmony swaps: the record is its")]
        else:
            steps = [(0, "_st_drums_in"), (max(1, L - DARKEN_BARS), "_st_darken"), (L, "_st_harmony"), (L + 1, "_st_done")]
            marks = [(0, None, ["drums"], "the new drums under the old harmony"), (max(1, L - DARKEN_BARS), L, ["other"], "the harmony darkens"),
                     (L, None, ["bass", "other"], "the harmony swaps: the record is its")]
        self._seam_marks = marks
        return steps

    def _seam_fire(self):
        sm = self.seam
        if sm is None:
            return
        for off, name in sm["steps"]:
            if off in sm["fired"]:
                continue
            if self.bar_n >= sm["bar"] + off - 1:
                sm["fired"].add(off)
                at = self._next_bar_clock()
                try:
                    getattr(self, name)(at)
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"seam {name} @{off}: {type(e).__name__}: {e}"
                if self.seam is None:
                    return

    def _ab(self):
        sm = self.seam
        return sm["a"], sm["b"], self.songs.get(sm["a"]), self.songs.get(sm["b"])

    def _st_vocal_in(self, at):
        a, b, sa, sb = self._ab()
        if sb is None:
            return
        if self._vocal_ok(b, at):
            frm = self.lanes.get("vocals")
            tag = self._cross("vocals", b, at, beats=2.0, shape=False)
            sb.voice_since = self.bar_n
            self._hygiene(at + int(self._beat_s() * RATE))
            self._move("voice_in", "vocals", self._song_id(frm), sb.track.id, f"{sb.track.title} arrives: its vocal over {sa.track.title[:24] if sa else '?'} (an acapella lead-in)", shape=tag)
        else:
            self._note("the acapella lead-in skipped: the vocal is not singing there")

    def _st_drums_in(self, at):
        a, b, sa, sb = self._ab()
        sm = self.seam
        if sb is None:
            return
        first = sm["order"] in ("drums",)
        tag = self._cross("drums", b, at, beats=4.0, shape=False)
        if sb.voice_since is None:
            sb.voice_since = self.bar_n
        self._hygiene(at + int(4 * self._beat_s() * RATE))
        if first:
            self._move("voice_in", "drums", self._song_id(a), sb.track.id,
                       f"{sb.track.title} arrives: its drums come in under {sa.track.title[:24] if sa else '?'}'s harmony (a {sm['len']}-bar {ORDER_WORDS[sm['order']]})", shape=tag)
        elif sm["order"] == "vocal":
            self._move("seam", "drums", self._song_id(a), sb.track.id, f"the drums of {sb.track.title} come in under {sa.track.title[:24] if sa else '?'}'s harmony", shape=tag)
        else:
            self._move("bed_to", "drums", self._song_id(a), sb.track.id, f"the drums of {sb.track.title} follow: the record is its", shape=tag)
            self._seam_roles(at)

    def _st_darken(self, at):
        a, b, sa, sb = self._ab()
        if sa is None:
            return
        self.submix.post({"at": at, "cmd": "eq", "deck": a, "high": DARKEN_HIGH, "ramp_s": DARKEN_BARS * 4 * self._beat_s()})
        self._note(f"{sa.track.title[:24]} darkens over {DARKEN_BARS} bars before the swap")

    def _st_harmony(self, at):
        a, b, sa, sb = self._ab()
        sm = self.seam
        if sb is None:
            return
        first = sm["order"] == "bass"
        if first:
            self.lane_boost["bass"] = (1.25, self.bar_n + self.change_bars)
        tag = self._cross("bass", b, at, beats=2.0, shape=False)
        if self._other_free_for(b, a):
            self._cross("other", b, at, beats=2.0, shape=False)
        gone = self._vocal_goes_with(b, a, at) if a in self.songs else ""
        if a in self.songs:
            self.submix.post({"at": at, "cmd": "eq", "deck": a, "high": 1.0, "ramp_s": self._beat_s()})
        self._hygiene(at + int(2 * self._beat_s() * RATE))
        if first:
            self._move("voice_in", "bass", self._song_id(a), sb.track.id,
                       f"{sb.track.title} arrives: its bassline and melody take over under {sa.track.title[:24] if sa else '?'}'s drums, loud (a {sm['len']}-bar bass swap){gone}", shape=tag)
        else:
            self._move("bed_to", "bass", self._song_id(a), sb.track.id, f"the bed passes to {sb.track.title}: its bassline and melody take over on the one{gone}", shape=tag)
            self._seam_roles(at)

    def _st_strip(self, at):
        a, b, sa, sb = self._ab()
        if sa is None:
            return
        self._cross("drums", None, at, beats=1.0, shape=False)
        self._cross("bass", None, at, beats=1.0, shape=False)
        self._hygiene(at + int(self._beat_s() * RATE))
        self._move("seam", None, self._song_id(a), None, f"the breakdown: {sa.track.title}'s drums and bass out for 4 bars, its melody stays")

    def _st_cut(self, at):
        a, b, sa, sb = self._ab()
        if sb is None:
            return
        from lib.dj.remix import DROP_BEATS
        tag = self._cross("drums", b, at, beats=DROP_BEATS, shape=False)
        self._cross("bass", b, at, beats=DROP_BEATS, shape=False)
        if self._other_free_for(b, a):
            self._cross("other", b, at, beats=DROP_BEATS, shape=False)
        gone = self._vocal_goes_with(b, a, at) if a in self.songs else ""
        self._hygiene(at + int(DROP_BEATS * self._beat_s() * RATE))
        self._move("bed_to", "drums", self._song_id(a), sb.track.id, f"the bed passes to {sb.track.title} on the one: a cut after the breakdown{gone}", shape=tag)
        self._seam_roles(at)

    def _seam_roles(self, at):
        a, b, sa, sb = self._ab()
        if sa is not None:
            sa.was_bed, sa.voice_since, sa.bed_since, sa.play_from_k = True, self.bar_n, None, None
            if not sa.held(self.lanes):
                self._leave_song(a, at + 2 * self._bar_clock())
        if sb is not None:
            sb.play_from_k = self.bar_n
            sb.bed_since, sb.was_bed, sb.voice_since = self.bar_n, False, None

    def _st_done(self, at):
        a, b, sa, sb = self._ab()
        sm = self.seam
        if sb is not None:
            t_b = self._song_time_at(b, at)
            bar_b = self._bar_s(sb.track, t_b)
            err = (t_b - sm["lm"]) / max(bar_b, 1e-6)
            self.seam_log.append({"a": sa.track.title if sa else "?", "b": sb.track.title, "order": sm["order"], "len": sm["len"], "moved": sm["moved"],
                                  "b_t": round(t_b, 1), "lm": round(sm["lm"], 1), "err_bars": round(err, 2), "on_lm": sm["on_lm"], "bar": self.bar_n})
            self._note(f"seam done: {sb.track.title[:24]} is the record at {t_b:.0f} s (its landmark {sm['lm']:.0f} s, {err:+.1f} bars)")
        self._seam_last_order = sm["order"]
        self.seam, self.plan = None, None
        if sb is not None:
            self._record_set(b)

    # -- words and pictures ----------------------------------------------------------------------------------------------
    def _seam_status(self):
        sm = self.seam
        if sm is None:
            return None
        sa, sb = self.songs.get(sm["a"]), self.songs.get(sm["b"])
        return {"a": sa.track.title if sa else "?", "b": sb.track.title if sb else "?", "bar": sm["bar"], "in_bars": sm["bar"] - self.bar_n,
                "len": sm["len"], "order": sm["order"], "kind": sm["kind"], "state": sm["state"], "moved": sm["moved"], "why": sm["why"],
                "rehearsed": sm.get("rehearsed"), "on_landmark": sm["on_lm"], "drop_after_bars": self._bars_after_seam(),
                "words": self._seam_words(),
                "marks": [{"off": o, "end_off": e, "lanes": ln, "text": tx} for (o, e, ln, tx) in getattr(self, "_seam_marks", [])]}

    def _record_status(self):
        s = self.songs.get(self.record_deck)
        if s is None:
            return None
        since = self.bar_n - s.play_from_k if s.play_from_k is not None else 0
        return {"deck": self.record_deck, "title": s.track.title, "since_bars": since, "min_bars": self._play_min_bars(),
                "settle_left": max(0, self._play_min_bars() - since), "kind": self._kind_at(self.record_deck, self._next_bar_clock())}
