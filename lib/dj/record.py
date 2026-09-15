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
import time

from lib.dj.submix import RATE

PLAY_MIN_S = 210.0                     # a record plays at least this long (the club default; PACE bends it)
SEAM_BARS = {"cut": 4, "morph": 16, "blend": 32}
DARKEN_HIGH = 0.7                      # the departing harmony's high shelf over the bars before the swap (a fade to darkness; 0.35 opened
                                       # a 10 dB hole under the new drums and a 13 dB lurch at the swap - the campaign)
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
        # THE EVENTS (docs/DJ_PAIRINGS.md): per record, from the pairing lists computed offline
        self.event_rate = 0                # events per record (the MOMENTS dial: rare 0 / some 1 / lots 2)
        self.pairings = None               # .stems/pairings.json, loaded when first needed
        self._pairings_mtime = None
        self._event_pending = None         # {"kind", "track_id", "bar", "bars", "why", "t"} staged and waiting to arm
        self._event = None                 # the event the armed moment reads (its bar and length)
        self.event_log = []

    def set_event_rate(self, n):
        self.event_rate = int(max(0, n))

    def _load_pairings(self):
        import json
        import os
        path = os.path.join(self.music_root, ".stems", "pairings.json")
        try:
            mt = os.path.getmtime(path)
        except OSError:
            self.pairings = None
            return None
        if self.pairings is None or mt != self._pairings_mtime:
            try:
                self.pairings = json.load(open(path, encoding="utf-8"))
                self._pairings_mtime = mt
            except Exception:  # noqa: BLE001
                self.pairings = None
        return self.pairings

    def pairings_for(self, track_id):
        """The record's two lists (VOCAL OVER, BASS UNDER) or None - for the songs list and the parts window."""
        p = self._load_pairings()
        if not p:
            return None
        return (p.get("records") or {}).get(str(int(track_id)))

    # -- the events: one or two per record, from the pairing lists -----------------------------------------------------
    def _event_step(self):
        """During a record's play, clear of its seam, the DJ arms one event (event_rate two) from the record's
        pairing lists: a VOCAL OVER becomes the ACAPELLA moment - the other song staged with its sung stretch
        cued to begin on a phrase where the record is instrumental; a BASS UNDER becomes the TAKEOVER moment
        with that song, which then is the next record."""
        rate = int(getattr(self, "event_rate", 0) or 0)
        if rate <= 0:
            return
        sa = self.songs.get(self.record_deck)
        if sa is None or sa.leaving or not sa.entered:
            return
        pend = self._event_pending
        bar = self.bar_n
        cb = self.change_bars
        if pend is not None:
            sm_ = self.seam
            if time.time() - pend["t"] > 150.0 or (sm_ is not None and bar > sm_["bar"] - 3 * cb):
                self._event_pending = None
                self._note("the event's song never became ready in time: dropped")
                return
            d = self.deck_of(pend["track_id"])
            s = self.songs.get(d) if d else None
            if s is None or s.staged_at is None or self._clock() < int(s.staged_at) + self._bar_clock():
                return
            if self._moment is not None or (self.seam is not None and self.seam["state"] == "running"):
                return
            if bar >= pend["bar"] - 1:
                # ready late (a decode is real seconds): the event moves to the next phrase; the acapella's song
                # jumps so its sung stretch still begins on that bar
                K2 = self._phrase_bar_after(bar + 3)
                if pend["kind"] == "acapella" and pend.get("want_t") is not None:
                    self.jump(d, float(pend["want_t"]), K2)
                pend["bar"] = K2
                if sm_ is not None and K2 > sm_["bar"] - 2 * cb:
                    self._event_pending = None
                    self._note("the event came too late for this record: dropped")
                return
            self._event = dict(pend)
            ok, msg = self.arm_moment(pend["kind"], auto=True, other=d)
            self._event_pending, self._event = None, None
            self.event_log.append({"kind": pend["kind"], "record": sa.track.title, "song": s.track.title, "ok": ok, "msg": msg, "bar": bar})
            self._move("event", None, self._song_id(self.record_deck), s.track.id,
                       (f"the event: {msg}" if ok else f"the event refused: {msg}"))
            return
        if getattr(sa, "events_done", 0) >= rate or self._moment is not None:
            return
        sm = self.seam
        if sm is None or sm["state"] == "running":
            return
        rec = self.pairings_for(sa.track.id)
        if not rec:
            sa.events_done = rate                     # nothing known for this record: no event
            return
        k0 = sa.play_from_k if sa.play_from_k is not None else bar
        # the window: from a phrase into the record - and FOUR phrases from now, a decode and a staging are real
        # seconds (at two phrases the gate's four planned events all came too late) - to two phrases before the seam
        start_min = max(bar + 4 * cb, k0 + cb)
        end_max = sm["bar"] - 2 * cb
        if end_max - start_min < 2 * cb:
            sa.events_done = rate                     # no room left in this record
            return
        at = self._next_bar_clock()
        bar_c = self._bar_clock()
        t_now = self._song_time_at(self.record_deck, at)
        bar_a = self._bar_s(sa.track, t_now)
        busy = {s.track.id for s in self.songs.values()} | {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        busy |= set(getattr(self, "avoid_ids", ()) or ())
        n_done = int(getattr(sa, "events_done", 0))
        kinds = ["vocal", "bass"]
        for kind in kinds:
            cands = rec.get("vocal_over" if kind == "vocal" else "bass_under") or []
            if kind == "bass":
                # a BASS UNDER changes the record: it is the record's EXIT, at the end of its minimum play, not a
                # phrase into it (one event per record at a phrase in halved the pace: 2.5 min per record - the
                # fifth campaign); it replaces the seam where the seam would have been
                start_min = max(start_min, k0 + self._play_min_bars() - 2 * cb)
                if end_max - start_min < cb:
                    continue
            for c in cands:
                if c["id"] in busy:
                    continue
                if kind == "vocal":
                    ga, gb = c["groove"]
                    # the first phrase boundary in the window where the record is inside that groove with eight bars to spare
                    K = None
                    for k in range(start_min, end_max + 1):
                        if (k - k0) % cb:
                            continue
                        tk = t_now + (k - bar - 1) * bar_a
                        if ga - 0.5 * bar_a <= tk and tk + 8 * bar_a <= gb + 0.5 * bar_a:
                            K = k
                            break
                    if K is None:
                        continue
                    tk = t_now + (K - bar - 1) * bar_a
                    room = int((gb - tk) / max(bar_a, 1e-6))
                    stretch = int((c["sung"][1] - c["sung"][0]) / max(bar_a * c.get("rate", 1.0), 1e-6))
                    bars = max(cb, min(4 * cb, room, stretch) // cb * cb)
                    want_t = float(c["sung"][0]) - bar_a * c.get("rate", 1.0)
                    ok, msg = self.stage_for_bar(c["id"], want_t, K)
                    if not ok:
                        continue
                    self._event_pending = {"kind": "acapella", "track_id": int(c["id"]), "bar": int(K), "bars": int(bars), "t": time.time(), "want_t": want_t,
                                           "why": f"the vocal of {c['title'][:24]} over this groove (fit {c['fit']})"}
                else:
                    K = None
                    for k in range(start_min, end_max + 1):
                        if (k - k0) % cb == 0 and self._track_steady(sa.track, t_now + (k - bar - 1) * bar_a, 2 * cb, bad=("build", "breakdown")) is not False:
                            K = k
                            break
                    if K is None:
                        continue
                    ok, msg = self.stage_for_bar(c["id"], float(c["landmark"]) - 2 * bar_a * c.get("rate", 1.0), K - 2)
                    if not ok:
                        continue
                    self._event_pending = {"kind": "takeover", "track_id": int(c["id"]), "bar": int(K), "bars": 0, "t": time.time(),
                                           "why": f"the bassline of {c['title'][:24]} under these drums at its drop (fit {c['fit']})"}
                sa.events_done = n_done + 1
                self._move("event_plan", None, self._song_id(self.record_deck), int(c["id"]),
                           f"event in {int(K) - bar} bars: {self._event_pending['why']}")
                return
        sa.events_done = rate                         # nothing usable for this record

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
        if self.wait_why and self.wait_why.startswith("NEXT"):
            self.wait_why = None                      # (a stale "NEXT: ..." sat in the WHY panel for ten minutes, 2026-09-14)
        self._room_never_empty()
        if sm is not None:
            sb = self.songs.get(sm["b"])
            if sm["a"] != self.record_deck or sb is None or sb.leaving or sb.evicted:
                self._seam_drop("the room changed")
                sm = None
            else:
                # YOUR pick outranks the conductor's: a staged song you put UP NEXT replaces the seam's next record
                yours = [d for d, s in self.songs.items() if d not in (self.record_deck, sm["b"]) and getattr(s, "user_pick", False)
                         and not s.leaving and s.staged_at is not None and s.env is not None and not s.was_bed]
                if yours and not getattr(sb, "user_pick", False):
                    self._seam_drop(f"your pick {self.songs[yours[0]].track.title[:22]} comes first")
                    sm = None
        if sm is None:
            b = self._next_record_deck()
            if b is not None:
                self._plan_seam(self.record_deck, b)
            return
        # the record is looping its last bars (its body has ended): the seam fires NOW whatever B's place - the
        # cut, B whole on the one - rather than the record being evicted into silence (2026-09-14)
        sa = self.songs.get(self.record_deck)
        if sa is not None and sa.loop is not None and sm["bar"] > bar + 1:
            sm["bar"], sm["drums_bar"], sm["order"], sm["len"] = bar + 1, bar + 1, "cut", 4
            sm["steps"] = self._seam_steps("cut", 4)
            self._seam_plan_mirror()
            self._move("plan", None, None, self._song_id(sm["b"]), f"the seam now, as a cut: {sa.track.title[:22]} is looping its last bars")
            sm["state"] = "running"
            self._seam_fire()
            return
        # the plan mirrors the seam (a moment's end clears the plan; the seam stands)
        if self.plan is None or self.plan.get("what") != "seam" or self.plan.get("bar") != sm["bar"]:
            self._seam_plan_mirror()
        try:
            self._event_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"event: {type(e).__name__}: {e}"
        if bar >= sm["bar"] - 1:
            if self.hold or self._moment is not None or self._break is not None or self._recall is not None:
                self._seam_move(self._phrase_bar_after(bar + 1), "held" if self.hold else "a moment has the room")
                return
            # the pre-flight: B must be where the plan put it (within a bar); else the seam moves a phrase and B
            # is cued again, once - then the seam is dropped and planned afresh
            sb = self.songs.get(sm["b"])
            if sb is not None and not sb.entered:
                at = self._next_bar_clock()
                want = sm["tb"] if sm["order"] != "vocal" else sm["tb"] - VOCAL_LEAD_BARS * self._bar_s(sb.track, sm["tb"])
                have = self._song_time_at(sm["b"], at)
                if abs(have - want) > 1.0 * self._bar_s(sb.track, want):
                    if sm["moved"] < 2:
                        self._seam_move(self._phrase_bar_after(bar + 3), f"{sb.track.title[:20]} was {have - want:+.0f} s from its place: cued again")
                    else:
                        self._seam_drop(f"{sb.track.title[:20]} never reached its place")
                    return
            sm["state"] = "running"
            self._seam_fire()

    def _room_never_empty(self):
        """The record plays WHOLE: its four lanes are its (the vocal lane too - the stem is silent where it does
        not sing, resting it changes nothing but risks missing its return; flag 'vocals' 2026-09-14). Drums or
        bass with nobody go to the record; with no record left, a staged song enters whole (an evicted record
        rested every lane and the room was silent for twenty seconds - 2026-09-14)."""
        if self._break is not None or self._recall is not None or (self._moment is not None and self._moment.get("started")):
            return
        rd = self.record_deck
        at = self._next_bar_clock()
        s = self.songs.get(rd)
        if s is not None and not s.leaving and s.entered:
            fixed = []
            for ln in ("drums", "bass", "other", "vocals"):
                h = self.lanes.get(ln)
                if h is None or h not in self.songs or self.songs[h].leaving:
                    if ln == "vocals" and self.seam is not None and self.seam["state"] == "running":
                        continue                      # a seam's vocal step may be about to place it
                    self._cross(ln, rd, at, beats=1.0, shape=False)
                    fixed.append(ln)
            if fixed:
                self._hygiene(at + int(self._beat_s() * RATE))
                self._move("return", fixed[0], None, self._song_id(rd), f"{', '.join(fixed)} back on {s.track.title[:26]} (the record plays whole)")
            return
        # no record: anyone entered, else a staged song, whole, now
        cands = [d for d, x in self.songs.items() if not x.leaving and x.staged_at is not None and x.entered]
        if not cands:
            cands = [d for d, x in self.songs.items() if not x.leaving and x.staged_at is not None and x.env is not None]
        if not cands:
            return
        d = max(cands, key=lambda dd: (len(self.songs[dd].held(self.lanes)), self.songs[dd].entered))
        for ln in ("drums", "bass", "other", "vocals"):
            if self.lanes.get(ln) != d:
                self._cross(ln, d, at, beats=1.0, shape=False)
        self._hygiene(at + int(self._beat_s() * RATE))
        self.songs[d].play_from_k = self.bar_n
        self._record_set(d, "the room had nobody")
        self._move("return", "drums", None, self._song_id(d), f"{self.songs[d].track.title} whole, now: the room had nobody")

    def _seam_clear_strays(self, at):
        """At a seam's first step: a vocal still carried from an earlier record goes out (a vocal rode across two
        seams over two harmonies it never fitted - flags 'clash' 2026-09-14)."""
        sm = self.seam
        if sm is None:
            return
        for ln in ("vocals", "other"):
            h = self.lanes.get(ln)
            if h is not None and h not in (sm["a"], sm["b"]) and h in self.songs:
                self._cross(ln, None, at, beats=2.0, shape=False)
                self._note(f"the {ln} of {self.songs[h].track.title[:22]} out: a seam begins and it belongs to neither record")

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

    def _seam_b_start(self, sb, L, lanes=("drums",)):
        """B's song time at the seam's first bar: its landmark minus the seam, when the stems the seam brings
        FIRST (`lanes`: its drums for a drums-first seam, its bass for a bass swap) carry from there; else the
        LATEST place before the landmark where they do (searched from its earliest groove, a record was cued to
        its intro with its drop 96 bars after the seam); else the first place they do (its IN point) - the
        landmark comes a little after the seam then. The bars before a drop ARE its build, and a build's drums
        under the old harmony, then the drop as the harmony swaps, is the classic mix into the drop: the labels
        are not asked here, the stems are. Never under two bars in: a cue needs two bars to lock, and cued to
        0.0 s a record arrived late at every attempt until the room fell silent (2026-09-14).
        Returns (t_start, landmark, lands_on_landmark, seam length)."""
        t = sb.track
        lm = self._landmark(t, sb.env)
        bar_b = self._bar_s(t, lm)
        floor_t = 2.0 * bar_b
        t0 = lm - L * bar_b
        if t0 >= floor_t and self._stem_carries(sb.env, t0, 0.5 * L * bar_b, lanes=lanes) is not False:
            return t0, lm, True, L
        step = 4 * bar_b
        tt = t0 - step
        while tt >= floor_t:
            if self._stem_carries(sb.env, tt, L * bar_b, lanes=lanes) is not False:
                return tt, lm, False, L
            tt -= step
        t_in = self._steady_from(t, sb.env, floor_t, L, strict=False, lanes=lanes)
        if t_in is None:
            t_in = max(floor_t, t0)
        t_in = max(t_in, floor_t)
        # its stems carry from later than a full seam before its landmark: a SHORTER seam that still completes
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
        if getattr(self, "_next_asap", False):
            earliest = bar + 1 + max(2, self.change_bars)          # NEXT was pressed with no next record ready: as soon as it is
            self._next_asap = False
        if order == "vocal":
            earliest = max(earliest, bar + 1 + VOCAL_LEAD_BARS + 2)
        if sa.loop is not None:
            return max(bar + 3, earliest if earliest - bar < 2 * self.change_bars else bar + 3), f"{t.title[:22]} is looping its last bars"
        from lib.dj.remix import RUNWAY_S
        end_a = float(t.duration_s) - RUNWAY_S - 2.0 * bar_a      # the runway loop must not fire inside the seam
        latest = bar + 1 + int(math.floor((end_a - t_now) / max(bar_a, 1e-6))) - L
        short = ""
        if latest < earliest:
            # a record too short for its minimum play AND a seam: the DJ mixes out EARLIER, at the best boundary
            # before the end, rather than run into the outro (seventeen of thirty campaign seams had fallen to
            # "before it runs out" here and landed in A's build or breakdown)
            earliest = max(bar + 1 + max(2, self.change_bars), latest - 4 * self.change_bars)
            short = " (a short record: before its minimum play)"
            if latest < earliest:
                return max(bar + 3, latest), f"before {t.title[:22]} runs out"
        pre = VOCAL_LEAD_BARS if order == "vocal" else 0             # the acapella lead-in is part of the seam too
        n_win = L + pre

        def unsteady(K):
            """How much of A's stretch under the seam is wrong for it: the share of bars in a build or a breakdown,
            plus a half when its drums do not carry through (0 = clean)."""
            tK = t_now + (K - pre - bar - 1) * bar_a
            bad = 0
            for i in range(n_win):
                if (t.section_at(tK + (i + 0.25) * bar_a) or {}).get("kind") in ("build", "breakdown"):
                    bad += 1
            car = self._stem_carries(sa.env, tK, n_win * bar_a, lanes=("drums",))
            return bad / max(n_win, 1) + (0.5 if car is False else 0.0)
        # 1. the seam COMPLETES at a section boundary of A - the new record takes over as A's breakdown, build or
        #    outro would begin (the DJ's mix-out point) - with A steady and its drums carrying all the way there
        #    (the first version put A's boundary at the seam's START and fell back to "before it runs out" for
        #    most records, which landed the seam in A's build or breakdown five times in twenty-two - the campaign)
        for sec in t.sections or []:
            ts = t.nearest_downbeat(float(sec["start_s"]))
            K = bar + 1 + int(round((ts - t_now) / max(bar_a, 1e-6))) - L
            if K < earliest or K > latest:
                continue
            if unsteady(K) == 0.0:
                return K, f"into {t.title[:22]}'s {sec.get('kind', 'section')}{short}"
        # 2. a phrase boundary with a clean stretch
        for K in range(earliest, latest + 1):
            if (K - k0) % self.change_bars == 0 and unsteady(K) == 0.0:
                return K, f"at the phrase{short}"
        # 3. the cleanest stretch there is
        scored = sorted(((round(unsteady(K), 3), K) for K in range(earliest, latest + 1)), key=lambda x: (x[0], x[1]))
        if scored:
            u, K = scored[0]
            if u == 0.0:
                return K, f"where it is steady{short}"
            return K, f"the steadiest stretch left ({int(round(100 * min(u, 1.0)))} % of it unsteady){short}"
        return latest, f"before {t.title[:22]} runs out"

    def _plan_seam(self, a, b):
        sa, sb = self.songs[a], self.songs[b]
        bar = self.bar_n
        L, kind = self._seam_len()
        tb, lm, on_lm, L2 = self._seam_b_start(sb, L)
        order = self._seam_order(a, b, L2, kind, tb)
        if order in ("bass", "cut"):
            # the bass swap (and the cut) bring B's BASS first: B must be where its bass carries, not merely its drums
            # (a bass swap from a bassless intro was a hole)
            tb, lm, on_lm, L2 = self._seam_b_start(sb, L, lanes=("bass", "drums"))
        L = L2
        K, why = self._seam_a_bar(a, L, order)
        # no clean stretch of A for this seam: a shorter, plainer one before a bad one (the acapella lead-in
        # over Mantra's breakdown came back in every campaign - dead air 0.07, a 12 dB lurch)
        if "unsteady" in why and order == "vocal":
            order = "drums"
            tb, lm, on_lm, L = self._seam_b_start(sb, L)
            K, why = self._seam_a_bar(a, L, order)
        if "unsteady" in why and L > 8:
            L = 8
            tb, lm, on_lm, L = self._seam_b_start(sb, L, lanes=(("bass", "drums") if order in ("bass", "cut") else ("drums",)))
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
        at = self._next_bar_clock()
        if self._clock() < int(sb.staged_at or 0) + self._bar_clock() // 2:
            # its staging cue and start land on the coming bar: a re-cue on the same bar lost to them (a record
            # arrived 72 bars past its landmark and ran out inside its own seam - the campaign); a bar later
            at += self._bar_clock()
            n -= 1
        if n < 2:
            return
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
        """NEXT is an order (2026-09-14): a running seam completes on the next bar; a planned seam starts at the
        next phrase; with no next record staged, the conductor stages its pick now and the seam comes as soon
        as it is ready, before the record's minimum play."""
        sm = self.seam
        bar = self.bar_n
        if sm is None:
            self._next_asap = True
            staged = [d for d, s in self.songs.items() if d != self.record_deck and not s.leaving and s.staged_at is not None]
            if not staged and self.auto > 0.0:
                free = [d for d in ("a", "b", "c", "d") if d not in self.songs and d not in self._decoding and d not in self._pending]
                nxt = self._pick_next() if free else None
                if nxt is not None:
                    self._decode(nxt, free[0])
                    self._move("plan", None, None, nxt.id, f"NEXT: {nxt.title[:24]} is being staged; the seam comes as soon as it is ready")
                    return None
            self._move("plan", None, None, None, "NEXT: the seam comes as soon as the next record is ready")
            return None
        if sm["state"] == "running":
            # complete it now: the last step before done fires on the next bar, the steps between are skipped
            done_off = max(off for off, name in sm["steps"] if name == "_st_done")
            last_off = max(off for off, name in sm["steps"] if name != "_st_done")
            for off, name in sm["steps"]:
                if off < last_off:
                    sm["fired"].add(off)
            sm["bar"] = bar + 2 - last_off
            sm["drums_bar"] = sm["bar"]
            self._seam_plan_mirror()
            self._move("plan", None, None, self._song_id(sm["b"]), "NEXT: the seam completes on the next bar")
            return None
        self._seam_move(self._phrase_bar_after(bar + 3), "NEXT")   # (three bars: the re-cue needs two to lock)
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

    def _seam_abandon(self, why):
        """A running seam whose new record is running out (looping its last bars, evicted, leaving): what it
        took goes back to the old record, it leaves, the seam is dropped. (An evicted song's harmony took the
        bed while its drums had gone back to the old record: the melody and the bass on different songs for the
        next seam - the campaign.)"""
        sm = self.seam
        a, b = sm["a"], sm["b"]
        at = self._next_bar_clock()
        back = []
        if a in self.songs and not self.songs[a].leaving:
            for ln in ("drums", "bass", "other", "vocals"):
                if self.lanes.get(ln) == b and (ln != "vocals" or self._vocal_ok(a, at)):
                    self._cross(ln, a, at, beats=1.0, shape=False)
                    back.append(ln)
                elif self.lanes.get(ln) == b:
                    self._cross(ln, None, at, beats=1.0, shape=False)
        if b in self.songs and not self.songs[b].leaving:
            self._leave_song(b, at + 2 * self._bar_clock())
        self._hygiene(at + int(self._beat_s() * RATE))
        self._move("seam", None, self._song_id(b), self._song_id(a), f"the seam abandoned: {why}" + (f" ({', '.join(back)} back)" if back else ""))
        self.seam, self.plan = None, None

    def _seam_fire(self):
        sm = self.seam
        if sm is None:
            return
        sb = self.songs.get(sm["b"])
        if sb is None or sb.leaving or sb.evicted or sb.loop is not None:
            self._seam_abandon(f"{sb.track.title[:22] if sb else 'the new record'} is running out")
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
        self._seam_clear_strays(at)
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
        if first:
            self._seam_clear_strays(at)
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
            self._seam_clear_strays(at)
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
        self._seam_clear_strays(at)
        # the cut lands B on its landmark: cued now, four bars ahead (the strip is its lock time), when it is
        # not sounding yet and not in place (an emergency cut took a record from deep in its body - the gate)
        if sb is not None and not sb.entered:
            sm = self.seam
            bar_b = self._bar_s(sb.track, sm["lm"])
            want = sm["lm"] - 4 * bar_b
            have = self._song_time_at(b, at)
            if want >= 0.0 and abs(have - want) > 1.0 * bar_b:
                self._cue_later.pop(b, None)
                self._post_cue(b, want, at)
                self.recue_log = (getattr(self, "recue_log", []) + [(self._tel().get("clock_s"), b, round(want, 1), self.bar_n + 4, "cut")])[-40:]
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
        pend = self._event_pending
        return {"deck": self.record_deck, "title": s.track.title, "since_bars": since, "min_bars": self._play_min_bars(),
                "settle_left": max(0, self._play_min_bars() - since), "kind": self._kind_at(self.record_deck, self._next_bar_clock()),
                "events_done": int(getattr(s, "events_done", 0)), "event_rate": int(getattr(self, "event_rate", 0)),
                "event": ({"kind": pend["kind"], "in_bars": pend["bar"] - self.bar_n, "why": pend["why"]} if pend else None)}
