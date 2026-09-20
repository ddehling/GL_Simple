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

MIN_RUNWAY_S = 90.0                    # (lib/dj/remix.py's: a landmark must leave this much body)
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
        self.event_pref = {"vocals": "some", "energy": "hold"}   # the VOCALS and ENERGY dials steer the events
        self.rehearsal_log = []            # every rehearsal's verdict (lib/dj/rehearse.py)
        self.silence_log = []              # the bars where the record itself was silent (its own breakdown)
        self._silent_since = None
        self._rehearsal_result = None
        self._rehearsing = None

    def set_event_rate(self, n):
        self.event_rate = int(max(0, n))

    def set_event_pref(self, vocals=None, energy=None):
        """VOCALS: none = no vocal events, some = a vocal first when one fits, lots = vocals first and two of them
        at MOMENTS lots; ENERGY: cool / amp lean the pairing toward a calmer / hotter song."""
        if vocals is not None:
            self.event_pref["vocals"] = vocals
        if energy is not None:
            self.event_pref["energy"] = energy

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
            d = self.deck_of(pend["track_id"])
            # dropped: its song never became ready, the seam it sits before is too close, it moved late three
            # times, or the ROOM CHANGED under it - the record is another song now, or the event's own song
            # became the record (your DROP onto it: the pending takeover then blocked every seam, and the record
            # ran out into 57 dead bars - the gate, 2026-09-16)
            gone = d is None and pend.get("staged_once") and not any(t.id == pend["track_id"] for t in self._decoding.values()) \
                and not any(p[0].id == pend["track_id"] for p in self._pending.values())
            if d is not None:
                pend["staged_once"] = True
            if time.time() - pend["t"] > 150.0 or (sm_ is not None and bar > sm_["bar"] - 3 * cb) \
                    or pend.get("record") != sa.track.id or d == self.record_deck or pend.get("late", 0) >= 3 or gone:
                self._event_pending = None
                why = ("its song left the decks" if gone else "the room changed" if (pend.get("record") != sa.track.id or d == self.record_deck)
                       else ("it came too late three times" if pend.get("late", 0) >= 3
                             else ("its seam is too close" if (sm_ is not None and bar > sm_["bar"] - 3 * cb) else "its song never became ready in time")))
                self._move("event", None, self._song_id(self.record_deck), int(pend["track_id"]), f"the event is dropped: {why}")
                return
            s = self.songs.get(d) if d else None
            if s is None or s.staged_at is None or self._clock() < int(s.staged_at) + self._bar_clock():
                return
            if self._moment is not None or (self.seam is not None and self.seam["state"] == "running"):
                return
            if bar < pend["bar"] - 2 * cb:
                return                                # armed two phrases before its bar (the countdown), not the moment its song is ready
            if bar >= pend["bar"] - 1:
                # ready late (a decode is real seconds): the event moves to the next phrase; the acapella's song
                # jumps so its sung stretch still begins on that bar
                K2 = self._phrase_bar_after(bar + 3)
                if pend["kind"] == "acapella" and pend.get("want_t") is not None:
                    self.jump(d, float(pend["want_t"]), K2)
                pend["bar"] = K2
                pend["late"] = int(pend.get("late", 0)) + 1
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
        pref_v = self.event_pref.get("vocals", "some")
        # VOCALS none: no vocal events; some: a vocal first when one fits, else the bass swap; lots: vocals only
        # (two acapellas at MOMENTS lots) - the bass swap only when no vocal fits at all
        kinds = ["bass"] if pref_v == "none" else ["vocal", "bass"]
        used = getattr(sa, "events_used", None)
        if used is None:
            used = sa.events_used = set()
        busy |= used
        lean = self.event_pref.get("energy", "hold")
        e_rec = None
        try:
            e_rec = float(sa.track.energy_proxy())
        except Exception:
            pass

        def _leaned(cands):
            # ENERGY cool / amp: the candidates re-ranked toward a calmer / hotter song (within the fit)
            if lean == "hold" or e_rec is None:
                return cands
            sgn = -1.0 if lean == "cool" else 1.0
            return sorted(cands, key=lambda c: -(c["score"] + 0.15 * sgn * ((c.get("energy") or e_rec) - e_rec)))
        seam_bar = sm["bar"]
        for kind in kinds:
            cands = _leaned(rec.get("vocal_over" if kind == "vocal" else "bass_under") or [])
            win_start, win_end = start_min, end_max
            if kind == "bass" and pref_v == "lots" and n_done == 0 and (rec.get("vocal_over") or []):
                continue                              # VOCALS lots: the first event is a vocal or nothing yet
            if kind == "bass":
                # a BASS UNDER changes the record: it is the record's EXIT, at the end of its minimum play, not a
                # phrase into it (one event per record at a phrase in halved the pace: 2.5 min per record - the
                # fifth campaign); it takes the seam's own bar - the seam is dropped when it is planned
                win_start = max(bar + 4 * cb, k0 + self._play_min_bars() - 2 * cb)
                win_end = seam_bar
                if win_end < win_start:
                    continue
            for c in cands:
                if c["id"] in busy:
                    continue
                if kind == "bass" and float(c.get("runway_s") or 0.0) < float(self.play_min_s) + 45.0:
                    # the takeover's song becomes the RECORD at its drop: it needs its minimum play and a seam after
                    # that, not the pairing file's 90 s floor (Glam Trap took over with three minutes of body,
                    # the seam after it moved twice and it looped its last bars - flag 'runout' 2026-09-19)
                    continue
                if kind == "vocal":
                    ga, gb = c["groove"]
                    # the first phrase boundary in the window where the record is inside that groove with eight bars to spare
                    K = None
                    for k in range(win_start, win_end + 1):
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
                                           "record": sa.track.id, "why": f"the vocal of {c['title'][:24]} over this groove (fit {c['fit']})"}
                else:
                    K = None
                    for k in range(win_start, win_end + 1):
                        if (k - k0) % cb == 0 and self._track_steady(sa.track, t_now + (k - bar - 1) * bar_a, 2 * cb, bad=("build", "breakdown")) is not False:
                            K = k
                            break
                    if K is None:
                        continue
                    ok, msg = self.stage_for_bar(c["id"], float(c["landmark"]) - 2 * bar_a * c.get("rate", 1.0), K - 2)
                    if not ok:
                        continue
                    self._event_pending = {"kind": "takeover", "track_id": int(c["id"]), "bar": int(K), "bars": 0, "t": time.time(),
                                           "record": sa.track.id, "why": f"the bassline of {c['title'][:24]} under these drums at its drop (fit {c['fit']})"}
                    self._seam_drop(f"the event is the exit: {c['title'][:22]} at its drop")   # (the takeover replaces the seam)
                sa.events_done = n_done + 1
                used.add(int(c["id"]))
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

    def _exit_event_coming(self):
        """A BASS UNDER event (the DJ's TAKEOVER) is the record's exit: while it is pending or in flight no seam
        is planned (the seam it replaced was planned again the next bar - the gate, 2026-09-16)."""
        pend = self._event_pending
        sa = self.songs.get(self.record_deck)
        if pend is not None and pend.get("kind") == "takeover" and sa is not None and pend.get("record") == sa.track.id:
            return True
        m = self._moment
        return m is not None and m.get("kind") == "takeover" and bool(m.get("auto"))

    def _record_silence_tick(self):
        """The record's OWN silence (a breakdown under -45 dBFS is the record, not a hole): noted once per stretch
        and kept in silence_log for the gate and the campaign, which count those bars apart."""
        s = self.songs.get(self.record_deck)
        if s is None or not s.env or not s.entered:
            self._silent_since = None
            return
        try:
            import numpy as np
            t = self._song_time_at(self.record_deck, self._next_bar_clock())
            bar = self._bar_s(s.track, t)
            i0, i1 = max(0, int(t * 2)), int((t + bar) * 2) + 1
            lv = [float(np.mean(np.asarray(e[i0:i1], dtype=np.float32))) for e in s.env.values() if e is not None and len(e) > i1]
        except Exception:
            return
        if not lv:
            return
        if max(lv) < 0.06:
            self.silence_log.append({"clock_s": self._tel().get("clock_s"), "title": s.track.title, "t": round(t, 1), "bar": self.bar_n})
            if getattr(self, "_silent_since", None) is None:
                self._silent_since = self.bar_n
                self._note(f"{s.track.title[:24]} is silent here ({t:.0f} s: its own breakdown)")
        else:
            self._silent_since = None

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
        self._record_silence_tick()
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
        # the events, BEFORE the seam check: a pending exit event (the takeover) has dropped its seam - stepped
        # after it, it never armed and blocked every seam for its whole timeout (the gate, 2026-09-16)
        try:
            self._event_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"event: {type(e).__name__}: {e}"
        sm = self.seam
        if sm is None:
            b = self._next_record_deck()
            if b is not None and not self._exit_event_coming():
                self._plan_seam(self.record_deck, b)
            return
        # the record is looping its last bars (its body has ended): the seam fires NOW whatever B's place - the
        # cut, B whole on the one - rather than the record being evicted into silence (2026-09-14)
        sa = self.songs.get(self.record_deck)
        if sa is not None and sa.loop is not None and sm["bar"] > bar + 1:
            # the cut NOW - no four-bar strip first: a looping record's loop expired inside its own strip and it
            # left three seconds in, six seconds of nothing until the cut (the control gate, seed 17); the next
            # record comes in whole on the next bar, cued to its landmark
            sm["bar"], sm["drums_bar"], sm["order"], sm["len"] = bar + 1, bar + 1, "cut", 4
            sm["steps"] = [(0, "_st_cut"), (1, "_st_done")]
            sm["moved"] += 1                          # (an emergency: its landmark lands where it can)
            sb = self.songs.get(sm["b"])
            if sb is not None and not sb.entered:
                at = self._next_bar_clock()
                if abs(self._song_time_at(sm["b"], at) - sm["lm"]) > self._bar_s(sb.track, sm["lm"]):
                    self._cue_later.pop(sm["b"], None)
                    self._post_cue(sm["b"], sm["lm"], at)
            self._seam_plan_mirror()
            self._move("plan", None, None, self._song_id(sm["b"]), f"the seam now, as a cut: {sa.track.title[:22]} is looping its last bars")
            sm["state"] = "running"
            self._seam_fire()
            return
        # the plan mirrors the seam (a moment's end clears the plan; the seam stands)
        if self.plan is None or self.plan.get("what") != "seam" or self.plan.get("bar") != sm["bar"]:
            self._seam_plan_mirror()
        try:
            self._rehearse_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"rehearsal: {type(e).__name__}: {e}"
        sm = self.seam
        if sm is None:
            return
        if bar >= sm["bar"] - 1:
            if self.hold or self._moment is not None or self._break is not None or self._recall is not None:
                # moved ONCE to after what has the room (a phrase per check under a sixteen-bar moment pushed the
                # plan back three times in a row: "potential new songs continuously pushed back", 2026-09-17)
                m = self._moment
                if m is not None:
                    end = int(m["L"]) + max((off for off, _fn in m["script"]), default=0) + 1
                    self._seam_move(self._phrase_bar_after(max(bar + 1, end)), f"after the {m['label']}")
                elif self._break is not None:
                    end = bar + 1 + int(math.ceil(max(0, self._break[0] - self._next_bar_clock()) / max(self._bar_clock(), 1)))
                    self._seam_move(self._phrase_bar_after(max(bar + 1, end)), "after the BREAK")
                else:
                    self._seam_move(self._phrase_bar_after(bar + 1), "held" if self.hold else "after the RECALL")
                return
            # the pre-flight: B must be where the plan put it (within a bar); else the seam moves a phrase and B
            # is cued again, once - then the seam is dropped and planned afresh
            sb = self.songs.get(sm["b"])
            if sb is not None and not sb.entered:
                at = self._next_bar_clock()
                cued = sm.get("cued_clock")
                fresh = cued is not None and self._clock() < cued + self._bar_clock()
                # a cue posted within the last bar has not executed (or locked) yet: the picture would be the
                # old position (a re-cue a bar after NEXT's read "-27 s from its place" and cued again - the
                # gate, 2026-09-16); the cue was computed for this very bar - it is trusted, not re-checked
                want = sm["tb"] if sm["order"] != "vocal" else sm["tb"] - VOCAL_LEAD_BARS * self._bar_s(sb.track, sm["tb"])
                have = self._song_time_at(sm["b"], at)
                if not fresh and abs(have - want) > 1.0 * self._bar_s(sb.track, want):
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
                if getattr(self.songs[h], "user_k", None) is not None:
                    # a part YOU brought is never a stray: it stays through the seam until its stretch ends or you
                    # take it out ("the system overrode my vocals choice after a few seconds" - flag 2026-09-19)
                    self._move("plan", ln, self._song_id(h), None, f"your {ln} of {self.songs[h].track.title[:22]} stays through the seam (your order)")
                    continue
                self._cross(ln, None, at, beats=2.0, shape=False)
                self._note(f"the {ln} of {self.songs[h].track.title[:22]} out: a seam begins and it belongs to neither record")

    # -- the rehearsal: the headphone cue ---------------------------------------------------------------------------
    def _rehearse_step(self):
        """Two phrases before a planned seam, render it from the decks' stems and judge it (lib/dj/rehearse.py);
        a seam that fails moves a phrase (twice at most), then changes its order, then falls back to the
        breakdown and cut. The verdict is kept on the seam and in rehearsal_log."""
        sm = self.seam
        cb = self.change_bars
        res = getattr(self, "_rehearsal_result", None)
        if res is not None:
            self._rehearsal_result = None
            seq, ok, why = res
            if sm is None or sm.get("seq") != seq or sm["state"] != "planned":
                return
            self.rehearsal_log.append({"seq": seq, "a": sm["a"], "b": sm["b"], "order": sm["order"], "len": sm["len"], "ok": ok, "why": why, "bar": self.bar_n})
            if ok is None:
                sm["rehearsed"] = f"not rehearsed: {why}"
                return
            if ok:
                sm["rehearsed"] = f"clean - {why}"
                self._note(f"rehearsed: {why}")
                return
            # a hole AT the swap bar is B's own quiet bar before its drop, not a place: the swap moves ON the
            # drop (B a bar later), once - moving a phrase would find the same hole
            last = {"cut": 4, "bass": sm["len"], "vocal": VOCAL_LEAD_BARS + sm["len"]}.get(sm["order"], sm["len"])
            hole_bar = None
            if "a hole at bar" in why:
                try:
                    hole_bar = int(why.split("a hole at bar")[1].split()[0])
                except (ValueError, IndexError):
                    hole_bar = None
            sb = self.songs.get(sm["b"])
            if hole_bar is not None and hole_bar >= last - 1 and not sm.get("swap_on_drop") and sb is not None and not sb.entered:
                sm["tb"] += self._bar_s(sb.track, sm["tb"])
                sm["swap_on_drop"], sm["rehearsed"] = True, None
                self._seam_cue_b()
                self._move("plan", None, None, self._song_id(sm["b"]), f"rehearsed {why}: the swap lands on its drop instead")
                return
            tries = int(sm.get("rehearsal_tries", 0)) + 1
            sm["rehearsal_tries"] = tries
            # THE SEAM STAYS WHERE IT IS: a failed rehearsal changes the seam's shape at its bar first (the other
            # order, then the breakdown and cut), and moves it a phrase later only when the record has the body
            # for it - moving first pushed the next song back a phrase at a time (three moves in thirty seconds),
            # ran the record out into its last-bars loop, and landed the next record deep in its body (flags
            # 'plan', 'runout' 2026-09-19)
            sa_ = self.songs.get(self.record_deck)
            room = False
            try:
                if sa_ is not None:
                    at_ = self._next_bar_clock()
                    t_end = self._song_time_at(self.record_deck, at_) + (sm["bar"] + int(sm["len"]) + 2 * cb - self.bar_n) * self._bar_s(sa_.track)
                    room = sa_.loop is None and t_end < float(sa_.track.duration_s) - 8.0
            except Exception:
                room = False
            if tries == 1 and sm["order"] in ("drums", "bass"):
                new_order = "bass" if sm["order"] == "drums" else "drums"
                sm["order"], sm["steps"], sm["rehearsed"] = new_order, self._seam_steps(new_order, sm["len"]), None
                self._seam_cue_b()
                self._move("plan", None, None, self._song_id(sm["b"]), f"the seam rehearsed {why}: as a {ORDER_WORDS[new_order]} instead, at the same bar")
            elif tries <= 2 and room:
                sm["rehearsed"] = None
                self._seam_move(self._phrase_bar_after(sm["bar"] + 1), f"rehearsed: {why}")
            elif sm["order"] != "cut":
                sm["order"], sm["len"], sm["steps"], sm["rehearsed"] = "cut", 4, self._seam_steps("cut", 4), f"fell back to the cut - {why}"
                sm["drums_bar"] = sm["bar"]
                self._move("plan", None, None, self._song_id(sm["b"]), f"the seam rehearsed {why}: the breakdown and a cut instead, at the same bar")
            else:
                sm["rehearsed"] = f"plays as rehearsed - {why}"
            return
        if sm is None or sm["state"] != "planned" or sm.get("rehearsed") is not None or getattr(self, "_rehearsing", None) == (sm.get("seq"), sm["bar"]):
            return
        left = sm["bar"] - self.bar_n
        if left > 2 * cb + 1 or left < cb - 1:
            return                                    # two phrases ahead, a phrase at least (a move re-rehearses if there is time)
        sb = self.songs.get(sm["b"])
        if sb is None or sb.entered or sb.staged_at is None or self._clock() < int(sb.staged_at):
            return
        from lib.dj.rehearse import rehearse
        self._rehearsing = (sm.get("seq"), sm["bar"])
        rehearse(self, sm, None)

    def _next_record_deck(self):
        """The next record: a song staged and waiting (your UP NEXT first), else a song you brought a part of
        in (it is heard; the seam morphs into it)."""
        cands = [(d, s) for d, s in self.songs.items() if d != self.record_deck and not s.leaving and not s.evicted
                 and s.staged_at is not None and s.env is not None and not s.was_bed and not getattr(s, "part_only", False)]
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

    def _seam_lm_off(self, order, L):
        """Bars from the seam's first bar to where B's LANDMARK should fall: one bar after the step that brings
        B's harmony, so the drop is B's own build resolving inside its harmony, not a step in the mix (the
        drop's 9-13 dB rise on the completion bar was the lurch class every campaign kept - v6)."""
        if order == "bass":
            return DARKEN_BARS + 1
        if order == "vocal":
            return VOCAL_LEAD_BARS + L + 1
        if order == "cut":
            return 4
        return L + 1

    def _seam_b_start(self, sb, L, lanes=("drums",), lm_off=None):
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
        off = L if lm_off is None else lm_off
        # the stems the seam brings first must carry through the bars they play under the old record - judged
        # BAR BY BAR (every bar's mean over 0.7 of the line): a pre-drop pattern on and off every bar passed the
        # two-second test and the campaign heard it as holes and lurches
        lead = max(2.0 * bar_b, min(off, L) * bar_b)
        t0 = lm - off * bar_b
        if t0 >= floor_t and self._stem_carries(sb.env, t0, lead, lanes=lanes, bar_s=bar_b) is not False:
            return t0, lm, True, L
        step = 4 * bar_b
        tt = t0 - step
        while tt >= floor_t:
            if self._stem_carries(sb.env, tt, lead, lanes=lanes, bar_s=bar_b) is not False:
                return tt, lm, False, L
            tt -= step
        if lm_off is None:
            t_in = self._steady_from(t, sb.env, floor_t, L, strict=False, lanes=lanes)
            if t_in is None:
                t_in = max(floor_t, t0)
            t_in = max(t_in, floor_t)
            # its stems carry from later than a full seam before its landmark: a SHORTER seam that still completes
            # on the landmark (eight bars at least), else the landmark falls inside the seam
            room = int(math.floor((lm - t_in) / max(bar_b, 1e-6) / 4.0)) * 4
            if room >= 8 and room < L and lm_off is None:
                return lm - room * bar_b, lm, True, room
            return t_in, lm, False, L
        # nothing carries before its landmark (a build with thin drums, a short groove between breakdowns -
        # Ruhi): a LATER landmark - the first place its leading stems carry for the whole lead AND the song is
        # steady for two phrases after the seam completes there. (The old fallback returned the first steady
        # place, which was the landmark itself: the drop landed on the seam's FIRST bar under the old harmony,
        # and the record ran into its silent breakdown eight bars after the swap - the gate, 2026-09-16.)
        # None when there is no such place: the caller falls back to the breakdown and cut.
        end = float(t.duration_s) - MIN_RUNWAY_S - 2 * self.change_bars * bar_b
        cands = sorted({t.nearest_downbeat(float(x["start_s"])) for x in (t.sections or [])
                        if x.get("kind") in ("groove", "intro") and float(x["start_s"]) >= floor_t}
                       | {t.nearest_downbeat(floor_t + 4 * k * bar_b) for k in range(0, int(max(0.0, end - floor_t) / (4 * bar_b)) + 1)})
        for c in cands:
            lm_c = c + off * bar_b
            if lm_c > end:
                break
            if self._stem_carries(sb.env, c, lead, lanes=lanes, bar_s=bar_b) is not False \
                    and self._steady_at(t, sb.env, lm_c, 2 * self.change_bars, False, ("drums", "bass")) is not False:
                return c, lm_c, True, L
        return None

    def _seam_b_cut(self, sb):
        """B's start for the breakdown and cut: it enters WHOLE on the cut bar, so nothing of it sounds before -
        only that it carries from its landmark for two phrases matters; else the first later place it does.
        Returns (t_start, landmark, on_landmark, 4) - the start four bars (the strip) before the landmark."""
        t = sb.track
        lm = self._landmark(t, sb.env)
        bar_b = self._bar_s(t, lm)
        floor_t = 2.0 * bar_b
        span = 2 * self.change_bars * bar_b

        def ok(x):
            return self._stem_carries(sb.env, x, span, lanes=("drums", "bass"), bar_s=bar_b) is not False
        if ok(lm):
            return max(floor_t, lm - 4 * bar_b), lm, True, 4
        end = float(t.duration_s) - MIN_RUNWAY_S - span
        cands = sorted({t.nearest_downbeat(float(x["start_s"])) for x in (t.sections or []) if x.get("kind") == "groove" and float(x["start_s"]) > lm}
                       | {t.nearest_downbeat(lm + 4 * k * bar_b) for k in range(1, int(max(0.0, end - lm) / (4 * bar_b)) + 1)})
        for c in cands:
            if c > end:
                break
            if ok(c):
                return c - 4 * bar_b, c, True, 4
        return max(floor_t, lm - 4 * bar_b), lm, False, 4

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
        lm0 = self._landmark(sb.track, sb.env)
        order = self._seam_order(a, b, L, kind, lm0 - (L + 1) * self._bar_s(sb.track, lm0))

        def place(order_, L_):
            # B's start, for the order: the stems it brings first must carry (the bass swap brings its BASS - a
            # bass swap from a bassless intro was a hole) and its landmark lands a bar after its harmony is in;
            # the cut brings B whole on the cut bar. None: no place for this order and length.
            if order_ == "cut":
                return self._seam_b_cut(sb)
            lanes_b = ("bass", "drums") if order_ == "bass" else ("drums",)
            r = self._seam_b_start(sb, L_, lanes=lanes_b, lm_off=self._seam_lm_off(order_, L_))
            if r is not None and r[2]:
                # the harmony swaps a bar before B's drop - unless B's harmony is OUT in that bar (the classic
                # silent bar before a drop): then the swap lands ON the drop (the rehearsal heard a hole at the
                # swap bar in half the seams of campaign v7)
                tb_, lm_, on_, Lr = r
                bar_b = self._bar_s(sb.track, lm_)
                if self._stem_carries(sb.env, lm_ - bar_b, bar_b, lanes=("bass", "other"), bar_s=bar_b) is False:
                    r = (tb_ + bar_b, lm_, on_, Lr)
            return r

        def placed(order_, L_):
            # the order and length that CAN be placed: this one, else drums first, else eight bars, else the cut
            r = place(order_, L_)
            if r is None and order_ in ("vocal", "bass"):
                order_ = "drums"
                r = place(order_, L_)
            if r is None and L_ > 8:
                L_ = 8
                r = place(order_, L_)
            if r is None:
                order_, L_ = "cut", 4
                r = place(order_, L_)
            return order_, r
        order, (tb, lm, on_lm, L) = placed(order, L)
        if order == "cut":
            kind = "cut"
        K, why = self._seam_a_bar(a, L, order)
        # no clean stretch of A for this seam: a shorter, plainer one before a bad one (the acapella lead-in
        # over Mantra's breakdown came back in every campaign - dead air 0.07, a 12 dB lurch)
        if "unsteady" in why and order == "vocal":
            order, (tb, lm, on_lm, L) = placed("drums", L)
            K, why = self._seam_a_bar(a, L, order)
        if "unsteady" in why and L > 8:
            order, (tb, lm, on_lm, L) = placed(order, 8)
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
        # a vocal YOU hold over the room is judged against the harmony that arrives with the seam, and said (it
        # stays: your order) - a vocal brought over one record rode into the next one's harmony unjudged (flags
        # 'clash', 'vocals muddy' 2026-09-19)
        try:
            vh = self.lanes.get("vocals")
            if vh is not None and vh not in (a, b) and vh in self.songs and getattr(self.songs[vh], "user_k", None) is not None:
                fit = self._pair_fit(vh, b)
                if fit is not None and fit < 0.55:
                    self._move("plan", "vocals", self._song_id(vh), sb.track.id,
                               f"your vocal of {self.songs[vh].track.title[:22]} against {sb.track.title[:22]}'s harmony from the seam: fit {fit:.2f}, a clash - it stays (your order); take it out or choose another next record")
        except Exception:
            pass
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
            sm["cued_clock"] = int(later[0])
            self.recue_log = (getattr(self, "recue_log", []) + [(self._tel().get("clock_s"), sm["b"], round(later[1], 1), K, "deferred")])[-40:]
            return
        now_t = self._song_time_at(sm["b"], at)
        if abs(now_t - cue) < 0.5 * self._bar_s(sb.track, cue):
            return
        self._post_cue(sm["b"], cue, at)
        sm["cued_clock"] = int(at)
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
            sm["moved"] += 1                          # (you moved it: its landmark lands where it can)
            self._seam_plan_mirror()
            self._move("plan", None, None, self._song_id(sm["b"]), "NEXT: the seam completes on the next bar")
            return None
        K = self._phrase_bar_after(bar + 3)               # (three bars: the re-cue needs two to lock)
        if K >= sm["bar"]:
            # already due within the phrase: SAID, not a silent no-op (the control gate, 2026-09-16)
            self._move("plan", None, None, self._song_id(sm["b"]), f"NEXT: the seam is already due in {sm['bar'] - bar} bars")
            return None
        # your NEXT is an order, but the seam it forces must still hold the room: where the record is in a
        # build or a breakdown at that bar, a morph leaves its silent drums under the new harmony (a bass swap
        # forced onto Eidothea's breakdown with Dissolution's intro bass: three bars at -65 dB - the control
        # gate, seed 24) - it becomes the CUT there, the next record whole at its landmark
        sa, sb = self.songs.get(self.record_deck), self.songs.get(sm["b"])
        if sa is not None and sb is not None and not sb.entered and sm["order"] != "cut":
            at = self._next_bar_clock()
            t_k = self._song_time_at(self.record_deck, at) + (K - bar - 1) * self._bar_s(sa.track)
            if self._track_steady(sa.track, t_k, int(sm["len"]), bad=("build", "breakdown", "outro")) is False:
                tb, lm, on_lm, L = self._seam_b_cut(sb)
                sm["order"], sm["len"], sm["kind"], sm["steps"] = "cut", L, "cut", self._seam_steps("cut", L)
                sm["tb"], sm["lm"], sm["on_lm"] = float(tb), float(lm), bool(on_lm)
                sm["drums_bar"] = sm["bar"]
                self._seam_move(K, f"NEXT ({sa.track.title[:20]} is in its {(sa.track.section_at(t_k) or {}).get('kind', 'break')} there: the cut)")
                return None
        self._seam_move(K, "NEXT")
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
        # what STAYS must be audible: the strip takes the drums and the bass only when the record's melody or
        # vocal is on its lane and sounding through the four bars; else the bass carries the breakdown (a strip
        # left one silent vocal stem - four bars at -79 dB - the control gate, seed 17)
        span = 4 * self._bar_clock()
        own = {ln for ln in ("bass", "other", "vocals") if self.lanes.get(ln) == a}
        stays = [ln for ln in ("other", "vocals") if ln in own
                 and ((self._sings_through(a, at, 4) is True) if ln == "vocals" else ((self._stem_level_span(a, ln, at, at + span) or 0.0) >= 0.2))]
        # the strip takes only what the RECORD holds: when another song's bassline sits under these drums (a
        # TAKEOVER's), taking that bass takes its melody with it (harmony travels together) and the song leaves -
        # the room was one silent vocal for four bars (the gate, seed 39, 2026-09-19); its bassline carries instead
        bh = self.lanes.get("bass")
        other_bass = bh is not None and bh != a and bh in self.songs and not self.songs[bh].leaving and self.songs[bh].entered
        if not stays and not other_bass and "bass" not in own:
            # nothing would carry a breakdown: the drums stay until the cut lands on the bar
            self._move("seam", None, self._song_id(a), None, f"no breakdown before the cut: nothing of {sa.track.title[:24]} would carry it; its drums stay to the bar")
            return
        self._cross("drums", None, at, beats=1.0, shape=False)
        take_bass = "bass" in own and bool(stays)
        if take_bass:
            self._cross("bass", None, at, beats=1.0, shape=False)
        self._hygiene(at + int(self._beat_s() * RATE))
        carried = (f"its {'melody' if 'other' in stays else 'vocal'} stays" if take_bass
                   else (f"{self.songs[bh].track.title[:20]}'s bassline carries it" if other_bass else ("its bass carries it" if "bass" in own else "what is left carries it")))
        self._move("seam", None, self._song_id(a), None,
                   f"the breakdown: {sa.track.title}'s drums{' and bass' if take_bass else ''} out for 4 bars, {carried}")

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
            # the landmark's bar is lm_off bars from the seam's first bar (one after the harmony swap: bar 5 of a
            # bass swap, L + 1 of a drums-first morph, 4 of a cut); this step fires done_off bars in - the error
            # is measured where the landmark was DUE (a bass swap read +28 bars at its done bar - campaign v8)
            done_off = max(off for off, name in sm["steps"] if name == "_st_done")
            due_off = self._seam_lm_off(sm["order"], sm["len"]) - (1 if sm.get("swap_on_drop") else 0)
            err = (t_b - (sm["lm"] + (done_off - due_off) * bar_b)) / max(bar_b, 1e-6)
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
        return {"a": sa.track.title if sa else "?", "b": sb.track.title if sb else "?", "bar": sm["bar"], "in_bars": sm["bar"] - self.bar_n, "seq": sm["seq"],
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
                "event": ({"kind": pend["kind"], "in_bars": pend["bar"] - self.bar_n, "why": pend["why"], "track_id": pend["track_id"], "bar": pend["bar"]} if pend else None)}
