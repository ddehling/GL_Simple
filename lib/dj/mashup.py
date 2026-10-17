"""MASHUP (docs/DJ_MASHUP.md): three songs taken apart and played as one.

A mixin for RemixConductor, like RecordPlay: when `policy == "mashup"` the conductor's step calls
`_mashup_step` instead of the record's. Three SLOTS hold three songs whose tempos and keys lock; a SCENE says
who holds the drums, who holds the harmony (bass + melody, always together - the clash rule) and who holds the
vocal (or nobody). A scene lasts `mash_bars` (the MOMENTS dial: 32 / 16 / 8); each new one changes one or two
of the three; every few scenes a shaped one - a DROP (everything but the vocal out for the last two beats,
then the scene slams in) or a BREAKDOWN (no drums for a scene, then the drop). A song that comes in is cued to
where its part sounds a bar ahead; a song that keeps a part is never jumped, except as a hot cue on a scene's
bar when it is running out. After ROTATE_SCENES scenes the slot used least gives way to a new song that fits
the two that stay. Your pads (`mash_pad`) move a part on the next bar; the routine carries on from there.
"""
from __future__ import annotations

import time

from lib.dj.submix import RATE

SLOTS = 3
ROTATE_SCENES = 12               # the set moves on: a slot changes song after this many scenes
SHAPE_EVERY = 4                  # a DROP every this many scenes (when the next scene changes the drums or the harmony)
BREAKDOWN_EVERY = 7              # ...and a BREAKDOWN scene (no drums) every this many, the drop after it
RUNOUT_S = 50.0                  # a song with less body than this ahead is hot-cued back to a groove on its next scene bar
HISTORY = 6


def scene_words(scene, titles):
    """'drums A · harmony B · vocal C' in words for THE PLAN and the story row."""
    if scene is None:
        return "-"

    def t(i):
        return titles[i][:20] if (i is not None and 0 <= i < len(titles) and titles[i]) else "-"
    d, h, v = scene.get("drums"), scene.get("harm"), scene.get("vocal")
    out = [f"drums {t(d)}" if d is not None else "no drums", f"harmony {t(h)}" if h is not None else "no harmony"]
    out.append(f"vocal {t(v)}" if v is not None else "no vocal")
    shape = scene.get("shape")
    return " · ".join(out) + (f"  ({shape})" if shape else "")


class MashupPlay:
    # -- state ---------------------------------------------------------------------------------------------------
    def _mash_init_state(self):
        self._mash = None
        self.mash_bars = 16
        self.mash_wish = None            # the Director's UP NEXT head: the next song to come in when a slot rotates

    def mash_reset(self):
        """Leaving the mode (LAYERS back to two / three): the record is the song holding the drums."""
        m = getattr(self, "_mash", None)
        self._mash = None
        if m is None:
            return
        d = self.lanes.get("drums") or self.lanes.get("bass") or self.master
        if d in self.songs:
            self.record_deck = d
            s = self.songs[d]
            s.play_from_k = self.bar_n
            for x in self.songs.values():
                x.part_only = x is not s and x.entered
            # the record plays WHOLE: its harmony and vocal come back to it over a bar (the room was left split three ways -
            # drums one song, harmony another, vocal a third - the mashup smoke's way back); the others leave once silent
            at = self._next_bar_clock()
            moved = []
            for ln in ("drums", "bass", "other", "vocals"):
                if self.lanes.get(ln) != d:
                    self._cross(ln, d, at, beats=4.0, shape=False)
                    moved.append(ln)
            if moved:
                self._hygiene(at + int(4 * self._beat_s() * RATE))
                self._move("mashup", None, s.track.id, None, f"MASHUP over: {s.track.title[:26]} is the record, whole ({', '.join(moved)} back to it)")

    # -- the helpers ---------------------------------------------------------------------------------------------
    def _mash_deck(self, i):
        m = self._mash
        tid = m["tids"][i] if m else None
        if tid is None:
            return None
        d = self.deck_of(tid)
        if d is None or d not in self.songs or self.songs[d].staged_at is None or self.songs[d].leaving:
            return None
        return d

    def _mash_ready(self):
        return [i for i in range(SLOTS) if self._mash_deck(i) is not None]

    def _mash_titles(self):
        m = self._mash
        return [(self._track_title(t) if t is not None else "") for t in (m["tids"] if m else [None] * SLOTS)]

    def _mash_sings(self, i):
        d = self._mash_deck(i)
        if d is None:
            return False
        s = self.songs[d]
        vr = float((s.stem_rms or {}).get("vocals") or 0.0)
        ref = float((self.ref_rms or {}).get("vocals") or 0.0) or 0.05
        return vr >= 0.25 * ref or bool(getattr(s.track, "hook", None))

    def _mash_lists(self, tid):
        try:
            return self.pairings_for(int(tid)) or {}
        except Exception:
            return {}

    def _mash_pick(self, keep_tids):
        """A song for a slot that fits the songs staying: in their pairing lists (a vocal that sits over one, a
        bassline that swaps under one), inside the tempo wall, its key fitting the room, not played tonight, not on a
        deck. The Director's UP NEXT head first when it fits."""
        from lib.dj.remix import ENTRY_FIT, RATE_MAX, RATE_MIN
        busy = {s.track.id for s in self.songs.values()} | {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        avoid = set(getattr(self, "avoid_ids", ()) or ()) | busy | set(getattr(self, "unfit_ids", ()) or ())

        def fits(t):
            if t is None or t.id in avoid or not t.bpm or not getattr(t, "has_stems", True):
                return False
            rate = float(self.master_bpm) / float(t.bpm)
            if not (RATE_MIN <= rate <= RATE_MAX):
                return False
            _sh, fit = self._key_fit(t)
            return fit is None or fit >= ENTRY_FIT
        wish = getattr(self, "mash_wish", None)
        if wish is not None:
            t = self._track(int(wish))
            if fits(t):
                return t
        score = {}
        for k in keep_tids:
            if k is None:
                continue
            rec = self._mash_lists(k)
            for x in (rec.get("vocal_over") or []):
                score[int(x["id"])] = score.get(int(x["id"]), 0.0) + 1.0 + 0.3 * float(x.get("score") or 0.0)
            for x in (rec.get("bass_under") or []):
                score[int(x["id"])] = score.get(int(x["id"]), 0.0) + 1.0 + 0.3 * float(x.get("score") or 0.0)
        for tid, _sc in sorted(score.items(), key=lambda z: -z[1]):
            t = self._track(tid)
            if fits(t):
                return t
        try:
            return self._pick_next()
        except Exception:
            return None

    # -- the step --------------------------------------------------------------------------------------------------
    def _mashup_step(self):
        if getattr(self, "_mash", None) is None:
            self._mash_start()
            return
        m = self._mash
        bar = self.bar_n
        if m.get("k") == bar:
            return
        m["k"] = bar
        self._mash_fill_slots()
        ready = self._mash_ready()
        if len(ready) < 2:
            # waiting for the songs: the first song plays whole meanwhile
            if bar >= m["next_bar"] - 1:
                m["next_bar"] = bar + 2
            return
        if m.get("next") is None:
            m["next"] = self._mash_next_scene(m["scene"])
            self._mash_say_next()
        nb = int(m["next_bar"])
        # one bar ahead: the songs coming in are cued to where their parts sound, landing on the scene's bar
        if bar >= nb - 2 and not m.get("cued"):
            m["cued"] = True
            self._mash_cue(m["next"], nb)
        if bar >= nb - 1 and m.get("applied") != nb:
            m["applied"] = nb
            self._mash_apply(m["next"], nb)
            m["scene"] = m["next"]
            m["history"] = (m.get("history") or [])[-(HISTORY - 1):] + [m["scene"]]
            m["n"] = int(m.get("n", 0)) + 1
            for slot in (m["scene"].get("drums"), m["scene"].get("harm"), m["scene"].get("vocal")):
                if slot is not None:
                    m["used"][slot] += 1
            m["next"], m["cued"] = None, False
            m["next_bar"] = nb + int(self.mash_bars if m["scene"].get("shape") != "breakdown" else max(8, self.mash_bars // 2))
            self._mash_rotate_step()

    def _mash_start(self):
        """The record playing becomes slot 0; two partners that fit it are staged as parts."""
        anchor = self.record_deck if self.record_deck in self.songs else self.master
        if anchor not in self.songs:
            return
        if getattr(self, "seam", None) is not None:
            try:
                self._seam_drop("MASHUP: three songs, not a seam")
            except Exception:
                self.seam = None
        a = self.songs[anchor]
        self._mash = {"tids": [a.track.id, None, None], "scene": {"drums": 0, "harm": 0, "vocal": 0 if self.lanes.get("vocals") == anchor else None},
                      "next": None, "next_bar": self.bar_n + max(8, int(self.mash_bars)), "n": 0, "used": [1, 0, 0], "history": [],
                      "rotations": 0, "since_rot": 0, "leaving": None, "k": None, "cued": False, "applied": None, "t0": time.time()}
        self._mash_fill_slots()
        self._move("mashup", None, a.track.id, None, f"MASHUP: {a.track.title[:28]} and two songs that fit it, taken apart and played as one - a scene every {self.mash_bars} bars")

    def _mash_fill_slots(self):
        """An empty slot gets a song: an entered or staged song already on a deck first (the next record staged by the
        record policy), else a fresh pick, staged as a part."""
        m = self._mash
        for i in range(SLOTS):
            if m["tids"][i] is not None:
                if self.deck_of(m["tids"][i]) is None and not any(t.id == m["tids"][i] for t in self._decoding.values()) \
                        and not any(p[0].id == m["tids"][i] for p in self._pending.values()):
                    if m.get("stage_t", {}).get(i) and time.time() - m["stage_t"][i] > 30.0:
                        self._move("mashup", None, None, m["tids"][i], f"MASHUP: {self._track_title(m['tids'][i])[:24]} did not reach a deck - another is picked")
                        m["tids"][i] = None
                continue
            keep = [t for t in m["tids"] if t is not None]
            on = [s for d, s in self.songs.items() if not s.leaving and s.staged_at is not None and s.track.id not in keep]
            t = on[0].track if on else self._mash_pick(keep)
            if t is None:
                continue
            m["tids"][i] = t.id
            m.setdefault("stage_t", {})[i] = time.time()
            if self.deck_of(t.id) is None:
                ok, msg = self.stage_track(t.id, as_next=False)
                if not ok:
                    m["tids"][i] = None
                    if not m.get("said_full"):
                        m["said_full"] = True
                        self._move("mashup", None, None, t.id, f"MASHUP: {t.title[:24]} could not be staged ({msg})")
                    continue
            self._move("mashup", None, None, t.id, f"MASHUP: {t.title[:28]} comes into slot {i + 1}")

    # -- the routine -----------------------------------------------------------------------------------------------
    def _mash_next_scene(self, cur):
        """The next combination from the current one: one or two of (drums, harmony, vocal) change; the vocal sits over
        a harmony its pairing lists name when they know; recent scenes are not repeated; the slots used least are
        preferred; a DROP every SHAPE_EVERY scenes, a BREAKDOWN every BREAKDOWN_EVERY."""
        m = self._mash
        ready = self._mash_ready()
        leaving = m.get("leaving")
        avail = [i for i in ready if i != leaving]
        if not avail:
            avail = ready
        n = int(m.get("n", 0)) + 1
        if cur is not None and cur.get("shape") == "breakdown":
            # after the breakdown: the drums slam back with the drop (the same harmony and vocal, the drums from the slot used least)
            d = min(avail, key=lambda i: (m["used"][i], i == cur.get("harm")))
            return {"drums": d, "harm": cur.get("harm") if cur.get("harm") in avail else avail[0], "vocal": cur.get("vocal") if cur.get("vocal") in avail else None, "shape": "drop"}
        if n % BREAKDOWN_EVERY == 0 and cur is not None and cur.get("harm") in avail:
            return {"drums": None, "harm": cur.get("harm"), "vocal": cur.get("vocal") if cur.get("vocal") in avail else None, "shape": "breakdown"}
        tids = m["tids"]
        recent = [(s.get("drums"), s.get("harm"), s.get("vocal")) for s in (m.get("history") or [])[-4:]]
        best, best_sc = None, -1e9
        sings = {i: self._mash_sings(i) for i in avail}
        for d in avail:
            for h in avail:
                for v in [None] + [i for i in avail if sings.get(i)]:
                    cand = (d, h, v)
                    changes = 0 if cur is None else sum(1 for a, b in zip(cand, (cur.get("drums"), cur.get("harm"), cur.get("vocal"))) if a != b)
                    if cur is not None and changes == 0:
                        continue
                    sc = 0.0
                    sc += {1: 1.0, 2: 1.2, 3: -0.8}.get(changes, 0.0)
                    if v is not None:
                        sc += 0.8
                        rec_h = self._mash_lists(tids[h]) if tids[h] is not None else {}
                        if v != h and any(int(x["id"]) == tids[v] for x in (rec_h.get("vocal_over") or [])):
                            sc += 1.0                 # the lists say this vocal sits over this harmony
                        if v == h:
                            sc += 0.2                 # a song's own vocal over its own harmony: safe
                    if d != h:
                        rec_d = self._mash_lists(tids[d]) if tids[d] is not None else {}
                        if any(int(x["id"]) == tids[h] for x in (rec_d.get("bass_under") or [])):
                            sc += 0.8                 # the lists say this bassline sits under these drums
                    if d == h == v:
                        sc -= 0.6                     # one song whole is the record mode, not a mashup
                    if cand in recent:
                        sc -= 1.5                     # (a recent scene is not repeated)
                    sc -= 0.25 * sum(m["used"][i] for i in set(x for x in cand if x is not None)) / max(1.0, float(n))
                    sc += 0.3 * self.rng.random()
                    if sc > best_sc:
                        best, best_sc = cand, sc
        if best is None:
            return dict(cur or {"drums": avail[0], "harm": avail[0], "vocal": None})
        d, h, v = best
        shape = None
        if n % SHAPE_EVERY == 0 and cur is not None and (d != cur.get("drums") or h != cur.get("harm")):
            shape = "drop"
        return {"drums": d, "harm": h, "vocal": v, "shape": shape}

    def _mash_say_next(self):
        m = self._mash
        if m.get("next") is None:
            return
        self._move("mashup_plan", None, None, None,
                   f"next scene in {max(0, int(m['next_bar']) - self.bar_n)} bars: {scene_words(m['next'], self._mash_titles())}")

    # -- making it sound -----------------------------------------------------------------------------------------
    def _mash_target_t(self, deck, lanes, at_bar):
        """Where the song on `deck` should be on the scene's bar for `lanes` to sound: its own position when they sound
        there, else the next place they carry (a groove for drums / harmony, a sung stretch for the vocal), wrapping to
        its first body when it runs out."""
        s = self.songs[deck]
        t = s.track
        at = self._next_bar_clock() + max(0, at_bar - self.bar_n - 1) * self._bar_clock()
        t_now = self._song_time_at(deck, at)
        bar = self._bar_s(t, t_now)
        span = max(8, int(self.mash_bars)) * bar
        dur = float(t.duration_s or 0.0)
        if "vocals" in lanes and len(lanes) == 1:
            env = (s.env or {}).get("vocals")
            if env is not None and len(env) > 8:
                import numpy as np
                e = np.asarray(env, dtype=np.float32)

                def sings_at(x):
                    i0, i1 = int(x * 2), int((x + 4 * bar) * 2)
                    seg = e[i0:i1]
                    return len(seg) > 4 and float(np.mean(seg >= 0.12)) >= 0.5
                x = t_now
                while x < dur - span:
                    if sings_at(x):
                        return t.nearest_downbeat(x), x != t_now
                    x += 4 * bar
                x = 0.0
                while x < t_now:
                    if sings_at(x):
                        return t.nearest_downbeat(x), True
                    x += 4 * bar
            return t_now, False
        need = ("drums",) if "drums" in lanes else ("bass",)
        if dur - t_now > span + RUNOUT_S and self._stem_carries(s.env, t_now, min(span, 4 * bar), lanes=need, bar_s=bar) is not False:
            return t_now, False
        c = self._steady_from(t, s.env, t_now, 4, strict=False, lanes=need)
        if c is None or dur - c < span + RUNOUT_S:
            c = self._steady_from(t, s.env, self._body_start(t, s.env), 4, strict=False, lanes=need)
        if c is None:
            return t_now, False
        return t.nearest_downbeat(c), abs(c - t_now) > 0.5 * bar

    def _mash_cue(self, scene, nb):
        """A bar ahead of the scene: every slot that will hold a part it does not hold now - and is silent now - is
        cued so its part sounds from the scene's bar. A slot that keeps a part but is running out is hot-cued on the
        scene's bar itself (in _mash_apply)."""
        m = self._mash
        want = self._mash_lanes_of(scene)
        for i in range(SLOTS):
            d = self._mash_deck(i)
            if d is None:
                continue
            lanes_new = [ln for ln, slot in want.items() if slot == i]
            if not lanes_new:
                continue
            held = self.songs[d].held(self.lanes)
            if held:
                continue
            tgt, move = self._mash_target_t(d, lanes_new, nb)
            if not move:
                continue
            at_land = self._next_bar_clock() + max(0, nb - self.bar_n - 1) * self._bar_clock()
            bar_b = self._bar_s(self.songs[d].track, tgt)
            at_cue = at_land - self._bar_clock()
            if at_cue <= self._clock() + int(0.25 * RATE):
                at_cue, tgt_c = at_land, tgt
            else:
                tgt_c = tgt - bar_b
            self._post_cue(d, max(0.0, tgt_c), at_cue)

    def _mash_lanes_of(self, scene):
        return {"drums": scene.get("drums"), "bass": scene.get("harm"), "other": scene.get("harm"), "vocals": scene.get("vocal")}

    def _mash_apply(self, scene, nb):
        """The scene's bar: every lane goes to its slot's song (quick: a beat; a DROP lands on the one after two beats of
        nothing but the vocal); a song keeping a part that is running out is hot-cued back to a groove on this bar."""
        m = self._mash
        at = self._next_bar_clock() + max(0, nb - self.bar_n - 1) * self._bar_clock()
        beat_c = int(self._beat_s() * RATE)
        want = self._mash_lanes_of(scene)
        titles = self._mash_titles()
        if scene.get("shape") == "drop":
            hole_at = at - 2 * beat_c
            if hole_at > self._clock() + int(0.2 * RATE):
                ev = []
                for ln, d in self.lanes.items():
                    if d is None or d not in self.songs or ln == "vocals":
                        continue
                    ev.append({"at": hole_at, "cmd": "stem_gains", "deck": d, "gains": {ln: 0.0}, "ramp_s": 0.05})
                if ev:
                    self.submix.post_many(ev)
        # running out: hot cue on the scene's bar (a held song jumps back to a groove - a DJ's cue point)
        for i in range(SLOTS):
            d = self._mash_deck(i)
            if d is None or not self.songs[d].held(self.lanes):
                continue
            s = self.songs[d]
            t_at = self._song_time_at(d, at)
            if float(s.track.duration_s) - t_at < max(8, int(self.mash_bars)) * self._bar_s(s.track, t_at) + RUNOUT_S:
                need = ("drums",) if want.get("drums") == i else ("bass",)
                c = self._steady_from(s.track, s.env, self._body_start(s.track, s.env), 4, strict=False, lanes=need)
                if c is not None:
                    self._post_cue(d, c, at)
                    self._move("mashup", None, s.track.id, None, f"{s.track.title[:24]} is running out: hot cue back to {int(c) // 60}:{int(c) % 60:02d} on the scene's bar")
        beats = 0.5 if scene.get("shape") == "drop" else 1.0
        changed = []
        for ln in ("drums", "bass", "other", "vocals"):
            slot = want[ln]
            d = self._mash_deck(slot) if slot is not None else None
            if self.lanes.get(ln) != d:
                self._cross(ln, d, at, beats=beats, shape=False)
                changed.append(ln)
            if d is not None:
                s = self.songs[d]
                if not s.entered:
                    s.entered, s.entered_k = True, self.bar_n
        self._hygiene(at + int(beats * beat_c))
        self._move("mashup", None, None, None, f"SCENE {int(m.get('n', 0)) + 1}: {scene_words(scene, titles)}" + (f" - {', '.join(changed)} move" if changed else ""))

    # -- the set moves on ----------------------------------------------------------------------------------------
    def _mash_rotate_step(self):
        """After ROTATE_SCENES scenes: the slot used least in the recent scenes (and not in the coming one) leaves at the
        next scene and a new song that fits the two that stay is staged into it."""
        m = self._mash
        m["since_rot"] = int(m.get("since_rot", 0)) + 1
        if m.get("leaving") is not None:
            i = m["leaving"]
            in_scene = i in (m["scene"].get("drums"), m["scene"].get("harm"), m["scene"].get("vocal"))
            if not in_scene:
                d = self._mash_deck(i)
                old = m["tids"][i]
                if d is not None:
                    self._leave_song(d, self._next_bar_clock())
                m["tids"][i], m["leaving"], m["since_rot"] = None, None, 0
                m["used"][i] = 0
                m["rotations"] = int(m.get("rotations", 0)) + 1
                self._move("mashup", None, old, None, f"MASHUP: {self._track_title(old)[:24]} leaves - a new song comes into slot {i + 1}")
                self._mash_fill_slots()
            return
        if m["since_rot"] < ROTATE_SCENES or len(self._mash_ready()) < SLOTS:
            return
        recent = (m.get("history") or [])[-4:]
        use = {i: sum(1 for s in recent for x in (s.get("drums"), s.get("harm"), s.get("vocal")) if x == i) for i in range(SLOTS)}
        i = min(range(SLOTS), key=lambda k: (use[k], -m["used"][k]))
        m["leaving"] = i
        self._move("mashup_plan", None, m["tids"][i], None, f"MASHUP: {self._track_title(m['tids'][i])[:24]} leaves after this scene (used least)")

    # -- your hands ----------------------------------------------------------------------------------------------
    def mash_pad(self, slot, lane):
        """A pad: `lane` (drums / bass / other / vocals) to the song in `slot` on the next bar; harmony travels together
        (bass or melody moves both); a lit pad takes the part out. The routine carries on from the new scene.
        Returns (ok, words)."""
        m = getattr(self, "_mash", None)
        if m is None:
            return False, "not in MASHUP (LAYERS: mashup)"
        if slot not in range(SLOTS) or lane not in ("drums", "bass", "other", "vocals"):
            return False, "no such pad"
        d = self._mash_deck(slot)
        if d is None:
            return False, f"slot {slot + 1} has no song ready yet"
        key = {"drums": "drums", "bass": "harm", "other": "harm", "vocals": "vocal"}[lane]
        scene = dict(m["scene"])
        scene["shape"] = None
        scene[key] = None if scene.get(key) == slot else slot
        if key == "vocal" and scene["vocal"] is not None and not self._mash_sings(slot):
            pass                                      # (your order: said in the move, not refused)
        nb = self.bar_n + 1
        self._mash_cue(scene, nb + 1)
        self._mash_apply(scene, nb + 1)
        m["scene"] = scene
        m["n"] = int(m.get("n", 0)) + 1
        m["history"] = (m.get("history") or [])[-(HISTORY - 1):] + [scene]
        m["next"], m["cued"] = None, False
        m["next_bar"] = max(int(m["next_bar"]), nb + 1 + max(4, int(self.mash_bars) // 2))   # (your move is heard before the routine moves again)
        title = self.songs[d].track.title[:24]
        what = {"drums": "drums", "harm": "harmony", "vocal": "vocal"}[key]
        return True, (f"the {what} of {title} out" if scene[key] is None else f"the {what} of {title} in on the next bar")

    def mash_status(self):
        """For the MASHUP tab and THE PLAN: the slots (title, deck, the parts each holds), the scene now and next."""
        m = getattr(self, "_mash", None)
        if m is None:
            return None
        titles = self._mash_titles()
        slots = []
        for i in range(SLOTS):
            d = self._mash_deck(i)
            held = self.songs[d].held(self.lanes) if d is not None else []
            slots.append({"slot": i, "tid": m["tids"][i], "title": titles[i], "deck": d, "ready": d is not None, "held": list(held),
                          "leaving": m.get("leaving") == i, "sings": self._mash_sings(i) if d is not None else False})
        return {"slots": slots, "scene": scene_words(m.get("scene"), titles), "next": scene_words(m.get("next"), titles) if m.get("next") else None,
                "next_in_bars": max(0, int(m["next_bar"]) - self.bar_n), "n": int(m.get("n", 0)), "bars": int(self.mash_bars),
                "rotate_in": (None if m.get("leaving") is not None else max(0, ROTATE_SCENES - int(m.get("since_rot", 0)))),
                "rotations": int(m.get("rotations", 0))}
