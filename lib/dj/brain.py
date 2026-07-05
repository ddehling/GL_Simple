"""The DJ brain: what to play next, and exactly how to get there.

Selection couples SONG choice to MIX quality: a candidate scores on tempo
fit (small stretch ratios preferred, half/double time considered), Camelot
key compatibility, energy fit to the theme's arc target, mood/spectral
match, recency penalties - AND the best available section-pair for the
transition. Two busy/vocal sections never blend over each other; if two
tracks have no quiet seam between them, that pairing loses points.

plan_transition() returns a fully-resolved plan (style, A's exit point, B's
entry point, blend length); build_events() compiles it into sample-stamped
submix automation using a telemetry snapshot for the clock mapping.
"""
import math
import random
import time

RATE = 44100

STYLES = ("long_blend", "bass_swap", "cut_at_drop", "loop_roll_exit",
          "bassline_layer", "double_drop", "loop_build", "long_fade")
STRETCH_MIN, STRETCH_MAX = 0.92, 1.08
GLIDE_PER_S = 0.0015             # post-transition rate->1.0 glide speed


# --------------------------------------------------------------------------
# Track wrapper
# --------------------------------------------------------------------------

class TrackInfo:
    """One library track hydrated with sections/loops/mix points."""

    def __init__(self, row, sections, loops, mix_points, cues=None,
                 user_tags=None):
        self.row = row
        self.id = row["id"]
        self.path = row["path"]
        self.title = row.get("title") or row["path"]
        self.artist = row.get("artist") or ""
        self.duration_s = row["duration_s"] or 0.0
        self.bpm = row["bpm"] or 0.0
        self.bpm_conf = row["bpm_conf"] or 0.0
        self.downbeat_offset = row["downbeat_offset"] or 0
        self.downbeat_conf = row["downbeat_conf"] or 0.0
        self.camelot = row["camelot"] or ""
        self.grid = row.get("beat_grid") or []
        self.gain_db = row.get("loudness_gain_db") or 0.0
        self.kick_offset_s = float(row.get("kick_offset_s") or 0.0)
        self.mood_hist = row.get("mood_hist") or {}
        self.rhythm_density = row.get("rhythm_density") or 0.0
        self.spectral = row.get("spectral") or {}
        self.sections = sections
        self.loops = loops
        self.axes = row.get("axes") or {}
        self.auto_tags = row.get("auto_tags") or []
        self.user_tags = list(user_tags or [])
        self.cues = list(cues or [])
        self.mix_ins = [p for p in mix_points if p["kind"] == "in"]
        self.mix_outs = [p for p in mix_points if p["kind"] == "out"]
        # USER-authored in/out cues override the analyzer's guesses: if any
        # exist for a direction, they become the only candidates (score 1.0,
        # so pair selection favors what the human marked).
        user_ins = [c for c in self.cues
                    if c["kind"] == "in" and c["source"] == "user"]
        user_outs = [c for c in self.cues
                     if c["kind"] == "out" and c["source"] == "user"]
        if user_ins:
            self.mix_ins = [{"kind": "in", "time_s": c["time_s"],
                             "score": 1.0, "style_hint": c.get("label")
                             or "blend"} for c in user_ins]
        if user_outs:
            self.mix_outs = [{"kind": "out", "time_s": c["time_s"],
                              "score": 1.0, "style_hint": c.get("label")
                              or "blend"} for c in user_outs]

    @property
    def all_tags(self):
        return sorted(set(self.auto_tags) | set(self.user_tags))

    @property
    def period_s(self):
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def section_at(self, t):
        for s in self.sections:
            if s["start_s"] <= t < s["end_s"]:
                return s
        return self.sections[-1] if self.sections else None

    def nearest_downbeat(self, t):
        """Downbeat time closest to t (from the main grid segment)."""
        g = None
        for seg in self.grid:
            if seg["start_s"] <= t <= seg["end_s"]:
                g = seg
                break
        if g is None and self.grid:
            g = self.grid[0]
        if g is None:
            return t
        bar = 4 * g["period_s"]
        first_down = g["first_beat_s"] + self.downbeat_offset * g["period_s"]
        k = round((t - first_down) / bar)
        return first_down + k * bar

    def energy_proxy(self):
        """Cross-track comparable energy in ~0..1."""
        mh = self.mood_hist
        mood_e = mh.get("peak", 0.0) + 0.6 * mh.get("groove", 0.0) \
            + 0.25 * mh.get("chill", 0.0)
        dens = min(self.rhythm_density / 3.0, 1.0)
        bass = self.spectral.get("bass_share", 0.33)
        return max(0.0, min(1.0, 0.45 * mood_e + 0.35 * dens + 0.2 * bass * 2.0))


def load_library(db):
    """Hydrate every playable track from the DB into TrackInfo objects."""
    out = []
    for row in db.all_tracks():
        if not row.get("bpm") or row["bpm"] <= 0 or not row.get("duration_s"):
            continue
        if row["duration_s"] < 90.0:
            continue
        out.append(TrackInfo(row, db.sections_for(row["id"]),
                             db.loops_for(row["id"]),
                             db.mix_points_for(row["id"]),
                             cues=db.cues_for(row["id"]),
                             user_tags=db.tags_for(row["id"])))
    return out


# --------------------------------------------------------------------------
# Key compatibility (Camelot wheel)
# --------------------------------------------------------------------------

def camelot_compat(c1, c2):
    if not c1 or not c2:
        return 0.6                      # unknown keys: mildly neutral
    try:
        n1, m1 = int(c1[:-1]), c1[-1]
        n2, m2 = int(c2[:-1]), c2[-1]
    except ValueError:
        return 0.6
    dn = min((n1 - n2) % 12, (n2 - n1) % 12)
    if dn == 0:
        return 1.0 if m1 == m2 else 0.92    # same / relative major-minor
    if dn == 1 and m1 == m2:
        return 0.9                          # neighbour on the wheel
    if dn == 2 and m1 == m2:
        return 0.55
    return 0.3


# --------------------------------------------------------------------------
# Brain
# --------------------------------------------------------------------------

class Brain:
    def __init__(self, library, theme, seed=None, stretch_max=STRETCH_MAX):
        self.library = list(library)
        self.theme = theme
        self.rng = random.Random(seed)
        self.stretch_max = min(stretch_max, STRETCH_MAX)
        self.stretch_min = max(2.0 - self.stretch_max, STRETCH_MIN)
        self.recent = []                # (wall_time, track_id, artist)

    # -- memory --------------------------------------------------------------
    def note_played(self, track, when=None):
        self.recent.append((when or time.time(), track.id, track.artist))
        cutoff = (when or time.time()) - 10 * 3600
        self.recent = [r for r in self.recent if r[0] > cutoff]

    def _recency_penalty(self, track, now=None):
        now = now or time.time()
        pen = 1.0
        for when, tid, artist in self.recent:
            age_h = (now - when) / 3600.0
            if tid == track.id:
                pen *= min(1.0, 0.05 + age_h / 6.0)     # ~6h to forgive a track
            elif artist and artist == track.artist:
                pen *= min(1.0, 0.55 + age_h / 2.0)     # ~1h for an artist
        return max(pen, 0.01)

    # -- tempo ---------------------------------------------------------------
    def rate_for(self, out_bpm, cand):
        """Best stretch rate to bring cand to out_bpm, allowing half/double
        time reads. Returns (rate, effective_bpm) or (None, 0)."""
        best = None
        for mult in (1.0, 2.0, 0.5):
            eff = cand.bpm * mult
            if eff <= 0:
                continue
            r = out_bpm / eff
            if self.stretch_min <= r <= self.stretch_max:
                if best is None or abs(math.log(r)) < abs(math.log(best[0])):
                    best = (r, eff)
        return best if best else (None, 0.0)

    # -- selection -----------------------------------------------------------
    def score(self, current, cand, arc_target, out_bpm, now=None):
        if cand.id == getattr(current, "id", None):
            return 0.0, None
        rate, eff_bpm = self.rate_for(out_bpm, cand)
        if rate is None:
            return 0.0, None
        lo, hi = self.theme.bpm_range
        if not (lo * 0.93 <= eff_bpm <= hi * 1.07):
            return 0.0, None
        s_rate = math.exp(-((abs(math.log(rate))) / 0.045) ** 2)
        s_key = camelot_compat(getattr(current, "camelot", ""), cand.camelot)
        s_energy = math.exp(-((cand.energy_proxy() - arc_target) / 0.3) ** 2)
        s_mood = 0.25 + sum(self.theme.mood_weights.get(m, 0.0) * f
                            for m, f in cand.mood_hist.items())
        s_spec = 1.0
        if self.theme.spectral_lean == "bass":
            s_spec = 0.7 + 0.6 * cand.spectral.get("bass_share", 0.33) * 2.0
        elif self.theme.spectral_lean == "high":
            s_spec = 0.7 + 0.6 * cand.spectral.get("high_share", 0.2) * 3.0
        pair = self.best_pair(current, cand) if current is not None else None
        s_pair = pair["score"] if pair else (0.5 if current is None else 0.15)
        total = (s_rate * s_key * s_energy * s_mood * s_spec
                 * self._recency_penalty(cand, now) * s_pair
                 * self.rng.uniform(0.9, 1.1))
        return total, {"rate": rate, "eff_bpm": eff_bpm, "pair": pair}

    def choose_next(self, current, arc_target, out_bpm, now=None):
        """Returns (TrackInfo, meta) or (None, None) when the library is dry."""
        best, best_score, best_meta = None, 0.0, None
        for cand in self.library:
            s, meta = self.score(current, cand, arc_target, out_bpm, now)
            if s > best_score:
                best, best_score, best_meta = cand, s, meta
        return best, best_meta

    def choose_first(self, arc_target, now=None):
        best, best_score = None, -1.0
        for cand in self.library:
            lo, hi = self.theme.bpm_range
            if not (lo * 0.93 <= cand.bpm <= hi * 1.07):
                continue
            s = math.exp(-((cand.energy_proxy() - arc_target) / 0.3) ** 2) \
                * (0.25 + sum(self.theme.mood_weights.get(m, 0.0) * f
                              for m, f in cand.mood_hist.items())) \
                * self._recency_penalty(cand, now) * self.rng.uniform(0.9, 1.1)
            if s > best_score:
                best, best_score = cand, s
        return best

    # -- section-pair mixability (the anti-garbage rule) -------------------------
    def best_pair(self, cur, cand, after_s=None):
        """Best (A-exit, B-entry) combination, or None. Never lets two
        busy/vocal sections blend over each other."""
        outs = [o for o in cur.mix_outs
                if after_s is None or o["time_s"] >= after_s]
        if not outs:
            return None
        # How good a section is to mix OUT of / IN to. The golden rule of
        # melodic-house mixing: bring the new track's INTRO (drums, no lead)
        # in over the old track's OUTRO/breakdown, so two lead melodies never
        # play at once. Kind drives this; busyness/vocalness refine it.
        def out_fit(sec):
            k = sec.get("kind", "groove")
            base = {"outro": 1.0, "breakdown": 0.85, "groove": 0.6,
                    "build": 0.3, "intro": 0.4}.get(k, 0.5)
            return base * (1.0 - 0.5 * (sec.get("vocalness") or 0.0))

        def in_fit(sec):
            k = sec.get("kind", "groove")
            base = {"intro": 1.0, "breakdown": 0.8, "groove": 0.55,
                    "build": 0.6, "outro": 0.2}.get(k, 0.5)
            return base * (1.0 - 0.6 * (sec.get("vocalness") or 0.0))

        best = None
        for o in outs[:8]:
            sec_a = cur.section_at(min(o["time_s"] + 1.0,
                                       cur.duration_s - 1.0))
            for i in cand.mix_ins[:8]:
                sec_b = cand.section_at(min(i["time_s"] + 1.0,
                                            cand.duration_s - 1.0))
                if sec_a is None or sec_b is None:
                    continue
                busy_a = sec_a.get("busyness") or 0.0
                busy_b = sec_b.get("busyness") or 0.0
                voc_a = sec_a.get("vocalness") or 0.0
                voc_b = sec_b.get("vocalness") or 0.0
                fit = out_fit(sec_a) * in_fit(sec_b)     # intro-over-outro
                # Prefer mixing the incoming in EARLIER (nearer its groove
                # start) over a deep point, but only a gentle lean - the
                # mix-in must still land where the track has energy, or the
                # blend goes quiet as the outgoing leaves.
                early_b = math.exp(-max(i["time_s"] - 20.0, 0.0) / 120.0)
                quiet = 1.0 - 0.5 * min(busy_a + busy_b, 1.6) / 1.6
                # BLEND WHERE THE BEATS ARE: a beat-matched blend is only
                # audible as beat-matched if BOTH sides carry rhythm and
                # comparable energy - otherwise it just reads as a fade.
                ra = sec_a.get("rhythm_density") or 0.0
                rb = sec_b.get("rhythm_density") or 0.0
                ea = sec_a.get("energy") or 0.0
                eb = sec_b.get("energy") or 0.0
                beaty = ra >= 1.2 and rb >= 1.2
                rhythm_fit = (1.3 if beaty else 0.55) \
                    * math.exp(-((ea - eb) ** 2) / (2 * 0.4 ** 2))
                # Two lead-carrying sections over each other = clash: heavy
                # penalty (not a hard reject, so there's always a best pair).
                clash = 1.0
                if busy_a > 0.6 and busy_b > 0.6:
                    clash *= 0.3
                if voc_a > 0.5 and voc_b > 0.5:
                    clash *= 0.25
                mp = 0.5 + 0.5 * max(o["score"], 0.0) * max(i["score"], 0.0)
                # Weighted-sum form so a mediocre pair stays ~0.05-1, never
                # collapsing to ~0 (which would zero the whole selection).
                score = ((0.25 + 0.75 * fit) * (0.6 + 0.4 * quiet)
                         * (0.4 + 0.6 * early_b) * rhythm_fit * clash * mp)
                if best is None or score > best["score"]:
                    best = {"out_s": o["time_s"], "in_s": i["time_s"],
                            "out_hint": o.get("style_hint", "blend"),
                            "in_hint": i.get("style_hint", "blend"),
                            "score": round(score, 5), "beaty": beaty,
                            "kinds": (sec_a.get("kind"), sec_b.get("kind")),
                            "busy": (round(busy_a, 2), round(busy_b, 2))}
        return best

    def _drop_after(self, track, after_s):
        """First DROP MOMENT (energy slams up at a boundary) at/after
        after_s, else the earliest one, or None."""
        from lib.dj.features import drop_moments
        drops = drop_moments(track.sections)
        if not drops:
            return None
        ahead = [t for t in drops if t >= after_s]
        return min(ahead) if ahead else min(drops)

    def _bass_loop(self, track, before_s, after_s=0.0):
        """Best loop in `track` to isolate as a repeating groove bed: a
        bass-heavy, low-vocal, repetitive loop before the exit. Only loops
        AHEAD of `after_s` qualify - a loop behind the live playhead would
        compile to already-past events (the whole transition fires at once
        and the loop window never wraps)."""
        best = None
        for l in track.loops:
            if l["start_s"] >= before_s or l["start_s"] < after_s:
                continue
            sec = track.section_at(l["start_s"] + 1.0)
            if sec is None:
                continue
            if sec.get("vocalness", 1.0) > 0.5:
                continue
            score = (l.get("score", 0.0) * (0.4 + sec.get("bass_share", 0.3))
                     * (0.5 + 0.5 * sec.get("repetitiveness", 0.0)))
            if best is None or score > best[0]:
                best = (score, l)
        return best[1] if best else None

    # -- transition planning -----------------------------------------------------
    def plan_transition(self, cur, cand, meta, after_s=None):
        """Resolve style + timing. Returns a plan dict (see build_events)."""
        pair = meta.get("pair") if meta else None
        if pair is None or (after_s is not None
                            and pair["out_s"] < after_s):
            pair = self.best_pair(cur, cand, after_s=after_s)
        if pair is None:
            # Last resort: exit on the last downbeat-aligned half minute.
            pair = {"out_s": max(cur.duration_s - 35.0, cur.duration_s * 0.6),
                    "in_s": cand.mix_ins[0]["time_s"] if cand.mix_ins else 0.0,
                    "out_hint": "blend", "in_hint": "blend", "score": 0.1}
        rate = meta["rate"] if meta else 1.0

        # Style menu, gated by analysis confidence.
        weights = dict(self.theme.style_weights)
        low_conf = (cur.bpm_conf < 0.5 or cand.bpm_conf < 0.5)
        if low_conf or not pair.get("beaty", True):
            # No confident grid, or the best seam is BEATLESS on one side:
            # a beat-matched blend there is inaudible as such and just
            # smears - do a deliberate clean fade on the phrase instead.
            style = "long_fade"
        else:
            if (cur.downbeat_conf < 0.15 or cand.downbeat_conf < 0.15):
                weights["cut_at_drop"] = 0.0
            # cut_at_drop needs a pre-drop entry in B - ANY of B's pre_drop
            # mix-ins qualifies, not just the best-scoring pair's (gating on
            # pair["in_hint"] starved the style to literally zero uses
            # across a 125-track library: pre_drop points rarely win the
            # generic pair scoring).
            pre_drops = [p for p in cand.mix_ins
                         if p.get("style_hint") == "pre_drop"]
            if not pre_drops:
                weights["cut_at_drop"] = 0.0
            # (loop_roll_exit rolls the 16 beats just before out_s - its
            # window is derived, so no after_s restriction needed here.)
            loop_ok = any(l["start_s"] < pair["out_s"] for l in cur.loops)
            if not loop_ok:
                weights["loop_roll_exit"] = 0.0
            # bassline_layer needs a bass-heavy, low-vocal loop in A that is
            # still AHEAD of the playhead when the transition arms.
            if self._bass_loop(cur, pair["out_s"],
                               after_s=after_s or 0.0) is None:
                weights["bassline_layer"] = 0.0
            # double_drop aligns A's drop onset with B's drop onset (the
            # drop boundary IS a downbeat, so this works even where the
            # global downbeat_offset is uncertain); the sync snap handles
            # beat phase. Needs a drop in each track.
            a_drop = self._drop_after(cur, pair["out_s"] - 8 * cur.period_s)
            b_drop = self._drop_after(cand, cand.duration_s * 0.15)
            if a_drop is None or b_drop is None:
                weights["double_drop"] = 0.0
            # loop_build stutters A into its own drop as a tension build,
            # then B cuts in on the drop. Needs a drop in A to build toward.
            if self._drop_after(cur, pair["out_s"] - 8 * cur.period_s) is None:
                weights["loop_build"] = 0.0
            weights["long_fade"] = 0.0
            menu = [(s, w) for s, w in weights.items() if w > 0]
            if not menu:
                menu = [("bass_swap", 1.0)]
            styles, ws = zip(*menu)
            style = self.rng.choices(styles, weights=ws, k=1)[0]

        beats = {"long_blend": 32, "bass_swap": 16, "cut_at_drop": 16,
                 "loop_roll_exit": 32, "bassline_layer": 16,
                 "double_drop": 16, "loop_build": 16, "long_fade": 0}[style]
        if style == "loop_build":
            # Exit ON A's drop; the stutter build fills the bars before it.
            a_drop = self._drop_after(cur, pair["out_s"] - 8 * cur.period_s)
            out_s = cur.nearest_downbeat(a_drop)
            in_s = cand.nearest_downbeat(pair["in_s"])
            return {"style": style, "rate": rate, "out_s": out_s,
                    "in_s": in_s, "beats": beats,
                    "pair_score": pair["score"], "cand_id": cand.id}
        if style == "double_drop":
            # A exits on ITS drop; B is cued so its drop lands on the same
            # beat. out_s/in_s become the two drop onsets.
            a_drop = self._drop_after(cur, pair["out_s"] - 8 * cur.period_s)
            b_drop = self._drop_after(cand, cand.duration_s * 0.15)
            out_s = cur.nearest_downbeat(a_drop)
            in_s = cand.nearest_downbeat(b_drop)
            plan = {"style": style, "rate": rate, "out_s": out_s,
                    "in_s": in_s, "beats": beats,
                    "pair_score": pair["score"], "cand_id": cand.id}
            return plan
        out_s = cur.nearest_downbeat(pair["out_s"])
        in_s = cand.nearest_downbeat(pair["in_s"])
        if style == "cut_at_drop":
            # Enter at B's best PRE-DROP point (the style's whole premise),
            # not the generic pair in-point.
            pd = max((p for p in cand.mix_ins
                      if p.get("style_hint") == "pre_drop"),
                     key=lambda p: p.get("score", 0.0), default=None)
            if pd is not None:
                in_s = cand.nearest_downbeat(pd["time_s"])
        plan = {"style": style, "rate": rate,
                "out_s": out_s, "in_s": in_s, "beats": beats,
                "pair_score": pair["score"], "cand_id": cand.id}
        if style == "loop_roll_exit":
            # Loop the 16 bars-worth just before the exit point: with the
            # window pinned to out_s the first wrap and both shrink moments
            # all land exactly on the grid (elapsed beats stay multiples of
            # the shrinking span).
            plan["loop_start_s"] = max(0.0, out_s - 16 * cur.period_s)
        if style == "bassline_layer":
            loop = self._bass_loop(cur, pair["out_s"],
                                   after_s=after_s or 0.0)
            beats_len = loop["beats"] if loop["beats"] in (8, 16) else 8
            plan["loop_start_s"] = cur.nearest_downbeat(loop["start_s"])
            plan["loop_beats"] = beats_len
            plan["layer_beats"] = 16          # bars both tracks play together
        return plan

    # -- automation compilation ------------------------------------------------
    def build_events(self, plan, snapshot, active, incoming, cur, cand):
        """Compile a plan into submix events. `snapshot` is submix telemetry;
        `active`/`incoming` are deck names; `cur`/`cand` TrackInfos.

        Returns (events, swap_at_clock, blend_start_clock)."""
        tel = snapshot["decks"][active]
        clock = snapshot["clock"]
        rate_a = max(tel["rate"], 1e-6)

        def clock_at(src_time_s):
            return clock + int((src_time_s - tel["time_s"]) / rate_a * RATE)

        # Nothing may schedule in the past: past events all fire in one
        # flush (run-ins vanish, sync snaps at full gain, loop windows can
        # land entirely behind the cursor and never wrap).
        now_guard = clock + int(0.3 * RATE)

        beat_out = cur.period_s / rate_a          # output-domain beat of A
        style = plan["style"]
        rate_b = plan["rate"]
        ev = []

        if style == "long_fade":
            S0 = clock_at(plan["out_s"])
            dur = 12.0                       # deliberate clean fade, phrase-tight
            ev += [
                {"at": S0, "cmd": "cue", "deck": incoming,
                 "time_s": plan["in_s"]},
                {"at": S0, "cmd": "rate", "deck": incoming, "value": 1.0},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.01},
                {"at": S0, "cmd": "start", "deck": incoming},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": dur},
                {"at": S0, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": dur},
                {"at": S0 + int((dur + 1) * RATE), "cmd": "stop",
                 "deck": active},
            ]
            return ev, S0 + int((dur + 1) * RATE), S0

        nb = plan["beats"]
        if style == "cut_at_drop":
            # The cut lands on B's drop downbeat; B rides in underneath first.
            S_cut = clock_at(plan["out_s"])
            # 16 B-beats of run-up, measured in OUTPUT time (period/rate) so
            # the launch lands 16 matched beats before the cut, not 16 source
            # beats (up to 8% off - a beat-and-a-third the PLL can't absorb).
            lead = int(16 * cand.period_s / rate_b * RATE)
            S0 = max(S_cut - lead, now_guard)
            # B must still ARRIVE at in_s exactly at the cut, however much
            # run-in survives the clamp.
            cue_b = max(0.0, plan["in_s"]
                        - (S_cut - S0) / RATE * rate_b)
            ev += [
                {"at": S0, "cmd": "cue", "deck": incoming, "time_s": cue_b},
                {"at": S0, "cmd": "rate", "deck": incoming, "value": rate_b},
                {"at": S0, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "ramp_s": 0.01},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.01},
                {"at": S0, "cmd": "start", "deck": incoming},
                {"at": S0, "cmd": "sync", "slave": incoming, "master": active},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.8,
                 "ramp_s": 12 * cand.period_s},
                {"at": S_cut, "cmd": "end_sync"},
                {"at": S_cut, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 0.04},
                {"at": S_cut, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "ramp_s": 0.04},
                {"at": S_cut, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 0.04},
                {"at": S_cut + int(0.5 * RATE), "cmd": "stop", "deck": active},
            ]
            swap_at = S_cut + int(0.5 * RATE)
            self._glide_home(ev, incoming, rate_b, swap_at)
            return ev, swap_at, S0

        if style == "loop_build":
            # Tension build: A stutters a loop that shrinks 8->4->2->1 beats
            # (all ending on its drop) accelerating into it, releases ON the
            # drop, and B cuts in - the loop-build-into-drop move. A's loop
            # end is pinned to the drop so release lands exactly on it.
            drop_s = plan["out_s"]
            per = cur.period_s
            # (beats_len, output beats to hold that stage)
            stages = [(8, 8), (4, 4), (2, 2), (1, 2)]
            S0 = max(clock_at(drop_s - stages[0][0] * per), now_guard)
            t = S0
            for length, hold in stages:
                ls = drop_s - length * per
                # (a late-fired loop is safe: the window END is the drop,
                # still ahead of the cursor, so the wrap engages normally)
                ev.append({"at": max(t, now_guard), "cmd": "loop",
                           "deck": active, "start_s": ls, "end_s": drop_s})
                # Filter up as it builds (rising tension), trim lows late.
                ev.append({"at": t, "cmd": "eq", "deck": active,
                           "high": 1.0, "mid": 1.0,
                           "low": 1.0 if length > 2 else 0.6, "ramp_s": 0.1})
                t += int(hold * beat_out * RATE)
            S_drop = t                                   # release = the drop
            cue_b = max(0.0, plan["in_s"] - 8 * cand.period_s)
            out = S_drop + int(8 * beat_out * RATE)
            ev += [
                # Pre-run B under the tail of the build, bass-cut + synced.
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "cue",
                 "deck": incoming, "time_s": cue_b},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "rate",
                 "deck": incoming, "value": rate_b},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "eq",
                 "deck": incoming, "low": 0.0, "mid": 0.5, "high": 0.6,
                 "ramp_s": 0.05},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "gain",
                 "deck": incoming, "value": 0.0, "ramp_s": 0.05},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "start",
                 "deck": incoming},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "sync",
                 "slave": incoming, "master": active},
                # THE DROP: release A's loop into it, B slams in full, A ducks.
                {"at": S_drop, "cmd": "release_loop", "deck": active},
                {"at": S_drop, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 0.06},
                {"at": S_drop, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 0.06},
                {"at": S_drop, "cmd": "eq", "deck": active, "low": 0.0,
                 "ramp_s": 0.06},
                {"at": S_drop, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 4 * beat_out},
                {"at": out, "cmd": "stop", "deck": active},
                {"at": out, "cmd": "clear_loop", "deck": active},
                {"at": out, "cmd": "end_sync"},
            ]
            self._glide_home(ev, incoming, rate_b, out)
            return ev, out, S0

        if style == "double_drop":
            # Align B's drop onset to A's drop onset: B runs in for 16 beats
            # bass-cut (build tension under A), both drops HIT together for 4
            # bars of full-range double-drop, then A exits and B rides on.
            run_in = 16
            S_drop = clock_at(plan["out_s"])            # A's drop moment
            b_period = cand.period_s
            S0 = max(S_drop - int(run_in * b_period / rate_b * RATE),
                     now_guard)
            # B's drop must land ON S_drop whatever run-in survives.
            cue_b = max(0.0, plan["in_s"]
                        - (S_drop - S0) / RATE * rate_b)
            both = S_drop + int(16 * beat_out * RATE)   # 4 bars of both
            out = both + int(4 * beat_out * RATE)
            ev += [
                {"at": S0, "cmd": "cue", "deck": incoming, "time_s": cue_b},
                {"at": S0, "cmd": "rate", "deck": incoming, "value": rate_b},
                {"at": S0, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "mid": 0.7, "high": 0.8, "ramp_s": 0.05},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.05},
                {"at": S0, "cmd": "start", "deck": incoming},
                {"at": S0, "cmd": "sync", "slave": incoming, "master": active},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.85,
                 "ramp_s": run_in * beat_out},
                # THE DROP: both full-range for 4 bars.
                {"at": S_drop, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 0.06},
                {"at": S_drop, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 0.06},
                # A ducks and drops its low so two kicks don't muddy or clip
                # (B is the star of the double drop); keeps some body. The
                # low cut is fast (no dual kick) but the GAIN duck rides
                # down over 2 beats, masked by B's drop - an instant -6 dB
                # duck at the drop reads as a level lurch (measured 7 dB).
                {"at": S_drop, "cmd": "eq", "deck": active, "low": 0.0,
                 "mid": 0.6, "ramp_s": 0.15},
                {"at": S_drop, "cmd": "gain", "deck": active, "value": 0.5,
                 "ramp_s": 2 * beat_out},
                # Hand over: A leaves, B's already full.
                {"at": both, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 4 * beat_out},
                {"at": out, "cmd": "stop", "deck": active},
                {"at": out, "cmd": "end_sync"},
            ]
            self._glide_home(ev, incoming, rate_b, out)
            return ev, out, S0

        if style == "bassline_layer":
            # Isolate A's groove as a looping bed and play B OVER it for an
            # extended stretch - two tracks genuinely playing together, not
            # a crossfade. A loops with mids/highs pulled out (its bassline +
            # kick repeat); B enters beat-locked with its bass cut, riding on
            # top with its melody/vocal/hats; after layer_beats the low end
            # hands over (A's loop releases, B's bass returns).
            ls = plan["loop_start_s"]
            loop_beats = plan["loop_beats"]
            layer = plan["layer_beats"]
            S0 = clock_at(ls)                 # engage the loop at its start
            if S0 < now_guard:
                # The chosen loop is already behind the playhead (armed
                # late). Rebase: loop the next grid-aligned window ahead -
                # same groove family in practice, and every event stays in
                # the future.
                ls = cur.nearest_downbeat(tel["time_s"] + 2 * cur.period_s)
                while clock_at(ls) < now_guard:
                    ls += 4 * cur.period_s
                S0 = clock_at(ls)
            bar = 4 * beat_out
            enter = S0 + int(loop_beats * beat_out * RATE)  # B in after 1 pass
            hand = enter + int(layer * beat_out * RATE)     # hand off the low
            out = hand + int(8 * beat_out * RATE)
            ev += [
                {"at": S0, "cmd": "loop", "deck": active, "start_s": ls,
                 "end_s": ls + loop_beats * cur.period_s},
                {"at": S0, "cmd": "eq", "deck": active, "high": 0.0,
                 "mid": 0.2, "low": 1.0, "ramp_s": bar},
                {"at": enter, "cmd": "cue", "deck": incoming,
                 "time_s": plan["in_s"]},
                {"at": enter, "cmd": "rate", "deck": incoming, "value": rate_b},
                {"at": enter, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 0.05},
                {"at": enter, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.05},
                {"at": enter, "cmd": "start", "deck": incoming},
                {"at": enter, "cmd": "sync", "slave": incoming,
                 "master": active},
                {"at": enter, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 8 * beat_out},
                # Hand the low end over: B's bass returns as A's loop leaves.
                {"at": hand, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "ramp_s": bar},
                {"at": hand, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 4 * beat_out},
                {"at": out, "cmd": "stop", "deck": active},
                {"at": out, "cmd": "clear_loop", "deck": active},
                {"at": out, "cmd": "end_sync"},
            ]
            self._glide_home(ev, incoming, rate_b, out)
            return ev, out, S0

        # Clean bass-swap EQ blend (long_blend / bass_swap / loop_roll_exit).
        # The golden rule: ONLY ONE BASSLINE AT A TIME. The incoming track
        # comes in with its low end fully cut and rides on top (we mix into
        # its intro/breakdown, so that's drums + atmosphere, not a clashing
        # lead); at the midpoint downbeat the bass swaps decisively in one
        # move; the outgoing track then leaves with its bass already gone.
        # No two-bass mud, no dueling low mids - the reliable pro default.
        # The blend COMPLETES at out_s - A's out point is the boundary where
        # its groove ends (that's why the seam scored there), so playing
        # 16-32 beats PAST it means A's own outro collapse lands mid-blend
        # (measured as 8-9 dB level lurches). Real DJs finish the blend ON
        # the boundary, riding A's last full-groove phrase.
        end = clock_at(plan["out_s"])
        S0 = max(end - int(nb * beat_out * RATE), now_guard)
        mid = (S0 + end) // 2
        half = (end - S0) / RATE / 2.0
        # Never swap the bass into a BASSLESS stretch of B: cutting A's low
        # while B enters on intro atmosphere collapses the mix floor ~8 dB
        # (measured). Time the swap to where B's content actually carries
        # bass, clamped inside the blend.
        b_bassy = None
        for sec in (cand.sections or []):
            if sec["end_s"] <= plan["in_s"] + 0.5:
                continue
            if sec.get("bass_share", 0.3) >= 0.28:
                b_bassy = max(sec["start_s"], plan["in_s"])
                break
        if b_bassy is not None:
            k = round((b_bassy - plan["in_s"]) / max(cand.period_s, 1e-6))
            mid = min(max(S0 + int(k * beat_out * RATE),
                          S0 + int(4 * beat_out * RATE)),
                      max(end - int(2 * beat_out * RATE), S0 + 1))
        # A's exit fade spans swap -> blend end however late the swap lands.
        half_exit = max((end - mid) / RATE, 2 * beat_out)
        ev += [
            {"at": S0, "cmd": "cue", "deck": incoming, "time_s": plan["in_s"]},
            {"at": S0, "cmd": "rate", "deck": incoming, "value": rate_b},
            # Incoming: bass fully cut, mids/highs open, fade up over 1st half.
            {"at": S0, "cmd": "eq", "deck": incoming, "low": 0.0,
             "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
            {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
             "ramp_s": 0.01},
            {"at": S0, "cmd": "start", "deck": incoming},
            {"at": S0, "cmd": "sync", "slave": incoming, "master": active},
            {"at": S0, "cmd": "gain", "deck": incoming, "value": 1.0,
             "ramp_s": half},
            # Swap downbeat: hand the bass over across ~1.5 beats. Decisive
            # to the ear, but on bass-dominant club material an instant
            # swap is a measured 8 dB RMS step whenever the two tracks'
            # low-end levels differ - the short crossfade keeps the floor.
            {"at": mid, "cmd": "eq", "deck": active, "low": 0.0,
             "ramp_s": 1.5 * beat_out},
            {"at": mid, "cmd": "eq", "deck": incoming, "low": 1.0,
             "ramp_s": 1.5 * beat_out},
            # Outgoing leaves over the rest of the blend (bass already gone).
            {"at": mid, "cmd": "gain", "deck": active, "value": 0.0,
             "ramp_s": half_exit},
        ]
        if style == "loop_roll_exit":
            ls = plan["loop_start_s"]
            ev += [
                {"at": S0, "cmd": "loop", "deck": active,
                 "start_s": ls, "end_s": ls + 16 * cur.period_s},
                {"at": S0 + int(16 * beat_out * RATE), "cmd": "loop",
                 "deck": active, "start_s": ls,
                 "end_s": ls + 8 * cur.period_s},
                {"at": S0 + int(24 * beat_out * RATE), "cmd": "loop",
                 "deck": active, "start_s": ls,
                 "end_s": ls + 4 * cur.period_s},
            ]
        stop_at = end + int(4 * beat_out * RATE)
        ev += [{"at": stop_at, "cmd": "stop", "deck": active},
               {"at": stop_at, "cmd": "end_sync"},
               {"at": stop_at, "cmd": "clear_loop", "deck": active}]
        self._glide_home(ev, incoming, rate_b, stop_at)
        return ev, stop_at, S0

    def preview_events(self, plan, cur, cand):
        """The EXACT automation a transition will run, timed from a zeroed
        clock with deck A cued at the blend start - for the planner's mix
        view and offline auditions. Returns (events, swap_at, blend_at) with
        'at' in samples where blend_at corresponds to plan['out_s']."""
        pre = 16 * cur.period_s          # drawing/audition run-up
        snapshot = {"clock": 0,
                    "decks": {"a": {"time_s": plan["out_s"] - pre,
                                    "rate": 1.0}}}
        return self.build_events(plan, snapshot, "a", "b", cur, cand)

    @staticmethod
    def _glide_home(ev, deck, rate, at):
        """After the swap the new dominant deck glides to its natural rate."""
        if abs(rate - 1.0) > 1e-4:
            ev.append({"at": at, "cmd": "rate", "deck": deck, "value": 1.0,
                       "ramp_s": abs(rate - 1.0) / GLIDE_PER_S})
