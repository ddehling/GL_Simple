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
          "bassline_layer", "double_drop", "long_fade")
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
        best = None
        for o in outs[:6]:
            sec_a = cur.section_at(min(o["time_s"] + 1.0,
                                       cur.duration_s - 1.0))
            for i in cand.mix_ins[:6]:
                sec_b = cand.section_at(min(i["time_s"] + 1.0,
                                            cand.duration_s - 1.0))
                if sec_a is None or sec_b is None:
                    continue
                busy_a = sec_a.get("busyness") or 0.0
                busy_b = sec_b.get("busyness") or 0.0
                voc_a = sec_a.get("vocalness") or 0.0
                voc_b = sec_b.get("vocalness") or 0.0
                if busy_a > 0.65 and busy_b > 0.65:
                    continue                     # two walls of sound: never
                if voc_a > 0.55 and voc_b > 0.55:
                    continue                     # two vocal lines: never
                quiet = 1.0 - 0.5 * min(busy_a + busy_b, 1.6) / 1.6
                clash = 1.0 - 0.4 * min(voc_a + voc_b, 1.4) / 1.4
                score = (max(o["score"], 0.05) * max(i["score"], 0.05)
                         * quiet * clash)
                if best is None or score > best["score"]:
                    best = {"out_s": o["time_s"], "in_s": i["time_s"],
                            "out_hint": o.get("style_hint", "blend"),
                            "in_hint": i.get("style_hint", "blend"),
                            "score": round(score, 4),
                            "busy": (round(busy_a, 2), round(busy_b, 2))}
        return best

    def _drop_after(self, track, after_s):
        """Start time of the first strong drop section at/after after_s
        (else the strongest drop anywhere), or None."""
        drops = [s for s in track.sections if s["kind"] == "drop"]
        if not drops:
            return None
        ahead = [s for s in drops if s["start_s"] >= after_s]
        pick = min(ahead, key=lambda s: s["start_s"]) if ahead else \
            max(drops, key=lambda s: s.get("energy", 0.0))
        return pick["start_s"]

    def _bass_loop(self, track, before_s):
        """Best loop in `track` to isolate as a repeating groove bed: a
        bass-heavy, low-vocal, repetitive loop before the exit. Returns the
        loop dict (+ resolved section) or None."""
        best = None
        for l in track.loops:
            if l["start_s"] >= before_s:
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
        if low_conf:
            style = "long_fade"
        else:
            if (cur.downbeat_conf < 0.15 or cand.downbeat_conf < 0.15):
                weights["cut_at_drop"] = 0.0
            if pair["in_hint"] != "pre_drop":
                weights["cut_at_drop"] = 0.0
            loop_ok = any(l["start_s"] < pair["out_s"] for l in cur.loops)
            if not loop_ok:
                weights["loop_roll_exit"] = 0.0
            # bassline_layer needs a bass-heavy, low-vocal loop in A to
            # isolate as the repeating groove.
            if self._bass_loop(cur, pair["out_s"]) is None:
                weights["bassline_layer"] = 0.0
            # double_drop aligns A's drop onset with B's drop onset (the
            # drop boundary IS a downbeat, so this works even where the
            # global downbeat_offset is uncertain); the sync snap handles
            # beat phase. Needs a drop in each track.
            a_drop = self._drop_after(cur, pair["out_s"] - 8 * cur.period_s)
            b_drop = self._drop_after(cand, cand.duration_s * 0.15)
            if a_drop is None or b_drop is None:
                weights["double_drop"] = 0.0
            weights["long_fade"] = 0.0
            menu = [(s, w) for s, w in weights.items() if w > 0]
            if not menu:
                menu = [("bass_swap", 1.0)]
            styles, ws = zip(*menu)
            style = self.rng.choices(styles, weights=ws, k=1)[0]

        beats = {"long_blend": 32, "bass_swap": 16, "cut_at_drop": 16,
                 "loop_roll_exit": 32, "bassline_layer": 16,
                 "double_drop": 16, "long_fade": 0}[style]
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
            loop = self._bass_loop(cur, pair["out_s"])
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

        beat_out = cur.period_s / rate_a          # output-domain beat of A
        style = plan["style"]
        rate_b = plan["rate"]
        ev = []

        if style == "long_fade":
            S0 = clock_at(plan["out_s"])
            dur = 20.0
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
            S0 = S_cut - lead
            cue_b = max(0.0, plan["in_s"] - 16 * cand.period_s)
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

        if style == "double_drop":
            # Align B's drop onset to A's drop onset: B runs in for 16 beats
            # bass-cut (build tension under A), both drops HIT together for 4
            # bars of full-range double-drop, then A exits and B rides on.
            run_in = 16
            S_drop = clock_at(plan["out_s"])            # A's drop moment
            b_period = cand.period_s
            S0 = S_drop - int(run_in * b_period / rate_b * RATE)
            cue_b = max(0.0, plan["in_s"] - run_in * b_period)
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
                # (B is the star of the double drop); keeps some body.
                {"at": S_drop, "cmd": "eq", "deck": active, "low": 0.0,
                 "mid": 0.6, "ramp_s": 0.06},
                {"at": S_drop, "cmd": "gain", "deck": active, "value": 0.5,
                 "ramp_s": 0.06},
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

        # Beat-matched blends: long_blend / bass_swap / loop_roll_exit.
        # v2 staged band pull-in, like riding a mixer's EQ knobs: the new
        # track's HIGHS ride in first (hats/sparkle announce it), MIDS open
        # a quarter in, the BASS swaps decisively at the midpoint downbeat,
        # and the outgoing track loses its top end as it leaves so the two
        # never fight for the same bands.
        S0 = clock_at(plan["out_s"])
        q1 = S0 + int(nb / 4 * beat_out * RATE)
        mid = S0 + int(nb / 2 * beat_out * RATE)
        q3 = S0 + int(3 * nb / 4 * beat_out * RATE)
        end = S0 + int(nb * beat_out * RATE)
        qlen = nb / 4 * beat_out
        ev += [
            {"at": S0, "cmd": "cue", "deck": incoming, "time_s": plan["in_s"]},
            {"at": S0, "cmd": "rate", "deck": incoming, "value": rate_b},
            {"at": S0, "cmd": "eq", "deck": incoming, "low": 0.0,
             "mid": 0.25, "high": 1.0, "ramp_s": 0.01},
            {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
             "ramp_s": 0.01},
            {"at": S0, "cmd": "start", "deck": incoming},
            {"at": S0, "cmd": "sync", "slave": incoming, "master": active},
            {"at": S0, "cmd": "gain", "deck": incoming, "value": 1.0,
             "ramp_s": nb * beat_out},
            {"at": q1, "cmd": "eq", "deck": incoming, "mid": 1.0,
             "ramp_s": qlen},
            {"at": q1, "cmd": "eq", "deck": active, "high": 0.55,
             "ramp_s": qlen},
            {"at": mid, "cmd": "eq", "deck": active, "low": 0.0,
             "ramp_s": 0.4},
            {"at": mid, "cmd": "eq", "deck": incoming, "low": 1.0,
             "ramp_s": 0.4},
            {"at": mid, "cmd": "gain", "deck": active, "value": 0.0,
             "ramp_s": nb / 2 * beat_out},
            {"at": q3, "cmd": "eq", "deck": active, "high": 0.25,
             "mid": 0.5, "ramp_s": qlen},
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
