"""Setlists: preplanned DJ sets, compiled against the library analysis.

A setlist is an ordered list of entries; each is an ANCHOR (must play, in
order, optionally near a target time offset) or a SUGGESTION (soft - the
live brain may swap it for something that fits the moment better). The
plan COMPILER resolves a setlist into a fully-checked plan: per-slot
timing estimates and per-seam transition plans (style, exit/entry points,
stretch rate, warnings) using the same Brain machinery the live DJ runs -
what you audition in the planner is what plays at night.

`optimize_order()` runs a BEAM SEARCH over suggestion orderings (anchors
pinned) maximizing the whole set's seam quality + arc fit. `autofill()` is
an anchor TIMING SOLVER: it picks how many fills each gap needs and gives
every entry a target_play_s so timed anchors actually land near their
offsets (compile_plan and the live order-mode both honor the hint).
`bridge()` finds the best 1-2 track chain between two remote tracks.
"""
import math
import time

from lib.dj.brain import Brain, camelot_compat
from lib.dj.rhythm import seam_rhythm, tempo_mult_for

# Styles whose blend runs BOTH tracks' low end open at some point - the
# ones a kick-pattern clash actually bites. bass_swap/stem_drum_swap keep
# one low bed at a time; cut_at_drop never overlaps; fades opt out.
_LOW_OPEN_STYLES = ("long_blend", "loop_roll_exit",
                    "loop_build", "drum_bridge")


def _make_edge(brain):
    """Cached seam-quality edge: score(a -> b at arc), floored so a dead
    edge never -inf's a whole ordering. The brain's rng jitter is baked
    into the first evaluation per key, so orderings compare consistently
    within one search.

    Floor 1e-12, NOT 1e-6: a legit-but-rough edge (a bridge leg beyond
    the stretch wall x a big energy gap) can land under 1e-6, and a high
    floor FLATTENS such edges equal - ordering vanishes and bridge()
    can't tell the honest connector from a truly dead pair (same failure
    class as the artist-penalty floor: penalties + a floor = ordering
    destruction). log(1e-12) is a harsh but finite -27.6 in the beam."""
    cache = {}

    def edge(a, b, arc):
        key = (a.id, b.id, round(arc, 1))
        v = cache.get(key)
        if v is None:
            s, _ = brain.score(a, b, arc, a.bpm)
            v = max(s, 1e-12)
            cache[key] = v
        return v
    return edge


def _play_estimate(theme, t):
    """Planning-time guess of how long a track will hold the floor."""
    return min(max((theme.min_play_s + theme.max_play_s) / 2.0, 60.0),
               max(t.duration_s - 45.0, 60.0))


# --------------------------------------------------------------------------
# CRUD (schema lives in db.py)
# --------------------------------------------------------------------------

def list_setlists(db):
    return [dict(r) for r in db.conn.execute(
        "SELECT s.*, COUNT(e.id) AS n_tracks FROM setlists s"
        " LEFT JOIN setlist_entries e ON e.setlist_id = s.id"
        " GROUP BY s.id ORDER BY s.updated_at DESC").fetchall()]


def create_setlist(db, name, theme="groove", notes=""):
    now = time.time()
    cur = db.conn.execute(
        "INSERT INTO setlists (name, theme, notes, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?)", (name, theme, notes, now, now))
    db.conn.commit()
    return cur.lastrowid


def delete_setlist(db, setlist_id):
    db.conn.execute("DELETE FROM setlists WHERE id = ?", (setlist_id,))
    db.conn.commit()


def get_setlist(db, setlist_id=None, name=None):
    q = "SELECT * FROM setlists WHERE " + \
        ("id = ?" if setlist_id is not None else "name = ?")
    row = db.conn.execute(
        q, (setlist_id if setlist_id is not None else name,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["entries"] = [dict(r) for r in db.conn.execute(
        "SELECT * FROM setlist_entries WHERE setlist_id = ?"
        " ORDER BY position", (out["id"],)).fetchall()]
    return out


def save_entries(db, setlist_id, entries):
    """Replace a setlist's entries. Each entry: {track_id, pin_type,
    target_offset_min, style_override, target_play_s} in play order."""
    db.conn.execute("DELETE FROM setlist_entries WHERE setlist_id = ?",
                    (setlist_id,))
    for pos, e in enumerate(entries):
        db.conn.execute(
            "INSERT INTO setlist_entries (setlist_id, position, track_id,"
            " pin_type, target_offset_min, style_override, target_play_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (setlist_id, pos, e["track_id"],
             e.get("pin_type", "suggestion"),
             e.get("target_offset_min"), e.get("style_override"),
             e.get("target_play_s")))
    db.conn.execute("UPDATE setlists SET updated_at = ? WHERE id = ?",
                    (time.time(), setlist_id))
    db.conn.commit()


# --------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------

def compile_plan(library, entries, theme, seed=0, pair_memory=None):
    """Resolve an ordered entry list into slot timings + seam plans.

    Returns {"slots": [...], "total_s", "warnings"}. Each slot:
        track (TrackInfo), entry, start_offset_s, play_s,
        transition (plan dict to NEXT slot | None), warnings [str],
        seam_info ({key_fit, d_off, floor, fade, pair_mem} | None)
    Timing model matches the live system: each track enters at the seam's
    in-point and exits at the next seam's out-point. pair_memory (the
    cross-night seam multipliers) is display/steering input the caller
    loads on its own thread - the compiler itself never touches the DB.
    """
    by_id = {t.id: t for t in library}
    brain = Brain(library, theme, seed=seed)
    if pair_memory:
        brain.pair_memory = dict(pair_memory)
    slots, warnings = [], []
    tracks = []
    for e in entries:
        t = by_id.get(e["track_id"])
        if t is None:
            warnings.append(f"entry #{e.get('position', '?')}: track "
                            f"{e['track_id']} missing from library - skipped")
            continue
        tracks.append((t, e))

    offset = 0.0
    # The OPENER enters at its EARLIEST mix-in, not its best-SCORING one:
    # mix_ins are ranked by seam score, and a mid-song winner opened the
    # set minutes deep - on tracks whose best in-point sits near the tail
    # the first exit could even land BEFORE the entry, and the preview
    # played a few seconds of song one before blending (user-reported).
    if tracks and tracks[0][0].mix_ins:
        entry_in_s = min(p["time_s"] for p in tracks[0][0].mix_ins)
    else:
        entry_in_s = 0.0
    entry_rate = 1.0                 # rate this track was brought in at
    # Aim each track to play a sensible stretch before mixing out, so the
    # set doesn't lurch through 30-second snippets.
    target_play = min(max(theme.min_play_s, 150.0), 300.0)
    from lib.dj.brain import ARATE_RAMP_PER_S, GLIDE_PER_S
    for i, (t, e) in enumerate(tracks):
        slot = {"track": t, "entry": e, "start_offset_s": offset,
                "in_s": entry_in_s,              # AT-SEAM source position
                "entry_rate": entry_rate,
                "transition": None, "seam_info": None, "warnings": []}
        in_s = entry_in_s
        # GLIDE SKEW: while this track glides home from entry_rate to 1.0
        # it consumes glide*(r-1)/2 source seconds more/less than wall
        # clock. Slot boundaries placed without this skew drift up to
        # ~200ms per seam at 2.5% stretch and CASCADE - the drawn beat
        # grids of adjacent lanes then sit rigidly misaligned (measured).
        skew = (entry_rate - 1.0) * abs(entry_rate - 1.0) / (2 * GLIDE_PER_S)
        if i + 1 < len(tracks):
            nxt, ne = tracks[i + 1]
            _, meta = brain.score(t, nxt, arc_target=0.6, out_bpm=t.bpm)
            if meta is None:
                rate, eff = brain.rate_for(t.bpm, nxt)
                if rate is None:
                    # Big gap: bend BOTH decks to a meeting tempo (the
                    # outgoing ramps into the blend, the incoming glides
                    # home after) before ever surrendering to a fade.
                    rate, eff, a_rate = brain.rate_for_dual(t.bpm, nxt)
                    if rate is not None:
                        slot["warnings"].append(
                            f"dual bend {t.bpm:.0f}->{t.bpm * a_rate:.0f}"
                            f"<-{nxt.bpm:.0f} bpm")
                        meta = {"rate": rate, "eff_bpm": eff, "pair": None,
                                "a_rate": a_rate}
                    else:
                        slot["warnings"].append(
                            f"tempo clash: {t.bpm:.0f} vs {nxt.bpm:.0f} bpm"
                            " - will long_fade")
                        meta = {"rate": 1.0, "eff_bpm": nxt.bpm,
                                "pair": None, "tempo_clash": True}
                else:
                    meta = {"rate": rate, "eff_bpm": eff, "pair": None}
            # Force the exit at least target_play after entry (but leave room
            # before the track ends), so it plays a real stretch first. An
            # autofill-stamped target_play_s (the anchor timing solver)
            # overrides the generic clamp.
            tp = float(e.get("target_play_s") or target_play)
            after = min(in_s + tp, max(t.duration_s - 50.0, in_s + 40))
            # A pinned seam style goes through plan_transition itself (NOT
            # a label overwrite after the fact - that skipped the style's
            # geometry: loop windows, drop anchoring, exit policy). The
            # live order-mode passes the same pin, so what compiles here
            # is what arms at night; a gate-refused pin is reported in
            # plan["diag"]["style_pin"].
            plan = brain.plan_transition(
                t, nxt, meta, after_s=after,
                force_style=ne.get("style_override"))
            # An EXPLICIT play hint (the anchor timing solver) is a timing
            # promise, not a suggestion: if the chosen seam overshoots it,
            # force the exit onto the nearest phrase - same trade the live
            # urgent exit makes. Drop-anchored styles keep their drop (the
            # exit IS the drop; moving it breaks the style's premise).
            if (e.get("target_play_s")
                    and plan["style"] not in ("cut_at_drop", "loop_build")
                    and plan["out_s"] > in_s + tp + 45.0):
                forced = t.nearest_phrase(in_s + tp)
                plan["out_s"] = min(max(forced, in_s + 40.0),
                                    max(t.duration_s - 30.0, in_s + 40.0))
                if plan["style"] == "loop_roll_exit":
                    plan["loop_start_s"] = max(
                        0.0, plan["out_s"] - 16 * t.period_s)
            pin = (plan.get("diag") or {}).get("style_pin")
            if pin and not pin.get("honored"):
                slot["warnings"].append(
                    f"style pin '{pin['want']}' refused"
                    f" ({pin.get('why_not') or 'gated'})"
                    f" - plays {plan['style']}")
            key_fit = camelot_compat(t.camelot, nxt.camelot)
            if key_fit < 0.5:
                slot["warnings"].append(
                    f"key clash {t.camelot} -> {nxt.camelot}")
            if abs(plan["rate"] - 1.0) > 0.05:
                slot["warnings"].append(
                    f"big stretch {plan['rate']:.3f}")
            if plan.get("pair_score", 0) < 0.05:
                slot["warnings"].append("weak seam (busy x busy?)")
            slot["transition"] = plan
            # SEAM SIGNALS for the planner cockpit - the same physics the
            # live selection leans on, surfaced per seam.
            slot["seam_info"] = {
                "key_fit": round(key_fit, 2),
                "d_off": round(abs(t.kick_offset_s - nxt.kick_offset_s), 3),
                "floor": round(brain._blend_floor(
                    t, plan["out_s"], nxt, plan["in_s"]), 2),
                "fade": plan["style"] == "long_fade",
                "pair_mem": brain.pair_memory.get((t.id, nxt.id)),
                # Tempo-multiple this seam reads B at (2.0 = B heard
                # double-time, 0.5 = half-time). Chip-worthy info even
                # when neither track has a rhythm signature.
                "mult": tempo_mult_for(t.bpm, nxt.bpm, plan["rate"]),
            }
            # RHYTHM TERMS (DB v13 signatures): the decomposed groove
            # compatibility behind the seam chips - kick agreement, swing
            # clash, flam risk, meter, tempo-multiple read. Stamped into
            # the plan by plan_transition (region-aware out-vs-in compare);
            # None when either side lacks a signature (evidence-gated
            # everywhere downstream).
            rt = plan.get("rhythm") or seam_rhythm(t, nxt, plan["rate"])
            if rt is not None:
                slot["seam_info"]["rhythm"] = rt
                # Warnings only when the estimate is trustworthy AND the
                # clash is audible in this style; a fade opts out of beat
                # physics (same rule the seam coloring applies).
                if rt["conf"] >= 0.5 and not slot["seam_info"]["fade"]:
                    if rt.get("meter_clash"):
                        slot["warnings"].append(
                            f"meter clash {rt.get('meter_a', 4)}/4 vs "
                            f"{rt.get('meter_b', 4)}/4")
                    if rt["swing_delta"] > 0.055:
                        slot["warnings"].append(
                            "swing clash (swung vs straight groove)")
                    if rt["kick_agreement"] < 0.35 \
                            and plan["style"] in _LOW_OPEN_STYLES:
                        slot["warnings"].append("kick pattern clash")
            # Dual-bend exit ramp: while THIS deck ramps to the meeting
            # tempo it also consumes ramp*(a-1)/2 extra source vs wall.
            # SHARES the event builder's ramp-rate constant - a stale
            # hardcoded 0.004 here (the pre-2026-07 ramp speed) modeled
            # the ramp ~5x shorter than the audio actually plays it, and
            # the skew error drifted every dual-bend slot boundary.
            a_r = plan.get("a_rate", 1.0) or 1.0
            exit_skew = (abs(a_r - 1.0) / ARATE_RAMP_PER_S) \
                * (a_r - 1.0) / 2.0 if abs(a_r - 1.0) > 1e-4 else 0.0
            # ...and the blend itself runs at a_rate: source consumed over
            # the blend = blend_wall*a_rate, wall = blend_wall.
            play = max(plan["out_s"] - in_s - skew - exit_skew, 40.0)
            # Blend-family overlaps play BEFORE the seam: by the time the
            # boundary (A's out point) arrives, B has already consumed the
            # blend's worth of source. slot["in_s"] is the AT-SEAM source
            # so drawing/click/playhead all agree with the audio.
            blend_wall = plan["beats"] * t.period_s / a_r \
                if plan["style"] in ("long_blend", "bass_swap",
                                     "filter_sweep", "loop_roll_exit",
                                     "stem_drum_swap", "acapella_out",
                                     "stem_bass_swap", "drum_bridge",
                                     "acapella_in", "melody_carry",
                                     "loop_in", "breakdown_swap") \
                else 0.0
            play -= blend_wall * (a_r - 1.0)  # blend runs at meeting tempo
            entry_in_s = plan["in_s"] + blend_wall * plan["rate"]
            entry_rate = plan["rate"]
        else:
            play = max(t.duration_s - in_s - skew, 40.0)
        slot["play_s"] = play
        slots.append(slot)
        offset += play
        warnings.extend(f"{t.title}: {w}" for w in slot["warnings"])

    # Anchor timing report: how far each anchor lands from its target.
    for s in slots:
        e = s["entry"]
        if e.get("pin_type") == "anchor" and e.get("target_offset_min"):
            err = s["start_offset_s"] / 60.0 - float(e["target_offset_min"])
            if abs(err) > 8.0:
                w = (f"anchor '{s['track'].title}' lands {abs(err):.0f} min "
                     f"{'late' if err > 0 else 'early'}")
                s["warnings"].append(w)
                warnings.append(w)
    return {"slots": slots, "total_s": offset, "warnings": warnings}


def suggest_set(library, theme, minutes, seed=0, start_track_id=None,
                pool_ids=None, flavor=None, unique=True):
    """PLAN MODE: generate a whole set from scratch. Inputs are the how-the-
    night-should-go controls: theme (bpm window, energy ARC, moods) and
    target length; the brain chains tracks so each seam is mixable and the
    energy tracks the arc. Returns entry dicts (all suggestions - pin what
    you care about afterwards).

    pool_ids: confine selection to these track ids (a genre/vibe subset) so
    the set is COHERENT, not a smattering of the whole library. flavor:
    {prefer_tags, avoid_tags, axis_targets} leaning within the pool."""
    brain = Brain(library, theme, seed=seed)
    if pool_ids is not None:
        brain.pool_ids = set(pool_ids)
    if flavor:
        brain.set_flavor(flavor)
    total_s = minutes * 60.0
    cur = None
    if start_track_id is not None:
        cur = next((t for t in library if t.id == start_track_id), None)
    if cur is None:
        cur = brain.choose_first(theme.arc_target(0.0))
    if cur is None:
        return []
    entries = [{"track_id": cur.id, "pin_type": "suggestion",
                "target_offset_min": None, "style_override": None}]
    brain.note_played(cur)
    if unique:
        # A PLANNED set must never repeat a track. The recency penalty is
        # deliberately soft (so the live all-night DJ can gracefully repeat
        # on a dry pool), so we hard-VETO each played track here: once used
        # it can never be chosen again, and when the pool is exhausted the
        # set simply ends (shorter) instead of looping the same songs.
        brain.veto_ids.add(cur.id)
    elapsed = 0.0
    while elapsed < total_s and len(entries) < 200:
        arc = theme.arc_target(min(elapsed / max(total_s, 60.0), 1.0))
        cand, meta = brain.choose_next(cur, arc, cur.bpm)
        if cand is None:
            break                        # pool/library exhausted - stop clean
        plan = brain.plan_transition(cur, cand, meta)
        in_s = cur.mix_ins[0]["time_s"] if cur.mix_ins else 0.0
        elapsed += max(plan["out_s"] - in_s, 60.0)
        entries.append({"track_id": cand.id, "pin_type": "suggestion",
                        "target_offset_min": None, "style_override": None,
                        "target_play_s": None})
        brain.note_played(cand)
        if unique:
            brain.veto_ids.add(cand.id)
        cur = cand
    return entries


def optimize_order(library, entries, theme, seed=0, beam=24):
    """Reorder the set's SUGGESTIONS for better seams and arc fit; anchors
    stay exactly at their positions.

    BEAM SEARCH over orderings (v3; was greedy): the objective is the sum
    of log seam scores across every consecutive pair at that slot's arc
    target - the same brain.score the live DJ selects with, so fade-risk,
    groove offsets, blend floors and pair memory all steer the order.
    Greedy grabbed the best next seam and painted the tail of the set into
    corners; the beam keeps the ~24 best partial orderings alive so a
    slightly worse seam now can buy a much better back half."""
    by_id = {t.id: t for t in library}
    valid = [e for e in entries if e["track_id"] in by_id]
    if len(valid) <= 2:
        return list(entries)
    brain = Brain(library, theme, seed=seed)
    edge = _make_edge(brain)
    est_total = max(sum(_play_estimate(theme, by_id[e["track_id"]])
                        for e in valid), 60.0)
    sugg = [e for e in valid if e.get("pin_type") != "anchor"]
    sugg_ids = {e["track_id"]: e for e in sugg}

    def arc_at(elapsed):
        return theme.arc_target(min(elapsed / est_total, 1.0))

    # State: (neg_score, picks tuple, used frozenset, last_track_id,
    #         elapsed). Dedup on (used, last) keeps the best prefix only.
    states = [(0.0, (), frozenset(), None, 0.0)]
    for slot in valid:
        anchor = slot.get("pin_type") == "anchor"
        nxt_states = {}
        for neg, picks, used, last_id, elapsed in states:
            if anchor:
                cands = [slot]
            else:
                cands = [e for tid, e in sugg_ids.items() if tid not in used]
                if len(cands) > 16 and last_id is not None:
                    # Cheap prefilter before the expensive full score:
                    # tempo reachability + energy fit keeps beam expansion
                    # bounded on very long sets.
                    prev_t = by_id[last_id]
                    a = arc_at(elapsed)

                    def cheap(e):
                        t = by_id[e["track_id"]]
                        r, _ = brain.rate_for(prev_t.bpm, t)
                        return ((0.0 if r is None else 1.0)
                                - abs(t.energy_proxy() - a))
                    cands = sorted(cands, key=cheap, reverse=True)[:16]
            for e in cands:
                t = by_id[e["track_id"]]
                a = arc_at(elapsed)
                if last_id is None:
                    s = max(1e-6, 1.0 - abs(t.energy_proxy() - a))
                else:
                    s = edge(by_id[last_id], t, a)
                key = (used | {e["track_id"]}, e["track_id"])
                cand_state = (neg - math.log(s), picks + (e,),
                              key[0], e["track_id"],
                              elapsed + _play_estimate(theme, t))
                best = nxt_states.get(key)
                if best is None or cand_state[0] < best[0]:
                    nxt_states[key] = cand_state
        states = sorted(nxt_states.values(), key=lambda st: st[0])[:beam]
        if not states:
            return list(entries)         # degenerate input - keep as-is
    return list(states[0][1])


def autofill(library, entries, theme, seed=0):
    """ANCHOR TIMING SOLVER (v3): insert brain-chosen suggestions between
    anchors AND size everyone's play length so each timed anchor actually
    lands near its target offset - not just "fill until the estimate says
    stop". For every gap it solves how many fills fit (each track can hold
    the floor for min_play..~max_play), chains them with the same scoring
    the live DJ uses (the last fill is weighted toward mixing INTO the
    anchor), and stamps target_play_s on the entries; compile_plan and the
    live order-mode both honor the stamp. Returns a NEW entry list
    (anchors kept in order and untouched)."""
    by_id = {t.id: t for t in library}
    brain = Brain(library, theme, seed=seed)
    edge = _make_edge(brain)
    anchors = [e for e in entries if e.get("pin_type") == "anchor"
               and e["track_id"] in by_id]
    if not anchors:
        return list(entries)
    used = {e["track_id"] for e in entries}
    lo_play = max(theme.min_play_s, 60.0)
    horizon = max((float(a.get("target_offset_min") or 0.0) * 60.0
                   for a in anchors), default=0.0) or 3600.0
    out = []
    offset = 0.0
    prev_track = None

    def hi_play(t):
        # A track can hold the floor a bit past the theme's usual draw
        # when the timeline needs it, but never into its own outro.
        return min(theme.max_play_s * 1.3, max(t.duration_s - 60.0, lo_play))

    for a in anchors:
        at = by_id[a["track_id"]]
        target_min = a.get("target_offset_min")
        if target_min is not None:
            gap = float(target_min) * 60.0 - offset
            # How many fills CAN occupy this gap given per-track bounds?
            k_min = int(math.ceil(gap / (theme.max_play_s * 1.3))) \
                if gap > 0 else 0
            k_max = int(gap // lo_play)
            k = min(max(int(round(gap / max(
                (theme.min_play_s + theme.max_play_s) / 2.0, 60.0))),
                k_min, 0), max(k_max, 0))
            for step in range(k):
                remaining = float(target_min) * 60.0 - offset
                want = remaining / (k - step)
                arc = theme.arc_target(min(offset / horizon, 1.0))
                best, best_s = None, -1.0
                for t in library:
                    if t.id in used or t.duration_s < lo_play + 60.0:
                        continue
                    s = (edge(prev_track, t, arc) if prev_track is not None
                         else max(1e-6, 1.0 - abs(t.energy_proxy() - arc)))
                    if step == k - 1:
                        # The last fill must also REACH the anchor - a
                        # great seam into a dead end still fades.
                        s *= edge(t, at, arc) ** 0.5
                    if s > best_s:
                        best, best_s = t, s
                if best is None:
                    break                # library truly dry for this gap
                play = min(max(want, lo_play), hi_play(best))
                out.append({"track_id": best.id, "pin_type": "suggestion",
                            "target_offset_min": None,
                            "style_override": None,
                            "target_play_s": round(play, 1)})
                used.add(best.id)
                brain.note_played(best)
                offset += play
                prev_track = best
        entry = dict(a)
        # Stamp the anchor's own play too: the solver's clock and the
        # compiler must run the same model, or every untimed anchor
        # drifts the timeline by (compiler default - solver estimate).
        if not entry.get("target_play_s"):
            entry["target_play_s"] = round(_play_estimate(theme, at), 1)
        out.append(entry)
        brain.note_played(at)
        if target_min is not None:
            # The anchor starts NOW; report drift via compile, but keep
            # the running clock honest either way.
            offset = max(offset, float(target_min) * 60.0)
        offset += float(entry["target_play_s"])
        prev_track = at
    return out


def order_by_shape(library, entries, theme, metric="tempo", shape="rise",
                   seed=0, beam=24):
    """Reorder a set so a chosen METRIC follows a target SHAPE across the
    night, while keeping seams beat-matchable. This is what makes a set
    actually BUILD - suggest_set/optimize_order chain by seam quality and
    won't produce a monotonic climb on their own.

    metric: 'tempo' (bpm) or 'energy'. shape: 'rise' (low->high, a build),
    'wind_down' (high->low), 'peak' (low->high->low), 'flat' (steady).

    Beam search minimizing per-slot distance to the target curve PLUS a seam
    penalty (an un-beat-matchable adjacency costs). For metric='tempo',
    'rise' naturally yields near-perfect seams (adjacent tempos are close).
    Anchors keep their flags but not their positions - shape owns the order;
    use autofill afterwards if you need timed anchors."""
    by_id = {t.id: t for t in library}
    items, seen = [], set()
    for e in entries:                    # dedupe - a planned set is unique
        tid = e["track_id"]
        if tid in by_id and tid not in seen:
            seen.add(tid)
            items.append((e, by_id[tid]))
    n = len(items)
    if n <= 2:
        return [e for e, _ in items]
    brain = Brain(library, theme, seed=seed)

    def mval(t):
        return t.bpm if metric == "tempo" else t.energy_proxy()
    # RANK-normalized metric: a shape is a claim about ORDER ("the
    # fastest records in the middle"), not about value distances. Value
    # normalization broke on skewed sets - tempos bunched high scored an
    # ascending order as well as a true crest for 'peak', because no
    # ordering could match the curve's raw VALUES; rank space always can.
    by_val = sorted(items, key=lambda et: mval(et[1]))
    nrm = {et[1].id: k / max(n - 1, 1) for k, et in enumerate(by_val)}

    def target(pos):
        f = pos / (n - 1)
        if shape == "rise":
            return f
        if shape == "wind_down":
            return 1.0 - f
        if shape == "peak":
            return 1.0 - abs(2.0 * f - 1.0)      # 0 -> 1 -> 0
        return 0.5                               # flat

    def seam_pen(a, b):
        r, _ = brain.rate_for(a.bpm, b)
        if r is None:
            return 1.0                           # fade only
        return 0.0 if abs(math.log(r)) < math.log(1.055) else 0.4

    def step_cost(last, t, tgt):
        pen = seam_pen(last, t) if last is not None else 0.0
        return abs(nrm[t.id] - tgt) + 0.6 * pen

    # State: (cost, picks tuple, used frozenset, last_track).
    states = [(0.0, (), frozenset(), None)]
    for pos in range(n):
        tgt = target(pos)
        nxt = {}
        for cost, picks, used, last in states:
            cands = [(e, t) for (e, t) in items if t.id not in used]
            # Truncate by the FULL step cost (arc fit AND seam), not arc
            # fit alone - arc-only truncation starved the beam of the
            # seam-viable near-target picks whenever a value cluster
            # crowded the top of the sort.
            cands.sort(key=lambda et: step_cost(last, et[1], tgt))
            # SHAPE CORRIDOR - HARD: shape OWNS the order (the tool's
            # whole contract); seams are optimized within it, never
            # traded against it. Without this the beam bought big shape
            # violations with cheap seams ('rise' ending on its two
            # lowest tracks because every seam onto those tempo-islands
            # was a fade the beam kept deferring). A state that strands
            # inventory outside the corridor simply dies; if the whole
            # beam dies, the rank baseline below wins with the exact
            # shape and pays its fades honestly.
            near = [et for et in cands
                    if abs(nrm[et[1].id] - tgt) <= 0.3]
            for e, t in near[:12]:
                c = cost + step_cost(last, t, tgt)
                key = (used | {t.id}, t.id)
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, picks + (e,), used | {t.id}, t)
        states = sorted(nxt.values(), key=lambda s: s[0])[:beam]
        if not states:
            break                        # corridor infeasible - ranked wins

    # RANK BASELINE, judged by the SAME objective as the beam. The beam
    # is myopic on sparse value tails (measured on the real library: a
    # 'rise' dumped its three slowest tracks at the END - clean early
    # seams outbid the tail's arc error until the outliers were the only
    # tracks left; the returned order cost 7.4 where plain value-sorting
    # cost 5.0). Assigning value-sorted tracks to target-sorted positions
    # realizes the requested shape EXACTLY (zero rank error by
    # construction); return whichever ordering the objective actually
    # scores cheaper.
    ranked = [None] * n
    for k, p in enumerate(sorted(range(n), key=target)):
        ranked[p] = by_val[k]

    def total(seq):
        c, last = 0.0, None
        for pos, (e, t) in enumerate(seq):
            c += step_cost(last, t, target(pos))
            last = t
        return c

    if states and len(states[0][1]) == n:
        beam_seq = [(e, by_id[e["track_id"]]) for e in states[0][1]]
        return [e for e, _ in min((beam_seq, ranked), key=total)]
    return [e for e, _ in ranked]


def bridge(library, a, b, theme, seed=0, exclude_ids=()):
    """Best 1-2 track chain CONNECTING a -> b: the repair move for two
    tempo/key-remote tracks (typically anchors) whose direct seam would
    fade. Maximizes the BOTTLENECK seam score of the chain - one great
    seam must not hide one dead one. Returns (tracks, bottleneck_score);
    ([], direct_score) when nothing beats the direct seam."""
    brain = Brain(library, theme, seed=seed)
    edge = _make_edge(brain)
    arc = (a.energy_proxy() + b.energy_proxy()) / 2.0
    skip = set(exclude_ids) | {a.id, b.id}
    direct = edge(a, b, arc)
    pool = [t for t in library if t.id not in skip]
    best_chain, best_v = [], direct
    singles = sorted(((min(edge(a, x, arc), edge(x, b, arc)), x)
                      for x in pool), key=lambda p: -p[0])
    if singles and singles[0][0] > best_v * 1.3:
        best_v, best_chain = singles[0][0], [singles[0][1]]
    # Two-track chains: only worth the extra floor time when they beat the
    # best single by a real margin. Search the top single-hop candidates.
    top_x = [x for _, x in sorted(((edge(a, x, arc), x) for x in pool),
                                  key=lambda p: -p[0])[:20]]
    top_y = [y for _, y in sorted(((edge(y, b, arc), y) for y in pool),
                                  key=lambda p: -p[0])[:20]]
    for x in top_x:
        ax = edge(a, x, arc)
        if ax <= best_v * 1.3:
            continue
        for y in top_y:
            if y.id == x.id:
                continue
            v = min(ax, edge(x, y, arc), edge(y, b, arc))
            if v > best_v * 1.3:
                best_v, best_chain = v, [x, y]
    return best_chain, best_v
