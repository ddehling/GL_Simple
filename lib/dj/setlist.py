"""Setlists: preplanned DJ sets, compiled against the library analysis.

A setlist is an ordered list of entries; each is an ANCHOR (must play, in
order, optionally near a target time offset) or a SUGGESTION (soft - the
live brain may swap it for something that fits the moment better). The
plan COMPILER resolves a setlist into a fully-checked plan: per-slot
timing estimates and per-seam transition plans (style, exit/entry points,
stretch rate, warnings) using the same Brain machinery the live DJ runs -
what you audition in the planner is what plays at night.

`autofill()` inserts brain-chosen suggestions between anchors until the
estimated timeline reaches each anchor's target offset.
"""
import time

from lib.dj.brain import Brain, camelot_compat


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
    target_offset_min, style_override} in play order."""
    db.conn.execute("DELETE FROM setlist_entries WHERE setlist_id = ?",
                    (setlist_id,))
    for pos, e in enumerate(entries):
        db.conn.execute(
            "INSERT INTO setlist_entries (setlist_id, position, track_id,"
            " pin_type, target_offset_min, style_override)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (setlist_id, pos, e["track_id"],
             e.get("pin_type", "suggestion"),
             e.get("target_offset_min"), e.get("style_override")))
    db.conn.execute("UPDATE setlists SET updated_at = ? WHERE id = ?",
                    (time.time(), setlist_id))
    db.conn.commit()


# --------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------

def compile_plan(library, entries, theme, seed=0):
    """Resolve an ordered entry list into slot timings + seam plans.

    Returns {"slots": [...], "total_s", "warnings"}. Each slot:
        track (TrackInfo), entry, start_offset_s, play_s,
        transition (plan dict to NEXT slot | None), warnings [str]
    Timing model matches the live system: each track enters at the seam's
    in-point and exits at the next seam's out-point.
    """
    by_id = {t.id: t for t in library}
    brain = Brain(library, theme, seed=seed)
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
    entry_in_s = tracks[0][0].mix_ins[0]["time_s"] \
        if tracks and tracks[0][0].mix_ins else 0.0
    entry_rate = 1.0                 # rate this track was brought in at
    # Aim each track to play a sensible stretch before mixing out, so the
    # set doesn't lurch through 30-second snippets.
    target_play = min(max(theme.min_play_s, 150.0), 300.0)
    from lib.dj.brain import GLIDE_PER_S
    for i, (t, e) in enumerate(tracks):
        slot = {"track": t, "entry": e, "start_offset_s": offset,
                "in_s": entry_in_s,              # AT-SEAM source position
                "entry_rate": entry_rate,
                "transition": None, "warnings": []}
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
                    slot["warnings"].append(
                        f"tempo clash: {t.bpm:.0f} vs {nxt.bpm:.0f} bpm - "
                        "will long_fade")
                    meta = {"rate": 1.0, "eff_bpm": nxt.bpm, "pair": None,
                            "tempo_clash": True}
                else:
                    meta = {"rate": rate, "eff_bpm": eff, "pair": None}
            # Force the exit at least target_play after entry (but leave room
            # before the track ends), so it plays a real stretch first.
            after = min(in_s + target_play, max(t.duration_s - 50.0, in_s + 40))
            plan = brain.plan_transition(t, nxt, meta, after_s=after)
            if ne.get("style_override"):
                plan["style"] = ne["style_override"]
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
            play = max(plan["out_s"] - in_s - skew, 40.0)
            # Blend-family overlaps play BEFORE the seam: by the time the
            # boundary (A's out point) arrives, B has already consumed the
            # blend's worth of source. slot["in_s"] is the AT-SEAM source
            # so drawing/click/playhead all agree with the audio.
            blend_wall = plan["beats"] * t.period_s \
                if plan["style"] in ("long_blend", "bass_swap",
                                     "filter_sweep", "loop_roll_exit") \
                else 0.0
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


def suggest_set(library, theme, minutes, seed=0, start_track_id=None):
    """PLAN MODE: generate a whole set from scratch. Inputs are the how-the-
    night-should-go controls: theme (bpm window, energy ARC, moods) and
    target length; the brain chains tracks so each seam is mixable and the
    energy tracks the arc. Returns entry dicts (all suggestions - pin what
    you care about afterwards)."""
    brain = Brain(library, theme, seed=seed)
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
    elapsed = 0.0
    while elapsed < total_s and len(entries) < 200:
        arc = theme.arc_target(min(elapsed / max(total_s, 60.0), 1.0))
        cand, meta = brain.choose_next(cur, arc, cur.bpm)
        if cand is None:
            break
        plan = brain.plan_transition(cur, cand, meta)
        in_s = cur.mix_ins[0]["time_s"] if cur.mix_ins else 0.0
        elapsed += max(plan["out_s"] - in_s, 60.0)
        entries.append({"track_id": cand.id, "pin_type": "suggestion",
                        "target_offset_min": None, "style_override": None})
        brain.note_played(cand)
        cur = cand
    return entries


def optimize_order(library, entries, theme, seed=0):
    """Reorder the set's SUGGESTIONS for better seams and arc fit; anchors
    stay exactly where they are. Greedy: at each position pick the pooled
    suggestion that scores best against the previous track at that moment's
    arc target."""
    by_id = {t.id: t for t in library}
    brain = Brain(library, theme, seed=seed)
    n = len(entries)
    if n <= 2:
        return list(entries)
    pool = [e for e in entries if e.get("pin_type") != "anchor"
            and e["track_id"] in by_id]
    est_total = sum(min(max(by_id[e["track_id"]].duration_s * 0.6, 90.0),
                        360.0) for e in entries if e["track_id"] in by_id)
    out, elapsed, prev = [], 0.0, None
    for i, slot in enumerate(entries):
        if slot.get("pin_type") == "anchor":
            pick = slot
        else:
            if not pool:
                continue
            arc = theme.arc_target(min(elapsed / max(est_total, 60.0), 1.0))
            best, best_s = None, -1.0
            for e in pool:
                t = by_id[e["track_id"]]
                if prev is None:
                    s = 1.0 - abs(t.energy_proxy() - arc)
                else:
                    s, _ = brain.score(prev, t, arc, prev.bpm)
                if s > best_s:
                    best, best_s = e, s
            pick = best
            pool.remove(best)
        out.append(pick)
        t = by_id.get(pick["track_id"])
        if t is not None:
            brain.note_played(t)
            elapsed += min(max(t.duration_s * 0.6, 90.0), 360.0)
            prev = t
    return out


def autofill(library, entries, theme, seed=0):
    """Insert brain-chosen suggestions between anchors so each timed anchor
    lands near its target offset. Returns a NEW entry list (anchors kept
    in order and untouched)."""
    by_id = {t.id: t for t in library}
    brain = Brain(library, theme, seed=seed)
    anchors = [e for e in entries if e.get("pin_type") == "anchor"]
    if not anchors:
        return list(entries)
    used = {e["track_id"] for e in entries}
    out = []
    offset = 0.0

    def track_play_estimate(t):
        return min(max(theme.min_play_s * 0.5 + theme.max_play_s * 0.5,
                       60.0), max(t.duration_s - 45.0, 60.0))

    prev_track = None
    for a in anchors:
        target_min = a.get("target_offset_min")
        at = by_id.get(a["track_id"])
        if at is None:
            continue
        # Fill until this anchor's target offset is nearly reached.
        while target_min is not None and \
                offset / 60.0 < float(target_min) - \
                track_play_estimate(at) / 60.0:
            arc = theme.arc_target(min(offset / max(
                float(anchors[-1].get("target_offset_min") or 60) * 60.0,
                60.0), 1.0))
            cand, meta = brain.choose_next(
                prev_track, arc, prev_track.bpm if prev_track else
                sum(theme.bpm_range) / 2.0)
            if cand is None or cand.id in used:
                # library dry for this moment - relax recency and move on
                break
            out.append({"track_id": cand.id, "pin_type": "suggestion",
                        "target_offset_min": None, "style_override": None})
            used.add(cand.id)
            brain.note_played(cand)
            offset += track_play_estimate(cand)
            prev_track = cand
        out.append(dict(a))
        brain.note_played(at)
        offset = max(offset + track_play_estimate(at),
                     (float(target_min) * 60.0 + track_play_estimate(at))
                     if target_min is not None else 0.0)
        prev_track = at
    return out
