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
    for i, (t, e) in enumerate(tracks):
        slot = {"track": t, "entry": e, "start_offset_s": offset,
                "transition": None, "warnings": []}
        in_s = t.mix_ins[0]["time_s"] if t.mix_ins else 0.0
        if i + 1 < len(tracks):
            nxt, ne = tracks[i + 1]
            _, meta = brain.score(t, nxt, arc_target=0.6, out_bpm=t.bpm)
            if meta is None:
                rate, eff = brain.rate_for(t.bpm, nxt)
                if rate is None:
                    slot["warnings"].append(
                        f"tempo clash: {t.bpm:.0f} vs {nxt.bpm:.0f} bpm - "
                        "will long_fade")
                    meta = {"rate": 1.0, "eff_bpm": nxt.bpm, "pair": None}
                else:
                    meta = {"rate": rate, "eff_bpm": eff, "pair": None}
            plan = brain.plan_transition(t, nxt, meta)
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
            play = max(plan["out_s"] - in_s, 30.0)
        else:
            play = max(t.duration_s - in_s, 30.0)
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
