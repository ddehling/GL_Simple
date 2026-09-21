"""Orders: what the user asked for, decided ONCE and executed verbatim (docs/DJ_ORDERS.md).

An order (PLAY THIS PAIR today; TAKEOVER / BRIDGE with a song, UP NEXT through the same door) is a SCORE:
every concrete decision, in the record's own seconds and in bars, computed once from the real inputs (the
record's stems) when they are known, refused then by name when they do not fit - and then the seam, the event
and the arm READ the score; they search nothing and re-decide nothing. THE PLAN and the picture render the
score object itself.

This module is pure: it decides, it does not touch decks. `plan_pair_entry` is shared by the conductor (when
the pair's record is staged) and the pair census (offline, over the library) - the list shows what the same
planner accepts.
"""
from __future__ import annotations

import math
import time

STATES = ("staging", "fixed", "running", "done", "refused", "failed")


def new_order(kind, record_tid, other_tid, want_t=None, want_end=None, s_start=None, s_end=None, words="", **extra):
    """A fresh order. `kind`: "acapella" (the other song's vocal over the record's groove at `want_t`..`want_end`;
    its sung stretch `s_start`..`s_end`) or "takeover" (its bassline under the record's drums at its drop)."""
    o = {"id": int(time.time() * 1000) % 10_000_000, "kind": kind, "record_tid": int(record_tid), "track_id": int(other_tid),
         "want_t": (float(want_t) if want_t is not None else None), "want_end": (float(want_end) if want_end is not None else None),
         "s_start": (float(s_start) if s_start is not None else None), "s_end": (float(s_end) if s_end is not None else None),
         "words": words, "t": time.time(), "state": "staging", "why": "planning: its record's stems",
         # decided once, from the stems (None until then)
         "entry": None, "start_t": None, "entry_t": None, "event_t": None, "bars": None, "cue_t": None, "cut": False,
         "event_bar": None, "t_fixed": None, "t_end": None}
    o.update(extra)
    return o


def _first_body(track):
    for s in (getattr(track, "sections", None) or []):
        if s.get("kind") != "intro":
            return float(s.get("start_s") or 0.0)
    return 0.0


def plan_pair_entry(track, env, want_t, want_end, change_bars, bar_s, stem_carries, s_start=None, s_end=None, s_rate=1.0):
    """Where a pair's RECORD enters so that its groove at `want_t`..`want_end` is inside its play with the vocal on
    it, decided from its stems. Returns a dict {ok, why, entry ("drums" | "cut"), start_t (B's time at the seam's
    first bar), entry_t (where it is first heard), event_t (the record's time the vocal comes), bars (the vocal's
    length), cue_t (the other song's time cued to the event's bar), groove_bars}.

    Three doors, in order: (1) five phrases before the groove, its drums carrying two phrases from there - the
    classic drums-first seam; (2) a CUT at the latest 4-bar line before the groove (two phrases before it, or
    earlier) where its drums carry two phrases - more of its intro under the room, then the drop with the vocal on
    it; (3) a cut ON the groove, the vocal four bars in (a cue needs two bars to lock) - only when the groove has
    eight bars left after that. Otherwise refused, by name. `stem_carries(env, t, span_s, lanes, bar_s)` is the
    conductor's own judge (RemixConductor._stem_carries), so the seam and this decision cannot disagree."""
    cb = int(change_bars)
    bar = float(bar_s)
    want = float(want_t)
    end = float(want_end) if want_end is not None else want + 4 * cb * bar
    groove_bars = (end - want) / max(bar, 1e-6)
    out = {"ok": False, "why": "", "entry": None, "start_t": None, "entry_t": None, "event_t": None, "bars": None, "cue_t": None,
           "groove_bars": round(groove_bars, 1)}
    nd = getattr(track, "nearest_downbeat", None) or (lambda x: x)
    title = (getattr(track, "title", "") or "")[:24]
    if groove_bars < cb - 0.5:
        out["why"] = f"{title}'s groove at {_mm(want)} is {groove_bars:.0f} bars: a phrase is needed"
        return out
    first = _first_body(track)
    floor_t = 2.0 * bar
    span = 2 * cb * bar

    def drums_carry(t):
        return stem_carries(env, t, span, lanes=("drums",), bar_s=bar) is not False

    # (1) drums first, five phrases early
    ps = nd(want - 5 * cb * bar)
    if ps >= max(floor_t, first - 0.5 * bar) and drums_carry(ps):
        out.update(ok=True, entry="drums", start_t=float(ps), entry_t=float(ps), event_t=float(nd(want)),
                   why=f"{title} enters at {_mm(ps)}, drums first, its groove five phrases on")
    else:
        # (2) a cut before the groove, its drums carrying from there
        tt = nd(want - 2 * cb * bar)
        found = None
        while tt >= floor_t:
            if drums_carry(tt):
                found = tt
                break
            tt = nd(tt - 4 * bar)
            if tt <= 0:
                break
        if found is not None:
            out.update(ok=True, entry="cut", start_t=float(max(floor_t, found - 4 * bar)), entry_t=float(found), event_t=float(nd(want)),
                       why=f"{title} enters whole at {_mm(found)} (a cut), its groove {int(round((want - found) / bar))} bars on")
        else:
            # (3) on the groove itself: the vocal four bars in
            ev_t = nd(want + 4 * bar)
            if (end - ev_t) / bar < cb - 0.5:
                out["why"] = (f"{title}'s drums carry nowhere before its groove at {_mm(want)} and the groove is {groove_bars:.0f} bars: "
                              f"entering on it, the vocal would have {max(0.0, (end - ev_t) / bar):.0f} bars")
                return out
            out.update(ok=True, entry="cut", start_t=float(max(floor_t, want - 4 * bar)), entry_t=float(nd(want)), event_t=float(ev_t),
                       why=f"{title} enters whole on its groove at {_mm(want)} (nothing of it carries before); the vocal four bars in")
    # the vocal's length: whole phrases inside the groove from the event, at most four, at least one - and no
    # longer than the other song's sung stretch
    room = int(math.floor((end - out["event_t"]) / bar + 0.5))
    bars = max(cb, min(4 * cb, room) // cb * cb)
    if s_start is not None and s_end is not None:
        stretch = int((float(s_end) - float(s_start)) / max(bar * float(s_rate or 1.0), 1e-6))
        if stretch >= cb:
            bars = max(cb, min(bars, stretch // cb * cb))
    if room < cb - 1:
        out.update(ok=False, why=f"{title}'s groove leaves {room} bars after {_mm(out['event_t'])}: a phrase is needed")
        return out
    out["bars"] = int(bars)
    out["cue_t"] = (float(s_start) - bar * float(s_rate or 1.0)) if s_start is not None else None
    return out


def order_words(o, title_of):
    """One line for THE PLAN: what the order is, in words."""
    R, S = title_of(o["record_tid"])[:22], title_of(o["track_id"])[:22]
    if o["kind"] == "acapella":
        return f"the vocal of {S} over {R}" + (f" from {_mm(o['want_t'])}" if o.get("want_t") is not None else "")
    return f"{S}'s bassline takes over {R} at its drop - then it is the record"


def order_state_words(o):
    """The state, for the row's second line."""
    st = o["state"]
    if st == "staging":
        return o.get("why") or "planning: its record's stems"
    if st == "fixed":
        return "planned: " + (o.get("why") or "")
    if st == "running":
        return "armed: " + (o.get("why") or "")
    if st == "done":
        return "done: " + (o.get("why") or "")
    if st == "refused":
        return "REFUSED: " + (o.get("why") or "")
    return "FAILED: " + (o.get("why") or "")


def _mm(t):
    t = max(0.0, float(t or 0.0))
    return f"{int(t) // 60}:{int(t) % 60:02d}"
