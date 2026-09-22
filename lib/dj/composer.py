"""THE COMPOSER (docs/DJ_THE_COMPOSER.md): every record gets a score of its own.

Pure: from a record's pairing lists (vocal over, bass under), its section map (grooves, its drop) and where it
will be heard first, `compose` returns two to four EVENTS placed on its grooves and its drop - a guest vocal, a
guest bassline for a phrase or two (then back), a build into its own drop - and, when the next record is in its
lists, the exit as a takeover at that record's drop. The conductor turns each event into an ORDER
(lib/dj/orders.py, RecordPlay.place_order) and executes it verbatim; THE PLAN shows the score the moment the
record is staged.

Density is the MOMENTS dial: rare = one event, some = two, lots = four. Variety: no two records in a row lead
with the same kind (`last_lead`), and a song used tonight is not used again.
"""
from __future__ import annotations

DENSITY = {"rare": 1, "some": 2, "lots": 4}
GUEST_BARS = 16              # a guest bassline stays two phrases, then the record's own comes back
LEAD_PHRASES = 4             # the first event no sooner than this after the record is heard (a decode is real seconds)
GAP_PHRASES = 1              # a phrase of the record alone between events
BUILD_LEAD_BARS = 17         # BUILD arms this many bars before the drop (its lead of 16, plus the arm bar)


def _grooves(track, from_t):
    """[(start, end)] of the record's grooves that begin at or after `from_t`, from its section map."""
    secs = track.sections or []
    out = []
    for i, s in enumerate(secs):
        if s.get("kind") != "groove":
            continue
        a = float(s.get("start_s") or 0.0)
        b = float(s.get("end_s") or (secs[i + 1]["start_s"] if i + 1 < len(secs) else track.duration_s))
        if b > from_t + 1.0:
            out.append((max(a, from_t), b))
    return out


def _drops(track, from_t):
    """The record's drops after `from_t`: a groove that follows a build or a breakdown (its own payoff)."""
    secs = track.sections or []
    out = []
    for i in range(1, len(secs)):
        if secs[i].get("kind") == "groove" and secs[i - 1].get("kind") in ("build", "breakdown"):
            t = float(secs[i].get("start_s") or 0.0)
            if t > from_t:
                out.append(t)
    return out


def compose(track, rec, entry_t, change_bars, bar_s, n_events, used_ids=(), avoid_ids=(), next_tid=None,
            last_lead=None, rec_end_t=None, on_deck_ids=()):
    """The score of `track` (the record) from its pairing lists `rec` ({"vocal_over": [...], "bass_under": [...]}),
    heard from `entry_t` (its own seconds) until `rec_end_t` (its seam out; None = its length minus a runway).
    Returns [{kind, other_tid, want_t, want_end, s_start, s_end, bars, why}] in time order, at most `n_events`:
      acapella  - a guest vocal over a groove (the vocal holds until it has sung its last)
      guest     - a guest bassline and melody under the record's drums for GUEST_BARS, then the record's own back
      build     - BUILD into the record's own drop (the eight bars before it once more, the bass pulled, the drop earned)
      takeover  - the exit: the next record's bassline takes over at its drop (only when `next_tid` is in the lists)
    `used_ids` / `avoid_ids`: songs not to use (used tonight, played, on the decks as records); `last_lead`: the kind the
    previous record led with (variety); `on_deck_ids`: songs already on a deck (they may come sooner)."""
    cb = int(change_bars)
    bar = float(bar_s)
    phrase = cb * bar
    dur = float(track.duration_s or 0.0)
    end_t = float(rec_end_t) if rec_end_t is not None else max(entry_t, dur - 60.0)
    used = set(int(x) for x in used_ids) | set(int(x) for x in avoid_ids)
    on_deck = set(int(x) for x in on_deck_ids)
    events = []
    t_free = float(entry_t) + LEAD_PHRASES * phrase          # the earliest a first event may come
    if n_events <= 0:
        return events
    vo = [x for x in (rec.get("vocal_over") or []) if int(x["id"]) not in used]
    bu = [x for x in (rec.get("bass_under") or []) if int(x["id"]) not in used]

    def take_vocal(t_min):
        # the best vocal whose groove has eight bars left after t_min (the groove's start, or where we are in it)
        best = None
        for x in vo:
            g = x.get("groove") or []
            if len(g) < 2:
                continue
            g0, g1 = float(g[0]), float(g[1])
            start = max(g0, t_min)
            if int(x["id"]) in on_deck:
                start = max(g0, t_min - (LEAD_PHRASES - 1) * phrase)   # already on a deck: one phrase of lead is enough
            if g1 - start < cb * bar or start > end_t - cb * bar:
                continue
            key = (float(x.get("score") or 0.0), -start)
            if best is None or key > best[0]:
                best = (key, x, start, g1)
        if best is None:
            return None
        _k, x, start, g1 = best
        sung = x.get("sung") or [0.0, 0.0]
        return {"kind": "acapella", "other_tid": int(x["id"]), "want_t": start, "want_end": g1, "s_start": float(sung[0]), "s_end": float(sung[1]),
                "bars": int(max(cb, min(4 * cb, (g1 - start) / bar) // cb * cb)), "fit": x.get("fit"),
                "why": f"the vocal of {str(x.get('title') or x['id'])[:24]} over its groove at {_mm(start)}"}

    def take_guest(t_min):
        # a guest bassline on a groove of the record with two phrases of room, from the bass-under list
        grooves = [(a, b) for a, b in _grooves(track, t_min) if b - max(a, t_min) >= (GUEST_BARS + cb) * bar and max(a, t_min) < end_t - GUEST_BARS * bar]
        if not grooves or not bu:
            return None
        a, b = grooves[0]
        start = max(a, t_min)
        x = max(bu, key=lambda z: float(z.get("score") or 0.0))
        return {"kind": "guest", "other_tid": int(x["id"]), "want_t": start, "want_end": min(b, start + GUEST_BARS * bar), "s_start": float(x.get("landmark") or 0.0),
                "s_end": None, "bars": GUEST_BARS, "fit": x.get("fit"),
                "why": f"{str(x.get('title') or x['id'])[:24]}'s bassline under its drums at {_mm(start)} for {GUEST_BARS} bars, then its own back"}

    def take_build(t_min):
        # BUILD into the record's own next drop, when there is room for its lead before it
        for d in _drops(track, t_min + BUILD_LEAD_BARS * bar):
            if d < end_t - 2 * cb * bar:
                return {"kind": "build", "other_tid": None, "want_t": d - BUILD_LEAD_BARS * bar, "want_end": d, "s_start": None, "s_end": None,
                        "bars": 2 * cb, "fit": None, "why": f"a BUILD into its own drop at {_mm(d)}"}
        return None
    # the lead: variety - not the kind the previous record led with, when there is a choice
    order = ["acapella", "guest", "build"]
    if last_lead in order and len(order) > 1:
        order.remove(last_lead)
        order.append(last_lead)
    t_min = t_free
    tried = set()
    while len(events) < n_events:
        made = None
        for kind in order:
            if kind in tried and kind != "guest":
                continue
            made = {"acapella": take_vocal, "guest": take_guest, "build": take_build}[kind](t_min)
            if made is not None:
                tried.add(kind)
                break
            tried.add(kind)
        if made is None:
            break
        events.append(made)
        if made.get("other_tid") is not None:
            used.add(int(made["other_tid"]))
            vo[:] = [x for x in vo if int(x["id"]) != int(made["other_tid"])]
            bu[:] = [x for x in bu if int(x["id"]) != int(made["other_tid"])]
        end_of = float(made["want_end"] or made["want_t"]) if made["kind"] != "acapella" else float(made["want_t"]) + made["bars"] * bar
        t_min = end_of + GAP_PHRASES * phrase
        if len(tried) >= 3 and made["kind"] == "build":
            break
    # the exit as a payoff: the next record's bassline takes over at its drop when it is in this record's lists
    if next_tid is not None and len(events) < n_events + 1:
        x = next((z for z in (rec.get("bass_under") or []) if int(z["id"]) == int(next_tid)), None)
        if x is not None:
            events.append({"kind": "takeover", "other_tid": int(next_tid), "want_t": None, "want_end": None, "s_start": float(x.get("landmark") or 0.0),
                           "s_end": None, "bars": 0, "fit": x.get("fit"), "exit": True,
                           "why": f"the exit: {str(x.get('title') or next_tid)[:24]}'s bassline takes over at its drop - then it is the record"})
    return events


def _mm(t):
    t = max(0.0, float(t or 0.0))
    return f"{int(t) // 60}:{int(t) % 60:02d}"
