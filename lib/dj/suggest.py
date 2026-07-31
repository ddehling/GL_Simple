"""Follow-the-set suggestion engine (was buried in the planner GUI).

suggest_followers() ranks what should FOLLOW a set-in-progress with the
same Brain scoring the live DJ selects with, plus a second FADE-REACHABLE
tier the brain itself doesn't have: candidates outside beat-match range
that would enter via the dipped long_fade, scored on what a fade actually
carries across (energy-vs-arc, theme mood, genre continuity). Living in
lib/dj makes that tier available to any future caller (the live horizon
included), not just the planner panel.

Pure function over snapshots - safe to call from a worker thread; it
never touches the DB.
"""
import math

from lib.dj import stretch_engine_name
from lib.dj.brain import (Brain, _title_root, camelot_compat,
                          chroma_key_compat)
from lib.dj.planner_util import track_genre, track_sig
from lib.dj.rhythm import seam_chips, seam_rhythm


def suggest_followers(library, entries, theme, compiled, pair_memory,
                      target_s=3600.0, n=7, anchor_idx=None):
    """Top-n candidates to FOLLOW the current set: scored with the live
    brain against the LAST track (seam quality, key/chroma, tempo reach,
    mood/genre coherence, pair memory) at the arc position the NEXT track
    would occupy, with an artist-variety lean against the last few
    entries. Empty set -> opener candidates (energy fit to the arc's
    start inside the theme's tempo window).

    ARC ANCHOR: progress toward the TARGET set length (target_s), NOT
    position within the current entries - the last slot's offset over the
    current total is ~1.0 by construction, which scored every suggestion
    at the arc's END and offered closers all night (user-reported). A
    3-track set of a planned 60 minutes is ~15% in, and the suggestions
    should sound like it.

    anchor_idx: follow the SELECTED slot instead of the set's last track
    (mid-set repair: 'what should come after slot 3?'). Arc position and
    the artist-variety window move to that slot; exclusions still cover
    the whole set."""
    by_id = {t.id: t for t in library}
    set_ids = [e["track_id"] for e in entries]
    used = set(set_ids)
    if anchor_idx is None or not (0 <= anchor_idx < len(set_ids)):
        anchor_idx = len(set_ids) - 1
    last = by_id.get(set_ids[anchor_idx]) if set_ids else None
    brain = Brain(library, theme)
    brain.pair_memory = dict(pair_memory or {})
    if last is None:
        arc0 = theme.arc_target(0.0)
        lo, hi = theme.bpm_range
        cands = [t for t in library
                 if any(lo * 0.95 <= t.bpm * m <= hi * 1.05
                        for m in (1.0, 2.0, 0.5))]
        cands.sort(key=lambda t: abs(t.energy_proxy() - arc0))
        return [{"id": t.id, "title": t.title, "artist": t.artist,
                 "genre": track_genre(t),
                 "bpm": t.bpm, "camelot": t.camelot, "fit": None,
                 "why": "opener", "beat": None, "key": None,
                 "theme": round(math.exp(
                     -((t.energy_proxy() - arc0) / 0.21) ** 2), 3),
                 "energy": round(t.energy_proxy(), 2),
                 "arc": round(arc0, 2)} for t in cands[:n]]
    brain.veto_ids.update(used)              # never suggest set members
    # Where would the track AFTER the anchor slot sit in the intended
    # night? Running length up to (and including) the anchor, plus half
    # a typical play, over the target length.
    est_play = 0.5 * (theme.min_play_s + theme.max_play_s)
    slots = (compiled or {}).get("slots") or []
    if anchor_idx < len(slots):
        s = slots[anchor_idx]
        total = s["start_offset_s"] + (s.get("play_s") or est_play)
    elif slots:
        total = (compiled or {}).get("total_s", 0.0)
    else:
        total = (anchor_idx + 1) * est_play
    progress = min((total + 0.5 * est_play) / max(target_s, 60.0), 1.0)
    arc = theme.arc_target(progress)
    recent_artists = {(by_id[i].artist or "").lower()
                      for i in set_ids[max(0, anchor_idx - 2):anchor_idx + 1]
                      if i in by_id and by_id[i].artist}
    vari = stretch_engine_name() == "vari"
    scored = []
    for t in library:
        if t.id in used:
            continue
        if brain.rate_for(last.bpm, t)[0] is None:   # cheap tempo gate
            continue
        s, meta = brain.score(last, t, arc, last.bpm)
        if s <= 0 or meta is None:
            continue
        if (t.artist or "").lower() in recent_artists:
            s *= 0.45                        # variety over artist streaks
        scored.append((s, t, meta))
    scored.sort(key=lambda x: -x[0])

    def components(t, meta):
        """The three per-dimension qualities shown in the panel, computed
        the same way score() weighs them."""
        rate = meta.get("rate") or 1.0
        beat = math.exp(-((abs(math.log(rate))) / 0.045) ** 2)
        pair = meta.get("pair")
        if pair and not pair.get("beaty", True):
            beat *= 0.5                  # one side beatless: it'd be a fade
        key = camelot_compat(last.camelot, t.camelot)
        semis = (12.0 * math.log(max(rate, 1e-6)) / math.log(2.0)
                 if vari else float(meta.get("pitch_st") or 0))
        sc = chroma_key_compat(getattr(last, "chroma", None),
                               getattr(t, "chroma", None), semis)
        if sc is not None:
            key = 0.45 * key + 0.55 * sc
        e = brain._arc_energy(t)
        s_en = math.exp(-((e - arc) / 0.21) ** 2)
        mood = sum(theme.mood_weights.get(m, 0.0) * f
                   for m, f in (t.mood_hist or {}).items())
        theme_q = 0.6 * s_en + 0.4 * min(1.0, mood / 0.35)
        # Groove compatibility vs the last track (region-aware; None when
        # either side has no rhythm signature). rate=None in the chips
        # call: the stretch is its own column here.
        rt = seam_rhythm(last, t, rate)
        return {"beat": round(beat, 3), "key": round(key, 3),
                "theme": round(theme_q, 3),
                "groove": round(rt["score"], 3) if rt else None,
                "groove_chips": seam_chips({"rate": None}, {"rhythm": rt})
                if rt else [],
                "stretch_pct": round((rate - 1.0) * 100.0, 1),
                "energy": round(e, 2), "arc": round(arc, 2)}

    # One suggestion per SONG, and never a song the SET already contains
    # in ANY copy/version: identity is title root + content hash + the
    # mangled-tag re-rip check ('02_Alex - Youth (feat_ ...)': same ~8s
    # duration bucket AND one flattened title containing the other).
    # Seeding the seen-sets from the set's own tracks makes set members
    # and their twins unsuggestable, not just their exact track ids.
    set_tracks = [by_id[i] for i in used if i in by_id]
    seen_roots = {(_title_root(t.title) or t.title.lower())
                  for t in set_tracks}
    seen_hashes = {h for _, _, h in map(track_sig, set_tracks) if h}
    kept = [(f, b) for f, b, _ in map(track_sig, set_tracks) if f]
    n_viable = max(len(scored), 1)
    out = []
    for rank, (s, t, meta) in enumerate(scored):
        root = _title_root(t.title) or t.title.lower()
        if root in seen_roots:
            continue
        flat, dur_b, chash = track_sig(t)
        if chash and chash in seen_hashes:
            continue
        if any(abs(dur_b - kb) <= 1 and flat and kf
               and (flat in kf or kf in flat)
               for kf, kb in kept):
            continue
        seen_roots.add(root)
        if chash:
            seen_hashes.add(chash)
        kept.append((flat, dur_b))
        out.append({"id": t.id, "title": t.title, "artist": t.artist,
                    "genre": track_genre(t),
                    "bpm": t.bpm, "camelot": t.camelot, "fit": round(s, 3),
                    # Where this candidate ranks among EVERYTHING viable -
                    # the fit's raw scale is a many-factor product (ceiling
                    # ~0.4) and reads misleadingly low on its own.
                    "top_pct": max(1, round(100 * (rank + 1) / n_viable)),
                    "n_viable": n_viable,
                    "rate": round((meta.get("rate") or 1.0), 3),
                    **components(t, meta)})
        if len(out) >= n:
            break

    # SECOND TIER: fade-reachable. The beat tier can only draw from the
    # ±8% tempo neighbourhood of the last track (63% of an eclectic
    # library is out of reach at any moment) - but the live DJ has a
    # deliberate entrance for exactly those: the dipped long_fade, where
    # beat and key don't overlap enough to matter. Score those on what a
    # fade DOES carry across: energy-vs-arc, theme mood, genre
    # continuity. Still inside the theme's tempo identity, still
    # variety-leaned, same song-dedup.
    lo, hi = theme.bpm_range
    fade_scored = []
    for t in library:
        if t.id in used:
            continue
        if brain.rate_for(last.bpm, t)[0] is not None:
            continue                          # that's the beat tier's job
        if not any(lo * 0.93 <= t.bpm * m <= hi * 1.07
                   for m in (1.0, 2.0, 0.5)):
            continue
        s_en = math.exp(-((brain._arc_energy(t) - arc) / 0.21) ** 2)
        mood = sum(theme.mood_weights.get(m, 0.0) * f
                   for m, f in (t.mood_hist or {}).items())
        s_mood = 0.25 + mood
        inter = len(t.genre_set & last.genre_set)
        base = min(len(t.genre_set), len(last.genre_set))
        s_coh = 0.84 + 0.16 * (inter / base if base else 0.0)
        s = s_en * s_mood * s_coh
        if (t.artist or "").lower() in recent_artists:
            s *= 0.45
        theme_q = 0.6 * s_en + 0.4 * min(1.0, mood / 0.35)
        fade_scored.append((s, t, s_en, theme_q))
    fade_scored.sort(key=lambda x: -x[0])
    n_fade = max(len(fade_scored), 1)
    for rank, (s, t, s_en, theme_q) in enumerate(fade_scored):
        if len(out) >= n + 7:
            break
        root = _title_root(t.title) or t.title.lower()
        if root in seen_roots:
            continue
        flat, dur_b, chash = track_sig(t)
        if chash and chash in seen_hashes:
            continue
        if any(abs(dur_b - kb) <= 1 and flat and kf
               and (flat in kf or kf in flat) for kf, kb in kept):
            continue
        seen_roots.add(root)
        if chash:
            seen_hashes.add(chash)
        kept.append((flat, dur_b))
        out.append({"id": t.id, "title": t.title, "artist": t.artist,
                    "genre": track_genre(t), "tier": "fade",
                    "bpm": t.bpm, "camelot": t.camelot, "fit": round(s, 3),
                    "top_pct": max(1, round(100 * (rank + 1) / n_fade)),
                    "n_viable": n_fade,
                    "beat": None, "key": None, "theme": round(theme_q, 3),
                    "energy": round(t.energy_proxy(), 2),
                    "arc": round(arc, 2)})
    return out
