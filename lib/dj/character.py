"""Derived track CHARACTER: danceability, valence (dark<->bright/happy),
brightness, and dynamics - computed from features the scanner already
measured (no re-analysis, no new dependency, no ML models).

This is the pragmatic answer to "the system needs mood/danceability info":
Essentia (the toolkit Spotify used for those descriptors) has no reliable
Windows build, so instead of a fragile dependency we derive the same class
of signals from the stored analysis - rhythm density, grid confidence, and
section repetitiveness for danceability; key mode + spectral brightness +
energy for valence. Heuristic, but grounded in real audio measurements and
always available. load_library percentile-ranks danceability/valence across
the library so they SPREAD 0..1 (like energy), and folds tags into
all_tags. An Essentia backend could replace these functions later.
"""


def _clamp(x):
    return max(0.0, min(1.0, x))


def brightness(t):
    """0 = dark/bassy, 1 = bright/airy. From the high-frequency share."""
    return _clamp((t.spectral.get("high_share", 0.25)) / 0.4)


def _avg_repetitiveness(sections):
    reps = [s.get("repetitiveness") or 0.0 for s in sections
            if s.get("kind") == "groove"] or \
           [s.get("repetitiveness") or 0.0 for s in sections]
    return sum(reps) / len(reps) if reps else 0.0


# Music2Emo mood tags that lean toward / away from a dance floor. Used only
# as a small nudge on the rhythm-based danceability heuristic when present.
_DANCE_UP = {"party", "groovy", "energetic", "upbeat", "fun", "sport",
             "sexy", "cool", "fast", "powerful"}
_DANCE_DOWN = {"soundscape", "ambient", "relaxing", "meditative", "sad",
               "melancholic", "dream", "nature", "film", "calm", "slow",
               "emotional", "deep"}


def danceability_raw(t):
    """How much the track drives a floor: steady rhythm + a clear beat grid
    + a repetitive groove. rhythm_density ~6-9 on grooves, ~0 in ambient.
    When the ML mood pass has run, party/ambient mood tags nudge it +/-."""
    dens = min((t.rhythm_density or 0.0) / 6.0, 1.0)
    grid = t.bpm_conf or 0.0
    rep = _avg_repetitiveness(t.sections)
    base = 0.45 * dens + 0.3 * grid + 0.25 * rep
    moods = set(getattr(t, "ml_moods", None) or [])
    if moods:
        bonus = 0.1 * len(moods & _DANCE_UP) - 0.1 * len(moods & _DANCE_DOWN)
        base += max(-0.2, min(0.2, bonus))
    return _clamp(base)


def valence_raw(t):
    """0 = dark/moody, 1 = bright/uplifting. PREFERS the Music2Emo ML valence
    when the mood pass has scored this track; otherwise falls back to the
    heuristic (major key + brightness + energy)."""
    mlv = getattr(t, "ml_valence", None)
    if mlv is not None:
        return _clamp(mlv)
    mode = getattr(t, "key_mode", None)
    mode_v = 1.0 if mode == "major" else (0.0 if mode == "minor" else 0.5)
    return _clamp(0.4 * mode_v + 0.35 * brightness(t)
                  + 0.25 * t._raw_energy())


def dynamics(t):
    """0 = steady/flat, 1 = dynamic (big breakdowns/builds). Std of the
    2 Hz energy curve."""
    curve = t.row.get("energy_curve") or []
    if len(curve) < 8:
        return 0.5
    m = sum(curve) / len(curve)
    var = sum((c - m) ** 2 for c in curve) / len(curve)
    return _clamp((var ** 0.5) * 3.0)


def character_tags(t):
    """A few browsable character tags (thresholds on the RANKED values, so
    they're library-relative)."""
    tags = []
    d = getattr(t, "danceability", None)
    if d is not None:
        if d >= 0.75:
            tags.append("danceable")
        elif d <= 0.25:
            tags.append("downtempo-feel")
    v = getattr(t, "valence", None)
    if v is not None:
        if v >= 0.72:
            tags.append("uplifting")
        elif v <= 0.28:
            tags.append("dark")
    if dynamics(t) >= 0.7:
        tags.append("dynamic")
    return tags


def rank_library(tracks):
    """Store library-relative (percentile-ranked) danceability + valence on
    each track so the values span 0..1. Arousal keeps its raw ML value on
    .arousal (for display) AND gets a library-percentile .arousal_rank
    (None when unscored) so the brain's energy arc can blend it onto the
    same uniform scale as energy_proxy. Called by load_library."""
    for t in tracks:
        t.arousal = getattr(t, "ml_arousal", None)
        t.arousal_rank = None
    scored = [t for t in tracks if getattr(t, "ml_arousal", None) is not None]
    if len(scored) >= 4:
        import bisect
        a_sorted = sorted(t.ml_arousal for t in scored)
        adenom = max(len(scored) - 1, 1)
        for t in scored:
            t.arousal_rank = bisect.bisect_left(a_sorted, t.ml_arousal) / adenom
    else:
        for t in scored:
            t.arousal_rank = t.ml_arousal
    if len(tracks) < 4:
        for t in tracks:
            t.danceability = danceability_raw(t)
            t.valence = valence_raw(t)
        return
    import bisect
    d_sorted = sorted(danceability_raw(t) for t in tracks)
    v_sorted = sorted(valence_raw(t) for t in tracks)
    denom = max(len(tracks) - 1, 1)
    for t in tracks:
        t.danceability = bisect.bisect_left(d_sorted,
                                            danceability_raw(t)) / denom
        t.valence = bisect.bisect_left(v_sorted, valence_raw(t)) / denom
