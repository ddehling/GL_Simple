"""Song-identity helpers shared by every dedup path in the planner.

"Is this the same song?" gets asked in three places - set dedup, the
suggestion panel's twin filter, the Discover tab's already-owned check -
and each had grown its own answer. One vocabulary lives here:

  flat_title   normalization that survives re-rips with mangled tags
  track_sig    (flat title, ~8s duration bucket, content hash)
  dup_keys     identity keys for union-find duplicate detection
  artist_tokens loose artist matching across metadata sources
"""
import re


def flat_title(title):
    """Lowercased title with everything non-alphanumeric stripped, so
    '02_Alex - Youth (feat_ ...)' and 'Alex - Youth (feat. ...)' meet."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def track_sig(t):
    """(flattened title, ~8s duration bucket, content hash) for a
    TrackInfo. The bucket keeps a radio edit from colliding with the
    extended mix of the same name."""
    row = getattr(t, "row", None) or {}
    return (flat_title(getattr(t, "title", "")),
            int((getattr(t, "duration_s", 0.0) or 0.0) // 8),
            (row.get("content_hash") or "").strip())


def dup_keys(t):
    """Identity keys for duplicate detection. A byte-identical copy in
    another directory shares the scan's content hash; a re-rip or
    re-encode falls back to normalized title/artist plus a coarse length
    bucket (so a radio edit never collides with the extended mix of the
    same name)."""
    keys = []
    _, bucket, h = track_sig(t)
    if h:
        keys.append(("h", h))
    ti = (getattr(t, "title", "") or "").strip().lower()
    ar = (getattr(t, "artist", "") or "").strip().lower()
    if ti:
        keys.append(("m", ti, ar, bucket))
    return keys


def track_genre(t):
    """Best single genre label for a track: MusicBrainz genre (from the
    Enrich pass) first, else the embedded file genre tag."""
    g = getattr(t, "genres", None)
    if g:
        return g[0]
    return (getattr(t, "file_genre", "") or "").split(",")[0] \
        .split("/")[0].strip()


def artist_tokens(artist):
    """Meaningful artist-name words for loose cross-source matching
    ('Artist A, Artist B' vs 'Artist A & Someone')."""
    return {w for w in (artist or "").lower().replace(",", " ").split()
            if len(w) > 2}
