"""Track enrichment from an external music database (MusicBrainz).

The scanner reads only title/artist/album from file tags and derives BPM,
key, structure, etc. from the audio. It never learns a track's GENRE,
release YEAR, era, or canonical identity. This module fills that gap by
matching each track against MusicBrainz - free, live, no API key - and
writing genre + decade tags (which immediately steer selection, flavor,
and the Set Copilot) plus a stored enrichment blob.

WHY MUSICBRAINZ and not "advanced acoustic features": the databases that
served Spotify-style danceability/valence/mood are gone - Spotify's
audio-features API returns 403 for new apps (Nov 2024) and AcousticBrainz
shut its API in 2022 (frozen dump only). MusicBrainz is the durable, open
source of real song metadata. For local acoustic descriptors the modern
path is on-device extraction (Essentia), not an API - a separate optional
pass, like the demucs vocal one.

Transport is injectable (`MusicBrainzClient(transport=...)`) so the gate
runs against canned JSON - no network. MusicBrainz asks for <=1 req/s and
a descriptive User-Agent; the default transport honors both.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

MB_BASE = "https://musicbrainz.org/ws/2/"
USER_AGENT = os.environ.get(
    "MB_USER_AGENT",
    "GL_Simple-DJ/1.0 ( https://github.com/ddehling/GL_Simple )")
_MIN_INTERVAL = 1.05            # MusicBrainz rate limit: ~1 req/s
_last_call = [0.0]


def urllib_transport(method, url, headers=None, params=None, timeout=20):
    """Rate-limited GET returning (status, parsed_json_or_None)."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json",
                      **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, {"error": str(e)}


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class MusicBrainzClient:
    def __init__(self, transport=urllib_transport):
        self.transport = transport

    def search_recording(self, artist, title, limit=5):
        q = f'artist:"{_mb_escape(artist)}" AND recording:"{_mb_escape(title)}"' \
            if artist else f'recording:"{_mb_escape(title)}"'
        status, payload = self.transport(
            "GET", MB_BASE + "recording/",
            params={"query": q, "fmt": "json", "limit": limit})
        if status != 200 or not payload:
            return []
        return payload.get("recordings", [])

    def recording_genres(self, mbid):
        """Structured genres for one recording (+ its artist)."""
        status, payload = self.transport(
            "GET", MB_BASE + f"recording/{mbid}",
            params={"inc": "genres+artist-rels+artists+tags", "fmt": "json"})
        if status != 200 or not payload:
            return []
        genres = _genre_names(payload.get("genres"))
        # Fall back to folksonomy tags when no curated genres.
        if not genres:
            genres = _genre_names(payload.get("tags"))
        # Artist-level genres broaden coverage for obscure tracks.
        for ac in payload.get("artist-credit") or []:
            art = ac.get("artist") or {}
            genres += _genre_names(art.get("genres"))
        seen, out = set(), []
        for g in genres:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out


def _mb_escape(s):
    return (s or "").replace("\\", "").replace('"', " ").strip()


def _genre_names(items):
    if not items:
        return []
    got = [(g.get("name"), g.get("count", 0)) for g in items
           if isinstance(g, dict) and g.get("name")]
    got.sort(key=lambda p: -(p[1] or 0))
    return [n for n, _ in got]


# --------------------------------------------------------------------------
# Matching + normalization
# --------------------------------------------------------------------------

def _norm(s):
    import re
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)          # drop (remix)/(feat...)
    s = re.sub(r"\b(feat|ft|featuring|remix|edit|mix|original)\b", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def best_match(recordings, artist, title, duration_s=None):
    """Pick the best recording for our track. Combines MB's own score with
    title/artist string agreement and duration proximity (the strongest
    disambiguator - two songs share a name, rarely a length). Returns
    (recording, confidence 0..1) or (None, 0)."""
    nt, na = _norm(title), _norm(artist)
    best, best_c = None, 0.0
    for r in recordings:
        rt = _norm(r.get("title"))
        ra = _norm(" ".join(ac.get("name", "") for ac in
                            r.get("artist-credit") or []))
        title_ok = nt and (nt == rt or nt in rt or rt in nt)
        artist_ok = (not na) or na in ra or ra in na
        if not (title_ok and artist_ok):
            continue
        c = 0.4 + 0.3 * (r.get("score", 0) / 100.0)
        c += 0.1 if nt == rt else 0.0
        if duration_s and r.get("length"):
            dd = abs(r["length"] / 1000.0 - duration_s)
            c += 0.2 if dd < 4 else (0.1 if dd < 10 else -0.1)
        if c > best_c:
            best, best_c = r, c
    return best, round(min(best_c, 1.0), 2)


def _year_of(recording):
    d = recording.get("first-release-date") or ""
    if not d:
        for rel in recording.get("releases") or []:
            d = rel.get("date") or d
            if d:
                break
    try:
        return int(d[:4]) if d[:4].isdigit() else None
    except (ValueError, TypeError):
        return None


def _label_of(recording):
    for rel in recording.get("releases") or []:
        for li in rel.get("label-info") or []:
            lab = (li.get("label") or {}).get("name")
            if lab:
                return lab
    return None


def enrich_track(track, mb=None, min_confidence=0.55):
    """Match one TrackInfo (or a dict with title/artist/duration_s) against
    MusicBrainz and return an enrichment blob, or None when no confident
    match. The blob carries mbid, canonical title/artist, year, decade,
    era, genres, isrc, label, confidence - and `tags` (the genre/decade
    strings the caller writes to the tags table)."""
    mb = mb or MusicBrainzClient()
    title = getattr(track, "title", None) or track.get("title")
    artist = getattr(track, "artist", None) or track.get("artist", "")
    dur = getattr(track, "duration_s", None) or track.get("duration_s")
    recs = mb.search_recording(artist, title)
    rec, conf = best_match(recs, artist, title, dur)
    if rec is None or conf < min_confidence:
        return None
    genres = mb.recording_genres(rec["id"])[:5]
    year = _year_of(rec)
    decade = f"{year // 10 * 10}s" if year else None
    era = None
    if year:
        era = ("classic" if year < 2000 else "2000s"
               if year < 2010 else "modern")
    isrc = (rec.get("isrcs") or [None])[0]
    tags = [g.lower() for g in genres]
    if decade:
        tags.append(decade)
    blob = {
        "source": "musicbrainz", "mbid": rec["id"], "confidence": conf,
        "canonical_title": rec.get("title"),
        "canonical_artist": " ".join(ac.get("name", "") for ac in
                                     rec.get("artist-credit") or []).strip(),
        "year": year, "decade": decade, "era": era,
        "genres": genres, "isrc": isrc, "label": _label_of(rec),
        "tags": tags, "enriched_at": int(time.time()),
    }
    return blob
