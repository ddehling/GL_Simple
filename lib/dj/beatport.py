"""Beatport integration for the DJ system: search the catalog, fit-score
candidates against your set/library, audition preview clips, and keep a
wishlist you buy from the browser.

WHAT THIS IS (and isn't). Beatport's PUBLIC v4 API exposes catalog search,
full track metadata (BPM, key, genre, price, ISRC), your purchased library,
and 2-minute preview audio. It exposes NO cart, hold-bin, or playlist-write
endpoints - those live only in Beatport's internal web API. So this module
is honest about the boundary: it discovers and fit-scores tracks, and the
"buy" step opens the track's page in your browser for a one-click manual
add-to-cart. No scraping, no unsanctioned automation.

AUTH. Official API-key registration is currently closed. Two paths, both
personal-use:
  - MANUAL TOKEN (zero-config, reliable): log into Beatport's API docs in a
    browser, copy the access-token JSON from the network tab, and hand it to
    `BeatportAuth.set_token_json(...)` (CLI: `dj_beatport.py login`).
  - PKCE (needs a client_id, e.g. the public docs-frontend one via
    BEATPORT_CLIENT_ID): loopback authorization-code flow. Refreshes
    automatically; the manual token can't refresh (re-paste when it lapses).

Transport is injectable (`BeatportClient(transport=...)`) so the whole thing
is testable with canned JSON - no network, no credentials. Only stdlib.
"""
import base64
import hashlib
import http.cookiejar
import json
import os
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.beatport.com/v4/"
DOCS_LOGIN_URL = "https://api.beatport.com/v4/docs/"
CLIENT_ID = os.environ.get("BEATPORT_CLIENT_ID", "")   # auto-scraped if empty
# Beatport's own OAuth post-message redirect (used by its docs frontend).
REDIRECT_URI = "https://api.beatport.com/v4/auth/o/post-message/"
_SCRIPT_SRC = re.compile(r"src=.(.*js)")
_CLIENT_ID = re.compile(r"API_CLIENT_ID: \'(.*)\'")

_NOTE_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
            "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
            "A#": 10, "BB": 10, "B": 11}


def _default_token_path():
    return os.path.join(os.path.expanduser("~"), ".gl_simple_beatport.json")


# --------------------------------------------------------------------------
# HTTP transport (injectable)
# --------------------------------------------------------------------------

def urllib_transport(method, url, headers=None, params=None, data=None,
                     timeout=30):
    """Return (status_code, parsed_json_or_None). Never raises on HTTP
    errors - returns the status so callers branch on 401 (refresh) etc."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers or {},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace") if e.fp else ""
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = {"detail": raw[:300]}
        return e.code, payload
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, {"detail": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# Auth / token store
# --------------------------------------------------------------------------

class BeatportAuth:
    """Persisted OAuth token with optional refresh. A token JSON is the
    standard `{access_token, refresh_token, expires_in, token_type,
    scope}` from the OAuth token endpoint; `obtained_at` is stamped on
    save so expiry survives restarts."""

    def __init__(self, token_path=None, client_id=None):
        self.token_path = token_path or _default_token_path()
        self.client_id = client_id or CLIENT_ID
        self.token = None
        self.load()

    def load(self):
        try:
            with open(self.token_path, encoding="utf-8") as f:
                self.token = json.load(f)
        except (OSError, ValueError):
            self.token = None
        return self.token

    def save(self):
        tmp = self.token_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.token, f)
        os.replace(tmp, self.token_path)

    def set_token_json(self, obj):
        """Accept any of: a token dict, the token JSON string, or a BARE
        access token (with or without a 'Bearer ' prefix) - whatever the
        user managed to copy out of Beatport's docs / DevTools."""
        if isinstance(obj, str):
            s = obj.strip()
            try:
                obj = json.loads(s)
            except ValueError:
                # Not JSON - treat it as a raw bearer token.
                tok = s.split(None, 1)[-1] if s.lower().startswith("bearer") \
                    else s
                obj = {"access_token": tok, "token_type": "bearer",
                       "expires_in": 3600}
        if not isinstance(obj, dict) or "access_token" not in obj:
            raise ValueError("couldn't find an access_token - paste the "
                             "token JSON or the raw Bearer token")
        obj.setdefault("obtained_at", self._now())
        self.token = obj
        self.save()

    def clear(self):
        self.token = None
        try:
            os.remove(self.token_path)
        except OSError:
            pass

    def _now(self):
        return int(time.time())

    def is_authenticated(self):
        return bool(self.token and self.token.get("access_token"))

    def expires_at(self):
        if not self.token:
            return 0
        return int(self.token.get("obtained_at", 0)) + \
            int(self.token.get("expires_in", 0))

    def is_expired(self, skew=60):
        exp = self.expires_at()
        return exp > 0 and self._now() >= exp - skew

    def bearer(self):
        if not self.is_authenticated():
            return None
        ttype = self.token.get("token_type", "Bearer").capitalize()
        return f"{ttype} {self.token['access_token']}"

    def refresh(self, transport=urllib_transport):
        """Refresh via the stored refresh_token (needs client_id). Returns
        True on success. Manual-paste tokens without a refresh_token or
        client_id can't refresh - the caller re-authenticates."""
        rt = (self.token or {}).get("refresh_token")
        if not (rt and self.client_id):
            return False
        status, payload = transport(
            "POST", API_BASE + "auth/o/token/",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": rt,
                  "client_id": self.client_id})
        if status == 200 and payload and payload.get("access_token"):
            payload.setdefault("refresh_token", rt)   # some servers omit it
            self.set_token_json(payload)
            return True
        return False


# --------------------------------------------------------------------------
# PKCE loopback login (optional; needs a client_id)
# --------------------------------------------------------------------------

def _pkce_pair():
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(40)).rstrip(
        b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def pkce_login(auth, client_id=None, scope="library",
               transport=urllib_transport, port=0, open_browser=True):
    """Loopback authorization-code + PKCE flow. Spins a one-shot localhost
    server to catch the redirect, exchanges the code, stores the token on
    `auth`. Requires a client_id whose registered redirect allows
    http://localhost. Returns True on success."""
    import http.server
    import threading
    import webbrowser

    client_id = client_id or auth.client_id or CLIENT_ID
    if not client_id:
        raise ValueError("PKCE login needs a client_id (BEATPORT_CLIENT_ID "
                         "or pass client_id=)")
    verifier, challenge = _pkce_pair()
    holder = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.urlparse(self.path).query
            holder.update(urllib.parse.parse_qs(q))
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h3>Beatport login complete - close this "
                             b"tab and return to the app.</h3>")

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    bound_port = server.server_address[1]
    redirect_uri = f"http://localhost:{bound_port}/callback"
    params = {"client_id": client_id, "response_type": "code",
              "redirect_uri": redirect_uri, "scope": scope,
              "code_challenge": challenge, "code_challenge_method": "S256"}
    auth_url = API_BASE + "auth/o/authorize/?" + urllib.parse.urlencode(params)
    if open_browser:
        webbrowser.open(auth_url)
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    t.join(timeout=300)
    server.server_close()
    code = (holder.get("code") or [None])[0]
    if not code:
        raise TimeoutError("no authorization code received (login timed out "
                           "or was denied)")
    status, payload = transport(
        "POST", API_BASE + "auth/o/token/",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "authorization_code", "code": code,
              "redirect_uri": redirect_uri, "client_id": client_id,
              "code_verifier": verifier})
    if status != 200 or not (payload and payload.get("access_token")):
        raise RuntimeError(f"token exchange failed ({status}): {payload}")
    auth.client_id = client_id
    auth.set_token_json(payload)
    return True


# --------------------------------------------------------------------------
# Username/password login (no token hunting)
#
# Faithful port of the proven beets-beatport4 flow: scrape the public
# client_id from Beatport's docs, POST your Beatport username+password to
# Beatport's OWN login endpoint (over HTTPS - the password never leaves
# Beatport, never touches us), follow the OAuth authorize redirect to get a
# code, exchange it for a token. We store ONLY the token, never the password.
# --------------------------------------------------------------------------

class BeatportSession:
    """Cookie-preserving HTTP session (urllib; no requests dependency).
    Injectable-free but factored into small methods so tests drive it."""

    def __init__(self):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [("User-Agent", "GL_Simple-DJ/1.0")]

    def get_text(self, url):
        with self.opener.open(url, timeout=30) as r:
            return r.read().decode("utf-8", "replace")

    def post_json(self, url, obj):
        data = json.dumps(obj).encode()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def get_location(self, url):
        """GET without following redirects; return the Location header."""
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar), _NoRedirect)
        opener.addheaders = self.opener.addheaders
        try:
            with opener.open(url, timeout=30) as r:
                return r.headers.get("Location"), r.read().decode(
                    "utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.headers.get("Location"), (e.read().decode("utf-8",
                                                               "replace")
                                               if e.fp else "")

    def post_form(self, url):
        req = urllib.request.Request(url, data=b"", method="POST")
        with self.opener.open(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace"))


def scrape_client_id(session):
    """Find Beatport's public API client_id via its docs page JS."""
    html = session.get_text(API_BASE + "docs/")
    for src in _SCRIPT_SRC.findall(html):
        try:
            js = session.get_text("https://api.beatport.com" + src)
        except Exception:
            continue
        m = _CLIENT_ID.findall(js)
        if m:
            return m[0]
    return None


def password_login(auth, username, password, session=None, client_id=None):
    """Log into Beatport with a username + password and store the token on
    `auth`. Returns the account dict {username, email, ...} on success.
    Raises BeatportError with Beatport's own message on failure."""
    session = session or BeatportSession()
    cid = client_id or auth.client_id or CLIENT_ID or scrape_client_id(session)
    if not cid:
        raise BeatportError("couldn't find Beatport's client_id")

    # 1. Login (password goes to Beatport's own endpoint only).
    acct = session.post_json(API_BASE + "auth/login/",
                             {"username": username, "password": password})
    if not isinstance(acct, dict) or "username" not in acct \
            or "email" not in acct:
        raise BeatportError(f"login failed: {acct}")

    # 2. Authorization code from the OAuth redirect.
    q = urllib.parse.urlencode({"response_type": "code", "client_id": cid,
                                "redirect_uri": REDIRECT_URI})
    location, body = session.get_location(API_BASE + "auth/o/authorize/?" + q)
    if not location:
        msg = re.findall(r"<p>(.*)</p>", body)
        raise BeatportError("OAuth authorize failed: "
                            + (msg[0] if msg else "no redirect"))
    code = urllib.parse.parse_qs(
        urllib.parse.urlparse(location).query).get("code")
    if not code:
        raise BeatportError(f"no authorization code in redirect: {location}")

    # 3. Exchange the code for a token.
    tq = urllib.parse.urlencode({
        "code": code[0], "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI, "client_id": cid})
    token = session.post_form(API_BASE + "auth/o/token/?" + tq)
    if not (isinstance(token, dict) and token.get("access_token")):
        raise BeatportError(f"token exchange failed: {token}")
    auth.client_id = cid
    auth.set_token_json(token)           # stores the TOKEN only, not the password
    return acct


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class BeatportError(Exception):
    pass


class BeatportClient:
    def __init__(self, auth=None, transport=urllib_transport,
                 token_path=None):
        self.auth = auth or BeatportAuth(token_path=token_path)
        self.transport = transport

    def available(self):
        return self.auth.is_authenticated()

    def _get(self, path, params=None, _retried=False):
        if not self.auth.is_authenticated():
            raise BeatportError("not authenticated - run the login flow "
                                "(dj_beatport.py login)")
        if self.auth.is_expired() and not _retried:
            self.auth.refresh(self.transport)
        status, payload = self.transport(
            "GET", API_BASE + path.lstrip("/"),
            headers={"Authorization": self.auth.bearer(),
                     "Accept": "application/json"},
            params=params)
        self.last = {"path": path, "params": params, "status": status,
                     "payload": payload}
        if os.environ.get("BEATPORT_DEBUG"):
            import json as _j
            print(f"[beatport] GET {path} params={params} -> {status}\n"
                  f"           {_j.dumps(payload)[:600]}")
        if status == 401 and not _retried:
            if self.auth.refresh(self.transport):
                return self._get(path, params, _retried=True)
            raise BeatportError("unauthorized - token expired; re-run login")
        if status != 200:
            raise BeatportError(f"GET {path} -> {status}: {payload}")
        return payload

    def search(self, query, type="tracks", per_page=25, page=1, **filters):
        """Catalog search. type: tracks|releases|artists|labels|charts.
        Extra keyword filters (bpm_low, bpm_high, genre_id, key_id, ...)
        pass through to the API. Returns the list of result dicts.

        NOTE: the API wants RANGE filters in colon syntax ("bpm":
        "110:128") and SILENTLY IGNORES bpm_low/bpm_high (verified live
        2026-07-20 - every bpm-boxed search since this integration
        shipped was actually unfiltered). Translated here so callers keep
        the readable kwargs. Empty query + genre_id is a valid paginated
        full-catalog browse."""
        lo = filters.pop("bpm_low", None)
        hi = filters.pop("bpm_high", None)
        if lo is not None or hi is not None:
            filters["bpm"] = (f"{int(lo if lo is not None else 40)}:"
                              f"{int(hi if hi is not None else 999)}")
        params = {"q": query, "type": type, "per_page": per_page,
                  "page": page, **filters}
        payload = self._get("catalog/search/", params) or {}
        # v4 returns {"tracks": [...], "releases": [...], ...} or a paged
        # {"results": [...]} - accept both.
        if type in payload:
            return payload[type]
        return payload.get("results", payload.get("data", []))

    def track(self, track_id):
        return self._get(f"catalog/tracks/{int(track_id)}/")

    def top(self, genre_id, per_page=100):
        payload = self._get(f"catalog/tracks/top/{int(genre_id)}/",
                            {"per_page": per_page}) or {}
        return payload.get("results", payload.get("data", []))

    def genres(self):
        """All Beatport genres, [{'id': ..., 'name': ...}, ...]."""
        payload = self._get("catalog/genres/", {"per_page": 200}) or {}
        return payload.get("results", payload.get("data", []))

    def my_tracks(self, page=1, per_page=100):
        """Your purchased/library tracks (scope: library)."""
        payload = self._get("my/beatport/tracks/",
                            {"page": page, "per_page": per_page}) or {}
        return payload.get("results", payload.get("data", []))

    def download_preview(self, track_json, dest_path, transport=None):
        """Fetch the 2-minute LOFI preview to dest_path. Preview URLs are
        unauthenticated CDN links. Returns dest_path."""
        url = preview_url(track_json)
        if not url:
            raise BeatportError("track has no preview/sample URL")
        req = urllib.request.Request(url, headers={"User-Agent": "GL_Simple"})
        with urllib.request.urlopen(req, timeout=60) as r, \
                open(dest_path, "wb") as f:
            f.write(r.read())
        return dest_path


# --------------------------------------------------------------------------
# Normalization: Beatport JSON -> library-row shape
# --------------------------------------------------------------------------

def _camelot_from_key(key):
    """Beatport key object -> Camelot code (e.g. '8A'). Prefers explicit
    camelot fields; falls back to parsing the note name + mode."""
    if not isinstance(key, dict):
        return ""
    num = key.get("camelot_number")
    letter = key.get("camelot_letter")
    if num and letter:
        return f"{int(num)}{str(letter).upper()}"
    name = (key.get("name") or "").strip()
    if not name:
        return ""
    # e.g. "A min", "F# maj", "Db Major", "C minor"
    from lib.dj.features import camelot_of
    toks = name.replace("♯", "#").replace("♭", "b").split()
    note = toks[0].upper() if toks else ""
    mode = "minor" if any("min" in t.lower() for t in toks) else "major"
    pc = _NOTE_PC.get(note)
    if pc is None:
        return ""
    return camelot_of(pc, mode)


def _first(*vals):
    for v in vals:
        if v:
            return v
    return None


def preview_url(track_json):
    """Best-effort preview URL from a Beatport track JSON across field
    shapes: `sample_url`, `preview.url`, or the geo-samples CDN pattern
    built from the track's `sample_uuid`/`uuid`."""
    direct = _first(track_json.get("sample_url"),
                    (track_json.get("preview") or {}).get("url")
                    if isinstance(track_json.get("preview"), dict) else None)
    if direct:
        return direct
    uuid = _first(track_json.get("sample_uuid"), track_json.get("uuid"))
    if uuid:
        return f"https://geo-samples.beatport.com/track/{uuid}.LOFI.mp3"
    return None


def track_url(track_json):
    """Public Beatport page for the track (where you add to cart / buy)."""
    slug = track_json.get("slug")
    tid = track_json.get("id")
    if tid and slug:
        return f"https://www.beatport.com/track/{slug}/{tid}"
    if tid:
        return f"https://www.beatport.com/track/-/{tid}"
    return None


def _artists(track_json):
    arts = track_json.get("artists") or []
    names = [a.get("name") for a in arts if isinstance(a, dict) and
             a.get("name")]
    return ", ".join(names)


def beatport_row(track_json):
    """Normalize a Beatport track JSON into a dict shaped like a library
    row (the subset the Brain scores on), plus Beatport-specific extras
    (price, url, preview). id is prefixed 'bp:' so it never collides with
    a local integer track id."""
    length_ms = _first(track_json.get("length_ms"),
                       track_json.get("sample_end_ms"))
    if not length_ms and track_json.get("length"):
        # "6:23" -> ms
        try:
            m, s = str(track_json["length"]).split(":")
            length_ms = (int(m) * 60 + int(s)) * 1000
        except (ValueError, AttributeError):
            length_ms = 0
    price = track_json.get("price")
    if isinstance(price, dict):
        price = _first(price.get("display"), price.get("value"))
    genre = track_json.get("genre")
    if isinstance(genre, dict):
        genre = genre.get("name")
    return {
        "id": f"bp:{track_json.get('id')}",
        "bp_id": track_json.get("id"),
        "title": _first(track_json.get("name"), track_json.get("title")) or "",
        "artist": _artists(track_json) or "",
        "bpm": float(track_json.get("bpm") or 0.0),
        "camelot": _camelot_from_key(track_json.get("key") or {}),
        "duration_s": (length_ms or 0) / 1000.0,
        "genre": genre or "",
        "price": price,
        "url": track_url(track_json),
        "preview": preview_url(track_json),
        "isrc": track_json.get("isrc"),
        "released": _first(track_json.get("new_release_date"),
                          track_json.get("publish_date")),
        "raw": track_json,
    }


# --------------------------------------------------------------------------
# Ghost TrackInfo + fit scoring against the local set/library
# --------------------------------------------------------------------------

def ghost_trackinfo(row):
    """A minimal TrackInfo built from Beatport metadata alone (bpm, key) -
    enough for tempo/key fit. No grid/sections, so bpm_conf is 0 and the
    seam-quality machinery treats it as a fade candidate until you analyze
    a preview (see deep_ghost)."""
    from lib.dj.brain import TrackInfo
    lib_row = {
        "id": row["id"], "path": row.get("url") or row["id"],
        "title": row["title"], "artist": row["artist"],
        "duration_s": row["duration_s"], "bpm": row["bpm"], "bpm_conf": 0.0,
        "downbeat_offset": 0, "downbeat_conf": 0.0, "camelot": row["camelot"],
        "beat_grid": [], "loudness_gain_db": 0.0, "kick_offset_s": 0.0,
        "phrase_beats": 0, "phrase_start_s": 0.0, "phrase_conf": 0.0,
        "mood_hist": {}, "rhythm_density": 0.0, "spectral": {},
        "axes": {}, "auto_tags": [], "content_hash": row["id"],
        "energy_curve": [],
    }
    return TrackInfo(lib_row, sections=[], loops=[], mix_points=[])


def deep_ghost(client, row, tmp_dir=None):
    """Download the preview clip and run the real analyzer on it, returning
    a fuller ghost (measured bpm/key/grid confidence from ~2 min of audio).
    Verifies Beatport's published bpm/key and unlocks real seam scoring."""
    import tempfile
    from lib.dj.brain import TrackInfo
    from lib.dj.features import analyze_samples, decode_file
    tmp_dir = tmp_dir or tempfile.gettempdir()
    dest = os.path.join(tmp_dir, f"bp_preview_{row['bp_id']}.mp3")
    client.download_preview(row["raw"], dest)
    try:
        analysis = analyze_samples(decode_file(dest), deep=False)
    finally:
        try:
            os.remove(dest)
        except OSError:
            pass
    lib_row = dict(analysis)
    lib_row.update({
        "id": row["id"], "path": row.get("url") or row["id"],
        "title": row["title"], "artist": row["artist"],
        "content_hash": row["id"],
        # Trust Beatport's published metadata over a 2-min lofi clip when
        # both exist; keep the clip's grid confidence + sections.
        "bpm": row["bpm"] or analysis.get("bpm"),
        "camelot": row["camelot"] or analysis.get("camelot"),
    })
    return TrackInfo(lib_row, sections=analysis.get("sections") or [],
                     loops=analysis.get("loops") or [],
                     mix_points=analysis.get("mix_points") or [])


def fit_vs_track(cur, cand, brain=None):
    """Compact tempo/key fit of `cand` mixing out of `cur`. Returns
    {mixable, stretch_pct, half_double, key_fit, verdict}."""
    from lib.dj.brain import Brain, camelot_compat
    if brain is None:
        brain = Brain([], _neutral_theme())
    rate, eff = brain.rate_for(cur.bpm, cand)
    key_fit = camelot_compat(cur.camelot, cand.camelot)
    if rate is None:
        return {"mixable": False, "stretch_pct": None, "half_double": None,
                "key_fit": round(key_fit, 2), "verdict": "fade only "
                "(tempo gap beyond +/-8%)"}
    import math
    pct = round((rate - 1.0) * 100.0, 1)
    hd = abs(eff - cand.bpm) > 1.0
    verdict = ("great" if abs(math.log(rate)) < math.log(1.03)
               and key_fit >= 0.9 else
               "good" if abs(math.log(rate)) < math.log(1.055)
               and key_fit >= 0.55 else "workable")
    return {"mixable": True, "stretch_pct": pct, "half_double": hd,
            "key_fit": round(key_fit, 2), "verdict": verdict}


def fit_between(a, b, ghost, brain=None):
    """CONNECTOR fit: how well `ghost` bridges the gap a -> X -> b.
    Both directions must work - X mixes out of a AND b mixes out of X.
    Returns {mixable, key_fit, verdict, in_fit, out_fit} where key_fit is
    the geometric mean of the two directional key fits and verdict is the
    WORSE of the two directions (a bridge is only as good as its weaker
    seam)."""
    from lib.dj.brain import Brain
    if brain is None:
        brain = Brain([], _neutral_theme())
    f_in = fit_vs_track(a, ghost, brain=brain)       # a -> X
    f_out = fit_vs_track(ghost, b, brain=brain)      # X -> b
    rank = {"great": 3, "good": 2, "workable": 1}
    v_in = rank.get(f_in["verdict"], 0)
    v_out = rank.get(f_out["verdict"], 0)
    worse = min((v_in, f_in["verdict"]), (v_out, f_out["verdict"]))[1]
    kf = (max(f_in["key_fit"], 0.0) * max(f_out["key_fit"], 0.0)) ** 0.5
    return {"mixable": f_in["mixable"] and f_out["mixable"],
            "key_fit": round(kf, 2),
            "verdict": worse if (f_in["mixable"] and f_out["mixable"])
            else "fade only (bridges one side, not both)",
            "in_fit": f_in, "out_fit": f_out,
            # fit_vs_track's shape so the Discover fit column renders it.
            "stretch_pct": f_in.get("stretch_pct"),
            "half_double": (f_in.get("half_double")
                            or f_out.get("half_double"))}


def connector_bpm_window(a, b, stretch=1.075):
    """BPM range a connector must sit in to be tempo-reachable from BOTH
    sides ([lo, hi], or a widened midpoint window when the two tracks'
    reachable ranges don't overlap - the two-step bridge mentality)."""
    lo = max(a.bpm / stretch, b.bpm / stretch)
    hi = min(a.bpm * stretch, b.bpm * stretch)
    if lo <= hi:
        return lo, hi
    mid = (a.bpm * b.bpm) ** 0.5
    return mid / stretch, mid * stretch


def fit_vs_library(library, cand, top=5):
    """Does this candidate have good neighbours in the library already? -
    the 'is it worth buying' question. Returns counts + best matches."""
    from lib.dj.brain import Brain, camelot_compat
    brain = Brain([], _neutral_theme())
    mixable = []
    for t in library:
        rate, _ = brain.rate_for(t.bpm, cand)      # cand mixes out of t
        r2, _ = brain.rate_for(cand.bpm, t)        # ...and into t
        if rate is None and r2 is None:
            continue
        kf = camelot_compat(t.camelot, cand.camelot)
        best_rate = min((abs(r - 1.0) for r in (rate, r2) if r is not None),
                        default=1.0)
        mixable.append((kf - best_rate, kf, t))
    mixable.sort(key=lambda x: -x[0])
    return {"mixable_neighbours": len(mixable),
            "library_size": len(library),
            "best": [{"title": t.title, "artist": t.artist,
                      "bpm": round(t.bpm, 1), "key": t.camelot,
                      "key_fit": round(kf, 2)}
                     for _, kf, t in mixable[:top]]}


def _neutral_theme():
    from lib.dj.themes import Theme
    return Theme("discover", bpm_range=(60.0, 200.0), energy_base=0.5,
                 energy_span=0.5, mood_weights={})


# --------------------------------------------------------------------------
# Wishlist (local; buy in the browser)
# --------------------------------------------------------------------------

class Wishlist:
    """A local buy-list. Beatport has no cart API, so this is the honest
    bridge: collect candidates here, then open each on Beatport to add to
    your real cart and purchase. Stored beside the library so it travels
    with it."""

    def __init__(self, music_root):
        self.path = os.path.join(os.path.abspath(music_root),
                                 "beatport_wishlist.json")
        self.items = []
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self.items = json.load(f)
        except (OSError, ValueError):
            self.items = []
        return self.items

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=1)
        os.replace(tmp, self.path)

    def add(self, row):
        """Add a normalized beatport_row (idempotent by bp_id)."""
        if any(it.get("bp_id") == row.get("bp_id") for it in self.items):
            return False
        self.items.append({k: row[k] for k in
                          ("bp_id", "title", "artist", "bpm", "camelot",
                           "genre", "price", "url", "preview")
                          if k in row})
        self.save()
        return True

    def remove(self, bp_id):
        n = len(self.items)
        self.items = [it for it in self.items if it.get("bp_id") != bp_id]
        if len(self.items) != n:
            self.save()
        return n - len(self.items)

    def remove_many(self, bp_ids):
        """Drop several items in ONE pass + ONE save - removing a large
        selection track-by-track would rewrite the file per item."""
        drop = set(bp_ids)
        if not drop:
            return 0
        n = len(self.items)
        self.items = [it for it in self.items if it.get("bp_id") not in drop]
        if len(self.items) != n:
            self.save()
        return n - len(self.items)

    def open_in_browser(self, bp_id=None):
        """Open one item's Beatport page (or all) for manual add-to-cart."""
        import webbrowser
        opened = 0
        for it in self.items:
            if bp_id is not None and it.get("bp_id") != bp_id:
                continue
            if it.get("url"):
                webbrowser.open(it["url"])
                opened += 1
        return opened
