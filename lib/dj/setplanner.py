"""THE NIGHT AS A PLAN: the set planner (docs/DJ_SETS.md, phase 1). Pure planning: no audio, no Director, no deck.

A THEME (a crate, leans, a story of chapters, variety, a length) becomes an ordered list of records for the whole
night, each placed where its chapter asks for it (the energy and tempo at that moment), every seam checked the way
the record player will check it, so the player never has to pick a planned record again:

  the pair table   records in tempo order, each meeting only its tempo neighbours (MAX_TEMPO_GAP, the stretcher's
                   limit); THE KEY VETO, pass/fail at KEY_BLEND through RecordPlayer.key_fit (decision 2 of
                   DJ_SETS.md: until the keys-only rescan keys get no score, no leans and no keys mode - the planner
                   only drops the pairs the player would refuse, so it reads whatever keys the DB holds tonight);
                   VOCAL ROOM from the stems' vocal envelope with the player's own VOX_QUIET / VOX_FLOOR meaning
                   (never the library's ML vocal axis, which read 0.0 on records that sang); the energy step; the
                   genres two records share
  the clock        THE PLAYER'S CLOCK (DJ_SETS.md section 3): a record plays from its cue to where the player
                   starts the next one - its strict placement (the first phrase at or after the minimum play and
                   LAST_FROM of the record, steady, vocals apart, best place wins) or its breakdown swap - computed
                   with the player's own pure functions; the prototype's "landmark to the end minus 30 s" made a
                   "90-minute" night play short and every chapter arrive early
  the search       the prototype's beam search (sizes configurable): candidate nights grow side by side, each
                   judged by its whole route so far; the result's slots carry 2-3 pre-checked alternates that fit
                   both neighbours (decision 1: a planned record the rules cannot place plays its alternate)

THE CUT IS THE USER'S: nothing here plans a cut. A re-plan keeps a fixed prefix (the record playing and slot +1,
which the player has already committed) and changes only what follows; a seam the planner offers is always one the
player's strict placement accepts. The play-to-the-end and relaxed clocks below exist only to TIME seams the user
made (UP NEXT) or that were already committed.

It runs in a SUBPROCESS so a search never holds the show's GIL (`PlannerProcess`, `python -m lib.dj.setplanner
--serve`): the show ships a JSON request with ids and gets JSON slots back; the library, the per-record facts and
the pair table per crate are built once in the child and cached there. `Planner(music_root).handle(request)` is
the same thing in-process (the headless tests).

    python -m lib.dj.setplanner --plan request.json --music D:/Devel/music      one request, the answer on stdout
    python -m lib.dj.setplanner --warm-vox --music D:/Devel/music               measure the uncertain vocal floors
"""
import json
import math
import os
import sys
import threading
import time

# Nothing heavy at the top (the boot-time rule): numpy, the library and the record player load in the functions.

PLAN_AT_S = 60.0          # the player decides a blend when the next record is decoded, about a minute after a record
                          # opens (its geometry is measured "now"; only the k_now bound and the grid segment of a
                          # two-tempo record depend on it, and neither moves a blend from its minimum play)
DEFAULT_PLAY_MIN_S = 210.0  # the Director's PACE "normal" (director.PLAY_MIN_S); a request passes the dial's value
BEAM = 60                 # the builder's search sizes (theme_builder plan(): 60 beams x 12); the request may change them
EXPAND = 12
MAX_STEPS = 400           # one record per step: 8 h is about 100 records
DONE_MAX = 400            # this many finished nights is enough to choose from (the builder's stop)
MAX_DURATION_S = 900.0    # a 15-minute file is a DJ mix or a live set, not a record (the prototype's crate rule)
LENGTH_MIN = (10.0, 480.0)  # LENGTH is capped at 8 h (DJ_SETS.md defaults)
SPACING = 6               # records between two by the same artist (the builder's default)
ALTS = 3                  # alternates per slot
ALTS_FEW_SHARE = 0.15     # more than this share of planned slots with fewer than 2 alternates: the plan says the crate is thin
WARN_MIN_OFF = 2          # a chapter warning needs at least this many records off what it asks for...
WARN_MIN_SHARE = 1.0 / 3  # ...and at least this share of the chapter's planned records (red: more than half)
SKIP_WHY = "skipped for your record"

# THE BUILDER'S SIX THEMES (2026-10-28, theme_builder.template.html PRESETS), as data: a crate (which records may
# play), leans (weighted words that rank what is left), chapters (a share of the night, an energy course, a tempo
# course as crate percentiles, chapter leans, optional mood / vocals / hypnotic / hardness courses) and the keys mode
# (stored, not scored until the rescan).


def _ch(name, share, energy, tempo, leans=None, dims=None):
    return {"name": name, "share": share, "energy": list(energy), "tempo": list(tempo), "leans": list(leans or []), "dims": dict(dims or {})}


def _crate(genres_out=(), bpm=(80, 135), energy=(0, 100), vocals="any", tags_out=()):
    return {"source": None, "genresIn": [], "genresOut": list(genres_out), "moodsIn": [], "tagsIn": [], "tagsOut": list(tags_out),
            "bpm": list(bpm), "energy": list(energy), "vocals": vocals}


PRESETS = [
    {"name": "Desert Night", "blurb": "Warm dusk, a percussive caravan, a dark sandstorm at the peak, and open sky to close.",
     "crate": _crate(["rock", "pop"], (80, 135)),
     "leans": {"desert": 2, "organic house": 1.2, "afro house": 1, "downtempo": 0.7, "hypnotic": 0.6, "percussive": 0.6, "travel": 0.5, "nature": 0.5, "warm": 0.4},
     "chapters": [_ch("Dusk", 0.2, (0.25, 0.42), (0.15, 0.35), ["downtempo", "calm", "warm", "nature"], {"vocals": [0, 0.1]}),
                  _ch("Caravan", 0.3, (0.45, 0.62), (0.4, 0.6), ["organic house", "afro house", "percussive", "hypnotic"]),
                  _ch("Sandstorm", 0.3, (0.66, 0.88), (0.6, 0.85), ["melodic house & techno", "dark", "driving"], {"mood": [0.35, 0.2], "hardness": [0.5, 0.8]}),
                  _ch("Starfield", 0.2, (0.6, 0.3), (0.55, 0.3), ["space", "dream", "deep"], {"mood": [0.45, 0.65]})], "keys": "close"},
    {"name": "Deep Space", "blurb": "Lift-off on melodic electronica, a long hypnotic orbit, a dark nebula at full power, a slow re-entry.",
     "crate": _crate(["rock", "pop", "disco"], (90, 135)),
     "leans": {"space": 1.2, "dream": 0.8, "melodic house & techno": 1, "progressive house": 0.8, "electronica": 0.6, "deep": 0.5, "hypnotic": 0.5},
     "chapters": [_ch("Launch", 0.2, (0.35, 0.55), (0.3, 0.45), ["uplifting", "melodic", "electronica"]),
                  _ch("Orbit", 0.3, (0.55, 0.66), (0.45, 0.55), ["hypnotic", "steady", "progressive house"], {"hypnotic": [0.7, 0.85], "vocals": [0, 0.1]}),
                  _ch("Nebula", 0.3, (0.7, 0.9), (0.6, 0.8), ["dark", "driving", "melodic house & techno"], {"mood": [0.3, 0.2]}),
                  _ch("Re-entry", 0.2, (0.6, 0.3), (0.5, 0.3), ["soundscape", "calm", "dream"])], "keys": "wander"},
    {"name": "Golden Hour", "blurb": "Deep house aperitivo, a nu-disco rooftop, an indie-dance sunset peak, and a warm afterglow.",
     "crate": _crate(["new age", "soundtracks"], (100, 130)),
     "leans": {"nu disco": 1.2, "indie dance": 1, "disco": 1, "house": 0.7, "deep house": 0.6, "summer": 0.8, "happy": 0.7, "groovy": 0.8, "upbeat": 0.6},
     "chapters": [_ch("Aperitivo", 0.25, (0.35, 0.5), (0.3, 0.45), ["deep house", "sexy", "cool"]),
                  _ch("Rooftop", 0.3, (0.5, 0.7), (0.45, 0.6), ["nu disco", "groovy", "summer"], {"mood": [0.6, 0.75]}),
                  _ch("Sunset peak", 0.3, (0.72, 0.9), (0.6, 0.8), ["indie dance", "party", "energetic"], {"mood": [0.7, 0.8], "vocals": [0.3, 0.4]}),
                  _ch("Afterglow", 0.15, (0.6, 0.45), (0.5, 0.4), ["love", "romantic", "happy"])], "keys": "lift"},
    {"name": "Night Shift", "blurb": "Down into the dark: a sparse descent, a hypnotic tunnel, the machine at full drive, then the surface.",
     "crate": _crate(["pop", "disco", "new age"], (110, 135), (20, 100), "some"),
     "leans": {"melodic house & techno": 1, "techno (peak time": 1, "dark": 1.2, "hypnotic": 1, "minor": 0.5, "driving": 0.6},
     "chapters": [_ch("Descent", 0.2, (0.4, 0.55), (0.35, 0.5), ["deep", "sparse", "instrumental"], {"mood": [0.3, 0.2]}),
                  _ch("Tunnel", 0.35, (0.55, 0.7), (0.5, 0.6), ["hypnotic", "steady"], {"hypnotic": [0.75, 0.9]}),
                  _ch("Machine", 0.3, (0.75, 0.95), (0.65, 0.85), ["driving", "hard", "bass-heavy"], {"hardness": [0.6, 0.9]}),
                  _ch("Surface", 0.15, (0.65, 0.45), (0.55, 0.45), ["melodic", "emotional"], {"mood": [0.35, 0.55]})], "keys": "close"},
    {"name": "Prog Odyssey", "blurb": "The classic journey: a melodic warm-up, a long climb, an epic summit, and an emotional way down.",
     "crate": _crate(["rock", "pop"], (110, 140)),
     "leans": {"progressive house": 1.2, "trance": 1, "psy-trance": 0.8, "melodic house & techno": 0.7, "uplifting": 0.8, "epic": 0.8, "emotional": 0.6},
     "chapters": [_ch("Warm-up", 0.25, (0.35, 0.55), (0.3, 0.45), ["melodic", "deep", "progressive house"]),
                  _ch("Climb", 0.3, (0.55, 0.75), (0.45, 0.65), ["driving", "uplifting", "dynamic"]),
                  _ch("Summit", 0.25, (0.8, 0.95), (0.7, 0.85), ["epic", "peaky", "trance", "powerful"], {"mood": [0.6, 0.75]}),
                  _ch("The way down", 0.2, (0.7, 0.45), (0.6, 0.4), ["emotional", "hopeful", "inspiring"])], "keys": "lift"},
    {"name": "Slow Morning", "blurb": "No peak on purpose: mist and soundscapes, warm organic grooves over coffee, then the sun comes up.",
     "crate": _crate(["psy-trance", "techno (peak time"], (70, 118), (0, 70), "any", ["hard"]),
     "leans": {"downtempo": 1.2, "organic house": 0.8, "new age": 0.8, "electronica": 0.6, "calm": 0.8, "soft": 0.6, "nature": 0.6, "relaxing": 0.6},
     "chapters": [_ch("Mist", 0.3, (0.12, 0.25), (0.1, 0.25), ["soundscape", "meditative", "new age"], {"vocals": [0, 0.05]}),
                  _ch("Coffee", 0.35, (0.3, 0.45), (0.3, 0.45), ["warm", "groovy", "organic house"]),
                  _ch("Sun", 0.35, (0.45, 0.58), (0.45, 0.6), ["uplifting", "happy", "organic house"], {"mood": [0.55, 0.75]})], "keys": "wander"},
]
CH_DIMS = ("mood", "vocals", "hypnotic", "hardness")
# the vocal rule of the crate, on the share of a record that sings (the builder's VOCALS_OK)
VOCALS_OK = {"any": lambda s: True, "none": lambda s: s <= 0.05, "some": lambda s: s <= 0.35, "lots": lambda s: s >= 0.25}
# THE KEYS MODE (2026-10-29, after the key v2 rescan: 0.825 exact against Beatport): theme["keys_mode"], a LIGHT factor
# on each seam's score in the search - never a veto of its own, and the player's veto (KEY_BLEND) stands in every mode.
# "veto" is today's planner exactly (no factor at all) and the only mode of a theme that does not set keys_mode: the
# presets' stored "keys" words are not read, so a preset plans as it did until you pick a mode.
#   close   Camelot distance <= 1 (same key, relative, a neighbour) over the diagonal the veto still lets through
#   wander  no preference beyond the veto
#   lift    at a chapter whose energy rises, the +1 move on the wheel (the energy boost); +2 fits 0.55, under the veto,
#           so the player never blends it and the planner never offers it
KEYS_MODES = ("veto", "close", "wander", "lift")
KEYS_WORDS = {"veto": "veto only (pass/fail at the player's key fit)", "close": "close (distance 1 or less preferred)",
              "wander": "wander (no preference beyond the veto)", "lift": "lift (+1 moves preferred where a chapter rises)"}
KEY_OFF = {"close": 0.82, "lift": 0.84, "lift_same": 0.92, "unknown": 0.92}   # the factor where a seam misses the mode's wish


def preset_names():
    return [p["name"] for p in PRESETS]


def get_preset(name):
    for p in PRESETS:
        if p["name"].lower() == str(name or "").strip().lower():
            return json.loads(json.dumps(p))
    return None


# ---- small pure helpers --------------------------------------------------------------------------------------------
def _squash(x, k):
    return 1.0 - math.exp(-max(x, 0.0) / k)


def _hash01(i, seed):
    """The builder's night noise (hash01), the same 32-bit arithmetic: a new seed reshuffles the near-ties."""
    m = 0xFFFFFFFF
    h = ((int(i) * 2654435761) & m) ^ ((int(seed) * 40503) & m)
    h = ((h ^ (h >> 16)) * 2246822507) & m
    h = ((h ^ (h >> 13)) * 3266489909) & m
    h ^= h >> 16
    return h / 4294967296.0


def _artist_key(artist):
    return (artist or "").split(",")[0].split(" feat")[0].strip().lower()


def _jround(x):
    """JavaScript's Math.round (half up), which the builder's percentile used; Python's round() is the banker's."""
    return int(math.floor(x + 0.5))


class _NoSubmix:
    """What the record player's pure geometry reads from a submix: nothing is playing (the planner has no audio)."""
    telemetry = {}
    clock = 0

    def post(self, *_a, **_k):
        pass

    def post_many(self, *_a, **_k):
        pass


# ---- the records -----------------------------------------------------------------------------------------------------
class _Fact:
    """One record, as the planner sees it (independent of the minimum play)."""
    __slots__ = ("id", "t", "title", "artist", "ak", "root", "bpm", "e", "words", "genres", "sung", "vr", "hyp", "hard",
                 "dims", "pl", "dur", "vox", "vox_exact", "rec")


class _Geo:
    """A record's geometry for one minimum play (the landmark needs play_min + RUNWAY_SLACK_S of body)."""
    __slots__ = ("why", "lm", "bar_b", "first", "cues", "cls", "build", "opens", "cue_in")


class _Open:
    """A record that opened at `cue`: the player's bars for planning its way out (RecordPlayer._geometry)."""
    __slots__ = ("cue", "t_now", "bar", "k_now", "open_bar", "k_min", "k_end", "entries", "bd", "memo", "kt", "solo")


_CUE_KEYS = None
_RP_MOD = None


def _rpm():
    """lib.dj.recordplay, imported on first use (not at the top: it pulls the submix and numpy) and kept, so the
    search's inner loop does not run an import statement per pair."""
    global _RP_MOD
    if _RP_MOD is None:
        import lib.dj.recordplay as m
        _RP_MOD = m
    return _RP_MOD


def _cue_keys():
    """(size, quiet bars) for every cue the strict placement may ask of the next record: the lead alone (L) or the
    whole overlap (L + the tail, 4 bars or 8 in an outro)."""
    global _CUE_KEYS
    if _CUE_KEYS is None:
        from lib.dj.recordplay import SIZES, TAIL_BARS, TAIL_OUTRO_BARS
        keys = []
        for nm in ("standard", "short"):
            L = SIZES[nm][0]
            for qb in (L, L + TAIL_BARS, L + TAIL_OUTRO_BARS):
                keys.append((nm, qb))
        _CUE_KEYS = tuple(keys)
    return _CUE_KEYS


VOX_TOP_FILE = "vox_top.json"   # .stems/vox_top.json: the vocal stem's loud level where the envelope cannot tell it


def _vox_cache_path(music_root):
    return os.path.join(music_root, ".stems", VOX_TOP_FILE)


def _load_vox_cache(music_root):
    try:
        with open(_vox_cache_path(music_root), encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _vocal_file(music_root, tid):
    try:
        from lib.dj.stems import stem_paths
        p = stem_paths(music_root, tid)
        return (p or {}).get("vocals")
    except Exception:
        return None


def _file_sig(path):
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return None


def _vox_measure(music_root, t, cache):
    """THE VOCAL ENVELOPE THE PLAYER USES (recordplay.vocal_envelope: the vocal stem's half-second RMS over its 95th
    percentile; silent everywhere when that loud level is under VOX_FLOOR of the mix's), read cheaply from the stem
    envelopes on disk (.stems/<id>/env.npz, the same half-second RMS over the same percentile) instead of decoding a
    stem per record. Returns (envelope float32 or None, exact).

    The floor test needs the stem's loud level in absolute terms, which env.npz does not store. Its body RMS gives an
    UPPER bound on it (the clipped half-seconds are at least the loud level), and the stems' levels give the mix's:
    under the floor at the bound means silent for sure. Otherwise the level measured by --warm-vox (vox_top.json,
    the player's own arithmetic on the decoded stem) decides; without it the envelope stands as it is - stricter than
    the player (it may say "sings" where the player hears bleed), never looser. Measured 2026-10-28 on 36 records
    against vocal_envelope: every singing / quiet half-second agreed except three sparse-vocal records the floor
    silences (Sequences, Abracadabra, Dunes) - the case the cache is for. No stems: None ("nothing is known, nothing
    is refused", the player's _vox_quiet)."""
    import numpy as np
    from lib.dj.recordplay import VOX_FLOOR, VOX_QUIET, RecordPlayer
    if not getattr(t, "has_stems", False):
        return None, True
    p = os.path.join(music_root, ".stems", str(t.id), "env.npz")
    try:
        z = np.load(p)
        files = set(z.files)
    except Exception:
        return None, True
    if "vocals" not in files:
        return None, True               # (17 records: their vocal stem decodes empty; the player's envelope is None too)
    v = np.asarray(z["vocals"], dtype=np.float32)
    if v.size < 2 or not bool(np.any(v >= VOX_QUIET)):
        return v, True                  # quiet everywhere: the floor cannot change that
    try:
        rms = [float(x) for x in z["rms"]]
    except Exception:
        rms = []
    body = RecordPlayer.body_start(t)
    pw, top_v = 0.0, None
    for i, nm in enumerate(("drums", "bass", "other", "vocals")):
        if nm not in files or i >= len(rms):
            continue
        e = np.asarray(z[nm], dtype=np.float32)
        n = len(e)
        if n < 2:
            continue
        k0 = int(body * 2)
        k1 = min(n, k0 + 240)
        if k1 - k0 < 2:
            k0, k1 = 0, min(n, 240)          # (stemload.stem_levels' body window)
        m = float(np.sqrt(np.mean(e[k0:k1] ** 2))) if k1 > k0 else 0.0
        top = rms[i] / m if m > 1e-9 else None
        if top is None:
            continue
        if nm == "vocals":
            top_v = top
        a = (e * top)[n // 4:3 * n // 4]       # (recordplay.mix_level: the mix over its middle half)
        if a.size:
            pw += float(np.mean(a * a))
    mix = math.sqrt(pw)
    if mix <= 0.0:
        return v, False
    if top_v is not None and top_v < VOX_FLOOR * mix:
        return np.zeros_like(v), True
    c = cache.get(str(t.id)) if cache else None
    if isinstance(c, dict) and c.get("sig") and c.get("sig") == _file_sig(_vocal_file(music_root, t.id) or ""):
        if float(c.get("top", 0.0)) < VOX_FLOOR * mix:
            return np.zeros_like(v), True
        return v, True
    return v, False


def _measure_top(music_root, tid):
    """The vocal stem's loud level exactly as recordplay.vocal_envelope measures it (decoded in this process)."""
    import numpy as np
    from lib.dj.stemload import decode_mix
    from lib.dj.submix import RATE
    path = _vocal_file(music_root, tid)
    if not path:
        return None, None
    v = decode_mix(path)
    hop = RATE // 2
    n = len(v) // hop
    if n < 2:
        return None, _file_sig(path)
    e = np.empty(n, dtype=np.float32)
    for j in range(0, n, 240):
        j1 = min(n, j + 240)
        blk = np.asarray(v[j * hop:j1 * hop], dtype=np.float32).reshape(j1 - j, hop, -1)
        e[j:j1] = np.sqrt((blk * blk).mean(axis=(1, 2)))
    return float(np.percentile(e, 95)), _file_sig(path)


def warm_vox(music_root, ids=None, log=print, stop=None):
    """Measure the vocal loud level of every record whose floor test env.npz cannot decide, into .stems/vox_top.json
    (about a second per record, once per machine; the file travels with the music folder and is re-measured when a
    stem changes). Returns how many were measured."""
    os.environ.setdefault("DJ_DECODE_INPROC", "1")       # (this runs in the planner's own process: no helper pool)
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    db = LibraryDB(music_root)
    lib = [t for t in B.load_library(db) if not t.excluded and (ids is None or t.id in ids)]
    cache = _load_vox_cache(music_root)
    todo = []
    for t in lib:
        _v, exact = _vox_measure(music_root, t, cache)
        if not exact:
            todo.append(t)
    if log:
        log(f"[setplanner] vocal floors: {len(todo)} of {len(lib)} records need the decoded stem")
    done = 0
    t0 = time.time()
    for t in todo:
        if stop is not None and stop.is_set():
            break
        try:
            top, sig = _measure_top(music_root, t.id)
        except Exception as e:  # noqa: BLE001
            if log:
                log(f"[setplanner] {t.id} {t.title[:30]}: {type(e).__name__}: {e}")
            continue
        if top is None or sig is None:
            continue
        cache[str(t.id)] = {"top": round(top, 6), "sig": sig}
        done += 1
        if done % 25 == 0 or done == len(todo):
            _save_vox_cache(music_root, cache)
            if log:
                log(f"[setplanner]   {done}/{len(todo)} ({time.time() - t0:.0f} s)")
    _save_vox_cache(music_root, cache)
    try:
        db.close()
    except Exception:
        pass
    return done


def _save_vox_cache(music_root, cache):
    path = _vox_cache_path(music_root)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError:
        pass


# ---- the planner -------------------------------------------------------------------------------------------------------
class Planner:
    """The whole planner, in-process. `handle(request) -> response` (both plain JSON-able dicts); see the module's
    request / response shapes in `handle`. Not thread-safe: one request at a time (the subprocess serves them in
    order)."""

    def __init__(self, music_root, db=None, library=None, log=None):
        self.music_root = music_root
        self._db = db
        self._library = library
        self.log = log
        self.facts = None
        self.plays = {}
        self.setlists = {}
        self._gen = 0
        self._geo = {}            # play_min -> {id: _Geo}
        self._tables = {}         # (crate signature, play_min, gen) -> _Table (a few kept)
        self._rps = {}
        # (id, cue rounded to the ms) -> where that record's phrases count from, for the request's fixed records a cut
        # landed past a silent start (the player's Rec.phrase_t; the Director sends it as fixed[].phrase_s, 2026-10-29)
        self._phr = {}
        self.load_s = 0.0
        self.lib_sig = None

    # -- loading ---------------------------------------------------------------------------------------------------------
    def _load(self, reload=False):
        if self.facts is not None and not reload:
            return
        t0 = time.time()
        from lib.dj import brain as B
        from lib.dj.recordplay import Rec
        db = self._db
        if db is None:
            from lib.dj.db import LibraryDB
            db = self._db = LibraryDB(self.music_root)
        lib = self._library if (self._library is not None and not reload) else B.load_library(db)
        lib = [t for t in lib if t.bpm and t.grid and not getattr(t, "excluded", False)]
        try:
            self.plays = db.play_counts(30.0)
        except Exception:
            self.plays = {}
        self.setlists = {}
        try:
            for sid, name in db.conn.execute("SELECT id, name FROM setlists").fetchall():
                ids = [int(r[0]) for r in db.conn.execute("SELECT track_id FROM setlist_entries WHERE setlist_id = ? ORDER BY position", (sid,)).fetchall()]
                self.setlists[str(name)] = ids
        except Exception:
            pass
        from lib.dj.recordplay import VOX_QUIET
        cache = _load_vox_cache(self.music_root)
        facts = {}
        for t in lib:
            f = _Fact()
            f.id, f.t = int(t.id), t
            f.title, f.artist = t.title or "", t.artist or ""
            f.ak = _artist_key(t.artist)
            try:
                f.root = B._title_root(t.title)
            except Exception:
                f.root = (t.title or "").lower()
            f.bpm = float(t.bpm)
            f.e = float(t.energy_proxy())
            f.genres = set(t.genre_set)
            f.words = set(x.lower() for x in t.all_tags) | f.genres | set(m.lower() for m in (t.ml_moods or []))
            f.dur = float(t.duration_s or 0.0)
            try:
                f.vox, f.vox_exact = _vox_measure(self.music_root, t, cache)
            except Exception:
                f.vox, f.vox_exact = None, True
            f.sung = round(float((f.vox >= VOX_QUIET).mean()), 3) if f.vox is not None and len(f.vox) else 0.0
            ar = t.axes_rank or {}
            f.hyp = None if ar.get("hypnotic") is None else float(ar["hypnotic"])
            f.hard = None if ar.get("hardness") is None else float(ar["hardness"])
            f.pl = int(self.plays.get(t.id, 0))
            f.vr = None
            r = Rec("p", t)
            r.vox = f.vox
            f.rec = r
            facts[f.id] = f
        # mood (valence) as a rank within the records, like energy (the builder's r.vr)
        vals = sorted((f for f in facts.values() if f.t.ml_valence is not None), key=lambda f: (float(f.t.ml_valence), f.id))
        for i, f in enumerate(vals):
            f.vr = i / max(1, len(vals) - 1)
        for f in facts.values():
            f.dims = {"mood": f.vr, "vocals": f.sung, "hypnotic": f.hyp, "hardness": f.hard}   # the chapter courses' measures
        self.facts = facts
        # what the plans depend on, in one number: a keys rescan or a tempo fix changes it (the Director sends reload)
        import zlib
        self.lib_sig = zlib.crc32(json.dumps(sorted((f.id, f.t.camelot or "", round(f.bpm, 3), round(f.dur, 1)) for f in facts.values())).encode())
        self._gen += 1
        self._geo, self._tables = {}, {}
        self.load_s = time.time() - t0
        if self.log:
            self.log(f"[setplanner] {len(facts)} records loaded in {self.load_s:.1f} s "
                     f"({sum(1 for f in facts.values() if not f.vox_exact)} vocal floors approximate)")

    def _rp(self, play_min):
        """The record player's pure geometry for one minimum play: an instance without its brain or submix (the
        methods used here read only play_min_s, the track and a Rec's vox - landmark, bar_s, _cue_for_lead,
        _steady_span, _section_kind, _bar_index, _bar_time, _first_down, fits_as_record, landmark_has_build)."""
        rp = self._rps.get(play_min)
        if rp is None:
            from lib.dj.recordplay import RecordPlayer
            rp = RecordPlayer.__new__(RecordPlayer)
            rp.play_min_s = float(play_min)
            rp.songs, rp.record, rp.submix = {}, None, _NoSubmix()
            self._rps[play_min] = rp
        return rp

    def _geo_of(self, pm, f):
        d = self._geo.setdefault(pm, {})
        g = d.get(f.id)
        if g is not None:
            return g
        rp = self._rp(pm)
        t = f.t
        g = _Geo()
        g.why = rp.fits_as_record(t)
        g.lm = float(rp.landmark(t))
        g.bar_b = float(rp.bar_s(t, g.lm))
        g.first = max(0.0, float(t.nearest_downbeat(0.0)))
        g.cues = {}
        from lib.dj.recordplay import SIZES
        for nm, qb in _cue_keys():
            c, on = rp._cue_for_lead(f.rec, SIZES[nm][0], g.bar_b, quiet_bars=qb)
            g.cues[(nm, qb)] = (None if c is None else float(c), bool(on))
        g.cls = tuple((g.cues[k][0] is not None, g.cues[k][1]) for k in _cue_keys())
        g.build = bool(rp.landmark_has_build(t))
        g.opens = {}
        # where it most likely starts when it comes in under another record: the cue the standard blend asks for
        g.cue_in = g.lm
        for k in (("standard", 20), ("standard", 16), ("short", 12), ("short", 8)):
            if g.cues.get(k, (None,))[0] is not None:
                g.cue_in = g.cues[k][0]
                break
        d[f.id] = g
        return g

    def _open_of(self, pm, f, g, cue):
        """RecordPlayer._geometry at the moment the plan is made, and every place _find_blend_strict would look at
        (lib/dj/recordplay.py ~738: phrase boundaries from the first at/after max(minimum play, LAST_FROM of the
        record), each size that fits the record's room, kept only where the record is steady), plus the breakdown
        _plan_breakdown would use (~949). Cached per cue."""
        key = round(float(cue), 3)
        ph = self._phr.get((f.id, key)) if self._phr else None
        if ph is not None:
            key = (key, round(float(ph), 3))           # (its phrases from the landmark a cut landed past: its own places)
        op = g.opens.get(key)
        if op is not None:
            return op
        from lib.dj.recordplay import (CUT_MIN_BARS, LAST_FROM, PHRASE_BARS, SIZES, TAIL_BARS, TAIL_OUTRO_BARS,
                                       BREAK_MIN_BARS, RecordPlayer)
        rp = self._rp(pm)
        t = f.t
        dur = f.dur
        op = _Open()
        op.cue = float(cue)
        op.t_now = min(op.cue + PLAN_AT_S, max(op.cue, dur - 1.0))
        t_now = op.t_now
        bar = float(rp.bar_s(t, t_now))
        fd = float(rp._first_down(t, t_now))
        op.bar = bar
        op.k_now = rp._bar_index(t, t_now)
        op.open_bar = rp._bar_index(t, op.cue if ph is None else float(ph), t_ref=t_now)
        op.k_min = int(math.ceil((op.cue + float(pm) - fd) / bar - 1e-6))
        op.k_end = int(math.floor((dur - 0.3 - fd) / bar))
        k_last = int(math.ceil((LAST_FROM * dur - fd) / bar))
        K = RecordPlayer._phrase_up(max(op.k_now + CUT_MIN_BARS, op.k_min, k_last), op.open_bar)
        entries = []
        while K + SIZES["short"][0] + 1 <= op.k_end - 1:
            tK = rp._bar_time(t, K, t_now)
            kind = rp._section_kind(t, tK + 0.25 * bar)
            per = {}
            for nm in ("standard", "short"):
                L = SIZES[nm][0]
                T = TAIL_OUTRO_BARS if rp._section_kind(t, rp._bar_time(t, K + L, t_now) + 0.25 * bar) == "outro" else TAIL_BARS
                if K + L + T > op.k_end - 1:
                    continue
                t1 = rp._bar_time(t, K + L + T, t_now)
                if not rp._steady_span(t, tK, t1):
                    continue
                per[nm] = (L, T, RecordPlayer._vox_quiet(f.rec, tK, t1))
            if per:
                entries.append((K, kind, per))
            K += PHRASE_BARS
        op.entries = entries
        # the breakdown _plan_breakdown would take (the first one that resolves inside the record's room)
        op.bd = None
        secs = t.sections or []
        for i, s in enumerate(secs):
            if s.get("kind") != "breakdown":
                continue
            bs = rp._bar_index(t, float(s["start_s"]), t_ref=t_now)
            be = rp._bar_index(t, float(s["end_s"]), t_ref=t_now)
            if be - bs < BREAK_MIN_BARS or bs < op.k_now + CUT_MIN_BARS + 4:
                continue
            D = be
            j = i + 1
            while j < len(secs) and secs[j].get("kind") in ("build", "breakdown"):
                j += 1
            if j < len(secs) and secs[j].get("kind") == "groove":
                dj = rp._bar_index(t, float(secs[j]["start_s"]), t_ref=t_now)
                if dj - be <= 16:
                    D = dj
            D = RecordPlayer._phrase_up(D, op.open_bar) if (D - op.open_bar) % PHRASE_BARS <= 1 else D
            if D < op.k_min or D > op.k_end - 2:
                continue
            op.bd = (bs, be, D)
            break
        op.memo = {}
        op.kt = {}
        op.solo = None
        g.opens[key] = op
        return op

    @staticmethod
    def _choose(op, variant, cls, cues):
        """_find_blend_strict's choice (recordplay.py ~738), replicated over the places _open_of found: every place
        and size scored (neither sings 3; the next record silent through the overlap while the record sings 1; the
        record quiet through it and the next quiet until the swap 2; +0.5 its drop on the swap; +0.5 the record's
        outro), the best wins, the earliest among equals, and the first 3.5 ends the search. It depends on the next
        record only through which of its cues exist (`cls`), so the answer is memoized per class. Returns
        (K, size, L, T, quiet_bars_used, score, on_landmark, kind) or None."""
        key = (variant, cls)
        if key in op.memo:
            return op.memo[key]
        sizes = ("short",) if variant == "short" else ("standard", "short")
        best, best_score = None, None
        out = None
        for K, kind, per in op.entries:
            for nm in sizes:
                e = per.get(nm)
                if e is None:
                    continue
                L, T, aq = e
                c, on = cues[(nm, L + T)]
                used = L + T
                if c is not None:
                    score = 3 if aq is not False else 1
                else:
                    if aq is False:
                        continue
                    c, on = cues[(nm, L)]
                    used = L
                    if c is None:
                        continue
                    score = 2
                if on:
                    score += 0.5
                if kind == "outro":
                    score += 0.5
                if best_score is None or score > best_score + 1e-9:
                    best, best_score = (K, nm, L, T, used, score, bool(on), kind), score
                if best_score >= 3.5:
                    out = best
                    break
            if out is not None:
                break
        if out is None:
            out = best
        op.memo[key] = out
        return out

    def _seam(self, ctx, a, cue_a, b, strict=True, fit=False, op=None):
        """How record `b` comes in after `a` (which opened at cue_a), on the player's clock: {kind, size, play (a's
        seconds from its cue to b's start), cue_b, placement score, ...}. `strict` (every seam the planner offers):
        None unless the player's strict placement takes the pair - the tempo within MAX_TEMPO_GAP, the key veto,
        a place that keeps the vocals apart. Not strict (a seam the user made or the player already committed): the
        player's own fallbacks time it - the breakdown swap, else the strict blend, else the relaxed blend at the
        record's way out, else the record plays to its end (recordplay.py _plan_blend / _plan_cut). `fit`: the key fit
        when the caller has it (the pair table's row), else RecordPlayer.key_fit is asked.
        NOT memoized per pair (2026-10-28: a per-pair memo held 160 MB per crate in the child): everything costly is
        shared - the record's places per cue (_open_of) and the choice per cue class of the next record (_choose)."""
        R = _rpm()
        pm = ctx["play_min"]
        geo = self._geo.get(pm) or {}
        gb = geo.get(b.id) or self._geo_of(pm, b)
        if op is None:
            ga = geo.get(a.id) or self._geo_of(pm, a)
            op = self._open_of(pm, a, ga, cue_a)
        gap = abs(a.bpm / max(b.bpm, 1e-6) - 1.0)
        if fit is False:
            fit = R.RecordPlayer.key_fit(a.t, b.t)
        res = None
        best = None
        if gap <= R.MAX_TEMPO_GAP and (fit is None or fit >= R.KEY_BLEND):
            key = ("short" if (fit is not None and fit < R.KEY_LONG) else "std", gb.cls)
            best = op.memo[key] if key in op.memo else self._choose(op, key[0], gb.cls, gb.cues)
        if best is None and strict:
            return None
        if (best is not None or gap <= R.MAX_TEMPO_GAP) and op.bd is not None and gb.build:
            # the player tries the breakdown swap first (step(): _plan_breakdown, then _plan_blend) - but only a pair
            # its strict probe passed gets that far, so the planner never offers a breakdown-only pair
            res = self._breakdown(pm, op, a, b, gb)
        if res is None and best is not None:
            K, nm, L, T, used, score, on, kind = best
            res = {"kind": "blend", "size": nm, "K": K, "t_out": self._kt(pm, op, a, K), "cue_b": gb.cues[(nm, used)][0], "on_landmark": on,
                   "placement": score, "a_section": kind or "groove"}
        if res is None and not strict:
            res = self._relaxed(pm, op, a, b, gb) if gap <= R.MAX_TEMPO_GAP else None
            if res is None:
                res = self._to_the_end(pm, op, a, b, gb, "a tempo gap no blend can bridge" if gap > R.MAX_TEMPO_GAP else "no phrase left for a blend")
        if res is not None:
            res["play"] = res["t_out"] - cue_a
            res["key_fit"] = fit
            res["gap"] = gap
        return res

    def _kt(self, pm, op, a, K):
        """The song time of bar K of the record, on the segment the plan counts on (cached per opening)."""
        t = op.kt.get(K)
        if t is None:
            t = op.kt[K] = float(self._rp(pm)._bar_time(a.t, K, op.t_now))
        return t

    def _breakdown(self, pm, op, a, b, gb):
        """_plan_breakdown (recordplay.py ~949), replicated: the record's breakdown, the next record's build into its
        landmark, its intro and build under the breakdown, not where it sings."""
        if op.bd is None or not gb.build:
            return None
        R = _rpm()
        bs, be, D = op.bd
        lm = gb.lm
        bar_b = gb.bar_b
        K = bs - 4
        lm_bars = int(math.floor(lm / max(bar_b, 1e-6) + 1e-6))
        if lm_bars < D - K:
            K = D - lm_bars
        if K < op.k_now + R.CUT_MIN_BARS:
            return None
        cue_b = max(0.0, float(b.t.nearest_downbeat(lm - (D - K) * bar_b)))
        if R.RecordPlayer._vox_quiet(b.rec, cue_b, lm) is False:
            return None
        return {"kind": "breakdown", "size": "breakdown", "K": K, "t_out": self._kt(pm, op, a, K), "cue_b": cue_b,
                "on_landmark": True, "placement": 4.0, "a_section": "breakdown"}

    def _relaxed(self, pm, op, a, b, gb):
        """_find_blend(relax=True) (recordplay.py ~712): the short blend at the record's last phrases."""
        from lib.dj.recordplay import CUT_MIN_BARS, PHRASE_BARS, SIZES, TAIL_BARS, RecordPlayer
        rp = self._rp(pm)
        L, T = SIZES["short"][0], TAIL_BARS
        K = RecordPlayer._phrase_up(max(op.k_now + CUT_MIN_BARS, op.k_min), op.open_bar)
        K_last = K
        while K_last + PHRASE_BARS + L + T <= op.k_end - 1:
            K_last += PHRASE_BARS
        K = K_last
        if K + L + T > op.k_end - 1:
            return None
        cue_b = gb.cues[("short", L)][0]
        if cue_b is None:
            c, _on = rp._quietest_cue(b.rec, L, gb.bar_b)
            cue_b = float(c)
        return {"kind": "relaxed", "size": "short", "K": K, "t_out": self._kt(pm, op, a, K), "cue_b": cue_b,
                "on_landmark": False, "placement": 0.0, "a_section": "outro"}

    def _to_the_end(self, pm, op, a, b, gb, why):
        """_plan_cut(at_song_t = the record's end) (recordplay.py ~1048): the record finishes, the next comes in whole at
        its landmark - never a rip."""
        rp = self._rp(pm)
        k = max(op.k_now + 1, int(math.ceil((a.dur - 0.3 - float(rp._first_down(a.t, op.t_now))) / op.bar - 1e-6)))
        k = min(k, max(op.k_now + 1, op.k_end))
        return {"kind": "end", "size": "end", "K": k, "t_out": self._kt(pm, op, a, k), "cue_b": gb.lm,
                "on_landmark": True, "placement": 0.0, "a_section": "end", "why": why}

    def _solo(self, pm, f, cue):
        """A record's play from `cue` when what follows is not known yet: its earliest quiet steady place (the next
        record assumed silent through the mix, its drop on the swap)."""
        g = self._geo_of(pm, f)
        op = self._open_of(pm, f, g, cue)
        if op.solo is None:
            keys = _cue_keys()
            best = self._choose(op, "std", tuple((True, True) for _ in keys), {k: (0.0, True) for k in keys})
            rp = self._rp(pm)
            if best is not None:
                tK = float(rp._bar_time(f.t, best[0], op.t_now))
            else:
                rel = self._relaxed(pm, op, f, f, g)
                tK = rel["t_out"] if rel is not None else self._to_the_end(pm, op, f, f, g, "")["t_out"]
            op.solo = max(30.0, tK - float(cue))
        return op.solo

    # -- the crate and its pair table -------------------------------------------------------------------------------------
    def _crate_ids(self, theme, pm):
        c = theme.get("crate") or {}
        src = c.get("source")
        src_ids = None
        if src:
            ids = self.setlists.get(str(src))
            src_ids = set(ids) if ids else set()
        lo, hi = (list(c.get("bpm") or [0, 999]) + [0, 999])[:2]
        elo, ehi = (list(c.get("energy") or [0, 100]) + [0, 100])[:2]
        vrule = VOCALS_OK.get(c.get("vocals") or "any", VOCALS_OK["any"])
        g_in = [w.lower() for w in c.get("genresIn") or []]
        g_out = [w.lower() for w in c.get("genresOut") or []]
        m_in = [w.lower() for w in c.get("moodsIn") or []]
        t_in = [w.lower() for w in c.get("tagsIn") or []]
        t_out = [w.lower() for w in c.get("tagsOut") or []]
        out = []
        for f in self.facts.values():
            if src_ids is not None and f.id not in src_ids:
                continue
            if f.dur > MAX_DURATION_S:
                continue
            if f.bpm < float(lo) or f.bpm > float(hi):
                continue
            if f.e * 100.0 < float(elo) or f.e * 100.0 > float(ehi):
                continue
            if not vrule(f.sung):
                continue
            w = f.words
            if g_in and not any(x in w for x in g_in):
                continue
            if any(x in w for x in g_out):
                continue
            if m_in and not any(x in w for x in m_in):
                continue
            if t_in and not all(x in w for x in t_in):
                continue
            if any(x in w for x in t_out):
                continue
            if self._geo_of(pm, f).why is not None:
                continue                              # too short for a record at this minimum play
            out.append(f.id)
        return out

    def _table(self, theme, pm, ctx):
        """The pair table, once per crate: every record meets only its tempo neighbours (a tempo-sorted walk), the key
        veto and the player's strict placement (from the cue it most likely enters at) drop the pairs the player
        would refuse, and what is left carries the tempo, energy step and genre parts of its score (the vocal part
        comes from the exact seam at search time)."""
        sig = (json.dumps(theme.get("crate") or {}, sort_keys=True), float(pm), self._gen)
        tb = self._tables.get(sig)
        if tb is not None:
            return tb
        t0 = time.time()
        ids = self._crate_ids(theme, pm)
        F = self.facts
        ids.sort(key=lambda i: (F[i].bpm, i))
        tb = {"ids": ids, "set": set(ids), "bpms": [F[i].bpm for i in ids], "rows": {}, "pairs": 0}
        for i in ids:
            self._row(tb, F[i], ctx)
        tb["build_s"] = time.time() - t0
        while len(self._tables) >= 4:
            self._tables.pop(next(iter(self._tables)))
        self._tables[sig] = tb
        return tb

    def _row(self, tb, a, ctx):
        """Record a's outgoing pairs inside the crate (built on demand for a record outside it, e.g. the user's)."""
        import bisect
        R = _rpm()
        row = tb["rows"].get(a.id)
        if row is not None:
            return row
        F = self.facts
        pm = ctx["play_min"]
        ga = self._geo_of(pm, a)
        cue_a = ga.cue_in
        lo = bisect.bisect_left(tb["bpms"], a.bpm / (1.0 + R.MAX_TEMPO_GAP) - 1e-6)
        hi = bisect.bisect_right(tb["bpms"], a.bpm / (1.0 - R.MAX_TEMPO_GAP) + 1e-6)
        row = []
        for j in tb["ids"][lo:hi]:
            if j == a.id:
                continue
            b = F[j]
            gap = abs(a.bpm / max(b.bpm, 1e-6) - 1.0)
            if gap > R.MAX_TEMPO_GAP:
                continue
            fit = R.RecordPlayer.key_fit(a.t, b.t)          # THE KEY VETO: the player's own fit, pass/fail
            if fit is not None and fit < R.KEY_BLEND:
                continue
            if self._seam(ctx, a, cue_a, b, fit=fit) is None:
                continue
            s_t = max(0.25, math.exp(-((gap / 0.04) ** 2)))
            d = b.e - a.e
            s_step = math.exp(-((max(0.0, abs(d) - 0.12) / 0.15) ** 2))
            if a.genres and b.genres:
                s_gen = 0.85 + 0.15 * len(a.genres & b.genres) / min(len(a.genres), len(b.genres))
            else:
                s_gen = 0.92
            row.append((j, s_t * s_step * s_gen, fit))
        tb["rows"][a.id] = row
        if a.id in tb["set"]:
            tb["pairs"] += len(row)                   # (the crate's own pairs only: a row built for a record outside it is not counted)
        return row

    def _loose_row(self, tb, a):
        """Record a's tempo neighbours in the crate that pass the key veto, without the strict placement (the one
        relaxed seam after a record the plan did not choose; not cached, it is rare)."""
        import bisect
        R = _rpm()
        F = self.facts
        lo = bisect.bisect_left(tb["bpms"], a.bpm / (1.0 + R.MAX_TEMPO_GAP) - 1e-6)
        hi = bisect.bisect_right(tb["bpms"], a.bpm / (1.0 - R.MAX_TEMPO_GAP) + 1e-6)
        out = []
        for j in tb["ids"][lo:hi]:
            b = F[j]
            gap = abs(a.bpm / max(b.bpm, 1e-6) - 1.0)
            fit = R.RecordPlayer.key_fit(a.t, b.t)
            if j == a.id or gap > R.MAX_TEMPO_GAP or (fit is not None and fit < R.KEY_BLEND):
                continue
            d = b.e - a.e
            s_gen = (0.85 + 0.15 * len(a.genres & b.genres) / min(len(a.genres), len(b.genres))) if (a.genres and b.genres) else 0.92
            out.append((j, max(0.25, math.exp(-((gap / 0.04) ** 2))) * math.exp(-((max(0.0, abs(d) - 0.12) / 0.15) ** 2)) * s_gen))
        return out

    # -- scoring a record at a moment of the night ---------------------------------------------------------------------
    def _context(self, req, theme):
        pm = float(req.get("play_min_s") or DEFAULT_PLAY_MIN_S)
        L = 60.0 * min(max(float(req.get("length_min") or theme.get("length") or 90), LENGTH_MIN[0]), LENGTH_MIN[1])
        ctx = {"play_min": pm, "L": L, "base": {}}
        chs = theme.get("chapters") or [_ch("The night", 1.0, (0.4, 0.7), (0.3, 0.7))]
        tot = sum(float(c.get("share") or 0.0) for c in chs) or 1.0
        acc = 0.0
        out = []
        for c in chs:
            share = float(c.get("share") or 0.0)
            out.append({"name": c.get("name") or "", "p0": acc / tot, "p1": (acc + share) / tot, "energy": [float(x) for x in c.get("energy") or (0.5, 0.5)],
                        "tempo": [float(x) for x in c.get("tempo") or (0.5, 0.5)], "leans": [str(w).lower() for w in c.get("leans") or []],
                        "dims": {k: [float(v[0]), float(v[1])] for k, v in (c.get("dims") or {}).items() if k in CH_DIMS and v}})
            acc += share
        ctx["chs"] = out
        leans = {str(k).lower(): float(v) for k, v in (theme.get("leans") or {}).items()}
        for k, v in (req.get("neg_leans") or {}).items():
            leans[str(k).lower()] = leans.get(str(k).lower(), 0.0) - abs(float(v))     # BAD: the record's tags count against
        ctx["leans"] = leans
        ctx["seed"] = int(req.get("seed") or theme.get("night") or 1)
        ctx["fresh"] = bool(theme.get("fresh", True) if req.get("fresh") is None else req.get("fresh"))
        ctx["spacing"] = int(req.get("spacing") if req.get("spacing") is not None else theme.get("spacing", SPACING))
        km = str(theme.get("keys_mode") or "veto").lower()
        ctx["keys"] = km if km in KEYS_MODES else "veto"
        ctx["rise"] = [c["energy"][-1] - c["energy"][0] > 0.04 for c in out]       # (lift: the chapters that climb)
        return ctx

    @staticmethod
    def _camelot(cam):
        try:
            return int(cam[:-1]), cam[-1].upper()
        except (TypeError, ValueError, IndexError):
            return None

    def _s_key(self, ctx, a, b, fit, p):
        """The keys mode's light factor on a seam a -> b that the veto already passed (1.0 = what the mode wants)."""
        mode = ctx["keys"]
        if mode == "close":
            if fit is None:
                return KEY_OFF["unknown"]
            return 1.0 if fit >= 0.9 - 1e-9 else KEY_OFF["close"]
        if mode == "lift":
            tg = ctx.get("tgt")
            ci = tg[min(max(int(p * self.TGT_STEPS + 0.5), 0), self.TGT_STEPS)][0] if tg else self._target(ctx, p)[0]
            if not ctx["rise"][ci]:
                return 1.0                            # (a flat or falling chapter: lift is wander there, unknown keys included)
            if fit is None:
                return KEY_OFF["unknown"]
            ka, kb = self._camelot(a.t.camelot), self._camelot(b.t.camelot)
            if ka is None or kb is None:
                return KEY_OFF["unknown"]
            if ka[1] == kb[1] and (kb[0] - ka[0]) % 12 in (1, 2):
                return 1.0
            return KEY_OFF["lift_same"] if ka[0] == kb[0] else KEY_OFF["lift"]
        return 1.0

    def _prepare_scores(self, ctx, ids):
        """Per record and chapter, everything in the builder's node() that does not depend on the moment: the feel
        (theme leans and chapter leans), the night's noise, the rest for records played lately, and BAD's tags."""
        leans = ctx["leans"]
        pos = {k: v for k, v in leans.items() if v > 0}
        neg = {k: -v for k, v in leans.items() if v < 0}
        has = bool(pos)
        chs = ctx["chs"]
        base = ctx["base"]
        for i in ids:
            if i in base:
                continue
            f = self.facts[i]
            tw = sum(v for k, v in pos.items() if k in f.words)
            ng = sum(v for k, v in neg.items() if k in f.words)
            noise = 0.86 + 0.28 * _hash01(f.id, ctx["seed"]) if ctx["seed"] > 1 else 1.0
            worn = 1.0 / (1.0 + 0.3 * min(f.pl, 5)) if ctx["fresh"] else 1.0
            row = []
            for c in chs:
                cl = c["leans"]
                cw = sum(1 for w in cl if w in f.words)
                if has or cl:
                    feel = 0.45 * (_squash(tw, 1.5) if has else 0.6) + 0.55 * (_squash(cw, 0.8) if cl else 0.6)
                else:
                    feel = 1.0
                row.append((0.25 + 0.75 * feel) * noise * worn * math.exp(-ng))
            base[i] = row

    def _target(self, ctx, p):
        """The story at fraction p of the night: (chapter index, energy, bpm, dims)."""
        chs = ctx["chs"]
        p = min(max(p, 0.0), 0.9999)
        ci = len(chs) - 1
        for k, c in enumerate(chs):
            if p < c["p1"]:
                ci = k
                break
        c = chs[ci]
        u = min(max((p - c["p0"]) / max(c["p1"] - c["p0"], 1e-6), 0.0), 1.0)
        e = c["energy"][0] + (c["energy"][1] - c["energy"][0]) * u
        q = c["tempo"][0] + (c["tempo"][1] - c["tempo"][0]) * u
        bp = ctx["bpms"]
        b = bp[min(max(_jround(q * (len(bp) - 1)), 0), len(bp) - 1)] if bp else 120.0
        dims = {k: v[0] + (v[1] - v[0]) * u for k, v in c["dims"].items()}
        return ci, e, b, dims

    TGT_STEPS = 4000          # the search reads the story from a table: 1/4000 of the night (1.4 s at 90 min) moves an
                              # energy target by well under 0.001 - and the search asks a million times on a 6 h night

    def _node(self, ctx, i, p, exact=False):
        """The builder's node(): how well record i fits the story at fraction p of the night."""
        if exact:
            ci, e_t, b_t, dims = self._target(ctx, p)
        else:
            tg = ctx.get("tgt")
            if tg is None:
                tg = ctx["tgt"] = [self._target(ctx, k / self.TGT_STEPS) for k in range(self.TGT_STEPS + 1)]
            ci, e_t, b_t, dims = tg[min(max(int(p * self.TGT_STEPS + 0.5), 0), self.TGT_STEPS)]
        f = self.facts[i]
        base = ctx["base"].get(i)
        if base is None:
            self._prepare_scores(ctx, [i])
            base = ctx["base"][i]
        s = base[ci] * max(0.04, math.exp(-(((f.e - e_t) / 0.13) ** 2))) * max(0.3, math.exp(-(((f.bpm - b_t) / 7.0) ** 2)))
        if dims:
            dv = f.dims
            for k, tv in dims.items():
                v = dv[k]
                s *= 0.8 if v is None else max(0.15, math.exp(-(((v - tv) / 0.18) ** 2)))
        return s

    @staticmethod
    def _s_vox(sm):
        """The vocal room of a seam as a score, in the player's own order of preference (its placement score:
        neither singing 3, one quiet 2, the next silent under a singing record 1; its drop on the swap and the
        record's outro +0.5 each); the breakdown swap is this genre's own move."""
        if sm["kind"] == "breakdown":
            return 1.0
        if sm["kind"] == "blend":
            return 0.6 + 0.4 * min(float(sm["placement"]), 4.0) / 4.0
        return 0.4

    # -- the search -------------------------------------------------------------------------------------------------------
    def _search(self, ctx, tb, prefix, starts, cues, banned, history, beam, expand):
        """The builder's beam search on the player's clock. `prefix`: records already decided (their starts and
        cues known); empty: every crate record is a candidate opener. Returns (best path, finished?)."""
        F = self.facts
        L = ctx["L"]
        spacing = max(0, ctx["spacing"])
        hist_ak = [F[i].ak for i in history if i in F]
        hist_roots = {F[i].root for i in history if i in F}
        pm = ctx["play_min"]
        cands_all = [i for i in tb["ids"] if i not in banned]
        self._prepare_scores(ctx, cands_all)
        if prefix:
            last = prefix[-1]
            st, cue = starts[-1], cues[-1]
            en = st + self._solo(pm, F[last], cue)
            beams = [(0.0, tuple(prefix), 0, st, cue, en)]
            if en >= L:
                return list(prefix), True
        else:
            beams = []
            st0 = starts[0] if starts else 0.0
            for i in cands_all:
                g = self._geo_of(pm, F[i])
                solo = self._solo(pm, F[i], g.first)
                s = self._node(ctx, i, (st0 + 0.5 * solo) / L)
                beams.append((math.log(max(s, 1e-9)), (i,), 1, st0, g.first, st0 + solo))
            beams.sort(key=lambda b: -b[0])
            beams = beams[:beam]
        done = []
        dead = None
        solos = {}
        keyed = ctx.get("keys") in ("close", "lift")          # (veto / wander: no factor at all - today's search exactly)
        for _step in range(MAX_STEPS):
            nxt = []
            for sc, path, nnew, st, cue, _en in beams:
                last = F[path[-1]]
                used = set(path)
                seq_ak = hist_ak + [F[i].ak for i in path]
                recent = set(x for x in seq_ak[-spacing:] if x) if spacing else set()
                roots = hist_roots | {F[i].root for i in path}
                row = self._row(tb, last, ctx)
                op = self._open_of(pm, last, self._geo_of(pm, last), cue)      # (once per beam: the record and its cue are fixed)
                cands = []
                for j, s_static, fit in row:
                    if j in used or j in banned:
                        continue
                    c = F[j]
                    if c.ak and c.ak in recent:
                        continue
                    if c.root in roots:
                        continue
                    sm = self._seam(ctx, last, cue, c, fit=fit, op=op)
                    if sm is None:
                        continue
                    st_j = st + sm["play"]
                    sk = (j, sm["cue_b"])
                    solo = solos.get(sk)
                    if solo is None:
                        solo = solos[sk] = self._solo(pm, c, sm["cue_b"])
                    s = self._node(ctx, j, (st_j + 0.5 * solo) / L) * s_static * self._s_vox(sm)
                    if keyed:
                        s *= self._s_key(ctx, last, c, fit, (st_j + 0.5 * solo) / L)
                    cands.append((sc + math.log(max(s, 1e-9)), j, st_j, sm["cue_b"], st_j + solo))
                if not cands and prefix and nnew == 0:
                    # NOTHING BLENDS STRICTLY AFTER A RECORD THE PLAN DID NOT CHOOSE (the user's UP NEXT, or a committed one):
                    # typically a four-minute record with no steady phrase left after its minimum play. The player would
                    # refuse every pick it probes and end on the relaxed blend anyway, so the plan takes that one seam
                    # relaxed (the slot's "in" says so: the Director offers it without the strict probe) and stays strict
                    # after it - better than leaving the rest of the night unplanned. When the record has no phrase left
                    # for even the short blend, the player plays it to its END whatever comes next (its own fallback,
                    # recordplay.py _plan_blend: "no phrase ... is left for a blend") - the plan only times that; it is
                    # never chosen while a blend exists, and nothing here makes a record stop early.
                    for allow_end in (False, True):
                        for j, s_static in self._loose_row(tb, last):
                            if j in used or j in banned:
                                continue
                            c = F[j]
                            if (c.ak and c.ak in recent) or c.root in roots:
                                continue
                            sm = self._seam(ctx, last, cue, c, strict=False, op=op)
                            if sm is None or (sm["kind"] == "end" and not allow_end):
                                continue
                            st_j = st + sm["play"]
                            solo = self._solo(pm, c, sm["cue_b"])
                            s = self._node(ctx, j, (st_j + 0.5 * solo) / L) * s_static * self._s_vox(sm)
                            cands.append((sc + math.log(max(s, 1e-9)), j, st_j, sm["cue_b"], st_j + solo))
                        if cands:
                            ctx["relaxed_after"] = (last.id, allow_end)
                            break
                cands.sort(key=lambda x: -x[0])
                for s2, j, st_j, cue_j, en_j in cands[:expand]:
                    item = (s2, path + (j,), nnew + 1, st_j, cue_j, en_j)
                    (done if en_j >= L else nxt).append(item)
            if not nxt:
                if not done:
                    dead = beams[0] if beams else None
                break
            nxt.sort(key=lambda b: -b[0] / max(1, b[2]))
            kept, per = [], {}
            for b in nxt:
                k = b[1][-1]
                if per.get(k, 0) >= 2:
                    continue                          # beam variety: few beams sharing a last record
                per[k] = per.get(k, 0) + 1
                kept.append(b)
                if len(kept) >= beam:
                    break
            beams = kept
            if len(done) > DONE_MAX:
                break
        if done:
            # a finished night must not overshoot by much (the builder's 0.002 per minute over)
            done.sort(key=lambda b: -(b[0] / max(1, b[2]) - 0.002 * max(0.0, b[5] - L) / 60.0))
            return list(done[0][1]), True
        best = dead or (beams[0] if beams else None)
        if best is None:
            return list(prefix), False
        return list(best[1]), False

    # -- the answer -------------------------------------------------------------------------------------------------------
    def _timeline(self, ctx, path, n_fixed, fixed_items, start0):
        """Starts, cues and seams along a path: fixed seams timed by the player's fallbacks, planned seams strict."""
        F = self.facts
        pm = ctx["play_min"]
        starts, cues, seams = [], [], []
        for k, i in enumerate(path):
            fx = fixed_items[k] if k < len(fixed_items) else {}
            if k == 0:
                st = float(fx.get("start_s") if fx.get("start_s") is not None else start0)
                cue = fx.get("cue_s")
                if cue is None:
                    cue = self._geo_of(pm, F[i]).first if fx.get("opener", True) else self._geo_of(pm, F[i]).cue_in
                seams.append(None)
            else:
                strict = k >= n_fixed
                sm = self._seam(ctx, F[path[k - 1]], cues[-1], F[i], strict=strict)
                if sm is None:
                    sm = self._seam(ctx, F[path[k - 1]], cues[-1], F[i], strict=False)
                st = starts[-1] + sm["play"]
                cue = sm["cue_b"]
                if fx.get("start_s") is not None:
                    st = float(fx["start_s"])
                if fx.get("cue_s") is not None:
                    cue = float(fx["cue_s"])
                seams.append(sm)
            starts.append(float(st))
            cues.append(float(cue))
        return starts, cues, seams

    def _slots(self, ctx, tb, path, n_fixed, fixed_items, start0, banned, history, user_ids=(), alts=True):
        F = self.facts
        L = ctx["L"]
        pm = ctx["play_min"]
        starts, cues, seams = self._timeline(ctx, path, n_fixed, fixed_items, start0)
        n = len(path)
        plays = [(starts[k + 1] - starts[k]) if k + 1 < n else self._solo(pm, F[path[k]], cues[k]) for k in range(n)]
        in_path = set(path)
        spacing = max(0, ctx["spacing"])
        hist_ak = [F[i].ak for i in history if i in F]
        slots = []
        pos_leans = [k for k, v in ctx["leans"].items() if v > 0]
        for k, i in enumerate(path):
            f = F[i]
            pmid = (starts[k] + 0.5 * plays[k]) / L
            ci, e_t, b_t, dims = self._target(ctx, pmid)
            sl = {"id": f.id, "title": f.title, "artist": f.artist, "bpm": round(f.bpm, 1), "camelot": f.t.camelot or "",
                  "energy": round(f.e, 3), "sung": f.sung, "chapter": ci, "chapter_name": ctx["chs"][ci]["name"],
                  "start_s": round(starts[k], 3), "play_s": round(plays[k], 3), "end_s": round(starts[k] + plays[k], 3),
                  # the cue is NOT rounded: it sits on a downbeat, and a cue rounded below it counts its phrases from the bar
                  # before (the player's _bar_index floors), which moved a planned blend by a bar or a whole phrase
                  "cue_s": cues[k], "fixed": k < n_fixed, "user": f.id in user_ids,
                  "target": {"energy": round(e_t, 3), "bpm": round(b_t, 1), "dims": {d: round(v, 3) for d, v in dims.items()}},
                  "score": round(self._node(ctx, i, pmid, exact=True), 4),
                  "matched": [w for w in pos_leans if w in f.words][:6] + [w for w in ctx["chs"][ci]["leans"] if w in f.words and w not in pos_leans][:3],
                  "in": self._seam_facts(seams[k], F[path[k - 1]] if k else None, f), "alts": []}
            if alts and k >= n_fixed:
                relaxed = seams[k]["kind"] if (seams[k] is not None and seams[k]["kind"] in ("relaxed", "end")) else None
                sl["alts"] = self._alternates(ctx, tb, path, k, starts, cues, plays, banned, hist_ak, spacing, in_path, relaxed)
            slots.append(sl)
        return slots

    @staticmethod
    def _vox_words(sm, a, b):
        """The seam's vocal room in words (the player's placement: which record may sing where)."""
        k = sm["kind"]
        if k == "breakdown":
            return f"{b.title[:22]}'s intro and build under {a.title[:22]}'s breakdown, its drop where the breakdown resolves"
        if k == "relaxed":
            return f"the relaxed blend at {a.title[:22]}'s way out (nothing blends after it by the rules)"
        if k == "end":
            return f"{a.title[:22]} plays to its end ({sm.get('why') or 'no blend'})"
        vocal = sm["placement"] - (0.5 if sm["on_landmark"] else 0.0) - (0.5 if sm["a_section"] == "outro" else 0.0)   # 3, 2 or 1
        if vocal >= 3:
            return "neither record sings through the mix"
        if vocal >= 2:
            return f"{a.title[:22]}'s vocal quiet through the mix, {b.title[:22]}'s until the swap"
        return f"{b.title[:22]}'s vocal silent through the mix ({a.title[:22]} may sing over its intro)"

    @classmethod
    def _seam_facts(cls, sm, a, b):
        if sm is None or a is None:
            return None
        fit = sm.get("key_fit")
        return {"kind": sm["kind"], "size": sm["size"], "key_fit": None if fit is None else round(fit, 2), "tempo_gap_pct": round(100.0 * sm["gap"], 1),
                "vox": cls._vox_words(sm, a, b), "placement": sm["placement"], "on_landmark": sm["on_landmark"], "blend_at_s": round(sm["t_out"], 3),
                "cue_s": sm["cue_b"], "step": round(b.e - a.e, 3), "genres_shared": len(a.genres & b.genres)}

    def _alternates(self, ctx, tb, path, k, starts, cues, plays, banned, hist_ak, spacing, in_path, relaxed=None):
        """Records that could take slot k instead: the seam from the record before it AND the seam into the record
        after it both pass the player's strict placement; the artist spacing and one version of a song still hold.
        `relaxed` ("relaxed" | "end"): the slot comes in on the player's fallback after the record before (nothing blends
        strictly after it), and so may its alternates."""
        F = self.facts
        L = ctx["L"]
        pm = ctx["play_min"]
        prev = path[k - 1] if k > 0 else None
        nxt = path[k + 1] if k + 1 < len(path) else None
        seq_ak = hist_ak + [F[i].ak for i in path]
        off = len(hist_ak)
        near = set(x for x in seq_ak[max(0, off + k - spacing):off + k] + seq_ak[off + k + 1:off + k + 1 + spacing] if x) if spacing else set()
        roots = {F[i].root for j, i in enumerate(path) if j != k}
        pmid = (starts[k] + 0.5 * plays[k]) / L
        if prev is None:
            cand = list(tb["ids"])
        elif relaxed:
            cand = [row[0] for row in self._loose_row(tb, F[prev])]
        else:
            cand = [row[0] for row in self._row(tb, F[prev], ctx)]
        out = []
        for j in cand:
            if j in in_path or j in banned:
                continue
            c = F[j]
            if (c.ak and c.ak in near) or c.root in roots:
                continue
            if prev is not None:
                s_in = self._seam(ctx, F[prev], cues[k - 1], c, strict=not relaxed)
                if s_in is None or (s_in["kind"] == "end" and relaxed != "end"):
                    continue
                cue_c = s_in["cue_b"]
                e_in = self._s_vox(s_in)
            else:
                s_in, cue_c, e_in = None, self._geo_of(pm, c).first, 1.0
            e_out = 1.0
            if nxt is not None:
                s_out = self._seam(ctx, c, cue_c, F[nxt])
                if s_out is None:
                    continue
                e_out = self._s_vox(s_out)
            out.append((self._node(ctx, j, pmid) * e_in * e_out, j, s_in))
        out.sort(key=lambda x: -x[0])
        return [{"id": j, "title": F[j].title, "artist": F[j].artist, "bpm": round(F[j].bpm, 1), "camelot": F[j].t.camelot or "",
                 "energy": round(F[j].e, 3), "score": round(s, 4), "in": self._seam_facts(s_in, F[prev] if prev is not None else None, F[j])}
                for s, j, s_in in out[:ALTS]]

    def _warnings(self, ctx, tb, slots, finished):
        """The builder's chapter warnings: only a chapter that really got off records is mentioned (none of its leans,
        or well away from the energy its moment asks for), with what the crate holds for it and how to fix it. A chapter
        warning carries, besides bad / chapter / text, off and of (its records that miss, of those placed) and off_ids."""
        F = self.facts
        warns = []
        ra = ctx.get("relaxed_after")
        if ra is not None and any(s["id"] == ra[0] for s in slots):
            if ra[1]:
                warns.append({"bad": False, "text": f"{F[ra[0]].title} has no phrase left for any blend after its minimum play: it plays to its "
                                                    "end and the next record comes in whole there (the player's own fallback); the plan blends "
                                                    "by the rules again after it."})
            else:
                warns.append({"bad": False, "text": f"Nothing blends after {F[ra[0]].title} by the rules (no steady phrase is left after its "
                                                    "minimum play): the next record comes in on the relaxed blend, and the plan is strict again "
                                                    "after it."})
        if not finished and slots and all(s["fixed"] for s in slots):
            # nothing in the crate follows the last fixed record by the player's strict placement (the user's record
            # sings everywhere, or sits at a tempo of its own): the DJ picks the next one, and once that is fixed a
            # re-plan carries on from it - the plan never offers a seam the player would have to relax
            warns.append({"bad": True, "text": f"Nothing in the crate blends after {slots[-1]['title']} by the rules: the DJ picks the next "
                                               "record, and the plan carries on from it."})
        elif not finished:
            last = slots[-1] if slots else None
            warns.append({"bad": True, "text": "The route runs out of records that blend at " + _hm(last["end_s"] if last else 0) +
                          ". Widen the crate's tempo or genres, or shorten the night."})
        for ci, c in enumerate(ctx["chs"]):
            placed = [s for s in slots if s["chapter"] == ci and not s["fixed"]]
            if not placed:
                continue
            off = []
            for s in placed:
                why = []
                if c["leans"] and not any(w in F[s["id"]].words for w in c["leans"]):
                    why.append("none of its leans")
                if abs(s["energy"] - s["target"]["energy"]) > 0.15:
                    why.append(f"energy {s['energy']:.2f} where it asks for {s['target']['energy']:.2f}")
                if why:
                    off.append((s, why))
            # A MARK MEANS SOMETHING (2026-10-29): one record a little off is the search's normal give, not news - the
            # chapter is said only when at least WARN_MIN_OFF records AND at least WARN_MIN_SHARE of it miss (red: more
            # than half); the records that miss are named (off_ids: the strip outlines them)
            if len(off) < WARN_MIN_OFF or len(off) < WARN_MIN_SHARE * len(placed) - 1e-9:
                continue
            lo = min(c["energy"]) - 0.08
            hi = max(c["energy"]) + 0.08
            fit = sum(1 for i in tb["ids"] if lo <= F[i].e <= hi and (not c["leans"] or any(w in F[i].words for w in c["leans"])))
            names = "; ".join(f"{s['title']} ({', '.join(w)})" for s, w in off[:6])
            warns.append({"bad": len(off) * 2 > len(placed), "chapter": ci, "off": len(off), "of": len(placed),
                          "off_ids": [int(s["id"]) for s, _w in off],
                          "text": f"\u201c{c['name']}\u201d: {len(off)} of its {len(placed)} records don't match what it asks for: {names}"
                                  + ("; \u2026" if len(off) > 6 else "") + ". The crate holds " + str(fit) + " records that"
                                  + (f" carry one of its leans ({', '.join(c['leans'])}) and" if c["leans"] else "")
                                  + f" sit in its energy range ({max(0.0, lo):.2f}\u2013{min(1.0, hi):.2f}), but not enough of them also blend"
                                  + f" with their neighbours and fit the tempo here. To fix it: add leans, widen the crate, or give \u201c{c['name']}\u201d less of the night."})
        # A THIN CRATE (2026-10-29): the alternates are every crate record that blends with both neighbours, so a slot
        # with fewer than two is the crate's own limit (Slow Morning: 311 records but about 13 blendable neighbours each,
        # where the other presets have 70-90) - said, not hidden: where a planned record is refused there, the DJ picks
        planned = [s for s in slots if not s["fixed"]]
        few = [s for s in planned if len(s.get("alts") or []) < 2]
        if planned and len(few) > ALTS_FEW_SHARE * len(planned):
            per = tb["pairs"] / max(1, len(tb["ids"]))
            warns.append({"bad": False, "alts": len(few),
                          "text": f"{len(few)} of the {len(planned)} planned records have fewer than 2 alternates: the crate's {len(tb['ids'])} "
                                  f"records hold {tb['pairs']} pairs that blend ({per:.0f} per record), so where one of those records is "
                                  "refused the DJ picks instead. Widen the crate's tempo or genres for more."})
        return warns

    def _arc(self, ctx):
        L = ctx["L"]
        out = []
        for k in range(61):
            ci, e, b, dims = self._target(ctx, k / 60.0)
            out.append({"t": round(k / 60.0 * L, 1), "energy": round(e, 3), "bpm": round(b, 1), "chapter": ci})
        return out

    def _theme_of(self, req):
        th = req.get("theme")
        if isinstance(th, str):
            th = get_preset(th)
        if not th and req.get("preset"):
            th = get_preset(req["preset"])
        if not th:
            th = get_preset(PRESETS[0]["name"])
        return th

    # -- requests ---------------------------------------------------------------------------------------------------------
    def handle(self, req):
        """One request -> one response (JSON-able dicts). See PlannerProcess for the shapes."""
        t0 = time.time()
        op = str(req.get("op") or "plan")
        self._phr = {}
        try:
            if op == "ping":
                return {"ok": True, "op": op, "seq": req.get("seq"), "loaded": self.facts is not None}
            if op == "presets":
                return {"ok": True, "op": op, "seq": req.get("seq"), "presets": json.loads(json.dumps(PRESETS))}
            self._load(reload=bool(req.get("reload")))
            if op in ("plan", "replan"):
                out = self._plan(req)
            elif op == "rejoin":
                out = self._rejoin(req)
            elif op == "seam":
                out = self._seam_req(req)
            elif op == "crate":
                out = self._crate_req(req)
            elif op == "infer":
                out = self._infer_req(req)
            else:
                return {"ok": False, "op": op, "seq": req.get("seq"), "error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001
            import traceback
            return {"ok": False, "op": op, "seq": req.get("seq"), "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        out.update({"ok": True, "op": op, "seq": req.get("seq"), "lib_sig": self.lib_sig})
        out.setdefault("timing", {})["total_s"] = round(time.time() - t0, 3)
        return out

    def _state(self, req, ctx):
        """The night so far: what is banned (played, skipped, unfit, fixed), the order that played (artist spacing),
        and the fixed prefix with its known starts and cues."""
        F = self.facts
        played = [int(i) for i in req.get("played_ids") or [] if int(i) in F]
        fixed_items = []
        for it in req.get("fixed") or []:
            if isinstance(it, dict):
                d = dict(it)
            else:
                d = {"id": it}
            d["id"] = int(d["id"])
            if d["id"] in F:
                fixed_items.append(d)
                if d.get("phrase_s") is not None and d.get("cue_s") is not None:
                    self._phr[(d["id"], round(float(d["cue_s"]), 3))] = float(d["phrase_s"])
        banned = set(played) | {int(i) for i in req.get("exclude_ids") or []} | {d["id"] for d in fixed_items}
        start0 = float(req.get("elapsed_s") or 0.0)
        if fixed_items and played and "cue_s" not in fixed_items[0]:
            fixed_items[0]["opener"] = False      # (a record that came in under another one: it starts at its entry cue)
        return played, fixed_items, banned, start0

    def _plan(self, req):
        theme = self._theme_of(req)
        ctx = self._context(req, theme)
        t0 = time.time()
        tb = self._table(theme, ctx["play_min"], ctx)
        t_table = time.time() - t0
        F = self.facts
        ctx["bpms"] = sorted(F[i].bpm for i in tb["ids"])
        played, fixed_items, banned, start0 = self._state(req, ctx)
        beam = int(req.get("beam") or BEAM)
        expand = int(req.get("expand") or EXPAND)
        warns = []
        if len(tb["ids"]) < 6:
            return {"slots": [], "warnings": [{"bad": True, "text": f"The crate holds {len(tb['ids'])} records. Loosen it to plan a night."}],
                    "chapters": self._chapters_out(ctx), "arc": [], "length_s": ctx["L"], "crate": len(tb["ids"]), "pairs": tb["pairs"],
                    "theme": theme.get("name"), "timing": {"table_s": round(t_table, 3), "load_s": round(self.load_s, 2)}}
        prefix = [d["id"] for d in fixed_items]
        starts, cues, _seams = self._timeline(ctx, prefix, len(prefix), fixed_items, start0) if prefix else ([start0], [], [])
        t1 = time.time()
        # A RE-PLAN THAT KEEPS WHAT STILL FITS (2026-10-29; DJ_SETS.md section 3, BAD: "slots that still fit and don't
        # carry those tags stay"; horizon rule 2: a slot changes only when it becomes impossible or you change it): with
        # the previous plan's slots (plan_ids), its order is kept from the fixed records on while each seam still passes
        # and no slot carries one of `bad_tags`; the search goes on from the first slot that breaks. Without plan_ids
        # (a dial, a theme, the end extension): the whole rest is searched again, as before.
        keep = [int(i) for i in req.get("plan_ids") or [] if int(i) in F] if prefix else []
        kept, kept_stop = None, None
        if keep:
            stop_words = {str(w).lower() for w in req.get("bad_tags") or [] if w}
            path, finished, kept, kept_stop = self._keep_chain(ctx, tb, prefix, fixed_items, start0, starts, cues, keep, banned, played,
                                                               stop_words, beam, expand)
        else:
            path, finished = self._search(ctx, tb, prefix, starts, cues, banned, played, beam, expand)
        t_search = time.time() - t1
        slots = self._slots(ctx, tb, path, len(prefix), fixed_items, start0, banned, played,
                            user_ids={int(i) for i in req.get("user_ids") or []})
        warns += self._warnings(ctx, tb, slots, finished)
        return {"theme": theme.get("name"), "keys": KEYS_WORDS[ctx["keys"]], "keys_mode": ctx["keys"], "slots": slots, "warnings": warns,
                "kept_ids": kept, "kept_stop": kept_stop,
                "chapters": self._chapters_out(ctx), "arc": self._arc(ctx), "length_s": ctx["L"],
                "end_s": slots[-1]["end_s"] if slots else start0, "finished": finished,
                "crate": len(tb["ids"]), "pairs": tb["pairs"], "play_min_s": ctx["play_min"],
                "crate_ids": list(tb["ids"]),             # (the Director's gap picks draw from the crate: the brain's pool_ids)
                "vox_approx": sum(1 for i in tb["ids"] if not F[i].vox_exact),
                "timing": {"table_s": round(t_table, 3), "search_s": round(t_search, 3), "load_s": round(self.load_s, 2)}}

    def _fits_after(self, ctx, played):
        """The variety rules the search keeps, for a chain kept by hand: no artist within the spacing, one version of a
        song (counting what played tonight)."""
        F = self.facts
        spacing = max(0, ctx["spacing"])
        hist = [F[i].ak for i in played if i in F]
        roots_played = {F[i].root for i in played if i in F}

        def fits_after(seq, sid):
            f = F[sid]
            recent = (hist + [F[i].ak for i in seq])[-spacing:] if spacing else []
            return not (f.ak and f.ak in recent) and f.root not in ({F[i].root for i in seq} | roots_played)
        return fits_after

    def _keep_chain(self, ctx, tb, prefix, fixed_items, start0, starts, cues, ids, banned, played, stop_words, beam, expand):
        """The previous plan's order kept after `prefix` (the fixed records, and the user's record on a rejoin): each slot
        of `ids` in turn while its seam from the record before still passes the player's strict placement, the variety
        rules hold, the night is not over and it carries none of `stop_words` (BAD's tags); the search carries on from
        the first one that breaks. Returns (path, finished, kept ids, {id, why} of the slot it stopped at or None)."""
        F = self.facts
        L = ctx["L"]
        pm = ctx["play_min"]
        fits_after = self._fits_after(ctx, played)
        chain = list(prefix)
        st, cue = starts[-1], cues[-1]
        stop = None
        for sid in ids:
            if sid in banned or sid in chain:
                continue                              # (played, skipped or fixed since: it is not the plan's any more)
            f = F[sid]
            hit = sorted(stop_words & f.words) if stop_words else []
            if hit:
                stop = {"id": sid, "why": "carries " + ", ".join(hit[:3])}
                break
            if not fits_after(chain, sid):
                stop = {"id": sid, "why": "the artist again too soon"}
                break
            sm = self._seam(ctx, F[chain[-1]], cue, f)
            if sm is None:
                stop = {"id": sid, "why": "no strict blend from the record before it now"}
                break                                 # the chain breaks here: the search carries on from the record before
            if st >= L:
                stop = {"id": sid, "why": "past the night's length"}
                break
            st, cue = st + sm["play"], sm["cue_b"]
            chain.append(sid)
        # the length: drop trailing records whose start is already past the night (each is a whole record over)
        ts, tc, _x = self._timeline(ctx, chain, len(prefix), fixed_items, start0)
        while len(chain) > len(prefix) and ts[-1] >= L:
            chain.pop()
            ts.pop()
            tc.pop()
        kept = chain[len(prefix):]
        path, finished = chain, True
        end = ts[-1] + self._solo(pm, F[chain[-1]], tc[-1])
        if end < L:
            path, finished = self._search(ctx, tb, chain, ts, tc, banned, played, beam, expand)
        return path, finished, kept, stop

    def _rejoin(self, req):
        """After the user's UP NEXT (decision 3): their record follows the fixed prefix, and the plan rejoins at the
        earliest remaining slot it connects to (the player's strict placement from the user's record), keeping the
        chapter clock; the slots in between are skipped for good. The chain is kept while each of its seams still
        passes from its new cues and the night's length is honoured (trimmed at the end, or extended by the search);
        when no slot connects, the rest is searched again from the user's record."""
        theme = self._theme_of(req)
        ctx = self._context(req, theme)
        t0 = time.time()
        tb = self._table(theme, ctx["play_min"], ctx)
        t_table = time.time() - t0
        F = self.facts
        ctx["bpms"] = sorted(F[i].bpm for i in tb["ids"])
        played, fixed_items, banned, start0 = self._state(req, ctx)
        q = int(req["queued_id"])
        if q not in F:
            raise ValueError(f"no such record {q}")
        why = self._geo_of(ctx["play_min"], F[q]).why
        if why is not None:
            # the player's hook refuses it ("cannot be the record", recordplay.py ~381) and the DJ picks instead: the plan
            # goes on as it was, re-planned from the fixed prefix (DJ_SETS.md section 3: the plan pops that slot)
            sub = dict(req, op="replan")
            sub.pop("queued_id", None)
            out = self._plan(sub)
            out["warnings"] = [{"bad": False, "text": f"{F[q].title} cannot be the record ({why}): the plan goes on without it."}] + out["warnings"]
            out.update({"rejoined": False, "rejoined_at": None, "skipped": [], "queued_refused": why})
            return out
        fixed_items = fixed_items + [{"id": q}]
        banned = banned | {q}
        user_ids = {int(i) for i in req.get("user_ids") or []} | {q}
        prefix = [d["id"] for d in fixed_items]
        starts, cues, _s = self._timeline(ctx, prefix, len(prefix), fixed_items, start0)
        plan_ids = [int(i) for i in req.get("plan_ids") or [] if int(i) in F]
        L = ctx["L"]
        stop_words = {str(w).lower() for w in req.get("bad_tags") or [] if w}     # (a BAD waiting with this UP NEXT)
        fits_after = self._fits_after(ctx, played)
        at = None
        for j, sid in enumerate(plan_ids):
            if sid in banned or not fits_after(prefix, sid):
                continue
            if self._seam(ctx, F[q], cues[-1], F[sid]) is not None:
                at = j
                break
        t1 = time.time()
        skipped = []
        kept, kept_stop = None, None
        beam, expand = int(req.get("beam") or BEAM), int(req.get("expand") or EXPAND)
        if at is None:
            path, finished = self._search(ctx, tb, prefix, starts, cues, banned, played, beam, expand)
            rejoined = False
        else:
            skipped = [sid for sid in plan_ids[:at] if sid not in banned]
            # the chain is kept from the slot it rejoins at while each seam still passes; the slots skipped for the user's
            # record are banned from the search that carries on after it (they do not come back)
            path, finished, kept, kept_stop = self._keep_chain(ctx, tb, prefix, fixed_items, start0, starts, cues, plan_ids[at:],
                                                               banned | set(skipped), played, stop_words, beam, expand)
            rejoined = True
        t_search = time.time() - t1
        ban_all = banned | set(skipped)
        slots = self._slots(ctx, tb, path, len(prefix), fixed_items, start0, ban_all, played, user_ids=user_ids)
        warns = self._warnings(ctx, tb, slots, finished)
        return {"theme": theme.get("name"), "keys": KEYS_WORDS[ctx["keys"]], "keys_mode": ctx["keys"], "slots": slots, "warnings": warns,
                "kept_ids": kept, "kept_stop": kept_stop,
                "rejoined": rejoined, "rejoined_at": at, "skipped": [{"id": s, "title": F[s].title, "why": SKIP_WHY} for s in skipped],
                "chapters": self._chapters_out(ctx), "arc": self._arc(ctx), "length_s": L,
                "end_s": slots[-1]["end_s"] if slots else start0, "finished": finished,
                "crate": len(tb["ids"]), "pairs": tb["pairs"], "play_min_s": ctx["play_min"], "crate_ids": list(tb["ids"]),
                "timing": {"table_s": round(t_table, 3), "search_s": round(t_search, 3), "load_s": round(self.load_s, 2)}}

    def _seam_req(self, req):
        """The Director's pre-offer check (DJ_SETS.md section 3, the third hook fix): would the player blend
        candidate b after the record a playing from cue_s? Returns the seam facts or None."""
        F = self.facts
        ctx = {"play_min": float(req.get("play_min_s") or DEFAULT_PLAY_MIN_S)}
        a, b = F.get(int(req["a"])), F.get(int(req["b"]))
        if a is None or b is None:
            return {"seam": None, "why": "no such record"}
        cue = req.get("cue_s")
        if cue is None:
            cue = self._geo_of(ctx["play_min"], a).cue_in
        sm = self._seam(ctx, a, float(cue), b)
        return {"seam": self._seam_facts(sm, a, b) if sm else None, "play_s": round(sm["play"], 1) if sm else None}

    def _crate_req(self, req):
        """The NIGHT tab's live count (2026-10-29): how many records a theme's crate holds and how many pairs among them
        blend by the player's rules, from the same pair table a plan of that crate uses (built once, then cached)."""
        theme = self._theme_of(req)
        pm = float(req.get("play_min_s") or DEFAULT_PLAY_MIN_S)
        t0 = time.time()
        tb = self._table(theme, pm, {"play_min": pm})
        F = self.facts
        n = len(tb["ids"])
        es = sorted(F[i].e for i in tb["ids"])
        return {"crate": n, "pairs": tb["pairs"], "per_record": round(tb["pairs"] / n, 1) if n else 0.0,
                "bpm": [round(tb["bpms"][0], 1), round(tb["bpms"][-1], 1)] if n else None,
                "energy": [round(es[0], 3), round(es[-1], 3)] if n else None, "timing": {"table_s": round(time.time() - t0, 3)}}

    def _infer_req(self, req):
        """"A night like these" (docs/DJ_SETS.md section 2, item 3): a theme inferred from example records, measured
        against this library (lib/dj/themestore.infer: ranges plus margins widened to about twice the night's need,
        leans by lift, vocals from the stems' envelope, the story from the examples' energy spread)."""
        from lib.dj.themestore import infer
        pm = float(req.get("play_min_s") or DEFAULT_PLAY_MIN_S)
        length_min = min(max(float(req.get("length_min") or 90.0), LENGTH_MIN[0]), LENGTH_MIN[1])

        out = infer(self.facts, [int(i) for i in req.get("example_ids") or []],
                    lambda crate: self._crate_ids({"crate": crate}, pm), length_min, pm, name=req.get("name"))
        th = out.get("theme")
        if th is not None:
            th["examples"] = [int(i) for i in req.get("example_ids") or []]
        return out

    @staticmethod
    def _chapters_out(ctx):
        L = ctx["L"]
        bp = ctx.get("bpms") or []

        def pct(q):
            return bp[min(max(_jround(q * (len(bp) - 1)), 0), len(bp) - 1)] if bp else None
        return [{"name": c["name"], "start_s": round(c["p0"] * L, 1), "end_s": round(c["p1"] * L, 1), "energy": c["energy"],
                 "tempo_bpm": [pct(c["tempo"][0]), pct(c["tempo"][1])], "leans": c["leans"], "dims": c["dims"]} for c in ctx["chs"]]


def _hm(s):
    s = max(0, int(round(s)))
    return f"{s // 3600}:{(s % 3600) // 60:02d}"


# ---- the subprocess -------------------------------------------------------------------------------------------------------
class PlannerProcess:
    """The planner in its own process (DJ_SETS.md section 3: a re-plan is too heavy for a thread in the show - a
    search would hold the GIL for seconds, and 85-119 ms stalls already hurt). One long-lived child, started on
    first use, below normal priority; one JSON line in, one JSON line out. NOT multiprocessing: spawn re-imports
    the parent's __main__ (all of Stories_OGL) in the child - lib/dj/tempo_worker.py has the story.

    Call `request()` from a worker thread (it blocks on the pipe, which releases the GIL), never from the audio
    callback or the render thread; `submit()` does that for you and calls back with the response.

    REQUEST (all optional except where noted):
      op            "plan" | "replan" | "rejoin" | "seam" | "crate" | "infer" | "presets" | "ping"
                    (crate: theme, play_min_s -> {crate, pairs, per_record, bpm, energy}, the NIGHT tab's live count;
                    infer: example_ids, length_min, play_min_s, name? -> {theme, widened, crate, need, missing},
                    "a night like these" - lib/dj/themestore.infer)
      theme.keys_mode  "veto" (the default and today's planner) | "close" | "wander" | "lift": a light factor on the
                    search's seam score, never a veto (KEYS_MODES above)
      seq           any; echoed, so a caller can drop a stale answer
      theme         a theme object (the builder's shape: name, crate, leans, chapters, keys) or a preset's name
      preset        a preset's name (when no theme)
      length_min    the night's length (10..480; default the theme's length, else 90)
      play_min_s    the record player's minimum play (the PACE dial; default 210)
      seed          the night seed (1 = no noise, as in the builder); fresh (rest records played in 30 days, default
                    true); spacing (records between two by one artist, default 6)
      beam, expand  the search's sizes (default 60, 12)
      played_ids    ids played tonight, in order (never again; the artist spacing reads them)
      exclude_ids   skips, "not tonight", records found unfit
      neg_leans     {word: weight}: BAD - a record carrying the word scores exp(-weight) as much
      fixed         [{id, cue_s?, start_s?, phrase_s?}, ...]: the record playing and slot +1 (kept as they are, in order);
                    start_s of the first = when it opened on the night's clock (else elapsed_s); phrase_s = where its
                    phrases count from when a cut landed it past a silent start (its landmark; else cue_s)
      elapsed_s     the night's clock at the first fixed record (or at the opener of a fresh plan)
      user_ids      ids the user asked for (marked "user" in the slots)
      reload        true: read the library again (after a rescan)
      rejoin only:  queued_id (required, the user's UP NEXT, placed after `fixed`), plan_ids (the previous plan's
                    slot ids after the fixed ones, in order)
      replan:       plan_ids (optional, as for rejoin): the previous plan's order is KEPT from the fixed records on
                    while each seam still passes and no slot carries one of bad_tags (BAD's own tags); the search
                    goes on from the first slot that breaks. Without plan_ids the whole rest is searched again.
      seam only:    a, b (ids), cue_s (a's cue), play_min_s

    RESPONSE: {ok, op, seq, error?, trace?, lib_sig (a checksum of the library the plan read: ids, keys, tempos,
      lengths - when the show's own library differs, send reload), theme, keys (the mode in words), keys_mode, slots: [SLOT], warnings: [{bad, text, chapter?}], chapters:
      [{name, start_s, end_s, energy:[from,to], tempo_bpm:[from,to], leans, dims}], arc: [{t, energy, bpm, chapter}]
      (61 samples), length_s, end_s, finished (false: the route ran out of records that blend), crate, pairs,
      play_min_s, vox_approx, timing: {load_s, table_s, search_s, total_s}; rejoin adds rejoined, rejoined_at (index
      into plan_ids or null), skipped: [{id, title, why}], and queued_refused (why) when the user's record cannot be a
      record (too short for the minimum play: the plan goes on without it). A replan with plan_ids and a rejoin add
      kept_ids (the previous plan's slots kept, in order) and kept_stop ({id, why}: the slot the kept order stopped
      at, or null). seam answers {seam: SEAM | null, play_s}.
    SLOT: {id, title, artist, bpm, camelot, energy, sung, chapter, chapter_name, start_s, play_s, end_s, cue_s
      (exact, not rounded: the Director may pass it back as fixed[].cue_s), fixed, user, target: {energy, bpm, dims},
      score, matched: [words], in: SEAM | null (the seam from the slot before), alts: [{id, title, artist, bpm,
      camelot, energy, score, in: SEAM}] (up to 3 on planned slots, fewer only where the crate has no more)}
    SEAM: {kind, size, key_fit, tempo_gap_pct, vox (words), placement, on_landmark, blend_at_s (the record before's
      song time where this one starts), cue_s (where this one starts), step, genres_shared}. kind is "blend" or
      "breakdown" on every seam the planner chose - the player's strict placement takes it, so the Director offers
      such a slot as (track, 'plan') with the strict probe. Only on the FIRST planned slot after the fixed records, and
      only when nothing at all blends strictly after the last fixed one (a warning says so), kind may be "relaxed" (the
      player's relaxed blend) or "end" (that record has no phrase left for any blend and plays to its end - the
      player's own fallback, whatever comes next): the Director offers that one slot without the strict probe. Seams
      between fixed records are timed by whatever the player will do (strict, breakdown, relaxed or end).
    """

    RESPAWN_S = 30.0          # a child that died is not started again within this (an import error after a code edit, a
                              # library that will not load: one spawn and a 4-6 s load per request was a storm, 2026-10-29)

    def __init__(self, music_root, log=print):
        self.music_root = music_root
        self.log = log
        self._proc = None
        self._lock = threading.Lock()
        self._seq = 0
        self._died_t = None

    def start(self):
        import subprocess
        import weakref
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            if self._proc is not None:
                self._proc, self._died_t = None, time.time()      # (it exited on its own)
            if self._died_t is not None and time.time() - self._died_t < self.RESPAWN_S:
                return False
            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env = dict(os.environ)
            env["DJ_NO_PLAY_LOG"] = "1"
            env["DJ_NO_MIDI"] = "1"
            env["DJ_DECODE_INPROC"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            kw = {}
            if sys.platform == "win32":
                kw["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x4000)
            try:
                self._proc = subprocess.Popen([sys.executable, "-m", "lib.dj.setplanner", "--serve", "--music", self.music_root],
                                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=repo, env=env, **kw)
            except Exception as e:  # noqa: BLE001
                self._proc, self._died_t = None, time.time()
                if self.log:
                    self.log(f"[setplanner] could not start the planner process: {type(e).__name__}: {e}")
                return False
            # the safety net: an owner dropped without close() kills its child when it is collected (about 400 MB each;
            # Director.close_planner is the deterministic way)
            weakref.finalize(self, PlannerProcess._kill_proc, self._proc)
            return True

    def request(self, req, timeout=300.0):
        """Send one request and wait for its answer (blocking: a worker thread only). A dead or silent child is killed
        and the answer is {ok: False, error}; the next request starts a fresh one."""
        if not self.start():
            return {"ok": False, "error": "the planner process is not available", "seq": req.get("seq")}
        with self._lock:
            proc = self._proc
            self._seq += 1
            req = dict(req)
            req.setdefault("seq", self._seq)
            timer = threading.Timer(float(timeout), self._kill, args=(proc,))
            timer.daemon = True
            timer.start()
            try:
                proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError) as e:
                line = b""
                err = f"{type(e).__name__}: {e}"
            else:
                err = "the planner process closed its pipe"
            finally:
                timer.cancel()
            if not line:
                self._kill(proc)
                return {"ok": False, "error": err, "seq": req.get("seq")}
            try:
                return json.loads(line.decode("utf-8"))
            except ValueError as e:
                return {"ok": False, "error": f"bad answer: {e}", "seq": req.get("seq")}

    def submit(self, req, on_done):
        """request() on a daemon thread; on_done(response) is called on that thread."""
        def work():
            resp = self.request(req)
            try:
                on_done(resp)
            except Exception as e:  # noqa: BLE001
                if self.log:
                    self.log(f"[setplanner] callback: {type(e).__name__}: {e}")
        th = threading.Thread(target=work, daemon=True, name="setplanner-request")
        th.start()
        return th

    @staticmethod
    def _kill_proc(proc):
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def _kill(self, proc):
        self._kill_proc(proc)
        if proc is self._proc:
            self._proc, self._died_t = None, time.time()

    def close(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.write(b'{"op": "quit"}\n')
            proc.stdin.flush()
            proc.stdin.close()
            proc.wait(timeout=3.0)
        except Exception:
            self._kill(proc)


def _serve(music_root):
    """The child's loop: stdout carries only the protocol (stray prints go to stderr), one answer per request, until
    the parent closes the pipe (it exits with its parent). The vocal floors env.npz cannot decide are measured in a
    background thread between requests, for the next start (the plans of this process stay as they began)."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    if sys.platform != "win32":
        try:
            os.nice(5)
        except Exception:
            pass
    planner = Planner(music_root, log=lambda s: print(s, file=sys.stderr, flush=True))
    stop = threading.Event()
    warm = {"th": None}
    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            resp = {"ok": False, "error": f"bad request: {e}"}
        else:
            if req.get("op") == "quit":
                break
            resp = planner.handle(req)
        try:
            out.write((json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8"))
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            break
        if warm["th"] is None and planner.facts is not None and any(not f.vox_exact for f in planner.facts.values()):
            warm["th"] = threading.Thread(target=warm_vox, args=(music_root,), kwargs={"log": None, "stop": stop}, daemon=True, name="vox-warm")
            warm["th"].start()
    stop.set()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default
    music = arg("--music", "D:/Devel/music")
    os.environ.setdefault("DJ_NO_PLAY_LOG", "1")
    os.environ.setdefault("DJ_NO_MIDI", "1")
    if "--serve" in argv:
        return _serve(music)
    if "--warm-vox" in argv:
        n = warm_vox(music)
        print(f"[setplanner] measured {n} vocal floors")
        return 0
    if "--plan" in argv:
        with open(arg("--plan"), encoding="utf-8") as fh:
            req = json.load(fh)
        resp = Planner(music, log=lambda s: print(s, file=sys.stderr, flush=True)).handle(req)
        sys.stdout.write(json.dumps(resp, indent=1, ensure_ascii=False) + "\n")
        return 0 if resp.get("ok") else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
