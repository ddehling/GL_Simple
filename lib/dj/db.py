"""Sqlite music-library database for the DJ subsystem.

One DB file lives INSIDE the music folder (music/dj_library.sqlite3) so the
library + its analysis travel together between machines. Track paths are
stored RELATIVE to the music root for the same reason.

Schema v1: tracks / sections / loops / mix_points / play_history plus the
set-planner tables (setlists / setlist_entries). JSON-typed columns hold the
richer analysis blobs (beat_grid, energy_curve, mood_hist, spectral).
"""
import json
import os
import sqlite3
import time

SCHEMA_VERSION = 15              # v15: seam_feedback.ver (engine
                                 # fingerprint, so evidence from
                                 # different engine eras stays separable)
#                                 v14: setlists.total_s/arc_json (compiled
                                 #      set duration + arc, honored live) +
                                 #      tracks.bpm_source/bpm_scan/
                                 #      bpm_verified_at (live tempo verify
                                 #      written back - see set_verified_tempo)
                                 # (v13: tracks.rhythm / rhythm signature)
                                 # (v12: tracks.chroma + tracks.structure)
                                 # (v11: tracks.excluded / do-not-use flag)
                                 # (v10: tracks.mood_ml / Music2Emo)
                                 # (v9: tracks.enrichment / MusicBrainz)
                                 # (v8: setlist_entries.target_play_s)
DB_FILENAME = "dj_library.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,          -- relative to the music root
    file_size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    content_hash TEXT,
    title TEXT, artist TEXT, album TEXT,
    duration_s REAL,
    bpm REAL, bpm_conf REAL,
    beat_grid TEXT,                     -- JSON list of tempo segments
    downbeat_offset INTEGER, downbeat_conf REAL,
    key_pc INTEGER, key_mode TEXT, camelot TEXT, key_conf REAL, key_name TEXT,
    loudness_gain_db REAL,
    kick_offset_s REAL,
    energy_curve TEXT,                  -- JSON, 2 Hz, 0..1.2
    band_curve TEXT,                    -- JSON v3 {low/mid/high: [2Hz]}
    mood_hist TEXT,                     -- JSON {mood: fraction}
    rhythm_density REAL,
    spectral TEXT,                      -- JSON {bass/mid/high_share}
    live_check TEXT,                    -- JSON live-pipeline cross-check
    axes TEXT,                          -- JSON v2 character axes (0..1)
    auto_tags TEXT,                     -- JSON v2 derived tag list
    enrichment TEXT,                    -- JSON v9 MusicBrainz enrichment blob
    mood_ml TEXT,                       -- JSON v10 Music2Emo {valence,arousal,moods}
    file_genre TEXT,                    -- v10 embedded genre tag (free, from container)
    excluded INTEGER NOT NULL DEFAULT 0, -- v11 do-not-use: hidden from all selection
    chroma TEXT,                        -- v12 JSON 12-bin A-origin pitch-class profile
    structure TEXT,                     -- v12 JSON ML segments {segments:[[s,e,label]..]}
    rhythm TEXT,                        -- v13 JSON beat-sync rhythm signature
    bpm_source TEXT,                    -- v14 'scan' (default) | 'live_verified'
    bpm_scan REAL,                      -- v14 original scanner bpm (pre-verify)
    bpm_verified_at REAL,               -- v14 when the live verify wrote back
    analysis_version INTEGER,
    analyzed_at REAL,
    error TEXT,
    missing INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    start_s REAL NOT NULL, end_s REAL NOT NULL,
    start_beat INTEGER, end_beat INTEGER,
    energy REAL, bass_share REAL, mid_share REAL, high_share REAL,
    rhythm_density REAL, repetitiveness REAL,
    busyness REAL, vocalness REAL, boundary_strength REAL
);
CREATE INDEX IF NOT EXISTS idx_sections_track ON sections(track_id);
CREATE TABLE IF NOT EXISTS loops (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    start_s REAL NOT NULL, beats INTEGER NOT NULL, score REAL
);
CREATE INDEX IF NOT EXISTS idx_loops_track ON loops(track_id);
CREATE TABLE IF NOT EXISTS mix_points (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                 -- 'in' | 'out'
    time_s REAL NOT NULL, score REAL, style_hint TEXT
);
CREATE INDEX IF NOT EXISTS idx_mix_points_track ON mix_points(track_id);
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    UNIQUE(track_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_tags_track ON tags(track_id);
CREATE TABLE IF NOT EXISTS cues (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                 -- 'in' | 'out' | 'interest'
    time_s REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',   -- 'user' | 'auto'
    label TEXT
);
CREATE INDEX IF NOT EXISTS idx_cues_track ON cues(track_id);
CREATE TABLE IF NOT EXISTS seam_feedback (
    id INTEGER PRIMARY KEY,
    a_id INTEGER NOT NULL,              -- outgoing track
    b_id INTEGER NOT NULL,              -- incoming track
    style TEXT,
    up INTEGER NOT NULL,                -- 1 thumbs-up / 0 thumbs-down
    at REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'user', -- 'user' thumbs | 'auto' assessment
    ver TEXT                            -- engine fingerprint (lib/dj/version)
);
CREATE INDEX IF NOT EXISTS idx_seam_fb ON seam_feedback(a_id, b_id);
CREATE TABLE IF NOT EXISTS play_history (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    started_at REAL NOT NULL, ended_at REAL,
    transition_style TEXT, theme TEXT,
    skipped INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_history_track ON play_history(track_id);
CREATE TABLE IF NOT EXISTS setlists (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    theme TEXT, notes TEXT,
    total_s REAL,                       -- v14 compiled set length (live arc clock)
    arc_json TEXT,                      -- v14 JSON [[progress, energy], ...] waypoints
    created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS setlist_entries (
    id INTEGER PRIMARY KEY,
    setlist_id INTEGER NOT NULL REFERENCES setlists(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    pin_type TEXT NOT NULL DEFAULT 'suggestion',   -- 'anchor' | 'suggestion'
    target_offset_min REAL,             -- anchor's desired minutes into the set
    style_override TEXT,
    target_play_s REAL                  -- autofill timing solver's play hint
);
CREATE INDEX IF NOT EXISTS idx_setlist_entries ON setlist_entries(setlist_id, position);
"""

_JSON_COLS = ("beat_grid", "energy_curve", "band_curve", "mood_hist",
              "spectral", "live_check", "axes", "auto_tags", "enrichment",
              "mood_ml", "chroma", "structure", "rhythm")
# JSON columns the SCANNER owns and writes. enrichment (MusicBrainz),
# mood_ml (Music2Emo) and structure (allin1 segments) come from SEPARATE
# passes and must NEVER be touched by upsert_track - a re-scan's analysis
# dict lacks them, so writing them would NULL them out (wiped genres + ML
# mood on every re-scan). chroma and rhythm ARE scanner-owned
# (analyze_samples emits both) but a rescan result that lacks one (error
# record) must not wipe a backfill, so upsert skips None values for them
# via _KEEP_IF_NONE. (A real rescan legitimately rewrites rhythm with the
# mix-derived version; the rhythm backfill re-upgrades to stem-derived.)
_SCAN_JSON_COLS = tuple(c for c in _JSON_COLS
                        if c not in ("enrichment", "mood_ml", "structure"))
_KEEP_IF_NONE = ("chroma", "rhythm")

_SECTION_COLS = ("kind", "start_s", "end_s", "start_beat", "end_beat",
                 "energy", "bass_share", "mid_share", "high_share",
                 "rhythm_density", "repetitiveness", "busyness",
                 "vocalness", "boundary_strength")


class LibraryDB:
    """All DB access. Not thread-safe per instance; open one per thread."""

    def __init__(self, music_root, db_path=None):
        self.music_root = os.path.abspath(music_root)
        self.db_path = db_path or os.path.join(self.music_root, DB_FILENAME)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # check_same_thread=False: DJSystem builds the library cache on the
        # caller's thread, then hands the connection to its planner thread
        # (accesses are sequential, never concurrent). Scanner/planner tools
        # each open their own instance.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _migrate(self):
        ver = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if ver < SCHEMA_VERSION:
            self.conn.executescript(_SCHEMA)     # tables are IF NOT EXISTS
            have = {r[1] for r in
                    self.conn.execute("PRAGMA table_info(tracks)")}
            for col, typ in (("axes", "TEXT"), ("auto_tags", "TEXT"),
                             ("band_curve", "TEXT"),
                             ("kick_offset_s", "REAL"),   # v2/v3/v4 columns
                             ("phrase_beats", "INTEGER"),  # v5: hypermeter
                             ("phrase_start_s", "REAL"),
                             ("phrase_conf", "REAL")):
                if col not in have:
                    self.conn.execute(
                        f"ALTER TABLE tracks ADD COLUMN {col} {typ}")
            have_fb = {r[1] for r in
                       self.conn.execute("PRAGMA table_info(seam_feedback)")}
            if have_fb and "source" not in have_fb:      # v7
                self.conn.execute(
                    "ALTER TABLE seam_feedback ADD COLUMN source TEXT"
                    " NOT NULL DEFAULT 'user'")
            have_se = {r[1] for r in self.conn.execute(
                "PRAGMA table_info(setlist_entries)")}
            if have_se and "target_play_s" not in have_se:   # v8
                self.conn.execute(
                    "ALTER TABLE setlist_entries ADD COLUMN"
                    " target_play_s REAL")
            if "enrichment" not in have:                     # v9
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN enrichment TEXT")
            if "mood_ml" not in have:                         # v10
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN mood_ml TEXT")
            if "file_genre" not in have:                      # v10
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN file_genre TEXT")
            if "excluded" not in have:                        # v11
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN excluded INTEGER "
                    "NOT NULL DEFAULT 0")
            if "chroma" not in have:                          # v12
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN chroma TEXT")
            if "structure" not in have:                       # v12
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN structure TEXT")
            if "rhythm" not in have:                          # v13
                self.conn.execute(
                    "ALTER TABLE tracks ADD COLUMN rhythm TEXT")
            for col, typ in (("bpm_source", "TEXT"),          # v14
                             ("bpm_scan", "REAL"),
                             ("bpm_verified_at", "REAL")):
                if col not in have:
                    self.conn.execute(
                        f"ALTER TABLE tracks ADD COLUMN {col} {typ}")
            have_sl = {r[1] for r in
                       self.conn.execute("PRAGMA table_info(setlists)")}
            for col, typ in (("total_s", "REAL"),             # v14
                             ("arc_json", "TEXT")):
                if have_sl and col not in have_sl:
                    self.conn.execute(
                        f"ALTER TABLE setlists ADD COLUMN {col} {typ}")
            have_fb = {r[1] for r in
                       self.conn.execute("PRAGMA table_info(seam_feedback)")}
            if have_fb and "ver" not in have_fb:            # v15
                self.conn.execute(
                    "ALTER TABLE seam_feedback ADD COLUMN ver TEXT")
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()

    def close(self):
        self.conn.close()

    # -- path helpers ------------------------------------------------------
    def rel(self, abs_path):
        return os.path.relpath(os.path.abspath(abs_path),
                               self.music_root).replace("\\", "/")

    def abs(self, rel_path):
        return os.path.join(self.music_root, rel_path.replace("/", os.sep))

    # -- scan bookkeeping ----------------------------------------------------
    def needs_scan(self, abs_path, analysis_version):
        """True if this file is new or changed since its last analysis."""
        st = os.stat(abs_path)
        row = self.conn.execute(
            "SELECT file_size, mtime, analysis_version, error FROM tracks"
            " WHERE path = ?", (self.rel(abs_path),)).fetchone()
        if row is None:
            return True
        return (row["file_size"] != st.st_size
                or abs(row["mtime"] - st.st_mtime) > 1.0
                or (row["analysis_version"] or 0) < analysis_version)

    def upsert_track(self, abs_path, analysis, content_hash=None):
        """Write one track's full analysis (or an error record)."""
        st = os.stat(abs_path)
        a = dict(analysis)
        vals = {
            "path": self.rel(abs_path),
            "file_size": st.st_size, "mtime": st.st_mtime,
            "content_hash": content_hash,
            "title": a.get("title"), "artist": a.get("artist"),
            "album": a.get("album"), "file_genre": a.get("genre"),
            "duration_s": a.get("duration_s"),
            "bpm": a.get("bpm"), "bpm_conf": a.get("bpm_conf"),
            "downbeat_offset": a.get("downbeat_offset"),
            "downbeat_conf": a.get("downbeat_conf"),
            "key_pc": a.get("key_pc"), "key_mode": a.get("key_mode"),
            "camelot": a.get("camelot"), "key_conf": a.get("key_conf"),
            "key_name": a.get("key_name"),
            "loudness_gain_db": a.get("loudness_gain_db"),
            "kick_offset_s": a.get("kick_offset_s"),
            "phrase_beats": a.get("phrase_beats"),
            "phrase_start_s": a.get("phrase_start_s"),
            "phrase_conf": a.get("phrase_conf"),
            "rhythm_density": a.get("rhythm_density"),
            "analysis_version": a.get("analysis_version", 0),
            "analyzed_at": time.time(),
            "error": a.get("error"), "missing": 0,
        }
        for col in _SCAN_JSON_COLS:  # NOT enrichment/mood_ml/structure (passes)
            if a.get(col) is None and col in _KEEP_IF_NONE:
                continue                 # absent value must not wipe a backfill
            vals[col] = json.dumps(a[col]) if a.get(col) is not None else None
        cols = ", ".join(vals)
        marks = ", ".join("?" for _ in vals)
        updates = ", ".join(f"{c}=excluded.{c}" for c in vals if c != "path")
        self.conn.execute(
            f"INSERT INTO tracks ({cols}) VALUES ({marks})"
            f" ON CONFLICT(path) DO UPDATE SET {updates}",
            tuple(vals.values()))
        track_id = self.conn.execute(
            "SELECT id FROM tracks WHERE path = ?",
            (vals["path"],)).fetchone()["id"]
        for table in ("sections", "loops", "mix_points"):
            self.conn.execute(f"DELETE FROM {table} WHERE track_id = ?",
                              (track_id,))
        # Rescans refresh AUTO cues but never touch user-authored ones.
        self.conn.execute(
            "DELETE FROM cues WHERE track_id = ? AND source = 'auto'",
            (track_id,))
        for c in a.get("cues") or []:
            self.conn.execute(
                "INSERT INTO cues (track_id, kind, time_s, source, label)"
                " VALUES (?, ?, ?, 'auto', ?)",
                (track_id, c["kind"], c["time_s"], c.get("label")))
        for s in a.get("sections") or []:
            self.conn.execute(
                f"INSERT INTO sections (track_id, {', '.join(_SECTION_COLS)})"
                f" VALUES (?{', ?' * len(_SECTION_COLS)})",
                (track_id,) + tuple(s.get(c) for c in _SECTION_COLS))
        for l in a.get("loops") or []:
            self.conn.execute(
                "INSERT INTO loops (track_id, start_s, beats, score)"
                " VALUES (?, ?, ?, ?)",
                (track_id, l["start_s"], l["beats"], l.get("score")))
        for p in a.get("mix_points") or []:
            self.conn.execute(
                "INSERT INTO mix_points (track_id, kind, time_s, score,"
                " style_hint) VALUES (?, ?, ?, ?, ?)",
                (track_id, p["kind"], p["time_s"], p.get("score"),
                 p.get("style_hint")))
        self.conn.commit()
        return track_id

    def set_enrichment(self, track_id, blob):
        """Store a track's MusicBrainz enrichment blob (JSON). Genres/era
        surface via TrackInfo.all_tags in memory, so we never touch the
        user-authored `tags` table."""
        self.conn.execute("UPDATE tracks SET enrichment = ? WHERE id = ?",
                          (json.dumps(blob) if blob is not None else None,
                           track_id))
        self.conn.commit()

    def enrichment_for(self, track_id):
        row = self.conn.execute(
            "SELECT enrichment FROM tracks WHERE id = ?",
            (track_id,)).fetchone()
        if row and row["enrichment"]:
            try:
                return json.loads(row["enrichment"])
            except ValueError:
                return None
        return None

    def set_mood_ml(self, track_id, blob):
        """Store a track's Music2Emo ML descriptor blob (JSON): normalized
        valence/arousal (0..1) + predicted mood tags. Moods surface via
        TrackInfo.all_tags in memory; we never touch the user `tags` table."""
        self.conn.execute("UPDATE tracks SET mood_ml = ? WHERE id = ?",
                          (json.dumps(blob) if blob is not None else None,
                           track_id))
        self.conn.commit()

    def mood_ml_for(self, track_id):
        row = self.conn.execute(
            "SELECT mood_ml FROM tracks WHERE id = ?",
            (track_id,)).fetchone()
        if row and row["mood_ml"]:
            try:
                return json.loads(row["mood_ml"])
            except ValueError:
                return None
        return None

    def set_chroma(self, track_id, profile):
        """Store a track's 12-bin A-origin pitch-class profile (JSON list).
        Written by the chroma backfill tool; new scans write it inline via
        upsert_track (analyze_samples emits it since v12)."""
        self.conn.execute("UPDATE tracks SET chroma = ? WHERE id = ?",
                          (json.dumps(profile) if profile is not None
                           else None, track_id))
        self.conn.commit()

    def set_structure(self, track_id, blob):
        """Store a track's ML structure blob (JSON):
        {segments: [[start_s, end_s, label], ...], source: 'allin1'}.
        Separate pass, mirrors set_mood_ml."""
        self.conn.execute("UPDATE tracks SET structure = ? WHERE id = ?",
                          (json.dumps(blob) if blob is not None else None,
                           track_id))
        self.conn.commit()

    def set_rhythm(self, track_id, blob):
        """Store a track's beat-sync rhythm signature (JSON, lib/dj/rhythm).
        Written by the rhythm backfill tool (stem-derived when stems exist);
        new scans write a mix-derived one inline via upsert_track."""
        self.conn.execute("UPDATE tracks SET rhythm = ? WHERE id = ?",
                          (json.dumps(blob) if blob is not None else None,
                           track_id))
        self.conn.commit()

    def structure_for(self, track_id):
        row = self.conn.execute(
            "SELECT structure FROM tracks WHERE id = ?",
            (track_id,)).fetchone()
        if row and row["structure"]:
            try:
                return json.loads(row["structure"])
            except ValueError:
                return None
        return None

    def set_setlist_meta(self, setlist_id, theme=None, notes=None,
                         total_s=None, arc_json=None):
        """Update a setlist's plan-level metadata. Only the fields passed
        non-None are written; total_s/arc_json are what the live system
        uses to run the set's OWN arc clock instead of the generic night
        cycle (system.arc_progress)."""
        sets, vals = [], []
        for col, v in (("theme", theme), ("notes", notes),
                       ("total_s", total_s), ("arc_json", arc_json)):
            if v is not None:
                sets.append(f"{col} = ?")
                vals.append(v)
        if not sets:
            return
        sets.append("updated_at = ?")
        vals += [time.time(), setlist_id]
        self.conn.execute(
            f"UPDATE setlists SET {', '.join(sets)} WHERE id = ?",
            tuple(vals))
        self.conn.commit()

    def set_verified_tempo(self, track_id, bpm, beat_grid, conf):
        """Write back a live-verified tempo (system._verify_tempo's groove
        re-measure). The scanner's original bpm is preserved once in
        bpm_scan so the correction is reversible; every future library
        load (planner compile AND live) then starts from the measured
        value instead of re-discovering the same disagreement nightly."""
        self.conn.execute(
            "UPDATE tracks SET bpm_scan = COALESCE(bpm_scan, bpm),"
            " bpm = ?, bpm_conf = ?, beat_grid = ?,"
            " bpm_source = 'live_verified', bpm_verified_at = ?"
            " WHERE id = ?",
            (bpm, conf,
             json.dumps(beat_grid) if beat_grid is not None else None,
             time.time(), track_id))
        self.conn.commit()

    def coverage_counts(self):
        """Per-pass analysis coverage over non-missing tracks:
        {pass_name: (done, total)}. The GUI's "is my library ready?"
        answer - previously only reachable via each CLI's --stats."""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE missing = 0"
            " AND error IS NULL").fetchone()[0]
        out = {"tracks": (total, total)}
        for name, col in (("grid", "beat_grid"), ("key", "camelot"),
                          ("chroma", "chroma"), ("rhythm", "rhythm"),
                          ("enrich", "enrichment"), ("mood", "mood_ml"),
                          ("structure", "structure")):
            done = self.conn.execute(
                f"SELECT COUNT(*) FROM tracks WHERE missing = 0"
                f" AND error IS NULL AND {col} IS NOT NULL").fetchone()[0]
            out[name] = (done, total)
        return out

    def set_excluded(self, track_id, flag):
        """Mark/unmark a track 'do not use'. Excluded tracks stay in the DB and
        the library browser but are removed from everything that auto-selects
        (set generation, the copilot pool, and the live autoDJ brain)."""
        self.conn.execute("UPDATE tracks SET excluded = ? WHERE id = ?",
                          (1 if flag else 0, track_id))
        self.conn.commit()

    def mark_missing(self, present_rel_paths):
        """Flag DB rows whose file vanished (history preserved, not deleted)."""
        rows = self.conn.execute("SELECT id, path FROM tracks").fetchall()
        present = set(present_rel_paths)
        n = 0
        for r in rows:
            miss = 0 if r["path"] in present else 1
            self.conn.execute("UPDATE tracks SET missing = ? WHERE id = ?",
                              (miss, r["id"]))
            n += miss
        self.conn.commit()
        return n

    # -- queries -------------------------------------------------------------
    @staticmethod
    def _hydrate(row):
        if row is None:
            return None
        d = dict(row)
        for col in _JSON_COLS:
            if d.get(col):
                d[col] = json.loads(d[col])
        return d

    def get_track(self, track_id=None, rel_path=None):
        q = ("SELECT * FROM tracks WHERE id = ?" if track_id is not None
             else "SELECT * FROM tracks WHERE path = ?")
        row = self.conn.execute(
            q, (track_id if track_id is not None else rel_path,)).fetchone()
        return self._hydrate(row)

    def all_tracks(self, include_missing=False, include_errors=False):
        q = "SELECT * FROM tracks"
        conds = []
        if not include_missing:
            conds.append("missing = 0")
        if not include_errors:
            conds.append("error IS NULL")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        return [self._hydrate(r) for r in self.conn.execute(q).fetchall()]

    _SEC_NUM = ("start_s", "end_s", "energy", "bass_share", "mid_share",
                "high_share", "rhythm_density", "repetitiveness",
                "busyness", "vocalness", "boundary_strength")

    def sections_for(self, track_id):
        # Sanitize on READ: a handful of real-library tracks carry NULL
        # section metrics (edge-case analyses); every consumer does math
        # on these fields and None crashes them (drop_moments took down
        # tag recalibration for a whole 580-track scan).
        out = []
        for r in self.conn.execute(
                "SELECT * FROM sections WHERE track_id = ? ORDER BY start_s",
                (track_id,)).fetchall():
            d = dict(r)
            for k in self._SEC_NUM:
                if d.get(k) is None:
                    d[k] = 0.0
            out.append(d)
        return out

    def loops_for(self, track_id):
        out = []
        for r in self.conn.execute(
                "SELECT * FROM loops WHERE track_id = ? ORDER BY score DESC",
                (track_id,)).fetchall():
            d = dict(r)
            for k in ("start_s", "score"):
                if d.get(k) is None:
                    d[k] = 0.0
            out.append(d)
        return out

    def mix_points_for(self, track_id):
        out = []
        for r in self.conn.execute(
                "SELECT * FROM mix_points WHERE track_id = ?"
                " ORDER BY kind, score DESC", (track_id,)).fetchall():
            d = dict(r)
            for k in ("time_s", "score"):
                if d.get(k) is None:
                    d[k] = 0.0
            out.append(d)
        return out

    def counts(self):
        row = self.conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,"
            " SUM(missing) AS missing FROM tracks").fetchone()
        return {"total": row["total"], "errors": row["errors"] or 0,
                "missing": row["missing"] or 0}

    # -- user tags + cues --------------------------------------------------------
    def tags_for(self, track_id):
        return [r["tag"] for r in self.conn.execute(
            "SELECT tag FROM tags WHERE track_id = ? ORDER BY tag",
            (track_id,)).fetchall()]

    def add_tag(self, track_id, tag):
        tag = tag.strip().lower()
        if tag:
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (track_id, tag) VALUES (?, ?)",
                (track_id, tag))
            self.conn.commit()

    def remove_tag(self, track_id, tag):
        self.conn.execute("DELETE FROM tags WHERE track_id = ? AND tag = ?",
                          (track_id, tag))
        self.conn.commit()

    def all_tags(self):
        return [r["tag"] for r in self.conn.execute(
            "SELECT DISTINCT tag FROM tags ORDER BY tag").fetchall()]

    def cues_for(self, track_id, kind=None):
        q = "SELECT * FROM cues WHERE track_id = ?"
        args = [track_id]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        return [dict(r) for r in self.conn.execute(
            q + " ORDER BY time_s", args).fetchall()]

    def add_cue(self, track_id, kind, time_s, label=None, source="user"):
        cur = self.conn.execute(
            "INSERT INTO cues (track_id, kind, time_s, source, label)"
            " VALUES (?, ?, ?, ?, ?)",
            (track_id, kind, float(time_s), source, label))
        self.conn.commit()
        return cur.lastrowid

    def remove_cue(self, cue_id):
        self.conn.execute("DELETE FROM cues WHERE id = ?", (cue_id,))
        self.conn.commit()

    # -- play history ----------------------------------------------------------
    def log_play_start(self, track_id, transition_style=None, theme=None):
        cur = self.conn.execute(
            "INSERT INTO play_history (track_id, started_at,"
            " transition_style, theme) VALUES (?, ?, ?, ?)",
            (track_id, time.time(), transition_style, theme))
        self.conn.commit()
        return cur.lastrowid

    def add_seam_feedback(self, a_id, b_id, style, up, source="user",
                          ver=None):
        # `ver` stamps WHICH ENGINE produced the seam this verdict judged
        # (lib/dj/version). Without it, evidence from before a transition
        # change is indistinguishable from evidence after it.
        if ver is None:
            try:
                from lib.dj.version import tag
                ver = tag()
            except Exception:
                ver = None
        self.conn.execute(
            "INSERT INTO seam_feedback (a_id, b_id, style, up, at, source,"
            " ver) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (a_id, b_id, style, 1 if up else 0, time.time(), source, ver))
        self.conn.commit()

    def seam_feedback_rows(self, days=90.0):
        """Raw recent seam feedback (a_id, b_id, style, up, source) - the
        brain generalizes these into class/style memory (see
        Brain.load_pair_memory); pair_stats keeps the exact-pair view."""
        since = time.time() - days * 86400.0
        return [dict(r) for r in self.conn.execute(
            "SELECT a_id, b_id, style, up, source FROM seam_feedback"
            " WHERE at > ?", (since,))]

    def pair_stats(self, days=90.0):
        """Cross-night pair memory inputs: thumbs per (a,b) pair and
        skipped-shortly-after-swap counts derived from play_history."""
        since = time.time() - days * 86400.0
        fb = {}
        for r in self.conn.execute(
                "SELECT a_id, b_id, up, source FROM seam_feedback"
                " WHERE at > ?", (since,)):
            k = (r["a_id"], r["b_id"])
            # The machine's self-assessment counts, but at half an
            # operator's vote - measured train-wrecks lean the memory,
            # human taste steers it.
            w = 0.5 if (r["source"] or "user") == "auto" else 1.0
            ups, downs = fb.get(k, (0.0, 0.0))
            fb[k] = (ups + w * r["up"], downs + w * (1 - r["up"]))
        skips = {}
        rows = self.conn.execute(
            "SELECT track_id, started_at, ended_at, skipped FROM"
            " play_history WHERE started_at > ? ORDER BY started_at",
            (since,)).fetchall()
        for i in range(1, len(rows)):
            prev, cur = rows[i - 1], rows[i]
            # same session, consecutive, bailed within 90s of the seam
            if cur["skipped"] and cur["ended_at"]                     and cur["started_at"] - (prev["ended_at"] or 0) < 60.0                     and cur["ended_at"] - cur["started_at"] < 90.0:
                k = (prev["track_id"], cur["track_id"])
                skips[k] = skips.get(k, 0) + 1
        return fb, skips

    def log_play_end(self, history_id, skipped=False):
        self.conn.execute(
            "UPDATE play_history SET ended_at = ?, skipped = ? WHERE id = ?",
            (time.time(), 1 if skipped else 0, history_id))
        self.conn.commit()

    def recent_plays(self, hours=12.0):
        return [dict(r) for r in self.conn.execute(
            "SELECT h.*, t.artist FROM play_history h"
            " JOIN tracks t ON t.id = h.track_id"
            " WHERE h.started_at > ? ORDER BY h.started_at DESC",
            (time.time() - hours * 3600.0,)).fetchall()]
