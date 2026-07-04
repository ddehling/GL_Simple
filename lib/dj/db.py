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

SCHEMA_VERSION = 2
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
    energy_curve TEXT,                  -- JSON, 2 Hz, 0..1.2
    mood_hist TEXT,                     -- JSON {mood: fraction}
    rhythm_density REAL,
    spectral TEXT,                      -- JSON {bass/mid/high_share}
    live_check TEXT,                    -- JSON live-pipeline cross-check
    axes TEXT,                          -- JSON v2 character axes (0..1)
    auto_tags TEXT,                     -- JSON v2 derived tag list
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
    created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS setlist_entries (
    id INTEGER PRIMARY KEY,
    setlist_id INTEGER NOT NULL REFERENCES setlists(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    pin_type TEXT NOT NULL DEFAULT 'suggestion',   -- 'anchor' | 'suggestion'
    target_offset_min REAL,             -- anchor's desired minutes into the set
    style_override TEXT
);
CREATE INDEX IF NOT EXISTS idx_setlist_entries ON setlist_entries(setlist_id, position);
"""

_JSON_COLS = ("beat_grid", "energy_curve", "mood_hist", "spectral",
              "live_check", "axes", "auto_tags")

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
            if ver == 1:                         # v1 -> v2: new track columns
                have = {r[1] for r in
                        self.conn.execute("PRAGMA table_info(tracks)")}
                for col in ("axes", "auto_tags"):
                    if col not in have:
                        self.conn.execute(
                            f"ALTER TABLE tracks ADD COLUMN {col} TEXT")
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
            "album": a.get("album"),
            "duration_s": a.get("duration_s"),
            "bpm": a.get("bpm"), "bpm_conf": a.get("bpm_conf"),
            "downbeat_offset": a.get("downbeat_offset"),
            "downbeat_conf": a.get("downbeat_conf"),
            "key_pc": a.get("key_pc"), "key_mode": a.get("key_mode"),
            "camelot": a.get("camelot"), "key_conf": a.get("key_conf"),
            "key_name": a.get("key_name"),
            "loudness_gain_db": a.get("loudness_gain_db"),
            "rhythm_density": a.get("rhythm_density"),
            "analysis_version": a.get("analysis_version", 0),
            "analyzed_at": time.time(),
            "error": a.get("error"), "missing": 0,
        }
        for col in _JSON_COLS:
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

    def sections_for(self, track_id):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM sections WHERE track_id = ? ORDER BY start_s",
            (track_id,)).fetchall()]

    def loops_for(self, track_id):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM loops WHERE track_id = ? ORDER BY score DESC",
            (track_id,)).fetchall()]

    def mix_points_for(self, track_id):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM mix_points WHERE track_id = ?"
            " ORDER BY kind, score DESC", (track_id,)).fetchall()]

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
