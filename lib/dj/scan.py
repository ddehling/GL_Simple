"""Incremental music-library scanner.

Walks the music root, decides per file whether (size, mtime,
analysis_version) changed, analyzes only what's new/changed on a
multiprocessing pool, and writes results into the library DB from the
parent process (sqlite stays single-writer). A per-track failure becomes an
`error` row - one corrupt mp3 never kills a scan. Progress is mirrored to
logs/dj_scan_progress.json so UIs can poll it.
"""
import hashlib
import json
import os
import time
import traceback

from lib.dj.db import LibraryDB
from lib.dj.features import ANALYSIS_VERSION, analyze_file

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRESS_PATH = os.path.join(_REPO_ROOT, "logs", "dj_scan_progress.json")


def list_audio_files(music_root):
    out = []
    for dirpath, dirnames, filenames in os.walk(music_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def quick_hash(path):
    """Cheap content fingerprint: size + first/last 256 KB."""
    h = hashlib.md5()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(262144))
        if size > 524288:
            f.seek(-262144, os.SEEK_END)
            h.update(f.read(262144))
    return h.hexdigest()


def _analyze_worker(abs_path):
    """Pool worker: never raises - errors come back as a record."""
    try:
        result = analyze_file(abs_path, deep=True)
        return abs_path, result, quick_hash(abs_path)
    except Exception as e:
        return abs_path, {"error": f"{type(e).__name__}: {e}",
                          "analysis_version": ANALYSIS_VERSION,
                          "trace": traceback.format_exc(limit=3)}, None


def write_progress(payload):
    try:
        os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
        with open(PROGRESS_PATH, "w") as f:
            json.dump(payload, f)
    except OSError:
        pass


def scan_library(music_root, workers=None, force=False, progress_cb=None,
                 vocals_pass=True, refine_grids=False):
    """Run one incremental scan. Returns a summary dict.

    progress_cb(done, total, current_name) is called from the parent after
    every finished track (CLI bar, Qt signal, whatever). After the feature
    scan, tracks not yet measured by the ML vocal pass get one (sequential,
    resumable; skipped silently when torch/demucs aren't installed).

    refine_grids=True additionally RE-analyzes every unchanged track whose
    stored bpm_conf sits below 0.75: the grid estimator's confidence now
    weights window votes by comb strength (quiet stretches no longer dilute
    agreement), so the low-confidence band is exactly where a re-run can
    promote tracks back into the precision transition styles. Vocal data is
    preserved as on any rescan; the summary reports the conf movement."""
    db = LibraryDB(music_root)
    files = list_audio_files(music_root)
    rel_present = [db.rel(p) for p in files]
    n_missing = db.mark_missing(rel_present)

    todo = [p for p in files
            if force or db.needs_scan(p, ANALYSIS_VERSION)]
    prior_conf = {}
    if refine_grids and not force:
        seen = {os.path.normcase(p) for p in todo}
        for r in db.conn.execute(
                "SELECT path, bpm_conf FROM tracks WHERE error IS NULL"
                " AND missing = 0 AND bpm_conf IS NOT NULL"
                " AND bpm_conf < 0.75"):
            p = db.abs(r["path"])
            if os.path.normcase(p) not in seen and os.path.exists(p):
                todo.append(p)
                prior_conf[db.rel(p)] = float(r["bpm_conf"])
    summary = {"found": len(files), "scanned": 0, "errors": 0,
               "skipped": len(files) - len(todo), "missing": n_missing,
               "started_at": time.time()}
    write_progress({**summary, "total": len(todo), "done": 0,
                    "current": "", "finished": False})

    if todo:
        if workers is None:
            workers = max(1, (os.cpu_count() or 4) - 1)
        results = _run_pool(todo, workers)
        done = 0
        for abs_path, result, chash in results:
            result.pop("trace", None)
            vocal_snap = _vocal_snapshot(db, abs_path)
            db.upsert_track(abs_path, result, content_hash=chash)
            if vocal_snap and not result.get("error"):
                _vocal_restore(db, abs_path, vocal_snap)
            done += 1
            summary["scanned"] += 1
            if result.get("error"):
                summary["errors"] += 1
            name = os.path.basename(abs_path)
            if progress_cb:
                progress_cb(done, len(todo), name)
            write_progress({**summary, "total": len(todo), "done": done,
                            "current": name, "finished": False})

    if prior_conf:
        deltas, promoted = [], 0
        for rel, old in prior_conf.items():
            row = db.conn.execute(
                "SELECT bpm_conf FROM tracks WHERE path = ?"
                " AND error IS NULL", (rel,)).fetchone()
            if row is None or row["bpm_conf"] is None:
                continue
            new = float(row["bpm_conf"])
            deltas.append(new - old)
            if old < 0.7 <= new:      # crossed the precision-style gate
                promoted += 1
        if deltas:
            summary["grid_refine"] = {
                "refined": len(deltas),
                "improved": sum(1 for d in deltas if d > 0.02),
                "regressed": sum(1 for d in deltas if d < -0.02),
                "mean_delta": round(sum(deltas) / len(deltas), 3),
                "promoted_070": promoted,
            }

    if vocals_pass:
        summary["vocals"] = vocal_pass(db, progress_cb=progress_cb,
                                       summary=summary, total=len(todo))
    _recalibrate_tags(db)
    summary["elapsed_s"] = round(time.time() - summary["started_at"], 1)
    summary["db_counts"] = db.counts()
    write_progress({**summary, "total": len(todo), "done": len(todo),
                    "current": "", "finished": True})
    db.close()
    return summary


def _vocal_snapshot(db, abs_path):
    """The ML vocal pass costs ~1 min/track of CPU - an analysis-version
    rescan must NOT wipe it. Snapshot the measured vocal data before the
    upsert (which rebuilds sections) so it can be re-applied."""
    row = db.conn.execute("SELECT id, axes FROM tracks WHERE path = ?",
                          (db.rel(abs_path),)).fetchone()
    if row is None or not row["axes"]:
        return None
    axes = json.loads(row["axes"])
    if not axes.get("vocal_src"):
        return None
    secs = db.conn.execute(
        "SELECT start_s, vocalness FROM sections WHERE track_id = ?",
        (row["id"],)).fetchall()
    return {"vocal": axes.get("vocal", 0.0), "src": axes["vocal_src"],
            "secs": [(r["start_s"], r["vocalness"]) for r in secs]}


def _vocal_restore(db, abs_path, snap):
    """Re-apply snapshotted vocal data onto the freshly-rebuilt rows.
    Sections are matched by start_s (same audio, same boundary logic);
    an unmatched new section falls back to the nearest old one."""
    row = db.conn.execute("SELECT id, axes FROM tracks WHERE path = ?",
                          (db.rel(abs_path),)).fetchone()
    if row is None:
        return
    axes = json.loads(row["axes"]) if row["axes"] else {}
    axes["vocal"] = snap["vocal"]
    axes["vocal_src"] = snap["src"]
    db.conn.execute("UPDATE tracks SET axes = ? WHERE id = ?",
                    (json.dumps(axes), row["id"]))
    if snap["secs"]:
        for sec in db.conn.execute(
                "SELECT id, start_s FROM sections WHERE track_id = ?",
                (row["id"],)).fetchall():
            old = min(snap["secs"], key=lambda s: abs(s[0] - sec["start_s"]))
            db.conn.execute("UPDATE sections SET vocalness = ? WHERE id = ?",
                            (old[1], sec["id"]))
    db.conn.commit()


def vocal_pass(db, progress_cb=None, summary=None, total=0):
    """ML vocal measurement for every track that doesn't have one yet.

    Sequential in this process (torch parallelizes internally; a Pool of
    model copies would just fight over cores and RAM) and resumable: each
    track commits on completion, and tracks with axes.vocal_src are
    skipped, so an interrupted pass continues where it stopped."""
    import json as _json
    from lib.dj import vocals
    if not vocals.available():
        return {"status": "unavailable",
                "hint": "pip install -r requirements-dj-vocals.txt"}
    rows = db.conn.execute(
        "SELECT id, path, axes FROM tracks"
        " WHERE error IS NULL AND missing = 0 AND axes IS NOT NULL"
        " ORDER BY path").fetchall()
    todo = [r for r in rows
            if not _json.loads(r["axes"]).get("vocal_src")]
    if not todo:
        return {"status": "done", "measured": 0}
    from lib.dj.features import decode_file
    analyzer = vocals.VocalAnalyzer()
    measured = errors = 0
    base = summary or {}
    for i, r in enumerate(todo):
        name = os.path.basename(r["path"])
        write_progress({**base, "total": total, "done": total,
                        "phase": "vocals", "vocals_total": len(todo),
                        "vocals_done": i, "current": name,
                        "finished": False})
        try:
            samples = decode_file(db.abs(r["path"]))
            analyzer.update_track(db, r["id"], samples)
            measured += 1
        except Exception as e:
            errors += 1
            print(f"  vocal pass failed on {name}: {type(e).__name__}: {e}")
        if progress_cb:
            progress_cb(i + 1, len(todo), f"vocals: {name}")
    return {"status": "done", "measured": measured, "errors": errors}


def _recalibrate_tags(db):
    """Library-relative auto tags. The per-track axes are built from
    per-track-normalized features, so absolute thresholds saturate (every
    club track reads 'hard' against a silence baseline). After a scan we
    re-derive comparative tags from PERCENTILES across the collection -
    'hard' means hard FOR THIS LIBRARY."""
    import json as _json
    import numpy as _np
    from lib.dj.vocals import vocal_tags
    rows = db.conn.execute(
        "SELECT id, bpm, axes, mood_hist, spectral, rhythm_density,"
        " key_mode, duration_s, energy_curve"
        " FROM tracks WHERE error IS NULL AND axes IS NOT NULL").fetchall()
    user_tags = {}
    for r in db.conn.execute("SELECT track_id, tag FROM tags"):
        user_tags.setdefault(r["track_id"], set()).add(r["tag"])
    if len(rows) < 8:
        # Tiny library: percentile tags are meaningless, but vocal tags
        # come from real per-track measurements - update just those.
        for r in rows:
            a = _json.loads(r["axes"])
            old = _json.loads(db.conn.execute(
                "SELECT auto_tags FROM tracks WHERE id = ?",
                (r["id"],)).fetchone()["auto_tags"] or "[]")
            tags = [t for t in old
                    if t not in ("vocal-heavy", "vocals", "instrumental")]
            tags.extend(vocal_tags(a, user_tags.get(r["id"], ())))
            db.conn.execute("UPDATE tracks SET auto_tags = ? WHERE id = ?",
                            (_json.dumps(tags), r["id"]))
        db.conn.commit()
        return
    axes = {r["id"]: _json.loads(r["axes"]) for r in rows}
    # The stored hardness axis clips at 1.0 and saturates on club music -
    # rebuild an UNCLIPPED score from stored ingredients so percentiles
    # actually discriminate.
    hard_raw = {}
    for r in rows:
        spec = _json.loads(r["spectral"] or "{}")
        mood = _json.loads(r["mood_hist"] or "{}")
        hard_raw[r["id"]] = (2.5 * spec.get("bass_share", 0.33)
                             + 0.8 * mood.get("peak", 0.0)
                             + 0.25 * min((r["rhythm_density"] or 0) / 3.0,
                                          1.5)
                             + 0.4 * axes[r["id"]].get("speed", 0.5))
    def pct(vals, q):
        return float(_np.percentile(list(vals), q))
    hi_hard = pct(hard_raw.values(), 72)
    lo_hard = pct(hard_raw.values(), 28)
    hi_en = pct((a.get("energy", 0.5) for a in axes.values()), 72)
    lo_en = pct((a.get("energy", 0.5) for a in axes.values()), 28)
    hi_hyp = pct((a.get("hypnotic", 0.5) for a in axes.values()), 80)

    # STRUCTURAL/TEXTURAL metrics from analysis we already store - the
    # auto vocabulary was ~11 thin tags, which dumped all real tagging on
    # the user. Everything here is library-relative (percentiles) or a
    # hard structural fact, so each tag discriminates on THIS collection.
    from lib.dj.features import drop_moments
    met = {}
    for r in rows:
        secs = db.sections_for(r["id"])
        dur = max(r["duration_s"] or 1.0, 1.0)
        intro = next((x for x in secs if x["kind"] == "intro"), None)
        bdown = sum(x["end_s"] - x["start_s"] for x in secs
                    if x["kind"] in ("breakdown", "build"))
        n_loops = db.conn.execute(
            "SELECT COUNT(*) c FROM loops WHERE track_id = ?",
            (r["id"],)).fetchone()["c"]
        curve = _json.loads(r["energy_curve"] or "[]")
        spec = _json.loads(r["spectral"] or "{}")
        met[r["id"]] = {
            "drops": len(drop_moments(secs)),
            "intro_s": (intro["end_s"] - intro["start_s"]) if intro else 0.0,
            "bdown_frac": bdown / dur,
            "n_loops": n_loops,
            "dyn": float(_np.std(curve)) if len(curve) > 8 else 0.0,
            "bass": spec.get("bass_share", 0.33),
            "midsh": spec.get("mid_share", 0.33),
            "high": spec.get("high_share", 0.25),
            "dens": r["rhythm_density"] or 0.0,
            "dur": dur,
        }
    def th(key, q):
        return pct((m[key] for m in met.values()), q)
    hi_dyn, lo_dyn = th("dyn", 75), th("dyn", 25)
    hi_bd = th("bdown_frac", 75)
    hi_loop = th("n_loops", 75)
    hi_bass, hi_mid, hi_high = th("bass", 75), th("midsh", 75), th("high", 75)
    lo_high = th("high", 25)
    hi_dens, lo_dens = th("dens", 75), th("dens", 25)
    hi_dur, lo_dur = th("dur", 78), th("dur", 22)
    for r in rows:
        a = axes[r["id"]]
        mood = _json.loads(r["mood_hist"] or "{}")
        tags = list(vocal_tags(a, user_tags.get(r["id"], ())))
        bpm = r["bpm"] or 0
        if bpm and bpm < 105:
            tags.append("slow")
        elif bpm and bpm > 123:
            tags.append("fast")
        if hard_raw[r["id"]] > hi_hard:
            tags.append("hard")
        elif hard_raw[r["id"]] < lo_hard:
            tags.append("gentle")
        if a.get("energy", 0.5) >= hi_en:
            tags.append("driving")
        elif a.get("energy", 0.5) <= lo_en:
            tags.append("mellow")
        if a.get("hypnotic", 0.5) >= hi_hyp:
            tags.append("hypnotic")
        if mood.get("peak", 0.0) > 0.25:
            tags.append("peaky")
        m = met[r["id"]]
        if r["key_mode"] in ("minor", "major"):
            tags.append(r["key_mode"])
        if m["drops"] >= 2:
            tags.append("drops")
        elif m["drops"] == 0:
            tags.append("no-drops")
        if m["intro_s"] >= 45.0:
            tags.append("long-intro")
        if m["dyn"] >= hi_dyn:
            tags.append("dynamic")
        elif m["dyn"] <= lo_dyn:
            tags.append("steady")
        if m["bdown_frac"] >= hi_bd:
            tags.append("breakdowny")
        if m["n_loops"] > hi_loop:
            tags.append("loopable")
        if m["bass"] >= hi_bass:
            tags.append("bass-heavy")
        if m["midsh"] >= hi_mid:
            tags.append("melodic")
        if m["high"] >= hi_high:
            tags.append("bright")
        elif m["high"] <= lo_high:
            tags.append("warm")
        if m["dens"] >= hi_dens:
            tags.append("percussive")
        elif m["dens"] <= lo_dens:
            tags.append("sparse")
        if m["dur"] >= hi_dur:
            tags.append("epic")
        elif m["dur"] <= lo_dur:
            tags.append("short")
        db.conn.execute("UPDATE tracks SET auto_tags = ? WHERE id = ?",
                        (_json.dumps(tags), r["id"]))
    db.conn.commit()


def _run_pool(todo, workers):
    """Yield worker results; falls back to in-process when workers==1."""
    if workers <= 1 or len(todo) == 1:
        for p in todo:
            yield _analyze_worker(p)
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(todo))) as pool:
        for r in pool.imap_unordered(_analyze_worker, todo):
            yield r
