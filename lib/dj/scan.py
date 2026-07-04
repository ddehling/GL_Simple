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


def scan_library(music_root, workers=None, force=False, progress_cb=None):
    """Run one incremental scan. Returns a summary dict.

    progress_cb(done, total, current_name) is called from the parent after
    every finished track (CLI bar, Qt signal, whatever)."""
    db = LibraryDB(music_root)
    files = list_audio_files(music_root)
    rel_present = [db.rel(p) for p in files]
    n_missing = db.mark_missing(rel_present)

    todo = [p for p in files
            if force or db.needs_scan(p, ANALYSIS_VERSION)]
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
            db.upsert_track(abs_path, result, content_hash=chash)
            done += 1
            summary["scanned"] += 1
            if result.get("error"):
                summary["errors"] += 1
            name = os.path.basename(abs_path)
            if progress_cb:
                progress_cb(done, len(todo), name)
            write_progress({**summary, "total": len(todo), "done": done,
                            "current": name, "finished": False})

    summary["elapsed_s"] = round(time.time() - summary["started_at"], 1)
    summary["db_counts"] = db.counts()
    write_progress({**summary, "total": len(todo), "done": len(todo),
                    "current": "", "finished": True})
    db.close()
    return summary


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
