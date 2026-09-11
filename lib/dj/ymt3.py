"""YourMT3+ as an `other` reader: notes that arrive already labelled with an
instrument (GM program), from a multi-instrument transcription model, run in
its own interpreter (tools/dj/ymt3_transcribe.py in the WSL venv ~/ymt3venv).

    transcribe(y) -> [(onset_s, offset_s, midi, program, is_drum, velocity)] or None

Why: the `other` stem's parts landing in shared voices ("the wrong instrument
on a part") was measured as the reconstruction's largest audible defect, and
no clustering rule separated them; a model trained to name the instrument
per note is the direct answer, judged on the truth set like every reader.
"""
import json
import os
import subprocess
import tempfile

import numpy as np

RATE = 44100
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tools", "dj", "ymt3_transcribe.py")
WSL_PYTHON = os.environ.get("YMT3_PYTHON", "~/ymt3venv/bin/python")
TIMEOUT_S = 1800


def _wsl_path(win_path):
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def available():
    return os.path.isfile(SCRIPT) and os.path.isdir(os.path.join(ROOT, "models", "yourmt3", "space"))


CACHE_DIR = os.path.join(ROOT, "logs", "ymt3_cache")   # a stem's transcription, keyed by its audio: a re-read of the
                                                       # same track (a version bump, a variant under test) skips the
                                                       # model's minute per stem


def _cache_key(a):
    import hashlib
    h = hashlib.sha1()
    h.update(str(len(a)).encode())
    h.update(np.ascontiguousarray(a[:: max(1, len(a) // 65536)]).astype(np.float32).tobytes())
    return h.hexdigest()[:24]


def transcribe(y, progress=None):
    import soundfile as sf
    if not available():
        return None
    a = np.asarray(y, dtype=np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, _cache_key(a) + ".json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            d = json.load(fh)
        if progress:
            progress(f"other: YourMT3+ {len(d['notes'])} notes (cached)")
        return [(n["onset"], n["offset"], n["pitch"], n["program"], n["is_drum"], n["velocity"]) for n in d["notes"]]
    tmp_dir = os.path.join(ROOT, "logs", "ymt3_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, wav = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
    os.close(fd)
    out = wav[:-4] + ".json"
    try:
        sf.write(wav, np.asarray(y, dtype=np.float32), RATE)
        if os.name == "nt":
            cmd = ["wsl", "-e", "bash", "-lc", f"{WSL_PYTHON} {_wsl_path(SCRIPT)} {_wsl_path(wav)} {_wsl_path(out)}"]
        else:
            cmd = [os.path.expanduser(WSL_PYTHON), SCRIPT, wav, out]
        if progress:
            progress("other: YourMT3+ transcription (its own interpreter)")
        try:
            # the other process needs the card: release this one's cached allocations first (the context stays)
            import sys as _sys
            torch = _sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        if r.returncode != 0 or not os.path.isfile(out):
            if progress:
                progress("other: YourMT3+ failed - " + (r.stderr or r.stdout).strip().splitlines()[-1][:120] if (r.stderr or r.stdout).strip() else "other: YourMT3+ failed")
            return None
        with open(out, encoding="utf-8") as fh:
            d = json.load(fh)
        try:
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(d, fh)
        except OSError:
            pass
        if progress:
            progress(f"other: YourMT3+ {len(d['notes'])} notes, {len(d.get('programs') or {})} programs in {d.get('seconds', 0):.0f} s")
        return [(n["onset"], n["offset"], n["pitch"], n["program"], n["is_drum"], n["velocity"]) for n in d["notes"]]
    finally:
        for p in (wav, out):
            try:
                os.remove(p)
            except OSError:
                pass
