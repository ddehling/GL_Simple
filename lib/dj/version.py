"""Engine fingerprint, stamped on every piece of learning evidence.

A verdict describes the audio the engine produced AT THE TIME. When the
transition code or the tuned constants change, older verdicts may
describe something the engine no longer plays - and silently mixing the
eras makes every statistic slightly wrong in a way nothing reveals.

Three parts, because they change at different rates and you want to
segment on different ones:

  code   - git HEAD short sha, plus '+dirty' when the working tree has
           uncommitted changes (during active development that is most
           of the time, so it is a coarse marker, not a guarantee)
  knobs  - hash of the EFFECTIVE execution knob table (defaults with the
           learned tuning applied), so a knob that moves marks a new era
  stretch - which time-stretch engine rendered it

Cheap enough to call per rating: the git lookup runs once per process,
the knob hash is a dict hash of ~39 floats.
"""
import hashlib
import os
import subprocess

_CODE = None


def _git_code():
    global _CODE
    if _CODE is not None:
        return _CODE
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        _CODE = (sha or "nogit") + ("+dirty" if dirty else "")
    except Exception:
        _CODE = "nogit"
    return _CODE


def knob_hash():
    """Short hash of the knob table the engine is actually mixing with."""
    try:
        from lib.dj import tuning
        from lib.dj.brain import TUNE_DEFAULTS
        eff = tuning.current(TUNE_DEFAULTS)
    except Exception:
        return "unknown"
    blob = ";".join(f"{k}={eff[k]:.6g}" for k in sorted(eff))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def engine_version():
    """{'code','knobs','stretch'} - stamp this on every rating."""
    try:
        from lib.dj import stretch_engine_name
        stretch = stretch_engine_name()
    except Exception:
        stretch = "?"
    return {"code": _git_code(), "knobs": knob_hash(), "stretch": stretch}


def tag(ver=None):
    """One short string for a DB column / grouping key."""
    v = ver or engine_version()
    return f"{v['code']}/{v['knobs']}"
