"""Learned execution tuning: the values the ratings have settled on.

`brain.TUNE_DEFAULTS` holds somebody's first guess at each execution
constant. This module holds what the evidence says instead, and
`build_events` reads THIS as its baseline - so a verdict in the Seam Lab
eventually changes how the live engine mixes, which is the only reason
collecting verdicts is worth doing.

Kept deliberately conservative:

  * a knob only moves when its randomised evidence clears 2 sigma,
  * it moves a damped FRACTION of the way to the suggestion (a noisy
    estimate that is followed all the way oscillates instead of
    converging),
  * it can never leave the narrow sampling range it was explored in,
  * every move is journalled with the evidence behind it, and
    `reset()` restores the original constant.

Stored as JSON at logs/seam_tuning.json - one small readable file you can
inspect, edit or delete.
"""
import json
import os
import threading
import time

_LOCK = threading.Lock()
_CACHE = {"mtime": None, "values": {}, "path": None}

# GRADIENT STEP, not a jump to an estimated optimum. The correlation
# between nudge and verdict IS the local gradient, and it vanishes at the
# optimum, so stepping along it converges and then stops. (Moving toward
# a score-weighted "best value" was tried first and crawls: that estimator
# only shifts by the asymmetry in the good-rate, so a knob 0.10 off its
# true optimum moved 0.005 per session.)
STEP = 0.15             # fraction of the explored range per unit of r
MAX_STEP = 0.22         # ...and never more than this fraction in one go


def path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs", "seam_tuning.json")


def _load():
    """{knob: value} as last written. Cheap: re-reads only on mtime change
    (build_events calls this per seam)."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        _CACHE.update(mtime=None, values={}, path=p)
        return {}
    if _CACHE["mtime"] == m and _CACHE["path"] == p:
        return _CACHE["values"]
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        vals = {k: float(v) for k, v in (doc.get("values") or {}).items()}
    except (OSError, ValueError, TypeError):
        vals = {}
    _CACHE.update(mtime=m, values=vals, path=p)
    return vals


def value(name, default):
    """The learned value for `name`, or `default` when nothing is learned."""
    return _load().get(name, default)


def current(defaults):
    """Full knob table: learned values over the supplied defaults."""
    out = dict(defaults)
    out.update({k: v for k, v in _load().items() if k in defaults})
    return out


def _read_doc():
    try:
        with open(path(), encoding="utf-8") as f:
            doc = json.load(f)
            if isinstance(doc, dict):
                return doc
    except (OSError, ValueError):
        pass
    return {"values": {}, "history": []}


def _write(doc):
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, p)
    _CACHE["mtime"] = None          # force a re-read


def apply_findings(findings, defaults, ranges, step=STEP, now=None):
    """Move every knob with solid randomised evidence toward its suggestion.

    `findings` are seamtune.knob_findings() rows. Returns the list of
    changes made, each {knob, was, now, r, n} - empty when nothing had
    enough evidence, which is the normal case early on."""
    with _LOCK:
        doc = _read_doc()
        vals = dict(doc.get("values") or {})
        changed = []
        for f in findings:
            k = f.get("knob")
            if f.get("thin") or not f.get("solid") or k not in defaults:
                continue
            r = float(f.get("r") or 0.0)
            lo, hi = ranges.get(k, (None, None))
            if lo is None or hi <= lo or not r:
                continue
            cur = float(vals.get(k, defaults[k]))
            delta = max(-MAX_STEP, min(MAX_STEP, step * r)) * (hi - lo)
            new = max(lo, min(hi, cur + delta))
            if abs(new - cur) < 1e-4:
                continue
            vals[k] = round(new, 4)
            changed.append({"knob": k, "was": round(cur, 4),
                            "now": vals[k], "r": round(r, 3),
                            "n": f.get("n")})
        if changed:
            doc["values"] = vals
            doc.setdefault("history", []).append(
                {"t": now or time.time(), "changes": changed})
            doc["history"] = doc["history"][-200:]
            _write(doc)
        return changed


def history(limit=12):
    return list(reversed((_read_doc().get("history") or [])[-limit:]))


def reset(knob=None):
    """Forget one knob (or everything) and go back to the constant."""
    with _LOCK:
        doc = _read_doc()
        vals = dict(doc.get("values") or {})
        if knob is None:
            vals = {}
        else:
            vals.pop(knob, None)
        doc["values"] = vals
        doc.setdefault("history", []).append(
            {"t": time.time(), "reset": knob or "all"})
        _write(doc)
