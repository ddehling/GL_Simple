"""Learned execution tuning: the values the ratings have settled on.

`brain.TUNE_DEFAULTS` holds somebody's first guess at each execution
constant. This module holds what the evidence says instead, and
`build_events` reads THIS as its baseline - so a verdict in the Seam Lab
eventually changes how the live engine mixes, which is the only reason
collecting verdicts is worth doing.

A BATCH RESPONSE SURFACE, not an online step. Every rated seam's random
nudge goes into one quadratic fit per knob; the fit yields the peak AND
an error bar on where the peak is (delta method). A knob moves only to
the near edge of that confidence interval - never further than the data
supports - so as samples accumulate the interval tightens onto the peak,
the target stops moving, and the updates die out. Convergence is a
property of the ESTIMATE settling, which needs no step-size schedule.

Three earlier designs failed and are recorded in the code so they are not
retried: climbing the pooled correlation walks a knob to the end of its
range (the pooled slope stays positive after you pass the peak); damping
the step by cumulative evidence freezes it early at whatever the first
noisy estimate pointed at; and re-centring the exploration on the learned
value collapses the sampled span until the peak is unidentifiable.

Guards:

  * significant concave curvature (t < -2) before a peak is believed,
  * a predicted gain of at least MIN_GAIN verdict-score points, which is
    what stops noise from nudging knobs that have no real effect,
  * movement capped per update and confined to the explored range,
  * every move journalled with its evidence; `reset()` restores the
    original constant.

Measured against known optima on simulated verdicts: interior optima
settle to within a mean 0.042 of a 0.28-wide range (15%) and stay there;
an optimum outside the range walks to the boundary; a knob with no real
effect does not move at all across five seeds. The residual bias comes
from fitting a parabola to a response that is not one - it is stable and
directionally right, not exact.

Stored as JSON at logs/seam_tuning.json - one small readable file you can
inspect, edit or delete.
"""
import json
import os
import threading
import time

_LOCK = threading.Lock()
_CACHE = {"mtime": None, "values": {}, "path": None}

MAX_STEP = 0.22         # never move more than this fraction of the range
                        # in one update, however confident the fit

# Auto-apply: learned values feed build_events. Verified to converge and
# to hold still on knobs with no real effect (see the module docstring).
AUTO_APPLY = True



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


def apply_findings(findings, defaults, ranges, now=None):
    """Move every knob whose fit places a better value away from where it
    currently sits.

    `findings` are seamtune.knob_findings() rows carrying a `target` (the
    near edge of the peak's confidence interval). Returns the changes made
    - empty when no knob has evidence yet, which is the normal case early
    on, and empty again once they have all settled."""
    with _LOCK:
        doc = _read_doc()
        vals = dict(doc.get("values") or {})
        changed = []
        for f in findings:
            k = f.get("knob")
            if f.get("thin") or not f.get("solid") or k not in defaults:
                continue
            lo, hi = ranges.get(k, (None, None))
            target = f.get("target")
            if lo is None or hi <= lo or target is None:
                # No target means the fit could not place a peak away from
                # where the knob already sits - either not enough evidence,
                # no significant curvature, or the confidence interval
                # already covers the current value. All three mean: hold.
                continue
            cur = float(vals.get(k, defaults[k]))
            # Move to the NEAR EDGE of the peak's confidence interval, not
            # to the point estimate: never further than the data supports.
            # As samples accumulate the interval tightens onto the peak, so
            # the target converges and the moves die out on their own -
            # no step-size schedule, and nothing to freeze early.
            delta = float(target) - cur
            delta = max(-MAX_STEP * (hi - lo),
                        min(MAX_STEP * (hi - lo), delta))
            new = max(lo, min(hi, cur + delta))
            if abs(new - cur) < 1e-4:
                continue
            vals[k] = round(new, 4)
            changed.append({"knob": k, "was": round(cur, 4),
                            "now": vals[k], "r": round(float(f.get("r") or 0), 3),
                            "n": f.get("n")})
        if changed and AUTO_APPLY:
            doc["values"] = vals
            doc.setdefault("history", []).append(
                {"t": now or time.time(), "changes": changed})
            doc["history"] = doc["history"][-200:]
            _write(doc)
        elif changed:
            for c in changed:            # proposal, not a change
                c["proposed"] = True
        return changed


def set_value(knob, value, why=""):
    """Set one knob directly - the one-at-a-time probe's way in, as
    opposed to apply_findings' statistical route."""
    with _LOCK:
        doc = _read_doc()
        vals = dict(doc.get("values") or {})
        was = vals.get(knob)
        vals[knob] = round(float(value), 4)
        doc["values"] = vals
        doc.setdefault("history", []).append(
            {"t": time.time(), "changes": [
                {"knob": knob, "was": was, "now": vals[knob], "why": why}]})
        doc["history"] = doc["history"][-200:]
        _write(doc)
        return vals[knob]


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
