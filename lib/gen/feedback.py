"""Preference memory: what the operator liked and did not, so the night
drifts toward their taste. Thumbs on the /gen page (or the director's
'more like this') record a snapshot of the musical state; bias() pulls
steering parameters toward the centroid of liked snapshots and away from
disliked ones. Persisted in logs/gen_prefs.json, bounded, and always a
NUDGE - never a lock - so the arc and the form stay in charge."""
from __future__ import annotations

import json
import os
import time

PARAMS = ("energy", "density", "swing", "brightness")


class PreferenceMemory:
    def __init__(self, path="logs/gen_prefs.json", capacity=400):
        self.path = path
        self.cap = capacity
        self.items = []
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                self.items = list(json.load(fh))[-self.cap:]
        except Exception:
            self.items = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.items[-self.cap:], fh)
        except Exception:
            pass

    def record(self, snapshot: dict, up: bool):
        rec = {"t": time.time(), "up": bool(up)}
        rec.update({k: snapshot.get(k) for k in ("style", "section", "key", "mode", "layers", "pattern_slots")})
        rec.update({k: float(snapshot.get(k)) for k in PARAMS if snapshot.get(k) is not None})
        self.items.append(rec)
        self.items = self.items[-self.cap:]
        self._save()
        return rec

    def counts(self):
        return {"up": sum(1 for r in self.items if r["up"]), "down": sum(1 for r in self.items if not r["up"])}

    def bias(self, style: str, strength: float = 0.35) -> dict:
        """Parameter deltas toward liked / away from disliked snapshots of
        this style. Empty when there is no evidence."""
        ups = [r for r in self.items if r["up"] and r.get("style") == style]
        downs = [r for r in self.items if not r["up"] and r.get("style") == style]
        if len(ups) + len(downs) < 2:
            return {}
        out = {}
        for k in PARAMS:
            u = [r[k] for r in ups if k in r]
            d = [r[k] for r in downs if k in r]
            if not u and not d:
                continue
            target = (sum(u) / len(u)) if u else None
            avoid = (sum(d) / len(d)) if d else None
            out[k] = {"target": target, "avoid": avoid, "n_up": len(u), "n_down": len(d)}
        # liked layers -> which slots to favour
        if ups:
            from collections import Counter
            c = Counter(s for r in ups for s in (r.get("layers") or []))
            out["favoured_layers"] = [s for s, n in c.most_common(6)]
        return out

    def section_bias(self, style: str) -> dict:
        """{section: weight multiplier} for the form grammar: sections the
        operator liked get up to 2x, disliked down to 0.5x. Empty without
        evidence. The form multiplies its grammar weights by this - a
        nudge on the dice, never a lock."""
        ups, downs = {}, {}
        for r in self.items:
            if r.get("style") != style or not r.get("section"):
                continue
            (ups if r["up"] else downs)[r["section"]] = (ups if r["up"] else downs).get(r["section"], 0) + 1
        out = {}
        for sec in set(ups) | set(downs):
            u, d = ups.get(sec, 0), downs.get(sec, 0)
            if u + d == 0:
                continue
            out[sec] = float(max(0.5, min(2.0, (1.0 + u) / (1.0 + d))))
        return out

    def nudges(self, style: str, current: dict, strength: float = 0.35) -> dict:
        """Concrete deltas to apply now: move each param a fraction toward
        the liked centroid and away from the disliked one."""
        b = self.bias(style)
        out = {}
        for k in PARAMS:
            info = b.get(k)
            if not info or current.get(k) is None:
                continue
            cur = float(current[k])
            delta = 0.0
            if info["target"] is not None:
                delta += (info["target"] - cur) * strength
            if info["avoid"] is not None and abs(info["avoid"] - cur) < 0.15:
                delta += (cur - info["avoid"]) * strength if cur != info["avoid"] else 0.1 * strength
            if abs(delta) > 1e-4:
                out[k] = delta
        return out
