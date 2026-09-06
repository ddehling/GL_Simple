"""Scenes: named snapshots of the steering surface (style, tempo, key,
energy bias, density, swing, brightness, level, muted layers, patterns)
that the operator saves from the page and recalls later. Stored as JSON
next to the night logs; applied through the same phrase-boundary steering
as everything else (an Intent), so recalling a scene is auditable."""
from __future__ import annotations

import json
import os
import time

SCENE_KEYS = ("style", "bpm", "key", "energy_bias", "density", "swing", "brightness",
              "master", "muted", "pattern", "slot_patterns")


class SceneStore:
    def __init__(self, path="logs/gen_scenes.json"):
        self.path = path
        self.scenes = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                self.scenes = dict(json.load(fh))
        except Exception:
            self.scenes = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.scenes, fh, indent=1)
        except Exception:
            pass

    def save(self, name, snapshot):
        name = str(name).strip()[:40]
        if not name:
            raise ValueError("scene needs a name")
        rec = {k: snapshot.get(k) for k in SCENE_KEYS}
        rec["saved_at"] = time.time()
        self.scenes[name] = rec
        self._save()
        return rec

    def get(self, name):
        return self.scenes.get(name)

    def delete(self, name):
        if name in self.scenes:
            del self.scenes[name]
            self._save()
            return True
        return False

    def listing(self):
        return [{"name": n, "style": r.get("style"), "bpm": r.get("bpm"), "key": r.get("key"),
                 "saved_at": r.get("saved_at")} for n, r in sorted(self.scenes.items())]


def snapshot(system):
    """The steering surface of a live GenSystem as a scene record."""
    st = system.status()
    return {"style": st["style"], "bpm": st["bpm"], "key": st["camelot"],
            "energy_bias": st["energy_bias"], "density": st["density"], "swing": st["swing"],
            "brightness": st["brightness"], "master": st["master"], "muted": list(st["muted"]),
            "pattern": st.get("pattern"), "slot_patterns": dict(system._slot_patterns)}


def scene_intent(scene, slots):
    """A scene as an Intent for lib.gen.director.apply_intent."""
    intent = {"say": "scene recalled", "set": {}}
    for k in ("energy_bias", "density", "swing", "brightness", "master", "bpm"):
        if scene.get(k) is not None:
            intent["set"][k] = float(scene[k])
    if scene.get("key"):
        intent["set"]["key"] = scene["key"]
    muted = set(scene.get("muted") or [])
    intent["layers"] = {"mute": sorted(muted & set(slots)), "unmute": sorted(set(slots) - muted)}
    if scene.get("slot_patterns"):
        intent["patterns"] = dict(scene["slot_patterns"])
    intent["pattern"] = scene.get("pattern") or ""
    return intent
