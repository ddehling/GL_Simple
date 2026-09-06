"""One-shot sample library: a manifest of named samples the styles can
reference as "oneshots:<name>" in a sample slot's "file".

Search order (first manifest that has the name wins):
  1. the active project's  projects/<id>/media/gen/oneshots/manifest.json
  2. the shared            media/oneshots/manifest.json

manifest.json: {"kick_a": {"file": "kick_a.wav", "base_midi": 36, "tags": ["kick"]}, ...}
(file paths relative to the manifest's folder). tools/gen/oneshots.py
scans a folder into a manifest, or bootstraps a starter set rendered
from the rack's own voices so the sample path works on a fresh box."""
from __future__ import annotations

import json
import os

_cache = None


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))          # lib/gen/synth
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _active_project():
    try:
        import yaml
        with open(os.path.join(_repo_root(), "active_project.yaml"), encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        return d.get("active") or d.get("project") or d.get("id")
    except Exception:
        return None


def manifests():
    """[(folder, manifest dict)] in search order."""
    global _cache
    if _cache is not None:
        return _cache
    out = []
    root = _repo_root()
    cands = []
    pid = _active_project()
    if pid:
        cands.append(os.path.join(root, "projects", str(pid), "media", "gen", "oneshots"))
    cands.append(os.path.join(root, "media", "oneshots"))
    for folder in cands:
        path = os.path.join(folder, "manifest.json")
        try:
            with open(path, encoding="utf-8") as fh:
                out.append((folder, json.load(fh)))
        except Exception:
            continue
    _cache = out
    return out


def reload():
    global _cache
    _cache = None


def resolve(ref: str):
    """"oneshots:<name>" -> (absolute file path, base_midi) or (None, 60)
    when unknown; a plain path is returned as-is."""
    if not ref:
        return None, 60
    if not ref.startswith("oneshots:"):
        return ref, 60
    name = ref.split(":", 1)[1].strip()
    for folder, man in manifests():
        entry = man.get(name)
        if entry:
            return os.path.join(folder, entry.get("file", name + ".wav")), int(entry.get("base_midi", 60))
    return None, 60


def names(tag: str | None = None):
    out = []
    for _, man in manifests():
        for k, v in man.items():
            if tag is None or tag in (v.get("tags") or []):
                if k not in out:
                    out.append(k)
    return out
