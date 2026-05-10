"""Project loader.

A Project bundles everything that is specific to one art piece: groups,
receivers, geometry, weather sets, event map, sounds, sensors. Phase 1 of
the multi-project refactor only wires the content-module pointers
(weather_sets_module, event_map_module); group/receiver/geometry fields
will populate later phases.

Selection: the active project id comes from (in order)
  1. ``--project <id>`` on the command line, if present
  2. ``project: <id>`` in the top-level config.yaml
  3. ``"fan"`` as the default
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml


PROJECTS_ROOT = Path(__file__).resolve().parent.parent / "projects"


@dataclass
class GroupSpec:
    """One logical canvas owned by a project. Effects render here; DMX
    strips sample from here."""
    id: str
    width: int
    height: int


@dataclass
class Project:
    id: str
    display_name: str
    weather_sets_module: str
    event_map_module: str
    groups: list = field(default_factory=list)  # list[GroupSpec]
    hooks: dict = field(default_factory=dict)   # capabilities (random_events_module, etc.)
    raw: dict = field(default_factory=dict)
    # Brightness setpoint (0..1) used by RenderPipeline's per-frame
    # limiter. Per-project so each piece can match its actual power
    # supply budget — Fan and WoL have very different LED counts and
    # PSU capacities. ``None`` means "use the pipeline default" so
    # legacy projects without the key keep their existing behaviour.
    brightness_limit: float | None = None
    # Target render frames per second. Read from project.yaml's
    # ``target_fps`` key. ``None`` means "use the engine default"
    # (40 fps). Per-project so a piece with heavy-render shaders or
    # bandwidth-limited DDP receivers can dial FPS down without
    # affecting other pieces.
    target_fps: float | None = None

    @property
    def root(self) -> Path:
        return PROJECTS_ROOT / self.id

    @property
    def media_root(self) -> Path:
        """Project-local media folder. Effects resolve relative paths
        (e.g. weather sets' ``narrative_script: 'sounds/foo/script.json'``)
        against this root. Created on first access if missing."""
        path = self.root / "media"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def shaders_root(self) -> Path:
        """Project-local shader package directory. Empty by default;
        project-specific effects live here. The shared
        ``renderer/effects/`` library remains importable for shared
        effects regardless of active project."""
        return self.root / "shaders"

    def load_event_map(self) -> dict:
        mod = importlib.import_module(self.event_map_module)
        return mod.EVENT_MAP

    def load_weather_module(self):
        return importlib.import_module(self.weather_sets_module)

    def load_hook(self, name: str):
        """Resolve and import a project capability hook by its key in the
        project.yaml ``hooks:`` block. Returns the imported module, or
        None if the project doesn't declare that hook.

        Example use::

            random_events_mod = project.load_hook("random_events")
            if random_events_mod is not None:
                random_events_mod.run(env_system)

        This pattern keeps Stories_OGL.py free of project-specific event
        scheduling — projects describe their own behavior via the hooks
        block.
        """
        spec = self.hooks.get(name)
        if not spec:
            return None
        try:
            return importlib.import_module(spec)
        except Exception as e:
            print(f"[Project] hook {name!r} import failed ({spec}): {e}")
            return None

    def group_to_frame_id(self) -> dict[str, int]:
        """Map group id → its index in ``self.groups`` (= the frame_id
        used by the existing event scheduler / effects API)."""
        return {g.id: i for i, g in enumerate(self.groups)}

    def load_geometry(self):
        """Construct the active GeometryProvider from the project's
        ``geometry:`` block. Defaults to a FanGeometryProvider with
        baked-in fan-installation defaults if nothing is specified.
        """
        from core.geometry.fan import FanGeometryProvider
        from core.geometry.multi_object import MultiObjectGeometryProvider

        block = (self.raw.get("geometry") or {}) if isinstance(self.raw, dict) else {}
        gtype = block.get("type", "fan")
        kwargs = {k: v for k, v in block.items() if k != "type"}
        if gtype == "fan":
            return FanGeometryProvider(**kwargs)
        if gtype == "multi_object":
            file_path = kwargs.pop("file", None)
            if file_path is None:
                raise ValueError(
                    f"multi_object geometry for project {self.id!r} requires a 'file:' "
                    "pointing at the geometry YAML."
                )
            # Resolve relative to the repo root (PROJECTS_ROOT.parent).
            full_path = (PROJECTS_ROOT.parent / file_path).resolve()
            return MultiObjectGeometryProvider(full_path, **kwargs)
        raise ValueError(
            f"Unknown geometry type {gtype!r} in project {self.id!r}. "
            "Known types: 'fan', 'multi_object'."
        )


def list_projects() -> list[dict]:
    """Scan ``projects/`` for project.yaml files and return their headers.

    Returns a list of ``{"id": ..., "display_name": ...}`` sorted by
    display name. Used by the web UI to populate the project switcher.
    Failed YAMLs are skipped with a warning rather than crashing.
    """
    out: list[dict] = []
    if not PROJECTS_ROOT.exists():
        return out
    for child in sorted(PROJECTS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        yaml_path = child / "project.yaml"
        if not yaml_path.exists():
            continue
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Project] list_projects: skipping {yaml_path}: {e}")
            continue
        pid = raw.get("id", child.name)
        out.append({"id": pid, "display_name": raw.get("display_name", pid)})
    out.sort(key=lambda d: d["display_name"].lower())
    return out


def load_project(project_id: str) -> Project:
    yaml_path = PROJECTS_ROOT / project_id / "project.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Project config not found: {yaml_path}. "
            f"Available: {[p.name for p in PROJECTS_ROOT.iterdir() if p.is_dir()]}"
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    pid = raw.get("id", project_id)

    # Groups: explicit ``groups:`` list wins. Otherwise fall back to a
    # single "main" group sized from the project's ``display:`` block (or
    # the legacy 128x300 default), so Fan-style projects without groups
    # keep working without edits.
    raw_groups = raw.get("groups")
    groups: list[GroupSpec] = []
    if isinstance(raw_groups, list):
        for g in raw_groups:
            if not isinstance(g, dict):
                continue
            groups.append(GroupSpec(
                id=str(g["id"]),
                width=int(g["width"]),
                height=int(g["height"]),
            ))
    if not groups:
        disp = raw.get("display", {}) if isinstance(raw, dict) else {}
        groups = [GroupSpec(
            id="main",
            width=int(disp.get("width", 128)),
            height=int(disp.get("height", 300)),
        )]

    hooks = raw.get("hooks") or {}
    if not isinstance(hooks, dict):
        hooks = {}

    bl = raw.get("brightness_limit")
    brightness_limit = float(bl) if bl is not None else None

    fps = raw.get("target_fps")
    target_fps = float(fps) if fps is not None else None

    return Project(
        id=pid,
        display_name=raw.get("display_name", pid),
        weather_sets_module=raw.get("weather_sets_module", f"projects.{pid}.weather_params"),
        event_map_module=raw.get("event_map_module", f"projects.{pid}.event_map"),
        groups=groups,
        hooks=hooks,
        raw=raw,
        brightness_limit=brightness_limit,
        target_fps=target_fps,
    )
