"""Project-local shader loader.

The shared library at ``renderer/effects/`` auto-imports every ``shader_*``
function and ``*Effect`` class on package init (see ``renderer/effects/
__init__.py``) and exposes them as attributes of the package — so callers
do ``from renderer import effects as fx; fx.shader_kelp(...)`` regardless
of which file inside the package defined the function.

This module extends the same convention to project-local shaders living
under ``projects/<id>/shaders/``. After ``load_project_shaders(project)``
runs:

  * every ``shader_*`` function and ``*Effect`` class found in the
    project's ``shaders/`` directory is imported into the
    ``renderer.effects`` namespace so existing ``fx.shader_xxx`` call
    sites resolve transparently;
  * the project's ``shaders`` package is added to Python's import path
    so its modules can do relative imports between sibling shaders
    (``from .bart_map import _haversine_km``) the same way they did when
    they all lived in ``renderer/effects/``.

This keeps the ``fx.shader_xxx`` API stable across the file moves.
Non-active projects' shaders are NOT loaded, so nothing leaks across
projects: a swap that goes Fan → WoL must call ``unload_project_shaders``
first to drop Fan's project-local symbols, then load WoL's (empty
today, but the API is symmetric).
"""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys

_currently_loaded: set[str] = set()  # ids we've added to renderer.effects


def load_project_shaders(project) -> None:
    """Auto-import the active project's project-local shaders into the
    ``renderer.effects`` namespace.

    Idempotent for a given project — re-calling for the same project is
    a no-op. Calling for a different project first unloads the previous
    project's symbols, so swapping projects never leaves a previous
    project's shader on the ``fx`` namespace.
    """
    if project.id in _currently_loaded:
        return
    _unload_all()

    shaders_dir = project.shaders_root
    if not shaders_dir.exists():
        return

    package_name = f"projects.{project.id}.shaders"
    try:
        package = importlib.import_module(package_name)
    except Exception as e:
        print(f"[ShaderLoader] failed to import {package_name}: {e}")
        return

    import renderer.effects as fx

    for _, mod_name, is_pkg in pkgutil.iter_modules([str(shaders_dir)]):
        if is_pkg or mod_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{package_name}.{mod_name}")
        except Exception as e:
            print(f"[ShaderLoader] {package_name}.{mod_name} failed to import: {e}")
            continue

        for attr_name in dir(module):
            if attr_name.startswith("shader_") or (
                attr_name.endswith("Effect") and attr_name != "ShaderEffect"
            ):
                attr = getattr(module, attr_name)
                # Set on the renderer.effects module so existing
                # ``fx.shader_xxx`` references resolve.
                setattr(fx, attr_name, attr)
                _currently_loaded_attrs.setdefault(project.id, []).append(attr_name)

    _currently_loaded.add(project.id)
    count = len(_currently_loaded_attrs.get(project.id, []))
    if count:
        print(f"[ShaderLoader] Loaded {count} project-local symbols from {package_name}")


# Per-project list of attrs we set on renderer.effects, so we can reverse them
# cleanly on swap.
_currently_loaded_attrs: dict[str, list[str]] = {}


def _unload_all() -> None:
    """Reverse every previous load_project_shaders call: pop attrs we set
    on renderer.effects and forget the cached project-shader modules.
    """
    if not _currently_loaded:
        return
    import renderer.effects as fx
    for pid, attrs in _currently_loaded_attrs.items():
        for name in attrs:
            if hasattr(fx, name):
                delattr(fx, name)
        # Drop the project's shader modules so a re-import sees fresh code.
        prefix = f"projects.{pid}.shaders"
        stale = [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]
        for k in stale:
            del sys.modules[k]
    _currently_loaded.clear()
    _currently_loaded_attrs.clear()
