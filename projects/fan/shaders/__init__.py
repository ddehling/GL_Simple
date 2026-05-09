"""Fan-project-local shader effects.

Shaders authored specifically for the Fan installation live here. The
shared shader library at ``renderer/effects/`` is still importable —
projects can mix shared + project-local effects in their event_map.

Today this directory is empty (all current shaders live in
``renderer/effects/`` and are project-agnostic). New Fan-only effects
go here so they don't pollute the shared library or the WoL project.
"""
