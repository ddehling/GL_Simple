"""Compile-test EVERY effect class in every project repo (plus engine
infrastructure) in a hidden GL context.

Broader sibling of _club_shader_compile_test: walks projects/*/shaders/,
imports each module, instantiates each *Effect class against a stub
viewport and calls compile_shader(). Catches broken imports (e.g. a
module still importing a shader that moved repos) and GLSL-ES failures
without launching the app.

Classes whose constructors require extra arguments are reported as
skips, not failures — add explicit entries for those if they need
coverage.

Usage: python tools/_all_shader_compile_test.py
Exit code 0 + "ALL PASS" when every instantiable shader compiles.
"""
import os
import sys
import importlib
import inspect
import pkgutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glfw
from OpenGL.GL import glGetString, GL_VERSION


class _StubViewport:
    width = 128
    height = 300


def main():
    if not glfw.init():
        print("FAIL: glfw.init()")
        sys.exit(1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "all-shader-compile-test", None, None)
    if win is None:
        print("FAIL: could not create a hidden GL context")
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(win)
    print(f"GL context: {glGetString(GL_VERSION).decode()}\n")

    from renderer.effects.base import ShaderEffect

    roots = []
    projects_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "projects")
    for pid in sorted(os.listdir(projects_dir)):
        folder = os.path.join(projects_dir, pid, "shaders")
        if os.path.isdir(folder):
            roots.append((f"projects.{pid}.shaders", folder))

    n_pass = 0
    skips = []
    failures = []
    vp = _StubViewport()
    for pkg_name, folder in roots:
        for _, mod_name, _ in pkgutil.iter_modules([folder]):
            if mod_name.startswith("_"):
                continue
            full = f"{pkg_name}.{mod_name}"
            try:
                mod = importlib.import_module(full)
            except Exception as e:
                failures.append(f"{full}: IMPORT {type(e).__name__}: {e}")
                continue
            for name, cls in inspect.getmembers(mod, inspect.isclass):
                if not (name.endswith("Effect") and name != "ShaderEffect"):
                    continue
                if not issubclass(cls, ShaderEffect) or cls.__module__ != full:
                    continue
                try:
                    eff = cls(vp)
                except TypeError:
                    skips.append(f"{full}.{name}")
                    continue
                except Exception as e:
                    failures.append(f"{full}.{name}: CTOR {type(e).__name__}: {e}")
                    continue
                try:
                    shader = eff.compile_shader()
                    if shader:
                        n_pass += 1
                    else:
                        skips.append(f"{full}.{name} (no shader)")
                except Exception as e:
                    failures.append(
                        f"{full}.{name}: COMPILE {type(e).__name__}: {str(e)[:200]}")

    glfw.terminate()
    print(f"\ncompiled OK: {n_pass}")
    if skips:
        print(f"skipped (ctor needs args / no shader): {len(skips)}")
        for s in skips:
            print(f"  skip: {s}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
