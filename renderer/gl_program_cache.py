"""Process-wide GLSL compile memoization — spawn-hitch eliminator.

Every event spawn builds a fresh ShaderEffect whose compile_shader() runs
driver compiles ON THE RENDER THREAD — 10-100ms per program on deployment
GPUs. The same GLSL sources get recompiled every time the event respawns
(weather cycles, club one-shots on drops...), which reads as a visual
hitch exactly when motion is fastest.

Importing this module (BEFORE anything star-imports OpenGL.GL — see the
top of Stories_OGL.py) patches PyOpenGL so:

- ``shaders.compileShader`` / ``shaders.compileProgram`` memoize by
  (GL context, source text / shader ids): the first spawn pays the
  driver compile, every respawn reuses the linked program. Subclass
  compile_shader() side effects (extra programs stored on self, uniform
  lookups...) still run — they just receive cached handles.
- ``glDeleteShader`` / ``glDeleteProgram`` skip cache-owned objects, so
  one instance's cleanup can't destroy a program other instances (or
  future spawns) share. Dynamic per-instance programs that bypass
  ``shaders.compileProgram`` (e.g. the live-shader slots' raw
  glCreateShader path) are untouched and delete normally.

Uniform locations are per-program, so base.ShaderEffect's uniform cache
works unchanged with shared programs.
"""
import time

from OpenGL import platform as _glplatform
import OpenGL.GL as _GL
from OpenGL.GL import shaders as _glshaders

_real_compileShader = _glshaders.compileShader
_real_compileProgram = _glshaders.compileProgram
_real_deleteShader = _GL.glDeleteShader
_real_deleteProgram = _GL.glDeleteProgram

_shader_cache = {}      # (ctx, source, type) -> shader object
_program_cache = {}     # (ctx, shader ids, opts) -> program object
_owned_shaders = set()  # ids the cache owns — deletion is a no-op
_owned_programs = set()
_stats = {'compiles': 0, 'links': 0, 'hits': 0, 'compile_ms': 0.0}


def _ctx():
    try:
        return int(_glplatform.GetCurrentContext() or 0)
    except Exception:
        return 0


def _src_key(source):
    if isinstance(source, (list, tuple)):
        return tuple(_src_key(s) for s in source)
    if isinstance(source, bytes):
        return source
    return str(source)


def _cached_compileShader(source, shaderType, **kw):
    key = (_ctx(), _src_key(source), int(shaderType))
    sh = _shader_cache.get(key)
    if sh is not None:
        _stats['hits'] += 1
        return sh
    t0 = time.perf_counter()
    sh = _real_compileShader(source, shaderType, **kw)
    _stats['compiles'] += 1
    _stats['compile_ms'] += (time.perf_counter() - t0) * 1000.0
    _shader_cache[key] = sh
    _owned_shaders.add(int(sh))
    return sh


def _cached_compileProgram(*shs, **named):
    key = (_ctx(), tuple(int(s) for s in shs),
           tuple(sorted(named.items())))
    prog = _program_cache.get(key)
    if prog is not None:
        _stats['hits'] += 1
        return prog
    t0 = time.perf_counter()
    prog = _real_compileProgram(*shs, **named)
    _stats['links'] += 1
    _stats['compile_ms'] += (time.perf_counter() - t0) * 1000.0
    _program_cache[key] = prog
    _owned_programs.add(int(prog))
    return prog


def _guarded_deleteShader(sh):
    try:
        if int(sh) in _owned_shaders:
            return None
    except (TypeError, ValueError):
        pass
    return _real_deleteShader(sh)


def _guarded_deleteProgram(prog):
    try:
        if int(prog) in _owned_programs:
            return None
    except (TypeError, ValueError):
        pass
    return _real_deleteProgram(prog)


def stats():
    """For /api/perf: hits vs. driver compiles and total compile time."""
    return dict(_stats)


_glshaders.compileShader = _cached_compileShader
_glshaders.compileProgram = _cached_compileProgram
_GL.glDeleteShader = _guarded_deleteShader
_GL.glDeleteProgram = _guarded_deleteProgram
print("[GLCache] shader/program compile memoization installed")
