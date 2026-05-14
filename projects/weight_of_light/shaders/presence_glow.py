"""WoL ground presence glow — extra brightness on a box when its
LD2412 radar detects something close, with a hue/shimmer character
driven by the static-vs-motion split.

Pattern B (fullscreen-quad atmospheric layer). Per-fragment the
shader reads the ``object_id`` atlas, looks up three per-object
signals in small uniform arrays, and emits a tinted glow:

  * **closeness** (0..1) — base alpha, derived from the radar's
    ``gate_peak_idx``: 1.0 at gate 0 (≤0.75 m), 0.0 at gate 13 (≥9.75 m).
    Empty rooms (``state == 0``) and offline sensors clamp to 0.

  * **motion_ratio** (0..1) — `sum(gm_above_baseline) /
    sum(gm_above_baseline + gs_above_baseline)`. Pulls the glow's
    hue between two anchor colors: 0 = fully ``color_warm`` (a still
    person reads as a warm settled glow), 1 = fully ``color_cool``
    (a walking person reads as a cool active glow).

  * **motion_amp** (0..1) — total motion energy normalized by
    ``motion_norm``. Drives a sinusoidal shimmer on the alpha so
    walking people produce a visibly modulated wash and standing
    people produce a steady one. Per-object phase offsets stop
    adjacent boxes shimmering in lockstep.

Smoothing: each of the three signals is EMA-smoothed CPU-side (the
LD2412 emits at ~5 Hz; without smoothing the visuals would step on
each packet). Time constants chosen so closeness and the shimmer
amplitude track motion within a fraction of a second while the hue
blend takes a beat to settle, making the warm/cool transition feel
more deliberate than reactive.

Reads ``state['radar']`` populated by ``projects.weight_of_light.radar``
each frame. If radar isn't installed or no record exists for an
object, that object's signals stay at 0 (no glow), which is the
right fail-safe.

Render priority 5.0 — above the back layer (ground_twinkle=1.0) and
the rainbow (4.0), so the glow reads as a foreground "presence
spotlight" while still composing nicely with the rest of the Ground
stack via the engine's standard alpha blend.
"""
from __future__ import annotations

import ctypes
import math
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


# Fixed cap on simultaneously-tracked objects. WoL has 9; keep some
# headroom so a project can grow without recompiling. Uniform-array
# size in the GLSL must match.
_MAX_OBJECTS = 16

# LD2412 reports detection in 14 range gates of 0.75 m each.
# gate 0 is closest, gate 13 is farthest. Mirror the constant from
# radar.py so the closeness math doesn't drift if that constant
# ever changes (would be silent otherwise — peak_idx is just an
# int, not self-describing about its range).
_N_GATES = 14


def shader_wol_presence_glow(state, outstate,
                             color_warm=(1.0, 0.55, 0.15),
                             color_cool=(0.20, 0.55, 1.0),
                             max_alpha: float = 1.0,
                             shimmer_freq: float = 1.6,
                             shimmer_max_amp: float = 0.18,
                             motion_norm: float = 150.0,
                             color_pivot_low: float = 0.10,
                             color_pivot_high: float = 0.55):
    """Extra brightness on each box's ground arc that brightens with
    radar closeness, shifts color with the static/motion split, and
    shimmers when motion is high.

    ``color_warm`` is the hue when the detection is purely static
    (a person standing); ``color_cool`` is the hue when it's purely
    moving (walking). The shader uses smoothstep over the
    ``motion_ratio`` between ``color_pivot_low`` and
    ``color_pivot_high`` to drive the lerp — so a real-world
    motion_ratio range of ~0.1–0.6 (a walking person rarely hits
    1.0 because the body always produces some static return) still
    gets the full warm→cool shift instead of barely budging from
    the midpoint.

    ``max_alpha`` caps the peak opacity (1.0 = fully opaque at the
    closest gate; drop to e.g. 0.6 for a subtler haze).

    ``shimmer_freq`` is the shimmer's temporal frequency in Hz;
    ``shimmer_max_amp`` is the multiplicative modulation depth at
    full motion (0 = no shimmer; 0.18 ≈ subtle breathing — earlier
    defaults of 0.40 × 3 Hz read as flicker through the brightness
    limiter, so shipping calmer numbers).

    ``motion_norm`` is the total above-baseline motion energy that
    maps to ``motion_amp = 1.0``; tune to the typical walking-
    person reading on your install.
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        # Resolve the active group's per-pixel object_id atlas. The
        # atlas tells the fragment shader which box it's painting,
        # so it can look up the right closeness slot.
        group_ids = getattr(shader_renderer, 'group_ids', None) or []
        if 0 <= frame_id < len(group_ids):
            gid = group_ids[frame_id]
        else:
            gid = "Ground"
        meta = (outstate.get("group_metadata") or {}).get(gid)
        if meta is None or meta.get("object_id") is None:
            print(f"[wol_presence_glow] no object_id atlas on group "
                  f"{gid!r}; effect inert")
            return
        try:
            eff = viewport.add_effect(
                WolPresenceGlowEffect,
                object_id_atlas=meta["object_id"],
                color_warm=tuple(color_warm),
                color_cool=tuple(color_cool),
                max_alpha=float(max_alpha),
                shimmer_freq=float(shimmer_freq),
                shimmer_max_amp=float(shimmer_max_amp),
                motion_norm=float(motion_norm),
                color_pivot_low=float(color_pivot_low),
                color_pivot_high=float(color_pivot_high),
            )
            state['effect'] = eff
            print(f"[wol_presence_glow] init on frame {frame_id} "
                  f"(gate-driven closeness, motion-driven hue + shimmer)")
        except Exception as e:
            print(f"[wol_presence_glow] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        # Refresh per-object RAW signals from the radar state. The
        # effect's update() smooths these into the values the shader
        # actually consumes — keeping the wrapper a thin pass-through.
        # Untouched slots (object_id beyond range, or radar
        # unavailable) stay at 0.
        radar = outstate.get('radar') or {}
        raw_close = [0.0] * _MAX_OBJECTS
        raw_ratio = [0.0] * _MAX_OBJECTS
        raw_amp   = [0.0] * _MAX_OBJECTS
        for oid, rec in radar.items():
            if not isinstance(oid, int) or oid < 0 or oid >= _MAX_OBJECTS:
                continue
            close, ratio, amp = _signals_for(rec, eff.motion_norm)
            raw_close[oid] = close
            raw_ratio[oid] = ratio
            raw_amp[oid]   = amp
        eff.raw_closeness   = raw_close
        eff.raw_motion_ratio = raw_ratio
        eff.raw_motion_amp  = raw_amp

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[wol_presence_glow] cleaned up on frame {frame_id}")


def _signals_for(rec: dict, motion_norm: float) -> tuple[float, float, float]:
    """Compute (closeness, motion_ratio, motion_amp) for one record.

    closeness: 1 - gate_peak_idx / 13. 0 if offline or state == 0.
    motion_ratio: sum(gm_above_baseline) / sum(both above-baseline arrays).
        0 if no above-baseline energy at all (avoids divide-by-zero).
    motion_amp: sum(gm_above_baseline) / motion_norm, clamped 0..1.

    All three return 0 when the sensor is offline or the radar
    reports state == 0 — the down-stream shader uses closeness as
    its alpha, so zeroing it cleanly removes any contribution."""
    if not rec.get('online'):
        return 0.0, 0.0, 0.0
    if int(rec.get('state', 0)) == 0:
        return 0.0, 0.0, 0.0

    peak_idx = int(rec.get('gate_peak_idx', _N_GATES - 1))
    if peak_idx < 0:
        peak_idx = 0
    elif peak_idx > _N_GATES - 1:
        peak_idx = _N_GATES - 1
    closeness = 1.0 - peak_idx / float(_N_GATES - 1)

    gs = rec.get('gs_above_baseline') or [0] * _N_GATES
    gm = rec.get('gm_above_baseline') or [0] * _N_GATES
    sum_s = float(sum(gs))
    sum_m = float(sum(gm))
    total = sum_s + sum_m
    motion_ratio = (sum_m / total) if total > 1e-3 else 0.0
    motion_amp = sum_m / max(motion_norm, 1e-3)
    if motion_amp < 0.0: motion_amp = 0.0
    elif motion_amp > 1.0: motion_amp = 1.0

    return closeness, motion_ratio, motion_amp


class WolPresenceGlowEffect(ShaderEffect):
    # Per-signal EMA time constants (seconds). Tuned by feel:
    #   * closeness reacts inside half a second so people approaching
    #     the box feel "noticed" promptly;
    #   * motion_amp follows similarly so the shimmer matches the
    #     observed activity level;
    #   * motion_ratio is slower so the warm/cool hue blend feels
    #     deliberate rather than twitching with every gait step.
    _TAU_CLOSENESS_S    = 0.30
    _TAU_MOTION_AMP_S   = 0.30
    _TAU_MOTION_RATIO_S = 0.50

    def __init__(self, viewport,
                 object_id_atlas: np.ndarray,
                 color_warm: tuple = (1.0, 0.55, 0.15),
                 color_cool: tuple = (0.20, 0.55, 1.0),
                 max_alpha: float = 1.0,
                 shimmer_freq: float = 1.6,
                 shimmer_max_amp: float = 0.18,
                 motion_norm: float = 150.0,
                 color_pivot_low: float = 0.10,
                 color_pivot_high: float = 0.55):
        super().__init__(viewport)
        # Above ground_twinkle (1.0) and rainbow (4.0). Reads as a
        # presence spotlight on top of the Ground stack.
        self.render_priority = 5.0

        self.color_warm = tuple(float(c) for c in color_warm)
        self.color_cool = tuple(float(c) for c in color_cool)
        self.max_alpha = float(max_alpha)
        self.shimmer_freq = float(shimmer_freq)
        self.shimmer_max_amp = float(shimmer_max_amp)
        self.motion_norm = float(motion_norm)
        self.color_pivot_low = float(color_pivot_low)
        self.color_pivot_high = float(color_pivot_high)

        # Wrapper writes RAW values each frame; update() advances the
        # smoothed values via per-signal EMAs.
        self.raw_closeness    = [0.0] * _MAX_OBJECTS
        self.raw_motion_ratio = [0.0] * _MAX_OBJECTS
        self.raw_motion_amp   = [0.0] * _MAX_OBJECTS
        # Smoothed → uploaded as uniforms each render.
        self.closeness    = [0.0] * _MAX_OBJECTS
        self.motion_ratio = [0.0] * _MAX_OBJECTS
        self.motion_amp   = [0.0] * _MAX_OBJECTS

        # CPU-integrated shimmer time. Mod 1e6 (~11.5 days) so float32
        # precision on the GPU stays comfortable at long uptimes.
        self._time = 0.0

        # object_id atlas → R32F texture. -1 marks "no strip", which
        # the fragment shader skips.
        self._atlas = np.ascontiguousarray(object_id_atlas, dtype=np.float32)
        self.atlas_tex = None

        self.position_VBO = None
        self.EBO = None
        self._vertices = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)
        self._indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        layout(location = 0) in vec2 position;
        out vec2 vUV;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vUV = position * 0.5 + 0.5;
        }
        """

    def get_fragment_shader(self):
        # Uniform-array sizes MUST match _MAX_OBJECTS on the Python
        # side. GLSL ES 310 supports dynamic indexing into uniform
        # arrays (no constant-index restriction), so the atlas's
        # variable object_id directly indexes the per-object signal
        # arrays.
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform sampler2D uAtlas;             // R32F per-pixel object_id (-1 empty)
        uniform float u_closeness[16];        // per-object 0..1 closeness
        uniform float u_motion_ratio[16];     // per-object 0..1 motion fraction
        uniform float u_motion_amp[16];       // per-object 0..1 shimmer amplitude
        uniform float u_time;                 // CPU-integrated seconds
        uniform float u_shimmer_freq;         // Hz
        uniform float u_shimmer_max_amp;      // peak modulation depth
        uniform float u_color_pivot_low;      // motion_ratio that still reads warm
        uniform float u_color_pivot_high;     // motion_ratio that's fully cool
        uniform vec3  u_color_warm;
        uniform vec3  u_color_cool;
        uniform float u_max_alpha;
        out vec4 fragColor;

        void main() {
            float oid = texture(uAtlas, vUV).r;
            if (oid < -0.5) { fragColor = vec4(0.0); return; }
            int idx = int(oid + 0.5);
            if (idx < 0 || idx >= 16) { fragColor = vec4(0.0); return; }

            float closeness = u_closeness[idx];
            if (closeness <= 0.001) { fragColor = vec4(0.0); return; }

            float motion_ratio = u_motion_ratio[idx];
            float motion_amp   = u_motion_amp[idx];

            // (B) Hue split. smoothstep with operator-controlled
            // pivots remaps the realistic motion_ratio range
            // (typically ~0.1 standing → ~0.55 walking, never
            // reaching 0 or 1) into a full 0..1 lerp factor —
            // otherwise a linear mix would only ever reach about
            // half-way to the cool color and the swing wouldn't
            // read.
            float t = smoothstep(u_color_pivot_low, u_color_pivot_high,
                                 motion_ratio);
            vec3 color = mix(u_color_warm, u_color_cool, t);

            // (C) Shimmer — sine modulation of effective alpha,
            // amplitude scales with motion. Per-object phase offset
            // (hashed from idx) keeps adjacent boxes from shimmering
            // in lockstep.
            float per_obj_phase = float(idx) * 1.91;
            float shim = sin(6.2831853 * u_shimmer_freq * u_time + per_obj_phase);
            float brightness_mod = 1.0 + shim * motion_amp * u_shimmer_max_amp;

            float alpha = closeness * brightness_mod * u_max_alpha;
            fragColor = vec4(color, clamp(alpha, 0.0, 1.0));
        }
        """

    def compile_shader(self):
        vert = shaders.compileShader(self.get_vertex_shader(), GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.get_fragment_shader(), GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)

    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self._vertices.nbytes,
                     self._vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes,
                     self._indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

        # Upload object_id atlas vertically flipped (same convention
        # as the other multi_object atlas-using shaders). Without
        # the flip, rows render to inverted output rows and the glow
        # lands on the wrong physical box.
        h, w = self._atlas.shape
        flipped = np.ascontiguousarray(np.flipud(self._atlas))
        self.atlas_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0,
                     GL_RED, GL_FLOAT, flipped)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    def update(self, dt: float, state: Dict):
        # Advance shimmer time monotonically (mod 1e6 to keep float32
        # precision on the GPU comfortable at long uptimes).
        self._time = (self._time + dt) % 1.0e6
        # EMA per signal; tau-aware so the response is dt-independent.
        if dt > 0.0:
            a_close = 1.0 - math.exp(-dt / self._TAU_CLOSENESS_S)
            a_amp   = 1.0 - math.exp(-dt / self._TAU_MOTION_AMP_S)
            a_ratio = 1.0 - math.exp(-dt / self._TAU_MOTION_RATIO_S)
        else:
            a_close = a_amp = a_ratio = 1.0
        for i in range(_MAX_OBJECTS):
            self.closeness[i]    += a_close * (self.raw_closeness[i]    - self.closeness[i])
            self.motion_amp[i]   += a_amp   * (self.raw_motion_amp[i]   - self.motion_amp[i])
            self.motion_ratio[i] += a_ratio * (self.raw_motion_ratio[i] - self.motion_ratio[i])

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        # Skip the draw entirely when no object has any glow — saves
        # a state-change burst when the room is empty.
        if not any(c > 0.001 for c in self.closeness):
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
            glUniform1i(glGetUniformLocation(self.shader, "uAtlas"), 0)
            # Upload the per-object signal arrays in one call each.
            # Location for ``u_<name>[0]`` resolves the whole array.
            glUniform1fv(
                glGetUniformLocation(self.shader, "u_closeness"),
                _MAX_OBJECTS,
                np.asarray(self.closeness, dtype=np.float32),
            )
            glUniform1fv(
                glGetUniformLocation(self.shader, "u_motion_ratio"),
                _MAX_OBJECTS,
                np.asarray(self.motion_ratio, dtype=np.float32),
            )
            glUniform1fv(
                glGetUniformLocation(self.shader, "u_motion_amp"),
                _MAX_OBJECTS,
                np.asarray(self.motion_amp, dtype=np.float32),
            )
            glUniform1f(glGetUniformLocation(self.shader, "u_time"),
                        float(self._time))
            glUniform1f(glGetUniformLocation(self.shader, "u_shimmer_freq"),
                        float(self.shimmer_freq))
            glUniform1f(glGetUniformLocation(self.shader, "u_shimmer_max_amp"),
                        float(self.shimmer_max_amp))
            glUniform1f(glGetUniformLocation(self.shader, "u_color_pivot_low"),
                        float(self.color_pivot_low))
            glUniform1f(glGetUniformLocation(self.shader, "u_color_pivot_high"),
                        float(self.color_pivot_high))
            glUniform3f(glGetUniformLocation(self.shader, "u_color_warm"),
                        *self.color_warm)
            glUniform3f(glGetUniformLocation(self.shader, "u_color_cool"),
                        *self.color_cool)
            glUniform1f(glGetUniformLocation(self.shader, "u_max_alpha"),
                        float(self.max_alpha))
            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)
        finally:
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)

    def cleanup(self):
        try:
            if self.atlas_tex is not None:
                glDeleteTextures(1, [self.atlas_tex])
                self.atlas_tex = None
            if self.position_VBO is not None:
                glDeleteBuffers(1, [self.position_VBO])
                self.position_VBO = None
            if self.EBO is not None:
                glDeleteBuffers(1, [self.EBO])
                self.EBO = None
            if self.VAO is not None:
                glDeleteVertexArrays(1, [self.VAO])
                self.VAO = None
            if self.shader:
                glDeleteProgram(self.shader)
                self.shader = None
        except Exception:
            pass
