"""WoL rainbow — subtle glimmer of color across the outer ground arcs.

Pattern B (fullscreen-quad atmospheric layer): paints a slow rainbow
hue band onto the Ground group canvas, but only on pixels belonging to
the outer-ring objects (object_id != 0 — center is skipped). Uses the
per-pixel object_id atlas published at boot (same data path as the
lightning flash + bouncing ball test) to gate the effect; the central
ground arc stays untouched.

Subtle by design: max alpha is small (~0.25 at full intensity) so the
underlying ground twinkle reads through. Animation: the hue offset
walks slowly around each arc, giving the appearance of a colorband
sliding around the ring.

Gated on ``rainbow_intensity`` from outstate so the WOL_CLEAR /
WOL_RAIN / WOL_STARS states make it invisible without ever pausing
the event. WOL_RAINBOW state sets it to 1.0.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


def shader_wol_rainbow(state, outstate,
                       max_alpha: float = 0.30,
                       walk_speed: float = 0.05,
                       glimmer_speed: float = 1.5,
                       glimmer_amp: float = 0.35):
    """Permanent rainbow background on the outer-ring ground arcs;
    intensity follows ``rainbow_intensity`` from outstate.

    Each box's ground strip shows a single solid hue at any instant
    (no within-strip color gradient). The hue rotates over time at
    ``walk_speed`` cycles/sec, with each object phase-offset so
    adjacent boxes don't display identical hues. ``max_alpha`` caps
    the opacity at full intensity (0..1). ``glimmer_speed`` /
    ``glimmer_amp`` add a subtle brightness shimmer per object so
    the rainbow doesn't read as a flat colorband.
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        # First call: resolve the active group's per-pixel object_id
        # atlas, then build the GPU effect. We need the atlas at
        # init-time so the texture upload happens once on the GL
        # thread; subsequent updates reuse it.
        group_ids = getattr(shader_renderer, 'group_ids', None) or []
        if 0 <= frame_id < len(group_ids):
            gid = group_ids[frame_id]
        else:
            gid = "Ground"
        meta = (outstate.get("group_metadata") or {}).get(gid)
        if meta is None:
            print(f"[wol_rainbow] no group_metadata for {gid!r}; rainbow inert")
            return
        object_id_atlas = meta.get("object_id")
        if object_id_atlas is None:
            print(f"[wol_rainbow] group {gid!r} has no object_id atlas; "
                  f"rainbow inert")
            return
        try:
            eff = viewport.add_effect(
                WolRainbowEffect,
                object_id_atlas=object_id_atlas,
                max_alpha=float(max_alpha),
                walk_speed=float(walk_speed),
                glimmer_speed=float(glimmer_speed),
                glimmer_amp=float(glimmer_amp),
            )
            state['effect'] = eff
            print(f"[wol_rainbow] init on frame {frame_id} (group={gid})")
        except Exception as e:
            print(f"[wol_rainbow] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        eff.intensity = float(np.clip(
            outstate.get('rainbow_intensity', 0.0), 0.0, 1.0))

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[wol_rainbow] cleaned up on frame {frame_id}")


class WolRainbowEffect(ShaderEffect):
    def __init__(self, viewport,
                 object_id_atlas: np.ndarray,
                 max_alpha: float = 0.30,
                 walk_speed: float = 0.05,
                 glimmer_speed: float = 1.5,
                 glimmer_amp: float = 0.35):
        super().__init__(viewport)
        # Above the ground twinkle (priority 1) so the rainbow reads
        # over the base. Below any future foreground events.
        self.render_priority = 4.0

        self.max_alpha = float(max_alpha)
        self.walk_speed = float(walk_speed)
        self.glimmer_speed = float(glimmer_speed)
        self.glimmer_amp = float(glimmer_amp)
        self.intensity = 0.0

        # Atlas → R32F texture. int32 atlas casts to float on CPU; -1
        # sentinel (no strip) and 0 (center object) are both excluded
        # in the fragment shader.
        self._atlas = np.ascontiguousarray(object_id_atlas, dtype=np.float32)
        self.atlas_tex = None

        self._walk_phase = 0.0
        self._glimmer_phase = 0.0

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
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform sampler2D uAtlas;
        uniform float u_intensity;
        uniform float u_walk;
        uniform float u_glimmer;
        uniform float u_glimmer_amp;
        uniform float u_max_alpha;
        out vec4 fragColor;

        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        void main() {
            if (u_intensity <= 0.001) { fragColor = vec4(0.0); return; }
            float oid = texture(uAtlas, vUV).r;
            // Skip empty pixels (oid < -0.5) and the center object
            // (oid == 0) — rainbow only paints the outer 8 boxes.
            if (oid < -0.5 || abs(oid) < 0.5) {
                fragColor = vec4(0.0);
                return;
            }
            // Hue is a function of (object_id, time) only — every
            // pixel along a given strip shares the same hue, so each
            // ground arc reads as one solid color at any moment.
            // The whole field's hue rotates over time (u_walk) and
            // each object is offset so adjacent boxes don't match.
            float per_obj = oid * 0.137;
            float hue = fract(u_walk + per_obj);
            vec3 rgb = hsv2rgb(vec3(hue, 0.85, 1.0));
            // Subtle brightness glimmer.
            float twink = 0.5 + 0.5 * sin(6.2831853 * (u_glimmer + per_obj));
            float br = (1.0 - u_glimmer_amp) + u_glimmer_amp * twink;
            float alpha = u_max_alpha * u_intensity * br;
            fragColor = vec4(rgb, alpha);
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

        # Upload object_id atlas vertically flipped — same convention
        # as the bouncing-ball / lightning shaders. Without this flip,
        # rows render to inverted output rows and the gating lands on
        # the wrong physical box (see the long comment in
        # test_bouncing_ball.py for the GL/numpy y-axis discussion).
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
        if not self.enabled:
            return
        self._walk_phase = (self._walk_phase + dt * self.walk_speed) % 1.0
        self._glimmer_phase = (self._glimmer_phase
                               + dt * self.glimmer_speed) % 1.0

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        if self.intensity <= 0.001:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
            glUniform1i(glGetUniformLocation(self.shader, "uAtlas"), 0)
            glUniform1f(glGetUniformLocation(self.shader, "u_intensity"),
                        float(self.intensity))
            glUniform1f(glGetUniformLocation(self.shader, "u_walk"),
                        float(self._walk_phase))
            glUniform1f(glGetUniformLocation(self.shader, "u_glimmer"),
                        float(self._glimmer_phase))
            glUniform1f(glGetUniformLocation(self.shader, "u_glimmer_amp"),
                        float(self.glimmer_amp))
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
