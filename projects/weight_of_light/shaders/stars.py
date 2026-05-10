"""WoL stars — twinkling pinpricks of light scattered up the sky.

Pattern B (fullscreen-quad atmospheric layer): renders a sparse field
of bright points on the Sky group canvas, each twinkling on its own
sine-wave phase. Gated by ``starryness`` from outstate so the WOL_CLEAR
and WOL_RAIN states make stars invisible without ever pausing the
event (smooth crossfades fall out of WeatherStateController's param
interpolation).

Star positions are deterministic per (row, slot) — they don't move.
What animates is each star's brightness, sinusoidally with a
per-star phase offset. Slow enough (default ~0.4 Hz) to read as
twinkle rather than flicker.

Color is mostly white with a faint hue jitter per star (warm yellow
↔ cool blue). Gives a more organic look than a flat white field.

Time integration: ``_phase`` is CPU-side; the GLSL never multiplies
``u_time × varying_uniform``. See docs/shader_info.txt §Time-based
Animation.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


def shader_wol_stars(state, outstate,
                     stars_per_strip: float = 4.0,
                     twinkle_speed: float = 0.4,
                     star_radius: float = 0.012):
    """Permanent star background; intensity follows ``starryness``.

    ``stars_per_strip`` is the steady-state visible-star count per
    physical strip when starryness=1. ``twinkle_speed`` is how fast
    each star's brightness oscillates (1.0 = full sine cycle per
    second). ``star_radius`` is the spatial extent of each star in
    canvas-x units (smaller = sharper pinpricks)."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            eff = viewport.add_effect(
                WolStarsEffect,
                stars_per_strip=float(stars_per_strip),
                twinkle_speed=float(twinkle_speed),
                star_radius=float(star_radius),
            )
            state['effect'] = eff
            print(f"[wol_stars] init on frame {frame_id}")
        except Exception as e:
            print(f"[wol_stars] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        eff.intensity = float(np.clip(outstate.get('starryness', 0.0), 0.0, 1.0))

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[wol_stars] cleaned up on frame {frame_id}")


class WolStarsEffect(ShaderEffect):
    def __init__(self, viewport,
                 stars_per_strip: float = 4.0,
                 twinkle_speed: float = 0.4,
                 star_radius: float = 0.012):
        super().__init__(viewport)
        # Above sky background (1) but below rain (5) so rain streaks
        # cross over stars rather than the other way around.
        self.render_priority = 4.0

        self.stars_per_strip = float(stars_per_strip)
        self.twinkle_speed = float(twinkle_speed)
        self.star_radius = float(star_radius)
        self.intensity = 0.0

        self._phase = 0.0   # CPU-side seconds tally (mod something
                            # large enough that float precision
                            # doesn't bite at long uptimes)

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
        # Each row owns ``u_stars`` star slots; each slot k generates
        # one star at hash-determined x with a hash-determined
        # twinkle phase + hue.
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform float u_intensity;
        uniform float u_phase;
        uniform float u_stars;
        uniform float u_radius;
        out vec4 fragColor;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        // Faint hue jitter — interpolates between warm yellow and cool
        // blue around white so the field doesn't read as one flat hue.
        vec3 starColor(float h) {
            // h in [0,1]; near 0 → warm yellow, near 1 → cool blue.
            vec3 warm = vec3(1.0, 0.92, 0.78);
            vec3 cool = vec3(0.85, 0.92, 1.0);
            return mix(warm, cool, h);
        }

        void main() {
            if (u_intensity <= 0.001) { fragColor = vec4(0.0); return; }
            float row = floor(vUV.y * 19.0);
            float maxAlpha = 0.0;
            vec3  bestColor = vec3(1.0);
            for (int k = 0; k < 8; k++) {
                if (float(k) >= u_stars) break;
                float kf = float(k);
                float x_pos    = hash(vec2(row, kf * 1.91 + 0.13));
                float t_offset = hash(vec2(row, kf * 4.27 + 0.71));
                float hue      = hash(vec2(row, kf * 9.83 + 0.47));
                // Distance from star center along the strip.
                float d = abs(vUV.x - x_pos);
                if (d > u_radius) continue;
                // Spatial profile: gaussian-ish softness.
                float spatial = 1.0 - smoothstep(0.0, u_radius, d);
                // Twinkle profile: sine, with phase offset per star.
                // 0.5 + 0.5*sin keeps it in [0,1]; minimum_brightness
                // floor keeps stars never quite invisible.
                float twink = 0.5 + 0.5 * sin(6.2831853 * (u_phase + t_offset));
                twink = 0.25 + 0.75 * twink;
                float a = spatial * twink;
                if (a > maxAlpha) {
                    maxAlpha = a;
                    bestColor = starColor(hue);
                }
            }
            fragColor = vec4(bestColor, maxAlpha * u_intensity);
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

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        # Wrap mod 1 — the inner sin uses 2π * (phase + offset), so
        # dropping integer revolutions keeps the float precision tight
        # while preserving phase relationships.
        self._phase = (self._phase + dt * self.twinkle_speed) % 1.0

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        if self.intensity <= 0.001:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glUniform1f(glGetUniformLocation(self.shader, "u_intensity"),
                        float(self.intensity))
            glUniform1f(glGetUniformLocation(self.shader, "u_phase"),
                        float(self._phase))
            glUniform1f(glGetUniformLocation(self.shader, "u_stars"),
                        float(self.stars_per_strip))
            glUniform1f(glGetUniformLocation(self.shader, "u_radius"),
                        float(self.star_radius))
            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glUseProgram(0)
        finally:
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)

    def cleanup(self):
        try:
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
