"""WoL sky day/night gradient — permanent, self-contained background.

Pattern B (fullscreen-quad atmospheric layer): paints a horizon→zenith
gradient onto the Sky group canvas, cycling continuously through dawn
→ day → dusk → night → dawn on its own internal phase counter. No
weather-state input — the cycle runs forever and weather effects (rain,
stars) overlay on top via separate higher-priority background events.

Each Sky strip's LED 0 sits at the bottom of the strip (the ground-arc
chord intersection); the last LED is at the top. Canvas x maps directly
to physical altitude: vUV.x = 0 → horizon, vUV.x = 1 → zenith.

The cycle period is set by ``cycle_seconds``. Default 600 s (10 min)
gives a brisk demo cadence; 3600 s would be a full real-time hour.
Phase is integrated on the CPU using ``dt`` so the animation never
reverses if the cycle speed gets adjusted at runtime.

Color scheme: 4 anchor color pairs (horizon, zenith) interpolated
linearly between phase quarters. Tuned for visibility in a small
lit room — bright enough at "day" to read as bright, dim enough at
"night" to feel like night without going pitch black.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


# Phase anchor → (horizon RGB, zenith RGB). Phase wraps 0..1.
# Eight anchors with intermediates so the color trajectory doesn't
# need to pass through muddy mid-points (e.g. dawn-pink → day-cyan
# bisecting through grey). Each segment is 1/8 of the cycle ≈ 3.75
# minutes at WoL's 30-min season; smoothstep within each segment
# (see _eval_anchors) holds near each anchor color longer than a
# pure linear interpolation would.
_SKY_ANCHORS = [
    # phase, horizon,                        zenith
    (0.000, (0.95, 0.55, 0.35),  (0.25, 0.30, 0.55)),  # dawn         — warm pink / deep blue
    (0.125, (0.88, 0.72, 0.62),  (0.22, 0.40, 0.78)),  # early morning — peach / sky blue
    (0.250, (0.70, 0.85, 0.95),  (0.20, 0.45, 0.85)),  # midday       — pale cyan / mid blue
    (0.375, (0.85, 0.68, 0.45),  (0.22, 0.32, 0.65)),  # afternoon    — warm wash / slate
    (0.500, (0.85, 0.30, 0.20),  (0.22, 0.10, 0.35)),  # dusk         — red-orange / purple
    (0.625, (0.40, 0.15, 0.18),  (0.08, 0.05, 0.18)),  # twilight     — deep maroon / dark navy
    (0.750, (0.04, 0.05, 0.10),  (0.01, 0.01, 0.04)),  # night        — near-black / deepest
    (0.875, (0.18, 0.18, 0.32),  (0.08, 0.10, 0.28)),  # pre-dawn     — deep blue / dark blue
]


def _eval_anchors(phase: float):
    """Smooth (horizon, zenith) color pair at ``phase`` (0..1, wraps).

    Linear interpolation between adjacent anchors with smoothstep
    ease-in/out applied to ``t``. The smoothstep guarantees the
    color's rate of change is zero at each anchor — eliminating the
    rate-discontinuity ("kink") that pure linear interpolation
    leaves at every anchor and reads as a "hard shift" in motion.
    """
    p = phase % 1.0
    n = len(_SKY_ANCHORS)
    # Find the bracketing anchors with phase wrap-around.
    for i in range(n):
        p0 = _SKY_ANCHORS[i][0]
        p1 = _SKY_ANCHORS[(i + 1) % n][0]
        span_lo = p0
        span_hi = p1 if p1 > p0 else p1 + 1.0
        p_test = p if p >= p0 else p + 1.0
        if span_lo <= p_test <= span_hi:
            t = (p_test - span_lo) / max(span_hi - span_lo, 1e-6)
            # Smoothstep ease-in/out: rate-of-change is zero at both
            # ends of the segment, so transitioning past an anchor
            # has C¹ continuity in color space.
            t = t * t * (3.0 - 2.0 * t)
            h0, z0 = _SKY_ANCHORS[i][1], _SKY_ANCHORS[i][2]
            h1, z1 = _SKY_ANCHORS[(i + 1) % n][1], _SKY_ANCHORS[(i + 1) % n][2]
            horizon = tuple(h0[k] + t * (h1[k] - h0[k]) for k in range(3))
            zenith  = tuple(z0[k] + t * (z1[k] - z0[k]) for k in range(3))
            return horizon, zenith
    # Defensive fallback (shouldn't reach with the wrap-aware bracketing)
    return _SKY_ANCHORS[0][1], _SKY_ANCHORS[0][2]


def shader_wol_sky_daynight(state, outstate,
                            cycle_seconds: float = 600.0,
                            shimmer: float = 0.06,
                            shimmer_speed: float = 0.05,
                            dim_strength: float = 0.85,
                            use_season: bool = True):
    """Permanent background day/night sky.

    Phase source (which point of the dawn→day→dusk→night→dawn loop
    the sky is currently rendering):

      * ``use_season=True`` (default): drive the phase from the
        engine's global ``season`` tracker (``state['season']``),
        which advances at ``season_speed × time / 1800``. With
        WoL's ``season_speed=1.0`` a full sky cycle takes 30 min
        and is synchronised with everything else season-driven
        (ambient_light dimming the ground twinkle, etc.). If
        ``state['season']`` is missing for some reason the shader
        falls back to its own internal timer.
      * ``use_season=False``: ignore season and run an independent
        timer at ``cycle_seconds`` per loop. Useful if you want
        the sky to cycle faster than the season for bring-up demo
        purposes.

    ``shimmer`` is the amplitude of a slow value-noise overlay for
    atmospheric texture (0 disables).

    ``dim_strength`` is how much the sky pulls back when other layers
    want priority. Driven each frame by ``celestial_visibility`` from
    outstate: at celestial=0 the gradient is at full brightness; at
    celestial=1 it dims to ``(1 - dim_strength)`` of full (default
    0.85 → 15 % residual at full night, leaving room for stars or
    other foreground events to read clearly without being washed
    out by the base sky).
    """
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
                WolSkyDayNightEffect,
                shimmer=float(shimmer),
                shimmer_speed=float(shimmer_speed),
                cycle_seconds=float(cycle_seconds),
                dim_strength=float(dim_strength),
                use_season=bool(use_season),
            )
            state['effect'] = eff
            src = "season" if use_season else f"internal {cycle_seconds}s"
            print(f"[wol_sky_daynight] init on frame {frame_id} "
                  f"(phase source: {src})")
        except Exception as e:
            print(f"[wol_sky_daynight] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        # Pull live celestial_visibility — high values = "stars want
        # to read", so the sky pulls back. Clamp to [0, 1].
        cv = float(outstate.get('celestial_visibility', 0.0))
        if cv < 0.0: cv = 0.0
        elif cv > 1.0: cv = 1.0
        eff.celestial_visibility = cv
        # Drive the day/night phase from the engine's season tracker
        # (state['season'] is already a float in [0, 1) advancing at
        # season_speed × time / 1800). Falling back to the internal
        # timer when state['season'] is missing keeps the shader
        # usable in standalone tests.
        if eff.use_season:
            season = outstate.get('season')
            if season is not None:
                eff.season_phase = float(season) % 1.0

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[wol_sky_daynight] cleaned up on frame {frame_id}")


class WolSkyDayNightEffect(ShaderEffect):
    """Pattern B fullscreen-quad gradient with a self-driven phase.

    Phase advances at ``1 / cycle_seconds`` per second on the CPU
    (never multiplied with a varying uniform on the GPU; see
    docs/shader_info.txt §Time-based Animation). Each frame the
    wrapper-side ``_eval_anchors`` resolves the current horizon /
    zenith colors for the current phase and uploads them as uniforms
    — keeps the shader trivial and the color anchors easy to tune
    in pure Python."""

    def __init__(self, viewport,
                 shimmer: float = 0.06,
                 shimmer_speed: float = 0.05,
                 cycle_seconds: float = 600.0,
                 dim_strength: float = 0.85,
                 use_season: bool = True):
        super().__init__(viewport)
        self.render_priority = 1.0   # back layer

        self.shimmer = float(shimmer)
        self.shimmer_speed = float(shimmer_speed)
        self.cycle_seconds = max(float(cycle_seconds), 1.0)
        self.dim_strength = float(dim_strength)
        self.celestial_visibility = 0.0   # set each frame by wrapper

        self.use_season = bool(use_season)
        self.season_phase = 0.0   # set by wrapper from outstate['season']

        self._phase = 0.0      # 0..1, wraps; advances on dt — internal-timer fallback
        self._drift = 0.0      # shimmer noise offset

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
        uniform vec3  u_horizon;
        uniform vec3  u_zenith;
        uniform float u_drift;
        uniform float u_shimmer;
        uniform float u_dim;            // 0..1; how much to pull back
        out vec4 fragColor;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        float vnoise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            vec2 u = f * f * (3.0 - 2.0 * f);
            return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x)
                                  + (d - b) * u.x * u.y;
        }

        void main() {
            float t = smoothstep(0.0, 1.0, vUV.x);
            vec3 sky = mix(u_horizon, u_zenith, t);

            float n = vnoise(vec2(vUV.x * 4.0 + u_drift, vUV.y * 12.0));
            sky *= (1.0 - u_shimmer * 0.5) + (u_shimmer * n);

            // Pull back when overlay events want priority (stars,
            // future themes). u_dim is precomputed CPU-side as
            // celestial_visibility * dim_strength.
            sky *= (1.0 - u_dim);

            fragColor = vec4(sky, 1.0);    // straight alpha, opaque
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
        # Internal phase always advances — used as a fallback when
        # use_season is False or state['season'] is missing. Cheap.
        self._phase = (self._phase + dt / self.cycle_seconds) % 1.0
        self._drift += dt * self.shimmer_speed

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        # Pick the phase source: season-driven (synced with engine's
        # global tick) or internal (independent timer).
        phase = self.season_phase if self.use_season else self._phase
        horizon, zenith = _eval_anchors(phase)
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glUniform3f(glGetUniformLocation(self.shader, "u_horizon"),
                        horizon[0], horizon[1], horizon[2])
            glUniform3f(glGetUniformLocation(self.shader, "u_zenith"),
                        zenith[0], zenith[1], zenith[2])
            glUniform1f(glGetUniformLocation(self.shader, "u_drift"),
                        float(self._drift))
            glUniform1f(glGetUniformLocation(self.shader, "u_shimmer"),
                        float(self.shimmer))
            glUniform1f(
                glGetUniformLocation(self.shader, "u_dim"),
                float(self.celestial_visibility * self.dim_strength),
            )
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
