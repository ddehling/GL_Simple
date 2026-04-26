"""Snowfall effect — drifting flakes from canopy (outer ring) toward floor (inner ring).

Procedural fragment shader: flakes are computed in a grid where each cell
hosts one flake whose y position cycles with time. Wind tilts the path.
Floor accumulates a frost tint driven by frost_level.

Renders to the rectangular FBO; uv.y=1 (top) is the outer ring of the fan
(sky/canopy) where flakes spawn, and uv.y=0 (bottom) is the inner ring
(forest floor) where they vanish.

Drives:
  snow_rate    -> flake density / fall speed
  wind         -> sideways drift
  frost_level  -> ground-frost tint depth at the floor
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_snowfall(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(SnowfallEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized snowfall for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize snowfall: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.rate = float(outstate.get('snow_rate', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))
        eff.frost = float(outstate.get('frost_level', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 5.0
        if elapsed < fade:
            f = elapsed / fade
        elif elapsed > total - fade:
            f = (total - elapsed) / fade
        else:
            f = 1.0
        eff.fade = float(np.clip(f, 0, 1))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


_VERT = """
#version 310 es
precision highp float;
in vec2 position;
out vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    // Front-of-mid; sits above rain_on_leaves (priority 4.0).
    gl_Position = vec4(position, 0.45, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_rate;
uniform float u_wind;
uniform float u_frost;
uniform float u_fade;
// Integrated phases: CPU-side accumulators of dt * current_rate / dt * current_wind.
// Used in place of `u_time * varying_rate` so motion stays monotonic when
// u_rate / u_wind are interpolated during weather state transitions.
uniform float u_rate_phase;
uniform float u_wind_phase;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 flakeLayer(vec2 uv, float layer_idx, float depth) {
    // Grid of cells, each hosts a single flake.
    // Layer_idx + depth give per-layer offsets so layers don't sync.
    float cols = mix(20.0, 32.0, depth);   // far layers: fewer flakes
    float rows = mix(14.0, 22.0, depth);

    // Speed: faster when "rate" is high; deeper layers fall slower (parallax).
    // Rate-dependent factor is integrated as u_rate_phase on the CPU so it
    // stays monotonic when u_rate is interpolated during state transitions.
    float depth_speed = mix(0.05, 0.18, depth);

    // Wind tilts the path; angular wrapping ensures continuity in x.
    float wind_offset = u_wind_phase * 0.06 * (0.5 + depth);

    // Flakes fall from canopy (uv.y=1) toward floor (uv.y=0).
    // To make a fixed cell-identity slide downward (uv.y decreasing) as
    // time grows, scroll p.y in the +flow direction with uv.y in plain
    // (non-inverted) form: uv.y * rows + flow * rows.
    float flow = u_rate_phase * depth_speed;
    vec2 p = vec2(
        uv.x * cols + wind_offset * cols + layer_idx * 13.7,
        uv.y * rows + flow * rows
    );

    vec2 cell = floor(p);
    vec2 frac = fract(p);

    // Per-cell flake center jitter and brightness
    float h1 = hash(cell + layer_idx * 7.13);
    float h2 = hash(cell + layer_idx * 11.7 + 1.0);
    vec2 center = vec2(h1, h2);

    // Squared-distance core (no sqrt) — exp has the same falloff.
    vec2 dv = frac - center;
    float d2 = dot(dv, dv);
    float radius = mix(0.05, 0.12, depth) * (0.6 + u_rate * 0.5);
    float core = exp(-d2 / (radius * radius * 0.6));

    // Each cell may or may not have a flake (density depends on rate).
    float threshold = 1.0 - clamp(u_rate * 0.55 + 0.1, 0.0, 0.95);
    float on = step(threshold, hash(cell + layer_idx * 17.3 + 5.5));

    // Subtle twinkle so flakes feel alive.
    float tw = 0.7 + 0.3 * sin(u_time * 4.0 + h1 * 30.0);

    float bright = core * on * tw * depth;
    return vec3(bright);
}

void main() {
    // Gate: invisible when neither snow nor frost is active.
    if (u_rate < 0.02 && u_frost < 0.02) discard;
    if (u_fade < 0.005) discard;

    vec2 uv = v_uv;
    vec3 col = vec3(0.0);

    // Three depth layers for parallax.
    col += flakeLayer(uv, 0.0, 0.45);
    col += flakeLayer(uv, 1.0, 0.75);
    col += flakeLayer(uv, 2.0, 1.00);

    float flake_intensity = clamp(col.r, 0.0, 1.0);

    // Floor frost: cool blue-white tint that grows toward the inner ring
    // (uv.y near 0). Strength scales with frost_level and fades upward.
    float frost_band = smoothstep(0.30, 0.0, uv.y);  // 1 at floor, 0 above 0.30
    // Subtle organic noise so the frost isn't a flat band.
    float frost_noise = 0.6 + 0.4 *
        sin(uv.x * 12.0 + sin(uv.y * 18.0 + u_time * 0.05) * 1.3);
    float frost_alpha = frost_band * frost_noise * clamp(u_frost, 0.0, 1.0) * 0.55;

    vec3 snow_col  = vec3(0.92, 0.95, 1.0);
    vec3 frost_col = vec3(0.78, 0.88, 1.00);

    vec3 final_col = snow_col * flake_intensity + frost_col * frost_alpha;
    float alpha = (flake_intensity * 0.85 + frost_alpha) * u_fade;
    if (alpha < 0.005) discard;
    fragColor = vec4(final_col * u_fade, alpha);
}
"""


class SnowfallEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 9.5  # Front snow
        self._time = 0.0
        self._rate_phase = 0.0   # integrated dt * (0.5 + rate * 1.5)
        self._wind_phase = 0.0   # integrated dt * wind
        self.rate = 0.5
        self.wind = 0.0
        self.frost = 0.0
        self.fade = 0.0

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(_FRAG, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        self.VBOs.append(vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        self._time += dt
        self._rate_phase += dt * (0.5 + self.rate * 1.5)
        self._wind_phase += dt * self.wind

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_rate"), self.rate)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_frost"), self.frost)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glUniform1f(glGetUniformLocation(self.shader, "u_rate_phase"), self._rate_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind_phase"), self._wind_phase)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
