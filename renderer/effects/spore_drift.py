"""Spore-drift effect — slow upward bioluminescent particles.

Soft glowing motes drift upward from the forest floor (uv.y=0, inner ring)
toward the canopy (uv.y=1, outer ring). Cool greenish-cyan glow. Used in
mushroom and pollen_drift states.

Drives:
  spore_density   -> particle count
  spore_color     -> 0..1 = green to gold (gold for pollen, green for spores)
  wind            -> sideways drift while ascending
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_spore_drift(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(SporeDriftEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized spore_drift for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize spore_drift: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.density = float(outstate.get('spore_density', 0.0))
        eff.color_shift = float(outstate.get('spore_color', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))

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
    // Mid-depth particles (priority 3.0).
    gl_Position = vec4(position, 0.65, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_density;
uniform float u_color_shift;
uniform float u_wind;
uniform float u_fade;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 sporeLayer(vec2 uv, float layer_idx, float depth, float burst) {
    // Larger cells and bigger soft cores → cloudy / hazy, not point-source
    // (so it doesn't read like fireflies).
    float cols = mix(8.0, 14.0, depth);
    float rows = mix(6.0, 10.0, depth);

    // Slow ascent — these are spores/pollen drifting, not bugs flying.
    float ascend_speed = mix(0.008, 0.022, depth) * (0.5 + u_density * 0.6 + burst * 0.4);
    float wind_drift = u_wind * u_time * 0.04 * (0.4 + depth) + burst * 0.05;

    float flow = u_time * ascend_speed;
    vec2 p = vec2(
        uv.x * cols + wind_drift * cols + layer_idx * 9.7,
        uv.y * rows - flow * rows
    );

    vec2 cell = floor(p);
    vec2 frac = fract(p);

    float h1 = hash(cell + layer_idx * 5.13);
    float h2 = hash(cell + layer_idx * 8.7 + 1.0);
    vec2 center = vec2(h1, h2);

    vec2 dv = frac - center;
    float d2 = dot(dv, dv);
    // Wider, softer core: spore "puffs" instead of pinpoint specks.
    float radius = mix(0.18, 0.32, depth);
    float core = exp(-d2 / (radius * radius * 1.4));

    float threshold = 1.0 - clamp(u_density * 0.45 + burst * 0.20, 0.0, 0.85);
    float on = step(threshold, hash(cell + layer_idx * 19.7 + 3.3));

    // Slow, gentle breathing — no rapid flicker. Period 5–10s per puff.
    float pulse = 0.65 + 0.35 *
        sin(u_time * (0.3 + h1 * 0.25) + h1 * 6.28);

    return vec3(core * on * pulse * depth);
}

void main() {
    vec2 uv = v_uv;

    // Hard gate: invisible when density is essentially zero.
    if (u_density < 0.02 || u_fade < 0.005) discard;

    // Burst events: a slow pulse on top of wind. Higher wind = more bursting.
    float burst = pow(0.5 + 0.5 * sin(u_time * 0.41 + 1.7), 8.0) * (0.3 + u_wind * 0.7);

    vec3 acc = vec3(0.0);
    acc += sporeLayer(uv, 0.0, 0.5,  burst);
    acc += sporeLayer(uv, 1.0, 0.85, burst);
    acc += sporeLayer(uv, 2.0, 1.0,  burst);

    float intensity = clamp(acc.r, 0.0, 1.0);

    // Color: cool green-cyan (spore) → warm gold (pollen).
    vec3 spore = vec3(0.45, 1.00, 0.65);
    vec3 pollen = vec3(1.00, 0.78, 0.30);
    vec3 col = mix(spore, pollen, clamp(u_color_shift, 0.0, 1.0));

    // Slight extra brightness near the floor (denser air there).
    float vertical = mix(0.7, 1.0, 1.0 - uv.y);

    float alpha = intensity * vertical * u_fade;
    if (alpha < 0.005) discard;
    fragColor = vec4(col * alpha, alpha * 0.85);
}
"""


class SporeDriftEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 3.0
        self._time = 0.0
        self.density = 0.5
        self.color_shift = 0.0
        self.wind = 0.1
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

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform1f(glGetUniformLocation(self.shader, "u_color_shift"), self.color_shift)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
