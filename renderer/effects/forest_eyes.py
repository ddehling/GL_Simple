"""Forest eyes effect — pairs of glinting animal eyes in the dark grove.

Eight pairs at fixed angular slots blink and shift over time. Each pair
appears in the mid-radius zone (uv.y around 0.2..0.55, between forest
floor and lower canopy on the fan). Eyes only appear in dark scenes
(starryness/spookyness driven).

Drives:
  eye_density     -> how many pairs are visible
  spookyness      -> color shift toward red, faster blinks
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_forest_eyes(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(ForestEyesEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized forest_eyes for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize forest_eyes: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.density = float(outstate.get('eye_density', 0.0))
        eff.spookyness = float(outstate.get('spookyness', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 6.0
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
    // Closest layer: foreground glints (priority 5.0).
    gl_Position = vec4(position, 0.30, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_density;
uniform float u_spookyness;
uniform float u_fade;

float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    if (u_density < 0.02 || u_fade < 0.005) discard;

    vec2 uv = v_uv;

    vec3 col = vec3(0.0);
    float total_alpha = 0.0;

    const int N_PAIRS = 8;
    float radius = 0.012;
    float r2 = radius * radius;

    for (int i = 0; i < N_PAIRS; ++i) {
        float fi = float(i);
        // Slow re-roll: each pair jumps to a new position on a long cycle.
        float reroll = floor(u_time / 18.0 + fi * 1.7);

        // Show probability based on density & per-pair seed.
        float show = step(1.0 - clamp(u_density, 0.0, 1.0), hash(reroll * 13.7 + fi * 7.7));
        if (show < 0.5) continue;

        // Center position. x is angular (0..1), y is mid-radius zone.
        // Cluster bias: every other pair spawns near the previous one,
        // creating "watching groups" instead of evenly-distributed eyes.
        float cx = hash(reroll * 7.13 + fi * 3.1);
        float cy = mix(0.18, 0.55, hash(reroll * 11.3 + fi * 5.7));
        float cluster = step(0.5, hash(reroll * 19.3 + fi));
        if (cluster > 0.5 && i > 0) {
            float anchor_x = hash(reroll * 7.13 + (fi - 1.0) * 3.1);
            float anchor_y = mix(0.18, 0.55, hash(reroll * 11.3 + (fi - 1.0) * 5.7));
            cx = anchor_x + (hash(reroll + fi * 5.5) - 0.5) * 0.06;
            cy = anchor_y + (hash(reroll + fi * 6.6) - 0.5) * 0.04;
        }

        // Pair offset: two eyes spaced apart horizontally.
        float spacing = 0.012 + 0.008 * hash(reroll + fi * 9.1);

        // Smooth blink: eye openness goes 1 → 0 → 1 over a short window
        // every blink period. Spookyness shortens the period.
        float blink_period = mix(4.5, 1.8, u_spookyness) + hash(reroll + fi) * 1.2;
        float blink_cycle  = mod(u_time + fi * 2.0, blink_period);
        // Blink lasts 0.18 seconds, smooth on/off.
        float blink_t = blink_cycle / 0.18;
        float openness = (blink_t < 1.0)
            ? (1.0 - exp(-pow(blink_t * 2.0 - 1.0, 2.0) * 4.0))
            : 1.0;

        // Slow gaze drift inside the assigned cell (eyes shift left/right
        // by a fraction of an eye width).
        float gaze = sin(u_time * 0.8 + fi * 1.7) * 0.003;

        // Two eye centers (left + right) — handle horizontal wrap continuity.
        for (int k = 0; k < 2; ++k) {
            float kf = float(k) * 2.0 - 1.0;  // -1 or 1
            vec2 ecenter = vec2(cx + kf * spacing + gaze, cy);

            float dx = abs(uv.x - ecenter.x);
            dx = min(dx, 1.0 - dx);  // wrap continuity
            float dy = uv.y - ecenter.y;
            float d2 = dx * dx + dy * dy;

            // Squared-distance falloff (no sqrt).
            float core = exp(-d2 / (r2 * 0.35)) * openness;
            float glow = exp(-d2 / (r2 * 1.6)) * 0.3 * openness;
            float bright = core + glow;

            // Color: yellow-green default, shifts to red with spookyness.
            vec3 normal_col = vec3(0.85, 1.0, 0.45);
            vec3 spooky_col = vec3(1.0, 0.18, 0.12);
            vec3 ec = mix(normal_col, spooky_col, u_spookyness);

            col += ec * bright;
            total_alpha = max(total_alpha, bright);
        }
    }

    float alpha = total_alpha * u_fade;
    if (alpha < 0.005) discard;
    fragColor = vec4(col * u_fade, alpha);
}
"""


class ForestEyesEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 5.0
        self._time = 0.0
        self.density = 0.6
        self.spookyness = 0.0
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
        glUniform1f(glGetUniformLocation(self.shader, "u_spookyness"), self.spookyness)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
