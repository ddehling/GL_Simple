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
// Integrated phases — see snowfall.py / rain_on_leaves.py for rationale.
// Replaces `u_time * varying_rate` so motion stays monotonic when the
// driving uniforms (density, wind) interpolate during state transitions.
uniform float u_ascend_phase;
uniform float u_wind_phase;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise.
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    if (u_density < 0.02 || u_fade < 0.005) discard;

    vec2 uv = v_uv;

    // Continuous density field via fbm — NO cellular grid, so spores
    // blend smoothly with each other and gradients are continuous.
    // ascend_speed and wind_drift are pre-integrated on the CPU as
    // u_ascend_phase / u_wind_phase to avoid time-warp under interpolation.
    float wind_drift = u_wind_phase * 0.04;

    // Two layered noise scales for depth/parallax. Both drift upward
    // (negative y term in p) and sideways with wind.
    vec2 p1 = vec2(uv.x *  6.0 + wind_drift *  6.0,
                   uv.y * 10.0 - u_ascend_phase * 10.0);
    vec2 p2 = vec2(uv.x * 12.0 + wind_drift * 12.0 * 1.4,
                   uv.y * 18.0 - u_ascend_phase * 18.0 * 0.7) + 31.4;
    float density_field = vnoise(p1) * 0.7 + vnoise(p2) * 0.3;

    // Smooth threshold (no hard step). Higher density param -> lower
    // threshold -> more spore coverage.
    float thresh = mix(0.85, 0.30, clamp(u_density, 0.0, 1.0));
    float spore_field = smoothstep(thresh - 0.10,
                                   thresh + 0.20,
                                   density_field);

    // Lifecycle fade: spores fade IN as they enter from the floor and
    // fade OUT as they reach the canopy. Eliminates pop-in/out at
    // screen edges.
    float fade_in  = smoothstep(0.00, 0.20, uv.y);
    float fade_out = smoothstep(1.00, 0.80, uv.y);
    float lifecycle = fade_in * fade_out;

    // Slow gentle breathing across the field — spatial variation rather
    // than temporal flicker. Mild low-freq oscillation only.
    float breath = 0.75 + 0.25 *
        sin(u_time * 0.25 + uv.x * 3.0 + uv.y * 2.0);

    float intensity = spore_field * lifecycle * breath;

    // Color palette — DISTINCT from fireflies (yellow-green):
    //   spore (default): pale lavender / blue-violet — mushroom-grove feel
    //   pollen (color_shift=1): warm gold
    vec3 spore_col  = vec3(0.62, 0.55, 0.95);   // pale lavender
    vec3 pollen_col = vec3(1.00, 0.78, 0.30);   // warm gold
    vec3 col = mix(spore_col, pollen_col, clamp(u_color_shift, 0.0, 1.0));

    // Use STRAIGHT alpha (NOT pre-multiplied) — global blend is
    // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), so output rgb directly.
    float alpha = clamp(intensity, 0.0, 1.0) * 0.65 * u_fade;
    if (alpha < 0.01) discard;
    fragColor = vec4(col, alpha);
}
"""


class SporeDriftEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 8.0  # Mid-air particles, in front of canopy/shadows
        self._time = 0.0
        self._ascend_phase = 0.0   # integrated dt * 0.025 * (0.5 + density * 0.6)
        self._wind_phase = 0.0     # integrated dt * wind
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
        self._ascend_phase += dt * 0.025 * (0.5 + self.density * 0.6)
        self._wind_phase += dt * self.wind

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform1f(glGetUniformLocation(self.shader, "u_color_shift"), self.color_shift)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glUniform1f(glGetUniformLocation(self.shader, "u_ascend_phase"), self._ascend_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind_phase"), self._wind_phase)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
