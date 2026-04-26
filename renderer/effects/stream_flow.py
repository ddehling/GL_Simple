"""Stream-flow effect — flowing water along the inner ring of the fan.

A horizontal band of moving caustics/ripples concentrated at the bottom
of the FBO (uv.y near 0 = inner ring of fan = forest floor). Adds a
"brook running through the grove" element. Wind nudges flow rate.

Drives:
  stream_flow_rate -> water flow speed
  wind             -> ripple speed multiplier
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_stream_flow(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(StreamFlowEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized stream_flow for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize stream_flow: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.rate = float(outstate.get('stream_flow_rate', 0.0))
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
    // Mid-back, just above dappled_shadows (priority 2.5).
    gl_Position = vec4(position, 0.78, 1.0);
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
uniform float u_fade;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

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
    vec2 uv = v_uv;

    // Hard gate: no stream when rate is essentially zero. This avoids the
    // base water tint being visible in non-stream states.
    if (u_rate < 0.02 || u_fade < 0.005) discard;

    // Confine flow to the bottom band (y in [0, 0.35]) — the inner-ring
    // forest floor on the fan. Discard above to avoid depth writes.
    float band_top = 0.35;
    if (uv.y > band_top + 0.05) discard;

    // Inside-band coordinate: 1 at bottom (water core), 0 at band edge.
    float band_t = clamp(1.0 - uv.y / band_top, 0.0, 1.0);

    // Flow direction along x (angular around fan). Speed = base + wind.
    float flow_speed = 0.18 * (0.4 + u_rate * 1.2) + u_wind * 0.06;
    float drift = u_time * flow_speed;

    // Layered caustics: two scales of noise scrolling at different speeds.
    float n1 = vnoise(vec2(uv.x * 14.0 + drift * 14.0, uv.y * 24.0 + u_time * 0.5));
    float n2 = vnoise(vec2(uv.x * 28.0 - drift * 18.0, uv.y * 40.0 - u_time * 0.7) + 17.0);
    float caustic = pow(n1 * 0.7 + n2 * 0.3, 2.0);

    // Bright ripple highlights where caustic peaks above threshold.
    float ripple = smoothstep(0.45, 0.95, caustic) * (0.5 + band_t);

    // Bubble pops: small bright spots that move along with the flow.
    vec2 bp = vec2(uv.x * 50.0 - drift * 50.0, uv.y * 18.0);
    vec2 b_cell = floor(bp);
    vec2 b_frac = fract(bp);
    float b_seed = hash(b_cell);
    float b_show = step(0.97, b_seed);
    vec2 b_center = vec2(hash(b_cell + 1.7), hash(b_cell + 5.3));
    float b_d = distance(b_frac, b_center);
    float b_life = fract(u_time * 0.6 + b_seed * 17.0);
    float b_bright = b_show * (1.0 - b_life) *
                     exp(-(b_d * b_d) / 0.004) * band_t;

    // Base water tint, deeper toward inner ring.
    vec3 water_deep = vec3(0.10, 0.20, 0.32);
    vec3 water_top  = vec3(0.45, 0.65, 0.78);
    vec3 base = mix(water_top, water_deep, band_t);

    // Highlight color: pale blue-white.
    vec3 highlight = vec3(0.85, 0.92, 1.0);
    vec3 col = base + highlight * (ripple * 0.6 + b_bright * 1.2);

    // Alpha tapers at band edge so the stream blends into the floor.
    // The base water tint scales with u_rate so the stream fades out
    // instead of leaving a faint haze when rate drops.
    float rate_gate = clamp(u_rate * 1.3, 0.0, 1.0);
    float alpha = (0.55 * rate_gate + ripple * 0.4 + b_bright) *
                  smoothstep(0.0, 0.08, band_top - uv.y);
    alpha *= u_fade;
    if (alpha < 0.005) discard;

    fragColor = vec4(col * alpha, alpha);
}
"""


class StreamFlowEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 2.0  # Bottom water layer; renders only at floor band
        self._time = 0.0
        self.rate = 0.5
        self.wind = 0.0
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
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_rate"), self.rate)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
