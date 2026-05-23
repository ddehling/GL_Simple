"""
Cyber neon signs — flickering tube-glow storefront signage along the
midden level. Each sign is a small horizontal bar with bloom; signs are
deterministically placed and have independent flicker timers. Hot pink
/ cyan / acid green palette. Reads `outstate['cyber_signage_density']`
(0..1).

Sparse bright features against dark = excellent under the limiter.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_neon_signs(state, outstate, density=0.6):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberNeonSignsEffect, density=density)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_neon_signs] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # Wrapper default 0.0 per docs/shader_info.txt.
    eff.density = float(outstate.get('cyber_signage_density', 0.0))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


VERTEX = """#version 310 es
precision highp float;
layout(location = 0) in vec2 position;
out vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT = """#version 310 es
precision highp float;
in vec2 v_uv;
uniform float u_time;
uniform float u_density;
out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Sign at center cx, cy with half-width w_h and half-height h_h. Returns
// 0..1 brightness with bloom. Bloom is a soft falloff outside the sign.
float sign_glow(vec2 uv, vec2 c, vec2 sz) {
    vec2 d = abs(uv - c) / sz;
    // Sharp rectangular core
    float core = 1.0 - smoothstep(0.85, 1.0, max(d.x, d.y));
    // Soft bloom
    float bloom = 1.0 - smoothstep(0.0, 2.5, length(d));
    return max(core, bloom * 0.45);
}

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float a = 0.0;

    // Sign grid: 8 columns x 6 rows of POSSIBLE sign positions. Each cell
    // hashes to whether it's occupied (per density), color, and flicker.
    const float COLS = 8.0;
    const float ROWS = 6.0;

    // Place signs only in the upper-middle band (y in [0.30, 0.85]) — they
    // hang on building faces, not on the ground or sky.
    // Loop counter `r` for row, `c` for column — NOT `col` (which would
    // shadow the outer `vec3 col` and produce a typed-assignment error
    // that some drivers compile but then fail validation on).
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 8; c++) {
            vec2 cell = vec2(float(c), float(r));
            float seed = hash(cell + 7.7);

            // Density gate
            if (seed > u_density) continue;

            // Cell center in uv space, jittered within cell
            float jx = hash(cell + 1.3) - 0.5;
            float jy = hash(cell + 2.7) - 0.5;
            float cx = (float(c) + 0.5 + jx * 0.6) / COLS;
            float cy = 0.30 + (float(r) + 0.5 + jy * 0.4) / ROWS * 0.55;

            // Sign size — small horizontal bar
            float w = 0.020 + hash(cell + 3.1) * 0.018;
            float h = 0.007 + hash(cell + 4.5) * 0.005;

            // Flicker: each sign has its own period and duty
            float fl_period = 0.4 + hash(cell + 5.9) * 2.5;
            float fl_phase = hash(cell + 6.3);
            float fl_t = mod(u_time * (1.0 / fl_period) + fl_phase, 1.0);
            // Mostly on (85%), occasional dropouts
            float lit = step(0.05, fl_t) * (1.0 - step(0.92, fl_t) * 0.8);

            // Color: hot pink / cyan / acid green / purple
            vec3 c1 = vec3(1.0, 0.0, 0.55);    // hot pink
            vec3 c2 = vec3(0.0, 0.96, 1.0);     // cyan
            vec3 c3 = vec3(0.50, 1.0, 0.10);    // acid green
            vec3 c4 = vec3(0.75, 0.0, 1.0);     // purple
            float ch = hash(cell + 8.1);
            vec3 sign_color;
            if (ch < 0.30) sign_color = c1;
            else if (ch < 0.60) sign_color = c2;
            else if (ch < 0.80) sign_color = c3;
            else sign_color = c4;

            float g = sign_glow(uv, vec2(cx, cy), vec2(w, h)) * lit;
            col += sign_color * g;
            a = max(a, g * 0.95);
        }
    }

    if (a < 0.02) discard;
    fragColor = vec4(col, clamp(a, 0.0, 1.0));
}
"""


class CyberNeonSignsEffect(ShaderEffect):
    def __init__(self, viewport, density: float = 0.6):
        super().__init__(viewport)
        self.render_priority = 7.0    # In front of skyline (6.0), behind holograms (7.5)
        self.density = density
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberNeonSigns compile error: {e}")
            raise

    def setup_buffers(self):
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        self.VBOs = [vbo]
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt

    def render(self, state: Dict):
        if not self.enabled or self.density < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
