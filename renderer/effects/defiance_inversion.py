"""
Defiance inversion — variable shader for `story_defiance`.

HIGH defiance (→1.0): bright electrical arcs cross the frame in random
directions; brief full-screen palette inversion flashes; horizontal
counter-content bands pulse aggressively.
LOW defiance (just above 0.2): sparse single sparks.

Threshold-gated at 0.2. Scales 0→1 as variable goes 0.2→1.0.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.2


def shader_defiance_inversion(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DefianceInversionEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[defiance_inversion] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.defiance_value = float(outstate.get('story_defiance', 0.0))

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
uniform float u_defiance;        // 0..1 after threshold
uniform float u_time;
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float hash1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float alpha = 0.0;

    // --- Electric arcs ---
    // Each arc is a thin jagged line between two random endpoints, lasting
    // ~0.15 seconds. Number of active arcs scales with defiance.
    int n_arcs = int(1.0 + u_defiance * 5.0);   // 1 to 6 arcs at max
    for (int i = 0; i < 6; i++) {
        if (i >= n_arcs) break;
        float fi = float(i);
        // Each arc has its own period; pick a fresh seed every period
        float period = 0.20 + hash1(fi * 7.0) * 0.30;
        float arc_t = mod(u_time + fi * period * 0.3, period) / period;   // 0..1
        // Only visible for first half of each period
        if (arc_t > 0.5) continue;
        float seed_t = floor((u_time + fi * 0.1) / period);
        // Endpoints
        vec2 p0 = vec2(hash1(seed_t * 13.0 + fi), hash1(seed_t * 17.0 + fi * 3.0));
        vec2 p1 = vec2(hash1(seed_t * 19.0 + fi * 5.0), hash1(seed_t * 23.0 + fi * 7.0));
        // Distance from uv to line segment
        vec2 ba = p1 - p0;
        vec2 pa = uv - p0;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        // Add per-position jitter to make the line jagged
        vec2 jitter = vec2(sin(uv.x * 80.0 + seed_t * 5.0) * 0.008,
                           sin(uv.y * 80.0 + seed_t * 7.0) * 0.008);
        float d = length(pa - ba * h + jitter);
        // Bright thin arc
        float arc = smoothstep(0.012, 0.0, d) * (1.0 - arc_t * 2.0);
        // Color: bright white-cyan
        col = max(col, vec3(0.9, 1.0, 1.0) * arc);
        alpha = max(alpha, arc * 0.95);
    }

    // --- Counter-content tiles ---
    // Small randomly-placed bright squares (ad-hijack icons), localized
    // not banded. NO full-width horizontal stripes — those produce ugly
    // concentric-ring artifacts in fan view and ugly solid horizontal
    // lines in flat view.
    int n_tiles = int(2.0 + u_defiance * 6.0);
    for (int ti = 0; ti < 8; ti++) {
        if (ti >= n_tiles) break;
        float fi = float(ti);
        float life = 0.6 + hash(vec2(fi, 11.7)) * 0.6;
        float tt = mod(u_time + fi * 0.31, life) / life;
        if (tt > 0.4) continue;
        float seed_t = floor((u_time + fi * 0.07) / life);
        vec2 center = vec2(hash(vec2(seed_t * 9.0,  fi * 5.0)),
                           hash(vec2(seed_t * 11.0, fi * 7.0)));
        vec2 size = vec2(0.030 + hash(vec2(seed_t * 3.1, fi)) * 0.020,
                         0.020 + hash(vec2(seed_t * 4.7, fi)) * 0.015);
        vec2 d = abs(uv - center) / size;
        if (max(d.x, d.y) > 1.0) continue;
        float edge = max(smoothstep(0.85, 0.98, d.x), smoothstep(0.85, 0.98, d.y));
        float body = (1.0 - smoothstep(0.0, 1.0, max(d.x, d.y))) * 0.40 + edge * 1.0;
        // Each tile picks ONE color, mostly hot pink, occasional acid green
        float ch = hash(vec2(seed_t, fi + 17.0));
        vec3 tile_col = mix(vec3(1.0, 0.05, 0.55), vec3(0.40, 1.0, 0.10),
                            step(0.80, ch));
        col = max(col, tile_col * body);
        alpha = max(alpha, body * 0.85);
    }

    // --- Full-screen inversion flash ---
    // Brief (< 0.10s) flashes that cover the whole screen with high-contrast
    // single color. Rare; frequency scales with defiance.
    float flash_period = 5.0 - u_defiance * 3.0;
    float flash_t = mod(u_time, flash_period) / flash_period;
    float flash = smoothstep(0.0, 0.005, flash_t) * smoothstep(0.04, 0.005, flash_t);
    flash *= u_defiance;
    col = max(col, vec3(1.0, 1.0, 1.0) * flash);
    alpha = max(alpha, flash * 0.8);

    alpha = clamp(alpha * u_defiance, 0.0, 1.0);
    if (alpha < 0.04) discard;
    fragColor = vec4(col, alpha);
}
"""


class DefianceInversionEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 11.0   # Topmost — sparks visible above everything
        self.defiance_value = 0.0
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"DefianceInversion compile error: {e}")
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
        if not self.enabled:
            return
        if self.defiance_value < THRESHOLD:
            return
        u_d = (self.defiance_value - THRESHOLD) / (1.0 - THRESHOLD)
        u_d = max(0.0, min(1.0, u_d))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_defiance"), u_d)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
