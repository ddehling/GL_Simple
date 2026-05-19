"""
Dissolution drift — variable shader for `story_dissolution`.

HIGH dissolution (→1.0): bright particles drift UPWARD across the entire
frame (fade-to-end), a desaturation wash creeps in from the corners
toward the center, edges of features soften.
LOW dissolution (just above 0.2): rare upward particles.

Threshold-gated at 0.2. Scales 0→1 as variable goes 0.2→1.0.

The motion is UPWARD — pixels leaving the frame, like consciousness
dispersing. Velocity rate is CPU-integrated (u_drift_phase).
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.2


def shader_dissolution_drift(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DissolutionDriftEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[dissolution_drift] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.dissolution_value = float(outstate.get('story_dissolution', 0.0))

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
uniform float u_dissolution;     // 0..1 after threshold
uniform float u_drift_phase;     // CPU-integrated drift
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float alpha = 0.0;

    // --- Upward-drifting particles ---
    // Particles live on a grid that scrolls upward. Each grid cell has
    // a per-cell probability of being a bright particle.
    const float COLS = 30.0;
    const float ROWS = 25.0;
    // Scroll the grid upward
    float scroll_y = u_drift_phase * 0.30;
    float gx = uv.x * COLS;
    float gy = uv.y * ROWS + scroll_y;
    float gxi = floor(gx);
    float gyi = floor(gy);
    float gxf = fract(gx);
    float gyf = fract(gy);
    // Each cell has its own particle visibility, position-within-cell,
    // and intensity. Cells reseed each grid period as they scroll past.
    float seed = hash(vec2(gxi, gyi));
    float on = step(0.85, seed) * u_dissolution * 1.4;   // sparse at low, denser at high
    on = clamp(on, 0.0, 1.0);

    // Particle position inside the cell (offset)
    float px = hash(vec2(gxi, gyi) + 7.7);
    float py = hash(vec2(gxi, gyi) + 13.1);
    vec2 cell_offset = vec2(px, py);
    float pd = length(vec2(gxf, gyf) - cell_offset);
    float particle = smoothstep(0.20, 0.0, pd) * on;

    // Slight smear upward (vertical motion-blur)
    float blur = smoothstep(0.40, 0.0, abs(gxf - cell_offset.x))
               * smoothstep(0.50, 0.0, max(gyf - cell_offset.y, 0.0))
               * on * 0.5;
    particle = max(particle, blur);

    // Cool white-blue particle color (the "draining" color)
    vec3 particle_color = vec3(0.85, 0.95, 1.00);
    col = particle_color * particle;
    alpha = particle;

    // --- Corner desaturation wash ---
    // A subtle gray overlay creeps in from the 4 corners; at max dissolution
    // it occupies roughly the outer 30% of the frame.
    float corner_x = min(uv.x, 1.0 - uv.x);
    float corner_y = min(uv.y, 1.0 - uv.y);
    float corner_d = min(corner_x, corner_y);
    float wash_reach = 0.05 + u_dissolution * 0.30;
    float wash = 1.0 - smoothstep(0.0, wash_reach, corner_d);
    wash *= u_dissolution;
    vec3 wash_color = vec3(0.4, 0.4, 0.5);
    if (wash > alpha) {
        col = wash_color;
        alpha = wash * 0.35;
    }

    alpha = clamp(alpha, 0.0, 1.0);
    if (alpha < 0.03) discard;
    fragColor = vec4(col, alpha);
}
"""


class DissolutionDriftEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 9.0    # Above most layers, below sparks
        self.dissolution_value = 0.0
        self._drift_phase = 0.0       # CPU-integrated rate

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"DissolutionDrift compile error: {e}")
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
        # Drift rate is constant per-class, but scaled by dissolution_value.
        # Phase form keeps the shader safe even though rate effectively
        # interpolates.
        rate = 0.5 + self.dissolution_value * 1.5
        self._drift_phase += dt * rate

    def render(self, state: Dict):
        if not self.enabled:
            return
        if self.dissolution_value < THRESHOLD:
            return
        u_d = (self.dissolution_value - THRESHOLD) / (1.0 - THRESHOLD)
        u_d = max(0.0, min(1.0, u_d))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_dissolution"), u_d)
        glUniform1f(glGetUniformLocation(self.shader, "u_drift_phase"), self._drift_phase)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
