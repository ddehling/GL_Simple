"""
Velocity streaks — variable shader for `story_velocity`.

HIGH velocity (→1.0): dense horizontal motion-blur streaks sweep across
the frame; particles trail with long lines; the whole screen reads as
kinetic motion.
LOW velocity (just above 0.2): a few short streaks.
ZERO/below threshold: dead still — perfect for interior nodes.

Threshold-gated at 0.2. Scales 0→1 as variable goes 0.2→1.0.

Direction is configurable via outstate['velocity_direction'] — a
2-element tuple (dx, dy) in unit vector space. Default: rightward.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.2


def shader_velocity_streaks(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(VelocityStreaksEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[velocity_streaks] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.velocity_value = float(outstate.get('story_velocity', 0.0))

    # Direction bias: a 2-tuple (dx, dy) in unit-vector space. Used by arcs
    # to set motion direction (e.g. Subroutine 9 = (0, -1) for upward,
    # Faraday Run = (1, 0) for rightward).
    direction = outstate.get('velocity_direction', (1.0, 0.0))
    if direction is not None and len(direction) == 2:
        eff.direction = (float(direction[0]), float(direction[1]))

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
uniform float u_velocity;        // 0..1 after threshold
uniform float u_flow_phase;      // CPU-integrated
uniform vec2 u_dir;              // unit-vector direction
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec2 uv = v_uv;

    // Per-pixel position in direction-aligned coords.
    vec2 perp = vec2(-u_dir.y, u_dir.x);
    float along = dot(uv, u_dir);
    float across = dot(uv, perp);

    // Each streak is a SHORT DASH at a random along-position, in a row
    // indexed by perp position. Each dash has independent spawn time so
    // dashes do not align into a continuous horizontal/vertical line.
    // Few rows, sparse spawns, short dashes — no full-width banding.
    float n_rows = 10.0 + u_velocity * 10.0;     // 10..20 rows
    float row = floor(across * n_rows);
    float row_frac = fract(across * n_rows);
    float row_seed = hash(vec2(row, 1.7));
    // Sparsity gate
    if (row_seed > 0.45 + u_velocity * 0.35) discard;

    // Per-row dash speed (varies the kinetic feel)
    float row_speed = 0.5 + hash(vec2(row, 3.1)) * 1.2;

    // Position of dash leading edge in 'along' axis. fract gives one
    // dash position per row, advancing with u_flow_phase. To prevent
    // the dash spanning the full width, we cap dash length below.
    float lead = fract(u_flow_phase * row_speed * 0.5 + hash(vec2(row, 5.3)));

    // Distance from THIS pixel to the dash leading edge (cyclic).
    float d_along = lead - along;
    // Wrap so a dash near the seam still works
    if (d_along > 0.5) d_along -= 1.0;
    if (d_along < -0.5) d_along += 1.0;

    // Dash length — short. Hard-capped so even at max velocity a dash
    // is at most ~15% of the screen, never a full-width bar.
    float dash_len = 0.04 + u_velocity * 0.10;     // 0.04..0.14

    // We want d_along in [0, dash_len] to be visible (dash tail behind
    // leading edge). Anything else: invisible.
    if (d_along < 0.0 || d_along > dash_len) discard;

    // Brightness fades from bright at leading edge (d_along=0) to zero
    // at tail (d_along=dash_len).
    float trail = 1.0 - (d_along / dash_len);

    // Cross-axis tightness — thin dashes
    float thickness = 0.30;
    float in_line = smoothstep(0.5 + thickness, 0.5 - thickness,
                                abs(row_frac - 0.5));

    float bright = trail * in_line;
    vec3 col = vec3(0.85, 0.95, 1.00);
    float alpha = bright * u_velocity * 0.90;
    if (alpha < 0.04) discard;
    fragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
"""


class VelocityStreaksEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 9.5    # Above dissolution, below sparks
        self.velocity_value = 0.0
        self.direction = (1.0, 0.0)
        self._flow_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"VelocityStreaks compile error: {e}")
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
        # Flow rate scales with velocity_value; phase-integrated since rate
        # interpolates with the variable
        rate = 0.5 + self.velocity_value * 3.0
        self._flow_phase += dt * rate

    def render(self, state: Dict):
        if not self.enabled:
            return
        if self.velocity_value < THRESHOLD:
            return
        u_v = (self.velocity_value - THRESHOLD) / (1.0 - THRESHOLD)
        u_v = max(0.0, min(1.0, u_v))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_velocity"), u_v)
        glUniform1f(glGetUniformLocation(self.shader, "u_flow_phase"), self._flow_phase)
        glUniform2f(glGetUniformLocation(self.shader, "u_dir"),
                    self.direction[0], self.direction[1])
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
