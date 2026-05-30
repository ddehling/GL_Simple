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
    # Guard: a transition to a state without velocity_direction (and no
    # default to fall back to) makes the controller produce scalar 0 here.
    direction = outstate.get('velocity_direction', (1.0, 0.0))
    try:
        if len(direction) == 2:
            eff.direction = (float(direction[0]), float(direction[1]))
    except TypeError:
        pass

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
    vec3 col   = vec3(0.0);
    float alpha = 0.0;

    // The transit-corridor shader already does directional dashes
    // streaming through the field — so velocity does NOT do that.
    // Instead, velocity expresses the SENSORY experience of speed:
    // Doppler color shift, tunnel vision, and a forward shockwave.

    vec2 uv_c   = uv - vec2(0.5);
    float forward_t = dot(uv_c, u_dir);                // signed along motion

    // --- Doppler color split ---
    // Leading half (forward_t > 0) tints cool cyan; trailing half
    // tints warm red-orange. Strength rises with distance from the
    // perpendicular midline through screen center, sharpened with pow
    // so the edges are saturated and the midline is clean.
    float d_strength = pow(clamp(abs(forward_t) * 2.0, 0.0, 1.0), 1.3) * u_velocity;
    vec3 cool = vec3(0.25, 0.85, 1.00);
    vec3 warm = vec3(1.00, 0.32, 0.15);
    vec3 doppler_col = (forward_t > 0.0) ? cool : warm;
    col   = max(col, doppler_col * d_strength * 0.85);
    alpha = max(alpha, d_strength * 0.65);

    // --- Tunnel vignette ---
    // Darkens the corners — gives a strong sense of "everything outside
    // my forward path is blurring out". Subtle on top of doppler.
    float r = length(uv_c);
    float vignette = smoothstep(0.18, 0.55, r) * u_velocity;
    col   = mix(col, vec3(0.0, 0.0, 0.02), vignette * 0.45);
    alpha = max(alpha, vignette * 0.40);

    // --- Forward shockwave ring ---
    // Pulsing concentric rings expanding outward from a point shifted
    // toward the leading direction. Reads as "wavefront I'm pushing
    // through". Two overlapping rings at different phases so there is
    // always one visible.
    vec2 lead_point = vec2(0.5) + u_dir * 0.30;
    float ring_r = length(uv - lead_point);
    for (int i = 0; i < 2; i++) {
        float fi = float(i);
        float phase  = fract(u_flow_phase * 0.45 + fi * 0.5);
        float radius = phase * 0.55;
        float thickness = 0.018;
        float ring = smoothstep(thickness, 0.0, abs(ring_r - radius));
        ring *= (1.0 - phase);                          // fades as it expands
        ring *= u_velocity;
        col   = max(col, vec3(0.95, 1.00, 1.00) * ring * 1.40);
        alpha = max(alpha, ring * 0.90);
    }

    // --- Bright burst at the leading point ---
    float burst = smoothstep(0.075, 0.0, length(uv - lead_point));
    burst *= u_velocity * 1.30;
    col   = max(col, vec3(1.0, 1.0, 1.0) * burst);
    alpha = max(alpha, burst);

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
