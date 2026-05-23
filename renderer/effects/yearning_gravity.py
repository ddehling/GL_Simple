"""
Yearning gravity — variable shader for `story_yearning`.

HIGH yearning (→1.0): a single warm pink/gold point of light, slow-
orbiting near a corner of the frame, with strong chromatic-soft bloom.
Everything else slightly darkens toward it (radial vignette).
LOW yearning (just above 0.2): faint warm spark, no orbit.

Threshold-gated at 0.2. Scales 0→1 as variable goes 0.2→1.0.
"""
import ctypes
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.2


def shader_yearning_gravity(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(YearningGravityEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[yearning_gravity] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.yearning_value = float(outstate.get('story_yearning', 0.0))

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
uniform float u_yearning;        // 0..1 after threshold
uniform vec2 u_focal;            // CPU-orbited focal point (uv space)
out vec4 fragColor;

void main() {
    vec2 uv = v_uv;

    vec2  to_pixel = uv - u_focal;
    float d        = length(to_pixel);

    // Core / bloom / halo — bolder radii AND intensities than before.
    float core  = smoothstep(0.035, 0.0, d);    // was 0.020
    float bloom = smoothstep(0.22,  0.0, d);    // was 0.18
    float halo  = smoothstep(0.55,  0.0, d);    // was 0.45

    // Radial sun-ray spokes emanating from the focal point — the
    // strongest "pulled toward this" iconography. Eight sharp spokes,
    // sharpened with pow(), faded with distance so the focal stays the
    // brightest thing.
    float angle = atan(to_pixel.y, to_pixel.x);
    float spoke = 0.5 + 0.5 * cos(angle * 8.0);
    float ray_falloff = smoothstep(0.55, 0.10, d);
    float rays = pow(spoke, 3.0) * ray_falloff;

    vec3 c_core  = vec3(1.00, 0.90, 0.75);
    vec3 c_bloom = vec3(1.00, 0.55, 0.40);
    vec3 c_halo  = vec3(0.85, 0.30, 0.45);     // rosier than 0.65/0.20/0.30
    vec3 c_rays  = vec3(1.00, 0.65, 0.50);

    vec3 color = c_halo  * halo  * 0.55         // was 0.35
               + c_bloom * bloom * 0.90         // was 0.70
               + c_core  * core  * 1.25         // was 1.00
               + c_rays  * rays  * 0.45;        // new layer

    float alpha = (core * 1.25 + bloom * 0.75 + halo * 0.35 + rays * 0.30)
                  * u_yearning;

    // Anti-halo at opposite side — the "gravity pull" leaving the rest
    // of the frame slightly dimmed. Wider reach and stronger tint than
    // the previous near-invisible version.
    vec2 opp_focal = vec2(1.0) - u_focal;
    float od = length(uv - opp_focal);
    float anti_halo = smoothstep(0.50, 0.12, od);
    if (anti_halo > 0.0 && alpha < anti_halo * 0.35) {
        color = mix(vec3(0.02, 0.0, 0.04), vec3(0.18, 0.05, 0.10), anti_halo);
        alpha = anti_halo * 0.42 * u_yearning;
    }

    if (alpha < 0.03) discard;
    fragColor = vec4(color, clamp(alpha, 0.0, 1.0));
}
"""


class YearningGravityEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 8.5    # Above hologram/data, below scan_lines
        self.yearning_value = 0.0
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"YearningGravity compile error: {e}")
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

    def _focal_position(self):
        """Slow orbit near the upper-right quadrant. Period ~30s."""
        omega = 2.0 * math.pi / 30.0
        # Center the orbit in upper-right (0.70, 0.30); orbit radius 0.15
        cx = 0.70 + 0.15 * math.cos(self._time * omega)
        cy = 0.30 + 0.10 * math.sin(self._time * omega * 0.7)
        return cx, cy

    def render(self, state: Dict):
        if not self.enabled:
            return
        if self.yearning_value < THRESHOLD:
            return
        u_y = (self.yearning_value - THRESHOLD) / (1.0 - THRESHOLD)
        u_y = max(0.0, min(1.0, u_y))

        fx, fy = self._focal_position()

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_yearning"), u_y)
        glUniform2f(glGetUniformLocation(self.shader, "u_focal"), fx, fy)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
