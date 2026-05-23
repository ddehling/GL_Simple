"""
Cyber transit flow — passing reflective light bands on the walls of a
maglev / tunnel corridor. Bands sweep horizontally across the screen
at high speed, with parallax depth bands. Reads
`outstate['cyber_transit_intensity']` (0..1).

For TRANSIT_CORRIDOR state and Faraday-Run / Repossessor arc motion.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_transit_flow(state, outstate, intensity=0.7):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberTransitFlowEffect,
                                          intensity=intensity)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_transit_flow] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # Wrapper default is 0.0 per docs/shader_info.txt — states that
    # want transit packets MUST set cyber_transit_intensity explicitly
    # (only CYBER_TRANSIT_CORRIDOR does, at 1.0).
    eff.intensity = float(outstate.get('cyber_transit_intensity', 0.0))

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
uniform float u_flow_phase;      // CPU-integrated outward flow
uniform float u_intensity;
out vec4 fragColor;

float hash1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    // FAN-NATIVE TRANSIT MOTION
    // The fan maps v_uv.x → angular sweep and v_uv.y → radius
    // (0 = fan origin / inner arc, 1 = outer arc / "far away").
    // So a vertical line in v_uv = a radial spoke on the fan.
    //
    // Each angular "lane" carries a light packet that flows FROM the
    // fan origin (small v_uv.y, vanishing point) OUT toward the outer
    // arc (large v_uv.y, near the camera). Packets grow as they
    // approach the outer arc — perspective scaling — selling the
    // "lights racing past in a tunnel" feel both in flat AND in fan.

    // 30 angular lanes across the wrap. v_uv.x is the angular coord,
    // multiplied by lane count to get an integer lane index + fractional
    // offset within the lane.
    const float N_LANES = 30.0;
    float lane_f = v_uv.x * N_LANES;
    float lane_idx = floor(lane_f);
    float lane_frac = fract(lane_f);

    // Per-lane attributes — random speed, phase, dash length
    float lane_seed = hash1(lane_idx);
    float lane_speed = 0.55 + lane_seed * 1.8;        // 0.55..2.35
    float lane_phase_off = hash1(lane_idx + 17.3);
    float dash_len = 0.18 + 0.18 * hash1(lane_idx + 31.7);   // 0.18..0.36

    // Packet leading-edge position along the lane's radial axis.
    // Wraps continuously — packets keep streaming outward; when one
    // exits the outer arc, another emerges at the inner arc.
    float head_y = fract(u_flow_phase * lane_speed + lane_phase_off);

    // Pixel's position relative to the packet head
    float dy = head_y - v_uv.y;

    // Visible region: this pixel must be IN the packet (head ± dash_len).
    // dy > 0 means pixel is INSIDE (closer to fan origin than head — the
    //   tail trails toward the vanishing point, which is correct).
    // dy < 0 means pixel is past the head — bright tip with small bloom.
    float tail_alive = step(0.0, dy) * step(dy, dash_len);
    float head_bloom = step(-0.012, dy) * step(dy, 0.0);

    if (tail_alive < 0.5 && head_bloom < 0.5) discard;

    // Brightness profile along the tail (head_t = 1 at head, 0 at tail end)
    float head_t = tail_alive > 0.5 ? (1.0 - dy / dash_len) : 1.0;

    // Lane thickness (cross-angular) — keep the spoke crisp
    float lane_thickness = 0.30;
    float in_lane = smoothstep(0.5 + lane_thickness, 0.5 - lane_thickness,
                                abs(lane_frac - 0.5));

    // Perspective scaling: packets at LARGE v_uv.y (outer arc on fan,
    // close to "camera") are brighter than packets at SMALL v_uv.y
    // (near vanishing point). Sells the racing-past-fast feel.
    float perspective = mix(0.25, 1.10, v_uv.y);

    float bright = head_t * in_lane * perspective;
    // White-hot head, cyan tail
    vec3 col = mix(vec3(0.30, 0.85, 1.00), vec3(0.95, 1.00, 1.00), head_t);

    float a = bright * u_intensity * 0.95;
    if (a < 0.04) discard;
    fragColor = vec4(col, clamp(a, 0.0, 1.0));
}
"""


class CyberTransitFlowEffect(ShaderEffect):
    def __init__(self, viewport, intensity: float = 0.7):
        super().__init__(viewport)
        # Render LATE — in CYBER_TRANSIT_CORRIDOR the transit flow IS the
        # dominant visual, so it has to draw on top of city_skyline (6.0),
        # neon_signs (7.0), hologram_billboards (7.5), data_rain (8.0).
        # Priority 2.0 (atmospheric base) buries the sparse bright edges
        # under every other Pattern B layer's alpha, making it invisible.
        self.render_priority = 8.6
        self.intensity = intensity
        self._flow_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberTransitFlow compile error: {e}")
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
        self._flow_phase += dt

    def render(self, state: Dict):
        if not self.enabled or self.intensity < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_flow_phase"), self._flow_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_intensity"), self.intensity)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
