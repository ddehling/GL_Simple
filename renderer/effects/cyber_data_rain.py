"""
Cyber data rain — descending bright "data packets" in vertical streams.

Replacement for an earlier Matrix-style cascading-glyph design that
fundamentally cannot read on the LED installation: at the per-receiver
resolution, character shapes turn into indistinguishable colored blocks
and the effect just becomes noisy multicolored flicker.

This redesign trades glyphs for clear discrete packets:
  • 16 streams across the wrap (sparser than before)
  • Each stream has a single bright HEAD (compact bright disc) with a
    short fading TAIL above it — reads as a falling point at any LED
    resolution.
  • Per-stream speed variation; density gated by `data_flow_rate`.
  • Cool palette (cyan-white head, fading green tail).

Pattern B fullscreen, with the head/tail rendered procedurally per
pixel. CPU-integrated fall phase scaled by data_flow_rate, so heavier
data flow = faster packets and more active streams.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_data_rain(state, outstate, density=0.6):
    """Cascading data-packet streams. The wrapper name is preserved so
    event_map / weather_set registrations keep working — only the visual
    has changed."""
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberDataRainEffect, density=density)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_data_rain] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.density = float(outstate.get('data_flow_rate', density))

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
uniform float u_fall_phase;     // CPU-integrated dt * rate
uniform float u_density;         // 0..1, gates how many streams are lit
uniform vec2 u_resolution;
out vec4 fragColor;

float hash(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    // 16 vertical stream columns across the wrap.
    const float COLS = 16.0;
    float col_x = v_uv.x * COLS;
    float col_idx = floor(col_x);
    float col_frac = fract(col_x);

    // Column gate — only ~density fraction of streams are lit at a time
    float col_seed = hash(col_idx * 1.31);
    if (col_seed > clamp(u_density * 1.3, 0.05, 1.0)) discard;

    // Per-stream attributes
    float speed_var = 0.7 + hash(col_idx * 7.7) * 1.6;   // 0.7..2.3
    float phase_off = hash(col_idx * 13.3);
    float jitter_x  = (hash(col_idx * 23.1) - 0.5) * 0.4;

    // The HEAD's y-position drifts down with monotonic phase. Per-stream
    // independent offset so packets don't all align in a row.
    float head_y = fract(u_fall_phase * speed_var + phase_off);

    // Tail length — short. Hard-capped so a tail is at most ~12% of screen.
    float tail_len = 0.06 + speed_var * 0.025;     // 0.08..0.12

    // Vertical distance from this pixel to the head. Positive = pixel
    // is above the head (in the tail region).
    float dy = head_y - v_uv.y;

    // Visible window: dy in [-0.004, tail_len].
    if (dy < -0.004 || dy > tail_len) discard;

    // Head intensity: peaks at dy=0, very narrow
    float head = exp(-dy * dy * 4000.0);            // tight gaussian

    // Tail intensity: linear fade from 0.5 at head to 0 at tail end
    float tail = max(0.0, 1.0 - dy / tail_len) * 0.55;
    tail *= step(0.0, dy);                          // tail is only above head

    // Combine — head dominates near dy=0, tail covers the trail.
    float bright = max(head, tail);

    // Cross-axis: thin packet, centered in column (with jitter).
    float center_x = 0.5 + jitter_x;
    float dx = abs(col_frac - center_x);
    float thickness = 0.12;
    float in_packet = smoothstep(thickness, 0.0, dx);

    bright *= in_packet;

    // Color: bright cyan-white head, fading toward medium green for tail
    vec3 head_col = vec3(0.85, 1.00, 0.95);
    vec3 tail_col = vec3(0.10, 0.95, 0.40);
    // head_t ~ 1 right at the head, falls off quickly along the tail
    float head_t = smoothstep(0.0, 0.012, -dy + 0.012);   // 1 at head, 0 a few pixels above
    vec3 col = mix(tail_col, head_col, head_t);

    float alpha = clamp(bright, 0.0, 1.0);
    if (alpha < 0.04) discard;
    fragColor = vec4(col, alpha);
}
"""


class CyberDataRainEffect(ShaderEffect):
    def __init__(self, viewport, density: float = 0.6):
        super().__init__(viewport)
        # Late in the stack so packets read clearly over the city/signs
        # backdrop, but below scan_lines (10.5) and the narrative-variable
        # sparks (defiance at 11.0).
        self.render_priority = 8.0
        self.density = density
        self._fall_phase = 0.0       # CPU-integrated rate

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberDataRain compile error: {e}")
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
        # Density modulates fall rate — denser data flows faster.
        # Rate is CPU-integrated so it stays monotonic across density
        # interpolations (per shader_info.txt rules).
        rate = 0.18 + self.density * 0.45
        self._fall_phase += dt * rate

    def render(self, state: Dict):
        if not self.enabled or self.density < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_fall_phase"), self._fall_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform2f(glGetUniformLocation(self.shader, "u_resolution"),
                    float(self.viewport.width), float(self.viewport.height))
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
