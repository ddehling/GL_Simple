"""
Signal carrier — variable shader for `story_signal`.

HIGH signal (→1.0): a coherent slow-scrolling waveform of clean
horizontal traces — the protagonist's broadcast/perception is clear.
LOW signal (just above the 0.2 gate): static, RGB tearing, dropouts.

Threshold-gated at 0.2 (below = invisible). Scales effect strength
linearly from 0 → 1 as the variable climbs from 0.2 → 1.0.

This is one of six NARRATIVE-VARIABLE shaders. It runs continuously
during cyberpunk arcs and responds to the active node's variable values.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.2


def shader_signal_carrier(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(SignalCarrierEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[signal_carrier] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.signal_value = float(outstate.get('story_signal', 0.0))

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
uniform float u_signal;          // 0..1 after threshold remap
uniform float u_scroll_phase;
uniform float u_time;
out vec4 fragColor;

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float alpha = 0.0;

    // THREE stacked oscilloscope traces — reads as a broadcast monitor
    // without crowding the field. Each trace has its own frequency /
    // scroll direction / amplitude so the bundle looks like a real
    // multi-channel scope. Was four traces but at full signal the
    // bundle was visually overwhelming; three breathes better.
    float ys[3];      ys[0]    = 0.25; ys[1]    = 0.50; ys[2]    = 0.75;
    float freqs[3];   freqs[0] = 18.0; freqs[1] = 30.0; freqs[2] = 22.0;
    float speeds[3];  speeds[0]=  2.8; speeds[1]= -3.4; speeds[2]=  4.0;
    float amps[3];    amps[0]  = 0.040; amps[1] = 0.028; amps[2] = 0.048;

    // Traces are notably THICKER than the original — at the small canvas
    // size, the old 0.011 thickness was effectively sub-pixel in fan
    // view. 0.022..0.034 reads as a real trace.
    float thickness = 0.022 + u_signal * 0.012;

    // Fan-radial reveal: in fan space, v_uv.y = 0 sits at the fan
    // origin (inner arc) and v_uv.y = 1 sits at the outer arc — so
    // "grow radially outward from the fan origin" maps to "grow
    // upward from v_uv.y = 0 in the rectangle". At low signal only a
    // small band near v_uv.y = 0 is visible; as signal climbs the band
    // expands toward v_uv.y = 1. Combined with the dropout-dashes
    // (below) this makes low signal feel like a faint broken inner-arc
    // patch and high signal like a full clean broadcast spanning the
    // fan, rather than all four traces popping in at once.
    float reveal_top    = 0.10 + u_signal * 0.95;    // 0.10 → 1.05
    float reveal_edge   = 0.12;
    float reveal_mask   = 1.0 - smoothstep(reveal_top - reveal_edge, reveal_top, uv.y);

    for (int i = 0; i < 3; i++) {
        float wave   = sin(uv.x * freqs[i] + u_scroll_phase * speeds[i]) * amps[i];
        float wave_y = ys[i] + wave;
        float dist   = abs(uv.y - wave_y);
        float band   = smoothstep(thickness, 0.0, dist);

        // Dropout-dashes at LOW signal: each trace breaks into gaps
        // when signal degrades. At u_signal=1 the trace is solid.
        float dash_period = fract(uv.x * 6.0 + float(i) * 0.4);
        float min_visible = (1.0 - u_signal) * 0.45;
        float continuity  = step(min_visible, dash_period);
        float trace = band * continuity * reveal_mask;

        // Color: cyan-white at high signal → dim phosphor green at low.
        // Alpha lowered (0.98 → 0.55) so the trace bundle sits as a
        // legible overlay instead of dominating the frame.
        vec3 trace_col = mix(vec3(0.10, 0.85, 0.35),
                             vec3(0.35, 1.00, 0.95), u_signal);
        col   = max(col, trace_col * trace * 0.95);
        alpha = max(alpha, trace * 0.55);
    }

    if (alpha < 0.03) discard;
    fragColor = vec4(col, alpha);
}
"""


class SignalCarrierEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 10.0
        self.signal_value = 0.0
        self._time = 0.0
        self._scroll_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"SignalCarrier compile error: {e}")
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
        self._scroll_phase += dt

    def render(self, state: Dict):
        if not self.enabled:
            return
        # Threshold gate: render nothing below 0.2
        if self.signal_value < THRESHOLD:
            return
        # Remap [THRESHOLD, 1.0] → [0, 1] for shader
        u_signal = (self.signal_value - THRESHOLD) / (1.0 - THRESHOLD)
        u_signal = max(0.0, min(1.0, u_signal))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_signal"), u_signal)
        glUniform1f(glGetUniformLocation(self.shader, "u_scroll_phase"), self._scroll_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
