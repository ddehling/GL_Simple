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

    // A smooth scrolling carrier waveform centered at mid-screen. Three
    // sines composed at different frequencies and scroll rates. The
    // signal value drives ONLY the color and slight thickness — no
    // per-frame jitter, no dropouts. The earlier jumpy/strobe-y
    // degradation effects were unpleasant; signal quality is now
    // expressed purely through the trace's color and thickness.

    float center_y = 0.5;

    // Smooth composite waveform — scroll_phase is CPU-integrated so
    // motion stays monotonic across signal-value interpolations.
    float wave = sin(uv.x * 25.0 - u_scroll_phase * 3.0) * 0.05
               + sin(uv.x * 12.0 + u_scroll_phase * 1.7) * 0.03
               + sin(uv.x * 42.0 - u_scroll_phase * 5.0) * 0.015;

    float wave_y = center_y + wave;
    float dist_to_wave = abs(uv.y - wave_y);

    // Trace thickness slightly increases at low signal (a "thicker
    // smear" reads as a degraded trace without strobing).
    float deg = 1.0 - u_signal;
    float thickness = 0.011 + deg * 0.006;
    float trace = smoothstep(thickness, 0.0, dist_to_wave);

    // Color: bright cyan-white at high signal, smoothly shifts toward
    // dimmer phosphor green at low signal. Pure interpolation — no
    // per-pixel hash, no flashes.
    vec3 hi  = vec3(0.20, 1.00, 0.95);
    vec3 lo  = vec3(0.10, 0.80, 0.35);
    vec3 col = mix(lo, hi, u_signal);

    float alpha = trace * 0.95;
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
