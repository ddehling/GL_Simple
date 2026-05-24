"""
Dread perimeter — variable shader for `story_dread`.

HIGH dread (→1.0): red/amber edge glow surrounds the entire frame; a
prowling searchlight cone sweeps across the field; brief red flash
pulses on the periphery — the city is looking AT the protagonist.
LOW dread (just above 0.2 gate): faint red edge tint, no sweeps.

Threshold-gated at 0.2. Scales 0→1 as variable goes 0.2→1.0.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.2


def shader_dread_perimeter(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DreadPerimeterEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[dread_perimeter] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.dread_value = float(outstate.get('story_dread', 0.0))

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
uniform float u_dread;            // 0..1 after threshold remap
uniform float u_sweep_phase;
uniform float u_time;
out vec4 fragColor;

void main() {
    vec2 uv = v_uv;
    vec3 col   = vec3(0.0);
    float alpha = 0.0;

    // --- RED CORNER VIGNETTE (always visible) ---
    // Subtle red wash radiating inward from the corners. Was 0.70× —
    // too prominent, painting half the screen red. Dialed back to 0.35
    // so it reads as a faint corner darken rather than a strong border.
    float r = length(uv - vec2(0.5)) * 2.0;          // 0 center, ~1.4 corners
    float vignette = smoothstep(0.45, 1.30, r);      // 0 in center half, 1 at corners
    float v_breath = 0.85 + 0.15 * sin(u_time * 1.0);
    float vign_a   = vignette * u_dread * v_breath * 0.35;
    col   = max(col, vec3(1.00, 0.08, 0.04) * vign_a * 1.10);
    alpha = max(alpha, vign_a);

    // --- Edge glow (thin red border) ---
    // Was 0.30 penetration and ×1.40 brightness — the red border was
    // overwhelming. Now 0.18 penetration (narrower band hugging the
    // edge) and dimmer overall.
    float dx = min(uv.x, 1.0 - uv.x);
    float dy = min(uv.y, 1.0 - uv.y);
    float edge_dist = min(dx, dy);
    float edge_band = 1.0 - smoothstep(0.0, 0.18, edge_dist);
    float breath = 0.92 + 0.08 * sin(u_time * 1.3);
    float edge_glow = edge_band * breath * u_dread;
    col   = max(col, vec3(1.0, 0.10, 0.05) * edge_glow * 0.85);
    alpha = max(alpha, edge_glow * 0.55);

    // --- Sweep beam (wider, brighter — no drones to compete with) ---
    float sweep_speed = 0.5 + u_dread * 0.5;
    float sweep_x = fract(u_sweep_phase * sweep_speed * 0.4);
    float beam_x  = abs(uv.x - sweep_x);
    beam_x = min(beam_x, 1.0 - beam_x);
    float beam_width = 0.045 + u_dread * 0.065;
    float sweep_beam = smoothstep(beam_width, 0.0, beam_x);
    sweep_beam *= mix(0.55, 1.0, 1.0 - uv.y);
    sweep_beam *= u_dread;
    col   = max(col, vec3(1.0, 0.20, 0.10) * sweep_beam * 1.30);
    alpha = max(alpha, sweep_beam * 0.92);

    // --- Alarm flash (faster cycle, brighter) ---
    float alarm_period = 3.0 - u_dread * 1.8;
    float alarm_t = fract(u_time / alarm_period);
    float alarm_pulse = smoothstep(0.0, 0.04, alarm_t) * smoothstep(0.16, 0.04, alarm_t);
    float alarm = alarm_pulse * edge_band * u_dread;
    col   = max(col, vec3(1.0, 0.05, 0.05) * alarm * 1.4);
    alpha = max(alpha, alarm * 1.2);

    if (alpha < 0.04) discard;
    fragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
"""


class DreadPerimeterEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 10.6   # HUD-style overlay: above skyline + most city, below ar_glitch
        self.dread_value = 0.0
        self._time = 0.0
        self._sweep_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"DreadPerimeter compile error: {e}")
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
        self._sweep_phase += dt

    def render(self, state: Dict):
        if not self.enabled:
            return
        if self.dread_value < THRESHOLD:
            return
        u_dread = (self.dread_value - THRESHOLD) / (1.0 - THRESHOLD)
        u_dread = max(0.0, min(1.0, u_dread))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_dread"), u_dread)
        glUniform1f(glGetUniformLocation(self.shader, "u_sweep_phase"), self._sweep_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
