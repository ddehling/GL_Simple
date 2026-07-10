"""
Cyber power-down — one-shot rolling grid failure for the transition
into CYBER_BLACKOUT.

The Bay Stack's grid dies in blocks, not all at once: a failure front
marches across the fan and each city sector stutters (brown-out
flicker) for a beat before going dark. Once the whole field is dark
the overlay holds, then lifts — reading as eyes adjusting to the
starlight the blackout state brings up underneath.

Scheduled via on_transition_events on the CYBER_BLACKOUT preset:

    "on_transition_events": [['cyber_power_down', 8, 0]],

The sweep is driven by u_progress = elapsed/duration integrated on the
CPU (never u_time x a varying uniform — see docs/shader_info.txt
"Time-based Animation"). u_time is used only at constant rate for the
flicker. The overlay only ever DARKENS (alpha-modulated near-black,
no additive brightness), so it is safe under the fan's brightness
limiter.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_power_down(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberPowerDownEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_power_down] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    duration = float(state.get('duration', 8.0)) or 8.0
    eff.progress = float(np.clip(state['elapsed_time'] / duration, 0.0, 1.0))

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
uniform float u_time;      // constant-rate flicker only
uniform float u_progress;  // 0..1 over the event, CPU-integrated
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    // The failure front crosses the field during the first 55% of the
    // event, everything holds dark until 70%, then the overlay lifts.
    float sweep = clamp(u_progress / 0.55, 0.0, 1.0);
    float lift = 1.0 - smoothstep(0.70, 1.0, u_progress);

    // City sectors: coarse blocks, each with its own failure delay and
    // flicker phase so the grid dies raggedly, not as a clean wipe.
    vec2 cell = floor(v_uv * vec2(24.0, 8.0));
    float jitter = hash(cell) * 0.18;
    float cell_pos = cell.x / 24.0;
    // Front travels 0 -> ~1.2 so the most-delayed cell still dies by
    // the end of the sweep window.
    float front = sweep * 1.25 - 0.02;
    float since_fail = front - (cell_pos + jitter);

    float dark = 0.0;
    if (since_fail > 0.0) {
        // Dead: darkness snaps in over a short slice of front travel.
        dark = smoothstep(0.0, 0.04, since_fail);
    } else if (since_fail > -0.06) {
        // Dying: brown-out stutter just ahead of the front. Alpha-only
        // modulation of the dark overlay — never adds light.
        float flick = step(0.5, fract(u_time * 13.0 + hash(cell + 7.0)));
        dark = 0.55 * flick * smoothstep(-0.06, 0.0, since_fail);
    }

    float alpha = dark * 0.85 * lift;
    if (alpha < 0.01) discard;
    // Not pure black: the faint blue of a city lit only by the sky.
    fragColor = vec4(vec3(0.010, 0.012, 0.030), alpha);
}
"""


class CyberPowerDownEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # In front of ar_glitch (10.7), below defiance sparks (11.0) —
        # the grid dying should darken everything the city draws.
        self.render_priority = 10.85
        self.progress = 0.0
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberPowerDown compile error: {e}")
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
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(self.uniform(self.shader, "u_time"), self._time)
        glUniform1f(self.uniform(self.shader, "u_progress"), self.progress)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
