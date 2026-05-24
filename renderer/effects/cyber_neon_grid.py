"""
Cyber neon grid — perspective ground-plane grid with cyan/magenta lines
receding toward a vanishing horizon. Lines pulse on audio
(`outstate['sound']`). Reads `outstate['cyber_neon_intensity']` for
overall brightness.

This is the iconic cyberpunk floor. Sparse lines (most of canvas is
black) + saturated colors = ideal under the brightness limiter.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_neon_grid(state, outstate, intensity=0.8):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberNeonGridEffect, intensity=intensity)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_neon_grid] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # Wrapper default 0.0 per docs/shader_info.txt.
    eff.intensity = float(outstate.get('neon_intensity', 0.0))
    # Audio energy (0..1 typical); some installations publish 'sound' as
    # a small float (the running mic envelope)
    sound = outstate.get('sound', 0.0)
    if isinstance(sound, (int, float)):
        eff.audio_level = float(sound)

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
uniform float u_scroll_phase;    // CPU-integrated scroll
uniform float u_intensity;
uniform float u_audio;
out vec4 fragColor;

void main() {
    // Grid lives in the bottom 55% of the screen (the "ground plane").
    // y_floor = 1.0 at the bottom of the visible grid, 0.0 at the horizon.
    float horizon = 0.55;             // v_uv.y of horizon
    if (v_uv.y < horizon) {
        discard;                       // above horizon = no grid
    }

    // Perspective remap: closer rows (bottom of screen) have wider line
    // spacing; far rows (near horizon) crowd toward the horizon.
    // floor at 0.18 (was 0.05) — that floor was small enough that lines
    // crowded to infinity at the horizon, merging into a single thick
    // bright cyan band. 0.18 caps perspective_z at ~5.5 so lines stay
    // resolvable all the way to the horizon.
    float t = (v_uv.y - horizon) / (1.0 - horizon);   // 0 at horizon, 1 at bottom
    float perspective_z = 1.0 / max(0.18, t);          // larger = farther

    // Horizontal lines (become concentric ARCS in fan view). THIN lines
    // for spatial intensity contrast: sparse bright pixels on a dark
    // ground reads stronger than thick washed-out bands. Per shader_info
    // contrast playbook: concentrate energy in features against darkness.
    float v_line_phase = perspective_z * 1.5;
    float v_line = abs(fract(v_line_phase) - 0.5) * 2.0;
    float v_thickness = 0.018 + t * 0.030;      // half as thick as before
    float v_line_mask = smoothstep(1.0 - v_thickness, 1.0, v_line);

    // Vertical lines (radial SPOKES in fan view). Thin spokes with
    // bright tips — high contrast against the dark ground.
    float h_line_phase = v_uv.x * 12.0;
    float h_line = abs(fract(h_line_phase) - 0.5) * 2.0;
    float h_thickness = 0.018 + (1.0 - t) * 0.030;
    float h_line_mask = smoothstep(1.0 - h_thickness, 1.0, h_line);

    // Audio pulse adds brightness uniformly to all lines
    float audio_boost = 1.0 + u_audio * 1.0;

    // Single color — uniform cyan across all lines. Alternating per-band
    // color creates rainbow rings under fan-view transformation.
    vec3 line_color = vec3(0.0, 0.85, 1.0);

    // Spokes (h_line) are 1.3x brighter than arcs (v_line) — radial spokes
    // are the cyberpunk-floor "look" in fan view; arcs are supporting.
    float line = max(v_line_mask * 0.55, h_line_mask);
    // Falloff toward horizon — distant lines dimmer
    line *= mix(0.4, 1.0, t);
    // Hard horizon fade: ramp lines up from 0 across the first 10%
    // past the horizon line. Without this the very first row of
    // visible pixels still has a bright line right at v_uv.y = 0.55,
    // forming a stripe at the discard boundary.
    line *= smoothstep(0.0, 0.10, t);

    float alpha = line * u_intensity * audio_boost * 0.85;
    if (alpha < 0.02) discard;

    fragColor = vec4(line_color, clamp(alpha, 0.0, 1.0));
}
"""


class CyberNeonGridEffect(ShaderEffect):
    def __init__(self, viewport, intensity: float = 0.8):
        super().__init__(viewport)
        # Render LATE — sparse line pattern gets washed out by the
        # building / sign / hologram alpha layers if drawn behind them.
        # Sits above buildings (6.0), signs (7.0), drones (6.5) but
        # below holograms (7.5) which can naturally float over the grid.
        self.render_priority = 7.2
        self.intensity = intensity
        self.audio_level = 0.0
        self._scroll_phase = 0.0      # CPU-integrated scroll

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberNeonGrid compile error: {e}")
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
        # Grid is static — scroll term removed from shader because under
        # perspective compression the apparent screen velocity is highly
        # non-uniform (slow at the bottom, accelerating toward the horizon)
        # and reads as jarring. Audio pulse provides motion.
        self._scroll_phase = 0.0

    def render(self, state: Dict):
        if not self.enabled or self.intensity < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_scroll_phase"), self._scroll_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_intensity"), self.intensity)
        glUniform1f(glGetUniformLocation(self.shader, "u_audio"), self.audio_level)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
