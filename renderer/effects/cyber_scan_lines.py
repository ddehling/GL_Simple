"""
Cyber scan-lines — fullscreen CRT-style horizontal-line overlay with
chromatic-edge aberration and occasional vertical-sync roll. Reads
`outstate['cyber_scan_intensity']` (0..1) for overall strength and is
modulated upward by `story_signal` low-state (a glitchy signal means more
visible scan lines).
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


# ─────────────────────────────────────────────────────────────────────────────
# Event wrapper
# ─────────────────────────────────────────────────────────────────────────────

def shader_cyber_scan_lines(state, outstate, intensity=0.6):
    """CRT overlay. Intensity is the baseline; story_signal low-state and
    cyber_scan_intensity outstate can override per state/arc."""
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberScanLinesEffect, intensity=intensity)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_scan_lines] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # State-level intensity (set by weather state via outstate)
    base = float(outstate.get('scan_line_intensity', intensity))
    # Low signal adds glitch: signal 1.0 = pure, signal 0.0 = max scan-line noise
    signal = float(outstate.get('story_signal', 1.0))
    glitch_boost = (1.0 - signal) * 0.4
    eff.current_intensity = max(0.0, min(1.0, base + glitch_boost))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Effect class
# ─────────────────────────────────────────────────────────────────────────────

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
uniform float u_time;
uniform float u_intensity;
uniform vec2 u_resolution;
out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // CRT overlay — visible-but-not-stroby. Three components, all
    // visually distinct:
    //   1. Bright cyan scan-line highlight on every 3rd pixel row
    //      (ADDITIVE — reads clearly on LED, doesn't just dim the scene)
    //   2. Strong slow vertical roll band (cyan, moving downward at
    //      one full pass every ~12 s)
    //   3. Rare chromatic tear flash (magenta, ~25% chance per 2 s
    //      bucket, visible for ~60 ms — not a 30 Hz strobe)

    float line_y = v_uv.y * u_resolution.y;

    // --- Bright scan-line highlights ---
    // Every 6th row gets a thin cyan-white emissive tint. Sparser than
    // every-3rd so it doesn't paint cyan over half the screen, but the
    // cyan is preserved — scan lines reading slightly cyan is part of
    // the cyberpunk-CRT look.
    float row_idx = mod(floor(line_y), 6.0);
    float line_lit = step(row_idx, 0.0);     // 1.0 for every 6th row
    vec3 line_color = vec3(0.55, 0.95, 1.00);
    float line_alpha = line_lit * 0.30;

    // --- Slow vertical roll ---
    // Narrow soft band drifting downward — classic vertical-sync hold
    // wobble. Slimmer than before (0.12 vs 0.18 reach) and dimmer so
    // it accents rather than dominates.
    float roll_pos = fract(u_time * 0.085);
    float roll_dist = abs(v_uv.y - roll_pos);
    roll_dist = min(roll_dist, 1.0 - roll_dist);   // wrap-safe
    float roll = smoothstep(0.12, 0.02, roll_dist);
    vec3 roll_color = vec3(0.85, 0.95, 1.00);
    float roll_alpha = roll * 0.35;

    // --- Rare chromatic tear ---
    // Single bright magenta line, brief flash inside a 2 s bucket. About
    // 25% of buckets fire — so roughly one tear every 8 seconds.
    float tear_bucket = floor(u_time * 0.5);
    float tear_seed = hash(vec2(tear_bucket, 0.0));
    float tear_y = fract(tear_seed * 7.13);
    float tear_dist = abs(v_uv.y - tear_y);
    float tear_active = step(fract(u_time * 0.5), 0.06);
    float tear_fire = step(0.75, tear_seed);
    float tear = smoothstep(0.010, 0.0, tear_dist) * tear_active * tear_fire;
    vec3 tear_color = vec3(1.0, 0.10, 0.55);
    float tear_alpha = tear * 0.95;

    // --- Compose — pick brightest component per pixel ---
    vec3 color;
    float alpha;
    if (tear_alpha >= max(line_alpha, roll_alpha)) {
        color = tear_color;
        alpha = tear_alpha;
    } else if (roll_alpha >= line_alpha) {
        color = roll_color;
        alpha = roll_alpha;
    } else {
        color = line_color;
        alpha = line_alpha;
    }

    alpha *= u_intensity;
    if (alpha < 0.02) discard;
    fragColor = vec4(color, clamp(alpha, 0.0, 1.0));
}
"""


class CyberScanLinesEffect(ShaderEffect):
    def __init__(self, viewport, intensity: float = 0.6):
        super().__init__(viewport)
        self.render_priority = 10.5  # Near top of the cyber stack
        self.intensity = intensity
        self.current_intensity = 0.0  # set by wrapper each frame
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberScanLines shader compile error: {e}")
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
        if not self.enabled or self.current_intensity < 0.01:
            return

        # Pattern B: disable depth, draw, restore
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)

        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_intensity"), self.current_intensity)
        glUniform2f(glGetUniformLocation(self.shader, "u_resolution"),
                    float(self.viewport.width), float(self.viewport.height))
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)

        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
