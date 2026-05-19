"""
Cyber AR glitch — augmented-reality malfunction visuals for the
CYBER_AR_BLOOM state (and any other future state with high
glitch_probability).

The AR layer of the Bay Stack is tearing. The visual signature:

  * Diagonal chromatic-aberration WAVES sweep across the frame at
    varying speeds, leaving brief RGB-split ghost trails.
  * GLITCH TILES — small bright rectangles in violet / magenta /
    cyan, randomly placed, that flash for ~0.15 s then die. They are
    the "impossible geometry" fragments leaking through.
  * REALITY-TEAR SEAMS — thin jagged horizontal-ish lines that briefly
    appear, jitter, and fade. The seams in the AR fabric.
  * Recursive-halo RINGS around a few random pivot points — concentric
    thin rings expanding outward (the AR layer "echoing" a real-world
    feature).

Reads `outstate['glitch_probability']` (0..1). Threshold-gated at 0.4
so it stays invisible during normal cyber states and only activates
when the AR layer is actually in trouble (AR_BLOOM, GLITCH_FOG, etc.).
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


THRESHOLD = 0.40


def shader_cyber_ar_glitch(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberArGlitchEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_ar_glitch] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.glitch = float(outstate.get('glitch_probability', 0.0))

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
uniform float u_time;
uniform float u_glitch;          // 0..1 after threshold remap
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float hash1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float alpha = 0.0;

    // ── 1. Diagonal chromatic-aberration WAVES ──────────────────────────
    // A wide band sweeps diagonally across the screen, leaving an RGB
    // split at its leading edge. Cycle period 4.5s, the wave at varying
    // angles. Up to 3 waves active simultaneously at high glitch.
    int n_waves = int(1.0 + u_glitch * 2.0);
    for (int wi = 0; wi < 3; wi++) {
        if (wi >= n_waves) break;
        float fi = float(wi);
        float wave_period = 4.0 + hash1(fi * 7.3) * 3.0;       // 4..7s
        float t_cycle = mod(u_time + fi * 1.7, wave_period) / wave_period;
        // Visible window: 25% of the cycle
        if (t_cycle > 0.25) continue;

        float seed_t = floor((u_time + fi * 0.31) / wave_period);
        float angle = hash1(seed_t + fi * 13.0) * 6.28318;
        vec2 dir = vec2(cos(angle), sin(angle));
        // The wave's centerline position in projection-along-dir space.
        // Marches from -0.4 to 1.4 across the wave cycle.
        float center = mix(-0.4, 1.4, t_cycle * 4.0);
        float along = dot(uv, dir);
        float d = along - center;

        // Wide leading bright band (the wave itself)
        float band = smoothstep(0.18, 0.0, abs(d));
        // RGB SPLIT — three channels offset along the perp of dir
        vec2 perp = vec2(-dir.y, dir.x);
        float split_r = dot(uv + perp * 0.025, dir) - center;
        float split_g = d;
        float split_b = dot(uv - perp * 0.025, dir) - center;
        float er = smoothstep(0.08, 0.0, abs(split_r));
        float eg = smoothstep(0.08, 0.0, abs(split_g));
        float eb = smoothstep(0.08, 0.0, abs(split_b));
        vec3 rgb_split = vec3(er, eg, eb);

        float strength = band * 0.4 + max(max(er, eg), eb) * 0.7;
        col = max(col, rgb_split * 0.85);
        alpha = max(alpha, strength * 0.65);
    }

    // ── 2. Glitch TILES (impossible-geometry fragments) ─────────────────
    // Small bright rectangles flash briefly at random positions. Hot
    // violet / magenta / cyan palette — the AR layer leaking through.
    int n_tiles = int(2.0 + u_glitch * 8.0);
    for (int ti = 0; ti < 10; ti++) {
        if (ti >= n_tiles) break;
        float fi = float(ti);
        float life = 0.25 + hash1(fi * 11.1) * 0.18;            // 0.25..0.43 s
        float t01 = mod(u_time + fi * 0.13, life) / life;
        if (t01 > 0.45) continue;                                // visible 45% of life
        float seed_t = floor((u_time + fi * 0.07) / life);
        vec2 c = vec2(hash(vec2(seed_t * 9.0,  fi * 5.0)),
                      hash(vec2(seed_t * 11.0, fi * 7.0)));
        vec2 size = vec2(0.020 + hash(vec2(seed_t * 3.1, fi)) * 0.025,
                         0.015 + hash(vec2(seed_t * 4.7, fi)) * 0.020);
        vec2 d2 = abs(uv - c) / size;
        if (max(d2.x, d2.y) > 1.0) continue;
        float edge = max(smoothstep(0.75, 0.98, d2.x),
                         smoothstep(0.75, 0.98, d2.y));
        float body = (1.0 - smoothstep(0.0, 1.0, max(d2.x, d2.y))) * 0.55 + edge;
        float ch = hash(vec2(seed_t, fi + 17.0));
        vec3 tile_col;
        if (ch < 0.40)      tile_col = vec3(0.85, 0.10, 0.95);  // hot magenta
        else if (ch < 0.70) tile_col = vec3(0.20, 0.95, 1.00);  // cyan
        else                tile_col = vec3(0.55, 0.20, 1.00);  // electric violet
        col = max(col, tile_col * body);
        alpha = max(alpha, body * 0.92);
    }

    // ── 3. Reality-TEAR seams ───────────────────────────────────────────
    // Brief jagged horizontal-ish lines flash and fade. Lifecycle ~0.5 s
    // each, spawned at random y positions. Up to 3 active at high glitch.
    int n_tears = int(1.0 + u_glitch * 3.0);
    for (int qi = 0; qi < 4; qi++) {
        if (qi >= n_tears) break;
        float fi = float(qi);
        float life = 0.45 + hash1(fi * 17.7) * 0.30;
        float t01 = mod(u_time + fi * 0.23, life) / life;
        if (t01 > 0.55) continue;
        float seed_t = floor((u_time + fi * 0.19) / life);
        float tear_y = hash1(seed_t + fi * 23.0);
        // Jagged displacement of the line as we go across x
        float jag = sin(uv.x * 30.0 + seed_t * 13.0) * 0.012
                  + sin(uv.x * 70.0 - seed_t * 7.0) * 0.005;
        float d_tear = abs(uv.y - tear_y - jag);
        float tear_intensity = (1.0 - t01 / 0.55);              // fade across lifetime
        float tear = smoothstep(0.010, 0.0, d_tear) * tear_intensity;
        vec3 tear_col = vec3(0.95, 0.85, 1.00);                  // near-white violet tint
        col = max(col, tear_col * tear);
        alpha = max(alpha, tear * 0.95);
    }

    // ── 4. Recursive halo RINGS around pivot points ─────────────────────
    // Each pivot point emits 3 expanding concentric rings over its
    // lifetime — the "AR echo" of a real-world feature being doubled.
    int n_pivots = int(0.0 + u_glitch * 3.0);
    for (int pi = 0; pi < 3; pi++) {
        if (pi >= n_pivots) break;
        float fi = float(pi);
        float life = 1.8 + hash1(fi * 31.3) * 1.0;               // 1.8..2.8 s
        float t01 = mod(u_time + fi * 0.41, life) / life;
        float seed_t = floor((u_time + fi * 0.37) / life);
        vec2 pivot = vec2(hash(vec2(seed_t * 5.0,  fi * 3.0)),
                          hash(vec2(seed_t * 11.0, fi * 9.0)));
        float dist = length(uv - pivot);
        float ring_intensity = 0.0;
        // Three rings, staggered start, each expanding
        for (int ri = 0; ri < 3; ri++) {
            float ring_t = clamp(t01 - float(ri) * 0.20, 0.0, 1.0);
            float ring_r = ring_t * 0.50;                        // radius 0..0.5
            float ring_fade = (1.0 - ring_t) * step(0.02, ring_t);
            ring_intensity = max(ring_intensity,
                                  smoothstep(0.006, 0.0, abs(dist - ring_r)) * ring_fade);
        }
        vec3 ring_col = vec3(0.55, 0.20, 1.00);                  // electric violet
        col = max(col, ring_col * ring_intensity * 0.9);
        alpha = max(alpha, ring_intensity * 0.75);
    }

    alpha = clamp(alpha * u_glitch, 0.0, 1.0);
    if (alpha < 0.04) discard;
    fragColor = vec4(col, alpha);
}
"""


class CyberArGlitchEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # In front of scan_lines (10.5), below defiance sparks (11.0).
        # AR glitches are dramatic — should ride the top of the stack.
        self.render_priority = 10.7
        self.glitch = 0.0
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberArGlitch compile error: {e}")
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
        # Threshold gate — invisible below 0.4 so normal cyber states
        # don't get glitched
        if self.glitch < THRESHOLD:
            return
        u_g = (self.glitch - THRESHOLD) / (1.0 - THRESHOLD)
        u_g = max(0.0, min(1.0, u_g))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_glitch"), u_g)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
