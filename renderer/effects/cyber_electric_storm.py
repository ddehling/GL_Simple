"""
Cyber electric storm — branching electric arcs striking from the sky,
electric-blue palette (distinct from the warm-yellow `lightning` shader
used in non-cyber sets). Reads:

  outstate['lightning_probability']  — strike-rate driver (0..1)
  outstate['electric_interference']  — adds horizontal "interference"
                                        arcs running across the buildings

Each strike is a vertical bolt with branching forks rendered procedurally
in the fragment shader. Up to 4 simultaneous strikes. Strikes fade over
~0.4 s; new ones spawn stochastically per frame at a rate driven by
lightning_probability.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_electric_storm(state, outstate):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberElectricStormEffect)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_electric_storm] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.lightning_prob = float(outstate.get('lightning_probability', 0.0))
    eff.interference = float(outstate.get('electric_interference', 0.0))

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
uniform float u_lightning_prob;     // 0..1
uniform float u_interference;        // 0..1
uniform vec4  u_bolts[4];            // (x, life_start, life_end, seed) per bolt
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float hash1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

// Distance from pixel uv to a vertical jagged bolt centered at bolt_x,
// running from top of screen (y=0) to y=ground. seed controls the jag.
float bolt_distance(vec2 uv, float bolt_x, float ground, float seed) {
    // Jag the x-position along the bolt's length
    float jag = sin(uv.y * 80.0 + seed * 13.0) * 0.010
              + sin(uv.y * 24.0 + seed * 7.0) * 0.018
              + sin(uv.y * 8.0 + seed * 3.0) * 0.025;
    // Side-fork branches: a few short diagonal arcs off the main bolt
    float fork_x = bolt_x + jag;
    float d_main = abs(uv.x - fork_x);

    // Branch 1 — diagonal off mid-bolt
    float b1_y = 0.30 + hash1(seed) * 0.30;
    float b1_dy = uv.y - b1_y;
    float b1_dir = (hash1(seed * 2.7) - 0.5) * 0.5;  // left or right
    float b1_x = fork_x + b1_dy * b1_dir;
    float b1_active = step(b1_y, uv.y) * step(uv.y, b1_y + 0.18);
    float d_b1 = mix(1.0, abs(uv.x - b1_x), b1_active);

    // Branch 2 — second fork lower
    float b2_y = 0.55 + hash1(seed * 3.1) * 0.25;
    float b2_dy = uv.y - b2_y;
    float b2_dir = (hash1(seed * 4.9) - 0.5) * 0.4;
    float b2_x = fork_x + b2_dy * b2_dir;
    float b2_active = step(b2_y, uv.y) * step(uv.y, b2_y + 0.14);
    float d_b2 = mix(1.0, abs(uv.x - b2_x), b2_active);

    float d = min(d_main, min(d_b1, d_b2));
    // Cut off below ground
    if (uv.y > ground) d = 1.0;
    return d;
}

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float alpha = 0.0;

    // --- Main strikes (up to 4) ---
    // Each bolt: u_bolts[i].x = column x, .y = strike start time,
    //            .z = strike end time, .w = random seed
    for (int i = 0; i < 4; i++) {
        vec4 b = u_bolts[i];
        if (b.y >= b.z) continue;                  // inactive slot
        if (u_time < b.y || u_time > b.z) continue; // outside life window

        // Life curve: bright spike at start, fast decay
        float t01 = (u_time - b.y) / max(b.z - b.y, 0.001);
        float strike_intensity = (1.0 - t01) * (1.0 - t01);    // 1.0..0.0 quad

        // Bolt LENGTH scales with lightning_probability: at low prob,
        // strikes are short (mostly upper half); at high prob, they extend
        // all the way down to the floor.
        float base_ground = mix(0.40, 0.85, u_lightning_prob);
        float ground = base_ground + hash1(b.w * 0.31) * 0.15;
        float d = bolt_distance(uv, b.x, ground, b.w);

        // Bolt THICKNESS also scales with lightning_probability — fatter
        // strikes when the storm is heavy.
        float core_size = mix(0.003, 0.010, u_lightning_prob);
        float glow_size = mix(0.025, 0.080, u_lightning_prob);
        float core = smoothstep(core_size, 0.0, d);
        float glow = smoothstep(glow_size, 0.0, d) * 0.45;
        float strike = max(core, glow) * strike_intensity;

        // Color: white-hot core, electric-blue glow
        vec3 core_color = vec3(0.95, 0.98, 1.0);
        vec3 glow_color = vec3(0.20, 0.55, 1.00);
        vec3 c = mix(glow_color, core_color, core);
        col = max(col, c * strike);
        alpha = max(alpha, strike * 0.95);
    }

    // --- Interference arcs (horizontal twitches) ---
    // Brief horizontal lines running across building tops when
    // electric_interference > 0. NOT random per frame — fixed-period
    // shuffle (every 0.35s) so it doesn't strobe.
    if (u_interference > 0.05) {
        float ib = floor(u_time / 0.35);
        float row_seed = hash(vec2(ib, 1.7));
        float row_y = 0.20 + row_seed * 0.45;          // top half of screen
        float life_t = mod(u_time, 0.35) / 0.35;
        // Visible for first 15% of the period
        float live = step(life_t, 0.15);
        float fire = step(1.0 - u_interference * 0.6, hash(vec2(ib, 5.3)));
        float dist = abs(uv.y - row_y);
        // Jagged horizontal
        float jx = sin(uv.x * 60.0 + ib * 13.0) * 0.004;
        float d_arc = abs(uv.y - row_y - jx);
        float arc = smoothstep(0.008, 0.0, d_arc) * live * fire;
        vec3 arc_color = vec3(0.30, 0.60, 1.0);
        col = max(col, arc_color * arc);
        alpha = max(alpha, arc * 0.85);
    }

    if (alpha < 0.03) discard;
    fragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
"""


class CyberElectricStormEffect(ShaderEffect):
    """Manages up to 4 simultaneous bolt slots on the CPU; the GPU only
    draws the active ones each frame."""

    NUM_BOLTS = 8

    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 9.5   # In front of city, behind sparks
        self.lightning_prob = 0.0
        self.interference = 0.0
        self._time = 0.0
        # Each bolt: [x, t_start, t_end, seed]. t_start >= t_end means slot
        # is inactive (won't render).
        self._bolts = np.zeros((self.NUM_BOLTS, 4), dtype=np.float32)
        # Random state for bolt spawn
        self._rng = np.random.RandomState(20251217)

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberElectricStorm compile error: {e}")
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

        # Spawn rate: lightning_prob 1.0 -> ~5 strikes/sec on average.
        # Probability per frame ~ rate * dt. Combined with up to 8
        # concurrent bolt slots and per-strike duration 0.25-0.45s, this
        # gives a dense, near-continuous flash at max probability.
        rate = self.lightning_prob * 5.0
        if self._rng.random() < rate * dt:
            # Find an inactive slot (t_start >= t_end OR past life end)
            for i in range(self.NUM_BOLTS):
                if self._bolts[i, 2] < self._time:   # past death
                    bolt_x = self._rng.uniform(0.05, 0.95)
                    life = self._rng.uniform(0.25, 0.45)   # strike duration
                    seed = self._rng.uniform(0.0, 100.0)
                    self._bolts[i] = [bolt_x, self._time, self._time + life, seed]
                    break

    def render(self, state: Dict):
        if not self.enabled:
            return
        # If nothing happening, skip entirely
        if self.lightning_prob < 0.01 and self.interference < 0.01:
            return

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_lightning_prob"), self.lightning_prob)
        glUniform1f(glGetUniformLocation(self.shader, "u_interference"), self.interference)
        # Upload bolt array (4 vec4s)
        bolts_loc = glGetUniformLocation(self.shader, "u_bolts")
        if bolts_loc != -1:
            glUniform4fv(bolts_loc, self.NUM_BOLTS, self._bolts.flatten())
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
