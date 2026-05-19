"""
Cyber underway glow — the flooded BART tunnel aesthetic. Bioluminescent
algae blooms along tunnel walls (left/right edges), water surface
caustics at ankle height across the bottom. Cool teal/cyan with hints
of green. Reads `outstate['cyber_underway_intensity']` (0..1).

This is used by UNDERSIDE_FLOOD and surfaces in other arcs (Saint of
the Underway, Listening Glass — the sleeper location).
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_underway_glow(state, outstate, intensity=0.8):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberUnderwayGlowEffect,
                                          intensity=intensity)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_underway_glow] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.intensity = float(outstate.get('cyber_underway_intensity', intensity))

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
uniform float u_drift_phase;
uniform float u_intensity;
out vec4 fragColor;

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = fract(sin(dot(i, vec2(12.9898, 78.233))) * 43758.5453);
    float b = fract(sin(dot(i + vec2(1, 0), vec2(12.9898, 78.233))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0, 1), vec2(12.9898, 78.233))) * 43758.5453);
    float d = fract(sin(dot(i + vec2(1, 1), vec2(12.9898, 78.233))) * 43758.5453);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float alpha = 0.0;

    // ── 1. WATERLINE ──────────────────────────────────────────────────
    // A clear horizontal band at y = 0.55 with a gentle wavy displacement.
    // This is the surface the viewer is wading through. On the fan view
    // this becomes a bright arc curving around the viewer — unambiguously
    // reads as "water level."
    float waterline_y = 0.55
                      + sin(uv.x * 12.0 + u_drift_phase * 1.5) * 0.012
                      + sin(uv.x * 27.0 - u_drift_phase * 0.9) * 0.006;
    float waterline_d = abs(uv.y - waterline_y);
    // Wider waterline so it reads from a distance — was 0.012, now 0.020
    float waterline = smoothstep(0.020, 0.0, waterline_d);
    // Trail of mist just above the surface — slightly stronger
    float mist = smoothstep(0.080, 0.020, waterline_y - uv.y)
               * step(uv.y, waterline_y) * 0.35;
    vec3 waterline_col = vec3(0.65, 1.0, 1.0);     // brighter cyan-white
    col = max(col, waterline_col * (waterline + mist));
    alpha = max(alpha, waterline + mist * 0.85);

    // ── 2. SUBMERGED CAUSTICS ─────────────────────────────────────────
    // Below the waterline: bright crisscross caustic patterns animated
    // like sunlight through rippling water (in this case, neon-light
    // through it). Covers the entire submerged region, not just a strip.
    if (uv.y > waterline_y) {
        float submerged_t = (uv.y - waterline_y) / max(1.0 - waterline_y, 0.001);
        // Caustic field
        vec2 cq1 = vec2(uv.x * 9.0 + u_drift_phase * 0.40,
                        uv.y * 18.0 + u_drift_phase * 0.20);
        vec2 cq2 = vec2(uv.x * 17.0 - u_drift_phase * 0.30 + 13.7,
                        uv.y * 11.0 + u_drift_phase * 0.50 + 7.3);
        float c1 = vnoise(cq1);
        float c2 = vnoise(cq2);
        // Caustics peak where the two noise fields constructively interfere
        float caustic = pow(max(c1 + c2 - 0.95, 0.0), 1.4) * 3.0;
        // Brighter near the surface, dimmer at depth
        float depth_fade = mix(1.0, 0.4, submerged_t);
        vec3 caustic_col = vec3(0.20, 0.90, 1.0);
        float caustic_bright = caustic * depth_fade;
        col = max(col, caustic_col * caustic_bright * 1.1);
        alpha = max(alpha, caustic_bright * 0.85);

        // Submerged ambient — a uniform tint everywhere below the waterline
        // so the lower half of the frame reads as "under water." Slight
        // darkening with depth so the bottom is gloomier than near the surface.
        vec3 ambient_col = vec3(0.04, 0.18, 0.26);
        float ambient = mix(0.45, 0.30, submerged_t);
        col = max(col, ambient_col * ambient);
        alpha = max(alpha, ambient * 0.55);
    }

    // ── 3. ABOVE-WATER MIST ───────────────────────────────────────────
    // Above the waterline: faint dark teal mist, so the air "above the
    // flood" reads as cold, dim, submarine-station-like.
    if (uv.y < waterline_y) {
        float above_t = (waterline_y - uv.y) / max(waterline_y, 0.001);
        vec3 above_col = vec3(0.05, 0.10, 0.14);
        float above_a = mix(0.20, 0.05, above_t);
        col = max(col, above_col * above_a);
        alpha = max(alpha, above_a * 0.35);
    }

    // ── 4. BIOLUMINESCENT ALGAE ───────────────────────────────────────
    // Scattered bright green points — algae blooms along the tunnel walls
    // (sides) AND scattered along the waterline. Not subtle: these are
    // the "pop" of the scene and the most distinctive signature.
    // 9 column × 7 row grid of POSSIBLE bloom positions, hash-gated to
    // ~40% lit, biased toward left/right thirds (the tunnel walls).
    const float COLS = 9.0;
    const float ROWS = 7.0;
    for (int rr = 0; rr < 7; rr++) {
        for (int cc = 0; cc < 9; cc++) {
            vec2 cell = vec2(float(cc), float(rr));
            float seed = hash(cell + 7.7);
            // Bias toward edges (tunnel walls) — center cells less likely
            float cx_norm = (float(cc) + 0.5) / COLS;
            float wall_bias = 1.0 - smoothstep(0.10, 0.50, abs(cx_norm - 0.5));
            // wall_bias = 1.0 at center, 0.0 at edges. We INVERT for the gate.
            float gate = 0.45 + wall_bias * 0.45;  // 0.45 at edges, 0.90 at center
            if (seed > 1.0 - gate * 0.55) continue;   // ~25-45% of cells lit

            // Cell center (jittered)
            float jx = (hash(cell + 1.3) - 0.5) * 0.5;
            float jy = (hash(cell + 2.7) - 0.5) * 0.5;
            float bx = (float(cc) + 0.5 + jx) / COLS;
            float by = (float(rr) + 0.5 + jy) / ROWS;

            // Bloom size + per-bloom slow pulse — larger, brighter to
            // shine through any smog or overlays drawn after this shader.
            float pulse = 0.75 + 0.25 * sin(u_drift_phase * 0.8 + seed * 13.7);
            float radius = 0.025 + hash(cell + 4.1) * 0.018;
            float d = length(uv - vec2(bx, by));
            float bloom = smoothstep(radius, 0.0, d) * pulse;
            // Outer glow halo — wider radius for visible bloom
            float halo = smoothstep(radius * 5.0, radius, d) * 0.55 * pulse;

            vec3 algae_col = vec3(0.20, 1.0, 0.55);    // bright bioluminescent green
            col = max(col, algae_col * (bloom + halo));
            alpha = max(alpha, bloom + halo * 0.55);
        }
    }

    alpha = clamp(alpha * u_intensity, 0.0, 1.0);
    if (alpha < 0.02) discard;
    fragColor = vec4(col, alpha);
}
"""


class CyberUnderwayGlowEffect(ShaderEffect):
    def __init__(self, viewport, intensity: float = 0.8):
        super().__init__(viewport)
        # Render LATE — the algae blooms / waterline / caustics are the
        # state's signature visual and need to read clearly. Previously
        # at 1.5 (atmospheric base) they got covered by everything above.
        # At 8.4 they draw in front of city/data/rain but below the
        # narrative-variable overlays (scan_lines, sparks).
        self.render_priority = 8.4
        self.intensity = intensity
        self._drift_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberUnderwayGlow compile error: {e}")
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
        self._drift_phase += dt

    def render(self, state: Dict):
        if not self.enabled or self.intensity < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_drift_phase"), self._drift_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_intensity"), self.intensity)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
