"""Forest canopy effect — leaf canopy band at the outer ring of the fan.

Renders bright leaf-cluster patches in the upper portion of the FBO
(uv.y > ~0.5), which on the fan corresponds to the outer ring (canopy
overhead). Sways gently with wind.

No tree trunks: vertical lines in FBO render as radial spokes on the
fan, which doesn't read as trees. The leaf-cluster band alone gives
the "forest canopy" feel.

Drives:
  canopy_density    -> leaf-cluster opacity / coverage
  wind              -> leaf-cluster sway
  season_preference -> color tint (cool dawn, warm midday/autumn)
  starryness        -> moonlit silver-green palette swap
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_forest_canopy(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(ForestCanopyEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized forest_canopy for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize forest_canopy: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.density = float(outstate.get('canopy_density', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))
        eff.season = float(outstate.get('season_preference', 0.5))
        eff.starryness = float(outstate.get('starryness', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 1.0  # Fast fade-in so canopy appears quickly.
        if elapsed < fade:
            f = elapsed / fade
        elif elapsed > total - fade:
            f = (total - elapsed) / fade
        else:
            f = 1.0
        eff.fade = float(np.clip(f, 0, 1))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


_VERT = """
#version 310 es
precision highp float;
in vec2 position;
out vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    // Depth 0.15: in front of clouds (depth 0.15-0.30) so clouds don't
    // occlude the canopy. Godrays (depth 0.10) render AFTER canopy and
    // alpha-blend on top, brightening shaft regions without erasing
    // the leaves/sky underneath.
    gl_Position = vec4(position, 0.15, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_density;
uniform float u_wind;
uniform float u_season;
uniform float u_starryness;
uniform float u_fade;

float hash(float n) { return fract(sin(n) * 43758.5453); }
float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise for leaf clusters.
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash2(i);
    float b = hash2(i + vec2(1.0, 0.0));
    float c = hash2(i + vec2(0.0, 1.0));
    float d = hash2(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// fbm — two octaves are plenty at 128px wide.
float fbm(vec2 p) {
    return vnoise(p) * 0.6 + vnoise(p * 2.0 + 5.7) * 0.4;
}

void main() {
    if (u_density < 0.02 || u_fade < 0.005) discard;

    vec2 uv = v_uv;
    float density = clamp(u_density, 0.0, 1.0);

    // Canopy band: covers top 80% of FBO = wide annulus on the outer
    // 80% of the fan's radial extent.
    if (uv.y < 0.20) discard;

    // Wind-swept noise sample.
    float leaf_sway = sin(u_time * 0.5) * u_wind * 0.04;
    vec2 leaf_p = vec2(uv.x * 12.0 + leaf_sway * 12.0,
                       uv.y * 22.0 + u_time * 0.04);
    float leaf_n = fbm(leaf_p);

    // Soft fade-in over a wider range (0.20 -> 0.55) so the inner edge
    // dissolves smoothly into open sky toward the fan's inner ring.
    float canopy_band = smoothstep(0.20, 0.55, uv.y);

    // Threshold: higher density → lower threshold → more pixels render
    // leaves. Range tuned so even max density leaves visible sky gaps.
    //   density=0.30 (aurora_grove) → thresh ~0.77 (very sparse, big gaps)
    //   density=0.85 (forest_dawn)  → thresh ~0.52 (dense with gaps)
    //   density=1.00 (forest_morning) → thresh ~0.45 (densest, still gapped)
    float leaf_thresh = mix(0.90, 0.45, density);
    float leaf_mask = smoothstep(leaf_thresh - 0.08,
                                 leaf_thresh + 0.12,
                                 leaf_n);

    // Only render LEAF cells. Gap cells discard so depth is not written
    // there — that lets canopy_godrays' sky backdrop (at depth 0.95)
    // remain visible AND allows clouds (depth 0.15-0.30) to drift in
    // front of the sky through the leaf gaps.
    float effective_mask = leaf_mask * canopy_band;
    if (effective_mask < 0.05) discard;

    float lit_amount = smoothstep(0.40, 0.90, leaf_n);

    // ---------- Leaf color ----------
    // Leaves stay GREEN across all daytime seasons so they contrast
    // against the warm/blue sky behind them.
    float warm_factor = 1.0 - smoothstep(0.0, 0.4, abs(u_season - 0.5));
    vec3 day_lit_midday = vec3(0.45, 0.95, 0.30);   // vivid midday green
    vec3 day_lit_dd     = vec3(0.70, 0.85, 0.25);   // yellow-green at dawn/dusk
    vec3 day_high = mix(day_lit_dd, day_lit_midday, warm_factor);
    vec3 day_dark = vec3(0.10, 0.30, 0.10);          // shadowed leaf interior
    vec3 day_col  = mix(day_dark, day_high, lit_amount);

    vec3 night_dark = vec3(0.06, 0.18, 0.12);
    vec3 night_lit  = vec3(0.30, 0.55, 0.32);
    vec3 night_col  = mix(night_dark, night_lit, lit_amount);

    vec3 leaf_col = mix(day_col, night_col, u_starryness);

    // Bright/lit leaves are nearly opaque; shadowed leaves slightly
    // translucent. effective_mask carries the soft band edge.
    float leaf_alpha = mix(0.50, 1.0, lit_amount);
    float max_leaf_alpha = mix(0.98, 0.85, u_starryness);
    float total_alpha = leaf_alpha * max_leaf_alpha * effective_mask * u_fade;
    if (total_alpha < 0.02) discard;

    fragColor = vec4(leaf_col, total_alpha);
}
"""


class ForestCanopyEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Foreground silhouettes — renders LATE so it alpha-blends on
        # top of the sky/aurora/sunrise/clouds underneath.
        self.render_priority = 7.0
        self._time = 0.0
        self.density = 0.0
        self.wind = 0.0
        self.season = 0.5
        self.starryness = 0.0
        self.fade = 0.0

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(_FRAG, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        self.VBOs.append(vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        self._time += dt

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        # Pure alpha-blend layering — never read or write depth.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, "u_starryness"), self.starryness)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
