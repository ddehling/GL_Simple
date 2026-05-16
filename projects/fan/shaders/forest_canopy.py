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
from renderer.effects.base import ShaderEffect


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
        # Scale incoming density 1.5x for a denser-looking canopy. The shader
        # clamps u_density to [0, 1] so high-density states cleanly saturate.
        eff.density = float(outstate.get('canopy_density', 0.0)) * 1.5
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

    // ---------- Per-region wind field ----------
    // Slow-varying 2D direction field — each region of the canopy has
    // its own drift angle from a low-freq noise lookup, so different
    // patches sway in different directions. The field itself slowly
    // evolves over time (the wind aloft isn't static). Combined with
    // a time-oscillating amplitude (gust/rest cycle) and modulated by
    // overall wind strength (still air → minimal sway, breezy → clear
    // motion), individual leaves in a region move together while
    // neighboring regions drift differently.
    vec2 wf_p = vec2(uv.x * 2.5, uv.y * 3.0) + vec2(u_time * 0.05,
                                                     u_time * 0.03);
    float wind_angle = vnoise(wf_p) * 6.2832;
    vec2 wind_dir = vec2(cos(wind_angle), sin(wind_angle));
    // Per-region gust amplitude — sinusoidal over time with a spatial
    // offset so different regions are at different phases of their
    // gust cycle. Multiplied by u_wind so calm air shows almost no
    // motion, breezy air shows clear sway.
    float gust = 0.35 + 0.65 * sin(u_time * 0.40 + uv.x * 2.7 + uv.y * 1.9);
    float sway_amp = gust * (0.10 + 0.90 * clamp(abs(u_wind), 0.0, 1.5)) * 0.10;
    vec2 sway = wind_dir * sway_amp;

    // HIGH-FREQUENCY leaf grain (small individual leaves, not cloud puffs).
    // Sway shifts the sample position; different canopy regions take
    // samples from different offsets, so each region's leaves move
    // independently. Slow vertical drift retained as ambient canopy
    // animation (subtle growth/movement even when wind=0).
    vec2 leaf_p = vec2(uv.x * 30.0 + sway.x * 30.0,
                       uv.y * 55.0 + sway.y * 55.0 + u_time * 0.06);
    float leaf_n = fbm(leaf_p);

    // Soft fade-in over a wider range (0.20 -> 0.55) so the inner edge
    // dissolves smoothly into open sky toward the fan's inner ring.
    float canopy_band = smoothstep(0.20, 0.55, uv.y);

    // SPARSE-BRIGHT RESTRUCTURE — render only the brighter noise peaks.
    // Below threshold → discard (zero output). Density drives the
    // visibility threshold. Range chosen so peaceful_forest at typical
    // density values (0.45..1.0) shows recognizable canopy coverage,
    // not just scattered flecks — operator-tested at the previous
    // (0.74, 0.58) values the canopy read as "really sparse":
    //   density=0.45 → tip_lo 0.65 (moderate)
    //   density=1.00 → tip_lo 0.45 (dense)
    float tip_lo = mix(0.65, 0.45, density);
    // VISIBILITY (alpha): wider smoothstep so leaves on the edge of
    // threshold fade gently rather than pop in binary on/off.
    float visibility = smoothstep(tip_lo, tip_lo + 0.12, leaf_n);
    float effective = visibility * canopy_band;
    if (effective < 0.02) discard;

    // ---------- Per-leaf intensity from noise value itself ----------
    // INTENSITY CONTRAST per individual leaf: how far above threshold
    // a leaf's noise value sits determines its brightness. Leaves just
    // barely above threshold are DIM; leaves at high noise peaks are
    // BRIGHT. Range 0.20..1.00 so dimmest visible leaves are 20% of
    // brightest — substantial luminance variation across the canopy.
    //
    // Previously used a slow exposure noise (4.5/6 freq) which couldn't
    // distinguish adjacent leaves; all leaves in a region got the same
    // exposure. Now intensity varies per individual leaf because it's
    // sourced from the same high-freq noise that defines the leaves.
    float leaf_strength = clamp((leaf_n - tip_lo) / max(0.001, 1.0 - tip_lo),
                                0.0, 1.0);
    // Per-leaf intensity floor at 0.35: dim leaves still recognizable.
    float leaf_intensity = 0.35 + 0.65 * leaf_strength;

    // ---------- Per-region exposure ----------
    // SPATIAL intensity contrast — adjacent canopy patches at different
    // brightness levels, like sunlit clearings vs deeper shade. Driven
    // by a mid-frequency exposure noise (slower than the leaf grain so
    // many leaves share a brightness band, but fast enough that
    // multiple bands are visible across the canopy). Slowly drifts so
    // the bright/dim regions migrate gently. Range 0.25..1.00 → 4x
    // regional brightness spread; multiplied with per-leaf intensity
    // gives roughly 11x total brightest-to-dimmest spread (bright leaf
    // in sunny region vs dim leaf in shadow region). Without this layer
    // every region of the canopy reads at uniform overall brightness,
    // which is what made forest_midday feel flat even with dappled
    // shadows on top.
    vec2 expose_p = vec2(uv.x * 3.0 + u_time * 0.02, uv.y * 5.0);
    float region_brightness = 0.25 + 0.75 * fbm(expose_p);

    float intensity = leaf_intensity * region_brightness;

    // ---------- Leaf color (naturalistic) ----------
    // Naturalistic forest palette — earlier "neon lime" and "silver-blue
    // fairy moonlight" tested as visually wrong. These read as forest
    // leaves at different times of day rather than as synthetic
    // theatre lighting, while still distinct enough from sky/godray
    // hues to give clean hue separation under the brightness limiter.
    float warm_factor = 1.0 - smoothstep(0.0, 0.4, abs(u_season - 0.5));
    vec3 day_lit_midday = vec3(0.15, 0.75, 0.20);   // clear emerald green
    vec3 day_lit_dd     = vec3(0.55, 0.60, 0.15);   // warm olive
    vec3 day_col  = mix(day_lit_dd, day_lit_midday, warm_factor);
    // Night palette pushed toward silver-green (was dark blue-grey).
    // The previous (0.12, 0.20, 0.28) was too dim AND too blue — at
    // intensity*0.35 floor it produced uniformly dark-blue dots
    // through the night/rain states. (0.22, 0.42, 0.30) reads as
    // moonlit forest: clearly green-leaning but cool-shifted, with
    // enough luminance that the brightest leaves are visible.
    vec3 night_col = vec3(0.22, 0.42, 0.30);
    vec3 leaf_base = mix(day_col, night_col, u_starryness);

    vec3 leaf_col = leaf_base * intensity;

    float max_alpha = mix(0.98, 0.85, u_starryness);
    float total_alpha = effective * max_alpha * u_fade;
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
