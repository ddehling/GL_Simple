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

    // Wind-swept HIGH-FREQUENCY leaf grain. Much higher spatial
    // frequency than the old version (12/22 -> 30/55) so blobs are
    // small and read as individual leaves, not cloud puffs. The
    // previous low-frequency mask produced smooth round patches that
    // looked like sparse green clouds — wrong shape for canopy.
    float leaf_sway = sin(u_time * 0.5) * u_wind * 0.04;
    vec2 leaf_p = vec2(uv.x * 30.0 + leaf_sway * 30.0,
                       uv.y * 55.0 + u_time * 0.06);
    float leaf_n = fbm(leaf_p);

    // Soft fade-in over a wider range (0.20 -> 0.55) so the inner edge
    // dissolves smoothly into open sky toward the fan's inner ring.
    float canopy_band = smoothstep(0.20, 0.55, uv.y);

    // SPARSE-BRIGHT RESTRUCTURE — render only the brightest noise peaks
    // (visible leaf clusters). Everything below the threshold discards
    // entirely (contributes zero output instead of "leaf body at dim
    // color"). Under brightness_limit=0.1 the un-painted pixels free
    // budget so visible leaves stay near-full brightness.
    //
    // Threshold transition is SHARP (0.06 wide instead of 0.16) so
    // each leaf reads as a distinct small feature with crisp edges,
    // not a soft puff. Density drives the visibility threshold:
    //   density=0.45 → tip_lo 0.74 (sparse)
    //   density=1.00 → tip_lo 0.58 (dense)
    float tip_lo = mix(0.74, 0.58, density);
    float tip_factor = smoothstep(tip_lo, tip_lo + 0.06, leaf_n);

    float effective = tip_factor * canopy_band;
    if (effective < 0.04) discard;

    // ---------- Per-leaf intensity variation ----------
    // INTENSITY CONTRAST comes from DIFFERENT LEAVES being at different
    // brightness levels — some leaves brightly sunlit, others in
    // partial shadow, etc. Driven by a slow-varying exposure noise
    // independent of the high-freq leaf mask. Range 0.35..1.00 means
    // dimmest leaves are at ~35% of brightest, big luminance variation
    // across the canopy without resorting to per-leaf white hot-spots
    // (which made each leaf look like a cloud puff).
    vec2 expose_p = vec2(uv.x * 4.5 + u_time * 0.015,
                         uv.y * 6.0);
    float expose = fbm(expose_p);
    float intensity = 0.35 + 0.65 * expose;

    // ---------- Leaf color ----------
    // Saturated base by season + day/night, scaled by per-leaf intensity.
    // No per-tip white hot-spot — that produced fluffy cloud-looking
    // blobs. Intensity variation across DIFFERENT leaves provides the
    // luminance contrast.
    float warm_factor = 1.0 - smoothstep(0.0, 0.4, abs(u_season - 0.5));
    vec3 day_lit_midday = vec3(0.28, 1.00, 0.16);   // saturated vivid green
    vec3 day_lit_dd     = vec3(0.85, 0.95, 0.20);   // saturated yellow-green
    vec3 day_col  = mix(day_lit_dd, day_lit_midday, warm_factor);
    vec3 night_col = vec3(0.25, 0.45, 0.65);        // moonlit silver-blue
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
