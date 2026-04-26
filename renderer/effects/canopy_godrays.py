"""Canopy sky + glow backdrop — sky color and warm light glow at the
outer-ring canopy band, behind the leaf layer.

Renders FIRST among canopy layers at deep depth (0.95) so clouds can
drift in front of it through the leaf gaps. forest_canopy then renders
leaves on top (depth 0.15) and only WHERE leaves exist, leaving the gap
cells with their sky/glow visible.

Drives:
  godray_strength    -> overall glow intensity (warm shaft tint)
  season_preference  -> sky color (dawn/dusk amber, midday blue)
  starryness         -> mixes toward dark night sky
  wind_speed         -> subtle drift in the glow texture
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_canopy_godrays(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CanopyGodraysEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized canopy_godrays for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize canopy_godrays: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.strength = float(outstate.get('godray_strength', 0.0))
        eff.season = float(outstate.get('season_preference', 0.5))
        eff.wind = float(outstate.get('wind', 0.0))
        eff.starryness = float(outstate.get('starryness', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 4.0
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
    // Depth 0.95: deep so this is the BACKDROP layer for the canopy
    // band. forest_canopy leaves at depth 0.15 render in front of it,
    // and clouds at depth 0.15-0.30 can drift in front of the sky in
    // gap cells (where forest_canopy discarded).
    gl_Position = vec4(position, 0.95, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_strength;
uniform float u_season;
uniform float u_wind;
uniform float u_starryness;
uniform float u_fade;
// Integrated phase: dt * (0.04 + u_wind * 0.12) accumulated CPU-side.
// Replaces `u_time * (0.04 + u_wind * 0.12)` so the warm-glow noise drift
// stays monotonic when u_wind interpolates during state transitions.
uniform float u_glow_drift_phase;

float hash1(float n) { return fract(sin(n) * 43758.5453); }
float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise — interpolated random per cell, not sinusoidal.
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

void main() {
    if (u_fade < 0.005) discard;

    vec2 uv = v_uv;
    float y = uv.y;

    // Restrict to the canopy band (matches forest_canopy's band).
    // Below uv.y=0.20 we discard so this doesn't write depth on the
    // forest floor — clouds and other effects render normally there.
    float band = smoothstep(0.20, 0.55, y);
    if (band < 0.01) discard;

    // ---------- SKY backdrop ----------
    // Time-of-day color. This layer provides the SKY visible through
    // canopy gaps (since forest_canopy now discards in gap pixels and
    // doesn't draw sky itself).
    float warm_factor = 1.0 - smoothstep(0.0, 0.4, abs(u_season - 0.5));
    vec3 day_sky  = vec3(0.45, 0.72, 0.95);   // midday blue
    vec3 dd_sky   = vec3(1.00, 0.55, 0.30);   // dawn/dusk amber
    vec3 daytime_sky = mix(dd_sky, day_sky, warm_factor);
    vec3 night_sky   = vec3(0.04, 0.06, 0.12);
    vec3 sky_col = mix(daytime_sky, night_sky, u_starryness);

    // ---------- Warm glow on top of sky ----------
    // Radial falloff brightest near the outer ring (y=1).
    float radial = pow(y, 2.0);
    // 2D noise modulates the glow so it has organic patchiness and
    // drifts with wind.
    vec2 np = vec2(uv.x * 4.0 + u_glow_drift_phase,
                   uv.y * 6.0 + u_time * 0.02);
    float texture_n = vnoise(np) * 0.6 + vnoise(np * 2.5 + 7.7) * 0.4;
    float glow_modulation = 0.55 + 0.45 * texture_n;

    // Glow color is brighter / warmer than the sky.
    vec3 dawn_glow = vec3(1.00, 0.60, 0.28);
    vec3 day_glow  = vec3(1.00, 0.95, 0.78);
    vec3 glow_col = mix(dawn_glow, day_glow, warm_factor);
    // Fades to nothing at night (no sun).
    float glow_amount = radial * glow_modulation * u_strength * (1.0 - u_starryness);

    // Combine: base sky + warm glow added on top.
    vec3 col = sky_col + glow_col * glow_amount * 0.5;

    // Sky opacity — solid in daytime, fades to nearly transparent at
    // night so stars / aurora rendered behind (deeper depth) can dominate
    // the canopy gaps. Clouds always render in front via depth 0.15-0.30.
    float sky_alpha = mix(0.85, 0.10, u_starryness);
    float total_alpha = sky_alpha * band * u_fade;
    if (total_alpha < 0.01) discard;
    fragColor = vec4(col, total_alpha);
}
"""


class CanopyGodraysEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Sky backdrop renders FIRST (back of atmospheric stack). With
        # depth writes disabled, render order is purely priority-driven.
        self.render_priority = 1.0
        self._time = 0.0
        self._glow_drift_phase = 0.0  # integrated dt * (0.04 + wind * 0.12)
        self.strength = 0.6
        self.season = 0.5
        self.wind = 0.2
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
        self._glow_drift_phase += dt * (0.04 + self.wind * 0.12)

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        # Post-process layering: always pass depth, never write depth.
        # Layer order is determined purely by render_priority.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"), self.strength)
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_starryness"), self.starryness)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glUniform1f(glGetUniformLocation(self.shader, "u_glow_drift_phase"), self._glow_drift_phase)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        # Restore default depth state for subsequent effects.
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
