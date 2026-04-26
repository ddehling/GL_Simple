"""Canopy glow effect — warm radial light filtering down from the canopy.

Replaces the older "vertical streaks" approach which rendered as ugly
radial spokes on the fan. Now renders as a soft brightness gradient
brightest at the outer-ring canopy band and fading inward, with low-
amplitude organic noise for texture and rare bright dust-mote sparkles.

Drives:
  godray_strength    -> overall glow intensity
  season_preference  -> time-of-day color (dawn/dusk = warm, midday = white)
  wind_speed         -> subtle drift in the noise texture
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
    // Depth 0.10: in front of canopy (depth 0.15) so godrays render
    // AFTER canopy and ALPHA-BLEND on top — adds warm shaft tint
    // without depth-erasing the leaves underneath. Canopy renders
    // first (priority 1.5), godrays second (priority 1.7).
    gl_Position = vec4(position, 0.10, 1.0);
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
uniform float u_fade;

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
    if (u_strength < 0.02 || u_fade < 0.005) discard;

    vec2 uv = v_uv;
    float y = uv.y;

    // RADIAL GRADIENT, no vertical streaks. Brightest at the outer-ring
    // canopy band (y near 1), fades quickly toward the inner ring (y=0).
    // Quadratic falloff gives a soft glow rather than a hard band.
    float radial = pow(y, 2.0);
    if (radial < 0.02) discard;

    // Subtle 2D noise modulates the glow so it's not a perfectly smooth
    // gradient — gives texture / "patchy light through canopy" feeling.
    // Wind slowly drifts the pattern.
    vec2 np = vec2(uv.x * 4.0 + u_time * (0.04 + u_wind * 0.12),
                   uv.y * 6.0 + u_time * 0.02);
    float texture_n = vnoise(np) * 0.6 + vnoise(np * 2.5 + 7.7) * 0.4;
    // Map noise to a 0.55-1.0 multiplier so even the dim spots still
    // contribute glow (no harsh dark gaps).
    float glow_modulation = 0.55 + 0.45 * texture_n;

    // Rare bright dust-mote sparkles, only in the upper part where the
    // glow is bright enough to see them.
    float mote_cell = floor(uv.x * 50.0) + floor(uv.y * 70.0) * 137.0;
    float mote = step(0.992, hash1(mote_cell + floor(u_time * 3.5)));
    float mote_brightness = mote * smoothstep(0.4, 1.0, radial);

    // Time-of-day color.
    float warm_factor = 1.0 - smoothstep(0.0, 0.4, abs(u_season - 0.5));
    vec3 dawn_dusk = vec3(1.00, 0.60, 0.28);
    vec3 midday    = vec3(1.00, 0.95, 0.78);
    vec3 glow_col = mix(dawn_dusk, midday, warm_factor);

    float intensity = radial * glow_modulation * u_strength;
    intensity += mote_brightness * 0.7 * u_strength;
    intensity = clamp(intensity, 0.0, 1.0);

    // Alpha capped low so this is a SOFT atmospheric layer, not a wash.
    float alpha = intensity * 0.55 * u_fade;
    if (alpha < 0.005) discard;
    fragColor = vec4(glow_col * intensity, alpha);
}
"""


class CanopyGodraysEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 1.7  # After canopy (1.5), so we alpha-blend on top.
        self._time = 0.0
        self.strength = 0.6
        self.season = 0.5
        self.wind = 0.2
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
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"), self.strength)
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
