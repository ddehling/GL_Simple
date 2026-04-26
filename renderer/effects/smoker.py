"""Hydrothermal-vent black-smoker shader (fan-aware).

Pattern B atmospheric layer. Six dark mineral plumes rise from chimneys
sitting on the inner ring of the fan (the sea floor). Each chimney is
sized in physical feet via FAN_COORDS_GLSL, so columns look like real
vertical pipes regardless of where they sit on the semicircle, instead
of pie-slices that get squashed at the rim. Three contribution layers
stack: dark sooty plume body, a hot orange glow at each chimney mouth,
and bright ember sparks scrolling upward through the smoke. Driven by
the `vent_activity` weather param so the layer stays invisible outside
of OCEAN_HYDROTHERMAL_VENT.

Drives:
  vent_activity -> overall plume intensity / opacity gate
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords


def shader_smoker(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(SmokerEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized smoker for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize smoker: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.vent_activity = float(outstate.get('vent_activity', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 5.0
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
    // Mid-back atmospheric layer; sits in front of base water tint but
    // behind foreground particle effects (bubbles, bioluminescence).
    gl_Position = vec4(position, 0.82, 1.0);
}
"""

_FRAG = f"""
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_vent_activity;
uniform float u_fade;
{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

float hash(vec2 p) {{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}}

float vnoise(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}}

void main() {{
    // uv.y = 0 is sea floor (inner ring of fan, r = 4 ft).
    // uv.y = 1 is sea surface (outer ring of fan, r = 20.6 ft).
    vec2 uv = v_uv;

    // Hard gate: invisible when vent isn't active or event is mid-fade
    // near zero. Avoids any base haze leaking into non-vent states.
    if (u_vent_activity < 0.005 || u_fade < 0.005) discard;
    if (uv.y > 0.92) discard;

    // Physical position in feet. We work entirely in physical units from
    // here on so columns look like actual vertical pipes regardless of
    // their angular position on the semicircle.
    vec2 phys = fan_uv_to_physical(uv);
    float r = fan_radius_ft(uv);
    float rise_ft = r - u_inner_r_ft;  // height above the seafloor

    // Heat shimmer: wobble the effective angular position. Strongest in
    // the bottom 5 ft, fading out as smoke "cools" on the rise.
    float shimmer_amp = 0.014 * (1.0 - smoothstep(0.0, 5.0, rise_ft));
    float shimmer = sin(rise_ft * 4.0 + u_time * 1.6) * shimmer_amp;
    float shimmered_u = uv.x + shimmer;

    // Six chimneys spread irregularly across the semicircle so the row
    // doesn't read as a grid. (angular_uv, base_width_ft).
    float chimney_u[6];
    float chimney_w[6];
    chimney_u[0] = 0.08; chimney_w[0] = 0.55;
    chimney_u[1] = 0.24; chimney_w[1] = 0.75;
    chimney_u[2] = 0.41; chimney_w[2] = 0.55;
    chimney_u[3] = 0.59; chimney_w[3] = 0.70;
    chimney_u[4] = 0.76; chimney_w[4] = 0.55;
    chimney_u[5] = 0.92; chimney_w[5] = 0.65;

    // Two-octave noise sampled in physical (x_ft, y_ft) space scrolling
    // upward over time. Gives turbulent billow that stays uniform across
    // the fan instead of looking pinched at the rim.
    float n1 = vnoise(vec2(phys.x * 0.45, (phys.y - u_time * 0.7) * 0.55));
    float n2 = vnoise(vec2(phys.x * 1.05, (phys.y - u_time * 1.1) * 1.20));
    float billow = mix(n1, n2, 0.4);
    billow = smoothstep(0.16, 0.78, billow);

    // Vertical envelope in feet: starts ~half a foot above the floor (so
    // the chimney mouth glow isn't smothered), rises through the water
    // column, fades out by ~14 ft above the seafloor. Outer ring is at
    // 16.6 ft above the floor, so plumes fade well before the surface.
    float env = smoothstep(0.4, 1.6, rise_ft)
              * (1.0 - smoothstep(8.0, 14.0, rise_ft));

    float smoke_density = 0.0;
    float glow_density  = 0.0;
    float ember_density = 0.0;

    for (int i = 0; i < 6; i++) {{
        float cu = chimney_u[i];
        float base_w = chimney_w[i];

        // Lateral arc length in feet between the fragment and the
        // chimney axis at the current radius. uv.x maps over PI radians,
        // so a delta in uv corresponds to that delta times PI in radians,
        // and arc length s = r * theta.
        float du = shimmered_u - cu;
        float lateral_ft = abs(du) * r * FAN_PI;

        // Column widens with rise — smoke disperses as it goes up.
        float col_w = base_w + rise_ft * 0.40;
        float dx = lateral_ft / col_w;
        float col = exp(-dx * dx);

        // Per-chimney slow surge so the row doesn't pulse in unison.
        // Bottoms out at 0.6 so no chimney ever fully vanishes.
        float surge = 0.60 + 0.40 *
                      (sin(u_time * 0.18 + cu * 17.3) * 0.5 + 0.5);

        smoke_density += col * billow * env * surge;

        // Hot orange glow at the chimney mouth — narrow column, only in
        // the bottom ~2 ft, with fast flicker.
        float glow_w = base_w * 0.55;
        float glow_dx = lateral_ft / glow_w;
        float glow_col = exp(-glow_dx * glow_dx);
        float glow_env = smoothstep(2.0, 0.0, rise_ft);
        float flicker = 0.7 + 0.3 *
                        vnoise(vec2(cu * 31.0, u_time * 2.6 + cu * 7.0));
        glow_density += glow_col * glow_env * flicker * surge;

        // Ember sparks: bright pinpoints rising through the plume. Sample
        // a high-frequency noise in physical coords scrolling upward fast
        // (faster than the smoke billow), threshold for sparse spikes,
        // gate by column proximity so embers only appear inside plumes.
        float ember_col = exp(-(dx * dx) * 1.8);
        float spark_n = vnoise(vec2(phys.x * 2.6 + cu * 13.7,
                                    (phys.y - u_time * 1.6) * 4.5));
        float spark = smoothstep(0.84, 0.94, spark_n);
        ember_density += ember_col * spark * env * surge;
    }}

    // Boost smoke density a touch so plumes read as substantial rather
    // than gauzy. Glow gets an extra boost so the chimney mouths punch
    // through even when other plumes overlap.
    smoke_density = clamp(smoke_density * u_vent_activity * 1.35, 0.0, 1.0);
    glow_density  = clamp(glow_density  * u_vent_activity * 1.50, 0.0, 1.2);
    ember_density = clamp(ember_density * u_vent_activity * 1.10, 0.0, 1.0);

    // Sooty almost-black for the cold upper plume, mineral-warm tint as
    // smoke nears the chimney mouth.
    vec3 smoke_dark = vec3(0.020, 0.012, 0.010);
    vec3 smoke_warm = vec3(0.28,  0.10,  0.04);
    vec3 smoke_color = mix(smoke_dark, smoke_warm,
                           smoothstep(8.0, 0.5, rise_ft));
    vec3 glow_color  = vec3(1.00, 0.30, 0.05);
    vec3 ember_color = vec3(1.00, 0.60, 0.15);

    // Mix the three contributions by relative density. Output is straight-
    // alpha (color is NOT premultiplied by alpha).
    float total = smoke_density + glow_density + ember_density + 0.001;
    vec3 color = (smoke_color  * smoke_density
                + glow_color   * glow_density
                + ember_color  * ember_density) / total;

    // Smoke contributes most of the body opacity; embers are bright but
    // not solid, glow is luminous so its alpha contribution is moderated.
    float alpha = clamp(smoke_density * 0.95
                      + glow_density  * 0.65
                      + ember_density * 0.95, 0.0, 1.0);
    alpha *= u_fade;

    if (alpha < 0.005) discard;

    fragColor = vec4(color, alpha);
}}
"""


class SmokerEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Mid-back: in front of base water tint (~1-2), behind foreground
        # particles like bubbles and bioluminescence.
        self.render_priority = 3.0
        self._time = 0.0
        self.vent_activity = 0.0
        self.fade = 0.0
        self.fan = FanCoords(viewport.width, viewport.height)

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
        # Pattern B: post-process-style fullscreen quad. Don't write depth
        # and pass the depth test unconditionally so the layer composes via
        # render_priority + alpha blending only.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_vent_activity"), self.vent_activity)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        # Fan-coords uniforms (inner_r_ft, outer_r_ft, num_cols, num_rows)
        # so the shader can convert UV -> physical feet.
        self.fan.set_uniforms(self.shader)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
