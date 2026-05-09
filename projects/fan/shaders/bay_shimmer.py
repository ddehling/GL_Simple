"""Bay water shimmer and tidal pulse effect for the bartiki weather set.

Adds animated sparkle to water pixels (bay and ocean) on the geographic
map.  Shimmer color shifts with time-of-day: white sun glints by day,
blue-silver moonlight at night, golden at dawn/dusk.  A slow tidal
breathing cycle modulates overall water brightness.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords


# ---------------------------------------------------------------------------
# Event wrapper
# ---------------------------------------------------------------------------

def shader_bay_shimmer(state, outstate):
    """Bay water shimmer background effect."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(BayShimmerEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized bay_shimmer for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize bay_shimmer: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        state['effect'].time_of_day = outstate.get('season', 0.5)

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

_VERT = """
#version 310 es
precision highp float;
in vec2 position;
out vec2 v_texcoord;
void main() {
    v_texcoord = position * 0.5 + 0.5;
    float depth = 97.5 / 100.0;
    gl_Position = vec4(position, depth, 1.0);
}
"""

_FRAG = f"""#version 310 es
precision highp float;

in vec2 v_texcoord;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

uniform float u_time;
uniform float u_time_of_day;
uniform sampler2D u_geo_tex;

// Hash for sparkle
float hash(vec2 p) {{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}}

void main() {{
    // Sample geo texture to detect water pixels
    vec3 geo = texture(u_geo_tex, v_texcoord).rgb;

    // Water detection: blue channel dominant (ocean and bay colors)
    // Ocean: (12,22,48)/255, Bay: (20,38,72)/255
    float blue_ratio = geo.b / max(geo.r + geo.g + geo.b, 0.001);
    bool is_water = blue_ratio > 0.42 && geo.b > 0.05;

    if (!is_water) discard;

    vec2 phys = fan_uv_to_physical(v_texcoord);

    // --- Shimmer: layered sine waves ---
    float shimmer = 0.0;
    shimmer += sin(phys.x * 3.0 + u_time * 2.0) * 0.3;
    shimmer += sin(phys.y * 4.0 - u_time * 1.5) * 0.25;
    shimmer += sin((phys.x + phys.y) * 5.0 + u_time * 3.0) * 0.15;

    // Sparkle: sharp random glints
    vec2 cell = floor(phys * 2.0 + u_time * 0.5);
    float sparkle = step(0.97, hash(cell));
    shimmer += sparkle * 1.5;

    shimmer = max(shimmer, 0.0);

    // --- Tidal pulse: slow breathing (~2 min cycle) ---
    float tidal = 0.85 + 0.15 * sin(u_time * 0.052);  // ~2 min period

    // --- Time-of-day color ---
    float dayNight = -cos(u_time_of_day * 6.28318);

    // Day: white sun glints
    vec3 dayColor = vec3(0.9, 0.95, 1.0);
    // Night: blue-silver moonlight
    vec3 nightColor = vec3(0.3, 0.5, 0.8);
    // Dawn/dusk: golden
    float dawnDist = min(abs(u_time_of_day - 0.20), abs(u_time_of_day - 1.20));
    float duskDist = abs(u_time_of_day - 0.80);
    float golden = exp(-dawnDist * dawnDist * 80.0) + exp(-duskDist * duskDist * 80.0);
    vec3 goldenColor = vec3(1.0, 0.7, 0.3);

    float dayFactor = max(dayNight, 0.0);
    vec3 shimmerColor = mix(nightColor, dayColor, dayFactor);
    shimmerColor = mix(shimmerColor, goldenColor, golden * 0.6);

    // Intensity: brighter during day, dimmer at night
    float intensity = 0.15 + 0.35 * dayFactor;

    float alpha = shimmer * intensity * tidal;
    if (alpha < 0.005) discard;

    fragColor = vec4(shimmerColor * alpha, alpha);
}}
"""


# ---------------------------------------------------------------------------
# Effect class
# ---------------------------------------------------------------------------

class BayShimmerEffect(ShaderEffect):
    """Fullscreen water shimmer overlay."""

    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 5.1  # just after BART background
        self.time_of_day = 0.5
        self._time = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)
        self._geo_texture = None

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(_FRAG, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
                        dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        self.VBOs.append(vbo)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # Set static fan coord uniforms
        glUseProgram(self.shader)
        self._fan.set_uniforms(self.shader)
        glUseProgram(0)

    def _find_geo_texture(self):
        """Find the BART map's geo texture from the viewport's effects."""
        if self._geo_texture is not None:
            return
        for effect in self.viewport.effects:
            if effect is self:
                continue
            if hasattr(effect, '_geo_texture') and effect._geo_texture:
                self._geo_texture = effect._geo_texture
                return

    def update(self, dt: float, state: Dict):
        self._time += dt
        if self._geo_texture is None:
            self._find_geo_texture()

    def render(self, state: Dict):
        super().render(state)
        if not self.shader or self._geo_texture is None:
            return

        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_time_of_day"), self.time_of_day)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._geo_texture)
        glUniform1i(glGetUniformLocation(self.shader, "u_geo_tex"), 0)

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    def cleanup(self):
        # Don't delete the geo texture — it belongs to the BART map effect
        self._geo_texture = None
        super().cleanup()
