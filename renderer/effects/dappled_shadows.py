"""Dappled-shadows effect — moving leaf-shadow patterns from canopy.

Procedural fragment shader: layered organic spots drift across the
display, their density/opacity strongest at the canopy (uv.y=1, outer
ring of fan) and softening toward the floor (uv.y=0, inner ring). Wind
modulates drift speed.

Drives:
  dapple_strength  -> overall shadow opacity
  wind             -> drift speed of patches
  season_preference -> shadow tint (cool greens midday, warm at dawn/dusk)
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_dappled_shadows(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DappledShadowsEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized dappled_shadows for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize dappled_shadows: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.strength = float(outstate.get('dapple_strength', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))
        eff.season = float(outstate.get('season_preference', 0.5))

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
    // Atmospheric layer above godrays (priority 2.0).
    gl_Position = vec4(position, 0.85, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_strength;
uniform float u_wind;
uniform float u_season;
uniform float u_fade;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise periodic in x via 2*pi mapping.
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    if (u_strength < 0.02 || u_fade < 0.005) discard;

    vec2 uv = v_uv;

    // Smooth drift only — no peaky gust pulse (which was producing the
    // jerky motion). Wind raises overall drift speed gently.
    float drift_speed = 0.025 + u_wind * 0.10;
    float drift_x = u_time * drift_speed;
    float drift_y = u_time * 0.018;

    // Two layers of organic blobs. Both drift in the same direction so
    // motion has a clear coherent flow rather than CCW vs CW competing
    // layers.
    float n1 = vnoise(uv * vec2(7.0, 10.0)  + vec2(drift_x * 7.0,  drift_y * 10.0));
    float n2 = vnoise(uv * vec2(15.0, 22.0) + vec2(drift_x * 15.0, drift_y * 22.0) + 31.4);
    float blobs = n1 * 0.65 + n2 * 0.35;

    // Soft shadow patches. Wide smoothstep range (0.35) for SOFT edges,
    // not hard blocks.
    float thresh = mix(0.65, 0.45, clamp(u_strength, 0.0, 1.0));
    float shadow = smoothstep(thresh, thresh - 0.35, blobs);

    // Strongest near the canopy, softer toward floor.
    float radial = mix(0.3, 1.0, uv.y);

    // Final alpha — kept moderate so shadows DIM rather than block.
    float alpha = shadow * radial * u_strength * u_fade * 0.55;
    if (alpha < 0.01) discard;

    // Output BLACK rgb with positive alpha. With standard
    // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) blend this cleanly dims the
    // destination toward black — that's what a shadow IS.
    // No more colored tints (the previous "amber dawn/dusk shadow"
    // read as bright orange blocks rather than shadow).
    fragColor = vec4(0.0, 0.0, 0.0, alpha);
}
"""


class DappledShadowsEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 2.0
        self._time = 0.0
        self.strength = 0.5
        self.wind = 0.2
        self.season = 0.5
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
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
