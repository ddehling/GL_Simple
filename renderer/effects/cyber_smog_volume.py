"""
Cyber smog volume — colored volumetric fog that replaces the generic
gray/white fog for cyberpunk states. Drifts horizontally, has internal
turbulence from layered sines, and tints to a state-driven color
(`outstate['cyber_smog_color']`). Density driven by
`outstate['cyber_smog_density']` (0..1).

Per the contrast playbook: a single tinted-smog wash spends most of the
energy budget — so we keep alpha modest (max ~0.55) and let the smog
PASS THROUGH dark areas so the skyline silhouettes show.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_smog_volume(state, outstate, density=0.5,
                              color=(0.55, 0.35, 0.20)):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberSmogVolumeEffect,
                                          density=density, color=color)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_smog_volume] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # Use pollution_level (existing param) for smog density
    eff.density = float(outstate.get('pollution_level', density))
    # Use fog_color (existing param) for smog tint
    c = outstate.get('fog_color', color)
    if c is not None and len(c) >= 3:
        eff.color = (float(c[0]), float(c[1]), float(c[2]))
    # Season-of-day for diurnal hue shift (0.0=midnight → 0.5=noon → 1.0=midnight)
    eff.season = float(outstate.get('season', 0.5))

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
uniform float u_drift_phase;   // CPU-integrated dt * drift_rate
uniform float u_swirl_phase;
uniform float u_density;
uniform vec3 u_color;
uniform float u_season;         // [0,1) time-of-day cycle
out vec4 fragColor;

// Season-driven smog tint — gives a continuous diurnal hue shift
// even when the weather state doesn't change.
//   0.00 = midnight     deep blue-violet
//   0.25 = dawn         warm rose / amber
//   0.50 = noon         cool blue-gray (washed)
//   0.75 = dusk         deep orange-red (sodium street-light)
vec3 season_tint(float s) {
    vec3 night = vec3(0.10, 0.05, 0.25);
    vec3 dawn  = vec3(0.85, 0.50, 0.40);
    vec3 noon  = vec3(0.45, 0.55, 0.70);
    vec3 dusk  = vec3(0.95, 0.40, 0.15);
    s = fract(s) * 4.0;
    if (s < 1.0) return mix(night, dawn,  s);
    if (s < 2.0) return mix(dawn,  noon,  s - 1.0);
    if (s < 3.0) return mix(noon,  dusk,  s - 2.0);
    return mix(dusk, night, s - 3.0);
}

// Cheap 2D value noise
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

void main() {
    vec2 uv = v_uv;
    // Smog drifts horizontally + has slow vertical lift; horizontal wraps
    vec2 q1 = vec2(uv.x * 4.0 + u_drift_phase * 0.08, uv.y * 3.5 - u_drift_phase * 0.02);
    vec2 q2 = vec2(uv.x * 8.0 - u_swirl_phase * 0.05, uv.y * 7.0 + u_swirl_phase * 0.03);

    float n1 = vnoise(q1);
    float n2 = vnoise(q2);
    float smog = n1 * 0.7 + n2 * 0.3;

    // Density curve: low density = mostly clear, high = thick band
    smog = smoothstep(1.0 - u_density, 1.0, smog + u_density * 0.3);

    // Smog is heavier at the bottom (settles in the Midden)
    float vertical = mix(0.6, 1.0, 1.0 - uv.y);
    smog *= vertical;

    // Alpha capped at 0.55 so silhouettes still read through
    float alpha = smog * 0.55;
    if (alpha < 0.01) discard;

    // Blend the state's fog_color with the season-of-day tint. State
    // identity is DOMINANT (85%); season only nudges 15% so it gives a
    // perceptible diurnal drift without painting brown over everything.
    vec3 col = mix(u_color, season_tint(u_season), 0.15);
    fragColor = vec4(col, alpha);
}
"""


class CyberSmogVolumeEffect(ShaderEffect):
    def __init__(self, viewport, density: float = 0.5,
                 color=(0.55, 0.35, 0.20)):
        super().__init__(viewport)
        self.render_priority = 1.0     # Atmospheric base
        self.density = density
        self.color = color
        self.season = 0.5              # set by wrapper from outstate['season']
        self._time = 0.0
        # CPU-integrated phases (drift/swirl rates are constants — safe to
        # multiply by u_time, but using phases keeps the convention uniform)
        self.drift_phase = 0.0
        self.swirl_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberSmogVolume compile error: {e}")
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
        self.drift_phase += dt          # constant rate; phase form for consistency
        self.swirl_phase += dt * 1.3

    def render(self, state: Dict):
        if not self.enabled or self.density < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_drift_phase"), self.drift_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_swirl_phase"), self.swirl_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform3f(glGetUniformLocation(self.shader, "u_color"),
                    self.color[0], self.color[1], self.color[2])
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
