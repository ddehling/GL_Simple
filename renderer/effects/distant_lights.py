"""Distant lights effect for the beloved (love) weather set.

A field of small, slow-drifting points of light — like windows seen from
across a city, lanterns across a valley, the lights of someone you can
no longer reach. Reads the four love variables from outstate.

  passion    -> warm hue (amber/red), slight twinkle rate
  tenderness -> base brightness floor, gentler twinkle
  longing    -> drift speed, parallax separation, blue shift
  devotion   -> count of visible lights, persistence
  sadness    -> overall dim, color desaturated toward gray-blue
  heartbreak -> lights go out (sparser field), occasional sharp blackouts
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_distant_lights(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DistantLightsEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized distant_lights for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize distant_lights: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.passion    = float(outstate.get('story_passion',    0.2))
        eff.tenderness = float(outstate.get('story_tenderness', 0.3))
        eff.longing    = float(outstate.get('story_longing',    0.4))
        eff.devotion   = float(outstate.get('story_devotion',   0.4))
        eff.sadness    = float(outstate.get('story_sadness',    0.0))
        eff.heartbreak = float(outstate.get('story_heartbreak', 0.0))

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
    gl_Position = vec4(position, 0.95, 1.0);   // far back
}
"""

_FRAG = """
#version 310 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_passion;
uniform float u_tenderness;
uniform float u_longing;
uniform float u_devotion;
uniform float u_sadness;
uniform float u_heartbreak;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    // Two parallax layers: far drifts slower than near. Longing widens the gap.
    float drift_far  = u_time * (0.005 + u_longing * 0.04);
    float drift_near = u_time * (0.015 + u_longing * 0.10);

    vec2 uv = v_uv;
    uv.y *= 1.6;

    vec3 accum = vec3(0.0);

    for (int layer = 0; layer < 2; ++layer) {
        float fl    = float(layer);
        float drift = (layer == 0) ? drift_far : drift_near;
        float depth = (layer == 0) ? 0.4 : 1.0;

        vec2 grid = vec2(28.0, 44.0);
        vec2 p    = uv * grid + vec2(drift * grid.x, 0.0);

        vec2 cell = floor(p);
        vec2 frac = fract(p);

        // Per-cell jitter
        vec2 jitter = vec2(hash(cell + fl * 5.1), hash(cell + fl * 13.7));
        vec2 center = jitter;

        // Cell on/off based on devotion. Heartbreak knocks out additional lights.
        float on_thresh = 1.0 - (0.05 + u_devotion * 0.18);
        on_thresh += u_heartbreak * 0.15;
        float on = step(on_thresh, hash(cell + fl * 17.3));

        float d = distance(frac, center);
        float r = 0.10 + u_tenderness * 0.05;
        float core = exp(-(d * d) / (r * r));

        // Twinkle: rate slightly tied to passion; tenderness damps it.
        float tw = 0.5 + 0.5 * sin(u_time * (2.0 + u_passion * 5.0) + hash(cell + fl) * 6.28);
        tw = mix(tw, 0.7, u_tenderness);

        float bright = core * on * (0.3 + 0.7 * tw) * depth;

        // Color: passion -> amber, longing -> blue, otherwise warm white.
        vec3 amber = vec3(1.0, 0.80, 0.45);
        vec3 white = vec3(1.0, 0.92, 0.80);
        vec3 blue  = vec3(0.55, 0.70, 1.0);
        vec3 gray  = vec3(0.45, 0.50, 0.60);
        vec3 col = mix(white, amber, u_passion);
        col      = mix(col, blue, u_longing);
        col      = mix(col, gray, u_sadness * 0.6);

        accum += col * bright;
    }

    // Sadness dims the whole field.
    accum *= mix(1.0, 0.45, u_sadness);

    float a = clamp(max(max(accum.r, accum.g), accum.b), 0.0, 1.0);
    fragColor = vec4(accum, a);
}
"""


class DistantLightsEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 1.0
        self._time = 0.0
        self.passion = 0.2
        self.tenderness = 0.3
        self.longing = 0.4
        self.devotion = 0.4
        self.sadness = 0.0
        self.heartbreak = 0.0

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

    def update(self, dt: float, state: Dict):
        self._time += dt

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"),       self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_passion"),    self.passion)
        glUniform1f(glGetUniformLocation(self.shader, "u_tenderness"), self.tenderness)
        glUniform1f(glGetUniformLocation(self.shader, "u_longing"),    self.longing)
        glUniform1f(glGetUniformLocation(self.shader, "u_devotion"),   self.devotion)
        glUniform1f(glGetUniformLocation(self.shader, "u_sadness"),    self.sadness)
        glUniform1f(glGetUniformLocation(self.shader, "u_heartbreak"), self.heartbreak)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
