"""Distant lights effect for the beloved (Weather of the Heart) set.

The GOAT / memory layer. A field of small points of light, like windows seen
from across a city or lanterns across a valley - the lights of remembered
moments. Reads the Goat's two love variables from outstate:

  goat_light (Cherishing) -> warm, numerous, present, steadily glowing lights;
                             the warm archive held close.
  goat_dark  (the Phantom) -> lights grow sparse, cool, and unreachable, drift
                             away faster, and flicker out; loving a memory that
                             is slipping its moorings.

Published by NarrativePlayer as story_goat_light / story_goat_dark.
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
        eff.goat_light = float(outstate.get('story_goat_light', 0.45))
        eff.goat_dark  = float(outstate.get('story_goat_dark',  0.10))

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
uniform float u_drift;       // accumulated horizontal drift (decoupled from u_time*uniform)
uniform float u_goat_light;  // cherishing: warm, numerous, steady, present
uniform float u_goat_dark;   // phantom: sparse, cool, receding, flickering out

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = v_uv;
    uv.y *= 1.6;

    vec3 accum = vec3(0.0);

    for (int layer = 0; layer < 2; ++layer) {
        float fl    = float(layer);
        float depth = (layer == 0) ? 0.45 : 1.0;
        float drift = u_drift * ((layer == 0) ? 0.4 : 1.0);

        vec2 grid = vec2(28.0, 44.0);
        vec2 p    = uv * grid + vec2(drift * grid.x, 0.0);

        vec2 cell = floor(p);
        vec2 frac = fract(p);

        // Per-cell jitter so the lights aren't grid-aligned.
        vec2 center = vec2(hash(cell + fl * 5.1), hash(cell + fl * 13.7));

        // Density: cherishing fills the field; the phantom thins it out.
        float on_thresh = 1.0 - (0.06 + u_goat_light * 0.22) + u_goat_dark * 0.28;
        float on = step(on_thresh, hash(cell + fl * 17.3));

        float d    = distance(frac, center);
        float r    = 0.10 + u_goat_light * 0.05;
        float core = exp(-(d * d) / (r * r));

        // Twinkle at a CONSTANT rate (no u_time*uniform coupling). The phantom
        // deepens the flicker until lights blink out; cherishing keeps them steady.
        float tw    = 0.5 + 0.5 * sin(u_time * 2.0 + hash(cell + fl) * 6.2831);
        float depth_flicker = mix(0.12, 0.95, u_goat_dark);
        float steady = (1.0 - depth_flicker) + depth_flicker * tw;

        // Phantom: occasional hard blackouts of whole cells.
        float blackout = step(1.0 - u_goat_dark * 0.5,
                              hash(cell + floor(u_time * 0.35) + fl * 3.3));
        steady *= 1.0 - blackout;

        float bright = core * on * steady * depth;

        // Color: warm white -> amber when cherishing; cool pale blue (unreachable)
        // when the lights become a phantom.
        vec3 white = vec3(1.0, 0.93, 0.82);
        vec3 amber = vec3(1.0, 0.84, 0.55);
        vec3 cold  = vec3(0.55, 0.66, 0.85);
        vec3 col = mix(white, amber, u_goat_light);
        col      = mix(col, cold, u_goat_dark * 0.85);

        accum += col * bright;
    }

    // The phantom pushes the whole field dimmer and farther away.
    accum *= mix(1.0, 0.55, u_goat_dark);

    float a = clamp(max(max(accum.r, accum.g), accum.b), 0.0, 1.0);
    fragColor = vec4(accum, a);
}
"""


class DistantLightsEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Beloved z-band: 0.95, matching gl_Position.z in the vertex
        # shader (behind engine stars at 0.90, in front of heart_sky 0.97).
        self.z_centroid = 0.95
        self._time = 0.0
        self._drift = 0.0          # accumulated drift phase
        self.goat_light = 0.45
        self.goat_dark = 0.10

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
        # Drift speeds up as the phantom takes over (memory slipping away).
        # Accumulated here so it never couples u_time to a ramping uniform.
        self._drift += dt * (0.01 + self.goat_dark * 0.10)

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        # Translucent layer: depth-TEST at z=0.95, depth-WRITE suppressed
        # (HARD RULE 2). Without this the quad stamps depth 0.95 across
        # every fragment - including fully transparent ones - the exact
        # pattern that punched star-holes into the bartiki map.
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"),       self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_drift"),      self._drift)
        glUniform1f(glGetUniformLocation(self.shader, "u_goat_light"), self.goat_light)
        glUniform1f(glGetUniformLocation(self.shader, "u_goat_dark"),  self.goat_dark)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthMask(GL_TRUE)
