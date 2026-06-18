"""Warm bloom effect for the beloved (Weather of the Heart) set.

The CAPYBARA / presence layer. Soft, slow radial blooms that swell and fade -
the ambient warmth-of-being-with. Reads the Capybara's two love variables:

  capybara_light (Abiding)      -> warm, full, frequent blooms that drift gently;
                                   the easy companionship that needs no occasion.
  capybara_dark  (the Still Water) -> blooms go cool, flat, sparse, and motionless;
                                   comfort curdled into numbness, presence that has
                                   become absence.

Published by NarrativePlayer as story_capybara_light / story_capybara_dark.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_warm_bloom(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(WarmBloomEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized warm_bloom for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize warm_bloom: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.capybara_light = float(outstate.get('story_capybara_light', 0.45))
        eff.capybara_dark  = float(outstate.get('story_capybara_dark',  0.10))

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
    gl_Position = vec4(position, 0.7, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_cycle;           // accumulated bloom cycle (decoupled from period)
uniform float u_capybara_light;  // abiding: warm, full, present blooms
uniform float u_capybara_dark;   // still water: cool, flat, sparse, motionless

float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    vec2 uv = v_uv;
    uv.y *= 1.6;

    // Abiding keeps the blooms present for more of each cycle; the still water
    // shrinks them and damps their motion almost to nothing.
    float live_frac = mix(0.45, 0.85, u_capybara_light);
    float radius0   = mix(0.10, 0.30, u_capybara_light) * mix(1.0, 0.6, u_capybara_dark);
    float drift_amp = mix(0.03, 0.005, u_capybara_dark);

    float total = 0.0;

    for (int i = 0; i < 6; ++i) {
        float fi = float(i);
        // Each slot reads the shared accumulated cycle, staggered. Using an
        // accumulated cycle (not u_time / period) avoids phase jumps when the
        // period changes as the variables ramp.
        float s    = u_cycle - fi * 0.37;
        float idx  = floor(s);
        float ph01 = fract(s);                 // 0..1 within this slot's cycle
        if (ph01 > live_frac) continue;

        float age = ph01 / live_frac;          // 0..1 over the bloom's life
        float env = sin(age * 3.1416);         // 0 -> 1 -> 0
        env *= mix(1.0, 0.5, u_capybara_dark); // still water flattens the swell

        vec2 c = vec2(hash(idx * 7.13 + fi * 1.7),
                      hash(idx * 11.7 + fi * 2.3) * 1.6);
        c.x += sin(u_time * 0.05 + fi) * drift_amp;
        c.y += cos(u_time * 0.04 + fi) * drift_amp;

        float radius = radius0 * (0.6 + 0.8 * env);
        float d      = distance(uv, c);
        float fall   = exp(-(d * d) / (radius * radius * 0.5));

        total += fall * env;
    }

    // Color: cool gray-blue (still water) -> warm amber-cream (abiding).
    vec3 cool  = vec3(0.42, 0.50, 0.62);
    vec3 warm  = vec3(1.0, 0.80, 0.58);
    vec3 muted = vec3(0.40, 0.43, 0.50);
    vec3 tint = mix(cool, warm, u_capybara_light);
    tint      = mix(tint, muted, u_capybara_dark * 0.6);

    // Straight alpha (NOT pre-multiplied): rgb is the full tint, alpha carries
    // the bloom intensity.
    float alpha = clamp(total, 0.0, 1.0);
    alpha *= (0.45 + u_capybara_light * 0.5);
    alpha *= mix(1.0, 0.55, u_capybara_dark);
    fragColor = vec4(tint, clamp(alpha, 0.0, 0.9));
}
"""


class WarmBloomEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Beloved z-band: 0.70, matching gl_Position.z in the vertex
        # shader so draw-order sort and depth test agree.
        self.z_centroid = 0.70
        self._time = 0.0
        self._cycle = 0.0          # accumulated bloom cycle
        self.capybara_light = 0.45
        self.capybara_dark = 0.10

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
        # Abiding blooms come a little more often; the still water spaces them
        # far apart. Accumulated as a cycle so changing period never jumps phase.
        period = mix_py(7.0, 4.5, self.capybara_light) * mix_py(1.0, 1.8, self.capybara_dark)
        self._cycle += dt / max(period, 0.5)

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        # Translucent layer: depth-TEST at z=0.70, depth-WRITE suppressed
        # (HARD RULE 2) so transparent fragments never stamp the depth
        # buffer over layers composed afterwards.
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"),           self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_cycle"),          self._cycle)
        glUniform1f(glGetUniformLocation(self.shader, "u_capybara_light"), self.capybara_light)
        glUniform1f(glGetUniformLocation(self.shader, "u_capybara_dark"),  self.capybara_dark)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthMask(GL_TRUE)


def mix_py(a: float, b: float, t: float) -> float:
    """Linear interpolation matching GLSL mix(), for use in Python update()."""
    return a + (b - a) * t
