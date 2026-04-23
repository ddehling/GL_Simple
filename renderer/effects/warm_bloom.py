"""Warm bloom effect for the beloved (love) weather set.

Soft, slow radial blooms that swell and fade across the display — the
ambient warmth-of-presence layer. Each bloom drifts gently and pulses
once over a few seconds.

  passion    -> bloom rate (more frequent), color shift toward red-orange
  tenderness -> bloom size, softness of falloff, brightness floor
  longing    -> color shift toward cool blue, slower swell
  devotion   -> bloom lifetime persistence (slower fade)
  sadness    -> muted gray-blue tint, dimmer overall
  heartbreak -> blooms cut short (sharp early decay)
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
        eff.passion    = float(outstate.get('story_passion',    0.2))
        eff.tenderness = float(outstate.get('story_tenderness', 0.5))
        eff.longing    = float(outstate.get('story_longing',    0.0))
        eff.devotion   = float(outstate.get('story_devotion',   0.3))
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
    gl_Position = vec4(position, 0.7, 1.0);
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

float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    vec2 uv = v_uv;
    uv.y *= 1.6;

    // Period between blooms shortens with passion; lifetime extends with devotion.
    float period   = mix(8.0, 2.0, u_passion);
    float lifetime = mix(2.5, 7.0, u_devotion);

    vec3 col = vec3(0.0);
    float total = 0.0;

    for (int i = 0; i < 6; ++i) {
        float fi = float(i);
        // Each bloom slot has its own drifting position.
        float t   = u_time + fi * 31.4;
        float idx = floor(t / period);
        float ph  = fract(t / period) * period;       // 0..period

        // New bloom every period: pick a fresh random center.
        vec2 c = vec2(
            hash(idx * 7.13 + fi * 1.7),
            hash(idx * 11.7 + fi * 2.3) * 1.6
        );

        // Slow drift
        c.x += sin(u_time * 0.05 + fi) * 0.02;
        c.y += cos(u_time * 0.04 + fi) * 0.02;

        // Bloom envelope: rises and falls smoothly.
        // Heartbreak truncates the lifetime — blooms get cut short.
        float effective_life = lifetime * mix(1.0, 0.35, u_heartbreak);
        float age = ph / effective_life;
        if (age > 1.0) continue;
        float env = sin(age * 3.1416);                 // 0 -> 1 -> 0

        // Slower swell when longing or sadness is high.
        env = pow(env, mix(1.0, 1.8, max(u_longing, u_sadness)));

        float radius = mix(0.10, 0.32, u_tenderness) * (0.6 + 0.8 * env);
        float d = distance(uv, c);
        float falloff = exp(-(d * d) / (radius * radius * 0.5));

        col   += vec3(falloff) * env;
        total += falloff * env;
    }

    // Color blend: passion = red-orange, tenderness = cream/pink, longing = cool blue
    vec3 base   = vec3(1.0, 0.85, 0.72);  // cream
    vec3 warm   = vec3(1.0, 0.45, 0.20);  // red-orange
    vec3 pink   = vec3(1.0, 0.62, 0.78);
    vec3 cool   = vec3(0.35, 0.55, 0.95);
    vec3 muted  = vec3(0.40, 0.45, 0.55); // gray-blue (sadness)
    vec3 tint = mix(base, pink, u_tenderness * 0.6);
    tint      = mix(tint, warm, u_passion);
    tint      = mix(tint, cool, u_longing);
    tint      = mix(tint, muted, u_sadness * 0.75);

    float intensity = clamp(total, 0.0, 1.4);
    intensity *= (0.5 + u_tenderness * 0.5);
    intensity *= mix(1.0, 0.45, u_sadness);
    fragColor = vec4(tint * intensity * 0.5, intensity * 0.6);
}
"""


class WarmBloomEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 2.0
        self._time = 0.0
        self.passion = 0.2
        self.tenderness = 0.5
        self.longing = 0.0
        self.devotion = 0.3
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
