"""Thread bonds effect for the beloved (love) weather set.

Glowing thread-like curves that connect pairs of anchor points across the
display, suggesting the felt threads between people. Reads the four love
variables from outstate.

  passion    -> color warmth, slight shimmer along the thread
  tenderness -> softer falloff, warmer cream undertones
  longing    -> threads bow apart, color cools toward indigo
  devotion   -> number of threads, brightness, persistence
  sadness    -> threads sag downward, dim and desaturate
  heartbreak -> threads break: dark gaps cut along their length
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect


def shader_thread_bonds(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(ThreadBondsEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized thread_bonds for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize thread_bonds: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.passion    = float(outstate.get('story_passion',    0.3))
        eff.tenderness = float(outstate.get('story_tenderness', 0.3))
        eff.longing    = float(outstate.get('story_longing',    0.0))
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
    gl_Position = vec4(position, 0.55, 1.0);
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

// Distance from p to a quadratic bezier (a, b, c). Approximate via samples.
float bezier_dist(vec2 p, vec2 a, vec2 b, vec2 c) {
    float best = 1e9;
    const int N = 12;
    for (int i = 0; i <= N; ++i) {
        float t = float(i) / float(N);
        vec2 q = mix(mix(a, b, t), mix(b, c, t), t);
        best = min(best, distance(p, q));
    }
    return best;
}

void main() {
    vec2 uv = v_uv;
    uv.y *= 1.6;

    float total = 0.0;
    vec3  accum = vec3(0.0);

    int max_threads = int(floor(2.0 + u_devotion * 6.0));  // up to 8

    for (int i = 0; i < 8; ++i) {
        if (i >= max_threads) break;
        float fi = float(i) * 1.731;

        // Two endpoints — drift slowly through time.
        vec2 a = vec2(hash(fi + 1.0), hash(fi + 2.0));
        vec2 b = vec2(hash(fi + 3.0), hash(fi + 4.0));
        a.x += sin(u_time * 0.10 + fi) * 0.05;
        b.x += cos(u_time * 0.08 + fi) * 0.05;
        a.y *= 1.6; b.y *= 1.6;

        // Control point bows outward when longing is high (threads pulled apart).
        // Sadness adds a downward sag to the control point (gravity of grief).
        vec2 mid    = mix(a, b, 0.5);
        vec2 normal = normalize(vec2(-(b.y - a.y), b.x - a.x));
        float bow   = (hash(fi + 5.0) - 0.5) * (0.05 + u_longing * 0.4);
        bow        += sin(u_time * 0.4 + fi) * 0.03;
        vec2 ctrl   = mid + normal * bow;
        ctrl.y     += u_sadness * 0.25;

        float d = bezier_dist(uv, a, ctrl, b);

        // Thread thickness — softer when tender.
        float thickness = mix(0.004, 0.010, u_tenderness);
        float core      = exp(-pow(d / thickness, 2.0));

        // Persistence: per-thread on/off based on devotion.
        float alive = step(1.0 - 0.1 - u_devotion * 0.85, hash(fi + 9.0));
        if (alive < 0.5) continue;

        // Slight shimmer travels along the thread; rate scales with passion.
        float along   = dot(uv - a, normalize(b - a));
        float shimmer = 0.3 + 0.7 * (0.5 + 0.5 * sin(along * 30.0 - u_time * (1.0 + u_passion * 6.0) + fi));

        // Heartbreak: punch dark gaps along the thread length.
        // Higher heartbreak -> wider gap windows fail.
        float seg = floor(along * 8.0 + fi * 3.7);
        float keep = step(u_heartbreak * 0.7, hash(fi + seg * 13.1));
        core *= keep;

        accum += vec3(core) * shimmer;
        total += core;
    }

    if (total < 0.001) { fragColor = vec4(0.0); return; }

    // Color: gold (devotion baseline) -> warm cream (tenderness) -> red-warm (passion) -> indigo (longing)
    vec3 gold   = vec3(1.0, 0.78, 0.32);
    vec3 cream  = vec3(1.0, 0.88, 0.72);
    vec3 warm   = vec3(1.0, 0.45, 0.30);
    vec3 indigo = vec3(0.30, 0.35, 0.80);
    vec3 col = mix(gold, cream, u_tenderness);
    col      = mix(col, warm,   u_passion * 0.5);
    col      = mix(col, indigo, u_longing);

    // Sadness: desaturate and dim.
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(lum) * 0.7, u_sadness * 0.7);

    float intensity = clamp(total, 0.0, 1.0) * (0.4 + u_devotion * 0.6);
    intensity *= mix(1.0, 0.4, u_sadness);
    fragColor = vec4(col * accum.r * intensity, intensity);
}
"""


class ThreadBondsEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 4.0
        self._time = 0.0
        self.passion = 0.3
        self.tenderness = 0.3
        self.longing = 0.0
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
