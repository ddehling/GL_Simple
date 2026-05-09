"""Heart pulse effect for the beloved (love) weather set.

Procedural cardioid glows arranged on a grid that pulse like heartbeats.
Reads the four love variables from outstate (published by NarrativePlayer
as story_passion, story_tenderness, story_longing, story_devotion).

  passion    -> pulse rate + sparks
  tenderness -> base brightness + softness
  longing    -> color shift toward blue/violet, vertical drift
  devotion   -> number of visible hearts + persistence between beats
  sadness    -> desaturates color, dims overall brightness
  heartbreak -> skipped beats and a fracturing notch through the heart
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect


# ---------------------------------------------------------------------------
# Event wrapper
# ---------------------------------------------------------------------------

def shader_heart_pulse(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(HeartPulseEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized heart_pulse for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize heart_pulse: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.passion    = float(outstate.get('story_passion',    0.3))
        eff.tenderness = float(outstate.get('story_tenderness', 0.3))
        eff.longing    = float(outstate.get('story_longing',    0.0))
        eff.devotion   = float(outstate.get('story_devotion',   0.3))
        eff.sadness    = float(outstate.get('story_sadness',    0.0))
        eff.heartbreak = float(outstate.get('story_heartbreak', 0.0))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ---------------------------------------------------------------------------
# Shader source
# ---------------------------------------------------------------------------

_VERT = """
#version 310 es
precision highp float;
in vec2 position;
out vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.5, 1.0);
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

// Implicit cardioid: (x^2 + y^2 - 1)^3 - x^2 y^3 = 0  (heart curve)
// Returns a soft mask that falls off outside the heart.
float heart_mask(vec2 p) {
    p.y = -p.y;                     // flip so heart is upright
    p.y += 0.25;                    // recenter
    float a = p.x*p.x + p.y*p.y - 1.0;
    float v = a*a*a - p.x*p.x * p.y*p.y*p.y;
    return smoothstep(0.6, -0.2, v);
}

void main() {
    // Tile the screen into cells; each cell hosts one heart.
    float cells_x = mix(2.0, 6.0, u_devotion);
    vec2 cell_id   = floor(v_uv * vec2(cells_x, cells_x * 1.6));
    vec2 cell_uv   = fract(v_uv * vec2(cells_x, cells_x * 1.6)) * 2.0 - 1.0;

    // Each cell skips with a probability inverse to devotion.
    float visible = step(1.0 - u_devotion * 0.9 - 0.1, hash(cell_id));
    if (visible < 0.5) { fragColor = vec4(0.0); return; }

    // Per-cell jitter so hearts aren't grid-aligned exactly.
    vec2 jitter = vec2(hash(cell_id + 1.3), hash(cell_id + 7.7)) - 0.5;
    cell_uv -= jitter * 0.6;

    // Vertical drift driven by longing (hearts ascend slowly when longing high)
    cell_uv.y += sin(u_time * 0.25 + hash(cell_id) * 6.28) * u_longing * 0.4;

    // Pulse: rate scales with passion. Phase varies per cell.
    float rate  = mix(0.5, 2.5, u_passion);
    float phase = hash(cell_id + 3.1) * 6.2831;
    float beat  = pow(max(0.0, sin(u_time * rate + phase)), 6.0);

    // Heartbreak: skip occasional beats per cell.
    float skip = step(1.0 - u_heartbreak * 0.6, hash(cell_id + floor(u_time * 0.6)));
    beat *= 1.0 - skip;

    // Persistence floor from devotion: hearts never fully vanish if devoted.
    float glow  = mix(beat, 0.4 + 0.6 * beat, u_devotion);

    // Heart geometry, scaled — softer when tender.
    float scale = 0.55 + u_tenderness * 0.25;
    float m = heart_mask(cell_uv / scale);

    // Heartbreak: carve a thin notch / crack down the right lobe.
    float crack_x = 0.05 + (hash(cell_id + 4.4) - 0.5) * 0.2;
    float crack_w = 0.015 + u_heartbreak * 0.04;
    float notch  = smoothstep(crack_w, 0.0, abs(cell_uv.x - crack_x));
    m *= 1.0 - notch * u_heartbreak;

    // Color: red (passion) -> pink (tenderness) -> violet/blue (longing)
    vec3 red    = vec3(1.0, 0.18, 0.25);
    vec3 pink   = vec3(1.0, 0.65, 0.78);
    vec3 violet = vec3(0.45, 0.30, 0.85);
    vec3 col = mix(red, pink, u_tenderness);
    col      = mix(col, violet, u_longing);

    // Sparks: small bright random bursts when passion is high.
    float spark = step(0.985 - u_passion * 0.05, hash(cell_id + floor(u_time * 6.0)));
    col += vec3(1.0, 0.7, 0.4) * spark * u_passion * 0.6;

    // Sadness: desaturate toward gray-blue, dim overall.
    vec3 gray = vec3(0.30, 0.32, 0.40);
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, mix(vec3(lum), gray, 0.5), u_sadness * 0.7);

    float intensity = m * glow * (0.35 + u_tenderness * 0.65);
    intensity *= mix(1.0, 0.45, u_sadness);
    fragColor = vec4(col * intensity, intensity);
}
"""


# ---------------------------------------------------------------------------
# Effect class
# ---------------------------------------------------------------------------

class HeartPulseEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 3.0
        self._time = 0.0
        self.passion = 0.3
        self.tenderness = 0.3
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
