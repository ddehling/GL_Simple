"""Pride flag effect — billowing rainbow overlay for the BarTiki Pride state.

Pattern B fullscreen-quad atmospheric layer. Cycles through seven LGBTQIA+
flags with smooth crossfades, sinusoidal "billowing" displacement, and
sparkle highlights. Sits high in the BarTiki render stack so it tints the
whole scene (map, bay, city lights) but stays below the cloud layer.

Drives:
  pride_intensity   -> overall opacity (0 = invisible, 1 = full saturation)
  wind              -> wave amplitude (more wind, more billow)
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


# Number of flags in the cycle; must match the GLSL switch in flag_color().
_NUM_FLAGS = 7
# Full cycle through all flags = 40 seconds. Per-flag slot is total / N,
# split into a hold and a crossfade. With 7 flags that's ~5.71 s/slot.
_CYCLE_TOTAL_S = 40.0
_FLAG_FADE_S = 1.5
_FLAG_HOLD_S = _CYCLE_TOTAL_S / _NUM_FLAGS - _FLAG_FADE_S


def shader_pride_flag(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(PrideFlagEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized pride_flag for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize pride_flag: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.intensity = float(outstate.get('pride_intensity', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 2.0
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
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_intensity;
uniform float u_fade;
uniform float u_wind;
uniform int   u_flag_a;
uniform int   u_flag_b;
uniform float u_flag_blend;   // 0 = a, 1 = b

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Look up the flag color for a given flag index and band fraction in [0,1].
// "Top" of the flag is band_t = 0; bottom is band_t = 1.
vec3 flag_color(int idx, float t) {
    t = clamp(t, 0.0, 0.9999);
    if (idx == 0) {
        // Rainbow Pride (Gilbert Baker classic 6-stripe).
        if (t < 1.0/6.0) return vec3(0.89, 0.01, 0.01);  // red
        if (t < 2.0/6.0) return vec3(1.00, 0.55, 0.00);  // orange
        if (t < 3.0/6.0) return vec3(1.00, 0.93, 0.00);  // yellow
        if (t < 4.0/6.0) return vec3(0.05, 0.55, 0.10);  // green
        if (t < 5.0/6.0) return vec3(0.00, 0.30, 0.85);  // blue
        return vec3(0.46, 0.02, 0.55);                   // purple
    } else if (idx == 1) {
        // Trans (light-blue / pink / white / pink / light-blue).
        if (t < 0.20) return vec3(0.36, 0.81, 0.98);
        if (t < 0.40) return vec3(0.96, 0.66, 0.72);
        if (t < 0.60) return vec3(1.00, 1.00, 1.00);
        if (t < 0.80) return vec3(0.96, 0.66, 0.72);
        return vec3(0.36, 0.81, 0.98);
    } else if (idx == 2) {
        // Bisexual (pink / purple / blue, with the purple stripe narrower).
        if (t < 0.40) return vec3(0.84, 0.01, 0.46);
        if (t < 0.60) return vec3(0.61, 0.31, 0.59);
        return vec3(0.00, 0.22, 0.69);
    } else if (idx == 3) {
        // Lesbian (5-stripe community flag).
        if (t < 0.20) return vec3(0.83, 0.18, 0.10);
        if (t < 0.40) return vec3(0.93, 0.51, 0.27);
        if (t < 0.60) return vec3(1.00, 1.00, 1.00);
        if (t < 0.80) return vec3(0.83, 0.38, 0.62);
        return vec3(0.65, 0.00, 0.40);
    } else if (idx == 4) {
        // Pansexual (pink / yellow / cyan).
        if (t < 1.0/3.0) return vec3(1.00, 0.13, 0.55);
        if (t < 2.0/3.0) return vec3(1.00, 0.84, 0.00);
        return vec3(0.13, 0.69, 1.00);
    } else if (idx == 5) {
        // Asexual (black / grey / white / purple).
        if (t < 0.25) return vec3(0.00, 0.00, 0.00);
        if (t < 0.50) return vec3(0.64, 0.64, 0.64);
        if (t < 0.75) return vec3(1.00, 1.00, 1.00);
        return vec3(0.50, 0.00, 0.50);
    } else {
        // Nonbinary (yellow / white / purple / black).
        if (t < 0.25) return vec3(1.00, 0.95, 0.00);
        if (t < 0.50) return vec3(1.00, 1.00, 1.00);
        if (t < 0.75) return vec3(0.61, 0.35, 0.83);
        return vec3(0.16, 0.16, 0.16);
    }
}

void main() {
    if (u_intensity < 0.02 || u_fade < 0.005) discard;

    vec2 uv = v_uv;

    // Billowing wave: two sinusoids mixed for a hand-stitched feel.
    // Amplitude scales with wind so windy states swing harder.
    float wave = sin(uv.x * 8.0  + u_time * 1.2) * 0.025;
    wave     += sin(uv.x * 14.0 - u_time * 0.7) * 0.015;
    wave     *= 1.0 + u_wind * 0.45;

    float band_t = clamp(uv.y + wave, 0.0, 1.0);

    vec3 color_a = flag_color(u_flag_a, band_t);
    vec3 color_b = flag_color(u_flag_b, band_t);
    vec3 color   = mix(color_a, color_b, u_flag_blend);

    // Sparkle highlights — a sparse twinkle of bright pinpoints over the
    // flag, like glitter catching the sun. Cell-based hash so each sparkle
    // has its own per-cell phase and only ~1.5% of cells light up at once.
    vec2 cell = floor(uv * vec2(60.0, 80.0));
    float phase = hash(cell);
    float spark = step(0.985, fract(u_time * 0.6 + phase));
    color = mix(color, vec3(1.0), spark * 0.65);

    // Boost saturation slightly so the wash reads strongly on top of the
    // BART map and city lights, then taper alpha at the very bottom of
    // the FBO so the inner ring (foreground) keeps its identity.
    float alpha = u_intensity * u_fade * 0.60;
    alpha *= smoothstep(0.0, 0.10, uv.y);

    fragColor = vec4(color, alpha);
}
"""


class PrideFlagEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Sit above bart_map (5), bay_shimmer (5.1), city_lights (5.3),
        # sunrise (5.4) and aurora (5.5), but below clouds (6) so weather
        # still drifts naturally over the scene.
        self.render_priority = 5.7
        self._time = 0.0
        self.intensity = 0.0
        self.wind = 0.0
        self.fade = 0.0

        # Flag-cycle state (CPU-driven; the shader just receives current
        # indices + crossfade blend).
        self._cycle_t = 0.0
        self._flag_a = 0
        self._flag_b = 1
        self._flag_blend = 0.0

        # Cached uniform locations populated in setup_buffers().
        self._u_time = -1
        self._u_intensity = -1
        self._u_fade = -1
        self._u_wind = -1
        self._u_flag_a = -1
        self._u_flag_b = -1
        self._u_flag_blend = -1

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

        # Cache uniform locations.
        self._u_time = glGetUniformLocation(self.shader, "u_time")
        self._u_intensity = glGetUniformLocation(self.shader, "u_intensity")
        self._u_fade = glGetUniformLocation(self.shader, "u_fade")
        self._u_wind = glGetUniformLocation(self.shader, "u_wind")
        self._u_flag_a = glGetUniformLocation(self.shader, "u_flag_a")
        self._u_flag_b = glGetUniformLocation(self.shader, "u_flag_b")
        self._u_flag_blend = glGetUniformLocation(self.shader, "u_flag_blend")

    def update(self, dt: float, state: Dict):
        self._time += dt
        # Advance the flag-cycle timer. One slot = HOLD then FADE; total
        # period = HOLD + FADE per flag. During the hold the blend is 0;
        # during the fade it ramps 0→1 and we rotate when it reaches 1.
        self._cycle_t += dt
        period = _FLAG_HOLD_S + _FLAG_FADE_S
        if self._cycle_t < _FLAG_HOLD_S:
            self._flag_blend = 0.0
        elif self._cycle_t < period:
            self._flag_blend = (self._cycle_t - _FLAG_HOLD_S) / _FLAG_FADE_S
        else:
            self._cycle_t -= period
            self._flag_a = self._flag_b
            self._flag_b = (self._flag_b + 1) % _NUM_FLAGS
            self._flag_blend = 0.0

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        # Pattern B: depth disabled, restored on exit.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(self._u_time, self._time)
        glUniform1f(self._u_intensity, self.intensity)
        glUniform1f(self._u_fade, self.fade)
        glUniform1f(self._u_wind, self.wind)
        glUniform1i(self._u_flag_a, self._flag_a)
        glUniform1i(self._u_flag_b, self._flag_b)
        glUniform1f(self._u_flag_blend, self._flag_blend)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
