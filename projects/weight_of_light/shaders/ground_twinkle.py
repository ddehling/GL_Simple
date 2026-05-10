"""WoL ground twinkle — per-pixel spawn + exponential decay.

Pattern B (fullscreen-quad atmospheric layer). Twinkle model is the
classic stars-at-night recipe rather than a slot-based animation:

  * each pixel along each ground arc has its own staggered "cycle"
    (cycle_seconds long)
  * once per cycle, the pixel rolls a hash-driven die — if the roll
    is below ``spawn_chance``, this pixel sparks this cycle
  * a sparking pixel ramps to full brightness instantly, then
    exponentially decays at ``decay_rate`` (units of e-folds per
    second) for the rest of the cycle
  * non-sparking pixels output zero alpha and stay dark

Per-pixel phase offsets (hashed from pixel coords) spread spawns
across the cycle window so the field churns continuously instead of
strobing all-pixels-at-once at cycle boundaries. Each spark picks its
hue from the green/brown earthy palette, except an ``echo_chance``
fraction adopt the active weather's color (blue when raining, warm
white when starry).

Spawn rate per arc ≈ (300 pixels × spawn_chance) / cycle_seconds.
Defaults give ~6/sec per arc — a steady twinkle without overcrowding.

Time integration: the GLSL gets a single CPU-integrated ``u_time``
that grows monotonically (mod 1e6 for float-precision hygiene).
Spawn windows + per-pixel ages all derive from u_time + per-pixel
hash offsets; no varying uniform is multiplied by u_time anywhere.
See docs/shader_info.txt §Time-based Animation.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


def shader_wol_ground_twinkle(state, outstate,
                              spawn_chance: float = 0.12,
                              cycle_seconds: float = 2.0,
                              decay_rate: float = 1.6,
                              max_brightness: float = 1.0,
                              echo_chance: float = 0.30):
    """Per-pixel ground twinkle: random sparks that fade out.

    ``spawn_chance`` is the per-pixel-per-cycle probability of
    sparking. ``cycle_seconds`` is the length of each pixel's
    spawn window (lower = sparks fire more often per pixel,
    higher = sparser). Combined: spawn rate per arc ≈
    300 × spawn_chance / cycle_seconds.

    ``decay_rate`` is how fast a spark fades out — 1.6 e-folds per
    second means a spark drops to 20 % brightness after 1 s. Increase
    for snappier flashes, decrease for lingering glows.

    ``echo_chance`` is the fraction of sparks that take the active
    weather's color instead of the earthy palette (blue during rain,
    warm white during stars).
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            eff = viewport.add_effect(
                WolGroundTwinkleEffect,
                spawn_chance=float(spawn_chance),
                cycle_seconds=float(cycle_seconds),
                decay_rate=float(decay_rate),
                max_brightness=float(max_brightness),
                echo_chance=float(echo_chance),
            )
            state['effect'] = eff
            print(f"[wol_ground_twinkle] init on frame {frame_id} "
                  f"(spawn_chance={spawn_chance}, cycle={cycle_seconds}s, "
                  f"decay={decay_rate}/s)")
        except Exception as e:
            print(f"[wol_ground_twinkle] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        # Pull the dominant active weather for the echo color.
        # ``rain_rate`` is exposed as ``rain`` in outstate (legacy
        # rename in get_state_output).
        rain = float(np.clip(outstate.get('rain', 0.0), 0.0, 1.0))
        stars = float(np.clip(outstate.get('starryness', 0.0), 0.0, 1.0))
        if rain >= stars and rain > 0.001:
            eff.echo_color = (0.55, 0.75, 1.0)   # rain blue
            eff.echo_intensity = rain
        elif stars > 0.001:
            eff.echo_color = (0.95, 0.95, 0.85)  # star warm-white
            eff.echo_intensity = stars
        else:
            eff.echo_intensity = 0.0
        # Ambient brightness floor (season-driven dim at night).
        ambient = float(outstate.get('ambient_light', 1.0))
        eff.brightness_mul = max(0.4, ambient)

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[wol_ground_twinkle] cleaned up on frame {frame_id}")


class WolGroundTwinkleEffect(ShaderEffect):
    def __init__(self, viewport,
                 spawn_chance: float = 0.12,
                 cycle_seconds: float = 2.0,
                 decay_rate: float = 1.6,
                 max_brightness: float = 1.0,
                 echo_chance: float = 0.30):
        super().__init__(viewport)
        self.render_priority = 1.0   # back layer of Ground stack

        self.spawn_chance = float(spawn_chance)
        self.cycle_seconds = max(float(cycle_seconds), 0.05)
        self.decay_rate = float(decay_rate)
        self.max_brightness = float(max_brightness)
        self.echo_chance = float(echo_chance)

        self.echo_color = (0.0, 0.0, 0.0)
        self.echo_intensity = 0.0
        self.brightness_mul = 1.0

        # Monotonic time accumulator. Modded mod 1e6 to keep float
        # precision tight at long uptimes; never wraps to 0 inside
        # any single pixel's cycle window.
        self._time = 0.0

        self.position_VBO = None
        self.EBO = None
        self._vertices = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)
        self._indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        layout(location = 0) in vec2 position;
        out vec2 vUV;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vUV = position * 0.5 + 0.5;
        }
        """

    def get_fragment_shader(self):
        # Per-pixel logic:
        #   1. Discretize the fragment to a "pixel id" (row, col) so
        #      the same physical LED gets the same hash across frames.
        #   2. Each pixel has its own cycle with phase offset; when
        #      cycle_id increments, roll a hash to decide spawn.
        #   3. If spawning, age = (time mod cycle) - spawn_offset;
        #      brightness = exp(-age * decay) clamped to [0,1].
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform float u_time;            // CPU-integrated seconds
        uniform float u_cycle;           // seconds per pixel cycle
        uniform float u_spawn_chance;    // 0..1 per-pixel-per-cycle
        uniform float u_decay;           // e-folds per second
        uniform float u_max_brightness;
        uniform float u_echo_chance;
        uniform float u_echo_intensity;
        uniform vec3  u_echo_color;
        uniform float u_brightness_mul;
        out vec4 fragColor;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        // Earthy base palette; h ∈ [0,1] picks moss / olive / brown.
        vec3 earthy(float h) {
            vec3 moss  = vec3(0.20, 0.55, 0.20);
            vec3 olive = vec3(0.60, 0.55, 0.18);
            vec3 brown = vec3(0.60, 0.32, 0.13);
            if (h < 0.5) {
                return mix(moss, olive, h * 2.0);
            } else {
                return mix(olive, brown, (h - 0.5) * 2.0);
            }
        }

        void main() {
            // Quantize to per-LED granularity. Ground canvas is
            // 300×9; floor maps each fragment to one of 300×9 cells.
            // Sub-pixel sampling collapses to the same pixel-id, so
            // hashes are stable per physical LED.
            vec2 pid = vec2(floor(vUV.x * 300.0), floor(vUV.y * 9.0));

            // Per-pixel phase offset spreads spawn windows so all
            // pixels in a row don't roll their dice at the same
            // wall-clock instant.
            float phase_offset = hash(pid + vec2(13.0, 71.0)) * u_cycle;
            float t_local = u_time + phase_offset;

            float cycle_id = floor(t_local / u_cycle);
            float t_in_cycle = t_local - cycle_id * u_cycle;

            // Roll fire die for THIS cycle on THIS pixel.
            float roll = hash(pid + vec2(cycle_id, 7.91));
            if (roll >= u_spawn_chance) {
                fragColor = vec4(0.0);
                return;
            }

            // The pixel sparks. Age = seconds since spawn (from the
            // start of this cycle).
            float age = t_in_cycle;
            float brightness = exp(-age * u_decay);
            if (brightness < 0.005) {
                fragColor = vec4(0.0);
                return;
            }

            // Pick color. Per-cycle hue + per-cycle echo roll, so the
            // same pixel firing in successive cycles looks fresh
            // each time.
            float hue = hash(pid + vec2(cycle_id, 33.7));
            float echo_roll = hash(pid + vec2(cycle_id, 91.3));
            vec3 color;
            if (echo_roll < u_echo_chance * u_echo_intensity) {
                color = u_echo_color;
            } else {
                color = earthy(hue);
            }

            float alpha = brightness * u_max_brightness * u_brightness_mul;
            fragColor = vec4(color, alpha);
        }
        """

    def compile_shader(self):
        vert = shaders.compileShader(self.get_vertex_shader(), GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.get_fragment_shader(), GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)

    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self._vertices.nbytes,
                     self._vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes,
                     self._indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time = (self._time + dt) % 1.0e6

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glUniform1f(glGetUniformLocation(self.shader, "u_time"),
                        float(self._time))
            glUniform1f(glGetUniformLocation(self.shader, "u_cycle"),
                        float(self.cycle_seconds))
            glUniform1f(glGetUniformLocation(self.shader, "u_spawn_chance"),
                        float(self.spawn_chance))
            glUniform1f(glGetUniformLocation(self.shader, "u_decay"),
                        float(self.decay_rate))
            glUniform1f(glGetUniformLocation(self.shader, "u_max_brightness"),
                        float(self.max_brightness))
            glUniform1f(glGetUniformLocation(self.shader, "u_echo_chance"),
                        float(self.echo_chance))
            glUniform1f(glGetUniformLocation(self.shader, "u_echo_intensity"),
                        float(self.echo_intensity))
            glUniform3f(glGetUniformLocation(self.shader, "u_echo_color"),
                        *self.echo_color)
            glUniform1f(glGetUniformLocation(self.shader, "u_brightness_mul"),
                        float(self.brightness_mul))
            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glUseProgram(0)
        finally:
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)

    def cleanup(self):
        try:
            if self.position_VBO is not None:
                glDeleteBuffers(1, [self.position_VBO])
                self.position_VBO = None
            if self.EBO is not None:
                glDeleteBuffers(1, [self.EBO])
                self.EBO = None
            if self.VAO is not None:
                glDeleteVertexArrays(1, [self.VAO])
                self.VAO = None
            if self.shader:
                glDeleteProgram(self.shader)
                self.shader = None
        except Exception:
            pass
