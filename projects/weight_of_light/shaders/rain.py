"""WoL rain — vertical streaks falling down each Sky strip.

Pattern B (fullscreen-quad atmospheric layer): renders animated
descending streaks on the Sky group canvas, gated by ``rain_rate``
from outstate. The shader runs continuously as a background event;
``rain_rate=0`` outputs full transparency, so the WOL_CLEAR /
WOL_STARS states make the rain invisible without ever pausing the
effect (smooth crossfades fall out of the standard
WeatherStateController interpolation for free).

Per-strip drops are independent. Each row of the canvas (which is
one physical sky strip) gets a phase offset so adjacent boxes don't
rain in lockstep.

"Down" the physical strip means "from the top toward the bottom of
the polyline." Polyline[0] (LED 0) sits at the chord-bottom of the
ground arc; polyline[-1] is at the top. Canvas x = 0 → bottom of
strip, canvas x = length-1 → top. So "rain falling down" maps to
streaks moving from canvas-x = high (top) toward canvas-x = low
(bottom) — i.e. negative x-velocity in canvas coords.

Time integration: ``_drop_phase`` and ``_speed_phase`` are CPU-side
to avoid the u_time × varying-uniform trap (see
docs/shader_info.txt §Time-based Animation).
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


def shader_wol_rain(state, outstate,
                    color_rgb=(0.85, 0.95, 1.0),
                    drops_per_strip: float = 3.0,
                    fall_speed: float = 0.8,
                    streak_length: float = 0.10):
    """Permanent rain background; intensity follows ``rain_rate``.

    ``drops_per_strip`` sets the steady-state simultaneous drop count
    per strip when ``rain_rate=1``. ``fall_speed`` is canvas units
    per second (canvas x is normalized to [0,1]; 0.8 = a drop crosses
    the strip in ~1.25 s). ``streak_length`` is the streak's length
    in canvas-x units."""
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
                WolRainEffect,
                color=tuple(color_rgb),
                drops_per_strip=float(drops_per_strip),
                fall_speed=float(fall_speed),
                streak_length=float(streak_length),
            )
            state['effect'] = eff
            print(f"[wol_rain] init on frame {frame_id}")
        except Exception as e:
            print(f"[wol_rain] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        # Pull live rain intensity. NB: ``get_state_output`` renames
        # the input param ``rain_rate`` to ``rain`` in outstate (see
        # lib/weather_state.py). Reading 'rain_rate' here always
        # returns the default and silently kills the effect.
        eff.intensity = float(np.clip(outstate.get('rain', 0.0), 0.0, 1.0))

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[wol_rain] cleaned up on frame {frame_id}")


class WolRainEffect(ShaderEffect):
    def __init__(self, viewport,
                 color=(0.85, 0.95, 1.0),
                 drops_per_strip: float = 3.0,
                 fall_speed: float = 0.8,
                 streak_length: float = 0.10):
        super().__init__(viewport)
        # Above background layers (sky=1, ground_twinkle=1, stars=4,
        # rainbow=4). Rain reads as foreground.
        self.render_priority = 5.0

        self.color = (float(color[0]), float(color[1]), float(color[2]))
        self.drops_per_strip = float(drops_per_strip)
        self.fall_speed = float(fall_speed)
        self.streak_length = float(streak_length)
        self.intensity = 0.0   # set by wrapper from rain_rate

        # Row count = number of physical strips on this canvas. Read
        # from the viewport's FBO height so the same shader works on
        # both Sky (19 strips) and Ground (9 strips) without code
        # changes — bumping a hardcoded literal would silently mess
        # up whichever group it wasn't sized for.
        self.rows = max(int(getattr(viewport, "height", 1) or 1), 1)

        # Monotonic seconds counter, mod 1e6 (~11.5 days) to keep
        # float32 precision on the GPU side from degrading at very
        # long uptimes. The earlier mod-1.0 phase wrapped every
        # ~1.25 s and synchronized a discontinuity across all drops
        # when per-slot speed multipliers were applied; using
        # unbounded time + per-slot ``mod()`` in GLSL avoids that
        # entirely.
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
        # Per-drop particle model:
        #   * each slot k on each row has its own constant cycle
        #     period (hashed → 3:1 spread of speeds across drops);
        #   * within each cycle, a single drop is born above the top
        #     of the strip and falls through to below the bottom;
        #   * when the cycle completes, the next iteration "respawns"
        #     a fresh drop at the top.
        # Critically, the CPU-side ``u_time`` is unbounded
        # (modulo a number large enough that wraps don't visibly
        # land inside any drop's cycle), and the modulo into
        # 0..cycle_period is computed PER-DROP via GLSL ``mod()``.
        # That stops the synchronized wrap-jerk that the previous
        # ``fract(offset + global_phase * spd)`` model had every
        # 1.25 s when its global phase rolled over.
        #
        # vUV.x walks along the strip — 0 = LED 0 (bottom / horizon),
        # 1 = last LED (top / zenith). Drops fall downward, so a
        # drop's head_x decreases as time advances within its cycle.
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform float u_intensity;       // 0..1 from rain_rate
        uniform float u_time;            // CPU-integrated seconds (unbounded)
        uniform float u_fall_speed;      // global rate scalar (drops/sec for spd=1)
        uniform float u_drops;           // drops per strip slot count
        uniform float u_streak;          // streak length in canvas-x
        uniform float u_rows;            // # of physical strips on this canvas
        uniform vec3  u_color;
        out vec4 fragColor;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            if (u_intensity <= 0.001) { fragColor = vec4(0.0); return; }
            float row = floor(vUV.y * u_rows);
            float n = u_drops;
            float maxAlpha = 0.0;
            for (int k = 0; k < 6; k++) {
                if (float(k) >= n) break;
                float kf = float(k);

                // Per-slot fall speed in [0.5, 1.5] of the global
                // u_fall_speed. Stable per (row, k) hash so a given
                // slot's drop falls at a constant rate (drops don't
                // accelerate / decelerate mid-fall) and adjacent
                // slots have visibly different paces.
                float spd_mult = 0.5 + hash(vec2(row, kf * 11.31 + 0.41));
                float cycle = 1.0 / (u_fall_speed * spd_mult);

                // Per-slot time offset spread over a few seconds so
                // slots don't all spawn drops at the same instant.
                // Multiplied by 5.0 to give a comfortable spread
                // larger than any single cycle, ensuring slots have
                // diverse phases.
                float offset = hash(vec2(row, kf * 7.13)) * 5.0;

                // Time within this slot's current cycle, in seconds.
                // ``mod`` is per-slot — no synchronized wrap.
                float t_in_cycle = mod(offset + u_time, cycle);
                float frac = t_in_cycle / cycle;       // 0..1 in cycle

                // Head walks from just above the top of the strip
                // (head_x = 1 + streak at frac=0) to just below the
                // bottom (head_x = -streak at frac=1). Total span =
                // 1 + 2*streak so the drop has clean off-screen
                // entry + exit margins.
                float total_span = 1.0 + 2.0 * u_streak;
                float head_x = 1.0 + u_streak - frac * total_span;

                float dx = head_x - vUV.x;
                if (dx < 0.0 || dx > u_streak) continue;
                float a = 1.0 - (dx / u_streak);
                maxAlpha = max(maxAlpha, a);
            }
            fragColor = vec4(u_color, maxAlpha * u_intensity);
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
        # Time advances regardless of intensity (drops don't slow
        # down when the rain weakens — they just become rarer /
        # more transparent because alpha scales with u_intensity).
        # mod 1e6 is hygiene against float32 precision rot at
        # multi-day uptimes; never visible in any single drop's
        # cycle since cycles are ~1 s, far shorter than the wrap.
        self._time = (self._time + dt) % 1.0e6

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        if self.intensity <= 0.001:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glUniform1f(glGetUniformLocation(self.shader, "u_intensity"),
                        float(self.intensity))
            glUniform1f(glGetUniformLocation(self.shader, "u_time"),
                        float(self._time))
            glUniform1f(glGetUniformLocation(self.shader, "u_fall_speed"),
                        float(self.fall_speed))
            glUniform1f(glGetUniformLocation(self.shader, "u_drops"),
                        float(self.drops_per_strip))
            glUniform1f(glGetUniformLocation(self.shader, "u_streak"),
                        float(self.streak_length))
            glUniform1f(glGetUniformLocation(self.shader, "u_rows"),
                        float(self.rows))
            glUniform3f(glGetUniformLocation(self.shader, "u_color"),
                        *self.color)
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
