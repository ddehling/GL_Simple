"""
Desert rain — fan-aware procedural rain that falls VERTICALLY in physical
(feet) space. Pattern B (fullscreen quad, no depth writes).

Why a desert-specific rain shader? The shared `rain` shader treats drops
as moving in buffer pixel space, which on the polar fan looks like rain
falling RADIALLY inward (toward the inner ring) rather than down. Here
we reason in physical feet via fan_uv_to_physical, so each drop's
trajectory is a real straight line in the room — and wind tilts it left
or right naturally.

Spawn density and visibility scale with `outstate['rain']` (which is the
weather state's rain_rate). Wind comes from `outstate['wind']`.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords

VERTEX_SHADER = """#version 310 es
precision highp float;

in vec2 position;
out vec2 v_uv;

void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = f"""#version 310 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

uniform float u_time;
uniform float u_rain_rate;       // 0..1 visibility/density
uniform float u_wind_tilt;       // tan of angle from vertical (-0.7..+0.7)
uniform float u_strength;        // 0..1 master alpha (fade in/out)

float hash11(float n) {{ return fract(sin(n) * 43758.5453); }}

void main() {{
    float master = u_rain_rate * u_strength;
    if (master < 0.005) discard;

    vec2 phys = fan_uv_to_physical(v_uv);

    // ----- Procedural rain -----
    // Tile the fan in physical feet. Each "column" hosts a single drop on
    // a continuously cycling vertical loop. Wind tilts the trajectory so
    // the streak angles naturally.
    //
    // Streak width scales with local pixel density so streaks stay visible
    // (no sub-pixel alias) without bleeding into giant blobs at the outer
    // ring. ~0.7 cols wide everywhere keeps it thin and distinct.
    float local_dx_ft = fan_arc_width_ft(v_uv);
    float streak_w = max(0.05, 0.7 * local_dx_ft);

    const float COL_W      = 0.55;    // ft between drops horizontally
    const float STREAK_LEN = 1.20;    // ft, length of motion-blur streak
    const float TOP_Y      = 21.5;    // ft, drops spawn above the fan
    const float BOTTOM_Y   = -2.0;    // ft, drops recycle below the fan
    const float FALL_SPEED = 14.0;    // ft/s — constant so u_time × FALL is safe

    // The streak trails BEHIND a falling drop (so above it), with the
    // wind-tilted axis pointing up-and-back relative to the drop's motion.
    vec2 v_streak = normalize(vec2(-u_wind_tilt, 1.0));
    vec2 perp     = vec2(-v_streak.y, v_streak.x);

    float fall_dist = TOP_Y - BOTTOM_Y;

    // **Wind-drift compensation**: a drop currently passing through this
    // fragment's y has, by the time it got there, drifted laterally by
    // tilt × (TOP_Y − phys.y) feet. So the column whose drop COULD pass
    // through (phys.x, phys.y) is offset upwind from phys.x. Without
    // this correction, fragments at low phys.y look at the wrong columns
    // (the original spawn column has long since drifted far downwind), so
    // rain visibly cuts out below a certain y when the wind is on.
    float effective_x = phys.x - u_wind_tilt * (TOP_Y - phys.y);
    float col_id_self = floor(effective_x / COL_W);

    float rain = 0.0;
    for (int ofs = -1; ofs <= 1; ofs++) {{
        float ci = col_id_self + float(ofs);

        // Sparse-out at low rain_rate via per-column hash gate.
        if (hash11(ci * 13.71) > u_rain_rate * 1.5) continue;

        float jitter_x = (hash11(ci * 7.13)  - 0.5) * COL_W * 0.6;
        float phase    =  hash11(ci * 11.3 + 5.7);

        // Drop's cycle position
        float t = fract(u_time * FALL_SPEED / fall_dist + phase);
        float drop_y = TOP_Y - t * fall_dist;
        float drop_x = (ci + 0.5) * COL_W + jitter_x
                     + u_wind_tilt * (TOP_Y - drop_y);

        vec2 d = phys - vec2(drop_x, drop_y);
        float along  = dot(d, v_streak);
        float across = dot(d, perp);

        if (along >= 0.0 && along <= STREAK_LEN && abs(across) < streak_w) {{
            float a = (1.0 - along / STREAK_LEN)
                    * smoothstep(streak_w, streak_w * 0.3, abs(across));
            rain = max(rain, a);
        }}
    }}

    rain *= master;
    if (rain < 0.005) discard;

    // Bright cyan-blue rain — distinct against both day and night sky.
    // Saturation matches the existing `rain` shader's character.
    vec3 col = vec3(0.55, 0.85, 1.00);
    fragColor = vec4(col, rain);   // straight alpha
}}
"""


def shader_desert_rain(state, outstate, fade_duration=4.0):
    """Background event: vertical-in-fan-space rain with wind tilt."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DesertRainEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing desert_rain: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.rain_rate = float(outstate.get('rain', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))

        elapsed = state['elapsed_time']
        duration = state.get('duration')
        if duration is None or duration <= 0:
            eff.fade = 1.0
        else:
            if elapsed < fade_duration:
                f = elapsed / fade_duration
            elif elapsed > duration - fade_duration:
                f = (duration - elapsed) / fade_duration
            else:
                f = 1.0
            eff.fade = float(np.clip(f, 0.0, 1.0))

    if state['count'] == -1:
        if 'effect' in state:
            eff = state['effect']
            if eff in viewport.effects:
                viewport.effects.remove(eff)
            eff.cleanup()
            del state['effect']


class DesertRainEffect(ShaderEffect):
    """Fullscreen-quad procedural rain in physical fan space."""

    def __init__(self, viewport):
        super().__init__(viewport)
        # Sits BETWEEN clouds (6) and dunes (7) so the dune silhouette
        # occludes the rain behind it — drops disappear behind the
        # foreground hills. (Pattern B layers compose by render_priority,
        # not depth, so this is purely a draw-order trick.)
        self.render_priority = 6.5
        self.rain_rate = 0.0
        self.wind = 0.0
        self.fade = 0.0
        self._time = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)

    def compile_shader(self):
        v = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        f = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(v, f)

    def setup_buffers(self):
        verts = np.array([-1, -1,  1, -1,  -1, 1,  1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        self.VBOs.append(vbo)
        loc = glGetAttribLocation(self.shader, "position")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        glUseProgram(self.shader)
        self._fan.set_uniforms(self.shader)
        glUseProgram(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt

    def render(self, state: Dict):
        if not self.enabled or self.fade < 0.01 or self.rain_rate < 0.005:
            return

        # Wind tilt = horizontal velocity / fall speed. Clamp to ±0.7
        # (~35° from vertical) so heavy wind doesn't make rain near-
        # horizontal. Wind in outstate is signed (`wind_speed * cos(...)`)
        # and ranges roughly [-2, +2] for desert states.
        tilt = max(-0.7, min(0.7, self.wind * 0.25))

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        self._fan.set_uniforms(self.shader)

        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_rain_rate"), self.rain_rate)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind_tilt"), tilt)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"), self.fade)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
