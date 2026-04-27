"""
Desert sky — split into two layered fullscreen-quad passes (Pattern B):

  - DesertSkyBackdropEffect  (priority -1)  — sky gradient, behind stars
  - DesertSunMoonEffect      (priority  5)  — sun/moon disc, in front of stars
                                              but behind clouds (6) and dunes (7)

Splitting matters because stars (priority 0) need to twinkle ON TOP OF the
sky gradient but BEHIND the moon disc — otherwise either the gradient covers
the stars (single high-priority pass) or the stars sit over the moon
(single low-priority pass).

Both passes reason in PHYSICAL fan-space (feet) via fan_uv_to_physical so
the gradient is a real horizon-to-zenith ramp and the disc is a real circle
on the semicircular fan.
"""
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords

INNER_R_FT = 4.0
OUTER_R_FT = 20.6

VERTEX_SHADER = """#version 310 es
precision highp float;

in vec2 position;
out vec2 v_uv;

void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Sky-backdrop fragment shader (gradient only, alpha = u_sky_strength)
# ---------------------------------------------------------------------------
BACKDROP_FRAGMENT = f"""#version 310 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

uniform vec3  u_zenith_color;
uniform vec3  u_horizon_color;
uniform float u_sky_strength;

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);
    // Vertical gradient in feet. Inner ring (4 ft) → horizon, outer ring
    // (20.6 ft) → zenith. Soft S-curve so the horizon band reads warmer.
    float t_y = clamp((phys.y - 4.0) / 16.0, 0.0, 1.0);
    float s   = smoothstep(0.0, 1.0, t_y);
    vec3 col  = mix(u_horizon_color, u_zenith_color, s);

    if (u_sky_strength < 0.005) discard;
    fragColor = vec4(col, u_sky_strength);   // straight alpha
}}
"""

# ---------------------------------------------------------------------------
# Sun/moon-disc fragment shader (disc only)
# ---------------------------------------------------------------------------
DISC_FRAGMENT = f"""#version 310 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

uniform vec2  u_disc_center_ft;
uniform float u_disc_radius_ft;
uniform float u_disc_corona_ft;
uniform vec3  u_disc_color;
uniform float u_disc_intensity;
uniform float u_time;

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);
    float r = length(phys - u_disc_center_ft);

    float core   = smoothstep(u_disc_radius_ft, u_disc_radius_ft * 0.85, r);
    float corona = smoothstep(u_disc_corona_ft, u_disc_radius_ft, r);

    float shimmer   = 1.0 + 0.05 * sin(u_time * 1.4 + phys.x * 1.6);
    vec3 disc_rgb   = u_disc_color * (1.05 + 0.05 * sin(u_time * 0.8));
    vec3 corona_rgb = u_disc_color * 0.65 * shimmer;

    float a = clamp(core + corona * 0.55, 0.0, 1.0) * u_disc_intensity;
    if (a < 0.005) discard;

    vec3 col = mix(corona_rgb, disc_rgb, core);
    fragColor = vec4(col, a);   // straight alpha
}}
"""


# ===========================================================================
# Event wrapper — creates BOTH effects and updates BOTH each frame
# ===========================================================================

def shader_desert_sky(state, outstate, fade_duration=4.0):
    """Background event: sky gradient + sun-or-moon disc tracking time-of-day.

    Adds two ShaderEffect instances at different render priorities so stars
    can sit between the gradient and the disc.
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
            backdrop = viewport.add_effect(DesertSkyBackdropEffect)
            disc     = viewport.add_effect(DesertSunMoonEffect)
            state['backdrop'] = backdrop
            state['disc']     = disc
        except Exception as e:
            import traceback
            print(f"ERROR initializing desert_sky: {e}")
            traceback.print_exc()
            return

    if 'backdrop' in state and 'disc' in state:
        # Sky tracks the GLOBAL day cycle (`season`), not any individual
        # weather state. The desert weather set is expected to provide enough
        # states across all season_preference bands that the random
        # transition logic naturally lands on a TOD-appropriate state.
        season = float(outstate.get('season', 0.5))
        cv     = float(outstate.get('celestial_visibility', 0.0))
        sp     = float(outstate.get('spookyness', 0.0))
        fc_raw = outstate.get('fog_color', (0.7, 0.7, 0.7))
        fc     = (float(fc_raw[0]), float(fc_raw[1]), float(fc_raw[2]))

        elapsed = state['elapsed_time']
        duration = state.get('duration')
        if duration is None or duration <= 0:
            fade = 1.0
        else:
            if elapsed < fade_duration:
                f = elapsed / fade_duration
            elif elapsed > duration - fade_duration:
                f = (duration - elapsed) / fade_duration
            else:
                f = 1.0
            fade = float(np.clip(f, 0.0, 1.0))

        for eff in (state['backdrop'], state['disc']):
            eff.season = season
            eff.celestial_visibility = cv
            eff.spookyness = sp
            eff.fog_color = fc
            eff.fade = fade

    if state['count'] == -1:
        for key in ('backdrop', 'disc'):
            if key in state:
                eff = state[key]
                if eff in viewport.effects:
                    viewport.effects.remove(eff)
                eff.cleanup()
                del state[key]


# ===========================================================================
# Shared CPU resolvers — same math both effects need
# ===========================================================================

def _lerp3(a, b, t):
    return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t)


def _resolve_sky(season, fog_color, spookyness):
    """Return (zenith_rgb, horizon_rgb) shaped by season + fog/spooky bias."""
    s = season

    # Night near-black so stars read as bright pinpoints. Day saturated.
    midnight_z = (0.005, 0.008, 0.020)
    midnight_h = (0.010, 0.012, 0.030)
    sunrise_z  = (0.30, 0.40, 0.70)
    sunrise_h  = (1.00, 0.55, 0.25)
    noon_z     = (0.30, 0.65, 1.00)
    noon_h     = (1.00, 0.92, 0.70)
    sunset_z   = (0.30, 0.30, 0.60)
    sunset_h   = (1.00, 0.45, 0.20)

    if s < 0.25:
        t = s / 0.25
        z = _lerp3(midnight_z, sunrise_z, t)
        h = _lerp3(midnight_h, sunrise_h, t)
    elif s < 0.5:
        t = (s - 0.25) / 0.25
        z = _lerp3(sunrise_z, noon_z, t)
        h = _lerp3(sunrise_h, noon_h, t)
    elif s < 0.75:
        t = (s - 0.5) / 0.25
        z = _lerp3(noon_z, sunset_z, t)
        h = _lerp3(noon_h, sunset_h, t)
    else:
        t = (s - 0.75) / 0.25
        z = _lerp3(sunset_z, midnight_z, t)
        h = _lerp3(sunset_h, midnight_h, t)

    fr, fg, fb = fog_color
    red_bias = max(0.0, fr - max(fg, fb))
    bias = min(1.0, red_bias * max(spookyness, 0.0) * 3.0)
    if bias > 0.01:
        h = _lerp3(h, (fr, fg, fb), 0.5 * bias)
        z = _lerp3(z, (fr * 0.4, fg * 0.4, fb * 0.4), 0.3 * bias)

    return z, h


def _resolve_disc(season, fog_color, spookyness):
    """Return (center_ft, radius_ft, corona_ft, color_rgb)."""
    s = season
    is_day = 0.25 <= s <= 0.75

    if is_day:
        t = (s - 0.25) / 0.5
    elif s >= 0.75:
        t = (s - 0.75) / 0.5
    else:
        t = (s + 0.25) / 0.5
    arc = math.sin(math.pi * t)

    cx_ft = -16.0 + 32.0 * t
    cy_ft = 6.0 + 12.0 * arc

    if is_day:
        warm  = (1.0, 0.45, 0.15)
        white = (1.0, 0.98, 0.85)
        color = (
            warm[0] * (1 - arc) + white[0] * arc,
            warm[1] * (1 - arc) + white[1] * arc,
            warm[2] * (1 - arc) + white[2] * arc,
        )
        radius_ft = 4.0 + 0.5 * arc
        corona_ft = radius_ft * 2.2
    else:
        color = (0.85, 0.88, 0.95)
        radius_ft = 2.0
        corona_ft = radius_ft * 1.6

        r, g, b = fog_color
        red_bias = max(0.0, r - max(g, b))
        blood = min(1.0, red_bias * 2.0 * max(spookyness, 0.0) * 4.0)
        if blood > 0.01:
            color = (
                color[0] * (1 - blood) + 0.95 * blood,
                color[1] * (1 - blood) + 0.10 * blood,
                color[2] * (1 - blood) + 0.10 * blood,
            )
            radius_ft += 1.6 * blood
            corona_ft += 1.6 * blood

    return (cx_ft, cy_ft), radius_ft, corona_ft, color


# ===========================================================================
# Effect classes
# ===========================================================================

class _DesertSkyBase(ShaderEffect):
    """Common quad/buffer setup for both passes."""

    FRAGMENT_SRC = ""

    def __init__(self, viewport):
        super().__init__(viewport)
        self.season = 0.5
        self.celestial_visibility = 0.0
        self.spookyness = 0.0
        self.fog_color = (0.7, 0.7, 0.7)
        self.fade = 0.0
        self._time = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)

    def compile_shader(self):
        v = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        f = shaders.compileShader(self.FRAGMENT_SRC, GL_FRAGMENT_SHADER)
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


class DesertSkyBackdropEffect(_DesertSkyBase):
    """Sky gradient — renders before stars."""
    FRAGMENT_SRC = BACKDROP_FRAGMENT

    def __init__(self, viewport):
        super().__init__(viewport)
        # Behind stars (default 0), clouds (~6) and dunes (~7).
        self.render_priority = -1.0

    def render(self, state: Dict):
        if not self.enabled or self.fade < 0.01:
            return

        zenith, horizon = _resolve_sky(self.season, self.fog_color, self.spookyness)

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        self._fan.set_uniforms(self.shader)

        glUniform3f(glGetUniformLocation(self.shader, "u_zenith_color"), *zenith)
        glUniform3f(glGetUniformLocation(self.shader, "u_horizon_color"), *horizon)
        glUniform1f(glGetUniformLocation(self.shader, "u_sky_strength"), self.fade)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)


class DesertSunMoonEffect(_DesertSkyBase):
    """Sun/moon disc — renders in front of stars but behind clouds and dunes."""
    FRAGMENT_SRC = DISC_FRAGMENT

    def __init__(self, viewport):
        super().__init__(viewport)
        # In front of stars (0). Behind clouds (~6) and dunes (~7), so dunes
        # naturally occlude a low-sitting horizon disc.
        self.render_priority = 5.0

    def render(self, state: Dict):
        if not self.enabled or self.fade < 0.01:
            return
        intensity = self.celestial_visibility * self.fade
        if intensity < 0.01:
            return

        (cx, cy), radius, corona, color = _resolve_disc(
            self.season, self.fog_color, self.spookyness
        )

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        self._fan.set_uniforms(self.shader)

        glUniform2f(glGetUniformLocation(self.shader, "u_disc_center_ft"), cx, cy)
        glUniform1f(glGetUniformLocation(self.shader, "u_disc_radius_ft"), radius)
        glUniform1f(glGetUniformLocation(self.shader, "u_disc_corona_ft"), corona)
        glUniform3f(glGetUniformLocation(self.shader, "u_disc_color"), *color)
        glUniform1f(glGetUniformLocation(self.shader, "u_disc_intensity"), intensity)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
