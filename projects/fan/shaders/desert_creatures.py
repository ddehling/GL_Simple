"""
Desert creatures — small things crawling on the dunes and occasional things
moving through the sky. Pattern B (fullscreen quad, no depth writes).

Four creature kinds, picked by time-of-day:
  - GROUND_LIZARD (day)   : small dark scurry along the dune ridge
  - GROUND_EYES   (night) : a pair of tiny glowing yellow eye glints
  - SKY_BIRD      (day)   : small V-shaped silhouette drifting across the sky
  - SKY_BAT       (night) : tiny fluttering silhouette zipping across

CPU manages a list of live creatures (position in physical feet, velocity,
type, lifetime, brightness fade). Each frame the list is encoded into a
uniform vec4 array and drawn by a single fullscreen-quad fragment shader
that iterates over the active creatures.

Spawn rates are modulated by `season` so day creatures appear during day
hours and night creatures at night, with a brief mix at dawn/dusk.
"""
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords

# Hard cap on creatures alive at once (matches the shader array size).
MAX_CREATURES = 32

# Type IDs (encoded in the .z slot of each creature uniform vec4).
T_GROUND_LIZARD = 0
T_GROUND_EYES   = 1
T_SKY_BIRD      = 2
T_SKY_BAT       = 3

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

#define MAX_CREATURES {MAX_CREATURES}

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

// Each creature is one vec4: (x_ft, y_ft, type_id, brightness)
//   type_id: 0=ground_lizard, 1=ground_eyes, 2=sky_bird, 3=sky_bat
//   brightness: 0..1 master alpha (handles spawn/despawn fade)
uniform vec4  u_creatures[MAX_CREATURES];
uniform int   u_active_count;
uniform float u_time;

// Render one creature contribution. Returns RGBA (straight alpha).
vec4 draw_creature(vec2 phys, vec4 c) {{
    int type = int(c.z + 0.5);
    // Brightness sign encodes facing direction for asymmetric creatures
    // (negative = facing left, positive = facing right); .a uses |w|.
    float facing = sign(c.w);
    if (facing == 0.0) facing = 1.0;
    float brightness = abs(c.w);
    if (brightness < 0.005) return vec4(0.0);
    vec2 center = c.xy;

    // Per-type rendering
    if (type == 0) {{
        // GROUND_LIZARD: head + body + tapering tail in local coords.
        // Local frame: +X = forward (direction of motion), +Y = up.
        vec2 lp = phys - center;
        lp.x *= facing;        // flip so the lizard always faces +X locally

        // Body ellipse: centered at (0.15, 0), radii (0.55, 0.13) ft
        vec2 body_d = (lp - vec2(0.15, 0.0)) / vec2(0.55, 0.13);
        float body = smoothstep(1.05, 0.85, length(body_d));

        // Head bulge: small ellipse at the front (0.65, 0), radii (0.18, 0.10)
        vec2 head_d = (lp - vec2(0.65, 0.0)) / vec2(0.18, 0.10);
        float head = smoothstep(1.05, 0.80, length(head_d));

        // Tail: thin tapering line from x=-0.40 back to x=-1.10. Width
        // narrows from 0.08 ft at the base to 0 at the tip.
        float tail_t = clamp((-0.40 - lp.x) / 0.70, 0.0, 1.0);   // 0 at base, 1 at tip
        float tail_width = mix(0.08, 0.005, tail_t);
        float tail_in_x = step(-1.10, lp.x) * step(lp.x, -0.40);
        float tail = tail_in_x * smoothstep(tail_width, tail_width * 0.4, abs(lp.y));

        float silh = max(max(body, head), tail);
        if (silh < 0.005) return vec4(0.0);
        vec3 col = vec3(0.20, 0.12, 0.06);   // dark earth-brown silhouette
        return vec4(col, silh * brightness);
    }}
    else if (type == 1) {{
        // GROUND_EYES: two tiny bright eye glints, side by side, slight twinkle
        vec2 eye_off = vec2(0.18, 0.0);
        float r_l = length(phys - (center - eye_off));
        float r_r = length(phys - (center + eye_off));
        float twinkle = 0.85 + 0.15 * sin(u_time * 6.0 + center.x * 11.0);
        float glow = smoothstep(0.18, 0.04, min(r_l, r_r));
        if (glow < 0.005) return vec4(0.0);
        vec3 col = vec3(1.0, 0.85, 0.30) * twinkle;   // warm yellow eye-shine
        return vec4(col, glow * brightness);
    }}
    else if (type == 2) {{
        // SKY_BIRD: V-silhouette. Two diagonal strokes meeting at the body.
        vec2 d = phys - center;
        // V opens upward — left wing slope -1, right wing slope +1
        float wing_l = abs(d.y - (-d.x * 0.4));
        float wing_r = abs(d.y - ( d.x * 0.4));
        float wing_extent = step(abs(d.x), 0.55);     // wingspan ~1.1 ft
        float wing = (smoothstep(0.07, 0.02, min(wing_l, wing_r))) * wing_extent;
        if (wing < 0.005) return vec4(0.0);
        vec3 col = vec3(0.10, 0.08, 0.06);  // very dark silhouette
        return vec4(col, wing * brightness);
    }}
    else {{
        // SKY_BAT: small fluttering dark spot. Vertical wobble drives "wings."
        float flutter = sin(u_time * 14.0 + center.x * 7.0) * 0.06;
        vec2 d = phys - vec2(center.x, center.y + flutter);
        d.x *= 0.7;
        float r = length(d);
        float core = smoothstep(0.32, 0.15, r);
        if (core < 0.005) return vec4(0.0);
        vec3 col = vec3(0.05, 0.05, 0.08);   // deep night silhouette
        return vec4(col, core * brightness);
    }}
}}

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);
    vec4 acc = vec4(0.0);

    // Iterate all creature slots; each is a no-op if brightness < 0.
    for (int i = 0; i < MAX_CREATURES; i++) {{
        if (i >= u_active_count) break;
        vec4 contrib = draw_creature(phys, u_creatures[i]);
        // Composite each creature using straight-alpha "over" math
        // (acc starts at (0,0,0,0); painting one creature at a time).
        float a_out = contrib.a + acc.a * (1.0 - contrib.a);
        if (a_out > 0.0) {{
            acc.rgb = (contrib.rgb * contrib.a + acc.rgb * acc.a * (1.0 - contrib.a)) / a_out;
            acc.a   = a_out;
        }}
    }}

    if (acc.a < 0.005) discard;
    fragColor = acc;        // straight alpha
}}
"""


# ---------------------------------------------------------------------------
# Event wrapper
# ---------------------------------------------------------------------------

def shader_desert_creatures(state, outstate, fade_duration=4.0):
    """Background event: spawns small creatures crawling on dunes and
    drifting through the sky, varying by time of day."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DesertCreaturesEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing desert_creatures: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.season = float(outstate.get('season', 0.5))
        eff.wind   = float(outstate.get('wind', 0.0))

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


# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------

class _Creature:
    """One live creature on the CPU side."""
    __slots__ = ('type', 'x', 'y', 'vx', 'vy', 'life', 'max_life')

    def __init__(self, type_id, x, y, vx, vy, max_life):
        self.type = type_id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = 0.0
        self.max_life = max_life


class DesertCreaturesEffect(ShaderEffect):
    """Fullscreen-quad creature painter."""

    # Physical fan dimensions (mirror FanCoords). Spawn just off the edges.
    FAN_X_HALF = 20.6
    SPAWN_X    = 21.0   # just outside the fan in feet

    def __init__(self, viewport):
        super().__init__(viewport)
        # In front of dunes (7) so creatures are visible on top of the
        # silhouette. Below post-process fog (1000).
        self.render_priority = 8.0
        self.season = 0.5
        self.wind = 0.0
        self.fade = 0.0
        self._time = 0.0
        # Wind-phase integrator that mirrors desert_dunes' integration so
        # ground creatures can track the migrating front-dune surface.
        # Both effects integrate the same `wind` value with the same
        # coefficient (6.5), so they stay in sync to floating-point drift.
        self._wind_phase = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)
        self._creatures: list[_Creature] = []
        # Spawn cooldown per kind (so we don't burst)
        self._spawn_acc = {
            T_GROUND_LIZARD: 0.0,
            T_GROUND_EYES:   0.0,
            T_SKY_BIRD:      0.0,
            T_SKY_BAT:       0.0,
        }

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

    # -- TOD shaping --------------------------------------------------------

    def _day_factor(self) -> float:
        """1.0 at noon (s=0.5), 0.0 at midnight, smooth in between."""
        return max(0.0, 1.0 - abs(self.season - 0.5) * 2.0)

    def _night_factor(self) -> float:
        return 1.0 - self._day_factor()

    # -- Dune surface tracking ---------------------------------------------
    #
    # These mirror the GLSL math in desert_dunes.py for the FRONT layer
    # only — the layer closest to the viewer, which is what the lizard
    # appears to walk on. Constants and integration MUST stay in sync with
    # that shader; if dune_dunes changes the front-layer baseline / wave /
    # warp parameters, update them here too.

    # Front-layer constants (mirror desert_dunes.py FR_*)
    _FR_BASE  = 4.4
    _FR_AMP   = 0.5
    _FR_WL    = 6.5    # mean separation; actual peaks 5..8 ft apart via warp
    _FR_WARP  = 0.75
    _FR_WFREQ = 0.30
    _FR_DRIFT_COEFF = -1.10  # matches `drift_fr = -u_wind_phase * 1.10`
                             # (negative so dunes migrate WITH wind direction)

    @staticmethod
    def _hash11(n: float) -> float:
        # Mirror GLSL `fract(sin(n) * 43758.5453)`
        return (math.sin(n) * 43758.5453) % 1.0

    @classmethod
    def _vnoise1(cls, x: float) -> float:
        i = math.floor(x)
        f = x - i
        u = f * f * (3.0 - 2.0 * f)
        return cls._hash11(i) * (1.0 - u) + cls._hash11(i + 1.0) * u

    def _front_dune_h(self, x_ft: float) -> float:
        """Front-layer dune ridge height (feet) at world x."""
        drift = self._wind_phase * self._FR_DRIFT_COEFF

        # x-warp (aperiodic spacing)
        n = self._vnoise1(x_ft * self._FR_WFREQ + drift * self._FR_WFREQ * 0.5) - 0.5
        wx = x_ft + self._FR_WARP * n * 2.0

        # Asymmetric dune wave at wx
        u = ((wx + drift) / self._FR_WL) % 1.0
        if u < 0.7:
            w = math.sin(u / 0.7 * (math.pi / 2.0))
        else:
            w = math.cos((u - 0.7) / 0.3 * (math.pi / 2.0))

        # Per-cycle amplitude jitter
        ci = math.floor((wx + drift) / self._FR_WL)
        j = 0.7 + 0.3 * self._hash11(ci * 7.13)

        # Per-dune sharpness bias (matches dune_personality .y)
        sharp = (self._hash11(ci * 23.17) - 0.5) * 0.6
        if w >= 0.0:
            w_shaped = w ** max(1e-3, 1.0 - sharp)
        else:
            w_shaped = w

        # NOTE: we omit the high-freq crest_detail term — it's small (<0.15
        # ft), and the lizard's CPU-side y-track doesn't need that level of
        # fidelity. The visible silhouette will absorb the small offset.
        return self._FR_BASE + self._FR_AMP * w_shaped * j

    # -- Spawning -----------------------------------------------------------

    def _try_spawn(self, dt: float):
        if len(self._creatures) >= MAX_CREATURES:
            return
        day = self._day_factor()
        night = self._night_factor()

        # Target rate (creatures-per-second) by kind, weighted by TOD.
        # Bird rate boosted (0.04 -> 0.12) so the day sky has 4-5 birds
        # active instead of 1-2 — populates the bright daytime sky with
        # dark silhouettes, which is the cheapest way to add brightness
        # contrast (the sky pays the energy cost, the silhouettes are
        # zero-output). See docs/shader_contrast_playbook.md "Silhouette
        # against a band" pattern.
        rates = {
            T_GROUND_LIZARD: 0.06 * day,
            T_GROUND_EYES:   0.10 * night,
            T_SKY_BIRD:      0.12 * day,
            T_SKY_BAT:       0.05 * night,
        }
        for kind, rate in rates.items():
            self._spawn_acc[kind] += dt * rate
            if self._spawn_acc[kind] >= 1.0 and len(self._creatures) < MAX_CREATURES:
                self._spawn_acc[kind] -= 1.0
                # Occasional flock spawning for birds — 25% chance a bird
                # spawn brings 2-3 wingmates with it at similar altitude
                # and close x. Singletons still dominate so the sky feels
                # alive rather than crowded.
                if kind == T_SKY_BIRD and np.random.random() < 0.25:
                    self._spawn_flock(kind)
                else:
                    self._spawn_one(kind)

    def _spawn_one(self, kind):
        # Direction: 50/50 left-to-right or right-to-left
        rtl = np.random.random() < 0.5
        x0 = self.SPAWN_X if rtl else -self.SPAWN_X
        sign = -1.0 if rtl else 1.0

        if kind == T_GROUND_LIZARD:
            # Y is set by update() to the live dune-surface height; we just
            # initialize to the surface at spawn x so the first frame isn't
            # mid-air.
            y = self._front_dune_h(x0) + 0.10
            speed = float(np.random.uniform(1.8, 3.2))  # ft/s, scurrying
            vx = sign * speed
            vy = 0.0
            life = 25.0
        elif kind == T_GROUND_EYES:
            y = self._front_dune_h(x0) + 0.04
            speed = float(np.random.uniform(0.6, 1.4))   # slow creep
            vx = sign * speed
            vy = 0.0
            life = 35.0
        elif kind == T_SKY_BIRD:
            y = float(np.random.uniform(11.0, 17.0))   # upper sky
            speed = float(np.random.uniform(1.0, 2.0))
            vx = sign * speed
            vy = float(np.random.uniform(-0.05, 0.05))
            life = 40.0
        else:  # T_SKY_BAT
            y = float(np.random.uniform(9.0, 16.0))
            speed = float(np.random.uniform(2.0, 3.5))
            vx = sign * speed
            vy = float(np.random.uniform(-0.10, 0.10))
            life = 18.0

        self._creatures.append(_Creature(kind, x0, y, vx, vy, life))

    def _spawn_flock(self, kind):
        """Spawn 2-3 creatures of the same kind in loose formation.

        All members share entry direction and approximate altitude,
        but each gets independent small jitter in y / speed so they
        don't move as a rigid clone — feels like a real flock. The
        leader spawns first at the normal entry x; followers spawn
        slightly behind, so they trail across the sky.
        """
        n_followers = int(np.random.randint(2, 4))
        # Leader determines direction + lane
        rtl = np.random.random() < 0.5
        sign = -1.0 if rtl else 1.0
        x_lead = self.SPAWN_X if rtl else -self.SPAWN_X
        y_lead = float(np.random.uniform(11.0, 17.0))
        speed_lead = float(np.random.uniform(1.0, 2.0))

        # Leader
        self._creatures.append(_Creature(
            kind, x_lead, y_lead, sign * speed_lead,
            float(np.random.uniform(-0.05, 0.05)), 40.0,
        ))
        # Followers, lagging behind by a few feet in -sign direction
        for _ in range(n_followers):
            if len(self._creatures) >= MAX_CREATURES:
                break
            x_off = float(np.random.uniform(1.0, 3.0)) * sign  # behind leader
            y_off = float(np.random.uniform(-0.6, 0.6))
            speed = speed_lead * float(np.random.uniform(0.95, 1.05))
            self._creatures.append(_Creature(
                kind, x_lead - x_off, y_lead + y_off, sign * speed,
                float(np.random.uniform(-0.05, 0.05)), 40.0,
            ))

    # -- Update + render ----------------------------------------------------

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt
        # Mirror desert_dunes' wind-phase integration (signed) so the
        # dune surface we track here matches the one being rendered.
        self._wind_phase += dt * 6.5 * self.wind

        # Advance + cull. For GROUND creatures, snap their y to the
        # current front-dune surface so they walk along the actual
        # silhouette instead of floating above or below it.
        survivors = []
        for c in self._creatures:
            c.x += c.vx * dt
            if c.type == T_GROUND_LIZARD:
                # Sit just on top of the dune ridge (small offset for body
                # half-height so the belly aligns with the surface).
                c.y = self._front_dune_h(c.x) + 0.10
            elif c.type == T_GROUND_EYES:
                # Eyes sit a touch lower than a lizard (animal in dune shadow,
                # peeking out near the ridge top).
                c.y = self._front_dune_h(c.x) + 0.04
            else:
                # Sky creatures keep their independent y motion
                c.y += c.vy * dt
            c.life += dt
            if c.life >= c.max_life:
                continue
            if abs(c.x) > self.SPAWN_X + 1.0:
                continue
            survivors.append(c)
        self._creatures = survivors
        # Spawn new
        self._try_spawn(dt)

    def render(self, state: Dict):
        if not self.enabled or self.fade < 0.01:
            return
        n = len(self._creatures)
        if n == 0:
            return

        # Pack creatures into a Nx4 float array.
        # Each row: (x_ft, y_ft, type_id, brightness*facing_sign)
        # The sign of `brightness` encodes facing direction for asymmetric
        # creatures (lizard, bat). Magnitude is the alpha; the shader uses
        # abs(w) for brightness and sign(w) for orientation.
        data = np.zeros((n, 4), dtype=np.float32)
        FADE = 1.5
        for i, c in enumerate(self._creatures):
            if c.life < FADE:
                b = c.life / FADE
            elif c.life > c.max_life - FADE:
                b = max(0.0, (c.max_life - c.life) / FADE)
            else:
                b = 1.0
            facing = -1.0 if c.vx < 0 else 1.0
            data[i, 0] = c.x
            data[i, 1] = c.y
            data[i, 2] = float(c.type)
            data[i, 3] = float(b * self.fade) * facing

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        self._fan.set_uniforms(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1i(glGetUniformLocation(self.shader, "u_active_count"), n)
        glUniform4fv(glGetUniformLocation(self.shader, "u_creatures"),
                     n, data.flatten())

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
