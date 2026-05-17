"""
Forest bioluminescence — glowing fungi / orbs scattered across the
forest floor. Pattern B (fullscreen quad, no depth writes).

A CPU-managed pool of glow points in physical fan-cartesian feet,
each with its own color (cool teal-green base, occasional warm
orange variant — different fungi species), radius, and pulse phase.
Renders at low altitude (forest-floor area) so the canopy still
dominates the upper sky.

Each glow point is a soft Gaussian falloff in physical space,
with a slow per-point pulse modulating intensity. Points fade in/out
over their lifetime so the field of glows shifts gently — never
the same arrangement twice.

Under brightness_limit=0.1 the bioluminescence is sparse-bright
(most pixels paint nothing, glow centers are saturated cool color
at high luminance) so the energy budget is concentrated on the
luminous features against an otherwise-dark forest floor — exactly
the high-contrast pattern docs/shader_contrast_playbook.md
prescribes.
"""
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords


MAX_GLOWS = 24

# Floor area in physical feet — below the canopy band (which dominates
# uv.y > 0.20 = physical r > 7.3 ft). Glows scatter through 0..7 ft
# altitude — the visible portion of the inner ring + lower fan.
ALT_MIN = 0.5
ALT_MAX = 6.5

# Horizontal range across the fan's physical width.
X_RANGE = 18.0


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

#define MAX_GLOWS {MAX_GLOWS}

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

// Each glow packed as two vec4s:
//   slot 2*i:    (cx_ft, cy_ft, radius_ft, life_fade)
//   slot 2*i+1:  (r, g, b, pulse_intensity)
uniform vec4 u_glows[2 * MAX_GLOWS];
uniform int u_active;
uniform float u_strength;       // 0..1 master alpha
uniform float u_obscuration;    // 0..1 storm obscuration (rare)

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);

    vec3 acc_rgb = vec3(0.0);
    float acc_a = 0.0;

    for (int i = 0; i < MAX_GLOWS; i++) {{
        if (i >= u_active) break;
        vec4 a = u_glows[2 * i];
        vec4 b = u_glows[2 * i + 1];
        vec2 c = a.xy;
        float r_ft = a.z;
        float life = a.w;
        vec3 col = b.rgb;
        float pulse = b.a;

        float d2 = dot(phys - c, phys - c);
        // Gaussian-ish falloff. Inside r_ft the glow is bright; falls
        // off rapidly outside. Squared distance / r² normalizes shape.
        float falloff = exp(-d2 / max(0.01, r_ft * r_ft * 0.30));
        if (falloff < 0.02) continue;

        float contrib = falloff * life * pulse;
        // Additive accumulation (glows compound brighter when they
        // overlap, like real luminous fungi clusters do).
        acc_rgb += col * contrib;
        acc_a   = max(acc_a, contrib);
    }}

    float a = acc_a * u_strength * (1.0 - u_obscuration);
    if (a < 0.01) discard;

    fragColor = vec4(acc_rgb * u_strength * (1.0 - u_obscuration), a);
}}
"""


# ===========================================================================
# Event wrapper
# ===========================================================================

def shader_forest_bioluminescence(state, outstate, density=1.0, fade_duration=4.0):
    """Background event: bioluminescent fungi/orbs on the forest floor."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(ForestBioluminescenceEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing forest_bioluminescence: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.density = float(density)
        eff.obscuration = float(outstate.get('storm_obscuration', 0.0))

        # Night-only gate. Bioluminescent fungi don't read against a
        # bright sky — and operator-tested it looked perpetual in the
        # midday states. Multiply the master strength by starryness so
        # the glow is invisible at noon (starryness 0) and full at
        # deep night (starryness 1). Cheap soft falloff in twilight.
        starryness = float(outstate.get('starryness', 0.0))
        night_gate = float(np.clip(starryness, 0.0, 1.0))

        elapsed = state['elapsed_time']
        duration = state.get('duration')
        if duration is None or duration <= 0:
            event_fade = 1.0
        else:
            if elapsed < fade_duration:
                f = elapsed / fade_duration
            elif elapsed > duration - fade_duration:
                f = (duration - elapsed) / fade_duration
            else:
                f = 1.0
            event_fade = float(np.clip(f, 0.0, 1.0))
        eff.strength = event_fade * night_gate

    if state['count'] == -1:
        if 'effect' in state:
            eff = state['effect']
            if eff in viewport.effects:
                viewport.effects.remove(eff)
            eff.cleanup()
            del state['effect']


# ===========================================================================
# Effect
# ===========================================================================

# Fungi species — base color palette. Mostly cool teal-green, with
# occasional warm-orange "lantern fungus" variants for variety.
_SPECIES = [
    # (r, g, b, weight)
    (0.10, 0.65, 0.45, 0.35),   # teal-green (common)
    (0.18, 0.85, 0.55, 0.25),   # bright lime-green
    (0.05, 0.45, 0.75, 0.20),   # deep cyan-blue
    (0.65, 0.40, 0.15, 0.10),   # warm orange-amber (rare lantern)
    (0.80, 0.30, 0.65, 0.10),   # magenta-pink (very rare)
]


class _Glow:
    __slots__ = ('x', 'y', 'vx', 'vy', 'r', 'color', 'pulse_phase',
                 'pulse_freq', 'life', 'max_life', 'fade_in', 'fade_out')

    def __init__(self, x, y, vx, vy, r, color, pulse_phase, pulse_freq,
                 max_life):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.r = float(r)
        self.color = color    # (r, g, b)
        self.pulse_phase = float(pulse_phase)
        self.pulse_freq = float(pulse_freq)
        self.life = 0.0
        self.max_life = float(max_life)
        self.fade_in = 6.0
        self.fade_out = 8.0


class ForestBioluminescenceEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # In front of forest floor stuff, behind canopy. Canopy is at 7.0;
        # using 4.0 puts glows above the forest_eyes (default 0) and any
        # ground-level shaders, but behind leaves.
        self.render_priority = 4.0
        self.density = 1.0
        self.obscuration = 0.0
        self.strength = 0.0
        self._time = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)
        self._glows: list = []
        self._spawn_acc = 0.0
        self._upload = np.zeros((2 * MAX_GLOWS, 4), dtype=np.float32)

        # Pre-build the species cumulative-weight table for fast picks.
        weights = np.array([s[3] for s in _SPECIES], dtype=np.float32)
        self._species_p = weights / weights.sum()

        # Initial population — seed with several glows so the scene
        # isn't empty on the first frame.
        for _ in range(MAX_GLOWS // 2):
            self._spawn_one(initial=True)

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

    def _spawn_one(self, initial: bool = False):
        idx = int(np.random.choice(len(_SPECIES), p=self._species_p))
        col = (_SPECIES[idx][0], _SPECIES[idx][1], _SPECIES[idx][2])
        x = float(np.random.uniform(-X_RANGE * 0.5, X_RANGE * 0.5))
        y = float(np.random.uniform(ALT_MIN, ALT_MAX))
        r = float(np.random.uniform(0.45, 0.95))
        # Slow drift so glows don't stay pinned to the same pixel
        # cluster — the eye reads them as ambient ground luminance
        # that moves rather than blobs turning on/off in fixed spots.
        # 0.05..0.25 ft/s, mostly horizontal with a tiny vertical
        # component (some fungi drift on a breeze, others stay
        # mostly put on the substrate).
        vx = float(np.random.uniform(-0.20, 0.20))
        vy = float(np.random.uniform(-0.06, 0.06))
        # Pulse damped: lower frequencies and the alpha modulation
        # itself in render() has a smaller swing (0.85..1.0 instead
        # of 0.55..1.0) — looks like soft luminance breathing rather
        # than visible on/off blinking.
        pulse_phase = float(np.random.uniform(0.0, 6.2832))
        pulse_freq = float(np.random.uniform(0.08, 0.22))
        # Longer lifetimes so individual glows don't churn rapidly.
        max_life = float(np.random.uniform(25.0, 55.0))
        g = _Glow(x, y, vx, vy, r, col, pulse_phase, pulse_freq, max_life)
        if initial:
            g.life = float(np.random.uniform(0.0, max_life * 0.5))
        self._glows.append(g)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt

        # Age, drift, prune. Glows wander slowly across the forest
        # floor so the field of light looks like ambient luminance
        # that moves, not blobs blinking on/off in fixed positions.
        alive = []
        for g in self._glows:
            g.life += dt
            g.x += g.vx * dt
            g.y += g.vy * dt
            # Reflect off the altitude bounds so glows stay in the
            # floor band rather than drifting off.
            if g.y < ALT_MIN and g.vy < 0.0:
                g.vy = -g.vy
            elif g.y > ALT_MAX and g.vy > 0.0:
                g.vy = -g.vy
            # Lose glows that drift well off-fan horizontally.
            if abs(g.x) > X_RANGE * 0.7:
                continue
            if g.life < g.max_life:
                alive.append(g)
        self._glows = alive

        # Refill based on density.
        target = int(round(MAX_GLOWS * max(0.0, min(1.0, self.density))))
        # Slowly approach target (spawn rate proportional to deficit)
        deficit = target - len(self._glows)
        if deficit > 0:
            self._spawn_acc += dt * (0.6 + 0.1 * deficit)
            while self._spawn_acc >= 1.0 and len(self._glows) < MAX_GLOWS:
                self._spawn_acc -= 1.0
                self._spawn_one()

    def render(self, state: Dict):
        if not self.enabled or self.strength < 0.01:
            return
        if self.obscuration >= 0.99:
            return
        if not self._glows:
            return

        n = min(len(self._glows), MAX_GLOWS)
        for i in range(n):
            g = self._glows[i]
            # Life fade — ramp in at start, ramp out at end.
            if g.life < g.fade_in:
                life_a = g.life / g.fade_in
            elif g.life > g.max_life - g.fade_out:
                life_a = max(0.0, (g.max_life - g.life) / g.fade_out)
            else:
                life_a = 1.0
            life_a = max(0.0, min(1.0, life_a))

            # Pulse: damped to 0.85..1.0 swing — soft luminance
            # breathing rather than visible on/off blinking. Previous
            # 0.55..1.0 swing read as deliberate cyclic flashing.
            pulse = 0.925 + 0.075 * math.sin(self._time * g.pulse_freq
                                             * 6.2832 + g.pulse_phase)

            self._upload[2 * i, 0] = g.x
            self._upload[2 * i, 1] = g.y
            self._upload[2 * i, 2] = g.r
            self._upload[2 * i, 3] = life_a
            self._upload[2 * i + 1, 0] = g.color[0]
            self._upload[2 * i + 1, 1] = g.color[1]
            self._upload[2 * i + 1, 2] = g.color[2]
            self._upload[2 * i + 1, 3] = pulse

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        self._fan.set_uniforms(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "u_active"), n)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"),
                    float(self.strength))
        glUniform1f(glGetUniformLocation(self.shader, "u_obscuration"),
                    float(self.obscuration))
        glUniform4fv(glGetUniformLocation(self.shader, "u_glows"),
                     2 * MAX_GLOWS, self._upload.flatten())

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
