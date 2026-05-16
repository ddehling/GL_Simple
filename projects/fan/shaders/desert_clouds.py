"""
Desert clouds — fan-aware cloud layer that renders in physical fan-space.

Pattern B (fullscreen quad, no depth writes). The shared
``renderer.effects.clouds`` is buffer-UV-native and produces
radial-looking smears on a fan layout; this shader reasons in
physical (x, y) feet via ``fan_uv_to_physical`` so cloud blobs are
correctly oriented as horizontal puffs in the viewer's frame, not
along the fan's radial direction.

A small fixed pool of cloud blobs (N_CLOUDS) drifts horizontally
with wind. Each blob has its own center, ellipse-radii (wider than
tall — clouds are stretched horizontally), opacity, and internal
noise seed. Soft elliptical Gaussian falloff at each blob keeps
edges fluffy without lookups.

Renders at priority 6.5 — between mountains (6) and dunes (7) so
the dune ridge always silhouettes against any cloud behind it.
Fades on storm_obscuration like the rest of the desert background
stack (during a sandstorm the sky reads as a wash, no individual
clouds visible).
"""
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords


# Pool size — also hard-coded in the fragment shader's array length.
N_CLOUDS = 8

# Cloud altitude band (physical feet). Sits in the upper sky, well
# above the dune ridges (~6 ft) but below the noon sun (~18 ft).
ALT_MIN = 11.0
ALT_MAX = 17.0

# Horizontal extent — clouds spawn somewhere across the fan's width
# and drift off the side, recycling from the opposite edge.
X_MIN = -22.0
X_MAX =  22.0
X_RANGE = X_MAX - X_MIN

# Per-cloud size (ellipse semi-axes in feet). Clouds are wider than
# they are tall — classic horizontal-puff shape.
HALF_W_MIN = 2.5
HALF_W_MAX = 5.0
HALF_H_MIN = 0.8
HALF_H_MAX = 1.6

# Base drift speed multiplier. Wind in outstate is signed in the
# ~[-2, +2] range; ``DRIFT_SCALE`` converts it to feet/second.
DRIFT_SCALE = 1.5


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

#define N_CLOUDS {N_CLOUDS}

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

// Each cloud packed as two vec4s in a single vec4 array of length
// 2*N_CLOUDS. Index 2*i: (cx_ft, cy_ft, half_w_ft, half_h_ft).
// Index 2*i+1: (opacity, seed, _, _). Avoids the alignment quirks of
// struct arrays in ES310.
uniform vec4 u_clouds[2 * N_CLOUDS];

uniform vec3  u_color;          // base cloud tint (gray-white biased)
uniform float u_strength;       // 0..1 master alpha (event fade)
uniform float u_obscuration;    // 0..1 storm obscuration
uniform float u_time;           // for very slow internal cloud animation

// Hash + noise for surface mottling so clouds aren't perfectly smooth
float hash21(vec2 p) {{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}}
float vnoise2(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1, 0));
    float c = hash21(i + vec2(0, 1));
    float d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}}
float fbm2(vec2 p) {{
    float a = 0.5, s = 0.0;
    for (int o = 0; o < 3; o++) {{
        s += a * vnoise2(p);
        p *= 2.1;
        a *= 0.5;
    }}
    return s;
}}

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);

    // Accumulate cloud coverage. Each blob contributes a Gaussian-ish
    // soft-ellipse falloff in physical space. Mottling perturbs the
    // edge so they don't read as smooth ovals.
    float coverage = 0.0;
    for (int i = 0; i < N_CLOUDS; i++) {{
        vec4 a = u_clouds[2 * i];
        vec4 b = u_clouds[2 * i + 1];
        vec2 c = a.xy;
        float hw = a.z;
        float hh = a.w;
        float op = b.x;
        float seed = b.y;

        if (op < 0.005) continue;

        // Normalized distance from blob center in ellipse space
        vec2 d = (phys - c) / vec2(max(hw, 0.1), max(hh, 0.1));
        float r2 = dot(d, d);
        if (r2 >= 1.2) continue;  // hard culling outside blob

        // Gaussian falloff with a perturbation from FBM noise so edges
        // are fluffy rather than perfectly elliptical. Internal time
        // drift makes the puffy edge breathe slowly.
        float mottle = fbm2(d * 1.8 + vec2(seed * 13.7, u_time * 0.04));
        float r2_perturbed = r2 + (mottle - 0.5) * 0.35;
        float falloff = exp(-2.5 * max(0.0, r2_perturbed));

        coverage = max(coverage, falloff * op);
    }}

    // Master alpha: event fade × (1 - obscuration). Same pattern as
    // mountains so the cloud layer fades during sandstorms.
    float a = coverage * u_strength * (1.0 - u_obscuration);
    if (a < 0.01) discard;

    fragColor = vec4(u_color, a);
}}
"""


# ===========================================================================
# Event wrapper
# ===========================================================================

def shader_desert_clouds(state, outstate, density=1.0, fade_duration=4.0):
    """Background event: fan-aware cloud layer for the desert.

    ``density`` (0..1+) scales how many of the N_CLOUDS slots are
    actually visible — at 0.0 the sky is clear; at 1.0 all slots
    are populated.
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
            effect = viewport.add_effect(DesertCloudsEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing desert_clouds: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.density = float(density)
        eff.wind = float(outstate.get('wind', 0.0))
        eff.obscuration = float(outstate.get('storm_obscuration', 0.0))
        fc = outstate.get('fog_color', (0.85, 0.85, 0.90))
        eff.fog_color = (float(fc[0]), float(fc[1]), float(fc[2]))

        elapsed = state['elapsed_time']
        duration = state.get('duration')
        if duration is None or duration <= 0:
            eff.strength = 1.0
        else:
            if elapsed < fade_duration:
                f = elapsed / fade_duration
            elif elapsed > duration - fade_duration:
                f = (duration - elapsed) / fade_duration
            else:
                f = 1.0
            eff.strength = float(np.clip(f, 0.0, 1.0))

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

class _Cloud:
    """One cloud blob on the CPU side."""
    __slots__ = ('x', 'y', 'half_w', 'half_h', 'seed')

    def __init__(self, x, y, half_w, half_h, seed):
        self.x = float(x)
        self.y = float(y)
        self.half_w = float(half_w)
        self.half_h = float(half_h)
        self.seed = float(seed)


class DesertCloudsEffect(ShaderEffect):
    """Fan-aware cloud layer effect."""

    def __init__(self, viewport):
        super().__init__(viewport)
        # Behind dunes (7) so dunes silhouette over clouds. Above
        # mountains (6) so clouds can drift in front of the distant
        # mountain ridges — feels more like sky depth.
        self.render_priority = 6.5
        self.density = 1.0
        self.wind = 0.0
        self.obscuration = 0.0
        self.strength = 0.0
        self.fog_color = (0.85, 0.85, 0.90)
        self._time = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)
        # Spawn cloud pool with random positions/sizes
        self._clouds = []
        for _ in range(N_CLOUDS):
            self._clouds.append(_Cloud(
                x=float(np.random.uniform(X_MIN, X_MAX)),
                y=float(np.random.uniform(ALT_MIN, ALT_MAX)),
                half_w=float(np.random.uniform(HALF_W_MIN, HALF_W_MAX)),
                half_h=float(np.random.uniform(HALF_H_MIN, HALF_H_MAX)),
                seed=float(np.random.uniform(0.0, 100.0)),
            ))
        # Reusable upload buffer for the 2*N_CLOUDS vec4 array
        self._upload = np.zeros((2 * N_CLOUDS, 4), dtype=np.float32)

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
        # Drift clouds horizontally with wind. Wind is signed so
        # direction reverses across the day cycle. Each cloud drifts
        # at the same speed (no per-cloud variation needed — the
        # noise mottling makes them feel individual visually).
        vx = self.wind * DRIFT_SCALE
        for c in self._clouds:
            c.x += vx * dt
            # Recycle off-screen clouds to the opposite edge with a
            # fresh size and altitude.
            if c.x > X_MAX + c.half_w:
                c.x = X_MIN - c.half_w
                c.y = float(np.random.uniform(ALT_MIN, ALT_MAX))
                c.half_w = float(np.random.uniform(HALF_W_MIN, HALF_W_MAX))
                c.half_h = float(np.random.uniform(HALF_H_MIN, HALF_H_MAX))
                c.seed = float(np.random.uniform(0.0, 100.0))
            elif c.x < X_MIN - c.half_w:
                c.x = X_MAX + c.half_w
                c.y = float(np.random.uniform(ALT_MIN, ALT_MAX))
                c.half_w = float(np.random.uniform(HALF_W_MIN, HALF_W_MAX))
                c.half_h = float(np.random.uniform(HALF_H_MIN, HALF_H_MAX))
                c.seed = float(np.random.uniform(0.0, 100.0))

    def render(self, state: Dict):
        if not self.enabled or self.strength < 0.01:
            return
        if self.obscuration >= 0.999:
            return

        # Pack cloud array. Opacity uses density to pick which slots
        # are active — first ``ceil(density * N_CLOUDS)`` slots get
        # opacity 1, the rest 0. Fractional density bleeds one slot.
        active_f = max(0.0, min(1.0, self.density)) * N_CLOUDS
        for i, c in enumerate(self._clouds):
            self._upload[2 * i, 0] = c.x
            self._upload[2 * i, 1] = c.y
            self._upload[2 * i, 2] = c.half_w
            self._upload[2 * i, 3] = c.half_h
            if i + 1 <= active_f:
                op = 1.0
            elif i < active_f:
                op = active_f - i
            else:
                op = 0.0
            self._upload[2 * i + 1, 0] = op
            self._upload[2 * i + 1, 1] = c.seed
            self._upload[2 * i + 1, 2] = 0.0
            self._upload[2 * i + 1, 3] = 0.0

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        self._fan.set_uniforms(self.shader)

        # Cloud tint — gray-white biased toward the active fog color
        # so clouds match the atmospheric palette. fog_color comes
        # in via outstate (which projects can override per weather
        # state); we damp it toward gray so clouds stay readable
        # against any sky color.
        fr, fg, fb = self.fog_color
        # Mix fog color halfway with neutral gray-white
        col = (
            0.50 + 0.50 * fr,
            0.50 + 0.50 * fg,
            0.50 + 0.55 * fb,
        )
        glUniform3f(glGetUniformLocation(self.shader, "u_color"), *col)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"),
                    float(self.strength))
        glUniform1f(glGetUniformLocation(self.shader, "u_obscuration"),
                    float(self.obscuration))
        glUniform1f(glGetUniformLocation(self.shader, "u_time"),
                    float(self._time))
        glUniform4fv(
            glGetUniformLocation(self.shader, "u_clouds"),
            2 * N_CLOUDS,
            self._upload.flatten(),
        )

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
