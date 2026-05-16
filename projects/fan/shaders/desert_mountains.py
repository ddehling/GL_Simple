"""
Desert mountains — distant silhouette layer behind the dunes.

Pattern B (fullscreen quad, no depth writes). Renders BETWEEN the
sun/moon disc (priority 5) and the dunes (priority 7) at priority 6,
so a low sun naturally sinks behind the mountains and the dunes
naturally occlude the mountain feet.

The body is deep-dark — pure silhouette against the brighter sky.
That's the high-value contrast pattern under Fan's fixed
``brightness_limit: 0.1``: the sky already paid the energy cost, and
this shader spends near-zero output (most pixels discarded; only the
mountain band paints, and even that paints at low brightness). See
``docs/shader_contrast_playbook.md`` "Silhouette against a band".

Two scene presets (``scene_id % 2``) so the desert can show different
horizon profiles across storms. Storm obscuration fades the silhouette
out so the scene swap happens behind weather rather than as a visible
pop. ``outstate['scene_id']`` and ``outstate['storm_obscuration']`` are
published by ``shader_background_director`` — that event must also be
in the active set's background_events for rotation to occur. Without
the director, scene_id stays at its seeded value and the mountains
hold a stable preset.
"""
import math
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

uniform float u_strength;        // 0..1 master alpha (event fade)
uniform float u_obscuration;     // 0..1 storm obscuration
uniform int   u_scene_id;        // monotonic; preset = scene_id mod 2
uniform vec3  u_silhouette;      // deep-dark mountain color

// 1D value-noise primitives (input in feet)
float hash11(float n) {{ return fract(sin(n) * 43758.5453); }}
float vnoise1(float x) {{
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), u);
}}

// Two scene presets selected by ``u_scene_id mod 2``. Each profile uses
// a different combination of frequencies + phase offsets so the
// silhouette has a clearly different character.
//
//   Preset 0 — gentle rolling ridges (low-freq sine + small noise jitter).
//   Preset 1 — angular spires (sharper, taller peaks via high-freq sharpen).
//
// Both stay within the same vertical band (8..14 ft) so they sit
// consistently above the dunes (~5..7 ft) and below the sun's daytime
// arc (~14..18 ft at noon, lower at dawn/dusk).
float mountain_height(float x_ft, int scene) {{
    // Modulo 2 select; mod() on int isn't ES310 portable so use a bool
    bool preset = (scene & 1) != 0;

    float baseline = 9.0;
    if (!preset) {{
        // Rolling: low-freq base + gentler noise overlay
        float low  = sin(x_ft * 0.18) * 0.9;
        float mid  = (vnoise1(x_ft * 0.55 + 13.7) - 0.5) * 1.6;
        float fine = (vnoise1(x_ft * 1.40 + 41.3) - 0.5) * 0.6;
        return baseline + low + mid + fine;
    }} else {{
        // Angular spires: sharper, sparser, much taller peaks. Sparse
        // tall spikes (up to ~17 ft) reach into the lower half of the
        // sun's mid-arc, so the sun is visibly silhouetted by spires
        // during morning/afternoon transitions. Baseline body stays
        // at ~9 ft so most of the horizon still reads as distant low
        // mountains; only the occasional spike sticks up dramatically.
        float low  = sin(x_ft * 0.22 + 5.1) * 0.7;
        float n    = vnoise1(x_ft * 0.85 + 91.2);
        // Sharpen toward the upper tail: 0.55 threshold (slightly more
        // peaks) then steep — gives ~15% of the horizon length some
        // amount of spike. Spike height up to 7.0 ft so tallest peaks
        // hit ~17 ft (baseline 9 + low 0.7 + spike 7.0 + jitter 0.25).
        float spike_mask = smoothstep(0.55, 0.85, n);
        float spike = pow(spike_mask, 1.4) * 7.0;
        float jitter = (vnoise1(x_ft * 2.30 + 173.9) - 0.5) * 0.5;
        return baseline + low + spike + jitter;
    }}
}}

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);

    float h = mountain_height(phys.x, u_scene_id);

    // Below-mountain pixels are part of the mountain body; above are sky.
    // 0.40-ft soft band keeps the ridge from aliasing on the LED grid.
    float a = smoothstep(h + 0.40, h - 0.20, phys.y);

    // Master alpha: event fade × (1 - obscuration). During a storm
    // u_obscuration approaches 1 and the mountains vanish; afterward
    // they ease back in (with possibly a different scene_id).
    a *= u_strength * (1.0 - u_obscuration);

    if (a < 0.01) discard;
    fragColor = vec4(u_silhouette, a);
}}
"""


# ===========================================================================
# Event wrapper
# ===========================================================================

def shader_desert_mountains(state, outstate, fade_duration=4.0):
    """Background event: distant mountain silhouette layer."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DesertMountainsEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing desert_mountains: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.scene_id = int(outstate.get('scene_id', 0))
        eff.obscuration = float(outstate.get('storm_obscuration', 0.0))

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

class DesertMountainsEffect(ShaderEffect):
    """Fullscreen-quad distant mountain silhouette."""

    def __init__(self, viewport):
        super().__init__(viewport)
        # Between sun/moon disc (5) and dunes (7) so dunes occlude the
        # mountain feet and a low sun sinks behind the ridges.
        self.render_priority = 6.0
        self.scene_id = 0
        self.obscuration = 0.0
        self.strength = 0.0
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
        # No CPU-side state to advance — the shader reads scene_id and
        # obscuration directly from uniforms set in render().
        pass

    def render(self, state: Dict):
        if not self.enabled or self.strength < 0.01:
            return
        # If obscured ≥ 1, nothing would be visible anyway. Bail.
        if self.obscuration >= 0.999:
            return

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        self._fan.set_uniforms(self.shader)

        # Deep-dark cool silhouette — distinct from the warm dune tones
        # so even when both layers are visible there's hue separation,
        # not just brightness contrast. Slightly blue-violet pushes the
        # silhouette toward "atmospheric distance" rather than "ink black".
        glUniform3f(glGetUniformLocation(self.shader, "u_silhouette"),
                    0.04, 0.05, 0.10)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"),
                    float(self.strength))
        glUniform1f(glGetUniformLocation(self.shader, "u_obscuration"),
                    float(self.obscuration))
        glUniform1i(glGetUniformLocation(self.shader, "u_scene_id"),
                    int(self.scene_id))

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
