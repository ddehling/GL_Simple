"""Rain-on-leaves effect — droplets running down leaf silhouettes.

Differs from generic rain.py: instead of straight falling streaks,
droplets accumulate at leaf edges, swell, then run down in short
trails. Best stacked behind/under the regular `rain` shader for
forest-storm states.

Drives:
  rain         -> droplet density and run-off rate
  wind         -> sideways drift on droplets
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_rain_on_leaves(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(RainOnLeavesEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized rain_on_leaves for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize rain_on_leaves: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.rain = float(outstate.get('rain', 0.0))
        eff.wind = float(outstate.get('wind', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 4.0
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
    // Mid-front; sits in front of spore_drift (priority 3.5).
    gl_Position = vec4(position, 0.55, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_rain;
uniform float u_wind;
uniform float u_fade;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    if (u_rain < 0.02 || u_fade < 0.005) discard;
    vec2 uv = v_uv;

    // Drip streams at periodic angular positions. Each column has a
    // unique phase + speed so they don't all run at once.
    float n_columns = 24.0;
    float col = floor(uv.x * n_columns);
    float col_phase = uv.x * n_columns - col;

    // Per-column random offsets (deterministic on column index).
    float seed = col * 0.137;
    float spawn_period = mix(2.0, 0.5, clamp(u_rain, 0.0, 1.0));
    float fall_speed   = mix(0.18, 0.45, clamp(u_rain, 0.0, 1.0));
    float trail_len    = mix(0.04, 0.15, clamp(u_rain, 0.0, 1.0));

    // Drips are events: each drip has a start time, falls down, then dies.
    float drip_alpha = 0.0;
    for (int k = 0; k < 3; ++k) {
        float fk = float(k);
        float local_t = u_time + fk * spawn_period * 0.33 + hash(vec2(col, fk)) * spawn_period;
        float drip_idx = floor(local_t / spawn_period);
        float drip_age = fract(local_t / spawn_period);  // 0..1

        // Pick a starting y near the canopy (top of FBO = outer ring of fan).
        // Drips fall toward the floor (uv.y decreasing toward 0).
        float start_y = 0.40 + 0.55 * hash(vec2(col, drip_idx + fk * 7.0));
        float y_pos   = start_y - drip_age * fall_speed * 1.6;
        if (y_pos < -0.05) continue;

        // Trail extends ABOVE the head (uv.y > y_pos) since the head moved down.
        float dy = uv.y - y_pos;
        float along_trail = dy / trail_len;  // positive above the head
        if (along_trail < 0.0 || along_trail > 1.0) continue;

        // Sideways offset from wind, scaled by trail position.
        float wind_offset = u_wind * 0.04 * (1.0 - along_trail);
        float dx = abs(uv.x - (col + 0.5) / n_columns - wind_offset);
        // Wrap dx for continuity.
        dx = min(dx, 1.0 - dx);

        float trail_width = 0.004 + 0.005 * along_trail;
        float across = 1.0 - smoothstep(0.0, trail_width, dx);

        // Brightness fades along trail; head is brightest.
        float fade_along = pow(1.0 - along_trail, 1.8);
        drip_alpha += across * fade_along;
    }

    // Bead at leaf edges: tiny static droplets that swell with rain.
    float bead_seed = hash(vec2(floor(uv.x * 80.0), floor(uv.y * 60.0)));
    float bead_pulse = 0.5 + 0.5 * sin(u_time * 1.5 + bead_seed * 31.4);
    float bead_alpha = step(0.985, bead_seed) * bead_pulse * u_rain * 0.6;

    // Splash hits at the floor — brief horizontal flashes (NOT vertical
    // expanding rings, which read as rising rain). Each splash is just
    // a short-lived horizontal smear at the very bottom of the band.
    float splash_alpha = 0.0;
    if (uv.y < 0.05 && u_rain > 0.05) {
        float scols = 24.0;
        float sx = uv.x * scols;
        float scell = floor(sx);
        float sfrac = fract(sx) - 0.5;  // -0.5..0.5
        float event_rate = 0.6 + u_rain * 1.2;
        float event_idx = floor(u_time * event_rate + scell * 0.13);
        float s_seed = hash(vec2(scell, event_idx));
        float splash_show = step(1.0 - clamp(u_rain * 0.45, 0.0, 0.6), s_seed);
        // Splash lifetime is brief (0..1 of cycle, but only the first 30%
        // looks like a hit; rest is invisible). No expanding ring → no
        // upward motion.
        float splash_t = fract(u_time * event_rate + scell * 0.13);
        float life = step(splash_t, 0.30) * (1.0 - splash_t / 0.30);
        // Horizontal smear: bright at cell center, fades sideways. No
        // vertical motion component at all.
        float horiz = exp(-pow(sfrac / 0.18, 2.0));
        // Slight vertical taper (brightest at floor, fading up tiny bit).
        float vert = 1.0 - smoothstep(0.0, 0.05, uv.y);
        splash_alpha = splash_show * life * horiz * vert * 0.55 * u_rain;
    }

    float total = clamp(drip_alpha + bead_alpha + splash_alpha, 0.0, 1.0) * u_fade;
    if (total < 0.005) discard;

    // Cool blue-white droplet color.
    vec3 col_rgb = vec3(0.55, 0.72, 0.85);
    fragColor = vec4(col_rgb * total, total * 0.85);
}
"""


class RainOnLeavesEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 3.5
        self._time = 0.0
        self.rain = 0.3
        self.wind = 0.0
        self.fade = 0.0

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(_FRAG, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
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
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_rain"), self.rain)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
