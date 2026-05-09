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
from renderer.effects.base import ShaderEffect


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
// Integrated splash event phase: dt * (0.6 + u_rain * 1.2) accumulated
// CPU-side. Replaces `u_time * event_rate` so splash event_idx / splash_t
// don't wind backward when u_rain interpolates during state transitions.
uniform float u_splash_phase;

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

    // IMPORTANT: spawn_period, fall_speed, and trail_len are CONSTANTS — they
    // must NOT depend on u_rain. They parameterize drip motion via the
    // accumulated u_time, and any change to them would shift drip_age
    // non-monotonically (since d(u_time/period)/dt = 1/period − u_time·dperiod/dt
    // / period² can go negative for large u_time). That manifests as drips
    // drifting upward during weather state transitions when u_rain is being
    // interpolated. Convey rain intensity via the per-drip visibility gate
    // below instead.
    const float spawn_period = 0.7;
    const float fall_speed   = 0.32;
    const float trail_len    = 0.10;

    // Drips are events: each drip has a start time, falls down, then dies.
    float drip_alpha = 0.0;
    for (int k = 0; k < 3; ++k) {
        float fk = float(k);
        float local_t = u_time + fk * spawn_period * 0.33 + hash(vec2(col, fk)) * spawn_period;
        float drip_idx = floor(local_t / spawn_period);
        float drip_age = fract(local_t / spawn_period);  // 0..1

        // Per-drip visibility gate: at low rain, only some drips appear.
        // This is the ONLY place u_rain modulates drip cadence.
        float visibility_thresh = 1.0 - clamp(u_rain * 1.4, 0.0, 1.0);
        if (hash(vec2(col + 0.31, drip_idx)) < visibility_thresh) continue;

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
        // event phase pre-integrated CPU-side as u_splash_phase to avoid
        // the same time-warp the drip code is protected against.
        float event_idx = floor(u_splash_phase + scell * 0.13);
        float s_seed = hash(vec2(scell, event_idx));
        float splash_show = step(1.0 - clamp(u_rain * 0.45, 0.0, 0.6), s_seed);
        // Splash lifetime is brief (0..1 of cycle, but only the first 30%
        // looks like a hit; rest is invisible). No expanding ring → no
        // upward motion.
        float splash_t = fract(u_splash_phase + scell * 0.13);
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
        self.render_priority = 9.0  # Front rain detail
        self._time = 0.0
        self._splash_phase = 0.0   # integrated dt * (0.6 + rain * 1.2)
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
        self._splash_phase += dt * (0.6 + self.rain * 1.2)

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_rain"), self.rain)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), self.wind)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glUniform1f(glGetUniformLocation(self.shader, "u_splash_phase"), self._splash_phase)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
