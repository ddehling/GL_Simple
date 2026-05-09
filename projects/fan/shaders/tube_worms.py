"""Giant tube worms (Riftia pachyptila) — colonies clustered at the
bases of hydrothermal vent smokers.

White calcareous tubes with crimson feathery plumes at the top. The
plume retracts/extends slowly ("breathing"), and the colony sways
gently with the bottom current. Tubes are placed at the same six
angular positions as the smoker chimneys so the worms read as
ecologically attached to the vent structure.

Pattern B atmospheric layer; renders just in front of the smoker
plumes so the worm tubes punch through the haze. Gated by
`vent_activity` so it only appears in OCEAN_HYDROTHERMAL_VENT.

Drives:
  vent_activity  -> visibility / overall opacity gate
  wave_speed     -> sway rate (integrated as u_sway_phase)
  marine_life_activity -> plume animation amplitude (subtle)
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect


def shader_tube_worms(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(TubeWormsEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized tube_worms for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize tube_worms: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.vent_activity = float(outstate.get('vent_activity', 0.0))
        eff.wave_speed = float(outstate.get('wave_speed', 0.0))
        eff.life = float(outstate.get('marine_life_activity', 0.0))

        elapsed = state['elapsed_time']
        total = state.get('duration', 60)
        fade = 5.0
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
    // Just in front of the smoker plumes (priority handles ordering;
    // depth is disabled in render() since this is Pattern B).
    gl_Position = vec4(position, 0.78, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_vent_activity;
uniform float u_wave_speed;
uniform float u_life;
uniform float u_fade;
// CPU-integrated wave phase — see docs/shader_info.txt; replaces the
// `u_time * u_wave_speed` anti-pattern so sway never reverses.
uniform float u_sway_phase;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float wrap_dx(float a, float b) {
    float d = abs(a - b);
    return min(d, 1.0 - d);
}

void main() {
    if (u_vent_activity < 0.005 || u_fade < 0.005) discard;
    vec2 uv = v_uv;

    // Tube worms only live in the low band — colonies grow on the seafloor
    // around the chimney bases. Tallest plume tip is ~0.39 of UV height
    // (tallest worm_h up to 0.275 + plume reach up to ~0.115).
    if (uv.y > 0.42) discard;

    // Five cluster positions, each in the GAP between two smoker chimneys
    // (smoker chimneys live at 0.05, 0.19, 0.36, 0.57, 0.74, 0.93). The
    // worms therefore don't share angular position with the mineral
    // plumes — they grow on the seafloor between vents, like real Riftia
    // colonies. Spacing is intentionally irregular (0.13 – 0.21 between
    // clusters) so the row doesn't read as a regular grid.
    float cluster_u[5];
    cluster_u[0] = 0.12;   // between chimneys 0 and 1
    cluster_u[1] = 0.27;   // between 1 and 2
    cluster_u[2] = 0.46;   // between 2 and 3
    cluster_u[3] = 0.66;   // between 3 and 4
    cluster_u[4] = 0.84;   // between 4 and 5

    float tube_mask = 0.0;
    float plume_mask = 0.0;
    float plume_shade = 0.0;   // for inner shadow on the plume

    // Outer loop: clusters. Inner loop: 3 worms per cluster, jittered
    // around the cluster center.
    for (int c = 0; c < 5; ++c) {
        float fc = float(c);
        float cu = cluster_u[c];
        float cseed = hash(vec2(fc, 1.0));

        // Three thicker worms per cluster (was five thin sticks).
        for (int i = 0; i < 3; ++i) {
            float fi = float(i);
            float wseed = hash(vec2(fc * 11.0 + fi, 7.0));

            // Per-worm horizontal offset within the cluster — tighter
            // than before since there are fewer worms to space out.
            float x_offset = (hash(vec2(fc, fi + 10.0)) - 0.5) * 0.030;

            // Per-worm height & width. Heights are STAGGERED by loop
            // index so each cluster has one short, one medium, and one
            // tall worm — that way the red plumes sit at three distinct
            // y-levels, and the tubes of taller worms pass through the
            // gaps between shorter worms' plumes. Combined with the
            // plume-mask carving below, tubes always read as visible
            // white columns "between" the red plume areas.
            float h_min, h_max;
            if (i == 0)      { h_min = 0.060; h_max = 0.105; }   // short
            else if (i == 1) { h_min = 0.130; h_max = 0.180; }   // medium
            else             { h_min = 0.220; h_max = 0.275; }   // tall
            float worm_h = h_min + (h_max - h_min) * hash(vec2(fc, fi + 20.0));
            // Worms are ~3× wider than the original so they read as
            // substantial calcareous tubes, not pencil lines.
            float worm_w = 0.014 + 0.010 * hash(vec2(fc, fi + 30.0));   // 0.014..0.024

            // Cantilever bend: tube anchored at base, swings at the top.
            // Bend phase has a u_time baseline (constant frequency, always
            // visibly moving even at low wave_speed) plus a small wave-
            // speed-modulated contribution. This is the fix for the worms
            // appearing static — u_sway_phase alone advances ~0.06 rad/sec
            // at default wave_speed=0.1, which is one cycle per 100 s.
            // Two octaves give organic, less metronome-y motion. Bend
            // amount scales with (y/h)^2 so the base stays still and the
            // tip swings the most. Independent per-worm phase so the
            // colony doesn't move as one block.
            float bend_phase = u_time * 1.20
                             + u_sway_phase * 0.55
                             + wseed * 6.28;
            float bend_amp_primary   = 0.024 + 0.012 * hash(vec2(fc, fi + 50.0));
            float bend_amp_secondary = 0.010;
            float y_norm = clamp(uv.y / max(worm_h, 1e-4), 0.0, 1.0);
            float bend_along = y_norm * y_norm;
            float bend_at_y = (sin(bend_phase) * bend_amp_primary
                             + sin(bend_phase * 1.7 + wseed * 3.1) * bend_amp_secondary)
                            * bend_along;
            // Tip-of-tube position used to anchor the plume.
            float bend_at_top = (sin(bend_phase) * bend_amp_primary
                               + sin(bend_phase * 1.7 + wseed * 3.1) * bend_amp_secondary);

            float worm_x_at_y = cu + x_offset + bend_at_y;
            float dx = wrap_dx(uv.x, worm_x_at_y);

            // Tube body: bent white tube from y=0 up to worm_h.
            if (uv.y < worm_h) {
                float across = 1.0 - smoothstep(worm_w, worm_w * 1.7, dx);
                // Darker shaded edge to give the tube some volume.
                float center_glow = 1.0 - smoothstep(0.0, worm_w * 1.2, dx);
                // Subtle vertical banding (calcareous texture).
                float band = 0.92 + 0.08 * sin(uv.y * 220.0 + wseed * 5.7);
                tube_mask = max(tube_mask, across * band);
                plume_shade = max(plume_shade, center_glow);
            }

            // Plume at the top: bushy red feathery cluster. Sits on top
            // of the bent tube, with its own slow wobble independent of
            // the tube sway, plus the breathing animation. All sin terms
            // use constant frequency — safe per shader_info.txt.
            float plume_dy = uv.y - worm_h;
            if (plume_dy > -0.006 && plume_dy < 0.170) {
                float breathe = 0.55 + 0.45 *
                                sin(u_time * 0.50 + wseed * 11.3);
                // Independent plume waving (separate from bend phase).
                float plume_wobble = sin(u_time * 0.85 + wseed * 17.9) * 0.010;
                // Marine-life-activity nudges plume size on top of the
                // baseline (only amplitude, not phase — safe).
                float life_boost = 0.85 + 0.30 * u_life;
                // Plumes scaled up to match the wider tubes (was 0.0140).
                float plume_w = 0.030 * (0.65 + 0.35 * breathe) * life_boost;
                // 2x taller plumes — was 0.048, now 0.096 baseline. The
                // red gill cluster is the visual focus of a tube worm,
                // so it should dominate vertically over the tube.
                float plume_h = 0.096 * (0.70 + 0.30 * breathe) * life_boost;

                // Plume anchored at tube tip (full bend) plus its own wobble.
                float plume_x_world = cu + x_offset + bend_at_top + plume_wobble;
                float pdx = wrap_dx(uv.x, plume_x_world);
                float pdy = plume_dy + 0.004;  // shift so center sits above tube
                // Half-ellipse, only above the tube.
                float ex = pdx / max(plume_w, 1e-4);
                float ey = pdy / max(plume_h, 1e-4);
                float r2 = ex * ex + ey * ey;
                float envelope = 1.0 - smoothstep(0.55, 1.05, r2);

                // Bushy / feathery noise so the plume looks like gill
                // filaments rather than a smooth blob. Slow scroll on y
                // simulates the filaments rippling. Constant-rate time
                // term (no rate-coupled uniform) — safe.
                float bushy = vnoise(vec2(uv.x * 130.0 + wseed * 9.0,
                                          uv.y * 90.0 + wseed * 13.0
                                          + u_time * 0.45));
                bushy = smoothstep(0.30, 0.78, bushy);

                float p = envelope * (0.45 + 0.55 * bushy);
                if (pdy < 0.0) p *= 0.6;  // tuck into top of tube

                plume_mask = max(plume_mask, p);
            }
        }
    }

    // Carve plumes out of any pixel that is part of a tube. This keeps
    // the white tubes visually "between" the red plumes — taller worms'
    // tubes pass through the gaps between shorter worms' plumes without
    // the red painting over the white. The 1.05 multiplier slightly
    // widens the carve so the plume edge doesn't sliver into the tube.
    plume_mask *= max(0.0, 1.0 - tube_mask * 1.05);

    // Composite: tubes are white, plumes are crimson.
    vec3 tube_col = vec3(0.92, 0.88, 0.80);              // cream-white
    vec3 tube_shadow = vec3(0.55, 0.50, 0.45);
    vec3 tube_final = mix(tube_shadow, tube_col, plume_shade);

    vec3 plume_deep = vec3(0.55, 0.05, 0.08);           // dark base of gills
    vec3 plume_bright = vec3(0.90, 0.18, 0.20);         // bright tip
    // Vary plume color along height — bright at top, deeper at base.
    // Range now spans the staggered worm heights (tube tops 0.06-0.275,
    // plume tips reach up to ~0.39).
    float plume_height_factor = clamp((uv.y - 0.06) / 0.34, 0.0, 1.0);
    vec3 plume_col = mix(plume_deep, plume_bright, plume_height_factor);
    // Mix in a touch of orange where intensity is highest (highlights).
    plume_col = mix(plume_col, vec3(1.00, 0.45, 0.25), plume_mask * 0.18);

    // Final color: paint tubes first, then plume on top.
    vec3 col_final = tube_final;
    col_final = mix(col_final, plume_col, plume_mask);

    // STRAIGHT alpha (not pre-multiplied). Combined alpha favors the
    // structures; plume is solid where it covers the tube top.
    float alpha = clamp(tube_mask + plume_mask, 0.0, 0.95);
    alpha *= u_vent_activity * u_fade;
    if (alpha < 0.01) discard;
    fragColor = vec4(col_final, alpha);
}
"""


class TubeWormsEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # In front of the smoker plumes (3.0) so the bright crimson plumes
        # punch through the haze.
        self.render_priority = 3.5
        self._time = 0.0
        self._sway_phase = 0.0     # integrated dt * wave_speed
        self.vent_activity = 0.0
        self.wave_speed = 0.0
        self.life = 0.0
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
        # Floor wave so the colony has readable sway even in calm water.
        effective_wave = max(self.wave_speed, 0.06)
        self._sway_phase += dt * effective_wave

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return
        # Pattern B fullscreen quad — disable depth, restore on exit.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_vent_activity"), self.vent_activity)
        glUniform1f(glGetUniformLocation(self.shader, "u_wave_speed"), self.wave_speed)
        glUniform1f(glGetUniformLocation(self.shader, "u_life"), self.life)
        glUniform1f(glGetUniformLocation(self.shader, "u_fade"), self.fade)
        glUniform1f(glGetUniformLocation(self.shader, "u_sway_phase"), self._sway_phase)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
