"""Coral reef effect — branching coral silhouettes along the seabed.

Renders along the inner ring (uv.y near 0 = inner radius = seabed). Per
angular column, a coral structure of randomized height/width/color, with
small pulsing polyps. Sway is driven by the integrated wave_speed phase
(see docs/shader_info.txt "Time-based Animation"), so it never reverses
during weather state transitions.

Drives:
  wave_speed             -> sway rate (integrated as u_sway_phase)
  marine_life_activity   -> polyp pulse intensity

Used as an on_transition_events entry for ocean_coral_reef. Fades in
over 5 s, holds, fades out over the last 5 s of its scheduled duration.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_coral(state, outstate):
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CoralEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized coral for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize coral: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.wave_speed = float(outstate.get('wave_speed', 0.0))
        eff.life = float(outstate.get('marine_life_activity', 0.0))

        # Fade in over first 5 s, fade out over last 5 s.
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
    // Mid-back foreground feature: in front of waves/water, behind
    // fish/bubbles. Pattern B layer, depth disabled in render() so this
    // value is for documentation only — render_priority controls order.
    gl_Position = vec4(position, 0.72, 1.0);
}
"""

_FRAG = """
#version 310 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_wave_speed;
uniform float u_life;
uniform float u_fade;
// CPU-integrated wave-phase. See docs/shader_info.txt — this replaces
// the anti-pattern `u_time * u_wave_speed` so sway stays monotonic when
// wave_speed interpolates.
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

float fbm(vec2 p) {
    return vnoise(p) * 0.55 + vnoise(p * 2.13 + 7.7) * 0.30
         + vnoise(p * 4.27 + 3.1) * 0.15;
}

// Distance from point p to line segment (a, b).
float seg_dist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t);
}

// Wrapped horizontal distance (handles seam at uv.x = 0/1).
float wrap_dx(float a, float b) {
    float dx = abs(a - b);
    return min(dx, 1.0 - dx);
}

// ============================================================
// Shape primitives. Each returns a 0..1 coverage mask. Each
// shape spans different silhouettes so the reef doesn't read
// as a row of identical sticks.
// ============================================================

// Tall branched coral (trunk + 2 branches + occasional sub-branches).
float shape_branched(vec2 p, vec2 c, float seed, float scale) {
    float h = (0.20 + 0.32 * hash(vec2(seed, 7.3))) * scale;
    float thickness = (0.010 + 0.013 * hash(vec2(seed, 11.1))) * scale;
    float lean = (hash(vec2(seed, 13.7)) - 0.5) * 0.05 * scale;

    // Use signed dx so branches in the wrap region still render correctly
    // by working in the local frame around the cell center.
    float dx = p.x - c.x;
    if (dx >  0.5) dx -= 1.0;
    if (dx < -0.5) dx += 1.0;
    vec2 lp = vec2(dx, p.y);

    vec2 trunk_a = vec2(0.0, 0.0);
    vec2 trunk_b = vec2(lean, h);
    float d = seg_dist(lp, trunk_a, trunk_b);

    float br1_dx = (-0.012 - 0.014 * hash(vec2(seed, 23.5))) * scale;
    float br2_dx = ( 0.012 + 0.014 * hash(vec2(seed, 27.7))) * scale;
    float br1_dy = (0.045 + 0.030 * hash(vec2(seed, 29.3))) * scale;
    float br2_dy = (0.045 + 0.030 * hash(vec2(seed, 31.1))) * scale;

    vec2 br1_a = mix(trunk_a, trunk_b, 0.35 + 0.20 * hash(vec2(seed, 17.5)));
    vec2 br1_b = br1_a + vec2(br1_dx, br1_dy);
    d = min(d, seg_dist(lp, br1_a, br1_b));

    vec2 br2_a = mix(trunk_a, trunk_b, 0.60 + 0.20 * hash(vec2(seed, 19.7)));
    vec2 br2_b = br2_a + vec2(br2_dx, br2_dy);
    d = min(d, seg_dist(lp, br2_a, br2_b));

    if (hash(vec2(seed, 41.7)) > 0.45) {
        vec2 sb1_b = br1_b + vec2(-0.010 * scale, 0.025 * scale);
        d = min(d, seg_dist(lp, br1_b, sb1_b));
    }
    if (hash(vec2(seed, 43.1)) > 0.45) {
        vec2 sb2_b = br2_b + vec2(0.010 * scale, 0.025 * scale);
        d = min(d, seg_dist(lp, br2_b, sb2_b));
    }
    return 1.0 - smoothstep(thickness * 0.5, thickness * 1.7, d);
}

// Round mound (brain-coral / boulder shape) — wide and low. Heights
// span very short to medium so we get range, not all squat.
float shape_mound(vec2 p, vec2 c, float seed, float scale) {
    float w = (0.040 + 0.045 * hash(vec2(seed, 51.3))) * scale;   // half-width
    float h = (0.030 + 0.090 * hash(vec2(seed, 53.7))) * scale;   // height
    float dx = wrap_dx(p.x, c.x);
    float dy = p.y - c.y;
    if (dy < 0.0) return 0.0;
    // Surface noise breaks the smooth ellipse into a brain-coral lumpy look
    float surf = vnoise(vec2(p.x * 28.0 + seed * 7.0, p.y * 22.0)) * 0.18;
    float ex = dx / max(w + surf * 0.012, 1e-4);
    float ey = dy / max(h, 1e-4);
    float r2 = ex * ex + ey * ey;
    float core = 1.0 - smoothstep(0.7, 1.0, r2);
    // Lumpy surface ridges (brain coral grooves)
    float ridges = 0.5 + 0.5 * sin(dx * 220.0 + dy * 80.0 + seed * 12.0);
    ridges = smoothstep(0.30, 0.65, ridges);
    return core * (0.65 + 0.35 * ridges);
}

// Table coral — flat horizontal disc on a thin stem. Stem heights vary.
float shape_table(vec2 p, vec2 c, float seed, float scale) {
    float stem_h = (0.06 + 0.18 * hash(vec2(seed, 71.5))) * scale;
    float disc_w = (0.045 + 0.030 * hash(vec2(seed, 73.1))) * scale;
    float disc_t = (0.012 + 0.008 * hash(vec2(seed, 77.9))) * scale; // disc thickness
    float stem_w = 0.006 * scale;
    float dx = wrap_dx(p.x, c.x);
    float dy = p.y - c.y;
    if (dy < 0.0) return 0.0;

    // Stem (vertical line up to the disc)
    float in_stem = step(dy, stem_h)
                  * (1.0 - smoothstep(stem_w * 0.5, stem_w * 1.5, dx));
    // Disc (horizontal slab centered at stem_h)
    float disc_dy = abs(dy - stem_h);
    float in_disc = (1.0 - smoothstep(disc_w * 0.75, disc_w, dx))
                  * (1.0 - smoothstep(disc_t * 0.6, disc_t, disc_dy));
    return max(in_stem * 0.9, in_disc);
}

// Sea fan — vertical fan, narrow at base, wider at top, with vertical ribs.
float shape_fan(vec2 p, vec2 c, float seed, float scale) {
    float h = (0.30 + 0.25 * hash(vec2(seed, 81.3))) * scale;
    float top_w = (0.050 + 0.030 * hash(vec2(seed, 83.7))) * scale;
    float dx = wrap_dx(p.x, c.x);
    float dy = p.y - c.y;
    if (dy < 0.0 || dy > h) return 0.0;
    // Trapezoidal envelope, 25% width at base
    float w_at_y = mix(top_w * 0.25, top_w, dy / h);
    float envelope = 1.0 - smoothstep(w_at_y * 0.7, w_at_y, dx);
    // Vertical ribs (the fan structure)
    float ribs = 0.5 + 0.5 * sin(dx * 320.0 + seed * 23.0);
    ribs = smoothstep(0.35, 0.70, ribs);
    return envelope * (0.45 + 0.55 * ribs);
}

// Boulder — chunky rounded encrustation, lumpier than mound, can be quite tall.
float shape_boulder(vec2 p, vec2 c, float seed, float scale) {
    float w = (0.055 + 0.040 * hash(vec2(seed, 91.3))) * scale;
    float h = (0.060 + 0.150 * hash(vec2(seed, 93.7))) * scale;
    float dx = wrap_dx(p.x, c.x);
    float dy = p.y - c.y;
    if (dy < 0.0) return 0.0;
    // Multi-octave perturbation gives lumpy outline
    float perturb = (vnoise(vec2(p.x * 18.0 + seed * 3.7, p.y * 16.0)) - 0.5) * 0.18
                  + (vnoise(vec2(p.x * 42.0 + seed * 1.1, p.y * 38.0)) - 0.5) * 0.08;
    float ex = dx / max(w * (1.0 + perturb), 1e-4);
    float ey = dy / max(h * (1.0 + perturb * 0.6), 1e-4);
    float r2 = ex * ex + ey * ey;
    return 1.0 - smoothstep(0.75, 1.05, r2);
}

// Cell dispatcher: picks one of the 5 shapes based on a per-cell type
// hash. Returns coverage mask. type_hash is 0..1.
float cell_shape(vec2 p, vec2 c, float seed, float scale, float type_hash) {
    if (type_hash < 0.22) {
        return shape_branched(p, c, seed, scale);
    } else if (type_hash < 0.50) {
        return shape_mound(p, c, seed, scale);
    } else if (type_hash < 0.68) {
        return shape_boulder(p, c, seed, scale);
    } else if (type_hash < 0.85) {
        return shape_table(p, c, seed, scale);
    } else {
        return shape_fan(p, c, seed, scale);
    }
}

void main() {
    if (u_fade < 0.005) discard;

    vec2 uv = v_uv;

    // Coral lives in the inner band (uv.y in [0, 0.62]). The seabed
    // corresponds to the inner ring of the fan (uv.y = 0).
    if (uv.y > 0.62) discard;

    // Whole-field sway from wave motion (low-frequency horizontal warp).
    // Uses integrated phase — never reverses on wave_speed transitions.
    float field_sway = sin(u_sway_phase * 0.5 + uv.x * 3.5) * 0.008
                     + sin(u_sway_phase * 0.3 + uv.x * 1.7) * 0.005;
    vec2 suv = vec2(uv.x + field_sway, uv.y);

    // ============================================================
    // Layer 1: Seabed strip — always-present rock/sand floor.
    // ============================================================
    float seabed_h = 0.040
                   + 0.045 * vnoise(vec2(suv.x * 5.5, 0.0))
                   + 0.025 * vnoise(vec2(suv.x * 14.0, 1.0));
    float seabed = smoothstep(seabed_h + 0.012, seabed_h - 0.012, uv.y);

    // ============================================================
    // Layer 2: Encrusting / mound coral — patchy organic mass that
    // fills the lower mid-band so the reef reads as a continuous
    // structure rather than isolated poles.
    // ============================================================
    float enc_y_gate = smoothstep(0.42, 0.0, uv.y);
    float enc_field = fbm(vec2(suv.x * 9.0, suv.y * 11.0 + 2.7));
    // Threshold drops with height so the mass tapers upward.
    float enc_thresh = mix(0.42, 0.62, uv.y / 0.42);
    float encrusting = smoothstep(enc_thresh + 0.08, enc_thresh - 0.04, enc_field)
                     * enc_y_gate;
    // Carve a small noise into the surface so it isn't a smooth blob.
    float enc_carve = vnoise(vec2(suv.x * 35.0, suv.y * 30.0 + 8.1));
    encrusting *= 0.55 + 0.45 * enc_carve;

    // ============================================================
    // Layer 3: Background coral structures — depth/parallax layer.
    // Smaller scale, dimmer, slightly offset cell positions. Reads
    // as further-away coral behind the foreground row.
    // ============================================================
    float bg_n = 18.0;
    float bg_cell = floor((uv.x + 0.022) * bg_n);
    float bg_seed = hash(vec2(bg_cell, 101.3));
    float bg_sway = sin(u_sway_phase * 0.65 + bg_seed * 6.28) * 0.008;
    vec2 bg_center = vec2((bg_cell + 0.5) / bg_n - 0.022 + bg_sway, 0.0);
    float bg_type = hash(vec2(bg_cell, 103.7));
    // Background scale ~0.65 of foreground for parallax effect.
    float bg_mask = cell_shape(uv, bg_center, bg_seed, 0.62, bg_type);

    // ============================================================
    // Layer 4: Foreground coral structures — fewer cells, bigger
    // shapes, full color saturation. Five shape types (branched,
    // mound, boulder, table, fan) at very different heights so the
    // reef silhouette varies dramatically across the band.
    // ============================================================
    float fg_n = 11.0;
    float fg_cell = floor(uv.x * fg_n);                 // STABLE — uv, not suv
    float fg_seed = hash(vec2(fg_cell, 1.0));
    // Per-cell independent sway so corals don't move as one block.
    float fg_sway = sin(u_sway_phase * 0.55 + fg_seed * 6.28) * 0.012;
    vec2 fg_center = vec2((fg_cell + 0.5) / fg_n + fg_sway, 0.0);
    float fg_type = hash(vec2(fg_cell, 23.5));
    // Foreground at full scale; range of heights baked into each shape.
    float fg_mask = cell_shape(uv, fg_center, fg_seed, 1.0, fg_type);
    // Slight per-pixel density variation suggests texture / polyp clusters.
    float fg_var = 0.75 + 0.25 * vnoise(vec2(uv.x * 50.0,
                                             uv.y * 40.0 + fg_seed * 9.0));
    fg_mask *= fg_var;

    // ============================================================
    // Composite all layers (back-to-front).
    // ============================================================
    float reef_mask = max(seabed,
                      max(encrusting,
                      max(bg_mask * 0.78, fg_mask)));

    // ============================================================
    // Color: per-region species + per-cell variation.
    // ============================================================
    float color_n = vnoise(vec2(suv.x * 4.0, suv.y * 6.0 + 7.7));
    float fg_species = hash(vec2(fg_cell, 17.5));
    float bg_species = hash(vec2(bg_cell, 19.7));

    vec3 sand_col   = vec3(0.55, 0.45, 0.32);
    vec3 enc_col_a  = vec3(0.62, 0.32, 0.42);
    vec3 enc_col_b  = vec3(0.55, 0.42, 0.55);
    vec3 enc_col    = mix(enc_col_a, enc_col_b, color_n);

    vec3 br_pink    = vec3(1.00, 0.45, 0.55);
    vec3 br_orange  = vec3(1.00, 0.58, 0.25);
    vec3 br_purple  = vec3(0.65, 0.32, 0.90);
    vec3 br_yellow  = vec3(0.95, 0.82, 0.40);
    vec3 br_red     = vec3(0.85, 0.25, 0.25);
    vec3 br_teal    = vec3(0.30, 0.65, 0.62);

    // Foreground species color
    vec3 fg_a = mix(br_pink, br_purple, smoothstep(0.0, 0.5, fg_species));
    vec3 fg_b = mix(br_orange, br_yellow, smoothstep(0.5, 1.0, fg_species));
    vec3 fg_col = mix(fg_a, fg_b, step(0.5, fg_species));
    fg_col = mix(fg_col, br_red, step(0.85, fg_species));
    fg_col = mix(fg_col, br_teal, step(0.92, fg_species));

    // Background species: dimmed/desaturated for depth feel
    vec3 bg_a = mix(br_pink, br_purple, smoothstep(0.0, 0.5, bg_species));
    vec3 bg_b = mix(br_orange, br_yellow, smoothstep(0.5, 1.0, bg_species));
    vec3 bg_col_raw = mix(bg_a, bg_b, step(0.5, bg_species));
    // Cool/desaturate to push it visually back
    vec3 bg_col = mix(bg_col_raw, vec3(0.35, 0.40, 0.50), 0.40) * 0.78;

    // Composite color: paint by feature priority — sand at the bottom,
    // encrusting, background coral (depth), foreground coral on top.
    vec3 col_final = sand_col;
    col_final = mix(col_final, enc_col, encrusting);
    col_final = mix(col_final, bg_col, bg_mask * 0.78);
    col_final = mix(col_final, fg_col, fg_mask);

    // ============================================================
    // Polyps: small bright spots that pulse with marine_life.
    // Constant-frequency time multiplier — safe.
    // ============================================================
    vec2 pp = vec2(suv.x * 75.0, uv.y * 55.0);
    vec2 p_cell = floor(pp);
    vec2 p_frac = fract(pp);
    float p_seed = hash(p_cell + 5.7);
    float p_show = step(0.91, p_seed) * reef_mask;
    float p_d = distance(p_frac, vec2(0.5));
    float p_pulse = 0.55 + 0.45 * sin(u_time * 1.8 + p_seed * 31.4);
    float p_int = p_show * exp(-(p_d * p_d) / 0.018) * p_pulse * u_life;

    vec3 polyp_col = vec3(0.95, 0.97, 1.00);

    vec3 final_col = col_final + polyp_col * p_int * 0.85;
    // STRAIGHT alpha — not pre-multiplied. Global blend is
    // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
    float final_alpha = clamp(reef_mask + p_int * 0.45, 0.0, 0.94) * u_fade;
    if (final_alpha < 0.01) discard;
    fragColor = vec4(final_col, final_alpha);
}
"""


class CoralEffect(ShaderEffect):
    def __init__(self, viewport):
        super().__init__(viewport)
        # Mid-back foreground: in front of waves/water (1-2), behind
        # mid-stack creatures (fish/bubbles). Adjust if fish render order
        # needs tuning.
        self.render_priority = 2.5
        self._time = 0.0
        self._sway_phase = 0.0   # integrated dt * wave_speed
        self.wave_speed = 0.3
        self.life = 0.5
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
        # Floor wave_speed slightly so a "calm" state still has readable sway.
        effective_wave = max(self.wave_speed, 0.08)
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
