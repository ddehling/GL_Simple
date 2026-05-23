"""
Cyber city skyline — silhouetted vertical towers with sparse window-light
grids. The buildings are dark bodies (zero output) against a smog-banded
sky strip behind them. Window lights are sparse, randomized, with
occasional flickers. Reads `outstate['cyber_skyline_density']` (0..1) for
building density and `outstate['cyber_light_pollution']` for sky-band
intensity.

This is the FOUNDATION shader of the cyberpunk set — it defines what
"the city" looks like. Per the contrast playbook: dark bodies + bright
sparse features (window lights) = high-contrast under the limiter.
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_city_skyline(state, outstate, density=0.7, light_pollution=0.5):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberCitySkylineEffect,
                                          density=density,
                                          light_pollution=light_pollution)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_city_skyline] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # Wrapper defaults are 0.0 per docs/shader_info.txt — misconfigured
    # states fail visibly to nothing rather than rendering a partial city.
    eff.density = float(outstate.get('cyber_skyline_density', 0.0))
    eff.light_pollution = float(outstate.get('light_pollution', 0.0))
    eff.season = float(outstate.get('season', 0.5))
    # Vertical position in the Stack — 0.0 = street level (full towers),
    # 1.0 = Crown (only rooftops visible). Default 0.0 keeps existing
    # states looking the same until they explicitly set it.
    eff.view_elevation = float(outstate.get('cyber_view_elevation', 0.0))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


VERTEX = """#version 310 es
precision highp float;
layout(location = 0) in vec2 position;
out vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT = """#version 310 es
precision highp float;
in vec2 v_uv;
uniform float u_time;
uniform float u_density;
uniform float u_light_pollution;
uniform float u_season;              // [0,1) time-of-day cycle
uniform float u_view_elevation;      // 0=street/full towers, 1=Crown/only rooftops
uniform vec2 u_resolution;
out vec4 fragColor;

float hash(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
float hash2(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// Per-column top edge — returns the v_uv.y value of the building's top
// at this x position. Four building types selected by per-column hash so
// the silhouette isn't a uniform row of rectangles:
//   Type 0 (flat rect, 50% of columns) — top edge is constant per column
//   Type 1 (stepped,  20%)             — center 50% is taller (notched silhouette)
//   Type 2 (pyramid,  15%)             — top edge curves to a peak in the middle
//   Type 3 (antenna,  15%)             — flat top + thin tall antenna spike at center
//
// Returns the building's top edge y for the pixel's x position within
// the column. Pixels with uv.y > top_y are INSIDE the building.
float column_top_y(float col_idx, float col_frac) {
    float h_seed   = hash(col_idx * 1.7 + 3.0);
    // View-elevation scale: 0=street (full towers, 0.30..0.80 of canvas),
    // 1=Crown (only rooftops poking up, 0.045..0.12). Same scaling
    // applied wherever base_top is computed so the silhouette stays
    // consistent across shape types and the window grid.
    // Piecewise so that 0.0 stays at exactly 1.0× (default), negative
    // values look UP from below (towers extend off the top of frame at
    // -1.0 → 1.6×), positive values look DOWN from above (Crown
    // rooftops at 1.0 → 0.15×).
    float height_scale = (u_view_elevation < 0.0)
        ? (1.0 - 0.6  * u_view_elevation)
        : (1.0 - 0.85 * u_view_elevation);
    float base_top = (0.30 + h_seed * 0.50) * height_scale;
    float base_y   = 1.0 - base_top;              // top edge in v_uv

    float type_seed = hash(col_idx * 5.3 + 11.0);

    if (type_seed < 0.50) {
        // Flat rectangle — top edge constant across the column
        return base_y;
    } else if (type_seed < 0.70) {
        // Stepped — taller "penthouse" rectangle in the center 50%
        float step_h = 0.05 + hash(col_idx * 7.1 + 13.0) * 0.06;
        bool in_center = (col_frac > 0.25 && col_frac < 0.75);
        return in_center ? base_y - step_h : base_y;
    } else if (type_seed < 0.85) {
        // Pyramidal — top edge curves up toward the center
        float center_d = abs(col_frac - 0.5) * 2.0;          // 0 center, 1 sides
        float taper    = (1.0 - center_d) * (0.05 + hash(col_idx * 9.3) * 0.06);
        return base_y - taper;
    } else {
        // Antenna — flat rectangle with a thin tall spike at center
        float antenna_h = 0.10 + hash(col_idx * 11.7) * 0.10;
        bool in_antenna = (col_frac > 0.46 && col_frac < 0.54);
        return in_antenna ? base_y - antenna_h : base_y;
    }
}

// Sky band hue cycles with time-of-day. Eight anchor points across the
// cycle give a much wider color range than four — there's purple at
// pre-dawn and late night, golden yellow at dawn / morning, intense
// blue at noon, and saturated red+violet at dusk. Lower vec3 of each
// pair is the horizon (just above building tops); upper is zenith.
void season_sky(float s, out vec3 horizon, out vec3 zenith) {
    s = fract(s) * 8.0;                    // 8 segments
    // 8 anchor stops at s = 0, 1, 2, ... 7 (then wraps to 0 again)
    //   0.000  midnight       deep indigo horizon  +  near-black zenith
    //   0.125  pre-dawn       saturated violet     +  deep blue zenith
    //   0.250  dawn           amber/gold           +  bright cyan-blue
    //   0.375  morning        pale yellow          +  intense pure blue
    //   0.500  noon           washed cyan-blue     +  deep saturated blue
    //   0.625  afternoon      warm cyan-amber      +  cool blue zenith
    //   0.750  dusk           red-orange           +  electric violet
    //   0.875  night          deep magenta         +  dark indigo
    vec3 h0 = vec3(0.05, 0.05, 0.30);   // midnight horizon  (deep indigo)
    vec3 z0 = vec3(0.02, 0.02, 0.10);   // midnight zenith
    vec3 h1 = vec3(0.45, 0.10, 0.60);   // pre-dawn horizon  (saturated violet)
    vec3 z1 = vec3(0.05, 0.05, 0.35);   // pre-dawn zenith
    vec3 h2 = vec3(1.00, 0.55, 0.15);   // dawn horizon      (gold)
    vec3 z2 = vec3(0.20, 0.50, 0.85);   // dawn zenith       (cyan-blue)
    vec3 h3 = vec3(1.00, 0.90, 0.50);   // morning horizon   (pale yellow)
    vec3 z3 = vec3(0.10, 0.55, 1.00);   // morning zenith    (intense blue)
    vec3 h4 = vec3(0.45, 0.70, 0.95);   // noon horizon      (cyan-blue)
    vec3 z4 = vec3(0.05, 0.30, 1.00);   // noon zenith       (deep pure blue)
    vec3 h5 = vec3(0.85, 0.55, 0.30);   // afternoon horizon (warm amber)
    vec3 z5 = vec3(0.10, 0.30, 0.80);   // afternoon zenith  (cool blue)
    vec3 h6 = vec3(1.00, 0.25, 0.05);   // dusk horizon      (red-orange)
    vec3 z6 = vec3(0.40, 0.10, 0.85);   // dusk zenith       (electric violet)
    vec3 h7 = vec3(0.55, 0.10, 0.50);   // night horizon     (deep magenta)
    vec3 z7 = vec3(0.05, 0.02, 0.25);   // night zenith      (dark indigo)

    // Pick segment and interpolate
    if      (s < 1.0) { horizon = mix(h0, h1, s);       zenith = mix(z0, z1, s); }
    else if (s < 2.0) { horizon = mix(h1, h2, s - 1.0); zenith = mix(z1, z2, s - 1.0); }
    else if (s < 3.0) { horizon = mix(h2, h3, s - 2.0); zenith = mix(z2, z3, s - 2.0); }
    else if (s < 4.0) { horizon = mix(h3, h4, s - 3.0); zenith = mix(z3, z4, s - 3.0); }
    else if (s < 5.0) { horizon = mix(h4, h5, s - 4.0); zenith = mix(z4, z5, s - 4.0); }
    else if (s < 6.0) { horizon = mix(h5, h6, s - 5.0); zenith = mix(z5, z6, s - 5.0); }
    else if (s < 7.0) { horizon = mix(h6, h7, s - 6.0); zenith = mix(z6, z7, s - 6.0); }
    else              { horizon = mix(h7, h0, s - 7.0); zenith = mix(z7, z0, s - 7.0); }
}

// Window-lit probability shifts across the cycle. More windows lit at
// night, fewer at noon (people are at work, not at home). Used as a
// multiplier on per-state density.
float window_lit_mult(float s) {
    // 1.4x at midnight, 0.5x at noon, 1.2x at dusk, 1.0x at dawn
    return 0.95 + 0.45 * cos(6.28318 * s);
}

// Per-window color, weighted by the building's lighting "type". Cities
// read cohesively because each building has a dominant lighting style:
// residential = warm, corporate = cool white, neon-clad = magenta/cyan,
// industrial = sodium yellow. We still allow outlier windows within
// each building so it's not uniform.
//
// All colors stay on cyberpunk theme (no purples / blood-reds / nature
// greens that would clash with the rest of the set).
vec3 window_color(float seed, float bldg_type) {
    // Palette
    vec3 warm    = vec3(1.00, 0.65, 0.30);   // tungsten / incandescent
    vec3 cyan    = vec3(0.30, 0.85, 1.00);   // LED / cool fluorescent
    vec3 white   = vec3(0.90, 0.94, 1.00);   // office fluorescent
    vec3 magenta = vec3(1.00, 0.35, 0.75);   // neon-pink bleed
    vec3 sodium  = vec3(1.00, 0.80, 0.35);   // older sodium-vapor street/factory
    vec3 acid    = vec3(0.55, 1.00, 0.40);   // terminal-green (rare accent)

    if (bldg_type < 0.45) {
        // 45% RESIDENTIAL: mostly warm, with cyan + occasional magenta
        if      (seed < 0.72) return warm;
        else if (seed < 0.90) return cyan;
        else                  return magenta;
    } else if (bldg_type < 0.70) {
        // 25% CORPORATE: mostly cool white + cyan, occasional warm
        if      (seed < 0.60) return white;
        else if (seed < 0.88) return cyan;
        else                  return warm;
    } else if (bldg_type < 0.88) {
        // 18% NEON-CLAD MID-RISE: magenta + cyan dominate, warm accents
        if      (seed < 0.45) return magenta;
        else if (seed < 0.82) return cyan;
        else                  return warm;
    } else {
        // 12% INDUSTRIAL/OLDER: sodium dominant, rare acid-green accents
        if      (seed < 0.70) return sodium;
        else if (seed < 0.92) return warm;
        else                  return acid;
    }
}

void main() {
    vec2 uv = v_uv;

    // Building columns: 24 across the wrap, with per-column shape driven
    // by hash. Wraps cleanly because we use floor(x * 24) which is integer-
    // valued at the seam.
    const float NUM_COLS = 24.0;
    float col_x = uv.x * NUM_COLS;
    float col_idx = floor(col_x);
    float col_frac = fract(col_x);

    // Per-column top edge (varies based on shape type — flat / stepped /
    // pyramidal / antenna). For window-grid placement and sky-band
    // gradient we still use the BASE rectangle top (sky band sits above
    // the tallest possible building edge for this column). Same
    // view-elevation height scale as column_top_y so window grid lines
    // up with the silhouette.
    float h_seed = hash(col_idx * 1.7 + 3.0);
    // Piecewise so that 0.0 stays at exactly 1.0× (default), negative
    // values look UP from below (towers extend off the top of frame at
    // -1.0 → 1.6×), positive values look DOWN from above (Crown
    // rooftops at 1.0 → 0.15×).
    float height_scale = (u_view_elevation < 0.0)
        ? (1.0 - 0.6  * u_view_elevation)
        : (1.0 - 0.85 * u_view_elevation);
    float bldg_top = (0.30 + h_seed * 0.50) * height_scale;
    float bldg_bottom_y = 1.0 - bldg_top;     // base rectangle top edge
    float top_edge_y = column_top_y(col_idx, col_frac);  // shape-aware

    // Sky band: gradient above all buildings, hue driven by the
    // time-of-day cycle. Light pollution scales overall brightness.
    // This is the ONLY part of the canvas that pays significant energy
    // — buildings are dark silhouettes.
    vec3 sky_horizon, sky_zenith;
    season_sky(u_season, sky_horizon, sky_zenith);
    float sky_t = clamp((bldg_bottom_y - uv.y) / max(0.05, bldg_bottom_y), 0.0, 1.0);
    vec3 sky = mix(sky_horizon, sky_zenith, sky_t) * u_light_pollution;

    // Are we in a building? Use the shape-aware top edge so steps,
    // pyramids, and antennae carve out the correct silhouette.
    bool in_building = (uv.y > top_edge_y);

    // Per-column edge gap so columns read as distinct buildings. Skip
    // the gap for antenna-spike pixels (the antenna is narrower than
    // the gap and should always read as solid).
    //
    // Setback: ~30% of FLAT-top buildings narrow in their upper third
    // (Tyrell-pyramid / Empire-State ziggurat silhouette). Independent
    // of shape type, only stacks with flat to avoid over-modulating
    // pyramidal/antenna columns which already break silhouette.
    float type_seed_again = hash(col_idx * 5.3 + 11.0);
    bool is_flat_shape   = (type_seed_again < 0.50);
    float setback_seed   = hash(col_idx * 4.7 + 23.0);
    bool has_setback     = is_flat_shape && (setback_seed < 0.30);
    float setback_height = bldg_bottom_y + (1.0 - bldg_bottom_y) * 0.35;
    float edge = 0.06;
    if (has_setback && uv.y < setback_height) {
        edge = 0.16;   // upper portion narrower than base
    }
    bool in_edge_gap = (col_frac < edge) || (col_frac > 1.0 - edge);
    if (in_building && in_edge_gap) {
        in_building = false;
    }

    // Write fragment depth so depth-aware effects (rain, pixel_spots,
    // anything Pattern A per docs/shader_info.txt) can occlude against
    // the city silhouette. Using the doc's standard z=0..100 scale
    // mapped through depth=z/100:
    //   sky      → z=95 (very far)
    //   building → z=70 (mid-far, like the skyline is 70m out)
    // So foreground particles with z < 70 (depth < 0.7) still render
    // IN FRONT of the buildings — rain in the foreground, drone
    // spotlights, etc. — and only background particles with z > 70
    // get occluded behind the city, which matches real-world depth.
    gl_FragDepth = in_building ? 0.70 : 0.95;

    vec3 col = sky;
    float alpha = 0.85;   // sky always opaque enough to occlude what's behind

    if (in_building) {
        // Building body: nearly zero output (silhouette).
        col = vec3(0.01, 0.01, 0.02);

        // Window grid — only inside the base RECTANGLE (uv.y >= bldg_bottom_y).
        // Pixels above the base rectangle (step / pyramid / antenna)
        // skip windows; the silhouette there is just dark spire.
        if (uv.y < bldg_bottom_y) {
            // antenna / step / pyramid extension — no windows here
            alpha = 0.95;
            fragColor = vec4(col, alpha);
            return;
        }

        // Window grid — 6 columns of windows per building, 18 rows down.
        float win_x = (col_frac - edge) / (1.0 - 2.0 * edge);   // 0..1 inside building
        float win_y_norm = (uv.y - bldg_bottom_y) / (1.0 - bldg_bottom_y);

        const float WINDOWS_X = 6.0;
        const float WINDOWS_Y = 18.0;
        float wx = win_x * WINDOWS_X;
        float wy = win_y_norm * WINDOWS_Y;
        float wxi = floor(wx);
        float wyi = floor(wy);
        float wxf = fract(wx);
        float wyf = fract(wy);

        // Window cell shape: small bright rectangle in each cell
        float in_window_x = step(0.25, wxf) * (1.0 - step(0.75, wxf));
        float in_window_y = step(0.20, wyf) * (1.0 - step(0.65, wyf));
        float in_window = in_window_x * in_window_y;

        // Per-window on/off pattern (deterministic per column+window).
        // Season multiplier: more lit windows at night, fewer at noon.
        // Baseline lowered from 0.20 to 0.05 so density=0 actually reads
        // as a near-blackout (a handful of emergency lights only); scale
        // bumped 0.45 → 0.60 so the max-lit case stays roughly where it
        // was at u_density=1.0.
        float win_seed = hash2(vec2(col_idx * 11.0 + wxi, wyi * 7.0));
        float on_prob = (0.05 + u_density * 0.60) * window_lit_mult(u_season);
        float lit = step(1.0 - on_prob, win_seed);

        // Per-window flicker — rare, ~once per 4 sec
        float flicker_t = u_time * 0.25 + win_seed * 13.7;
        float flicker = step(0.92, fract(flicker_t));   // 8% of cycles
        lit *= (1.0 - flicker * 0.7);

        // Per-building lighting "type" — each column gets one of four
        // dominant palettes so the city reads as distinct buildings with
        // character rather than uniformly speckled random colors.
        float bldg_type = hash(col_idx * 3.1 + 17.0);
        vec3 wcol = window_color(win_seed, bldg_type);

        // Service floor bands — every N floors, a row of lit windows
        // marks a mechanical / transfer floor. KEEPS per-window x gaps
        // so each band reads as several distinct lit windows in a row
        // rather than a solid horizontal strip. Also gated on u_density
        // so the bands fade out alongside the rest of the city in low-
        // density states (blackout, tunnel interiors, etc.).
        //
        // ~40% of buildings have them; period varies 4/5/6 floors.
        float band_seed     = hash(col_idx * 6.3 + 31.0);
        float band_gate     = smoothstep(0.10, 0.55, u_density);     // 0 at blackout, 1 at typical
        bool  building_has_bands = (band_seed < 0.40);
        if (building_has_bands && band_gate > 0.01 && wyi > 0.5) {
            float band_period = 4.0 + floor(hash(col_idx * 8.9 + 41.0) * 3.0);
            bool is_band_row = (mod(wyi, band_period) < 0.5);
            if (is_band_row) {
                // Row of distinct windows — reuse the existing per-window
                // x gating (in_window_x) so each band is several lit cells
                // separated by the building's mullion gaps, not a solid bar.
                float band_y = step(0.20, wyf) * (1.0 - step(0.65, wyf));
                in_window = band_y * in_window_x;
                lit       = band_gate;                  // fades with density
                wcol      = wcol * 0.85 + 0.15;         // slightly brighter / desaturated
            }
        }

        col += wcol * in_window * lit * 1.0;
        alpha = 0.95;
    }

    // Crown beacon — small pulsing red light atop antenna-type towers
    // and ~12% of tall flat towers. Sits just above the building edge,
    // additive over sky. Classic aviation-warning red, slow asymmetric
    // pulse so multiple towers blink out of phase.
    bool has_antenna_type = (type_seed_again >= 0.85);
    bool has_beacon = has_antenna_type ||
                      (hash(col_idx * 13.1 + 53.0) < 0.12 && is_flat_shape && h_seed > 0.55);
    if (has_beacon) {
        // Beacon position: just above top_edge_y, centered horizontally
        // in the column. Squished horizontally so it reads as a small dot.
        vec2 d = vec2((col_frac - 0.5) * 14.0,
                      (top_edge_y - 0.010 - uv.y) * 80.0);
        float r = length(d);
        float dot_intensity = 1.0 - smoothstep(0.0, 1.0, r);
        if (dot_intensity > 0.001) {
            // Asymmetric pulse: bright flash every ~1.3s, mostly dark between
            float pulse_t = fract(u_time * 0.75 + col_idx * 0.137);
            float pulse = 0.20 + 0.80 * smoothstep(0.82, 0.95, pulse_t)
                              * (1.0 - smoothstep(0.95, 1.00, pulse_t));
            vec3 beacon_col = vec3(1.00, 0.18, 0.12);
            col += beacon_col * dot_intensity * pulse * 1.6;
            alpha = max(alpha, dot_intensity * pulse);
        }
    }

    fragColor = vec4(col, alpha);
}
"""


class CyberCitySkylineEffect(ShaderEffect):
    def __init__(self, viewport, density: float = 0.7, light_pollution: float = 0.5):
        super().__init__(viewport)
        self.render_priority = 6.0   # Behind hologram/signs, in front of smog
        self.density = density
        self.light_pollution = light_pollution
        self.season = 0.5            # set by wrapper from outstate['season']
        self.view_elevation = 0.0    # 0=street level, 1=Crown rooftops
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberCitySkyline shader compile error: {e}")
            raise

    def setup_buffers(self):
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        self.VBOs = [vbo]
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt

    def render(self, state: Dict):
        if not self.enabled:
            return
        # GL_ALWAYS so we still paint regardless of what was drawn below
        # (the sky band must always overlay smog, underway glow, etc.).
        # depth MASK on TRUE so the fragment shader's gl_FragDepth writes
        # land in the depth buffer — that's what lets pixel_spots and
        # other later effects depth-test against building silhouettes.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_TRUE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform1f(glGetUniformLocation(self.shader, "u_light_pollution"), self.light_pollution)
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, "u_view_elevation"), self.view_elevation)
        glUniform2f(glGetUniformLocation(self.shader, "u_resolution"),
                    float(self.viewport.width), float(self.viewport.height))
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
