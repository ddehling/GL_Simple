"""
Desert dunes — parallax sand-ridge silhouette as a horizon band across the
lower portion of the fan. Pattern B (fullscreen quad, no depth writes).

Reasons in PHYSICAL space (feet) via fan_coords.fan_uv_to_physical so that
ridge lines look like real horizon lines on the semicircular LED fan.

Each of three depth layers uses an asymmetric dune wave (gentle windward
rise, sharp leeward drop) that literally migrates downwind via a CPU-
integrated wind phase. Slope-based Lambert shading from a sun direction
(derived from season) gives the dunes visible 3D form.
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

uniform float u_wind_phase;     // CPU-integrated dt * wind drift in feet
uniform float u_ripple_phase;   // fast surface-ripple phase, scales with wind
uniform float u_wind;           // raw wind magnitude (~0..2)
uniform float u_season;         // 0..1 time-of-day (0=midnight, 0.5=noon)
uniform float u_strength;       // 0..1 master alpha
uniform vec3  u_tint;           // base dune color tint (from fog_color)
uniform vec2  u_light_dir;      // normalized 2D direction TOWARD the sun/moon

float hash11(float n) {{ return fract(sin(n) * 43758.5453); }}
float hash21(vec2 p) {{ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }}

// Smooth 1D value-noise (input in feet)
float vnoise1(float x) {{
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), u);
}}

// 2D value-noise for surface texture
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

// ===== Asymmetric dune wave =====
// Spends 70% of each cycle on the gentle WINDWARD rise, 30% on the sharp
// LEEWARD drop — the actual silhouette of a sand dune. Moving the phase with
// wind makes the dunes literally migrate downwind.
float dune_wave(float x_ft, float wavelength_ft, float phase_ft) {{
    float u = fract((x_ft + phase_ft) / wavelength_ft);
    if (u < 0.7) {{
        // Gentle windward face: 0 → 1 over 70% of the cycle (sin quarter)
        return sin(u / 0.7 * 1.5708);
    }} else {{
        // Sharp leeward face: 1 → 0 over 30% (cos quarter)
        return cos((u - 0.7) / 0.3 * 1.5708);
    }}
}}

// Per-cycle amplitude jitter so adjacent dunes aren't clones
float dune_jitter(float x_ft, float wavelength_ft, float phase_ft) {{
    float ci = floor((x_ft + phase_ft) / wavelength_ft);
    return 0.7 + 0.3 * hash11(ci * 7.13);
}}

// X-axis warp for APERIODIC spacing. Stretches and compresses x non-uniformly
// using a slow noise so dunes are NOT evenly spaced. The warp also drifts
// with the wind so the irregular spacing migrates with the dunes themselves.
// `warp_scale_ft` controls how much x is shifted (~ how irregular the
// spacing looks); a low spatial frequency keeps the warp slow so it doesn't
// destroy the dune shape itself.
float warp_x(float x_ft, float drift_ft, float warp_scale_ft, float warp_freq) {{
    float n = vnoise1(x_ft * warp_freq + drift_ft * warp_freq * 0.5) - 0.5;
    return x_ft + warp_scale_ft * n * 2.0;
}}

// Per-cycle personality — each individual dune has its own roughness and
// crest sharpness, so adjacent dunes don't all read identically.
// Returns (rough_amount, sharp_bias) where rough_amount in [0.4, 1.6]
// scales the high-freq crest detail, and sharp_bias in [-0.3, +0.3] biases
// the apex pointiness.
vec2 dune_personality(float x_ft, float wavelength_ft, float drift_ft) {{
    float ci = floor((x_ft + drift_ft) / wavelength_ft);
    float rough = 0.4 + 1.2 * hash11(ci * 13.71);
    float sharp = (hash11(ci * 23.17) - 0.5) * 0.6;
    return vec2(rough, sharp);
}}

// Multi-octave crest detail. Builds up lateral structure on the crest
// (sub-peaks, notches, irregular topography along the ridge length).
// `detail_scale` amplifies the structure — back/mid pass 1.0; the front
// (foreground) layer passes a larger value to read as the most-detailed,
// closest-to-viewer silhouette with extra high-frequency micro-bumps.
float crest_detail(float x_ft, float w, float drift_ft, float amp_ft,
                   float rough, float detail_scale) {{
    if (w <= 0.0) return 0.0;

    // Base four octaves (always on)
    float n = 0.0;
    n += 1.00 * (vnoise1(x_ft * 0.45 + drift_ft * 0.7              ) - 0.5);
    n += 0.65 * (vnoise1(x_ft * 1.10 + drift_ft * 0.9 + 17.3        ) - 0.5);
    n += 0.40 * (vnoise1(x_ft * 2.10 + drift_ft * 1.2 + 41.7        ) - 0.5);
    n += 0.22 * (vnoise1(x_ft * 3.20 + drift_ft * 1.5 + 73.1        ) - 0.5);

    // Extra high-frequency octaves for foreground detail. Activated when
    // detail_scale > 1.0 (front layer) so back/mid stay visually quieter.
    float extra = max(0.0, detail_scale - 1.0);
    if (extra > 0.0) {{
        n += 0.55 * extra * (vnoise1(x_ft * 5.30 + drift_ft * 2.0 + 113.7) - 0.5);
        n += 0.35 * extra * (vnoise1(x_ft * 8.20 + drift_ft * 2.5 + 157.1) - 0.5);
        n += 0.20 * extra * (vnoise1(x_ft * 12.5 + drift_ft * 3.0 + 191.3) - 0.5);
    }}

    // Sharp "notch" — occasional dunes get a deep saddle in their crest.
    float notch_phase = vnoise1(x_ft * 0.30 + drift_ft * 0.4 + 91.5);
    float notch = smoothstep(0.78, 0.92, notch_phase)
                * smoothstep(0.92, 0.78, notch_phase);

    // Secondary smaller notches/sub-peaks for high-detail layers
    float sub_notch = 0.0;
    if (extra > 0.0) {{
        float p2 = vnoise1(x_ft * 0.85 + drift_ft * 0.6 + 211.7);
        sub_notch = smoothstep(0.82, 0.93, p2)
                  * smoothstep(0.93, 0.82, p2) * extra;
    }}

    // Mask: cover most of the upper face so detail shapes the whole crest
    float mask = max(0.0, w);

    float detail = n * amp_ft * 0.85 * mask * rough * detail_scale;
    detail -= notch * amp_ft * 0.55 * mask * detail_scale;
    detail -= sub_notch * amp_ft * 0.35 * mask;
    return detail;
}}

// Full ridge height (feet) for a layer. `wavelength_ft` is a NOMINAL
// spacing — actual dune-to-dune distances vary via x-warping. Each dune
// carries its own crest detail. `detail_scale` controls how much
// high-freq structure the layer gets: 1.0 for distant layers, larger
// values for the foreground so close-up dunes have more visible detail.
float ridge_h(float x_ft, float baseline_ft, float amp_ft,
              float wavelength_ft, float drift_ft,
              float warp_scale_ft, float warp_freq,
              float detail_scale) {{
    float wx = warp_x(x_ft, drift_ft, warp_scale_ft, warp_freq);
    float w  = dune_wave(wx, wavelength_ft, drift_ft);
    float j  = dune_jitter(wx, wavelength_ft, drift_ft);
    vec2  p  = dune_personality(wx, wavelength_ft, drift_ft);
    float w_shaped = w >= 0.0 ? pow(w, 1.0 - p.y) : w;
    float base = baseline_ft + amp_ft * w_shaped * j;
    base += crest_detail(x_ft, max(0.0, w), drift_ft, amp_ft, p.x, detail_scale);
    return base;
}}

// Slope dh/dx via central difference (in feet)
float ridge_slope(float x_ft, float baseline_ft, float amp_ft,
                  float wavelength_ft, float drift_ft,
                  float warp_scale_ft, float warp_freq,
                  float detail_scale) {{
    float dx = 0.25;
    float h_p = ridge_h(x_ft + dx, baseline_ft, amp_ft, wavelength_ft,
                        drift_ft, warp_scale_ft, warp_freq, detail_scale);
    float h_m = ridge_h(x_ft - dx, baseline_ft, amp_ft, wavelength_ft,
                        drift_ft, warp_scale_ft, warp_freq, detail_scale);
    return (h_p - h_m) / (2.0 * dx);
}}

float dune_alpha(vec2 phys, float h) {{
    return smoothstep(h + 0.30, h - 0.10, phys.y);
}}

// Sand surface texture shading. Returns a luminance multiplier ~[0.55, 1.5].
// The streaks are stretched horizontally so they read as wind-blown striations.
// flow_offset drifts with wind for visible motion.
float sand_surface(vec2 phys, float flow_offset, float ripple_off) {{
    // Long horizontal streaks (streched in x, tight in y)
    vec2 q_streak = vec2((phys.x + flow_offset) * 0.6, phys.y * 2.4);
    float streak = vnoise2(q_streak);

    // Cross-cutting wind ripple bands — narrow but visible at any wind > ~0
    float band_phase = (phys.x + flow_offset * 1.3 + ripple_off) * 5.0 + phys.y * 1.5;
    float ripple = pow(0.5 + 0.5 * sin(band_phase), 3.0);   // wider than pow(,6)
    float ripple_gate = clamp(u_wind * 0.9 + 0.1, 0.0, 1.2);

    // Fine grain that always shimmers slightly — keeps surface alive at wind=0
    vec2 q_grain = vec2(phys.x * 8.0 + ripple_off * 0.4, phys.y * 8.0);
    float grain = vnoise2(q_grain);

    // Combine. Streak contrast is meaningful (±0.4) so the surface reads as
    // textured rather than flat. Ripples brighten where bands sit.
    float lit = 1.0
              + (streak - 0.5) * 0.55
              + ripple * 0.40 * ripple_gate
              + (grain  - 0.5) * 0.12;
    return clamp(lit, 0.55, 1.55);
}}

// Lifted grit above front crest at high wind
vec2 lifted_grit(vec2 phys, float crest_h, float flow_offset) {{
    float above = crest_h + 1.0 - phys.y;
    if (above <= 0.0 || phys.y < crest_h - 0.05) return vec2(0.0);
    vec2 q = vec2((phys.x + flow_offset * 1.4) * 6.5, phys.y * 5.0);
    float n = vnoise2(q);
    float band = smoothstep(0.0, 0.4, above) * (1.0 - smoothstep(0.4, 1.0, above));
    float wind_gate = smoothstep(0.5, 1.4, u_wind);
    float on = step(0.78, n) * band * wind_gate;
    return vec2(on * (0.6 + 0.4 * n), on * 0.55);
}}

// Slope-based 3D shading — Lambert from the sun direction. Returns a
// per-fragment lightness multiplier so the windward face reads bright and
// the leeward (shadow) face reads dark, giving the dunes visible 3D form.
float slope_shading(float slope) {{
    // Surface normal in 2D side view: n = normalize(vec2(-slope, 1.0)).
    // Light direction (toward sun): u_light_dir, already normalized on CPU.
    float inv_len = inversesqrt(1.0 + slope * slope);
    vec2 n = vec2(-slope * inv_len, inv_len);
    float lambert = dot(n, u_light_dir);
    // Map to [0.35, 1.30] so even shadow faces have some color
    return 0.35 + 0.95 * clamp(lambert * 0.5 + 0.5, 0.0, 1.0);
}}

// Per-layer parameters. Wavelengths are NOMINAL spacings — actual dune-to-
// dune distances vary because the input x is warped by low-freq noise
// (warp_scale_ft, warp_freq). Larger warp_scale → more aperiodic spacing.
const float BACK_BASE  = 5.6;
const float BACK_AMP   = 0.9;
const float BACK_WL    = 14.0;
const float BACK_WARP  = 4.0;    // ft, warp magnitude
const float BACK_WFREQ = 0.05;   // 1/ft, low spatial freq
const float MID_BASE   = 4.9;
const float MID_AMP    = 0.7;
const float MID_WL     = 7.0;
const float MID_WARP   = 2.5;
const float MID_WFREQ  = 0.10;
// Foreground dune spacing (peak-to-peak distance) varies in roughly 5..8 ft.
// Mean WL = 6.5 ft; warp_scale 0.75 ft per peak → adjacent-peak shifts up to
// ±1.5 ft → actual separations in [WL-1.5, WL+1.5] = [5.0, 8.0] ft.
// warp_freq 0.30 = ~3.3 ft per warp cycle so adjacent peaks see decorrelated
// warp values (without that, every peak shifts the same amount and the
// separation just stays at WL).
const float FR_BASE    = 4.4;
const float FR_AMP     = 0.5;
const float FR_WL      = 6.5;
const float FR_WARP    = 0.75;
const float FR_WFREQ   = 0.30;

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);

    // Drift offsets in feet — front layer migrates fastest, back slowest.
    // u_wind_phase is strictly proportional to signed wind. Coefficients
    // are NEGATIVE because dune_wave(x + drift) shifts the wave shape in
    // the -x direction as drift grows, so a negative drift coefficient
    // makes the dunes (and their ripples / lifted grit) migrate in the
    // SAME direction the wind is blowing.
    float drift_back = -u_wind_phase * 0.30;
    float drift_mid  = -u_wind_phase * 0.65;
    float drift_fr   = -u_wind_phase * 1.10;

    // Per-layer detail amplifier. Front kept at 1.0 — the four base
    // octaves only (no high-freq "extras"). With FR_AMP=0.5 that gives
    // ±0.43 ft of crest wiggle: visible horizontal undulation without
    // the spiky high-freq stack of the larger detail_scale values.
    const float BACK_DETAIL = 0.9;
    const float MID_DETAIL  = 1.2;
    const float FR_DETAIL   = 1.0;

    // Heights and slopes
    float h_back = ridge_h    (phys.x, BACK_BASE, BACK_AMP, BACK_WL, drift_back, BACK_WARP, BACK_WFREQ, BACK_DETAIL);
    float h_mid  = ridge_h    (phys.x, MID_BASE,  MID_AMP,  MID_WL,  drift_mid,  MID_WARP,  MID_WFREQ,  MID_DETAIL);
    float h_fr   = ridge_h    (phys.x, FR_BASE,   FR_AMP,   FR_WL,   drift_fr,   FR_WARP,   FR_WFREQ,   FR_DETAIL);
    float s_back = ridge_slope(phys.x, BACK_BASE, BACK_AMP, BACK_WL, drift_back, BACK_WARP, BACK_WFREQ, BACK_DETAIL);
    float s_mid  = ridge_slope(phys.x, MID_BASE,  MID_AMP,  MID_WL,  drift_mid,  MID_WARP,  MID_WFREQ,  MID_DETAIL);
    float s_fr   = ridge_slope(phys.x, FR_BASE,   FR_AMP,   FR_WL,   drift_fr,   FR_WARP,   FR_WFREQ,   FR_DETAIL);

    float a_back = dune_alpha(phys, h_back);
    float a_mid  = dune_alpha(phys, h_mid);
    float a_fr   = dune_alpha(phys, h_fr);

    // Day/night base color
    float day = 1.0 - abs(u_season - 0.5) * 2.0;
    vec3 warm = vec3(0.85, 0.55, 0.32);
    vec3 cool = vec3(0.10, 0.12, 0.22);
    vec3 base = mix(cool, warm, day);
    vec3 dune_color = mix(base, u_tint, 0.35);

    // Atmospheric perspective per layer
    vec3 col_back = dune_color * 1.10;
    vec3 col_mid  = dune_color * 0.90;
    vec3 col_fr   = dune_color * 0.65;

    // Slope shading (3D form)
    col_back *= slope_shading(s_back);
    col_mid  *= slope_shading(s_mid);
    col_fr   *= slope_shading(s_fr);

    // Surface texture (wind-driven streaks/ripples)
    col_back *= sand_surface(phys, drift_back, u_ripple_phase * 0.08);
    col_mid  *= sand_surface(phys, drift_mid,  u_ripple_phase * 0.16);
    col_fr   *= sand_surface(phys, drift_fr,   u_ripple_phase * 0.28);

    // Composite back-to-front
    vec4 c = vec4(0.0);
    c = mix(c, vec4(col_back, 1.0), a_back);
    c = mix(c, vec4(col_mid,  1.0), a_mid);
    c = mix(c, vec4(col_fr,   1.0), a_fr);

    // Lifted grit above the front crest at high wind. We're still operating
    // in pre-multiplied space here; treat grit as a coverage layer too.
    vec2 grit = lifted_grit(phys, h_fr, drift_fr);
    if (grit.y > 0.0) {{
        vec3 grit_color = dune_color * 1.4 + vec3(0.05);
        // Composite grit as a layer over c (in pre-multiplied space)
        c.rgb = c.rgb * (1.0 - grit.y) + grit_color * grit.y;
        c.a   = c.a   * (1.0 - grit.y) + grit.y;
    }}

    // Composition above accumulated in pre-multiplied form (mix() against a
    // (color, 1.0) layer with coverage `a` produces rgb = col*a, alpha = a).
    // The global blend function expects STRAIGHT alpha, so convert back here
    // — otherwise crest edges with 0 < a < 1 get dimmed by alpha² (see
    // docs/shader_info.txt "Alpha Output Rules").
    if (c.a > 0.001) c.rgb /= c.a;

    c.a *= u_strength;
    if (c.a < 0.01) discard;
    fragColor = c;        // straight alpha, per docs
}}
"""


def shader_desert_dunes(state, outstate, fade_duration=4.0):
    """Background event: layered dune silhouette across the horizon."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(DesertDunesEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing desert_dunes: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        eff.wind = float(outstate.get('wind', 0.0))
        # Track the GLOBAL day cycle so dune lighting stays in sync with the
        # sun's actual position in the sky (desert_sky uses the same signal).
        eff.season = float(outstate.get('season', 0.5))
        fog_color = outstate.get('fog_color', (0.7, 0.55, 0.35))
        eff.tint = (float(fog_color[0]), float(fog_color[1]), float(fog_color[2]))

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


class DesertDunesEffect(ShaderEffect):
    """Fullscreen-quad parallax dunes in physical fan space."""

    def __init__(self, viewport):
        super().__init__(viewport)
        # Foreground silhouette in front of sky+clouds (matches forest_canopy at 7.0)
        self.render_priority = 7.0
        self.wind = 0.0
        self.season = 0.5
        self.tint = (0.7, 0.55, 0.35)
        self.strength = 0.0
        # CPU-integrated phases — both monotonic by construction so transitions
        # in `wind` never cause backward drift (see docs/shader_info.txt
        # "Time-based Animation"). _wind_phase drives slow dune drift and
        # surface flow; _ripple_phase drives fast sub-second ripple shimmer.
        self._wind_phase = 0.0
        self._ripple_phase = 0.0
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

        # Static fan-coord uniforms
        glUseProgram(self.shader)
        self._fan.set_uniforms(self.shader)
        glUseProgram(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        # `outstate['wind']` is signed (`wind_speed * cos(2π(season-0.125))`)
        # — i.e., direction reverses across the day cycle. Use it directly
        # so dunes drift one way during morning winds and the other during
        # evening winds. Magnitude is what matters for visible motion, so
        # ripple_phase uses |wind|.
        w = self.wind
        self._wind_phase   += dt * 6.5 * w
        self._ripple_phase += dt * 9.0 * abs(w)

    def _light_direction(self):
        """Return a normalized 2D (lx, ly) pointing TOWARD the sun/moon,
        used by the shader for Lambert slope shading.

        The arc mirrors desert_sky's disc placement so dunes are lit from
        wherever the visible body is.
        """
        s = self.season
        if 0.25 <= s <= 0.75:
            t = (s - 0.25) / 0.5            # 0 sunrise → 1 sunset
        elif s >= 0.75:
            t = (s - 0.75) / 0.5             # 0 dusk → 0.5 midnight
        else:
            t = (s + 0.25) / 0.5             # 0.5 midnight → 1 dawn
        arc = math.sin(math.pi * t)          # 0..1..0
        # Sun position in feet (matches desert_sky._resolve_disc geometry)
        lx = -16.0 + 32.0 * t
        ly = 6.0 + 12.0 * arc
        # Vector from a reference dune (origin in physical space) to light
        n = math.hypot(lx, ly)
        if n < 1e-3:
            return (0.0, 1.0)
        return (lx / n, ly / n)

    def render(self, state: Dict):
        if not self.enabled or self.strength < 0.01:
            return

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        self._fan.set_uniforms(self.shader)

        lx, ly = self._light_direction()

        glUniform1f(glGetUniformLocation(self.shader, "u_wind_phase"), self._wind_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_ripple_phase"), self._ripple_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_wind"), float(self.wind))
        glUniform1f(glGetUniformLocation(self.shader, "u_season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, "u_strength"), self.strength)
        glUniform3f(glGetUniformLocation(self.shader, "u_tint"), *self.tint)
        glUniform2f(glGetUniformLocation(self.shader, "u_light_dir"), lx, ly)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
