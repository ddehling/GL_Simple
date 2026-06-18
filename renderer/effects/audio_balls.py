"""Audio-reactive balls with lightning - rendered in REAL fan space.

Each ball is tied to a frequency band and lives at a physical (x, y) position
in FEET on the semicircular fan (4 ft inner radius, 20.6 ft outer). A single
fullscreen pass converts every FBO pixel to its real fan position via
``renderer.fan_coords`` and draws the balls as true circles there - so on the
physical fan (and the fan preview) they read as round orbs, not the stretched
ellipses the old flat-pixel-space version produced. Lightning arcs between
high-energy balls are drawn as real-space line glows in the same pass.

This is Pattern B (fullscreen quad). It supersedes the old instanced-quad +
``squish_top_width`` approximation, which only crudely faked the fan widening.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FanCoords, FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL

MAX_BALLS = 32
MAX_BOLTS = 16

# ============================================================================
# Event Wrapper Function
# ============================================================================

def shader_audio_balls(state, outstate, sensitivity=2.0, base_size=8.0,
                       lightning_threshold=0.5, lightning_probability=0.3,
                       num_balls=16, depth=0.55,
                       mixer_op_key=None, mixer_tint_key=None):
    """Audio-reactive balls (real fan-space) with lightning arcs.

    Args:
        sensitivity: Multiplier for audio response.
        base_size: Legacy size handle; mapped to a ~1 ft base ball radius
            (base_size/8 -> ~1.1 ft) so existing call sites keep working.
        lightning_threshold / lightning_probability: lightning behaviour.
        num_balls: number of balls (max 32, one per frequency band).
        mixer_op_key / mixer_tint_key: opt-in VJ-mixer keys. When set (the club
            set passes "op_balls"/"tint_balls"), opacity follows that outstate
            value (default 0 -> invisible until the fader is raised) and the
            ball hues shift by the tint. When None (every other set), the balls
            stay fully visible with their natural per-band colours.
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    audio_data = outstate.get('sound')

    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(
                AudioBallsEffect,
                sensitivity=sensitivity,
                base_size=base_size,
                lightning_threshold=lightning_threshold,
                lightning_probability=lightning_probability,
                num_balls=num_balls,
                depth=depth,
            )
            state['effect'] = effect
            print(f"  [OK] Initialized audio_balls (real fan-space) for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize audio_balls: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        # VJ-mixer controls (opt-in): opacity from the fader, hue from the knob.
        if mixer_op_key:
            eff.set_opacity(float(outstate.get(mixer_op_key, 0.0)))
        else:
            eff.set_opacity(1.0)
        if mixer_tint_key:
            eff.set_tint(float(outstate.get(mixer_tint_key, 0.0))
                         + float(outstate.get('club_palette', 0.0)))
        else:
            eff.set_tint(0.0)

    if 'effect' in state and audio_data is not None:
        bands = audio_data['norm_short'][0]   # (32,)
        state['effect'].update_from_audio(bands)

        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        fade_duration = 2.0
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        state['effect'].fade_factor = float(np.clip(fade_factor, 0, 1))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ============================================================================
# Rendering Class
# ============================================================================

class AudioBallsEffect(ShaderEffect):
    """Audio balls + lightning, positioned in real fan feet, drawn as a
    single fullscreen real-space SDF pass."""

    def __init__(self, viewport, sensitivity: float = 2.0, base_size: float = 8.0,
                 lightning_threshold: float = 0.5, lightning_probability: float = 0.3,
                 num_balls: int = 16, depth: float = 0.55):
        super().__init__(viewport)
        self.sensitivity = sensitivity
        self.lightning_threshold = lightning_threshold
        self.lightning_probability = lightning_probability
        self.num_balls = int(min(num_balls, MAX_BALLS))
        self.fade_factor = 0.0
        self.z_centroid = float(depth)
        self._depth = float(depth)

        # Fan coordinate mapping (feet). Matches the physical install + every
        # other fan-aware effect, so the balls land where you'd expect.
        self.fan = FanCoords(viewport.width, viewport.height)
        self.inner_ft = self.fan.inner_r_ft        # 4.0
        self.outer_ft = self.fan.outer_r_ft        # 20.6
        # Base ball radius in feet (legacy base_size 8 px -> ~1.1 ft).
        self.base_radius_ft = max(0.4, (base_size / 8.0) * 1.1)

        self.band_step = max(1, 32 // self.num_balls)

        # Per-ball state.
        n = self.num_balls
        self.base_theta = np.zeros(n, dtype=np.float64)     # rad, [0..pi]
        self.theta = np.zeros(n, dtype=np.float64)
        self.base_radius = np.zeros(n, dtype=np.float64)    # feet
        self.radius = np.zeros(n, dtype=np.float64)         # feet
        self.ang_speed = np.zeros(n, dtype=np.float64)      # rad/s
        self.wave_freq = np.linspace(1.0, 3.0, n)
        self.wave_amp_ft = 2.0
        self.positions_ft = np.zeros((n, 2), dtype=np.float32)
        self.colors = np.ones((n, 3), dtype=np.float32)
        self.alphas = np.full(n, 0.5, dtype=np.float32)
        self.sizes_ft = np.full(n, self.base_radius_ft, dtype=np.float32)
        self.energies = np.zeros(n, dtype=np.float32)
        self.smoothed_energies = np.zeros(n, dtype=np.float32)
        self.energy_smoothing = 0.15

        # VJ-mixer master opacity + hue tint. Default opacity 1.0 so non-club
        # sets show the balls normally; the club layer sets it from the fader.
        self.opacity = 1.0
        self.tint = 0.0
        self.base_hues = np.zeros(n, dtype=np.float32)

        self.surface_time = 0.0
        self.active_lightning = []   # (i, j, intensity, age)

        # Local RNG so per-frame lightning rolls don't shift the engine's
        # global np.random stream (see np_random_global_state_pollution).
        self._rng = np.random.default_rng()

        self.VAO = None
        self._initialize_balls()

    def _initialize_balls(self):
        n = self.num_balls
        frac = np.arange(n, dtype=np.float64) / max(n - 1, 1)
        # Spread across the arc (left pi -> right 0), avoiding the extreme edges.
        self.base_theta[:] = np.pi * (0.92 - frac * 0.84)
        self.theta[:] = self.base_theta
        # Spread base radius inner -> outer (a diagonal sweep across the fan).
        self.base_radius[:] = (self.inner_ft + 2.0) + frac * (self.outer_ft - self.inner_ft - 4.0)
        self.radius[:] = self.base_radius
        # Drift speed in angle (rad/s); negative -> sweep toward the right.
        self.ang_speed[:] = -self._rng.uniform(0.10, 0.30, size=n)

        band_indices = np.minimum((np.arange(n) * self.band_step).astype(np.int32), 31)
        self.base_hues[:] = (band_indices / 32.0) % 1.0
        self._apply_tint()
        self._recompute_positions()

    def set_opacity(self, v):
        self.opacity = float(v)

    def set_tint(self, v):
        if abs(float(v) - self.tint) > 1e-4:
            self.tint = float(v)
            self._apply_tint()

    def _apply_tint(self):
        """Recompute ball colours as base per-band hue shifted by the tint knob."""
        self.colors[:] = self._hsv_to_rgb_vectorized(
            (self.base_hues + self.tint) % 1.0, s=0.85, v=1.0)

    def _recompute_positions(self):
        self.positions_ft[:, 0] = (self.radius * np.cos(self.theta)).astype(np.float32)
        self.positions_ft[:, 1] = (self.radius * np.sin(self.theta)).astype(np.float32)

    def _hsv_to_rgb_vectorized(self, h, s=1.0, v=1.0):
        h = np.asarray(h, dtype=np.float32).flatten() % 1.0
        s = np.full_like(h, s) if np.ndim(s) == 0 else np.asarray(s, np.float32)
        v = np.full_like(h, v) if np.ndim(v) == 0 else np.asarray(v, np.float32)
        i = (h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        rgb = np.zeros((len(h), 3), dtype=np.float32)
        for k, cols in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                  (p, q, v), (t, p, v), (v, p, q)]):
            m = i == k
            rgb[m] = np.column_stack([cols[0][m], cols[1][m], cols[2][m]])
        return rgb

    # -- audio + animation --------------------------------------------------

    def update_from_audio(self, bands: np.ndarray):
        band_indices = np.minimum(
            (np.arange(self.num_balls) * self.band_step).astype(np.int32), 31)
        self.energies[:] = bands[band_indices]
        self.smoothed_energies[:] = (
            self.smoothed_energies * (1.0 - self.energy_smoothing)
            + self.energies * self.energy_smoothing)
        smoothed = self.smoothed_energies
        # Ball radius (feet) and brightness grow with band energy.
        self.sizes_ft[:] = self.base_radius_ft * (1.0 + smoothed * self.sensitivity * 0.3)
        self.alphas[:] = 0.3 + 0.5 * np.clip(smoothed, 0, 1)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self.surface_time += dt
        # Drift in angle, wrap into [0, pi] (sweep across the fan arc).
        self.theta[:] = np.mod(self.theta + self.ang_speed * dt, np.pi)
        # Bob in radius; keep inside the fan annulus.
        bob = self.wave_amp_ft * np.sin(2 * np.pi * self.wave_freq * self.surface_time / 10.0)
        self.radius[:] = np.clip(self.base_radius + bob,
                                 self.inner_ft + 0.5, self.outer_ft - 0.5)
        self._recompute_positions()
        self._update_lightning(dt)

    def _update_lightning(self, dt: float):
        # Decay existing arcs.
        if self.active_lightning:
            kept = []
            for (i, j, inten, age) in self.active_lightning:
                inten *= 0.8
                age += dt
                if inten > 0.05 and age < 0.5:
                    kept.append((i, j, inten, age))
            self.active_lightning = kept

        high = np.where(self.energies >= self.lightning_threshold)[0]
        if len(high) < 2:
            return
        pos = self.positions_ft[high]                      # (m, 2) feet
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))          # feet
        idx_diff = high[:, None] - high[None, :]
        upper = np.triu(np.ones_like(dist, dtype=bool), k=1)
        nearby = (dist < 8.0) | (np.abs(idx_diff) == 1)    # within ~8 ft or adjacent band
        vi, vj = np.where(upper & nearby)
        if len(vi) == 0:
            return
        sel = self._rng.random(len(vi)) < self.lightning_probability
        vi, vj = vi[sel], vj[sel]
        existing = {(min(a, b), max(a, b)) for a, b, _, _ in self.active_lightning}
        for a_idx, b_idx in zip(high[vi], high[vj]):
            pair = (min(int(a_idx), int(b_idx)), max(int(a_idx), int(b_idx)))
            if pair in existing or len(self.active_lightning) >= MAX_BOLTS:
                continue
            inten = float(np.clip(self.energies[a_idx] * self.energies[b_idx], 0, 1))
            self.active_lightning.append((pair[0], pair[1], inten, 0.0))
            existing.add(pair)

    def _build_bolt_uniforms(self):
        if not self.active_lightning:
            return 0, None, None
        bolts = self.active_lightning[:MAX_BOLTS]
        k = len(bolts)
        seg = np.zeros((k, 4), dtype=np.float32)
        col = np.zeros((k, 4), dtype=np.float32)
        for n, (i, j, inten, age) in enumerate(bolts):
            seg[n, 0:2] = self.positions_ft[i]
            seg[n, 2:4] = self.positions_ft[j]
            hue = ((i + j) / (self.num_balls * 2.0)) % 1.0
            rgb = self._hsv_to_rgb_vectorized([hue], s=0.5, v=1.0)[0]
            rgb = rgb * (1.0 - inten * 0.4) + np.float32(1.0) * (inten * 0.4)  # toward white
            col[n, 0:3] = rgb
            col[n, 3] = inten * 0.8 * self.fade_factor
        return k, seg, col

    # -- GL -----------------------------------------------------------------

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(self._VERT, GL_VERTEX_SHADER),
            shaders.compileShader(self._FRAG, GL_FRAGMENT_SHADER),
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

    def render(self, state: Dict):
        super().render(state)
        if not self.shader or self.opacity < 0.004:
            return  # faded out -> skip entirely
        # Foreground real-space overlay: always composite, never write depth.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        self.fan.set_uniforms(self.shader)
        glUniform1f(self.uniform(self.shader, "u_depth"), self._depth)
        glUniform1f(self.uniform(self.shader, "u_time"), self.surface_time)
        glUniform1f(self.uniform(self.shader, "u_opacity"), self.opacity)

        n = self.num_balls
        ball = np.zeros((n, 4), dtype=np.float32)
        ball[:, 0:2] = self.positions_ft
        ball[:, 2] = self.sizes_ft
        ball[:, 3] = self.alphas * self.fade_factor
        glUniform1i(self.uniform(self.shader, "u_numBalls"), n)
        glUniform4fv(self.uniform(self.shader, "u_ball"), n, ball.flatten())
        glUniform3fv(self.uniform(self.shader, "u_ballColor"), n, self.colors.flatten())

        k, seg, col = self._build_bolt_uniforms()
        glUniform1i(self.uniform(self.shader, "u_numBolts"), k)
        if k > 0:
            glUniform4fv(self.uniform(self.shader, "u_bolt"), k, seg.flatten())
            glUniform4fv(self.uniform(self.shader, "u_boltColor"), k, col.flatten())

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)

    # -- shaders ------------------------------------------------------------

    _VERT = """
    #version 310 es
    precision highp float;
    in vec2 position;
    out vec2 v_uv;
    uniform float u_depth;
    void main() {
        v_uv = position * 0.5 + 0.5;
        gl_Position = vec4(position, u_depth, 1.0);
    }
    """

    _FRAG = f"""
    #version 310 es
    precision highp float;
    in vec2 v_uv;
    out vec4 fragColor;

    {FAN_COORDS_UNIFORMS}

    uniform float u_time;
    uniform float u_opacity;                // layer master opacity (MIDI slider)
    uniform int  u_numBalls;
    uniform vec4 u_ball[{MAX_BALLS}];       // xy=center ft, z=radius ft, w=alpha
    uniform vec3 u_ballColor[{MAX_BALLS}];
    uniform int  u_numBolts;
    uniform vec4 u_bolt[{MAX_BOLTS}];       // p1.xy, p2.xy (ft)
    uniform vec4 u_boltColor[{MAX_BOLTS}];  // rgb, alpha

    {FAN_COORDS_GLSL}

    float hash1(float n) {{ return fract(sin(n) * 43758.5453123); }}
    float noise1(float x) {{
        float i = floor(x); float f = fract(x);
        float u = f * f * (3.0 - 2.0 * f);
        return mix(hash1(i), hash1(i + 1.0), u);
    }}
    float distSeg(vec2 p, vec2 a, vec2 b) {{
        vec2 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        return length(pa - ba * h);
    }}

    void main() {{
        vec2 phys = fan_uv_to_physical(v_uv);   // feet
        vec3 col = vec3(0.0);
        float a = 0.0;

        for (int i = 0; i < {MAX_BALLS}; i++) {{
            if (i >= u_numBalls) break;
            float R = max(u_ball[i].z, 0.05);
            vec2 local = (phys - u_ball[i].xy) / R;
            float d = length(local);
            if (d > 1.7) continue;

            float sphere = max(0.3, 1.0 - d * d * 0.5);
            float ang = atan(local.y, local.x);
            float ripple = sin(d * 8.0 - u_time * 3.0) * 0.25
                         + sin(ang * 5.0 + u_time * 2.5) * 0.15
                         + noise1(u_time * 0.7 + float(i)) * 0.12;
            sphere = clamp(sphere + ripple * (1.0 - d), 0.3, 1.0);

            float spec = exp(-(d - 0.55) * (d - 0.55) * 14.0);
            spec *= smoothstep(0.3, 0.1, abs(sin((ang - u_time * 2.0) * 3.0)));
            spec += exp(-d * d * 7.0) * 0.4;

            vec3 c = u_ballColor[i] * sphere + vec3(0.4) * spec;
            float core = smoothstep(1.05, 0.9, d);
            float glow = exp(-(d * d)) * 0.45;
            float inten = clamp(core + glow, 0.0, 1.5);

            col += c * inten * u_ball[i].w;
            a = max(a, inten * u_ball[i].w);
        }}

        for (int i = 0; i < {MAX_BOLTS}; i++) {{
            if (i >= u_numBolts) break;
            float d = distSeg(phys, u_bolt[i].xy, u_bolt[i].zw);
            float w = 0.5;  // line half-width feet
            float line = exp(-(d * d) / (w * w));
            col += u_boltColor[i].rgb * line * u_boltColor[i].a;
            a = max(a, line * u_boltColor[i].a);
        }}

        a = clamp(a, 0.0, 1.0) * u_opacity;
        if (a < 0.01) discard;
        fragColor = vec4(col, a);
    }}
    """

    def cleanup(self):
        super().cleanup()
