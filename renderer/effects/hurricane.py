"""
Hurricane shader effect - A rotating cyclonic storm system.

Renders a logarithmic-spiral band structure around a calm eye, with roiling
fBm texture in the bands and a dark green-grey storm palette. Rotates
steadily; wind speed from global state pushes the rotation faster and
strengthens band contrast. Designed for the Storm World weather set.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FanCoords, FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL


# ============================================================================
# Event Wrapper
# ============================================================================

def shader_hurricane(state, outstate, eye_x_ft=0.0, eye_y_ft=0.0,
                     num_arms=3, spiral_tightness=1.4, intensity=1.0,
                     depth=75.0):
    """
    Hurricane shader effect compatible with EventScheduler.

    Args:
        state: Event state dict
        outstate: Global state dict (reads 'wind' and 'lightning_flash')
        eye_x_ft: Horizontal position of the eye in PHYSICAL FEET on the fan
                  (default 0.0 = fan center / apex of the semicircle)
        eye_y_ft: Vertical position of the eye in PHYSICAL FEET on the fan
                  (default 0.0 = on the fan's diameter, so the eye sits at
                  the bottom of the visible semicircle)
        num_arms: Number of spiral bands (default 3)
        spiral_tightness: Log-spiral winding factor (higher = tighter coil)
        intensity: Overall opacity multiplier
        depth: Z-depth (0=near, 100=far)
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')

    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return

    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return

    if state['count'] == 0:
        print(f"Initializing hurricane for frame {frame_id}")
        try:
            effect = viewport.add_effect(
                HurricaneEffect,
                eye_x_ft=eye_x_ft,
                eye_y_ft=eye_y_ft,
                num_arms=num_arms,
                spiral_tightness=spiral_tightness,
                intensity=intensity,
                depth=depth,
            )
            state['effect'] = effect
            print(f"[OK] Initialized hurricane for frame {frame_id}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize hurricane: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        elapsed = state.get('elapsed_time', 0)
        total = state.get('duration', 60)
        fade_d = 4.0
        if elapsed < fade_d:
            fade = elapsed / fade_d
        elif elapsed > total - fade_d:
            fade = (total - elapsed) / fade_d
        else:
            fade = 1.0
        eff.fade_factor = float(np.clip(fade, 0.0, 1.0))

        # ---- Dynamic coupling ----
        # Rotation follows the SIGNED wind so the hurricane always turns
        # the same way the wind is pushing clouds -- the eye is the wind
        # field, it shouldn't fight it. To avoid the jarring "snap reverse"
        # when the seasonal wind crosses zero, the rate is low-pass
        # filtered in the effect itself: it smoothly decelerates, drifts
        # through zero, and spins up in the new direction.
        wind_signed = float(outstate.get('wind', 0.0))
        wind_abs    = abs(wind_signed)
        rain        = float(outstate.get('rain', 0.0))
        cloudyness  = float(outstate.get('cloudyness', 0.5))

        # Ferocity (unsigned) drives non-rotation effects: band contrast,
        # eye pinch, palette darkness.
        ferocity = np.clip(0.45 * wind_abs + 0.55 * rain + 0.15 * cloudyness,
                           0.0, 1.0)
        eff.ferocity = float(ferocity)
        eff.wind_speed = wind_abs
        eff.rain_rate = rain

        # Target rotation rate in rad/s. Signed by wind so the swirl
        # direction matches cloud drift; rain adds a magnitude kick in
        # the same direction as the wind.
        wind_sign = 1.0 if wind_signed >= 0.0 else -1.0
        eff.target_rate = float(wind_signed * 1.5 + wind_sign * rain * 0.9)

        # Optional: a 0..1 flash value from the lightning effect if wired up.
        eff.lightning_flash = float(outstate.get('lightning_flash', 0.0))

    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up hurricane for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"[OK] Cleaned up hurricane for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class HurricaneEffect(ShaderEffect):
    """Swirling cyclonic storm with a calm eye and rotating spiral arms."""

    def __init__(self, viewport, eye_x_ft=0.0, eye_y_ft=0.0, num_arms=4,
                 spiral_tightness=2.2, intensity=1.0, depth=75.0):
        super().__init__(viewport)
        self.fan = FanCoords(viewport.width, viewport.height)
        self.eye_x_ft = float(eye_x_ft)
        self.eye_y_ft = float(eye_y_ft)
        self.num_arms = int(num_arms)
        self.spiral_tightness = float(spiral_tightness)
        self.base_intensity = float(intensity)
        self.depth = float(depth)
        self.time = 0.0
        self.fade_factor = 0.0
        self.wind_speed = 0.0
        self.rain_rate = 0.0
        self.ferocity = 0.0
        # Accumulated rotation angle in radians, integrated on the CPU.
        # Current rate is low-pass filtered toward target_rate so that
        # when the seasonal wind crosses zero, the storm smoothly decel
        # to a pause and spins up the other way rather than snapping.
        self.rot_angle = 0.0
        self.rot_rate = 0.0
        self.target_rate = 0.0
        self.lightning_flash = 0.0
        # Render behind most foreground effects but above the plain sky.
        self.render_priority = 4

    def compile_shader(self):
        vertex_src = """
        #version 310 es
        precision highp float;
        layout(location = 0) in vec2 position;
        uniform float depth;
        out vec2 vUV;
        void main() {
            vUV = position * 0.5 + 0.5;
            gl_Position = vec4(position, clamp(depth / 100.0, 0.0, 1.0), 1.0);
        }
        """

        fragment_src = f"""
        #version 310 es
        precision highp float;

        in vec2 vUV;
        out vec4 fragColor;

        uniform float u_time;
        uniform float u_fade;
        uniform float u_intensity;
        uniform float u_wind;
        uniform float u_rain;
        uniform float u_ferocity;
        uniform float u_rot;        // CPU-integrated rotation angle (radians)
        uniform float u_flash;
        uniform vec2  u_eye_ft;
        uniform float u_num_arms;
        uniform float u_spiral;

        {FAN_COORDS_UNIFORMS}
        {FAN_COORDS_GLSL}

        const float TAU = 6.28318530718;

        float hash(vec2 p) {{
            p = fract(p * vec2(443.897, 441.423));
            p += dot(p, p + vec2(19.19, 23.41));
            return fract(sin(p.x * p.y) * 43758.5453);
        }}
        float noise(vec2 p) {{
            vec2 i = floor(p), f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(mix(hash(i),               hash(i + vec2(1, 0)), f.x),
                       mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
        }}
        float fbm(vec2 p) {{
            float v = 0.0, a = 0.5;
            for (int i = 0; i < 5; i++) {{
                v += a * noise(p);
                p  = p * 2.05 + vec2(3.7, 1.3);
                a *= 0.5;
            }}
            return v;
        }}

        void main() {{
            // Work entirely in PHYSICAL FEET on the fan. This makes the
            // hurricane geometrically correct on the semicircle: the eye is
            // a round feature in real-world space, bands curve with physical
            // distance from the eye, and scales stay consistent no matter
            // where the eye is positioned.
            vec2  xy_ft = fan_uv_to_physical(vUV);
            vec2  d_ft  = xy_ft - u_eye_ft;
            float r     = length(d_ft);               // feet from eye
            float theta = atan(d_ft.y, d_ft.x);

            // Rotation comes pre-integrated from the CPU (u_rot, radians),
            // so direction reversals driven by wind-sign flips are smooth
            // rather than phase-jumping.
            float rot = u_rot;

            float logR = log(max(r, 0.25));

            // Base arm coordinate along a log spiral. All radii rotate at
            // the same rate -- earlier code applied a radius-dependent
            // rotation boost to inner arms, which looked great for a few
            // seconds but accumulated over time until the spiral wound
            // itself into concentric rings. Fixed rate keeps the spiral
            // spiral-shaped indefinitely.
            float arm_raw = theta - logR * u_spiral + rot;

            // ---- Domain-warp the arm coordinate with fBm ----
            // This is what actually makes it stop looking programmatic.
            // Instead of following a perfect mathematical spiral, the arms
            // wiggle organically -- the turbulence deflects the arm
            // coordinate itself, so the resulting bands aren't just
            // TEXTURED with noise, they're SHAPED by it.
            vec2  warpP = vec2(arm_raw * 0.35 + u_time * 0.03, logR * 1.2);
            float warp1 = fbm(warpP) - 0.5;
            float warp2 = fbm(warpP * 2.3 + vec2(7.1, 3.9) + u_time * 0.07) - 0.5;
            float arm   = arm_raw + (warp1 * 1.4 + warp2 * 0.55);

            // ---- Rain bands: thin asymmetric crescents ----
            float armCos  = cos(arm * u_num_arms);
            float armEdge = pow(max(armCos, 0.0), 2.2);

            // ---- Arm breakage ----
            // Gate the crescents by a noise mask so the bands fragment
            // into puffs and gaps instead of being continuous ribbons.
            vec2  breakP    = vec2(arm_raw * 1.6 + u_time * 0.14, logR * 2.1);
            float breakMask = smoothstep(0.28, 0.72, fbm(breakP));
            armEdge        *= mix(0.20, 1.10, breakMask);

            // ---- Lopsided storm ----
            // Slowly drifting one-lobed mask that thickens bands on one
            // side of the eye at any given moment, just like a real
            // hurricane's asymmetric rain shield.
            float asym = 0.5 + 0.5 * cos(theta - u_time * 0.08);
            armEdge *= 0.70 + 0.55 * asym;

            // ---- Inner / outer cirrus layer ----
            // Outer high cirrus drifts with a looser pitch and slower
            // rotation than the rain bands, sliding as a separate layer.
            float outerArm  = theta - logR * (u_spiral * 0.55) + rot * 0.55;
            float cirrusWav = pow(max(cos(outerArm * (u_num_arms + 2.0)), 0.0), 3.2);

            // ---- Multi-scale turbulence ----
            // Sample noise in (log r, arm) so cells stay roughly the same
            // physical size along the spiral.
            vec2  p1 = vec2(logR * 2.6 + u_time * 0.05, arm * 0.65);
            float n1 = fbm(p1);
            vec2  p2 = vec2(logR * 7.5 + u_time * 0.11, arm * 1.7);
            float n2 = fbm(p2);

            // ---- Radial masks (physical feet) ----
            // With the eye at the fan origin (0,0), the fan's inner edge
            // sits at r = 4 ft. The eye itself lives inside the fan's
            // inner hole (invisible), so the eyewall is the first feature
            // the fan actually renders -- we anchor it right at the inner
            // edge for a dramatic bright ring across the apex of the
            // display.
            //
            // Eye: opaque dark disc extending just past the inner hole so
            // you "see down into" the clear eye at the apex of the display.
            // Ferocity pinches the eye tighter (a more intense storm has a
            // smaller, sharper eye) and shoves the eyewall slightly inward.
            float eyePinch = 1.0 - 0.25 * u_ferocity;
            float eyeOpen = smoothstep(3.5 * eyePinch, 4.8 * eyePinch, r);
            float inEye   = 1.0 - eyeOpen;

            // Eyewall: narrow bright gaussian ring sitting exactly at the
            // fan's inner edge -- the single most iconic hurricane feature.
            // Ferocity narrows the wall (sharper, brighter edge).
            float eyewallR_ft = 5.0 * eyePinch;
            float eyewallW_ft = 1.1 * (1.0 - 0.30 * u_ferocity);
            float eyewallD    = (r - eyewallR_ft) / eyewallW_ft;
            float eyewall     = exp(-eyewallD * eyewallD)
                              * (1.0 + 0.5 * u_ferocity);

            // Central Dense Overcast: thick solid cloud deck just outside
            // the eyewall, trailing out to ~10 ft.
            float cdo = (1.0 - smoothstep(4.5, 10.0, r));

            // Overall outer falloff (feet). The fan's outer edge is at
            // 20.6 ft, so the storm should still be carrying some mass at
            // that radius but start fading a few feet before.
            float outerFade = 1.0 - smoothstep(17.0, 22.5, r);

            // ---- Compose cloud mass ----
            // Rain bands live just past the eyewall out to the outer rim.
            float bandAnnulus = smoothstep(4.8, 7.0, r) * (1.0 - smoothstep(15.0, 20.0, r));
            float bandMass    = armEdge * (0.55 + 0.55 * n1) * bandAnnulus;
            // Ferocity dramatically boosts band contrast and density:
            // calm weather has soft low-contrast bands, a full storm has
            // sharp dense ones. Rain rate adds a separate density kick.
            bandMass *= 0.55 + 0.85 * u_ferocity + 0.25 * u_rain;

            // Cirrus outflow: thin wispy density at outer radii.
            float cirrusMass = cirrusWav * smoothstep(9.0, 16.0, r) * (0.35 + 0.65 * n2)
                              * outerFade;

            // Core mass: thick CDO + eyewall highlight.
            float coreMass = (cdo * (0.55 + 0.35 * n1) + eyewall * 0.95) * eyeOpen;

            float density = coreMass + bandMass * 0.85 + cirrusMass * 0.50;
            // Fine detail sugar on the densest regions.
            density += n2 * 0.18 * density;
            density  = clamp(density * outerFade, 0.0, 1.0);

            // ---- Palette ----
            // Stormy gradient from near-black through blue-grey to bright
            // cloud-top white. Olive tint in the rain bands gives the
            // classic "tornado sky" hue without recoloring the whole frame.
            vec3 deep      = vec3(0.02, 0.03, 0.06);
            vec3 cloudLow  = vec3(0.10, 0.12, 0.17);
            vec3 cloudMid  = vec3(0.32, 0.38, 0.44);
            vec3 cloudTop  = vec3(0.78, 0.82, 0.86);
            vec3 olive     = vec3(0.18, 0.20, 0.10);

            vec3 color = mix(deep, cloudLow, density);
            color = mix(color, cloudMid, smoothstep(0.28, 0.62, density));
            color = mix(color, cloudTop, smoothstep(0.68, 0.96, density));
            color = mix(color, olive, clamp(bandMass, 0.0, 1.0) * 0.35);
            // Ferocity darkens the palette -- an intense storm reads
            // more ominous and slate-grey than a calm overcast.
            color *= 1.0 - 0.30 * u_ferocity;

            // Eyewall gets an extra bright rim because those walls of
            // cumulonimbus catch the most light.
            color = mix(color, vec3(0.90, 0.92, 0.95), eyewall * 0.55);

            // Pure dark over the eye: peering down through the clear center.
            vec3 eyeDark = vec3(0.02, 0.04, 0.08);
            color = mix(color, eyeDark, inEye);

            // Lightning tint: if a flash is active, the whole cloud mass
            // brightens and shifts cool-blue, as if lit from inside.
            vec3 flashTint = vec3(0.6, 0.75, 1.0);
            float flashAmt = clamp(u_flash, 0.0, 1.0);
            color = mix(color, flashTint, flashAmt * 0.40 * density);

            // ---- Alpha ----
            // Dense cloud opaque; the eye is also opaque (so you see through
            // to the "sea" beneath) rather than transparent back to the sky.
            float alpha = clamp(density * 1.15 + eyewall * 0.30 + inEye * outerFade * 0.70,
                                0.0, 1.0);
            alpha *= u_fade * u_intensity;

            fragColor = vec4(color, alpha);
        }}
        """

        return shaders.compileProgram(
            shaders.compileShader(vertex_src, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(vbo)
        glBindVertexArray(0)

        # Upload static fan-geometry uniforms so fan_uv_to_physical works.
        glUseProgram(self.shader)
        self.fan.set_uniforms(self.shader)
        glUseProgram(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self.time += dt

        # Low-pass filter the rate toward target_rate. Tau ~ 3s means
        # a sign flip takes about 10 seconds to fully reverse, with the
        # storm visibly slowing through zero.
        tau = 3.0
        alpha = 1.0 - float(np.exp(-dt / tau))
        self.rot_rate += (self.target_rate - self.rot_rate) * alpha

        # Small always-on spin in the current direction so the storm
        # never looks frozen even during low-wind moments. Sign of this
        # bias follows the sign of rot_rate so it keeps momentum through
        # the zero-crossing instead of stalling.
        sign = 1.0 if self.rot_rate >= 0.0 else -1.0
        bias = sign * 0.20
        # Breathing wobble (always small, preserves sign).
        breath = 0.10 * sign * float(np.sin(self.time * 0.35))

        self.rot_angle += dt * (self.rot_rate + bias + breath)

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return

        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        def _u1(name, v):
            loc = glGetUniformLocation(self.shader, name)
            if loc != -1:
                glUniform1f(loc, v)

        def _u2(name, x, y):
            loc = glGetUniformLocation(self.shader, name)
            if loc != -1:
                glUniform2f(loc, x, y)

        _u1("depth", self.depth)
        _u1("u_time", self.time)
        _u1("u_fade", self.fade_factor)
        _u1("u_intensity", self.base_intensity)
        _u1("u_wind", self.wind_speed)
        _u1("u_rain", self.rain_rate)
        _u1("u_ferocity", self.ferocity)
        _u1("u_rot", self.rot_angle)
        _u1("u_flash", self.lightning_flash)
        _u2("u_eye_ft", self.eye_x_ft, self.eye_y_ft)
        _u1("u_num_arms", float(self.num_arms))
        _u1("u_spiral", self.spiral_tightness)

        glDepthMask(GL_FALSE)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDepthMask(GL_TRUE)

        glBindVertexArray(0)
        glUseProgram(0)
