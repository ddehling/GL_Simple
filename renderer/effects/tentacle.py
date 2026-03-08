"""
Tentacle shader effect - Multiple jellyfish tentacles drifting from the top.

Renders 8 thin, translucent tentacles via SDF on a fullscreen quad.
Each tentacle is a spine curve with traveling-wave undulation (waves propagate
toward the tip, as in a real cephalopod/jellyfish). Purple/violet body with
bioluminescent cyan tip glow. No mesh geometry needed.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

N_TENTACLES = 8

# ============================================================================
# Event Wrapper
# ============================================================================

def shader_tentacle(state, outstate, depth=50.0, intensity=1.0, speed=1.0,
                    width_ratio=0.15, extend_duration=10.0, withdraw_duration=10.0):
    """
    Multiple jellyfish tentacle effect compatible with EventScheduler.

    Creates 8 thin translucent tentacles that extend from the top of screen,
    undulate with traveling waves, and withdraw at the end.

    Args:
        state: Event state dict
        outstate: Global state dict
        depth: Z-depth (0=near, 100=far, default: 50)
        intensity: Brightness multiplier (default: 1.0)
        speed: Animation speed multiplier (default: 1.0)
        width_ratio: Unused (kept for API compatibility)
        extend_duration: Time in seconds to fully extend (default: 10.0)
        withdraw_duration: Time in seconds to withdraw (default: 10.0)
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
        print(f"Initializing tentacle for frame {frame_id}")
        try:
            effect = viewport.add_effect(
                Tentacle,
                depth=depth,
                intensity=intensity,
                speed=speed,
                extend_duration=extend_duration,
                withdraw_duration=withdraw_duration,
            )
            state['effect'] = effect
            print(f"[OK] Initialized tentacle for frame {frame_id}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize tentacle: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        elapsed       = state['elapsed_time']
        total         = state.get('duration', 60)

        if elapsed < extend_duration:
            t          = elapsed / extend_duration
            extension  = t * t * (3.0 - 2.0 * t)      # smoothstep
        elif elapsed > (total - withdraw_duration):
            t          = (total - elapsed) / withdraw_duration
            extension  = t * t * (3.0 - 2.0 * t)
        else:
            extension  = 1.0

        state['effect'].extension_factor = extension
        state['effect'].base_intensity   = outstate.get('tentacle_intensity', intensity)
        state['effect'].base_speed       = outstate.get('tentacle_speed', speed)

    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up tentacle for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"[OK] Cleaned up tentacle for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class Tentacle(ShaderEffect):
    """
    Fullscreen SDF tentacle renderer.

    8 thin tentacles are defined by analytical spine curves; the fragment
    shader evaluates each pixel's distance to each spine and composites
    using premultiplied Porter-Duff over.
    """

    def __init__(self, viewport, depth: float = 50.0, intensity: float = 1.0,
                 speed: float = 1.0, extend_duration: float = 10.0,
                 withdraw_duration: float = 10.0):
        super().__init__(viewport)
        self.depth            = depth
        self.base_intensity   = intensity
        self.base_speed       = speed
        self.extend_duration  = extend_duration
        self.withdraw_duration = withdraw_duration
        self.time             = 0.0
        self.extension_factor = 0.0
        self.fade_factor      = 1.0   # required by ShaderEffect render contract
        self.y_start          = 0.0   # UV y where tentacles attach (top of screen)
        self.max_length       = 0.72  # maximum UV length they extend downward

        rng = np.random.default_rng(17)

        # Base X positions in UV [0, 1] — evenly spaced with slight jitter
        self.base_x = (np.linspace(0.08, 0.92, N_TENTACLES)
                       + rng.uniform(-0.04, 0.04, N_TENTACLES))
        self.base_x = np.clip(self.base_x, 0.04, 0.96).astype(np.float32)

        # Per-tentacle random phase offsets for independent motion
        self.phases = rng.uniform(0.0, 2.0 * np.pi, N_TENTACLES).astype(np.float32)

    # ------------------------------------------------------------------
    # Shader
    # ------------------------------------------------------------------

    def compile_shader(self):
        n = N_TENTACLES

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

        uniform float time;
        uniform float speed;
        uniform float fadeAlpha;
        uniform float intensity;
        uniform float extensionFactor;
        uniform float tentBaseX[{n}];
        uniform float tentPhase[{n}];
        uniform float tentStart;    // UV y where tentacles originate
        uniform float tentMaxLen;   // maximum UV length they can extend

        void main() {{
            float px = vUV.x;
            float py = vUV.y;

            // Accumulated output in premultiplied alpha (Porter-Duff over, back-to-front)
            vec4 dst = vec4(0.0);

            for (int i = 0; i < {n}; i++) {{
                // t: normalised depth along this tentacle [0=base, 1=full extension]
                float t   = (py - tentStart) / tentMaxLen;
                float ph  = tentPhase[i];
                float spd = speed;

                // Pixel is above the attachment point or below the current tip
                if (t < 0.0 || t > extensionFactor) continue;

                // ---- Spine position ----
                // Three traveling-wave harmonics + a slow whole-body sway.
                // The phase term "-time*spd*freq" makes waves propagate downward
                // (increasing t direction), giving true tentacle undulation.
                float sway = 0.058 * sin(time * spd * 0.27 + ph) * t;
                float w1   = 0.046 * sin(3.8  * t - time * spd * 1.05 + ph);
                float w2   = 0.023 * sin(7.3  * t - time * spd * 0.70 + ph * 1.58 + 1.2);
                float w3   = 0.010 * sin(12.6 * t - time * spd * 1.32 + ph * 0.84 + 2.5);
                // Amplitude grows as sqrt(t): tip swings more than base
                float xSpine = tentBaseX[i] + sway + (w1 + w2 + w3) * sqrt(t + 0.04);

                // ---- Radius: wide at base, tapers to a fine tip ----
                float radius = mix(0.016, 0.0030, t);

                // ---- SDF distances ----
                float dist = abs(px - xSpine);
                float sdf  = clamp(1.0 - dist / radius,          0.0, 1.0);
                float glow = clamp(1.0 - dist / (radius * 5.5),  0.0, 1.0);
                glow = pow(glow, 2.2);

                // Early-out: pixel not near this tentacle
                if (sdf < 0.001 && glow < 0.004) continue;

                // ---- Fades ----
                // Tip grows in as extensionFactor increases; soft fade at growing end
                float tipFade  = smoothstep(extensionFactor + 0.02,
                                            extensionFactor - 0.04, t);
                float baseFade = smoothstep(0.0, 0.022, t);
                float fade     = tipFade * baseFade;

                // ---- Subtle segmentation bands (move slowly along tentacle) ----
                float segs = 0.88 + 0.12 * sin(t * 48.0 - time * spd * 0.42 + ph);

                // ---- Colour ----
                // Body: purple-violet, fades to translucent cyan at the tip
                float tipFrac = pow(clamp(t / max(extensionFactor, 0.01), 0.0, 1.0), 2.0);
                vec3 bodyCol  = mix(vec3(0.40, 0.10, 0.62), vec3(0.20, 0.05, 0.40), t);
                vec3 tipCol   = vec3(0.35, 0.80, 0.94);   // bioluminescent cyan-white
                bodyCol = mix(bodyCol, tipCol, tipFrac * 0.55);

                // Rim highlight: light scattering through translucent tissue
                float rim = sdf * (1.0 - sdf) * 4.0;      // peaks at the edge midpoint
                bodyCol  += vec3(0.55, 0.20, 0.88) * rim * 0.55;

                // Glow halo colour
                vec3 glowCol = mix(vec3(0.20, 0.04, 0.45), tipCol * 0.5, tipFrac);

                // Composite body + halo into one colour/alpha for this tentacle
                vec3  col   = bodyCol * sdf + glowCol * glow * (1.0 - sdf * 0.8);
                float alpha = (sdf * 0.82 + glow * 0.18) * fade * segs;

                // ---- Premultiplied Porter-Duff over ----
                vec4 src = vec4(col * alpha, alpha);
                dst = src + dst * (1.0 - alpha);
            }}

            // Unpack premultiplied alpha for final output
            vec3  color     = dst.a > 0.001 ? dst.rgb / dst.a : vec3(0.0);
            float finalA    = clamp(dst.a, 0.0, 0.90) * fadeAlpha * intensity;
            fragColor = vec4(color, finalA);
        }}
        """

        return shaders.compileProgram(
            shaders.compileShader(vertex_src,   GL_VERTEX_SHADER),
            shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        vertices = np.array([-1, -1,  1, -1,  -1, 1,  1, 1], dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(vbo)

        glBindVertexArray(0)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self.time += dt

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return

        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glUniform1f(glGetUniformLocation(self.shader, "depth"),           self.depth)
        glUniform1f(glGetUniformLocation(self.shader, "time"),            self.time)
        glUniform1f(glGetUniformLocation(self.shader, "speed"),           self.base_speed)
        glUniform1f(glGetUniformLocation(self.shader, "fadeAlpha"),       self.fade_factor)
        glUniform1f(glGetUniformLocation(self.shader, "intensity"),       self.base_intensity)
        glUniform1f(glGetUniformLocation(self.shader, "extensionFactor"), self.extension_factor)
        glUniform1f(glGetUniformLocation(self.shader, "tentStart"),      self.y_start)
        glUniform1f(glGetUniformLocation(self.shader, "tentMaxLen"),     self.max_length)

        loc = glGetUniformLocation(self.shader, "tentBaseX")
        if loc != -1:
            glUniform1fv(loc, N_TENTACLES, self.base_x)

        loc = glGetUniformLocation(self.shader, "tentPhase")
        if loc != -1:
            glUniform1fv(loc, N_TENTACLES, self.phases)

        glDepthMask(GL_FALSE)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDepthMask(GL_TRUE)

        glBindVertexArray(0)
        glUseProgram(0)
