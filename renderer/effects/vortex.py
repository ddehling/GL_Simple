"""
Vortex shader effect - Multiple interacting point vortices.

Physics: 2D Biot-Savart point vortex dynamics on the CPU (numpy), periodic BCs.
Rendering: Vortex-driven domain warp of FBM in plain UV space (no periodic wrap,
no aspect correction) — seam-free by construction. Ocean colour palette.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

N_VORTICES = 14

# ============================================================================
# Event Wrapper
# ============================================================================

def shader_vortex(state, outstate, depth=50.0, intensity=1.0):
    """
    Multiple interacting point vortex effect compatible with EventScheduler.

    Args:
        state: Event state dict
        outstate: Global state dict
        depth: Z-depth (0=near, 100=far, default: 50)
        intensity: Brightness multiplier (default: 1.0)
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
        print(f"Initializing vortex for frame {frame_id}")
        try:
            effect = viewport.add_effect(VortexEffect, depth=depth, intensity=intensity)
            state['effect'] = effect
            print(f"[OK] Initialized vortex for frame {frame_id}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize vortex: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        elapsed = state['elapsed_time']
        total   = state.get('duration', 60)
        fade_d  = 3.0
        if elapsed < fade_d:
            fade = elapsed / fade_d
        elif elapsed > total - fade_d:
            fade = (total - elapsed) / fade_d
        else:
            fade = 1.0
        state['effect'].fade_factor = float(np.clip(fade, 0, 1))

    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up vortex for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"[OK] Cleaned up vortex for frame {frame_id}")


# ============================================================================
# Rendering / Physics Class
# ============================================================================

class VortexEffect(ShaderEffect):
    """
    Point vortex dynamics + turbulent ocean rendering.

    positions: (N, 2) array in UV space [0, 1]
    strengths: (N,)   signed floats; + = CCW, - = CW
    """

    # Physics constants
    PHYSICS_CORE = 0.012   # smoothed core radius (normalised height units)
    SUBSTEPS     = 6       # integration substeps per frame

    # Visual constants (UV space, no aspect correction)
    VISUAL_CORE  = 0.06    # warp falloff radius around each vortex core
    SPD_SCALE    = 0.30    # Michaelis-Menten half-saturation for the swirl speed

    def __init__(self, viewport, depth: float = 50.0, intensity: float = 1.0):
        super().__init__(viewport)
        self.depth          = depth
        self.base_intensity = intensity
        self.time           = 0.0
        self.fade_factor    = 0.0
        self.n              = N_VORTICES
        self._init_vortices()

    def _init_vortices(self):
        rng = np.random.default_rng(31)

        angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False) + np.pi / 7
        r = 0.27
        self.positions = (0.5 + r * np.column_stack([np.cos(angles), np.sin(angles)])
                          + rng.uniform(-0.05, 0.05, (self.n, 2)))
        self.positions = np.mod(self.positions, 1.0)

        signs = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)[:self.n]
        mags  = rng.uniform(0.011, 0.016, self.n)
        self.strengths = signs * mags

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _physics_step(self, sub_dt: float, aspect: float):
        pos  = self.positions
        d_uv = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d_uv -= np.round(d_uv)                         # periodic wrap [-0.5, 0.5]
        d    = d_uv * np.array([aspect, 1.0])          # aspect-corrected distances

        r2 = np.einsum('...k,...k->...', d, d)
        np.fill_diagonal(r2, np.inf)
        r2 = np.maximum(r2, self.PHYSICS_CORE ** 2)

        perp    = np.stack([-d[..., 1], d[..., 0]], axis=-1)
        contrib = self.strengths[np.newaxis, :, np.newaxis] * perp / r2[..., np.newaxis]
        vel_phys = contrib.sum(axis=1)

        vel_uv = vel_phys / np.array([aspect, 1.0])
        self.positions = np.mod(self.positions + vel_uv * sub_dt, 1.0)

    # ------------------------------------------------------------------
    # Shader
    # ------------------------------------------------------------------

    def compile_shader(self):
        n = self.n
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
        uniform float fadeAlpha;
        uniform float intensity;
        uniform vec2  vortexPos[{n}];
        uniform float vortexStr[{n}];

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
                p  = p * 2.1 + vec2(3.7, 1.3);
                a *= 0.5;
            }}
            return v;
        }}

        #define PHYS_CORE {self.PHYSICS_CORE}
        #define VIS_CORE  {self.VISUAL_CORE}
        #define SPD_SCALE {self.SPD_SCALE}

        void main() {{
            // ----------------------------------------------------------
            // Swirl field — plain UV space, NO periodic wrap, NO aspect.
            // Every term is a continuous function of vUV, so there are
            // no seam discontinuities anywhere on screen.
            // ----------------------------------------------------------
            vec2  swirl    = vec2(0.0);
            float coreGlow = 0.0;

            for (int i = 0; i < {n}; i++) {{
                vec2  d   = vUV - vortexPos[i];      // UV space, no wrap
                float r2  = dot(d, d);
                float r2s = r2 + PHYS_CORE * PHYS_CORE;
                float str = vortexStr[i];

                // Biot-Savart-like swirl (tangential, 1/r²)
                swirl += str * vec2(-d.y, d.x) / r2s;

                // Core indicator: Lorentzian in UV space
                float cR2 = VIS_CORE * VIS_CORE;
                coreGlow += abs(str) * 16.0 * cR2 / (r2 + cR2 * 0.3);
            }}

            // Soft-clamp the warp magnitude so near-core values don't blow up FBM
            float swLen = length(swirl);
            vec2  warp  = swirl / (1.0 + swLen * 2.5);   // magnitude → [0, ~0.4)

            // Speed brightness — Michaelis-Menten on unclamped swirl length
            float speedB = swLen / (swLen + SPD_SCALE);

            // ----------------------------------------------------------
            // Domain-warped FBM: two passes.
            // warp drives the UV offset so the turbulence organises around
            // vortex positions without using any directional velocity.
            // ----------------------------------------------------------
            vec2 p0 = vUV * 5.5 + vec2(time * 0.025, 0.0);
            // First warp layer: coarse structure
            vec2 w0 = vec2(fbm(p0 + warp * 1.8),
                           fbm(p0 + warp * 1.8 + vec2(5.2, 1.3)));
            float n1 = fbm(p0 + warp * 1.4 + w0 * 0.6);

            // Second layer: finer, slightly offset in time
            vec2 p1  = vUV * 9.5 - vec2(time * 0.018, time * 0.011);
            float n2 = fbm(p1 + warp * 0.9 + vec2(n1 * 0.5, 0.0));

            // Thin filaments where both layers simultaneously peak
            float filament = clamp(n1 * n2 * 4.2 - 0.30, 0.0, 1.0);

            // Turbulence visible only where there is swirl
            float turbulence = filament * speedB;

            // Foam at very high swirl + bright filament
            float foam = pow(speedB, 2.5) * filament;

            // ----------------------------------------------------------
            // Ocean colour palette
            // ----------------------------------------------------------
            vec3 deep    = vec3(0.01, 0.04, 0.12);   // near-black deep ocean
            vec3 current = vec3(0.03, 0.20, 0.38);   // dark teal-blue
            vec3 seafoam = vec3(0.68, 0.86, 0.94);   // pale seafoam

            vec3 color = mix(deep, current, clamp(turbulence * 1.6 + speedB * 0.07, 0.0, 1.0));
            color = mix(color, seafoam, clamp(foam * 0.72, 0.0, 1.0));
            color += vec3(0.04, 0.28, 0.48) * coreGlow * 0.30;  // subtle core teal

            // ----------------------------------------------------------
            // Alpha — transparent background, filaments carry the light
            // ----------------------------------------------------------
            float alpha = turbulence * 0.75 + speedB * 0.12 + coreGlow * 0.07;
            alpha = clamp(alpha * fadeAlpha * intensity, 0.0, 0.85);

            fragColor = vec4(color, alpha);
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
        aspect = self.viewport.width / self.viewport.height
        sub = dt / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            self._physics_step(sub, aspect)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return

        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glUniform1f(glGetUniformLocation(self.shader, "depth"),     self.depth)
        glUniform1f(glGetUniformLocation(self.shader, "time"),      self.time)
        glUniform1f(glGetUniformLocation(self.shader, "fadeAlpha"), self.fade_factor)
        glUniform1f(glGetUniformLocation(self.shader, "intensity"), self.base_intensity)

        pos_flat = self.positions.astype(np.float32).flatten()
        str_flat = self.strengths.astype(np.float32)

        loc = glGetUniformLocation(self.shader, "vortexPos")
        if loc != -1:
            glUniform2fv(loc, self.n, pos_flat)

        loc = glGetUniformLocation(self.shader, "vortexStr")
        if loc != -1:
            glUniform1fv(loc, self.n, str_flat)

        glDepthMask(GL_FALSE)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDepthMask(GL_TRUE)

        glBindVertexArray(0)
        glUseProgram(0)
