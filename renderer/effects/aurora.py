"""
Aurora shader effect - Flowing northern lights at top of screen
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_aurora(state, outstate, height_ratio=0.90, depth=96.0,
                  intensity=1.0, speed=1.0):
    """
    Shader-based aurora effect compatible with EventScheduler.

    Creates flowing northern lights at the top of the screen with sparse
    bright zones (horizontal envelope), fine vertical filaments, and a slow,
    mostly-per-column color flow through green / cyan / purple / pink.

    Usage:
        scheduler.schedule_event(0, 60, shader_aurora, height_ratio=0.3,
                                 depth=75, frame_id=0)

    Args:
        state: Event state dict (start_time, elapsed_time, count, frame_id).
        outstate: Global state dict (from EventScheduler).
        height_ratio: Height of aurora as proportion of viewport height.
        depth: Z-depth of aurora (0=near, 100=far).
        intensity: Base brightness multiplier.
        speed: Base animation speed multiplier.
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

    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing aurora for frame {frame_id}")

        try:
            effect = viewport.add_effect(
                Aurora,
                height_ratio=height_ratio,
                depth=depth,
                intensity=intensity,
                speed=speed,
            )
            state['effect'] = effect
            print(f"✓ Initialized shader aurora for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize aurora: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        # Update from global state (optional)
        state['effect'].base_intensity = outstate.get('aurora_intensity', intensity)
        state['effect'].base_speed = outstate.get('aurora_speed', speed)

        # Fade in/out
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

    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up aurora for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader aurora for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class Aurora(ShaderEffect):
    """
    Aurora borealis effect with sparse bright zones and vertical filaments.

    The horizontal envelope concentrates brightness into a few traveling
    regions instead of running uniformly across the strip. Filaments are
    fine vertical striations that only appear inside the bright zones, and
    color is mostly stable per-column (a column reads as one hue top to
    bottom) so the eye sees coherent curtains rather than rainbow gradients.
    """

    def __init__(self, viewport, height_ratio: float = 0.90, depth: float = 96.0,
                 intensity: float = 1.0, speed: float = 1.0):
        super().__init__(viewport)
        self.render_priority = 5.5  # After BART map (5), before clouds (6)
        self.height_ratio = height_ratio
        self.height = int(viewport.height * height_ratio)
        self.depth = depth
        self.base_intensity = intensity
        self.base_speed = speed
        self.fade_factor = 0.0

        # CPU-integrated phases. base_speed is a varying uniform (interpolates
        # during weather transitions), so we must NOT multiply it by an
        # in-shader u_time — we integrate dt * base_speed here and pass the
        # monotonic phase to the shader. See docs/shader_info.txt
        # "Time-based Animation".
        self.wave_phase = 0.0   # drives wave displacement & band motion
        self.color_phase = 0.0  # drives color cycle & envelope drift

        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.EBO = None

        # Mesh resolution (higher = smoother waves)
        self.segments_x = 100
        self.segments_y = 20

        self._initialize_data()

    def _initialize_data(self):
        """Initialize mesh data for aurora plane"""
        width = self.viewport.width

        vertices = []
        for y in range(self.segments_y + 1):
            y_pos = (y / self.segments_y) * self.height
            for x in range(self.segments_x + 1):
                x_pos = (x / self.segments_x) * width
                vertices.append([x_pos, y_pos])

        self.vertices = np.array(vertices, dtype=np.float32)

        # Triangle-strip indices with degenerate triangles between rows
        indices = []
        for y in range(self.segments_y):
            for x in range(self.segments_x + 1):
                indices.append(y * (self.segments_x + 1) + x)
                indices.append((y + 1) * (self.segments_x + 1) + x)
            if y < self.segments_y - 1:
                indices.append((y + 1) * (self.segments_x + 1) + self.segments_x)
                indices.append((y + 1) * (self.segments_x + 1))

        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = len(self.indices)

    def compile_shader(self):
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()

        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Aurora shader compilation error: {e}")
            raise

    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;

        layout(location = 0) in vec2 position;

        uniform vec2 resolution;
        uniform float depth;
        uniform float wavePhase;
        uniform float height;

        out vec2 vTexCoord;
        out float vWave;

        void main() {
            vec2 pos = position;

            float normY = pos.y / height;          // 0 at top, 1 at bottom
            float normX = pos.x / resolution.x;

            // Three layered horizontal waves shape the aurora's lower edge.
            // Anchored at the top (multiplied by normY) so the top edge stays
            // straight and the bottom edge undulates.
            float wave1 = sin(normX * 6.28318 * 2.0 + wavePhase * 0.5) * 0.30;
            float wave2 = sin(normX * 6.28318 * 3.0 - wavePhase * 0.7) * 0.20;
            float wave3 = sin(normX * 6.28318 * 1.5 + wavePhase * 0.3) * 0.25;

            float totalWave = (wave1 + wave2 + wave3) * normY * 30.0;
            pos.y += totalWave;

            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;

            float mappedDepth = clamp(depth / 100.0, 0.0, 1.0);
            gl_Position = vec4(clipPos, mappedDepth, 1.0);

            vTexCoord = vec2(normX, normY);
            vWave = (wave1 + wave2 + wave3) * 0.5 + 0.5;  // 0..1
        }
        """

    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;

        in vec2 vTexCoord;
        in float vWave;

        uniform float wavePhase;
        uniform float colorPhase;
        uniform float intensity;
        uniform float fadeAlpha;

        out vec4 outColor;

        // Aurora palette — green dominant with cyan / purple / pink fringes.
        vec3 getAuroraColor(float t) {
            vec3 cGreen  = vec3(0.05, 1.00, 0.40);
            vec3 cCyan   = vec3(0.10, 0.85, 1.00);
            vec3 cPurple = vec3(0.55, 0.10, 1.00);
            vec3 cPink   = vec3(1.00, 0.25, 0.85);

            float p = fract(t) * 4.0;
            if (p < 1.0) return mix(cGreen,  cCyan,   p);
            if (p < 2.0) return mix(cCyan,   cPurple, p - 1.0);
            if (p < 3.0) return mix(cPurple, cPink,   p - 2.0);
            return mix(cPink, cGreen, p - 3.0);
        }

        void main() {
            float x = vTexCoord.x;
            float y = vTexCoord.y;
            float xw = mod(x, 1.0);  // wrap-safe x

            // ---- Horizontal envelope: traveling bright zones with floor ----
            // Sum of low-frequency sines, smoothstepped for shape, then a
            // baseline floor so the aurora is always visibly present across
            // the full width. Peaks still reach 1.0 in the brightest zones,
            // but the dimmest regions stay at ~0.45 — no dead gaps.
            float e1 = sin(xw * 6.28318 * 0.7 + colorPhase * 1.2);
            float e2 = sin(xw * 6.28318 * 1.4 - colorPhase * 0.7 + 1.7);
            float e3 = sin(xw * 6.28318 * 0.4 + colorPhase * 0.5 + 3.1);
            float envelope = (e1 + e2 + e3) / 3.0;
            envelope = smoothstep(-0.4, 0.5, envelope);
            envelope = max(envelope, 0.45);

            // ---- Sheet: continuous broad glow (the body of the curtain) ----
            // Smooth large-scale variation in x with a baseline so the sheet
            // never goes to zero inside a bright zone — this is what was
            // missing before (the old `band` term went all the way to 0
            // between cycles, leaving stripey gaps). 0.55 baseline keeps the
            // glow continuous; the sin terms breathe slowly across width.
            float sheet = 0.55
                        + 0.25 * sin(xw * 6.28318         + wavePhase * 0.20 + vWave * 1.5)
                        + 0.15 * sin(xw * 6.28318 * 2.0   - wavePhase * 0.13 + 1.7);
            sheet = clamp(sheet, 0.0, 1.0);

            // ---- Folds: broad mid-width vertical drapes ----
            // 6 soft-edged folds across width, with a gentle y-dependent
            // curl. Tighter smoothstep range gives crisper drape edges so
            // bright/dim folds read clearly on a low-resolution display.
            float foldArg = xw * 6.0 + wavePhase * 0.05
                          + sin(y * 3.0 + wavePhase * 0.25) * 0.15;
            float folds = sin(foldArg * 6.28318) * 0.5 + 0.5;
            folds = smoothstep(0.30, 0.80, folds);

            // ---- Filaments: irregularly-spaced vertical striations ----
            // FM-warp the stripe coordinate so spacing bunches and stretches
            // along x — this is what kills the even-comb look. Integer
            // frequencies keep the pattern continuous across the wrap.
            // Total warp amplitude (0.075) is ~1.65 stripe-widths, so peaks
            // visibly cluster; max derivative stays < 1 so stripes never
            // reverse direction.
            float warp = sin(xw * 6.28318         + colorPhase * 0.30) * 0.050
                       + sin(xw * 6.28318 * 2.0   - colorPhase * 0.50 + 1.1) * 0.025;
            // Y-dependent micro-warp gives each stripe a gentle curl as it
            // descends (follows the field-line shape of real aurorae).
            float wavyX = xw + warp + sin(y * 5.0 + wavePhase * 0.4) * 0.006;

            // 22 nominal stripes per width — bunched/stretched by warp.
            // pow exponent 2.5 (was 4.0) gives slightly wider, softer
            // filaments so they read as glowing accents on the sheet
            // rather than razor-thin lines.
            float stripeArg = wavyX * 22.0 + wavePhase * 0.15;
            float stripe = pow(max(sin(stripeArg * 6.28318), 0.0), 2.5);

            // Per-stripe brightness variation: each integer stripe index
            // hashes to a different intensity so the curtain has bright/dim
            // filaments side by side. mod-22 keeps the wrap seam clean.
            float stripeIdx = mod(floor(stripeArg), 22.0);
            float perStripe = 0.25 + 0.75 * fract(sin(stripeIdx * 12.9898 + 7.13) * 43758.5453);

            // Mid-frequency cluster mask drifts slowly so clumps fade in/out.
            float stripeMask = 0.4 + 0.6 * (sin(xw * 6.28318 * 5.0 - wavePhase * 0.25) * 0.5 + 0.5);

            float filaments = stripe * perStripe * stripeMask;
            // Stronger near the top of the curtain, fading downward.
            filaments *= 0.6 + 0.4 * (1.0 - y);

            // ---- Vertical fade ----
            // Softer falloff (exponent 0.35 vs 0.7) keeps the curtain bright
            // through more of its height so the aurora reads as a tall
            // luminous body, not just a thin band at the top. Bottom edge
            // softens only in the last ~12% of the strip.
            float verticalFade = pow(1.0 - y, 0.35);
            verticalFade *= smoothstep(1.0, 0.88, y);

            // ---- Color: mostly per-column, slow time drift ----
            // Each vertical column reads as a single hue. The y term is
            // intentionally tiny — just enough to prevent dead-flat columns.
            float colorT = colorPhase * 0.25 + xw * 0.6 + y * 0.05;
            vec3 color = getAuroraColor(colorT);

            // ---- Combine ----
            // Sheet is the continuous body of the curtain. Folds drape
            // texture across the sheet. Filaments are bright accents on
            // top — pushed hard enough that peak brightness saturates
            // color channels above 1.0, giving near-white hotspots that
            // read sharply against the colored sheet (high contrast on a
            // low-resolution LED display).
            float structure = sheet * 0.85
                            + folds * sheet * 0.65
                            + filaments * sheet * 1.8;
            // Built-in 1.5x boost so the aurora reads as a dominant layer
            // rather than a subtle accent. `intensity` is still the user
            // knob on top of this.
            float brightness = envelope * structure * verticalFade * intensity * 1.5;

            // Soft inner glow at fold peaks — pushes drape ridges further.
            float glow = smoothstep(0.4, 0.85, folds) * (1.0 - smoothstep(0.85, 1.0, folds));
            brightness += glow * envelope * verticalFade * 0.6;

            // Straight (NOT pre-multiplied) alpha — see shader_info.txt.
            // Aggressive midtone lift (exponent 0.55) means even moderate
            // brightness regions read as nearly opaque, so the curtain
            // body punches through whatever is behind it.
            float alpha = envelope * structure * verticalFade;
            alpha = pow(clamp(alpha, 0.0, 1.0), 0.55) * 0.97;
            alpha *= fadeAlpha;

            if (alpha < 0.01) discard;

            outColor = vec4(color * brightness, alpha);
        }
        """

    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        # Phase integration — base_speed can interpolate, so we integrate
        # rate * dt rather than multiplying u_time by base_speed in the shader.
        self.wave_phase += dt * self.base_speed
        self.color_phase += dt * self.base_speed * 0.15

    def render(self, state: Dict):
        if not self.enabled:
            return

        glUseProgram(self.shader)

        glUniform2f(glGetUniformLocation(self.shader, "resolution"),
                    float(self.viewport.width), float(self.viewport.height))
        glUniform1f(glGetUniformLocation(self.shader, "depth"), self.depth)
        glUniform1f(glGetUniformLocation(self.shader, "wavePhase"), self.wave_phase)
        glUniform1f(glGetUniformLocation(self.shader, "colorPhase"), self.color_phase)
        glUniform1f(glGetUniformLocation(self.shader, "height"), float(self.height))
        glUniform1f(glGetUniformLocation(self.shader, "intensity"), self.base_intensity)
        glUniform1f(glGetUniformLocation(self.shader, "fadeAlpha"), self.fade_factor)

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLE_STRIP, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glUseProgram(0)

    def cleanup(self):
        if self.position_VBO is not None:
            glDeleteBuffers(1, [self.position_VBO])
        if self.EBO is not None:
            glDeleteBuffers(1, [self.EBO])
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
