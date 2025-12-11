"""
Vortex shader effect - Swirling particle-based fluid dynamics
Adapted from Shadertoy vortex simulation with audio reactivity
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_vortex(state, outstate, depth=50.0, intensity=1.0, particle_count=64,
                  strength=1.5, bass_sensitivity=0.8, mid_sensitivity=0.5, 
                  high_sensitivity=0.3):
    """
    Shader-based vortex particle effect compatible with EventScheduler
    
    Creates swirling particle-based fluid dynamics using Biot-Savart velocity
    field with vortex particles. Responds to audio with particle spawning (bass),
    strength modulation (mids), and color shifts (highs).
    
    Usage:
        scheduler.schedule_event(0, 60, shader_vortex, particle_count=64, 
                               strength=0.3, bass_sensitivity=0.8, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        depth: Z-depth of vortex (0=near, 100=far, default: 50)
        intensity: Base brightness multiplier (default: 1.0)
        particle_count: Number of vortex particles (power of 2, max 256, default: 64)
        strength: Velocity field strength (default: 1.5)
        bass_sensitivity: How much bass frequencies affect particle spawning (default: 0.8)
        mid_sensitivity: How much mid frequencies affect strength (default: 0.5)
        high_sensitivity: How much high frequencies affect colors (default: 0.3)
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
        print(f"Initializing vortex for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                VortexEffect,
                depth=depth,
                intensity=intensity,
                particle_count=particle_count,
                strength=strength,
                bass_sensitivity=bass_sensitivity,
                mid_sensitivity=mid_sensitivity,
                high_sensitivity=high_sensitivity
            )
            state['effect'] = effect
            print(f"✓ Initialized shader vortex for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize vortex: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from audio data and global state
    if 'effect' in state:
        audio_data = outstate.get('sound')
        
        # Process audio data if available
        if audio_data is not None:
            # Get current normalized bands (use short-term for beat response)
            bands = audio_data['norm_short'][0]  # Shape: (32,)
            
            # Extract frequency ranges
            bass_energy = np.mean(bands[0:8])      # Bass: 40-300 Hz
            mid_energy = np.mean(bands[8:20])      # Mids: 300-2000 Hz
            high_energy = np.mean(bands[20:32])    # Highs: 2000-16000 Hz
            
            # Apply audio modulation to effect parameters
            state['effect'].audio_bass = bass_energy * bass_sensitivity
            state['effect'].audio_mid = mid_energy * mid_sensitivity
            state['effect'].audio_high = high_energy * high_sensitivity
        else:
            # No audio - zero out audio modulation
            state['effect'].audio_bass = 0.0
            state['effect'].audio_mid = 0.0
            state['effect'].audio_high = 0.0
        
        # Update from global state (optional)
        state['effect'].base_intensity = outstate.get('vortex_intensity', intensity)
        state['effect'].base_strength = outstate.get('vortex_strength', strength)
        
        # Implement fade in/out
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        fade_duration = 2.0
        
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = np.clip(fade_factor, 0, 1)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up vortex for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader vortex for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class VortexEffect(ShaderEffect):
    """
    Vortex particle fluid dynamics effect
    
    Creates swirling particle-based fluid simulation using Biot-Savart
    velocity field induced by vortex particles with audio reactivity.
    """
    
    def __init__(self, viewport, depth: float = 50.0, intensity: float = 1.0,
                 particle_count: int = 64, strength: float = 1.5,
                 bass_sensitivity: float = 0.8, mid_sensitivity: float = 0.5,
                 high_sensitivity: float = 0.3):
        super().__init__(viewport)
        self.depth = depth
        self.base_intensity = intensity
        self.base_strength = strength
        self.particle_count = min(particle_count, 256)  # Clamp to max
        self.N = int(np.sqrt(self.particle_count))  # Grid size (NxN particles)
        self.Nvort = self.particle_count
        self.time = 0.0
        self.fade_factor = 0.0
        
        # Audio sensitivity parameters
        self.bass_sensitivity = bass_sensitivity
        self.mid_sensitivity = mid_sensitivity
        self.high_sensitivity = high_sensitivity
        
        # Audio modulation values (updated from wrapper)
        self.audio_bass = 0.0
        self.audio_mid = 0.0
        self.audio_high = 0.0
        
        # Smoothed audio values for stable modulation
        self.audio_bass_smooth = 0.0
        self.audio_mid_smooth = 0.0
        self.audio_high_smooth = 0.0
        
        # Framebuffer objects for multi-pass rendering
        self.bufferA_FBO = None
        self.bufferB_FBO = None
        self.bufferC_FBO = None
        self.bufferD_FBO = None
        
        self.bufferA_tex = None
        self.bufferB_tex = None
        self.bufferC_tex = None
        self.bufferD_tex = None
        
        # Shader programs
        self.shader_bufferA = None
        self.shader_bufferB = None
        self.shader_bufferC = None
        self.shader_bufferD = None
        self.shader_image = None
        
        # Quad for fullscreen rendering
        self.quad_VAO = None
        self.quad_VBO = None
        
        self.frame_count = 0
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize quad mesh for fullscreen passes"""
        # Fullscreen quad
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        
        self.quad_vertices = quad_vertices
    
    def compile_shader(self):
        """Compile all shader passes"""
        try:
            # Compile all buffer shaders
            self.shader_bufferA = self._compile_program(
                self.get_vertex_shader_fullscreen(),
                self.get_fragment_shader_bufferA()
            )
            
            self.shader_bufferB = self._compile_program(
                self.get_vertex_shader_fullscreen(),
                self.get_fragment_shader_bufferB()
            )
            
            self.shader_bufferC = self._compile_program(
                self.get_vertex_shader_fullscreen(),
                self.get_fragment_shader_bufferC()
            )
            
            self.shader_bufferD = self._compile_program(
                self.get_vertex_shader_fullscreen(),
                self.get_fragment_shader_bufferD()
            )
            
            self.shader_image = self._compile_program(
                self.get_vertex_shader_display(),
                self.get_fragment_shader_image()
            )
            
            # Return the final display shader as main shader
            return self.shader_image
            
        except Exception as e:
            print(f"Vortex shader compilation error: {e}")
            raise
    
    def _compile_program(self, vert_src, frag_src):
        """Helper to compile a shader program"""
        try:
            vert = shaders.compileShader(vert_src, GL_VERTEX_SHADER)
        except Exception as e:
            print(f"Vertex shader compilation error: {e}")
            print("Vertex source:")
            for i, line in enumerate(vert_src.split('\n')[:20], 1):
                print(f"{i:3}: {line}")
            raise
        
        try:
            frag = shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
        except Exception as e:
            print(f"Fragment shader compilation error: {e}")
            print("Fragment source:")
            for i, line in enumerate(frag_src.split('\n')[:20], 1):
                print(f"{i:3}: {line}")
            raise
        
        try:
            prog = shaders.compileProgram(vert, frag)
            print(f"✓ Shader program compiled successfully (ID: {prog})")
            return prog
        except Exception as e:
            print(f"Program linking error: {e}")
            raise
    
    def get_common_defines(self):
        """Common shader definitions"""
        return f"""
        #version 310 es
        precision highp float;
        
        #define N {self.N}
        #define Nf float(N)
        #define Nvort {self.Nvort}
        #define R vec2(iResolution.xy)
        #define STRENGTH {self.base_strength}
        #define CYCLE 2
        #define BINARY 0
        
        #define TAU 6.2831853
        #define l2(x) dot(x,x)
        
        vec2 rand2(vec2 U) {{
            return fract(1e5 * sin(mat2(17.1, 191.7, -31.1, 241.7) * U));
        }}
        
        int IHash(int a) {{
            a = (a ^ 61) ^ (a >> 16);
            a = a + (a << 3);
            a = a ^ (a >> 4);
            a = a * 0x27d4eb2d;
            a = a ^ (a >> 15);
            return a;
        }}
        """
    
    def get_vertex_shader_fullscreen(self):
        """Vertex shader for fullscreen quad buffer passes"""
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 fragCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragCoord = (position + 1.0) * 0.5;
        }
        """
    
    def get_vertex_shader_display(self):
        """Vertex shader for final display with depth"""
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        uniform float depth;
        uniform vec2 resolution;
        
        out vec2 fragCoord;
        
        void main() {
            // Map depth to clip space z
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(position, mappedDepth, 1.0);
            fragCoord = (position + 1.0) * 0.5;
        }
        """
    
    def get_fragment_shader_bufferA(self):
        """Buffer A: First half-step of Verlet integration"""
        common = self.get_common_defines()
        return common + """
        uniform vec2 iResolution;
        uniform float iTime;
        uniform float iTimeDelta;
        uniform int iFrame;
        uniform sampler2D iChannel0;  // Self (previous frame)
        uniform sampler2D iChannel1;  // Buffer B output
        uniform float audioStrength;  // Mid frequencies affect strength
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define T0(U) texture(iChannel0, (U) / iResolution)
        #define T1(U) texture(iChannel1, mod((U), iResolution) / iResolution)
        #define W(P) T1(P + vec2(0.0, Nf)).z
        
        #define A(n, T) T0(T + vec2(float((int(n) - 1) % N), float((int(n) - 1) / N)))
        
        void mainImage(out vec4 O, vec2 U) {
            O = vec4(0.0);
            vec2 T = floor(U / Nf);
            
            // Tile mapping - persist vorticity tile
            if (T == vec2(0.0, 1.0)) {
                O = T1(U);
                return;
            }
            
            U = mod(U, Nf);
            
            float dt = iFrame < 10 ? 1.0/60.0 : iTimeDelta;
            
            // Only process main particle tile
            if (T != vec2(0.0, 0.0)) {
                return;
            }
            
            // Half time step
            dt *= 0.5;
            
            // Get current state from Buffer B (previous frame)
            O = T1(U);
            
            // Evaluate Biot-Savart velocity field
            vec2 F = vec2(0.0);
            float modStrength = STRENGTH * (1.0 + audioStrength * 2.0);
            
            for (int k = 0; k < Nvort; k++) {
                vec2 P = vec2(float(k % N), float(k / N));
                vec2 d = T1(P).xy - O.xy;
                float w = W(P);
                
                // Cycling world
                d = (fract(0.5 + d / R) - 0.5) * R;
                
                float l = dot(d, d);
                if (l > 1e-5) {
                    F += vec2(-d.y, d.x) * w / l;
                }
            }
            
            O.zw = modStrength * F;
            O.xy += O.zw * dt;
            O.xy = mod(O.xy, R);
        }
        
        void main() {
            mainImage(outColor, fragCoord * iResolution);
        }
        """
    
    def get_fragment_shader_bufferB(self):
        """Buffer B: Full Verlet integration step with audio particle spawning"""
        common = self.get_common_defines()
        return common + """
        uniform vec2 iResolution;
        uniform float iTime;
        uniform float iTimeDelta;
        uniform int iFrame;
        uniform sampler2D iChannel0;  // Buffer A output
        uniform sampler2D iChannel1;  // Self (previous frame)
        uniform float audioBass;      // Bass affects particle spawning
        uniform float audioStrength;  // Mid frequencies affect strength
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define T0(U) texture(iChannel0, (U) / iResolution)
        #define T1(U) texture(iChannel1, mod((U), iResolution) / iResolution)
        #define W(P) T1(P + vec2(0.0, Nf)).z
        
        void mainImage(out vec4 O, vec2 U) {
            O = vec4(0.0);
            vec2 T = floor(U / Nf);
            
            if (iFrame < 1) {
                // Initialize particles with better distribution
                O = vec4(R * rand2(U), 2.0 * rand2(U + 7.13) - 1.0);
                
                if (T == vec2(0.0, 1.0)) {
                    int idx = int(U.x) + N * (int(U.y) % N);
                    if (idx > Nvort) {
                        O.z = 0.0;  // Passive marker
                    } else {
                        // Active vortices with alternating signs for interesting flow
                        O.z = (idx % 2 == 0) ? 1.0 : -1.0;
                    }
                }
                return;
            }
            
            // Persist vorticity tile
            if (T == vec2(0.0, 1.0)) {
                O = T1(U);
                return;
            }
            
            U = mod(U, Nf);
            
            // Only process main particle tile
            if (T != vec2(0.0, 0.0)) {
                return;
            }
            
            float dt = iFrame < 10 ? 1.0/60.0 : iTimeDelta;
            
            // Get state from Buffer A (half-step)
            O = T0(U);
            
            // Evaluate Biot-Savart velocity field with updated positions
            vec2 F = vec2(0.0);
            float modStrength = STRENGTH * (1.0 + audioStrength * 2.0);
            
            for (int k = 0; k < Nvort; k++) {
                vec2 P = vec2(float(k % N), float(k / N));
                vec2 d = T0(P).xy - O.xy;  // Use Buffer A positions
                float w = W(P);
                
                d = (fract(0.5 + d / R) - 0.5) * R;
                
                float l = dot(d, d);
                if (l > 1e-5) {
                    F += vec2(-d.y, d.x) * w / l;
                }
            }
            
            // Update velocity and position
            O.zw = modStrength * F;
            // Get original position from previous frame
            vec2 oldPos = T1(U).xy;
            O.xy = oldPos + O.zw * dt;
            O.xy = mod(O.xy, R);
            
            // Audio-driven particle spawning (bass affects marker placement)
            if (int(U.x) + int(U.y) * N > Nvort && 
                fract(1e4 * sin(dot(U - iTime, vec2(131.7, -97.3)))) < 0.1 * (1.0 + audioBass * 3.0)) {
                // Spawn particles in circular patterns
                float angle = iTime * 2.0 + float(int(U.x) + int(U.y) * N) * 0.5;
                float radius = R.y * 0.3;
                O.xy = R * 0.5 + vec2(cos(angle), sin(angle)) * radius;
            }
        }
        
        void main() {
            mainImage(outColor, fragCoord * iResolution);
        }
        """
    
    def get_fragment_shader_bufferC(self):
        """Buffer C: Voronoi tracking structure"""
        common = self.get_common_defines()
        return common + """
        uniform vec2 iResolution;
        uniform int iFrame;
        uniform sampler2D iChannel0;  // Buffer B (particle data)
        uniform sampler2D iChannel1;  // Self (previous frame)
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define T0(U) texture(iChannel0, mod((U), iResolution) / iResolution)
        #define T1(U) texture(iChannel1, (U) / iResolution)
        
        #define A(n, T) T0(T + vec2(float((int(n) - 1) % N), float((int(n) - 1) / N)))
        
        void list_insert(inout vec4 i, inout vec4 d, float i_, float d_) {
            if (i_ == 0.0) return;
            if (any(equal(vec4(i_), i))) return;
            
            if (d_ < d[0]) {
                i = vec4(i_, i.xyz);
                d = vec4(d_, d.xyz);
            } else if (d_ < d[1]) {
                i = vec4(i.x, i_, i.yz);
                d = vec4(d.x, d_, d.yz);
            } else if (d_ < d[2]) {
                i = vec4(i.xy, i_, i.z);
                d = vec4(d.xy, d_, d.z);
            } else if (d_ < d[3]) {
                i = vec4(i.xyz, i_);
                d = vec4(d.xyz, d_);
            }
        }
        
        float dist(float idx, vec2 I) {
            vec2 D = mod(A(idx, vec2(0.0)).xy - I + R / 2.0, R) - R / 2.0;
            return dot(D, D);
        }
        
        void mainImage(out vec4 O, vec2 I) {
            vec4 i = vec4(0.0);
            vec4 i0 = T1(I);
            vec4 ia = T1(I + vec2(1.0, 0.0));
            vec4 ib = T1(I + vec2(0.0, 1.0));
            vec4 ic = T1(I + vec2(-1.0, 0.0));
            vec4 id = T1(I + vec2(0.0, -1.0));
            
            vec4 d = vec4(1e9);
            
            for (int k = 0; k < 4; k++) {
                list_insert(i, d, i0[k], dist(i0[k], I));
                list_insert(i, d, ia[k], dist(ia[k], I));
                list_insert(i, d, ib[k], dist(ib[k], I));
                list_insert(i, d, ic[k], dist(ic[k], I));
                list_insert(i, d, id[k], dist(id[k], I));
            }
            
            for (int k = 0; k < 8; k++) {
                int r = IHash(int(I.x) + int(I.y) * 2141 + iFrame * 2141 * 2141 + k * 11131);
                int i_ = 1 + r % (N * N);
                list_insert(i, d, float(i_), dist(float(i_), I));
            }
            
            O = vec4(i);
        }
        
        void main() {
            mainImage(outColor, fragCoord * iResolution);
        }
        """
    
    def get_fragment_shader_bufferD(self):
        """Buffer D: Motion blur display buffer"""
        common = self.get_common_defines()
        return common + """
        uniform vec2 iResolution;
        uniform sampler2D iChannel0;  // Buffer B (particle data)
        uniform sampler2D iChannel1;  // Self (previous frame)
        uniform sampler2D iChannel2;  // Buffer C (Voronoi)
        uniform float audioHue;       // High frequencies affect color
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define T0(U) texture(iChannel0, (U) / iResolution)
        #define T1(U) texture(iChannel1, (U) / iResolution)
        #define T2(U) texture(iChannel2, (U) / iResolution)
        
        #define A(n, T) T0(T + vec2(float((int(n) - 1) % N), float((int(n) - 1) / N)))
        
        #define Rv 1200.0
        #define Rm 600.0
        #define Wv 3.0
        
        vec3 hueShift(vec3 color, float shift) {
            float angle = shift * TAU;
            vec3 k = vec3(0.57735, 0.57735, 0.57735);
            float cosAngle = cos(angle);
            return color * cosAngle + cross(k, color) * sin(angle) + k * dot(k, color) * (1.0 - cosAngle);
        }
        
        void mainImage(out vec4 O, vec2 U) {
            O = vec4(0.0);
            vec4 a = T2(U);
            
            for (int i = 0; i < 4; i++) {
                if (a[i] == 0.0) break;
                
                vec4 P = A(a[i], vec2(0.0));
                float l = l2(U - P.xy);
                float w = A(a[i], vec2(0.0, Nf)).z;
                
                if (w == 0.0) {
                    O += smoothstep(Rm, Rm / 4.0, l) * 0.6 * 0.3;
                } else {
                    l = smoothstep(Rv, Rv / 4.0, l);
                    if (w > 0.0) {
                        O.r += w * l * Wv;
                    } else {
                        O.b += -w * l * Wv;
                    }
                }
            }
            
            // Apply hue shift based on audio
            O.rgb = hueShift(O.rgb, audioHue * 0.3);
            
            // Motion blur with less fade for more visibility
            O = max(O / 0.5, 0.92 * T1(U));
        }
        
        void main() {
            mainImage(outColor, fragCoord * iResolution);
        }
        """
    
    def get_fragment_shader_image(self):
        """Final image shader with depth and alpha - with debug particle visualization"""
        return """
        #version 310 es
        precision highp float;
        
        uniform vec2 iResolution;
        uniform sampler2D iChannel0;  // Buffer D (display buffer)
        uniform sampler2D iChannel1;  // Buffer B (particle data) for debugging
        uniform float fadeAlpha;
        uniform float intensity;
        uniform int iFrame;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define N 8
        #define Nf 8.0
        
        void main() {
            vec2 uv = fragCoord;
            vec2 pixelCoord = uv * iResolution;
            vec4 col = texture(iChannel0, uv);
            
            // DEBUG: Directly visualize particles from Buffer B
            // Read particle positions and draw them
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    vec2 particleUV = (vec2(float(x), float(y)) + 0.5) / Nf;
                    vec4 particleData = texture(iChannel1, particleUV);
                    vec2 particlePos = particleData.xy;
                    vec2 particleVel = particleData.zw;
                    
                    // Draw a blob at particle position
                    float dist = length(pixelCoord - particlePos);
                    if (dist < 30.0) {
                        float intensity = 1.0 - (dist / 30.0);
                        // Show velocity as color (green for moving particles)
                        float speed = length(particleVel);
                        col.rgb += mix(vec3(0.5), vec3(0.0, 1.0, 0.0), clamp(speed * 0.1, 0.0, 1.0)) * intensity;
                    }
                }
            }
            
            // Read vorticity data
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    vec2 vortUV = (vec2(float(x), float(y) + Nf) + 0.5) / iResolution;
                    vec4 vortData = texture(iChannel1, vortUV);
                    vec2 vortPos = texture(iChannel1, vec2(float(x), float(y)) / Nf).xy;
                    float vorticity = vortData.z;
                    
                    // Draw red/blue blobs for vortices
                    float dist = length(pixelCoord - vortPos);
                    if (dist < 40.0 && abs(vorticity) > 0.01) {
                        float intensity = (1.0 - (dist / 40.0)) * abs(vorticity);
                        if (vorticity > 0.0) {
                            col.rgb += vec3(1.0, 0.0, 0.0) * intensity * 2.0;
                        } else {
                            col.rgb += vec3(0.0, 0.0, 1.0) * intensity * 2.0;
                        }
                    }
                }
            }
            
            // Boost visibility
            col.rgb *= intensity;
            
            // Calculate alpha from color brightness
            float brightness = length(col.rgb);
            col.a = clamp(brightness * 0.6, 0.0, 0.85) * fadeAlpha;
            
            outColor = col;
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers and framebuffers"""
        print(f"Setting up vortex buffers (viewport: {self.viewport.width}x{self.viewport.height})")
        
        # Setup quad VAO
        self.quad_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.quad_VAO)
        
        self.quad_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, self.quad_vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        print(f"✓ Quad VAO/VBO created (VAO: {self.quad_VAO}, VBO: {self.quad_VBO})")
        
        # Create framebuffer textures
        width = self.viewport.width
        height = self.viewport.height
        
        self.bufferA_tex = self._create_texture(width, height)
        self.bufferB_tex = self._create_texture(width, height)
        self.bufferC_tex = self._create_texture(width, height)
        self.bufferD_tex = self._create_texture(width, height)
        print(f"✓ Created 4 framebuffer textures ({width}x{height})")
        
        # Create framebuffers
        self.bufferA_FBO = self._create_framebuffer(self.bufferA_tex)
        self.bufferB_FBO = self._create_framebuffer(self.bufferB_tex)
        self.bufferC_FBO = self._create_framebuffer(self.bufferC_tex)
        self.bufferD_FBO = self._create_framebuffer(self.bufferD_tex)
        print(f"✓ Created 4 framebuffers")
        
        # Initialize textures to zero
        self._clear_texture(self.bufferA_FBO)
        self._clear_texture(self.bufferB_FBO)
        self._clear_texture(self.bufferC_FBO)
        self._clear_texture(self.bufferD_FBO)
        print(f"✓ Cleared all framebuffer textures")
    
    def _create_texture(self, width, height):
        """Create a framebuffer texture"""
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        return tex
    
    def _create_framebuffer(self, texture):
        """Create a framebuffer object"""
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"ERROR: Framebuffer is not complete! Status: {status}")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return fbo
    
    def _clear_texture(self, fbo):
        """Clear a framebuffer texture to zero"""
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        self.time += dt
        self.frame_count += 1
        
        # Smooth audio values with exponential decay
        attack_factor = 1.0 - np.exp(-dt / 0.05)
        decay_factor = 1.0 - np.exp(-dt / 0.3)
        
        # Bass: quick attack for spawning
        if self.audio_bass > self.audio_bass_smooth:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * attack_factor
        else:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * decay_factor
        
        # Mid: affects strength
        if self.audio_mid > self.audio_mid_smooth:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * attack_factor
        else:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * decay_factor
        
        # High: affects hue
        if self.audio_high > self.audio_high_smooth:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * attack_factor
        else:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * decay_factor
    
    def _render_pass(self, shader, output_fbo, input_textures, uniforms):
        """Render a single shader pass"""
        glBindFramebuffer(GL_FRAMEBUFFER, output_fbo)
        glUseProgram(shader)
        
        # Set resolution
        res_loc = glGetUniformLocation(shader, "iResolution")
        glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        # Set time
        time_loc = glGetUniformLocation(shader, "iTime")
        if time_loc != -1:
            glUniform1f(time_loc, self.time)
        
        # Set delta time
        dt_loc = glGetUniformLocation(shader, "iTimeDelta")
        if dt_loc != -1:
            glUniform1f(dt_loc, 1.0 / 60.0)
        
        # Set frame
        frame_loc = glGetUniformLocation(shader, "iFrame")
        if frame_loc != -1:
            glUniform1i(frame_loc, self.frame_count)
        
        # Bind input textures
        for i, tex in enumerate(input_textures):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, tex)
            chan_loc = glGetUniformLocation(shader, f"iChannel{i}")
            if chan_loc != -1:
                glUniform1i(chan_loc, i)
        
        # Set custom uniforms
        for name, value in uniforms.items():
            loc = glGetUniformLocation(shader, name)
            if loc != -1:
                glUniform1f(loc, value)
        
        # Render quad
        glBindVertexArray(self.quad_VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
    
    def render(self, state: Dict):
        """Render the vortex effect with multi-pass rendering"""
        if not self.enabled:
            return
        
        # Debug output on first few frames
        if self.frame_count < 3:
            print(f"Vortex render frame {self.frame_count}: enabled={self.enabled}, fade={self.fade_factor:.2f}")
            print(f"  Audio: bass={self.audio_bass_smooth:.2f}, mid={self.audio_mid_smooth:.2f}, high={self.audio_high_smooth:.2f}")
            print(f"  Buffers: A={self.bufferA_tex}, B={self.bufferB_tex}, C={self.bufferC_tex}, D={self.bufferD_tex}")
        
        # Save current viewport and state
        current_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        
        # Disable depth testing for buffer passes
        depth_test_enabled = glIsEnabled(GL_DEPTH_TEST)
        if depth_test_enabled:
            glDisable(GL_DEPTH_TEST)
        
        # Pass 1: Buffer A (half-step integration)
        self._render_pass(
            self.shader_bufferA,
            self.bufferA_FBO,
            [self.bufferA_tex, self.bufferB_tex],
            {'audioStrength': self.audio_mid_smooth}
        )
        
        # Pass 2: Buffer B (full integration)
        self._render_pass(
            self.shader_bufferB,
            self.bufferB_FBO,
            [self.bufferA_tex, self.bufferB_tex],
            {
                'audioBass': self.audio_bass_smooth,
                'audioStrength': self.audio_mid_smooth
            }
        )
        
        # Pass 3: Buffer C (Voronoi tracking)
        self._render_pass(
            self.shader_bufferC,
            self.bufferC_FBO,
            [self.bufferB_tex, self.bufferC_tex],
            {}
        )
        
        # Pass 4: Buffer D (motion blur)
        self._render_pass(
            self.shader_bufferD,
            self.bufferD_FBO,
            [self.bufferB_tex, self.bufferD_tex, self.bufferC_tex],
            {'audioHue': self.audio_high_smooth}
        )
        
        # Re-enable depth test for final pass
        if depth_test_enabled:
            glEnable(GL_DEPTH_TEST)
        
        # Final pass: Render to viewport with depth
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo)
        glUseProgram(self.shader_image)
        
        if self.frame_count < 3:
            print(f"Final pass: FBO={current_fbo}, shader={self.shader_image}, depth={self.depth}")
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader_image, "iResolution")
        glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        depth_loc = glGetUniformLocation(self.shader_image, "depth")
        glUniform1f(depth_loc, self.depth)
        
        fade_loc = glGetUniformLocation(self.shader_image, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        intensity_loc = glGetUniformLocation(self.shader_image, "intensity")
        glUniform1f(intensity_loc, self.base_intensity)
        
        frame_loc = glGetUniformLocation(self.shader_image, "iFrame")
        glUniform1i(frame_loc, self.frame_count)
        
        # Bind buffer D texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bufferD_tex)
        chan_loc = glGetUniformLocation(self.shader_image, "iChannel0")
        glUniform1i(chan_loc, 0)
        
        # Also bind Buffer B for debug visualization
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.bufferB_tex)
        chan_loc = glGetUniformLocation(self.shader_image, "iChannel1")
        glUniform1i(chan_loc, 1)
        
        # Render final quad
        glBindVertexArray(self.quad_VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.quad_VBO is not None:
            glDeleteBuffers(1, [self.quad_VBO])
        if self.quad_VAO is not None:
            glDeleteVertexArrays(1, [self.quad_VAO])
        
        # Delete framebuffers
        if self.bufferA_FBO is not None:
            glDeleteFramebuffers(1, [self.bufferA_FBO])
        if self.bufferB_FBO is not None:
            glDeleteFramebuffers(1, [self.bufferB_FBO])
        if self.bufferC_FBO is not None:
            glDeleteFramebuffers(1, [self.bufferC_FBO])
        if self.bufferD_FBO is not None:
            glDeleteFramebuffers(1, [self.bufferD_FBO])
        
        # Delete textures
        if self.bufferA_tex is not None:
            glDeleteTextures(1, [self.bufferA_tex])
        if self.bufferB_tex is not None:
            glDeleteTextures(1, [self.bufferB_tex])
        if self.bufferC_tex is not None:
            glDeleteTextures(1, [self.bufferC_tex])
        if self.bufferD_tex is not None:
            glDeleteTextures(1, [self.bufferD_tex])
        
        # Delete shader programs
        if self.shader_bufferA is not None:
            glDeleteProgram(self.shader_bufferA)
        if self.shader_bufferB is not None:
            glDeleteProgram(self.shader_bufferB)
        if self.shader_bufferC is not None:
            glDeleteProgram(self.shader_bufferC)
        if self.shader_bufferD is not None:
            glDeleteProgram(self.shader_bufferD)
        if self.shader_image is not None:
            glDeleteProgram(self.shader_image)
