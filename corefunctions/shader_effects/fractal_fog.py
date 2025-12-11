"""
Fractal Fog shader effect - Volumetric raymarched fog with noise distortion
Adapted from Shadertoy with audio reactivity
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_fractal_fog(state, outstate, depth=50.0, intensity=1.0, speed=1.0,
                       bass_sensitivity=0.5, mid_sensitivity=0.3, 
                       high_sensitivity=0.2):
    """
    Shader-based fractal fog effect compatible with EventScheduler
    
    Creates volumetric raymarched fog with fractal noise distortion.
    Responds to audio with density pulsing (bass), rotation speed (mids),
    and distortion amount (highs).
    
    Usage:
        scheduler.schedule_event(0, 60, shader_fractal_fog, 
                               intensity=1.5, fog_density=1.2, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        depth: Z-depth of fog (0=near, 100=far, default: 50)
        intensity: Base brightness multiplier (default: 2.0)
        speed: Animation speed multiplier (default: 1.0)
        fog_density: Density of fog effect (default: 2.0)
        bass_sensitivity: How much bass frequencies affect density (default: 2.0)
        mid_sensitivity: How much mid frequencies affect rotation (default: 1.5)
        high_sensitivity: How much high frequencies affect distortion (default: 1.0)
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
        print(f"Initializing fractal fog for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                FractalFogEffect,
                depth=depth,
                intensity=intensity,
                speed=speed,
                bass_sensitivity=bass_sensitivity,
                mid_sensitivity=mid_sensitivity,
                high_sensitivity=high_sensitivity
            )
            state['effect'] = effect
            print(f"✓ Initialized shader fractal fog for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fractal fog: {e}")
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
        state['effect'].base_intensity = outstate.get('fog_intensity', intensity)
        state['effect'].base_speed = outstate.get('fog_speed', speed)
        
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
            print(f"Cleaning up fractal fog for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader fractal fog for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class FractalFogEffect(ShaderEffect):
    """
    Fractal fog volumetric raymarching effect
    
    Creates atmospheric fog with fractal noise distortion and audio reactivity.
    """
    
    def __init__(self, viewport, depth: float = 50.0, intensity: float = 1.0,
                 speed: float = 1.0,
                 bass_sensitivity: float = 0.5, mid_sensitivity: float = 0.3,
                 high_sensitivity: float = 0.2):
        super().__init__(viewport)
        self.depth = depth
        self.base_intensity = intensity
        self.base_speed = speed
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
        
        # Quad for fullscreen rendering
        self.quad_VAO = None
        self.quad_VBO = None
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize quad mesh for fullscreen rendering"""
        # Fullscreen quad
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        
        self.quad_vertices = quad_vertices
    
    def compile_shader(self):
        """Compile fog shader"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Fractal fog shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
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
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        uniform vec2 iResolution;
        uniform float iTime;
        uniform float speed;
        uniform float intensity;
        uniform float fadeAlpha;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define T (iTime * speed)
        
        // Rotation macro - exactly as in original
        #define r(v,t) { float a = (t)*T; float c=cos(a); float s=sin(a); v*=mat2(c,s,-s,c); }
        
        // Hash function
        float hash(float n) {
            return fract(sin(n) * 43758.5453);
        }
        
        // 3D noise function
        float noise(in vec3 x) {
            vec3 p = floor(x);
            vec3 f = fract(x);
            f = f * f * (3.0 - 2.0 * f);
            float n = p.x + p.y * 57.0 + 113.0 * p.z;
            float res = mix(
                mix(
                    mix(hash(n + 0.0), hash(n + 1.0), f.x),
                    mix(hash(n + 57.0), hash(n + 58.0), f.x),
                    f.y
                ),
                mix(
                    mix(hash(n + 113.0), hash(n + 114.0), f.x),
                    mix(hash(n + 170.0), hash(n + 171.0), f.x),
                    f.y
                ),
                f.z
            );
            return res;
        }
        
        const mat3 m = mat3(
            0.00,  0.80,  0.60,
           -0.80,  0.36, -0.48,
           -0.60, -0.48,  0.64
        );
        
        float fbm(vec3 p) {
            float f;
            f  = 0.5000 * noise(p); p = m * p * 2.02;
            f += 0.2500 * noise(p); p = m * p * 2.03;
            f += 0.1250 * noise(p); p = m * p * 2.01;
            f += 0.0625 * noise(p);
            return f;
        }
        
        #define snoise(x) (2.0 * noise(x) - 1.0)
        
        float sfbm(vec3 p) {
            float f;
            f  = 0.5000 * snoise(p); p = m * p * 2.02;
            f += 0.2500 * snoise(p); p = m * p * 2.03;
            f += 0.1250 * snoise(p); p = m * p * 2.01;
            f += 0.0625 * snoise(p);
            return f;
        }
        
        #define sfbm3(p) vec3(sfbm(p), sfbm(p - 327.67), sfbm(p + 327.67))
        
        void mainImage(out vec4 f, vec2 w) {
            // Viewport scaling - normalize by height for both dimensions to maintain aspect
            vec4 p = vec4(w, 0.0, 1.0) / iResolution.xyxy - 0.5;
            vec4 d, c;
            
            d = p;
            p.z += 10.0;
            
            vec4 bg = vec4(0.0, 0.2, 0.0, 0.0);
            f = bg;
            
            float x1, x2, x = 1e9;
            
            for (float i = 1.0; i > 0.0; i -= 0.01) {
                if (f.x >= 0.99) break;
                
                vec4 u = 0.03 * floor(p / vec4(8.0, 8.0, 1.0, 1.0) + 3.5);
                vec4 t = p;
                r(t.xy, u.x);
                r(t.xz, u.y);
                
                // Distortion exactly as original
                t.xyz += sfbm3(t.xyz / 2.0 + vec3(0.5 * T, 0.0, 0.0)) * (0.6 + 8.0 * (0.5 - 0.5 * cos(T / 16.0)));
                
                // Color from procedural texture (replacing iChannel0)
                float fbm_val = fbm(t.xyz * 0.3);
                c = vec4(vec3(5.0 * fbm_val), 1.0);
                
                x = abs(mod(length(t.xyz), 1.0) - 0.5);
                x1 = length(t.xyz) - 7.0;
                x = max(x, x1);
                
                if ((x1 > 0.1) && (p.z < 0.0)) break;
                
                if (x < 0.01) {
                    f += (1.0 - f) * 0.2 * mix(bg, c, i * i);
                    x = 0.1;
                }
                
                p += d * x;
            }
        }
        
        void main() {
            vec2 uv = fragCoord * iResolution;
            vec4 col;
            mainImage(col, uv);
            
            // Apply intensity and fade
            col.rgb *= intensity;
            
            // Alpha from brightness
            float brightness = col.r + col.g + col.b;
            col.a = clamp(brightness * 0.4, 0.0, 0.9) * fadeAlpha;
            
            outColor = col;
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Setup quad VAO
        self.quad_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.quad_VAO)
        
        self.quad_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, self.quad_vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        self.time += dt
        
        # Smooth audio values with exponential decay
        attack_factor = 1.0 - np.exp(-dt / 0.05)
        decay_factor = 1.0 - np.exp(-dt / 0.3)
        
        # Bass: affects density
        if self.audio_bass > self.audio_bass_smooth:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * attack_factor
        else:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * decay_factor
        
        # Mid: affects rotation
        if self.audio_mid > self.audio_mid_smooth:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * attack_factor
        else:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * decay_factor
        
        # High: affects distortion
        if self.audio_high > self.audio_high_smooth:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * attack_factor
        else:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * decay_factor
    
    def render(self, state: Dict):
        """Render the fractal fog effect"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        
        # Set uniforms
        resolution_loc = glGetUniformLocation(self.shader, "iResolution")
        glUniform2f(resolution_loc, float(self.viewport.width), float(self.viewport.height))
        
        time_loc = glGetUniformLocation(self.shader, "iTime")
        glUniform1f(time_loc, self.time)
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        speed_loc = glGetUniformLocation(self.shader, "speed")
        glUniform1f(speed_loc, self.base_speed)
        
        intensity_loc = glGetUniformLocation(self.shader, "intensity")
        glUniform1f(intensity_loc, self.base_intensity)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        # Render quad
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
