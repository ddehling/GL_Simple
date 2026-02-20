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
            state['effect'].audio_bass = bass_energy
            state['effect'].audio_mid = mid_energy
            state['effect'].audio_high = high_energy
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
        uniform float audioBass;
        uniform float audioHigh;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        #define T (iTime * speed)
        
        // Color shift based on audio and time
        vec3 getAudioColor(float t, float bassLevel, float highLevel) {
            // Base color cycles through spectrum over time
            float hue = fract(t * 0.05 + bassLevel * 0.3);
            
            // High frequencies add color variation
            hue += highLevel * 0.2;
            
            // Convert HSV to RGB (increased saturation and value for wider range)
            float h6 = hue * 6.0;
            float x = 1.0 - abs(mod(h6, 2.0) - 1.0);
            vec3 rgb;
            
            if (h6 < 1.0) rgb = vec3(1.0, x, 0.0);
            else if (h6 < 2.0) rgb = vec3(x, 1.0, 0.0);
            else if (h6 < 3.0) rgb = vec3(0.0, 1.0, x);
            else if (h6 < 4.0) rgb = vec3(0.0, x, 1.0);
            else if (h6 < 5.0) rgb = vec3(x, 0.0, 1.0);
            else rgb = vec3(1.0, 0.0, x);
            
            // Increase saturation and brightness for wider color range
            float saturation = 0.75;  // Increased from 0.4
            float brightness = 0.85;  // Increased from 0.5
            vec3 gray = vec3(0.5);
            
            return mix(gray, rgb, saturation) * brightness;
        }
        
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
            
            // Dark background (no tint)
            vec4 bg = vec4(0.0, 0.0, 0.0, 0.0);
            f = bg;
            
            float x1, x2, x = 1e9;
            
            for (float i = 1.0; i > 0.0; i -= 0.01) {
                if (f.x >= 0.99) break;
                
                vec4 u = 0.03 * floor(p / vec4(8.0, 8.0, 1.0, 1.0) + 3.5);
                vec4 t = p;
                r(t.xy, u.x);
                r(t.xz, u.y);
                
                // Audio-reactive distortion: bass increases base amount, highs add variation
                float distortion_amount = 0.6 + 8.0 * (0.5 - 0.5 * cos(T / 16.0));
                distortion_amount *= (1.0 + audioBass * 0.3);  // Reduced bass influence
                distortion_amount += audioHigh * 0.8;  // Reduced high influence
                
                t.xyz += sfbm3(t.xyz / 2.0 + vec3(0.5 * T, 0.0, 0.0)) * distortion_amount;
                
                // Color from procedural texture with spatial color variation
                float fbm_val = fbm(t.xyz * 0.3);
                
                // Create discrete color domains using floor() for sharp boundaries
                // This gives each "blob" of fog a consistent color rather than gradients
                
                // Domain ID based on quantized position (creates distinct regions)
                vec3 domainPos = floor(t.xyz * 0.3);  // Quantize space into cells
                float domainID = fbm(domainPos);  // Each cell gets a unique ID
                
                // Domain 1: Large blobs - warm colors, bass reactive
                float domain1ID = floor(fbm(floor(t.xyz * 0.15)) * 5.0);
                vec3 domain1Color = getAudioColor(
                    domain1ID + T * 0.1, 
                    audioBass * 2.0, 
                    0.1
                );
                
                // Domain 2: Medium blobs - cool colors, high reactive
                float domain2ID = floor(fbm(floor(t.xyz * 0.25 + vec3(100.0, 50.0, 25.0))) * 5.0);
                vec3 domain2Color = getAudioColor(
                    domain2ID + T * 0.15,
                    0.1,
                    audioHigh * 2.0
                );
                
                // Domain 3: Small blobs - varied colors, mixed reactive
                float domain3ID = floor(fbm(floor(t.xyz * 0.4 + vec3(200.0, 100.0, 50.0))) * 5.0);
                vec3 domain3Color = getAudioColor(
                    domain3ID + T * 0.2,
                    audioBass * 0.8,
                    audioHigh * 0.8
                );
                
                // Domain 4: Depth layers - slow shifting
                float domain4ID = floor(length(floor(t.xyz * 0.2)) * 0.5);
                vec3 domain4Color = getAudioColor(
                    domain4ID + T * 0.05,
                    audioBass * 0.5,
                    audioHigh * 0.5
                );
                
                // Use noise to select which domain controls each blob
                float selector = fbm(floor(t.xyz * 0.2));
                vec3 fogColor;
                
                if (selector < 0.25) {
                    fogColor = domain1Color;
                } else if (selector < 0.5) {
                    fogColor = domain2Color;
                } else if (selector < 0.75) {
                    fogColor = domain3Color;
                } else {
                    fogColor = domain4Color;
                }
                
                // Apply brightness
                c = vec4(fogColor * (1.5 * fbm_val), 1.0);
                
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
            col.rgb *= intensity * 2.0;
            
            // Alpha from brightness (increased for better visibility)
            float brightness = col.r + col.g + col.b;
            col.a = clamp(brightness * 2.5, 0.0, 1.0) * fadeAlpha;
            
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
        
        # Smooth audio values with exponential decay (slower for less reactive)
        attack_factor = 1.0 - np.exp(-dt / 0.15)
        decay_factor = 1.0 - np.exp(-dt / 0.5)
        
        # Bass: affects density
        if self.audio_bass > self.audio_bass_smooth:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * attack_factor
        else:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * decay_factor
        
        # Mid: affects rotation speed
        if self.audio_mid > self.audio_mid_smooth:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * attack_factor
        else:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * decay_factor
        
        # High: affects distortion amount
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
        
        # Modulate speed with mid frequencies (reduced sensitivity)
        speed_mod = self.base_speed * (1.0 + self.audio_mid_smooth * self.mid_sensitivity * 0.3)
        speed_loc = glGetUniformLocation(self.shader, "speed")
        glUniform1f(speed_loc, speed_mod)
        
        # Modulate intensity with bass (reduced sensitivity)
        intensity_mod = self.base_intensity * (1.0 + self.audio_bass_smooth * self.bass_sensitivity * 0.2)
        intensity_loc = glGetUniformLocation(self.shader, "intensity")
        glUniform1f(intensity_loc, intensity_mod)
        
        # Pass audio values to shader for distortion (reduced)
        audio_bass_loc = glGetUniformLocation(self.shader, "audioBass")
        glUniform1f(audio_bass_loc, self.audio_bass_smooth * self.bass_sensitivity * 0.25)
        
        audio_high_loc = glGetUniformLocation(self.shader, "audioHigh")
        glUniform1f(audio_high_loc, self.audio_high_smooth * self.high_sensitivity * 0.25)
        
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
