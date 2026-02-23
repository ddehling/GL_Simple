"""
Tunnel shader effect - Raymarched fractal tunnel with camera movement
Converted from Shadertoy: https://www.shadertoy.com/view/llXXzf
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_tunnel(state, outstate, depth=50.0, intensity=1.0, speed=1.0,
                  bass_sensitivity=0.5, mid_sensitivity=0.3, high_sensitivity=0.2):
    """
    Shader-based tunnel effect compatible with EventScheduler
    
    Creates a raymarched fractal tunnel with rotating camera movement.
    Responds to audio with red channel (bass), green channel (mids), 
    and blue channel (highs) color modulation.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_tunnel, depth=50, 
                               bass_sensitivity=0.8, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        depth: Z-depth of tunnel (0=near, 100=far, default: 50)
        intensity: Base brightness multiplier (default: 1.0)
        speed: Base animation speed multiplier (default: 1.0)
        bass_sensitivity: How much bass frequencies affect red channel (default: 0.5)
        mid_sensitivity: How much mid frequencies affect green channel (default: 0.3)
        high_sensitivity: How much high frequencies affect blue channel (default: 0.2)
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
        print(f"Initializing tunnel for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                TunnelEffect,
                depth=depth,
                intensity=intensity,
                speed=speed,
                bass_sensitivity=bass_sensitivity,
                mid_sensitivity=mid_sensitivity,
                high_sensitivity=high_sensitivity
            )
            state['effect'] = effect
            print(f"✓ Initialized shader tunnel for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize tunnel: {e}")
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
        state['effect'].base_intensity = outstate.get('tunnel_intensity', intensity)
        state['effect'].base_speed = outstate.get('tunnel_speed', speed)
        
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
            print(f"Cleaning up tunnel for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader tunnel for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class TunnelEffect(ShaderEffect):
    """
    Raymarched tunnel effect with fractal geometry
    
    Creates a 3D tunnel using raymarching with fractal-like patterns
    and camera rotation.
    """
    
    def __init__(self, viewport, depth: float = 50.0, intensity: float = 1.0, 
                 speed: float = 1.0, bass_sensitivity: float = 0.5, 
                 mid_sensitivity: float = 0.3, high_sensitivity: float = 0.2):
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
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.EBO = None
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize fullscreen quad for raymarching"""
        # Fullscreen quad vertices (clip space coordinates)
        self.vertices = np.array([
            [-1.0, -1.0],  # Bottom-left
            [ 1.0, -1.0],  # Bottom-right
            [ 1.0,  1.0],  # Top-right
            [-1.0,  1.0],  # Top-left
        ], dtype=np.float32)
        
        # Two triangles forming a quad
        self.indices = np.array([
            0, 1, 2,  # First triangle
            0, 2, 3   # Second triangle
        ], dtype=np.uint32)
    
    def compile_shader(self):
        """Compile and link tunnel shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Tunnel shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 vUV;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vUV = position * 0.5 + 0.5;  // Convert from [-1,1] to [0,1]
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vUV;
        
        uniform vec2 resolution;
        uniform float time;
        uniform float depth;
        uniform float intensity;
        uniform float speed;
        uniform float fadeAlpha;
        uniform float audioBass;   // Bass energy for speed modulation
        uniform float audioMid;    // Mid energy for brightness
        uniform float audioHigh;   // High energy for color intensity
        
        out vec4 outColor;
        
        const float pi = 3.14159265359;
        
        // Rotation matrices
        mat3 xrot(float t) {
            return mat3(1.0, 0.0, 0.0,
                       0.0, cos(t), -sin(t),
                       0.0, sin(t), cos(t));
        }
        
        mat3 yrot(float t) {
            return mat3(cos(t), 0.0, -sin(t),
                       0.0, 1.0, 0.0,
                       sin(t), 0.0, cos(t));
        }
        
        mat3 zrot(float t) {
            return mat3(cos(t), -sin(t), 0.0,
                       sin(t), cos(t), 0.0,
                       0.0, 0.0, 1.0);
        }
        
        // Distance field function
        vec2 map(vec3 p) {
            p.x += sin(p.z);
            p *= zrot(p.z);
            float d = 1000.0;
            vec3 q = fract(p) * 2.0 - 1.0;
            float idx = 0.0;
            
            for (int i = 0; i < 3; ++i) {
                q = sign(q) * (1.0 - 1.0 / (1.0 + abs(q) * 0.8));
                
                float md = length(q) - 0.5;
                
                float ss = 0.5 + 0.5 * sin(p.z + md * float(i) * 6.0);
                float cyl = length(p.xy) - 0.5 - ss;
                
                md = max(md, -cyl);
                
                if (md < d) {
                    d = md;
                    idx = float(i);
                }
            }
            return vec2(d, idx);
        }
        
        // Calculate normal using finite differences
        vec3 normal(vec3 p) {
            vec3 o = vec3(0.1, 0.0, 0.0);
            return normalize(vec3(
                map(p + o.xyy).x - map(p - o.xyy).x,
                map(p + o.yxy).x - map(p - o.yxy).x,
                map(p + o.yyx).x - map(p - o.yyx).x
            ));
        }
        
        // Raymarching
        float trace(vec3 o, vec3 r) {
            float t = 0.0;
            for (int i = 0; i < 64; ++i) {
                vec3 p = o + r * t;
                float d = map(p).x;
                t += d * 0.3;
            }
            return t;
        }
        
        // Procedural texture (replacing iChannel0)
        vec3 proceduralTexture(vec3 p) {
            // Create a procedural pattern similar to what a texture might provide
            vec3 ta = vec3(
                0.5 + 0.5 * sin(p.y * 3.0 + p.z * 2.0),
                0.5 + 0.5 * sin(p.y * 2.5 + p.z * 1.5),
                0.5 + 0.5 * sin(p.y * 2.0 + p.z * 3.0)
            );
            vec3 tb = vec3(
                0.5 + 0.5 * sin(p.x * 2.5 + p.z * 2.5),
                0.5 + 0.5 * sin(p.x * 3.0 + p.z * 1.0),
                0.5 + 0.5 * sin(p.x * 1.5 + p.z * 2.0)
            );
            vec3 tc = vec3(
                0.5 + 0.5 * sin(p.x * 2.0 + p.y * 2.0),
                0.5 + 0.5 * sin(p.x * 1.5 + p.y * 3.0),
                0.5 + 0.5 * sin(p.x * 3.0 + p.y * 1.5)
            );
            return (ta + tb + tc) / 3.0;
        }
        
        // Environment map approximation (replacing iChannel1)
        vec3 environmentMap(vec3 dir) {
            // Create a simple gradient-based environment
            float y = dir.y * 0.5 + 0.5;
            vec3 sky = mix(vec3(0.1, 0.2, 0.4), vec3(0.4, 0.6, 1.0), y);
            return sky * 0.3;
        }
        
        void main() {
            vec2 uv = vUV * 2.0 - 1.0;
            uv.x *= resolution.x / resolution.y;
            
            // Ray direction with subtle fish-eye effect
            vec3 r = normalize(vec3(uv, 1.0 - dot(uv, uv) * 0.33));
            
            // Camera rotation - constant speed
            r *= zrot(time * speed * 0.25) * yrot(-sin(time * speed));
            
            // Camera position
            vec3 o = vec3(0.0, 0.0, 0.0);
            o.z += time * speed;
            o.x += -sin(o.z);
            
            // Raymarch the scene
            float t = trace(o, r);
            vec3 w = o + r * t;
            vec3 sn = normal(w);
            vec2 fd = map(w);
            vec3 ref = reflect(r, sn);
            
            // Color based on surface index
            vec3 diff = vec3(0.0);
            if (fd.y == 0.0) {
                diff = vec3(1.0, 0.0, 0.0);
            } else if (fd.y == 1.0) {
                diff = vec3(0.0, 1.0, 0.0);
            } else if (fd.y == 2.0) {
                diff = vec3(0.0, 0.0, 1.0);
            } else {
                diff = vec3(1.0, 1.0, 1.0);
            }
            
            // Add procedural texture
            diff += proceduralTexture(w) * 0.3;
            
            // Add environment reflection
            diff += environmentMap(ref) * 0.5;
            
            // Mix based on surface normal
            diff = mix(diff, vec3(1.0), abs(sn.y));
            diff = mix(vec3(0.8, 0.0, 0.0), diff, abs(sn.y));
            
            // Lighting
            float prod = max(dot(sn, -r), 0.0);
            diff *= prod;
            
            // Fog
            float fog = 1.0 / (1.0 + t * t * 0.1 + fd.x * 100.0);
            vec3 fc = diff * fog * intensity;
            
            // Audio modulation affects individual color channels
            fc.r *= (1.0 + audioBass * 1.2);
            fc.g *= (1.0 + audioMid * 1.2);
            fc.b *= (1.0 + audioHigh * 1.2);
            
            // Gamma correction
            fc = sqrt(fc);
            
            // Map depth for proper 3D ordering
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            gl_FragDepth = mappedDepth;
            
            // Apply fade
            outColor = vec4(fc, fadeAlpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Generate VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Generate and bind position VBO
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Generate and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        # Update animation time
        self.time += dt
        
        # Smooth audio values with exponential decay
        attack_factor = 1.0 - np.exp(-dt / 0.05)
        decay_factor = 1.0 - np.exp(-dt / 0.3)
        
        # Bass: quick attack, medium decay
        if self.audio_bass > self.audio_bass_smooth:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * attack_factor
        else:
            self.audio_bass_smooth += (self.audio_bass - self.audio_bass_smooth) * decay_factor
        
        # Mid: quick attack, medium decay
        if self.audio_mid > self.audio_mid_smooth:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * attack_factor
        else:
            self.audio_mid_smooth += (self.audio_mid - self.audio_mid_smooth) * decay_factor
        
        # High: quick attack, medium decay
        if self.audio_high > self.audio_high_smooth:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * attack_factor
        else:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * decay_factor
    
    def render(self, state: Dict):
        """Render the tunnel effect"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        
        # Set uniforms
        resolution_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(resolution_loc, float(self.viewport.width), float(self.viewport.height))
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        intensity_loc = glGetUniformLocation(self.shader, "intensity")
        glUniform1f(intensity_loc, self.base_intensity)
        
        speed_loc = glGetUniformLocation(self.shader, "speed")
        glUniform1f(speed_loc, self.base_speed)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        # Set audio modulation uniforms (using smoothed values)
        bass_loc = glGetUniformLocation(self.shader, "audioBass")
        glUniform1f(bass_loc, self.audio_bass_smooth)
        
        mid_loc = glGetUniformLocation(self.shader, "audioMid")
        glUniform1f(mid_loc, self.audio_mid_smooth)
        
        high_loc = glGetUniformLocation(self.shader, "audioHigh")
        glUniform1f(high_loc, self.audio_high_smooth)
        
        # Render fullscreen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.position_VBO is not None:
            glDeleteBuffers(1, [self.position_VBO])
        if self.EBO is not None:
            glDeleteBuffers(1, [self.EBO])
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
