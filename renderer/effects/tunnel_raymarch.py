"""
Raymarched tunnel effect - animated 3D tunnel with distance field rendering
Creates a flowing tunnel with modulated geometry based on distance field raymarching
Based on shader by Stephane Cuillerdier - @Aiekick/2016 and shane's shader
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_tunnel_raymarch(state, outstate, speed=1.26, rotation_speed=0.375,
                           fade_duration=10.0, audio_reactive=True, 
                           audio_sensitivity=5.0, bass_sensitivity=5.0):
    """
    Audio-reactive raymarched tunnel effect compatible with EventScheduler
    
    Creates an animated 3D tunnel using distance field raymarching with audio reactivity
    
    Usage:
        scheduler.schedule_event(0, 60, shader_tunnel_raymarch, 
                               speed=1.5, rotation_speed=0.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        speed: Forward movement speed (default 1.26)
        rotation_speed: Camera rotation speed (default 0.375)
        fade_duration: Duration of fade in/out in seconds (default 10.0)
        audio_reactive: Enable audio reactivity (default True)
        audio_sensitivity: How much audio affects visuals (default 4.0, was 2.0)
        bass_sensitivity: How much bass affects tunnel width (default 5.0, was 2.5)
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    audio_data = outstate.get('sound')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing tunnel raymarch effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                TunnelRaymarchEffect,
                speed=speed,
                rotation_speed=rotation_speed,
                audio_reactive=audio_reactive,
                audio_sensitivity=audio_sensitivity,
                bass_sensitivity=bass_sensitivity
            )
            state['effect'] = effect
            state['smoothed_bands'] = np.zeros(32, dtype=np.float32)
            print(f"✓ Initialized shader tunnel_raymarch for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize tunnel raymarch: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters
    if 'effect' in state:
        state['effect'].speed = outstate.get('tunnel_speed', speed)
        state['effect'].rotation_speed = outstate.get('tunnel_rotation', rotation_speed)
        
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = np.clip(fade_factor, 0, 1)
        
        # Audio reactivity - pass all 32 bands
        if audio_reactive and audio_data is not None:
            bands = audio_data['norm_short'][0]  # Shape: (32,)
            
            # Initialize smoothed bands if first time
            if 'smoothed_bands' not in state:
                state['smoothed_bands'] = np.zeros(32, dtype=np.float32)
            
            # Smooth each band individually - faster response
            smoothing = 0.35  # Increased from 0.15 for more immediate response
            state['smoothed_bands'] = smoothing * bands + (1 - smoothing) * state['smoothed_bands']
            
            # Apply sensitivity and clamp to reasonable range
            # Use smoothed bands directly (no peak tracking for more dynamic response)
            state['effect'].audio_bands = np.clip(state['smoothed_bands'] * audio_sensitivity, 0, 5)  # Increased max to 5
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up tunnel raymarch effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader tunnel_raymarch for frame {frame_id}")

# ============================================================================
# Tunnel Raymarch Effect - Post-Processing
# ============================================================================

class TunnelRaymarchEffect(ShaderEffect):
    """Fullscreen post-processing effect with raymarched 3D tunnel"""
    
    def __init__(self, viewport, speed: float = 1.26, rotation_speed: float = 0.375,
                 audio_reactive: bool = True, audio_sensitivity: float = 5.0,
                 bass_sensitivity: float = 5.0):
        super().__init__(viewport)
        self.speed = speed
        self.rotation_speed = rotation_speed
        self.audio_reactive = audio_reactive
        self.audio_sensitivity = audio_sensitivity
        self.bass_sensitivity = bass_sensitivity
        self.fade_factor = 0.0
        
        # Audio reactivity - store all 32 frequency bands
        self.audio_bands = np.zeros(32, dtype=np.float32)
        
        self.time = 0.0
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position * 2.0 - 1.0, 0.5, 1.0);
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform vec2 resolution;
        uniform float time;
        uniform float speed;
        uniform float rotationSpeed;
        uniform float fadeAlpha;
        uniform float audioBands[32];  // All 32 frequency bands for per-layer assignment
        
        // Global distance step factor (accumulates during raymarching)
        float dstepf;
        
        // Get rotation matrix around Z axis
        mat3 getRotZMat(float a) {
            return mat3(
                cos(a), -sin(a), 0.0,
                sin(a), cos(a), 0.0,
                0.0, 0.0, 1.0
            );
        }
        
        // Distance field function - defines the tunnel geometry
        float map(vec3 p) {
            // Modulate position based on Z (creates wavy tunnel) - reduced for stability
            p.x += sin(p.z * 1.2) * 0.5;
            p.y += cos(p.z * 0.15) * sin(p.x * 0.5) * 0.5;
            
            // Rotate based on position only - stable, no audio wiggle
            float rotFactor = p.z * 0.4 + sin(p.x) * 0.3 + cos(p.y) * 0.3;
            p *= getRotZMat(rotFactor);
            
            // Create repeating cells (tunnel segments) - fixed size
            float cellSize = 0.3;
            p.xy = mod(p.xy, cellSize) - cellSize * 0.5;
            
            // Accumulate distance step factor
            dstepf += 0.003;
            
            // Return distance to nearest tunnel wall
            return length(p.xy);
        }
        
        void main() {
            // Normalize coordinates (centered, aspect-corrected)
            vec2 uv = (fragCoord * resolution - resolution * 0.5) / resolution.y;
            
            // Calculate aspect ratio for better scaling
            float aspectRatio = resolution.x / resolution.y;
            
            // Adjust FOV based on aspect ratio to prevent stretching
            // Wider screens get wider FOV, narrower screens get adjusted
            float fov = 1.4;  // Increased for zoom out effect
            if (aspectRatio > 1.0) {
                // Wide screen - scale FOV to maintain proper proportions
                fov = 1.2 + (aspectRatio - 1.0) * 0.2;
            }
            
            // Ray direction with minimal fisheye distortion for stability
            vec3 rd = normalize(vec3(uv * fov, (1.0 - dot(uv, uv) * 0.1) * 0.8));
            
            // Ray origin - moves forward over time
            vec3 ro = vec3(0.0, 0.0, time * speed);
            
            // Rotate camera based on time
            float cs = cos(time * rotationSpeed);
            float si = sin(time * rotationSpeed);
            rd.xz = mat2(cs, si, -si, cs) * rd.xz;
            
            vec3 col = vec3(0.0);  // Accumulated color
            vec3 sp;               // Sample point
            float t = 0.06;        // Ray distance
            float layers = 0.0;    // Layer counter
            float d = 0.0;         // Distance to surface
            float aD;              // Accumulated distance
            float thD = 0.02;      // Threshold distance
            dstepf = 0.0;          // Reset distance step factor
            
            // Raymarching loop
            for (float i = 0.0; i < 250.0; i++) {
                // Early exit conditions
                if (layers > 4.0 || col.x > 1.0 || t > 5.6) break;
                
                // Sample point along ray
                sp = ro + rd * t;
                
                // Get distance to nearest surface
                d = map(sp);
                
                // Calculate contribution based on proximity to surface
                aD = (thD - abs(d) * 15.0 / 16.0) / thD;
                
                // If close to surface, accumulate color
                if (aD > 0.0) {
                    // Assign each layer to spread across frequency spectrum
                    // Layer 0 -> band 0 (sub-bass), Layer 1 -> band 10 (low-mids), 
                    // Layer 2 -> band 20 (mids-highs), Layer 3 -> band 28 (highs)
                    int bandIndex = int(layers * 8.0);  // Spread across spectrum (0, 8, 16, 24)
                    bandIndex = min(bandIndex, 31);  // Clamp to valid range
                    float bandEnergy = audioBands[bandIndex];
                    
                    // Generate unique color per layer - darker base colors
                    vec3 layerColor = 0.15 + 0.25 * sin(  // Even darker base
                        vec3(0.0, 2.1, 4.2) + 
                        t * 3.0 + 
                        sp.z * 2.0 + 
                        layers * 1.5 +  // Increased spacing for more color variety (was 0.8)
                        bandEnergy * 4.0  // Increased for more dramatic color shifts (was 3.0)
                    );
                    
                    // Smoothstep for smooth contribution
                    float contribution = aD * aD * (3.0 - 2.0 * aD);
                    
                    // Fade with distance
                    float distanceFade = 1.0 / (1.0 + t * t * 0.25);
                    contribution *= distanceFade * 0.1;  // Much lower base contribution
                    
                    // Band energy modulates thickness DRAMATICALLY
                    contribution *= (0.3 + bandEnergy * 1.2);  // Range: 0.3 to 6.3 (more contrast)
                    
                    // Band energy affects brightness - MORE responsive
                    float audioBrightness = 0.08 + bandEnergy * 0.7;  // Range: 0.08 to 3.58 (more dynamic)
                    layerColor *= audioBrightness;
                    
                    col += layerColor * contribution;
                    layers++;
                }
                
                // Step along ray (adaptive step size)
                t += max(d * 0.7, thD * 1.5) * dstepf;
            }
            
            // Clamp color to prevent oversaturation
            col = min(col, vec3(1.0));
            
            // Calculate alpha based on accumulated color (transparent background)
            float colorBrightness = max(col.r, max(col.g, col.b));  // Use brightest channel
            
            // Boost alpha so tunnels are more visible even when dim
            // Alpha is higher than color brightness to maintain visibility
            float alpha = colorBrightness * 2.5;  // Multiply by 2.5 for better visibility
            alpha = clamp(alpha, 0.0, 1.0) * fadeAlpha;  // Clamp and apply fade factor
            
            outColor = vec4(col, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Setup fullscreen quad for post-processing"""
        # Fullscreen quad vertices (texture coordinates)
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update animation time"""
        if not self.enabled:
            return
        
        self.time += dt
    
    def render(self, state: Dict):
        """Render fullscreen post-processing effect"""
        if not self.enabled or not self.shader:
            return
        
        # Post-process exception: Always pass depth test without writing
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        if res_loc != -1:
            glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        if time_loc != -1:
            glUniform1f(time_loc, self.time)
        
        speed_loc = glGetUniformLocation(self.shader, "speed")
        if speed_loc != -1:
            glUniform1f(speed_loc, self.speed)
        
        rot_loc = glGetUniformLocation(self.shader, "rotationSpeed")
        if rot_loc != -1:
            glUniform1f(rot_loc, self.rotation_speed)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if fade_loc != -1:
            glUniform1f(fade_loc, self.fade_factor)
        
        # Audio uniforms - pass all 32 bands
        bands_loc = glGetUniformLocation(self.shader, "audioBands")
        if bands_loc != -1:
            if self.audio_reactive:
                glUniform1fv(bands_loc, 32, self.audio_bands)
            else:
                glUniform1fv(bands_loc, 32, np.zeros(32, dtype=np.float32))
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore default depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
