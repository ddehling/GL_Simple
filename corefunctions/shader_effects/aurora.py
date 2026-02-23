"""
Aurora shader effect - Flowing northern lights at top of screen
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_aurora(state, outstate, height_ratio=0.75, depth=75.0, intensity=1.0, speed=1.0,
                  bass_sensitivity=0.5, mid_sensitivity=0.3, high_sensitivity=0.2):
    """
    Shader-based aurora effect compatible with EventScheduler
    
    Creates flowing northern lights at the top of the screen with animated
    waves and color gradients. Responds to audio with height pulsing (bass),
    brightness changes (mids), and speed modulation (highs).
    
    Usage:
        scheduler.schedule_event(0, 60, shader_aurora, height_ratio=0.3, depth=75, 
                               bass_sensitivity=0.8, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        height_ratio: Height of aurora as proportion of viewport height (0.0-1.0, default: 0.25)
        depth: Z-depth of aurora (0=near, 100=far, default: 75)
        intensity: Base brightness multiplier (default: 1.0)
        speed: Base animation speed multiplier (default: 1.0)
        bass_sensitivity: How much bass frequencies affect height (default: 0.5)
        mid_sensitivity: How much mid frequencies affect brightness (default: 0.3)
        high_sensitivity: How much high frequencies affect speed (default: 0.2)
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
                bass_sensitivity=bass_sensitivity,
                mid_sensitivity=mid_sensitivity,
                high_sensitivity=high_sensitivity
            )
            state['effect'] = effect
            print(f"✓ Initialized shader aurora for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize aurora: {e}")
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
        state['effect'].base_intensity = outstate.get('aurora_intensity', intensity)
        state['effect'].base_speed = outstate.get('aurora_speed', speed)
        
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
            print(f"Cleaning up aurora for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader aurora for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class Aurora(ShaderEffect):
    """
    Aurora borealis effect with flowing waves and color gradients
    
    Creates animated northern lights at the top of the screen using
    layered sine waves with varying frequencies and amplitudes.
    """
    
    def __init__(self, viewport, height_ratio: float = 0.75, depth: float = 75.0, 
                 intensity: float = 1.0, speed: float = 1.0,
                 bass_sensitivity: float = 0.5, mid_sensitivity: float = 0.3,
                 high_sensitivity: float = 0.2):
        super().__init__(viewport)
        self.height_ratio = height_ratio
        self.height = int(viewport.height * height_ratio)  # Calculate actual pixel height
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
        
        # Smoothed audio values for stable modulation (with decay)
        self.audio_bass_smooth = 0.0
        self.audio_mid_smooth = 0.0
        self.audio_high_smooth = 0.0
        
        # Color flow offset - only moves forward, never backward
        self.color_flow_offset = 0.0
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.EBO = None
        
                # Mesh resolution (higher = smoother waves)
        self.segments_x = 100  # Horizontal segments
        self.segments_y = 20   # Vertical segments
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize mesh data for aurora plane"""
        # Create a grid mesh at the top of the screen
        width = self.viewport.width
        
        # Generate vertex positions for a grid
        vertices = []
        for y in range(self.segments_y + 1):
            y_pos = (y / self.segments_y) * self.height
            for x in range(self.segments_x + 1):
                x_pos = (x / self.segments_x) * width
                vertices.append([x_pos, y_pos])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Generate indices for triangle strip rendering
        indices = []
        for y in range(self.segments_y):
            for x in range(self.segments_x + 1):
                # Two vertices per column (current row and next row)
                indices.append(y * (self.segments_x + 1) + x)
                indices.append((y + 1) * (self.segments_x + 1) + x)
            # Degenerate triangles to connect strips
            if y < self.segments_y - 1:
                indices.append((y + 1) * (self.segments_x + 1) + self.segments_x)
                indices.append((y + 1) * (self.segments_x + 1))
        
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = len(self.indices)
    
    def compile_shader(self):
        """Compile and link aurora shaders"""
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
        uniform float time;
        uniform float height;
        uniform float speed;
        uniform float audioBass;  // Bass energy for height modulation
        uniform float colorFlowOffset;  // Monotonically increasing color flow
        
        out vec2 vTexCoord;
        out float vWave;
        out float vCurtainDepth;
        out float vColorFlow;
        
        void main() {
            vec2 pos = position;
            
            // Calculate normalized coordinates
            float normY = pos.y / height;  // 0 at top, 1 at bottom
            float normX = pos.x / resolution.x;
            
            // Create flowing wave pattern with multiple layers
            float wave1 = sin(normX * 6.28318 * 2.0 + time * speed * 0.5) * 0.3;
            float wave2 = sin(normX * 6.28318 * 3.0 - time * speed * 0.7) * 0.2;
            float wave3 = sin(normX * 6.28318 * 1.5 + time * speed * 0.3) * 0.25;
            
            // Add curtain-like vertical waves that flow down
            float curtainWave1 = sin(normX * 6.28318 * 8.0 + normY * 12.0 - time * speed * 1.2) * 0.15;
            float curtainWave2 = sin(normX * 6.28318 * 5.0 - normY * 8.0 + time * speed * 0.8) * 0.12;
            
            // Create rippling effect that gives depth perception
            float ripple = sin(normX * 6.28318 * 10.0 + time * speed) * 
                          cos(normY * 6.28318 * 2.0 - time * speed * 0.5) * 0.08;
            
            // Modulate wave amplitude with bass - more bass = bigger waves
            float bassModulation = 1.0 + audioBass * 2.0;
            float totalWave = (wave1 + wave2 + wave3 + curtainWave1 + curtainWave2 + ripple) 
                             * normY * 30.0 * bassModulation;
            pos.y += totalWave;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, mappedDepth, 1.0);
            
            // Pass data to fragment shader
            vTexCoord = vec2(normX, normY);
            vWave = (wave1 + wave2 + wave3) * 0.5 + 0.5;  // Normalize to 0-1
            vCurtainDepth = (curtainWave1 + curtainWave2) * 0.5 + 0.5;
            // Color flows downward - subtract normY so colors advance as y increases
            vColorFlow = colorFlowOffset - normY * 2.0;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vTexCoord;
        in float vWave;
        in float vCurtainDepth;
        in float vColorFlow;
        
        uniform float time;
        uniform float intensity;
        uniform float speed;
        uniform float fadeAlpha;
        uniform float audioMid;   // Mid energy for brightness
        uniform float audioHigh;  // High energy for speed/shimmer
        
        out vec4 outColor;
        
        // Aurora color palette
        vec3 getAuroraColor(float flowPosition, float wave) {
            // Blend between aurora colors: green, blue, purple, pink
            vec3 color1 = vec3(0.0, 1.0, 0.4);    // Bright green
            vec3 color2 = vec3(0.0, 0.8, 1.0);    // Cyan
            vec3 color3 = vec3(0.5, 0.0, 1.0);    // Purple
            vec3 color4 = vec3(1.0, 0.2, 0.8);    // Pink
            
            // Add wave influence to flow position
            float waveInfluence = wave * 0.3;
            
            // Seamless cyclic blending through all 4 colors
            // flowPosition only increases, so colors only flow forward
            float blendPos = fract(flowPosition + waveInfluence) * 4.0;
            
            vec3 color;
            if (blendPos < 1.0) {
                color = mix(color1, color2, blendPos);
            } else if (blendPos < 2.0) {
                color = mix(color2, color3, blendPos - 1.0);
            } else if (blendPos < 3.0) {
                color = mix(color3, color4, blendPos - 2.0);
            } else {
                // Smooth transition back to color1 to complete the cycle
                color = mix(color4, color1, blendPos - 3.0);
            }
            
            return color;
        }
        
        void main() {
            float x = vTexCoord.x;
            float y = vTexCoord.y;
            
            // Create multiple overlapping band layers for depth
            // Use mod to ensure seamless wrapping
            float xWrapped = mod(x, 1.0);
            float bands1 = sin(xWrapped * 6.28318 * 4.0 + time * speed * 0.8 + vWave * 2.0) * 0.5 + 0.5;
            float bands2 = sin(xWrapped * 6.28318 * 6.0 - time * speed * 0.5 + y * 3.0) * 0.5 + 0.5;
            float bands3 = sin(xWrapped * 6.28318 * 3.0 + time * speed * 0.3 - y * 2.0) * 0.5 + 0.5;
            
            // Combine bands with different weights for layered effect
            float bands = (bands1 * 0.5 + bands2 * 0.3 + bands3 * 0.2);
            bands = pow(bands, 1.3);
            
            // Add vertical curtain structures
            float curtainPattern = sin(xWrapped * 6.28318 * 12.0 + y * 8.0 - time * speed) * 
                                  sin(xWrapped * 6.28318 * 7.0 - y * 5.0 + time * speed * 0.7);
            curtainPattern = curtainPattern * 0.5 + 0.5;
            curtainPattern = smoothstep(0.3, 0.7, curtainPattern);
            
            // Create ray-like structures emanating downward
            float rays = 0.0;
            for (float i = 0.0; i < 5.0; i += 1.0) {
                float offset = i * 0.2 + time * speed * 0.1;
                float rayX = fract(xWrapped * 3.0 + offset);
                float rayIntensity = exp(-20.0 * pow(rayX - 0.5, 2.0));
                rays += rayIntensity * (1.0 - y * 0.7);
            }
            rays *= 0.3;
            
            // Vertical gradient - brighter at top, fade toward bottom
            float verticalFade = 1.0 - y;
            verticalFade = pow(verticalFade, 0.7);
            
            // Add soft edge falloff at the bottom to prevent sharp edges on short viewports
            // This creates a smooth transition in the bottom 30% of the aurora
            float edgeFalloff = smoothstep(1.0, 0.7, y);
            verticalFade *= edgeFalloff;
            
            // Add shimmer effect - enhanced by high frequencies
            float shimmerSpeed = 1.0 + audioHigh * 2.0;
            float shimmer = sin(xWrapped * 20.0 + time * speed * 2.0 * shimmerSpeed) * 
                          sin(y * 15.0 - time * speed * 1.5) * 0.1 + 0.9;
            
            // Create spatial variation in color across the aurora
            float colorVariation = sin(xWrapped * 6.28318 * 2.5) * 0.2;
            
            // Get aurora color - flows downward only via vColorFlow
            vec3 color = getAuroraColor(vColorFlow + xWrapped * 0.5 + colorVariation, 
                                       vWave * 0.7 + vCurtainDepth * 0.3);
            
            // Combine all structural elements
            float structure = bands * (0.7 + curtainPattern * 0.3) + rays;
            
            // Combine effects - mid frequencies boost brightness
            float audioBrightness = 1.0 + audioMid * 1.5;
            float brightness = structure * verticalFade * shimmer * intensity * audioBrightness;
            
            // Add subtle glow at edges of bands
            float edgeGlow = smoothstep(0.3, 0.7, bands) * (1.0 - smoothstep(0.7, 1.0, bands));
            brightness += edgeGlow * 0.3;
            
            // Add depth-based brightness variation (curtain depth affects intensity)
            brightness *= (0.8 + vCurtainDepth * 0.4);
            
            // Calculate alpha with more variation
            float alpha = verticalFade * structure * 0.7;
            alpha = clamp(alpha, 0.0, 0.85);
            
            // Apply fade in/out
            alpha *= fadeAlpha;
            
            outColor = vec4(color * brightness, alpha);
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
        
        # Smooth audio values with exponential decay (prevents flickering)
        # Attack time: 0.05s, Decay time: 0.3s
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
        
        # High: quick attack, slower decay (for smoother color shifts)
        high_decay_factor = 1.0 - np.exp(-dt / 0.5)
        if self.audio_high > self.audio_high_smooth:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * attack_factor
        else:
            self.audio_high_smooth += (self.audio_high - self.audio_high_smooth) * high_decay_factor
        
        # Update color flow offset - always moves forward, speed affected by audio
        flow_speed = self.base_speed * (1.0 + self.audio_high_smooth * 0.5)
        self.color_flow_offset += dt * flow_speed * 0.15
    
    def render(self, state: Dict):
        """Render the aurora effect"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        
        # Set uniforms
        resolution_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(resolution_loc, float(self.viewport.width), float(self.viewport.height))
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        height_loc = glGetUniformLocation(self.shader, "height")
        glUniform1f(height_loc, self.height)
        
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
        
        # Set color flow offset (monotonically increasing)
        color_flow_loc = glGetUniformLocation(self.shader, "colorFlowOffset")
        glUniform1f(color_flow_loc, self.color_flow_offset)
        
        # Render mesh
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLE_STRIP, self.index_count, GL_UNSIGNED_INT, None)
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
