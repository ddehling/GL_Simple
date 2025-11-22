"""
Audio-reactive curve shader - draws a flowing waveform based on frequency data
The curve spans the entire screen and leaves a fading trail
GPU-accelerated: Most computation happens in shaders, CPU only passes audio bands and time
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
import ctypes

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_audio_curve(state, outstate, sensitivity=2.0, amplitude=300.0, trail_length=30):
    """
    Audio-reactive curve that flows across the screen
    
    Usage:
        scheduler.schedule_event(0, 60, shader_audio_curve, 
                               sensitivity=1.5, amplitude=100, trail_length=30, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        sensitivity: Multiplier for audio response (default 1.0, lower=less saturated)
        amplitude: Maximum vertical displacement in pixels (default 80.0)
        trail_length: Number of frames to keep in trail (default 30, higher=longer history)
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
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing audio_curve for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                AudioCurveEffect,
                sensitivity=sensitivity,
                amplitude=amplitude,
                trail_length=trail_length
            )
            state['effect'] = effect
            print(f"✓ Initialized shader audio_curve for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize audio_curve: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from audio data every frame
    if 'effect' in state and audio_data is not None:
        # Get norm_long_relu data (only above-average energy)
        bands = audio_data['norm_long_relu'][0]  # Shape: (32,)
        
        # Update effect with current audio data
        state['effect'].update_from_audio(bands)
        
        # Debug: Log first few frames
        if state['count'] < 5:
            print(f"Audio curve frame {state['count']}: bands min={bands.min():.3f} max={bands.max():.3f} mean={bands.mean():.3f}")
        
        # Optional: Implement fade in/out
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
            print(f"Cleaning up audio_curve for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader audio_curve for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class AudioCurveEffect(ShaderEffect):
    """Audio-reactive curve with fading trail effect - Full GPU acceleration with history"""
    
    def __init__(self, viewport, sensitivity: float = 2.0, amplitude: float = 100.0, trail_length: int = 30):
        super().__init__(viewport)
        self.sensitivity = sensitivity
        self.amplitude = amplitude
        self.trail_length = trail_length
        self.fade_factor = 0.0
        
        # Curve parameters
        self.num_points = 64  # Number of points along the curve
        self.line_thickness = 3.0
        
        # Visual parameters (passed to GPU)
        self.current_hue = 0.0  # Current hue value (0-1)
        self.hue_speed = 0.01  # How fast to rotate through colors
        self.depth = 50.0  # Mid-depth for the curve
        
        # Baseline oscillation parameters
        self.baseline_amplitude = 15.0  # Amplitude of baseline wave (reduced for more range)
        self.baseline_frequency = 2.0  # Frequency of baseline oscillation
        self.baseline_phase = 0.0  # Phase offset for animation
        self.baseline_speed = 2  # Speed of baseline animation
        
        # Frame counter for circular buffer
        self.frame_counter = 0
        
        # GPU-side circular buffer for audio history
        # Texture: (32 bands, trail_length frames) = 32x(trail_length) texture
        self.audio_history_texture = None
        self.audio_history_write_index = 0
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize GPU-side history buffer"""
        # Will be created in setup_buffers after shader is compiled
        self.audio_bands_current = np.zeros(32, dtype=np.float32)
    
    def update_from_audio(self, bands: np.ndarray):
        """Update GPU audio history - only 32 floats"""
        self.audio_bands_current = np.array(bands, dtype=np.float32).clip(0, 1)
        
        # Advance hue
        self.current_hue = (self.current_hue + self.hue_speed) % 1.0
        
        # Update will happen in render() to keep GPU buffer current
    
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
            print(f"AudioCurveEffect shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Input: which point on curve (0-63) and which frame in history (0-trail_length-1)
        layout(location = 0) in vec2 curvePoint;  // x=pointIndex, y=frameAge
        // Input: offset perpendicular to line
        layout(location = 1) in float perpOffset;
        
        // Uniforms
        uniform vec2 resolution;
        uniform float depth;
        uniform float baselineAmplitude;
        uniform float baselineFrequency;
        uniform float baselinePhase;
        uniform float sensitivity;
        uniform float amplitude;
        uniform float hue;
        uniform int trailLength;
        uniform int currentWriteIndex;
        
        // Audio history texture
        uniform sampler2D audioHistoryTex;
        
        out vec3 vColor;
        out float vAlpha;
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        // Sample audio band from history texture
        float sampleAudioBand(float bandIndex, int frameOffset) {
            float u = (bandIndex + 0.5) / 32.0;
            int textureFrame = (currentWriteIndex - frameOffset + trailLength) % trailLength;
            float v = (float(textureFrame) + 0.5) / float(trailLength);
            return texture(audioHistoryTex, vec2(u, v)).r;
        }
        
        // Compute Y position for a given point and frame
        float computeY(float pointIndex, int frameAge) {
            float numPoints = 64.0;
            float bandIndex = (pointIndex / numPoints) * 31.0;
            float audioBand = sampleAudioBand(bandIndex, frameAge);
            
            // Moderate compression: 0-0.6 range allows good dynamic range
            // while still preventing extreme saturation
            audioBand = audioBand * 0.6;
            
            // Apply sensitivity and amplitude
            // Audio pushes DOWN (increases Y), so multiply directly
            float audioDisplacement = audioBand * sensitivity * amplitude;
            
            // Calculate baseline oscillation
            float t = (pointIndex / numPoints) * 2.0 * 3.14159 * baselineFrequency;
            float baseline = sin(t + baselinePhase) * baselineAmplitude;
            
            // Base y position: start at TOP of screen
            // When quiet (low audio): stays near top
            // When loud (high audio): drops down toward bottom
            float baseY = resolution.y * 0.05;
            
            // Return Y without perpendicular offset (for continuity calculation)
            // baseY is at top, audioDisplacement pushes DOWN
            return baseY + baseline + audioDisplacement;
        }
        
        void main() {
            float pointIndex = curvePoint.x;
            int frameAge = int(curvePoint.y);
            float numPoints = 64.0;
            
            // Map point index to curve position
            float xPos = (pointIndex / (numPoints - 1.0)) * resolution.x;
            
            // Get Y position for this point
            float yPos = computeY(pointIndex, frameAge);
            
            // For better line continuity at high slopes, use interpolated perpendicular
            // Sample adjacent points to compute actual curve direction
            float yPos_prev = pointIndex > 0.0 ? computeY(pointIndex - 1.0, frameAge) : yPos;
            float yPos_next = pointIndex < (numPoints - 1.0) ? computeY(pointIndex + 1.0, frameAge) : yPos;
            
            // Compute curve tangent direction
            float dx = 1.0;  // Normalized x-spacing between points
            float dy = (yPos_next - yPos_prev) / 2.0;
            float length = sqrt(dx * dx + dy * dy);
            
            // Perpendicular (rotated 90 degrees): (-dy, dx)
            float perpX = -dy / length;
            float perpY = dx / length;
            
            // Apply perpendicular offset with proper direction
            // perpOffset is -1 or +1, scaled by thickness
            float actualThickness = perpOffset * 1.5;  // Slightly thicker lines
            yPos += perpY * actualThickness;
            float xPos_offset = xPos + perpX * actualThickness;
            
            // Convert to clip space
            vec2 clipPos = (vec2(xPos_offset, yPos) / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depthValue = depth / 100.0;
            gl_Position = vec4(clipPos, depthValue, 1.0);
            
            // Color: shift hue based on age for visual history
            // Newest frames: current hue
            // Old frames: shift toward blue/cyan
            float normalizedAge = float(frameAge) / float(trailLength);
            float ageHue = mix(hue, hue - 0.3, normalizedAge);  // Shift 0.3 toward blue
            vec3 hsv = vec3(ageHue, 0.7, 1.0);  // Reduced saturation (0.7 instead of 1.0)
            vColor = hsv2rgb(hsv);
            
            // Alpha: extremely slow fade over time
            // Use very aggressive power curve for minimal fade effect
            float agePower = pow(normalizedAge, 0.1);  // 10th root for extremely slow fade
            vAlpha = 1.0 - (agePower * 0.7);  // Never fully transparent (min 0.3 alpha)
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec3 vColor;
        in float vAlpha;
        
        uniform float fadeAlpha;
        
        out vec4 outColor;
        
        void main() {
            // Apply both trail alpha and global fade alpha
            float finalAlpha = vAlpha * fadeAlpha;
            outColor = vec4(vColor, finalAlpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers and textures"""
        # Create circular buffer texture for audio history
        # Size: 32 bands x trail_length frames
        self.audio_history_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.audio_history_texture)
        
        # Initialize with zeros (pre-fill entire circular buffer)
        history_data = np.zeros((self.trail_length, 32), dtype=np.float32)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 32, self.trail_length, 0, GL_RED, GL_FLOAT, history_data)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Generate vertex data for all curves in history using TRIANGLES (not TRIANGLE_STRIP)
        # This prevents unwanted connections between frames
        # Each segment between two curve points becomes two triangles (a quad)
        vertices = []
        indices = []
        
        for frame_age in range(self.trail_length):
            for point_idx in range(self.num_points - 1):  # Segments between points
                frame_age_f = float(frame_age)
                p1 = float(point_idx)
                p2 = float(point_idx + 1)
                offset = self.line_thickness / 2.0
                
                # Quad: 4 corners for this segment
                # p1_left, p1_right, p2_left, p2_right
                base_idx = len(vertices)
                
                # Left side of p1
                vertices.append([p1, frame_age_f, -offset])
                # Right side of p1
                vertices.append([p1, frame_age_f, offset])
                # Right side of p2
                vertices.append([p2, frame_age_f, offset])
                # Left side of p2
                vertices.append([p2, frame_age_f, -offset])
                
                # Triangle 1: p1_left, p1_right, p2_right
                indices.extend([base_idx + 0, base_idx + 1, base_idx + 2])
                # Triangle 2: p1_left, p2_right, p2_left
                indices.extend([base_idx + 0, base_idx + 2, base_idx + 3])
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create VBO
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Curve Point attribute (x=pointIndex, y=frameAge)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        
        # Perp Offset attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(2 * 4))
        
        glBindVertexArray(0)
        
        # Cache counts
        self.vertex_count = len(vertices)
        self.index_count = len(indices)
        
        # Cache uniform locations
        self.uniform_resolution = glGetUniformLocation(self.shader, "resolution")
        self.uniform_depth = glGetUniformLocation(self.shader, "depth")
        self.uniform_baseline_amp = glGetUniformLocation(self.shader, "baselineAmplitude")
        self.uniform_baseline_freq = glGetUniformLocation(self.shader, "baselineFrequency")
        self.uniform_baseline_phase = glGetUniformLocation(self.shader, "baselinePhase")
        self.uniform_sensitivity = glGetUniformLocation(self.shader, "sensitivity")
        self.uniform_amplitude = glGetUniformLocation(self.shader, "amplitude")
        self.uniform_hue = glGetUniformLocation(self.shader, "hue")
        self.uniform_fade = glGetUniformLocation(self.shader, "fadeAlpha")
        self.uniform_audio_tex = glGetUniformLocation(self.shader, "audioHistoryTex")
        self.uniform_trail_length = glGetUniformLocation(self.shader, "trailLength")
        self.uniform_write_index = glGetUniformLocation(self.shader, "currentWriteIndex")
    
    def render(self, state: Dict):
        """Render the audio curve - GPU does all work, CPU updates circular buffer"""
        if not self.enabled or self.index_count == 0:
            return
        
        # Update circular buffer texture with current audio frame
        if hasattr(self, 'audio_bands_current'):
            glBindTexture(GL_TEXTURE_2D, self.audio_history_texture)
            # Write to current frame in circular buffer (as a 1D row)
            glTexSubImage2D(
                GL_TEXTURE_2D, 0,
                0, self.audio_history_write_index,  # x, y offset
                32, 1,  # width, height
                GL_RED, GL_FLOAT, self.audio_bands_current
            )
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Bind audio history texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.audio_history_texture)
        glUniform1i(self.uniform_audio_tex, 0)
        
        # Set uniforms
        glUniform2f(self.uniform_resolution, self.viewport.width, self.viewport.height)
        glUniform1f(self.uniform_depth, self.depth)
        glUniform1f(self.uniform_baseline_amp, self.baseline_amplitude)
        glUniform1f(self.uniform_baseline_freq, self.baseline_frequency)
        glUniform1f(self.uniform_baseline_phase, self.baseline_phase)
        glUniform1f(self.uniform_sensitivity, self.sensitivity)
        glUniform1f(self.uniform_amplitude, self.amplitude)
        glUniform1f(self.uniform_hue, self.current_hue)
        glUniform1f(self.uniform_fade, self.fade_factor)
        glUniform1i(self.uniform_trail_length, self.trail_length)
        glUniform1i(self.uniform_write_index, self.audio_history_write_index)
        
        # Draw using indexed triangles (no unwanted connections)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Advance write index for next frame
        self.audio_history_write_index = (self.audio_history_write_index + 1) % self.trail_length
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        # Animate baseline oscillation (GPU will use this)
        self.baseline_phase += self.baseline_speed * dt
