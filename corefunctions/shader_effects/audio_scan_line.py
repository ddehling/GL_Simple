"""
Audio Scan Line - Sound-reactive scanning line with fading trail
Creates a vertical line that travels from left to right, controlled by audio levels
Leaves a fading residue trail behind it and correctly wraps across boundaries
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

def shader_audio_scan_line(state, outstate, scan_speed=100.0, trail_length=60,
                           intensity_sensitivity=3.0, width_sensitivity=1.5,
                           base_width=4.0, max_width=40.0, color_hue=0.5,
                           fade_duration=3.0, fade_rate=None):
    """
    Audio-reactive scanning line with fading trail
    
    Creates a vertical line that moves from left to right, with width and
    intensity controlled by audio levels. Leaves a fading trail behind.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_audio_scan_line, 
                               scan_speed=150.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        scan_speed: Speed of line movement in pixels/second (default 100.0)
        trail_length: Number of trail segments to keep (default 60)
        intensity_sensitivity: How much audio affects brightness (default 2.0)
        width_sensitivity: How much audio affects line width (default 0.5)
        base_width: Minimum line width in pixels (default 8.0)
        max_width: Maximum line width in pixels (default 30.0)
        color_hue: Color hue 0-1 (default 0.5 = cyan)
        fade_duration: Duration of fade in/out in seconds (default 3.0)
        fade_rate: DEPRECATED - Use trail_length instead
    """
    # Handle deprecated fade_rate parameter
    if fade_rate is not None:
        print("WARNING: fade_rate is deprecated, use trail_length instead")
    
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
        print(f"Initializing audio scan line effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                AudioScanLineEffect,
                scan_speed=scan_speed,
                trail_length=trail_length,
                intensity_sensitivity=intensity_sensitivity,
                width_sensitivity=width_sensitivity,
                base_width=base_width,
                max_width=max_width,
                color_hue=color_hue
            )
            state['effect'] = effect
            state['smoothed_intensity'] = 0.0
            state['smoothed_width'] = base_width
            print(f"✓ Initialized shader audio_scan_line for frame {frame_id}")
            print(f"  - Viewport size: {viewport.width}x{viewport.height}")
            print(f"  - Scan speed: {scan_speed} px/s")
            print(f"  - Base width: {base_width}, Max width: {max_width}")
        except Exception as e:
            print(f"✗ Failed to initialize audio scan line: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters
    if 'effect' in state:
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
        
        # Audio reactivity - use norm_long_relu DIRECTLY for maximum contrast
        if audio_data is not None:
            # Use norm_long_relu for above-baseline detection
            # This gives 0 when at baseline, >0 when above baseline
            bands = audio_data['norm_long_relu'][0]
            
            # Clip to reasonable range
            bands = bands.clip(0, 3.0)  # Allow values up to 3x baseline
            
            # Use bass for intensity (most impactful)
            bass_energy = np.mean(bands[0:8])  # Pure bass
            mid_energy = np.mean(bands[8:20])  # Mids for width
            high_energy = np.mean(bands[20:32])  # Highs
            
            # Determine dominant frequency for color
            # Bass = warm (red/orange, hue 0.0-0.1)
            # Mids = cool (cyan/green, hue 0.4-0.6)
            # Highs = blue/purple (hue 0.6-0.8)
            energies = np.array([bass_energy, mid_energy, high_energy])
            dominant_idx = np.argmax(energies)
            dominant_energy = energies[dominant_idx]
            
            # Map to hue based on which frequency is loudest
            if dominant_idx == 0:  # Bass dominant
                base_hue = 0.05  # Red/orange
            elif dominant_idx == 1:  # Mids dominant
                base_hue = 0.5   # Cyan
            else:  # Highs dominant
                base_hue = 0.7   # Blue/purple
            
            # Vary hue slightly based on energy level
            hue_variation = (dominant_energy % 1.0) * 0.1
            calculated_hue = (base_hue + hue_variation) % 1.0
            
            # NO SMOOTHING - direct response for maximum contrast
            # Apply sensitivity multipliers directly
            state['effect'].audio_intensity = bass_energy * intensity_sensitivity
            state['effect'].audio_width = mid_energy * width_sensitivity
            state['effect'].current_hue = calculated_hue  # Dynamic color
        else:
            state['effect'].audio_intensity = 0.0
            state['effect'].audio_width = 0.0
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up audio scan line effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader audio_scan_line for frame {frame_id}")

# ============================================================================
# Audio Scan Line Effect - Instanced Line Rendering with Trail
# ============================================================================

class AudioScanLineEffect(ShaderEffect):
    """Scanning line with fading trail using instanced rendering"""
    
    def __init__(self, viewport, scan_speed: float = 100.0, trail_length: int = 60,
                 intensity_sensitivity: float = 2.0, width_sensitivity: float = 0.5,
                 base_width: float = 8.0, max_width: float = 30.0, color_hue: float = 0.5):
        super().__init__(viewport)
        self.scan_speed = scan_speed
        self.trail_length = trail_length
        self.intensity_sensitivity = intensity_sensitivity
        self.width_sensitivity = width_sensitivity
        self.base_width = base_width
        self.max_width = max_width
        self.color_hue = color_hue
        
        self.fade_factor = 0.0
        self.audio_intensity = 0.0
        self.audio_width = 0.0
        self.current_hue = color_hue  # Dynamic hue based on dominant frequency
        self.line_position = 0.0
        
        # Trail history: [position, width, intensity, age, hue]
        self.trail_history = []
        
        self.instance_VBO = None
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1 in Y, -0.5 to 0.5 in X)
        layout(location = 1) in vec4 instanceData;  // x=linePos, y=width, z=intensity, w=age
        layout(location = 2) in float instanceHue;  // Hue for this segment
        
        uniform vec2 resolution;
        out float vIntensity;
        out float vAge;
        out float vHue;
        
        void main() {
            float linePos = instanceData.x;
            float width = instanceData.y;
            float intensity = instanceData.z;
            float age = instanceData.w;
            
            // Position quad: X at linePos ± width/2, Y full screen height
            // position.x ranges from -0.5 to 0.5
            // position.y ranges from -1.0 to 1.0 (covers full height)
            vec2 pos;
            pos.x = linePos + position.x * width;
            pos.y = (position.y * 0.5 + 0.5) * resolution.y;  // Map -1..1 to 0..height
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            gl_Position = vec4(clipPos, 0.5, 1.0);
            vIntensity = intensity;
            vAge = age;
            vHue = instanceHue;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in float vIntensity;
        in float vAge;
        in float vHue;
        out vec4 outColor;
        
        uniform float fadeAlpha;
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            // Fade based on age (0 = newest, 1 = oldest)
            float ageFade = 1.0 - vAge;
            
            // Combine intensity and age fade
            float brightness = vIntensity * ageFade;
            
            // Convert hue to RGB (use per-segment hue)
            float saturation = mix(0.6, 1.0, brightness);  // More saturated when brighter
            vec3 hsv = vec3(vHue, saturation, 1.0);
            vec3 color = hsv2rgb(hsv) * brightness;
            float alpha = brightness * fadeAlpha;
            
            outColor = vec4(color, alpha);
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
        """Setup quad for line rendering"""
        # Simple quad vertices (will be stretched to line width)
        vertices = np.array([
            -0.5, -1.0,  # Bottom left
             0.5, -1.0,  # Bottom right
             0.5,  1.0,  # Top right
            -0.5,  1.0   # Top left
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
        
        # Instance buffer (will be updated each frame)
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.trail_length * 20, None, GL_DYNAMIC_DRAW)  # 5 floats * 4 bytes
        self.VBOs.append(self.instance_VBO)
        
        # Instance attributes (vec4: position, width, intensity, age)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)  # Advance once per instance
        
        # Instance hue attribute (float: hue)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)  # Advance once per instance
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update line position and trail"""
        if not self.enabled:
            return
        
        # Calculate current line parameters with DRAMATIC audio response
        # Width: base_width when silent, up to max_width when loud
        current_width = self.base_width + np.clip(self.audio_width, 0.0, 3.0) * (self.max_width - self.base_width)
        
        # Intensity: 0.2 when silent (barely visible), 1.0 when loud (full brightness)
        current_intensity = 0.2 + np.clip(self.audio_intensity, 0.0, 3.0) * 0.8
        
        # Add current position to trail history with current hue
        self.trail_history.append([self.line_position, current_width, current_intensity, 0.0, self.current_hue])
        
        # Keep only the most recent trail_length entries
        if len(self.trail_history) > self.trail_length:
            self.trail_history.pop(0)
        
        # Update ages (0 = newest, 1 = oldest)
        trail_len = len(self.trail_history)
        for i, entry in enumerate(self.trail_history):
            entry[3] = 1.0 - (i / max(trail_len - 1, 1))
        
        # Move line from left to right
        self.line_position += self.scan_speed * dt
        
        # Wrap around when reaching right edge
        if self.line_position > self.viewport.width:
            self.line_position = 0.0
            # Don't clear trail - let it persist across wrap
    
    def render(self, state: Dict):
        """Render scanning line with trail"""
        if not self.enabled or not self.shader or len(self.trail_history) == 0:
            return
        
        # Post-processing: Always pass depth test without writing
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        if res_loc != -1:
            glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        fade_alpha_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if fade_alpha_loc != -1:
            glUniform1f(fade_alpha_loc, self.fade_factor)
        
        # Handle horizontal wrapping: duplicate trail segments near edges
        wrap_margin = self.max_width  # Use max width as margin
        trail_data = []
        
        for entry in self.trail_history:
            pos, width, intensity, age, hue = entry
            # Add original segment
            trail_data.append([pos, width, intensity, age, hue])
            
            # If near right edge, duplicate on left
            if pos > (self.viewport.width - wrap_margin):
                wrapped_pos = pos - self.viewport.width
                trail_data.append([wrapped_pos, width, intensity, age, hue])
            
            # If near left edge, duplicate on right
            elif pos < wrap_margin:
                wrapped_pos = pos + self.viewport.width
                trail_data.append([wrapped_pos, width, intensity, age, hue])
        
        # Update instance buffer with trail data (including wrapped segments)
        if len(trail_data) > 0:
            instance_data = np.array(trail_data, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
            
            # Expand buffer if needed
            if len(trail_data) > self.trail_length:
                glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
            else:
                glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
            
            # Draw instanced quads (one per trail segment including wrapped)
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(trail_data))
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
