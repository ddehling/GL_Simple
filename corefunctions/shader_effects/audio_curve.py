"""
Audio-reactive curve shader - draws a flowing waveform based on frequency data
The curve spans the entire screen and leaves a fading trail
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_audio_curve(state, outstate, sensitivity=2.0, amplitude=100.0, trail_length=15):
    """
    Audio-reactive curve that flows across the screen
    
    Usage:
        scheduler.schedule_event(0, 60, shader_audio_curve, 
                               sensitivity=2.5, amplitude=120, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        sensitivity: Multiplier for audio response (default 2.0)
        amplitude: Maximum vertical displacement in pixels (default 100.0)
        trail_length: Number of frames to keep in trail (default 30)
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
    """Audio-reactive curve with fading trail effect"""
    
    def __init__(self, viewport, sensitivity: float = 2.0, amplitude: float = 100.0, trail_length: int = 30):
        super().__init__(viewport)
        self.sensitivity = sensitivity
        self.amplitude = amplitude
        self.trail_length = trail_length
        self.fade_factor = 0.0
        
        # Curve parameters
        self.num_points = 64  # Number of points along the curve (matches 2x frequency bands)
        self.curve_history = []  # Store previous curve positions for trail
        self.color_history = []  # Store colors for each curve
        self.current_audio = np.zeros(32, dtype=np.float32)  # Current audio data
        
        # Visual parameters
        self.current_hue = 0.0  # Current hue value (0-1)
        self.hue_speed = 0.01  # How fast to rotate through colors
        self.depth = 50.0  # Mid-depth for the curve
        
        # Baseline oscillation parameters
        self.baseline_amplitude = 30.0  # Amplitude of baseline wave
        self.baseline_frequency = 2.0  # Frequency of baseline oscillation
        self.baseline_phase = 0.0  # Phase offset for animation
        self.baseline_speed = 2  # Speed of baseline animation
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize curve geometry"""
        # Create base curve points (x positions evenly distributed)
        x_positions = np.linspace(0, self.viewport.width, self.num_points, dtype=np.float32)
        
        # Store x positions (these don't change)
        self.x_positions = x_positions
        
        # Calculate baseline curve (oscillating wave)
        self.baseline_y = self._calculate_baseline()
        
        # Initialize y positions at baseline
        self.y_positions = self.baseline_y.copy()
        
        # Quad vertices for thick line segments (2 triangles per segment)
        # We'll build this dynamically in update_buffers
        self.line_thickness = 3.0
    
    def _calculate_baseline(self):
        """Calculate oscillating baseline curve"""
        # Base position - closer to top (0.25 = 25% from top)
        base_y = self.viewport.height * 0.15
        
        # Add sine wave oscillation
        t = (self.x_positions / self.viewport.width) * 2 * np.pi * self.baseline_frequency
        oscillation = np.sin(t + self.baseline_phase) * self.baseline_amplitude
        
        return base_y + oscillation
    
    def _hsv_to_rgb(self, h, s=1.0, v=1.0):
        """Convert HSV to RGB (h in 0-1 range)"""
        h = h % 1.0
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        
        i = i % 6
        if i == 0:
            return np.array([v, t, p], dtype=np.float32)
        elif i == 1:
            return np.array([q, v, p], dtype=np.float32)
        elif i == 2:
            return np.array([p, v, t], dtype=np.float32)
        elif i == 3:
            return np.array([p, q, v], dtype=np.float32)
        elif i == 4:
            return np.array([t, p, v], dtype=np.float32)
        else:
            return np.array([v, p, q], dtype=np.float32)
    
    def update_from_audio(self, bands: np.ndarray):
        """Update curve shape from audio frequency bands"""
        # bands is shape (32,) - we need to interpolate to match our num_points
        
        # Interpolate audio bands to match curve points
        band_indices = np.linspace(0, len(bands) - 1, self.num_points)
        interpolated = np.interp(band_indices, np.arange(len(bands)), bands)
        
        # Apply sensitivity and amplitude
        self.current_audio = interpolated * self.sensitivity
        
        # Update baseline oscillation
        self.baseline_y = self._calculate_baseline()
        
        # Calculate new y positions (baseline + audio displacement)
        displacement = self.current_audio * self.amplitude
        self.y_positions = self.baseline_y + displacement
        
        # Update color for this frame
        current_color = self._hsv_to_rgb(self.current_hue)
        
        # Add current curve and color to history
        self.curve_history.append(self.y_positions.copy())
        self.color_history.append(current_color)
        
        # Advance hue
        self.current_hue = (self.current_hue + self.hue_speed) % 1.0
        
        # Limit history length
        if len(self.curve_history) > self.trail_length:
            self.curve_history.pop(0)
            self.color_history.pop(0)
    
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
        
        layout(location = 0) in vec2 position;  // Vertex position
        layout(location = 1) in vec3 color;      // RGB color
        layout(location = 2) in float alpha;     // Alpha for trail fade
        
        uniform vec2 resolution;
        uniform float depth;
        
        out vec3 vColor;
        out float vAlpha;
        
        void main() {
            vec2 clipPos = (position / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depthValue = depth / 100.0;
            
            gl_Position = vec4(clipPos, depthValue, 1.0);
            vColor = color;
            vAlpha = alpha;
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
        """Initialize OpenGL buffers"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create VBO for dynamic curve data
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        
        # Allocate space (we'll update it each frame)
        # Each segment needs 6 vertices (2 triangles) with 6 floats (x, y, r, g, b, alpha)
        # Include space for wrap-around segments: trail_length * num_points segments total
        max_vertices = self.trail_length * self.num_points * 6
        glBufferData(GL_ARRAY_BUFFER, max_vertices * 6 * 4, None, GL_DYNAMIC_DRAW)
        
        # Position attribute (x, y)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        
        # Color attribute (r, g, b)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))
        
        # Alpha attribute
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(5 * 4))
        
        glBindVertexArray(0)
    
    def _build_curve_geometry(self):
        """Build thick line geometry from curve history"""
        if len(self.curve_history) == 0:
            return np.array([], dtype=np.float32), 0
        
        vertices = []
        
        # Render each curve in history (oldest to newest for proper blending)
        for history_idx, y_positions in enumerate(self.curve_history):
            # Calculate alpha based on age (older = more transparent)
            age_factor = history_idx / max(len(self.curve_history) - 1, 1)
            alpha = age_factor  # 0 (oldest) to 1 (newest)
            
            # Get color for this curve
            color = self.color_history[history_idx]
            
            # Build thick line segments
            for i in range(len(self.x_positions) - 1):
                x1, y1 = self.x_positions[i], y_positions[i]
                x2, y2 = self.x_positions[i + 1], y_positions[i + 1]
                
                # Calculate perpendicular offset for thickness
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # Perpendicular unit vector
                    px = -dy / length
                    py = dx / length
                    
                    # Offset by half thickness
                    offset = self.line_thickness / 2
                    
                    # Four corners of the quad (x, y, r, g, b, alpha)
                    p1 = [x1 + px * offset, y1 + py * offset, color[0], color[1], color[2], alpha]
                    p2 = [x1 - px * offset, y1 - py * offset, color[0], color[1], color[2], alpha]
                    p3 = [x2 + px * offset, y2 + py * offset, color[0], color[1], color[2], alpha]
                    p4 = [x2 - px * offset, y2 - py * offset, color[0], color[1], color[2], alpha]
                    
                    # Two triangles for the quad
                    vertices.extend([p1, p2, p3])
                    vertices.extend([p2, p4, p3])
        
        # Handle wrapping: draw wrap-around segment
        if len(self.curve_history) > 0:
            for history_idx, y_positions in enumerate(self.curve_history):
                age_factor = history_idx / max(len(self.curve_history) - 1, 1)
                alpha = age_factor
                
                # Get color for this curve
                color = self.color_history[history_idx]
                
                # Connect last point to first point (wrapping)
                x1, y1 = self.x_positions[-1], y_positions[-1]
                x2, y2 = self.x_positions[0] + self.viewport.width, y_positions[0]  # Wrap
                
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    px = -dy / length
                    py = dx / length
                    offset = self.line_thickness / 2
                    
                    p1 = [x1 + px * offset, y1 + py * offset, color[0], color[1], color[2], alpha]
                    p2 = [x1 - px * offset, y1 - py * offset, color[0], color[1], color[2], alpha]
                    p3 = [x2 + px * offset, y2 + py * offset, color[0], color[1], color[2], alpha]
                    p4 = [x2 - px * offset, y2 - py * offset, color[0], color[1], color[2], alpha]
                    
                    vertices.extend([p1, p2, p3])
                    vertices.extend([p2, p4, p3])
        
        vertex_array = np.array(vertices, dtype=np.float32)
        vertex_count = len(vertices)
        
        return vertex_array, vertex_count
    
    def render(self, state: Dict):
        """Render the audio curve with trail"""
        if not self.enabled or len(self.curve_history) == 0:
            return
        
        # Build geometry from curve history
        vertex_data, vertex_count = self._build_curve_geometry()
        
        if vertex_count == 0:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Update VBO with new geometry
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        # Draw the curve
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        # Animate baseline oscillation
        self.baseline_phase += self.baseline_speed * dt
