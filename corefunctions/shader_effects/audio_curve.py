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
        
        # Performance optimization: pre-allocate geometry arrays
        self.line_thickness = 3.0
        self._preallocate_geometry_arrays()
        
        self._initialize_data()
    
    def _preallocate_geometry_arrays(self):
        """Pre-allocate arrays for vectorized geometry building"""
        # Maximum number of segments (including wrap-around)
        max_segments = self.trail_length * self.num_points
        
        # Pre-allocate vertex array (6 vertices per segment, 6 floats per vertex)
        self.vertex_buffer = np.zeros((max_segments, 6, 6), dtype=np.float32)
        
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
        
        # Cache uniform locations for faster access
        self.uniform_resolution = glGetUniformLocation(self.shader, "resolution")
        self.uniform_depth = glGetUniformLocation(self.shader, "depth")
        self.uniform_fade = glGetUniformLocation(self.shader, "fadeAlpha")
    
    def _build_curve_geometry(self):
        """Build thick line geometry from curve history - FULLY VECTORIZED"""
        if len(self.curve_history) == 0:
            return np.array([], dtype=np.float32), 0
        
        num_curves = len(self.curve_history)
        num_segments = self.num_points - 1
        offset = self.line_thickness / 2.0
        
        # Stack all curves into arrays - Shape: (num_curves, num_points)
        all_y_positions = np.array(self.curve_history, dtype=np.float32)
        all_colors = np.array(self.color_history, dtype=np.float32)  # Shape: (num_curves, 3)
        
        # Calculate alpha values for each curve (age-based fade)
        if num_curves > 1:
            alphas = np.linspace(0, 1, num_curves, dtype=np.float32)
        else:
            alphas = np.array([1.0], dtype=np.float32)
        
        # ==== FULLY VECTORIZED: Process ALL curves at once ====
        
        # Broadcast x positions for all curves - Shape: (num_curves, num_segments)
        x1 = np.broadcast_to(self.x_positions[:-1], (num_curves, num_segments))
        x2 = np.broadcast_to(self.x_positions[1:], (num_curves, num_segments))
        
        # Y positions for all curves - Shape: (num_curves, num_segments)
        y1 = all_y_positions[:, :-1]
        y2 = all_y_positions[:, 1:]
        
        # Calculate perpendicular vectors for ALL segments in ALL curves at once
        dx = x2 - x1
        dy = y2 - y1
        lengths = np.sqrt(dx*dx + dy*dy)
        
        # Avoid division by zero
        lengths = np.where(lengths > 0.001, lengths, 1.0)
        
        # Perpendicular unit vectors - Shape: (num_curves, num_segments)
        px = -dy / lengths
        py = dx / lengths
        
        # Calculate quad corners for ALL segments at once
        p1_x = x1 + px * offset
        p1_y = y1 + py * offset
        p2_x = x1 - px * offset
        p2_y = y1 - py * offset
        p3_x = x2 + px * offset
        p3_y = y2 + py * offset
        p4_x = x2 - px * offset
        p4_y = y2 - py * offset
        
        # Broadcast colors and alphas to match segment dimensions
        # Shape: (num_curves, num_segments, 3)
        colors_broadcast = all_colors[:, np.newaxis, :].repeat(num_segments, axis=1)
        # Shape: (num_curves, num_segments, 1)
        alphas_broadcast = alphas[:, np.newaxis, np.newaxis].repeat(num_segments, axis=1)
        
        # Build vertex data for all segments: (num_curves, num_segments, 6 vertices, 6 floats)
        # We need: [x, y, r, g, b, a] for each vertex
        
        # Create array for 6 vertices per segment: (num_curves, num_segments, 6, 6)
        vertices = np.zeros((num_curves, num_segments, 6, 6), dtype=np.float32)
        
        # Triangle 1: p1, p2, p3
        vertices[:, :, 0, 0] = p1_x  # p1 x
        vertices[:, :, 0, 1] = p1_y  # p1 y
        vertices[:, :, 0, 2:5] = colors_broadcast  # p1 rgb
        vertices[:, :, 0, 5] = alphas_broadcast.squeeze(-1)  # p1 alpha
        
        vertices[:, :, 1, 0] = p2_x  # p2 x
        vertices[:, :, 1, 1] = p2_y  # p2 y
        vertices[:, :, 1, 2:5] = colors_broadcast  # p2 rgb
        vertices[:, :, 1, 5] = alphas_broadcast.squeeze(-1)  # p2 alpha
        
        vertices[:, :, 2, 0] = p3_x  # p3 x
        vertices[:, :, 2, 1] = p3_y  # p3 y
        vertices[:, :, 2, 2:5] = colors_broadcast  # p3 rgb
        vertices[:, :, 2, 5] = alphas_broadcast.squeeze(-1)  # p3 alpha
        
        # Triangle 2: p2, p4, p3
        vertices[:, :, 3, 0] = p2_x  # p2 x
        vertices[:, :, 3, 1] = p2_y  # p2 y
        vertices[:, :, 3, 2:5] = colors_broadcast  # p2 rgb
        vertices[:, :, 3, 5] = alphas_broadcast.squeeze(-1)  # p2 alpha
        
        vertices[:, :, 4, 0] = p4_x  # p4 x
        vertices[:, :, 4, 1] = p4_y  # p4 y
        vertices[:, :, 4, 2:5] = colors_broadcast  # p4 rgb
        vertices[:, :, 4, 5] = alphas_broadcast.squeeze(-1)  # p4 alpha
        
        vertices[:, :, 5, 0] = p3_x  # p3 x
        vertices[:, :, 5, 1] = p3_y  # p3 y
        vertices[:, :, 5, 2:5] = colors_broadcast  # p3 rgb
        vertices[:, :, 5, 5] = alphas_broadcast.squeeze(-1)  # p3 alpha
        
        # ==== Add wrap-around segments (vectorized) ====
        
        # Wrap coordinates for all curves at once
        x1_wrap = np.full(num_curves, self.x_positions[-1], dtype=np.float32)
        y1_wrap = all_y_positions[:, -1]
        x2_wrap = np.full(num_curves, self.x_positions[0] + self.viewport.width, dtype=np.float32)
        y2_wrap = all_y_positions[:, 0]
        
        # Calculate perpendiculars for wrap segments
        dx_wrap = x2_wrap - x1_wrap
        dy_wrap = y2_wrap - y1_wrap
        length_wrap = np.sqrt(dx_wrap*dx_wrap + dy_wrap*dy_wrap)
        length_wrap = np.where(length_wrap > 0.001, length_wrap, 1.0)
        
        px_wrap = -dy_wrap / length_wrap
        py_wrap = dx_wrap / length_wrap
        
        # Wrap segment corners
        p1_x_wrap = x1_wrap + px_wrap * offset
        p1_y_wrap = y1_wrap + py_wrap * offset
        p2_x_wrap = x1_wrap - px_wrap * offset
        p2_y_wrap = y1_wrap - py_wrap * offset
        p3_x_wrap = x2_wrap + px_wrap * offset
        p3_y_wrap = y2_wrap + py_wrap * offset
        p4_x_wrap = x2_wrap - px_wrap * offset
        p4_y_wrap = y2_wrap - py_wrap * offset
        
        # Create wrap vertices: (num_curves, 6, 6)
        wrap_vertices = np.zeros((num_curves, 6, 6), dtype=np.float32)
        
        # Triangle 1
        wrap_vertices[:, 0, 0] = p1_x_wrap
        wrap_vertices[:, 0, 1] = p1_y_wrap
        wrap_vertices[:, 0, 2:5] = all_colors
        wrap_vertices[:, 0, 5] = alphas
        
        wrap_vertices[:, 1, 0] = p2_x_wrap
        wrap_vertices[:, 1, 1] = p2_y_wrap
        wrap_vertices[:, 1, 2:5] = all_colors
        wrap_vertices[:, 1, 5] = alphas
        
        wrap_vertices[:, 2, 0] = p3_x_wrap
        wrap_vertices[:, 2, 1] = p3_y_wrap
        wrap_vertices[:, 2, 2:5] = all_colors
        wrap_vertices[:, 2, 5] = alphas
        
        # Triangle 2
        wrap_vertices[:, 3, 0] = p2_x_wrap
        wrap_vertices[:, 3, 1] = p2_y_wrap
        wrap_vertices[:, 3, 2:5] = all_colors
        wrap_vertices[:, 3, 5] = alphas
        
        wrap_vertices[:, 4, 0] = p4_x_wrap
        wrap_vertices[:, 4, 1] = p4_y_wrap
        wrap_vertices[:, 4, 2:5] = all_colors
        wrap_vertices[:, 4, 5] = alphas
        
        wrap_vertices[:, 5, 0] = p3_x_wrap
        wrap_vertices[:, 5, 1] = p3_y_wrap
        wrap_vertices[:, 5, 2:5] = all_colors
        wrap_vertices[:, 5, 5] = alphas
        
        # Reshape and concatenate: (num_curves * num_segments * 6, 6) + (num_curves * 6, 6)
        main_vertices = vertices.reshape(-1, 6)
        wrap_vertices_flat = wrap_vertices.reshape(-1, 6)
        
        # Combine main segments and wrap segments
        all_vertices = np.vstack([main_vertices, wrap_vertices_flat])
        total_vertices = len(all_vertices)
        
        return all_vertices, total_vertices
    
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
        
        # Set uniforms (using cached locations)
        glUniform2f(self.uniform_resolution, self.viewport.width, self.viewport.height)
        glUniform1f(self.uniform_depth, self.depth)
        glUniform1f(self.uniform_fade, self.fade_factor)
        
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
