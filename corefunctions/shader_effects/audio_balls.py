"""
Audio-reactive balls with lightning - 16 balls respond to unique frequency bands
Each ball moves left to right with speed, size, color, depth, and transparency
driven by sound. Lightning occasionally arcs between nearby balls based on energy.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import ctypes
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_audio_balls(state, outstate, sensitivity=2.0, base_size=8.0, 
                       lightning_threshold=0.5, lightning_probability=0.3,
                       num_balls=16):
    """
    Audio-reactive balls with lightning arcs between them
    
    Usage:
        scheduler.schedule_event(0, 60, shader_audio_balls, 
                               sensitivity=2.5, base_size=10.0, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        sensitivity: Multiplier for audio response (default 2.0)
        base_size: Base size of each ball in pixels (default 8.0)
        lightning_threshold: Minimum normalized energy for lightning (0-1, default 0.5)
        lightning_probability: Chance of lightning between nearby high-energy balls (default 0.3)
        num_balls: Number of balls to create (default 16)
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
        print(f"Initializing audio_balls for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                AudioBallsEffect,
                sensitivity=sensitivity,
                base_size=base_size,
                lightning_threshold=lightning_threshold,
                lightning_probability=lightning_probability,
                num_balls=num_balls
            )
            state['effect'] = effect
            print(f"✓ Initialized shader audio_balls for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize audio_balls: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from audio data every frame
    if 'effect' in state and audio_data is not None:
        # Get short-term normalized data for beat response
        bands = audio_data['norm_short'][0]  # Shape: (32,)
        
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
            print(f"Cleaning up audio_balls for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader audio_balls for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class AudioBallsEffect(ShaderEffect):
    """Audio-reactive balls with lightning between high-energy pairs"""
    
    def __init__(self, viewport, sensitivity: float = 2.0, base_size: float = 8.0,
                 lightning_threshold: float = 0.5, lightning_probability: float = 0.3,
                 num_balls: int = 16):
        super().__init__(viewport)
        self.sensitivity = sensitivity
        self.base_size = base_size
        self.lightning_threshold = lightning_threshold
        self.lightning_probability = lightning_probability
        self.num_balls = min(num_balls, 32)  # Max 32 (one per frequency band)
        self.fade_factor = 0.0
        self.wrap_margin = self.base_size * (1.0 + self.sensitivity * 0.3) + 10  # Wrapping margin
        
        # Audio data mapping - each ball gets every nth frequency band
        # 32 bands total, so step = 32 / num_balls
        self.band_step = max(1, 32 // self.num_balls)
        
        # Ball state arrays
        self.positions = np.zeros((self.num_balls, 3), dtype=np.float32)  # x, y, z
        self.sizes = np.full(self.num_balls, self.base_size, dtype=np.float32)
        self.colors = np.ones((self.num_balls, 3), dtype=np.float32)  # RGB
        self.alphas = np.ones(self.num_balls, dtype=np.float32)
        self.speeds = np.zeros(self.num_balls, dtype=np.float32)  # Horizontal movement
        self.energies = np.zeros(self.num_balls, dtype=np.float32)  # Current audio energy
        self.smoothed_energies = np.zeros(self.num_balls, dtype=np.float32)  # Smoothed energy for slower reaction
        self.energy_smoothing = 0.15  # Smoothing factor (lower = slower reaction)
        
        # Surface animation state
        self.surface_time = 0.0
        
        # Lightning state
        self.active_lightning = []  # List of (ball_i, ball_j, intensity, age) tuples
        self.lightning_line_thickness = 2.0
        
        self._initialize_balls()
        # NOTE: Do NOT call setup_buffers() here!
    
    def _initialize_balls(self):
        """Initialize ball positions and properties (vectorized)"""
        # Vectorized position calculation
        indices = np.arange(self.num_balls, dtype=np.float32)
        
        if self.num_balls > 1:
            x_positions = (indices / (self.num_balls - 1)) * self.viewport.width
            y_positions = (indices / (self.num_balls - 1)) * (self.viewport.height - 60) + 30
        else:
            x_positions = np.full(self.num_balls, self.viewport.width / 2, dtype=np.float32)
            y_positions = np.full(self.num_balls, self.viewport.height / 2, dtype=np.float32)
        
        self.positions[:, 0] = x_positions
        self.positions[:, 1] = y_positions
        self.positions[:, 2] = 50  # Mid-depth
        
        # Vectorized speed generation
        self.speeds[:] = np.random.uniform(10, 30, size=self.num_balls)
        
        # Vectorized color generation from hue
        band_indices = (indices * self.band_step).astype(np.int32)
        band_indices = np.minimum(band_indices, 31)
        hues = (band_indices / 32.0) % 1.0
        self.colors[:] = self._hsv_to_rgb_vectorized(hues, s=0.8, v=1.0)
    
    def _hsv_to_rgb(self, h, s=1.0, v=1.0):
        """Convert HSV to RGB (h in 0-1 range, scalar version)"""
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
    
    def _hsv_to_rgb_vectorized(self, h, s=1.0, v=1.0):
        """Convert HSV to RGB (vectorized for arrays)
        
        Args:
            h: Array of hues in 0-1 range
            s: Saturation (scalar)
            v: Value (scalar)
        
        Returns:
            Array of shape (n, 3) with RGB values
        """
        h = np.asarray(h, dtype=np.float32) % 1.0
        i = np.asarray(h * 6.0, dtype=np.int32)
        f = h * 6.0 - i
        
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        
        i = i % 6
        n = len(h)
        
        # Vectorized RGB selection based on hue sector
        rgb = np.zeros((n, 3), dtype=np.float32)
        
        mask0 = i == 0
        mask1 = i == 1
        mask2 = i == 2
        mask3 = i == 3
        mask4 = i == 4
        mask5 = i == 5
        
        # Element-wise assignment with proper broadcasting of scalars
        rgb[mask0, 0] = v
        rgb[mask0, 1] = t[mask0]
        rgb[mask0, 2] = p
        
        rgb[mask1, 0] = q[mask1]
        rgb[mask1, 1] = v
        rgb[mask1, 2] = p
        
        rgb[mask2, 0] = p
        rgb[mask2, 1] = v
        rgb[mask2, 2] = t[mask2]
        
        rgb[mask3, 0] = p
        rgb[mask3, 1] = q[mask3]
        rgb[mask3, 2] = v
        
        rgb[mask4, 0] = t[mask4]
        rgb[mask4, 1] = p
        rgb[mask4, 2] = v
        
        rgb[mask5, 0] = v
        rgb[mask5, 1] = p
        rgb[mask5, 2] = q[mask5]
        
        return rgb
    
    def update_from_audio(self, bands: np.ndarray):
        """Update balls from audio frequency bands (vectorized)
        
        Args:
            bands: Array of shape (32,) with normalized frequency data
        """
        # Vectorized band index selection
        band_indices = np.minimum(
            (np.arange(self.num_balls) * self.band_step).astype(np.int32),
            31
        )
        
        # Extract energies from bands for all balls at once
        self.energies[:] = bands[band_indices]
        
        # Vectorized energy smoothing
        self.smoothed_energies[:] = (
            self.smoothed_energies * (1.0 - self.energy_smoothing) +
            self.energies * self.energy_smoothing
        )
        
        # Vectorized visual updates using smoothed energy
        smoothed = self.smoothed_energies
        
        # Size increases with energy
        self.sizes[:] = self.base_size * (1.0 + smoothed * self.sensitivity * 0.3)
        
        # Alpha increases with energy
        self.alphas[:] = 0.3 + 0.5 * np.clip(smoothed, 0, 1)
        
        # Depth varies with energy
        self.positions[:, 2] = 50 - smoothed * 20  # Range 30-50
    
    def _update_lightning(self, dt: float):
        """Update lightning state and generate new arcs (optimized)"""
        # Decay existing lightning and filter in one pass
        updated_lightning = []
        for ball_i, ball_j, intensity, age in self.active_lightning:
            new_intensity = intensity * 0.8
            new_age = age + dt
            # Keep lightning if intensity > 0.05 and age < 0.5
            if new_intensity > 0.05 and new_age < 0.5:
                updated_lightning.append((ball_i, ball_j, new_intensity, new_age))
        self.active_lightning = updated_lightning
        
        # Check for new lightning between nearby high-energy balls (vectorized distance)
        high_energy_mask = self.energies >= self.lightning_threshold
        high_energy_indices = np.where(high_energy_mask)[0]
        
        for i in high_energy_indices:
            # Find nearby balls with high energy using vectorized distance calc
            j_indices = np.arange(i + 1, self.num_balls)
            high_j = j_indices[self.energies[j_indices] >= self.lightning_threshold]
            
            if len(high_j) == 0:
                continue
            
            # Vectorized distance calculation
            dx = self.positions[high_j, 0] - self.positions[i, 0]
            dy = self.positions[high_j, 1] - self.positions[i, 1]
            distances = np.sqrt(dx*dx + dy*dy)
            
            # Find nearby balls (within 200 pixels or adjacent)
            adjacent_mask = np.abs(high_j - i) == 1
            nearby_mask = (distances < 200) | adjacent_mask
            nearby_j = high_j[nearby_mask]
            
            # Randomly create lightning to a nearby ball
            if len(nearby_j) > 0 and np.random.random() < self.lightning_probability:
                j = nearby_j[np.random.randint(len(nearby_j))]
                intensity = np.clip(self.energies[i] * self.energies[j], 0, 1)
                
                # Check if this lightning already exists
                exists = any((a == i and b == j) or (a == j and b == i) 
                           for a, b, _, _ in self.active_lightning)
                
                if not exists:
                    self.active_lightning.append((i, j, intensity, 0))
    
    def _add_ball_segment(self, vertices, offsets, colors, alphas, sphere_radii, ball_id, energy,
                          pos, size, color, alpha, angle1, angle2, x_offset=0.0):
        """Helper: Add a single triangle segment for a ball
        
        Args:
            x_offset: Horizontal offset for seamless wrapping
        """
        adjusted_pos = pos.copy()
        adjusted_pos[0] += x_offset
        
        # Center vertex
        vertices.append([pos[0] + x_offset, pos[1]])
        offsets.append(adjusted_pos)
        colors.append(color)
        alphas.append(alpha)
        sphere_radii.append(0.0)
        
        # Perimeter vertex 1
        x1 = pos[0] + x_offset + size * np.cos(angle1)
        y1 = pos[1] + size * np.sin(angle1)
        vertices.append([x1, y1])
        offsets.append(adjusted_pos)
        colors.append(color)
        alphas.append(alpha)
        sphere_radii.append(1.0 + (ball_id / 100.0))
        
        # Perimeter vertex 2
        x2 = pos[0] + x_offset + size * np.cos(angle2)
        y2 = pos[1] + size * np.sin(angle2)
        vertices.append([x2, y2])
        offsets.append(adjusted_pos)
        colors.append(color)
        alphas.append(alpha)
        sphere_radii.append(1.0 + (ball_id / 100.0))
    
    def _build_ball_geometry(self):
        """Build geometry for all balls as spheres with seamless wrapping"""
        vertices = []
        offsets = []
        colors = []
        alphas = []
        sphere_radii = []
        
        # Create a circular mesh for each ball
        segments = 24
        for i in range(self.num_balls):
            # Get ball properties
            pos = self.positions[i]
            size = self.sizes[i]
            color = self.colors[i]
            alpha = self.alphas[i] * self.fade_factor
            energy = self.smoothed_energies[i]
            
            # Determine which positions to render (for seamless wrapping)
            render_x_offsets = [0.0]  # Always render original
            
            # Check if ball is near left edge
            if pos[0] < self.wrap_margin:
                # Create duplicate on right side
                render_x_offsets.append(self.viewport.width)
            
            # Check if ball is near right edge
            if pos[0] > (self.viewport.width - self.wrap_margin):
                # Create duplicate on left side
                render_x_offsets.append(-self.viewport.width)
            
            # Generate sphere vertices for each position
            for x_offset in render_x_offsets:
                for seg in range(segments):
                    angle1 = (seg / segments) * 2 * np.pi
                    angle2 = ((seg + 1) / segments) * 2 * np.pi
                    
                    self._add_ball_segment(vertices, offsets, colors, alphas, sphere_radii,
                                          i, energy,
                                          pos, size, color, alpha, angle1, angle2, x_offset)
        
        if not vertices:
            return None, 0
        
        vertex_data = np.column_stack([
            np.array(vertices, dtype=np.float32),
            np.array(offsets, dtype=np.float32),
            np.array(colors, dtype=np.float32),
            np.array(alphas, dtype=np.float32),
            np.array(sphere_radii, dtype=np.float32)
        ]).astype(np.float32)
        
        return vertex_data, len(vertices)
    
    def _lightning_color_from_indices(self, i, j, intensity):
        """Generate varied lightning color based on ball indices and intensity"""
        # Use ball indices to determine base hue
        hue = ((i + j) / (self.num_balls * 2)) % 1.0
        
        # Convert to RGB
        rgb = self._hsv_to_rgb(hue, s=0.6 + 0.4 * intensity, v=1.0)
        
        # Blend towards white based on intensity
        rgb = rgb * (1.0 - intensity * 0.3) + np.array([1.0, 1.0, 1.0]) * (intensity * 0.3)
        
        return rgb.astype(np.float32)
    
    def _build_lightning_geometry(self):
        """Build geometry for all active lightning bolts with seamless wrapping"""
        if not self.active_lightning:
            return None, 0
        
        vertices = []
        offsets1 = []
        colors = []
        alphas = []
        
        for ball_i, ball_j, intensity, age in self.active_lightning:
            p1 = self.positions[ball_i]
            p2 = self.positions[ball_j]
            
            # Determine wrapped positions for lightning (use shortest path)
            p1_wrapped = p1.copy()
            p2_wrapped = p2.copy()
            
            # Handle horizontal wrapping: use shortest path
            dx = p2[0] - p1[0]
            if abs(dx) > self.viewport.width / 2:
                if dx > 0:
                    # p2 is far right; wrap p1 to right
                    p1_wrapped[0] += self.viewport.width
                else:
                    # p2 is far left; wrap p1 to left
                    p1_wrapped[0] -= self.viewport.width
            
            # Add jagged path between balls
            mid_x = (p1_wrapped[0] + p2_wrapped[0]) / 2 + np.random.uniform(-10, 10)
            mid_y = (p1_wrapped[1] + p2_wrapped[1]) / 2 + np.random.uniform(-10, 10)
            mid_z = (p1_wrapped[2] + p2_wrapped[2]) / 2
            
            # Generate varied lightning color based on ball indices
            lightning_color = self._lightning_color_from_indices(ball_i, ball_j, intensity)
            
            # Segment 1: p1 to mid
            vertices.append(np.array([p1_wrapped[0], p1_wrapped[1]], dtype=np.float32))
            vertices.append(np.array([mid_x, mid_y], dtype=np.float32))
            offsets1.append(p1_wrapped)
            offsets1.append(np.array([mid_x, mid_y, mid_z], dtype=np.float32))
            colors.append(lightning_color)
            colors.append(lightning_color)
            alphas.append(intensity * 0.8)
            alphas.append(intensity * 0.8)
            
            # Segment 2: mid to p2
            vertices.append(np.array([mid_x, mid_y], dtype=np.float32))
            vertices.append(np.array([p2_wrapped[0], p2_wrapped[1]], dtype=np.float32))
            offsets1.append(np.array([mid_x, mid_y, mid_z], dtype=np.float32))
            offsets1.append(p2_wrapped)
            colors.append(lightning_color)
            colors.append(lightning_color)
            alphas.append(intensity * 0.8)
            alphas.append(intensity * 0.8)
        
        vertex_data = np.column_stack([
            np.array(vertices, dtype=np.float32),
            np.array(offsets1, dtype=np.float32),
            np.array(colors, dtype=np.float32),
            np.array(alphas, dtype=np.float32)
        ]).astype(np.float32)
        
        return vertex_data, len(vertices)
    
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
            print(f"AudioBallsEffect shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 offset;  // x, y, z
        layout(location = 2) in vec3 color;
        layout(location = 3) in float alpha;
        layout(location = 4) in float sphereRadius;  // 0.0 = center, 1.0 = edge
        
        uniform vec2 resolution;
        uniform float time;
        
        out vec3 vColor;
        out float vAlpha;
        out vec2 vLocalPos;
        out float vBallId;
        
        void main() {
            vec2 pos = position;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depth = offset.z / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            vColor = color;
            vAlpha = alpha;
            vLocalPos = position - offset.xy;
            vBallId = sphereRadius - 1.0;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec3 vColor;
        in float vAlpha;
        in vec2 vLocalPos;
        in float vBallId;
        
        uniform float time;
        
        out vec4 outColor;
        
        // Simple noise function
        float hash(float n) {
            return fract(sin(n) * 43758.5453123);
        }
        
        float noise(float x) {
            float i = floor(x);
            float f = fract(x);
            float u = f * f * (3.0 - 2.0 * f);
            return mix(hash(i), hash(i + 1.0), u);
        }
        
        void main() {
            // Calculate distance from fragment to sphere center
            float distFromCenter = length(vLocalPos);
            
            // Base sphere shading: darker at edges
            float sphereShadow = 1.0 - (distFromCenter * distFromCenter * 0.5);
            sphereShadow = max(0.3, sphereShadow);
            
            // Animated surface pattern - ripples and waves
            float angle = atan(vLocalPos.y, vLocalPos.x);
            float radius = distFromCenter;
            
            // Create ripple effect that moves around
            float ripple = sin(radius * 8.0 - time * 3.0) * 0.3;
            ripple += sin(angle * 5.0 + time * 2.5) * 0.2;
            ripple += noise(time * 0.7 + vBallId) * 0.15;
            
            // Apply ripple to shading
            sphereShadow += ripple * (1.0 - distFromCenter);
            sphereShadow = clamp(sphereShadow, 0.3, 1.0);
            
            // Apply shading
            vec3 shadedColor = vColor * sphereShadow;
            
            // Animated specular highlight - moves around surface
            float highlightAngle = angle - time * 2.0;
            float specular = exp(-(distFromCenter - 0.6) * (distFromCenter - 0.6) * 15.0);
            specular *= smoothstep(0.3, 0.1, abs(sin(highlightAngle * 3.0)));
            specular += exp(-distFromCenter * distFromCenter * 8.0) * 0.5;
            
            shadedColor += vec3(0.4, 0.4, 0.4) * specular;
            
            outColor = vec4(shadedColor, vAlpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Ball VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 500000, None, GL_DYNAMIC_DRAW)
        
        # Position (x, y)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        
        # Offset (x, y, z)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(2 * 4))
        
        # Color (r, g, b)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(5 * 4))
        
        # Alpha
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(8 * 4))
        
        # Sphere radius
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(9 * 4))
        
        glBindVertexArray(0)
        
        # Lightning VAO
        self.lightning_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.lightning_VAO)
        
        self.lightning_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.lightning_VBO)
        glBufferData(GL_ARRAY_BUFFER, 100000, None, GL_DYNAMIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(2 * 4))
        
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(5 * 4))
        
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(8 * 4))
        
        glBindVertexArray(0)
    
    def render(self, state: Dict):
        """Render balls and lightning"""
        if not self.enabled or self.shader is None:
            return
        
        glUseProgram(self.shader)
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        # Pass time uniform for surface animations
        time_loc = glGetUniformLocation(self.shader, "time")
        if time_loc != -1:
            glUniform1f(time_loc, self.surface_time)
        
        # Render balls
        ball_data, ball_count = self._build_ball_geometry()
        if ball_data is not None:
            glBindVertexArray(self.VAO)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, ball_data.nbytes, ball_data)
            glDrawArrays(GL_TRIANGLES, 0, ball_count)
        
        # Render lightning
        lightning_data, lightning_count = self._build_lightning_geometry()
        if lightning_data is not None:
            glBindVertexArray(self.lightning_VAO)
            glBindBuffer(GL_ARRAY_BUFFER, self.lightning_VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, lightning_data.nbytes, lightning_data)
            
            old_line_width = glGetFloatv(GL_LINE_WIDTH)
            glLineWidth(self.lightning_line_thickness)
            glDrawArrays(GL_LINES, 0, lightning_count)
            glLineWidth(old_line_width)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame (vectorized)"""
        if not self.enabled:
            return
        
        # Update surface animation time
        self.surface_time += dt
        
        # Vectorized ball movement with wrapping
        self.positions[:, 0] += self.speeds * dt
        
        # Vectorized wrapping with modulo
        self.positions[:, 0] = np.where(
            self.positions[:, 0] > self.viewport.width,
            self.positions[:, 0] - self.viewport.width,
            self.positions[:, 0]
        )
        self.positions[:, 0] = np.where(
            self.positions[:, 0] < 0,
            self.positions[:, 0] + self.viewport.width,
            self.positions[:, 0]
        )
        
        # Update lightning
        self._update_lightning(dt)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        try:
            if self.VAO:
                glDeleteVertexArrays(1, [self.VAO])
            if self.lightning_VAO:
                glDeleteVertexArrays(1, [self.lightning_VAO])
            if self.VBO:
                glDeleteBuffers(1, [self.VBO])
            if self.lightning_VBO:
                glDeleteBuffers(1, [self.lightning_VBO])
        except:
            pass
        super().cleanup()
