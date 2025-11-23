"""
Audio-reactive balls with lightning - 16 balls respond to unique frequency bands
Each ball moves left to right with speed, size, color, depth, and transparency
driven by sound. Lightning occasionally arcs between nearby balls based on energy.

GPU-based rendering: Uses instanced rendering with all geometry generation,
animation, and wrapping handled by shaders.
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
    """Audio-reactive balls with lightning between high-energy pairs - GPU instanced rendering"""
    
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
        
        # Ball state arrays (uploaded as instance attributes to GPU)
        self.positions = np.zeros((self.num_balls, 3), dtype=np.float32)  # x, y, z
        self.sizes = np.full(self.num_balls, self.base_size, dtype=np.float32)
        self.colors = np.ones((self.num_balls, 3), dtype=np.float32)  # RGB
        self.alphas = np.ones(self.num_balls, dtype=np.float32)
        self.speeds = np.zeros(self.num_balls, dtype=np.float32)  # Horizontal movement
        self.energies = np.zeros(self.num_balls, dtype=np.float32)  # Current audio energy
        self.smoothed_energies = np.zeros(self.num_balls, dtype=np.float32)  # Smoothed energy for slower reaction
        self.energy_smoothing = 0.15  # Smoothing factor (lower = slower reaction)
        self.ball_ids = np.arange(self.num_balls, dtype=np.float32)  # Unique ID for each ball
        
        # Surface animation state
        self.surface_time = 0.0
        
        # Base y positions for sinusoidal height variation
        self.base_y_positions = np.zeros(self.num_balls, dtype=np.float32)
        # Sinusoidal frequency for each ball (Hz) - varies from 1.0 to 3.0
        self.wave_frequencies = np.linspace(1.0, 3.0, self.num_balls, dtype=np.float32)
        # Height variation amplitude in pixels
        self.wave_amplitude = 30.0
        
        # Lightning state
        self.active_lightning = []  # List of (ball_i, ball_j, intensity, age) tuples
        self.lightning_line_thickness = 2.0
        
        # OpenGL buffer objects
        self.VAO = None
        self.instance_VBO = None
        self.lightning_VAO = None
        self.lightning_VBO = None
        
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
        
        # Store base y positions for sinusoidal animation
        self.base_y_positions[:] = y_positions
        
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
            s: Saturation (scalar or array)
            v: Value (scalar or array)
        
        Returns:
            Array of shape (n, 3) with RGB values
        """
        h = np.asarray(h, dtype=np.float32).flatten() % 1.0
        s = np.asarray(s, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        
        # Ensure s and v are broadcastable
        if s.shape == ():
            s = np.full_like(h, s)
        if v.shape == ():
            v = np.full_like(h, v)
        
        i = (h * 6.0).astype(np.int32)
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
        
        # Now p, q, t, v are all arrays of same shape, so indexing works correctly
        rgb[mask0] = np.column_stack([v[mask0], t[mask0], p[mask0]])
        rgb[mask1] = np.column_stack([q[mask1], v[mask1], p[mask1]])
        rgb[mask2] = np.column_stack([p[mask2], v[mask2], t[mask2]])
        rgb[mask3] = np.column_stack([p[mask3], q[mask3], v[mask3]])
        rgb[mask4] = np.column_stack([t[mask4], p[mask4], v[mask4]])
        rgb[mask5] = np.column_stack([v[mask5], p[mask5], q[mask5]])
        
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
        """Update lightning state and generate new arcs (fully vectorized)"""
        # Vectorized decay of existing lightning
        if self.active_lightning:
            # Convert to numpy arrays for vectorized operations
            indices_i = np.array([x[0] for x in self.active_lightning], dtype=np.int32)
            indices_j = np.array([x[1] for x in self.active_lightning], dtype=np.int32)
            intensities = np.array([x[2] for x in self.active_lightning], dtype=np.float32)
            ages = np.array([x[3] for x in self.active_lightning], dtype=np.float32)
            
            # Vectorized decay and age update
            new_intensities = intensities * 0.8
            new_ages = ages + dt
            
            # Filter: keep only if intensity > 0.05 and age < 0.5
            keep_mask = (new_intensities > 0.05) & (new_ages < 0.5)
            
            # Rebuild list from filtered arrays
            self.active_lightning = [
                (int(indices_i[i]), int(indices_j[i]), float(new_intensities[i]), float(new_ages[i]))
                for i in np.where(keep_mask)[0]
            ]
        
        # Generate new lightning using fully vectorized approach
        high_energy_mask = self.energies >= self.lightning_threshold
        high_energy_indices = np.where(high_energy_mask)[0]
        
        if len(high_energy_indices) < 2:
            return  # Need at least 2 high-energy balls
        
        # Create all pairwise combinations efficiently
        # Use broadcasting to compute all distances at once
        pos_high = self.positions[high_energy_indices]  # Shape: (n_high, 3)
        energies_high = self.energies[high_energy_indices]  # Shape: (n_high,)
        
        # Compute pairwise distances using broadcasting
        # pos_high[:, np.newaxis, :] has shape (n_high, 1, 3)
        # pos_high[np.newaxis, :, :] has shape (1, n_high, 3)
        diff = pos_high[:, np.newaxis, :2] - pos_high[np.newaxis, :, :2]  # Shape: (n_high, n_high, 2)
        distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n_high, n_high)
        
        # Create index differences for adjacency check
        idx_diff = high_energy_indices[:, np.newaxis] - high_energy_indices[np.newaxis, :]  # Shape: (n_high, n_high)
        
        # Find valid pairs: upper triangle (i < j), nearby (distance < 200 or adjacent)
        upper_triangle_mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        adjacent_mask = np.abs(idx_diff) == 1
        nearby_mask = (distances < 200) | adjacent_mask
        valid_pairs_mask = upper_triangle_mask & nearby_mask
        
        # Get indices of valid pairs
        valid_i_idx, valid_j_idx = np.where(valid_pairs_mask)
        
        if len(valid_i_idx) == 0:
            return  # No valid pairs
        
        # Randomly select pairs for lightning based on probability
        rand_vals = np.random.random(len(valid_i_idx))
        selected_mask = rand_vals < self.lightning_probability
        selected_i_idx = valid_i_idx[selected_mask]
        selected_j_idx = valid_j_idx[selected_mask]
        
        if len(selected_i_idx) == 0:
            return  # No pairs selected
        
        # Convert back to original ball indices
        selected_i = high_energy_indices[selected_i_idx]
        selected_j = high_energy_indices[selected_j_idx]
        
        # Compute intensities for selected pairs (vectorized)
        selected_intensities = np.clip(
            self.energies[selected_i] * self.energies[selected_j],
            0, 1
        )
        
        # Filter out pairs that already exist in active_lightning
        # Create a set of existing pairs for O(1) lookup
        if self.active_lightning:
            existing_pairs = set()
            for a, b, _, _ in self.active_lightning:
                existing_pairs.add((min(a, b), max(a, b)))
        else:
            existing_pairs = set()
        
        # Add new lightning that doesn't already exist
        for i, j, intensity in zip(selected_i, selected_j, selected_intensities):
            pair = (min(i, j), max(i, j))
            if pair not in existing_pairs:
                self.active_lightning.append((int(i), int(j), float(intensity), 0.0))
                existing_pairs.add(pair)  # Prevent duplicates within same frame
    
    def _prepare_lightning_instances(self):
        """Prepare instance data for lightning bolts (GPU will generate geometry)"""
        if not self.active_lightning:
            return None
        
        n_bolts = len(self.active_lightning)
        
        # Extract lightning data
        indices_i = np.array([x[0] for x in self.active_lightning], dtype=np.int32)
        indices_j = np.array([x[1] for x in self.active_lightning], dtype=np.int32)
        intensities = np.array([x[2] for x in self.active_lightning], dtype=np.float32)
        
        # Get ball positions for each lightning bolt endpoint
        p1_positions = self.positions[indices_i]  # Shape: (n_bolts, 3)
        p2_positions = self.positions[indices_j]  # Shape: (n_bolts, 3)
        
        # Calculate colors (vectorized HSV to RGB)
        hues = ((indices_i + indices_j) / (self.num_balls * 2)) % 1.0
        colors = self._hsv_to_rgb_vectorized(hues, s=0.6 + 0.4 * intensities, v=1.0)
        
        # Blend towards white based on intensity
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        blend_factor = (intensities * 0.3)[:, np.newaxis]
        colors = colors * (1.0 - blend_factor) + white * blend_factor
        
        # Pack instance data: p1.xyz, p2.xyz, color.rgb, intensity
        instance_data = np.column_stack([
            p1_positions,
            p2_positions,
            colors,
            intensities * 0.8  # Alpha
        ]).astype(np.float32)
        
        return instance_data
    
    def compile_shader(self):
        """Compile and link shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            # Compile lightning shader too
            lightning_vert = shaders.compileShader(self.get_lightning_vertex_shader(), GL_VERTEX_SHADER)
            lightning_frag = shaders.compileShader(self.get_lightning_fragment_shader(), GL_FRAGMENT_SHADER)
            self.lightning_shader = shaders.compileProgram(lightning_vert, lightning_frag)
            
            return shader
        except Exception as e:
            print(f"AudioBallsEffect shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Quad vertex (same for all instances)
        layout(location = 0) in vec2 quadVertex;  // -1 to 1
        
        // Per-instance attributes
        layout(location = 1) in vec3 ballPosition;  // x, y, z
        layout(location = 2) in float ballSize;
        layout(location = 3) in vec3 ballColor;
        layout(location = 4) in float ballAlpha;
        layout(location = 5) in float ballId;
        
        uniform vec2 resolution;
        uniform float time;
        
        out vec3 vColor;
        out float vAlpha;
        out vec2 vLocalPos;
        out float vBallId;
        
        void main() {
            // Calculate world position of this vertex
            // No wrapping needed - CPU already handles wrapped instances
            vec2 worldPos = ballPosition.xy + quadVertex * ballSize;
            
            // Convert to clip space
            vec2 clipPos = (worldPos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Depth from z coordinate (0-100 range mapped to 0-1)
            float depth = ballPosition.z / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Pass attributes to fragment shader
            vColor = ballColor;
            vAlpha = ballAlpha;
            vLocalPos = quadVertex;  // Normalized position (-1 to 1)
            vBallId = ballId;
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
            
            // Discard fragments outside the circle
            if (distFromCenter > 1.0) discard;
            
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
    
    def get_lightning_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec3 position;  // x, y, z
        layout(location = 1) in vec3 color;
        layout(location = 2) in float alpha;
        
        uniform vec2 resolution;
        
        out vec3 vColor;
        out float vAlpha;
        
        void main() {
            vec2 clipPos = (position.xy / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depth = position.z / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            vColor = color;
            vAlpha = alpha;
        }
        """
    
    def get_lightning_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec3 vColor;
        in float vAlpha;
        
        out vec4 outColor;
        
        void main() {
            outColor = vec4(vColor, vAlpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced rendering"""
        # Ball rendering with instancing
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Static quad geometry (-1 to 1, covers ball area)
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        # Quad VBO (shared by all instances)
        self.quad_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_VBO)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Instance VBO (per-ball data: position, size, color, alpha, id)
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        # Allocate space: position(3) + size(1) + color(3) + alpha(1) + id(1) = 9 floats per instance
        max_instances = self.num_balls * 3  # Allow for wrapping (original + 2 wrapped copies)
        glBufferData(GL_ARRAY_BUFFER, max_instances * 9 * 4, None, GL_DYNAMIC_DRAW)
        
        # Set up instance attributes
        stride = 9 * 4  # 9 floats
        
        # Ball position (vec3)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)  # Advance per instance
        
        # Ball size (float)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glVertexAttribDivisor(2, 1)
        
        # Ball color (vec3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(4 * 4))
        glVertexAttribDivisor(3, 1)
        
        # Ball alpha (float)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(7 * 4))
        glVertexAttribDivisor(4, 1)
        
        # Ball ID (float)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8 * 4))
        glVertexAttribDivisor(5, 1)
        
        glBindVertexArray(0)
        
        # Lightning VAO (line rendering)
        self.lightning_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.lightning_VAO)
        
        self.lightning_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.lightning_VBO)
        # Lightning vertices: position.xyz(3) + color.rgb(3) + alpha(1) = 7 floats per vertex
        glBufferData(GL_ARRAY_BUFFER, 1000 * 7 * 4, None, GL_DYNAMIC_DRAW)
        
        # Position (x, y, z)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
        
        # Color (r, g, b)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
        
        # Alpha
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(6 * 4))
        
        glBindVertexArray(0)
    
    def render(self, state: Dict):
        """Render balls using instanced rendering"""
        if not self.enabled or self.shader is None:
            return
        
        glUseProgram(self.shader)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        if time_loc != -1:
            glUniform1f(time_loc, self.surface_time)
        
        viewport_width_loc = glGetUniformLocation(self.shader, "viewportWidth")
        if viewport_width_loc != -1:
            glUniform1f(viewport_width_loc, self.viewport.width)
        
        # Prepare instance data for balls (fully vectorized)
        ball_alphas = self.alphas * self.fade_factor
        
        # Vectorized wrapping detection
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Count instances needed
        n_originals = self.num_balls
        n_left_wraps = np.sum(left_edge_mask)
        n_right_wraps = np.sum(right_edge_mask)
        total_instances = n_originals + n_left_wraps + n_right_wraps
        
        # Pre-allocate instance array
        instance_data = np.zeros((total_instances, 9), dtype=np.float32)
        
        # Fill original instances (vectorized)
        instance_data[:n_originals, 0:3] = self.positions  # position xyz
        instance_data[:n_originals, 3] = self.sizes  # size
        instance_data[:n_originals, 4:7] = self.colors  # color rgb
        instance_data[:n_originals, 7] = ball_alphas  # alpha
        instance_data[:n_originals, 8] = self.ball_ids  # id
        
        # Add wrapped instances (vectorized)
        offset = n_originals
        if n_left_wraps > 0:
            left_indices = np.where(left_edge_mask)[0]
            instance_data[offset:offset+n_left_wraps, 0:3] = self.positions[left_indices]
            instance_data[offset:offset+n_left_wraps, 0] += self.viewport.width  # Wrap x to right
            instance_data[offset:offset+n_left_wraps, 3] = self.sizes[left_indices]
            instance_data[offset:offset+n_left_wraps, 4:7] = self.colors[left_indices]
            instance_data[offset:offset+n_left_wraps, 7] = ball_alphas[left_indices]
            instance_data[offset:offset+n_left_wraps, 8] = self.ball_ids[left_indices]
            offset += n_left_wraps
        
        if n_right_wraps > 0:
            right_indices = np.where(right_edge_mask)[0]
            instance_data[offset:offset+n_right_wraps, 0:3] = self.positions[right_indices]
            instance_data[offset:offset+n_right_wraps, 0] -= self.viewport.width  # Wrap x to left
            instance_data[offset:offset+n_right_wraps, 3] = self.sizes[right_indices]
            instance_data[offset:offset+n_right_wraps, 4:7] = self.colors[right_indices]
            instance_data[offset:offset+n_right_wraps, 7] = ball_alphas[right_indices]
            instance_data[offset:offset+n_right_wraps, 8] = self.ball_ids[right_indices]
        
        if total_instances > 0:
            instance_data = instance_data.flatten()
            
            # Upload instance data
            glBindVertexArray(self.VAO)
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
            
            # Draw instanced quads
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, total_instances)
            glBindVertexArray(0)
        
        # Render lightning
        lightning_data = self._prepare_lightning_instances()
        if lightning_data is not None and len(lightning_data) > 0:
            # Switch to lightning shader
            glUseProgram(self.lightning_shader)
            
            res_loc = glGetUniformLocation(self.lightning_shader, "resolution")
            glUniform2f(res_loc, self.viewport.width, self.viewport.height)
            
            # Build line geometry from lightning data (vectorized)
            n_bolts = len(lightning_data)
            
            # Extract components (vectorized)
            p1_positions = lightning_data[:, 0:3]
            p2_positions = lightning_data[:, 3:6]
            colors = lightning_data[:, 6:9]
            alphas = lightning_data[:, 9]
            
            # Handle wrapping (vectorized)
            dx = p2_positions[:, 0] - p1_positions[:, 0]
            wrap_threshold = self.viewport.width / 2
            
            p1_wrapped = p1_positions.copy()
            p2_wrapped = p2_positions.copy()
            
            # Vectorized wrapping logic
            wrap_right_mask = (np.abs(dx) > wrap_threshold) & (dx > 0)
            wrap_left_mask = (np.abs(dx) > wrap_threshold) & (dx < 0)
            p1_wrapped[wrap_right_mask, 0] += self.viewport.width
            p1_wrapped[wrap_left_mask, 0] -= self.viewport.width
            
            # Calculate midpoints with jitter (vectorized)
            jitter = np.random.uniform(-10, 10, size=(n_bolts, 2)).astype(np.float32)
            mid_xy = (p1_wrapped[:, :2] + p2_wrapped[:, :2]) / 2 + jitter
            mid_z = (p1_wrapped[:, 2] + p2_wrapped[:, 2]) / 2
            mid_positions = np.column_stack([mid_xy, mid_z])
            
            # Build vertex data: 4 vertices per bolt (p1->mid, mid->p2)
            # Format: position.xyz(3) + color.rgb(3) + alpha(1) = 7 floats per vertex
            n_vertices = n_bolts * 4
            vertex_data = np.zeros((n_vertices, 7), dtype=np.float32)
            
            # First vertex of each pair (p1 and mid)
            vertex_data[0::4, 0:3] = p1_wrapped  # p1 position
            vertex_data[0::4, 3:6] = colors  # color
            vertex_data[0::4, 6] = alphas  # alpha
            
            vertex_data[1::4, 0:3] = mid_positions  # mid position
            vertex_data[1::4, 3:6] = colors  # color
            vertex_data[1::4, 6] = alphas  # alpha
            
            # Second vertex of each pair (mid and p2)
            vertex_data[2::4, 0:3] = mid_positions  # mid position
            vertex_data[2::4, 3:6] = colors  # color
            vertex_data[2::4, 6] = alphas  # alpha
            
            vertex_data[3::4, 0:3] = p2_wrapped  # p2 position
            vertex_data[3::4, 3:6] = colors  # color
            vertex_data[3::4, 6] = alphas  # alpha
            
            if n_vertices > 0:
                vertex_data = vertex_data.flatten()
                
                glBindVertexArray(self.lightning_VAO)
                glBindBuffer(GL_ARRAY_BUFFER, self.lightning_VBO)
                glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
                
                old_line_width = glGetFloatv(GL_LINE_WIDTH)
                glLineWidth(self.lightning_line_thickness)
                glDrawArrays(GL_LINES, 0, n_vertices)
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
        
        # Apply sinusoidal height variation to each ball (vectorized)
        # Each ball oscillates at a different frequency
        time_scaled = 2 * np.pi * self.wave_frequencies * self.surface_time/10
        y_offsets = self.wave_amplitude * np.sin(time_scaled)
        self.positions[:, 1] = self.base_y_positions + y_offsets
        
        # Update lightning
        self._update_lightning(dt)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        try:
            if hasattr(self, 'VAO') and self.VAO:
                glDeleteVertexArrays(1, [self.VAO])
            if hasattr(self, 'lightning_VAO') and self.lightning_VAO:
                glDeleteVertexArrays(1, [self.lightning_VAO])
            if hasattr(self, 'quad_VBO') and self.quad_VBO:
                glDeleteBuffers(1, [self.quad_VBO])
            if hasattr(self, 'instance_VBO') and self.instance_VBO:
                glDeleteBuffers(1, [self.instance_VBO])
            if hasattr(self, 'lightning_VBO') and self.lightning_VBO:
                glDeleteBuffers(1, [self.lightning_VBO])
            if hasattr(self, 'lightning_shader') and self.lightning_shader:
                glDeleteProgram(self.lightning_shader)
        except:
            pass
        super().cleanup()
