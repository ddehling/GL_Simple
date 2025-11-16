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
        
        # Base y positions for sinusoidal height variation
        self.base_y_positions = np.zeros(self.num_balls, dtype=np.float32)
        # Sinusoidal frequency for each ball (Hz) - varies from 1.0 to 3.0
        self.wave_frequencies = np.linspace(1.0, 3.0, self.num_balls, dtype=np.float32)
        # Height variation amplitude in pixels
        self.wave_amplitude = 30.0
        
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
        """Update lightning state and generate new arcs (fully vectorized)"""
        # Vectorized decay of existing lightning
        if self.active_lightning:
            # Convert to numpy arrays for vectorized operations
            lightning_array = np.array(self.active_lightning, dtype=object)
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
    
    def _generate_ball_vertices_vectorized(self, pos, size, color, alpha, ball_id, segments, x_offset=0.0):
        """Generate all vertices for a single ball using vectorized operations
        
        Args:
            pos: Ball position (x, y, z)
            size: Ball radius
            color: Ball color (r, g, b)
            alpha: Ball alpha
            ball_id: Ball identifier
            segments: Number of segments in the circle
            x_offset: Horizontal offset for seamless wrapping
        
        Returns:
            Tuple of (vertices, offsets, colors, alphas, sphere_radii) as numpy arrays
        """
        # Pre-compute all angles (vectorized)
        seg_indices = np.arange(segments, dtype=np.float32)
        angles1 = (seg_indices / segments) * 2 * np.pi
        angles2 = ((seg_indices + 1) / segments) * 2 * np.pi
        
        # Compute cos and sin for all angles at once
        cos1 = np.cos(angles1)
        sin1 = np.sin(angles1)
        cos2 = np.cos(angles2)
        sin2 = np.sin(angles2)
        
        # Adjusted position for offset
        adjusted_pos = pos.copy()
        adjusted_pos[0] += x_offset
        
        # Each segment creates 3 vertices (center, perimeter1, perimeter2)
        n_vertices = segments * 3
        
        # Pre-allocate arrays
        vertices = np.zeros((n_vertices, 2), dtype=np.float32)
        offsets = np.zeros((n_vertices, 3), dtype=np.float32)
        colors_arr = np.zeros((n_vertices, 3), dtype=np.float32)
        alphas_arr = np.zeros(n_vertices, dtype=np.float32)
        sphere_radii = np.zeros(n_vertices, dtype=np.float32)
        
        # Fill arrays using advanced indexing
        # Center vertices (every 3rd vertex starting at 0)
        vertices[0::3, 0] = pos[0] + x_offset
        vertices[0::3, 1] = pos[1]
        offsets[0::3] = adjusted_pos
        colors_arr[0::3] = color
        alphas_arr[0::3] = alpha
        sphere_radii[0::3] = 0.0
        
        # Perimeter vertex 1 (every 3rd vertex starting at 1)
        vertices[1::3, 0] = pos[0] + x_offset + size * cos1
        vertices[1::3, 1] = pos[1] + size * sin1
        offsets[1::3] = adjusted_pos
        colors_arr[1::3] = color
        alphas_arr[1::3] = alpha
        sphere_radii[1::3] = 1.0 + (ball_id / 100.0)
        
        # Perimeter vertex 2 (every 3rd vertex starting at 2)
        vertices[2::3, 0] = pos[0] + x_offset + size * cos2
        vertices[2::3, 1] = pos[1] + size * sin2
        offsets[2::3] = adjusted_pos
        colors_arr[2::3] = color
        alphas_arr[2::3] = alpha
        sphere_radii[2::3] = 1.0 + (ball_id / 100.0)
        
        return vertices, offsets, colors_arr, alphas_arr, sphere_radii
    
    def _build_ball_geometry(self):
        """Build geometry for all balls as spheres with seamless wrapping (vectorized)"""
        segments = 24
        
        # Calculate alphas for all balls at once (vectorized)
        ball_alphas = self.alphas * self.fade_factor
        
        # Determine wrapping requirements for all balls (vectorized)
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Count total number of ball instances (original + wrapped duplicates)
        wrap_counts = np.ones(self.num_balls, dtype=np.int32)
        wrap_counts += left_edge_mask.astype(np.int32)
        wrap_counts += right_edge_mask.astype(np.int32)
        total_instances = np.sum(wrap_counts)
        
        # Pre-allocate arrays for all vertices
        vertices_per_ball = segments * 3
        total_vertices = total_instances * vertices_per_ball
        
        all_vertices = np.zeros((total_vertices, 2), dtype=np.float32)
        all_offsets = np.zeros((total_vertices, 3), dtype=np.float32)
        all_colors = np.zeros((total_vertices, 3), dtype=np.float32)
        all_alphas = np.zeros(total_vertices, dtype=np.float32)
        all_sphere_radii = np.zeros(total_vertices, dtype=np.float32)
        
        # Fill arrays by processing each ball
        vertex_offset = 0
        
        for i in range(self.num_balls):
            pos = self.positions[i]
            size = self.sizes[i]
            color = self.colors[i]
            alpha = ball_alphas[i]
            
            # Determine x_offsets for this ball
            x_offsets = [0.0]  # Always render original
            if left_edge_mask[i]:
                x_offsets.append(self.viewport.width)
            if right_edge_mask[i]:
                x_offsets.append(-self.viewport.width)
            
            # Generate vertices for each wrapped instance of this ball
            for x_offset in x_offsets:
                verts, offs, cols, alphs, radii = self._generate_ball_vertices_vectorized(
                    pos, size, color, alpha, i, segments, x_offset
                )
                
                # Copy into pre-allocated arrays
                end_offset = vertex_offset + vertices_per_ball
                all_vertices[vertex_offset:end_offset] = verts
                all_offsets[vertex_offset:end_offset] = offs
                all_colors[vertex_offset:end_offset] = cols
                all_alphas[vertex_offset:end_offset] = alphs
                all_sphere_radii[vertex_offset:end_offset] = radii
                
                vertex_offset = end_offset
        
        if total_vertices == 0:
            return None, 0
        
        # Combine into final vertex data using column_stack (single operation)
        vertex_data = np.column_stack([
            all_vertices,
            all_offsets,
            all_colors,
            all_alphas,
            all_sphere_radii
        ]).astype(np.float32)
        
        return vertex_data, total_vertices
    
    def _lightning_color_from_indices(self, i, j, intensity):
        """Generate varied lightning color based on ball indices and intensity (scalar version)"""
        # Use ball indices to determine base hue
        hue = ((i + j) / (self.num_balls * 2)) % 1.0
        
        # Convert to RGB
        rgb = self._hsv_to_rgb(hue, s=0.6 + 0.4 * intensity, v=1.0)
        
        # Blend towards white based on intensity
        rgb = rgb * (1.0 - intensity * 0.3) + np.array([1.0, 1.0, 1.0]) * (intensity * 0.3)
        
        return rgb.astype(np.float32)
    
    def _lightning_color_from_indices_vectorized(self, indices_i, indices_j, intensities):
        """Generate varied lightning colors based on ball indices and intensities (vectorized)
        
        Args:
            indices_i: Array of first ball indices
            indices_j: Array of second ball indices
            intensities: Array of intensity values
        
        Returns:
            Array of shape (n, 3) with RGB values
        """
        # Vectorized hue calculation
        hues = ((indices_i + indices_j) / (self.num_balls * 2)) % 1.0
        
        # Vectorized saturation values
        saturations = 0.6 + 0.4 * intensities
        
        # Convert HSV to RGB for all lightning bolts
        # Vectorized HSV conversion
        h = hues % 1.0
        i = (h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        
        # Broadcast saturations for vectorized operations
        s = saturations
        v = 1.0
        
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        
        i = i % 6
        n = len(h)
        
        # Vectorized RGB selection
        rgb = np.zeros((n, 3), dtype=np.float32)
        
        mask0 = i == 0
        mask1 = i == 1
        mask2 = i == 2
        mask3 = i == 3
        mask4 = i == 4
        mask5 = i == 5
        
        rgb[mask0, 0] = v
        rgb[mask0, 1] = t[mask0]
        rgb[mask0, 2] = p[mask0]
        
        rgb[mask1, 0] = q[mask1]
        rgb[mask1, 1] = v
        rgb[mask1, 2] = p[mask1]
        
        rgb[mask2, 0] = p[mask2]
        rgb[mask2, 1] = v
        rgb[mask2, 2] = t[mask2]
        
        rgb[mask3, 0] = p[mask3]
        rgb[mask3, 1] = q[mask3]
        rgb[mask3, 2] = v
        
        rgb[mask4, 0] = t[mask4]
        rgb[mask4, 1] = p[mask4]
        rgb[mask4, 2] = v
        
        rgb[mask5, 0] = v
        rgb[mask5, 1] = p[mask5]
        rgb[mask5, 2] = q[mask5]
        
        # Vectorized blend towards white based on intensity
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        blend_factor = (intensities * 0.3)[:, np.newaxis]
        rgb = rgb * (1.0 - blend_factor) + white * blend_factor
        
        return rgb
    
    def _build_lightning_geometry(self):
        """Build geometry for all active lightning bolts with seamless wrapping (vectorized)"""
        if not self.active_lightning:
            return None, 0
        
        n_bolts = len(self.active_lightning)
        
        # Extract all lightning data into arrays
        indices_i = np.array([x[0] for x in self.active_lightning], dtype=np.int32)
        indices_j = np.array([x[1] for x in self.active_lightning], dtype=np.int32)
        intensities = np.array([x[2] for x in self.active_lightning], dtype=np.float32)
        ages = np.array([x[3] for x in self.active_lightning], dtype=np.float32)
        
        # Vectorized position extraction
        p1_positions = self.positions[indices_i]  # Shape: (n_bolts, 3)
        p2_positions = self.positions[indices_j]  # Shape: (n_bolts, 3)
        
        # Vectorized wrapping calculation
        p1_wrapped = p1_positions.copy()
        p2_wrapped = p2_positions.copy()
        
        # Handle horizontal wrapping: use shortest path (vectorized)
        dx = p2_positions[:, 0] - p1_positions[:, 0]
        wrap_threshold = self.viewport.width / 2
        
        # Wrap p1 to right where p2 is far right
        wrap_right_mask = (np.abs(dx) > wrap_threshold) & (dx > 0)
        p1_wrapped[wrap_right_mask, 0] += self.viewport.width
        
        # Wrap p1 to left where p2 is far left
        wrap_left_mask = (np.abs(dx) > wrap_threshold) & (dx < 0)
        p1_wrapped[wrap_left_mask, 0] -= self.viewport.width
        
        # Vectorized midpoint calculation with random jitter
        mid_jitter = np.random.uniform(-10, 10, size=(n_bolts, 2)).astype(np.float32)
        mid_xy = (p1_wrapped[:, :2] + p2_wrapped[:, :2]) / 2 + mid_jitter  # Shape: (n_bolts, 2)
        mid_z = (p1_wrapped[:, 2] + p2_wrapped[:, 2]) / 2  # Shape: (n_bolts,)
        mid_positions = np.column_stack([mid_xy, mid_z])  # Shape: (n_bolts, 3)
        
        # Vectorized color generation
        lightning_colors = self._lightning_color_from_indices_vectorized(
            indices_i, indices_j, intensities
        )  # Shape: (n_bolts, 3)
        
        # Vectorized alpha calculation
        lightning_alphas = intensities * 0.8  # Shape: (n_bolts,)
        
        # Build vertex data efficiently using NumPy operations
        # Each lightning bolt has 2 segments, each segment has 2 vertices = 4 vertices per bolt
        n_vertices = n_bolts * 4
        
        # Pre-allocate arrays
        vertices = np.zeros((n_vertices, 2), dtype=np.float32)
        offsets = np.zeros((n_vertices, 3), dtype=np.float32)
        colors = np.zeros((n_vertices, 3), dtype=np.float32)
        alphas = np.zeros(n_vertices, dtype=np.float32)
        
        # Use advanced indexing to fill arrays efficiently
        # Segment 1: p1 to mid (vertices 0, 1 for each bolt)
        vertices[0::4] = p1_wrapped[:, :2]  # p1 start
        vertices[1::4] = mid_xy  # mid end
        
        offsets[0::4] = p1_wrapped  # p1 offset
        offsets[1::4] = mid_positions  # mid offset
        
        colors[0::4] = lightning_colors
        colors[1::4] = lightning_colors
        
        alphas[0::4] = lightning_alphas
        alphas[1::4] = lightning_alphas
        
        # Segment 2: mid to p2 (vertices 2, 3 for each bolt)
        vertices[2::4] = mid_xy  # mid start
        vertices[3::4] = p2_wrapped[:, :2]  # p2 end
        
        offsets[2::4] = mid_positions  # mid offset
        offsets[3::4] = p2_wrapped  # p2 offset
        
        colors[2::4] = lightning_colors
        colors[3::4] = lightning_colors
        
        alphas[2::4] = lightning_alphas
        alphas[3::4] = lightning_alphas
        
        # Combine into final vertex data
        vertex_data = np.column_stack([
            vertices,
            offsets,
            colors,
            alphas
        ]).astype(np.float32)
        
        return vertex_data, n_vertices
    
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
