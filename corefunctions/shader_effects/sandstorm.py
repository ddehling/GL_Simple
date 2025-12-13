"""
Sandstorm shader effect - GPU-accelerated sand particles
Instanced rendering with wind-driven physics and horizontal wrapping
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_sandstorm(state, outstate, fade_duration=5.0, max_particles=600, 
                     particle_size=2.0, squish_top_width=1.0):
    """
    Wind-driven sandstorm effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_sandstorm, density=1.5, 
                               max_particles=200, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        fade_duration: Duration of fade in/out in seconds (default 5.0)
        max_particles: Maximum number of sand particles (default 200)
        particle_size: Base size of sand particles in pixels (default 2.0)
        squish_top_width: Horizontal width multiplier at top of viewport (default 1.0)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        try:
            effect = viewport.add_effect(
                SandstormEffect,
                max_particles=max_particles, 
                particle_size=particle_size, 
                squish_top_width=squish_top_width
            )
            state['sandstorm_effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing sandstorm effect: {e}")
            traceback.print_exc()
            return
    
    # Update effect parameters from outstate
    if 'sandstorm_effect' in state:
        effect = state['sandstorm_effect']
        
        # Get wind and sand_density from outstate
        effect.wind_strength = outstate.get('wind', 0.0)
        effect.sand_density = outstate.get('sand_density', 0.0)
        
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)  # Default 60s if not set
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            # Fade in
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            # Fade out
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            # Full visibility
            fade_factor = 1.0
        
        # Update effect's fade factor (clip to 0-1 range)
        effect.fade_factor = np.clip(fade_factor, 0, 1)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'sandstorm_effect' in state:
            effect = state['sandstorm_effect']
            if effect in viewport.effects:
                viewport.effects.remove(effect)
            effect.cleanup()
            del state['sandstorm_effect']


# ============================================================================
# Sandstorm Effect Class
# ============================================================================

class SandstormEffect(ShaderEffect):
    """GPU-based sandstorm effect using instanced rendering"""
    
    def __init__(self, viewport, max_particles: int = 600, particle_size: float = 2.0,
                 squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.max_particles = max_particles
        self.particle_size = particle_size
        self.squish_top_width = squish_top_width
        self.viewport_height = viewport.height
        self.instance_VBO = None
        self.fade_factor = 0.0  # For fade in/out (updated by event wrapper)
        
        # Environmental parameters (updated by event wrapper)
        self.wind_strength = 0.0   # From outstate['wind']
        self.sand_density = 0.0    # From outstate['sand_density']
        
        # Vectorized particle data
        self.positions = np.zeros((0, 2), dtype=np.float32)  # [x, y]
        self.velocities = np.zeros((0, 2), dtype=np.float32)  # [vx, vy]
        self.sizes = np.zeros(0, dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)  # [r, g, b]
        self.alphas = np.zeros(0, dtype=np.float32)
        self.lifetimes = np.zeros(0, dtype=np.float32)
        self.distances = np.zeros(0, dtype=np.float32)  # Depth (10-40) for 3D ordering
        self.squish_factors = np.zeros(0, dtype=np.float32)  # Horizontal width multipliers
        self.turbulence_phases = np.zeros(0, dtype=np.float32)  # Individual turbulence
        
        # Time tracking
        self.time = 0.0
        
        # Horizontal wrapping margin
        self.wrap_margin = 50  # Should exceed max particle size
        
        # Spawn initial particles across the entire viewport for immediate visibility
        initial_count = min(50, max_particles)
        if initial_count > 0:
            self._spawn_initial_particles(initial_count)
        
    def _spawn_particles(self, count: int):
        """Spawn new sand particles"""
        if count <= 0:
            return
            
        # Spawn from full horizontal range, distributed across full viewport height
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, count),  # Full horizontal range
            np.random.uniform(0, self.viewport.height, count)  # Across full height
        ])
        
        # Base velocities (primarily horizontal from wind, minimal vertical)
        wind_speed = self.wind_strength * 150.0  # Stronger horizontal motion
        new_velocities = np.column_stack([
            np.random.uniform(wind_speed * 0.8, wind_speed * 1.2, count),  # Horizontal (wind-driven)
            np.random.uniform(-5.0, 5.0, count)  # Minimal vertical drift
        ])
        
        # Random distances (depth) between 10 and 40
        new_distances = np.random.uniform(10.0, 40.0, count)
        
        # Size scaled by distance (closer = larger)
        base_sizes = np.random.uniform(self.particle_size * 0.5, self.particle_size * 1.5, count)
        new_sizes = base_sizes * (10.0 / new_distances)  # Scale by distance
        
        # Sand colors (yellowish-brown tones)
        hue_variation = np.random.uniform(-0.05, 0.05, count)
        saturation = np.random.uniform(0.3, 0.5, count)
        brightness = np.random.uniform(0.6, 0.9, count)
        
        # Convert HSV to RGB for sand colors (hue around 0.11 = yellow-orange)
        from skimage import color as skcolor
        new_colors = np.zeros((count, 3), dtype=np.float32)
        for i in range(count):
            hsv = np.array([0.11 + hue_variation[i], saturation[i], brightness[i]])
            rgb = skcolor.hsv2rgb(hsv.reshape(1, 1, 3)).reshape(3)
            new_colors[i] = rgb
        
        # Adjust alpha based on distance (farther = more transparent)
        base_alphas = np.random.uniform(0.3, 0.6, count)
        new_alphas = base_alphas * (40.0 - new_distances) / 30.0  # Farther particles more transparent
        
        # Lifetime based on viewport height and fall speed
        new_lifetimes = np.ones(count) * 10.0  # Particles last ~10 seconds
        
        # Random turbulence phases
        new_turbulence_phases = np.random.uniform(0, 2 * np.pi, count)
        
        # Calculate squish factors based on y position
        y_normalized = (self.viewport_height - new_positions[:, 1]) / self.viewport_height
        new_squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Concatenate with existing arrays
        self.positions = np.vstack([self.positions, new_positions]) if len(self.positions) > 0 else new_positions
        self.velocities = np.vstack([self.velocities, new_velocities]) if len(self.velocities) > 0 else new_velocities
        self.sizes = np.concatenate([self.sizes, new_sizes]) if len(self.sizes) > 0 else new_sizes
        self.colors = np.vstack([self.colors, new_colors]) if len(self.colors) > 0 else new_colors
        self.alphas = np.concatenate([self.alphas, new_alphas]) if len(self.alphas) > 0 else new_alphas
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes]) if len(self.lifetimes) > 0 else new_lifetimes
        self.distances = np.concatenate([self.distances, new_distances]) if len(self.distances) > 0 else new_distances
        self.squish_factors = np.concatenate([self.squish_factors, new_squish_factors]) if len(self.squish_factors) > 0 else new_squish_factors
        self.turbulence_phases = np.concatenate([self.turbulence_phases, new_turbulence_phases]) if len(self.turbulence_phases) > 0 else new_turbulence_phases
    
    def _spawn_initial_particles(self, count: int):
        """Spawn initial particles distributed across the entire viewport for immediate visibility"""
        if count <= 0:
            return
        
        # Spawn from random positions across the entire viewport height
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, count),  # Full horizontal range
            np.random.uniform(0, self.viewport.height, count)  # Across full height
        ])
        
        # Base velocities (primarily horizontal, minimal vertical)
        wind_speed = 0.5 * 150.0  # Use moderate wind for initial particles
        new_velocities = np.column_stack([
            np.random.uniform(wind_speed * 0.8, wind_speed * 1.2, count),
            np.random.uniform(-5.0, 5.0, count)  # Minimal vertical drift
        ])
        
        # Random distances (depth) between 10 and 40
        new_distances = np.random.uniform(10.0, 40.0, count)
        
        # Size scaled by distance
        base_sizes = np.random.uniform(self.particle_size * 0.5, self.particle_size * 1.5, count)
        new_sizes = base_sizes * (10.0 / new_distances)
        
        # Sand colors
        hue_variation = np.random.uniform(-0.05, 0.05, count)
        saturation = np.random.uniform(0.3, 0.5, count)
        brightness = np.random.uniform(0.6, 0.9, count)
        
        from skimage import color as skcolor
        new_colors = np.zeros((count, 3), dtype=np.float32)
        for i in range(count):
            hsv = np.array([0.11 + hue_variation[i], saturation[i], brightness[i]])
            rgb = skcolor.hsv2rgb(hsv.reshape(1, 1, 3)).reshape(3)
            new_colors[i] = rgb
        
        # Adjust alpha based on distance
        base_alphas = np.random.uniform(0.4, 0.7, count)
        new_alphas = base_alphas * (40.0 - new_distances) / 30.0
        
        # Lifetime
        new_lifetimes = np.ones(count) * 10.0
        
        # Random turbulence phases
        new_turbulence_phases = np.random.uniform(0, 2 * np.pi, count)
        
        # Calculate squish factors based on y position
        y_normalized = (self.viewport_height - new_positions[:, 1]) / self.viewport_height
        new_squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Initialize arrays
        self.positions = new_positions
        self.velocities = new_velocities
        self.sizes = new_sizes
        self.colors = new_colors
        self.alphas = new_alphas
        self.lifetimes = new_lifetimes
        self.distances = new_distances
        self.squish_factors = new_squish_factors
        self.turbulence_phases = new_turbulence_phases
    
    def update(self, dt: float, state: Dict):
        """Update particle positions and properties"""
        if not self.enabled:
            return
        
        self.time += dt
        
        # Spawn new particles based on sand_density
        if len(self.positions) < self.max_particles:
            # More aggressive spawn rate - sand_density=1.0 spawns ~15 particles per frame
            spawn_rate = self.sand_density * 15.0 * 60.0  # Scale density to particles per second
            particles_to_spawn = int(spawn_rate * dt)  # Convert to particles this frame
            particles_to_spawn = min(particles_to_spawn, self.max_particles - len(self.positions))
            if particles_to_spawn > 0:
                self._spawn_particles(particles_to_spawn)
        
        if len(self.positions) == 0:
            return
        
        # Update turbulence phases
        self.turbulence_phases += dt * 2.0
        
        # Calculate wind-driven turbulence (horizontal and vertical variation)
        turbulence_x = np.sin(self.turbulence_phases) * self.wind_strength * 50.0
        turbulence_y = np.cos(self.turbulence_phases * 1.3) * 15.0  # Vertical drift
        
        # Update velocities with wind influence (primarily horizontal)
        wind_speed = self.wind_strength * 150.0
        self.velocities[:, 0] = wind_speed + turbulence_x
        self.velocities[:, 1] = turbulence_y  # Mainly turbulence-driven vertical
        
        # Update positions
        self.positions += self.velocities * dt
        
        # Wrap horizontally (particles crossing edges appear on opposite side)
        self.positions[:, 0] = self.positions[:, 0] % self.viewport.width
        
        # Update lifetimes
        self.lifetimes -= dt
        
        # Remove particles that are out of bounds or expired
        valid_mask = (
            (self.positions[:, 0] < self.viewport.width + 10) &  # Before right edge
            (self.positions[:, 1] > -10) &  # Not above top
            (self.positions[:, 1] < self.viewport.height + 10) &  # Not below bottom
            (self.lifetimes > 0)  # Not expired
        )
        
        self.positions = self.positions[valid_mask]
        self.velocities = self.velocities[valid_mask]
        self.sizes = self.sizes[valid_mask]
        self.colors = self.colors[valid_mask]
        self.alphas = self.alphas[valid_mask]
        self.lifetimes = self.lifetimes[valid_mask]
        self.distances = self.distances[valid_mask]
        self.squish_factors = self.squish_factors[valid_mask]
        self.turbulence_phases = self.turbulence_phases[valid_mask]
        
        # Update squish factors based on current y position
        if len(self.positions) > 0:
            y_normalized = (self.viewport_height - self.positions[:, 1]) / self.viewport_height
            y_normalized = np.clip(y_normalized, 0, 1)
            self.squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
    
    def render(self, state: Dict):
        """Render sand particles with horizontal wrapping"""
        if not self.enabled:
            return
        if len(self.positions) == 0:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        # Prepare instance data with horizontal wrapping
        all_positions = []
        all_sizes = []
        all_colors = []
        all_distances = []
        all_squish_factors = []
        
        # Detect particles near edges for wrapping
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Add original particles
        all_positions.append(self.positions)
        all_sizes.append(self.sizes)
        all_colors.append(np.column_stack([self.colors, self.alphas]))
        all_distances.append(self.distances)
        all_squish_factors.append(self.squish_factors)
        
        # Add duplicates for particles near left edge (appear on right)
        if np.any(left_edge_mask):
            left_indices = np.where(left_edge_mask)[0]
            duplicate_pos = self.positions[left_indices].copy()
            duplicate_pos[:, 0] += self.viewport.width
            all_positions.append(duplicate_pos)
            all_sizes.append(self.sizes[left_indices])
            all_colors.append(np.column_stack([self.colors[left_indices], self.alphas[left_indices]]))
            all_distances.append(self.distances[left_indices])
            all_squish_factors.append(self.squish_factors[left_indices])
        
        # Add duplicates for particles near right edge (appear on left)
        if np.any(right_edge_mask):
            right_indices = np.where(right_edge_mask)[0]
            duplicate_pos = self.positions[right_indices].copy()
            duplicate_pos[:, 0] -= self.viewport.width
            all_positions.append(duplicate_pos)
            all_sizes.append(self.sizes[right_indices])
            all_colors.append(np.column_stack([self.colors[right_indices], self.alphas[right_indices]]))
            all_distances.append(self.distances[right_indices])
            all_squish_factors.append(self.squish_factors[right_indices])
        
        # Combine all particles (originals + duplicates)
        positions_combined = np.vstack(all_positions)
        sizes_combined = np.concatenate(all_sizes)
        colors_combined = np.vstack(all_colors)
        distances_combined = np.concatenate(all_distances)
        squish_combined = np.concatenate(all_squish_factors)
        
        # Build instance data array
        # Layout: [x, y, size, r, g, b, a, distance, squish_factor]
        instance_data = np.column_stack([
            positions_combined,
            sizes_combined,
            colors_combined,
            distances_combined,
            squish_combined
        ]).astype(np.float32)
        
        # Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Draw instanced particles
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(instance_data))
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec2 offset;    // Particle position (x, y)
        layout(location = 2) in float size;     // Particle size
        layout(location = 3) in vec4 color;     // Color (r, g, b, alpha)
        layout(location = 4) in float distance; // Depth value (10-40)
        layout(location = 5) in float squishFactor; // Horizontal width multiplier
        
        out vec4 fragColor;
        out vec2 fragPos;  // Position within quad (-1 to 1)
        uniform vec2 resolution;
        uniform float fadeAlpha;  // Global fade factor
        
        void main() {
            fragPos = position;
            
            // Scale by particle size with squish applied to horizontal
            vec2 scaled = vec2(
                position.x * size * squishFactor,
                position.y * size
            );
            
            // Translate to particle position
            vec2 pos = scaled + offset;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping: z = 0-100 -> depth = 0.0-1.0
            // distance 10 (near) -> depth 0.10
            // distance 40 (far) -> depth 0.40
            float depth = distance / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Apply fade factor to alpha
            fragColor = vec4(color.rgb, color.a * fadeAlpha);
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;  // Position within quad (-1 to 1)
        out vec4 outColor;
        
        void main() {
            // Create circular particles with soft edges
            float dist = length(fragPos);
            float alpha = smoothstep(1.0, 0.6, dist);
            
            if (alpha < 0.01) {
                discard;
            }
            
            // Add some texture variation
            float noise = fract(sin(dot(fragPos * 10.0, vec2(12.9898, 78.233))) * 43758.5453);
            vec3 color = fragColor.rgb * (0.9 + noise * 0.2);
            
            outColor = vec4(color, fragColor.a * alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link sandstorm shaders - REQUIRED by ShaderEffect base class"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vertex = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            fragment = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vertex, fragment)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced rendering"""
        # Quad vertices
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer (quad)
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
        
        # Instance buffer (particles)
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        self.VBOs.append(self.instance_VBO)
        
        # Instance attributes: offset (vec2), size (float), color (vec4), distance (float), squish (float)
        stride = 4 * 9  # 9 floats per instance
        
        # offset (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # size (location 2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # color (location 3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # distance (location 4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # squish_factor (location 5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
        glBindVertexArray(0)
