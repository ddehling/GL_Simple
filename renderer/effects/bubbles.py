"""
Bubble effect - rising bubbles from the bottom
GPU-accelerated bubble system with size variation and wobble
"""
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_bubbles(state, outstate, density=1.0, audio_sensitivity=0.5):
    """
    Shader-based bubble effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_bubbles, density=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Bubble spawn rate multiplier
        audio_sensitivity: Audio reactivity multiplier
    """
    # Get the viewport
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    squish_top_width = outstate.get('scale', 1.0)  # Get scale from state
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize bubble effect on first call
    if state['count'] == 0:
        # Start with base number (will adjust dynamically)
        num_bubbles = 200
        print(f"Initializing bubble effect for frame {frame_id} (base: {num_bubbles} bubbles)")
        
        try:
            bubble_effect = viewport.add_effect(
                BubbleEffect,
                num_bubbles=num_bubbles,
                density=density,
                squish_top_width=squish_top_width
            )
            state['bubble_effect'] = bubble_effect
            print(f"✓ Initialized shader bubbles for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize bubbles: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update density every frame
    if 'bubble_effect' in state:
        bubble_density = outstate.get('bubble_density', density)
        state['bubble_effect'].set_target_density(bubble_density)

        # Update squish width from scale
        state['bubble_effect'].squish_top_width = squish_top_width

        # Update tide level (controls water line/beach boundary)
        tide_level = outstate.get('tide_level', 0.5)
        state['bubble_effect'].set_tide_level(tide_level)

        # Current strength: positive wind drifts bubbles right, negative drifts left.
        state['bubble_effect'].wind = float(outstate.get('wind', 0.0))
    
    # On close event, clean up
    if state['count'] == -1:
        if 'bubble_effect' in state:
            print(f"Cleaning up bubble effect for frame {frame_id}")
            viewport.effects.remove(state['bubble_effect'])
            state['bubble_effect'].cleanup()
            print(f"✓ Cleaned up shader bubbles for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class BubbleEffect(ShaderEffect):
    """GPU-based bubble effect using instanced rendering"""
    
    def __init__(self, viewport, num_bubbles: int = 200, density: float = 1.0, squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.num_bubbles = num_bubbles
        self.base_num_bubbles = num_bubbles
        self.target_bubbles = num_bubbles
        self.density = density
        self.instance_VBO = None
        
        # Width scaling based on vertical position (bottom = 1.0, top = squish_top_width)
        self.squish_top_width = squish_top_width
        
        # Tide level controls water line (0.0 = low tide, 1.0 = high tide)
        self.tide_level = 0.5
        self.water_line_y = self.viewport.height * self.tide_level
        
        # Vectorized bubble data
        self.positions = None
        self.velocities = None
        self.sizes = None
        self.base_sizes = None
        self.squish_factors = None  # Width scaling per bubble
        self.wobble_phases = None
        self.wobble_speeds = None
        self.alphas = None
        self.colors = None
        
        # Audio reactivity
        self.audio_energy = 0.0
        self.audio_sensitivity = 0.5

        # Current drift: set from outstate['wind']; pushes bubbles sideways
        # as they rise, so strong currents visibly sweep the bubble column.
        self.wind = 0.0

        # Cached uniform locations (populated in setup_buffers).
        self._uniform_resolution = -1
        
        self._initialize_bubbles()
        
    def _initialize_bubbles(self):
        """Initialize all bubble data as numpy arrays"""
        n = self.num_bubbles
        
        # Positions: x, y, z
        # Spawn bubbles from bottom up to 30% of the water line height
        max_spawn_y = min(self.water_line_y * 0.3, self.viewport.height * 0.3)
        self.positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n),  # x
            np.random.uniform(-20, max_spawn_y, n),  # y (start below water)
            np.random.uniform(0, 100, n)  # z (depth)
        ])
        
        # Velocities (bubbles rise)
        self.velocities = np.random.uniform(15, 40, n)
        
        # Bubble sizes based on depth
        depth_factors = 1.0 - (self.positions[:, 2] / 100.0)  # 1.0 (near) to 0.0 (far)
        base_sizes = np.random.uniform(2.0, 8.0, n)
        self.sizes = base_sizes * (0.3 + 0.7 * depth_factors)
        self.base_sizes = self.sizes.copy()
        
        # Wobble effect for horizontal movement
        self.wobble_phases = np.random.uniform(0, 2 * np.pi, n)
        self.wobble_speeds = np.random.uniform(1.0, 3.0, n)
        
        # Calculate squish factors based on y position (bottom = 1.0, top = squish_top_width)
        # Normalize y: 0 at bottom, 1 at top
        y_normalized = self.positions[:, 1] / self.viewport.height
        self.squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Alpha based on depth (more transparent overall)
        self.alphas = 0.15 + 0.25 * depth_factors  # Range: 0.15-0.4
        
        # Colors: slight blue/cyan tint with shimmer
        self.colors = np.column_stack([
            np.random.uniform(0.6, 0.9, n),  # R
            np.random.uniform(0.8, 1.0, n),  # G
            np.ones(n)  # B: full blue
        ])
        
    def _reset_bubbles(self, mask):
        """Reset bubbles that have risen above the screen OR remove if over target"""
        n_reset = np.sum(mask)
        if n_reset == 0:
            return
        
        # Check if we're over the target and need to remove some bubbles
        current_count = len(self.positions)
        if current_count > self.target_bubbles:
            # Remove bubbles instead of resetting them
            n_to_remove = min(n_reset, current_count - self.target_bubbles)
            
            # Get indices of bubbles to reset
            reset_indices = np.where(mask)[0]
            
            # Remove the first n_to_remove bubbles
            remove_indices = reset_indices[:n_to_remove]
            keep_mask = np.ones(current_count, dtype=bool)
            keep_mask[remove_indices] = False
            
            self.positions = self.positions[keep_mask]
            self.velocities = self.velocities[keep_mask]
            self.sizes = self.sizes[keep_mask]
            self.base_sizes = self.base_sizes[keep_mask]
            self.wobble_phases = self.wobble_phases[keep_mask]
            self.wobble_speeds = self.wobble_speeds[keep_mask]
            self.squish_factors = self.squish_factors[keep_mask]
            self.alphas = self.alphas[keep_mask]
            self.colors = self.colors[keep_mask]
            
            self.num_bubbles = len(self.positions)
            
            # Update the mask for remaining resets
            if n_to_remove < n_reset:
                remaining_reset_indices = reset_indices[n_to_remove:]
                for i, old_idx in enumerate(remaining_reset_indices):
                    adjustment = np.sum(remove_indices < old_idx)
                    remaining_reset_indices[i] = old_idx - adjustment
                
                new_mask = np.zeros(len(self.positions), dtype=bool)
                new_mask[remaining_reset_indices] = True
                mask = new_mask
                n_reset = n_reset - n_to_remove
            else:
                return
        
        # Reset remaining bubbles below water line
        if n_reset > 0:
            max_spawn_y = min(self.water_line_y * 0.3, self.viewport.height * 0.3)
            self.positions[mask, 0] = np.random.uniform(0, self.viewport.width, n_reset)
            self.positions[mask, 1] = np.random.uniform(-20, max_spawn_y, n_reset)
            self.positions[mask, 2] = np.random.uniform(0, 100, n_reset)
            
            self.velocities[mask] = np.random.uniform(15, 40, n_reset)
            
            depth_factors = 1.0 - (self.positions[mask, 2] / 100.0)
            base_sizes = np.random.uniform(2.0, 8.0, n_reset)
            self.sizes[mask] = base_sizes * (0.3 + 0.7 * depth_factors)
            self.base_sizes[mask] = self.sizes[mask]
            
            self.wobble_phases[mask] = np.random.uniform(0, 2 * np.pi, n_reset)
            self.wobble_speeds[mask] = np.random.uniform(1.0, 3.0, n_reset)
            
            self.alphas[mask] = 0.15 + 0.25 * depth_factors  # Match initialization
        
    def set_target_density(self, density):
        """Set target bubble density and adjust bubble count"""
        self.density = density
        self.target_bubbles = int(self.base_num_bubbles * density)
    
    def set_tide_level(self, tide_level):
        """Set tide level (0.0 = low tide/beach high, 1.0 = high tide/beach low)"""
        self.tide_level = tide_level
        # Water line: low tide (0.0) = 60% up screen, high tide (1.0) = 95% up screen
        self.water_line_y = self.viewport.height * (0.6 + 0.35 * tide_level)
    
    def update_audio(self, bass_energy):
        """Update audio energy for bubble generation"""
        self.audio_energy = bass_energy
    
    def _add_bubbles(self, n_to_add):
        """Add new bubbles to the simulation"""
        if n_to_add <= 0:
            return
        
        # Create new bubble positions below water line
        max_spawn_y = min(self.water_line_y * 0.3, self.viewport.height * 0.3)
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n_to_add),
            np.random.uniform(-20, max_spawn_y, n_to_add),
            np.random.uniform(0, 100, n_to_add)
        ])
        
        new_velocities = np.random.uniform(15, 40, n_to_add)
        
        depth_factors = 1.0 - (new_positions[:, 2] / 100.0)
        base_sizes = np.random.uniform(2.0, 8.0, n_to_add)
        new_sizes = base_sizes * (0.3 + 0.7 * depth_factors)
        
        new_wobble_phases = np.random.uniform(0, 2 * np.pi, n_to_add)
        new_wobble_speeds = np.random.uniform(1.0, 3.0, n_to_add)
        
        # Calculate squish factors based on y position
        y_normalized = new_positions[:, 1] / self.viewport.height
        new_squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        new_alphas = 0.15 + 0.25 * depth_factors  # Match initialization
        
        new_colors = np.column_stack([
            np.random.uniform(0.6, 0.9, n_to_add),
            np.random.uniform(0.8, 1.0, n_to_add),
            np.ones(n_to_add)
        ])
        
        # Concatenate with existing bubbles
        self.positions = np.vstack([self.positions, new_positions])
        self.velocities = np.concatenate([self.velocities, new_velocities])
        self.sizes = np.concatenate([self.sizes, new_sizes])
        self.base_sizes = np.concatenate([self.base_sizes, new_sizes])
        self.wobble_phases = np.concatenate([self.wobble_phases, new_wobble_phases])
        self.wobble_speeds = np.concatenate([self.wobble_speeds, new_wobble_speeds])
        self.squish_factors = np.concatenate([self.squish_factors, new_squish_factors])
        self.alphas = np.concatenate([self.alphas, new_alphas])
        self.colors = np.vstack([self.colors, new_colors])
        
        self.num_bubbles = len(self.positions)
        
    def compile_shader(self):
        """Compile vertex and fragment shaders for bubble rendering"""
        vertex_shader = """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Circle vertex position
        layout(location = 1) in vec3 instance_pos;  // Instance: x, y, z
        layout(location = 2) in float instance_size;
        layout(location = 3) in float instance_alpha;
        layout(location = 4) in vec3 instance_color;
        layout(location = 5) in float instance_squish;  // Width scaling factor
        
        out float fragAlpha;
        out vec3 fragColor;
        out vec2 fragCoord;
        
        uniform vec2 resolution;
        
        void main() {
            // Apply size with squish factor (horizontal scaling)
            vec2 scaled_pos = vec2(position.x * instance_squish, position.y) * instance_size;
            vec2 world_pos = scaled_pos + instance_pos.xy;
            
            // Convert to clip space (-1 to 1)
            vec2 clipPos = (world_pos / resolution) * 2.0 - 1.0;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            fragAlpha = instance_alpha;
            fragColor = instance_color;
            fragCoord = position;  // -1 to 1 for circle masking
        }
        """
        
        fragment_shader = """
        #version 310 es
        precision highp float;
        
        in float fragAlpha;
        in vec3 fragColor;
        in vec2 fragCoord;
        
        out vec4 FragColor;
        
        void main() {
            // Create circular bubble shape
            float dist = length(fragCoord);
            if (dist > 1.0) discard;
            
            // 3D sphere normal calculation
            // Assume bubble is a sphere, calculate z coordinate
            float z = sqrt(max(0.0, 1.0 - dist * dist));
            vec3 normal = normalize(vec3(fragCoord.x, fragCoord.y, z));
            
            // Light direction (from top-right)
            vec3 lightDir = normalize(vec3(0.5, 0.7, 1.0));
            
            // Diffuse lighting
            float diffuse = max(dot(normal, lightDir), 0.0);
            
            // Specular highlight (view from front)
            vec3 viewDir = vec3(0.0, 0.0, 1.0);
            vec3 reflectDir = reflect(-lightDir, normal);
            float specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            
            // Rim lighting for bubble edge
            float rim = pow(1.0 - z, 2.5);
            
            // Fresnel effect - edges are more reflective
            float fresnel = pow(1.0 - abs(dot(normal, viewDir)), 3.0);
            
            // Base color with subtle lighting
            vec3 color = fragColor * (0.3 + diffuse * 0.3);
            
            // Add specular highlights
            color += vec3(1.0) * specular * 0.8;
            
            // Add rim glow
            color += vec3(0.9, 0.95, 1.0) * rim * 0.5;
            
            // Add fresnel reflection
            color += vec3(1.0) * fresnel * 0.3;
            
            // Fade edges smoothly
            float edgeFade = 1.0 - smoothstep(0.85, 1.0, dist);
            float alpha = fragAlpha * edgeFade;
            
            FragColor = vec4(color, alpha);
        }
        """
        
        return shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
    
    def setup_buffers(self):
        """Set up VAO and VBOs for instanced rendering"""
        # Create circle mesh (will be instanced for each bubble)
        segments = 16
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        circle_vertices = np.column_stack([
            np.cos(angles),
            np.sin(angles)
        ]).astype(np.float32)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Circle vertex VBO (shared by all instances)
        circle_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, circle_VBO)
        glBufferData(GL_ARRAY_BUFFER, circle_vertices.nbytes, circle_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(circle_VBO)
        
        # Instance data VBO (position, size, alpha, color, squish)
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        
        # Allocate space (will be updated each frame)
        instance_data_size = self.num_bubbles * (3 + 1 + 1 + 3 + 1) * 4  # pos(3) + size(1) + alpha(1) + color(3) + squish(1)
        glBufferData(GL_ARRAY_BUFFER, instance_data_size, None, GL_DYNAMIC_DRAW)
        
        # Instance position (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        
        # Instance size (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(3 * 4))
        glVertexAttribDivisor(2, 1)
        
        # Instance alpha (location 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(4 * 4))
        glVertexAttribDivisor(3, 1)
        
        # Instance color (location 4)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(5 * 4))
        glVertexAttribDivisor(4, 1)
        
        # Instance squish factor (location 5)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(8 * 4))
        glVertexAttribDivisor(5, 1)
        
        self.VBOs.append(self.instance_VBO)

        glBindVertexArray(0)

        # Cache uniform locations so render() doesn't re-look them up each frame.
        self._uniform_resolution = glGetUniformLocation(self.shader, "resolution")

    def update(self, dt: float, state: Dict):
        """Update bubble positions and wobble"""
        if self.num_bubbles == 0:
            return
        
        # Add bubbles if we need more
        current_bubbles = len(self.positions)
        if self.target_bubbles > current_bubbles:
            n_to_add = self.target_bubbles - current_bubbles
            self._add_bubbles(n_to_add)
        
        # Update wobble phases
        self.wobble_phases += self.wobble_speeds * dt
        
        # Calculate horizontal wobble
        wobble_x = np.sin(self.wobble_phases) * 5.0
        
        # Update positions (rise + wobble + current drift)
        self.positions[:, 1] += self.velocities * dt  # Rise
        self.positions[:, 0] += wobble_x * dt  # Wobble horizontally
        # Horizontal current push. 25 px/s per unit of wind at wind = 1
        # visibly sweeps the bubble column without overwhelming the wobble.
        self.positions[:, 0] += self.wind * 25.0 * dt
        # Wrap around horizontally so drifting bubbles don't pile up on a
        # single edge and leave the other side empty.
        self.positions[:, 0] = np.mod(self.positions[:, 0], self.viewport.width)
        
        # Update squish factors based on current y position (bottom = 1.0, top = squish_top_width)
        y_normalized = self.positions[:, 1] / self.viewport.height
        self.squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Reset or remove bubbles that have reached the water line (beach boundary)
        above_water = self.positions[:, 1] > self.water_line_y
        self._reset_bubbles(above_water)
        
    def render(self, state: Dict):
        """Render all bubbles using instanced rendering"""
        super().render(state)
        
        if self.num_bubbles == 0:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        if self._uniform_resolution != -1:
            glUniform2f(self._uniform_resolution,
                        float(self.viewport.width), float(self.viewport.height))
        
        # Sort bubbles back-to-front by depth (z) for proper alpha blending
        sort_indices = np.argsort(-self.positions[:, 2])
        
        # Build instance data (sorted back-to-front)
        instance_data = np.column_stack([
            self.positions[sort_indices],    # x, y, z (3 floats)
            self.sizes[sort_indices],        # size (1 float)
            self.alphas[sort_indices],       # alpha (1 float)
            self.colors[sort_indices],       # r, g, b (3 floats)
            self.squish_factors[sort_indices]  # squish (1 float)
        ]).astype(np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        
        # Disable depth writes for proper transparency blending
        # Bubbles still READ from depth buffer but don't write to it
        glDepthMask(GL_FALSE)
        
        # Draw instanced
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 17, self.num_bubbles)
        
        # Restore depth writes for next effect
        glDepthMask(GL_TRUE)
        
        glBindVertexArray(0)
        glUseProgram(0)
