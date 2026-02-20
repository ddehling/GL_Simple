"""
Fish swimming effect - marine life swimming across the scene
GPU-accelerated fish system with schooling behavior and depth
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

def shader_fish(state, outstate, activity=1.0):
    """
    Shader-based fish swimming effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_fish, activity=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        activity: Marine life activity multiplier
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
    
    # Initialize fish effect on first call
    if state['count'] == 0:
        # Start with base number (will adjust dynamically)
        num_fish = 30
        print(f"Initializing fish effect for frame {frame_id} (base: {num_fish} fish)")
        
        try:
            fish_effect = viewport.add_effect(
                FishEffect,
                num_fish=num_fish,
                activity=activity,
                squish_top_width=squish_top_width
            )
            state['fish_effect'] = fish_effect
            print(f"✓ Initialized shader fish for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fish: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update activity every frame
    if 'fish_effect' in state:
        marine_life_activity = outstate.get('marine_life_activity', activity)
        state['fish_effect'].set_target_activity(marine_life_activity)
        
        # Update squish width from scale
        state['fish_effect'].squish_top_width = squish_top_width
    
    # On close event, clean up
    if state['count'] == -1:
        if 'fish_effect' in state:
            print(f"Cleaning up fish effect for frame {frame_id}")
            viewport.effects.remove(state['fish_effect'])
            state['fish_effect'].cleanup()
            print(f"✓ Cleaned up shader fish for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

# Fish species definitions
FISH_SPECIES = {
    # Common schooling fish - small, fast, travel in groups
    'silverside': {
        'rarity': 0.35,  # 35% of fish
        'size_range': (4.0, 8.0),
        'speed_range': (25, 35),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.7, 0.9, n),  # Silver-white
            np.random.uniform(0.7, 0.9, n),
            np.random.uniform(0.8, 1.0, n)
        ]),
        'schooling': True,
        'school_distance': 40.0,
        'school_strength': 0.5
    },
    
    # Common tropical fish - medium size, colorful
    'tang': {
        'rarity': 0.25,  # 25% of fish
        'size_range': (8.0, 14.0),
        'speed_range': (12, 20),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.2, 0.4, n),  # Blue
            np.random.uniform(0.5, 0.7, n),
            np.random.uniform(0.8, 1.0, n)
        ]),
        'schooling': False,
        'school_distance': 0.0,
        'school_strength': 0.0
    },
    
    # Clownfish - small/medium, orange, independent
    'clownfish': {
        'rarity': 0.15,  # 15% of fish
        'size_range': (6.0, 10.0),
        'speed_range': (10, 16),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.9, 1.0, n),  # Orange
            np.random.uniform(0.4, 0.6, n),
            np.random.uniform(0.1, 0.3, n)
        ]),
        'schooling': False,
        'school_distance': 0.0,
        'school_strength': 0.0
    },
    
    # Damselfish - small, fast, some schooling
    'damselfish': {
        'rarity': 0.12,  # 12% of fish
        'size_range': (5.0, 9.0),
        'speed_range': (16, 28),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.3, 0.5, n),  # Teal/green
            np.random.uniform(0.6, 0.8, n),
            np.random.uniform(0.5, 0.7, n)
        ]),
        'schooling': True,
        'school_distance': 35.0,
        'school_strength': 0.45
    },
    
    # Grouper - large, slow, solitary
    'grouper': {
        'rarity': 0.08,  # 8% of fish (less common)
        'size_range': (16.0, 28.0),
        'speed_range': (6, 10),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.4, 0.6, n),  # Brown/grey
            np.random.uniform(0.4, 0.6, n),
            np.random.uniform(0.3, 0.5, n)
        ]),
        'schooling': False,
        'school_distance': 0.0,
        'school_strength': 0.0
    },
    
    # Angelfish - rare, medium, slow, beautiful
    'angelfish': {
        'rarity': 0.05,  # 5% of fish (rare)
        'size_range': (12.0, 18.0),
        'speed_range': (8, 14),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.8, 1.0, n),  # Yellow/gold
            np.random.uniform(0.7, 0.9, n),
            np.random.uniform(0.2, 0.4, n)
        ]),
        'schooling': False,
        'school_distance': 0.0,
        'school_strength': 0.0
    }
}

class FishEffect(ShaderEffect):
    """GPU-based fish swimming effect using instanced rendering"""
    
    def __init__(self, viewport, num_fish: int = 30, activity: float = 1.0, squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.num_fish = num_fish
        self.base_num_fish = num_fish
        self.target_fish = num_fish
        self.activity = activity
        self.instance_VBO = None
        
        # Width scaling based on vertical position
        self.squish_top_width = squish_top_width
        
        # Vectorized fish data
        self.positions = None
        self.velocities = None  # (vx, vy) velocity vectors
        self.sizes = None
        self.base_sizes = None
        self.base_speeds = None  # Base swimming speed per fish
        self.directions = None  # 1 = right, -1 = left
        self.swim_phases = None  # Animation phase
        self.swim_speeds = None  # Speed of tail animation
        self.squish_factors = None
        self.alphas = None
        self.colors = None
        self.species = None  # Species index for each fish
        self.species_names = list(FISH_SPECIES.keys())
        
        self._initialize_fish()
        
    def _initialize_fish(self):
        """Initialize all fish data as numpy arrays with species variety"""
        n = self.num_fish
        
        # Assign species based on rarity weights
        rarities = [FISH_SPECIES[sp]['rarity'] for sp in self.species_names]
        self.species = np.random.choice(len(self.species_names), size=n, p=rarities)
        
        # Positions: x, y, z (spawn in bottom 65% - y:0 is top, y:height is bottom)
        self.positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n),  # x
            np.random.uniform(0, self.viewport.height * 0.65, n),  # y (bottom 65%: 0 to 0.65*height)
            np.random.uniform(0, 100, n)  # z (depth)
        ])
        
        # Initialize arrays
        self.base_speeds = np.zeros(n)
        self.directions = np.random.choice([-1, 1], n)
        self.sizes = np.zeros(n)
        self.colors = np.zeros((n, 3))
        
        # Set attributes based on species
        for species_idx, species_name in enumerate(self.species_names):
            mask = self.species == species_idx
            n_species = np.sum(mask)
            if n_species == 0:
                continue
            
            spec = FISH_SPECIES[species_name]
            
            # Size based on species
            self.sizes[mask] = np.random.uniform(spec['size_range'][0], spec['size_range'][1], n_species)
            
            # Speed based on species
            self.base_speeds[mask] = np.random.uniform(spec['speed_range'][0], spec['speed_range'][1], n_species)
            
            # Color based on species
            self.colors[mask] = spec['color'](n_species)
        
        # Adjust size by depth
        depth_factors = 1.0 - (self.positions[:, 2] / 100.0)  # 1.0 (near) to 0.0 (far)
        self.sizes = self.sizes * (0.3 + 0.7 * depth_factors)
        self.base_sizes = self.sizes.copy()
        
        # Velocities based on species speed and direction
        self.velocities = np.column_stack([
            self.base_speeds * self.directions,  # vx
            np.random.uniform(-5, 5, n)  # vy (slight vertical drift)
        ])
        
        # Swim animation - faster tail for faster fish
        self.swim_phases = np.random.uniform(0, 2 * np.pi, n)
        self.swim_speeds = 8.0 + (self.base_speeds / 10.0)  # Tail speed scales with swim speed
        
        # Calculate squish factors based on y position
        y_normalized = self.positions[:, 1] / self.viewport.height
        self.squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Alpha based on depth
        self.alphas = 0.4 + 0.5 * depth_factors  # Range: 0.4-0.9
        
    def _reset_fish(self, mask):
        """Reset fish that have swum off-screen"""
        n_reset = np.sum(mask)
        if n_reset == 0:
            return
        
        # Check if we're over the target and need to remove some fish
        current_count = len(self.positions)
        if current_count > self.target_fish:
            n_to_remove = min(n_reset, current_count - self.target_fish)
            
            reset_indices = np.where(mask)[0]
            remove_indices = reset_indices[:n_to_remove]
            keep_mask = np.ones(current_count, dtype=bool)
            keep_mask[remove_indices] = False
            
            self.positions = self.positions[keep_mask]
            self.velocities = self.velocities[keep_mask]
            self.sizes = self.sizes[keep_mask]
            self.base_sizes = self.base_sizes[keep_mask]
            self.base_speeds = self.base_speeds[keep_mask]
            self.directions = self.directions[keep_mask]
            self.swim_phases = self.swim_phases[keep_mask]
            self.swim_speeds = self.swim_speeds[keep_mask]
            self.squish_factors = self.squish_factors[keep_mask]
            self.alphas = self.alphas[keep_mask]
            self.colors = self.colors[keep_mask]
            self.species = self.species[keep_mask]
            
            self.num_fish = len(self.positions)
            
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
        
        # Reset remaining fish on opposite side
        if n_reset > 0:
            # Get species info for fish being reset
            reset_species = self.species[mask]
            
            # Fish swimming right start on left, fish swimming left start on right
            new_x = np.where(self.directions[mask] > 0, -20, self.viewport.width + 20)
            self.positions[mask, 0] = new_x
            self.positions[mask, 1] = np.random.uniform(0, self.viewport.height * 0.65, n_reset)
            self.positions[mask, 2] = np.random.uniform(0, 100, n_reset)
            
            # Reset attributes based on species
            for species_idx, species_name in enumerate(self.species_names):
                species_mask = reset_species == species_idx
                n_species = np.sum(species_mask)
                if n_species == 0:
                    continue
                
                spec = FISH_SPECIES[species_name]
                
                # Get indices in the full array
                full_indices = np.where(mask)[0][species_mask]
                
                # Reset speed based on species
                self.base_speeds[full_indices] = np.random.uniform(
                    spec['speed_range'][0], spec['speed_range'][1], n_species
                )
                self.velocities[full_indices, 0] = self.base_speeds[full_indices] * self.directions[full_indices]
                self.velocities[full_indices, 1] = np.random.uniform(-5, 5, n_species)
                
                # Reset size based on species and depth
                depth_factors = 1.0 - (self.positions[full_indices, 2] / 100.0)
                base_sizes = np.random.uniform(spec['size_range'][0], spec['size_range'][1], n_species)
                self.sizes[full_indices] = base_sizes * (0.3 + 0.7 * depth_factors)
                self.base_sizes[full_indices] = self.sizes[full_indices]
                
                # Reset color based on species
                self.colors[full_indices] = spec['color'](n_species)
                
                # Reset alpha
                self.alphas[full_indices] = 0.4 + 0.5 * depth_factors
            
            # Reset animation for all reset fish
            self.swim_phases[mask] = np.random.uniform(0, 2 * np.pi, n_reset)
            self.swim_speeds[mask] = 8.0 + (self.base_speeds[mask] / 10.0)
            
    def set_target_activity(self, activity):
        """Set target marine life activity and adjust fish count"""
        self.activity = activity
        self.target_fish = int(self.base_num_fish * activity)
    
    def _add_fish(self, n_to_add):
        """Add new fish to the simulation"""
        if n_to_add <= 0:
            return
        
        # Create new fish (spawn in bottom 65% - y:0 to 0.65*height)
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n_to_add),
            np.random.uniform(0, self.viewport.height * 0.65, n_to_add),
            np.random.uniform(0, 100, n_to_add)
        ])
        
        # Assign species based on rarity
        rarities = [FISH_SPECIES[sp]['rarity'] for sp in self.species_names]
        new_species = np.random.choice(len(self.species_names), size=n_to_add, p=rarities)
        
        # Initialize arrays
        new_base_speeds = np.zeros(n_to_add)
        new_directions = np.random.choice([-1, 1], n_to_add)
        new_sizes = np.zeros(n_to_add)
        new_colors = np.zeros((n_to_add, 3))
        
        # Set attributes based on species
        for species_idx, species_name in enumerate(self.species_names):
            mask = new_species == species_idx
            n_species = np.sum(mask)
            if n_species == 0:
                continue
            
            spec = FISH_SPECIES[species_name]
            
            # Size based on species
            new_sizes[mask] = np.random.uniform(spec['size_range'][0], spec['size_range'][1], n_species)
            
            # Speed based on species
            new_base_speeds[mask] = np.random.uniform(spec['speed_range'][0], spec['speed_range'][1], n_species)
            
            # Color based on species
            new_colors[mask] = spec['color'](n_species)
        
        # Adjust size by depth
        depth_factors = 1.0 - (new_positions[:, 2] / 100.0)
        new_sizes = new_sizes * (0.3 + 0.7 * depth_factors)
        
        # Velocities
        new_velocities = np.column_stack([
            new_base_speeds * new_directions,
            np.random.uniform(-5, 5, n_to_add)
        ])
        
        # Animation
        new_swim_phases = np.random.uniform(0, 2 * np.pi, n_to_add)
        new_swim_speeds = 8.0 + (new_base_speeds / 10.0)
        
        # Squish factors
        y_normalized = new_positions[:, 1] / self.viewport.height
        new_squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Alphas
        new_alphas = 0.4 + 0.5 * depth_factors
        
        # Concatenate
        self.positions = np.vstack([self.positions, new_positions])
        self.velocities = np.vstack([self.velocities, new_velocities])
        self.sizes = np.concatenate([self.sizes, new_sizes])
        self.base_sizes = np.concatenate([self.base_sizes, new_sizes])
        self.base_speeds = np.concatenate([self.base_speeds, new_base_speeds])
        self.directions = np.concatenate([self.directions, new_directions])
        self.swim_phases = np.concatenate([self.swim_phases, new_swim_phases])
        self.swim_speeds = np.concatenate([self.swim_speeds, new_swim_speeds])
        self.squish_factors = np.concatenate([self.squish_factors, new_squish_factors])
        self.alphas = np.concatenate([self.alphas, new_alphas])
        self.colors = np.vstack([self.colors, new_colors])
        self.species = np.concatenate([self.species, new_species])
        
        self.num_fish = len(self.positions)
    
    def compile_shader(self):
        """Compile vertex and fragment shaders for fish rendering"""
        vertex_shader = """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Fish shape vertices
        layout(location = 1) in vec3 instance_pos;  // Instance: x, y, z
        layout(location = 2) in float instance_size;
        layout(location = 3) in float instance_alpha;
        layout(location = 4) in vec3 instance_color;
        layout(location = 5) in float instance_squish;
        layout(location = 6) in float instance_direction;  // 1 or -1
        layout(location = 7) in float instance_swim_phase;  // Tail animation
        
        out float fragAlpha;
        out vec3 fragColor;
        out vec2 fragCoord;
        
        uniform vec2 resolution;
        
        void main() {
            // Apply tail animation (wiggle the back of the fish)
            float tail_wiggle = sin(instance_swim_phase) * 0.15;
            vec2 animated_pos = position;
            // Only wiggle the tail (negative x values)
            if (position.x < 0.0) {
                animated_pos.y += tail_wiggle * abs(position.x);
            }
            
            // Flip fish based on direction
            vec2 flipped_pos = vec2(animated_pos.x * instance_direction, animated_pos.y);
            
            // Apply size with squish factor
            vec2 scaled_pos = vec2(flipped_pos.x * instance_squish, flipped_pos.y) * instance_size;
            vec2 world_pos = scaled_pos + instance_pos.xy;
            
            // Convert to clip space
            vec2 clipPos = (world_pos / resolution) * 2.0 - 1.0;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            fragAlpha = instance_alpha;
            fragColor = instance_color;
            fragCoord = position;
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
            // Check if we're in tail region (x < -0.6)
            bool inTail = fragCoord.x < -0.6;
            
            // Fish body shape (elongated oval) - only check body region
            float dist = length(vec2(fragCoord.x * 1.5, fragCoord.y));
            if (!inTail && dist > 1.0) discard;
            
            // For tail, use different bounds check
            if (inTail) {
                // Allow tail to extend further back with wider spread
                float tailDist = abs(fragCoord.y);
                if (tailDist > 0.6) discard;  // Tail can be wider
            }
            
            // Start with base color
            vec3 color = fragColor;
            
            // Dorsal stripe (darker stripe along the back)
            float dorsalStripe = smoothstep(0.15, 0.2, fragCoord.y) * smoothstep(0.35, 0.3, fragCoord.y);
            color = mix(color, color * 0.6, dorsalStripe * 0.7);
            
            // Side stripes (vertical bands)
            float stripePattern = sin(fragCoord.x * 8.0) * 0.5 + 0.5;
            float stripeMask = smoothstep(0.3, 0.4, stripePattern) * smoothstep(0.7, 0.6, stripePattern);
            color = mix(color, color * 0.75, stripeMask * 0.4);
            
            // Belly highlight (lighter underside)
            float bellyGradient = smoothstep(0.0, -0.4, fragCoord.y);
            color = mix(color, color * 1.3, bellyGradient * 0.5);
            
            // Centerline highlight (realistic fish sheen)
            float centerHighlight = 1.0 - abs(fragCoord.y) * 2.0;
            centerHighlight = smoothstep(0.3, 0.8, centerHighlight);
            color += vec3(0.15, 0.15, 0.2) * centerHighlight * 0.3;
            
            // Eye spot (small dark circle on front)
            vec2 eyePos = vec2(0.5, 0.15);
            float eyeDist = length(fragCoord - eyePos);
            float eye = smoothstep(0.08, 0.05, eyeDist);
            color = mix(color, vec3(0.1, 0.1, 0.15), eye);
            
            // Eye highlight (tiny white dot)
            float eyeGlint = smoothstep(0.03, 0.01, length(fragCoord - (eyePos + vec2(-0.015, 0.015))));
            color = mix(color, vec3(1.0, 1.0, 1.0), eyeGlint * 0.9);
            
            // Tail fin markings (darker at the tail)
            if (fragCoord.x < -0.3) {
                float tailPattern = sin(fragCoord.y * 12.0) * 0.5 + 0.5;
                float tailMask = smoothstep(0.4, 0.6, tailPattern);
                float tailIntensity = smoothstep(-0.3, -0.8, fragCoord.x);
                color = mix(color, color * 0.7, tailMask * tailIntensity * 0.5);
            }
            
            // Overall body gradient (darker at edges)
            float bodyGradient = 1.0 - dist * 0.3;
            color *= bodyGradient;
            
            // Fade edges for smooth appearance
            float alpha = fragAlpha * (1.0 - smoothstep(0.85, 1.0, dist));
            
            FragColor = vec4(color, alpha);
        }
        """
        
        return shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
    
    def setup_buffers(self):
        """Set up VAO and VBOs for instanced rendering"""
        # For TRIANGLE_FAN, first vertex is the center, all others radiate from it
        # We'll create the body as a fan from center, then add tail separately
        
        # Center point of fish body
        center = np.array([[0.0, 0.0]])
        
        # Create body outline - full ellipse around the body
        body_segments = 16
        # Go around the body, but leave the back open
        angles = np.linspace(-np.pi * 0.7, np.pi * 0.7, body_segments)
        
        x_coords = np.cos(angles)
        y_coords = np.sin(angles) * 0.5  # Flatten vertically
        # Taper toward back (negative x)
        x_coords = x_coords * (0.6 + 0.4 * (1 + x_coords) / 2)
        
        body_outline = np.column_stack([x_coords, y_coords])
        
        # Tail vertices - creates the forked tail
        tail_vertices = np.array([
            [-0.6, 0.25],      # Top connection to body
            [-0.9, 0.5],       # Upper tail spread
            [-1.1, 0.55],      # Upper tail tip
            [-0.9, 0.0],       # Center of tail fork
            [-1.1, -0.55],     # Lower tail tip
            [-0.9, -0.5],      # Lower tail spread
            [-0.6, -0.25],     # Bottom connection to body
        ])
        
        # Combine: center + body + tail
        # TRIANGLE_FAN will draw from center to each successive vertex
        fish_vertices = np.vstack([center, body_outline, tail_vertices]).astype(np.float32)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Fish shape VBO
        fish_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, fish_VBO)
        glBufferData(GL_ARRAY_BUFFER, fish_vertices.nbytes, fish_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(fish_VBO)
        
        # Instance data VBO
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        
        # Allocate space: pos(3) + size(1) + alpha(1) + color(3) + squish(1) + direction(1) + swim_phase(1) = 11 floats
        instance_data_size = self.num_fish * 11 * 4
        glBufferData(GL_ARRAY_BUFFER, instance_data_size, None, GL_DYNAMIC_DRAW)
        
        # Instance position (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        
        # Instance size (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(3 * 4))
        glVertexAttribDivisor(2, 1)
        
        # Instance alpha (location 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(4 * 4))
        glVertexAttribDivisor(3, 1)
        
        # Instance color (location 4)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(5 * 4))
        glVertexAttribDivisor(4, 1)
        
        # Instance squish (location 5)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(8 * 4))
        glVertexAttribDivisor(5, 1)
        
        # Instance direction (location 6)
        glEnableVertexAttribArray(6)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(9 * 4))
        glVertexAttribDivisor(6, 1)
        
        # Instance swim phase (location 7)
        glEnableVertexAttribArray(7)
        glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, 11 * 4, ctypes.c_void_p(10 * 4))
        glVertexAttribDivisor(7, 1)
        
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update fish positions and animation"""
        if self.num_fish == 0:
            return
        
        # Add fish if we need more
        current_fish = len(self.positions)
        if self.target_fish > current_fish:
            n_to_add = self.target_fish - current_fish
            self._add_fish(n_to_add)
        
        # Update swim animation
        self.swim_phases += self.swim_speeds * dt
        
        # Apply schooling behavior for species that school
        for species_idx, species_name in enumerate(self.species_names):
            spec = FISH_SPECIES[species_name]
            if not spec['schooling']:
                continue
            
            mask = self.species == species_idx
            if np.sum(mask) < 2:
                continue  # Need at least 2 fish to school
            
            # Get positions and velocities of this species
            species_positions = self.positions[mask]
            species_velocities = self.velocities[mask]
            species_indices = np.where(mask)[0]
            
            # For each fish of this species, find nearby school mates
            school_distance = spec['school_distance']
            school_strength = spec['school_strength']
            
            for i, idx in enumerate(species_indices):
                # Calculate distances to other fish of same species
                pos = self.positions[idx]
                other_positions = np.delete(species_positions, i, axis=0)
                other_velocities = np.delete(species_velocities, i, axis=0)
                
                if len(other_positions) == 0:
                    continue
                
                # Calculate distances
                deltas = other_positions - pos
                distances = np.linalg.norm(deltas[:, :2], axis=1)  # Only x, y distance
                
                # Find nearby fish within school distance
                nearby = distances < school_distance
                if np.any(nearby):
                    # Align velocity with nearby school mates (not position attraction!)
                    nearby_velocities = other_velocities[nearby]
                    avg_velocity = np.mean(nearby_velocities[:, :2], axis=0)
                    
                    # Blend current velocity with school average velocity
                    self.velocities[idx, 0] = (self.velocities[idx, 0] * (1.0 - school_strength) + 
                                               avg_velocity[0] * school_strength)
                    self.velocities[idx, 1] = (self.velocities[idx, 1] * (1.0 - school_strength) + 
                                               avg_velocity[1] * school_strength)
                    
                    # Maintain speed based on species
                    speed = np.linalg.norm(self.velocities[idx])
                    if speed > 0:
                        target_speed = self.base_speeds[idx]
                        self.velocities[idx] = self.velocities[idx] / speed * target_speed
        
        # Random direction/speed changes for natural movement
        # Each fish has a small chance to adjust course
        change_probability = dt * 0.3  # About 30% chance per second
        change_mask = np.random.random(len(self.positions)) < change_probability
        
        if np.any(change_mask):
            # Add random velocity adjustments
            speed_variation = np.random.uniform(-0.2, 0.2, np.sum(change_mask))
            angle_variation = np.random.uniform(-0.5, 0.5, np.sum(change_mask))
            
            # Adjust velocities
            current_vx = self.velocities[change_mask, 0]
            current_vy = self.velocities[change_mask, 1]
            
            # Add angular change
            cos_angle = np.cos(angle_variation)
            sin_angle = np.sin(angle_variation)
            new_vx = current_vx * cos_angle - current_vy * sin_angle
            new_vy = current_vx * sin_angle + current_vy * cos_angle
            
            # Add speed variation
            self.velocities[change_mask, 0] = new_vx * (1.0 + speed_variation)
            self.velocities[change_mask, 1] = new_vy * (1.0 + speed_variation)
            
            # Normalize to maintain species speed
            speeds = np.linalg.norm(self.velocities[change_mask], axis=1)
            speed_mask = speeds > 0
            if np.any(speed_mask):
                local_indices = np.where(change_mask)[0][speed_mask]
                target_speeds = self.base_speeds[local_indices]
                self.velocities[local_indices] = (
                    self.velocities[local_indices] / speeds[speed_mask, np.newaxis] * target_speeds[:, np.newaxis]
                )
        
        # Update positions
        self.positions[:, 0] += self.velocities[:, 0] * dt
        self.positions[:, 1] += self.velocities[:, 1] * dt
        
        # Keep fish from going below 70% height (clamp y to max 0.7*height)
        max_y = self.viewport.height * 0.7
        too_deep = self.positions[:, 1] > max_y
        if np.any(too_deep):
            self.positions[too_deep, 1] = max_y
            # Reverse vertical velocity if they hit the boundary
            self.velocities[too_deep, 1] = -np.abs(self.velocities[too_deep, 1])
        
        # Update squish factors based on y position
        y_normalized = self.positions[:, 1] / self.viewport.height
        self.squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Reset fish that have swum off-screen
        off_screen = ((self.directions > 0) & (self.positions[:, 0] > self.viewport.width + 20)) | \
                     ((self.directions < 0) & (self.positions[:, 0] < -20))
        self._reset_fish(off_screen)
        
    def render(self, state: Dict):
        """Render all fish using instanced rendering"""
        super().render(state)
        
        if self.num_fish == 0:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        # Sort fish back-to-front by depth
        sort_indices = np.argsort(-self.positions[:, 2])
        
        # Build instance data (sorted back-to-front)
        instance_data = np.column_stack([
            self.positions[sort_indices],
            self.sizes[sort_indices],
            self.alphas[sort_indices],
            self.colors[sort_indices],
            self.squish_factors[sort_indices],
            self.directions[sort_indices],
            self.swim_phases[sort_indices]
        ]).astype(np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        
        # Disable depth writes for transparency
        glDepthMask(GL_FALSE)
        
        # Draw instanced (1 center + 16 body + 7 tail = 24 vertices)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 24, self.num_fish)
        
        # Restore depth writes
        glDepthMask(GL_TRUE)
        
        glBindVertexArray(0)
        glUseProgram(0)
