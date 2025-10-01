"""
Complete falling leaves effect - GPU-accelerated shader version
Instanced rendering with realistic leaf shapes, colors, and physics
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import sys
from pathlib import Path

# Add parent path for imports
ParentPath = Path(__file__).parent.parent
sys.path.insert(0, str(ParentPath))

from corefunctions.shader_effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Functions - Integrate with EventScheduler
# ============================================================================

def shader_falling_leaves(state, outstate, density=1.0):
    """
    Shader-based falling leaves effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_falling_leaves, density=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Leaf spawn rate multiplier
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
        print(f"Initializing falling leaves effect for frame {frame_id}")
        
        try:
            leaves_effect = viewport.add_effect(
                FallingLeavesEffect,
                density=density,
                max_leaves=25
            )
            state['leaves_effect'] = leaves_effect
            print(f"✓ Initialized shader falling leaves for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize falling leaves: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update density if it changes in global state
    if 'leaves_effect' in state:
        state['leaves_effect'].density = outstate.get('leaf_density', density)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'leaves_effect' in state:
            print(f"Cleaning up falling leaves effect for frame {frame_id}")
            viewport.effects.remove(state['leaves_effect'])
            state['leaves_effect'].cleanup()
            print(f"✓ Cleaned up shader falling leaves for frame {frame_id}")


def shader_secondary_falling_leaves(state, outstate, density=1.0):
    """
    Shader-based secondary (polar) falling leaves effect
    
    Usage:
        scheduler.schedule_event(0, 60, shader_secondary_falling_leaves, density=1.5, frame_id=1)
    """
    frame_id = state.get('frame_id', 1)
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
        print(f"Initializing secondary falling leaves effect for frame {frame_id}")
        
        try:
            leaves_effect = viewport.add_effect(
                SecondaryFallingLeavesEffect,
                density=density,
                max_leaves=40
            )
            state['leaves_effect'] = leaves_effect
            print(f"✓ Initialized shader secondary leaves for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize secondary leaves: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update density if it changes in global state
    if 'leaves_effect' in state:
        state['leaves_effect'].density = outstate.get('leaf_density', density)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'leaves_effect' in state:
            print(f"Cleaning up secondary leaves effect for frame {frame_id}")
            viewport.effects.remove(state['leaves_effect'])
            state['leaves_effect'].cleanup()
            print(f"✓ Cleaned up shader secondary leaves for frame {frame_id}")


# ============================================================================
# Main Falling Leaves Effect (Cartesian Coordinates)
# ============================================================================

class FallingLeavesEffect(ShaderEffect):
    """GPU-based falling leaves effect using instanced rendering"""
    
    def __init__(self, viewport, density: float = 1.0, max_leaves: int = 25):
        super().__init__(viewport)
        self.density = density
        self.max_leaves = max_leaves
        self.instance_VBO = None
        
        # Vectorized leaf data
        self.positions = np.zeros((0, 2), dtype=np.float32)  # [x, y]
        self.velocities = np.zeros((0, 2), dtype=np.float32)  # [vx, vy]
        self.sizes = np.zeros(0, dtype=np.float32)
        self.rotations = np.zeros(0, dtype=np.float32)
        self.rotation_speeds = np.zeros(0, dtype=np.float32)
        self.flutter_phases = np.zeros(0, dtype=np.float32)
        self.flutter_amplitudes = np.zeros(0, dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)  # [r, g, b] in RGB
        self.alphas = np.zeros(0, dtype=np.float32)
        self.lifetimes = np.zeros(0, dtype=np.float32)
        self.leaf_types = np.zeros(0, dtype=np.int32)  # NEW: leaf shape type
        
    def _spawn_leaves(self, count: int, season: float = 0.625):
        """Spawn new leaves at random positions"""
        if count <= 0:
            return
            
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, count),
            np.random.uniform(-5, 0, count)  # Start above screen
        ])
        
        new_velocities = np.column_stack([
            np.random.uniform(-0.5, 0.5, count),
            np.random.uniform(1.0, 2.0, count)
        ])
        
        new_sizes = np.random.uniform(2.0, 3.5, count)
        new_rotations = np.random.uniform(0, 2 * np.pi, count)
        new_rotation_speeds = np.random.uniform(-1.0, 1.0, count)
        new_flutter_phases = np.random.uniform(0, 2 * np.pi, count)
        new_flutter_amplitudes = np.random.uniform(0.5, 1.2, count)
        new_alphas = np.random.uniform(0.9, 1.0, count)
        new_lifetimes = np.ones(count)
        
        # NEW: Assign random leaf types (0-4 for 5 different shapes)
        new_leaf_types = np.random.randint(0, 5, count)
        
        # Generate colors based on season
        new_colors = self._generate_leaf_colors(count, season)
        
        # Concatenate with existing arrays
        self.positions = np.vstack([self.positions, new_positions]) if len(self.positions) > 0 else new_positions
        self.velocities = np.vstack([self.velocities, new_velocities]) if len(self.velocities) > 0 else new_velocities
        self.sizes = np.concatenate([self.sizes, new_sizes]) if len(self.sizes) > 0 else new_sizes
        self.rotations = np.concatenate([self.rotations, new_rotations]) if len(self.rotations) > 0 else new_rotations
        self.rotation_speeds = np.concatenate([self.rotation_speeds, new_rotation_speeds]) if len(self.rotation_speeds) > 0 else new_rotation_speeds
        self.flutter_phases = np.concatenate([self.flutter_phases, new_flutter_phases]) if len(self.flutter_phases) > 0 else new_flutter_phases
        self.flutter_amplitudes = np.concatenate([self.flutter_amplitudes, new_flutter_amplitudes]) if len(self.flutter_amplitudes) > 0 else new_flutter_amplitudes
        self.colors = np.vstack([self.colors, new_colors]) if len(self.colors) > 0 else new_colors
        self.alphas = np.concatenate([self.alphas, new_alphas]) if len(self.alphas) > 0 else new_alphas
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes]) if len(self.lifetimes) > 0 else new_lifetimes
        self.leaf_types = np.concatenate([self.leaf_types, new_leaf_types]) if len(self.leaf_types) > 0 else new_leaf_types


    
    def _generate_leaf_colors(self, count: int, season: float) -> np.ndarray:
        """Generate leaf colors based on season (RGB format)"""
        from skimage import color as skcolor
        
        colors_hsv = np.zeros((count, 3), dtype=np.float32)
        
        # Calculate distance from spring and fall
        spring_distance = min(abs(season - 0.125), 1 - abs(season - 0.125))
        fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
        
        spring_factor = max(0, 1 - spring_distance * 4)
        fall_factor = max(0, 1 - fall_distance * 4)
        
        if spring_factor > 0.5:
            # All green leaves in spring
            colors_hsv[:, 0] = np.random.uniform(0.25, 0.35, count)  # Green hue
            colors_hsv[:, 1] = np.random.uniform(0.7, 0.9, count)
            colors_hsv[:, 2] = np.random.uniform(0.3, 0.5, count)
        else:
            # Seasonal mix with fall colors
            color_types = np.random.random(count)
            
            # Red leaves
            red_proportion = 0.1 + 0.3 * fall_factor
            red_mask = color_types < red_proportion
            colors_hsv[red_mask, 0] = np.random.uniform(0.00, 0.05, np.sum(red_mask))
            colors_hsv[red_mask, 1] = np.random.uniform(0.8, 0.95, np.sum(red_mask))
            colors_hsv[red_mask, 2] = np.random.uniform(0.4, 0.6, np.sum(red_mask))
            
            # Orange leaves
            orange_proportion = red_proportion + (0.1 + 0.2 * fall_factor)
            orange_mask = (color_types >= red_proportion) & (color_types < orange_proportion)
            colors_hsv[orange_mask, 0] = np.random.uniform(0.05, 0.10, np.sum(orange_mask))
            colors_hsv[orange_mask, 1] = np.random.uniform(0.85, 0.95, np.sum(orange_mask))
            colors_hsv[orange_mask, 2] = np.random.uniform(0.45, 0.65, np.sum(orange_mask))
            
            # Yellow leaves
            yellow_proportion = orange_proportion + (0.2 + 0.1 * fall_factor)
            yellow_mask = (color_types >= orange_proportion) & (color_types < yellow_proportion)
            colors_hsv[yellow_mask, 0] = np.random.uniform(0.10, 0.15, np.sum(yellow_mask))
            colors_hsv[yellow_mask, 1] = np.random.uniform(0.8, 0.9, np.sum(yellow_mask))
            colors_hsv[yellow_mask, 2] = np.random.uniform(0.5, 0.7, np.sum(yellow_mask))
            
            # Brown leaves
            brown_proportion = yellow_proportion + (0.05 + 0.15 * fall_factor)
            brown_mask = (color_types >= yellow_proportion) & (color_types < brown_proportion)
            colors_hsv[brown_mask, 0] = np.random.uniform(0.07, 0.12, np.sum(brown_mask))
            colors_hsv[brown_mask, 1] = np.random.uniform(0.6, 0.8, np.sum(brown_mask))
            colors_hsv[brown_mask, 2] = np.random.uniform(0.3, 0.4, np.sum(brown_mask))
            
            # Green leaves
            green_mask = color_types >= brown_proportion
            colors_hsv[green_mask, 0] = np.random.uniform(0.25, 0.35, np.sum(green_mask))
            colors_hsv[green_mask, 1] = np.random.uniform(0.7, 0.9, np.sum(green_mask))
            colors_hsv[green_mask, 2] = np.random.uniform(0.3, 0.5, np.sum(green_mask))
        
        # Convert HSV to RGB
        colors_rgb = np.zeros_like(colors_hsv)
        for i in range(len(colors_hsv)):
            rgb = skcolor.hsv2rgb(colors_hsv[i:i+1].reshape(1, 1, 3))
            colors_rgb[i] = rgb.flatten()
        
        return colors_rgb
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec2 offset;    // Leaf position (x, y)
        layout(location = 2) in float size;     // Leaf size
        layout(location = 3) in float rotation; // Leaf rotation
        layout(location = 4) in vec4 color;     // Color (r, g, b, alpha)
        layout(location = 5) in float leafType; // Leaf shape type
        
        out vec4 fragColor;
        out vec2 fragPos;  // Position within quad (-1 to 1)
        flat out int fragLeafType;
        uniform vec2 resolution;
        
        void main() {
            fragPos = position;
            fragLeafType = int(leafType);
            
            // Apply rotation to quad
            float c = cos(rotation);
            float s = sin(rotation);
            vec2 rotated = vec2(
                position.x * c - position.y * s,
                position.x * s + position.y * c
            );
            
            // Scale by leaf size
            vec2 scaled = rotated * size * 3.0;
            
            // Translate to leaf position
            vec2 pos = scaled + offset;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            gl_Position = vec4(clipPos, 0.5, 1.0);
            fragColor = color;
        }
        """


    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;  // Position within quad (-1 to 1)
        flat in int fragLeafType;
        out vec4 outColor;
        
        // Leaf type 0: Oak-style (rounded lobes)
        float oak_leaf(float nx, float ny) {
            float dist = abs(ny * 0.5);
            float width = (1.0 - nx * nx) * 0.5;
            width *= smoothstep(-0.9, -0.3, nx);
            width *= smoothstep(0.95, 0.3, nx);
            
            // Add lobes
            float lobe = 0.1 * sin(nx * 12.0) * (1.0 - nx * nx);
            width += lobe;
            
            return step(dist, width);
        }
        
        // Leaf type 1: Maple-style (pointed lobes)
        float maple_leaf(float nx, float ny) {
            float angle = atan(ny, nx);
            float r = length(vec2(nx, ny));
            
            // Create 5 pointed lobes
            float lobes = 0.6 + 0.3 * cos(angle * 2.5);
            return step(r, lobes * 0.8);
        }
        
        // Leaf type 2: Willow-style (long and narrow)
        float willow_leaf(float nx, float ny) {
            float dist = abs(ny * 0.3);  // Very narrow
            float width = (1.0 - nx * nx * 0.8) * 0.3;
            width *= smoothstep(-0.95, -0.5, nx);
            width *= smoothstep(0.98, 0.5, nx);
            return step(dist, width);
        }
        
        // Leaf type 3: Birch-style (triangular with serrated edge)
        float birch_leaf(float nx, float ny) {
            float dist = abs(ny * 0.6);
            float width = (1.0 - nx) * 0.4;
            width *= smoothstep(-0.9, -0.2, nx);
            
            // Serrated edges
            float serration = 0.05 * sin(nx * 25.0);
            width += serration;
            
            return step(dist, width);
        }
        
        // Leaf type 4: Aspen-style (circular with small point)
        float aspen_leaf(float nx, float ny) {
            float r = length(vec2(nx * 1.2, ny));
            float width = 0.75;
            
            // Add point at tip
            if (nx > 0.5) {
                width *= smoothstep(0.95, 0.6, nx);
            }
            
            return step(r, width);
        }
        
        void main() {
            float nx = fragPos.x;
            float ny = fragPos.y;
            
            // Select leaf shape based on type
            float leaf_mask = 0.0;
            if (fragLeafType == 0) {
                leaf_mask = oak_leaf(nx, ny);
            } else if (fragLeafType == 1) {
                leaf_mask = maple_leaf(nx, ny);
            } else if (fragLeafType == 2) {
                leaf_mask = willow_leaf(nx, ny);
            } else if (fragLeafType == 3) {
                leaf_mask = birch_leaf(nx, ny);
            } else {
                leaf_mask = aspen_leaf(nx, ny);
            }
            
            if (leaf_mask < 0.5) {
                discard;
            }
            
            // Common vein structure
            float main_vein = smoothstep(0.02, 0.0, abs(ny * 0.5));
            
            float side_veins = 0.0;
            for (float i = -0.6; i <= 0.6; i += 0.15) {
                float vx = i;
                float vein_y = (nx - vx) * 0.4;
                float vein_dist = abs(ny * 0.5 - vein_y);
                float vein_fade = smoothstep(0.8, 0.0, abs(nx - vx)) * step(vx, nx);
                side_veins = max(side_veins, smoothstep(0.008, 0.0, vein_dist) * vein_fade);
            }
            
            float veins = max(main_vein, side_veins * 0.5);
            
            // Texture variation
            float color_var = fract(sin(dot(fragPos * 30.0, vec2(12.9898, 78.233))) * 43758.5453);
            color_var = (color_var - 0.5) * 0.08;
            
            // Soft edge
            float edge_dist = min(
                min(1.0 - abs(nx), 1.0 - abs(ny)),
                leaf_mask
            );
            float edge = smoothstep(0.0, 0.2, edge_dist);
            
            vec3 final_color = fragColor.rgb * (1.0 - veins * 0.35 + color_var);
            float alpha = fragColor.a * edge;
            
            outColor = vec4(final_color, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link leaf shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            # Set resolution uniform
            glUseProgram(shader)
            loc = glGetUniformLocation(shader, "resolution")
            if loc != -1:
                glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
            glUseProgram(0)
            
            return shader
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
        
        # Vertex buffer
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
        
        # Instance buffer
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        """Update leaf positions and properties"""
        if not self.enabled:
            return
        
        # Get environment parameters
        wind = state.get('wind', 0.0)
        whomp = state.get('whomp', 0.0)
        season = state.get('season', 0.625)
        
        # Calculate fall factor for spawn rate
        fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
        fall_factor = 1 - 1.9 * fall_distance
        
        # Spawn new leaves
        if len(self.positions) < self.max_leaves:
            leaf_rate = (0.25 + 0.2 * abs(wind) + 0.3 * whomp) * self.density * fall_factor
            if np.random.random() < leaf_rate:
                spawn_count = min(np.random.randint(1, 3), self.max_leaves - len(self.positions))
                if spawn_count > 0:
                    self._spawn_leaves(spawn_count, season)
        
        if len(self.positions) == 0:
            return
        
        # Update flutter phase
        self.flutter_phases += 0.1 * dt * 60
        
        # Calculate flutter effect
        flutter_x = np.sin(self.flutter_phases) * self.flutter_amplitudes
        
        # Update velocities
        movement_multiplier = 1.0 + whomp * 12.0
        self.velocities[:, 0] = flutter_x + wind * 5
        
        # Update positions
        self.positions += self.velocities * dt * 5 * movement_multiplier
        self.positions[:, 1] *= (1.0 - whomp * 1.5 * dt * 5)
        
        # Update rotations
        self.rotations += self.rotation_speeds * dt * 2
        
        # Decrease lifetimes
        self.lifetimes -= 0.001 * dt * 60
        
        # NEW: Wrap horizontally when leaves exit left or right
        max_leaf_size = 3.5 * 3.0  # max size * scale factor
        margin = max_leaf_size + 5
        
        # Wrap from right to left
        wrap_right_mask = self.positions[:, 0] > self.viewport.width + margin
        self.positions[wrap_right_mask, 0] -= (self.viewport.width + 2 * margin)
        
        # Wrap from left to right
        wrap_left_mask = self.positions[:, 0] < -margin
        self.positions[wrap_left_mask, 0] += (self.viewport.width + 2 * margin)
        
        # Filter out-of-bounds leaves - only remove if below screen or lifetime expired
        valid_mask = (
            (self.positions[:, 1] < self.viewport.height + margin) & 
            (self.lifetimes > 0)
        )
        
        if not np.all(valid_mask):
            self.positions = self.positions[valid_mask]
            self.velocities = self.velocities[valid_mask]
            self.sizes = self.sizes[valid_mask]
            self.rotations = self.rotations[valid_mask]
            self.rotation_speeds = self.rotation_speeds[valid_mask]
            self.flutter_phases = self.flutter_phases[valid_mask]
            self.flutter_amplitudes = self.flutter_amplitudes[valid_mask]
            self.colors = self.colors[valid_mask]
            self.alphas = self.alphas[valid_mask]
            self.lifetimes = self.lifetimes[valid_mask]
            self.leaf_types = self.leaf_types[valid_mask]

    def render(self, state: Dict):
        """Render all leaves using instancing"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        # Build instance data
        instance_data = np.hstack([
            self.positions,
            self.sizes[:, np.newaxis],
            self.rotations[:, np.newaxis],
            self.colors,
            self.alphas[:, np.newaxis]
        ]).astype(np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 8 * 4  # 8 floats per instance
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Render
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(self.positions))
        
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)

