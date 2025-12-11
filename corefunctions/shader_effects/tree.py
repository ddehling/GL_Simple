"""
Tree shader effect - Procedural branching tree with leaves
Features recursive branching structure with seasonal leaf colors
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
import sys
from pathlib import Path

# Add parent path for imports
ParentPath = Path(__file__).parent.parent
sys.path.insert(0, str(ParentPath))

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_tree(state, outstate, x_position=0.5, scale=1.0, fade_duration=5.0):
    """
    Shader-based tree effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_tree, x_position=0.5, scale=1.2, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        x_position: Horizontal position (0-1, where 0.5 is center)
        scale: Tree size multiplier (default 1.0)
        fade_duration: Duration of fade in/out in seconds (default 5.0)
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
        print(f"Initializing tree effect for frame {frame_id}")
        
        try:
            tree_effect = viewport.add_effect(
                TreeEffect,
                x_position=x_position,
                scale=scale
            )
            state['tree_effect'] = tree_effect
            print(f"✓ Initialized shader tree for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize tree: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters
    if 'tree_effect' in state:
        # Get season from global state
        season = outstate.get('season', 0.625)
        state['tree_effect'].season = season
        
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
        
        state['tree_effect'].fade_factor = np.clip(fade_factor, 0, 1)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'tree_effect' in state:
            print(f"Cleaning up tree effect for frame {frame_id}")
            viewport.effects.remove(state['tree_effect'])
            state['tree_effect'].cleanup()
            print(f"✓ Cleaned up shader tree for frame {frame_id}")


# ============================================================================
# Tree Effect Class
# ============================================================================

class TreeEffect(ShaderEffect):
    """GPU-based tree with procedural branches and leaves"""
    
    def __init__(self, viewport, x_position: float = 0.5, scale: float = 1.0):
        super().__init__(viewport)
        self.x_position = x_position
        self.scale = scale
        self.season = 0.625  # Fall by default
        self.fade_factor = 0.0
        self.sway_time = 0.0  # For animation
        self.growth_time = 0.0  # For growth animation
        self.growth_duration = 15.0  # Seconds to fully grow
        
        self.branch_VBO = None
        self.leaf_VBO = None
        
        # Generate tree structure (with growth timestamps)
        self._generate_tree()
    
    def _generate_tree(self):
        """Generate recursive branch structure with leaves (upward perspective)"""
        # Tree parameters - branches originate from bottom, spread upward
        bottom_y = self.viewport.height  # Bottom of screen
        
        # Storage for all branches (start_x, start_y, end_x, end_y, start_width, end_width, depth, growth_start)
        self.branches = []
        
        # Storage for all leaves (x, y, size, rotation, color_r, color_g, color_b, leaf_type, depth)
        self.leaves = []
        
        # Generate multiple main branches from bottom
        num_main_branches = np.random.randint(4, 7)  # Fewer main branches
        
        for i in range(num_main_branches):
            # Branches start at random x positions along bottom
            start_x = np.random.uniform(0, self.viewport.width)
            
            # Angle pointing generally upward (-60 to -120 degrees, with -90 being straight up)
            angle = np.random.uniform(-2.4, -0.7)  # Radians
            
            # Main branch length - reach about 90% of viewport height
            branch_length = np.random.uniform(0.75, 0.9) * self.viewport.height * self.scale
            
            # End position
            end_x = start_x + np.cos(angle) * branch_length
            end_y = bottom_y + np.sin(angle) * branch_length
            
            # Main branch width (tapers from base to tip)
            branch_start_width = np.random.uniform(4, 7) * self.scale
            branch_end_width = branch_start_width * 0.6  # Tapers to 60%
            
            # Depth varies per branch
            branch_depth = np.random.uniform(40, 60)
            
            # Generate this main branch and its sub-branches
            self._generate_branch(
                start_x, bottom_y,  # Start at bottom
                end_x, end_y,  # End position
                branch_start_width,  # Start width
                branch_end_width,  # End width
                0,  # Current depth/generation
                3,  # Max depth
                branch_depth,  # Depth coordinate (z)
                0.0  # Growth start time
            )
        
        # Convert to numpy arrays
        self.branches = np.array(self.branches, dtype=np.float32)
        self.leaves = np.array(self.leaves, dtype=np.float32)
        
        # Horizontal wrapping margin (larger than largest leaf)
        self.wrap_margin = 30
        
        print(f"Generated tree with {len(self.branches)} branches and {len(self.leaves)} leaves")
    
    def _generate_branch(self, start_x, start_y, end_x, end_y, start_width, end_width, depth, max_depth, z_depth, growth_start):
        """Recursively generate branches with tapering and growth timing"""
        # Add this branch with start and end widths for tapering, plus growth timing
        self.branches.append([start_x, start_y, end_x, end_y, start_width, end_width, z_depth, growth_start])
        
        # Add leaves along all branches starting from the base
        if depth >= 0:  # Add leaves to all branches including main ones
            self._add_leaves_to_branch(start_x, start_y, end_x, end_y, z_depth)
        
        # Stop if we've reached max depth
        if depth >= max_depth:
            return
        
        # Calculate branch properties
        branch_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        branch_angle = np.arctan2(end_y - start_y, end_x - start_x)
        
        # Number of sub-branches (2-3 for better spread)
        num_subbranches = np.random.randint(2, 4)
        
        # Generate sub-branches
        for i in range(num_subbranches):
            # Sub-branch starts at random point along parent (40-90% along)
            t = np.random.uniform(0.4, 0.9)
            sub_start_x = start_x + (end_x - start_x) * t
            sub_start_y = start_y + (end_y - start_y) * t
            
            # Interpolate width at branch point
            branch_width_at_point = start_width + (end_width - start_width) * t
            
            # Sub-branch angle (spread outward from parent)
            angle_offset = np.random.uniform(-1.0, 1.0)  # Wider spread
            sub_angle = branch_angle + angle_offset
            
            # Sub-branch length (40-65% of parent)
            sub_length = branch_length * np.random.uniform(0.4, 0.65)
            
            # Sub-branch end position
            sub_end_x = sub_start_x + np.cos(sub_angle) * sub_length
            sub_end_y = sub_start_y + np.sin(sub_angle) * sub_length
            
            # Sub-branch width (starts at parent's width at branch point, tapers to 60%)
            sub_start_width = branch_width_at_point * np.random.uniform(0.7, 0.9)
            sub_end_width = sub_start_width * 0.6
            
            # Depth varies slightly (branches closer to viewer are nearer)
            sub_z_depth = z_depth + np.random.uniform(-5, 5)
            sub_z_depth = np.clip(sub_z_depth, 5, 95)
            
            # Growth timing - sub-branches grow after their parent
            # Each depth level adds delay
            sub_growth_start = growth_start + (depth + 1) * 0.8 + np.random.uniform(0, 0.3)
            
            # Recurse
            self._generate_branch(
                sub_start_x, sub_start_y,
                sub_end_x, sub_end_y,
                sub_start_width,
                sub_end_width,
                depth + 1,
                max_depth,
                sub_z_depth,
                sub_growth_start
            )
    
    def _add_leaves_to_branch(self, start_x, start_y, end_x, end_y, z_depth):
        """Add leaves along a branch"""
        num_leaves = np.random.randint(3, 7)  # Fewer leaves with more spacing
        
        for i in range(num_leaves):
            # Position along branch
            t = np.random.uniform(0.2, 1.0)
            leaf_x = start_x + (end_x - start_x) * t
            leaf_y = start_y + (end_y - start_y) * t
            
            # Offset perpendicular to branch (both sides)
            branch_angle = np.arctan2(end_y - start_y, end_x - start_x)
            offset_dist = np.random.uniform(-6, 6) * self.scale
            leaf_x += np.cos(branch_angle + np.pi/2) * offset_dist
            leaf_y += np.sin(branch_angle + np.pi/2) * offset_dist
            
            # Leaf properties - smaller leaves
            leaf_size = np.random.uniform(1.5, 3.0) * self.scale
            leaf_rotation = np.random.uniform(0, 2 * np.pi)
            leaf_type = np.random.randint(0, 5)
            
            # Depth varies around branch depth (leaves slightly in front of branches)
            leaf_z = z_depth + np.random.uniform(-5, -2)
            leaf_z = np.clip(leaf_z, 5, 95)
            
            # Generate leaf color based on season
            color = self._generate_leaf_color()
            
            # Add leaf (x, y, size, rotation, r, g, b, leaf_type, depth)
            self.leaves.append([
                leaf_x, leaf_y, leaf_size, leaf_rotation,
                color[0], color[1], color[2], leaf_type, leaf_z
            ])
    
    def _generate_leaf_color(self):
        """Generate leaf color based on season (RGB)"""
        from skimage import color as skcolor
        
        season = self.season
        
        # Calculate distance from spring and fall
        spring_distance = min(abs(season - 0.125), 1 - abs(season - 0.125))
        fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
        
        spring_factor = max(0, 1 - spring_distance * 4)
        fall_factor = max(0, 1 - fall_distance * 4)
        
        if spring_factor > 0.5:
            # Green leaves in spring
            h = np.random.uniform(0.25, 0.35)
            s = np.random.uniform(0.7, 0.9)
            v = np.random.uniform(0.3, 0.5)
        else:
            # Fall color distribution
            color_rand = np.random.random()
            
            red_proportion = 0.1 + 0.3 * fall_factor
            if color_rand < red_proportion:
                # Red
                h = np.random.uniform(0.00, 0.05)
                s = np.random.uniform(0.8, 0.95)
                v = np.random.uniform(0.4, 0.6)
            elif color_rand < red_proportion + (0.1 + 0.2 * fall_factor):
                # Orange
                h = np.random.uniform(0.05, 0.10)
                s = np.random.uniform(0.85, 0.95)
                v = np.random.uniform(0.45, 0.65)
            elif color_rand < red_proportion + (0.3 + 0.3 * fall_factor):
                # Yellow
                h = np.random.uniform(0.10, 0.15)
                s = np.random.uniform(0.8, 0.9)
                v = np.random.uniform(0.5, 0.7)
            elif color_rand < red_proportion + (0.35 + 0.45 * fall_factor):
                # Brown
                h = np.random.uniform(0.07, 0.12)
                s = np.random.uniform(0.6, 0.8)
                v = np.random.uniform(0.3, 0.4)
            else:
                # Green
                h = np.random.uniform(0.25, 0.35)
                s = np.random.uniform(0.7, 0.9)
                v = np.random.uniform(0.3, 0.5)
        
        # Convert HSV to RGB
        hsv = np.array([[[h, s, v]]], dtype=np.float32)
        rgb = skcolor.hsv2rgb(hsv)
        return rgb.flatten()
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Vertex position
        layout(location = 1) in vec4 data1;     // Branch: (start_x, start_y, end_x, end_y)
        layout(location = 2) in vec4 data2;     // Branch: (start_width, end_width, depth, growth_start)
        
        uniform vec2 resolution;
        uniform float growthTime;
        
        out vec4 fragColor;
        out vec2 fragPos;
        out float growthFactor;
        
        void main() {
            // BRANCH RENDERING
            vec2 start = data1.xy;
            vec2 end = data1.zw;
            float start_width = data2.x;
            float end_width = data2.y;
            float depth = data2.z;
            float growth_start = data2.w;
            
            // Calculate growth factor (0 to 1)
            float growth_duration = 2.0;  // Each branch takes 2 seconds to grow
            growthFactor = clamp((growthTime - growth_start) / growth_duration, 0.0, 1.0);
            
            // If not grown yet, don't render
            if (growthFactor <= 0.0) {
                gl_Position = vec4(0.0, 0.0, -10.0, 1.0);  // Move off-screen
                return;
            }
            
            // Interpolate end position based on growth
            vec2 grown_end = mix(start, end, growthFactor);
            
            // Calculate branch direction
            vec2 dir = normalize(grown_end - start);
            vec2 perp = vec2(-dir.y, dir.x);
            
            // Interpolate width along branch for tapering effect
            float width = mix(start_width, end_width, position.y);
            
            // Branch quad vertices (position.x: -1 or 1 for width, position.y: 0 or 1 for length)
            vec2 offset = perp * width * position.x * 0.5;
            vec2 along = mix(start, grown_end, position.y);
            vec2 pos = along + offset;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Map depth to 0.0-1.0 range
            float depthNorm = depth / 100.0;
            depthNorm = clamp(depthNorm, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depthNorm, 1.0);
            
            // Branch color (brown bark)
            fragColor = vec4(0.3, 0.2, 0.1, 1.0);
            fragPos = position;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;
        in float growthFactor;
        out vec4 outColor;
        
        uniform float fadeAlpha;
        
        void main() {
            // BRANCH RENDERING - simple bark texture
            float noise = fract(sin(dot(fragPos * 50.0, vec2(12.9898, 78.233))) * 43758.5453);
            vec3 bark_color = fragColor.rgb * (0.9 + noise * 0.2);
            
            // Fade in as branch grows
            float alpha = fadeAlpha * smoothstep(0.0, 0.2, growthFactor);
            outColor = vec4(bark_color, alpha);
        }
        """
    
    def get_leaf_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec2 offset;    // Leaf position (x, y)
        layout(location = 2) in float size;     // Leaf size
        layout(location = 3) in float rotation; // Leaf rotation
        layout(location = 4) in vec3 color;     // Color (r, g, b)
        layout(location = 5) in float leafType; // Leaf shape type
        layout(location = 6) in float distance; // Depth value
        
        out vec4 fragColor;
        out vec2 fragPos;
        flat out int fragLeafType;
        uniform vec2 resolution;
        uniform float fadeAlpha;
        uniform float swayTime;
        
        void main() {
            fragPos = position;
            fragLeafType = int(leafType);
            
            // Apply swaying motion
            float swayPhase = swayTime * 0.5 + offset.x * 0.01;
            float swayAmount = sin(swayPhase) * 2.0;
            vec2 swayed_offset = offset + vec2(swayAmount, 0.0);
            
            // Apply rotation (including sway-induced rotation)
            float sway_rotation = sin(swayPhase) * 0.1;
            float total_rotation = rotation + sway_rotation;
            float c = cos(total_rotation);
            float s = sin(total_rotation);
            vec2 rotated = vec2(
                position.x * c - position.y * s,
                position.x * s + position.y * c
            );
            
            // Scale by leaf size
            vec2 scaled = rotated * size * 3.0;
            
            // Translate to leaf position (with sway)
            vec2 pos = scaled + swayed_offset;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Map depth
            float depth = distance / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            fragColor = vec4(color, fadeAlpha);
        }
        """
    
    def get_leaf_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;
        flat in int fragLeafType;
        out vec4 outColor;
        
        // Leaf type 0: Oak-style (rounded lobes)
        float oak_leaf(float nx, float ny) {
            float dist = abs(ny * 0.5);
            float width = (1.0 - nx * nx) * 0.5;
            width *= smoothstep(-0.9, -0.3, nx);
            width *= smoothstep(0.95, 0.3, nx);
            float lobe = 0.1 * sin(nx * 12.0) * (1.0 - nx * nx);
            width += lobe;
            return step(dist, width);
        }
        
        // Leaf type 1: Maple-style (pointed lobes)
        float maple_leaf(float nx, float ny) {
            float angle = atan(ny, nx);
            float r = length(vec2(nx, ny));
            float lobes = 0.6 + 0.3 * cos(angle * 2.5);
            return step(r, lobes * 0.8);
        }
        
        // Leaf type 2: Willow-style (long and narrow)
        float willow_leaf(float nx, float ny) {
            float dist = abs(ny * 0.3);
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
            float serration = 0.05 * sin(nx * 25.0);
            width += serration;
            return step(dist, width);
        }
        
        // Leaf type 4: Aspen-style (circular with small point)
        float aspen_leaf(float nx, float ny) {
            float r = length(vec2(nx * 1.2, ny));
            float width = 0.75;
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
            
            // Vein structure
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
            float edge_dist = min(min(1.0 - abs(nx), 1.0 - abs(ny)), leaf_mask);
            float edge = smoothstep(0.0, 0.2, edge_dist);
            
            vec3 final_color = fragColor.rgb * (1.0 - veins * 0.35 + color_var);
            float alpha = fragColor.a * edge;
            
            outColor = vec4(final_color, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile branch and leaf shaders"""
        # Compile branch shader
        branch_vert = self.get_vertex_shader()
        branch_frag = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(branch_vert, GL_VERTEX_SHADER)
            frag = shaders.compileShader(branch_frag, GL_FRAGMENT_SHADER)
            self.branch_shader = shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"Branch shader compilation error: {e}")
            raise
        
        # Compile leaf shader
        leaf_vert = self.get_leaf_vertex_shader()
        leaf_frag = self.get_leaf_fragment_shader()
        
        try:
            vert = shaders.compileShader(leaf_vert, GL_VERTEX_SHADER)
            frag = shaders.compileShader(leaf_frag, GL_FRAGMENT_SHADER)
            self.leaf_shader = shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"Leaf shader compilation error: {e}")
            raise
        
        # Return branch shader as primary (base class expects this)
        return self.branch_shader
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for branches and leaves"""
        # === BRANCH BUFFERS ===
        # Quad vertices for branch (x: width offset, y: position along branch)
        branch_vertices = np.array([
            -1.0, 0.0,  # Bottom left
             1.0, 0.0,  # Bottom right
             1.0, 1.0,  # Top right
            -1.0, 1.0   # Top left
        ], dtype=np.float32)
        
        branch_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create branch VAO
        self.branch_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.branch_VAO)
        
        # Branch vertex buffer
        branch_vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, branch_vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, branch_vertices.nbytes, branch_vertices, GL_STATIC_DRAW)
        self.VBOs.append(branch_vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Branch element buffer
        branch_EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, branch_EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, branch_indices.nbytes, branch_indices, GL_STATIC_DRAW)
        
        # Branch instance buffer
        self.branch_VBO = glGenBuffers(1)
        self.VBOs.append(self.branch_VBO)
        
        glBindVertexArray(0)
        
        # === LEAF BUFFERS ===
        # Quad vertices for leaves
        leaf_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        leaf_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create leaf VAO
        self.leaf_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.leaf_VAO)
        
        # Leaf vertex buffer
        leaf_vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, leaf_vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, leaf_vertices.nbytes, leaf_vertices, GL_STATIC_DRAW)
        self.VBOs.append(leaf_vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Leaf element buffer
        leaf_EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, leaf_EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, leaf_indices.nbytes, leaf_indices, GL_STATIC_DRAW)
        
        # Leaf instance buffer
        self.leaf_VBO = glGenBuffers(1)
        self.VBOs.append(self.leaf_VBO)
        
        glBindVertexArray(0)
        
        # Set primary VAO for base class
        self.VAO = self.branch_VAO
    
    def update(self, dt: float, state: Dict):
        """Update tree animation (swaying and growth)"""
        if not self.enabled:
            return
        
        # Update sway animation time
        self.sway_time += dt
        
        # Update growth time (clamp at growth_duration)
        self.growth_time = min(self.growth_time + dt, self.growth_duration)
    
    def _apply_branch_wrapping(self, branches):
        """Apply horizontal wrapping to branches by duplicating edge branches"""
        if len(branches) == 0:
            return branches
        
        render_branches = branches.copy()
        
        # Check each branch - if any part is near an edge, duplicate it
        for i in range(len(branches)):
            start_x = branches[i, 0]
            start_y = branches[i, 1]
            end_x = branches[i, 2]
            end_y = branches[i, 3]
            start_width = branches[i, 4]
            end_width = branches[i, 5]
            depth = branches[i, 6]
            growth_start = branches[i, 7]
            
            # Check if any point of the branch is near left edge
            min_x = min(start_x, end_x)
            max_x = max(start_x, end_x)
            
            if min_x < self.wrap_margin:
                # Duplicate on right side
                dup_branch = np.array([[
                    start_x + self.viewport.width,
                    start_y,
                    end_x + self.viewport.width,
                    end_y,
                    start_width,
                    end_width,
                    depth,
                    growth_start
                ]], dtype=np.float32)
                render_branches = np.vstack([render_branches, dup_branch])
            
            # Check if any point of the branch is near right edge
            if max_x > (self.viewport.width - self.wrap_margin):
                # Duplicate on left side
                dup_branch = np.array([[
                    start_x - self.viewport.width,
                    start_y,
                    end_x - self.viewport.width,
                    end_y,
                    start_width,
                    end_width,
                    depth,
                    growth_start
                ]], dtype=np.float32)
                render_branches = np.vstack([render_branches, dup_branch])
        
        return render_branches
    
    def _apply_leaf_wrapping(self, leaves):
        """Apply horizontal wrapping to leaves by duplicating edge leaves"""
        if len(leaves) == 0:
            return leaves
        
        # Leaves near left edge need duplicates on right
        left_edge_mask = leaves[:, 0] < self.wrap_margin
        # Leaves near right edge need duplicates on left
        right_edge_mask = leaves[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Start with original leaves
        render_leaves = leaves.copy()
        
        # Add duplicates for left edge leaves (appear on right side)
        if np.any(left_edge_mask):
            left_indices = np.where(left_edge_mask)[0]
            duplicate_leaves = leaves[left_indices].copy()
            duplicate_leaves[:, 0] += self.viewport.width  # Shift x to right side
            render_leaves = np.vstack([render_leaves, duplicate_leaves])
        
        # Add duplicates for right edge leaves (appear on left side)
        if np.any(right_edge_mask):
            right_indices = np.where(right_edge_mask)[0]
            duplicate_leaves = leaves[right_indices].copy()
            duplicate_leaves[:, 0] -= self.viewport.width  # Shift x to left side
            render_leaves = np.vstack([render_leaves, duplicate_leaves])
        
        return render_leaves
    
    def render(self, state: Dict):
        """Render branches and leaves with horizontal wrapping"""
        if not self.enabled or not self.branch_shader or not self.leaf_shader:
            return
        
        # === RENDER BRANCHES WITH WRAPPING ===
        glUseProgram(self.branch_shader)
        
        res_loc = glGetUniformLocation(self.branch_shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        fade_loc = glGetUniformLocation(self.branch_shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        growth_loc = glGetUniformLocation(self.branch_shader, "growthTime")
        glUniform1f(growth_loc, self.growth_time)
        
        # Apply horizontal wrapping to branches
        render_branches = self._apply_branch_wrapping(self.branches)
        
        # Upload branch instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.branch_VBO)
        glBufferData(GL_ARRAY_BUFFER, render_branches.nbytes, render_branches, GL_STATIC_DRAW)
        
        glBindVertexArray(self.branch_VAO)
        
        # Setup branch instance attributes (8 floats per instance)
        stride = 8 * 4
        
        # Attribute 1: data1 (start_x, start_y, end_x, end_y)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Attribute 2: data2 (start_width, end_width, depth, growth_start)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Draw branches
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(render_branches))
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # === RENDER LEAVES WITH WRAPPING ===
        if len(self.leaves) > 0:
            glUseProgram(self.leaf_shader)
            
            res_loc = glGetUniformLocation(self.leaf_shader, "resolution")
            glUniform2f(res_loc, self.viewport.width, self.viewport.height)
            
            fade_loc = glGetUniformLocation(self.leaf_shader, "fadeAlpha")
            glUniform1f(fade_loc, self.fade_factor)
            
            sway_loc = glGetUniformLocation(self.leaf_shader, "swayTime")
            glUniform1f(sway_loc, self.sway_time)
            
            # Apply horizontal wrapping to leaves
            render_leaves = self._apply_leaf_wrapping(self.leaves)
            
            # Upload leaf instance data
            glBindBuffer(GL_ARRAY_BUFFER, self.leaf_VBO)
            glBufferData(GL_ARRAY_BUFFER, render_leaves.nbytes, render_leaves, GL_STATIC_DRAW)
            
            glBindVertexArray(self.leaf_VAO)
            
            # Setup leaf instance attributes (9 floats per instance)
            stride = 9 * 4
            
            # Attribute 1: offset (x, y)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribDivisor(1, 1)
            
            # Attribute 2: size
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
            glEnableVertexAttribArray(2)
            glVertexAttribDivisor(2, 1)
            
            # Attribute 3: rotation
            glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(3)
            glVertexAttribDivisor(3, 1)
            
            # Attribute 4: color (r, g, b)
            glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
            glEnableVertexAttribArray(4)
            glVertexAttribDivisor(4, 1)
            
            # Attribute 5: leaf type
            glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
            glEnableVertexAttribArray(5)
            glVertexAttribDivisor(5, 1)
            
            # Attribute 6: distance (depth)
            glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
            glEnableVertexAttribArray(6)
            glVertexAttribDivisor(6, 1)
            
            # Draw leaves
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(render_leaves))
            
            glBindVertexArray(0)
            glUseProgram(0)
