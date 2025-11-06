"""
Lightning shader effect - Bolts striking from the top of the screen
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import time
from typing import Dict
from .base import ShaderEffect


class LightningEffect(ShaderEffect):
    """
    Lightning bolts that strike down from the top of the screen.
    Multiple bolts can exist at different depths with varying intensities.
    """
    
    def __init__(self, viewport, bolt_interval=2.0, bolt_duration=0.3, 
                 num_segments=15, jaggedness=30.0, max_bolts=5,
                 branch_probability=0.4, max_branch_depth=2, branch_length_ratio=0.5):
        """
        Args:
            viewport: The viewport to render to
            bolt_interval: Time between lightning strikes (seconds)
            bolt_duration: How long each bolt lasts (seconds)
            num_segments: Number of line segments per bolt
            jaggedness: How much the bolt zigzags
            max_bolts: Maximum simultaneous bolts
            branch_probability: Chance of branching at each segment (0.0-1.0)
            max_branch_depth: Maximum recursion depth for branches
            branch_length_ratio: How long branches are relative to parent (0.0-1.0)
        """
        super().__init__(viewport)
        self.bolt_interval = bolt_interval
        self.bolt_duration = bolt_duration
        self.num_segments = num_segments
        self.jaggedness = jaggedness
        self.max_bolts = max_bolts
        self.branch_probability = branch_probability
        self.max_branch_depth = max_branch_depth
        self.branch_length_ratio = branch_length_ratio
        self.wrap_margin = 100  # For horizontal wrapping
        
        # VBO handles
        self.vbo_positions = None
        self.vbo_offsets = None
        self.vbo_brightness = None
        
        # Active bolts storage
        self.bolts = []  # List of dicts with bolt data
        self.last_spawn_time = time.time()
        
        # NOTE: Do NOT call setup_buffers() here!
        # It will be called automatically by init() after shader compilation
    
    def compile_shader(self):
        """Compile and link shaders - REQUIRED METHOD"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Lightning shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers - Called automatically after shader compilation"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create VBOs
        self.vbo_positions = glGenBuffers(1)
        self.vbo_offsets = glGenBuffers(1)
        self.vbo_brightness = glGenBuffers(1)
        
        # Position attribute (vertex positions - relative to bolt)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        # Offset attribute (x, y, z position per vertex)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_offsets)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        # Brightness attribute (per vertex for fade effect)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_brightness)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindVertexArray(0)
    
    def generate_branch(self, start_point, direction, length_ratio, depth):
        """
        Recursively generate a lightning branch
        
        Args:
            start_point: [x, y] starting position
            direction: 1 for down/right, -1 for down/left
            length_ratio: How long this branch is relative to full bolt
            depth: Current recursion depth
        
        Returns:
            List of [x, y] points making up the branch
        """
        if depth > self.max_branch_depth:
            return []
        
        points = [start_point.copy()]
        x, y = start_point
        
        # Calculate branch parameters
        num_segments = max(3, int(self.num_segments * length_ratio))
        segment_height = (self.viewport.height * length_ratio) / num_segments
        jaggedness = self.jaggedness * length_ratio
        
        # Branch grows down and to the side
        for i in range(num_segments):
            # Move down and sideways
            x += np.random.uniform(-jaggedness, jaggedness) + (direction * jaggedness * 0.5)
            y += segment_height
            points.append([x, y])
            
            # Chance to create sub-branches (decreases with depth)
            branch_chance = self.branch_probability * (0.5 ** depth)
            if np.random.random() < branch_chance:
                # Create a smaller branch
                sub_branch = self.generate_branch(
                    np.array([x, y]),
                    np.random.choice([-1, 1]),
                    length_ratio * self.branch_length_ratio,
                    depth + 1
                )
                points.extend(sub_branch)
        
        return points
    
    def generate_bolt_path(self, start_x, start_y):
        """
        Generate a jagged lightning bolt path with branches
        
        Returns:
            Dict with 'main' path and list of 'branches'
        """
        # Generate main bolt path
        main_points = []
        x = 0.0  # Relative to bolt position
        y = 0.0
        
        segment_height = self.viewport.height / self.num_segments
        
        for i in range(self.num_segments + 1):
            main_points.append([x, y])
            
            if i < self.num_segments:
                # Add random horizontal offset for zigzag
                x += np.random.uniform(-self.jaggedness, self.jaggedness)
                y += segment_height
        
        # Generate branches from main bolt
        branches = []
        for i in range(1, len(main_points) - 1):  # Don't branch from endpoints
            # Check if we should branch here
            if np.random.random() < self.branch_probability:
                branch_start = np.array(main_points[i])
                direction = np.random.choice([-1, 1])  # Left or right
                
                # Calculate remaining length ratio
                remaining_ratio = 1.0 - (i / len(main_points))
                branch_length = remaining_ratio * self.branch_length_ratio
                
                # Generate branch
                branch_points = self.generate_branch(
                    branch_start,
                    direction,
                    branch_length,
                    depth=1
                )
                
                if len(branch_points) > 1:
                    branches.append(np.array(branch_points, dtype=np.float32))
        
        return {
            'main': np.array(main_points, dtype=np.float32),
            'branches': branches
        }
        
    def spawn_bolt(self):
        """Create a new lightning bolt with branches"""
        if len(self.bolts) >= self.max_bolts:
            return
        
        # Random horizontal position
        x = np.random.uniform(0, self.viewport.width)
        y = 0  # Start from top
        
        # Random depth (some bolts in front, some behind)
        z = np.random.uniform(20, 80)  # Mid-range depths
        
        # Generate the bolt path with branches (relative coordinates)
        path_data = self.generate_bolt_path(x, y)
        
        bolt = {
            'main_path': path_data['main'],
            'branches': path_data['branches'],
            'position': np.array([x, y, z], dtype=np.float32),
            'spawn_time': time.time(),
            'brightness': 1.0
        }
        
        self.bolts.append(bolt)
    
    def update_bolts(self):
        """Update bolt states and remove expired ones"""
        current_time = time.time()
        
        # Remove expired bolts
        self.bolts = [
            bolt for bolt in self.bolts
            if (current_time - bolt['spawn_time']) < self.bolt_duration
        ]
        
        # Update brightness (fade out)
        for bolt in self.bolts:
            elapsed = current_time - bolt['spawn_time']
            fade_progress = elapsed / self.bolt_duration
            bolt['brightness'] = 1.0 - fade_progress
        
        # Spawn new bolt if interval has passed
        if current_time - self.last_spawn_time >= self.bolt_interval:
            self.spawn_bolt()
            self.last_spawn_time = current_time
    
    def add_path_to_buffers(self, path, position, brightness, all_vertices, all_offsets, all_brightness, branch_brightness_multiplier=1.0):
        """Helper to add a path (main or branch) to render buffers"""
        # Convert path to line segments (pairs of consecutive points)
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # Add both vertices of the line segment
            all_vertices.append(p1)
            all_vertices.append(p2)
            
            # Add offset and brightness for both vertices
            all_offsets.append(position)
            all_offsets.append(position)
            adjusted_brightness = brightness * branch_brightness_multiplier
            all_brightness.append(adjusted_brightness)
            all_brightness.append(adjusted_brightness)
    
    def build_render_data(self):
        """Build vertex data for all bolts including branches and wrapped duplicates"""
        all_vertices = []
        all_offsets = []
        all_brightness = []
        
        for bolt in self.bolts:
            main_path = bolt['main_path']
            branches = bolt['branches']
            position = bolt['position']
            brightness = bolt['brightness']
            
            # Add main bolt path
            self.add_path_to_buffers(main_path, position, brightness, 
                                    all_vertices, all_offsets, all_brightness)
            
            # Add all branches (slightly dimmer than main bolt)
            for branch in branches:
                self.add_path_to_buffers(branch, position, brightness,
                                        all_vertices, all_offsets, all_brightness,
                                        branch_brightness_multiplier=0.7)
            
            # Handle wrapping - duplicate near edges
            x = position[0]
            
            # Near left edge - duplicate on right
            if x < self.wrap_margin:
                wrapped_pos = position.copy()
                wrapped_pos[0] += self.viewport.width
                
                # Duplicate main path
                self.add_path_to_buffers(main_path, wrapped_pos, brightness,
                                        all_vertices, all_offsets, all_brightness)
                
                # Duplicate branches
                for branch in branches:
                    self.add_path_to_buffers(branch, wrapped_pos, brightness,
                                            all_vertices, all_offsets, all_brightness,
                                            branch_brightness_multiplier=0.7)
            
            # Near right edge - duplicate on left
            if x > (self.viewport.width - self.wrap_margin):
                wrapped_pos = position.copy()
                wrapped_pos[0] -= self.viewport.width
                
                # Duplicate main path
                self.add_path_to_buffers(main_path, wrapped_pos, brightness,
                                        all_vertices, all_offsets, all_brightness)
                
                # Duplicate branches
                for branch in branches:
                    self.add_path_to_buffers(branch, wrapped_pos, brightness,
                                            all_vertices, all_offsets, all_brightness,
                                            branch_brightness_multiplier=0.7)
        
        if not all_vertices:
            return None, None, None, 0
        
        vertices = np.array(all_vertices, dtype=np.float32)
        offsets = np.array(all_offsets, dtype=np.float32)
        brightness_data = np.array(all_brightness, dtype=np.float32)
        vertex_count = len(vertices)
        
        return vertices, offsets, brightness_data, vertex_count
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        self.update_bolts()
    
    def render(self, state):
        """Render all active lightning bolts"""
        if not self.enabled or self.shader is None:
            return
        
        if not self.bolts:
            return
        
        # Build render data
        vertices, offsets, brightness_data, vertex_count = self.build_render_data()
        
        if vertex_count == 0:
            return
        
        # NO depth test toggling per shader_info.txt guidelines!
        glUseProgram(self.shader)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, b"resolution")
        if res_loc >= 0:
            glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        # Upload geometry
        glBindVertexArray(self.VAO)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_offsets)
        glBufferData(GL_ARRAY_BUFFER, offsets.nbytes, offsets, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_brightness)
        glBufferData(GL_ARRAY_BUFFER, brightness_data.nbytes, brightness_data, GL_DYNAMIC_DRAW)
        
        # Render as lines
        old_line_width = glGetFloatv(GL_LINE_WIDTH)
        glLineWidth(3.0)
        glDrawArrays(GL_LINES, 0, vertex_count)
        glLineWidth(old_line_width)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def get_vertex_shader(self):
        return """
#version 310 es
precision highp float;

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 offset;
layout(location = 2) in float brightness;

uniform vec2 resolution;

out float v_brightness;

void main() {
    vec2 pos = position + offset.xy;
    vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
    clipPos.y = -clipPos.y;
    
    float depth = offset.z / 100.0;
    depth = clamp(depth, 0.0, 1.0);
    
    gl_Position = vec4(clipPos, depth, 1.0);
    v_brightness = brightness;
}
"""
    
    def get_fragment_shader(self):
        return """
#version 310 es
precision highp float;

in float v_brightness;
out vec4 outColor;

void main() {
    vec3 innerColor = vec3(1.0, 1.0, 1.0);
    vec3 outerColor = vec3(0.5, 0.7, 1.0);
    vec3 boltColor = mix(outerColor, innerColor, 0.7);
    float alpha = v_brightness * 0.9;
    outColor = vec4(boltColor * v_brightness, alpha);
}
"""
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if hasattr(self, 'VAO') and self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if hasattr(self, 'vbo_positions') and self.vbo_positions:
            glDeleteBuffers(1, [self.vbo_positions])
        if hasattr(self, 'vbo_offsets') and self.vbo_offsets:
            glDeleteBuffers(1, [self.vbo_offsets])
        if hasattr(self, 'vbo_brightness') and self.vbo_brightness:
            glDeleteBuffers(1, [self.vbo_brightness])
        super().cleanup()


# Event wrapper function for EventScheduler
def shader_lightning(state, outstate, bolt_interval=2.0, bolt_duration=0.3,
                     num_segments=15, jaggedness=30.0, max_bolts=5,
                     branch_probability=0.4, max_branch_depth=2, branch_length_ratio=0.5):
    """
    Lightning bolts shader effect with branching - compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_lightning, 
                               bolt_interval=2.0, 
                               branch_probability=0.5,
                               frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        bolt_interval: Time between lightning strikes (seconds)
        bolt_duration: How long each bolt lasts (seconds)
        num_segments: Number of line segments per bolt (more = more detail)
        jaggedness: How much the bolt zigzags horizontally
        max_bolts: Maximum number of simultaneous bolts
        branch_probability: Chance of branching at each segment (0.0-1.0, default 0.4)
        max_branch_depth: Maximum recursion depth for branches (default 2)
        branch_length_ratio: How long branches are relative to parent (0.0-1.0, default 0.5)
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
    
    if state['count'] == 0:
        print(f"Initializing lightning effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                LightningEffect,
                bolt_interval=bolt_interval,
                bolt_duration=bolt_duration,
                num_segments=num_segments,
                jaggedness=jaggedness,
                max_bolts=max_bolts,
                branch_probability=branch_probability,
                max_branch_depth=max_branch_depth,
                branch_length_ratio=branch_length_ratio
            )
            state['effect'] = effect
            print(f"✓ Initialized shader lightning for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize lightning: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up lightning effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader lightning for frame {frame_id}")