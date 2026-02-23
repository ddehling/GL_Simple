"""
Test shader effect - 5 stationary orange circles at different depths
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_test_circles(state, outstate):
    """
    Test shader with 5 stationary orange circles
    
    Usage:
        scheduler.schedule_event(0, 60, shader_test_circles, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
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
    
    # Initialize circles on first call
    if state['count'] == 0:
        print(f"Initializing test circles for frame {frame_id}")
        
        try:
            circles_effect = viewport.add_effect(TestCirclesEffect)
            state['circles_effect'] = circles_effect
            print(f"✓ Initialized test circles for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize test circles: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # On close event, clean up
    if state['count'] == -1:
        if 'circles_effect' in state:
            print(f"Cleaning up test circles for frame {frame_id}")
            viewport.effects.remove(state['circles_effect'])
            state['circles_effect'].cleanup()
            print(f"✓ Cleaned up test circles for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class TestCirclesEffect(ShaderEffect):
    """5 stationary orange circles at different depths"""
    
    def __init__(self, viewport):
        super().__init__(viewport)
        self.num_circles = 5
        self.instance_VBO = None
        
        # Circle data (all stored as numpy arrays)
        self.positions = None  # [x, y, z] - shape (5, 3)
        self.radii = None  # [radius] - shape (5,)
        self.colors = None  # [r, g, b, a] - shape (5, 4)
        
        self._initialize_circles()
        
    def _initialize_circles(self):
        """Initialize 5 circles at different positions and depths"""
        # Fixed positions across the viewport
        # Let's spread them out nicely
        w = self.viewport.width
        h = self.viewport.height
        
        self.positions = np.array([
            [w * 0.2, h * 0.3, 10.0],   # Near top-left, far
            [w * 0.5, h * 0.5, 50.0],   # Center, mid-depth
            [w * 0.8, h * 0.3, 90.0],   # Near top-right, close
            [w * 0.3, h * 0.7, 30.0],   # Lower-left, far-ish
            [w * 0.7, h * 0.7, 70.0],   # Lower-right, close-ish
        ], dtype=np.float32)
        
        # Size based on depth (closer = bigger)
        depth_factors = self.positions[:, 2] / 100.0  # 0.0 (far) to 1.0 (near)
        base_radius = 30.0
        self.radii = base_radius * (0.3 + 0.7 * depth_factors)
        
        # Orange colors with varying brightness based on depth
        # Closer circles are brighter
        self.colors = np.column_stack([
            np.ones(5) * 0.1,  # R: full red
            np.ones(5) * 0.1,  # G: orange range
            np.ones(5) * 0.1,  # B: no blue
            1 + 0 * depth_factors  # A: more opaque when closer
        ]).astype(np.float32)
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad: -1 to 1
        layout(location = 1) in vec3 offset;  // x, y, z
        layout(location = 2) in float radius;
        layout(location = 3) in vec4 color;

        out vec4 fragColor;
        out vec2 vertPos;
        uniform vec2 resolution;

        void main() {
            // Pass position to fragment shader (-1 to 1 range)
            vertPos = position;
            
            // Scale quad by radius
            vec2 scaled = position * radius;
            
            // Translate to circle position
            vec2 pos = scaled + offset.xy;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use depth for z-sorting
            float depth = offset.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
            fragColor = color;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 vertPos;  // -1 to 1
        out vec4 outColor;

        void main() {
            // Calculate distance from center
            float dist = length(vertPos);
            
            // Discard fragments outside circle
            if (dist > 1.0) {
                discard;
            }
            
            // Smooth edge with anti-aliasing
            float edge = 1.0 - smoothstep(0.9, 1.0, dist);
            
            // Radial gradient for depth effect
            float gradient = 1.0 - dist * 0.3;
            
            outColor = vec4(fragColor.rgb * gradient, fragColor.a * edge);
        }
        """
    
    def compile_shader(self):
        """Compile and link circle shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
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
        # Quad vertices - from -1 to 1 for easy circle calculation
        vertices = np.array([
            -1.0, -1.0,  # Bottom left
             1.0, -1.0,  # Bottom right
             1.0,  1.0,  # Top right
            -1.0,  1.0   # Top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer (quad template)
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
        """No updates needed - circles are stationary"""
        pass

    def render(self, state: Dict):
        """Render all circles using instancing (back-to-front sorted)"""
        if not self.enabled or not self.shader:
            return
        
        # Sort circles back-to-front (far to near) for proper alpha blending
        sort_indices = np.argsort(-self.positions[:, 2])  # High Z first
        
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        # Build instance data (sorted back-to-front)
        instance_data = np.hstack([
            self.positions[sort_indices],  # x, y, z (3 floats)
            self.radii[sort_indices, np.newaxis],  # radius (1 float)
            self.colors[sort_indices],  # r, g, b, a (4 floats)
        ]).astype(np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 8 * 4  # 8 floats * 4 bytes
        
        # Offset (location 1) - vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Radius (location 2) - float
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color (location 3) - vec4
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Draw all circles in one call
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_circles)
        
        glBindVertexArray(0)
        glUseProgram(0)