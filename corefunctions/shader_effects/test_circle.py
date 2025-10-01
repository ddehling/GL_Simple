"""
Simple circle for testing depth blending
"""
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function
# ============================================================================

def shader_test_circle(state, outstate, x=60, y=30, radius=20, z=50, color=(1.0, 0.0, 0.0, 0.8)):
    """
    Simple circle at specified depth for testing
    
    Usage:
        scheduler.schedule_event(0, 60, shader_test_circle, x=60, y=30, z=50, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        x, y: Center position
        radius: Circle radius
        z: Depth (0=far, 100=near)
        color: RGBA tuple
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
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing test circle at z={z}")
        
        try:
            circle_effect = viewport.add_effect(
                TestCircleEffect,
                x=x, y=y, radius=radius, z=z, color=color
            )
            state['circle_effect'] = circle_effect
            print(f"✓ Initialized test circle")
        except Exception as e:
            print(f"✗ Failed to initialize circle: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # On close event, clean up
    if state['count'] == -1:
        if 'circle_effect' in state:
            print(f"Cleaning up test circle")
            viewport.effects.remove(state['circle_effect'])
            state['circle_effect'].cleanup()
            print(f"✓ Cleaned up test circle")


# ============================================================================
# Rendering Class
# ============================================================================

class TestCircleEffect(ShaderEffect):
    """Simple filled circle at specified depth"""
    
    def __init__(self, viewport, x=60, y=30, radius=20, z=50, color=(1.0, 0.0, 0.0, 0.8)):
        super().__init__(viewport)
        self.x = x
        self.y = y
        self.radius = radius
        self.z = z
        self.color = color
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Vertex position (-1 to 1)
        
        out vec2 uv;
        
        uniform vec2 resolution;
        uniform vec3 circlePos;  // x, y, z
        uniform float radius;

        void main() {
            // Scale vertex by radius and offset to circle position
            vec2 pos = position * radius + circlePos.xy;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Normalize Z to 0-1 range for depth buffer
            float depth = circlePos.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
            uv = position;  // Pass through for fragment shader
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 uv;
        out vec4 outColor;
        
        uniform vec4 color;

        void main() {
            // Draw filled circle
            float dist = length(uv);
            if (dist > 1.0) {
                discard;  // Outside circle
            }
            
            // Smooth edge
            float alpha = color.a * smoothstep(1.0, 0.95, dist);
            outColor = vec4(color.rgb, alpha);
        }
        """
    
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
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Quad vertices centered at origin (-1 to 1)
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
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
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update (nothing to do for static circle)"""
        pass
    
    def render(self, state: Dict):
        """Render the circle"""
        if not self.enabled or not self.shader:
            return
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending
        
        # Enable depth testing AND depth writes
        # The circle will write to the depth buffer, blocking objects behind it
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)  # Write to depth buffer so we block things behind us
            
        glUseProgram(self.shader)
        
        # Set uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "circlePos")
        if loc != -1:
            glUniform3f(loc, self.x, self.y, self.z)
        
        loc = glGetUniformLocation(self.shader, "radius")
        if loc != -1:
            glUniform1f(loc, self.radius)
        
        loc = glGetUniformLocation(self.shader, "color")
        if loc != -1:
            glUniform4f(loc, self.color[0], self.color[1], self.color[2], self.color[3])
        
        # Draw
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Disable blending (depth mask already GL_TRUE)
        glDisable(GL_BLEND)