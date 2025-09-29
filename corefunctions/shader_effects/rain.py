"""
Complete rain effect - rendering + event integration
Everything needed for rain in one place!
"""
import numpy as np
import random
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_rain(state, outstate, intensity=1.0, wind=0.0):
    """
    Shader-based rain effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_rain, intensity=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        intensity: Rain intensity multiplier (affects number of drops)
        wind: Wind effect (-1 to 1, affects drop angle)
    """
    # Get the viewport
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize rain effect on first call
    if state['count'] == 0:
        num_drops = int(100 * intensity)
        print(f"Initializing rain effect for frame {frame_id} with {num_drops} drops")
        
        try:
            rain_effect = viewport.add_effect(
                RainEffect,
                num_raindrops=num_drops,
                wind=wind
            )
            state['rain_effect'] = rain_effect
            print(f"✓ Initialized shader rain for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize rain: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update wind if it changes in global state
    if 'rain_effect' in state:
        state['rain_effect'].wind = outstate.get('wind', wind)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'rain_effect' in state:
            print(f"Cleaning up rain effect for frame {frame_id}")
            viewport.effects.remove(state['rain_effect'])
            state['rain_effect'].cleanup()
            print(f"✓ Cleaned up shader rain for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class Raindrop:
    """Individual raindrop particle"""
    def __init__(self, surface_width, surface_height, wind=0.0):
        self.surface_width = surface_width
        self.surface_height = surface_height
        self.wind = wind
        self.reset(True)
        
    def reset(self, randomize_y=False):
        """Reset raindrop to top of screen"""
        self.x = random.uniform(0, self.surface_width)
        self.y = random.uniform(0, self.surface_height) if randomize_y else -10
        self.z = random.uniform(0, 100)  # NEW: Depth from 0 (far) to 100 (near)
        self.speed = random.uniform(100, 300)
        self.width = random.uniform(1.0, 2.0)
        self.length = random.uniform(10, 20)
        
        # NEW: Scale and alpha based on depth (closer = larger, more opaque)
        depth_factor = self.z / 100.0  # 0.0 (far) to 1.0 (near)
        self.width *= (0.3 + 0.7 * depth_factor)  # 30% to 100% size
        self.length *= (0.3 + 0.7 * depth_factor)
        self.alpha = 0.2 + 0.6 * depth_factor  # 0.2 to 0.8 opacity

        
    def update(self, dt):
        """Update raindrop position"""
        self.y += self.speed * dt
        self.x += self.wind * 50 * dt  # Wind effect
        
        # Wrap around horizontally
        if self.x < -10:
            self.x = self.surface_width + 10
        elif self.x > self.surface_width + 10:
            self.x = -10
            
        # Reset when off bottom
        if self.y > self.surface_height + 10:
            self.reset()


class RainEffect(ShaderEffect):
    """GPU-based rain effect using instanced rendering"""
    
    def __init__(self, viewport, num_raindrops: int = 100, wind: float = 0.0):
        super().__init__(viewport)
        self.num_raindrops = num_raindrops
        self.wind = wind
        self.raindrops = []
        self.instance_VBO = None
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 offset;  // x, y, z
        layout(location = 2) in vec2 size;
        layout(location = 3) in vec4 color;

        out vec4 fragColor;
        uniform vec2 resolution;

        void main() {
            // Apply size to quad and add screen-space offset
            vec2 pos = position * size + offset.xy;
            
            // Convert screen coordinates to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use Z for depth buffer (normalize to 0-1 range)
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
        out vec4 outColor;

        void main() {
            outColor = fragColor;
        }
        """
    
    def compile_shader(self):
        """Compile and link rain shaders"""
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


    def _setup_projection(self, shader):
        """Create orthographic projection for 3D rendering"""
        width = self.viewport.width
        height = self.viewport.height
        near = 0.0
        far = 100.0
        
        # Orthographic projection matrix
        # Maps: x:[0,width] y:[0,height] z:[0,100] to clip space [-1,1]
        projection = np.array([
            [2.0/width,  0.0,         0.0,        -1.0],
            [0.0,       -2.0/height,  0.0,         1.0],
            [0.0,        0.0,        -2.0/(far-near), -(far+near)/(far-near)],
            [0.0,        0.0,         0.0,         1.0]
        ], dtype=np.float32)
        
        loc = glGetUniformLocation(shader, "projection")
        if loc != -1:
            glUniformMatrix4fv(loc, 1, GL_FALSE, projection) 

    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced rendering"""
        # Create raindrops
        self.raindrops = [
            Raindrop(self.viewport.width, self.viewport.height, self.wind) 
            for _ in range(self.num_raindrops)
        ]

        # Sort by Z (far to near) for proper depth rendering
        self.raindrops.sort(key=lambda d: d.z)

        # Quad vertices (will be instanced)
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

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
        
        # Instance buffer (will be updated each frame)
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)

    
    def update(self, dt: float, state: Dict):
        """Update raindrop positions"""
        if not self.enabled:
            return
            
        # Update wind for all drops
        for drop in self.raindrops:
            drop.wind = self.wind
            drop.update(dt)
    

    def render(self, state: Dict):
        """Render all raindrops using instancing"""
        if not self.enabled or not self.shader:
            return
        
        # NEW: Clear depth buffer
        glClear(GL_DEPTH_BUFFER_BIT)
            
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        
        # Prepare instance data (offset with Z, size, color for each drop)
        instance_data = []
        for drop in self.raindrops:
            instance_data.extend([
                drop.x, drop.y, drop.z,  # offset (now 3D)
                drop.width, drop.length,  # size
                0.3, 0.7, 1.0, drop.alpha # color
            ])
        
        if not instance_data:
            return
            
        instance_array = np.array(instance_data, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_array.nbytes, instance_array, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 9 * 4  # 9 floats * 4 bytes (changed from 8)
        
        # Offset (location 1) - now vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Size (location 2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color (location 3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Draw all drops in one call
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_raindrops)
        
        glBindVertexArray(0)
        glUseProgram(0)
