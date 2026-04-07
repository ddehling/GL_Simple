"""
Test pattern shader - alternating red, green, blue stripes
Useful for debugging LED alignment and testing viewport output
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_test_pattern(state, outstate, orientation='horizontal', stripe_width=3, pattern='rgb_stripes'):
    """
    Test pattern with alternating RGB stripes
    
    Usage:
        scheduler.schedule_event(0, 60, shader_test_pattern, 
                               orientation='vertical', stripe_width=3, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        orientation: 'horizontal' or 'vertical' stripe direction
        stripe_width: Width of each color stripe in pixels (default 3)
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
        print(f"Initializing test_pattern for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                TestPatternEffect,
                orientation=orientation,
                stripe_width=stripe_width,
                pattern=pattern
            )
            state['effect'] = effect
            print(f"✓ Initialized shader test_pattern ({orientation}, width={stripe_width}) for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize test_pattern: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update orientation if changed
    if 'effect' in state:
        new_orientation = outstate.get('test_pattern_orientation', orientation)
        new_stripe_width = outstate.get('test_pattern_stripe_width', stripe_width)
        
        if state['effect'].orientation != new_orientation:
            state['effect'].orientation = new_orientation
            print(f"Changed test pattern orientation to {new_orientation}")
        
        if state['effect'].stripe_width != new_stripe_width:
            state['effect'].stripe_width = new_stripe_width
            print(f"Changed test pattern stripe width to {new_stripe_width}")

        # Switch pattern based on the current weather state
        current_state = outstate.get('current_weather_state', '')
        if current_state == 'test_rgb':
            desired_pattern = 'rgb_stripes'
        elif current_state == 'test_hue_bin':
            desired_pattern = 'hue_binary'
        else:
            desired_pattern = state['effect'].pattern  # leave alone
        if state['effect'].pattern != desired_pattern:
            state['effect'].pattern = desired_pattern
            print(f"Changed test pattern to {desired_pattern}")
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up test_pattern for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader test_pattern for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class TestPatternEffect(ShaderEffect):
    """Fullscreen test pattern with RGB stripes"""
    
    def __init__(self, viewport, orientation: str = 'horizontal', stripe_width: int = 3,
                 pattern: str = 'rgb_stripes'):
        super().__init__(viewport)
        self.orientation = orientation  # 'horizontal' or 'vertical'
        self.stripe_width = stripe_width
        self.pattern = pattern  # 'rgb_stripes' or 'hue_binary'
        self.depth = 50.0  # Mid-depth
    
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
            print(f"TestPatternEffect shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 vTexCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            // Convert from [-1, 1] to [0, 1]
            vTexCoord = position * 0.5 + 0.5;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;

        in vec2 vTexCoord;

        uniform vec2 resolution;
        uniform float depth;
        uniform int orientation;   // 0=horizontal, 1=vertical
        uniform float stripeWidth;
        uniform int patternMode;   // 0=rgb_stripes, 1=hue_binary

        out vec4 outColor;

        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        void main() {
            vec2 pixelCoord = vTexCoord * resolution;

            if (patternMode == 0) {
                // ----- RGB stripes (legacy) -----
                float coord = (orientation == 0) ? pixelCoord.x : pixelCoord.y;
                int stripeIndex = int(floor(coord / stripeWidth)) % 3;
                vec3 color;
                if (stripeIndex == 0) {
                    color = vec3(1.0, 0.0, 0.0);
                } else if (stripeIndex == 1) {
                    color = vec3(0.0, 1.0, 0.0);
                } else {
                    color = vec3(0.0, 0.0, 1.0);
                }
                outColor = vec4(color, 1.0);
            } else {
                // ----- Horizontal hue gradient + binary column index -----
                int col  = int(floor(pixelCoord.x));   // 0..127
                int yPix = int(floor(pixelCoord.y));   // 0..299, FBO bottom = 0

                float hue = (float(col) / 127.0) * 0.66;
                vec3 rgb = hsv2rgb(vec3(hue, 1.0, 1.0));

                // First 10 LEDs of each strip = top 10 rows of FBO
                // (get_frame() flips Y on readback).
                int topRow = int(resolution.y) - 1 - yPix;
                if (topRow >= 0 && topRow < 10) {
                    int bit = (col >> (9 - topRow)) & 1;
                    rgb = (bit == 1) ? vec3(1.0) : vec3(0.0);
                }

                outColor = vec4(rgb, 1.0);
            }
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for fullscreen quad"""
        # Fullscreen quad vertices (clip space)
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create VBO
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        
        # Cache uniform locations
        self.uniform_resolution = glGetUniformLocation(self.shader, "resolution")
        self.uniform_depth = glGetUniformLocation(self.shader, "depth")
        self.uniform_orientation = glGetUniformLocation(self.shader, "orientation")
        self.uniform_stripe_width = glGetUniformLocation(self.shader, "stripeWidth")
        self.uniform_pattern_mode = glGetUniformLocation(self.shader, "patternMode")
    
    def render(self, state: Dict):
        """Render test pattern"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        glUniform2f(self.uniform_resolution, self.viewport.width, self.viewport.height)
        glUniform1f(self.uniform_depth, self.depth)
        
        # Convert orientation string to int (0=horizontal, 1=vertical)
        orientation_int = 1 if self.orientation == 'vertical' else 0
        glUniform1i(self.uniform_orientation, orientation_int)
        glUniform1f(self.uniform_stripe_width, float(self.stripe_width))
        glUniform1i(self.uniform_pattern_mode, 1 if self.pattern == 'hue_binary' else 0)
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def update(self, dt: float, state: Dict):
        """No animation needed for static test pattern"""
        pass
