import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# Vertex shader for fullscreen quad
VERTEX_SHADER = """#version 310 es
precision highp float;

in vec2 position;
out vec2 v_texcoord;

void main() {
    v_texcoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader with depth-based fog and blur
FRAGMENT_SHADER = """#version 310 es
precision highp float;

in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D u_color_texture;
uniform sampler2D u_depth_texture;
uniform vec3 u_fog_color;
uniform float u_fog_strength;
uniform float u_fog_near;
uniform float u_fog_far;
uniform vec2 u_resolution;

// Linearize depth value
float linearize_depth(float depth) {
    float near = u_fog_near;
    float far = u_fog_far;
    float z = depth * 2.0 - 1.0; // Back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

// Gaussian blur based on depth
vec4 blur_at_depth(vec2 uv, float depth_factor) {
    vec2 texel_size = 1.0 / u_resolution;
    float blur_amount = depth_factor * 8.0; // Max blur radius
    
    if (blur_amount < 0.5) {
        return texture(u_color_texture, uv);
    }
    
    vec4 result = vec4(0.0);
    float total_weight = 0.0;
    
    // 5x5 gaussian kernel (simplified for performance)
    int radius = int(ceil(blur_amount));
    radius = min(radius, 4); // Limit max radius
    
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            float dist = length(vec2(float(x), float(y)));
            float weight = exp(-dist * dist / (2.0 * blur_amount * blur_amount));
            
            result += texture(u_color_texture, uv + offset) * weight;
            total_weight += weight;
        }
    }
    
    return result / total_weight;
}

void main() {
    // Early exit if fog strength is effectively zero
    if (u_fog_strength < 0.001) {
        fragColor = texture(u_color_texture, v_texcoord);
        return;
    }
    
    // Sample depth
    float depth = texture(u_depth_texture, v_texcoord).r;
    
    // Linearize depth to get distance from camera
    float linear_depth = linearize_depth(depth);
    
    // Calculate fog factor based on depth (0 = no fog, 1 = full fog)
    float depth_range = u_fog_far - u_fog_near;
    float fog_factor = clamp((linear_depth - u_fog_near) / depth_range, 0.0, 1.0);
    fog_factor = fog_factor * u_fog_strength * u_fog_strength;
    
    // Apply depth-based blur
    vec4 blurred_color = blur_at_depth(v_texcoord, fog_factor);
    
    // Mix blurred color with fog color based on fog factor
    vec3 final_color = mix(blurred_color.rgb, u_fog_color, fog_factor * 0.3);
    
    fragColor = vec4(final_color, blurred_color.a);
}
"""


def shader_fog(state, outstate, strength=0.0, color=(0.7, 0.7, 0.8), 
               fog_near=10.0, fog_far=100.0):
    """
    Depth-based fog post-processing effect compatible with EventScheduler
    
    Creates atmospheric fog with depth-based blur. Distant objects appear
    foggy and blurred, while near objects remain clear.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_fog, strength=0.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        strength: Fog intensity (0.0 = no fog, 1.0 = full fog)
        color: RGB tuple for fog color (default light blue-gray)
        fog_near: Distance where fog starts
        fog_far: Distance where fog reaches maximum
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
        print(f"Initializing fog effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                ShaderFog,
                strength=strength,
                color=color,
                fog_near=fog_near,
                fog_far=fog_far
            )
            effect._managed_by_wrapper = True  # Mark as managed by wrapper
            state['effect'] = effect
            print(f"✓ Initialized shader fog for frame {frame_id} (strength={strength})")
        except Exception as e:
            print(f"✗ Failed to initialize fog: {e}")
            import traceback
            traceback.print_exc()
            return
    
                        # Update fog parameters from global state (optional dynamic control)
    if 'effect' in state:
        # Allow dynamic fog strength control from outstate
        if 'fog_strength' in outstate:
            state['effect'].base_strength = outstate['fog_strength']
        
        # Allow dynamic fog color control from outstate
        if 'fog_color' in outstate:
            state['effect'].fog_color = outstate['fog_color']
        
        # Calculate current_strength based on fade (if duration specified)
        if state.get('duration') is not None:
            elapsed_time = state['elapsed_time']
            total_duration = state['duration']
            fade_duration = 2.0  # Fade in/out over 2 seconds
            
            # Calculate fade factor (0.0 to 1.0)
            if elapsed_time < fade_duration:
                # Fade in during first N seconds
                fade_factor = elapsed_time / fade_duration
            elif elapsed_time > (total_duration - fade_duration):
                # Fade out during last N seconds
                fade_factor = (total_duration - elapsed_time) / fade_duration
            else:
                # Full strength in the middle
                fade_factor = 1.0
            
            # Apply fade to base strength parameter
            state['effect'].current_strength = state['effect'].base_strength * np.clip(fade_factor, 0, 1)
        else:
            # No fade if duration not specified - use base_strength directly
            state['effect'].current_strength = state['effect'].base_strength
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up fog effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader fog for frame {frame_id}")


class ShaderFog(ShaderEffect):
    """Post-processing fog effect with depth-based blur"""
    
    def __init__(self, viewport, strength=0.0, color=(0.7, 0.7, 0.8), 
                 fog_near=10.0, fog_far=100.0):
        super().__init__(viewport)
        self.base_strength = strength  # Base strength set at initialization
        self.current_strength = strength  # Current strength (affected by fade)
        self.fog_color = color
        self.fog_near = fog_near
        self.fog_far = fog_far
        
        # OpenGL resources (initialized in setup_buffers)
        self.VAO = None
        self.VBO = None
    
    @property
    def strength(self):
        """Getter for strength (for backward compatibility)"""
        return self.current_strength
    
    @strength.setter
    def strength(self, value):
        """Setter for strength - updates base_strength"""
        self.base_strength = value
        # Note: current_strength is updated by the event wrapper based on fade
        # If no wrapper is used, this ensures current_strength matches
        if not hasattr(self, '_managed_by_wrapper'):
            self.current_strength = value

    def compile_shader(self):
        """Compile and link shaders - REQUIRED METHOD"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader_program = shaders.compileProgram(vert, frag)
            return shader_program
        except Exception as e:
            print(f"ShaderFog compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        """Return vertex shader source"""
        return VERTEX_SHADER
    
    def get_fragment_shader(self):
        """Return fragment shader source"""
        return FRAGMENT_SHADER
    
    def setup_buffers(self):
        """Initialize OpenGL buffers - Called automatically after shader compilation"""
        # Create fullscreen quad
        vertices = np.array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ], dtype=np.float32)
        
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        pos_loc = glGetAttribLocation(self.shader, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update fog parameters each frame"""
        if not self.enabled:
            return
        
        # If not managed by wrapper, check state dict for dynamic updates
        # This allows real-time control without using the event wrapper
        if not hasattr(self, '_managed_by_wrapper') or not self._managed_by_wrapper:
            if 'fog_strength' in state:
                self.base_strength = state['fog_strength']
                self.current_strength = state['fog_strength']
            
            if 'fog_color' in state:
                self.fog_color = state['fog_color']
    
    def render(self, state: Dict):
        """Apply fog as post-processing effect"""
        if not self.enabled:
            return
        
        # Skip rendering entirely if fog strength is effectively zero
        # This prevents any GPU processing when fog is disabled
        if self.current_strength < 0.001:
            return
        
        # Post-process exception: Use GL_ALWAYS to render on top without toggling depth test
        glDepthFunc(GL_ALWAYS)  # Always pass depth test (render in front)
        glDepthMask(GL_FALSE)   # Don't write to depth buffer
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Bind color texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.viewport.color_texture)
        glUniform1i(glGetUniformLocation(self.shader, "u_color_texture"), 0)
        
        # Bind depth texture from viewport
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.viewport.depth_texture)
        glUniform1i(glGetUniformLocation(self.shader, "u_depth_texture"), 1)
        
        # Set uniforms - use current_strength which includes fade
        glUniform3f(glGetUniformLocation(self.shader, "u_fog_color"),
                   self.fog_color[0], self.fog_color[1], self.fog_color[2])
        glUniform1f(glGetUniformLocation(self.shader, "u_fog_strength"),
                   self.current_strength)
        glUniform1f(glGetUniformLocation(self.shader, "u_fog_near"),
                   self.fog_near)
        glUniform1f(glGetUniformLocation(self.shader, "u_fog_far"),
                   self.fog_far)
        glUniform2f(glGetUniformLocation(self.shader, "u_resolution"),
                   float(self.viewport.width), float(self.viewport.height))
        
        # Draw fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore default depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
    
    def cleanup(self):
        """Clean up resources"""
        if self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if self.VBO:
            glDeleteBuffers(1, [self.VBO])
        if self.shader:
            glDeleteProgram(self.shader)
