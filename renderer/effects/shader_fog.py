import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords

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

# Fragment shader with depth-based fog, blur, and spatial clustering
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
uniform float u_time;

""" + FAN_COORDS_UNIFORMS + FAN_COORDS_GLSL + """

// Bay Area fog spatial density map.
// Fog rolls in from the Pacific (west/negative x) through the Golden Gate
// and settles along the coast and SF peninsula. East Bay stays clearer.
float bay_fog_density(vec2 phys) {
    // Convert physical coords to approximate lat/lon
    float nx = (phys.x - (-20.6)) / 41.2;  // 0-1 across fan
    float ny = (phys.y - 0.0) / 20.6;
    float lon = -123.01 + nx * 1.78;
    float lat = 37.32 + ny * 0.70;

    float density = 0.0;

    // Ocean fog — strongest over the Pacific (lon < -122.5)
    float ocean_dist = (-122.50 - lon) * 4.0;  // positive = further into ocean
    density += clamp(ocean_dist, 0.0, 0.6);

    // Golden Gate corridor fog (lon ~ -122.45, lat ~ 37.81)
    float gate_dist = length(vec2((lon - (-122.45)) * 8.0, (lat - 37.81) * 12.0));
    density += 0.5 * exp(-gate_dist * gate_dist * 1.5);

    // SF peninsula coastal fog (west side, lon < -122.42)
    float sf_coast = clamp((-122.42 - lon) * 3.0, 0.0, 0.4);
    float sf_lat_band = exp(-pow((lat - 37.75) * 4.0, 2.0));  // centered on SF
    density += sf_coast * sf_lat_band;

    // Fog thins rapidly east of the Bay (lon > -122.2)
    float east_clear = clamp((lon - (-122.2)) * 3.0, 0.0, 1.0);
    density *= (1.0 - east_clear * 0.8);

    // Animate: slow drift and pulse
    float drift = sin(u_time * 0.15 + phys.x * 0.3) * 0.1;
    density += drift;

    return clamp(density, 0.0, 1.0);
}

// Gaussian blur with proper weighting
vec4 apply_gaussian_blur(vec2 uv, float radius) {
    if (radius < 0.5) {
        return texture(u_color_texture, uv);
    }

    vec2 texel_size = 1.0 / u_resolution;
    vec4 result = vec4(0.0);
    float total_weight = 0.0;

    int r = int(ceil(radius));
    r = clamp(r, 1, 12);

    float sigma = radius / 2.5;

    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            float dist = length(vec2(float(x), float(y)));
            float weight = exp(-(dist * dist) / (2.0 * sigma * sigma));
            result += texture(u_color_texture, uv + offset) * weight;
            total_weight += weight;
        }
    }

    return result / total_weight;
}

void main() {
    // Spatial fog density from Bay Area geography — purely geographic,
    // independent of depth buffer (avoids artifacts from aurora/sky layers)
    vec2 phys = fan_uv_to_physical(v_texcoord);
    float spatial_fog = bay_fog_density(phys);

    float combined = u_fog_strength * spatial_fog;

    float blur_radius = combined * 4.0;
    vec4 blurred_color = apply_gaussian_blur(v_texcoord, blur_radius);

    float fog_mix = combined * 0.5;
    vec3 final_color = mix(blurred_color.rgb, u_fog_color, fog_mix);

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
        self.render_priority = 1000  # Post-processing: render LAST
        self.base_strength = strength  # Base strength set at initialization
        self.current_strength = strength  # Current strength (affected by fade)
        self.fog_color = color
        self.fog_near = fog_near
        self.fog_far = fog_far
        self._time = 0.0
        self._fan = FanCoords(viewport.width, viewport.height)

        # OpenGL resources (initialized in setup_buffers)
        self.VAO = None
        self.VBO = None
        self.temp_texture = None  # Temporary texture for post-processing
        self.temp_fbo = None  # Temporary framebuffer for copying
    
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
        
        # Create temporary texture for post-processing
        self.temp_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.temp_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.viewport.width, self.viewport.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create temporary framebuffer
        self.temp_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.temp_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, self.temp_texture, 0)
        
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Fog temp framebuffer incomplete: {status}")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        print(f"    ✓ Fog temp framebuffer created")
    
    def update(self, dt: float, state: Dict):
        """Update fog parameters each frame"""
        if not self.enabled:
            return

        self._time += dt

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
        if self.current_strength < 0.001:
            return
        
        # Step 1: Blit framebuffer color to temp FBO using glBlitFramebuffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.viewport.fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.temp_fbo)
        glBlitFramebuffer(
            0, 0, self.viewport.width, self.viewport.height,
            0, 0, self.viewport.width, self.viewport.height,
            GL_COLOR_BUFFER_BIT, GL_NEAREST
        )
        
        # Step 2: Render fog back to the original framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.viewport.fbo)
        
        # Disable depth testing for fullscreen quad
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Bind temp texture (copied scene)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.temp_texture)
        glUniform1i(glGetUniformLocation(self.shader, "u_color_texture"), 0)
        
        # Bind depth texture
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.viewport.depth_texture)
        glUniform1i(glGetUniformLocation(self.shader, "u_depth_texture"), 1)
        
        # Set fog parameters
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
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        self._fan.set_uniforms(self.shader)
        
        # Draw fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Re-enable depth testing and blend
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
    
    def cleanup(self):
        """Clean up resources"""
        if self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if self.VBO:
            glDeleteBuffers(1, [self.VBO])
        if self.temp_texture:
            glDeleteTextures(1, [self.temp_texture])
        if self.temp_fbo:
            glDeleteFramebuffers(1, [self.temp_fbo])
        if self.shader:
            glDeleteProgram(self.shader)
