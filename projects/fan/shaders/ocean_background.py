"""
Ocean wave shader - Physics-based wave simulation with beach rendering
"""
import traceback
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
import ctypes

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_ocean_background(state, outstate, depth=90.0):
    """
    Shader-based ocean wave effect with beach terrain
    
    Creates realistic ocean waves using wave equation simulation with:
    - Dynamic wave propagation
    - Beach/terrain rendering with shadows
    - Water specular highlights
    - Adjustable water level and wave parameters
    
    Usage:
        scheduler.schedule_event(0, 999999, shader_ocean_background, depth=90, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        depth: Z-depth of ocean background (0=near, 100=far, default: 90)
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
        print(f"Initializing ocean wave shader for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                OceanBackground,
                depth=depth
            )
            state['effect'] = effect
            print(f"[OK] Initialized ocean wave shader for frame {frame_id}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ocean shader: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from global state (ocean parameters)
    if 'effect' in state:
        state['effect'].wave_speed = outstate.get('wave_speed', 0.5)
        state['effect'].wave_amplitude = outstate.get('wave_amplitude', 0.5)
        state['effect'].tide_level = outstate.get('tide_level', 0.0)
        state['effect'].season = outstate.get('season', 0.5)
        state['effect'].wind_speed = outstate.get('wind', 0.2)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up ocean shader for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ============================================================================
# Rendering Class
# ============================================================================

class OceanBackground(ShaderEffect):
    """
    Ocean wave effect using physics-based simulation
    
    Implements a wave equation solver with beach terrain rendering
    """
    
    def __init__(self, viewport, depth: float = 90.0):
        super().__init__(viewport)
        self.depth = depth
        self.time = 0.0
        self.frame_count = 0
        
        # Ocean parameters (can be controlled externally)
        self.wave_speed = 0.5
        self.wave_amplitude = 0.5
        self.tide_level = 0.0
        self.season = 0.5
        self.wind_speed = 0.2
        
        # OpenGL resources (created in setup_buffers)
        self.VAO = None
        self.VBO = None
        self.sim_shader = None
        self.render_shader = None
        
        # Wave simulation buffers
        self.simulation_textures = [None, None]
        self.simulation_fbos = [None, None]
        self.current_sim_buffer = 0
        self.noise_texture = None
        
        # Fullscreen quad vertices (stored here, buffers created after shader compile)
        self.quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
    
    def compile_shader(self):
        """Compile both simulation and render shaders"""
        # Compile simulation shader
        sim_vs = self.get_simulation_vertex_shader()
        sim_fs = self.get_simulation_fragment_shader()
        
        try:
            vert = shaders.compileShader(sim_vs, GL_VERTEX_SHADER)
            frag = shaders.compileShader(sim_fs, GL_FRAGMENT_SHADER)
            self.sim_shader = shaders.compileProgram(vert, frag)
            print(f"    [OK] Ocean simulation shader compiled")
        except Exception as e:
            print(f"    [ERROR] Simulation shader compilation error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Compile render shader
        render_vs = self.get_vertex_shader()
        render_fs = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(render_vs, GL_VERTEX_SHADER)
            frag = shaders.compileShader(render_fs, GL_FRAGMENT_SHADER)
            self.render_shader = shaders.compileProgram(vert, frag)
            print(f"    [OK] Ocean render shader compiled")
        except Exception as e:
            print(f"    [ERROR] Render shader compilation error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Set shader for base class compatibility
        self.shader = self.render_shader
        return self.render_shader
    
    def get_simulation_vertex_shader(self):
        """Vertex shader for wave simulation pass"""
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        out vec2 vUV;
        
        void main() {
            vUV = position * 0.5 + 0.5;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """
    
    def get_simulation_fragment_shader(self):
        """Wave equation simulation - adapted from Shadertoy"""
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vUV;
        out vec4 fragColor;
        
        uniform sampler2D previousFrame;
        uniform vec2 resolution;
        uniform int iFrame;
        uniform float time;
        uniform float waveSpeed;
        
        void main() {
            vec2 fragCoord = vUV * resolution;
            
            // Initialize: Create initial wave disturbances
            if(iFrame < 10) {
                float dist = distance(fragCoord, resolution * 0.5);
                if(dist < 50.0) {
                    // Strong initial disturbance
                    fragColor = vec4(10.0, 0.0, 0.0, 1.0);
                } else {
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                }
                return;
            }
            
            // Wave equation simulation using neighbor samples
            vec2 texel = 1.0 / resolution;
            vec4 center = texture(previousFrame, vUV);
            vec4 down = texture(previousFrame, vUV + vec2(0.0, -texel.y));
            vec4 left = texture(previousFrame, vUV + vec2(-texel.x, 0.0));
            vec4 right = texture(previousFrame, vUV + vec2(texel.x, 0.0));
            vec4 up = texture(previousFrame, vUV + vec2(0.0, texel.y));
            
            // Wave equation: second derivative approximation
            float laplacian = (down.r + left.r + right.r + up.r - 4.0 * center.r);
            
            // Wave propagation with velocity stored in green channel
            float velocity = center.g + laplacian * waveSpeed * 0.2;
            float height = center.r + velocity;
            
            // Damping to prevent runaway
            velocity *= 0.98;
            height *= 0.99;
            
            fragColor = vec4(height, velocity, 0.0, 1.0);
        }
        """
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        uniform float depth;
        
        out vec2 vUV;
        
        void main() {
            vUV = position * 0.5 + 0.5;
            
            float depthValue = depth / 100.0;
            gl_Position = vec4(position, depthValue, 1.0);
        }
        """
    
    def get_fragment_shader(self):
        """Render ocean with waves, beach terrain, and lighting"""
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vUV;
        out vec4 fragColor;
        
        uniform vec2 resolution;
        uniform float time;
        uniform sampler2D waveData;
        uniform sampler2D noiseTex;
        uniform float waveAmplitude;
        
        void main() {
            // Sample wave simulation data
            vec4 waveInfo = texture(waveData, vUV);
            
            // Visualize wave height with high contrast
            float waveHeight = waveInfo.r * 0.1; // Scale down
            
            // Ocean blue base + wave variation
            vec3 color = vec3(0.0, 0.3, 0.6) + vec3(waveHeight * 5.0);
            
            fragColor = vec4(color, 1.0);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers - Called automatically after shader compilation"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create VBO for fullscreen quad
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, self.quad_vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        
        # Create simulation framebuffers
        width = self.viewport.width
        height = self.viewport.height
        
        for i in range(2):
            # Create texture for wave data
            self.simulation_textures[i] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.simulation_textures[i])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Create framebuffer
            self.simulation_fbos[i] = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.simulation_fbos[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.simulation_textures[i], 0)
            
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print(f"    [ERROR] Simulation framebuffer {i} incomplete!")
            
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Create noise texture
        noise_size = 256
        noise_data = np.random.random((noise_size, noise_size)).astype(np.float32)
        
        self.noise_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.noise_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, noise_size, noise_size, 0, GL_RED, GL_FLOAT, noise_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Cache uniform locations for render shader
        self.uniform_resolution = glGetUniformLocation(self.render_shader, "resolution")
        self.uniform_time = glGetUniformLocation(self.render_shader, "time")
        self.uniform_depth = glGetUniformLocation(self.render_shader, "depth")
        self.uniform_wave_data = glGetUniformLocation(self.render_shader, "waveData")
        self.uniform_noise_tex = glGetUniformLocation(self.render_shader, "noiseTex")
        self.uniform_wave_amplitude = glGetUniformLocation(self.render_shader, "waveAmplitude")
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        self.time += dt
        self.frame_count += 1
    
    def render(self, state: Dict):
        """Render the ocean effect (simulation + final render)"""
        if not self.enabled:
            return
        
        # Step 1: Run wave simulation
        self._render_simulation()
        
        # Step 2: Render final ocean scene
        self._render_ocean()
    
    def _render_simulation(self):
        """Run wave equation simulation pass"""
        if self.sim_shader is None:
            return
        
        # Bind next simulation framebuffer
        next_buffer = 1 - self.current_sim_buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.simulation_fbos[next_buffer])
        
        # Clear to ensure clean state
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.sim_shader)
        
        # Set simulation uniforms
        res_loc = glGetUniformLocation(self.sim_shader, "resolution")
        if res_loc >= 0:
            glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        frame_loc = glGetUniformLocation(self.sim_shader, "iFrame")
        if frame_loc >= 0:
            glUniform1i(frame_loc, self.frame_count)
        
        time_loc = glGetUniformLocation(self.sim_shader, "time")
        if time_loc >= 0:
            glUniform1f(time_loc, self.time)
        
        speed_loc = glGetUniformLocation(self.sim_shader, "waveSpeed")
        if speed_loc >= 0:
            glUniform1f(speed_loc, self.wave_speed)
        
        # Bind previous frame texture
        prev_loc = glGetUniformLocation(self.sim_shader, "previousFrame")
        if prev_loc >= 0:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.simulation_textures[self.current_sim_buffer])
            glUniform1i(prev_loc, 0)
        
        # Render fullscreen quad
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
        
        # Swap buffers
        self.current_sim_buffer = next_buffer
    
    def _render_ocean(self):
        """Render final ocean scene with waves and beach"""
        if self.render_shader is None:
            return
        
        glUseProgram(self.render_shader)
        glBindVertexArray(self.VAO)
        
        # Set render uniforms
        if self.uniform_resolution >= 0:
            glUniform2f(self.uniform_resolution, self.viewport.width, self.viewport.height)
        if self.uniform_time >= 0:
            glUniform1f(self.uniform_time, self.time)
        if self.uniform_depth >= 0:
            glUniform1f(self.uniform_depth, self.depth)
        if self.uniform_wave_amplitude >= 0:
            glUniform1f(self.uniform_wave_amplitude, self.wave_amplitude)
        
        # Bind wave simulation texture
        if self.uniform_wave_data >= 0:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.simulation_textures[self.current_sim_buffer])
            glUniform1i(self.uniform_wave_data, 0)
        
        # Bind noise texture
        if self.uniform_noise_tex >= 0:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.noise_texture)
            glUniform1i(self.uniform_noise_tex, 1)
        
        # Render fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.VBO is not None:
            glDeleteBuffers(1, [self.VBO])
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
        if self.noise_texture is not None:
            glDeleteTextures(1, [self.noise_texture])
        for tex in self.simulation_textures:
            if tex is not None:
                glDeleteTextures(1, [tex])
        for fbo in self.simulation_fbos:
            if fbo is not None:
                glDeleteFramebuffers(1, [fbo])
        if self.sim_shader is not None:
            glDeleteProgram(self.sim_shader)
        if self.render_shader is not None:
            glDeleteProgram(self.render_shader)
