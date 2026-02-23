"""
Pond ripples shader effect - Interactive water wave simulation
Simulates realistic ripples on a pond surface using the 2D wave equation
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_pond_ripples(state, outstate, depth=30.0, intensity=1.0, 
                        damping=0.98, wave_speed=1.5, ripple_frequency=0.5,
                        bass_sensitivity=1.2, water_color=None):
    """
    Shader-based pond ripple effect compatible with EventScheduler
    
    Creates realistic water ripples using 2D wave equation simulation.
    Ripples spawn randomly and respond to audio (bass creates bigger ripples).
    
    Usage:
        scheduler.schedule_event(0, 60, shader_pond_ripples, 
                               damping=0.98, wave_speed=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        depth: Z-depth of pond surface (0=near, 100=far, default: 30)
        intensity: Visual brightness multiplier (default: 1.0)
        damping: Wave damping factor 0-1 (default: 0.98, higher = less damping)
        wave_speed: Ripple propagation speed (default: 1.5)
        ripple_frequency: How often new ripples spawn per second (default: 0.5)
        bass_sensitivity: How much bass frequencies trigger ripples (default: 1.2)
        water_color: RGB tuple for water color (default: None for blue-green)
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
        print(f"Initializing pond ripples for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                PondRipplesEffect,
                depth=depth,
                intensity=intensity,
                damping=damping,
                wave_speed=wave_speed,
                ripple_frequency=ripple_frequency,
                bass_sensitivity=bass_sensitivity,
                water_color=water_color
            )
            state['effect'] = effect
            print(f"✓ Initialized shader pond_ripples for frame {frame_id}")
        except Exception as e:
            import traceback
            print(f"✗ Failed to initialize pond_ripples: {e}")
            traceback.print_exc()
            return
    
    # Update from audio data and global state
    if 'effect' in state:
        audio_data = outstate.get('sound')
        
        # Process audio data if available
        if audio_data is not None:
            # Use norm_long_relu which highlights above-average activity (already time-averaged)
            # This avoids double averaging - norm_long_relu = ReLU(norm_long - 1)
            bands_relu = audio_data['norm_long_relu'][0]
            
            # Extract frequency ranges - these values are 0 when at/below average,
            # and positive when above average (no additional smoothing needed)
            bass_energy = np.mean(bands_relu[0:8])      # Bass: 40-300 Hz (ripple triggers)
            mid_energy = np.mean(bands_relu[8:20])      # Mids: 300-2000 Hz (wave intensity)
            high_energy = np.mean(bands_relu[20:32])    # Highs: 2000-16000 Hz (shimmer)
            
            state['effect'].audio_bass = bass_energy
            state['effect'].audio_mid = mid_energy
            state['effect'].audio_high = high_energy
            
            # Overall loudness level (above average)
            long_term_level = np.mean(bands_relu)
            state['effect'].audio_long_term_level = long_term_level
        else:
            state['effect'].audio_bass = 0.0
            state['effect'].audio_mid = 0.0
            state['effect'].audio_high = 0.0
            state['effect'].audio_long_term_level = 0.0
        
        # Update from global state (optional)
        state['effect'].base_intensity = outstate.get('ripple_intensity', intensity)
        
        # Implement fade in/out
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        fade_duration = 3.0
        
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = np.clip(fade_factor, 0, 1)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            del state['effect']
            print(f"✓ Cleaned up shader pond_ripples for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class PondRipplesEffect(ShaderEffect):
    """
    Pond ripple water simulation effect
    
    Simulates realistic 2D wave propagation using the wave equation
    with interactive ripple spawning and audio reactivity.
    """
    
    def __init__(self, viewport, depth: float = 30.0, intensity: float = 1.0,
                 damping: float = 0.92, wave_speed: float = 1.5,
                 ripple_frequency: float = 0.5, bass_sensitivity: float = 3.0,
                 water_color=None):
        super().__init__(viewport)
        self.depth = depth
        self.base_intensity = intensity
        self.damping = damping
        self.wave_speed = wave_speed * 0.3  # Scale down for stability
        self.ripple_frequency = ripple_frequency * 0.3  # Reduce baseline automatic ripples
        self.bass_sensitivity = bass_sensitivity
        self.water_color = water_color if water_color else (0.2, 0.5, 0.8)
        
        self.time = 0.0
        self.fade_factor = 0.0
        self.ripple_timer = 0.0
        
        # Audio reactivity
        self.audio_bass = 0.0
        self.audio_mid = 0.0
        self.audio_high = 0.0
        self.audio_long_term_level = 0.0  # Long-term average for silence detection
        self.last_bass = 0.0
        
        # Framebuffer objects for wave simulation (ping-pong buffers)
        self.buffer_current_FBO = None
        self.buffer_previous_FBO = None
        
        self.buffer_current_tex = None
        self.buffer_previous_tex = None
        
        # Shader programs
        self.shader_wave = None  # Wave equation solver
        self.shader_image = None  # Final display
        
        # Quad for fullscreen rendering
        self.quad_VAO = None
        self.quad_VBO = None
        
        self.frame_count = 0
        self.swap_buffers = False  # For ping-pong
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize quad mesh for fullscreen passes"""
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        
        self.quad_vertices = quad_vertices
    
    def compile_shader(self):
        """Compile all shader passes"""
        try:
            # Wave equation solver shader
            vert_wave = self.get_vertex_shader_fullscreen()
            frag_wave = self.get_fragment_shader_wave()
            self.shader_wave = self._compile_program(vert_wave, frag_wave)
            print(f"✓ Compiled wave shader (program: {self.shader_wave})")
            
            # Final display shader
            vert_image = self.get_vertex_shader_display()
            frag_image = self.get_fragment_shader_image()
            self.shader_image = self._compile_program(vert_image, frag_image)
            print(f"✓ Compiled image shader (program: {self.shader_image})")
            
            # Return main shader (required by base class)
            return self.shader_image
            
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def _compile_program(self, vert_src, frag_src):
        """Helper to compile a shader program"""
        try:
            vert = shaders.compileShader(vert_src, GL_VERTEX_SHADER)
        except Exception as e:
            print(f"Vertex shader compilation error: {e}")
            print("Vertex source:")
            print(vert_src)
            raise
        
        try:
            frag = shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
        except Exception as e:
            print(f"Fragment shader compilation error: {e}")
            print("Fragment source:")
            print(frag_src)
            raise
        
        try:
            program = shaders.compileProgram(vert, frag)
            return program
        except Exception as e:
            print(f"Shader program linking error: {e}")
            raise
    
    def get_vertex_shader_fullscreen(self):
        """Vertex shader for fullscreen quad buffer passes"""
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 fragCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragCoord = (position + 1.0) * 0.5;
        }
        """
    
    def get_vertex_shader_display(self):
        """Vertex shader for final display with depth"""
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        uniform float depth;
        
        out vec2 fragCoord;
        
        void main() {
            // Map depth to clip space z (0-100 -> 0.0-1.0)
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(position, mappedDepth, 1.0);
            fragCoord = (position + 1.0) * 0.5;
        }
        """
    
    def get_fragment_shader_wave(self):
        """Wave equation solver - implements 2D wave propagation"""
        return """
        #version 310 es
        precision highp float;
        
        uniform vec2 iResolution;
        uniform float iTime;
        uniform float iTimeDelta;
        uniform sampler2D iChannel0;  // Current wave heights
        uniform sampler2D iChannel1;  // Previous wave heights
        uniform float damping;
        uniform float waveSpeed;
        uniform vec3 ripplePos;       // x, y, strength (0 = no new ripple)
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        void main() {
            vec2 uv = fragCoord;
            vec2 pixelSize = 1.0 / iResolution;
            
            // Sample current and previous wave heights
            float current = texture(iChannel0, uv).r;
            float previous = texture(iChannel1, uv).r;
            
            // Sample neighbors for Laplacian (5-point stencil)
            float left = texture(iChannel0, uv + vec2(-pixelSize.x, 0.0)).r;
            float right = texture(iChannel0, uv + vec2(pixelSize.x, 0.0)).r;
            float up = texture(iChannel0, uv + vec2(0.0, -pixelSize.y)).r;
            float down = texture(iChannel0, uv + vec2(0.0, pixelSize.y)).r;
            
            // Diagonal samples for better accuracy
            float upleft = texture(iChannel0, uv + vec2(-pixelSize.x, -pixelSize.y)).r;
            float upright = texture(iChannel0, uv + vec2(pixelSize.x, -pixelSize.y)).r;
            float downleft = texture(iChannel0, uv + vec2(-pixelSize.x, pixelSize.y)).r;
            float downright = texture(iChannel0, uv + vec2(pixelSize.x, pixelSize.y)).r;
            
            // 9-point Laplacian for smoother waves
            float laplacian = (left + right + up + down) * 0.5 + 
                             (upleft + upright + downleft + downright) * 0.25 - 
                             current * 4.0;
            
            // Wave equation with proper time stepping
            // Using Verlet integration: x_new = 2*x_current - x_previous + acceleration * dt^2
            float dt = 0.016;  // Fixed timestep for stability (60 FPS)
            float velocity2 = waveSpeed * waveSpeed;
            float newHeight = 2.0 * current - previous + velocity2 * laplacian * dt * dt;
            
            // Apply damping (energy loss)
            newHeight *= damping;
            
            // Add new ripple if ripplePos.z > 0
            if (ripplePos.z > 0.0) {
                vec2 pos = uv * iResolution;
                float dist = length(pos - ripplePos.xy);
                
                // Smooth ripple pulse
                float rippleRadius = 30.0;
                float rippleStrength = ripplePos.z;
                float ripple = rippleStrength * exp(-dist * dist / (rippleRadius * rippleRadius));
                newHeight += ripple;
            }
            
            // Clamp to prevent overflow
            newHeight = clamp(newHeight, -2.0, 2.0);
            
            outColor = vec4(newHeight, 0.0, 0.0, 1.0);
        }
        """
    
    def get_fragment_shader_image(self):
        """Final display shader with water rendering and depth"""
        return """
        #version 310 es
        precision highp float;
        
        uniform vec2 iResolution;
        uniform sampler2D iChannel0;  // Wave heights
        uniform float fadeAlpha;
        uniform float intensity;
        uniform vec3 waterColor;
        uniform float iTime;
        uniform float audioBass;
        uniform float audioMid;
        uniform float audioHigh;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        // Simple hash for noise
        float hash(vec2 p) {
            p = fract(p * vec2(123.34, 456.21));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
        }
        
        // Smooth noise
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        // Fractal noise
        float fbm(vec2 p) {
            float value = 0.0;
            float amplitude = 0.5;
            for (int i = 0; i < 4; i++) {
                value += amplitude * noise(p);
                p *= 2.0;
                amplitude *= 0.5;
            }
            return value;
        }
        
        void main() {
            vec2 uv = fragCoord;
            vec2 pixelSize = 1.0 / iResolution;
            
            // Sample wave height
            float height = texture(iChannel0, uv).r;
            
            // Compute gradient for normal mapping (lighting effect)
            float left = texture(iChannel0, uv + vec2(-pixelSize.x, 0.0)).r;
            float right = texture(iChannel0, uv + vec2(pixelSize.x, 0.0)).r;
            float up = texture(iChannel0, uv + vec2(0.0, -pixelSize.y)).r;
            float down = texture(iChannel0, uv + vec2(0.0, pixelSize.y)).r;
            
            vec2 gradient = vec2(right - left, down - up);
            
            // Almost uniform background - tiny random variation only
            float depth = 0.85 + noise(uv * 50.0 + iTime * 0.01) * 0.15;
            
            // Normal from gradient
            vec3 normal = normalize(vec3(-gradient * 20.0, 1.0));
            
            // Lighting
            vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
            float diffuse = max(dot(normal, lightDir), 0.0);
            
            vec3 reflectDir = reflect(-lightDir, vec3(0.0, 0.0, 1.0));
            float specular = pow(max(dot(reflectDir, normal), 0.0), 32.0);
            
            // Clean water color - almost uniform
            vec3 color = waterColor * depth * (0.6 + 0.4 * diffuse);
            color += specular * 0.5;
            color += waterColor * height * 0.4;
            
            // Ripple rings emanating from wave peaks
            float distFromPeak = length(gradient);
            float ringPattern = fract(distFromPeak * 15.0 - iTime * 3.0);
            float rings = smoothstep(0.4, 0.5, ringPattern) - smoothstep(0.5, 0.6, ringPattern);
            
            // Only show rings where there's significant wave activity
            float waveStrength = abs(height);
            rings *= smoothstep(0.05, 0.2, waveStrength);
            
            // Add rings to color (white/light blue)
            color += vec3(0.9, 0.95, 1.0) * rings * 0.6;
            
            // Apply intensity
            color *= intensity;
            
            // Calculate alpha based on wave activity and base transparency
            float waveActivity = abs(height) * 2.0 + length(gradient) * 1.0;
            float alpha = mix(0.6, 0.9, clamp(waveActivity * 2.0, 0.0, 1.0));
            
            // Apply fade factor
            alpha *= fadeAlpha;
            
            outColor = vec4(color, alpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers and framebuffers"""
        print(f"Setting up pond ripples buffers (viewport: {self.viewport.width}x{self.viewport.height})")
        
        # Setup quad VAO
        self.quad_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.quad_VAO)
        
        self.quad_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, self.quad_vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        print(f"✓ Quad VAO/VBO created")
        
        # Create framebuffer textures (ping-pong buffers for wave simulation)
        width = self.viewport.width
        height = self.viewport.height
        
        self.buffer_current_tex = self._create_texture(width, height)
        self.buffer_previous_tex = self._create_texture(width, height)
        print(f"✓ Created 2 wave simulation textures ({width}x{height})")
        
        # Create framebuffers
        self.buffer_current_FBO = self._create_framebuffer(self.buffer_current_tex)
        self.buffer_previous_FBO = self._create_framebuffer(self.buffer_previous_tex)
        print(f"✓ Created 2 framebuffers")
        
        # Initialize textures to zero (calm water)
        self._clear_texture(self.buffer_current_FBO)
        self._clear_texture(self.buffer_previous_FBO)
        print(f"✓ Cleared wave buffers to calm state")
    
    def _create_texture(self, width, height):
        """Create a framebuffer texture for wave heights"""
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        # Use R32F for single-channel float (wave height)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        return tex
    
    def _create_framebuffer(self, texture):
        """Create a framebuffer object"""
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer incomplete: {status}")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return fbo
    
    def _clear_texture(self, fbo):
        """Clear a framebuffer texture to zero"""
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        self.time += dt
        self.frame_count += 1
        self.ripple_timer += dt
        
        # No additional smoothing needed - norm_long_relu is already time-averaged
        # and highlights above-average activity directly
        
        # Detect bass hits - audio_bass is now from norm_long_relu (0 = at/below average)
        bass_threshold = 0.1 * self.bass_sensitivity  # Higher threshold for fewer ripples
        # Trigger when bass is significantly above average AND increasing substantially
        if self.audio_bass > bass_threshold and self.audio_bass > self.last_bass + 0.08:
            # Bass hit - spawn ripple (reduced strength multiplier)
            self._spawn_ripple(strength=self.audio_bass * 2.0)
        
        self.last_bass = self.audio_bass
    
    def _spawn_ripple(self, strength=0.3):
        """Spawn a new ripple at a random location"""
        # Random position
        x = np.random.uniform(self.viewport.width * 0.2, self.viewport.width * 0.8)
        y = np.random.uniform(self.viewport.height * 0.2, self.viewport.height * 0.8)
        
        # Store for next render call - scale strength for visibility
        self.pending_ripple = (x, y, strength * 2.0)
    
    def render(self, state: Dict):
        """Render the pond ripples effect with wave simulation"""
        if not self.enabled:
            return
        
        # Save current state
        current_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        
        # Disable depth testing for buffer passes
        depth_test_enabled = glIsEnabled(GL_DEPTH_TEST)
        if depth_test_enabled:
            glDisable(GL_DEPTH_TEST)
        
        # Determine which buffers to use (ping-pong)
        if self.swap_buffers:
            read_current = self.buffer_previous_tex
            read_previous = self.buffer_current_tex
            write_fbo = self.buffer_current_FBO
            display_tex = self.buffer_current_tex
        else:
            read_current = self.buffer_current_tex
            read_previous = self.buffer_previous_tex
            write_fbo = self.buffer_previous_FBO
            display_tex = self.buffer_previous_tex
        
        # Check for pending ripple (from audio triggers only)
        ripple_pos = (0.0, 0.0, 0.0)
        if hasattr(self, 'pending_ripple'):
            ripple_pos = self.pending_ripple
            delattr(self, 'pending_ripple')
        
        # Wave simulation pass
        glBindFramebuffer(GL_FRAMEBUFFER, write_fbo)
        glUseProgram(self.shader_wave)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader_wave, "iResolution")
        glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        time_loc = glGetUniformLocation(self.shader_wave, "iTime")
        glUniform1f(time_loc, self.time)
        
        dt_loc = glGetUniformLocation(self.shader_wave, "iTimeDelta")
        glUniform1f(dt_loc, 1.0 / 60.0)  # Assume 60 FPS
        
        # Audio-modulated damping (mid frequencies reduce damping = more active waves)
        # Use raw audio_mid directly (no smoothing needed, already from norm_long_relu)
        audio_damping = self.damping * (1.0 - self.audio_mid * 0.15)
        damping_loc = glGetUniformLocation(self.shader_wave, "damping")
        glUniform1f(damping_loc, audio_damping)
        
        # Audio-modulated wave speed (mid frequencies increase speed)
        audio_speed = self.wave_speed * (1.0 + self.audio_mid * 0.5)
        speed_loc = glGetUniformLocation(self.shader_wave, "waveSpeed")
        glUniform1f(speed_loc, audio_speed)
        
        ripple_loc = glGetUniformLocation(self.shader_wave, "ripplePos")
        glUniform3f(ripple_loc, ripple_pos[0], ripple_pos[1], ripple_pos[2])
        
        # Bind input textures (ping-pong)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, read_current)
        chan0_loc = glGetUniformLocation(self.shader_wave, "iChannel0")
        glUniform1i(chan0_loc, 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, read_previous)
        chan1_loc = glGetUniformLocation(self.shader_wave, "iChannel1")
        glUniform1i(chan1_loc, 1)
        
        # Render quad
        glBindVertexArray(self.quad_VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
        
        # Swap ping-pong buffers
        self.swap_buffers = not self.swap_buffers
        
        # Re-enable depth test for final pass
        if depth_test_enabled:
            glEnable(GL_DEPTH_TEST)
        
        # Final display pass with depth
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo)
        glUseProgram(self.shader_image)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader_image, "iResolution")
        glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        depth_loc = glGetUniformLocation(self.shader_image, "depth")
        glUniform1f(depth_loc, self.depth)
        
        fade_loc = glGetUniformLocation(self.shader_image, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        intensity_loc = glGetUniformLocation(self.shader_image, "intensity")
        glUniform1f(intensity_loc, self.base_intensity)
        
        water_color_loc = glGetUniformLocation(self.shader_image, "waterColor")
        glUniform3f(water_color_loc, self.water_color[0], self.water_color[1], self.water_color[2])
        
        time_loc = glGetUniformLocation(self.shader_image, "iTime")
        glUniform1f(time_loc, self.time)
        
        # Audio uniforms for visual effects (use direct values, no smoothing)
        bass_loc = glGetUniformLocation(self.shader_image, "audioBass")
        glUniform1f(bass_loc, self.audio_bass)
        
        mid_loc = glGetUniformLocation(self.shader_image, "audioMid")
        glUniform1f(mid_loc, self.audio_mid)
        
        high_loc = glGetUniformLocation(self.shader_image, "audioHigh")
        glUniform1f(high_loc, self.audio_high)
        
        # Bind wave texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, display_tex)
        chan0_loc = glGetUniformLocation(self.shader_image, "iChannel0")
        glUniform1i(chan0_loc, 0)
        
        # Render quad
        glBindVertexArray(self.quad_VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.quad_VAO:
            glDeleteVertexArrays(1, [self.quad_VAO])
        if self.quad_VBO:
            glDeleteBuffers(1, [self.quad_VBO])
        if self.buffer_current_tex:
            glDeleteTextures(1, [self.buffer_current_tex])
        if self.buffer_previous_tex:
            glDeleteTextures(1, [self.buffer_previous_tex])
        if self.buffer_current_FBO:
            glDeleteFramebuffers(1, [self.buffer_current_FBO])
        if self.buffer_previous_FBO:
            glDeleteFramebuffers(1, [self.buffer_previous_FBO])
        if self.shader_wave:
            glDeleteProgram(self.shader_wave)
        if self.shader_image:
            glDeleteProgram(self.shader_image)
