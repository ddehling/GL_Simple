"""
Bioluminescence shader effect - Glowing particles in ocean waters
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_bioluminescence(state, outstate, intensity=1.0, depth=50.0):
    """
    Shader-based bioluminescence effect compatible with EventScheduler
    
    Creates glowing bioluminescent particles floating through the water,
    like plankton, jellyfish, or other marine organisms.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_bioluminescence, intensity=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        intensity: Base brightness multiplier (default: 1.0)
        depth: Z-depth of particles (0=near, 100=far, default: 50)
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
        print(f"Initializing bioluminescence for frame {frame_id}")
        
        try:
            viewport.add_effect(BioluminescenceEffect, intensity=intensity, depth=depth)
            # Get the effect that was just added
            state['effect'] = viewport.effects[-1]
            # Set initial fade to 0 so it starts invisible
            state['effect'].fade_factor = 0.0
            print(f"Bioluminescence effect initialized successfully")
        except Exception as e:
            print(f"ERROR initializing bioluminescence effect: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from global state - read ocean weather parameters
    if 'effect' in state:
        # Use ocean weather set parameter names
        state['effect'].target_bio_level = outstate.get('bioluminescence', 0.0)
        state['effect'].wave_speed = outstate.get('wave_speed', 0.5)
        state['effect'].current_strength = outstate.get('wind', 0.0)
        state['effect'].tide_level = outstate.get('tide_level', 0.5)
        state['effect'].season = outstate.get('season', 0.5)
        
        # Implement fade in/out
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        fade_duration = 2.0  # 2 second fade
        
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = float(np.clip(fade_factor, 0, 1))
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"Bioluminescence effect cleaned up for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class BioluminescenceEffect(ShaderEffect):
    """
    Bioluminescent particles floating in water
    
    Creates glowing spots that drift through the water column with
    pulsing, flickering light like real bioluminescent organisms.
    """
    
    def __init__(self, viewport, intensity: float = 1.0, depth: float = 50.0):
        super().__init__(viewport)
        self.depth = depth
        self.base_intensity = intensity
        
        # Ocean parameters (updated from global state)
        self.target_bio_level = 0.0
        self.smoothed_bio_level = 0.0
        self.wave_speed = 0.5
        self.current_strength = 0.0
        self.tide_level = 0.5
        self.season = 0.5
        self.fade_factor = 0.0
        self.time = 0.0
        
        # Particle parameters
        self.max_particles = 1000
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.particle_data_VBO = None
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """Initialize bioluminescent particle positions and properties"""
        width = self.viewport.width
        height = self.viewport.height
        
        # Generate particle positions (randomly distributed in water only, not beach)
        np.random.seed(123)  # Consistent positions
        # Keep particles in bottom 85% of screen (water region, avoiding beach at top)
        self.particle_positions = np.column_stack([
            np.random.uniform(0, width, self.max_particles),
            np.random.uniform(height * 0.15, height, self.max_particles),  # Start at 15% from top
        ]).astype(np.float32)
        
        # Particle properties: [size, pulse_phase, pulse_speed, color_hue]
        self.particle_properties = np.column_stack([
            np.random.uniform(2.0, 8.0, self.max_particles),  # size
            np.random.uniform(0, 2 * np.pi, self.max_particles),  # pulse phase
            np.random.uniform(0.5, 2.0, self.max_particles),  # pulse speed
            np.random.uniform(0, 1.0, self.max_particles),  # color variation
        ]).astype(np.float32)
        
        # Create point sprite vertices (each particle is a point)
        self.vertices = np.zeros((self.max_particles, 2), dtype=np.float32)
        
        # Particle data: [particle_id]
        self.particle_data = np.arange(self.max_particles, dtype=np.float32).reshape(-1, 1)
    
    def compile_shader(self):
        """Compile and link bioluminescence shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            return shaders.compileProgram(
                shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in float particleId;
        
        uniform vec2 resolution;
        uniform float depth;
        uniform float time;
        uniform int maxParticles;
        uniform float bioLevel;
        uniform float waveSpeed;
        uniform float currentStrength;
        uniform float tideLevel;
        uniform float season;  // Time of day: 0=midnight, 0.5=noon
        
        out float vBrightness;
        out vec3 vColor;
        out float vParticleAlpha;
        
        // Hash function for randomness
        float hash(float n) {
            return fract(sin(n) * 43758.5453123);
        }
        
        float hash2(vec2 p) {
            p = fract(p * vec2(443.897, 441.423));
            p += dot(p, p + vec2(19.19, 23.41));
            return fract(sin(p.x * p.y) * 43758.5453);
        }
        
        void main() {
            int pid = int(particleId);
            float pidFloat = float(pid);
            
            // Get particle properties
            vec2 basePos = vec2(
                hash2(vec2(pidFloat * 0.1, 1.0)) * resolution.x,
                hash2(vec2(pidFloat * 0.1, 2.0)) * resolution.y * 0.85 + resolution.y * 0.15
            );
            
            float size = 3.0 + hash(pidFloat * 1.5) * 9.0;
            float pulsePhase = hash(pidFloat * 2.1) * 6.28318;
            float pulseSpeed = 0.5 + hash(pidFloat * 2.7) * 1.5;
            float colorHue = hash(pidFloat * 3.3);
            
            // Each particle has a threshold for visibility
            float particleThreshold = hash(pidFloat * 4.4);
            
            // Fade in/out based on bioLevel
            float fadeWidth = 0.15;
            float fadeStart = max(0.0, particleThreshold - fadeWidth);
            float fadeEnd = particleThreshold + fadeWidth;
            vParticleAlpha = smoothstep(fadeStart, fadeEnd, bioLevel);
            
            // Cull invisible particles
            if (vParticleAlpha < 0.01 || bioLevel < 0.001) {
                gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
                gl_PointSize = 0.0;
                return;
            }
            
            // Drift motion - particles float through water
            float driftSpeed = hash(pidFloat * 5.5) * 0.5 + 0.5;
            vec2 pos = basePos;
            
            // Calculate water region based on tide level
            // Match ocean_waves shader beach calculation exactly
            // Tide level: 0=low, 0.5=normal, 1.0=high
            float tideFactor = (tideLevel - 0.5) * 0.3;  // Map 0-1 to -0.15 to +0.15
            float beachStartNorm = clamp(0.92 + tideFactor, 0.0, 0.99);  // Normalized Y (0=bottom, 1=top after inversion)
            
            // Convert to screen coordinates (0=top, height=bottom)
            // In ocean_waves: normY=1.0 is at screen y=0 (top), normY=0.0 is at screen y=height (bottom)
            // So: screen_y = height * (1.0 - normY)
            // Beach starts at: screen_y = height * (1.0 - beachStartNorm)
            float waterTop = resolution.y * (1.0 - beachStartNorm);
            float waterBottom = resolution.y;
            float waterHeight = waterBottom - waterTop;
            
            // Vertical drift (slow rise like plankton) - constrain to water region
            pos.y -= time * 15.0 * driftSpeed * waveSpeed;
            
            // Wrap within water region (adjusted for tide)
            pos.y = waterTop + mod(pos.y - waterTop, waterHeight);
            
            // Cull particles that ended up in beach zone
            if (pos.y < waterTop) {
                gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
                gl_PointSize = 0.0;
                return;
            }
            
            // Horizontal current drift
            pos.x += currentStrength * time * 5.0;
            pos.x += sin(time * 0.5 + pidFloat * 0.1) * 10.0 * waveSpeed;
            
            // Wrap horizontally
            pos.x = mod(pos.x, resolution.x);
            
            // Pulsing brightness
            float pulse = sin(time * pulseSpeed + pulsePhase) * 0.5 + 0.5;
            pulse = pow(pulse, 2.0);  // Sharper pulses
            
            // Add flicker
            float flicker = hash(time * 10.0 + pidFloat) * 0.3 + 0.7;
            
            // Night brightness - bright at night, dim during day
            // season: 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset
            float dayNightCycle = cos(season * 2.0 * 3.14159);
            float nightBrightness = 0.3 + dayNightCycle * 1.2;  // midnight=1.5, noon=0.3
            
            vBrightness = pulse * flicker * nightBrightness;
            
            // Color - blue-green bioluminescence
            if (colorHue < 0.4) {
                vColor = vec3(0.2, 0.6, 1.0);  // Cyan-blue
            } else if (colorHue < 0.7) {
                vColor = vec3(0.3, 1.0, 0.7);  // Green-cyan
            } else {
                vColor = vec3(0.5, 0.8, 1.0);  // Light blue
            }
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Depth
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, mappedDepth, 1.0);
            gl_PointSize = size * (vBrightness * 0.5 + 0.5);  // Size varies with brightness
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in float vBrightness;
        in vec3 vColor;
        in float vParticleAlpha;
        
        uniform float fadeAlpha;
        uniform float bioIntensity;
        
        out vec4 outColor;
        
        void main() {
            // Create circular glow
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            // Soft glow falloff
            float glow = 1.0 - smoothstep(0.0, 0.5, dist);
            glow = pow(glow, 1.5);
            
            // Bright core
            float core = 1.0 - smoothstep(0.0, 0.2, dist);
            core = pow(core, 3.0);
            
            float brightness = (glow * 0.8 + core * 0.6) * vBrightness * bioIntensity;
            
            vec3 color = vColor * brightness;
            
            // Alpha - glowing particles are semi-transparent
            float alpha = (glow * 0.7 + core * 0.8) * vBrightness;
            alpha *= vParticleAlpha;
            alpha *= fadeAlpha;
            
            alpha = clamp(alpha, 0.0, 1.0);
            
            outColor = vec4(color, alpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Generate VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Position VBO (not actually used, particles positioned in shader)
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Particle data VBO
        self.particle_data_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_data_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.particle_data.nbytes, self.particle_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        self.time += dt
        
        # Smooth bio level transitions
        smoothing_speed = 0.5
        self.smoothed_bio_level += (self.target_bio_level - self.smoothed_bio_level) * smoothing_speed * dt
        self.bio_level = self.smoothed_bio_level
    
    def render(self, state: Dict):
        """Render bioluminescent particles"""
        if self.shader is None or self.VAO is None:
            return
        
        # Transparent particles don't write to depth buffer
        glDepthMask(GL_FALSE)
        
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniform2f(glGetUniformLocation(self.shader, b"resolution"),
                    self.viewport.width, self.viewport.height)
        glUniform1f(glGetUniformLocation(self.shader, b"depth"), self.depth)
        glUniform1f(glGetUniformLocation(self.shader, b"time"), self.time)
        glUniform1i(glGetUniformLocation(self.shader, b"maxParticles"), self.max_particles)
        glUniform1f(glGetUniformLocation(self.shader, b"bioLevel"), self.bio_level)
        glUniform1f(glGetUniformLocation(self.shader, b"waveSpeed"), self.wave_speed)
        glUniform1f(glGetUniformLocation(self.shader, b"currentStrength"), self.current_strength)
        glUniform1f(glGetUniformLocation(self.shader, b"tideLevel"), self.tide_level)
        glUniform1f(glGetUniformLocation(self.shader, b"season"), self.season)
        glUniform1f(glGetUniformLocation(self.shader, b"fadeAlpha"), self.fade_factor)
        glUniform1f(glGetUniformLocation(self.shader, b"bioIntensity"), self.base_intensity)
        
        # Render particles as points
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_POINTS, 0, self.max_particles)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Restore depth writes
        glDepthMask(GL_TRUE)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
        if self.position_VBO is not None:
            glDeleteBuffers(1, [self.position_VBO])
        if self.particle_data_VBO is not None:
            glDeleteBuffers(1, [self.particle_data_VBO])
        if self.shader is not None:
            glDeleteProgram(self.shader)
