"""
Audio-reactive fireworks shader - fireworks launch from bottom and explode at random heights
Launches respond to bass frequencies, explosion intensity responds to mid/high frequencies
GPU-accelerated particle system with trails
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
import ctypes

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_fireworks(state, outstate, launch_rate=2.0, bass_sensitivity=1.5, explosion_sensitivity=1.2):
    """
    Audio-reactive fireworks that launch from bottom and explode
    
    Usage:
        scheduler.schedule_event(0, 60, shader_fireworks, 
                               launch_rate=2.0, bass_sensitivity=1.5, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        launch_rate: Average fireworks per second (default 2.0)
        bass_sensitivity: Multiplier for bass-triggered launches (default 1.5)
        explosion_sensitivity: Multiplier for explosion intensity (default 1.2)
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    audio_data = outstate.get('sound')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing fireworks for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                FireworksEffect,
                launch_rate=launch_rate,
                bass_sensitivity=bass_sensitivity,
                explosion_sensitivity=explosion_sensitivity
            )
            state['effect'] = effect
            state['prev_bass'] = 0.0
            print(f"✓ Initialized shader fireworks for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fireworks: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from audio data every frame
    if 'effect' in state and audio_data is not None:
        # Get audio bands - use both norm_short for beats and norm_long_relu for peaks
        bands_short = audio_data['norm_short'][0]
        bands_relu = audio_data['norm_long_relu'][0]  # Only above-average energy
        
        # Extract frequency ranges from short-term (for beat detection)
        bass_energy = np.mean(bands_short[0:8])      # Bass: 40-300 Hz
        mid_energy = np.mean(bands_short[8:20])      # Mids: 300-2000 Hz
        high_energy = np.mean(bands_short[20:32])    # Highs: 2000-16000 Hz
        
        # Extract peak energy from relu (only above-average sounds)
        bass_peak = np.mean(bands_relu[0:8])
        mid_peak = np.mean(bands_relu[8:20])
        high_peak = np.mean(bands_relu[20:32])
        
        # Beat detection: sudden increase in bass OR significant bass peak
        prev_bass = state.get('prev_bass', 0.0)
        bass_delta = bass_energy - prev_bass
        
        # If bass spike detected OR strong bass peak, trigger launch
        # Lowered threshold from 0.3 to 0.2 for more responsive launches
        if bass_delta > 0.2 or bass_peak > 0.4:
            state['effect'].trigger_bass_launch()
        
        state['prev_bass'] = bass_energy
        
        # Update explosion intensity from mid/high frequencies
        # Pass both average energy and peak energy for dynamic range
        state['effect'].update_audio(bass_energy, mid_energy, high_energy, bass_peak, mid_peak, high_peak)
        
        # Optional: Implement fade in/out
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
            print(f"Cleaning up fireworks for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader fireworks for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class FireworksEffect(ShaderEffect):
    """GPU-accelerated fireworks with launch trails and explosions"""
    
    def __init__(self, viewport, launch_rate: float = 2.0, bass_sensitivity: float = 1.5, 
                 explosion_sensitivity: float = 1.2):
        super().__init__(viewport)
        self.launch_rate = launch_rate
        self.bass_sensitivity = bass_sensitivity
        self.explosion_sensitivity = explosion_sensitivity
        self.fade_factor = 0.0
        
        # Particle system parameters
        self.max_fireworks = 50  # Maximum concurrent fireworks
        self.particles_per_firework = 64  # Explosion particles
        self.max_particles = self.max_fireworks * self.particles_per_firework
        
        # Audio reactivity
        self.bass_energy = 0.0
        self.mid_energy = 0.0
        self.high_energy = 0.0
        self.bass_raw = 0.0
        self.mid_raw = 0.0
        self.high_raw = 0.0
        self.bass_launch_triggered = False
        
        # Launch timing
        self.time_since_launch = 0.0
        self.launch_interval = 1.0 / launch_rate
        
        # Firework state arrays (CPU-side, uploaded to GPU)
        # Each firework has: position (x, y, z), velocity (vx, vy), state, color (hue), lifetime
        self.firework_positions = np.zeros((self.max_fireworks, 3), dtype=np.float32)  # x, y, z
        self.firework_velocities = np.zeros((self.max_fireworks, 2), dtype=np.float32)  # vx, vy
        self.firework_states = np.zeros(self.max_fireworks, dtype=np.float32)  # 0=inactive, 1=launching, 2=exploding
        self.firework_colors = np.zeros(self.max_fireworks, dtype=np.float32)  # Hue (0-1)
        self.firework_lifetimes = np.zeros(self.max_fireworks, dtype=np.float32)  # Time since spawn
        self.firework_explosion_heights = np.zeros(self.max_fireworks, dtype=np.float32)  # Y position to explode at
        
        # Particle state (generated from firework data in shader)
        # Each particle knows which firework it belongs to
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize firework data"""
        # All fireworks start inactive
        self.firework_states[:] = 0.0
        
        # Random colors
        self.firework_colors[:] = np.random.random(self.max_fireworks)
    
    def trigger_bass_launch(self):
        """Trigger a firework launch on bass hit"""
        self.bass_launch_triggered = True
    
    def update_audio(self, bass: float, mid: float, high: float, 
                     bass_peak: float = 0.0, mid_peak: float = 0.0, high_peak: float = 0.0):
        """Update audio energy levels"""
        # Use peak values for stronger contrast - they only spike on above-average sounds
        self.bass_energy = bass_peak * self.bass_sensitivity * 2.0  # Double multiplier for peaks
        self.mid_energy = mid_peak * self.explosion_sensitivity * 2.0
        self.high_energy = high_peak * self.explosion_sensitivity * 2.0
        
        # Store raw values for color variation
        self.bass_raw = bass
        self.mid_raw = mid
        self.high_raw = high
    
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
            print(f"FireworksEffect shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Input: particle index (0 to max_particles-1)
        layout(location = 0) in float particleIndex;
        
        // Uniforms
        uniform vec2 resolution;
        uniform float time;
        uniform int maxFireworks;
        uniform int particlesPerFirework;
        
        // Firework data arrays
        uniform vec3 fireworkPositions[50];
        uniform vec2 fireworkVelocities[50];
        uniform float fireworkStates[50];
        uniform float fireworkColors[50];
        uniform float fireworkLifetimes[50];
        uniform float fireworkExplosionHeights[50];
        
        uniform float bassEnergy;
        uniform float midEnergy;
        uniform float highEnergy;
        
        out vec3 vColor;
        out float vAlpha;
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        // Hash function for particle randomness
        float hash(float n) {
            return fract(sin(n) * 43758.5453123);
        }
        
        void main() {
            int particleIdx = int(particleIndex);
            int fireworkIdx = particleIdx / particlesPerFirework;
            int particleOffset = particleIdx % particlesPerFirework;
            
            // Get firework data
            vec3 fwPos = fireworkPositions[fireworkIdx];
            vec2 fwVel = fireworkVelocities[fireworkIdx];
            float fwState = fireworkStates[fireworkIdx];
            float fwHue = fireworkColors[fireworkIdx];
            float fwLifetime = fireworkLifetimes[fireworkIdx];
            float explosionHeight = fireworkExplosionHeights[fireworkIdx];
            
            vec3 position = vec3(0.0);
            float alpha = 0.0;
            vec3 color = vec3(1.0);
            
            // State 0: Inactive - don't render
            if (fwState < 0.5) {
                gl_Position = vec4(0.0, 0.0, -1.0, 1.0);  // Clip away
                gl_PointSize = 0.0;
                vAlpha = 0.0;
                vColor = vec3(0.0);
                return;
            }
            
            // State 1: Launching - draw trail particles
            if (fwState < 1.5) {
                // Only use first few particles as trail
                if (particleOffset > 8) {
                    gl_Position = vec4(0.0, 0.0, -1.0, 1.0);
                    gl_PointSize = 0.0;
                    vAlpha = 0.0;
                    vColor = vec3(0.0);
                    return;
                }
                
                // Trail particles follow behind
                float trailOffset = float(particleOffset) * 0.1;
                position = fwPos;
                position.y -= trailOffset * 30.0;  // Trail spacing
                
                // Color: bright yellow-orange for launch
                color = hsv2rgb(vec3(0.1, 0.9, 1.0));
                alpha = 0.8 - (trailOffset * 0.5);  // Fade trail
                gl_PointSize = 4.0 - (trailOffset * 2.0);
            }
            
            // State 2: Exploding - draw explosion particles
            else {
                // Compute particle angle and speed
                float angle = (float(particleOffset) / float(particlesPerFirework)) * 6.28318;
                float angleVariation = hash(float(particleIdx)) * 0.5 - 0.25;
                angle += angleVariation;
                
                float speedVariation = hash(float(particleIdx + 1000));
                float baseSpeed = 100.0 + speedVariation * 150.0;
                
                // Audio reactivity: explosion speed scales with mid/high energy
                // Increased multiplier from 0.5 to 1.5 for more dramatic contrast
                float audioBoost = 1.0 + (midEnergy + highEnergy) * 1.5;
                float speed = baseSpeed * audioBoost;
                
                // Explosion physics
                float explosionTime = fwLifetime;
                vec2 explosionVel = vec2(cos(angle), sin(angle)) * speed;
                
                // Gravity affects explosion particles
                float gravity = 200.0;
                vec2 explosionPos = explosionVel * explosionTime;
                explosionPos.y -= 0.5 * gravity * explosionTime * explosionTime;
                
                position = fwPos;
                position.xy += explosionPos;
                
                // Color: use firework hue, vary brightness by particle
                float brightness = 0.8 + hash(float(particleIdx + 2000)) * 0.2;
                // Boost brightness with audio energy for more visual pop
                float audioBrightness = 1.0 + (midEnergy + highEnergy) * 0.5;
                brightness *= audioBrightness;
                brightness = clamp(brightness, 0.5, 1.5);  // Allow overbright particles
                color = hsv2rgb(vec3(fwHue, 0.9, brightness));
                
                // Fade out over time
                alpha = 1.0 - (explosionTime / 2.0);  // 2 second fade
                alpha = clamp(alpha, 0.0, 1.0);
                
                // Particle size shrinks over time
                gl_PointSize = 3.0 * (1.0 - explosionTime / 2.0);
            }
            
            // Convert to clip space
            vec2 clipPos = (position.xy / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depthValue = position.z / 100.0;
            depthValue = clamp(depthValue, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depthValue, 1.0);
            
            vColor = color;
            vAlpha = alpha;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec3 vColor;
        in float vAlpha;
        
        uniform float fadeAlpha;
        
        out vec4 outColor;
        
        void main() {
            // Circular particle shape
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;
            
            // Soft edge
            float edge = 1.0 - smoothstep(0.3, 0.5, dist);
            
            float finalAlpha = vAlpha * fadeAlpha * edge;
            outColor = vec4(vColor, finalAlpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create particle index array (0 to max_particles-1)
        particle_indices = np.arange(self.max_particles, dtype=np.float32)
        
        # Create VBO for particle indices
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, particle_indices.nbytes, particle_indices, GL_STATIC_DRAW)
        
        # Particle index attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        
        # Cache uniform locations
        self.uniform_resolution = glGetUniformLocation(self.shader, "resolution")
        self.uniform_time = glGetUniformLocation(self.shader, "time")
        self.uniform_max_fireworks = glGetUniformLocation(self.shader, "maxFireworks")
        self.uniform_particles_per = glGetUniformLocation(self.shader, "particlesPerFirework")
        self.uniform_bass = glGetUniformLocation(self.shader, "bassEnergy")
        self.uniform_mid = glGetUniformLocation(self.shader, "midEnergy")
        self.uniform_high = glGetUniformLocation(self.shader, "highEnergy")
        self.uniform_fade = glGetUniformLocation(self.shader, "fadeAlpha")
        
        # Firework data uniform locations
        self.uniform_fw_positions = glGetUniformLocation(self.shader, "fireworkPositions")
        self.uniform_fw_velocities = glGetUniformLocation(self.shader, "fireworkVelocities")
        self.uniform_fw_states = glGetUniformLocation(self.shader, "fireworkStates")
        self.uniform_fw_colors = glGetUniformLocation(self.shader, "fireworkColors")
        self.uniform_fw_lifetimes = glGetUniformLocation(self.shader, "fireworkLifetimes")
        self.uniform_fw_explosion_heights = glGetUniformLocation(self.shader, "fireworkExplosionHeights")
    
    def spawn_firework(self):
        """Spawn a new firework at the bottom"""
        # Find inactive firework slot
        inactive_idx = np.where(self.firework_states == 0.0)[0]
        if len(inactive_idx) == 0:
            return  # No slots available
        
        idx = inactive_idx[0]
        
        # Random X position
        x = np.random.random() * self.viewport.width
        y = self.viewport.height  # Start at bottom
        z = 40.0 + np.random.random() * 20.0  # Random depth 40-60
        
        self.firework_positions[idx] = [x, y, z]
        
        # Upward velocity with slight horizontal randomness
        vx = (np.random.random() - 0.5) * 50.0  # Horizontal drift
        vy = -300.0 - np.random.random() * 150.0  # Upward speed (negative Y is up)
        
        self.firework_velocities[idx] = [vx, vy]
        
        # Random explosion height (30-70% up the screen)
        explosion_y = self.viewport.height * (0.3 + np.random.random() * 0.4)
        self.firework_explosion_heights[idx] = explosion_y
        
        # Set state to launching
        self.firework_states[idx] = 1.0
        
        # Random color
        self.firework_colors[idx] = np.random.random()
        
        # Reset lifetime
        self.firework_lifetimes[idx] = 0.0
    
    def render(self, state: Dict):
        """Render fireworks particles"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        glUniform2f(self.uniform_resolution, self.viewport.width, self.viewport.height)
        glUniform1f(self.uniform_time, state.get('time', 0.0))
        glUniform1i(self.uniform_max_fireworks, self.max_fireworks)
        glUniform1i(self.uniform_particles_per, self.particles_per_firework)
        glUniform1f(self.uniform_bass, self.bass_energy)
        glUniform1f(self.uniform_mid, self.mid_energy)
        glUniform1f(self.uniform_high, self.high_energy)
        glUniform1f(self.uniform_fade, self.fade_factor)
        
        # Upload firework data arrays
        glUniform3fv(self.uniform_fw_positions, self.max_fireworks, self.firework_positions.flatten())
        glUniform2fv(self.uniform_fw_velocities, self.max_fireworks, self.firework_velocities.flatten())
        glUniform1fv(self.uniform_fw_states, self.max_fireworks, self.firework_states)
        glUniform1fv(self.uniform_fw_colors, self.max_fireworks, self.firework_colors)
        glUniform1fv(self.uniform_fw_lifetimes, self.max_fireworks, self.firework_lifetimes)
        glUniform1fv(self.uniform_fw_explosion_heights, self.max_fireworks, self.firework_explosion_heights)
        
        # Draw all particles as points
        # Note: GL_PROGRAM_POINT_SIZE is not needed in OpenGL ES 3.1
        # Point size is set directly in vertex shader via gl_PointSize
        glDrawArrays(GL_POINTS, 0, self.max_particles)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def update(self, dt: float, state: Dict):
        """Update firework simulation"""
        if not self.enabled:
            return
        
        # Spawn fireworks on timer
        self.time_since_launch += dt
        if self.time_since_launch >= self.launch_interval:
            self.spawn_firework()
            self.time_since_launch = 0.0
        
        # Spawn on bass hit
        if self.bass_launch_triggered:
            self.spawn_firework()
            self.bass_launch_triggered = False
        
        # Update all active fireworks
        gravity = 200.0  # Pixels per second squared
        
        for i in range(self.max_fireworks):
            if self.firework_states[i] == 0.0:
                continue  # Inactive
            
            # Update lifetime
            self.firework_lifetimes[i] += dt
            
            # State 1: Launching
            if self.firework_states[i] == 1.0:
                # Apply velocity and gravity
                self.firework_positions[i, 0] += self.firework_velocities[i, 0] * dt
                self.firework_positions[i, 1] += self.firework_velocities[i, 1] * dt
                
                # Apply gravity (positive Y is down)
                self.firework_velocities[i, 1] += gravity * dt
                
                # Check if reached explosion height
                if self.firework_positions[i, 1] <= self.firework_explosion_heights[i]:
                    # Transition to exploding
                    self.firework_states[i] = 2.0
                    self.firework_lifetimes[i] = 0.0  # Reset lifetime for explosion
            
            # State 2: Exploding
            elif self.firework_states[i] == 2.0:
                # Explosion lasts 2 seconds
                if self.firework_lifetimes[i] > 2.0:
                    # Deactivate
                    self.firework_states[i] = 0.0
