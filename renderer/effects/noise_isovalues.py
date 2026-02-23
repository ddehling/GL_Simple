"""
Noise-based isovalues effect - animated contour lines with procedural Perlin noise
Creates flowing patterns with smooth contour lines based on 3D Perlin noise
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_noise_isovalues(state, outstate, scale=8.0, speed=1, frequency=5.0, 
                           fade_duration=10.0, audio_reactive=True, audio_sensitivity=2.0,
                           bass_sensitivity=2.0, high_sensitivity=1.5):
    """
    Audio-reactive noise-based isovalues effect compatible with EventScheduler
    
    Creates animated contour lines based on procedural Perlin noise with audio reactivity
    
    Usage:
        scheduler.schedule_event(0, 60, shader_noise_isovalues, 
                               scale=8.0, speed=0.1, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        scale: Noise scale (default 8.0, larger = more zoomed out)
        speed: Animation speed (default 0.1)
        frequency: Contour line frequency (default 5.0, fewer lines)
        fade_duration: Duration of fade in/out in seconds (default 10.0)
        audio_reactive: Enable audio reactivity (default True)
        audio_sensitivity: How much mids affect frequency (default 2.0)
        bass_sensitivity: How much bass affects line width (default 2.5)
        high_sensitivity: How much highs affect color shift (default 1.5)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing noise isovalues effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                NoiseIsovaluesEffect,
                scale=scale,
                speed=speed,
                frequency=frequency,
                audio_reactive=audio_reactive,
                audio_sensitivity=audio_sensitivity,
                bass_sensitivity=bass_sensitivity,
                high_sensitivity=high_sensitivity
            )
            state['effect'] = effect
            state['smoothed_mid'] = 0.0
            state['smoothed_bass'] = 0.0
            state['smoothed_high'] = 0.0
            state['peak_mid'] = 0.0      # Peak tracker for one-directional response
            state['peak_bass'] = 0.0
            state['peak_high'] = 0.0
            print(f"✓ Initialized shader noise_isovalues for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize noise isovalues: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters
    if 'effect' in state:
        state['effect'].scale = outstate.get('noise_scale', scale)
        state['effect'].frequency = outstate.get('noise_frequency', frequency)
        
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = np.clip(fade_factor, 0, 1)
        
        # Audio reactivity - multi-band response (one-directional)
        if audio_reactive and audio_data is not None:
            bands = audio_data['norm_short'][0]
            
            # Extract frequency bands
            bass_energy = np.mean(bands[0:8])       # Bass: 40-300 Hz
            mid_energy = np.mean(bands[8:20])       # Mids: 300-2000 Hz
            high_energy = np.mean(bands[20:32])     # Highs: 2000-16000 Hz
            
            # Smooth audio for less jitter but keep it responsive
            smoothing = 0.15  # Faster response for more dynamic feel
            state['smoothed_bass'] = smoothing * bass_energy + (1 - smoothing) * state['smoothed_bass']
            state['smoothed_mid'] = smoothing * mid_energy + (1 - smoothing) * state['smoothed_mid']
            state['smoothed_high'] = smoothing * high_energy + (1 - smoothing) * state['smoothed_high']
            
            # One-directional response: track peaks and only decay slowly
            # When energy increases, update immediately
            # When energy decreases, decay slowly so effect only progresses forward
            decay_rate = 0.95  # Faster decay for more dynamic response
            
            if state['smoothed_bass'] > state['peak_bass']:
                state['peak_bass'] = state['smoothed_bass']
            else:
                state['peak_bass'] *= decay_rate
            
            if state['smoothed_mid'] > state['peak_mid']:
                state['peak_mid'] = state['smoothed_mid']
            else:
                state['peak_mid'] *= decay_rate
            
            if state['smoothed_high'] > state['peak_high']:
                state['peak_high'] = state['smoothed_high']
            else:
                state['peak_high'] *= decay_rate
            
            # Update effect with peak values (one-directional)
            state['effect'].audio_bass = np.clip(state['peak_bass'] * bass_sensitivity, 0, 5)
            state['effect'].audio_mid = np.clip(state['peak_mid'] * audio_sensitivity, 0, 5)
            state['effect'].audio_high = np.clip(state['peak_high'] * high_sensitivity, 0, 3)
            
            # Beat detection using norm_long_relu for better contrast (above-average bass only)
            relu_bands = audio_data['norm_long_relu'][0]
            bass_relu = np.mean(relu_bands[0:8])  # Only above-average bass energy
            
            # Detect bass hits with lower threshold on relu (already highlights peaks)
            if bass_relu > 0.3:  # Hit detected when significantly above baseline
                state['effect'].trigger_beat_pulse()
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up noise isovalues effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader noise_isovalues for frame {frame_id}")

# ============================================================================
# Noise Isovalues Effect - Post-Processing
# ============================================================================

class NoiseIsovaluesEffect(ShaderEffect):
    """Fullscreen post-processing effect with audio-reactive noise-based contour lines"""
    
    def __init__(self, viewport, scale: float = 8.0, speed: float = 0.1, 
                 frequency: float = 5.0, audio_reactive: bool = True, 
                 audio_sensitivity: float = 2.0, bass_sensitivity: float = 2.5,
                 high_sensitivity: float = 1.5):
        super().__init__(viewport)
        self.scale = scale
        self.speed = speed
        self.frequency = frequency
        self.audio_reactive = audio_reactive
        self.audio_sensitivity = audio_sensitivity
        self.bass_sensitivity = bass_sensitivity
        self.high_sensitivity = high_sensitivity
        self.fade_factor = 0.0
        
        # Multi-band audio reactivity
        self.audio_bass = 0.0      # Bass affects line width
        self.audio_mid = 0.0       # Mids affect frequency
        self.audio_high = 0.0      # Highs affect color shift
        self.beat_pulse = 0.0      # Beat detection pulse
        
        self.time = 0.0
        self.drift_x = 0.0    # Horizontal drift
        self.drift_y = 0.0    # Vertical drift
    
    def trigger_beat_pulse(self):
        """Trigger a visual pulse on beat detection"""
        self.beat_pulse = 2.0  # Much stronger pulse (was 1.0)
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position * 2.0 - 1.0, 0.5, 1.0);
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform vec2 resolution;
        uniform float time;
        uniform float scale;
        uniform float frequency;
        uniform float fadeAlpha;
        uniform float audioBass;      // Bass energy (affects line width)
        uniform float audioMid;       // Mid energy (affects frequency)
        uniform float audioHigh;      // High energy (affects color)
        uniform float beatPulse;      // Beat pulse intensity
        uniform vec2 drift;           // Slow 2D drift for moving peaks
        
        // Hash function for pseudo-random values
        float hash3(vec3 p) {
            return fract(sin(1e3 * dot(p, vec3(1.0, 57.0, -13.7))) * 4375.5453);
        }
        
        // 3D Perlin noise (adapted from IQ)
        float noise3(vec3 x) {
            vec3 p = floor(x);
            vec3 f = fract(x);
            
            // Smoothstep for continuous derivative at borders
            f = f * f * (3.0 - 2.0 * f);
            
            // Trilinear interpolation
            return mix(
                mix(
                    mix(hash3(p + vec3(0.0, 0.0, 0.0)), hash3(p + vec3(1.0, 0.0, 0.0)), f.x),
                    mix(hash3(p + vec3(0.0, 1.0, 0.0)), hash3(p + vec3(1.0, 1.0, 0.0)), f.x),
                    f.y
                ),
                mix(
                    mix(hash3(p + vec3(0.0, 0.0, 1.0)), hash3(p + vec3(1.0, 0.0, 1.0)), f.x),
                    mix(hash3(p + vec3(0.0, 1.0, 1.0)), hash3(p + vec3(1.0, 1.0, 1.0)), f.x),
                    f.y
                ),
                f.z
            );
        }
        
        // Improved pseudo-Perlin (double sampling)
        float noise(vec3 x) {
            return (noise3(x) + noise3(x + 11.5)) / 2.0;
        }
        
        void main() {
            vec2 uv = fragCoord * resolution;
            
            // Normalize coordinates by height (makes aspect ratio correct)
            vec2 noiseCoord = uv / resolution.y;
            
            // For horizontal wrapping, use sine/cosine mapping to create periodic input
            // Map x from [0, aspectRatio] to angle [0, 2π], then to circle coords
            float aspectRatio = resolution.x / resolution.y;
            float angle = (noiseCoord.x / aspectRatio) * 6.28318;  // 2π
            
            // Map to 3D torus-like coordinates for seamless wrapping
            float wrapRadius = aspectRatio * scale / 6.28318;
            vec3 wrappedCoord = vec3(
                cos(angle) * wrapRadius,
                sin(angle) * wrapRadius,
                noiseCoord.y * scale
            );
            
            // Multi-band audio reactivity
            // Mids affect contour line frequency (more energetic = more lines)
            float activeFreq = frequency * (1.0 + audioMid * 0.4);  // Increased from 0.2
            
            // Calculate noise value with wrapped coordinates, time, and drift
            // Drift shifts the entire noise field to make peaks/valleys move around
            float n = noise(wrappedCoord + vec3(drift * 0.5, 0.1 * time));
            
            // Create contour lines
            float v = sin(6.28318 * activeFreq * n);
            
            // Bass affects line width (stronger bass = wider lines)
            // Beat pulse adds sudden width increase on beats
            float lineWidthMod = 0.6 + 0.7 * fract(n * 10.0); // Base variation (increased from 0.3)
            lineWidthMod *= (1.0 + audioBass * 0.35 + beatPulse * 0.8);  // Stronger beat pulse (was 0.5)
            
            // Anti-aliased contour line using derivative with variable width
            // This creates the "smeared" depth effect from the original
            float lineGradient = 0.5 * abs(v) / fwidth(v);
            v = smoothstep(1.0, 0.0, lineGradient / lineWidthMod);  // Divide instead of multiply for wider gradient
            
            // Create color gradient based on noise
            // High frequencies shift colors (hue rotation effect)
            vec3 noiseColor = 0.5 + 0.5 * sin(12.0 * n + vec3(0.0, 2.1, -2.1) + audioHigh * 1.5);  // Increased from 0.8
            
            // Create flowing background for smear/depth effect (like original)
            vec2 flowOffset = uv + vec2(1.0, sin(time)) * resolution.y;
            vec2 bgUV = flowOffset / resolution.y;
            
            // Generate procedural background pattern
            float bgNoise = noise(vec3(bgUV * 2.0, 0.05 * time));
            vec3 bgPattern = 0.5 + 0.5 * sin(12.0 * bgNoise + vec3(0.0, 2.1, -2.1) + audioHigh * 1.5);
            
            // Apply subtle fade to background (creates depth)
            float bgFade = exp(-33.0 / resolution.y);
            vec3 bgColor = bgPattern * bgFade * 0.6;  // Brighter background for more visible smear
            
            // Mix background with contour colors (this creates the smear effect)
            vec3 finalColor = mix(bgColor, noiseColor, v);
            
            // Audio reactivity adds brightness pulse and beat flash
            finalColor *= 1.0 + audioMid * 0.12 + beatPulse * 0.5;  // Stronger beat flash (was 0.3)
            
            // Calculate alpha: make background visible for smear effect
            // Lines are opaque (v=1), background shows through where v=0
            float finalAlpha = fadeAlpha * (0.4 + 0.6 * v);  // Higher base alpha for more visible background
            
            outColor = vec4(finalColor, finalAlpha);
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
        """Setup fullscreen quad for post-processing"""
        # Fullscreen quad vertices (texture coordinates)
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
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
        """Update animation time and beat pulse decay"""
        if not self.enabled:
            return
        
        self.time += dt * self.speed
        
        # Drift pattern with forward bias (more forward than back-and-forth)
        # Forward drift dominates, with gentle side-to-side wandering
        self.drift_x += dt * 0.5  # Constant forward drift
        self.drift_y += dt * 0.3 * np.sin(self.time * 0.13)  # Gentle vertical oscillation
        
        # Decay beat pulse quickly for subtle effect
        self.beat_pulse *= 0.85  # Slower decay (was 0.75) for more visible beats
    
    def render(self, state: Dict):
        """Render fullscreen post-processing effect"""
        if not self.enabled or not self.shader:
            return
        
        # Post-process exception: Always pass depth test without writing
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        if res_loc != -1:
            glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        if time_loc != -1:
            glUniform1f(time_loc, self.time)
        
        drift_loc = glGetUniformLocation(self.shader, "drift")
        if drift_loc != -1:
            glUniform2f(drift_loc, self.drift_x, self.drift_y)
        
        scale_loc = glGetUniformLocation(self.shader, "scale")
        if scale_loc != -1:
            glUniform1f(scale_loc, self.scale)
        
        freq_loc = glGetUniformLocation(self.shader, "frequency")
        if freq_loc != -1:
            glUniform1f(freq_loc, self.frequency)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if fade_loc != -1:
            glUniform1f(fade_loc, self.fade_factor)
        
        # Audio uniforms
        if self.audio_reactive:
            bass_loc = glGetUniformLocation(self.shader, "audioBass")
            if bass_loc != -1:
                glUniform1f(bass_loc, self.audio_bass)
            
            mid_loc = glGetUniformLocation(self.shader, "audioMid")
            if mid_loc != -1:
                glUniform1f(mid_loc, self.audio_mid)
            
            high_loc = glGetUniformLocation(self.shader, "audioHigh")
            if high_loc != -1:
                glUniform1f(high_loc, self.audio_high)
            
            pulse_loc = glGetUniformLocation(self.shader, "beatPulse")
            if pulse_loc != -1:
                glUniform1f(pulse_loc, self.beat_pulse)
        else:
            # Set to zero if not audio reactive
            bass_loc = glGetUniformLocation(self.shader, "audioBass")
            if bass_loc != -1:
                glUniform1f(bass_loc, 0.0)
            
            mid_loc = glGetUniformLocation(self.shader, "audioMid")
            if mid_loc != -1:
                glUniform1f(mid_loc, 0.0)
            
            high_loc = glGetUniformLocation(self.shader, "audioHigh")
            if high_loc != -1:
                glUniform1f(high_loc, 0.0)
            
            pulse_loc = glGetUniformLocation(self.shader, "beatPulse")
            if pulse_loc != -1:
                glUniform1f(pulse_loc, 0.0)
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore default depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
