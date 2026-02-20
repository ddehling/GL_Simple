"""
Sound-reactive sunrise shader with tentacles
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_sunrise(state, outstate, tentacle_count=8, color_intensity=1.0, audio_sensitivity=2.0):
    """
    Sound-reactive sunrise with animated tentacles
    
    Usage:
        scheduler.schedule_event(0, 120, shader_sunrise, 
                               tentacle_count=8, audio_sensitivity=2.5, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        tentacle_count: Number of tentacles emanating from sun
        color_intensity: Brightness multiplier for colors
        audio_sensitivity: How strongly audio affects tentacle movement
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    audio_data = outstate.get('sound')
    scale = outstate.get('scale', 1.0)
    squish_top_width = 1.0 / scale if scale != 0 else 1.0  # Get inverse scale from state
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing sunrise for frame {frame_id}")
        print(f"  Duration: {state['duration']} seconds")
        print(f"  Elapsed: {state['elapsed_time']} seconds")
        
        try:
            effect = viewport.add_effect(
                SunriseEffect,
                tentacle_count=tentacle_count,
                color_intensity=color_intensity,
                audio_sensitivity=audio_sensitivity,
                squish_top_width=squish_top_width
            )
            state['effect'] = effect
            print(f"✓ Initialized shader sunrise for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize sunrise: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from audio data and time every frame
    if 'effect' in state:
        elapsed_time = state['elapsed_time']
        total_duration = state['duration']

        # Calculate rise/set position (rises first 10s, sets last 10s)
        rise_duration = 10.0
        set_duration = 10.0
        set_start_time = total_duration - set_duration
        
        if elapsed_time < rise_duration:
            # Rising phase (0 to 1)
            rise_factor = elapsed_time / rise_duration
        elif elapsed_time > set_start_time:
            # Setting phase (1 to 0)
            rise_factor = 1.0 - ((elapsed_time - set_start_time) / set_duration)
        else:
            # Fully risen
            rise_factor = 1.0
        
        state['effect'].rise_factor = np.clip(rise_factor, 0, 1)
        
        # Update from audio data
        if audio_data is not None:
            # Use short-term normalized for responsive tentacle movement
            bands = audio_data['norm_short'][0]
            
            # Bass drives tentacle intensity
            bass_energy = np.mean(bands[0:8])
            mid_energy = np.mean(bands[8:20])
            high_energy = np.mean(bands[20:32])
            
            state['effect'].bass_intensity = bass_energy * audio_sensitivity
            state['effect'].mid_intensity = mid_energy * audio_sensitivity
            state['effect'].high_intensity = high_energy * audio_sensitivity
            
            # Overall energy affects sun glow
            overall_energy = np.mean(bands)
            state['effect'].audio_energy = overall_energy
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up sunrise for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader sunrise for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class SunriseEffect(ShaderEffect):
    """Sound-reactive sunrise with animated tentacles"""
    
    def __init__(self, viewport, tentacle_count: int = 8, 
                 color_intensity: float = 1.0, audio_sensitivity: float = 2.0,
                 squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.tentacle_count = tentacle_count
        self.color_intensity = color_intensity
        self.audio_sensitivity = audio_sensitivity
        self.squish_top_width = squish_top_width
        
        # Animation state
        self.rise_factor = 0.0  # 0 = below screen, 1 = fully risen
        self.time = 0.0
        
        # Audio response
        self.bass_intensity = 0.0
        self.mid_intensity = 0.0
        self.high_intensity = 0.0
        self.audio_energy = 0.0
        
        # Fade control
        self.fade_factor = 1.0
        
        # Initialize geometry data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize fullscreen quad for rendering"""
        # Fullscreen quad vertices (covers entire screen)
        self.quad_vertices = np.array([
            [-1.0, -1.0],  # Bottom-left
            [ 1.0, -1.0],  # Bottom-right
            [ 1.0,  1.0],  # Top-right
            [-1.0,  1.0],  # Top-left
        ], dtype=np.float32)
        
        # Indices for two triangles
        self.quad_indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)
    
    def compile_shader(self):
        """Compile and link shaders - REQUIRED METHOD"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"SunriseEffect shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position, 0.99, 1.0);  // depth = 0.99 (z=99)
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
        uniform float riseFactor;
        uniform int tentacleCount;
        uniform float bassIntensity;
        uniform float midIntensity;
        uniform float highIntensity;
        uniform float audioEnergy;
        uniform float colorIntensity;
        uniform float fadeAlpha;
        uniform float squishTopWidth;
        
        // Noise function for tentacle animation
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        
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
        
        void main() {
            // Convert to pixel coordinates
            vec2 uv = (fragCoord + 1.0) * 0.5;  // 0 to 1
            vec2 pixel = uv * resolution;
            
            // Apply horizontal squish based on y position (bottom = 1.0, top = squishTopWidth)
            // Normalize y: 0 at bottom, 1 at top
            float y_normalized = pixel.y / resolution.y;
            float squish_factor = 1.0 + (squishTopWidth - 1.0) * y_normalized;
            
            // Apply squish to horizontal coordinate by scaling from center
            float centerX = resolution.x * 0.5;
            pixel.x = centerX + (pixel.x - centerX) * squish_factor;
            
            // Scale sun based on width - sun+halo+tentacles should be ~50% of total width
            float targetTotalWidth = resolution.x * 0.5;
            float haloRadius = targetTotalWidth * 0.5;  // Halo is 50% of total (full diameter)
            float sunRadius = haloRadius * 0.35;  // Sun is 35% of halo radius
            
            // Sun becomes a flat gradient at the top when fully risen
            float sunHeight = sunRadius + 20.0;  // Added 20 pixels
            float blueGradientHeight = 150.0;  // Blue to black gradient below sun colors (tripled)
            float totalGradientHeight = sunHeight + blueGradientHeight;
            
            // Two-phase animation:
            // Phase 1 (riseFactor 0.0 to 0.7): Sun rises and exits screen completely
            // Phase 2 (riseFactor 0.7 to 1.0): Gradient band rises from top
            
            float phase1End = 0.7;
            float sunMask = 0.0;
            float isFlat = 0.0;
            vec2 sunCenter;
            float distToSun;
            float angle;
            
            if (riseFactor < phase1End) {
                // PHASE 1: Circular sun rising and exiting
                float phase1Progress = riseFactor / phase1End;
                
                // Sun travels from below screen to completely above screen (including halo)
                // Extra margin ensures sun and glow are completely off screen before disappearing
                float totalRiseDistance = resolution.y + haloRadius * 2.0 + sunRadius * 2.0;
                float sunY = resolution.y + sunRadius + haloRadius - (phase1Progress * totalRiseDistance);
                sunCenter = vec2(resolution.x * 0.5, sunY);
                
                // Distance from sun center
                vec2 toSun = pixel - sunCenter;
                distToSun = length(toSun);
                angle = atan(toSun.y, toSun.x);
                
                // Circular sun
                sunMask = smoothstep(sunRadius * 1.0, sunRadius * 0.1, distToSun);
            } else {
                // PHASE 2: Flat gradient rising from top
                isFlat = 1.0;
                
                // Gradient rises over the remaining time
                float phase2Progress = (riseFactor - phase1End) / (1.0 - phase1End);
                
                // Gradient extends down from top, growing as it rises
                float currentGradientHeight = totalGradientHeight * phase2Progress;
                
                float topEdge = 0.0;
                float bottomEdge = currentGradientHeight;
                
                // Fade distance at the leading edge (in pixels)
                float fadeDistance = 50.0;
                
                if (pixel.y < bottomEdge) {
                    // Calculate fade at the leading edge
                    float distFromEdge = bottomEdge - pixel.y;
                    float edgeFade = smoothstep(0.0, fadeDistance, distFromEdge);
                    
                    // Set sunMask with edge fade
                    sunMask = edgeFade;
                    distToSun = pixel.y;
                }
                
                // Set dummy values for tentacle calculation (won't be used when isFlat=1)
                sunCenter = vec2(resolution.x * 0.5, -haloRadius * 2.0);
                distToSun = 9999.0;
                angle = 0.0;
            }
            
            // No separate sun glow - just smooth falloff built into sunMask
            
            // Add smooth radial falloff for pure radial gradient around sun
            float radialGlow = 0.0;
            if (isFlat < 0.5) {
                // Create radial glow extending from sun - constrained to haloRadius
                radialGlow = smoothstep(haloRadius, sunRadius, distToSun);
            }
            
            // Tentacles - flowing extensions like fog beings
            float tentacleInfluence = 0.0;
            
            if (distToSun > sunRadius * 0.9 && isFlat < 0.5) {
                float angleStep = 6.28318530718 / float(tentacleCount);
                
                for (int i = 0; i < 32; i++) {
                    if (i >= tentacleCount) break;
                    
                    float tentacleAngle = float(i) * angleStep;
                    
                    // Distance along this tentacle direction
                    float tentacleDist = distToSun - sunRadius;
                    float maxTentacleLength = haloRadius - sunRadius;  // Tentacles extend to halo edge
                    
                    if (tentacleDist < maxTentacleLength) {
                        // Calculate angular distance from this tentacle's centerline
                        float angleDiff = abs(angle - tentacleAngle);
                        angleDiff = min(angleDiff, 6.28318530718 - angleDiff);
                        
                        // Wavy tentacle path - oscillates as it extends
                        float waveFreq = 0.02;
                        float waveAmp = 0.3;
                        float waveOffset = sin(tentacleDist * waveFreq + time * 0.5 + float(i)) * waveAmp;
                        
                        // Tentacle width that tapers with distance
                        float ratio = tentacleDist / maxTentacleLength;
                        float baseWidth = 0.15;
                        float tentacleWidth = baseWidth * (1.0 - ratio * 0.5);
                        
                        // Apply wave offset to angular position
                        float effectiveAngleDiff = abs(angleDiff - waveOffset);
                        
                        if (effectiveAngleDiff < tentacleWidth) {
                            // Distance-based fadeout
                            float distanceFade = 1.0 - smoothstep(0.0, maxTentacleLength, tentacleDist);
                            
                            // Width-based fadeout (center is brightest)
                            float widthFade = 1.0 - (effectiveAngleDiff / tentacleWidth);
                            
                            // Combine fades
                            float intensity = widthFade * distanceFade * 0.6;
                            
                            tentacleInfluence = max(tentacleInfluence, intensity);
                        }
                    }
                }
            }
            
            // Combine: radial glow as base, tentacles as additions
            float totalMask = max(sunMask, radialGlow + tentacleInfluence * 0.4);
            
            // Color gradient: red at sun -> orange -> purple -> blue -> black
            vec3 color;
            
            if (totalMask > 0.01) {
                // Distance-based color
                float distFactor;
                
                if (isFlat > 0.5) {
                    // Use Y position for flat gradient
                    // Map the entire gradient height to color range
                    distFactor = pixel.y / totalGradientHeight;
                    distFactor = clamp(distFactor, 0.0, 1.0);
                } else {
                    // Use radial distance for circular sun - normalized to haloRadius
                    distFactor = distToSun / haloRadius;
                }
                
                distFactor = clamp(distFactor, 0.0, 1.0);
                
                // Sun core: bright yellow-orange
                vec3 sunColor = vec3(1.0, 0.6, 0.2);
                
                // Near sun: red-orange
                vec3 nearColor = vec3(1.0, 0.3, 0.1);
                
                // Mid distance: purple
                vec3 midColor = vec3(0.6, 0.2, 0.8);
                
                // Far: deep blue
                vec3 farColor = vec3(0.1, 0.2, 0.6);
                
                // Black background
                vec3 bgColor = vec3(0.0, 0.0, 0.0);
                
                // Multi-stage gradient
                if (distFactor < 0.15) {
                    color = mix(sunColor, nearColor, distFactor / 0.15);
                } else if (distFactor < 0.35) {
                    color = mix(nearColor, midColor, (distFactor - 0.15) / 0.20);
                } else if (distFactor < 0.55) {
                    // Transition from purple to deep blue
                    color = mix(midColor, farColor, (distFactor - 0.35) / 0.20);
                } else {
                    // Extended blue to black gradient
                    color = mix(farColor, bgColor, (distFactor - 0.55) / 0.45);
                }
                
                // Audio modulation: high frequencies add brightness
                color += vec3(highIntensity * 0.15);
                
                // Apply color intensity
                color *= colorIntensity*0.5;
                
                // Apply mask
                color *= totalMask;
                
                // Extra brightness for sun core (only when circular)
                if (sunMask > 0.7 && isFlat < 0.5) {
                    color += sunColor * sunMask * 0.3;
                }
            } else {
                // Pure black background
                color = vec3(0.0);
            }
            
            // Output with alpha based on intensity
            float alpha = clamp(totalMask, 0.0, 1.0);
            
            // Alpha fade in the blue gradient region
            if (isFlat > 0.5 && totalMask > 0.01) {
                float distFactor = pixel.y / totalGradientHeight;
                if (distFactor > 0.55) {
                    // Fade alpha from 1.0 to 0.0 across the blue gradient
                    float blueFadeFactor = (distFactor - 0.55) / 0.45;
                    alpha *= (1.0 - blueFadeFactor);
                }
            }
            
            alpha *= fadeAlpha;
            
            // Set alpha to 0 for very low intensities to prevent black artifacts
            if (totalMask < 0.02) {
                alpha = 0.0;
            }
            
            outColor = vec4(color, alpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers - Called automatically after shader compilation"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create vertex buffer
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, 
                     self.quad_vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Create element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.quad_indices.nbytes,
                     self.quad_indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        self.time += dt
    
    def render(self, state: Dict):
        """Render sunrise effect"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        rise_loc = glGetUniformLocation(self.shader, "riseFactor")
        glUniform1f(rise_loc, self.rise_factor)
        
        tentacle_loc = glGetUniformLocation(self.shader, "tentacleCount")
        glUniform1i(tentacle_loc, self.tentacle_count)
        
        bass_loc = glGetUniformLocation(self.shader, "bassIntensity")
        glUniform1f(bass_loc, self.bass_intensity)
        
        mid_loc = glGetUniformLocation(self.shader, "midIntensity")
        glUniform1f(mid_loc, self.mid_intensity)
        
        high_loc = glGetUniformLocation(self.shader, "highIntensity")
        glUniform1f(high_loc, self.high_intensity)
        
        energy_loc = glGetUniformLocation(self.shader, "audioEnergy")
        glUniform1f(energy_loc, self.audio_energy)
        
        intensity_loc = glGetUniformLocation(self.shader, "colorIntensity")
        glUniform1f(intensity_loc, self.color_intensity)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        squish_loc = glGetUniformLocation(self.shader, "squishTopWidth")
        glUniform1f(squish_loc, self.squish_top_width)
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
