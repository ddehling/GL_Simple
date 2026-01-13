"""
Ocean waves shader effect - Waves crashing against shore at top of screen
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_ocean_waves(state, outstate, height_ratio=0.9, depth=85.0, 
                       wave_speed=1.0, wave_height=1.0, foam_amount=1.0):
    """
    Shader-based ocean waves effect compatible with EventScheduler
    
    Creates realistic ocean waves that crash against a shore at the top of
    the screen. Waves roll upward from the bottom with natural motion and
    foam dynamics.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_ocean_waves, wave_speed=1.2, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        height_ratio: Height of ocean as proportion of viewport (0.0-1.0, default: 0.9)
        depth: Z-depth of ocean (0=near, 100=far, default: 85)
        wave_speed: Animation speed multiplier (default: 1.0)
        wave_height: Wave amplitude multiplier (default: 1.0)
        foam_amount: Foam intensity multiplier (default: 1.0)
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
        print(f"Initializing ocean_waves for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                OceanWaves,
                height_ratio=height_ratio,
                depth=depth,
                wave_speed=wave_speed,
                wave_height=wave_height,
                foam_amount=foam_amount
            )
            state['effect'] = effect
            print(f"✓ Initialized shader ocean_waves for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize ocean_waves: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from global state - read ocean weather parameters
    if 'effect' in state:
        # Use ocean weather set parameter names
        new_wave_speed = outstate.get('wave_speed', wave_speed)
        new_wave_height = outstate.get('wave_amplitude', wave_height)
        new_tide_level = outstate.get('tide_level', 0.5)
        
        # Debug output every 60 frames (~once per second)
        if state['count'] % 60 == 0:
            print(f"Ocean: speed={new_wave_speed:.3f}, amp={new_wave_height:.3f}, tide={new_tide_level:.3f}")
        
        state['effect'].wave_speed = new_wave_speed
        state['effect'].wave_height = new_wave_height
        state['effect'].tide_level = new_tide_level
        
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
            print(f"Cleaning up ocean_waves for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader ocean_waves for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class OceanWaves(ShaderEffect):
    """
    Ocean waves effect with realistic water motion and foam
    
    Creates animated waves that roll upward toward a shore at the top
    of the screen, with foam dynamics and natural water movement.
    """
    
    def __init__(self, viewport, height_ratio: float = 0.9, depth: float = 85.0,
                 wave_speed: float = 1.0, wave_height: float = 1.0, foam_amount: float = 1.0,
                 tide_level: float = 0.0):
        super().__init__(viewport)
        self.height_ratio = height_ratio
        self.height = viewport.height  # Use full viewport height
        self.depth = depth
        self.wave_speed = wave_speed
        self.wave_height = wave_height
        self.foam_amount = foam_amount
        self.tide_level = tide_level  # Controls water level: -1=low tide, 0=normal, 1=high tide
        self.time = 0.0
        self.phase = 0.0  # Accumulated phase for smooth wave speed transitions
        self.fade_factor = 0.0
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.EBO = None
        
        # Mesh resolution (higher = smoother waves)
        self.segments_x = 100  # Horizontal segments
        self.segments_y = 30   # Vertical segments
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize mesh data for ocean surface"""
        width = self.viewport.width
        
        # Extend mesh MUCH further beyond viewport to handle wave displacement
        # Start at -100 pixels and go to height + 100 pixels
        self.mesh_start = -100.0
        self.mesh_end = float(self.height) + 100.0
        mesh_height = self.mesh_end - self.mesh_start
        
        # Create a grid mesh covering the extended area
        vertices = []
        for y in range(self.segments_y + 1):
            y_pos = self.mesh_start + (y / self.segments_y) * mesh_height
            for x in range(self.segments_x + 1):
                x_pos = (x / self.segments_x) * width
                vertices.append([x_pos, y_pos])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Generate indices for triangle strip rendering
        indices = []
        for y in range(self.segments_y):
            for x in range(self.segments_x + 1):
                # Two vertices per column (current row and next row)
                indices.append(y * (self.segments_x + 1) + x)
                indices.append((y + 1) * (self.segments_x + 1) + x)
            # Degenerate triangles to connect strips
            if y < self.segments_y - 1:
                indices.append((y + 1) * (self.segments_x + 1) + self.segments_x)
                indices.append((y + 1) * (self.segments_x + 1))
        
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = len(self.indices)
    
    def compile_shader(self):
        """Compile and link ocean shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Ocean waves shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        uniform vec2 resolution;
        uniform float depth;
        uniform float time;
        uniform float phase;  // Accumulated phase for smooth transitions
        uniform float height;
        uniform float waveSpeed;
        uniform float waveHeight;
        uniform float tideLevel;
        uniform float meshStart;
        uniform float meshEnd;
        
        out vec2 vTexCoord;
        out float vWaveIntensity;
        out float vDistanceFromShore;
        out vec2 vWorldPos;
        
        // Simple noise function
        float hash(vec2 p) {
            p = fract(p * vec2(123.45, 678.90));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
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
            vec2 pos = position;
            
            // Normalize coordinates based on VIEWPORT, not mesh bounds
            // Clamp to 0-1 range for areas within viewport
            float normY = clamp((pos.y - 0.0) / height, 0.0, 1.0);
            // Invert Y to rotate 180 degrees (beach at bottom, waves go down)
            normY = 1.0 - normY;
            float normX = pos.x / resolution.x;
            
            // Distance from shore: 0 at shore (top), 1 at deep water (bottom)
            // Apply tide level: tideLevel expected in 0-1 range, map to -0.5 to 0.5 (0.5 = normal)
            float tideFactor = (tideLevel - 0.5) * 0.3;  // Map 0-1 to -0.15 to +0.15
            float shoreDistance = 1.0 - (normY + tideFactor);
            shoreDistance = clamp(shoreDistance, 0.0, 1.0);
            
            // Create wave fronts that roll toward shore
            // Wave phase moves with time AND position to create motion
            // Use accumulated phase to avoid discontinuities when speed changes
            float wavePhase = shoreDistance * 8.0 + phase;
            
            // Add horizontal variation with noise to break up vertical stripes
            // Use phase * 0.125 to slow down horizontal variation (phase is 8x faster than old time calc)
            float horizontalVariation = noise(vec2(normX * 3.0, phase * 0.125)) * 2.0;
            wavePhase += horizontalVariation;
            
            // Multiple wave layers with different scales
            float wave1 = sin(wavePhase) * 0.5 + 0.5;
            float wave2 = sin(wavePhase * 1.7 + 1.3) * 0.5 + 0.5;
            float wave3 = sin(wavePhase * 2.3 - 0.7) * 0.5 + 0.5;
            
            // Power function to create peaked waves (sharp crests, flat troughs)
            wave1 = pow(wave1, 0.4);
            wave2 = pow(wave2, 0.5);
            wave3 = pow(wave3, 0.6);
            
            // Combine waves
            float waveHeight_combined = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2);
            
            // Add noise-based turbulence for organic motion
            // Use phase with appropriate scaling
            float turbulence = noise(vec2(normX * 3.0 + phase * 0.375,
                                          shoreDistance * 4.0 - phase * 0.25));
            waveHeight_combined += turbulence * 0.15;
            
            // Waves grow taller as they approach shore (shoaling)
            float shoaling = 1.0 + pow(1.0 - shoreDistance, 1.5) * 2.5;
            waveHeight_combined *= shoaling;
            
            // Breaking wave at shore - waves curl over and crash
            if (shoreDistance < 0.15) {
                // Extreme steepening and chaos at shore
                float breakingIntensity = (0.15 - shoreDistance) / 0.15;
                float breakingNoise = noise(vec2(normX * 5.0, phase * 1.25));
                waveHeight_combined += breakingIntensity * breakingNoise * 1.5;
            }
            
            // Beach zone - only dampen at the very bottom where beach is (after Y inversion)
            // normY = 1 is bottom now, so dampen when normY > 0.97
            float beachZone = smoothstep(0.97, 0.99, normY);
            waveHeight_combined *= (1.0 - beachZone);
            
            // Apply wave height displacement
            // waveHeight expected in 0-1 range, use power curve to reduce dramatic high-end changes
            // pow(x, 1.5) compresses high values: 0.3→0.16, 0.5→0.35, 0.8→0.72, 1.0→1.0
            float scaledHeight = pow(waveHeight, 1.5) * 1.5;  // Range: 0 to 1.5
            float totalHeight = waveHeight_combined * 20.0 * scaledHeight;
            pos.y += totalHeight;
            
            // No clamping needed - mesh extends beyond viewport
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, mappedDepth, 1.0);
            
            // Pass data to fragment shader
            vTexCoord = vec2(normX, normY);
            vWaveIntensity = waveHeight_combined;
            vDistanceFromShore = shoreDistance;
            vWorldPos = pos;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vTexCoord;
        in float vWaveIntensity;
        in float vDistanceFromShore;
        in vec2 vWorldPos;
        
        uniform float time;
        uniform float phase;  // Accumulated phase for smooth transitions
        uniform float waveSpeed;
        uniform float foamAmount;
        uniform float fadeAlpha;
        uniform float tideLevel;
        
        out vec4 outColor;
        
        // Simple noise
        float hash(vec2 p) {
            p = fract(p * vec2(123.45, 678.90));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
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
        
        // Ocean color palette
        vec3 getOceanColor(float depth, float waveIntensity, float foam) {
            vec3 deepColor = vec3(0.0, 0.15, 0.3);
            vec3 midColor = vec3(0.0, 0.4, 0.5);
            vec3 shallowColor = vec3(0.1, 0.6, 0.7);
            vec3 foamColor = vec3(0.9, 0.95, 1.0);
            
            vec3 baseColor;
            if (depth > 0.7) {
                baseColor = mix(midColor, deepColor, (depth - 0.7) / 0.3);
            } else if (depth > 0.3) {
                baseColor = mix(shallowColor, midColor, (depth - 0.3) / 0.4);
            } else {
                baseColor = shallowColor;
            }
            
            baseColor = mix(baseColor, foamColor, foam);
            return baseColor;
        }
        
        void main() {
            float x = vTexCoord.x;
            float y = vTexCoord.y;
            float depth = vDistanceFromShore;
            
            // Beach zone at shore (top of screen where normY is close to 1)
            // Beach extends from normY 0.92 to 1.0 (top 8% of screen)
            // Tide level shifts beach boundary: tideLevel in 0-1 range, map to -0.5 to 0.5 (0.5 = normal)
            float tideFactor = (tideLevel - 0.5) * 0.3;  // Map 0-1 to -0.15 to +0.15
            float beachStart = 0.92 + tideFactor;
            float beachZone = smoothstep(beachStart, 1.0, y);
            
            // Sand colors
            vec3 drySandColor = vec3(0.85, 0.75, 0.55);
            vec3 wetSandColor = vec3(0.55, 0.45, 0.30);
            vec3 underwaterSandColor = vec3(0.45, 0.38, 0.25);
            
            // Sand texture
            float sandNoise1 = noise(vWorldPos * 0.15);
            float sandNoise2 = noise(vWorldPos * 0.5);
            float sandTexture = sandNoise1 * 0.7 + sandNoise2 * 0.3;
            sandTexture = sandTexture * 0.3 + 0.7;
            
            // Determine sand wetness based on position
            vec3 sandColor;
            float dryThreshold = 0.97 + tideFactor;
            float wetThreshold = 0.94 + tideFactor;
            if (y > dryThreshold) {
                sandColor = drySandColor;
            } else if (y > wetThreshold) {
                float blend = (y - wetThreshold) / 0.03;
                sandColor = mix(wetSandColor, drySandColor, blend);
            } else {
                float blend = (y - beachStart) / (0.94 - beachStart);
                sandColor = mix(underwaterSandColor, wetSandColor, blend);
            }
            sandColor *= sandTexture;
            
            // Create foam texture
            vec2 foamCoord = vWorldPos * 0.1 + vec2(phase * 0.625, -phase * 0.375);
            float foamNoise1 = noise(foamCoord);
            float foamNoise2 = noise(foamCoord * 2.3 + vec2(1.7, 3.2));
            float foamTexture = foamNoise1 * 0.6 + foamNoise2 * 0.4;
            
            float waveCrestFoam = smoothstep(0.65, 0.95, vWaveIntensity);
            float shoreFoam = smoothstep(0.25, 0.0, depth);
            
            float breakingFoam = 0.0;
            if (depth < 0.15) {
                float breakingNoise = noise(vWorldPos * 0.2 + vec2(0.0, phase * 2.5));
                breakingFoam = (0.15 - depth) / 0.15 * breakingNoise;
            }
            
            float foamFactor = (waveCrestFoam * 0.4 + shoreFoam * 0.6 + breakingFoam * 0.8) * foamTexture * foamAmount;
            foamFactor = clamp(foamFactor, 0.0, 1.0);
            
            // Water color
            vec3 waterColor = getOceanColor(depth, vWaveIntensity, foamFactor);
            float shimmer = noise(vWorldPos * 0.3 + vec2(phase * 1.25, 0.0)) * 0.15 + 0.85;
            waterColor *= shimmer;
            float lighting = 0.75 + vWaveIntensity * 0.5;
            waterColor *= lighting;
            
            // Blend beach and water
            vec3 color;
            float alpha;
            
            if (beachZone > 0.01) {  // Render beach in entire zone, not just beachZone > 0.5
                // Mostly beach
                color = sandColor;
                
                // Add foam line at water's edge (adjusted for tide)
                float foamLineStart = wetThreshold;
                float foamLineEnd = dryThreshold - 0.01;
                float foamLine = smoothstep(foamLineStart - 0.02, foamLineStart, y) * smoothstep(foamLineEnd + 0.02, foamLineStart, y);
                float foamLineNoise = noise(vec2(x * 20.0, phase * 2.5));
                foamLine *= foamLineNoise * 0.5 + 0.5;
                vec3 foamColor = vec3(0.95, 0.95, 1.0);
                color = mix(color, foamColor, foamLine * 0.7);
                
                // Underwater sand has water overlay (adjusted for tide)
                if (y < wetThreshold) {
                    float waterOverlay = smoothstep(wetThreshold, beachStart, y);
                    color = mix(color, waterColor, waterOverlay * 0.6);
                    alpha = mix(0.95, 0.75, waterOverlay);
                } else {
                    alpha = 0.95;
                }
            } else {
                // Pure water
                color = waterColor;
                alpha = 0.5 + (1.0 - depth) * 0.3;
                alpha += foamFactor * 0.3;
                alpha = clamp(alpha, 0.0, 0.95);
            }
            
            alpha *= fadeAlpha;
            
            outColor = vec4(color, alpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Generate VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Generate and bind position VBO
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Generate and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        # Update animation time
        self.time += dt
        
        # Accumulate phase based on current wave_speed to avoid discontinuities
        # Scale wave_speed (0-1 range) to appropriate phase rate
        phase_rate = self.wave_speed * 2.0 * 0.8  # Same scaling as before
        self.phase += dt * phase_rate
    
    def render(self, state: Dict):
        """Render the ocean waves effect"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        
        # Set uniforms
        resolution_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(resolution_loc, float(self.viewport.width), float(self.viewport.height))
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        phase_loc = glGetUniformLocation(self.shader, "phase")
        glUniform1f(phase_loc, self.phase)
        
        height_loc = glGetUniformLocation(self.shader, "height")
        glUniform1f(height_loc, float(self.height))
        
        wave_speed_loc = glGetUniformLocation(self.shader, "waveSpeed")
        glUniform1f(wave_speed_loc, self.wave_speed)
        
        mesh_start_loc = glGetUniformLocation(self.shader, "meshStart")
        glUniform1f(mesh_start_loc, self.mesh_start)
        
        mesh_end_loc = glGetUniformLocation(self.shader, "meshEnd")
        glUniform1f(mesh_end_loc, self.mesh_end)
        
        wave_height_loc = glGetUniformLocation(self.shader, "waveHeight")
        glUniform1f(wave_height_loc, self.wave_height)
        
        tide_level_loc = glGetUniformLocation(self.shader, "tideLevel")
        glUniform1f(tide_level_loc, self.tide_level)
        
        foam_loc = glGetUniformLocation(self.shader, "foamAmount")
        glUniform1f(foam_loc, self.foam_amount)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        # Render mesh
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLE_STRIP, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.position_VBO is not None:
            glDeleteBuffers(1, [self.position_VBO])
        if self.EBO is not None:
            glDeleteBuffers(1, [self.EBO])
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
