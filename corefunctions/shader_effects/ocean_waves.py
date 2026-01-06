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
    
    # Update from global state (optional)
    if 'effect' in state:
        state['effect'].wave_speed = outstate.get('ocean_speed', wave_speed)
        state['effect'].wave_height = outstate.get('ocean_height', wave_height)
        
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
                 wave_speed: float = 1.0, wave_height: float = 1.0, foam_amount: float = 1.0):
        super().__init__(viewport)
        self.height_ratio = height_ratio
        self.height = int(viewport.height * height_ratio)
        self.depth = depth
        self.wave_speed = wave_speed
        self.wave_height = wave_height
        self.foam_amount = foam_amount
        self.time = 0.0
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
        
        # Create a grid mesh covering the viewport
        vertices = []
        for y in range(self.segments_y + 1):
            y_pos = (y / self.segments_y) * self.height
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
        uniform float height;
        uniform float waveSpeed;
        uniform float waveHeight;
        
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
            
            // Normalize coordinates (0-1)
            float normY = pos.y / height;  // 0 at bottom, 1 at top (shore)
            float normX = pos.x / resolution.x;
            
            // Distance from shore: 0 at shore (top), 1 at deep water (bottom)
            float shoreDistance = 1.0 - normY;
            
            // Create wave fronts that roll toward shore
            // Wave phase moves with time AND position to create motion
            float wavePhase = shoreDistance * 8.0 - time * waveSpeed * 1.5;
            
            // Add horizontal variation with noise to break up vertical stripes
            float horizontalVariation = noise(vec2(normX * 3.0, time * waveSpeed * 0.1)) * 2.0;
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
            float turbulence = noise(vec2(normX * 5.0 + time * waveSpeed * 0.5, 
                                          shoreDistance * 4.0 - time * waveSpeed * 0.3));
            waveHeight_combined += turbulence * 0.15;
            
            // Waves grow taller as they approach shore (shoaling)
            float shoaling = 1.0 + pow(1.0 - shoreDistance, 1.5) * 2.5;
            waveHeight_combined *= shoaling;
            
            // Breaking wave at shore - waves curl over and crash
            if (shoreDistance < 0.15) {
                // Extreme steepening and chaos at shore
                float breakingIntensity = (0.15 - shoreDistance) / 0.15;
                float breakingNoise = noise(vec2(normX * 15.0, time * waveSpeed * 3.0));
                waveHeight_combined += breakingIntensity * breakingNoise * 1.5;
            }
            
            // Apply wave height displacement
            float totalHeight = waveHeight_combined * 30.0 * waveHeight;
            pos.y += totalHeight;
            
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
        uniform float waveSpeed;
        uniform float foamAmount;
        uniform float fadeAlpha;
        
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
            // Deep ocean: dark blue-green
            vec3 deepColor = vec3(0.0, 0.15, 0.3);
            
            // Mid water: teal
            vec3 midColor = vec3(0.0, 0.4, 0.5);
            
            // Shallow/shore: cyan
            vec3 shallowColor = vec3(0.1, 0.6, 0.7);
            
            // Foam: white with slight blue tint
            vec3 foamColor = vec3(0.9, 0.95, 1.0);
            
            // Blend based on distance from shore (depth)
            vec3 baseColor;
            if (depth > 0.7) {
                baseColor = mix(midColor, deepColor, (depth - 0.7) / 0.3);
            } else if (depth > 0.3) {
                baseColor = mix(shallowColor, midColor, (depth - 0.3) / 0.4);
            } else {
                baseColor = shallowColor;
            }
            
            // Add foam highlights on wave crests
            baseColor = mix(baseColor, foamColor, foam);
            
            return baseColor;
        }
        
        void main() {
            float x = vTexCoord.x;
            float depth = vDistanceFromShore;
            
            // Create foam texture using noise (no vertical stripes!)
            vec2 foamCoord = vWorldPos * 0.1 + vec2(time * waveSpeed * 0.5, -time * waveSpeed * 0.3);
            float foamNoise1 = noise(foamCoord);
            float foamNoise2 = noise(foamCoord * 2.3 + vec2(1.7, 3.2));
            float foamTexture = foamNoise1 * 0.6 + foamNoise2 * 0.4;
            
            // Foam appears on wave crests
            float waveCrestFoam = smoothstep(0.65, 0.95, vWaveIntensity);
            
            // Heavy foam at shore where waves break
            float shoreFoam = smoothstep(0.25, 0.0, depth);
            
            // Breaking wave foam (chaotic white water right at shore)
            float breakingFoam = 0.0;
            if (depth < 0.15) {
                float breakingNoise = noise(vWorldPos * 0.2 + vec2(0.0, time * waveSpeed * 2.0));
                breakingFoam = (0.15 - depth) / 0.15 * breakingNoise;
            }
            
            // Combine all foam sources
            float foamFactor = (waveCrestFoam * 0.4 + shoreFoam * 0.6 + breakingFoam * 0.8) * foamTexture * foamAmount;
            foamFactor = clamp(foamFactor, 0.0, 1.0);
            
            // Get ocean color with foam
            vec3 color = getOceanColor(depth, vWaveIntensity, foamFactor);
            
            // Add subtle shimmer on wave surfaces
            float shimmer = noise(vWorldPos * 0.3 + vec2(time * waveSpeed, 0.0)) * 0.15 + 0.85;
            color *= shimmer;
            
            // Lighting based on wave height (peaks are brighter)
            float lighting = 0.75 + vWaveIntensity * 0.5;
            color *= lighting;
            
            // Calculate alpha - more opaque at shore, semi-transparent in deep water
            float alpha = 0.5 + (1.0 - depth) * 0.3;
            alpha += foamFactor * 0.3;  // Foam is more opaque
            alpha = clamp(alpha, 0.0, 0.95);
            
            // Apply fade in/out
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
        
        height_loc = glGetUniformLocation(self.shader, "height")
        glUniform1f(height_loc, float(self.height))
        
        wave_speed_loc = glGetUniformLocation(self.shader, "waveSpeed")
        glUniform1f(wave_speed_loc, self.wave_speed)
        
        wave_height_loc = glGetUniformLocation(self.shader, "waveHeight")
        glUniform1f(wave_height_loc, self.wave_height)
        
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
