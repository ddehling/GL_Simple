"""
Twinkling stars effect - rendering + event integration
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_stars(state, outstate, num_stars=1000, twinkle_speed=1.0, drift_x=1.0, drift_y=0.0, audio_sensitivity=1.5):
    """
    Shader-based twinkling stars effect compatible with EventScheduler
    
    Stars twinkle based on audio input using norm_long_relu bands 0-15.
    Each star is assigned to one of 16 frequency bands and reacts to that band.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_stars, num_stars=150, drift_x=5.0, 
                                drift_y=-2.0, audio_sensitivity=2.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        num_stars: Number of stars to render
        twinkle_speed: Speed multiplier for base twinkling animation
                drift_x: Horizontal drift base speed (default 1.0, stars will move 0.1-0.5 px/s)
        drift_y: Vertical drift speed (pixels per second, default 0.0)
        audio_sensitivity: Multiplier for audio reactivity (0 = no audio, higher = more reactive)
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
    
    # Initialize stars effect on first call
    if state['count'] == 0:
        print(f"Initializing stars effect for frame {frame_id} with {num_stars} stars")
        
        try:
            stars_effect = viewport.add_effect(
                TwinklingStarsEffect,
                num_stars=num_stars,
                twinkle_speed=twinkle_speed,
                drift_x=drift_x,
                drift_y=drift_y,
                audio_sensitivity=audio_sensitivity
            )
            state['stars_effect'] = stars_effect
            print(f"✓ Initialized shader stars for frame {frame_id} (depth: 99.99, audio bands: 0-15)")
        except Exception as e:
            print(f"✗ Failed to initialize stars: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update parameters from global state and audio data
    if 'stars_effect' in state:
        state['stars_effect'].twinkle_speed = outstate.get('twinkle_speed', twinkle_speed)
        state['stars_effect'].drift_x = outstate.get('drift_x', drift_x)
        state['stars_effect'].drift_y = outstate.get('drift_y', drift_y)
        state['stars_effect'].starryness = outstate.get('starryness', 1.0)
        state['stars_effect'].audio_sensitivity = outstate.get('audio_sensitivity', audio_sensitivity)
        
        # Update audio bands for twinkling (if audio data available)
        if audio_data is not None:
            # Get norm_long_relu bands 0-15 (current frame)
            audio_bands = audio_data['norm_long_relu'][0][0:16]  # Shape: (16,)
            state['stars_effect'].audio_bands = audio_bands
    # On close event, clean up
    if state['count'] == -1:
        if 'stars_effect' in state:
            print(f"Cleaning up stars effect for frame {frame_id}")
            viewport.effects.remove(state['stars_effect'])
            state['stars_effect'].cleanup()
            print(f"✓ Cleaned up shader stars for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class TwinklingStarsEffect(ShaderEffect):
    """GPU-based twinkling stars using instanced rendering with audio reactivity
    
    Stars are placed at depth z=99.99 (far back in the scene).
    Each star is assigned to one of 16 audio bands (0-15) and twinkles
    based on the energy in that band using norm_long_relu.
    """
    
    def __init__(self, viewport, num_stars: int = 100, twinkle_speed: float = 1.0, 
                 drift_x: float = 0.0, drift_y: float = 0.0, depth: float = 99.99, 
                 audio_sensitivity: float = 1.5):
        super().__init__(viewport)
        self.num_stars = num_stars
        self.twinkle_speed = twinkle_speed
        self.drift_x = drift_x  # Pixels per second
        self.drift_y = drift_y  # Pixels per second
        self.depth = depth  # Z depth (default 99.99 = far back)
        self.starryness = 1.0  # Global brightness scalar
        self.audio_sensitivity = audio_sensitivity  # Audio reactivity multiplier
        self.instance_VBO = None
        self.time = 0.0
        
        # Audio reactivity
        self.audio_bands = np.zeros(16, dtype=np.float32)  # Current audio energy for bands 0-15
        
        # Vectorized star data
        self.positions = None  # [x, y, z] - shape (N, 3)
        self.sizes = None  # Base sizes - shape (N,)
        self.colors = None  # [r, g, b] - shape (N, 3)
        self.audio_band_indices = None  # Which audio band (0-15) each star reacts to - shape (N,)
        self.twinkle_phases = None  # Phase offset for each star - shape (N,)
        self.twinkle_frequencies = None  # How fast each star twinkles - shape (N,)
        self.twinkle_amplitudes = None  # How much each star twinkles - shape (N,)
        self.drift_multipliers = None  # Speed multiplier for each star - shape (N,)
        
        self._initialize_stars()
        
    def _initialize_stars(self):
        """Initialize all star data as numpy arrays"""
        n = self.num_stars
        
        # Random positions across screen with Z at specified depth
        self.positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n),
            np.random.uniform(0, self.viewport.height, n),
            np.full(n, self.depth)  # Use configurable depth
        ])
        # Star sizes - mix of small and larger stars
        # 70% small, 20% medium, 10% large
        size_types = np.random.random(n)
        self.sizes = np.where(
            size_types < 0.7,
            np.random.uniform(0.5, 1.0, n),  # Small stars
            np.where(
                size_types < 0.9,
                np.random.uniform(1.0, 2.0, n),  # Medium stars
                np.random.uniform(2.0, 3.0, n)   # Large stars
            )
        )
        
        # Colors - mostly white with slight tints
        color_types = np.random.random(n)
        self.colors = np.where(
            color_types[:, np.newaxis] < 0.6,
            # 60% pure white
            np.column_stack([np.ones(n), np.ones(n), np.ones(n)]),
            np.where(
                color_types[:, np.newaxis] < 0.85,
                # 25% warm white (slight yellow)
                np.column_stack([
                    np.random.uniform(0.6, 1.0, n),
                    np.random.uniform(0.6, 1.0, n),
                    np.random.uniform(0.6, 0.9, n)
                ]),
                # # 15% cool white (slight blue)
                # np.column_stack([
                #     np.random.uniform(0.8, 0.95, n),
                #     np.random.uniform(0.85, 0.95, n),
                #     np.ones(n)
                # ])
                # ,
                # 15% cool white (slight blue)
                np.column_stack([
                    np.random.uniform(0.15, 0.95, n),
                    np.random.uniform(0.15, 0.95, n),
                    np.random.uniform(0.15, 0.95, n)
                ])
            )
        )
        
                # Audio band assignment - each star assigned to one of 16 bands
        self.audio_band_indices = np.random.randint(0, 16, n)
        
        # Twinkling parameters - each star has unique behavior
        self.twinkle_phases = np.random.uniform(0, 2 * np.pi, n)
        self.twinkle_frequencies = np.random.uniform(0.5, 2.0, n)
        self.twinkle_amplitudes = np.random.uniform(0.3, 0.7, n)
        
                # Drift speed multipliers - random speed between 0.1 to 0.5 pixels per second
        # When combined with drift_x=1.0, gives individual speeds of 0.1-0.5 px/s
        self.drift_multipliers = np.random.uniform(0.1, 0.5, n)
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (centered -1 to 1)
        layout(location = 1) in vec3 starPos;   // Star position with Z
        layout(location = 2) in float starSize; // Star size
        layout(location = 3) in vec3 starColor; // Star color
        layout(location = 4) in float alpha;    // Twinkle alpha

        out vec4 fragColor;
        out vec2 texCoord;  // For creating star shape
        uniform vec2 resolution;

        void main() {
            // Scale quad by star size
            vec2 scaled = position * starSize;
            
            // Translate to star position (x, y)
            vec2 pos = scaled + starPos.xy;
            
            // Convert screen coordinates to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use Z for depth (normalize to 0-1 range)
            float depth = starPos.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Pass color with alpha
            fragColor = vec4(starColor, alpha);
            
            // Pass texture coordinates for star shape
            texCoord = position;  // -1 to 1 range
        }
        """

        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 texCoord;
        out vec4 outColor;

        void main() {
            // Create a star shape using distance from center
            float dist = length(texCoord);
            
            // Soft circular falloff for glow effect
            float glow = 1.0 - smoothstep(0.0, 1.0, dist);
            
            // Sharper center for star core
            float core = 1.0 - smoothstep(0.0, 0.3, dist);
            
            // Combine core and glow
            float intensity = core + glow * 0.5;
            
            // Apply to color and alpha
            vec3 finalColor = fragColor.rgb * intensity;
            float finalAlpha = fragColor.a * intensity;
            
            outColor = vec4(finalColor, finalAlpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link star shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            # Set resolution uniform
            glUseProgram(shader)
            loc = glGetUniformLocation(shader, "resolution")
            if loc != -1:
                glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
            glUseProgram(0)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise

    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced rendering"""
        # Quad vertices - centered from -1 to 1 for proper circular star shape
        vertices = np.array([
            -1.0, -1.0,  # Bottom left
             1.0, -1.0,  # Bottom right
             1.0,  1.0,  # Top right
            -1.0,  1.0   # Top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer (quad template)
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
        
        # Instance buffer (will be updated each frame)
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        """Update star twinkling animation and drift"""
        if not self.enabled:
            return
        
        # Update time for twinkling
        self.time += dt * self.twinkle_speed
        
        # Apply horizontal drift with individual star speeds (0.1-0.5 px/s)
        # Stars move left to right and wrap around
        self.positions[:, 0] += self.drift_x * dt * self.drift_multipliers
        
        # Apply vertical drift if specified
        if self.drift_y != 0.0:
            self.positions[:, 1] += self.drift_y * dt * self.drift_multipliers
        
        # Wrap around screen edges using modulo
        self.positions[:, 0] = self.positions[:, 0] % self.viewport.width
        self.positions[:, 1] = self.positions[:, 1] % self.viewport.height


    def render(self, state: Dict):
        """Render all stars using instancing"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        # NO depth test or blend toggling - these are set globally!
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
                # Calculate base twinkling brightness for each star
        twinkle_values = np.sin(
            self.time * self.twinkle_frequencies + self.twinkle_phases
        )
        # Map from [-1, 1] to brightness range
        base_alphas = 0.6 + self.twinkle_amplitudes * twinkle_values * 0.4
        
        # Add audio reactivity - get audio energy for each star's assigned band
        audio_modulation = self.audio_bands[self.audio_band_indices] * self.audio_sensitivity
        # Clamp audio contribution to reasonable range (0 to 0.4 additional brightness)
        audio_modulation = np.clip(audio_modulation, 0, 0.4)
        
        # Combine base twinkling with audio reactivity
        alphas = base_alphas + audio_modulation
        
        # Apply global starryness brightness scalar
        alphas = alphas * self.starryness
        
        # Clamp final alpha to valid range
        alphas = np.clip(alphas, 0, 1)
        
        # Build instance data - interleave all attributes
        instance_data = np.hstack([
            self.positions,  # x, y, z (3 floats)
            self.sizes[:, np.newaxis],  # size (1 float)
            self.colors,  # r, g, b (3 floats)
            alphas[:, np.newaxis]  # alpha (1 float)
        ]).astype(np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 8 * 4  # 8 floats * 4 bytes
        
        # Star position (location 1) - vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Star size (location 2) - float
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Star color (location 3) - vec3
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Alpha (location 4) - float
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
                # Draw all stars in one call
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_stars)
        
        glBindVertexArray(0)
        glUseProgram(0)
