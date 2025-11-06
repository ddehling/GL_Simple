"""
Aurora shader effect - Flowing northern lights at top of screen
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_aurora(state, outstate, height=200, depth=50.0, intensity=1.0, speed=1.0):
    """
    Shader-based aurora effect compatible with EventScheduler
    
    Creates flowing northern lights at the top of the screen with animated
    waves and color gradients.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_aurora, height=250, depth=50, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        height: Height of aurora effect in pixels (default: 200)
        depth: Z-depth of aurora (0=near, 100=far, default: 50)
        intensity: Brightness multiplier (default: 1.0)
        speed: Animation speed multiplier (default: 1.0)
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
        print(f"Initializing aurora for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                Aurora,
                height=height,
                depth=depth,
                intensity=intensity,
                speed=speed
            )
            state['effect'] = effect
            print(f"✓ Initialized shader aurora for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize aurora: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from global state (optional)
    if 'effect' in state:
        state['effect'].intensity = outstate.get('aurora_intensity', intensity)
        state['effect'].speed = outstate.get('aurora_speed', speed)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up aurora for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader aurora for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class Aurora(ShaderEffect):
    """
    Aurora borealis effect with flowing waves and color gradients
    
    Creates animated northern lights at the top of the screen using
    layered sine waves with varying frequencies and amplitudes.
    """
    
    def __init__(self, viewport, height: float = 200, depth: float = 50.0, 
                 intensity: float = 1.0, speed: float = 1.0):
        super().__init__(viewport)
        self.height = height
        self.depth = depth
        self.intensity = intensity
        self.speed = speed
        self.time = 0.0
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.EBO = None
        
                # Mesh resolution (higher = smoother waves)
        self.segments_x = 100  # Horizontal segments
        self.segments_y = 20   # Vertical segments
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize mesh data for aurora plane"""
        # Create a grid mesh at the top of the screen
        width = self.viewport.width
        
        # Generate vertex positions for a grid
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
        """Compile and link aurora shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Aurora shader compilation error: {e}")
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
        uniform float speed;
        
        out vec2 vTexCoord;
        out float vWave;
        
        void main() {
            vec2 pos = position;
            
            // Calculate normalized coordinates
            float normY = pos.y / height;  // 0 at top, 1 at bottom
            float normX = pos.x / resolution.x;
            
            // Create flowing wave pattern
            float wave1 = sin(normX * 6.28318 * 2.0 + time * speed * 0.5) * 0.3;
            float wave2 = sin(normX * 6.28318 * 3.0 - time * speed * 0.7) * 0.2;
            float wave3 = sin(normX * 6.28318 * 1.5 + time * speed * 0.3) * 0.25;
            
            float totalWave = (wave1 + wave2 + wave3) * normY * 30.0;
            pos.y += totalWave;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, mappedDepth, 1.0);
            
            // Pass data to fragment shader
            vTexCoord = vec2(normX, normY);
            vWave = (wave1 + wave2 + wave3) * 0.5 + 0.5;  // Normalize to 0-1
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vTexCoord;
        in float vWave;
        
        uniform float time;
        uniform float intensity;
        uniform float speed;
        
        out vec4 outColor;
        
        // Aurora color palette
        vec3 getAuroraColor(float t, float wave) {
            // Blend between aurora colors: green, blue, purple, pink
            vec3 color1 = vec3(0.0, 1.0, 0.4);    // Bright green
            vec3 color2 = vec3(0.0, 0.8, 1.0);    // Cyan
            vec3 color3 = vec3(0.5, 0.0, 1.0);    // Purple
            vec3 color4 = vec3(1.0, 0.2, 0.8);    // Pink
            
            // Animate color shifts
            float colorShift = sin(time * speed * 0.2 + t * 3.14159) * 0.5 + 0.5;
            float waveInfluence = wave * 0.3;
            
            vec3 color;
            float blendPos = mod(t + colorShift + waveInfluence, 1.0) * 3.0;
            
            if (blendPos < 1.0) {
                color = mix(color1, color2, blendPos);
            } else if (blendPos < 2.0) {
                color = mix(color2, color3, blendPos - 1.0);
            } else {
                color = mix(color3, color4, blendPos - 2.0);
            }
            
            return color;
        }
        
        void main() {
            float x = vTexCoord.x;
            float y = vTexCoord.y;
            
            // Create flowing bands
            float bands = sin(x * 6.28318 * 4.0 + time * speed * 0.8 + vWave * 2.0) * 0.5 + 0.5;
            bands = pow(bands, 1.5);
            
            // Vertical gradient - brighter at top, fade toward bottom
            float verticalFade = 1.0 - y;
            verticalFade = pow(verticalFade, 0.7);
            
            // Add shimmer effect
            float shimmer = sin(x * 20.0 + time * speed * 2.0) * 
                          sin(y * 15.0 - time * speed * 1.5) * 0.1 + 0.9;
            
            // Get aurora color
            vec3 color = getAuroraColor(x * 2.0 + time * speed * 0.1, vWave);
            
            // Combine effects
            float brightness = bands * verticalFade * shimmer * intensity;
            
            // Add subtle glow at edges of bands
            float edgeGlow = smoothstep(0.3, 0.7, bands) * (1.0 - smoothstep(0.7, 1.0, bands));
            brightness += edgeGlow * 0.3;
            
            // Calculate alpha - more transparent at bottom
            float alpha = verticalFade * bands * 0.7;
            alpha = clamp(alpha, 0.0, 0.85);
            
            outColor = vec4(color * brightness, alpha);
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
        """Render the aurora effect"""
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
        glUniform1f(height_loc, self.height)
        
        intensity_loc = glGetUniformLocation(self.shader, "intensity")
        glUniform1f(intensity_loc, self.intensity)
        
        speed_loc = glGetUniformLocation(self.shader, "speed")
        glUniform1f(speed_loc, self.speed)
        
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
