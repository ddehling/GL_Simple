"""
Aurora effect - shader-based implementation
Converts the CPU-based Aurora effect to GPU rendering using OpenGL shaders
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import time
from corefunctions.shader_effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_aurora(state, outstate,side='top'):
    """
    Shader-based aurora effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_aurora, frame_id=0, side='top')
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id, side)
        outstate: Global state dict (from EventScheduler)
    """
    # Get the viewport
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize aurora effect on first call
    if state['count'] == 0:
        print(f"Initializing shader aurora effect for frame {frame_id}")
        
        # Get side parameter (top, bottom, left, right)
        side = side
        
        try:
            aurora_effect = viewport.add_effect(AuroraEffect, side=side)
            state['aurora_effect'] = aurora_effect
            state['start_time'] = time.time()
            
            # Initialize wave parameters - make them periodic/circular
            num_points = 20
            state['wave_points'] = np.linspace(0, 1, num_points + 1)[:-1]  # Exclude last point to avoid duplication
            
            # Generate random offsets for each wave point
            state['wave_offsets'] = np.random.uniform(0, 2 * np.pi, num_points)
            state['wave_amplitudes'] = np.random.uniform(0.15, 0.25, num_points)
            
            state['base_hue'] = 0.3 + np.random.random() * 0.4
            state['time_offset'] = time.time()
            state['whomp'] = 0.0
            
            print(f"✓ Initialized shader aurora for frame {frame_id} (side: {side})")
        except Exception as e:
            print(f"✗ Failed to initialize aurora: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update aurora parameters from global state
    if 'aurora_effect' in state:
        current_time = time.time()
        elapsed_time = current_time - state['start_time']
        total_duration = state.get('duration', 30)
        
        # Calculate fade factor
        fade_in_duration = total_duration * 0.2
        fade_out_start = total_duration * 0.8
        
        if elapsed_time < fade_in_duration:
            fade_factor = elapsed_time / fade_in_duration
        elif elapsed_time > fade_out_start:
            fade_factor = (total_duration - elapsed_time) / (total_duration * 0.2)
        else:
            fade_factor = 1.0
        fade_factor = np.clip(fade_factor, 0, 1)
        
        # Update whomp (audio reactive parameter)
        whompin = outstate.get('whomp', 0.0)
        state['whomp'] = whompin * 0.3 + 0.7 * state.get('whomp', 0.0)
        
        # Update wave heights with smooth movement - using sine waves for smooth periodic motion
        wave_speed = 0.5
        time_factor = (current_time - state['time_offset']) * wave_speed
        
        # Generate wave heights using sine functions with the pre-generated offsets
        # This creates a naturally periodic pattern
        num_points = len(state['wave_offsets'])
        base_wave = 0.45  # Center position
        wave_heights = base_wave + np.array([
            state['wave_amplitudes'][i] * np.sin(time_factor + state['wave_offsets'][i])
            for i in range(num_points)
        ])
        
        state['wave_heights'] = np.clip(wave_heights, 0.25, 0.7)
        
        # Update effect parameters
        effect = state['aurora_effect']
        effect.fade_factor = fade_factor
        effect.intensity = outstate.get('aurora_intensity', 1.0)
        effect.base_hue = state['base_hue']
        effect.wave_heights = state['wave_heights']
        effect.wave_points = state['wave_points']
        effect.whomp = state['whomp']
        effect.time_factor = time_factor
    
    # On close event, clean up
    if state['count'] == -1:
        if 'aurora_effect' in state:
            print(f"Cleaning up aurora effect for frame {frame_id}")
            viewport.effects.remove(state['aurora_effect'])
            state['aurora_effect'].cleanup()
            print(f"✓ Cleaned up shader aurora for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class AuroraEffect(ShaderEffect):
    """GPU-based aurora effect using procedural shader rendering"""
    
    def __init__(self, viewport, side='top', depth=49.9):
        super().__init__(viewport)
        
        # Aurora properties
        self.side = side  # 'top', 'bottom', 'left', 'right'
        self.depth = depth  # Z depth (0-100, lower = closer)
        self.fade_factor = 0.0
        self.intensity = 1.0
        
        # Wave parameters
        self.base_hue = 0.5
        self.wave_points = np.linspace(0, 1, 20)
        self.wave_heights = np.random.uniform(0.3, 0.6, 20)
        self.whomp = 0.0
        self.time_factor = 0.0
        
        # Start time
        self.start_time = time.time()
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Full-screen quad
        
        out vec2 fragCoord;  // Pass screen coordinate to fragment shader
        
        uniform float auroraDepth;  // Z depth
        
        void main() {
            // Normalize depth to 0-1 range (0 = near, 1 = far)
            float depth = auroraDepth / 100.0;
            
            gl_Position = vec4(position, depth, 1.0);
            
            // Convert from clip space (-1,1) to screen space (0,1)
            fragCoord = (position + 1.0) * 0.5;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform vec2 resolution;
        uniform int side;  // 0=top, 1=bottom, 2=left, 3=right
        uniform float fadeAlpha;
        uniform float intensity;
        uniform float baseHue;
        uniform float waveHeights[20];  // Wave control points
        uniform float timeFactor;
        uniform float whomp;
        
        // Random function for noise
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        
        // Convert HSV to RGB
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        // Interpolate wave heights with wrapping
        float getWaveHeight(float x) {
            // x is in 0-1 range
            // Wrap x to ensure continuity
            x = fract(x);  // Ensure x is in 0-1
            
            float index = x * 20.0;  // 20 points
            int i0 = int(floor(index)) % 20;
            int i1 = (i0 + 1) % 20;  // Wrap around to create continuous pattern
            float t = fract(index);
            
            return mix(waveHeights[i0], waveHeights[i1], t);
        }
        
        void main() {
            vec2 uv = fragCoord;
            float wavePos, fadePos;
            
            // Transform UV based on side
            if (side == 0) {
                // Top: aurora comes down from top
                wavePos = uv.x;
                fadePos = 1.0 - uv.y;  // 1.0 at top, 0.0 at bottom
            } else if (side == 1) {
                // Bottom: aurora comes up from bottom
                wavePos = uv.x;
                fadePos = uv.y;  // 1.0 at bottom, 0.0 at top
            } else if (side == 2) {
                // Left: aurora comes right from left
                wavePos = uv.y;
                fadePos = uv.x;  // 1.0 at left, 0.0 at right
            } else {
                // Right: aurora comes left from right
                wavePos = uv.y;
                fadePos = 1.0 - uv.x;  // 1.0 at right, 0.0 at left
            }
            
            // Get wave height for this position (now wraps seamlessly)
            float waveHeight = getWaveHeight(wavePos);
            
            // Calculate how far the aurora extends into the screen
            // waveHeight controls how far it reaches (0.25 to 0.7)
            float verticalDist = waveHeight - fadePos;
            float verticalFalloff = clamp(verticalDist / 0.35, 0.0, 1.0);
            
            // Add vertical streaks (also make them continuous)
            float streaks = sin(wavePos * 60.0 + timeFactor) * 0.1;
            verticalFalloff *= (1.0 + streaks);
            
            // Add noise for organic look
            float noise = random(uv + vec2(timeFactor * 0.1)) * 0.1;
            verticalFalloff = clamp(verticalFalloff + noise, 0.0, 1.0);
            
            // If no aurora at this position, discard
            if (verticalFalloff < 0.01) {
                discard;
            }
            
            // Create color variations (also continuous)
            float hueVariation = sin(wavePos * 6.283 + timeFactor) * 0.05;
            float hue = baseHue + hueVariation;
            float saturation = 0.8;
            float value = verticalFalloff * intensity * 0.7;
            
            // Convert to RGB
            vec3 color = hsv2rgb(vec3(hue, saturation, value));
            
            // Calculate alpha with whomp effect
            float alpha = verticalFalloff * 0.4 * intensity * fadeAlpha;
            alpha *= clamp(1.0 + whomp, 0.0, 2.0);
            
            outColor = vec4(color, alpha);
        }
        """

    
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
            print(f"Shader compilation error: {e}")
            raise

    def setup_buffers(self):
        """Initialize OpenGL buffers for full-screen quad"""
        # Full-screen quad in clip space
        vertices = np.array([
            -1.0, -1.0,  # Bottom left
             1.0, -1.0,  # Bottom right
             1.0,  1.0,  # Top right
            -1.0,  1.0   # Top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
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
        """Update aurora animation (called every frame)"""
        if not self.enabled:
            return
        
        # Most updates are handled in the shader_aurora() wrapper function
        # This is called by the viewport's update loop
        pass

    def render(self, state: Dict):
        """Render the aurora using shader"""
        if not self.enabled or not self.shader:
            return
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(self.shader)
        
        # Set uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        # Map side string to integer
        side_map = {'top': 0, 'bottom': 1, 'left': 2, 'right': 3}
        side_int = side_map.get(self.side, 0)
        
        loc = glGetUniformLocation(self.shader, "side")
        if loc != -1:
            glUniform1i(loc, side_int)
        
        loc = glGetUniformLocation(self.shader, "auroraDepth")
        if loc != -1:
            glUniform1f(loc, self.depth)
        
        loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        loc = glGetUniformLocation(self.shader, "intensity")
        if loc != -1:
            glUniform1f(loc, self.intensity)
        
        loc = glGetUniformLocation(self.shader, "baseHue")
        if loc != -1:
            glUniform1f(loc, self.base_hue)
        
        loc = glGetUniformLocation(self.shader, "timeFactor")
        if loc != -1:
            glUniform1f(loc, self.time_factor)
        
        loc = glGetUniformLocation(self.shader, "whomp")
        if loc != -1:
            glUniform1f(loc, self.whomp)
        
        # Set wave heights array
        loc = glGetUniformLocation(self.shader, "waveHeights")
        if loc != -1:
            glUniform1fv(loc, len(self.wave_heights), self.wave_heights.astype(np.float32))
        
        # Draw full-screen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        glDisable(GL_BLEND)