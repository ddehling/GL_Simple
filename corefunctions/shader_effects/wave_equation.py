"""
Wave Equation shader effect - Mathematical wave terrain with color gradients
Based on Shadertoy shader by sofiane benchaa (sben/2015)
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_wave_equation(state, outstate, fade_duration=5.0, field_scale=20.0, 
                         height_offset=0.7, iterations=2, tone_r=0.5, tone_g=0.2, tone_b=0.3):
    """
    Mathematical wave equation effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_wave_equation, 
                               field_scale=20.0, iterations=2, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        fade_duration: Duration of fade in/out in seconds (default 5.0)
        field_scale: Scale of the wave field (default 20.0)
        height_offset: Height offset for waves (default 0.7)
        iterations: Number of wave iterations (default 2)
        tone_r: Red component of tone color (default 0.5)
        tone_g: Green component of tone color (default 0.2)
        tone_b: Blue component of tone color (default 0.3)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing wave_equation for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                WaveEquationEffect,
                field_scale=field_scale,
                height_offset=height_offset,
                iterations=iterations,
                tone_color=(tone_r, tone_g, tone_b)
            )
            state['wave_equation_effect'] = effect
            print(f"✓ Initialized shader wave_equation for frame {frame_id}")
        except Exception as e:
            import traceback
            print(f"ERROR initializing wave_equation effect: {e}")
            traceback.print_exc()
            return
    
    # Update effect parameters from outstate (optional audio reactivity)
    if 'wave_equation_effect' in state:
        effect = state['wave_equation_effect']
        
        # Optional: Get audio data for reactivity
        audio_data = outstate.get('sound')
        if audio_data is not None:
            # React to bass frequencies
            bands = audio_data['norm_short'][0]
            bass_energy = np.mean(bands[0:8])
            effect.audio_intensity = bass_energy
        
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)  # Default 60s if not set
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            # Fade in
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            # Fade out
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            # Full visibility
            fade_factor = 1.0
        
        # Update effect's fade factor (clip to 0-1 range)
        effect.fade_factor = np.clip(fade_factor, 0, 1)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'wave_equation_effect' in state:
            effect = state['wave_equation_effect']
            if effect in viewport.effects:
                viewport.effects.remove(effect)
            effect.cleanup()
            del state['wave_equation_effect']
            print(f"✓ Cleaned up shader wave_equation for frame {frame_id}")


# ============================================================================
# Wave Equation Effect Class
# ============================================================================

class WaveEquationEffect(ShaderEffect):
    """GPU-based wave equation effect rendering fullscreen mathematical waves"""
    
    def __init__(self, viewport, field_scale: float = 20.0, height_offset: float = 0.7,
                 iterations: int = 2, tone_color: tuple = (0.5, 0.2, 0.3)):
        super().__init__(viewport)
        self.field_scale = field_scale
        self.height_offset = height_offset
        self.iterations = iterations
        self.tone_color = tone_color
        self.fade_factor = 0.0  # For fade in/out
        self.audio_intensity = 0.0  # For audio reactivity
        
        # Time tracking
        self.time = 0.0
    
    def update(self, dt: float, state: Dict):
        """Update time"""
        if not self.enabled:
            return
        
        self.time += dt
    
    def render(self, state: Dict):
        """Render fullscreen wave equation effect"""
        if not self.enabled:
            return
        
        # This is a post-processing effect (fullscreen quad)
        # Use glDepthFunc instead of toggling depth test
        glDepthFunc(GL_ALWAYS)  # Always pass depth test
        glDepthMask(GL_FALSE)   # Don't write to depth buffer
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        field_loc = glGetUniformLocation(self.shader, "fieldScale")
        glUniform1f(field_loc, self.field_scale)
        
        height_loc = glGetUniformLocation(self.shader, "heightOffset")
        glUniform1f(height_loc, self.height_offset)
        
        iterations_loc = glGetUniformLocation(self.shader, "iterations")
        glUniform1i(iterations_loc, self.iterations)
        
        tone_loc = glGetUniformLocation(self.shader, "toneColor")
        glUniform3f(tone_loc, self.tone_color[0], self.tone_color[1], self.tone_color[2])
        
        audio_loc = glGetUniformLocation(self.shader, "audioIntensity")
        glUniform1f(audio_loc, self.audio_intensity)
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore default depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position, 0.0, 1.0);
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
        uniform float fadeAlpha;
        uniform float fieldScale;
        uniform float heightOffset;
        uniform int iterations;
        uniform vec3 toneColor;
        uniform float audioIntensity;
        
        // Wave equation function
        float eq(vec2 p, float t) {
            float x = sin(p.y + cos(t + p.x * 0.2)) * cos(p.x - t);
            x *= acos(clamp(x, -1.0, 1.0));
            return -x * abs(x - 0.5) * p.x / p.y;
        }
        
        void main() {
            // Convert screen coordinates to pixel space
            vec2 U = (fragCoord * 0.5 + 0.5) * resolution;
            
            // Initialize output
            vec4 O = vec4(0.0);
            vec4 X = vec4(0.0);
            
            // Calculate initial position exactly as original
            vec2 p = fieldScale * (U / resolution.xy + 0.5);
            
            float t = time;
            float hs = fieldScale * (heightOffset + cos(t) * 0.1 * (1.0 + audioIntensity));
            
            float x = eq(p, t);
            float y = p.y - x;
            
            // Multi-iteration wave accumulation
            for(float i = 0.0; i < float(iterations); i += 1.0) {
                p.x *= 2.0;
                
                X = vec4(
                    x,
                    eq(p, t + i + 1.0),
                    eq(p, t + i + 2.0),
                    0.0
                );
                x = X.z + X.y;
                
                // Accumulate color based on distance from wave (divide by 0.5 to make lines 2x thicker)
                O += vec4(toneColor, 0.0) / (abs(y - X - hs) * 0.5);
            }
            
            // Calculate alpha based on color intensity (where there's color, there's alpha)
            float intensity = length(O.rgb);
            O.a = min(intensity, 1.0) * fadeAlpha;
            
            // Apply fade factor to RGB
            O.rgb *= fadeAlpha;
            
            // Clamp RGB to prevent overflow
            O.rgb = clamp(O.rgb, 0.0, 1.0);
            
            outColor = O;
        }
        """
    
    def compile_shader(self):
        """Compile and link wave equation shaders - REQUIRED by ShaderEffect base class"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vertex = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            fragment = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vertex, fragment)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for fullscreen quad rendering"""
        # Fullscreen quad vertices (normalized device coordinates)
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
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
