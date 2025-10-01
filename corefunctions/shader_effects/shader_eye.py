"""
Complete eye effect - rendering + event integration using shaders
Shader-based implementation following rain.py template
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

def shader_eye(state, outstate):
    """
    Shader-based eye effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_eye, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
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
    
    # Initialize eye effect on first call
    if state['count'] == 0:
        print(f"Initializing shader eye effect for frame {frame_id}")
        
        try:
            eye_effect = viewport.add_effect(EyeEffect)
            state['eye_effect'] = eye_effect
            state['start_time'] = time.time()
            print(f"✓ Initialized shader eye for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize eye: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update eye parameters from global state
    if 'eye_effect' in state:
        current_time = time.time()
        elapsed_time = current_time - state['start_time']
        total_duration = state.get('duration', 30)
        
        # Calculate fade factor
        fade_duration = 5.0
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        fade_factor = np.clip(fade_factor, 0, 1)
        
        state['eye_effect'].fade_factor = fade_factor
        state['eye_effect'].pupil_size = outstate.get('eye_pupil_size', 1.0)
        state['eye_effect'].movement_interval = outstate.get('eye_movement_interval', 3.0)
        state['eye_effect'].movement_speed = outstate.get('eye_movement_speed', 2.0)
        state['eye_effect'].blink_interval = outstate.get('eye_blink_interval', 7.0)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'eye_effect' in state:
            print(f"Cleaning up eye effect for frame {frame_id}")
            viewport.effects.remove(state['eye_effect'])
            state['eye_effect'].cleanup()
            print(f"✓ Cleaned up shader eye for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class EyeEffect(ShaderEffect):
    """GPU-based eye effect using procedural shader rendering"""
    
    def __init__(self, viewport, position=(0.5, 0.5), scale=2.0, depth=25.0):
        super().__init__(viewport)
        
        # Eye properties
        self.position = position  # Normalized screen position (0-1)
        self.scale = scale  # Size multiplier
        self.depth = depth  # Z depth (0-100, lower = closer)
        self.fade_factor = 0.0
        self.pupil_size = 1.0
        
        # Movement parameters
        self.movement_interval = 3.0
        self.movement_speed = 2.0
        self.blink_interval = 7.0
        
        # Movement state
        self.start_time = time.time()
        self.last_movement_time = time.time()
        self.last_update_time = time.time()
        self.target_x = 0.0
        self.target_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Full-screen quad
        
        out vec2 fragCoord;  // Pass screen coordinate to fragment shader
        
        uniform vec2 resolution;
        uniform vec2 eyePosition;  // Eye center in screen space (0-1)
        uniform float eyeScale;
        uniform float eyeDepth;  // Z depth
        
        void main() {
            // Normalize depth to 0-1 range (0 = near, 1 = far)
            float depth = eyeDepth / 100.0;
            
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
        uniform vec2 eyePosition;  // Normalized position (0-1)
        uniform float eyeScale;
        uniform float fadeAlpha;
        uniform float pupilSize;
        uniform vec2 irisOffset;  // Iris movement offset
        uniform float time;
        
        // Convert HSVA to RGBA
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            // Convert to pixel coordinates and center on eye
            vec2 aspect = vec2(resolution.x / resolution.y, 1.0);
            vec2 uv = (fragCoord - eyePosition) * aspect * 2.0;  // Scale to make eye visible
            uv /= eyeScale;
            
            // Eye dimensions (ellipse)
            float eyeWidth = 2.0;
            float eyeHeight = 1.0;
            
            // Distance from center (normalized to ellipse)
            float dist = length(vec2(uv.x / eyeWidth, uv.y / eyeHeight));
            
            // Discard pixels outside the eye
            if (dist > 1.0) {
                discard;
            }
            
            // Sclera (white of eye)
            vec3 color = vec3(0.95, 0.95, 0.98);
            float alpha = fadeAlpha * 0.15;
            
            // Calculate distortion based on horizontal position
            float h_stretch = 1.0;
            float v_stretch = 1.0 + (0.4 * abs(irisOffset.x));
            
            // Iris position
            vec2 irisCenter = irisOffset * vec2(1.2, 0.5);  // Scale movement
            vec2 toIris = uv - irisCenter;
            float irisDist = length(vec2(toIris.x / h_stretch, toIris.y / v_stretch));
            
            float irisRadius = 0.5;
            
            if (irisDist < irisRadius) {
                // Iris pattern
                float angle = atan(toIris.y / v_stretch, toIris.x / h_stretch);
                float distRatio = irisDist / irisRadius;
                float pattern = (sin(angle * 8.0) * 0.1) + (distRatio * 0.2);
                
                // Use HSV for iris color
                vec3 hsv = vec3(0.55 + pattern, 0.7, 0.5);
                color = hsv2rgb(hsv);
                alpha = fadeAlpha;
            }
            
            // Pupil with breathing and blink variation
            float breathingVar = sin(time * 1.5) * 0.1;
            float blinkPhase = mod(time, 7.0) / 7.0;
            float blinkVar = 0.0;
            if (blinkPhase < 0.1) {
                blinkVar = -0.3 * sin(blinkPhase * 31.416);  // ~10*pi
            }
            
            float currentPupilSize = clamp(pupilSize + breathingVar + blinkVar, 0.3, 1.0);
            float pupilRadius = 0.25 * currentPupilSize;
            
            vec2 toPupil = uv - irisCenter;
            float pupilDist = length(vec2(toPupil.x / h_stretch, toPupil.y / v_stretch));
            
            if (pupilDist < pupilRadius) {
                // Black pupil
                color = vec3(0.0, 0.0, 0.0);
                alpha = fadeAlpha * 0.5;
                
                // Highlight
                vec2 highlightPos = irisCenter + vec2(-pupilRadius * 0.5, -pupilRadius * 0.5);
                vec2 toHighlight = uv - highlightPos;
                float highlightDist = length(vec2(toHighlight.x / h_stretch, toHighlight.y / v_stretch));
                float highlightRadius = 0.08;
                
                if (highlightDist < highlightRadius) {
                    float intensity = 1.0 - (highlightDist / highlightRadius);
                    color = mix(color, vec3(1.0), intensity);
                    alpha = fadeAlpha * intensity;
                }
            }
            
            outColor = vec4(color, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link eye shaders"""
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
        """Update eye movement"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Check if it's time to get a new target position
        if current_time - self.last_movement_time > self.movement_interval:
            angle = np.random.random() * 2 * np.pi
            max_radius = 0.7
            r = np.random.random() * max_radius
            
            self.target_x = r * np.cos(angle) * 1.5
            self.target_y = r * np.sin(angle)
            
            self.last_movement_time = current_time
        
        # Smoothly move current position toward target
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0.001:
            move_amount = min(distance, self.movement_speed * dt)
            if distance > 0:
                self.current_x += (dx / distance) * move_amount
                self.current_y += (dy / distance) * move_amount
        
        self.last_update_time = current_time

    def render(self, state: Dict):
        """Render the eye using shader"""
        if not self.enabled or not self.shader:
            return
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(self.shader)
        
        # Set uniforms
        current_time = time.time() - self.start_time
        
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "eyePosition")
        if loc != -1:
            glUniform2f(loc, self.position[0], self.position[1])
        
        loc = glGetUniformLocation(self.shader, "eyeScale")
        if loc != -1:
            glUniform1f(loc, self.scale)
        
        loc = glGetUniformLocation(self.shader, "eyeDepth")
        if loc != -1:
            glUniform1f(loc, self.depth)
        
        loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        loc = glGetUniformLocation(self.shader, "pupilSize")
        if loc != -1:
            glUniform1f(loc, self.pupil_size)
        
        loc = glGetUniformLocation(self.shader, "irisOffset")
        if loc != -1:
            glUniform2f(loc, self.current_x, self.current_y)
        
        loc = glGetUniformLocation(self.shader, "time")
        if loc != -1:
            glUniform1f(loc, current_time)
        
        # Draw full-screen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        glDisable(GL_BLEND)
