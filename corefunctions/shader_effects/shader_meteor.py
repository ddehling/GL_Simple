"""Meteor Shower effect - shader-based implementation
Converts the CPU-based meteor shower effect to GPU rendering using OpenGL shaders
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
import time
import random
import math
from pathlib import Path

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_meteor(state, outstate):
    """
    Shader-based meteor shower effect compatible with EventScheduler
    
    Meteors move diagonally from top-right to bottom-left.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_meteor, frame_id=0)
    
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
    
    # Initialize meteor effect on first call
    if state['count'] == 0:
        print(f"Initializing shader meteor effect for frame {frame_id}")
        
        try:
            meteor_effect = viewport.add_effect(MeteorEffect)
            state['meteor_effect'] = meteor_effect
            state['start_time'] = time.time()
            state['meteors'] = []
            
            print(f"✓ Initialized shader meteor for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize meteor: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update meteor parameters
    if 'meteor_effect' in state:
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
        
        # Generate new meteors based on meteor_rate
        meteor_rate = outstate.get('meteor_rate', 1) / 2
        if random.random() < meteor_rate:
            # Get viewport dimensions
            vp_width = viewport.width
            vp_height = viewport.height
            
            # Spawn from top-right, moving down-left diagonally
            meteor = {
                'x': random.uniform(vp_width * 0.67, vp_width * 1.17),  # 80-140 scaled to viewport
                'y': random.uniform(vp_height * -0.33, vp_height * 0.33),  # -20-20 scaled to viewport
                'angle': math.radians(random.uniform(200, 250)),  # Down-left diagonal
                'speed': random.uniform(4.0, 8.0),
                'size': random.uniform(0.6, 1.5),
                'trail_length': vp_width + random.random() * vp_height,  # Scale trail to viewport
                'life': 2,
                'hue': random.random()  # Random hue (0-1) for color variety
            }
            state['meteors'].append(meteor)
            
            # Add whoosh sound occasionally
            if random.random() < 0.1:
                sound_path = Path('sounds')
                whoosh_path = sound_path / 'Whoosh By 04.wav'
                if 'soundengine' in outstate and whoosh_path.exists():
                    outstate['soundengine'].schedule_event(
                        whoosh_path,
                        time.time(),
                        2
                    )
        
        # Update meteor positions
        new_meteors = []
        vp_width = viewport.width
        vp_height = viewport.height
        
        for meteor in state['meteors']:
            # Update position
            meteor['x'] += math.cos(meteor['angle']) * meteor['speed']
            meteor['y'] += math.sin(meteor['angle']) * meteor['speed']
            meteor['life'] -= 0.015  # Slower fade for longer trails
            
            # Keep meteor if still alive and on screen (expanded bounds for longer trails)
            if (meteor['life'] > 0 and 
                meteor['y'] > -vp_height * 1.33 and meteor['y'] < vp_height * 2.33 and 
                meteor['x'] > -vp_width * 0.67 and meteor['x'] < vp_width * 1.67):
                new_meteors.append(meteor)
        
        state['meteors'] = new_meteors
        
        # Update effect parameters
        effect = state['meteor_effect']
        effect.fade_factor = fade_factor
        effect.meteors = state['meteors']
    
    # On close event, clean up
    if state['count'] == -1:
        if 'meteor_effect' in state:
            print(f"Cleaning up meteor effect for frame {frame_id}")
            viewport.effects.remove(state['meteor_effect'])
            state['meteor_effect'].cleanup()
            print(f"✓ Cleaned up shader meteor for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class MeteorEffect(ShaderEffect):
    """GPU-based meteor shower effect using procedural shader rendering
    
    Renders meteors as a full-screen post-processing pass at z=75.9 (mid-depth).
    Meteors are drawn procedurally in the fragment shader, allowing unlimited
    meteor count while keeping the effect lightweight.
    
    Depth System:
        - z = 75.9 → depth = 0.759 → Mid-depth layer (between near objects and far background)
        - Participates in global depth testing and alpha blending
        - No state toggling per shader_info.txt requirements
    """
    
    MAX_METEORS = 16  # Maximum number of meteors the shader can handle
    
    def __init__(self, viewport, depth=75.9):
        super().__init__(viewport)
        
        # Meteor properties
        self.depth = depth  # Z depth: 0=near, 100=far (standard range per shader_info.txt)
        self.fade_factor = 0.0
        self.meteors = []
        
        # Start time
        self.start_time = time.time()
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Full-screen quad
        
        out vec2 fragCoord;  // Pass screen coordinate to fragment shader
        
        uniform float meteorDepth;  // Z depth
        
        void main() {
            // Normalize depth to 0-1 range (0 = near, 1 = far)
            // Standard mapping: z = 0-100 where 0 is near, 100 is far
            float depth = clamp(meteorDepth / 100.0, 0.0, 1.0);
            
            gl_Position = vec4(position, depth, 1.0);
            
            // Convert from clip space (-1,1) to screen space (0,1)
            fragCoord = (position + 1.0) * 0.5;
        }
        """
        
    def get_fragment_shader(self):
        return f"""
        #version 310 es
        precision highp float;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform vec2 resolution;
        uniform vec2 screenSize;  // Meteor screen dimensions (120x60)
        uniform float fadeAlpha;
        uniform int meteorCount;
        
        // Meteor data (positions in meteor screen space)
        uniform vec2 meteorPos[{self.MAX_METEORS}];
        uniform float meteorAngle[{self.MAX_METEORS}];
        uniform float meteorSpeed[{self.MAX_METEORS}];
        uniform float meteorSize[{self.MAX_METEORS}];
        uniform float meteorTrailLength[{self.MAX_METEORS}];
        uniform float meteorLife[{self.MAX_METEORS}];
        uniform float meteorHue[{self.MAX_METEORS}];  // Base hue for each meteor
        
        // Random function for noise
        float random(vec2 st) {{
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }}
        
        // Convert HSV to RGB
        vec3 hsv2rgb(vec3 c) {{
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }}
        
        void main() {{
            // Convert fragment coordinate to meteor screen space
            vec2 meteorScreenPos = fragCoord * screenSize;
            
            vec3 finalColor = vec3(0.0);
            float finalAlpha = 0.0;
            float maxBrightness = 0.0;
            
            // Check each meteor
            for (int m = 0; m < meteorCount && m < {self.MAX_METEORS}; m++) {{
                vec2 mPos = meteorPos[m];
                float angle = meteorAngle[m];
                float size = meteorSize[m];
                float trailLength = meteorTrailLength[m];
                float life = meteorLife[m];
                float baseHue = meteorHue[m];  // Get this meteor's base hue
                
                // Calculate trail direction
                vec2 trailDir = vec2(cos(angle), sin(angle));
                
                // Check multiple points along the trail (longer trails need more samples)
                int maxTrailPoints = int(trailLength);
                for (int i = 0; i < maxTrailPoints && i < 200; i++) {{
                    // Trail point position
                    vec2 trailPos = mPos - trailDir * float(i) * 0.4;  // Reduced spacing for smoother trails
                    
                    // Distance from current pixel to trail point
                    vec2 diff = meteorScreenPos - trailPos;
                    float dist = length(diff);
                    
                    // Calculate trail intensity falloff
                    float trailFactor = 1.0 - float(i) / trailLength;
                    float intensity = trailFactor * life;
                    
                    // Pixel size based on meteor size and trail position
                    // Make core bigger and trail thicker
                    float pixelSize = max(1.5, size * (0.5 + trailFactor * 0.8) * 2.5);
                    
                    // Is this pixel within the meteor/trail?
                    if (dist <= pixelSize) {{
                        float distFactor = 1.0 - (dist / pixelSize);
                        float pixelIntensity = intensity * distFactor;
                        
                        // Determine color based on position in trail
                        bool isCore = i < 3;
                        
                        if (isCore) {{
                            // Core: bright colored center with slight desaturation
                            vec3 coreColor = hsv2rgb(vec3(baseHue, 0.6, pixelIntensity * 1.2));
                            float coreBrightness = pixelIntensity * 1.2;
                            
                            if (coreBrightness > maxBrightness) {{
                                finalColor = coreColor;
                                finalAlpha = pixelIntensity;
                                maxBrightness = coreBrightness;
                            }}
                        }} else {{
                            // Trail: shift hue slightly based on trail position for variety
                            float hueShift = (1.0 - trailFactor) * 0.15;  // Shift up to 15% along spectrum
                            float trailHue = fract(baseHue + hueShift);  // Wrap around at 1.0
                            float trailSaturation = 0.85 + trailFactor * 0.15;  // More saturated at head
                            float trailValue = pixelIntensity * (0.7 + trailFactor * 0.3);
                            vec3 trailColor = hsv2rgb(vec3(trailHue, trailSaturation, trailValue));
                            float trailAlpha = pixelIntensity * 0.99;
                            float trailBrightness = trailValue;
                            
                            if (trailBrightness > maxBrightness) {{
                                finalColor = trailColor;
                                finalAlpha = trailAlpha;
                                maxBrightness = trailBrightness;
                            }}
                        }}
                    }}
                }}
            }}
            
            // Apply global fade
            finalAlpha *= fadeAlpha;
            
            // Discard if fully transparent
            if (finalAlpha < 0.01) {{
                discard;
            }}
            
            outColor = vec4(finalColor, finalAlpha);
        }}
        """

    def compile_shader(self):
        """Compile and link meteor shaders"""
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
        """Initialize OpenGL buffers for full-screen quad
        
        Note: This is a post-processing effect that renders a full-screen quad.
        Per shader_info.txt, depth testing is globally enabled and should NOT be toggled.
        """
        # Full-screen quad in clip space
        vertices = np.array([
            -1.0, -1.0,  # Bottom left
             1.0, -1.0,  # Bottom right
             1.0,  1.0,  # Top right
            -1.0,  1.0   # Top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # NO depth test toggling - global state is always active per shader_info.txt
        # Depth testing and alpha blending are set globally in create_window()
        
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
        """Update meteor animation (called every frame)"""
        if not self.enabled:
            return
        
        # Most updates are handled in the shader_meteor() wrapper function
        pass

    def render(self, state: Dict):
        """Render the meteors using shader
        
        Note: Per shader_info.txt, global OpenGL state (depth test + alpha blend) is always active.
        This effect participates in depth ordering at z=75.9 (mid-depth layer).
        NO depth test or blend state toggling is performed.
        """
        if not self.enabled or not self.shader:
            return
        
        # Skip rendering if no meteors
        if not self.meteors or len(self.meteors) == 0:
            return
        
        # NO blend state toggling - global state handles this per shader_info.txt
        # glEnable(GL_BLEND) and glBlendFunc are set globally in create_window()
        
        glUseProgram(self.shader)
        
        # Set basic uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "screenSize")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "meteorDepth")
        if loc != -1:
            glUniform1f(loc, self.depth)
        
        loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        # Set meteor count (limited to MAX_METEORS)
        meteor_count = min(len(self.meteors), self.MAX_METEORS)
        loc = glGetUniformLocation(self.shader, "meteorCount")
        if loc != -1:
            glUniform1i(loc, meteor_count)
        
        # Prepare meteor data arrays
        # Prepare meteor data arrays
        positions = np.zeros((self.MAX_METEORS, 2), dtype=np.float32)
        angles = np.zeros(self.MAX_METEORS, dtype=np.float32)
        speeds = np.zeros(self.MAX_METEORS, dtype=np.float32)
        sizes = np.zeros(self.MAX_METEORS, dtype=np.float32)
        trail_lengths = np.zeros(self.MAX_METEORS, dtype=np.float32)
        lives = np.zeros(self.MAX_METEORS, dtype=np.float32)
        hues = np.zeros(self.MAX_METEORS, dtype=np.float32)  # Add hue array
        
        # Fill arrays with meteor data
        for i in range(meteor_count):
            meteor = self.meteors[i]
            positions[i] = [meteor['x'], meteor['y']]
            angles[i] = meteor['angle']
            speeds[i] = meteor['speed']
            sizes[i] = meteor['size']
            trail_lengths[i] = meteor['trail_length']
            lives[i] = meteor['life']
            hues[i] = meteor['hue']  # Store hue value
        
        # Set meteor data uniforms
        loc = glGetUniformLocation(self.shader, "meteorPos")
        if loc != -1:
            glUniform2fv(loc, self.MAX_METEORS, positions.flatten())
        
        loc = glGetUniformLocation(self.shader, "meteorAngle")
        if loc != -1:
            glUniform1fv(loc, self.MAX_METEORS, angles)
        
        loc = glGetUniformLocation(self.shader, "meteorSpeed")
        if loc != -1:
            glUniform1fv(loc, self.MAX_METEORS, speeds)
        
        loc = glGetUniformLocation(self.shader, "meteorSize")
        if loc != -1:
            glUniform1fv(loc, self.MAX_METEORS, sizes)
        
        loc = glGetUniformLocation(self.shader, "meteorTrailLength")
        if loc != -1:
            glUniform1fv(loc, self.MAX_METEORS, trail_lengths)
        
        loc = glGetUniformLocation(self.shader, "meteorLife")
        if loc != -1:
            glUniform1fv(loc, self.MAX_METEORS, lives)
        
        loc = glGetUniformLocation(self.shader, "meteorHue")
        if loc != -1:
            glUniform1fv(loc, self.MAX_METEORS, hues)

        # Draw full-screen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        # NO state cleanup - global state is maintained per shader_info.txt