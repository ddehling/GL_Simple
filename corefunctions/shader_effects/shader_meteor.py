"""
Meteor Shower effect - shader-based implementation
Converts the CPU-based meteor shower effect to GPU rendering using OpenGL shaders
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import time
import random
import math
from pathlib import Path
from corefunctions.shader_effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_meteor(state, outstate, direction='top-right'):
    """
    Shader-based meteor shower effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_meteor, frame_id=0, direction='top-right')
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        direction: Direction meteors come from:
            'top-right' (default) - diagonal from top-right
            'top-left' - diagonal from top-left  
            'bottom-right' - diagonal from bottom-right
            'bottom-left' - diagonal from bottom-left
            'top' - straight down
            'bottom' - straight up
            'left' - straight right
            'right' - straight left
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
        print(f"Initializing shader meteor effect for frame {frame_id} (direction: {direction})")
        
        try:
            meteor_effect = viewport.add_effect(MeteorEffect, direction=direction)
            state['meteor_effect'] = meteor_effect
            state['start_time'] = time.time()
            state['meteors'] = []
            state['direction'] = direction
            
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
            # Get spawn parameters based on direction
            spawn_params = _get_spawn_params(state['direction'])
            
            # Create new meteor with direction-based parameters
            meteor = {
                'x': spawn_params['x'],
                'y': spawn_params['y'],
                'angle': spawn_params['angle'],
                'speed': random.uniform(2.0, 4.0),  # Increased speed for longer trails
                'size': random.uniform(0.6, 1.5),  # Larger meteors
                'trail_length': 120 + random.random() * 60,  # Much longer trails (80-140)
                'life': 1.2
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
        for meteor in state['meteors']:
            # Update position
            meteor['x'] += math.cos(meteor['angle']) * meteor['speed']
            meteor['y'] += math.sin(meteor['angle']) * meteor['speed']
            meteor['life'] -= 0.015  # Slower fade for longer trails
            
            # Keep meteor if still alive and on screen (expanded bounds for longer trails)
            if meteor['life'] > 0 and meteor['y'] > -80 and meteor['y'] < 140 and meteor['x'] > -80 and meteor['x'] < 200:
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


def _get_spawn_params(direction):
    """
    Get spawn position and angle based on direction
    Returns dict with 'x', 'y', 'angle' keys
    """
    params = {}
    
    if direction == 'top-right':
        # Spawn from top-right, moving down-left
        params['x'] = random.uniform(80, 140)
        params['y'] = random.uniform(-20, 20)
        params['angle'] = math.radians(random.uniform(200, 250))  # Down-left diagonal
        
    elif direction == 'top-left':
        # Spawn from top-left, moving down-right
        params['x'] = random.uniform(-20, 40)
        params['y'] = random.uniform(-20, 20)
        params['angle'] = math.radians(random.uniform(290, 340))  # Down-right diagonal
        
    elif direction == 'bottom-right':
        # Spawn from bottom-right, moving up-left
        params['x'] = random.uniform(80, 140)
        params['y'] = random.uniform(40, 80)
        params['angle'] = math.radians(random.uniform(110, 160))  # Up-left diagonal
        
    elif direction == 'bottom-left':
        # Spawn from bottom-left, moving up-right
        params['x'] = random.uniform(-20, 40)
        params['y'] = random.uniform(40, 80)
        params['angle'] = math.radians(random.uniform(20, 70))  # Up-right diagonal
        
    elif direction == 'top':
        # Spawn from top, moving straight down
        params['x'] = random.uniform(0, 120)
        params['y'] = random.uniform(-20, 0)
        params['angle'] = math.radians(random.uniform(85, 95))  # Straight down (90°)
        
    elif direction == 'bottom':
        # Spawn from bottom, moving straight up
        params['x'] = random.uniform(0, 120)
        params['y'] = random.uniform(60, 80)
        params['angle'] = math.radians(random.uniform(265, 275))  # Straight up (270°)
        
    elif direction == 'left':
        # Spawn from left, moving straight right
        params['x'] = random.uniform(-20, 0)
        params['y'] = random.uniform(0, 60)
        params['angle'] = math.radians(random.uniform(-5, 5))  # Straight right (0°)
        
    elif direction == 'right':
        # Spawn from right, moving straight left
        params['x'] = random.uniform(120, 140)
        params['y'] = random.uniform(0, 60)
        params['angle'] = math.radians(random.uniform(175, 185))  # Straight left (180°)
        
    else:
        # Default to top-right
        params['x'] = random.uniform(80, 140)
        params['y'] = random.uniform(-20, 20)
        params['angle'] = math.radians(random.uniform(200, 250))
    
    return params


# ============================================================================
# Rendering Class
# ============================================================================

class MeteorEffect(ShaderEffect):
    """GPU-based meteor shower effect using procedural shader rendering"""
    
    MAX_METEORS = 32  # Maximum number of meteors the shader can handle
    
    def __init__(self, viewport, direction='top-right', depth=49.9):
        super().__init__(viewport)
        
        # Meteor properties
        self.direction = direction
        self.depth = depth  # Z depth (same as aurora)
        self.fade_factor = 0.0
        self.meteors = []
        
        # Screen dimensions (matching meteor_window size)
        self.screen_width = 120.0
        self.screen_height = 60.0
        
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
            float depth = meteorDepth / 100.0;
            
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
        uniform vec2 meteorPos[{self.MAX_METEORS}];  // x, y position
        uniform float meteorAngle[{self.MAX_METEORS}];
        uniform float meteorSpeed[{self.MAX_METEORS}];
        uniform float meteorSize[{self.MAX_METEORS}];
        uniform float meteorTrailLength[{self.MAX_METEORS}];
        uniform float meteorLife[{self.MAX_METEORS}];
        
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
                        bool isCore = i < 3;  // Larger core
                        
                        if (isCore) {{
                            // Core: bright white/yellow
                            vec3 coreColor = hsv2rgb(vec3(0.15, 0.3, pixelIntensity * 1.2));
                            float coreBrightness = pixelIntensity * 1.2;
                            
                            // Blend with existing color (brightest wins)
                            if (coreBrightness > maxBrightness) {{
                                finalColor = coreColor;
                                finalAlpha = pixelIntensity;
                                maxBrightness = coreBrightness;
                            }}
                        }} else {{
                            // Trail: orange/red gradient with more variation
                            float hueVariation = 0.05 + (1.0 - trailFactor) * 0.1;  // Orange to red
                            float trailValue =  pixelIntensity * (0.7 + trailFactor * 0.3);
                            vec3 trailColor = hsv2rgb(vec3(hueVariation, 0.9, trailValue));
                            float trailAlpha = pixelIntensity * 0.99;
                            float trailBrightness = trailValue;
                            
                            // Blend with existing color (brightest wins)
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
        """Update meteor animation (called every frame)"""
        if not self.enabled:
            return
        
        # Most updates are handled in the shader_meteor() wrapper function
        pass

    def render(self, state: Dict):
        """Render the meteors using shader"""
        if not self.enabled or not self.shader:
            return
        
        # Skip rendering if no meteors
        if not self.meteors or len(self.meteors) == 0:
            return
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(self.shader)
        
        # Set basic uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "screenSize")
        if loc != -1:
            glUniform2f(loc, self.screen_width, self.screen_height)
        
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
        positions = np.zeros((self.MAX_METEORS, 2), dtype=np.float32)
        angles = np.zeros(self.MAX_METEORS, dtype=np.float32)
        speeds = np.zeros(self.MAX_METEORS, dtype=np.float32)
        sizes = np.zeros(self.MAX_METEORS, dtype=np.float32)
        trail_lengths = np.zeros(self.MAX_METEORS, dtype=np.float32)
        lives = np.zeros(self.MAX_METEORS, dtype=np.float32)
        
        # Fill arrays with meteor data
        for i in range(meteor_count):
            meteor = self.meteors[i]
            positions[i] = [meteor['x'], meteor['y']]
            angles[i] = meteor['angle']
            speeds[i] = meteor['speed']
            sizes[i] = meteor['size']
            trail_lengths[i] = meteor['trail_length']
            lives[i] = meteor['life']
        
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
        
        # Draw full-screen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        glDisable(GL_BLEND)