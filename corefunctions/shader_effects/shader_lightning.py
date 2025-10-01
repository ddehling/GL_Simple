"""
Lightning Strike effect - shader-based implementation
GPU-rendered lightning with procedural branching using OpenGL shaders
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

def shader_lightning(state, outstate, side='bottom', intensity=1.0, color='random', position=None):
    """
    Shader-based lightning effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 2, shader_lightning, frame_id=0, side='top', intensity=1.5, color='purple', position=0.3)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id, side)
        outstate: Global state dict (from EventScheduler)
        side: Direction of lightning ('top', 'bottom', 'left', 'right')
        intensity: Brightness multiplier (default 1.0)
        color: Lightning color ('blue', 'purple', 'red', 'green', 'yellow', 'white', 'cyan', 'orange', 'random')
        position: Horizontal position 0-1 (None = random)
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
    
    # Initialize lightning effect on first call
    if state['count'] == 0:
        print(f"Initializing shader lightning effect for frame {frame_id}")
        
        try:
            lightning_effect = viewport.add_effect(LightningEffect, side=side)
            state['lightning_effect'] = lightning_effect
            state['start_time'] = time.time()
            
            # Determine position (random if not specified)
            if position is None:
                lightning_position = np.random.uniform(0.2, 0.8)
            else:
                lightning_position = np.clip(position, 0.0, 1.0)
            
            state['lightning_position'] = lightning_position
            
            # Generate random lightning path parameters
            # Main bolt path (random offsets from center position)
            num_segments = 12
            base_offsets = np.random.uniform(-0.15, 0.15, num_segments)
            base_offsets[0] = 0.0  # Start at position
            
            # Smooth the path slightly
            for i in range(2):
                base_offsets = np.convolve(
                    base_offsets, 
                    [0.25, 0.5, 0.25], 
                    mode='same'
                )
            
            state['bolt_offsets'] = base_offsets
            
            # Branch parameters
            num_branches = np.random.randint(3, 7)
            state['branch_positions'] = np.random.uniform(0.2, 0.8, num_branches)
            state['branch_angles'] = np.random.uniform(-0.8, 0.8, num_branches)
            state['branch_lengths'] = np.random.uniform(0.1, 0.3, num_branches)
            
            # Determine color
            color_presets = {
                'blue': (0.7, 0.85, 1.0),
                'purple': (0.8, 0.6, 1.0),
                'red': (1.0, 0.5, 0.5),
                'green': (0.5, 1.0, 0.6),
                'yellow': (1.0, 1.0, 0.5),
                'white': (1.0, 1.0, 1.0),
                'cyan': (0.5, 1.0, 1.0),
                'orange': (1.0, 0.7, 0.4),
                'pink': (1.0, 0.6, 0.8)
            }
            
            if color == 'random':
                chosen_color = list(color_presets.keys())[np.random.randint(0, len(color_presets))]
                state['lightning_color'] = color_presets[chosen_color]
            elif color in color_presets:
                state['lightning_color'] = color_presets[color]
            else:
                # Default to blue
                state['lightning_color'] = color_presets['blue']
            
            # Random seed for this strike
            state['random_seed'] = np.random.random() * 1000.0
            
            # Flash timing
            state['flash_intensity'] = intensity
            state['flicker_offset'] = np.random.random() * 10.0
            
            print(f"✓ Initialized shader lightning for frame {frame_id} (side: {side}, color: {color}, pos: {lightning_position:.2f})")
        except Exception as e:
            print(f"✗ Failed to initialize lightning: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update lightning parameters
    if 'lightning_effect' in state:
        current_time = time.time()
        elapsed_time = current_time - state['start_time']
        total_duration = state.get('duration', 2.0)
        
        # Lightning flash timing (quick flash, slow fade)
        if elapsed_time < 0.05:
            # Initial flash
            fade_factor = elapsed_time / 0.05
        elif elapsed_time < 0.15:
            # Peak brightness with flicker
            flicker = 0.8 + 0.2 * np.sin((elapsed_time - 0.05) * 50 + state['flicker_offset'])
            fade_factor = flicker
        else:
            # Fade out
            fade_factor = 1.0 - (elapsed_time - 0.15) / (total_duration - 0.15)
        
        fade_factor = np.clip(fade_factor, 0, 1)
        
        # Update effect parameters
        effect = state['lightning_effect']
        effect.fade_factor = fade_factor
        effect.intensity = state['flash_intensity']
        effect.lightning_position = state['lightning_position']
        effect.bolt_offsets = state['bolt_offsets']
        effect.branch_positions = state['branch_positions']
        effect.branch_angles = state['branch_angles']
        effect.branch_lengths = state['branch_lengths']
        effect.lightning_color = state['lightning_color']
        effect.random_seed = state['random_seed']
        effect.time_factor = elapsed_time
        
        # Audio reactive
        effect.whomp = outstate.get('whomp', 0.0)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'lightning_effect' in state:
            print(f"Cleaning up lightning effect for frame {frame_id}")
            viewport.effects.remove(state['lightning_effect'])
            state['lightning_effect'].cleanup()
            print(f"✓ Cleaned up shader lightning for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class LightningEffect(ShaderEffect):
    """GPU-based lightning strike effect with procedural branching"""
    
    def __init__(self, viewport, side='top', depth=45.0):
        super().__init__(viewport)
        
        # Lightning properties
        self.side = side  # 'top', 'bottom', 'left', 'right'
        self.depth = depth  # Z depth (0-100, lower = closer)
        self.fade_factor = 0.0
        self.intensity = 1.0
        
        # Lightning path parameters
        self.lightning_position = 0.5  # 0-1 position along perpendicular axis
        self.bolt_offsets = np.zeros(12)
        self.branch_positions = np.array([])
        self.branch_angles = np.array([])
        self.branch_lengths = np.array([])
        self.lightning_color = (0.7, 0.85, 1.0)  # Default blue
        self.random_seed = 0.0
        self.time_factor = 0.0
        self.whomp = 0.0
        
        # Start time
        self.start_time = time.time()
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Full-screen quad
        
        out vec2 fragCoord;  // Pass screen coordinate to fragment shader
        
        uniform float lightningDepth;  // Z depth
        
        void main() {
            // Normalize depth to 0-1 range (0 = near, 1 = far)
            float depth = lightningDepth / 100.0;
            
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
        uniform float lightningPosition;  // 0-1 position
        uniform float boltOffsets[12];  // Main bolt path offsets
        uniform float branchPositions[8];  // Branch start positions
        uniform float branchAngles[8];  // Branch angles
        uniform float branchLengths[8];  // Branch lengths
        uniform int numBranches;
        uniform vec3 lightningColor;  // RGB color
        uniform float randomSeed;
        uniform float timeFactor;
        uniform float whomp;
        
        // Hash function for pseudo-random values
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7)) + randomSeed) * 43758.5453123);
        }
        
        // Get bolt offset at a given position with interpolation
        float getBoltOffset(float t) {
            // t is 0-1 along the bolt
            float index = t * 11.0;  // 12 points (0-11)
            int i0 = int(floor(index));
            int i1 = min(i0 + 1, 11);
            float frac = fract(index);
            
            return mix(boltOffsets[i0], boltOffsets[i1], frac);
        }
        
        // Distance from point to line segment
        float distanceToSegment(vec2 p, vec2 a, vec2 b) {
            vec2 pa = p - a;
            vec2 ba = b - a;
            float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            return length(pa - ba * h);
        }
        
        // Generate jagged lightning path
        float lightningPath(vec2 uv, float verticalPos, float horizontalPos) {
            float minDist = 1000.0;
            
            // Main bolt - draw segments between control points
            int segments = 11;
            for (int i = 0; i < segments; i++) {
                float t0 = float(i) / float(segments);
                float t1 = float(i + 1) / float(segments);
                
                vec2 p0, p1;
                if (side == 0 || side == 1) {
                    // Vertical lightning (top/bottom)
                    p0 = vec2(lightningPosition + getBoltOffset(t0), t0);
                    p1 = vec2(lightningPosition + getBoltOffset(t1), t1);
                } else {
                    // Horizontal lightning (left/right)
                    p0 = vec2(t0, lightningPosition + getBoltOffset(t0));
                    p1 = vec2(t1, lightningPosition + getBoltOffset(t1));
                }
                
                float dist = distanceToSegment(uv, p0, p1);
                minDist = min(minDist, dist);
                
                // Add micro-jitters to main bolt
                float jitter = hash(vec2(float(i) * 0.1, randomSeed)) * 0.002;
                minDist -= jitter;
            }
            
            return minDist;
        }
        
        // Add branches to the lightning
        float addBranches(vec2 uv, float mainDist) {
            float minDist = mainDist;
            
            for (int i = 0; i < numBranches && i < 8; i++) {
                float branchStart = branchPositions[i];
                float angle = branchAngles[i];
                float length = branchLengths[i];
                
                // Get position on main bolt where branch starts
                vec2 startPos;
                if (side == 0 || side == 1) {
                    startPos = vec2(lightningPosition + getBoltOffset(branchStart), branchStart);
                } else {
                    startPos = vec2(branchStart, lightningPosition + getBoltOffset(branchStart));
                }
                
                // Calculate branch end position
                vec2 branchDir = vec2(cos(angle), sin(angle)) * length;
                vec2 endPos = startPos + branchDir;
                
                // Draw branch
                float dist = distanceToSegment(uv, startPos, endPos);
                minDist = min(minDist, dist);
            }
            
            return minDist;
        }
        
        void main() {
            vec2 uv = fragCoord;
            
            // Transform UV based on side
            if (side == 1) {
                // Bottom: flip vertically
                uv.y = 1.0 - uv.y;
            } else if (side == 2) {
                // Left: no transform needed for horizontal
            } else if (side == 3) {
                // Right: flip horizontally for horizontal
                uv.x = 1.0 - uv.x;
            }
            
            // Calculate distance to lightning path
            float dist = lightningPath(uv, uv.y, uv.x);
            
            // Add branches
            dist = addBranches(uv, dist);
            
            // Convert distance to intensity
            float boltThickness = 0.002 * (1.0 + whomp * 0.5);
            float glowThickness = 0.03 * (1.0 + whomp * 0.5);
            
            // Core bolt (bright white)
            float coreBolt = 1.0 - smoothstep(0.0, boltThickness, dist);
            
            // Inner glow (bright colored)
            float innerGlow = 1.0 - smoothstep(boltThickness, boltThickness * 3.0, dist);
            
            // Outer glow (colored fade)
            float outerGlow = 1.0 - smoothstep(boltThickness * 3.0, glowThickness, dist);
            
            // If no lightning visible, discard
            if (outerGlow < 0.01) {
                discard;
            }
            
            // Create color gradient (white core, colored glow)
            vec3 coreColor = vec3(1.0, 1.0, 1.0);
            vec3 glowColor = lightningColor;
            
            vec3 color = mix(glowColor, coreColor, coreBolt);
            color = mix(glowColor * 0.5, color, innerGlow);
            
            // Calculate alpha
            float alpha = max(coreBolt, max(innerGlow * 0.7, outerGlow * 0.3));
            alpha *= intensity * fadeAlpha;
            
            // Add flickering to the glow
            float flicker = 0.8 + 0.2 * sin(timeFactor * 30.0 + dist * 50.0);
            alpha *= flicker;
            
            outColor = vec4(color * alpha, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link lightning shaders"""
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
        """Update lightning animation (called every frame)"""
        if not self.enabled:
            return
        
        # Most updates are handled in the shader_lightning() wrapper function
        pass

    def render(self, state: Dict):
        """Render the lightning using shader"""
        if not self.enabled or not self.shader:
            return
        
        # Enable additive blending for bright lightning
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
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
        
        loc = glGetUniformLocation(self.shader, "lightningDepth")
        if loc != -1:
            glUniform1f(loc, self.depth)
        
        loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        loc = glGetUniformLocation(self.shader, "intensity")
        if loc != -1:
            glUniform1f(loc, self.intensity)
        
        loc = glGetUniformLocation(self.shader, "lightningPosition")
        if loc != -1:
            glUniform1f(loc, self.lightning_position)
        
        loc = glGetUniformLocation(self.shader, "lightningColor")
        if loc != -1:
            glUniform3f(loc, self.lightning_color[0], self.lightning_color[1], self.lightning_color[2])
        
        loc = glGetUniformLocation(self.shader, "randomSeed")
        if loc != -1:
            glUniform1f(loc, self.random_seed)
        
        loc = glGetUniformLocation(self.shader, "timeFactor")
        if loc != -1:
            glUniform1f(loc, self.time_factor)
        
        loc = glGetUniformLocation(self.shader, "whomp")
        if loc != -1:
            glUniform1f(loc, self.whomp)
        
        # Set bolt offsets array
        loc = glGetUniformLocation(self.shader, "boltOffsets")
        if loc != -1:
            glUniform1fv(loc, len(self.bolt_offsets), self.bolt_offsets.astype(np.float32))
        
        # Set branch parameters (pad to size 8)
        num_branches = len(self.branch_positions)
        loc = glGetUniformLocation(self.shader, "numBranches")
        if loc != -1:
            glUniform1i(loc, num_branches)
        
        # Pad arrays to size 8
        branch_pos_padded = np.zeros(8, dtype=np.float32)
        branch_angles_padded = np.zeros(8, dtype=np.float32)
        branch_lengths_padded = np.zeros(8, dtype=np.float32)
        
        if num_branches > 0:
            branch_pos_padded[:num_branches] = self.branch_positions[:min(num_branches, 8)]
            branch_angles_padded[:num_branches] = self.branch_angles[:min(num_branches, 8)]
            branch_lengths_padded[:num_branches] = self.branch_lengths[:min(num_branches, 8)]
        
        loc = glGetUniformLocation(self.shader, "branchPositions")
        if loc != -1:
            glUniform1fv(loc, 8, branch_pos_padded)
        
        loc = glGetUniformLocation(self.shader, "branchAngles")
        if loc != -1:
            glUniform1fv(loc, 8, branch_angles_padded)
        
        loc = glGetUniformLocation(self.shader, "branchLengths")
        if loc != -1:
            glUniform1fv(loc, 8, branch_lengths_padded)
        
        # Draw full-screen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Reset blend mode
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)