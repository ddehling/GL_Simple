"""
Complete firefly effect - rendering + event integration
GPU-accelerated firefly system with glow effects and depth
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_firefly(state, outstate, density=1.0):
    """
    Shader-based firefly effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_firefly, density=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Firefly spawn rate multiplier
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
    
    # Initialize firefly effect on first call
    if state['count'] == 0:
        print(f"Initializing firefly effect for frame {frame_id}")
        
        try:
            firefly_effect = viewport.add_effect(
                FireflyEffect,
                density=density
            )
            state['firefly_effect'] = firefly_effect
            print(f"✓ Initialized shader fireflies for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fireflies: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update density if it changes in global state
    if 'firefly_effect' in state:
        state['firefly_effect'].density = outstate.get('firefly_density', density)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'firefly_effect' in state:
            print(f"Cleaning up firefly effect for frame {frame_id}")
            viewport.effects.remove(state['firefly_effect'])
            state['firefly_effect'].cleanup()
            print(f"✓ Cleaned up shader fireflies for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class FireflyEffect(ShaderEffect):
    """GPU-based firefly effect using instanced rendering with vectorized updates"""
    
    def __init__(self, viewport, density: float = 1.0, max_fireflies: int = 200):
        super().__init__(viewport)
        self.density = density
        self.max_fireflies = max_fireflies
        self.instance_VBO = None
        
        # Depth range
        self.min_depth = 10.0
        self.max_depth = 100.0
        
        # Vectorized firefly data (all stored as numpy arrays)
        self.positions = np.zeros((0, 3), dtype=np.float32)  # [x, y, z]
        self.phases = np.zeros(0, dtype=np.float32)  # Animation phase
        self.speeds = np.zeros(0, dtype=np.float32)  # XY movement speed
        self.z_speeds = np.zeros(0, dtype=np.float32)  # Z movement speed
        self.z_phases = np.zeros(0, dtype=np.float32)  # Separate phase for Z oscillation
        self.lifetimes = np.zeros(0, dtype=np.float32)  # Remaining lifetime (0-1)
        self.colors = np.zeros((0, 3), dtype=np.float32)  # [h, s, v] in HSV
        self.base_sizes = np.zeros(0, dtype=np.float32)  # Base size (before depth scaling)
        
    def _spawn_fireflies(self, count: int):
        """Spawn new fireflies at random positions"""
        if count <= 0:
            return
            
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, count),
            np.random.uniform(0, self.viewport.height, count),
            np.random.uniform(self.min_depth, self.max_depth, count)
        ])
        
        new_phases = np.random.uniform(0, 2 * np.pi, count)
        new_z_phases = np.random.uniform(0, 2 * np.pi, count)
        new_speeds = np.random.uniform(0.1, 0.4, count)
        new_z_speeds = np.random.uniform(0.02, 0.08, count)  # Slower Z movement
        new_lifetimes = np.ones(count)
        
        # Yellow-green color range (H: 0.1-0.25 = yellow to green)
        new_colors = np.column_stack([
            np.random.uniform(0.1, 0.25, count),  # Hue: yellow-green
            np.random.uniform(0.8, 1.0, count),   # Saturation: vibrant
            np.ones(count)                         # Value: full brightness
        ])
        
        new_base_sizes = np.random.uniform(1.0, 3.0, count)
        
        # Concatenate with existing arrays
        self.positions = np.vstack([self.positions, new_positions]) if len(self.positions) > 0 else new_positions
        self.phases = np.concatenate([self.phases, new_phases]) if len(self.phases) > 0 else new_phases
        self.z_phases = np.concatenate([self.z_phases, new_z_phases]) if len(self.z_phases) > 0 else new_z_phases
        self.speeds = np.concatenate([self.speeds, new_speeds]) if len(self.speeds) > 0 else new_speeds
        self.z_speeds = np.concatenate([self.z_speeds, new_z_speeds]) if len(self.z_speeds) > 0 else new_z_speeds
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes]) if len(self.lifetimes) > 0 else new_lifetimes
        self.colors = np.vstack([self.colors, new_colors]) if len(self.colors) > 0 else new_colors
        self.base_sizes = np.concatenate([self.base_sizes, new_base_sizes]) if len(self.base_sizes) > 0 else new_base_sizes
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec3 offset;    // Firefly position (x, y, z)
        layout(location = 2) in float size;     // Firefly size (scaled by depth)
        layout(location = 3) in vec4 color;     // Color (r, g, b, brightness)
        
        out vec4 fragColor;
        out vec2 fragPos;  // Position within quad (-1 to 1)
        uniform vec2 resolution;
        
        void main() {
            // Pass through for glow calculation
            fragPos = position;
            
            // Scale the quad by firefly size
            vec2 scaled = position * size;
            
            // Translate to firefly XY position
            vec2 pos = scaled + offset.xy;
            
            // Convert screen coordinates to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use Z for depth buffer (normalize to 0-1 range)
            // Map z from [10, 100] to depth [0.1, 1.0] (far to near)
            float depth = offset.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
            fragColor = color;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;  // Position within quad (-1 to 1)
        out vec4 outColor;
        
        void main() {
            // Calculate distance from center of quad
            float dist = length(fragPos);
            
            // Discard fragments outside the circle for cleaner edges
            if (dist > 1.0) {
                discard;
            }
            
            // Create bright core with soft glow falloff
            float core = 1.0 - smoothstep(0.0, 0.3, dist);
            float glow = 1.0 - smoothstep(0.3, 1.0, dist);
            
            // Combine core and glow
            float intensity = core + glow * 0.6;
            
            // Apply brightness from vertex color
            intensity *= fragColor.a;
            
            // Output with smooth alpha falloff
            outColor = vec4(fragColor.rgb, intensity);
        }
        """
    
    def compile_shader(self):
        """Compile and link firefly shaders"""
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
        # Quad vertices - square from -1 to 1 for glow effect
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
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
        """Update firefly positions and properties (vectorized)"""
        if not self.enabled:
            return
        
        # Spawn new fireflies based on density
        if len(self.positions) < self.max_fireflies:
            spawn_probability = self.density * 0.025
            if np.random.random() < spawn_probability:
                spawn_count = min(5, self.max_fireflies - len(self.positions))
                self._spawn_fireflies(spawn_count)
        
        if len(self.positions) == 0:
            return
        
        # Get whomp factor for dramatic movement
        whomp = state.get('whomp', 0.0)
        movement_multiplier = 1.0 + whomp * 12.0
        
        # Update phases for all fireflies
        self.phases += 0.1 * dt * 60  # Normalize for frame rate
        self.z_phases += 0.05 * dt * 60  # Slower Z oscillation
        
        # Calculate movement angles from phases
        angles = self.phases * 0.1
        
        # Update XY positions (smooth wandering motion)
        self.positions[:, 0] += np.cos(angles) * self.speeds * movement_multiplier
        self.positions[:, 1] += np.sin(angles) * self.speeds * movement_multiplier
        
        # Update Z positions (oscillating wandering between min and max depth)
        # Use sine wave with phase offset for smooth back-and-forth motion
        z_center = (self.min_depth + self.max_depth) / 2
        z_range = (self.max_depth - self.min_depth) / 2
        self.positions[:, 2] = z_center + np.sin(self.z_phases) * z_range * 0.8
        
        # Add some Z drift for more organic movement
        self.positions[:, 2] += np.cos(self.z_phases * 0.7) * self.z_speeds * 5 * movement_multiplier
        
        # Clamp Z to valid range
        self.positions[:, 2] = np.clip(self.positions[:, 2], self.min_depth, self.max_depth)
        
        # Wrap around screen edges for XY
        self.positions[:, 0] %= self.viewport.width
        self.positions[:, 1] %= self.viewport.height
        
        # Decrease lifetimes
        self.lifetimes -= 0.001 * dt * 60
        
        # Remove dead fireflies
        alive_mask = self.lifetimes > 0
        if not np.all(alive_mask):
            self.positions = self.positions[alive_mask]
            self.phases = self.phases[alive_mask]
            self.z_phases = self.z_phases[alive_mask]
            self.speeds = self.speeds[alive_mask]
            self.z_speeds = self.z_speeds[alive_mask]
            self.lifetimes = self.lifetimes[alive_mask]
            self.colors = self.colors[alive_mask]
            self.base_sizes = self.base_sizes[alive_mask]

    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;  // Position within quad (-1 to 1)
        out vec4 outColor;
        
        void main() {
            // Calculate distance from center of quad
            float dist = length(fragPos);
            
            // Discard fragments outside the circle for cleaner edges
            if (dist > 1.0) {
                discard;
            }
            
            // Create bright core with soft glow falloff
            float core = 1.0 - smoothstep(0.0, 0.3, dist);
            float glow = 1.0 - smoothstep(0.3, 1.0, dist);
            
            // Combine core and glow
            float intensity = core + glow * 0.6;
            
            // Apply brightness from vertex color
            intensity *= fragColor.a;
            
            // Use alpha threshold to control depth writes
            // Only write depth for bright areas (core), not faint glow
            // This is done by discarding very faint fragments before depth write
            if (intensity < 0.1) {
                discard;
            }
            
            // Output with smooth alpha falloff
            outColor = vec4(fragColor.rgb, intensity);
        }
        """

    def render(self, state: Dict):
        """Render all fireflies using instancing"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        # Render in two passes for proper depth + blending
        # Pass 1: Render solid cores with depth writes (for occlusion)
        # Pass 2: Render full glow without depth writes (for blending)
        
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        # Calculate depth-based size scaling (closer = bigger)
        depth_range = self.max_depth - self.min_depth
        depth_factors = 2.0 - 1.5 * (self.positions[:, 2] - self.min_depth) / depth_range
        scaled_sizes = self.base_sizes * depth_factors
        
        # Calculate brightness based on phase (pulsing effect)
        brightness = 0.8 + 0.2 * np.sin(self.phases)
        brightness *= self.lifetimes  # Fade out as lifetime decreases
        brightness *= depth_factors * 0.5 + 0.5  # Scale brightness with depth
        
        # Convert HSV colors to RGB (vectorized)
        from skimage import color as skcolor
        
        hsv_colors = self.colors.copy()
        hsv_colors[:, 2] = brightness
        
        rgb_colors = np.zeros_like(hsv_colors)
        for i in range(len(hsv_colors)):
            rgb = skcolor.hsv2rgb(hsv_colors[i:i+1].reshape(1, 1, 3))
            rgb_colors[i] = rgb.flatten()
        
        # Sort by depth (far to near)
        depth_order = np.argsort(self.positions[:, 2])[::-1]
        
        # Build instance data
        instance_data = np.hstack([
            self.positions[depth_order],
            scaled_sizes[depth_order, np.newaxis],
            rgb_colors[depth_order],
            brightness[depth_order, np.newaxis]
        ]).astype(np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 8 * 4
        
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # === PASS 1: Solid core with depth writes (no blending) ===
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)  # Write depth for cores
        
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(self.positions))
        
        # === PASS 2: Full glow with blending (no depth writes) ===
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending
        glDepthFunc(GL_LEQUAL)  # Use LEQUAL to allow same-depth blending
        glDepthMask(GL_FALSE)  # Don't write depth for glow
        
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(self.positions))
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore state
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)