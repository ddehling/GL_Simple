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

def shader_firefly(state, outstate, density=1.0, audio_sensitivity=1.0):
    """
    Shader-based firefly effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_firefly, density=1.5, audio_sensitivity=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Firefly spawn rate multiplier
        audio_sensitivity: Audio reactivity multiplier (0.0 = no audio, 1.0 = normal, 2.0 = double)
    """
    # Get the viewport
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
    
    # Initialize firefly effect on first call
    if state['count'] == 0:
        print(f"Initializing firefly effect for frame {frame_id}")
        
        try:
            firefly_effect = viewport.add_effect(
                FireflyEffect,
                density=density,
                audio_sensitivity=audio_sensitivity
            )
            state['firefly_effect'] = firefly_effect
            print(f"✓ Initialized shader fireflies for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fireflies: {e}")
            import traceback
            traceback.print_exc()
            return
    
        # Update density and audio data every frame
    if 'firefly_effect' in state:
        state['firefly_effect'].density = outstate.get('firefly_density', density)
        state['firefly_effect'].audio_sensitivity = audio_sensitivity
        
        # Initialize audio smoothing buffers on first update
        if 'audio_history' not in state:
            state['audio_history'] = {
                'bass': [],
                'mid': [],
                'high': []
            }
            state['smoothing_frames'] = 2  # Average over last 15 frames (~0.375s at 40fps)
        
        # Pass audio data to effect for reactivity
        if audio_data is not None:
            # Extract norm_long_relu for above-average sound detection
            current_bands = audio_data['norm_long_relu'][0]  # Shape: (32,)
            
            # Split into frequency ranges
            bass_energy = np.mean(current_bands[0:8])      # Bass: 40-300 Hz
            mid_energy = np.mean(current_bands[8:20])      # Mids: 300-2000 Hz
            high_energy = np.mean(current_bands[20:32])    # Highs: 2000-16000 Hz
            
            # Add to history buffers
            state['audio_history']['bass'].append(bass_energy)
            state['audio_history']['mid'].append(mid_energy)
            state['audio_history']['high'].append(high_energy)
            
            # Keep only last N frames
            max_frames = state['smoothing_frames']
            if len(state['audio_history']['bass']) > max_frames:
                state['audio_history']['bass'] = state['audio_history']['bass'][-max_frames:]
                state['audio_history']['mid'] = state['audio_history']['mid'][-max_frames:]
                state['audio_history']['high'] = state['audio_history']['high'][-max_frames:]
            
            # Calculate smoothed averages
            smoothed_bass = np.mean(state['audio_history']['bass']) if state['audio_history']['bass'] else 0.0
            smoothed_mid = np.mean(state['audio_history']['mid']) if state['audio_history']['mid'] else 0.0
            smoothed_high = np.mean(state['audio_history']['high']) if state['audio_history']['high'] else 0.0
            
            # Pass smoothed values to effect
            state['firefly_effect'].audio_bass = smoothed_bass * audio_sensitivity
            state['firefly_effect'].audio_mid = smoothed_mid * audio_sensitivity
            state['firefly_effect'].audio_high = smoothed_high * audio_sensitivity
        else:
            # No audio data available
            state['firefly_effect'].audio_bass = 0.0
            state['firefly_effect'].audio_mid = 0.0
            state['firefly_effect'].audio_high = 0.0
    
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
    
    def __init__(self, viewport, density: float = 1.0, max_fireflies: int = 150, audio_sensitivity: float = 1.0):
        super().__init__(viewport)
        self.density = density
        self.max_fireflies = max_fireflies
        self.instance_VBO = None
        self.audio_sensitivity = audio_sensitivity
        
        # Audio reactivity values (updated from wrapper)
        self.audio_bass = 0.0
        self.audio_mid = 0.0
        self.audio_high = 0.0
        
        # Depth range
        self.min_depth = 10.0
        self.max_depth = 100.0
        
        # Horizontal wrapping support
        self.wrap_margin = 50.0  # Should be larger than largest firefly size
        
        # Vectorized firefly data (all stored as numpy arrays)
        self.positions = np.zeros((0, 3), dtype=np.float32)  # [x, y, z]
        self.phases = np.zeros(0, dtype=np.float32)  # Animation phase
        self.speeds = np.zeros(0, dtype=np.float32)  # XY movement speed
        self.z_speeds = np.zeros(0, dtype=np.float32)  # Z movement speed
        self.z_phases = np.zeros(0, dtype=np.float32)  # Separate phase for Z oscillation
        self.lifetimes = np.zeros(0, dtype=np.float32)  # Remaining lifetime (0-1)
        self.colors = np.zeros((0, 3), dtype=np.float32)  # [h, s, v] in HSV
        self.base_sizes = np.zeros(0, dtype=np.float32)  # Base size (before depth scaling)
        self.audio_bands = np.zeros(0, dtype=np.int32)  # Which frequency band each firefly responds to (0=bass, 1=mid, 2=high)
        
        # Spawn initial fireflies immediately
        initial_count = int(max_fireflies * 0.6)  # Start with 60% of max
        if initial_count > 0:
            self._spawn_fireflies(initial_count)
        
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
        new_speeds = np.random.uniform(0.025, 0.1, count)
        new_z_speeds = np.random.uniform(0.02, 0.08, count)  # Slower Z movement
        new_lifetimes = np.ones(count)
        
        # Yellow-green color range (H: 0.1-0.25 = yellow to green)
        new_colors = np.column_stack([
            np.random.uniform(0.1, 0.25, count),  # Hue: yellow-green
            np.random.uniform(0.8, 1.0, count),   # Saturation: vibrant
            np.ones(count)                         # Value: full brightness
        ])
        
        new_base_sizes = np.random.uniform(3.0, 6.0, count)  # Firefly size
        
        # Assign each firefly to a frequency band (0=bass, 1=mid, 2=high)
        new_audio_bands = np.random.randint(0, 3, count)
        
        # Concatenate with existing arrays
        self.positions = np.vstack([self.positions, new_positions]) if len(self.positions) > 0 else new_positions
        self.phases = np.concatenate([self.phases, new_phases]) if len(self.phases) > 0 else new_phases
        self.z_phases = np.concatenate([self.z_phases, new_z_phases]) if len(self.z_phases) > 0 else new_z_phases
        self.speeds = np.concatenate([self.speeds, new_speeds]) if len(self.speeds) > 0 else new_speeds
        self.z_speeds = np.concatenate([self.z_speeds, new_z_speeds]) if len(self.z_speeds) > 0 else new_z_speeds
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes]) if len(self.lifetimes) > 0 else new_lifetimes
        self.colors = np.vstack([self.colors, new_colors]) if len(self.colors) > 0 else new_colors
        self.base_sizes = np.concatenate([self.base_sizes, new_base_sizes]) if len(self.base_sizes) > 0 else new_base_sizes
        self.audio_bands = np.concatenate([self.audio_bands, new_audio_bands]) if len(self.audio_bands) > 0 else new_audio_bands
        
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
            // Create elongated firefly shape (slightly oval)
            vec2 stretched = fragPos * vec2(1.0, 0.8);  // Slightly compressed vertically
            float dist = length(stretched);
            
            // Discard fragments outside the shape
            if (dist > 1.0) {
                discard;
            }
            
            // Create very bright, tight core (firefly's bioluminescent spot)
            float core = 1.0 - smoothstep(0.0, 0.15, dist);  // Smaller, brighter core
            core = pow(core, 2.0);  // More concentrated brightness
            
            // Add subtle glow around the core
            float glow = 1.0 - smoothstep(0.15, 0.6, dist);
            glow = pow(glow, 1.5);  // Softer falloff
            
            // Combine with strong core emphasis
            float intensity = core * 2.0 + glow * 0.4;
            
                        // Apply brightness from vertex color (pulsing)
            intensity *= fragColor.a;
            
            // Boost color saturation in the core for more vibrant appearance
            vec3 finalColor = fragColor.rgb;
            float coreBoost = core * 0.3;
            finalColor = mix(finalColor, finalColor * 1.5, coreBoost);
            
            // Output with alpha blending - let global blend state handle transparency
            // The alpha value controls blending, no need to discard
            outColor = vec4(finalColor, intensity);
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
        
                # Spawn new fireflies based on density (much higher rate)
        if len(self.positions) < self.max_fireflies:
            spawn_probability = self.density * 0.05  # 5% chance per frame instead of 0.5%
            if np.random.random() < spawn_probability:
                spawn_count = min(3, self.max_fireflies - len(self.positions))
                self._spawn_fireflies(spawn_count)
        
        if len(self.positions) == 0:
            return
        
        # Get whomp factor for dramatic movement
        whomp = state.get('whomp', 0.0)
        movement_multiplier = 1.0 + whomp * 4.0
        
                # Update phases for all fireflies (slower pulsing)
        self.phases += 0.03 * dt * 60  # Slower pulse rate
        self.z_phases += 0.05 * dt * 60  # Slower Z oscillation
        
        # Calculate movement angles from phases
        angles = self.phases * 0.1
        
                # Get audio energy for speed modulation
        audio_energies = np.zeros(len(self.positions))
        audio_energies[self.audio_bands == 0] = self.audio_bass
        audio_energies[self.audio_bands == 1] = self.audio_mid
        audio_energies[self.audio_bands == 2] = self.audio_high
        audio_energies = np.clip(audio_energies, 0, 2)
        
        # SPEED MODULATION: Increase speed with audio energy
        audio_speed_multiplier = 1.0 + audio_energies * 0.5  # Up to 1.5x speed boost
        
        # Update XY positions (smooth wandering motion with audio reactivity)
        self.positions[:, 0] += np.cos(angles) * self.speeds * movement_multiplier * audio_speed_multiplier
        self.positions[:, 1] += np.sin(angles) * self.speeds * movement_multiplier * audio_speed_multiplier
        
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
            self.audio_bands = self.audio_bands[alive_mask]

    def render(self, state: Dict):
        """Render all fireflies using instancing with horizontal wrapping"""
        if not self.enabled or not self.shader:
            return
        
        if len(self.positions) == 0:
            return
        
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
                # Get audio energy for each firefly based on their assigned frequency band
        audio_energies = np.zeros(len(self.positions))
        audio_energies[self.audio_bands == 0] = self.audio_bass   # Bass fireflies
        audio_energies[self.audio_bands == 1] = self.audio_mid    # Mid fireflies
        audio_energies[self.audio_bands == 2] = self.audio_high   # High fireflies
        
        # Clamp audio energies to reasonable range (0-2)
        audio_energies = np.clip(audio_energies, 0, 2)
        
        # Calculate depth-based size scaling (closer = bigger)
        depth_range = self.max_depth - self.min_depth
        depth_factors = 2.0 - 1.5 * (self.positions[:, 2] - self.min_depth) / depth_range
        
        # SIZE MODULATION: Scale size based on audio energy
        audio_size_multiplier = 1.0 + audio_energies * 0.8  # Up to 1.8x size boost
        scaled_sizes = self.base_sizes * depth_factors * audio_size_multiplier
        
        # Calculate brightness based on phase (smooth bright to dim)
        # Use 0.5 offset to keep fireflies always somewhat visible
        pulse = np.sin(self.phases)
        brightness = 0.4 + 0.6 * (pulse * 0.5 + 0.5)  # Range: 0.4 (dim) to 1.0 (bright)
        
        # BRIGHTNESS MODULATION: Boost brightness with audio energy
        audio_brightness_boost = audio_energies * 0.3  # Up to 30% brighter
        brightness = np.clip(brightness + audio_brightness_boost, 0.4, 1.0)
        
        brightness *= self.lifetimes  # Fade out as lifetime decreases
        brightness *= depth_factors * 0.4 + 0.6  # Scale brightness with depth
        
                # Convert HSV colors to RGB (vectorized) with audio modulation
        from skimage import color as skcolor
        
        hsv_colors = self.colors.copy()
        hsv_colors[:, 2] = brightness
        
        # COLOR MODULATION: Shift hue slightly based on audio energy
        # Bass: shift toward red/orange, Mid: neutral, High: shift toward green/blue
        hue_shift = np.zeros(len(self.positions))
        hue_shift[self.audio_bands == 0] = -audio_energies[self.audio_bands == 0] * 0.03  # Bass: redder
        hue_shift[self.audio_bands == 1] = 0.0  # Mid: no shift
        hue_shift[self.audio_bands == 2] = audio_energies[self.audio_bands == 2] * 0.03   # High: greener
        hsv_colors[:, 0] = (hsv_colors[:, 0] + hue_shift) % 1.0  # Wrap hue to 0-1
        
        # Boost saturation with audio energy for more vibrant colors
        audio_saturation_boost = audio_energies * 0.1  # Up to 10% more saturated
        hsv_colors[:, 1] = np.clip(hsv_colors[:, 1] + audio_saturation_boost, 0.0, 1.0)
        
        rgb_colors = np.zeros_like(hsv_colors)
        for i in range(len(hsv_colors)):
            rgb = skcolor.hsv2rgb(hsv_colors[i:i+1].reshape(1, 1, 3))
            rgb_colors[i] = rgb.flatten()
        
        # ============================================================
        # Horizontal Wrapping: Duplicate fireflies at screen edges
        # ============================================================
        
        # Identify fireflies near left and right edges
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Start with original fireflies
        all_positions = [self.positions]
        all_sizes = [scaled_sizes]
        all_colors = [rgb_colors]
        all_brightness = [brightness]
        
        # Duplicate fireflies near left edge to right side
        if np.any(left_edge_mask):
            left_indices = np.where(left_edge_mask)[0]
            duplicate_pos = self.positions[left_indices].copy()
            duplicate_pos[:, 0] += self.viewport.width  # Shift to right
            
            all_positions.append(duplicate_pos)
            all_sizes.append(scaled_sizes[left_indices])
            all_colors.append(rgb_colors[left_indices])
            all_brightness.append(brightness[left_indices])
        
        # Duplicate fireflies near right edge to left side
        if np.any(right_edge_mask):
            right_indices = np.where(right_edge_mask)[0]
            duplicate_pos = self.positions[right_indices].copy()
            duplicate_pos[:, 0] -= self.viewport.width  # Shift to left
            
            all_positions.append(duplicate_pos)
            all_sizes.append(scaled_sizes[right_indices])
            all_colors.append(rgb_colors[right_indices])
            all_brightness.append(brightness[right_indices])
        
        # Combine all fireflies (originals + duplicates)
        combined_positions = np.vstack(all_positions)
        combined_sizes = np.concatenate(all_sizes)
        combined_colors = np.vstack(all_colors)
        combined_brightness = np.concatenate(all_brightness)
        
        # Sort by depth (far to near) for proper depth ordering
        depth_order = np.argsort(combined_positions[:, 2])[::-1]
        
        # Build instance data with wrapping duplicates
        instance_data = np.hstack([
            combined_positions[depth_order],
            combined_sizes[depth_order, np.newaxis],
            combined_colors[depth_order],
            combined_brightness[depth_order, np.newaxis]
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
        
                # ============================================================
        # Single-pass render with global blend state
        # Disable depth writes for proper transparency blending
        # ============================================================
        
        # Disable depth writes so fireflies blend properly with objects behind them
        # They still READ from depth buffer (for occlusion by objects in front)
        glDepthMask(GL_FALSE)
        
        # Draw all fireflies (originals + wrapped duplicates) in one call
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(combined_positions))
        
        # Restore depth writes for next effect
        glDepthMask(GL_TRUE)
        
        glBindVertexArray(0)
        glUseProgram(0)