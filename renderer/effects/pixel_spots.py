"""
Pixel spots shader effect - small spots that appear and fade based on audio
Spots spawn at random positions/depths with colors tied to audio frequency bands
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import colorsys
import traceback
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_pixel_spots(state, outstate, intensity=1.0, audio_sensitivity=1.5,
                       audio_reactive=True):
    """
    Shader-based pixel spots effect compatible with EventScheduler

    Usage:
        scheduler.schedule_event(0, 60, shader_pixel_spots, intensity=1.0, frame_id=0)

    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        intensity: Overall intensity multiplier (affects spawn rate)
        audio_sensitivity: Multiplier for audio reactivity (default 1.5)
        audio_reactive: True (default) = spawn rate follows bass energy —
            near-silent baseline with bursts on pulses (WoL leaves canvas).
            False = steady twinkle at a constant rate, no audio coupling
            (cyberpunk: with the AGC'd bass signal hovering near 0, the
            reactive mode spent most of its life invisible and then
            burst, which read as broken rather than musical).

    Global Parameters (from outstate):
        spot_rate: Rate multiplier for spot appearance (default 1.0)
        spot_decay: Speed multiplier for spot decay (default 1.0)
        spot_saturation: HSV saturation for spots (0.0-1.0, default 0.8)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing pixel spots effect for frame {frame_id}")
        
        try:
            spots_effect = viewport.add_effect(
                PixelSpotsEffect,
                intensity=intensity,
                audio_sensitivity=audio_sensitivity,
                audio_reactive=audio_reactive
            )
            state['effect'] = spots_effect
            print(f"✓ Initialized shader pixel spots for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize pixel spots: {e}")
            traceback.print_exc()
            return
    
    # Update effect with global parameters and audio data every frame
    if 'effect' in state:
        effect = state['effect']
        
        # Get global parameters from outstate (with defaults)
        effect.spot_rate = outstate.get('spot_rate', 2.0)
        effect.spot_decay = outstate.get('spot_decay', 1)
        effect.spot_saturation = outstate.get('spot_saturation', 1.0)
        
        # Pass current time for rolling hue
        effect.current_time = state.get('elapsed_time', 0.0)
        
        # Pass audio data to effect (only bass; skipped in steady mode)
        audio_data = outstate.get('sound')
        if audio_reactive and audio_data is not None:
            # Get norm_long_relu for current frame [0]
            audio_bands = audio_data['norm_long_relu'][0]
            
            # Only use bass: bands 0-10 (40-300 Hz)
            bass_energy = np.mean(audio_bands[0:10])
            
            effect.update_audio(bass_energy)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up pixel spots effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader pixel spots for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class PixelSpotsEffect(ShaderEffect):
    """GPU-based pixel spots effect using instanced rendering"""
    
    def __init__(self, viewport, intensity: float = 1.0, audio_sensitivity: float = 1.5,
                 audio_reactive: bool = True):
        super().__init__(viewport)
        self.intensity = intensity
        self.audio_sensitivity = audio_sensitivity
        self.audio_reactive = bool(audio_reactive)

        # Must render AFTER cyber_city_skyline (priority 6.0) so the
        # building silhouette's depth values (written by skyline as
        # gl_FragDepth=0 for buildings, 0.95 for sky) are present in
        # the depth buffer when we depth-test against them — that's
        # what makes spots visible in the sky and occluded by buildings.
        # Still below city-particle foreground layers (signs 7.0,
        # holograms 7.5, data_rain 8.0, etc.) which overdraw spots
        # with their usual GL_ALWAYS rendering.
        self.render_priority = 6.4

        # Global parameters (updated from outstate)
        self.spot_rate = 1.0
        self.spot_decay = 1.0
        self.spot_saturation = 0.8
        
        # Audio energy (updated from event wrapper)
        self.bass_energy = 0.0
        
        # Time-based rolling hue
        self.current_time = 0.0
        self.hue_speed = 30.0  # Degrees per second (completes full cycle every 12 seconds)
        
        # Spot data arrays
        self.positions = np.empty((0, 3), dtype=np.float32)  # x, y, z
        self.colors = np.empty((0, 3), dtype=np.float32)     # RGB
        self.sizes = np.empty((0,), dtype=np.float32)        # Spot size
        self.lifetimes = np.empty((0,), dtype=np.float32)    # Current lifetime (0-max)
        self.max_lifetimes = np.empty((0,), dtype=np.float32)  # Max lifetime
        
        # Spawn timing
        self.spawn_accumulator = 0.0
        
        # Wrap margin for horizontal wrapping
        self.wrap_margin = 10.0
        
        # Instance VBO
        self.instance_VBO = None
        
    def update_audio(self, bass: float):
        """Update audio energy from event wrapper"""
        self.bass_energy = bass * self.audio_sensitivity
    
    def _spawn_spots(self, count: int):
        """Spawn new spots with rolling hue based on current time"""
        if count <= 0:
            return
        
        # Generate random positions. Z range is intentionally restricted
        # to 72..94 (depth 0.72..0.94 after vertex-shader z/100) so spots
        # sit BEHIND the city silhouette (which cyber_city_skyline writes
        # at depth 0.70) and IN FRONT of the sky depth (0.95). With the
        # global GL_LESS depth test:
        #   - over sky pixels (depth 0.95): spot 0.72..0.94 < 0.95 → pass (visible)
        #   - over building pixels (depth 0.70): spot 0.72..0.94 > 0.70 → fail (hidden)
        # See docs/shader_info.txt for the z=0..100 depth model.
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, count),   # x
            np.random.uniform(0, self.viewport.height, count),  # y
            np.random.uniform(72, 94, count)                    # z (sky depth band)
        ]).astype(np.float32)
        
        # Calculate current hue based on time (rolling through 0-360 degrees)
        current_hue = (self.current_time * self.hue_speed) % 360.0
        
        # Generate colors - all spots spawned at this instant get the same hue
        new_colors = np.zeros((count, 3), dtype=np.float32)
        h = current_hue / 360.0
        s = self.spot_saturation
        v = 1.0  # Start at full brightness
        rgb = colorsys.hsv_to_rgb(h, s, v)
        new_colors[:] = rgb  # All spots get same color
        
        # All spots are 1 pixel in size
        new_sizes = np.ones(count, dtype=np.float32)
        
        # Generate lifetimes
        new_max_lifetimes = np.random.uniform(0.5, 2.0, count).astype(np.float32)
        new_lifetimes = new_max_lifetimes.copy()
        
        # Append to arrays
        self.positions = np.vstack([self.positions, new_positions])
        self.colors = np.vstack([self.colors, new_colors])
        self.sizes = np.concatenate([self.sizes, new_sizes])
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes])
        self.max_lifetimes = np.concatenate([self.max_lifetimes, new_max_lifetimes])
    
    def update(self, dt: float, state: Dict):
        """Update spot lifetimes and spawn new spots"""
        if not self.enabled:
            return
        
        # Decay existing spots
        decay_rate = self.spot_decay * dt
        self.lifetimes -= decay_rate
        
        # Remove dead spots
        alive_mask = self.lifetimes > 0
        if not np.all(alive_mask):
            self.positions = self.positions[alive_mask]
            self.colors = self.colors[alive_mask]
            self.sizes = self.sizes[alive_mask]
            self.lifetimes = self.lifetimes[alive_mask]
            self.max_lifetimes = self.max_lifetimes[alive_mask]
        
        base_rate = self.spot_rate * self.intensity * 50.0  # Base spots per second

        if self.audio_reactive:
            # Audio-driven spawn rate with dramatic response to loud pulses
            spawn_rate = base_rate * (0.05 + self.bass_energy * 15.0)  # Very high sensitivity to bass
        else:
            # Steady twinkle: constant rate, no audio coupling. 0.6 of
            # base (~60 spots/s at defaults) with 0.5-2s lifetimes keeps
            # ~75 spots alive - a calm, continuous shimmer.
            spawn_rate = base_rate * 0.6
        
        # Accumulate spawn probability
        self.spawn_accumulator += dt
        
        # Spawn spots
        if self.spawn_accumulator > 0:
            # Calculate how many spots to spawn
            spot_count = int(spawn_rate * self.spawn_accumulator + np.random.random())
            
            # Spawn all spots with current rolling hue
            self._spawn_spots(spot_count)
            
            # Reset accumulator
            self.spawn_accumulator = 0.0
    
    def compile_shader(self):
        """Compile and link spot shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vs = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            fs = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vs, fs)
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            traceback.print_exc()
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec3 position;  // x, y, z (world position)
        layout(location = 1) in vec3 color;     // RGB color
        layout(location = 2) in float alpha;    // Fade alpha
        
        out vec4 fragColor;
        
        uniform vec2 resolution;
        
        void main() {
            // Convert to clip space
            vec2 clipPos = (position.xy / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Map depth (z=0 near, z=100 far)
            float depth = position.z / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            gl_PointSize = 1.0;  // Single pixel
            
            fragColor = vec4(color, alpha);
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        
        out vec4 outColor;
        
        void main() {
            // Get brightness (V in HSV) from RGB color
            float brightness = max(max(fragColor.r, fragColor.g), fragColor.b);
            
            // Discard if brightness is below threshold
            if (brightness < 0.01) {
                discard;
            }
            
            outColor = fragColor;
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for point rendering"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer (will be updated each frame with point data)
        self.vertex_VBO = glGenBuffers(1)
        self.VBOs.append(self.vertex_VBO)
        
        glBindVertexArray(0)
    
    def render(self, state: Dict):
        """Render all spots with seamless wrapping"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        # Original fast linear fade — spots pop in and decay quickly,
        # matching the snappy "spawn and die" feel the user wants.
        fade_factors = np.clip(self.lifetimes / self.max_lifetimes, 0.0, 1.0)

        # Brightness boost: multiply final RGB by 1.8 and clip at 1.0.
        # Because the framebuffer clamps each channel at 1.0, a spot
        # whose unboosted RGB is >= 1/1.8 = 0.56 reads as saturated
        # white-bright. So during roughly the upper half of each spot's
        # life it sits at full saturation, then falls to 0 quickly over
        # the lower half. Net effect: brighter pop without slowing the
        # fade's overall pace. Single-pixel spots can't go higher than
        # this without enlarging them.
        COLOR_BOOST = 1.8

        # Fade colors by reducing brightness (V) in HSV space, then boost.
        faded_colors = np.zeros_like(self.colors)
        for i in range(len(self.colors)):
            # Convert RGB to HSV
            r, g, b = self.colors[i]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            # Reduce brightness based on fade factor
            v_faded = v * fade_factors[i]

            # Convert back to RGB and apply the saturating boost
            rgb_faded = colorsys.hsv_to_rgb(h, s, v_faded)
            faded_colors[i] = np.minimum(np.array(rgb_faded) * COLOR_BOOST, 1.0)
        
        # Use full alpha (fade is handled by color brightness)
        alphas = np.ones(len(self.positions), dtype=np.float32)
        
        # Identify spots near boundaries that need duplicates
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Create duplicates for wrapping
        duplicate_positions_left = []
        duplicate_indices_left = []
        duplicate_positions_right = []
        duplicate_indices_right = []
        
        if np.any(left_edge_mask):
            left_indices = np.where(left_edge_mask)[0]
            duplicate_pos = self.positions[left_indices].copy()
            duplicate_pos[:, 0] += self.viewport.width  # Shift to right side
            duplicate_positions_right.append(duplicate_pos)
            duplicate_indices_right.extend(left_indices)
        
        if np.any(right_edge_mask):
            right_indices = np.where(right_edge_mask)[0]
            duplicate_pos = self.positions[right_indices].copy()
            duplicate_pos[:, 0] -= self.viewport.width  # Shift to left side
            duplicate_positions_left.append(duplicate_pos)
            duplicate_indices_left.extend(right_indices)
        
        # Combine primary spots with duplicates
        all_positions = [self.positions]
        all_indices = [np.arange(len(self.positions))]
        
        if duplicate_positions_right:
            all_positions.extend(duplicate_positions_right)
            all_indices.append(np.array(duplicate_indices_right))
        
        if duplicate_positions_left:
            all_positions.extend(duplicate_positions_left)
            all_indices.append(np.array(duplicate_indices_left))
        
        combined_positions = np.vstack(all_positions)
        combined_indices = np.concatenate(all_indices)
        
        # Get attributes for all spots (duplicates reference same attributes)
        combined_colors = faded_colors[combined_indices]
        combined_sizes = self.sizes[combined_indices]
        combined_alphas = alphas[combined_indices]
        
        # Sort back-to-front for proper alpha blending
        sort_indices = np.argsort(-combined_positions[:, 2])
        
        # Build vertex data (sorted back-to-front)
        vertex_data = np.hstack([
            combined_positions[sort_indices],          # position (vec3)
            combined_colors[sort_indices],             # color (vec3)
            combined_alphas[sort_indices, np.newaxis]  # alpha (float)
        ]).astype(np.float32)
        
        # Pattern A — particles with per-vertex depth. Per
        # docs/shader_info.txt: do NOT touch glDepthFunc / glDepthMask.
        # The renderer's global GL_LESS + GL_TRUE state handles ordering
        # against the city skyline (which writes building/sky depths)
        # and against other Pattern A particles (rain, stars, etc.).

        # Upload vertex data
        glUseProgram(self.shader)

        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))

        glBindVertexArray(self.VAO)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)
        
        # Setup vertex attributes
        stride = 7 * 4  # 7 floats * 4 bytes (vec3 + vec3 + float)
        
        # Position (location 0) - vec3
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color (location 1) - vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        
        # Alpha (location 2) - float
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        
        # Draw all spots as points
        total_spots = len(combined_positions)
        glDrawArrays(GL_POINTS, 0, total_spots)

        glBindVertexArray(0)
        glUseProgram(0)
