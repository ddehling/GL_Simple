"""
Complete falling leaves effect - GPU-accelerated shader version
Instanced rendering with realistic leaf shapes, colors, and physics
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Functions - Integrate with EventScheduler
# ============================================================================

def shader_falling_leaves(state, outstate, density=2.5, fade_duration=10.0, 
                          bass_sensitivity=8.0, mid_sensitivity=5.0):
    """
    Audio-reactive falling leaves effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_falling_leaves, density=2.5, 
                               bass_sensitivity=2.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Leaf spawn rate multiplier (default 2.5, increased from 1.0)
        fade_duration: Duration of fade in/out in seconds (default 10.0)
                        bass_sensitivity: How much bass affects spawn rate (default 8.0, very reactive)
        mid_sensitivity: How much mids affect flutter/rotation (default 5.0, very reactive)
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    audio_data = outstate.get('sound')  # Audio analysis data
    squish_top_width = outstate.get('scale', 1.0)  # Get scale from state
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing audio-reactive falling leaves effect for frame {frame_id}")
        
        try:
            leaves_effect = viewport.add_effect(
                FallingLeavesEffect,
                density=density,
                max_leaves=30,  # Gentle amount of leaves
                squish_top_width=squish_top_width
            )
            state['leaves_effect'] = leaves_effect
            state['smoothed_bass'] = 0.0  # For audio smoothing
            state['smoothed_mid'] = 0.0
            print(f"✓ Initialized shader falling leaves for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize falling leaves: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters and audio reactivity
    if 'leaves_effect' in state:
        state['leaves_effect'].density = outstate.get('leaf_density', density)
        
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)  # Default 60s if not set
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            # Fade in during first N seconds
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            # Fade out during last N seconds
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            # Full opacity in the middle
            fade_factor = 1.0
        
        # Update effect's fade factor (clip to 0-1 range)
        state['leaves_effect'].fade_factor = np.clip(fade_factor, 0, 1)
        
        # Audio reactivity - update every frame if audio data available
        if audio_data is not None:
            # Use short-term normalized bands for reactive response
            bands = audio_data['norm_short'][0]
            
            # Extract frequency ranges
            bass_energy = np.mean(bands[0:8])       # Bass: 40-300 Hz (deep sounds)
            mid_energy = np.mean(bands[8:20])       # Mids: 300-2000 Hz (most music)
            high_energy = np.mean(bands[20:32])     # Highs: 2000-16000 Hz (cymbals, etc.)
            
                        # Smooth audio values for less jittery response
            smoothing = 0.25  # 0-1, higher = faster response (increased for more reactivity)
            state['smoothed_bass'] = smoothing * bass_energy + (1 - smoothing) * state['smoothed_bass']
            state['smoothed_mid'] = smoothing * mid_energy + (1 - smoothing) * state['smoothed_mid']
            
            # Apply sensitivity multipliers and update effect (MUCH higher limits)
            state['leaves_effect'].audio_bass = np.clip(state['smoothed_bass'] * bass_sensitivity, 0, 10)
            state['leaves_effect'].audio_mid = np.clip(state['smoothed_mid'] * mid_sensitivity, 0, 10)
            state['leaves_effect'].audio_high = np.clip(high_energy * 2.0, 0, 5)  # High freq for sparkle
            
                                    # Detect bass hits for burst spawning (LOWER THRESHOLD for more triggers)
            if len(state.get('prev_bass_hist', [])) >= 3:
                # Check if current bass significantly higher than recent average
                recent_avg = np.mean(state['prev_bass_hist'][-3:])
                if bass_energy > recent_avg + 0.15:  # Bass hit threshold (lowered for more triggers)
                    state['leaves_effect'].trigger_bass_burst()
            
            # Store bass history for beat detection (keep last 5 frames)
            if 'prev_bass_hist' not in state:
                state['prev_bass_hist'] = []
            state['prev_bass_hist'].append(bass_energy)
            if len(state['prev_bass_hist']) > 5:
                state['prev_bass_hist'].pop(0)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'leaves_effect' in state:
            print(f"Cleaning up falling leaves effect for frame {frame_id}")
            viewport.effects.remove(state['leaves_effect'])
            state['leaves_effect'].cleanup()
            print(f"✓ Cleaned up shader falling leaves for frame {frame_id}")



# ============================================================================
# Main Falling Leaves Effect (Cartesian Coordinates)
# ============================================================================

class FallingLeavesEffect(ShaderEffect):
    """GPU-based audio-reactive falling leaves effect using instanced rendering"""
    
    # 11 floats per instance: pos.xy(2) + size(1) + rotation(1) + color.rgb(3) + alpha(1) + leaf_type(1) + distance(1) + squish(1)
    INSTANCE_FLOATS = 11

    def __init__(self, viewport, density: float = 2.5, max_leaves: int = 100, squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.density = density
        self.max_leaves = max_leaves
        self.squish_top_width = squish_top_width
        self.viewport_height = viewport.height  # Store for squish calculation
        self.instance_VBO = None
        self.fade_factor = 0.0  # For fade in/out (updated by event wrapper)

        # Pre-allocated CPU mirror of the instance VBO (worst-case 3x for wrap
        # duplicates on both edges). Written via slice assignment in render()
        # to avoid per-frame np.hstack allocation.
        self._instance_capacity = 0
        self._instance_buffer = None
        # Cached uniform locations populated in setup_buffers().
        self._uniform_resolution = -1
        self._uniform_fade = -1
        
        # Audio reactivity parameters (updated by event wrapper)
        self.audio_bass = 0.0      # Bass energy (affects spawn rate)
        self.audio_mid = 0.0       # Mid energy (affects flutter/rotation)
        self.audio_high = 0.0      # High energy (affects brightness)
        self.bass_burst = 0.0      # Burst effect on bass hits
        
        # Vectorized leaf data
        self.positions = np.zeros((0, 2), dtype=np.float32)  # [x, y]
        self.velocities = np.zeros((0, 2), dtype=np.float32)  # [vx, vy]
        self.sizes = np.zeros(0, dtype=np.float32)
        self.rotations = np.zeros(0, dtype=np.float32)
        self.rotation_speeds = np.zeros(0, dtype=np.float32)
        self.flutter_phases = np.zeros(0, dtype=np.float32)
        self.flutter_amplitudes = np.zeros(0, dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)  # [r, g, b] in RGB
        self.alphas = np.zeros(0, dtype=np.float32)
        self.lifetimes = np.zeros(0, dtype=np.float32)
        self.leaf_types = np.zeros(0, dtype=np.int32)
        self.distances = np.zeros(0, dtype=np.float32)  # Depth (5-25) for 3D ordering
        self.squish_factors = np.zeros(0, dtype=np.float32)  # Horizontal width multipliers
        self.wind_phases = np.zeros(0, dtype=np.float32)  # Individual wind turbulence phases
        
        # Wind simulation parameters
        self.wind_time = 0.0
        self.wind_gust_phase = 0.0
        
        # Horizontal wrapping margin (larger than largest leaf)
        self.wrap_margin = 50  # Should exceed max leaf size
        
    def _spawn_leaves(self, count: int, season: float = 0.625):
        """Spawn new leaves at random positions"""
        if count <= 0:
            return
            
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, count),
            np.random.uniform(-5, 0, count)  # Start above screen
        ])
        
        new_velocities = np.column_stack([
            np.zeros(count),  # No initial horizontal velocity (wind will control this)
            np.random.uniform(15.0, 25.0, count)  # Natural fall speed (pixels per second)
        ])
        
        # NEW: Generate random distances (depth) between 5 and 25
        new_distances = np.random.uniform(5.0, 25.0, count)
        
        # Base size scaled by distance (closer = larger)
        base_sizes = np.random.uniform(2.0, 3.5, count)
        new_sizes = base_sizes * (5.0 / new_distances)  # Scale by distance
        
        new_rotations = np.random.uniform(0, 2 * np.pi, count)
        new_rotation_speeds = np.random.uniform(-0.3, 0.3, count)  # Slower, more natural rotation
        new_flutter_phases = np.random.uniform(0, 2 * np.pi, count)
        new_flutter_amplitudes = np.random.uniform(0.3, 0.8, count)  # Reduced amplitude for gentler movement
        new_wind_phases = np.random.uniform(0, 2 * np.pi, count)  # Random wind turbulence phases
        
        # Adjust alpha based on distance (farther = more transparent)
        base_alphas = np.random.uniform(0.9, 1.0, count)
        new_alphas = base_alphas 
        
        # Increased lifetime: leaves last much longer now
        new_lifetimes = np.ones(count) * 5.0  # 5x longer lifetime
        
        # Assign random leaf types (0-4 for 5 different shapes)
        new_leaf_types = np.random.randint(0, 5, count)
        
        # Generate colors based on season
        new_colors = self._generate_leaf_colors(count, season)
        
        # Calculate squish factors based on y position (bottom = 1.0, top = squish_top_width)
        y_normalized = (self.viewport_height - new_positions[:, 1]) / self.viewport_height
        new_squish_factors = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Concatenate with existing arrays
        self.positions = np.vstack([self.positions, new_positions]) if len(self.positions) > 0 else new_positions
        self.velocities = np.vstack([self.velocities, new_velocities]) if len(self.velocities) > 0 else new_velocities
        self.sizes = np.concatenate([self.sizes, new_sizes]) if len(self.sizes) > 0 else new_sizes
        self.rotations = np.concatenate([self.rotations, new_rotations]) if len(self.rotations) > 0 else new_rotations
        self.rotation_speeds = np.concatenate([self.rotation_speeds, new_rotation_speeds]) if len(self.rotation_speeds) > 0 else new_rotation_speeds
        self.flutter_phases = np.concatenate([self.flutter_phases, new_flutter_phases]) if len(self.flutter_phases) > 0 else new_flutter_phases
        self.flutter_amplitudes = np.concatenate([self.flutter_amplitudes, new_flutter_amplitudes]) if len(self.flutter_amplitudes) > 0 else new_flutter_amplitudes
        self.colors = np.vstack([self.colors, new_colors]) if len(self.colors) > 0 else new_colors
        self.alphas = np.concatenate([self.alphas, new_alphas]) if len(self.alphas) > 0 else new_alphas
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes]) if len(self.lifetimes) > 0 else new_lifetimes
        self.leaf_types = np.concatenate([self.leaf_types, new_leaf_types]) if len(self.leaf_types) > 0 else new_leaf_types
        self.distances = np.concatenate([self.distances, new_distances]) if len(self.distances) > 0 else new_distances
        self.squish_factors = np.concatenate([self.squish_factors, new_squish_factors]) if len(self.squish_factors) > 0 else new_squish_factors
        self.wind_phases = np.concatenate([self.wind_phases, new_wind_phases]) if len(self.wind_phases) > 0 else new_wind_phases
    
    def _generate_leaf_colors(self, count: int, season: float) -> np.ndarray:
        """Generate leaf colors based on season (RGB format)"""
        from skimage import color as skcolor
        
        colors_hsv = np.zeros((count, 3), dtype=np.float32)
        
        # Calculate distance from spring and fall
        spring_distance = min(abs(season - 0.125), 1 - abs(season - 0.125))
        fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
        
        spring_factor = max(0, 1 - spring_distance * 4)
        fall_factor = max(0, 1 - fall_distance * 4)
        
        if spring_factor > 0.5:
            # All green leaves in spring
            colors_hsv[:, 0] = np.random.uniform(0.25, 0.35, count)  # Green hue
            colors_hsv[:, 1] = np.random.uniform(0.7, 0.9, count)
            colors_hsv[:, 2] = np.random.uniform(0.3, 0.5, count)
        else:
            # Seasonal mix with fall colors
            color_types = np.random.random(count)
            
            # Red leaves
            red_proportion = 0.1 + 0.3 * fall_factor
            red_mask = color_types < red_proportion
            colors_hsv[red_mask, 0] = np.random.uniform(0.00, 0.05, np.sum(red_mask))
            colors_hsv[red_mask, 1] = np.random.uniform(0.8, 0.95, np.sum(red_mask))
            colors_hsv[red_mask, 2] = np.random.uniform(0.4, 0.6, np.sum(red_mask))
            
            # Orange leaves
            orange_proportion = red_proportion + (0.1 + 0.2 * fall_factor)
            orange_mask = (color_types >= red_proportion) & (color_types < orange_proportion)
            colors_hsv[orange_mask, 0] = np.random.uniform(0.05, 0.10, np.sum(orange_mask))
            colors_hsv[orange_mask, 1] = np.random.uniform(0.85, 0.95, np.sum(orange_mask))
            colors_hsv[orange_mask, 2] = np.random.uniform(0.45, 0.65, np.sum(orange_mask))
            
            # Yellow leaves
            yellow_proportion = orange_proportion + (0.2 + 0.1 * fall_factor)
            yellow_mask = (color_types >= orange_proportion) & (color_types < yellow_proportion)
            colors_hsv[yellow_mask, 0] = np.random.uniform(0.10, 0.15, np.sum(yellow_mask))
            colors_hsv[yellow_mask, 1] = np.random.uniform(0.8, 0.9, np.sum(yellow_mask))
            colors_hsv[yellow_mask, 2] = np.random.uniform(0.5, 0.7, np.sum(yellow_mask))
            
            # Brown leaves
            brown_proportion = yellow_proportion + (0.05 + 0.15 * fall_factor)
            brown_mask = (color_types >= yellow_proportion) & (color_types < brown_proportion)
            colors_hsv[brown_mask, 0] = np.random.uniform(0.07, 0.12, np.sum(brown_mask))
            colors_hsv[brown_mask, 1] = np.random.uniform(0.6, 0.8, np.sum(brown_mask))
            colors_hsv[brown_mask, 2] = np.random.uniform(0.3, 0.4, np.sum(brown_mask))
            
            # Green leaves
            green_mask = color_types >= brown_proportion
            colors_hsv[green_mask, 0] = np.random.uniform(0.25, 0.35, np.sum(green_mask))
            colors_hsv[green_mask, 1] = np.random.uniform(0.7, 0.9, np.sum(green_mask))
            colors_hsv[green_mask, 2] = np.random.uniform(0.3, 0.5, np.sum(green_mask))
        
        # Convert HSV to RGB
        colors_rgb = np.zeros_like(colors_hsv)
        for i in range(len(colors_hsv)):
            rgb = skcolor.hsv2rgb(colors_hsv[i:i+1].reshape(1, 1, 3))
            colors_rgb[i] = rgb.flatten()
        
        return colors_rgb
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec2 offset;    // Leaf position (x, y)
        layout(location = 2) in float size;     // Leaf size
        layout(location = 3) in float rotation; // Leaf rotation
        layout(location = 4) in vec4 color;     // Color (r, g, b, alpha)
        layout(location = 5) in float leafType; // Leaf shape type
        layout(location = 6) in float distance; // Depth value (5-25)
        layout(location = 7) in float squishFactor; // Horizontal width multiplier
        
        out vec4 fragColor;
        out vec2 fragPos;  // Position within quad (-1 to 1)
        flat out int fragLeafType;
        uniform vec2 resolution;
        uniform float fadeAlpha;  // Global fade factor for fade in/out
        
        void main() {
            fragPos = position;
            fragLeafType = int(leafType);
            
            // Apply rotation to quad
            float c = cos(rotation);
            float s = sin(rotation);
            vec2 rotated = vec2(
                position.x * c - position.y * s,
                position.x * s + position.y * c
            );
            
            // Scale by leaf size with squish applied to horizontal
            vec2 scaled = vec2(
                rotated.x * size * 3.0 * squishFactor,
                rotated.y * size * 3.0
            );
            
            // Translate to leaf position
            vec2 pos = scaled + offset;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping: z = 0-100 -> depth = 0.0-1.0
            // distance 5 (near) -> depth 0.05 (close to camera)
            // distance 25 (far) -> depth 0.25 (farther from camera)
            float depth = distance / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Apply fade factor to alpha
            fragColor = vec4(color.rgb, color.a * fadeAlpha);
        }
        """



    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;  // Position within quad (-1 to 1)
        flat in int fragLeafType;
        out vec4 outColor;
        
        // Leaf type 0: Oak-style (rounded lobes)
        float oak_leaf(float nx, float ny) {
            float dist = abs(ny * 0.5);
            float width = (1.0 - nx * nx) * 0.5;
            width *= smoothstep(-0.9, -0.3, nx);
            width *= smoothstep(0.95, 0.3, nx);
            
            // Add lobes
            float lobe = 0.1 * sin(nx * 12.0) * (1.0 - nx * nx);
            width += lobe;
            
            return step(dist, width);
        }
        
        // Leaf type 1: Maple-style (pointed lobes)
        float maple_leaf(float nx, float ny) {
            float angle = atan(ny, nx);
            float r = length(vec2(nx, ny));
            
            // Create 5 pointed lobes
            float lobes = 0.6 + 0.3 * cos(angle * 2.5);
            return step(r, lobes * 0.8);
        }
        
        // Leaf type 2: Willow-style (long and narrow)
        float willow_leaf(float nx, float ny) {
            float dist = abs(ny * 0.3);  // Very narrow
            float width = (1.0 - nx * nx * 0.8) * 0.3;
            width *= smoothstep(-0.95, -0.5, nx);
            width *= smoothstep(0.98, 0.5, nx);
            return step(dist, width);
        }
        
        // Leaf type 3: Birch-style (triangular with serrated edge)
        float birch_leaf(float nx, float ny) {
            float dist = abs(ny * 0.6);
            float width = (1.0 - nx) * 0.4;
            width *= smoothstep(-0.9, -0.2, nx);
            
            // Serrated edges
            float serration = 0.05 * sin(nx * 25.0);
            width += serration;
            
            return step(dist, width);
        }
        
        // Leaf type 4: Aspen-style (circular with small point)
        float aspen_leaf(float nx, float ny) {
            float r = length(vec2(nx * 1.2, ny));
            float width = 0.75;
            
            // Add point at tip
            if (nx > 0.5) {
                width *= smoothstep(0.95, 0.6, nx);
            }
            
            return step(r, width);
        }
        
        void main() {
            float nx = fragPos.x;
            float ny = fragPos.y;
            
            // Select leaf shape based on type
            float leaf_mask = 0.0;
            if (fragLeafType == 0) {
                leaf_mask = oak_leaf(nx, ny);
            } else if (fragLeafType == 1) {
                leaf_mask = maple_leaf(nx, ny);
            } else if (fragLeafType == 2) {
                leaf_mask = willow_leaf(nx, ny);
            } else if (fragLeafType == 3) {
                leaf_mask = birch_leaf(nx, ny);
            } else {
                leaf_mask = aspen_leaf(nx, ny);
            }
            
            if (leaf_mask < 0.5) {
                discard;
            }
            
            // Common vein structure
            float main_vein = smoothstep(0.02, 0.0, abs(ny * 0.5));
            
            float side_veins = 0.0;
            for (float i = -0.6; i <= 0.6; i += 0.15) {
                float vx = i;
                float vein_y = (nx - vx) * 0.4;
                float vein_dist = abs(ny * 0.5 - vein_y);
                float vein_fade = smoothstep(0.8, 0.0, abs(nx - vx)) * step(vx, nx);
                side_veins = max(side_veins, smoothstep(0.008, 0.0, vein_dist) * vein_fade);
            }
            
            float veins = max(main_vein, side_veins * 0.5);
            
            // Texture variation
            float color_var = fract(sin(dot(fragPos * 30.0, vec2(12.9898, 78.233))) * 43758.5453);
            color_var = (color_var - 0.5) * 0.08;
            
            // Soft edge
            float edge_dist = min(
                min(1.0 - abs(nx), 1.0 - abs(ny)),
                leaf_mask
            );
            float edge = smoothstep(0.0, 0.2, edge_dist);
            
            vec3 final_color = fragColor.rgb * (1.0 - veins * 0.35 + color_var);
            float alpha = fragColor.a * edge;
            
            outColor = vec4(final_color, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link leaf shaders - REQUIRED by ShaderEffect base class"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"FallingLeavesEffect shader compilation error: {e}")
            raise

    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced rendering"""
        # Quad vertices
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
        
        # Instance buffer — pre-allocate at worst-case capacity so render()
        # uses glBufferSubData (no GPU re-allocation per frame).
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)

        # Worst case: every leaf is near both edges → 3 copies. With max_leaves
        # ~100 this is 300 instances * 44 bytes = 13 KB, trivially small.
        self._instance_capacity = self.max_leaves * 3
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER,
                     self._instance_capacity * self.INSTANCE_FLOATS * 4,
                     None, GL_DYNAMIC_DRAW)
        self._instance_buffer = np.empty(
            (self._instance_capacity, self.INSTANCE_FLOATS), dtype=np.float32)

        glBindVertexArray(0)

        # Cache uniform locations.
        self._uniform_resolution = glGetUniformLocation(self.shader, "resolution")
        self._uniform_fade = glGetUniformLocation(self.shader, "fadeAlpha")

    def trigger_bass_burst(self):
        """Trigger a burst effect on bass hit"""
        self.bass_burst = 8.0  # Very strong burst for obvious effect
    
    def update(self, dt: float, state: Dict):
        """Update leaf positions and properties with audio reactivity"""
        if not self.enabled:
            return
        
        # Get environment parameters
        wind = state.get('wind', 0.0)
        whomp = state.get('whomp', 0.0)
        season = state.get('season', 0.625)
        
        # Calculate fall factor for spawn rate
        fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
        fall_factor = 1 - 1.9 * fall_distance
        
        # Gentle spawn rate with subtle audio influence
        audio_spawn_multiplier = 1.0 + self.audio_bass * 0.3 + self.bass_burst * 0.5
        
        # Spawn new leaves gently
        if len(self.positions) < self.max_leaves:
            leaf_rate = 0.02 * self.density * fall_factor * audio_spawn_multiplier  # Much lower base rate
            if np.random.random() < leaf_rate:
                # Spawn 1-2 leaves at a time (occasionally 3 on bass hit)
                spawn_count = min(
                    1 + int(np.random.random() < 0.3) + int(self.bass_burst > 1.0),
                    self.max_leaves - len(self.positions)
                )
                if spawn_count > 0:
                    self._spawn_leaves(spawn_count, season)
        
                # Decay bass burst effect (slower decay = longer visible effect)
        self.bass_burst *= 0.90
        
        if len(self.positions) == 0:
            return
        
        # Update wind simulation time
        self.wind_time += dt
        self.wind_gust_phase += dt * 0.5
        
        # Generate wind gusts (smooth varying wind strength)
        wind_gust = np.sin(self.wind_gust_phase) * 0.5 + 0.5  # 0.0 to 1.0
        wind_gust += np.sin(self.wind_gust_phase * 2.3) * 0.3  # Add complexity
        wind_gust = np.clip(wind_gust, 0, 1)
        
        # Base wind force with gusts
        base_wind = wind * (0.3 + wind_gust * 0.7)  # Wind varies from 30% to 100% strength
        
        # Individual leaf turbulence (each leaf experiences slightly different wind)
        self.wind_phases += dt * 3.0  # Update turbulence phases
        turbulence_x = np.sin(self.wind_phases) * 0.4
        turbulence_y = np.cos(self.wind_phases * 1.7) * 0.2
        
        # Apply wind with turbulence to horizontal velocity
        wind_force_x = base_wind + turbulence_x * np.abs(wind) * 0.5
        self.velocities[:, 0] = wind_force_x * 30.0  # Scale to pixel velocity
        
        # Wind affects vertical movement (leaves sway up/down in gusts)
        wind_vertical = turbulence_y * np.abs(wind) * 15.0
        
        # Audio subtly affects fall rate (bass makes them fall slightly faster/slower)
        audio_fall_modifier = 1.0 + (self.audio_bass - 0.5) * 0.1  # Subtle ±10% variation
        
        # Update positions (apply audio modifier directly, don't modify velocities)
        movement_multiplier = 1.0 + whomp * 0.5  # Much gentler whomp effect
        self.positions[:, 0] += self.velocities[:, 0] * dt * movement_multiplier
        self.positions[:, 1] += (self.velocities[:, 1] + wind_vertical) * dt * movement_multiplier * audio_fall_modifier
        self.positions[:, 1] *= (1.0 - whomp * 0.1 * dt)
        
        # Update squish factors based on current y positions
        y_normalized = (self.viewport_height - self.positions[:, 1]) / self.viewport_height
        self.squish_factors[:] = 1.0 + (self.squish_top_width - 1.0) * y_normalized
        
        # Wind affects rotation speed (leaves spin faster in stronger wind)
        wind_rotation_influence = np.abs(wind) * 0.5 * wind_gust
        rotation_speed_mult = 1.0 + self.audio_mid * 0.2 + wind_rotation_influence
        self.rotations += self.rotation_speeds * dt * 2 * rotation_speed_mult
        
        # Update flutter phases for subtle brightness variation later
        self.flutter_phases += 0.02 * dt * 60  # Slow phase for brightness variation
        
                # Decrease lifetimes more slowly (leaves last longer)
        self.lifetimes -= 0.0002 * dt * 60  # 5x slower decay (was 0.001)
        
        # Wrap leaves that go completely off screen (teleport to other side)
        # This is different from rendering duplicates - this handles leaves that are fully off-screen
        off_screen_margin = self.wrap_margin * 2
        
        wrap_left_mask = self.positions[:, 0] < -self.wrap_margin
        if np.any(wrap_left_mask):
            self.positions[wrap_left_mask, 0] += self.viewport.width
        
        # Teleport leaves that have moved completely off the right side  
        # Original at x = width + 60, duplicate was at x = (width + 60) - width = 60
        # Teleport original to where duplicate was: x = 60
        wrap_right_mask = self.positions[:, 0] > self.viewport.width + self.wrap_margin  
        if np.any(wrap_right_mask):
            self.positions[wrap_right_mask, 0] -= self.viewport.width
        
        # Filter out-of-bounds leaves - only remove if below screen or lifetime expired
        # NO horizontal filtering - wrapping handles horizontal bounds
        valid_mask = (
            (self.positions[:, 1] < self.viewport.height + 100) &
            (self.lifetimes > 0)
        )
        
        if not np.all(valid_mask):
            self.positions = self.positions[valid_mask]
            self.velocities = self.velocities[valid_mask]
            self.sizes = self.sizes[valid_mask]
            self.rotations = self.rotations[valid_mask]
            self.rotation_speeds = self.rotation_speeds[valid_mask]
            self.flutter_phases = self.flutter_phases[valid_mask]
            self.flutter_amplitudes = self.flutter_amplitudes[valid_mask]
            self.colors = self.colors[valid_mask]
            self.alphas = self.alphas[valid_mask]
            self.lifetimes = self.lifetimes[valid_mask]
            self.leaf_types = self.leaf_types[valid_mask]
            self.distances = self.distances[valid_mask]
            self.squish_factors = self.squish_factors[valid_mask]
            self.wind_phases = self.wind_phases[valid_mask]

    def render(self, state: Dict):
        """Render all leaves using instancing with horizontal wrapping"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        # DO NOT toggle depth test or blending - global state handles this!
        # Depth testing and alpha blending are ALWAYS enabled globally
        
        glUseProgram(self.shader)

        if self._uniform_resolution != -1:
            glUniform2f(self._uniform_resolution,
                        float(self.viewport.width), float(self.viewport.height))
        if self._uniform_fade != -1:
            glUniform1f(self._uniform_fade, self.fade_factor)

        # === HORIZONTAL WRAPPING: queue edge leaves as duplicates ===
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)

        # Audio brightness/warmth applied during the slice write so we don't
        # need an intermediate `adjusted_colors` array.
        audio_brightness = 1.0 + self.audio_high * 0.15
        audio_color_shift = self.audio_mid * 0.08

        # Grow the pre-allocated instance buffer if the leaf population
        # outgrew the worst-case headroom (e.g. max_leaves was bumped up).
        n_orig = len(self.positions)
        num_left = int(np.sum(left_edge_mask))
        num_right = int(np.sum(right_edge_mask))
        total_instances = n_orig + num_left + num_right
        if total_instances > self._instance_capacity:
            new_capacity = max(total_instances, self._instance_capacity * 2)
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
            glBufferData(GL_ARRAY_BUFFER,
                         new_capacity * self.INSTANCE_FLOATS * 4,
                         None, GL_DYNAMIC_DRAW)
            self._instance_capacity = new_capacity
            self._instance_buffer = np.empty(
                (new_capacity, self.INSTANCE_FLOATS), dtype=np.float32)

        buf = self._instance_buffer

        def _write_block(dst_start, src_idx, x_offset):
            n = len(src_idx) if src_idx is not None else n_orig
            if n == 0:
                return
            end = dst_start + n
            if src_idx is None:
                buf[dst_start:end, 0:2] = self.positions
                buf[dst_start:end, 2] = self.sizes
                buf[dst_start:end, 3] = self.rotations
                buf[dst_start:end, 4:7] = self.colors
                buf[dst_start:end, 7] = self.alphas
                buf[dst_start:end, 8] = self.leaf_types
                buf[dst_start:end, 9] = self.distances
                buf[dst_start:end, 10] = self.squish_factors
            else:
                buf[dst_start:end, 0:2] = self.positions[src_idx]
                buf[dst_start:end, 2] = self.sizes[src_idx]
                buf[dst_start:end, 3] = self.rotations[src_idx]
                buf[dst_start:end, 4:7] = self.colors[src_idx]
                buf[dst_start:end, 7] = self.alphas[src_idx]
                buf[dst_start:end, 8] = self.leaf_types[src_idx]
                buf[dst_start:end, 9] = self.distances[src_idx]
                buf[dst_start:end, 10] = self.squish_factors[src_idx]
            if x_offset:
                buf[dst_start:end, 0] += x_offset
            # Apply audio brightness to RGB columns in-place on the slice.
            buf[dst_start:end, 4:7] *= audio_brightness
            np.clip(buf[dst_start:end, 4] + audio_color_shift, 0.0, 1.0,
                    out=buf[dst_start:end, 4])

        _write_block(0, None, 0.0)
        idx = n_orig
        if num_left:
            _write_block(idx, np.where(left_edge_mask)[0], self.viewport.width)
            idx += num_left
        if num_right:
            _write_block(idx, np.where(right_edge_mask)[0], -self.viewport.width)
            idx += num_right

        # Sub-data upload reuses the existing GPU allocation (no realloc).
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        total_instances * self.INSTANCE_FLOATS * 4,
                        buf[:total_instances])
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes (11 floats per instance)
        stride = 11 * 4  # 11 floats * 4 bytes
        
        # Attribute 1: offset (x, y)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Attribute 2: size
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Attribute 3: rotation
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Attribute 4: color (r, g, b, alpha)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Attribute 5: leaf type
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
        # Attribute 6: distance (depth)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)
        
        # Attribute 7: squishFactor
        glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(40))
        glEnableVertexAttribArray(7)
        glVertexAttribDivisor(7, 1)
        
        # Render all leaf instances (originals + duplicates for seamless wrapping)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, total_instances)
        
        glBindVertexArray(0)
        glUseProgram(0)