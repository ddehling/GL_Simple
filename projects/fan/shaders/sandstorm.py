"""
Sandstorm shader effect - GPU-accelerated sand particles
Instanced rendering with wind-driven physics and horizontal wrapping
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_sandstorm(state, outstate, fade_duration=5.0, max_particles=600, 
                     particle_size=2.0, squish_top_width=1.0):
    """
    Wind-driven sandstorm effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_sandstorm, density=1.5, 
                               max_particles=200, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        fade_duration: Duration of fade in/out in seconds (default 5.0)
        max_particles: Maximum number of sand particles (default 200)
        particle_size: Base size of sand particles in pixels (default 2.0)
        squish_top_width: Horizontal width multiplier at top of viewport (default 1.0)
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
        try:
            effect = viewport.add_effect(
                SandstormEffect,
                max_particles=max_particles, 
                particle_size=particle_size, 
                squish_top_width=squish_top_width
            )
            state['sandstorm_effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing sandstorm effect: {e}")
            traceback.print_exc()
            return
    
    # Update effect parameters from outstate
    if 'sandstorm_effect' in state:
        effect = state['sandstorm_effect']
        
        # Get wind and sand_density from outstate
        effect.wind_strength = outstate.get('wind', 0.0)
        effect.sand_density = outstate.get('sand_density', 0.0)
        
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
        if 'sandstorm_effect' in state:
            effect = state['sandstorm_effect']
            if effect in viewport.effects:
                viewport.effects.remove(effect)
            effect.cleanup()
            del state['sandstorm_effect']


# ============================================================================
# Sandstorm Effect Class
# ============================================================================

class SandstormEffect(ShaderEffect):
    """GPU-based sandstorm effect using instanced rendering"""
    
    def __init__(self, viewport, max_particles: int = 600, particle_size: float = 2.0,
                 squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.max_particles = max_particles
        self.particle_size = particle_size
        self.squish_top_width = squish_top_width
        self.viewport_height = viewport.height
        self.instance_VBO = None
        self.fade_factor = 0.0  # For fade in/out (updated by event wrapper)

        # FanCoords for converting between buffer pixels and physical feet.
        # Particles are stored in PHYSICAL coords so their trajectories are
        # straight lines in real fan space (not curved spirals in buffer
        # space). Imported lazily via FanCoords to avoid module cycles.
        from renderer.fan_coords import FanCoords
        self._fan = FanCoords(viewport.width, viewport.height)
        # Physical fan extents — used for spawn / cull bounds.
        self._fan_x_half = self._fan.outer_r_ft + 1.0     # 21.6 ft
        self._fan_y_max  = self._fan.outer_r_ft + 1.0     # 21.6 ft
        self._fan_y_min  = -2.0                            # off-fan margin

        # Environmental parameters (updated by event wrapper)
        self.wind_strength = 0.0   # From outstate['wind'] (ft/s scale internally)
        self.sand_density = 0.0    # From outstate['sand_density']

        # Vectorized particle data — positions and velocities are now in
        # PHYSICAL FEET, NOT viewport pixels. Render-time we convert to
        # pixel coords via the FanCoords mapper.
        self.positions = np.zeros((0, 2), dtype=np.float32)  # [x_ft, y_ft]
        self.velocities = np.zeros((0, 2), dtype=np.float32)  # [vx_ft/s, vy_ft/s]
        self.sizes = np.zeros(0, dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)  # [r, g, b]
        self.alphas = np.zeros(0, dtype=np.float32)
        self.lifetimes = np.zeros(0, dtype=np.float32)
        self.distances = np.zeros(0, dtype=np.float32)  # Depth (10-40) for 3D ordering
        # squish_factors removed — vertex shader computes squish from
        # the radial position in `offset` directly.
        self.turbulence_phases = np.zeros(0, dtype=np.float32)  # Individual turbulence

        # Time tracking
        self.time = 0.0

        # No unconditional initial spawn — particles are spawned by update()
        # based on the current `sand_density`. This avoids 50 ghost grains
        # appearing for the first ~10 seconds during states with no wind /
        # sand activity. With density ≥ 0.5 the particle population reaches
        # steady-state in well under a second, so the warm-up is invisible.
    
    def _hsv_to_rgb_vectorized(self, h, s, v):
        """Convert HSV to RGB using vectorized NumPy operations
        
        Args:
            h: Hue (0-1), can be scalar or array
            s: Saturation (0-1), can be scalar or array  
            v: Value/Brightness (0-1), can be scalar or array
            
        Returns:
            RGB array of shape (N, 3) where N is the number of colors
        """
        # Ensure inputs are arrays
        h = np.atleast_1d(h)
        s = np.atleast_1d(s)
        v = np.atleast_1d(v)
        
        # Initialize output
        n = len(h)
        rgb = np.zeros((n, 3), dtype=np.float32)
        
        # Calculate chroma
        c = v * s
        
        # Calculate hue sector (0-6)
        h_prime = (h * 6.0) % 6.0
        
        # Calculate x (second largest component)
        x = c * (1 - np.abs(h_prime % 2 - 1))
        
        # Calculate m (amount to add to match value)
        m = v - c
        
        # Assign RGB based on hue sector
        for i in range(n):
            sector = int(h_prime[i])
            if sector == 0:
                rgb[i] = [c[i], x[i], 0]
            elif sector == 1:
                rgb[i] = [x[i], c[i], 0]
            elif sector == 2:
                rgb[i] = [0, c[i], x[i]]
            elif sector == 3:
                rgb[i] = [0, x[i], c[i]]
            elif sector == 4:
                rgb[i] = [x[i], 0, c[i]]
            else:  # sector == 5
                rgb[i] = [c[i], 0, x[i]]
            
            # Add m to match value
            rgb[i] += m[i]
        
        return rgb
        
    def _spawn_particles(self, count: int):
        """Spawn new sand particles. Positions are PHYSICAL FEET so their
        trajectories are straight lines on the fan.

        Spawns INSIDE the fan annulus using polar sampling — picks a
        random radius in [inner, outer] and a theta biased toward the
        upwind edge (so wind blows particles ACROSS the fan rather than
        out of it). Cartesian-band spawning was causing many particles to
        spawn outside the annulus, where the vertex shader clamps them to
        the outer ring and then they drift inward radially — that was the
        "moving along radials" symptom.
        """
        if count <= 0:
            return

        # Uniform-AREA sampling across the fan annulus so particle density
        # is spatially even (uniform-r sampling biased density toward the
        # inner ring because area scales with r dr dθ; the upwind/outer
        # corner ended up under-populated). Theta now spans the full half-
        # circle so spawns happen across the whole fan; the wind continues
        # to push them along straight horizontal lines either way, so the
        # visible flow still reads as wind-driven.
        rs = np.sqrt(np.random.uniform(
            self._fan.inner_r_ft ** 2,
            self._fan.outer_r_ft ** 2,
            count,
        ))
        thetas = np.random.uniform(0.0, np.pi, count)
        new_positions = np.column_stack([
            rs * np.cos(thetas),
            rs * np.sin(thetas),
        ])

        # Wind-driven horizontal velocity in FEET per second.
        # 12 ft/s at wind=1 gives visibly fast dust without warping; per-
        # particle 0.8x..1.2x scatter stays constant for straight lines.
        wind_speed_ft_s = self.wind_strength * 12.0
        new_velocities = np.column_stack([
            np.random.uniform(wind_speed_ft_s * 0.8, wind_speed_ft_s * 1.2, count),
            np.random.uniform(-0.15, 0.15, count),  # gentle vertical drift
        ])
        
        # Random distances (depth) between 10 and 40
        new_distances = np.random.uniform(10.0, 40.0, count)
        
        # Size scaled by distance (closer = larger)
        base_sizes = np.random.uniform(self.particle_size * 0.5, self.particle_size * 1.5, count)
        new_sizes = base_sizes * (10.0 / new_distances)  # Scale by distance
        
        # Sand colors (yellowish-brown tones)
        hue_variation = np.random.uniform(-0.05, 0.05, count)
        saturation = np.random.uniform(0.3, 0.5, count)
        brightness = np.random.uniform(0.6, 0.9, count)
        
        # Convert HSV to RGB for sand colors (hue around 0.11 = yellow-orange)
        # Native NumPy implementation (no skimage dependency)
        new_colors = self._hsv_to_rgb_vectorized(
            0.11 + hue_variation, saturation, brightness
        )
        
        # Adjust alpha based on distance (farther = more transparent)
        base_alphas = np.random.uniform(0.3, 0.6, count)
        new_alphas = base_alphas * (40.0 - new_distances) / 30.0  # Farther particles more transparent
        
        # Lifetime based on viewport height and fall speed
        new_lifetimes = np.ones(count) * 10.0  # Particles last ~10 seconds
        
        # Random turbulence phases
        new_turbulence_phases = np.random.uniform(0, 2 * np.pi, count)
        
        # squish_factor used to be precomputed per particle here. The
        # vertex shader now computes it from the radial component of the
        # particle's physical position, so this CPU step is gone.

        # Concatenate with existing arrays
        self.positions = np.vstack([self.positions, new_positions]) if len(self.positions) > 0 else new_positions
        self.velocities = np.vstack([self.velocities, new_velocities]) if len(self.velocities) > 0 else new_velocities
        self.sizes = np.concatenate([self.sizes, new_sizes]) if len(self.sizes) > 0 else new_sizes
        self.colors = np.vstack([self.colors, new_colors]) if len(self.colors) > 0 else new_colors
        self.alphas = np.concatenate([self.alphas, new_alphas]) if len(self.alphas) > 0 else new_alphas
        self.lifetimes = np.concatenate([self.lifetimes, new_lifetimes]) if len(self.lifetimes) > 0 else new_lifetimes
        self.distances = np.concatenate([self.distances, new_distances]) if len(self.distances) > 0 else new_distances
        self.turbulence_phases = np.concatenate([self.turbulence_phases, new_turbulence_phases]) if len(self.turbulence_phases) > 0 else new_turbulence_phases
    
    def _spawn_initial_particles(self, count: int):
        """Spawn initial particles distributed across the entire viewport for immediate visibility"""
        if count <= 0:
            return
        
        # Spawn at random PHYSICAL positions across the visible fan area
        new_positions = np.column_stack([
            np.random.uniform(-self._fan_x_half, self._fan_x_half, count),
            np.random.uniform(0, self._fan_y_max, count),
        ])

        # Moderate initial wind, in FEET PER SECOND.
        wind_speed_ft_s = 0.5 * 12.0
        new_velocities = np.column_stack([
            np.random.uniform(wind_speed_ft_s * 0.8, wind_speed_ft_s * 1.2, count),
            np.random.uniform(-0.15, 0.15, count),
        ])
        
        # Random distances (depth) between 10 and 40
        new_distances = np.random.uniform(10.0, 40.0, count)
        
        # Size scaled by distance
        base_sizes = np.random.uniform(self.particle_size * 0.5, self.particle_size * 1.5, count)
        new_sizes = base_sizes * (10.0 / new_distances)
        
        # Sand colors
        hue_variation = np.random.uniform(-0.05, 0.05, count)
        saturation = np.random.uniform(0.3, 0.5, count)
        brightness = np.random.uniform(0.6, 0.9, count)
        
        # Native NumPy implementation (no skimage dependency)
        new_colors = self._hsv_to_rgb_vectorized(
            0.11 + hue_variation, saturation, brightness
        )
        
        # Adjust alpha based on distance
        base_alphas = np.random.uniform(0.4, 0.7, count)
        new_alphas = base_alphas * (40.0 - new_distances) / 30.0
        
        # Lifetime
        new_lifetimes = np.ones(count) * 10.0
        
        # Random turbulence phases
        new_turbulence_phases = np.random.uniform(0, 2 * np.pi, count)
        
        # squish_factor is computed in the vertex shader now (no CPU step).

        # Initialize arrays
        self.positions = new_positions
        self.velocities = new_velocities
        self.sizes = new_sizes
        self.colors = new_colors
        self.alphas = new_alphas
        self.lifetimes = new_lifetimes
        self.distances = new_distances
        self.turbulence_phases = new_turbulence_phases
    
    def update(self, dt: float, state: Dict):
        """Update particle positions and properties"""
        if not self.enabled:
            return
        
        self.time += dt
        
        # Spawn new particles based on sand_density
        if len(self.positions) < self.max_particles:
            # More aggressive spawn rate - sand_density=1.0 spawns ~15 particles per frame
            spawn_rate = self.sand_density * 15.0 * 60.0  # Scale density to particles per second
            particles_to_spawn = int(spawn_rate * dt)  # Convert to particles this frame
            particles_to_spawn = min(particles_to_spawn, self.max_particles - len(self.positions))
            if particles_to_spawn > 0:
                self._spawn_particles(particles_to_spawn)
        
        if len(self.positions) == 0:
            return
        
        # Velocities are FIXED for the lifetime of each particle — set once
        # at spawn (with 0.8x..1.2x scatter around the wind baseline) and
        # never modified. Constant velocity → exact straight-line motion in
        # physical fan space.
        self.positions += self.velocities * dt

        # Update lifetimes
        self.lifetimes -= dt

        # ----- Wrap particles that exit the fan annulus -----
        # Instead of culling at the downwind edge (which left a sand
        # gradient where particles spawned mid-screen and drifted off),
        # wrap the position to the OPPOSITE side at the same physical y.
        # The teleport direction follows the velocity sign so the particle
        # keeps moving the way the wind blows it: a particle that exits
        # the right edge re-enters at the left, etc.
        inner_r2 = self._fan.inner_r_ft ** 2
        outer_r2 = self._fan.outer_r_ft ** 2
        r2 = self.positions[:, 0] ** 2 + self.positions[:, 1] ** 2

        # Wrap across outer ring: particle exited at the fan rim → reappear
        # at the rim on the other side at the same y.
        out_of_outer = r2 > outer_r2
        if np.any(out_of_outer):
            ys = self.positions[out_of_outer, 1]
            max_x_at_y = np.sqrt(np.maximum(0.0, outer_r2 - ys * ys))
            vx_vals = self.velocities[out_of_outer, 0]
            # Moving +x → teleport to -max_x_at_y; moving -x → +max_x_at_y.
            self.positions[out_of_outer, 0] = np.where(
                vx_vals > 0, -max_x_at_y, max_x_at_y,
            )

        # Teleport across inner-ring cutout: particle drifted into the
        # central hole → pop out the other side along its current velocity.
        # Recompute r² because we just moved some particles.
        r2 = self.positions[:, 0] ** 2 + self.positions[:, 1] ** 2
        in_cutout = r2 < inner_r2
        if np.any(in_cutout):
            ys = self.positions[in_cutout, 1]
            cut_x = np.sqrt(np.maximum(0.0, inner_r2 - ys * ys))
            vx_vals = self.velocities[in_cutout, 0]
            # Moving +x → exit on the +x side of the cutout, etc. Add a
            # tiny epsilon so we land just OUTSIDE the cutout.
            self.positions[in_cutout, 0] = np.where(
                vx_vals > 0, cut_x + 0.05, -cut_x - 0.05,
            )

        # Cull only on lifetime expiry and on y going outside the fan.
        # (Vertical drift is ±0.15 ft/s so over a 10 s lifetime particles
        # only drift ~1.5 ft vertically — y-cull rarely fires but is a
        # safety net for transitioning states.)
        valid_mask = (
            (self.positions[:, 1] > self._fan_y_min - 0.5) &
            (self.positions[:, 1] < self._fan_y_max + 1.5) &
            (self.lifetimes > 0)
        )
        
        self.positions = self.positions[valid_mask]
        self.velocities = self.velocities[valid_mask]
        self.sizes = self.sizes[valid_mask]
        self.colors = self.colors[valid_mask]
        self.alphas = self.alphas[valid_mask]
        self.lifetimes = self.lifetimes[valid_mask]
        self.distances = self.distances[valid_mask]
        self.turbulence_phases = self.turbulence_phases[valid_mask]
        # Squish factor recomputation is gone — the vertex shader derives
        # it from each particle's `offset.{x,y}` directly each frame.
    
    def render(self, state: Dict):
        """Render sand particles with horizontal wrapping"""
        if not self.enabled:
            return
        if len(self.positions) == 0:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)

        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)

        # Fan-geometry uniforms. The vertex shader uses these to do the
        # polar→buffer-pixel conversion in-place (used to be on CPU via
        # FanCoords.physical_to_uv_np). Imported lazily to avoid a hard
        # dependency cycle when this module is reused.
        from renderer.fan_geometry import FanGeometry
        glUniform1f(glGetUniformLocation(self.shader, "u_inner_r_ft"),
                    FanGeometry.PHYSICAL_INNER_FT)
        glUniform1f(glGetUniformLocation(self.shader, "u_outer_r_ft"),
                    FanGeometry.PHYSICAL_OUTER_FT)
        glUniform1f(glGetUniformLocation(self.shader, "u_squish_top"),
                    float(self.squish_top_width))

        # Drop particles that have wandered into the inner-ring cutout
        # (where r < inner_r_ft) or below the visible diameter line. Cheap
        # squared-distance test, no sqrt needed.
        r2 = self.positions[:, 0] ** 2 + self.positions[:, 1] ** 2
        in_fan = (r2 >= self._fan.inner_r_ft ** 2) & (self.positions[:, 1] >= 0.0)
        if not np.any(in_fan):
            glBindVertexArray(0)
            glUseProgram(0)
            return

        # Build instance data — layout: [x_ft, y_ft, size, r, g, b, a, distance]
        # 8 floats per particle. Positions stay in PHYSICAL FEET; the
        # vertex shader converts to clip space.
        instance_data = np.column_stack([
            self.positions[in_fan],
            self.sizes[in_fan],
            self.colors[in_fan],
            self.alphas[in_fan],
            self.distances[in_fan],
        ]).astype(np.float32)
        
        # Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Draw instanced particles
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(instance_data))
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;

        layout(location = 0) in vec2 position;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec2 offset;    // Particle position in PHYSICAL FEET (x, y)
        layout(location = 2) in float size;     // Particle size hint
        layout(location = 3) in vec4 color;     // Color (r, g, b, alpha)
        layout(location = 4) in float distance; // Depth value (10-40)

        out vec4 fragColor;
        out vec2 fragPos;
        uniform vec2 resolution;
        uniform float fadeAlpha;
        // Fan geometry uniforms — particle is in PHYSICAL coordinates;
        // the vertex shader does the polar→buffer-pixel conversion that
        // used to live on the CPU as FanCoords.physical_to_uv_np().
        uniform float u_inner_r_ft;
        uniform float u_outer_r_ft;
        uniform float u_squish_top;   // multiplier for outer-ring particles

        const float FAN_PI = 3.14159265358979;

        void main() {
            fragPos = position;

            // ---- Convert PHYSICAL position (feet) → BUFFER pixels ----
            // Mirrors FanCoords.physical_to_uv: theta is the polar angle,
            // u maps π-left→0-right, v maps inner-ring (0)→outer-ring (1).
            float r_ft   = length(offset);
            float theta  = atan(offset.y, offset.x);   // GLSL atan(y, x) = atan2
            float u      = clamp(1.0 - theta / FAN_PI, 0.0, 1.0);
            float v_part = clamp((r_ft - u_inner_r_ft) /
                                 (u_outer_r_ft - u_inner_r_ft), 0.0, 1.0);
            vec2 offset_px = vec2(u * resolution.x, v_part * resolution.y);

            // ---- Per-particle anisotropic scale (circular in fan space) ----
            // Use the actual radial distance r_ft (not the v-derived approx)
            // so partially-clamped positions still get the right column scale.
            float dx_ft_per_col = max(r_ft, u_inner_r_ft) * FAN_PI / resolution.x;
            float dy_ft_per_row = (u_outer_r_ft - u_inner_r_ft) / resolution.y;

            // Particle radius in PHYSICAL feet. `size=2` (default wrapper
            // hint) → ~0.20 ft radius — visible everywhere on the fan.
            float size_ft = size * 0.10;
            float w_px = size_ft / max(dx_ft_per_col, 1e-4);
            float h_px = size_ft / max(dy_ft_per_row, 1e-4);

            // FLOOR the per-axis pixel size to ~1.5 buffer pixels so the
            // particle's quad always covers more than a single fragment.
            // Without this, the outer-ring particles end up sub-pixel-
            // wide (because cols at the outer ring span 0.51 ft each, so
            // a 0.20 ft particle is ~0.4 cols), the smoothstep edge falls
            // entirely on at most one fragment, and as the particle moves
            // across pixel centres it alternates which LED is lit — that
            // shows up as the particles appearing to "blink".
            const float MIN_PX = 1.5;
            w_px = max(MIN_PX, w_px);
            h_px = max(MIN_PX, h_px);

            // ---- Squish from radial position (was CPU-computed) ----
            float squish = 1.0 + (u_squish_top - 1.0) * v_part;

            vec2 scaled = vec2(position.x * w_px * squish,
                               position.y * h_px);

            vec2 pos = scaled + offset_px;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            // No y-flip here. `offset_px.y` is now derived from the
            // physical_to_uv `v` axis where v=0 is the inner ring (bottom
            // of FBO in OpenGL texture-coord convention) and v=1 is the
            // outer ring (top). That already matches clip-space y-up;
            // adding a flip would mirror particles radially and produce
            // dumbbell-shaped trajectories instead of straight lines.

            float depth = clamp(distance / 100.0, 0.0, 1.0);
            gl_Position = vec4(clipPos, depth, 1.0);

            fragColor = vec4(color.rgb, color.a * fadeAlpha);
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
            // Create circular particles with soft edges
            float dist = length(fragPos);
            float alpha = smoothstep(1.0, 0.6, dist);
            
            if (alpha < 0.01) {
                discard;
            }
            
            // Add some texture variation
            float noise = fract(sin(dot(fragPos * 10.0, vec2(12.9898, 78.233))) * 43758.5453);
            vec3 color = fragColor.rgb * (0.9 + noise * 0.2);
            
            outColor = vec4(color, fragColor.a * alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link sandstorm shaders - REQUIRED by ShaderEffect base class"""
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
        
        # Vertex buffer (quad)
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
        
        # Instance buffer (particles)
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        self.VBOs.append(self.instance_VBO)
        
        # Instance attributes (8 floats / 32 bytes per particle):
        #   offset(vec2 ft) + size(float) + color(vec4) + distance(float)
        # Squish is no longer per-instance — the vertex shader computes
        # it from the radial position derived from `offset`.
        stride = 4 * 8

        # offset (location 1) — particle position in PHYSICAL FEET
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)

        # size (location 2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)

        # color (location 3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)

        # distance (location 4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)

        glBindVertexArray(0)
