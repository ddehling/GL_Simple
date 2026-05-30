"""
Lightning shader effect - Bolts striking from the top of the screen
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import time
from typing import Dict
from .base import ShaderEffect
from pathlib import Path

class LightningEffect(ShaderEffect):
    """
    Lightning bolts that strike down from the top of the screen.
    Multiple bolts can exist at different depths with varying intensities.
    """
    
    def __init__(self, viewport, bolt_interval=2.0, bolt_duration=0.3,
                 num_segments=15, jaggedness=30.0, max_bolts=5,
                 branch_probability=0.4, max_branch_depth=2, branch_length_ratio=0.5,
                 fan_aware=False):
        """
        Args:
            viewport: The viewport to render to
            bolt_interval: Time between lightning strikes (seconds)
            bolt_duration: How long each bolt lasts (seconds)
            num_segments: Number of line segments per bolt
            jaggedness: How much the bolt zigzags
            max_bolts: Maximum simultaneous bolts
            branch_probability: Chance of branching at each segment (0.0-1.0)
            max_branch_depth: Maximum recursion depth for branches
            branch_length_ratio: How long branches are relative to parent (0.0-1.0)
            fan_aware: If True, generate bolts in PHYSICAL FEET via FanCoords
                so the bolt path is a true vertical strike on the polar fan
                display (top of fan = sky, inner ring = horizon). When False
                (default) the bolt is generated directly in buffer pixel
                space — fine for rectangular viewports but on the fan a
                "vertical" pixel path reads as a RADIAL spoke from the
                inner ring outward. Pass True for any fan-realm caller.
        """
        super().__init__(viewport)
        self.bolt_interval = bolt_interval
        self.bolt_duration = bolt_duration
        self.num_segments = num_segments
        self.jaggedness = jaggedness
        self.max_bolts = max_bolts
        self.branch_probability = branch_probability
        self.max_branch_depth = max_branch_depth
        self.branch_length_ratio = branch_length_ratio
        self.wrap_margin = 100  # For horizontal wrapping
        # Fan-aware mode: generate geometry in physical fan FEET and
        # convert to pixel coords via FanCoords. Imported lazily so this
        # generic shader still works on projects that don't have fan
        # support in renderer/fan_coords.
        self.fan_aware = bool(fan_aware)
        self._fan = None
        if self.fan_aware:
            try:
                from renderer.fan_coords import FanCoords
                self._fan = FanCoords(viewport.width, viewport.height)
            except Exception as e:
                print(f"[Lightning] fan_aware requested but FanCoords unavailable: {e}")
                self.fan_aware = False
        # Physical-feet bounds for fan_aware bolt generation. The fan
        # spans roughly x ∈ [-20.6, +20.6] ft, y ∈ [0, 20.6] ft with an
        # inner ring at ~4 ft. Bolts strike from the cloud band (~19 ft)
        # down to just above the horizon (~5 ft) so they don't drop into
        # the inner-ring hole at the bottom of the fan.
        self._fan_x_range_ft = 13.0   # ±13 ft anchor range
        self._fan_top_ft     = 19.0
        self._fan_bottom_ft  =  5.0
        self._fan_jitter_ft  = 1.2    # ±1.2 ft per-segment horizontal jitter
        
        # VBO handles
        self.vbo_positions = None
        self.vbo_offsets = None
        self.vbo_brightness = None
        
        # Active bolts storage
        self.bolts = []  # List of dicts with bolt data
        self.last_spawn_time = time.time()
        self.first_bolt_spawned = False  # Track if initial bolt has been spawned
        
        # NOTE: Do NOT call setup_buffers() here!
        # It will be called automatically by init() after shader compilation
    
    def compile_shader(self):
        """Compile and link shaders - REQUIRED METHOD"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Lightning shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers - Called automatically after shader compilation"""
        # Create VAO for bolt lines
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # Create VBOs
        self.vbo_positions = glGenBuffers(1)
        self.vbo_offsets = glGenBuffers(1)
        self.vbo_brightness = glGenBuffers(1)

        # Position attribute (vertex positions - relative to bolt)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        # Offset attribute (x, y, z position per vertex)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_offsets)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Brightness attribute (per vertex for fade effect)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_brightness)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        # Fullscreen-flash VAO: a single oversize triangle covering clip-space.
        # Rendered first each strike to brighten the whole sky (the signature
        # atmospheric wash that real lightning produces).
        self.flash_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.flash_VAO)
        self.flash_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.flash_VBO)
        flash_verts = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, flash_verts.nbytes, flash_verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # Simple shader dedicated to the fullscreen flash pass.
        self.flash_shader = self._compile_flash_shader()

    def _compile_flash_shader(self):
        vs = """
#version 310 es
precision highp float;
layout(location = 0) in vec2 position;
void main() { gl_Position = vec4(position, 0.999, 1.0); }
"""
        fs = """
#version 310 es
precision highp float;
uniform float u_intensity;
uniform vec3 u_flash_color;
out vec4 outColor;
void main() {
    outColor = vec4(u_flash_color * u_intensity, 1.0);
}
"""
        return shaders.compileProgram(
            shaders.compileShader(vs, GL_VERTEX_SHADER),
            shaders.compileShader(fs, GL_FRAGMENT_SHADER),
        )

    @staticmethod
    def _flicker_envelope(t):
        """Multi-peak lightning envelope, t in [0, 1].

        Three gaussian-ish peaks: the initial strike, a bright re-strike ~35%
        into the event, and a weaker echo near the end. This is what gives
        the effect its "flash, pause, flash again" signature rather than a
        single dull fade.
        """
        if t < 0.0 or t >= 1.0:
            return 0.0
        peaks = ((0.00, 1.00, 0.07),
                 (0.38, 0.80, 0.06),
                 (0.72, 0.45, 0.09))
        val = 0.0
        for center, amp, width in peaks:
            d = (t - center) / width
            v = amp * float(np.exp(-d * d))
            if v > val:
                val = v
        return val
    
    def generate_branch(self, start_point, direction, length_ratio, depth):
        """
        Recursively generate a lightning branch
        
        Args:
            start_point: [x, y] starting position
            direction: 1 for down/right, -1 for down/left
            length_ratio: How long this branch is relative to full bolt
            depth: Current recursion depth
        
        Returns:
            List of [x, y] points making up the branch
        """
        if depth > self.max_branch_depth:
            return []
        
        points = [start_point.copy()]
        x, y = start_point
        
        # Calculate branch parameters
        num_segments = max(3, int(self.num_segments * length_ratio))
        segment_height = (self.viewport.height * length_ratio) / num_segments
        jaggedness = self.jaggedness * length_ratio
        
        # Branch grows down and to the side
        for i in range(num_segments):
            # Move down and sideways
            x += np.random.uniform(-jaggedness, jaggedness) + (direction * jaggedness * 0.5)
            y += segment_height
            points.append([x, y])
            
            # Chance to create sub-branches (decreases with depth)
            branch_chance = self.branch_probability * (0.5 ** depth)
            if np.random.random() < branch_chance:
                # Create a smaller branch
                sub_branch = self.generate_branch(
                    np.array([x, y]),
                    np.random.choice([-1, 1]),
                    length_ratio * self.branch_length_ratio,
                    depth + 1
                )
                points.extend(sub_branch)
        
        return points
    
    def generate_bolt_path(self, start_x, start_y):
        """
        Generate a jagged lightning bolt path with branches

        Returns:
            Dict with 'main' path and list of 'branches'
        """
        # Generate main bolt path
        main_points = []
        x = 0.0  # Relative to bolt position
        y = 0.0

        segment_height = self.viewport.height / self.num_segments

        for i in range(self.num_segments + 1):
            main_points.append([x, y])

            if i < self.num_segments:
                # Add random horizontal offset for zigzag
                x += np.random.uniform(-self.jaggedness, self.jaggedness)
                y += segment_height

        # Generate branches from main bolt
        branches = []
        for i in range(1, len(main_points) - 1):  # Don't branch from endpoints
            # Check if we should branch here
            if np.random.random() < self.branch_probability:
                branch_start = np.array(main_points[i])
                direction = np.random.choice([-1, 1])  # Left or right

                # Calculate remaining length ratio
                remaining_ratio = 1.0 - (i / len(main_points))
                branch_length = remaining_ratio * self.branch_length_ratio

                # Generate branch
                branch_points = self.generate_branch(
                    branch_start,
                    direction,
                    branch_length,
                    depth=1
                )

                if len(branch_points) > 1:
                    branches.append(np.array(branch_points, dtype=np.float32))

        return {
            'main': np.array(main_points, dtype=np.float32),
            'branches': branches
        }

    # ------------------------------------------------------------------
    # Fan-aware variants — generate bolt geometry in PHYSICAL FAN FEET
    # then convert each point to absolute pixel coords via FanCoords.
    # Returned arrays are ABSOLUTE pixel positions, so callers in fan
    # mode set the bolt's `position` (offset attribute) to (0, 0, z)
    # rather than the bolt anchor pixel — the absolute coords already
    # carry the geometry.
    # ------------------------------------------------------------------

    def _generate_fan_branch(self, start_phys, direction, length_ratio, depth):
        """Recursive branch in physical feet. Mirrors generate_branch but
        works in feet so the branch follows real top-down geometry on
        the polar fan rather than radial spokes.

        start_phys: (phys_x_ft, phys_y_ft) — branch origin
        direction:  +1 (right) or -1 (left)
        length_ratio: fraction of the full sky band this branch covers
        """
        if depth > self.max_branch_depth:
            return []

        full_height_ft = self._fan_top_ft - self._fan_bottom_ft
        num_segments = max(3, int(self.num_segments * length_ratio))
        segment_height_ft = (full_height_ft * length_ratio) / num_segments
        jit_ft = self._fan_jitter_ft * length_ratio

        points = [tuple(start_phys)]
        x_ft, y_ft = float(start_phys[0]), float(start_phys[1])
        for _ in range(num_segments):
            # Step DOWN in phys_y (toward the horizon) with a sideways drift.
            x_ft += np.random.uniform(-jit_ft, jit_ft) + (direction * jit_ft * 0.5)
            y_ft -= segment_height_ft
            points.append((x_ft, y_ft))
            # Recursive sub-branches with diminishing probability.
            branch_chance = self.branch_probability * (0.5 ** depth)
            if np.random.random() < branch_chance:
                sub = self._generate_fan_branch(
                    (x_ft, y_ft),
                    int(np.random.choice([-1, 1])),
                    length_ratio * self.branch_length_ratio,
                    depth + 1,
                )
                points.extend(sub)
        return points

    def _generate_fan_bolt_path(self, anchor_phys_x_ft):
        """Generate the main bolt path + branches in physical feet, then
        convert every point to ABSOLUTE PIXEL coords for the renderer.
        """
        fan = self._fan
        full_height_ft = self._fan_top_ft - self._fan_bottom_ft
        segment_height_ft = full_height_ft / self.num_segments

        # Main bolt: start at the cloud band, zigzag DOWN to the horizon.
        main_phys = []
        x_ft = float(anchor_phys_x_ft)
        y_ft = self._fan_top_ft
        for i in range(self.num_segments + 1):
            main_phys.append((x_ft, y_ft))
            if i < self.num_segments:
                x_ft += np.random.uniform(-self._fan_jitter_ft, self._fan_jitter_ft)
                y_ft -= segment_height_ft

        # Branches: from each interior main-bolt vertex with branch_probability.
        branches_phys = []
        for i in range(1, len(main_phys) - 1):
            if np.random.random() < self.branch_probability:
                remaining_ratio = 1.0 - (i / float(len(main_phys)))
                branch_length = remaining_ratio * self.branch_length_ratio
                sub = self._generate_fan_branch(
                    main_phys[i],
                    int(np.random.choice([-1, 1])),
                    branch_length,
                    depth=1,
                )
                if len(sub) > 1:
                    branches_phys.append(sub)

        # Convert all (phys_x_ft, phys_y_ft) → absolute (pixel_x, pixel_y).
        def phys_to_px_array(pts):
            out = []
            for px, py in pts:
                ix, iy = fan.physical_to_px(float(px), float(py))
                out.append([float(ix), float(iy)])
            return np.array(out, dtype=np.float32)

        return {
            'main':     phys_to_px_array(main_phys),
            'branches': [phys_to_px_array(b) for b in branches_phys],
        }
        
    def spawn_bolt(self):
        """Create a new lightning bolt with branches"""
        if len(self.bolts) >= self.max_bolts:
            return

        # Random depth (some bolts in front, some behind)
        z = np.random.uniform(20, 80)  # Mid-range depths

        if self.fan_aware and self._fan is not None:
            # Anchor in PHYSICAL FEET; path is generated and returned as
            # ABSOLUTE pixel coords, so the per-vertex offset (the bolt
            # position uniform/attribute) is zeroed — the geometry already
            # carries the screen coords.
            anchor_phys_x = float(np.random.uniform(
                -self._fan_x_range_ft, self._fan_x_range_ft))
            path_data = self._generate_fan_bolt_path(anchor_phys_x)
            position = np.array([0.0, 0.0, z], dtype=np.float32)
        else:
            # Original buffer-pixel-space behaviour.
            x = np.random.uniform(0, self.viewport.width)
            y = 0  # Start from top
            path_data = self.generate_bolt_path(x, y)
            position = np.array([x, y, z], dtype=np.float32)

        bolt = {
            'main_path': path_data['main'],
            'branches': path_data['branches'],
            'position': position,
            'spawn_time': time.time(),
            'brightness': 1.0
        }

        self.bolts.append(bolt)
    
    def update_bolts(self):
        """Update bolt states and remove expired ones"""
        current_time = time.time()
        
        # Spawn first bolt immediately on first update
        if not self.first_bolt_spawned:
            self.spawn_bolt()
            self.first_bolt_spawned = True
            self.last_spawn_time = current_time
        
        # Remove expired bolts
        self.bolts = [
            bolt for bolt in self.bolts
            if (current_time - bolt['spawn_time']) < self.bolt_duration
        ]
        
        # Multi-peak flicker envelope: strike, re-strike, echo.
        for bolt in self.bolts:
            elapsed = current_time - bolt['spawn_time']
            bolt['brightness'] = self._flicker_envelope(elapsed / self.bolt_duration)
        
        # Spawn new bolt if interval has passed
        if current_time - self.last_spawn_time >= self.bolt_interval:
            self.spawn_bolt()
            self.last_spawn_time = current_time
    
    def add_path_to_buffers(self, path, position, brightness, all_vertices, all_offsets, all_brightness, branch_brightness_multiplier=1.0):
        """Helper to add a path (main or branch) to render buffers"""
        # Convert path to line segments (pairs of consecutive points)
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # Add both vertices of the line segment
            all_vertices.append(p1)
            all_vertices.append(p2)
            
            # Add offset and brightness for both vertices
            all_offsets.append(position)
            all_offsets.append(position)
            adjusted_brightness = brightness * branch_brightness_multiplier
            all_brightness.append(adjusted_brightness)
            all_brightness.append(adjusted_brightness)
    
    def build_render_data(self):
        """Build vertex data for all bolts including branches and wrapped duplicates"""
        all_vertices = []
        all_offsets = []
        all_brightness = []
        
        for bolt in self.bolts:
            main_path = bolt['main_path']
            branches = bolt['branches']
            position = bolt['position']
            brightness = bolt['brightness']
            
            # Add main bolt path
            self.add_path_to_buffers(main_path, position, brightness, 
                                    all_vertices, all_offsets, all_brightness)
            
            # Add all branches (slightly dimmer than main bolt)
            for branch in branches:
                self.add_path_to_buffers(branch, position, brightness,
                                        all_vertices, all_offsets, all_brightness,
                                        branch_brightness_multiplier=0.7)
            
            # Handle wrapping - duplicate near edges. Skipped in fan_aware
            # mode because the fan has no left/right wrap; the path is
            # already in absolute pixel coords and position is zeroed.
            if self.fan_aware:
                continue
            x = position[0]

            # Near left edge - duplicate on right
            if x < self.wrap_margin:
                wrapped_pos = position.copy()
                wrapped_pos[0] += self.viewport.width
                
                # Duplicate main path
                self.add_path_to_buffers(main_path, wrapped_pos, brightness,
                                        all_vertices, all_offsets, all_brightness)
                
                # Duplicate branches
                for branch in branches:
                    self.add_path_to_buffers(branch, wrapped_pos, brightness,
                                            all_vertices, all_offsets, all_brightness,
                                            branch_brightness_multiplier=0.7)
            
            # Near right edge - duplicate on left
            if x > (self.viewport.width - self.wrap_margin):
                wrapped_pos = position.copy()
                wrapped_pos[0] -= self.viewport.width
                
                # Duplicate main path
                self.add_path_to_buffers(main_path, wrapped_pos, brightness,
                                        all_vertices, all_offsets, all_brightness)
                
                # Duplicate branches
                for branch in branches:
                    self.add_path_to_buffers(branch, wrapped_pos, brightness,
                                            all_vertices, all_offsets, all_brightness,
                                            branch_brightness_multiplier=0.7)
        
        if not all_vertices:
            return None, None, None, 0
        
        vertices = np.array(all_vertices, dtype=np.float32)
        offsets = np.array(all_offsets, dtype=np.float32)
        brightness_data = np.array(all_brightness, dtype=np.float32)
        vertex_count = len(vertices)
        
        return vertices, offsets, brightness_data, vertex_count
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        self.update_bolts()

        # Publish the current strike intensity so other shaders (e.g. the
        # hurricane cloud mass) can react to the flash. 0 when no bolt is
        # active, peaks at 1 during the main strike.
        if self.bolts:
            state['lightning_flash'] = max(b['brightness'] for b in self.bolts)
        else:
            state['lightning_flash'] = 0.0
    
    def render(self, state):
        """Render all active lightning bolts"""
        if not self.enabled or self.shader is None:
            return

        if not self.bolts:
            return

        # Build render data
        vertices, offsets, brightness_data, vertex_count = self.build_render_data()

        if vertex_count == 0:
            return

        # Overall strike intensity for this frame (max of all active bolts).
        strike_intensity = max(b['brightness'] for b in self.bolts)

        # Switch to ADDITIVE blending so both the sky flash and the bolt
        # stack on top of whatever sky/cloud effects rendered before us.
        # Without this the bolt just alpha-blends over the sky and never
        # reads as a luminous discharge.
        glBlendFunc(GL_ONE, GL_ONE)

        # ---- Pass 1: fullscreen sky flash ----
        # The whole scene briefly gets washed in cool white-blue, which is
        # the single biggest "this looks like lightning" cue. Scaled well
        # below 1 so it brightens the sky without blowing out everything.
        if strike_intensity > 0.01:
            glUseProgram(self.flash_shader)
            glBindVertexArray(self.flash_VAO)
            loc = glGetUniformLocation(self.flash_shader, b"u_intensity")
            if loc >= 0:
                glUniform1f(loc, strike_intensity * 0.45)
            col_loc = glGetUniformLocation(self.flash_shader, b"u_flash_color")
            if col_loc >= 0:
                glUniform3f(col_loc, 0.55, 0.70, 1.00)
            glDrawArrays(GL_TRIANGLES, 0, 3)

        # ---- Pass 2: bolt halo + hot core ----
        glUseProgram(self.shader)

        res_loc = glGetUniformLocation(self.shader, b"resolution")
        if res_loc >= 0:
            glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_offsets)
        glBufferData(GL_ARRAY_BUFFER, offsets.nbytes, offsets, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_brightness)
        glBufferData(GL_ARRAY_BUFFER, brightness_data.nbytes, brightness_data, GL_DYNAMIC_DRAW)

        old_line_width = glGetFloatv(GL_LINE_WIDTH)
        glow_loc = glGetUniformLocation(self.shader, b"u_glow")

        # Wide cool halo underneath.
        if glow_loc >= 0:
            glUniform1f(glow_loc, 1.0)
        glLineWidth(6.0)
        glDrawArrays(GL_LINES, 0, vertex_count)

        # Thin hot white core on top.
        if glow_loc >= 0:
            glUniform1f(glow_loc, 0.0)
        glLineWidth(2.5)
        glDrawArrays(GL_LINES, 0, vertex_count)
        glLineWidth(old_line_width)

        glBindVertexArray(0)
        glUseProgram(0)

        # Restore the renderer-wide default alpha blending.
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def get_vertex_shader(self):
        return """
#version 310 es
precision highp float;

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 offset;
layout(location = 2) in float brightness;

uniform vec2 resolution;

out float v_brightness;

void main() {
    vec2 pos = position + offset.xy;
    vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
    clipPos.y = -clipPos.y;
    
    float depth = offset.z / 100.0;
    depth = clamp(depth, 0.0, 1.0);
    
    gl_Position = vec4(clipPos, depth, 1.0);
    v_brightness = brightness;
}
"""
    
    def get_fragment_shader(self):
        return """
#version 310 es
precision highp float;

in float v_brightness;
uniform float u_glow;
out vec4 outColor;

void main() {
    // Core is hot white pushed WAY past 1.0; additive blend + LED clamping
    // gives a saturated searing-white line. Halo pass (u_glow = 1.0) is a
    // bright cool blue wider stroke that reads as scattered atmospheric glow.
    vec3 coreColor = vec3(3.5, 3.5, 3.8);
    vec3 haloColor = vec3(0.55, 0.85, 1.60);
    vec3 boltColor = mix(coreColor, haloColor, u_glow);
    float gain = (u_glow > 0.5) ? 1.0 : 1.6;
    outColor = vec4(boltColor * v_brightness * gain, 1.0);
}
"""
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if hasattr(self, 'VAO') and self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if hasattr(self, 'vbo_positions') and self.vbo_positions:
            glDeleteBuffers(1, [self.vbo_positions])
        if hasattr(self, 'vbo_offsets') and self.vbo_offsets:
            glDeleteBuffers(1, [self.vbo_offsets])
        if hasattr(self, 'vbo_brightness') and self.vbo_brightness:
            glDeleteBuffers(1, [self.vbo_brightness])
        if hasattr(self, 'flash_VAO') and self.flash_VAO:
            glDeleteVertexArrays(1, [self.flash_VAO])
        if hasattr(self, 'flash_VBO') and self.flash_VBO:
            glDeleteBuffers(1, [self.flash_VBO])
        if hasattr(self, 'flash_shader') and self.flash_shader:
            glDeleteProgram(self.flash_shader)
        super().cleanup()


# Event wrapper function for EventScheduler
def shader_lightning(state, outstate, bolt_interval=2.0, bolt_duration=0.3,
                     num_segments=15, jaggedness=30.0, max_bolts=5,
                     branch_probability=0.4, max_branch_depth=2, branch_length_ratio=0.5,
                     fan_aware=False):
    """
    Lightning bolts shader effect with branching - compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_lightning, 
                               bolt_interval=2.0, 
                               branch_probability=0.5,
                               frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        bolt_interval: Time between lightning strikes (seconds)
        bolt_duration: How long each bolt lasts (seconds)
        num_segments: Number of line segments per bolt (more = more detail)
        jaggedness: How much the bolt zigzags horizontally
        max_bolts: Maximum number of simultaneous bolts
        branch_probability: Chance of branching at each segment (0.0-1.0, default 0.4)
        max_branch_depth: Maximum recursion depth for branches (default 2)
        branch_length_ratio: How long branches are relative to parent (0.0-1.0, default 0.5)
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
    
    if state['count'] == 0:
        print(f"Initializing lightning effect for frame {frame_id}")
        spath=np.random.choice(['Thunder Clap Loud.wav','loud-thunder-192165.mp3','thunder-307513.mp3','peals-of-thunder-191992.mp3'])



        # Resolve the boom sample relative to the active project's media
        # folder; falls back to legacy ``<repo>/media/sounds`` only if
        # the active project hasn't seeded ``media_root`` yet.
        media_root = outstate.get('media_root')
        if media_root:
            sound_path = Path(media_root) / 'sounds'
        else:
            sound_path = Path(__file__).parent.parent.parent / 'media' / 'sounds'
        boom_path = sound_path / spath
        outstate['soundengine'].schedule_event(boom_path, duration=10.0)
        try:
            effect = viewport.add_effect(
                LightningEffect,
                bolt_interval=bolt_interval,
                bolt_duration=bolt_duration,
                num_segments=num_segments,
                jaggedness=jaggedness,
                max_bolts=max_bolts,
                branch_probability=branch_probability,
                max_branch_depth=max_branch_depth,
                branch_length_ratio=branch_length_ratio,
                fan_aware=fan_aware,
            )
            state['effect'] = effect
            print(f"✓ Initialized shader lightning for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize lightning: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up lightning effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader lightning for frame {frame_id}")