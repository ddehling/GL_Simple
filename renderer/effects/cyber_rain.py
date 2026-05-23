"""
Cyber rain — neon-tinted vertical rain drops for the cyberpunk set.

Direct port of `rain.py`'s instanced 3D-particle approach (Pattern A) so
drops have real depth, sort back-to-front, and write to the depth buffer.
A Pattern B fullscreen-quad approach was tried first and rendered behind
other shaders (no depth) — the user saw nothing.

Adds:
  • per-state `cyber_rain_color` (RGB) overrides the default blue
  • per-drop brightness variation so neighboring drops aren't identical

Reads from frame state / outstate:
  state['rain']                 — rain intensity (rain_rate, published by
                                  lib/weather_state.py:get_state_output)
  outstate['cyber_rain_color']  — RGB tuple chosen by the active state
  outstate['wind']              — wind direction (slight slant on drops)
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


# Default if no state sets a color: bright neon cyan
DEFAULT_COLOR = (0.30, 0.95, 1.00)


def shader_cyber_rain(state, outstate, intensity=1.0, wind=0.0,
                       color=DEFAULT_COLOR):
    """Cyberpunk-tinted rain — color picked per-state via cyber_rain_color."""
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        num_drops = int(300 * intensity)
        try:
            effect = viewport.add_effect(
                CyberRainEffect,
                num_raindrops=num_drops,
                wind=wind,
                color=color,
            )
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_rain] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    # Pick up wind + color from outstate every frame
    eff.wind = float(outstate.get('wind', wind))
    c = outstate.get('cyber_rain_color', color)
    if c is not None and len(c) >= 3:
        eff.color = (float(c[0]), float(c[1]), float(c[2]))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


class CyberRainEffect(ShaderEffect):
    """GPU-instanced rain effect with state-driven color tint.

    Mirrors RainEffect (renderer/effects/rain.py) so it gets the same
    depth-correct motion and seamless wrap. The only meaningful difference
    is `self.color` (RGB tuple): each drop's RGB is set to
    `color * per_drop_brightness_variation` instead of the fixed blue
    distribution the original used.
    """

    def __init__(self, viewport, num_raindrops: int = 300, wind: float = 0.0,
                 color=DEFAULT_COLOR):
        super().__init__(viewport)
        # Render LATE so drops draw on top of city_skyline (6.0),
        # neon_signs (7.0), hologram_billboards (7.5), data_rain (8.0).
        # The narrative-variable layers (8.5+) still go in front.
        # Without this, drops render first (priority 0) and every Pattern
        # B shader's alpha overlay dims and obscures them.
        self.render_priority = 8.2
        self.num_raindrops = num_raindrops
        self.base_num_raindrops = num_raindrops
        self.target_raindrops = num_raindrops
        self.fade_multiplier = 1.0
        self.wind = wind
        self.color = color
        self.instance_VBO = None
        self.wrap_margin = 50

        self.positions = None
        self.velocities = None
        self.base_velocities = None
        self.dimensions = None
        self.alphas = None
        self.colors = None

        self._initialize_raindrops()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _drop_colors(self, n: int) -> np.ndarray:
        """RGB array of length n, biased around self.color with brightness
        variation per drop (0.7..1.15). Head/tail color shifting is done in
        the fragment shader."""
        r, g, b = self.color
        bright = np.random.uniform(0.7, 1.15, n).astype(np.float32)
        return np.column_stack([
            np.clip(bright * r, 0.0, 1.0),
            np.clip(bright * g, 0.0, 1.0),
            np.clip(bright * b, 0.0, 1.0),
        ])

    def _initialize_raindrops(self):
        n = self.num_raindrops

        self.positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n),
            np.random.uniform(-self.viewport.height, 0, n),
            np.random.uniform(0, 100, n),
        ])
        self.velocities = np.random.uniform(150, 400, n)   # px/sec
        self.base_velocities = self.velocities.copy()

        depth_factors = 1.0 - (self.positions[:, 2] / 100.0)
        base_widths = np.random.uniform(1.0, 2.0, n)
        base_lengths = np.random.uniform(12, 22, n)
        self.dimensions = np.column_stack([
            base_widths * (0.3 + 0.7 * depth_factors),
            base_lengths * (0.3 + 0.7 * depth_factors),
        ])
        # Brighter than the standard rain (0.25 + 0.65 * df) — cyber drops
        # need to punch through the busy alpha overlay stack. Range 0.55..1.0.
        self.alphas = 0.55 + 0.45 * depth_factors
        self.colors = self._drop_colors(n)

    def _reset_raindrops(self, mask):
        n_reset = int(np.sum(mask))
        if n_reset == 0:
            return

        current_count = len(self.positions)
        if current_count > self.target_raindrops:
            n_to_remove = min(n_reset, current_count - self.target_raindrops)
            reset_indices = np.where(mask)[0]
            remove_indices = reset_indices[:n_to_remove]
            keep_mask = np.ones(current_count, dtype=bool)
            keep_mask[remove_indices] = False
            self.positions = self.positions[keep_mask]
            self.velocities = self.velocities[keep_mask]
            self.base_velocities = self.base_velocities[keep_mask]
            self.dimensions = self.dimensions[keep_mask]
            self.alphas = self.alphas[keep_mask]
            self.colors = self.colors[keep_mask]
            if n_to_remove < n_reset:
                remaining = reset_indices[n_to_remove:]
                for i, old_idx in enumerate(remaining):
                    adj = int(np.sum(remove_indices < old_idx))
                    remaining[i] = old_idx - adj
                new_mask = np.zeros(len(self.positions), dtype=bool)
                new_mask[remaining] = True
                mask = new_mask
                n_reset = n_reset - n_to_remove
            else:
                return

        if n_reset > 0:
            self.positions[mask, 0] = np.random.uniform(0, self.viewport.width, n_reset)
            self.positions[mask, 1] = -10
            self.positions[mask, 2] = np.random.uniform(0, 100, n_reset)
            self.velocities[mask] = np.random.uniform(150, 400, n_reset)
            self.base_velocities[mask] = self.velocities[mask]
            depth_factors = 1.0 - (self.positions[mask, 2] / 100.0)
            base_widths = np.random.uniform(1.0, 2.0, n_reset)
            base_lengths = np.random.uniform(12, 22, n_reset)
            self.dimensions[mask, 0] = base_widths * (0.3 + 0.7 * depth_factors)
            self.dimensions[mask, 1] = base_lengths * (0.3 + 0.7 * depth_factors)
            self.alphas[mask] = 0.55 + 0.45 * depth_factors
            self.colors[mask] = self._drop_colors(n_reset)

    def _add_raindrops(self, n_new):
        if n_new <= 0:
            return
        new_positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n_new),
            np.random.uniform(-self.viewport.height, -10, n_new),
            np.random.uniform(0, 100, n_new),
        ])
        new_velocities = np.random.uniform(150, 400, n_new)
        depth_factors = 1.0 - (new_positions[:, 2] / 100.0)
        base_widths = np.random.uniform(1.0, 2.0, n_new)
        base_lengths = np.random.uniform(12, 22, n_new)
        new_dimensions = np.column_stack([
            base_widths * (0.3 + 0.7 * depth_factors),
            base_lengths * (0.3 + 0.7 * depth_factors),
        ])
        new_alphas = 0.55 + 0.45 * depth_factors
        new_colors = self._drop_colors(n_new)

        self.positions = np.vstack([self.positions, new_positions])
        self.velocities = np.concatenate([self.velocities, new_velocities])
        self.base_velocities = np.concatenate([self.base_velocities, new_velocities])
        self.dimensions = np.vstack([self.dimensions, new_dimensions])
        self.alphas = np.concatenate([self.alphas, new_alphas])
        self.colors = np.vstack([self.colors, new_colors])

    # ── Shaders ───────────────────────────────────────────────────────────

    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;

        layout(location = 0) in vec2 position;   // centered -0.5..0.5
        layout(location = 1) in vec3 offset;     // x, y, z
        layout(location = 2) in vec2 size;
        layout(location = 3) in vec4 color;
        layout(location = 4) in float rotation;

        out vec4 fragColor;
        out vec2 vertPos;
        uniform vec2 resolution;

        void main() {
            vertPos = position + 0.5;
            vec2 scaled = position * size;
            float cosR = cos(rotation);
            float sinR = sin(rotation);
            vec2 rotated = vec2(
                scaled.x * cosR - scaled.y * sinR,
                scaled.x * sinR + scaled.y * cosR
            );
            vec2 pos = rotated + offset.xy;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            float depth = offset.z / 100.0;
            gl_Position = vec4(clipPos, depth, 1.0);
            fragColor = color;
        }
        """

    def get_fragment_shader(self):
        # Cyber-rain fragment: bright "head" at the bottom of the drop
        # quad (vertPos.y = 0), fading toward the top. Adds a slight
        # white-hot push at the head so neon drops have a strong leading
        # point even at low alpha.
        return """
        #version 310 es
        precision highp float;

        in vec4 fragColor;
        in vec2 vertPos;
        out vec4 outColor;

        void main() {
            // vertPos.y: 0 at top of drop, 1 at bottom.
            // We want HEAD bright (bottom = leading), TAIL faded (top).
            float head_t = vertPos.y;                 // 1 at head, 0 at tail
            float alpha = fragColor.a * head_t;

            if (alpha < 0.01) discard;

            // Push head toward white-hot — neon drops should have a
            // visible bright leading point.
            float head_boost = smoothstep(0.6, 1.0, head_t);
            vec3 col = mix(fragColor.rgb, vec3(1.0), head_boost * 0.45);

            // Slight brightness boost at low alpha (compensates for
            // alpha-blending darkening of saturated colors).
            float lift = mix(1.0, 1.4, 1.0 - head_t);
            col *= lift;

            outColor = vec4(col, alpha);
        }
        """

    def compile_shader(self):
        vert = shaders.compileShader(self.get_vertex_shader(), GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.get_fragment_shader(), GL_FRAGMENT_SHADER)
        prog = shaders.compileProgram(vert, frag)
        glUseProgram(prog)
        loc = glGetUniformLocation(prog, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        glUseProgram(0)
        return prog

    def setup_buffers(self):
        vertices = np.array([
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        glBindVertexArray(0)

    # ── Update / Render ───────────────────────────────────────────────────

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return

        # `state` here is the global frame state dict (renderer passes it).
        # rain intensity is published as state['rain'] (= rain_rate) by
        # lib/weather_state.py:get_state_output.
        rain_intensity = float(state.get('rain', 0.0))

        # Target drop count scales with intensity + fade factor
        self.target_raindrops = int(self.base_num_raindrops
                                     * rain_intensity * self.fade_multiplier)

        # Gradually grow toward target (max 5 new per frame)
        current = len(self.positions)
        if self.target_raindrops > current:
            n_to_add = min(self.target_raindrops - current, 5)
            self._add_raindrops(n_to_add)

        # Per-drop velocity varies only across a narrow 75–100% band so the
        # dominant cue for rain intensity is the *number* of drops (spawn
        # rate, controlled by target_raindrops above), not their speed.
        # Light rain reads as sparse-and-slightly-slower, heavy rain as
        # dense-and-full-speed.
        eff_factor = 0.75 + rain_intensity * 0.25
        self.velocities = self.base_velocities * eff_factor

        # Horizontal velocity from wind (px/sec)
        horiz_vel = self.wind * 60.0

        # Integrate position
        self.positions[:, 1] += self.velocities * dt
        self.positions[:, 0] += horiz_vel * dt

        # Wrap horizontally
        left_mask = self.positions[:, 0] < 0
        right_mask = self.positions[:, 0] >= self.viewport.width
        self.positions[left_mask, 0] += self.viewport.width
        self.positions[right_mask, 0] -= self.viewport.width

        # Reset drops that went off bottom
        bottom_mask = self.positions[:, 1] > self.viewport.height + 10
        self._reset_raindrops(bottom_mask)

        self.num_raindrops = len(self.positions)

    def render(self, state: Dict):
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return

        # Seam-safe duplicates near left/right edges
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)

        dup_positions = []
        dup_indices = []
        if np.any(left_edge_mask):
            idx = np.where(left_edge_mask)[0]
            d = self.positions[idx].copy()
            d[:, 0] += self.viewport.width
            dup_positions.append(d); dup_indices.append(idx)
        if np.any(right_edge_mask):
            idx = np.where(right_edge_mask)[0]
            d = self.positions[idx].copy()
            d[:, 0] -= self.viewport.width
            dup_positions.append(d); dup_indices.append(idx)

        all_positions = [self.positions] + dup_positions
        all_indices = [np.arange(len(self.positions))] + dup_indices

        combined_positions = np.vstack(all_positions)
        combined_indices = np.concatenate(all_indices)

        combined_dimensions = self.dimensions[combined_indices]
        combined_alphas = self.alphas[combined_indices]
        combined_colors = self.colors[combined_indices]

        # Back-to-front sort for proper alpha
        sort_indices = np.argsort(-combined_positions[:, 2])

        glUseProgram(self.shader)
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))

        # Rotation: align drop quad with motion vector
        horiz_vel = self.wind * 60.0
        vert_vel = self.velocities[combined_indices]
        velocity_angle = np.arctan2(horiz_vel, vert_vel)
        rotations = (np.pi - velocity_angle).astype(np.float32)

        instance_data = np.hstack([
            combined_positions[sort_indices],                   # 3 floats
            combined_dimensions[sort_indices],                  # 2 floats
            combined_colors[sort_indices],                      # 3 floats
            combined_alphas[sort_indices, np.newaxis],          # 1 float
            rotations[sort_indices, np.newaxis],                # 1 float
        ]).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)

        glBindVertexArray(self.VAO)
        stride = 10 * 4

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribDivisor(1, 1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2); glVertexAttribDivisor(2, 1)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3); glVertexAttribDivisor(3, 1)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(4); glVertexAttribDivisor(4, 1)

        total_drops = len(combined_positions)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, total_drops)

        glBindVertexArray(0)
        glUseProgram(0)
