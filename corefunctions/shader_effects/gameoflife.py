"""
Conway's Game of Life shader effect

Implements the classic cellular automaton with:
- Standard Conway's rules (B3/S23)
- Horizontal wrapping for seamless LED display
- Configurable cell size and update rate
- Color coding based on cell age
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_gameoflife(state, outstate, cell_size=3, update_rate=10, 
                      initial_density=0.4, depth=20.0, 
                      audio_reactive=True, bass_spawn_sensitivity=2.0):
    """
    Conway's Game of Life shader effect compatible with EventScheduler
    
    Audio Reactivity:
    - Bass hits spawn new cells (creates bursts of life)
    - Mid frequencies modulate color intensity (adds energy to visuals)
    - High frequencies speed up evolution (accelerates patterns on cymbals)
    
    Usage:
        scheduler.schedule_event(0, 60, shader_gameoflife, 
                                cell_size=8, update_rate=3, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        cell_size: Size of each cell in pixels (default: 5)
        update_rate: Frames between updates (default: 5)
        initial_density: Initial probability of live cells (default: 0.4)
        depth: Z-depth for rendering (default: 20.0, lower=closer)
        audio_reactive: Enable audio reactivity (default: True)
        bass_spawn_sensitivity: How sensitive bass spawning is (default: 2.0)
    """
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
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing Audio-Reactive Game of Life for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                GameOfLifeEffect,
                cell_size=cell_size,
                update_rate=update_rate,
                initial_density=initial_density,
                depth=depth
            )
            state['effect'] = effect
            state['bass_spawn_sensitivity'] = bass_spawn_sensitivity
            state['audio_reactive'] = audio_reactive
            print(f"✓ Initialized shader gameoflife for frame {frame_id}")
            print(f"  Grid: {effect.grid_width}x{effect.grid_height} cells")
            print(f"  Cell size: {cell_size}px, Update rate: {update_rate} frames")
        except Exception as e:
            print(f"✗ Failed to initialize gameoflife: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect from audio data
    if 'effect' in state and state.get('audio_reactive', False) and audio_data is not None:
        # Get current normalized bands (use short-term for beat response)
        bands = audio_data['norm_short'][0]
        
        # Bass energy (40-300 Hz) - triggers cell spawning
        bass_energy = np.mean(bands[0:8])
        
        # Mid energy (300-2000 Hz) - modulates color intensity
        mid_energy = np.mean(bands[8:20])
        
        # High energy (2000-16000 Hz) - speeds up evolution
        high_energy = np.mean(bands[20:32])
        
        # Update effect parameters
        state['effect'].audio_bass = bass_energy * state['bass_spawn_sensitivity']
        state['effect'].audio_mid = mid_energy
        state['effect'].audio_high = high_energy
        
        # Initialize previous values
        if 'prev_bass' not in state:
            state['prev_bass'] = bass_energy
        if 'prev_mid' not in state:
            state['prev_mid'] = mid_energy
        if 'prev_high' not in state:
            state['prev_high'] = high_energy
        
        # Spawn cells on strong bass transients
        bass_delta = bass_energy - state['prev_bass']
        if bass_delta > 0.3:  # Bass hit threshold
            state['effect'].trigger_bass_spawn()
        
        # Roll grid on high frequency hits
        high_delta = high_energy - state['prev_high']
        
        if abs(high_delta) > 0.1:  # Use absolute value, trigger on any significant change
            state['effect'].trigger_x_roll()
        
        state['prev_bass'] = bass_energy
        state['prev_mid'] = mid_energy
        state['prev_high'] = high_energy
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up Game of Life for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader gameoflife for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class GameOfLifeEffect(ShaderEffect):
    """Conway's Game of Life cellular automaton"""
    
    def __init__(self, viewport, cell_size: int = 10, update_rate: int = 5,
                 initial_density: float = 0.4, depth: float = 20.0):
        super().__init__(viewport)
        self.cell_size = cell_size
        self.base_update_rate = update_rate  # Base frames between updates
        self.update_rate = update_rate
        self.initial_density = initial_density
        self.depth = depth
        self.frame_counter = 0
        
        # Audio reactivity parameters
        self.audio_bass = 0.0
        self.audio_mid = 0.0
        self.audio_high = 0.0
        self.bass_spawn_pending = False
        self.x_roll_pending = False
        self.x_roll_amount = 0  # Cells to roll left
        
        # Horizontal wrapping margin (should be larger than cell_size)
        self.wrap_margin = max(50, cell_size * 2)
        
        # Calculate grid dimensions
        self.grid_width = (viewport.width + cell_size - 1) // cell_size
        self.grid_height = (viewport.height + cell_size - 1) // cell_size
        
        # OpenGL buffer handles
        self.instance_VBO = None
        self.quad_VBO = None
        self.EBO = None
        
        # Initialize game state
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize Game of Life grid with random state"""
        # Game state: 0 = dead, >0 = alive (value = age)
        self.grid = np.random.random((self.grid_height, self.grid_width)) < self.initial_density
        self.grid = self.grid.astype(np.int32)
        
        # Cell ages for color coding - initialize live cells to age 1
        self.cell_ages = self.grid.astype(np.float32)
        
        # Working buffer for next generation
        self.next_grid = np.zeros_like(self.grid)
        
        # Debug: Report initialization
        live_count = np.sum(self.grid > 0)
        print(f"Game of Life initialized: {live_count} live cells out of {self.grid_width * self.grid_height}")
    
    def trigger_bass_spawn(self):
        """Trigger spawning of new cells on bass hit"""
        self.bass_spawn_pending = True
    
    def trigger_x_roll(self):
        """Trigger horizontal scrolling left of grid on mid frequency hit"""
        self.x_roll_pending = True
        # Roll amount based on mid energy: 1-3 cells for subtle effect
        self.x_roll_amount = int(1 + self.audio_mid)
    
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
            print(f"{self.__class__.__name__} shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Quad vertices (per-vertex)
        layout(location = 0) in vec2 position;
        
        // Instance data (per-cell)
        layout(location = 1) in vec2 cell_position;  // Grid position
        layout(location = 2) in float cell_age;      // Age for coloring
        
        uniform vec2 resolution;
        uniform float cell_size;
        uniform float depth;
        uniform float audio_mid;  // Mid frequency energy for color boost
        
        out float v_age;
        out float v_audio_mid;
        
        void main() {
            // Scale quad to cell size and position it
            vec2 pos = position * cell_size + cell_position;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping
            float depth_normalized = depth / 100.0;
            depth_normalized = clamp(depth_normalized, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth_normalized, 1.0);
            v_age = cell_age;
            v_audio_mid = audio_mid;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in float v_age;
        in float v_audio_mid;
        out vec4 outColor;
        
        void main() {
            // Color based on cell age with multiple life stages
            // Newborn (0-3): Bright cyan
            // Young (3-8): Green
            // Adult (8-15): Yellow
            // Mature (15-25): Orange
            // Elder (25-40): Red
            // Ancient (40+): Deep purple
            
            vec3 color;
            float age = v_age;
            
            if (age < 3.0) {
                // Newborn: cyan
                color = vec3(0.0, 1.0, 1.0);
            } else if (age < 8.0) {
                // Young: cyan -> green
                float t = (age - 3.0) / 5.0;
                vec3 cyan = vec3(0.0, 1.0, 1.0);
                vec3 green = vec3(0.0, 1.0, 0.3);
                color = mix(cyan, green, t);
            } else if (age < 15.0) {
                // Adult: green -> yellow
                float t = (age - 8.0) / 7.0;
                vec3 green = vec3(0.0, 1.0, 0.3);
                vec3 yellow = vec3(1.0, 1.0, 0.0);
                color = mix(green, yellow, t);
            } else if (age < 25.0) {
                // Mature: yellow -> orange
                float t = (age - 15.0) / 10.0;
                vec3 yellow = vec3(1.0, 1.0, 0.0);
                vec3 orange = vec3(1.0, 0.5, 0.0);
                color = mix(yellow, orange, t);
            } else if (age < 40.0) {
                // Elder: orange -> red
                float t = (age - 25.0) / 15.0;
                vec3 orange = vec3(1.0, 0.5, 0.0);
                vec3 red = vec3(1.0, 0.0, 0.0);
                color = mix(orange, red, t);
            } else {
                // Ancient: red -> deep purple
                float t = clamp((age - 40.0) / 20.0, 0.0, 1.0);
                vec3 red = vec3(1.0, 0.0, 0.0);
                vec3 purple = vec3(0.5, 0.0, 0.8);
                color = mix(red, purple, t);
            }
            
            // Audio reactivity: boost color intensity with mid frequencies
            float audio_boost = 1.0 + v_audio_mid * 0.5;  // Up to 1.5x brighter
            color *= audio_boost;
            
            // Fully opaque cells
            outColor = vec4(color, 1.0);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers - Called automatically after shader compilation"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Quad vertices (0,0 to 1,1 square that will be scaled by cell_size)
        quad_vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        # Quad indices (two triangles)
        quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # Create quad VBO (location 0)
        self.quad_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_VBO)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Create EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        
        # Create instance VBO (location 1 and 2)
        # Will be updated each frame with live cell positions
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        
        # cell_position (vec2) at location 1
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)  # Advance once per instance
        
        # cell_age (float) at location 2
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(8))
        glVertexAttribDivisor(2, 1)  # Advance once per instance
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update Game of Life state each frame (audio-reactive)"""
        if not self.enabled:
            return
        
        # Audio-reactive update rate: high frequencies speed up evolution
        # Range: base_update_rate when quiet, down to 1 frame on loud highs
        high_factor = np.clip(self.audio_high, 0, 2)  # 0-2x multiplier
        self.update_rate = max(1, int(self.base_update_rate * (1.0 - high_factor * 0.7)))
        
        self.frame_counter += 1
        
        # Only update game state at specified rate
        if self.frame_counter >= self.update_rate:
            self.frame_counter = 0
            self._update_game_state()
        
        # Handle mid-triggered X-axis roll (left) - do AFTER game state update
        if self.x_roll_pending:
            self._roll_grid_x()
            self.x_roll_pending = False
        
        # Handle bass-triggered cell spawning - do AFTER roll
        if self.bass_spawn_pending:
            self._spawn_bass_cells()
            self.bass_spawn_pending = False
    
    def _update_game_state(self):
        """Apply Conway's Game of Life rules with wrapping (vectorized)"""
        # Count live neighbors for each cell (with wrapping) - fully vectorized
        alive = (self.grid > 0).astype(np.int32)
        
        # Sum all 8 neighbors using rolled arrays
        neighbors = (
            np.roll(alive, (-1, -1), axis=(0, 1)) +  # top-left
            np.roll(alive, (-1, 0), axis=(0, 1)) +   # top
            np.roll(alive, (-1, 1), axis=(0, 1)) +   # top-right
            np.roll(alive, (0, -1), axis=(0, 1)) +   # left
            np.roll(alive, (0, 1), axis=(0, 1)) +    # right
            np.roll(alive, (1, -1), axis=(0, 1)) +   # bottom-left
            np.roll(alive, (1, 0), axis=(0, 1)) +    # bottom
            np.roll(alive, (1, 1), axis=(0, 1))      # bottom-right
        )
        
        # Apply Conway's rules
        # Birth: dead cell with exactly 3 neighbors becomes alive
        # Survival: live cell with 2-3 neighbors survives
        birth = (self.grid == 0) & (neighbors == 3)
        survival = (self.grid > 0) & ((neighbors == 2) | (neighbors == 3))
        
        self.next_grid[:] = 0
        self.next_grid[birth] = 1
        self.next_grid[survival] = self.grid[survival] + 1  # Increment age
        
        # Occasionally add random cells to prevent stagnation (1% chance per update)
        if np.random.random() < 0.01:
            num_new_cells = np.random.randint(3, 8)  # Add 3-7 random cells
            for _ in range(num_new_cells):
                row = np.random.randint(0, self.grid_height)
                col = np.random.randint(0, self.grid_width)
                if self.next_grid[row, col] == 0:  # Only spawn in dead cells
                    self.next_grid[row, col] = 1
        
        # Swap buffers
        self.grid, self.next_grid = self.next_grid, self.grid
        
        # Update cell ages (vectorized)
        self.cell_ages = self.grid.astype(np.float32)
    
    def _spawn_bass_cells(self):
        """Spawn new cells adjacent to existing live cells in response to bass hit"""
        # Find all live cells
        live_rows, live_cols = np.where(self.grid > 0)
        
        if len(live_rows) == 0:
            # No live cells to spawn near, spawn randomly instead
            num_cells = int(5 + self.audio_bass * 10)
            num_cells = min(num_cells, 30)
            for _ in range(num_cells):
                row = np.random.randint(0, self.grid_height)
                col = np.random.randint(0, self.grid_width)
                if self.grid[row, col] == 0:
                    self.grid[row, col] = 1
                    self.cell_ages[row, col] = 1.0
            return
        
        # Spawn cells adjacent to existing live cells
        num_cells = int(5 + self.audio_bass * 10)
        num_cells = min(num_cells, 30)
        
        spawned = 0
        max_attempts = num_cells * 3  # Prevent infinite loops
        attempts = 0
        
        while spawned < num_cells and attempts < max_attempts:
            attempts += 1
            
            # Pick a random live cell
            idx = np.random.randint(0, len(live_rows))
            center_row = live_rows[idx]
            center_col = live_cols[idx]
            
            # Pick a random adjacent cell (8 neighbors)
            dr = np.random.randint(-1, 2)  # -1, 0, or 1
            dc = np.random.randint(-1, 2)
            
            if dr == 0 and dc == 0:
                continue  # Skip the center cell
            
            # Apply wrapping
            new_row = (center_row + dr) % self.grid_height
            new_col = (center_col + dc) % self.grid_width
            
            # Spawn in dead cells only
            if self.grid[new_row, new_col] == 0:
                self.grid[new_row, new_col] = 1
                self.cell_ages[new_row, new_col] = 1.0
                spawned += 1
    
    def _roll_grid_x(self):
        """Roll the grid horizontally to the left"""
        # Roll left (negative shift on axis 1 = columns)
        roll_amount = -self.x_roll_amount
        
        # Apply roll to both grid and ages - wraps naturally
        self.grid = np.roll(self.grid, roll_amount, axis=1)
        self.cell_ages = np.roll(self.cell_ages, roll_amount, axis=1)
    
    def render(self, state: Dict):
        """Render live cells with horizontal wrapping"""
        if not self.enabled:
            return
        
        # Get positions of all live cells
        live_rows, live_cols = np.where(self.grid > 0)
        
        if len(live_rows) == 0:
            # No live cells, reseed immediately
            print("Game of Life: All cells died, reseeding...")
            self._initialize_data()
            return
        
        # Build instance data: [x, y, age] for each live cell
        positions = np.column_stack([
            live_cols * self.cell_size,  # X position
            live_rows * self.cell_size,  # Y position
            self.cell_ages[live_rows, live_cols]  # Age
        ]).astype(np.float32)
        
        # Horizontal wrapping: detect cells near edges
        left_edge_mask = positions[:, 0] < self.wrap_margin
        right_edge_mask = positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        # Collect all positions to render (originals + duplicates)
        all_positions = [positions]
        
        # Cells near left edge need duplicates on right
        if np.any(left_edge_mask):
            left_duplicates = positions[left_edge_mask].copy()
            left_duplicates[:, 0] += self.viewport.width  # Shift to right side
            all_positions.append(left_duplicates)
        
        # Cells near right edge need duplicates on left
        if np.any(right_edge_mask):
            right_duplicates = positions[right_edge_mask].copy()
            right_duplicates[:, 0] -= self.viewport.width  # Shift to left side
            all_positions.append(right_duplicates)
        
        # Combine all positions (originals + wrapped duplicates)
        instance_data = np.vstack(all_positions)
        
        # Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Render
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, float(self.viewport.width), float(self.viewport.height))
        
        size_loc = glGetUniformLocation(self.shader, "cell_size")
        glUniform1f(size_loc, float(self.cell_size))
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        audio_mid_loc = glGetUniformLocation(self.shader, "audio_mid")
        glUniform1f(audio_mid_loc, self.audio_mid)
        
        # Draw instanced (includes wrapped duplicates)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(instance_data))
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if self.quad_VBO:
            glDeleteBuffers(1, [self.quad_VBO])
        if self.instance_VBO:
            glDeleteBuffers(1, [self.instance_VBO])
        if self.EBO:
            glDeleteBuffers(1, [self.EBO])
        if self.shader:
            glDeleteProgram(self.shader)
