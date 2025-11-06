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

def shader_gameoflife(state, outstate, cell_size=10, update_rate=5, 
                      initial_density=0.3, depth=50.0):
    """
    Conway's Game of Life shader effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_gameoflife, 
                                cell_size=8, update_rate=3, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        cell_size: Size of each cell in pixels (default: 10)
        update_rate: Frames between updates (default: 5)
        initial_density: Initial probability of live cells (default: 0.3)
        depth: Z-depth for rendering (default: 50.0)
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
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing Game of Life for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                GameOfLifeEffect,
                cell_size=cell_size,
                update_rate=update_rate,
                initial_density=initial_density,
                depth=depth
            )
            state['effect'] = effect
            print(f"✓ Initialized shader gameoflife for frame {frame_id}")
            print(f"  Grid: {effect.grid_width}x{effect.grid_height} cells")
            print(f"  Cell size: {cell_size}px, Update rate: {update_rate} frames")
        except Exception as e:
            print(f"✗ Failed to initialize gameoflife: {e}")
            import traceback
            traceback.print_exc()
            return
    
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
                 initial_density: float = 0.3, depth: float = 50.0):
        super().__init__(viewport)
        self.cell_size = cell_size
        self.update_rate = update_rate  # Frames between updates
        self.initial_density = initial_density
        self.depth = depth
        self.frame_counter = 0
        
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
        
        # Cell ages for color coding
        self.cell_ages = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Working buffer for next generation
        self.next_grid = np.zeros_like(self.grid)
    
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
        
        out float v_age;
        
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
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in float v_age;
        out vec4 outColor;
        
        void main() {
            // Color based on cell age
            // Young cells: bright cyan -> older cells: deep blue
            vec3 young_color = vec3(0.0, 1.0, 1.0);    // Cyan
            vec3 old_color = vec3(0.0, 0.3, 0.8);      // Deep blue
            
            float age_factor = clamp(v_age / 10.0, 0.0, 1.0);
            vec3 color = mix(young_color, old_color, age_factor);
            
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
        """Update Game of Life state each frame"""
        if not self.enabled:
            return
        
        self.frame_counter += 1
        
        # Only update game state at specified rate
        if self.frame_counter >= self.update_rate:
            self.frame_counter = 0
            self._update_game_state()
    
    def _update_game_state(self):
        """Apply Conway's Game of Life rules with wrapping"""
        # Count live neighbors for each cell (with wrapping)
        neighbors = np.zeros_like(self.grid)
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # Shift grid with wrapping
                shifted = np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)
                neighbors += (shifted > 0).astype(np.int32)
        
        # Apply Conway's rules
        # Birth: dead cell with exactly 3 neighbors becomes alive
        # Survival: live cell with 2-3 neighbors survives
        # Death: all other cases
        
        birth = (self.grid == 0) & (neighbors == 3)
        survival = (self.grid > 0) & ((neighbors == 2) | (neighbors == 3))
        
        self.next_grid[:] = 0
        self.next_grid[birth] = 1
        self.next_grid[survival] = self.grid[survival] + 1  # Increment age
        
        # Swap buffers
        self.grid, self.next_grid = self.next_grid, self.grid
        
        # Update cell ages
        self.cell_ages[self.grid > 0] = self.grid[self.grid > 0].astype(np.float32)
        self.cell_ages[self.grid == 0] = 0.0
    
    def render(self, state: Dict):
        """Render live cells"""
        if not self.enabled:
            return
        
        # Get positions of all live cells
        live_rows, live_cols = np.where(self.grid > 0)
        
        if len(live_rows) == 0:
            # No live cells, optionally reseed
            if np.random.random() < 0.01:  # 1% chance per frame to reseed
                self._initialize_data()
            return
        
        # Build instance data: [x, y, age] for each live cell
        instance_data = np.zeros((len(live_rows), 3), dtype=np.float32)
        instance_data[:, 0] = live_cols * self.cell_size  # X position
        instance_data[:, 1] = live_rows * self.cell_size  # Y position
        instance_data[:, 2] = self.cell_ages[live_rows, live_cols]  # Age
        
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
        
        # Draw instanced
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(live_rows))
        
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
