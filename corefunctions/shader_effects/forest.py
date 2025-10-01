"""
Complete forest effect - GPU-based rendering + event integration
Shader-based pine forest with seasonal variations and wind animation
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import random
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_forest(state, outstate, season=0.0, density=0.8):
    """
    Shader-based forest effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_forest, season=0.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        season: Seasonal variation (0=spring, 0.25=summer, 0.5=fall, 0.75=winter)
        density: Forest density multiplier (0.5 to 2.0)
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
    
    # Initialize forest effect on first call
    if state['count'] == 0:
        print(f"Initializing forest effect for frame {frame_id}")
        
        try:
            forest_effect = viewport.add_effect(
                ForestEffect,
                season=season,
                density=density
            )
            state['forest_effect'] = forest_effect
            print(f"✓ Initialized shader forest for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize forest: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update parameters from global state
    if 'forest_effect' in state:
        effect = state['forest_effect']
        effect.wind = outstate.get('wind', 0.0)
        
        # Update season if it changes
        new_season = outstate.get('season', season)
        if abs(new_season - effect.season) > 0.01:
            effect.update_season(new_season)
    
    # Calculate fade based on duration
    if state.get('duration'):
        elapsed = state['elapsed_time']
        duration = state['duration']
        fade_duration = 6.0
        
        if elapsed < fade_duration:
            fade = elapsed / fade_duration
        elif elapsed > (duration - fade_duration):
            fade = (duration - elapsed) / fade_duration
        else:
            fade = 1.0
        
        if 'forest_effect' in state:
            state['forest_effect'].fade_factor = max(0.0, min(1.0, fade))
    
    # Clean up on close
    if state['count'] == -1:
        if 'forest_effect' in state:
            print(f"Cleaning up forest effect for frame {frame_id}")
            viewport.effects.remove(state['forest_effect'])
            state['forest_effect'].cleanup()
            print(f"✓ Cleaned up shader forest for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class ForestEffect(ShaderEffect):
    """GPU-based forest effect using instanced rendering"""
    
    def __init__(self, viewport, season: float = 0.0, density: float = 0.8):
        super().__init__(viewport)
        self.season = season
        self.density = density
        self.wind = 0.0
        self.fade_factor = 1.0
        self.time = 0.0
        
        # Buffers
        self.tree_instance_VBO = None
        self.segment_instance_VBO = None
        self.ground_VAO = None
        
        # Data arrays
        self.trees = None
        self.ground_texture = None
        
        self._generate_forest()
        self._generate_ground()
        
    def _generate_forest(self):
        """Generate forest tree data"""
        width = self.viewport.width
        height = self.viewport.height
        ground_y = height * 5 // 6
        
        num_trees = int((width // 8) * self.density)
        
        # Color palettes based on season
        palette = self._get_seasonal_palette()
        
        trees = []
        for _ in range(num_trees):
            x = random.uniform(0, width)
            y = random.uniform(ground_y - 2, ground_y + 2)
            tree_height = random.uniform(20, 35)
            base_width = tree_height * random.uniform(0.6, 0.8)
            num_segments = random.randint(5, 8)
            
            # Colors from palette
            trunk_hue = 0.08 + random.random() * 0.04
            trunk_sat = random.uniform(*palette['trunk_sat_range'])
            trunk_val = random.uniform(*palette['trunk_val_range'])
            
            needle_hue = random.uniform(*palette['hue_range'])
            needle_sat = random.uniform(*palette['sat_range'])
            needle_val = random.uniform(*palette['val_range'])
            
            sway_amount = random.uniform(0.4, 1.2)
            sway_phase = random.uniform(0, 6.28)
            
            trees.append({
                'x': x, 'y': y,
                'height': tree_height,
                'base_width': base_width,
                'segments': num_segments,
                'trunk_hue': trunk_hue,
                'trunk_sat': trunk_sat,
                'trunk_val': trunk_val,
                'needle_hue': needle_hue,
                'needle_sat': needle_sat,
                'needle_val': needle_val,
                'sway_amount': sway_amount,
                'sway_phase': sway_phase
            })
        
        # Sort by depth
        trees.sort(key=lambda t: t['y'])
        self.trees = trees
        
    def _get_seasonal_palette(self):
        """Get color palette based on season"""
        palettes = [
            # Spring (0-0.25)
            {"hue_range": (0.25, 0.30), "sat_range": (0.7, 0.9), "val_range": (0.4, 0.6),
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.5)},
            # Summer (0.25-0.5)
            {"hue_range": (0.28, 0.35), "sat_range": (0.75, 0.9), "val_range": (0.25, 0.4),
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.5)},
            # Fall (0.5-0.75)
            {"hue_range": (0.10, 0.15), "sat_range": (0.8, 0.95), "val_range": (0.5, 0.65),
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
            # Winter (0.75-1.0)
            {"hue_range": (0.3, 0.4), "sat_range": (0.05, 0.4), "val_range": (0.3, 0.5),
             "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.35)},
        ]
        
        idx = int(self.season * 4) % 4
        return palettes[idx]
    
    def _generate_ground(self):
        """Generate ground texture data"""
        width = self.viewport.width
        height = self.viewport.height
        ground_y = height * 5 // 6
        
        # Simple ground texture (can be enhanced)
        self.ground_height = ground_y
        
    def update_season(self, new_season):
        """Update season and regenerate forest colors"""
        self.season = new_season
        palette = self._get_seasonal_palette()
        
        # Update tree colors
        for tree in self.trees:
            tree['needle_hue'] = random.uniform(*palette['hue_range'])
            tree['needle_sat'] = random.uniform(*palette['sat_range'])
            tree['needle_val'] = random.uniform(*palette['val_range'])
            tree['trunk_sat'] = random.uniform(*palette['trunk_sat_range'])
            tree['trunk_val'] = random.uniform(*palette['trunk_val_range'])
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Vertex attributes
        layout(location = 0) in vec2 position;  // Base quad vertices
        
        // Instance attributes (per tree segment)
        layout(location = 1) in vec3 offset;     // x, y, depth
        layout(location = 2) in vec2 size;       // width, height
        layout(location = 3) in vec3 color_hsv;  // hue, saturation, value
        layout(location = 4) in float alpha;
        layout(location = 5) in float wind_factor; // How much this segment sways
        layout(location = 6) in float sway_phase;
        layout(location = 7) in float segment_type; // 0=trunk, 1=needles
        
        out vec4 fragColor;
        out vec2 vertPos;
        out float segmentType;
        
        uniform vec2 resolution;
        uniform float time;
        uniform float wind;
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            vertPos = position + 0.5;  // Convert to 0-1 range
            segmentType = segment_type;
            
            // Calculate wind displacement
            float wind_displacement = 0.0;
            if (segment_type > 0.5) {  // Only needles sway
                float sway = sin(time * 0.5 + sway_phase) * wind * wind_factor;
                wind_displacement = sway * 8.0;
            }
            
            // Scale and translate
            vec2 scaled = position * size;
            vec2 pos = scaled + offset.xy + vec2(wind_displacement, 0.0);
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depth = offset.z / 100.0;
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Convert HSV to RGB
            vec3 rgb = hsv2rgb(color_hsv);
            fragColor = vec4(rgb, alpha);
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 vertPos;
        in float segmentType;
        
        out vec4 outColor;
        
        uniform float fade;
        
        void main() {
            float alpha = fragColor.a;
            
            // For needle segments, create triangular fade
            if (segmentType > 0.5) {
                // Fade based on distance from center
                float dist_from_center = abs(vertPos.x - 0.5) * 2.0;
                float edge_fade = 1.0 - smoothstep(0.7, 1.0, dist_from_center);
                
                // Fade from top to bottom
                float vertical_fade = vertPos.y;
                
                alpha *= edge_fade * vertical_fade;
            }
            
            // Apply global fade
            alpha *= fade;
            
            outColor = vec4(fragColor.rgb, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile forest shaders"""
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
        """Initialize OpenGL buffers"""
        # Quad vertices (centered)
        vertices = np.array([
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5
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
        
        # Instance buffer (will be updated each frame)
        self.segment_instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.segment_instance_VBO)
        
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

    def update(self, dt: float, state: Dict):
        """Update animation time"""
        if not self.enabled:
            return
        
        self.time += dt
        self.wind = state.get('wind', 0.0)
        
    def render(self, state: Dict):
        """Render forest using instanced rendering"""
        if not self.enabled or not self.shader:
            return
            
        glUseProgram(self.shader)
        
        # Set uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "time")
        if loc != -1:
            glUniform1f(loc, self.time)
            
        loc = glGetUniformLocation(self.shader, "wind")
        if loc != -1:
            glUniform1f(loc, self.wind)
            
        loc = glGetUniformLocation(self.shader, "fade")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        # Build instance data for all tree segments
        instances = []
        
        for tree in self.trees:
            trunk_height = tree['height'] * 0.2
            trunk_width = tree['base_width'] * 0.11
            
            # Trunk (segment_type = 0)
            depth = 50.0  # Mid-depth
            instances.append([
                tree['x'], tree['y'] - trunk_height/2, depth,  # offset
                trunk_width, trunk_height,  # size
                tree['trunk_hue'], tree['trunk_sat'], tree['trunk_val'],  # color
                1.0,  # alpha
                0.0,  # wind_factor (trunk doesn't sway)
                0.0,  # sway_phase
                0.0   # segment_type (trunk)
            ])
            
            # Needle segments (segment_type = 1)
            segment_height = (tree['height'] - trunk_height) / tree['segments']
            
            for i in range(tree['segments']):
                width_factor = (tree['segments'] - i) / tree['segments']
                segment_width = tree['base_width'] * np.power(width_factor, 1.5)
                segment_y = tree['y'] - trunk_height - (i + 0.5) * segment_height
                
                # Color variation
                hue_var = (i / tree['segments']) * 0.05
                
                # Wind effect increases with height
                wind_factor = tree['sway_amount'] * ((i + 1) / tree['segments']) ** 2
                
                instances.append([
                    tree['x'], segment_y, depth,
                    segment_width, segment_height,
                    tree['needle_hue'] + hue_var, tree['needle_sat'], tree['needle_val'],
                    0.9,  # alpha
                    wind_factor,
                    tree['sway_phase'],
                    1.0   # segment_type (needles)
                ])
        
        if not instances:
            glUseProgram(0)
            return
            
        instance_data = np.array(instances, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.segment_instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 12 * 4  # 12 floats * 4 bytes
        
        # Offset (location 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Size (location 2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color HSV (location 3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Alpha (location 4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Wind factor (location 5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
        # Sway phase (location 6)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(40))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)
        
        # Segment type (location 7)
        glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(44))
        glEnableVertexAttribArray(7)
        glVertexAttribDivisor(7, 1)
        
        # Draw all segments
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(instances))
        
        glBindVertexArray(0)
        glUseProgram(0)