"""
Complete forest effect - GPU-based rendering + event integration
Shader-based pine forest with seasonal variations and wind animation
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import random
import cv2
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
        season: Seasonal variation (0-1 continuous, wraps around)
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
    
    # Get current season from outstate (can change over time)
    current_season = outstate.get('season', season)
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
        fade=max(0.0, min(1.0, fade))
        
            

    # Initialize forest effect on first call
    if state['count'] == 0:
        print(f"Initializing forest effect for frame {frame_id}, season={current_season}")
        outstate['tree'] = True
        try:
            forest_effect = viewport.add_effect(
                ForestEffect,
                season=current_season,
                density=density
            )
            forest_effect.fade_factor = fade
            state['forest_effect'] = forest_effect
            print(f"✓ Initialized shader forest for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize forest: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update wind from global state
    if 'forest_effect' in state:
        effect = state['forest_effect']
        effect.wind = outstate.get('wind', 0.0)
        state['forest_effect'].fade_factor =fade

    
    # Clean up on close
    if state['count'] == -1:
        outstate['tree'] = False
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
        self.segment_instance_VBO = None
        self.ground_VAO = None
        self.ground_VBO = None
        self.ground_EBO = None
        
        # Data arrays
        self.trees = None
        self.ground_vertices = None
        
        self._generate_forest()
        self._generate_ground()
        
    def _generate_forest(self):
        """Generate forest tree data with weighted seasonal palette selection"""
        width = self.viewport.width
        height = self.viewport.height
        ground_y = height * 5 // 6
        
        num_trees = int((width // 8) * self.density)
        
        # Get palette weights based on continuous season value
        palette_weights = self._get_seasonal_weights()
        palettes = self._get_all_palettes()
        
        trees = []
        for _ in range(num_trees):
            x = random.uniform(0, width)
            y = random.uniform(ground_y - 2, ground_y + 2)
            tree_height = random.uniform(20, 35)
            base_width = tree_height * random.uniform(0.6, 0.8)
            num_segments = random.randint(5, 8)
            
            # Random depth between 50-55
            depth = random.uniform(50.0, 55.0)
            
            # Choose palette based on seasonal weights (like original code)
            palette = random.choices(palettes, weights=palette_weights, k=1)[0]
            
            # Colors from weighted palette
            trunk_hue = 0.08 + random.random() * 0.04
            trunk_sat = random.uniform(*palette['trunk_sat_range'])
            trunk_val = random.uniform(*palette['trunk_val_range'])
            
            needle_hue = random.uniform(*palette['hue_range'])
            needle_sat = random.uniform(*palette['sat_range'])
            needle_val = random.uniform(*palette['val_range'])
            
            # Color variation range for this tree
            hue_variation = random.uniform(0.02, 0.08)
            sat_variation = random.uniform(0.1, 0.3)
            val_variation = random.uniform(0.1, 0.25)
            
            sway_amount = random.uniform(0.4, 1.2)
            sway_phase = random.uniform(0, 6.28)
            
            # Random seed for this tree's noise pattern
            noise_seed = random.uniform(0, 1000)
            
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
                'hue_variation': hue_variation,
                'sat_variation': sat_variation,
                'val_variation': val_variation,
                'sway_amount': sway_amount,
                'sway_phase': sway_phase,
                'depth': depth,
                'noise_seed': noise_seed
            })
        
        # Sort by depth (back to front)
        trees.sort(key=lambda t: t['depth'])
        self.trees = trees
        
    def _get_all_palettes(self):
        """Get all seasonal color palettes"""
        return [
            # Spring/Summer Palettes
            {"hue_range": (0.25, 0.30), "sat_range": (0.7, 0.9), "val_range": (0.4, 0.6), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.5)},
            {"hue_range": (0.28, 0.35), "sat_range": (0.75, 0.9), "val_range": (0.25, 0.4), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.5)},
            {"hue_range": (0.35, 0.43), "sat_range": (0.7, 0.85), "val_range": (0.3, 0.45), 
             "trunk_sat_range": (0.45, 0.65), "trunk_val_range": (0.25, 0.4)},
            {"hue_range": (0.22, 0.28), "sat_range": (0.7, 0.9), "val_range": (0.35, 0.5), 
             "trunk_sat_range": (0.55, 0.75), "trunk_val_range": (0.3, 0.45)},
            {"hue_range": (0.30, 0.35), "sat_range": (0.8, 0.95), "val_range": (0.2, 0.3), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.25, 0.4)},
            
            # Fall Palettes
            {"hue_range": (0.15, 0.20), "sat_range": (0.8, 0.9), "val_range": (0.45, 0.6), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
            {"hue_range": (0.10, 0.15), "sat_range": (0.8, 0.95), "val_range": (0.5, 0.65), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
            {"hue_range": (0.05, 0.10), "sat_range": (0.85, 0.95), "val_range": (0.45, 0.6), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
            {"hue_range": (0.02, 0.07), "sat_range": (0.85, 0.95), "val_range": (0.4, 0.55), 
             "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
            
            # Winter Palettes
            {"hue_range": (0.35, 0.45), "sat_range": (0.6, 0.8), "val_range": (0.2, 0.35), 
             "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.35)},
            {"hue_range": (0.33, 0.38), "sat_range": (0.7, 0.85), "val_range": (0.1, 0.25), 
             "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.3)},
            {"hue_range": (0.3, 0.4), "sat_range": (0.05, 0.4), "val_range": (0.3, 0.5), 
             "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.35)},
        ]
    
    def _get_seasonal_weights(self):
        """Calculate palette weights based on continuous season value (0-1)"""
        season = self.season % 1.0  # Wrap around
        
        # Season centers (matching original code)
        spring_center = 0.125
        summer_center = 0.375
        fall_center = 0.625
        winter_center = 0.875
        season_width = 0.25
        
        # Helper function for circular distance
        def circular_distance(a, b):
            direct_distance = abs(a - b)
            return min(direct_distance, 1 - direct_distance)
        
        # Initialize weights
        palette_weights = [0] * 12
        
        # Spring weights (palette 0)
        spring_influence = max(0, 1 - circular_distance(season, spring_center) / season_width)
        palette_weights[0] = 30 * spring_influence
        
        # Summer weights (palettes 1-4)
        summer_influence = max(0, 1 - circular_distance(season, summer_center) / season_width)
        palette_weights[1] = 20 * summer_influence
        palette_weights[2] = 15 * summer_influence
        palette_weights[3] = 15 * summer_influence
        palette_weights[4] = 10 * summer_influence
        
        # Fall weights (palettes 5-8)
        fall_influence = max(0, 1 - circular_distance(season, fall_center) / season_width)
        palette_weights[5] = 15 * fall_influence
        palette_weights[6] = 20 * fall_influence
        palette_weights[7] = 15 * fall_influence
        palette_weights[8] = 10 * fall_influence
        
        # Winter weights (palettes 9-11)
        winter_influence = max(0, 1 - circular_distance(season, winter_center) / season_width)
        palette_weights[9] = 20 * winter_influence
        palette_weights[10] = 15 * winter_influence
        palette_weights[11] = 100 * winter_influence
        
        # Add baseline to avoid zero probabilities
        palette_weights = [max(1, w) for w in palette_weights]
        
        return palette_weights
        
    def _generate_ground(self):
        """Generate ground texture as a textured quad"""
        width = self.viewport.width
        height = self.viewport.height
        ground_y = height * 5 // 6
        
        # Create multi-scale noise for ground texture
        scale_1 = np.random.uniform(-1, 1, (height//8, width//8))
        scale_2 = np.random.uniform(-1, 1, (height//4, width//4))
        scale_3 = np.random.uniform(-1, 1, (height//2, width//2))
        
        # Resize to full dimensions
        scale_1 = cv2.resize(scale_1, (width, height))
        scale_2 = cv2.resize(scale_2, (width, height))
        scale_3 = cv2.resize(scale_3, (width, height))
        
        # Combine scales
        noise = scale_1 * 0.5 + scale_2 * 0.3 + scale_3 * 0.2
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Create ground vertices with texture coordinates
        # Each vertex: [x, y, u, v, hue, sat, val]
        vertices = []
        
        # Sample ground at regular intervals
        sample_step = max(1, width // 120)  # Adjust resolution
        
        for y in range(ground_y, height, sample_step):
            for x in range(0, width, sample_step):
                # Position
                px = float(x)
                py = float(y)
                
                # Texture coordinates (0-1)
                u = x / width
                v = (y - ground_y) / (height - ground_y)
                
                # Ground color based on noise
                noise_val = noise[min(y, height-1), min(x, width-1)]
                
                # Calculate ground factor for depth shading
                ground_factor = v
                
                # Base ground color (brown)
                hue = 0.10 + noise_val * 0.05
                sat = 0.4 + noise_val * 0.2
                val = 0.3 - ground_factor * 0.1 + noise_val * 0.1
                
                # Add green patches
                if noise_val > 0.7 and y < ground_y + 3:
                    hue = 0.3 + noise_val * 0.05
                    sat = 0.5 + noise_val * 0.2
                
                vertices.append([px, py, u, v, hue, sat, val])
        
        # Convert to numpy array
        self.ground_vertices = np.array(vertices, dtype=np.float32)
        
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
        layout(location = 7) in float segment_type; // 0=trunk, 1=needles, 2=branch
        layout(location = 8) in vec3 color_variation; // hue_var, sat_var, val_var
        layout(location = 9) in float noise_seed;
        
        out vec4 fragColor;
        out vec2 vertPos;
        out vec2 worldPos;
        flat out float segmentType;
        flat out vec3 colorVariation;
        flat out float noiseSeed;
        
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
            colorVariation = color_variation;
            noiseSeed = noise_seed;
            
            // Calculate wind displacement
            float wind_displacement = 0.0;
            if (segment_type > 0.5) {  // Needles and branches sway
                float sway = sin(time * 0.5 + sway_phase) * wind * wind_factor;
                wind_displacement = sway * 8.0;
            }
            
            // Scale and translate
            vec2 scaled = position * size;
            vec2 pos = scaled + offset.xy + vec2(wind_displacement, 0.0);
            
            worldPos = pos;  // Pass world position for noise
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            float depth = offset.z / 100.0;
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Pass base color to fragment shader
            vec3 rgb = hsv2rgb(color_hsv);
            fragColor = vec4(rgb, alpha);
        }
        """
    
    def get_ground_vertex_shader(self):
        """Separate shader for ground rendering"""
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // x, y
        layout(location = 1) in vec2 texCoord;  // u, v
        layout(location = 2) in vec3 color_hsv; // hue, sat, val
        
        out vec3 fragColorHSV;
        
        uniform vec2 resolution;
        
        void main() {
            // Convert to clip space
            vec2 clipPos = (position / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Ground at depth 49
            float depth = 49.0 / 100.0;
            gl_Position = vec4(clipPos, depth, 1.0);
            
            fragColorHSV = color_hsv;
        }
        """
    
    def get_ground_fragment_shader(self):
        """Separate fragment shader for ground"""
        return """
        #version 310 es
        precision highp float;
        
        in vec3 fragColorHSV;
        out vec4 outColor;
        
        uniform float fade;
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            vec3 rgb = hsv2rgb(fragColorHSV);
            outColor = vec4(rgb, fade);
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 vertPos;
        in vec2 worldPos;
        flat in float segmentType;
        flat in vec3 colorVariation;
        flat in float noiseSeed;
        
        out vec4 outColor;
        
        uniform float fade;
        uniform float time;
        
        // Simple 2D noise function
        float hash(vec2 p) {
            p = fract(p * vec2(443.897, 441.423));
            p += dot(p, p.yx + 19.19);
            return fract(p.x * p.y);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        float fbm(vec2 p) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;
            
            for(int i = 0; i < 4; i++) {
                value += amplitude * noise(p * frequency);
                frequency *= 2.0;
                amplitude *= 0.5;
            }
            return value;
        }
        
        // RGB to HSV
        vec3 rgb2hsv(vec3 c) {
            vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
            vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
            vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
            
            float d = q.x - min(q.w, q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
        }
        
        // HSV to RGB
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            float alpha = fragColor.a;
            vec3 baseColor = fragColor.rgb;
            
            // Generate noise based on world position and tree seed
            vec2 noisePos = worldPos * 0.1 + vec2(noiseSeed);
            float noiseValue = fbm(noisePos);
            
            // Add finer detail noise
            float detailNoise = noise(worldPos * 0.5 + vec2(noiseSeed * 2.0));
            
            // For needle segments, create organic texture
            if (segmentType > 0.5 && segmentType < 1.5) {
                // Distance from center for edge detection
                float dist_from_center = abs(vertPos.x - 0.5) * 2.0;
                
                // Create needle clusters using noise
                float clusterNoise = fbm(worldPos * 0.3 + vec2(noiseSeed));
                float needlePattern = fbm(worldPos * 2.0 + vec2(noiseSeed * 3.0));
                
                // Create gaps and clusters - MORE VISIBLE
                float density = clusterNoise * 0.4 + 0.6;  // Increased base density
                float needleAlpha = smoothstep(0.2, 0.5, needlePattern * density);  // Easier threshold
                
                // Edge fade - SOFTER
                float edge_fade = 1.0 - smoothstep(0.7, 1.0, dist_from_center);  // Wider visible area
                
                // Vertical fade (darker at top, lighter at bottom) - LESS AGGRESSIVE
                float vertical_fade = vertPos.y * 0.5 + 0.5;  // More uniform
                
                alpha *= needleAlpha * edge_fade * vertical_fade * 1.5;  // Boost overall alpha
                
                // Apply color variation using HSV
                vec3 hsv = rgb2hsv(baseColor);
                hsv.x += (noiseValue - 0.5) * colorVariation.x; // Hue variation
                hsv.y += (detailNoise - 0.5) * colorVariation.y; // Saturation variation
                hsv.z += (noiseValue - 0.5) * colorVariation.z;  // Value variation
                
                // Add depth-based darkening - LESS EXTREME
                hsv.z *= 0.7 + vertical_fade * 0.3;
                
                baseColor = hsv2rgb(hsv);
            }
            // For branch segments
            else if (segmentType > 1.5) {
                // Branches have less alpha and more variation
                float branchNoise = noise(worldPos * 0.8 + vec2(noiseSeed * 1.5));
                alpha *= 0.6 + branchNoise * 0.4;
                
                // Darker branches
                vec3 hsv = rgb2hsv(baseColor);
                hsv.z *= 0.5 + branchNoise * 0.3;
                baseColor = hsv2rgb(hsv);
            }
            // For trunk
            else {
                // Add bark texture
                float barkNoise = fbm(worldPos * 0.2 + vec2(noiseSeed));
                float barkDetail = noise(worldPos * 1.5);
                
                vec3 hsv = rgb2hsv(baseColor);
                hsv.z *= 0.8 + barkNoise * 0.4;
                hsv.y *= 0.7 + barkDetail * 0.3;
                baseColor = hsv2rgb(hsv);
            }
            
            // Discard very transparent pixels
            if (alpha < 0.05) {
                discard;
            }
            
            // Apply global fade
            alpha *= fade;
            
            outColor = vec4(baseColor, alpha);
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
            
            # Compile ground shader
            ground_vert = shaders.compileShader(self.get_ground_vertex_shader(), GL_VERTEX_SHADER)
            ground_frag = shaders.compileShader(self.get_ground_fragment_shader(), GL_FRAGMENT_SHADER)
            self.ground_shader = shaders.compileProgram(ground_vert, ground_frag)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise

    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Tree quad vertices (centered)
        vertices = np.array([
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO for trees
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
        
        # Create VAO for ground
        self.ground_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.ground_VAO)
        
        # Ground vertex buffer
        self.ground_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ground_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.ground_vertices.nbytes, self.ground_vertices, GL_STATIC_DRAW)
        
        stride = 7 * 4  # 7 floats * 4 bytes
        
        # Position (location 0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Tex coords (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        
        # Color HSV (location 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        
        glBindVertexArray(0)
        
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
        # Render ground first (at depth 49)
        self._render_ground()
        
        # Then render trees (at depth 50-55)
        self._render_trees()
    
    def _render_ground(self):
        """Render ground plane"""
        if self.ground_shader is None or self.ground_vertices is None:
            return
        
        # Ensure depth testing and writing are enabled
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)  # Allow writing to depth buffer
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(self.ground_shader)
        
        # Set uniforms
        loc = glGetUniformLocation(self.ground_shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.ground_shader, "fade")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        glBindVertexArray(self.ground_VAO)
        glDrawArrays(GL_POINTS, 0, len(self.ground_vertices))
        glBindVertexArray(0)
        glUseProgram(0)
    
    def _render_trees(self):
        """Render tree segments with branches"""
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
            depth = tree['depth']
            
            # Get ground height
            ground_y = self.viewport.height * 5 // 6
            
            # Trunk - bottom should always be at ground level
            instances.append([
                tree['x'], ground_y - trunk_height/2, depth,  # Use ground_y instead of tree['y']
                trunk_width, trunk_height,  # size
                tree['trunk_hue'], tree['trunk_sat'], tree['trunk_val'],  # color
                1.0,  # alpha
                0.0,  # wind_factor (trunk doesn't sway)
                0.0,  # sway_phase
                0.0,  # segment_type (trunk)
                0.02, 0.1, 0.15,  # color_variation (subtle for trunk)
                tree['noise_seed']
            ])
            
            # BRANCHES DISABLED - keeping trunk and foliage only
            
            # Now add needle segments
            segment_height = (tree['height'] - trunk_height) / tree['segments']
            
            for i in range(tree['segments']):
                width_factor = (tree['segments'] - i) / tree['segments']
                segment_width = tree['base_width'] * np.power(width_factor, 1.5)
                segment_y = ground_y - trunk_height - (i + 0.5) * segment_height 
            

            # Now add needle segments
            for i in range(tree['segments']):
                width_factor = (tree['segments'] - i) / tree['segments']
                segment_width = tree['base_width'] * np.power(width_factor, 1.5)
                segment_y = tree['y'] - trunk_height - (i + 0.5) * segment_height
                
                # Color variation (more at top)
                hue_var = (i / tree['segments']) * 0.05
                
                # Wind effect increases with height
                wind_factor = tree['sway_amount'] * ((i + 1) / tree['segments']) ** 2
                
                # Main foliage segment
                instances.append([
                    tree['x'], segment_y, depth,
                    segment_width, segment_height,
                    tree['needle_hue'] + hue_var, tree['needle_sat'], tree['needle_val'],
                    0.9,  # alpha
                    wind_factor,
                    tree['sway_phase'],
                    1.0,  # segment_type (needles)
                    tree['hue_variation'], tree['sat_variation'], tree['val_variation'],
                    tree['noise_seed']
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
        stride = 16 * 4  # 16 floats * 4 bytes
        
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
        
        # Color variation (location 8)
        glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(48))
        glEnableVertexAttribArray(8)
        glVertexAttribDivisor(8, 1)
        
        # Noise seed (location 9)
        glVertexAttribPointer(9, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(60))
        glEnableVertexAttribArray(9)
        glVertexAttribDivisor(9, 1)
        
        # Draw all segments
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(instances))
        
        glBindVertexArray(0)
        glUseProgram(0)