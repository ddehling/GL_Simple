"""
Complete cloud effect - rendering + event integration
GPU-accelerated drifting clouds with procedural generation and depth
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from scipy.ndimage import gaussian_filter

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_drifting_clouds(state, outstate, density=1.0):
    """
    Shader-based drifting clouds effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_drifting_clouds, density=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Cloud spawn rate multiplier
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
    
    # Initialize cloud effect on first call
    if state['count'] == 0:
        outstate['has_clouds'] = True
        print(f"Initializing cloud effect for frame {frame_id}")
        
        try:
            cloud_effect = viewport.add_effect(
                CloudEffect,
                density=density,
                max_clouds=20
            )
            state['cloud_effect'] = cloud_effect
            print(f"✓ Initialized shader clouds for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize clouds: {e}")
            import traceback
            traceback.print_exc()
            return
    
        # Update wind, fog, and cloudyness from global state
    if 'cloud_effect' in state:
        effect = state['cloud_effect']
        effect.wind = outstate.get('wind', 0) * 50
        effect.fog_level = outstate.get('fog_level', 0)
        effect.cloudyness = outstate.get('cloudyness', 0.5)  # 0.0 to 1.0
        
        # Calculate fade with 4 second fade in/out
        fade_duration = 7.0
        total_duration = state.get('duration', 60)
        elapsed = state.get('elapsed_time', 0)
        
        if elapsed < fade_duration:
            # Fade in
            effect.fade_factor = elapsed / fade_duration
        elif elapsed > (total_duration - fade_duration):
            # Fade out
            effect.fade_factor = (total_duration - elapsed) / fade_duration
        else:
            # Fully visible
            effect.fade_factor = 1.0
        
        effect.fade_factor = np.clip(effect.fade_factor, 0, 1)
    
    # On close event, clean up
    if state['count'] == -1:
        outstate['has_clouds'] = False
        if 'cloud_effect' in state:
            print(f"Cleaning up cloud effect for frame {frame_id}")
            viewport.effects.remove(state['cloud_effect'])
            state['cloud_effect'].cleanup()
            print(f"✓ Cleaned up shader clouds for frame {frame_id}")


# ============================================================================
# Cloud Texture Generator
# ============================================================================

class CloudTextureGenerator:
    """Generates procedural cloud textures"""
    
    @staticmethod
    def generate_cloud(width, height, pattern_type=None):
        """Generate a single cloud texture with varied shape patterns"""
        
        if pattern_type is None:
            pattern_type = np.random.randint(0, 4)
        
        # Create density field
        density = np.zeros((height, width))
        y, x = np.mgrid[0:height, 0:width]
        
        # Number of blobs for varied shape
        num_blobs = np.random.randint(10, 20)
        
        # Pattern-specific parameters
        if pattern_type == 0:  # Horizontal stretched
            main_axis_ratio = np.random.uniform(1.5, 3.0)
            secondary_axis_ratio = np.random.uniform(0.6, 1.2)
            blob_x = np.random.beta(2, 5, num_blobs) * width
            blob_y = np.random.normal(height/2, height/4, num_blobs)
        elif pattern_type == 1:  # Vertical stretched
            main_axis_ratio = np.random.uniform(0.6, 1.2)
            secondary_axis_ratio = np.random.uniform(1.5, 3.0)
            blob_x = np.random.normal(width/2, width/4, num_blobs)
            blob_y = np.random.beta(2, 5, num_blobs) * height
        elif pattern_type == 2:  # Clustered multi-center
            main_axis_ratio = np.random.uniform(0.8, 1.5)
            secondary_axis_ratio = np.random.uniform(0.8, 1.5)
            centers = [(np.random.uniform(0.2, 0.8) * width,
                       np.random.uniform(0.2, 0.8) * height)
                      for _ in range(np.random.randint(2, 4))]
            center_idx = np.random.randint(0, len(centers), num_blobs)
            blob_x = np.array([centers[i][0] + np.random.normal(0, width/5) for i in center_idx])
            blob_y = np.array([centers[i][1] + np.random.normal(0, height/5) for i in center_idx])
        else:  # Random scattered
            main_axis_ratio = np.random.uniform(0.7, 1.8)
            secondary_axis_ratio = np.random.uniform(0.7, 1.8)
            blob_x = np.random.uniform(0.1, 0.9, num_blobs) * width
            blob_y = np.random.uniform(0.1, 0.9, num_blobs) * height
        
        # Add central dense blobs
        num_central = np.random.randint(3, 6)
        for i in range(num_central):
            cx = width * (0.5 + np.random.uniform(-0.2, 0.2))
            cy = height * (0.5 + np.random.uniform(-0.2, 0.2))
            rx = np.random.uniform(0.2, 0.5) * width * main_axis_ratio
            ry = np.random.uniform(0.2, 0.5) * height * secondary_axis_ratio
            
            angle = np.random.uniform(0, np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = (x - cx) * cos_a - (y - cy) * sin_a
            dy = (x - cx) * sin_a + (y - cy) * cos_a
            dist = np.sqrt((dx/rx)**2 + (dy/ry)**2)
            
            falloff = np.random.uniform(1.5, 3.0)
            blob_density = np.exp(-dist**falloff)
            density += blob_density * np.random.uniform(0.7, 1.0)
        
        # Add peripheral blobs
        for i in range(num_blobs):
            cx, cy = blob_x[i], blob_y[i]
            edge_factor = max(0.001, min(cx/width, (width-cx)/width, cy/height, (height-cy)/height))
            size_factor = 0.3 + 0.7 * (edge_factor ** 0.5)
            
            rx = np.random.uniform(0.05, 0.3) * width * size_factor
            ry = np.random.uniform(0.05, 0.3) * height * size_factor
            
            angle = np.random.uniform(0, np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = (x - cx) * cos_a - (y - cy) * sin_a
            dy = (x - cx) * sin_a + (y - cy) * cos_a
            
            if np.random.random() > 0.7:
                dist = (dx/rx)**2 + (dy/ry)**2
            else:
                dist = np.sqrt((dx/rx)**2 + (dy/ry)**2)
            
            falloff = np.random.uniform(1.2, 4.0)
            blob_density = np.exp(-dist**falloff)
            density += blob_density * np.random.uniform(0.3, 0.9)
        
        # Normalize
        density = np.clip(density / (np.max(density) + 1e-10), 0, 1)
        
        # Add noise texture
        noise = gaussian_filter(np.random.random((height, width)), sigma=1.0)
        
        # Apply wispy edges
        edge_mask = (density > 0.1) & (density < 0.5)
        if np.any(edge_mask):
            density[edge_mask] *= (0.6 + 0.8 * noise[edge_mask])
        
        # Blur for smoothness
        density = gaussian_filter(density, sigma=0.8)
        
        # Create brightness with vertical gradient
        vertical_gradient = 1.0 - y / height * 0.3
        brightness = (220 + noise * 35) * vertical_gradient
        
        # Create alpha with strong edge falloff
        alpha = (density**2 * 255).astype(np.float32)
        
        # Enhanced edge softening
        soft_edge = (density > 0) & (density < 0.6)
        if np.any(soft_edge):
            edge_factor = (density[soft_edge] / 0.6) ** 1.5
            edge_alpha = alpha[soft_edge]
            edge_multiplier = edge_factor**2.0 * (0.4 + 0.6 * noise[soft_edge])
            edge_alpha *= edge_multiplier
            
            extreme_edge = density[soft_edge] < 0.2
            if np.any(extreme_edge):
                extreme_factor = (density[soft_edge][extreme_edge] / 0.2) ** 3.0
                edge_alpha[extreme_edge] *= extreme_factor
            
            alpha[soft_edge] = np.clip(edge_alpha, 0, 255)
        
        # Strong blur for smooth edges
        alpha = gaussian_filter(alpha, sigma=1.5)
        
        # Force zero-alpha border
        border_width = 3
        if height > 2*border_width and width > 2*border_width:
            alpha[:border_width, :] = 0
            alpha[-border_width:, :] = 0
            alpha[:, :border_width] = 0
            alpha[:, -border_width:] = 0
            
            # Gradient falloff
            gradient_width = border_width
            if border_width + gradient_width <= height:
                rows = np.arange(gradient_width)
                factors = (rows / gradient_width).reshape(-1, 1)
                alpha[border_width:border_width + gradient_width, :] = (
                    alpha[border_width:border_width + gradient_width, :] * factors
                ).astype(np.uint8)
                alpha[-(border_width + gradient_width):-border_width, :] = (
                    alpha[-(border_width + gradient_width):-border_width, :] * factors[::-1]
                ).astype(np.uint8)
            
            if border_width + gradient_width <= width:
                cols = np.arange(gradient_width)
                factors = (cols / gradient_width).reshape(1, -1)
                alpha[:, border_width:border_width + gradient_width] = (
                    alpha[:, border_width:border_width + gradient_width] * factors
                ).astype(np.uint8)
                alpha[:, -(border_width + gradient_width):-border_width] = (
                    alpha[:, -(border_width + gradient_width):-border_width] * factors[::-1]
                ).astype(np.uint8)
        
        alpha = alpha.astype(np.uint8)
        
        # Create RGBA image
        cloud_img = np.zeros((height, width, 4), dtype=np.uint8)
        valid = alpha > 0
        for c in range(3):
            cloud_img[:, :, c][valid] = np.clip(brightness[valid], 0, 255).astype(np.uint8)
        cloud_img[:, :, 3] = alpha
        
        return cloud_img


# ============================================================================
# Rendering Class
# ============================================================================

class CloudEffect(ShaderEffect):
    """GPU-based cloud effect using instanced rendering with textures"""
    
    def __init__(self, viewport, density: float = 1.0, max_clouds: int = 8):
        super().__init__(viewport)
        self.density = density
        self.max_clouds = max_clouds
        self.wind = 0.0
        self.fog_level = 0.0
        self.fade_factor = 0.0  # Start faded out
        self.cloudyness = 0.5  # Cloudyness factor (0.0 to 1.0)
        
                # Cloud depth
        self.cloud_depth = 40.0
        
        # Horizontal wrapping support (margin should exceed largest cloud size)
        # Max cloud width = 60px * max scale (0.9) = 54px
        self.wrap_margin = 60.0  # Should be >= max possible cloud width
        
                # Vectorized cloud data
        self.positions = np.zeros((0, 2), dtype=np.float32)  # [x, y]
        self.speeds = np.zeros(0, dtype=np.float32)
        self.base_opacities = np.zeros(0, dtype=np.float32)
        self.current_opacities = np.zeros(0, dtype=np.float32)
        self.sizes = np.zeros(0, dtype=np.float32)
        self.z_indices = np.zeros(0, dtype=np.float32)
        self.is_fading_out = np.zeros(0, dtype=bool)  # Track clouds that are fading out for removal
        self.lifetime = np.zeros(0, dtype=np.float32)  # How long cloud has existed
        self.turbulence_phases = np.zeros((0, 3), dtype=np.float32)
        self.turbulence_speeds = np.zeros((0, 3), dtype=np.float32)
        self.turbulence_amounts = np.zeros((0, 3), dtype=np.float32)
        self.subpixel_offsets = np.zeros((0, 2), dtype=np.float32)
        self.noise_offsets = np.zeros((0, 2), dtype=np.float32)
        
        # Texture storage
        self.cloud_textures = []
        self.texture_ids = []
        self.texture_indices = np.zeros(0, dtype=np.int32)
        
        # Animation time
        self.noise_time = 0.0
        self.start_time = 0.0
        
    def _spawn_cloud(self):
        """Spawn a single new cloud"""
        # Generate smaller cloud texture
        width = np.random.randint(30, 80)
        height = np.random.randint(15, 60)
        cloud_img = CloudTextureGenerator.generate_cloud(width, height)
        
        # Upload texture to GPU
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, cloud_img)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        self.cloud_textures.append((width, height))
        self.texture_ids.append(texture_id)
        texture_idx = len(self.texture_ids) - 1
        
                # Initial position - span full viewport height
        start_x = np.random.uniform(-width * 1.5, self.viewport.width + width * 1.5)
        start_y = np.random.uniform(0, self.viewport.height)
        new_position = np.array([[start_x, start_y]], dtype=np.float32)
        
                                # Movement parameters - much slower speed
        new_speed = np.array([np.random.uniform(0.5, 1.5)], dtype=np.float32)
        new_base_opacity = np.array([np.random.uniform(0.5, 0.9)], dtype=np.float32)
        
        # Always start at zero opacity and fade in naturally
        # This prevents sudden appearance when spawning
        new_current_opacity = np.array([0.0], dtype=np.float32)
        
        # Smaller size scaling
        new_size = np.array([np.random.uniform(0.5, 0.9)], dtype=np.float32)
        new_z_index = np.array([start_y + np.random.uniform(-5, 5)], dtype=np.float32)
        
        # Turbulence
        new_turb_phases = np.random.uniform(0, 2*np.pi, (1, 3)).astype(np.float32)
        new_turb_speeds = np.random.uniform(0.1, 0.3, (1, 3)).astype(np.float32)
        new_turb_amounts = np.column_stack([
            np.random.uniform(0.5, 1.5, 1),
            np.random.uniform(0.3, 0.8, 1),
            np.random.uniform(0.1, 0.3, 1)
        ]).astype(np.float32)
        
        new_subpixel = np.zeros((1, 2), dtype=np.float32)
        new_noise_offset = np.random.uniform(0, 10, (1, 2)).astype(np.float32)
        new_texture_idx = np.array([texture_idx], dtype=np.int32)
        new_is_fading = np.array([False], dtype=bool)
        new_lifetime = np.array([0.0], dtype=np.float32)
        
        # Concatenate
        self.positions = np.vstack([self.positions, new_position]) if len(self.positions) > 0 else new_position
        self.speeds = np.concatenate([self.speeds, new_speed]) if len(self.speeds) > 0 else new_speed
        self.base_opacities = np.concatenate([self.base_opacities, new_base_opacity]) if len(self.base_opacities) > 0 else new_base_opacity
        self.current_opacities = np.concatenate([self.current_opacities, new_current_opacity]) if len(self.current_opacities) > 0 else new_current_opacity
        self.sizes = np.concatenate([self.sizes, new_size]) if len(self.sizes) > 0 else new_size
        self.z_indices = np.concatenate([self.z_indices, new_z_index]) if len(self.z_indices) > 0 else new_z_index
        self.turbulence_phases = np.vstack([self.turbulence_phases, new_turb_phases]) if len(self.turbulence_phases) > 0 else new_turb_phases
        self.turbulence_speeds = np.vstack([self.turbulence_speeds, new_turb_speeds]) if len(self.turbulence_speeds) > 0 else new_turb_speeds
        self.turbulence_amounts = np.vstack([self.turbulence_amounts, new_turb_amounts]) if len(self.turbulence_amounts) > 0 else new_turb_amounts
        self.subpixel_offsets = np.vstack([self.subpixel_offsets, new_subpixel]) if len(self.subpixel_offsets) > 0 else new_subpixel
        self.noise_offsets = np.vstack([self.noise_offsets, new_noise_offset]) if len(self.noise_offsets) > 0 else new_noise_offset
        self.texture_indices = np.concatenate([self.texture_indices, new_texture_idx]) if len(self.texture_indices) > 0 else new_texture_idx
        self.is_fading_out = np.concatenate([self.is_fading_out, new_is_fading]) if len(self.is_fading_out) > 0 else new_is_fading
        self.lifetime = np.concatenate([self.lifetime, new_lifetime]) if len(self.lifetime) > 0 else new_lifetime
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertices (0 to 1)
        layout(location = 1) in vec3 offset;    // Cloud position (x, y, z)
        layout(location = 2) in vec2 size;      // Cloud texture size (width, height)
        layout(location = 3) in float scale;    // Cloud scale factor
        layout(location = 4) in float opacity;  // Cloud opacity
        
        out vec2 texCoord;
        out float fragOpacity;
        uniform vec2 resolution;
        
        void main() {
            texCoord = position;
            fragOpacity = opacity;
            
            // Scale by texture size and scale factor
            vec2 scaled = position * size * scale;
            
            // Translate to cloud position
            vec2 pos = scaled + offset.xy;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use Z for depth
            float depth = offset.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 texCoord;
        in float fragOpacity;
        out vec4 outColor;
        
        uniform sampler2D cloudTexture;
        uniform float noiseTime;
        uniform float fadeFactor;
        
        // Simple 2D noise function
        float noise(vec2 p) {
            return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        }
        
        void main() {
            vec4 texColor = texture(cloudTexture, texCoord);
            
            // Add animated noise to alpha for dynamic wispy effect
            vec2 noiseCoord = texCoord * 5.0 + vec2(noiseTime * 0.1, noiseTime * 0.07);
            float noiseVal = noise(noiseCoord) * 0.7 + 0.6;
            
            // Apply noise more strongly at edges
            float alphaVal = texColor.a / 255.0;
            float edgeFactor = pow(alphaVal, 0.4);
            float noiseImpact = noiseVal * (1.0 - edgeFactor * 0.95);
            
            // Apply fade factor for smooth transitions
            float finalAlpha = texColor.a * fragOpacity * noiseImpact * fadeFactor;
            
            outColor = vec4(texColor.rgb, finalAlpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link cloud shaders"""
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
        """Initialize OpenGL buffers"""
        # Quad vertices (0 to 1 for texture coordinates)
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # NOTE: Depth testing and blending are enabled globally - DO NOT toggle here
        
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
        
        # Instance buffer
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
        
                # Don't spawn initial clouds - they will be added dynamically based on cloudyness
        # This allows smooth fade-in from zero clouds
        
        self.start_time = 0.0

    def update(self, dt: float, state: Dict):
        """Update cloud positions and properties"""
        if not self.enabled:
            return
        
        # Update noise time
        self.noise_time += dt * 0.2
        self.start_time += dt
        
        # Global wave for coordinated motion
        global_wave_y = np.sin(self.start_time * 0.05) * 0.3 + np.sin(self.start_time * 0.13) * 0.1
        
        # Dynamic cloud count based on cloudyness (0.0 to 1.0)
        # cloudyness=0.0 -> 0 clouds, cloudyness=1.0 -> max_clouds
        target_cloud_count = int(np.clip(self.cloudyness, 0, 1) * self.max_clouds)
        current_active_clouds = np.sum(~self.is_fading_out) if len(self.is_fading_out) > 0 else 0
        
        # Add clouds if we're below target (smooth spawn with fade-in)
        if current_active_clouds < target_cloud_count:
            # Spawn rate: one cloud every 2-4 seconds when below target
            spawn_chance = dt * 0.3  # ~30% chance per second
            if np.random.random() < spawn_chance:
                self._spawn_cloud()
                print(f"Spawned cloud (cloudyness={self.cloudyness:.2f}, active={current_active_clouds+1}/{target_cloud_count})")
        
        # Remove clouds if we're above target (smooth fade-out)
        elif current_active_clouds > target_cloud_count:
            # Find clouds that aren't already fading out and mark one for removal
            active_indices = np.where(~self.is_fading_out)[0]
            if len(active_indices) > 0:
                # Prioritize clouds that are older and in safe middle zone for fading
                removal_candidate = None
                for idx in active_indices:
                    tex_idx = self.texture_indices[idx]
                    cloud_width = self.cloud_textures[tex_idx][0] * self.sizes[idx]
                    cloud_x = self.positions[idx, 0]
                    
                    middle_safe_zone_left = self.wrap_margin + cloud_width
                    middle_safe_zone_right = self.viewport.width - self.wrap_margin - cloud_width
                    is_in_middle = (cloud_x > middle_safe_zone_left and 
                                   cloud_x < middle_safe_zone_right)
                    
                    if is_in_middle and self.lifetime[idx] > 3.0:  # At least 3 seconds old
                        removal_candidate = idx
                        break
                
                # If no ideal candidate in middle, just pick the oldest
                if removal_candidate is None:
                    removal_candidate = active_indices[np.argmax(self.lifetime[active_indices])]
                
                self.is_fading_out[removal_candidate] = True
                print(f"Fading out cloud (cloudyness={self.cloudyness:.2f}, active={current_active_clouds-1}/{target_cloud_count})")
        
        # If no clouds exist yet, don't process the rest
        if len(self.positions) == 0:
            return
        
        # Update lifetime for all clouds
        self.lifetime += dt
        
        # Smooth opacity transitions
        # For clouds fading out, force opacity to zero
        # For normal clouds, fade toward base opacity
        target_opacities = np.where(self.is_fading_out, 0.0, self.base_opacities)
        opacity_diff = target_opacities - self.current_opacities
        
        mask = np.abs(opacity_diff) > 0.01
        if np.any(mask):
            # Slow fade-in (3-4 seconds), faster fade-out (2-3 seconds) for smooth transitions
            transition_speed = np.where(self.is_fading_out, 0.4, 0.3)  # Slower transitions
            self.current_opacities += opacity_diff * transition_speed * dt
            self.current_opacities = np.clip(self.current_opacities, 0, 1)
        
        # Update turbulence
        self.turbulence_phases += self.turbulence_speeds * dt
        
        # Update subpixel offsets
        self.subpixel_offsets[:, 0] = (self.subpixel_offsets[:, 0] + self.speeds * dt) % 1.0
        self.subpixel_offsets[:, 1] = (self.subpixel_offsets[:, 1] + dt * 0.1) % 1.0
        
                # Update positions
        self.positions[:, 0] += (self.speeds + self.wind) * dt
        self.positions[:, 1] += global_wave_y * dt
        
        # Horizontal wrapping - immediate, no gaps (same as rain effect)
        left_mask = self.positions[:, 0] < 0
        right_mask = self.positions[:, 0] >= self.viewport.width
        self.positions[left_mask, 0] += self.viewport.width
        self.positions[right_mask, 0] -= self.viewport.width
        
        # Clamp vertical position to viewport height
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.viewport.height)
        
        # Update z-indices based on y position
        self.z_indices = self.positions[:, 1] * 0.3 + np.sin(self.turbulence_phases[:, 2]) * 3
        
                                                                # Manage cloud lifecycle: fade-out and recycling
        for i in range(len(self.positions)):
            tex_idx = self.texture_indices[i]
            cloud_width = self.cloud_textures[tex_idx][0] * self.sizes[i]
            cloud_x = self.positions[i, 0]
            
                        # Check if cloud has any wrapped duplicates being rendered
            # Duplicate on left: rendered when cloud's left edge is near/past right edge
            has_duplicate_on_left = (cloud_x > self.viewport.width - self.wrap_margin)
            # Duplicate on right: rendered when cloud is near left edge  
            has_duplicate_on_right = (cloud_x < self.wrap_margin)
            
            # Cloud is in the MIDDLE of screen (safe zone for fading)
            # Not near edges, no duplicates being rendered
            middle_safe_zone_left = self.wrap_margin + cloud_width
            middle_safe_zone_right = self.viewport.width - self.wrap_margin - cloud_width
            is_in_middle = (cloud_x > middle_safe_zone_left and 
                           cloud_x < middle_safe_zone_right and
                           not has_duplicate_on_left and 
                           not has_duplicate_on_right)
            
            min_lifetime = 30.0  # Minimum 30 seconds before natural removal
            
            if not self.is_fading_out[i]:
                # Natural lifecycle: fade-out when old and in middle (visible)
                # This ensures fade happens where user can see it naturally
                # and wrapping at edges is ALWAYS seamless
                # NOTE: Cloudyness-based removal is handled separately above
                if self.lifetime[i] > min_lifetime and is_in_middle:
                    # Start fading out (visible in middle of screen)
                    self.is_fading_out[i] = True
            else:
                # Already fading out - check if fully faded and can be removed
                if self.current_opacities[i] < 0.01:
                    # Cloud has faded to invisible - REMOVE IT (don't recycle)
                    # This allows dynamic cloud count adjustment
                    # Mark for deletion by setting lifetime to negative
                    self.lifetime[i] = -1.0
        
        # Remove clouds marked for deletion (those with lifetime < 0)
        if len(self.lifetime) > 0:
            keep_mask = self.lifetime >= 0
            if not np.all(keep_mask):
                num_removed = np.sum(~keep_mask)
                # Remove from all arrays
                self.positions = self.positions[keep_mask]
                self.speeds = self.speeds[keep_mask]
                self.base_opacities = self.base_opacities[keep_mask]
                self.current_opacities = self.current_opacities[keep_mask]
                self.sizes = self.sizes[keep_mask]
                self.z_indices = self.z_indices[keep_mask]
                self.turbulence_phases = self.turbulence_phases[keep_mask]
                self.turbulence_speeds = self.turbulence_speeds[keep_mask]
                self.turbulence_amounts = self.turbulence_amounts[keep_mask]
                self.subpixel_offsets = self.subpixel_offsets[keep_mask]
                self.noise_offsets = self.noise_offsets[keep_mask]
                
                # For texture indices, also clean up unused textures
                removed_tex_indices = self.texture_indices[~keep_mask]
                self.texture_indices = self.texture_indices[keep_mask]
                
                # Delete GPU textures that are no longer used by any cloud
                for tex_idx in removed_tex_indices:
                    if tex_idx not in self.texture_indices:
                        if tex_idx < len(self.texture_ids):
                            glDeleteTextures(1, [self.texture_ids[tex_idx]])
                            # Mark as deleted (set to 0)
                            self.texture_ids[tex_idx] = 0
                
                self.is_fading_out = self.is_fading_out[keep_mask]
                self.lifetime = self.lifetime[keep_mask]
                
                print(f"Removed {num_removed} fully faded cloud(s)")

    def render(self, state: Dict):
        """Render all clouds with seamless wrapping using duplicates (vectorized)"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        # Calculate actual cloud positions with subpixel offsets
        actual_positions = self.positions - self.subpixel_offsets
        
        # Calculate actual cloud widths (vectorized)
        cloud_widths = np.array([self.cloud_textures[idx][0] * self.sizes[i] 
                                  for i, idx in enumerate(self.texture_indices)], dtype=np.float32)
        
        # Identify clouds near boundaries that need duplicates
        # For left edge: check if cloud position is within wrap_margin
        left_edge_mask = actual_positions[:, 0] < self.wrap_margin
        
        # For right edge: check if cloud's RIGHT edge (position + width) is approaching right boundary
        right_edge_mask = (actual_positions[:, 0] + cloud_widths) > (self.viewport.width - self.wrap_margin)
        
        # DEBUG: Print wrapping info
        # if np.any(left_edge_mask) or np.any(right_edge_mask):
        #     print(f"\n=== CLOUD WRAPPING DEBUG ===")
        #     print(f"Viewport width: {self.viewport.width}, Wrap margin: {self.wrap_margin}")
        #     print(f"Left edge triggers: {np.sum(left_edge_mask)}")
        #     print(f"Right edge triggers: {np.sum(right_edge_mask)}")
        #     if np.any(left_edge_mask):
        #         left_idx = np.where(left_edge_mask)[0]
        #         for idx in left_idx:
        #             print(f"  Cloud {idx}: x={actual_positions[idx, 0]:.1f}, width={cloud_widths[idx]:.1f} (LEFT EDGE)")
        #     if np.any(right_edge_mask):
        #         right_idx = np.where(right_edge_mask)[0]
        #         for idx in right_idx:
        #             print(f"  Cloud {idx}: x={actual_positions[idx, 0]:.1f}, right_edge={actual_positions[idx, 0] + cloud_widths[idx]:.1f}, width={cloud_widths[idx]:.1f} (RIGHT EDGE)")
        
        # Create duplicate positions for seamless wrapping
        duplicate_positions_left = []
        duplicate_indices_left = []
        duplicate_positions_right = []
        duplicate_indices_right = []
        
        if np.any(left_edge_mask):
            # Clouds near left edge need duplicates on the right
            left_indices = np.where(left_edge_mask)[0]
            duplicate_pos = actual_positions[left_indices].copy()
            duplicate_pos[:, 0] += self.viewport.width  # Shift to right side
            duplicate_positions_right.append(duplicate_pos)
            duplicate_indices_right.append(left_indices)
        
        if np.any(right_edge_mask):
            # Clouds near right edge need duplicates on the left
            right_indices = np.where(right_edge_mask)[0]
            duplicate_pos = actual_positions[right_indices].copy()
            duplicate_pos[:, 0] -= self.viewport.width  # Shift to left side
            duplicate_positions_left.append(duplicate_pos)
            duplicate_indices_left.append(right_indices)
        
        # Combine primary clouds with duplicates
        all_positions = [actual_positions]
        all_indices = [np.arange(len(actual_positions))]
        
        if duplicate_positions_right:
            all_positions.extend(duplicate_positions_right)
            all_indices.extend(duplicate_indices_right)
        
        if duplicate_positions_left:
            all_positions.extend(duplicate_positions_left)
            all_indices.extend(duplicate_indices_left)
        
        combined_positions = np.vstack(all_positions)
        combined_indices = np.concatenate(all_indices)
        
        # Get attributes for all clouds (primary + duplicates reference the same attributes)
        combined_sizes = self.sizes[combined_indices]
        combined_opacities = self.current_opacities[combined_indices]
        combined_z_indices = self.z_indices[combined_indices]
        combined_texture_indices = self.texture_indices[combined_indices]
        
        # Sort back-to-front for proper alpha blending
        sort_indices = np.argsort(combined_z_indices)
        
        glUseProgram(self.shader)
        
        # Update uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "noiseTime")
        if loc != -1:
            glUniform1f(loc, self.noise_time)
        
        loc = glGetUniformLocation(self.shader, "fadeFactor")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        glBindVertexArray(self.VAO)
        
        # Disable depth writes for proper alpha blending
        # Clouds should blend with each other, not block each other
        glDepthMask(GL_FALSE)
        
        # Render each cloud (sorted back-to-front)
        # Note: Unlike rain which uses single instanced call, clouds need per-texture rendering
        for sort_idx in sort_indices:
            cloud_idx = combined_indices[sort_idx]
            tex_idx = combined_texture_indices[sort_idx]
            tex_width, tex_height = self.cloud_textures[tex_idx]
            
            # Build instance data for this cloud
            instance_data = np.array([
                combined_positions[sort_idx, 0],
                combined_positions[sort_idx, 1],
                self.cloud_depth,
                tex_width, tex_height,
                combined_sizes[sort_idx],
                combined_opacities[sort_idx]
            ], dtype=np.float32)
            
            # Upload instance data
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
            glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
            
            # Setup attributes
            stride = 7 * 4
            
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribDivisor(1, 1)
            
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(2)
            glVertexAttribDivisor(2, 1)
            
            glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
            glEnableVertexAttribArray(3)
            glVertexAttribDivisor(3, 1)
            
            glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
            glEnableVertexAttribArray(4)
            glVertexAttribDivisor(4, 1)
            
            # Bind cloud texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[tex_idx])
            loc = glGetUniformLocation(self.shader, "cloudTexture")
            if loc != -1:
                glUniform1i(loc, 0)
            
                        # Draw this cloud
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, 1)
        
        # Re-enable depth writes for other effects
        glDepthMask(GL_TRUE)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up GPU resources"""
        # Delete textures
        if self.texture_ids:
            glDeleteTextures(len(self.texture_ids), self.texture_ids)
            self.texture_ids = []
            self.cloud_textures = []
        
        # Call parent cleanup for VAO/VBO/EBO
        super().cleanup()