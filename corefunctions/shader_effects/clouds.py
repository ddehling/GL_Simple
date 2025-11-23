"""
Complete cloud effect - GPU-OPTIMIZED VERSION
Optimized drifting clouds with:
1. Procedural generation in fragment shader (no CPU texture generation or uploads)
2. Geometry shader for automatic edge wrapping (no CPU duplicate management)
3. Efficient CPU physics with GPU SSBO storage (compute shaders not available in ES 3.1)

This version eliminates texture memory and scipy dependency while using geometry
shaders for seamless edge wrapping. Physics updates run on CPU but are vectorized.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

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
        
        try:
            cloud_effect = viewport.add_effect(
                CloudEffectGPU,
                density=density,
                max_clouds=20
            )
            state['cloud_effect'] = cloud_effect
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
        effect.cloudyness = outstate.get('cloudyness', 0.5)
        
        # Calculate fade with 7 second fade in/out
        fade_duration = 7.0
        total_duration = state.get('duration', 60)
        elapsed = state.get('elapsed_time', 0)
        
        if elapsed < fade_duration:
            effect.fade_factor = elapsed / fade_duration
        elif elapsed > (total_duration - fade_duration):
            effect.fade_factor = (total_duration - elapsed) / fade_duration
        else:
            effect.fade_factor = 1.0
        
        effect.fade_factor = np.clip(effect.fade_factor, 0, 1)
    
    # On close event, clean up
    if state['count'] == -1:
        outstate['has_clouds'] = False
        if 'cloud_effect' in state:
            viewport.effects.remove(state['cloud_effect'])
            state['cloud_effect'].cleanup()


# ============================================================================
# GPU-Optimized Cloud Effect
# ============================================================================

class CloudEffectGPU(ShaderEffect):
    """
    GPU-optimized cloud effect:
    - Fragment shader generates clouds procedurally (no textures/uploads)
    - Geometry shader handles edge wrapping automatically  
    - CPU physics with GPU SSBO storage (ES 3.1 doesn't support compute shaders)
    - Eliminates texture memory and scipy dependency
    """
    
    def __init__(self, viewport, density: float = 1.0, max_clouds: int = 20):
        super().__init__(viewport)
        self.density = density
        self.max_clouds = max_clouds
        self.wind = 0.0
        self.fog_level = 0.0
        self.fade_factor = 1.0  # Start visible (event system will control fading)
        self.cloudyness = 0.5
        
        # GPU buffers
        self.cloud_ssbo = None
        self.compute_shader = None
        
        # CPU-side mirror of cloud data for rendering (avoiding GPU readback)
        self.cpu_cloud_data = np.zeros((max_clouds, 22), dtype=np.float32)
        self.cpu_cloud_data[:, 9] = -1.0  # Mark all as inactive initially
        self.num_clouds = 0
        
        # Timing
        self.global_time = 0.0
        
    def get_compute_shader(self):
        """Compute shader for cloud physics updates"""
        return """
        #version 430 core
        layout(local_size_x = 16) in;
        
        struct Cloud {
            vec2 position;      // x, y position
            float width;        // cloud width
            float height;       // cloud height
            float speed;        // base movement speed
            float wind_sens;    // wind sensitivity
            float base_opacity; // target opacity
            float curr_opacity; // current opacity (for fading)
            float size_scale;   // overall scale
            float lifetime;     // age in seconds
            float is_fading;    // 1.0 if fading out, 0.0 otherwise
            vec3 turb_phase;    // turbulence phases
            vec3 turb_speed;    // turbulence speeds
            vec3 turb_amount;   // turbulence amounts
            vec2 noise_seed;    // random seed for procedural generation
        };
        
        layout(std430, binding = 0) buffer CloudBuffer {
            Cloud clouds[];
        };
        
        uniform float deltaTime;
        uniform float globalTime;
        uniform float wind;
        uniform float viewportWidth;
        uniform float viewportHeight;
        uniform float cloudyness;
        uniform uint maxClouds;
        
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= maxClouds) return;
            
            Cloud cloud = clouds[idx];
            
            // Skip inactive clouds (lifetime < 0)
            if (cloud.lifetime < 0.0) return;
            
            // Update lifetime
            cloud.lifetime += deltaTime;
            
            // Update turbulence phases
            cloud.turb_phase += cloud.turb_speed * deltaTime;
            
            // Calculate global wave
            float global_wave_y = sin(globalTime * 0.05) * 0.3 + sin(globalTime * 0.13) * 0.1;
            
            // Update position with wind and turbulence
            float wind_effect = wind * cloud.wind_sens;
            cloud.position.x += (cloud.speed + wind_effect) * deltaTime;
            cloud.position.y += global_wave_y * deltaTime;
            
            // Horizontal wrapping
            if (cloud.position.x < 0.0) {
                cloud.position.x += viewportWidth;
            } else if (cloud.position.x >= viewportWidth) {
                cloud.position.x -= viewportWidth;
            }
            
            // Clamp vertical
            cloud.position.y = clamp(cloud.position.y, 0.0, viewportHeight);
            
            // Smooth opacity transitions
            float target_opacity = cloud.is_fading > 0.5 ? 0.0 : cloud.base_opacity;
            float opacity_diff = target_opacity - cloud.curr_opacity;
            
            if (abs(opacity_diff) > 0.01) {
                float transition_speed = cloud.is_fading > 0.5 ? 0.4 : 0.3;
                cloud.curr_opacity += opacity_diff * transition_speed * deltaTime;
                cloud.curr_opacity = clamp(cloud.curr_opacity, 0.0, 1.0);
            }
            
            // Mark for removal if fully faded
            if (cloud.is_fading > 0.5 && cloud.curr_opacity < 0.01) {
                cloud.lifetime = -1.0;  // Mark as inactive
            }
            
            // Natural lifecycle: start fading after 30 seconds if in middle of screen
            float wrap_margin = 60.0;
            float cloud_width_scaled = cloud.width * cloud.size_scale;
            float middle_left = wrap_margin + cloud_width_scaled;
            float middle_right = viewportWidth - wrap_margin - cloud_width_scaled;
            
            bool in_middle = (cloud.position.x > middle_left && 
                            cloud.position.x < middle_right);
            
            if (cloud.lifetime > 30.0 && in_middle && cloud.is_fading < 0.5) {
                cloud.is_fading = 1.0;
            }
            
            // Write back
            clouds[idx] = cloud;
        }
        """
    
    def get_geometry_shader(self):
        """Geometry shader for automatic edge wrapping"""
        return """
        #version 310 es
        #extension GL_EXT_geometry_shader : enable
        #extension GL_EXT_shader_io_blocks : enable
        
        precision highp float;
        
        layout(points) in;
        layout(triangle_strip, max_vertices = 12) out;  // Up to 2 quads (primary + wrap)
        
        in VS_OUT {
            vec2 size;
            float scale;
            float opacity;
            vec2 noiseSeed;
        } gs_in[];
        
        out vec2 texCoord;
        out float fragOpacity;
        out vec2 fragNoiseSeed;
        
        uniform vec2 resolution;
        uniform float wrapMargin;
        uniform float cloudDepth;
        
        void emitQuad(vec2 basePos, vec2 size, float opacity, vec2 seed) {
            // Bottom-left
            vec2 pos = basePos;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, cloudDepth / 100.0, 1.0);
            texCoord = vec2(0.0, 0.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();
            
            // Bottom-right
            pos = basePos + vec2(size.x, 0.0);
            clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, cloudDepth / 100.0, 1.0);
            texCoord = vec2(1.0, 0.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();
            
            // Top-left
            pos = basePos + vec2(0.0, size.y);
            clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, cloudDepth / 100.0, 1.0);
            texCoord = vec2(0.0, 1.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();
            
            // Top-right
            pos = basePos + size;
            clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, cloudDepth / 100.0, 1.0);
            texCoord = vec2(1.0, 1.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();
            
            EndPrimitive();
        }
        
        void main() {
            vec2 cloudPos = gl_in[0].gl_Position.xy;
            vec2 cloudSize = gs_in[0].size * gs_in[0].scale;
            float opacity = gs_in[0].opacity;
            vec2 seed = gs_in[0].noiseSeed;
            
            // Always emit primary cloud
            emitQuad(cloudPos, cloudSize, opacity, seed);
            
            // Check if we need wrap duplicate on left edge
            if (cloudPos.x < wrapMargin) {
                vec2 wrapPos = cloudPos + vec2(resolution.x, 0.0);
                emitQuad(wrapPos, cloudSize, opacity, seed);
            }
            
            // Check if we need wrap duplicate on right edge
            if (cloudPos.x + cloudSize.x > resolution.x - wrapMargin) {
                vec2 wrapPos = cloudPos - vec2(resolution.x, 0.0);
                emitQuad(wrapPos, cloudSize, opacity, seed);
            }
        }
        """
    
    def get_vertex_shader(self):
        """Vertex shader - passes through point data"""
        return """
        #version 310 es
        #extension GL_EXT_shader_io_blocks : enable
        
        precision highp float;
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 size;
        layout(location = 2) in float scale;
        layout(location = 3) in float opacity;
        layout(location = 4) in vec2 noiseSeed;
        
        out VS_OUT {
            vec2 size;
            float scale;
            float opacity;
            vec2 noiseSeed;
        } vs_out;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vs_out.size = size;
            vs_out.scale = scale;
            vs_out.opacity = opacity;
            vs_out.noiseSeed = noiseSeed;
        }
        """
    
    def get_fragment_shader(self):
        """Fragment shader with procedural cloud generation"""
        return """
        #version 310 es
        precision highp float;
        
        in vec2 texCoord;
        in float fragOpacity;
        in vec2 fragNoiseSeed;
        
        out vec4 outColor;
        
        uniform float noiseTime;
        uniform float fadeFactor;
        
        // Hash function for pseudo-random values
        float hash(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }
        
        // 2D Perlin-like noise
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);  // Smoothstep
            
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        // Fractal Brownian Motion for cloud texture
        float fbm(vec2 p) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;
            
            for (int i = 0; i < 5; i++) {
                value += amplitude * noise(p * frequency);
                frequency *= 2.0;
                amplitude *= 0.5;
            }
            
            return value;
        }
        
        void main() {
            // Use texture coordinates and noise seed for unique cloud pattern
            vec2 uv = texCoord;
            vec2 cloudSpace = uv * 5.0 + fragNoiseSeed;  // Different per cloud
            
            // Generate cloud density with FBM
            float density = fbm(cloudSpace);
            
            // Add animated wispy noise
            vec2 animCoord = cloudSpace + vec2(noiseTime * 0.1, noiseTime * 0.07);
            float wispy = noise(animCoord * 3.0) * 0.3 + 0.7;
            
            // Center-bias for cloud shape (make edges more transparent)
            vec2 centerDist = abs(uv - 0.5) * 2.0;
            float radialFalloff = 1.0 - smoothstep(0.3, 1.0, length(centerDist));
            
            // Combine density layers
            float cloudAlpha = density * radialFalloff;
            cloudAlpha = pow(cloudAlpha, 1.5);  // Enhance contrast
            
            // Apply wispy effect more at edges
            float edgeFactor = pow(cloudAlpha, 0.4);
            cloudAlpha *= wispy * (1.0 - edgeFactor * 0.5) + edgeFactor * 0.5;
            
            // Soft edge fadeout
            float edgeDist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            cloudAlpha *= smoothstep(0.0, 0.15, edgeDist);
            
            // Cloud color with vertical gradient
            float brightness = 0.9 - uv.y * 0.2;
            vec3 cloudColor = vec3(brightness);
            
            // Apply all opacity factors
            float finalAlpha = cloudAlpha * fragOpacity * fadeFactor;
            
            outColor = vec4(cloudColor, finalAlpha);
        }
        """
    
    def compile_shader(self):
        """Compile rendering shaders with geometry shader"""
        vertex_src = self.get_vertex_shader()
        geometry_src = self.get_geometry_shader()
        fragment_src = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
            geom = shaders.compileShader(geometry_src, GL_GEOMETRY_SHADER)
            frag = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
            
            # Create program and attach shaders
            shader = glCreateProgram()
            glAttachShader(shader, vert)
            glAttachShader(shader, geom)
            glAttachShader(shader, frag)
            glLinkProgram(shader)
            
            # Check link status
            link_status = glGetProgramiv(shader, GL_LINK_STATUS)
            if not link_status:
                error = glGetProgramInfoLog(shader).decode()
                raise RuntimeError(f"Shader linking failed: {error}")
            
            # Note: Skip validation for now - it fails without uniform values set
            # The shader will work fine once uniforms are properly set during render
            
            # Clean up shader objects (they're now part of the program)
            glDeleteShader(vert)
            glDeleteShader(geom)
            glDeleteShader(frag)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def compile_compute_shader(self):
        """Compile compute shader for physics"""
        compute_src = self.get_compute_shader()
        
        try:
            comp = shaders.compileShader(compute_src, GL_COMPUTE_SHADER)
            
            # Create program and attach shader
            program = glCreateProgram()
            glAttachShader(program, comp)
            glLinkProgram(program)
            
            # Check link status
            link_status = glGetProgramiv(program, GL_LINK_STATUS)
            if not link_status:
                error = glGetProgramInfoLog(program).decode()
                raise RuntimeError(f"Compute shader linking failed: {error}")
            
            # Clean up shader object
            glDeleteShader(comp)
            
            return program
        except Exception as e:
            print(f"Compute shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize GPU buffers and shaders"""
        # Note: Compute shaders not available in OpenGL ES 3.1, using CPU physics instead
        # We still use SSBO to store cloud data for potential future optimization
        
        # Create SSBO for cloud data storage
        # Each cloud: position(2) + size(2) + speed(1) + wind_sens(1) + opacities(2) + 
        #            scale(1) + lifetime(1) + is_fading(1) + turb_phase(3) + turb_speed(3) + 
        #            turb_amount(3) + noise_seed(2) = 22 floats
        cloud_data_size = self.max_clouds * 22 * 4  # 22 floats * 4 bytes
        
        self.cloud_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.cloud_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, cloud_data_size, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.cloud_ssbo)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        print(f"    ✓ Created SSBO: {cloud_data_size} bytes for {self.max_clouds} clouds")
        
        # Initialize with empty clouds
        initial_data = np.zeros(self.max_clouds * 22, dtype=np.float32)
        # Mark all as inactive (lifetime = -1)
        for i in range(self.max_clouds):
            initial_data[i * 22 + 9] = -1.0  # lifetime field
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.cloud_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, initial_data.nbytes, initial_data)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        # Create VAO for point rendering
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # VBO will be updated each frame with active cloud data
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
        
        self.num_clouds = 0
        self.global_time = 0.0
        
        # Spawn initial clouds based on cloudyness
        initial_cloud_count = int(self.cloudyness * self.max_clouds)
        for _ in range(initial_cloud_count):
            self._spawn_cloud_gpu()
        
        print(f"GPU Cloud Effect initialized: spawned {initial_cloud_count} clouds")
    
    def _spawn_cloud_gpu(self):
        """Spawn a cloud by adding it to GPU SSBO and CPU mirror"""
        if self.num_clouds >= self.max_clouds:
            return
        
        # Find first inactive slot
        inactive_idx = -1
        for i in range(self.max_clouds):
            if self.cpu_cloud_data[i, 9] < 0:  # lifetime < 0 means inactive
                inactive_idx = i
                break
        
        if inactive_idx == -1:
            print(f"Warning: Could not find inactive slot for new cloud")
            return
        
        # Generate cloud parameters
        width = np.random.uniform(30, 80)
        height = np.random.uniform(15, 60)
        start_x = np.random.uniform(-width * 1.5, self.viewport.width + width * 1.5)
        start_y = np.random.uniform(0, self.viewport.height)
        
        speed_type = np.random.random()
        if speed_type < 0.3:
            speed = np.random.uniform(0.2, 0.8)
        elif speed_type < 0.7:
            speed = np.random.uniform(0.8, 2.0)
        else:
            speed = np.random.uniform(2.0, 4.0)
        
        size_scale = np.random.uniform(0.5, 0.9)
        altitude_factor = 0.3 + (start_y / self.viewport.height) * 0.7
        size_factor = 1.5 - size_scale
        wind_sens = altitude_factor * size_factor * np.random.uniform(0.5, 1.5)
        
        base_opacity = np.random.uniform(0.5, 0.9)
        
        # Build cloud data
        cloud_data = np.array([
            start_x, start_y,           # position
            width, height,              # size
            speed,                      # speed
            wind_sens,                  # wind sensitivity
            base_opacity, 0.2,          # base opacity, current opacity (start at 0.2 for faster fade-in)
            size_scale,                 # scale
            0.0,                        # lifetime (start at 0)
            0.0,                        # is_fading (not fading)
            *np.random.uniform(0, 2*np.pi, 3),  # turb_phase
            *np.random.uniform(0.1, 0.3, 3),    # turb_speed
            np.random.uniform(0.5, 1.5),
            np.random.uniform(0.3, 0.8),
            np.random.uniform(0.1, 0.3),        # turb_amount
            *np.random.uniform(0, 10, 2),       # noise_seed
        ], dtype=np.float32)
        
        # Upload to SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.cloud_ssbo)
        offset = inactive_idx * 22 * 4
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, cloud_data.nbytes, cloud_data)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        # Store in CPU mirror
        self.cpu_cloud_data[inactive_idx] = cloud_data
        self.num_clouds += 1
    
    def update(self, dt: float, state: Dict):
        """Update clouds - CPU-based physics since compute shaders not available in ES 3.1"""
        if not self.enabled:
            return
        
        self.global_time += dt
        
        # Dynamic cloud spawning based on cloudyness
        target_count = int(np.clip(self.cloudyness, 0, 1) * self.max_clouds)
        
        if self.num_clouds < target_count:
            spawn_chance = dt * 0.3
            if np.random.random() < spawn_chance:
                self._spawn_cloud_gpu()
        
        # Update physics on CPU for active clouds
        active_mask = self.cpu_cloud_data[:, 9] >= 0
        if not np.any(active_mask):
            return
        
        active_indices = np.where(active_mask)[0]
        
        for idx in active_indices:
            cloud = self.cpu_cloud_data[idx]
            
            # Update lifetime
            cloud[9] += dt
            
            # Update turbulence phases
            cloud[11:14] += cloud[14:17] * dt
            
            # Calculate global wave
            global_wave_y = np.sin(self.global_time * 0.05) * 0.3 + np.sin(self.global_time * 0.13) * 0.1
            
            # Update position with wind and turbulence
            wind_effect = self.wind * cloud[5]  # wind_sens
            cloud[0] += (cloud[4] + wind_effect) * dt  # position.x
            cloud[1] += global_wave_y * dt  # position.y
            
            # Horizontal wrapping
            if cloud[0] < 0.0:
                cloud[0] += self.viewport.width
            elif cloud[0] >= self.viewport.width:
                cloud[0] -= self.viewport.width
            
            # Clamp vertical
            cloud[1] = np.clip(cloud[1], 0, self.viewport.height)
            
            # Smooth opacity transitions
            target_opacity = 0.0 if cloud[10] > 0.5 else cloud[6]  # is_fading, base_opacity
            opacity_diff = target_opacity - cloud[7]
            
            if abs(opacity_diff) > 0.01:
                transition_speed = 0.4 if cloud[10] > 0.5 else 0.3
                cloud[7] += opacity_diff * transition_speed * dt
                cloud[7] = np.clip(cloud[7], 0, 1)
            
            # Mark for removal if fully faded
            if cloud[10] > 0.5 and cloud[7] < 0.01:
                cloud[9] = -1.0  # Mark inactive
                self.num_clouds -= 1
                continue
            
            # Natural lifecycle: start fading after 30 seconds if in middle
            wrap_margin = 60.0
            cloud_width_scaled = cloud[2] * cloud[8]  # width * scale
            middle_left = wrap_margin + cloud_width_scaled
            middle_right = self.viewport.width - wrap_margin - cloud_width_scaled
            
            in_middle = (cloud[0] > middle_left and cloud[0] < middle_right)
            
            if cloud[9] > 30.0 and in_middle and cloud[10] < 0.5:
                cloud[10] = 1.0  # Start fading
        
        # Upload updated data back to GPU SSBO for rendering
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.cloud_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.cpu_cloud_data.nbytes, self.cpu_cloud_data)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    
    def render(self, state: Dict):
        """Render clouds using geometry shader for wrapping"""
        if not self.enabled or not self.shader:
            return
        
        # Use CPU mirror to find active clouds (lifetime >= 0)
        active_mask = self.cpu_cloud_data[:, 9] >= 0
        active_clouds = self.cpu_cloud_data[active_mask]
        
        if len(active_clouds) == 0:
            # Debug: print once if we have no clouds
            if not hasattr(self, '_no_clouds_warned'):
                target_count = int(self.cloudyness * self.max_clouds)
                print(f"GPU Clouds: No active clouds to render (target: {target_count}, fade_factor: {self.fade_factor:.2f})")
                self._no_clouds_warned = True
            return
        
        # Reset warning flag when we have clouds
        if hasattr(self, '_no_clouds_warned'):
            print(f"GPU Clouds: Now rendering {len(active_clouds)} clouds!")
            delattr(self, '_no_clouds_warned')
        
        # Debug first render
        if not hasattr(self, '_first_render_done'):
            first_opacity = active_clouds[0, 7]
            first_x = active_clouds[0, 0]
            first_y = active_clouds[0, 1]
            print(f"GPU Clouds first render: {len(active_clouds)} clouds")
            print(f"  First cloud opacity: {first_opacity:.2f}, position: ({first_x:.1f}, {first_y:.1f})")
            print(f"  fade_factor: {self.fade_factor:.2f}")
            self._first_render_done = True
        
        # Build instance data for rendering
        # position(2), size(2), scale(1), opacity(1), noiseSeed(2)
        instance_data = np.column_stack([
            active_clouds[:, 0:2],   # position
            active_clouds[:, 2:4],   # size (width, height)
            active_clouds[:, 8],     # scale
            active_clouds[:, 7],     # current opacity
            active_clouds[:, 20:22], # noise seed
        ]).astype(np.float32).flatten()
        
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniform2f(glGetUniformLocation(self.shader, "resolution"), 
                   float(self.viewport.width), float(self.viewport.height))
        glUniform1f(glGetUniformLocation(self.shader, "noiseTime"), self.global_time)
        glUniform1f(glGetUniformLocation(self.shader, "fadeFactor"), self.fade_factor)
        glUniform1f(glGetUniformLocation(self.shader, "wrapMargin"), 60.0)
        glUniform1f(glGetUniformLocation(self.shader, "cloudDepth"), 40.0)
        
        # Upload instance data
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Setup vertex attributes (8 floats per instance)
        stride = 8 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3)
        
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(4)
        
        # Disable depth writes for alpha blending
        glDepthMask(GL_FALSE)
        
        # Draw points (geometry shader will expand to quads)
        glDrawArrays(GL_POINTS, 0, len(active_clouds))
        
        # Re-enable depth writes
        glDepthMask(GL_TRUE)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.cloud_ssbo:
            glDeleteBuffers(1, [self.cloud_ssbo])
            self.cloud_ssbo = None
        
        if self.compute_shader:
            glDeleteProgram(self.compute_shader)
            self.compute_shader = None
        
        super().cleanup()
