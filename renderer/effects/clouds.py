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
from renderer.fan_coords import FanCoords

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
            # Always allocate max buffer for high cloudiness, actual spawning scales with cloudyness
            cloud_effect = viewport.add_effect(
                CloudEffectGPU,
                density=density,
                max_clouds=40  # Fixed allocation for dynamic cloudiness changes
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
        self.render_priority = 6  # Render AFTER BART map (5) so clouds blend over background
        self._fan = FanCoords(viewport.width, viewport.height)

        # GPU buffers
        self.cloud_ssbo = None
        self.compute_shader = None

        # CPU-side mirror of cloud data for rendering (avoiding GPU readback)
        self.cpu_cloud_data = np.zeros((max_clouds, 24), dtype=np.float32)
        self.cpu_cloud_data[:, 9] = -1.0  # Mark all as inactive initially
        self.num_clouds = 0

        # Timing
        self.global_time = 0.0
        self.last_fade_time = 0.0  # Track when we last marked a cloud for fading
        
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
        """Geometry shader for automatic edge wrapping with fan compensation"""
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
            float depth;
        } gs_in[];

        out vec2 texCoord;
        out float fragOpacity;
        out vec2 fragNoiseSeed;

        uniform vec2 resolution;
        uniform float wrapMargin;
        uniform float u_inner_r_ft;
        uniform float u_outer_r_ft;

        // Fan arc-width compensation: scale cloud width based on current Y position
        // so clouds appear equal physical size on the fan regardless of radial position.
        // Cloud Y is flipped: Y=0 = top = outer radius, Y=resolution.y = bottom = inner.
        // Inner radius = 4ft, outer = 20.6ft, ratio = 5.15x
        float fanWidthScale(float cloudY) {
            // normalized: 0 at top (outer), 1 at bottom (inner)
            float t = clamp(cloudY / resolution.y, 0.0, 1.0);
            // Top (t=0, outer radius): pixels span wide arc, shrink cloud
            // Bottom (t=1, inner radius): pixels span narrow arc, keep full size
            return mix(0.2, 1.0, t);
        }

        void emitQuad(vec2 basePos, vec2 size, float opacity, vec2 seed, float depth) {
            // Bottom-left
            vec2 pos = basePos;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, depth / 100.0, 1.0);
            texCoord = vec2(0.0, 0.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();

            // Bottom-right
            pos = basePos + vec2(size.x, 0.0);
            clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, depth / 100.0, 1.0);
            texCoord = vec2(1.0, 0.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();

            // Top-left
            pos = basePos + vec2(0.0, size.y);
            clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, depth / 100.0, 1.0);
            texCoord = vec2(0.0, 1.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();

            // Top-right
            pos = basePos + size;
            clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            gl_Position = vec4(clipPos, depth / 100.0, 1.0);
            texCoord = vec2(1.0, 1.0);
            fragOpacity = opacity;
            fragNoiseSeed = seed;
            EmitVertex();

            EndPrimitive();
        }

        void main() {
            vec2 cloudPos = gl_in[0].gl_Position.xy;

            // Apply fan compensation to width based on current Y position
            float wScale = fanWidthScale(cloudPos.y);
            vec2 cloudSize = gs_in[0].size * gs_in[0].scale;
            cloudSize.x *= wScale;
            float opacity = gs_in[0].opacity;
            vec2 seed = gs_in[0].noiseSeed;
            float depth = gs_in[0].depth;
            
            // Always emit primary cloud
            emitQuad(cloudPos, cloudSize, opacity, seed, depth);
            
            // Check if we need wrap duplicate on left edge
            if (cloudPos.x < wrapMargin) {
                vec2 wrapPos = cloudPos + vec2(resolution.x, 0.0);
                emitQuad(wrapPos, cloudSize, opacity, seed, depth);
            }
            
            // Check if we need wrap duplicate on right edge
            if (cloudPos.x + cloudSize.x > resolution.x - wrapMargin) {
                vec2 wrapPos = cloudPos - vec2(resolution.x, 0.0);
                emitQuad(wrapPos, cloudSize, opacity, seed, depth);
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
        layout(location = 5) in float depth;
        
        out VS_OUT {
            vec2 size;
            float scale;
            float opacity;
            vec2 noiseSeed;
            float depth;
        } vs_out;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vs_out.size = size;
            vs_out.scale = scale;
            vs_out.opacity = opacity;
            vs_out.noiseSeed = noiseSeed;
            vs_out.depth = depth;
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

            // Discard low-alpha fragments so they don't write to depth buffer
            // and create visible edges around clouds
            if (finalAlpha < 0.05) discard;

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
        #            turb_amount(3) + noise_seed(2) + depth(1) + max_lifetime(1) = 24 floats
        cloud_data_size = self.max_clouds * 24 * 4  # 24 floats * 4 bytes
        
        self.cloud_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.cloud_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, cloud_data_size, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.cloud_ssbo)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        print(f"    ✓ Created SSBO: {cloud_data_size} bytes for {self.max_clouds} clouds")
        
        # Initialize with empty clouds
        initial_data = np.zeros(self.max_clouds * 24, dtype=np.float32)
        # Mark all as inactive (lifetime = -1)
        for i in range(self.max_clouds):
            initial_data[i * 24 + 9] = -1.0  # lifetime field
        
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
        start_x = np.random.uniform(0, self.viewport.width)
        start_y = np.random.uniform(0, self.viewport.height)
        # Clouds are atmospheric background — render behind BART lines (z=50)
        # and trains (z=25) but in front of the geo map (z=98) and stars (z=99.99)
        depth = np.random.uniform(85, 95)

        # Width/height in base pixel units — fan compensation is done in
        # the geometry shader based on current Y position each frame.
        width = np.random.uniform(30, 80)
        height = np.random.uniform(15, 60)
        
        speed_type = np.random.random()
        if speed_type < 0.3:
            speed = np.random.uniform(0.2, 0.8)
        elif speed_type < 0.7:
            speed = np.random.uniform(0.8, 2.0)
        else:
            speed = np.random.uniform(2.0, 4.0)

        size_scale = np.random.uniform(0.5, 0.9)
        size_factor = 1.5 - size_scale
        wind_sens = size_factor * np.random.uniform(0.5, 1.5)
        
        base_opacity = np.random.uniform(0.7, 1.0)
        max_lifetime = np.random.uniform(40, 100)  # Variable lifespan
        
        # Randomize starting lifetime to prevent synchronized fading
        # Give clouds a random "age" between 0 and 30 seconds
        starting_lifetime = np.random.uniform(0, 30)
        
        # Build cloud data
        cloud_data = np.array([
            start_x, start_y,           # position
            width, height,              # size
            speed,                      # speed
            wind_sens,                  # wind sensitivity
            base_opacity, 0.2,          # base opacity, current opacity (start at 0.2 for faster fade-in)
            size_scale,                 # scale
            starting_lifetime,          # lifetime (randomized starting age)
            0.0,                        # is_fading (not fading)
            *np.random.uniform(0, 2*np.pi, 3),  # turb_phase
            *np.random.uniform(0.1, 0.3, 3),    # turb_speed
            np.random.uniform(0.5, 1.5),
            np.random.uniform(0.3, 0.8),
            np.random.uniform(0.1, 0.3),        # turb_amount
            *np.random.uniform(0, 10, 2),       # noise_seed
            depth,                      # depth
            max_lifetime,               # max lifetime
        ], dtype=np.float32)
        
        # Upload to SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.cloud_ssbo)
        offset = inactive_idx * 24 * 4
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
            # Aggressive spawning when below target - spawn multiple per second
            deficit = target_count - self.num_clouds
            spawn_rate = min(2.0, deficit * 0.3)  # Up to 2 clouds/sec, faster when further from target
            spawn_chance = dt * spawn_rate
            
            # Try to spawn (might spawn multiple in one frame if dt is large)
            while spawn_chance > 0:
                if np.random.random() < min(spawn_chance, 1.0):
                    self._spawn_cloud_gpu()
                spawn_chance -= 1.0
                
        elif self.num_clouds > target_count:
            # Gradually mark excess clouds for fading (not all at once)
            # Only mark 1 cloud every 2-4 seconds to create smooth, natural reduction
            time_since_last_fade = self.global_time - self.last_fade_time
            fade_interval = np.random.uniform(2.0, 4.0)  # Random interval for natural variation
            
            if time_since_last_fade >= fade_interval:
                active_indices = np.where(self.cpu_cloud_data[:, 9] >= 0)[0]
                not_fading_mask = self.cpu_cloud_data[active_indices, 10] < 0.5
                not_fading = active_indices[not_fading_mask]
                
                if len(not_fading) > 0:
                    # Mark just one random cloud for fading
                    fade_idx = np.random.choice(not_fading)
                    self.cpu_cloud_data[fade_idx, 10] = 1.0
                    self.last_fade_time = self.global_time
        
        # Update physics on CPU for active clouds using vectorized operations
        active_mask = self.cpu_cloud_data[:, 9] >= 0
        if not np.any(active_mask):
            return
        
        # Get all active clouds at once
        active_clouds = self.cpu_cloud_data[active_mask]
        
        # Update lifetime (vectorized)
        active_clouds[:, 9] += dt
        
        # Update turbulence phases (vectorized)
        active_clouds[:, 11:14] += active_clouds[:, 14:17] * dt
        
        # Calculate global wave (scalar, same for all clouds)
        global_wave_y = np.sin(self.global_time * 0.05) * 0.3 + np.sin(self.global_time * 0.13) * 0.1
        
        # Fan-compensate horizontal speed: inner radius pixels are narrower,
        # so clouds there need more px/sec for the same physical speed.
        # Cloud Y is flipped: Y=0=outer, Y=height=inner.
        cloud_v = 1.0 - np.clip(active_clouds[:, 1] / self.viewport.height, 0, 1)
        cloud_r = self._fan.inner_r_ft + cloud_v * (self._fan.outer_r_ft - self._fan.inner_r_ft)
        mid_r = (self._fan.inner_r_ft + self._fan.outer_r_ft) * 0.5
        speed_comp = mid_r / np.maximum(cloud_r, 0.1)  # >1 at inner, <1 at outer

        # Update position with wind and turbulence (vectorized)
        wind_effect = self.wind * active_clouds[:, 5]  # wind_sens
        active_clouds[:, 0] += (active_clouds[:, 4] + wind_effect) * speed_comp * dt  # position.x
        active_clouds[:, 1] += global_wave_y * dt  # position.y
        
        # Horizontal wrapping (vectorized)
        wrap_left = active_clouds[:, 0] < 0.0
        active_clouds[wrap_left, 0] += self.viewport.width
        wrap_right = active_clouds[:, 0] >= self.viewport.width
        active_clouds[wrap_right, 0] -= self.viewport.width
        
        # Clamp vertical (vectorized)
        active_clouds[:, 1] = np.clip(active_clouds[:, 1], 0, self.viewport.height)
        
        # Smooth opacity transitions (vectorized)
        is_fading = active_clouds[:, 10] > 0.5
        target_opacity = np.where(is_fading, 0.0, active_clouds[:, 6])  # base_opacity or 0
        opacity_diff = target_opacity - active_clouds[:, 7]
        
        # Update opacity where diff is significant
        needs_update = np.abs(opacity_diff) > 0.01
        transition_speed = np.where(is_fading, 0.4, 0.3)
        active_clouds[needs_update, 7] += opacity_diff[needs_update] * transition_speed[needs_update] * dt
        active_clouds[:, 7] = np.clip(active_clouds[:, 7], 0, 1)
        
        # Mark for removal if fully faded (vectorized)
        fully_faded = (active_clouds[:, 10] > 0.5) & (active_clouds[:, 7] < 0.01)
        active_clouds[fully_faded, 9] = -1.0  # Mark inactive
        removed_count = np.sum(fully_faded)
        self.num_clouds -= removed_count
        
        # Natural lifecycle: fade clouds after their individual max_lifetime (40-100 seconds)
        # This ensures clouds eventually cycle regardless of position
        max_lifetimes = active_clouds[:, 23]  # max_lifetime field
        should_fade = (active_clouds[:, 9] > max_lifetimes) & (active_clouds[:, 10] < 0.5)
        active_clouds[should_fade, 10] = 1.0  # Start fading
        
        # Write back active clouds to main array
        self.cpu_cloud_data[active_mask] = active_clouds
        
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
            return

        # Build instance data for rendering
        # position(2), size(2), scale(1), opacity(1), noiseSeed(2), depth(1)
        instance_data = np.column_stack([
            active_clouds[:, 0:2],   # position
            active_clouds[:, 2:4],   # size (width, height)
            active_clouds[:, 8],     # scale
            active_clouds[:, 7],     # current opacity
            active_clouds[:, 20:22], # noise seed
            active_clouds[:, 22],    # depth
        ]).astype(np.float32).flatten()
        
        glUseProgram(self.shader)

        # Set uniforms
        glUniform2f(glGetUniformLocation(self.shader, "resolution"),
                   float(self.viewport.width), float(self.viewport.height))
        glUniform1f(glGetUniformLocation(self.shader, "noiseTime"), self.global_time)
        glUniform1f(glGetUniformLocation(self.shader, "fadeFactor"), self.fade_factor)
        glUniform1f(glGetUniformLocation(self.shader, "wrapMargin"), 60.0)
        # Fan coord uniforms for geometry shader width compensation
        self._fan.set_uniforms(self.shader)
        
        # Upload instance data
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Setup vertex attributes (9 floats per instance)
        stride = 9 * 4
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
        
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(5)
        
        # Draw points (geometry shader will expand to quads)
        # NO state toggling — global state handles depth test + blend
        glDrawArrays(GL_POINTS, 0, len(active_clouds))

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
