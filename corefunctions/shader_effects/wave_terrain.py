"""
Wave Terrain - Raymarched animated wave terrain
Creates a 3D wave surface with animated undulating ridges
Based on heightmap raymarching with sinusoidal displacement
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_wave_terrain(state, outstate, field_of_view=60.0, camera_height=10.0,
                        wave_speed=1.0, wave_scale=3.0, spacing=0.2,
                        fade_duration=5.0, audio_reactive=False, audio_sensitivity=1.5):
    """
    Raymarched wave terrain effect compatible with EventScheduler
    
    Creates a 3D animated wave surface with sinusoidal displacement
    
    Usage:
        scheduler.schedule_event(0, 60, shader_wave_terrain, 
                               wave_speed=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        field_of_view: Camera FOV in degrees (default 60.0)
        camera_height: Height of camera above terrain (default 10.0)
        wave_speed: Speed of wave animation (default 1.0)
        wave_scale: Scale of wave displacement (default 3.0)
        spacing: Spacing between wave ridges (default 0.2)
        fade_duration: Duration of fade in/out in seconds (default 5.0)
        audio_reactive: Enable audio reactivity (default False)
        audio_sensitivity: How much audio affects waves (default 1.5)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing wave terrain effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                WaveTerrainEffect,
                field_of_view=field_of_view,
                camera_height=camera_height,
                wave_speed=wave_speed,
                wave_scale=wave_scale,
                spacing=spacing,
                audio_reactive=audio_reactive,
                audio_sensitivity=audio_sensitivity
            )
            state['effect'] = effect
            state['smoothed_bass'] = 0.0
            print(f"✓ Initialized shader wave_terrain for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize wave terrain: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters
    if 'effect' in state:
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = np.clip(fade_factor, 0, 1)
        
        # Audio reactivity - use bass frequencies to modulate wave intensity
        if audio_reactive and audio_data is not None:
            # Use bass bands (0-8) for wave modulation
            bass_energy = np.mean(audio_data['norm_short'][0][0:8])
            
            # Smooth the audio response
            smoothing = 0.15
            state['smoothed_bass'] = smoothing * bass_energy + (1 - smoothing) * state['smoothed_bass']
            
            # Apply to effect
            state['effect'].audio_intensity = state['smoothed_bass'] * audio_sensitivity
        else:
            state['effect'].audio_intensity = 0.0
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up wave terrain effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader wave_terrain for frame {frame_id}")

# ============================================================================
# Wave Terrain Effect - Post-Processing Raymarcher
# ============================================================================

class WaveTerrainEffect(ShaderEffect):
    """Fullscreen post-processing raymarched wave terrain"""
    
    def __init__(self, viewport, field_of_view: float = 60.0, camera_height: float = 10.0,
                 wave_speed: float = 1.0, wave_scale: float = 3.0, spacing: float = 0.2,
                 audio_reactive: bool = False, audio_sensitivity: float = 1.5):
        super().__init__(viewport)
        self.field_of_view = field_of_view
        self.camera_height = camera_height
        self.wave_speed = wave_speed
        self.wave_scale = wave_scale
        self.spacing = spacing
        self.audio_reactive = audio_reactive
        self.audio_sensitivity = audio_sensitivity
        self.fade_factor = 0.0
        self.audio_intensity = 0.0
        self.time = 0.0
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position * 2.0 - 1.0, 0.5, 1.0);
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform vec2 resolution;
        uniform float time;
        uniform float fieldOfView;
        uniform float cameraHeight;
        uniform float waveSpeed;
        uniform float waveScale;
        uniform float spacing;
        uniform float fadeAlpha;
        uniform float audioIntensity;
        
        #define MAX_MARCHING_STEPS 128
        #define MIN_FLOAT 1e-6
        #define MAX_FLOAT 1e6
        #define EPSILON 0.0001
        
        const float PI = acos(-1.0);
        
        // Ray direction calculation
        vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
            vec2 xy = fragCoord - size / 2.0;
            float z = size.y / tan(radians(fieldOfView) / 2.0);
            return normalize(vec3(xy, -z));
        }
        
        // View matrix calculation
        mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
            vec3 f = normalize(center - eye);
            vec3 s = normalize(cross(f, up));
            vec3 u = cross(s, f);
            return mat4(vec4(s, 0.0), vec4(u, 0.0), vec4(-f, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
        }
        
        // Height at position - creates wave pattern with Gaussian profile
        float heightAtPos(vec3 p) {
            // Audio modulates the wave amplitude
            float audioMod = 1.0 + audioIntensity * 0.5;
            
            // Gaussian parameters (animated over time and z-position)
            float height = waveScale * audioMod * 0.75;  // Peak height
            float width = 0.15;  // Standard deviation (controls spread) - reduced significantly
            float center = sin(p.z * 0.5 + time * waveSpeed) * (waveScale * audioMod * 0.125);  // Animated center position
            
            // Gaussian function: height * exp(-((x - center)^2) / (2 * width^2))
            float distance = p.x - center;
            return height * exp(-(distance * distance) / (2.0 * width * width));
        }
        
        // Subtraction operation for CSG
        float opSubtraction(float d1, float d2) {
            return max(-d1, d2);
        }
        
        // World SDF - defines the wave terrain geometry
        float world(vec3 p) {
            vec3 mp = p;
            float v = mod(mp.z, spacing) - spacing * 0.5;
            return opSubtraction(-p.y + heightAtPos(p), opSubtraction(v + 0.001, v - 0.001));
        }
        
        // Raymarching function
        float march(vec3 eye, vec3 marchingDirection) {
            const float precis = 0.001;
            float t = 0.0;
            
            for(int i = 0; i < MAX_MARCHING_STEPS; i++) {
                vec3 p = eye + marchingDirection * t;
                float hit = world(p);
                if(hit < precis) return t;
                t += hit * 0.25;
                if(t > 100.0) break; // Max distance
            }
            return -1.0;
        }
        
        // Calculate normal using gradient
        vec3 calcNormal(vec3 p) {
            const float h = 0.0001;
            const vec2 k = vec2(1, -1);
            return normalize(
                k.xyy * world(p + k.xyy * h) + 
                k.yyx * world(p + k.yyx * h) + 
                k.yxy * world(p + k.yxy * h) + 
                k.xxx * world(p + k.xxx * h)
            );
        }
        
        // Color calculation
        vec3 color(vec3 camPos, vec3 rayDir) {
            vec3 col = vec3(0.0);
            vec3 pos = camPos;
            
            float dis = march(pos, rayDir);
            if(dis >= 0.0) {
                pos += rayDir * dis;
                float h = heightAtPos(pos);
                
                // Create sharp edge effect
                float edge = smoothstep(0.05, 0.0, distance(pos.y, h - 0.05));
                
                // Calculate normal for lighting
                vec3 normal = calcNormal(pos);
                
                // Simple directional lighting
                vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
                float diffuse = max(0.0, dot(normal, lightDir));
                
                // Add ambient
                float ambient = 0.3;
                
                // Audio reactive color shift
                float hue = audioIntensity * 0.3;
                vec3 waveColor = vec3(0.2 + hue, 0.6, 0.9 - hue * 0.3);
                
                // Combine lighting with edge detection
                col = edge * waveColor * (ambient + diffuse * 0.7);
            }
            
            return col;
        }
        
        // Main color generation
        vec3 makeColor(vec2 fragCoord) {
            vec3 viewDir = rayDirection(fieldOfView, resolution.xy, fragCoord);
            vec3 origin = vec3(0.0, cameraHeight, 10.0);
            mat4 viewToWorld = viewMatrix(origin, vec3(0.0), vec3(0.0, 1.0, 0.0));
            vec3 dir = (viewToWorld * vec4(viewDir, 1.0)).xyz;
            
            return color(origin, dir);
        }
        
        void main() {
            // Simple antialiasing with 2x2 grid
            vec4 finalColor = vec4(0.0);
            const int AA = 2;
            
            for(int y = 0; y < AA; ++y) {
                for(int x = 0; x < AA; ++x) {
                    vec2 offset = vec2(float(x), float(y)) / float(AA);
                    vec2 coord = gl_FragCoord.xy + offset;
                    finalColor.rgb += clamp(makeColor(coord), 0.0, 1.0);
                }
            }
            
            finalColor.rgb /= float(AA * AA);
            
            // Apply fade and set alpha
            float alpha = (length(finalColor.rgb) > 0.01) ? fadeAlpha : 0.0;
            outColor = vec4(finalColor.rgb * fadeAlpha, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link shaders"""
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
        """Setup fullscreen quad for post-processing"""
        # Fullscreen quad vertices (texture coordinates)
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
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
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update animation time"""
        if not self.enabled:
            return
        
        self.time += dt
    
    def render(self, state: Dict):
        """Render fullscreen post-processing effect"""
        if not self.enabled or not self.shader:
            return
        
        # Post-process exception: Always pass depth test without writing
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        if res_loc != -1:
            glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        if time_loc != -1:
            glUniform1f(time_loc, self.time)
        
        fov_loc = glGetUniformLocation(self.shader, "fieldOfView")
        if fov_loc != -1:
            glUniform1f(fov_loc, self.field_of_view)
        
        cam_height_loc = glGetUniformLocation(self.shader, "cameraHeight")
        if cam_height_loc != -1:
            glUniform1f(cam_height_loc, self.camera_height)
        
        wave_speed_loc = glGetUniformLocation(self.shader, "waveSpeed")
        if wave_speed_loc != -1:
            glUniform1f(wave_speed_loc, self.wave_speed)
        
        wave_scale_loc = glGetUniformLocation(self.shader, "waveScale")
        if wave_scale_loc != -1:
            glUniform1f(wave_scale_loc, self.wave_scale)
        
        spacing_loc = glGetUniformLocation(self.shader, "spacing")
        if spacing_loc != -1:
            glUniform1f(spacing_loc, self.spacing)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if fade_loc != -1:
            glUniform1f(fade_loc, self.fade_factor)
        
        audio_loc = glGetUniformLocation(self.shader, "audioIntensity")
        if audio_loc != -1:
            glUniform1f(audio_loc, self.audio_intensity)
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore default depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
