"""
Voronoi Sphere - Raymarched 3D Voronoi pattern on a sphere with glowing kernel
Based on shader by Stephane Cuillerdier - Aiekick/2015
Creates a rotating 3D sphere with Voronoi cell structure and internal glowing core
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_voronoi_sphere(state, outstate, rotation_speed=0.14, elevation=1.0, 
                          distance=1.0, fade_duration=5.0, audio_reactive=True,
                          audio_sensitivity=2.5):
    """
    Raymarched Voronoi sphere effect compatible with EventScheduler
    
    Creates a 3D rotating sphere with Voronoi cell structure and glowing kernel
    Each Voronoi cell is assigned a frequency band and color, saturation modulates with audio
    
    Usage:
        scheduler.schedule_event(0, 60, shader_voronoi_sphere, 
                               rotation_speed=0.14, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        rotation_speed: Rotation speed multiplier (default 0.14)
        elevation: Camera elevation (default 1.0)
        distance: Camera distance from origin (default 1.0)
        fade_duration: Duration of fade in/out in seconds (default 5.0)
        audio_reactive: Enable audio reactivity (default True)
        audio_sensitivity: How much audio affects cell saturation (default 2.5)
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
        print(f"Initializing Voronoi sphere effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                VoronoiSphereEffect,
                rotation_speed=rotation_speed,
                elevation=elevation,
                distance=distance,
                audio_reactive=audio_reactive,
                audio_sensitivity=audio_sensitivity
            )
            state['effect'] = effect
            state['audio_bands'] = np.zeros(32, dtype=np.float32)  # Store all 32 bands
            print(f"✓ Initialized shader voronoi_sphere for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize voronoi sphere: {e}")
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
        
        # Audio reactivity - send all 32 frequency bands to shader
        if audio_reactive and audio_data is not None:
            # Use norm_long_relu for stronger, more dramatic response (only above-average energy)
            bands = audio_data['norm_long_relu'][0]  # All 32 bands - highlights peaks
            
            # Lighter smoothing for more responsive peaks
            smoothing = 0.35
            state['audio_bands'] = smoothing * bands + (1 - smoothing) * state['audio_bands']
            
            # Scale bands - norm_long_relu gives cleaner peaks so we can scale more aggressively
            state['effect'].audio_bands = np.clip(state['audio_bands'] * audio_sensitivity, 0, 3)
        else:
            # No audio - set bands to zero
            state['effect'].audio_bands = np.zeros(32, dtype=np.float32)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up Voronoi sphere effect for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader voronoi_sphere for frame {frame_id}")

# ============================================================================
# Voronoi Sphere Effect - Post-Processing Raymarcher
# ============================================================================

class VoronoiSphereEffect(ShaderEffect):
    """Fullscreen post-processing raymarched Voronoi sphere"""
    
    def __init__(self, viewport, rotation_speed: float = 0.14, elevation: float = 1.0,
                 distance: float = 1.0, audio_reactive: bool = True, 
                 audio_sensitivity: float = 2.5):
        super().__init__(viewport)
        self.rotation_speed = rotation_speed
        self.elevation = elevation
        self.distance = distance
        self.audio_reactive = audio_reactive
        self.audio_sensitivity = audio_sensitivity
        self.fade_factor = 0.0
        
        # Audio reactivity - 32 frequency bands
        self.audio_bands = np.zeros(32, dtype=np.float32)
        
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
        uniform float rotationSpeed;
        uniform float elevation;
        uniform float camDistance;
        uniform float fadeAlpha;
        uniform float audioBands[32];  // All 32 frequency bands
        
        #define shape(p) length(p)-2.8
        
        // Hash function with variation for cell colors
        vec3 hash3(vec3 p) {
            p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                     dot(p, vec3(269.5, 183.3, 246.1)),
                     dot(p, vec3(113.5, 271.9, 124.6)));
            return fract(sin(p) * 43758.5453123);
        }
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        // by shane - Voronoi-like noise function with cell ID output
        vec2 VoronesqueWithID(in vec3 p, out vec3 cellID) {
            vec3 i = floor(p + dot(p, vec3(0.333333)));
            vec3 baseCell = i;
            p -= i - dot(i, vec3(0.166666));
            vec3 i1 = step(0., p - p.yzx);
            vec3 i2 = max(i1, 1.0 - i1.zxy);
            i1 = min(i1, 1.0 - i1.zxy);
            vec3 p1 = p - i1 + 0.166666;
            vec3 p2 = p - i2 + 0.333333;
            vec3 p3 = p - 0.5;
            vec3 rnd = vec3(7., 157., 113.);
            vec4 v = max(0.5 - vec4(dot(p, p), dot(p1, p1), dot(p2, p2), dot(p3, p3)), 0.);
            vec4 d = vec4(dot(i, rnd), dot(i + i1, rnd), dot(i + i2, rnd), dot(i + 1., rnd));
            vec4 cellIDs = d;
            d = fract(sin(d) * 262144.) * v * 2.;
            
            // Find which cell is closest and get its base cell ID
            float maxDist = max(max(d.x, d.y), max(d.z, d.w));
            if (d.x >= maxDist) cellID = baseCell;
            else if (d.y >= maxDist) cellID = baseCell + i1;
            else if (d.z >= maxDist) cellID = baseCell + i2;
            else cellID = baseCell + 1.0;
            
            v.x = max(d.x, d.y);
            v.y = max(d.z, d.w);
            v.z = max(min(d.x, d.y), min(d.z, d.w));
            v.w = min(v.x, v.y);
            
            float voroValue = max(v.x, v.y) - max(v.z, v.w);
            return vec2(voroValue, 0.0);  // Return 0, we'll use cellID directly
        }
        
        vec3 map(vec3 p, out vec3 cellID) {
            vec3 res = vec3(0.);
            
            vec2 voroResult = VoronesqueWithID(p, cellID);
            float voro = voroResult.x;
            
            float sphere = shape(p);
            float sphereOut = sphere - voro;
            float sphereIn = sphere + voro * 0.5;
            
            float dist = max(-sphereIn, sphereOut + 0.29);
            
            res = vec3(dist, 1., 0.0);
            
            float kernel = sphere + 2.2;
            
            if (kernel < res.x)
                res = vec3(kernel, 2., 0.0);
            
            return res;
        }
        
        vec3 nor(vec3 pos, float prec) {
            vec2 e = vec2(prec, 0.);
            vec3 dummy;
            vec3 n = vec3(
                map(pos + e.xyy, dummy).x - map(pos - e.xyy, dummy).x,
                map(pos + e.yxy, dummy).x - map(pos - e.yxy, dummy).x,
                map(pos + e.yyx, dummy).x - map(pos - e.yyx, dummy).x
            );
            return normalize(n);
        }
        
        vec3 cam(vec2 uv, vec3 ro, vec3 cu, vec3 cv) {
            vec3 rov = normalize(cv - ro);
            vec3 u = normalize(cross(cu, rov));
            vec3 v = normalize(cross(rov, u));
            vec3 rd = normalize(rov + u * uv.x + v * uv.y);
            return rd;
        }
        
        // light position for simple shading
        const vec3 LPos = vec3(-0.6, 0.7, -0.5);
        
        void main() {
            vec2 si = resolution.xy;
            float t = time;
            
            // Constant rotation speed (no audio modulation)
            float ca = t * rotationSpeed;
            float ce = elevation;
            float cd = camDistance;
            
            vec3 cu = vec3(0., 1., 0.);
            vec3 cv = vec3(0., 0., 0.);
            
            // Shift viewport down so sphere center is at very bottom of screen
            vec2 uv = (gl_FragCoord.xy + gl_FragCoord.xy - si) / min(si.x, si.y);
            
            // Much larger shift to push sphere all the way to bottom
            uv.y += 2.5;  // Large shift to put sphere center at bottom of tall viewports
            
            vec3 ro = vec3(sin(ca) * cd, ce + 1., cos(ca) * cd);
            vec3 rd = cam(uv, ro, cu, cv);
            
            float d = 0.0;
            float s = 1.0;
            vec3 p = ro;
            vec3 cellID = vec3(0.);
            
            // Raymarching loop
            for (int i = 0; i < 200; i++) {
                if (d * d / s > 1e8) break;
                vec3 mapResult = map(p, cellID);
                s = mapResult.x;
                d += s * 0.5;
                p = ro + rd * d;
            }
            
            vec3 finalColor = vec3(0.);
            
            if (d < 5.0) {
                float nPrec = 0.1;
                vec3 n = nor(p, nPrec);
                vec3 finalMapResult = map(p, cellID);
                float mat = finalMapResult.y;
                
                if (mat < 1.5) { // rock/voronoi cells
                    // Get unique hash for this cell using the cellID
                    vec3 cellHash3 = hash3(cellID);
                    
                    // Assign frequency band based on cell hash (0-31)
                    int bandIndex = int(cellHash3.x * 31.0);
                    float audioEnergy = audioBands[bandIndex];
                    
                    // Assign hue based on cell hash for unique color per cell
                    float hue = cellHash3.y;  // Use second component for hue variation
                    
                    // Saturation strongly modulated by audio energy
                    // Start with high base saturation so colors are always visible
                    float baseSaturation = 0.7;
                    float saturation = baseSaturation + audioEnergy * 0.3;
                    saturation = clamp(saturation, 0.0, 1.0);
                    
                    // Get subtle lighting for depth
                    vec3 ldif = normalize(LPos - p);
                    float lightIntensity = max(0.3, dot(ldif, n)) * 0.5 + 0.5;
                    
                    // Brightness modulated by audio
                    float brightness = lightIntensity * (0.5 + audioEnergy * 0.5);
                    brightness = clamp(brightness, 0.3, 1.0);
                    
                    // Convert HSV to RGB
                    vec3 cellColor = hsv2rgb(vec3(hue, saturation, brightness));
                    
                    finalColor = cellColor;
                } else if (mat < 2.5) { // kernel - glowing center
                    float b = dot(n, normalize(ro - p)) * 0.9;
                    
                    // Simple warm glow for kernel
                    vec3 warmColor = vec3(1.0, 0.6, 0.3);
                    vec3 kernelColor = (b * warmColor + pow(b, 0.2) * vec3(1.0)) * (1.0 - d * 0.01);
                    
                    // Kernel reacts to bass frequencies (bands 0-7)
                    float bassEnergy = 0.0;
                    for (int i = 0; i < 8; i++) {
                        bassEnergy += audioBands[i];
                    }
                    bassEnergy /= 8.0;
                    
                    // Audio makes kernel pulse brighter
                    kernelColor *= 1.0 + bassEnergy * 0.5;
                    
                    finalColor = kernelColor;
                }
            } else {
                // Transparent background (no starfield)
                finalColor = vec3(0.);
            }
            
            // Set alpha to 0 for background, fadeAlpha for sphere
            float alpha = (d < 5.0) ? fadeAlpha : 0.0;
            outColor = vec4(finalColor * fadeAlpha, alpha);
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
        
        rot_loc = glGetUniformLocation(self.shader, "rotationSpeed")
        if rot_loc != -1:
            glUniform1f(rot_loc, self.rotation_speed)
        
        elev_loc = glGetUniformLocation(self.shader, "elevation")
        if elev_loc != -1:
            glUniform1f(elev_loc, self.elevation)
        
        dist_loc = glGetUniformLocation(self.shader, "camDistance")
        if dist_loc != -1:
            glUniform1f(dist_loc, self.distance)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if fade_loc != -1:
            glUniform1f(fade_loc, self.fade_factor)
        
        # Audio uniforms - send all 32 bands as array
        if self.audio_reactive:
            bands_loc = glGetUniformLocation(self.shader, "audioBands")
            if bands_loc != -1:
                glUniform1fv(bands_loc, 32, self.audio_bands)
        else:
            # Set all bands to zero if not audio reactive
            bands_loc = glGetUniformLocation(self.shader, "audioBands")
            if bands_loc != -1:
                glUniform1fv(bands_loc, 32, np.zeros(32, dtype=np.float32))
        
        # Draw fullscreen quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore default depth state
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
