"""
Chromatic fog beings effect - GPU-accelerated shader implementation
Organic entities with metaball physics, tentacles, and communication pulses
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List
from .base import ShaderEffect
import time

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_chromatic_fog_beings(state, outstate, num_beings=4, depth_layer=50.0):
    """
        Shader-based chromatic fog beings effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_chromatic_fog_beings, num_beings=5, 
                                depth_layer=50.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        num_beings: Number of fog beings to spawn (1-6, default 4)
        depth_layer: Depth layer for beings (0=near, 100=far, default 50.0)
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
    
    # Initialize fog beings on first call
    if state['count'] == 0:
        print(f"Initializing chromatic fog beings for frame {frame_id}")
        
        try:
            fog_effect = viewport.add_effect(
                ChromaticFogBeingsEffect,
                num_beings=num_beings,
                depth_layer=depth_layer
            )
            state['fog_effect'] = fog_effect
            state['start_time'] = time.time()
            print(f"✓ Initialized {num_beings} fog beings at depth {depth_layer} for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fog beings: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update fade factor based on elapsed time
    if 'fog_effect' in state:
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        fade_duration = 10.0
        
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['fog_effect'].fade_factor = np.clip(fade_factor, 0, 1)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'fog_effect' in state:
            print(f"Cleaning up fog beings for frame {frame_id}")
            viewport.effects.remove(state['fog_effect'])
            state['fog_effect'].cleanup()
            print(f"✓ Cleaned up fog beings for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class Being:
    """Data structure for a single fog being"""
    def __init__(self, viewport_width, viewport_height):
        self.position = np.array([
            np.random.uniform(20, viewport_width - 20),
            np.random.uniform(15, viewport_height - 15)
        ], dtype=np.float32)
        
        self.velocity = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ], dtype=np.float32)
        
        self.size = np.random.uniform(3, 6)
        self.base_hue = np.random.uniform(0, 1)
        self.hue_drift_rate = np.random.uniform(0.05, 0.2)
        self.hue_drift_phase = np.random.uniform(0, 2 * np.pi)
        
        self.shape_complexity = np.random.uniform(2, 5)
        self.shape_phase = np.random.uniform(0, 2 * np.pi)
        self.shape_evolution_rate = np.random.uniform(0.1, 0.3)
        
        self.target_behavior = np.random.randint(0, 4)  # 0=wander, 1=seek, 2=mimic, 3=avoid
        self.target_entity = None
        self.last_behavior_change = time.time()
        self.behavior_duration = np.random.uniform(5, 15)
        
        self.tentacles = np.random.randint(2, 5)  # More tentacles, min 2
        self.tentacle_params = []
        
        for _ in range(self.tentacles):
                        self.tentacle_params.append({
                'angle': np.random.uniform(0, 2 * np.pi),
                'angle_velocity': np.random.uniform(-0.08, 0.08),  # Slower, smoother rotation
                'rotation_phase': np.random.uniform(0, 2 * np.pi),  # For organic oscillation
                'rotation_rate': np.random.uniform(0.15, 0.3),  # Slower oscillation
                'length': np.random.uniform(10, 20),  # Much longer tentacles
                'wave_rate': np.random.uniform(0.3, 1.0),  # Slower wave motion
                'wave_phase': np.random.uniform(0, 2 * np.pi)
            })
        
        self.color_pulses = []  # Active communication pulses
        self.blorps = []  # Active blorps (balls sent to other beings)



class ChromaticFogBeingsEffect(ShaderEffect):
    """GPU-based chromatic fog beings using metaball rendering"""
    
    def __init__(self, viewport, num_beings: int = 4, depth_layer: float = 50.0):
        super().__init__(viewport)
        self.num_beings = np.clip(num_beings, 1, 6)
        self.beings: List[Being] = []  # Keep for compatibility with render method
        self.fade_factor = 0.0
        self.next_communication = time.time() + np.random.uniform(3, 8)
        self.depth_layer = depth_layer  # Depth in range 0-100
        
        # Wrap margin for horizontal wrapping
        self.wrap_margin = 60.0  # Should be larger than max being size + tentacle length
        
        # === VECTORIZED DATA STORAGE ===
        # Positions and velocities (N, 2)
        self.positions = np.random.uniform([20, 15], [viewport.width - 20, viewport.height - 15], 
                                          (self.num_beings, 2)).astype(np.float32)
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_beings, 2)).astype(np.float32)
        
        # Sizes and appearance
        self.sizes = np.random.uniform(3, 6, self.num_beings).astype(np.float32)
        self.base_hues = np.random.uniform(0, 1, self.num_beings).astype(np.float32)
        self.hue_drift_rates = np.random.uniform(0.05, 0.2, self.num_beings).astype(np.float32)
        self.hue_drift_phases = np.random.uniform(0, 2*np.pi, self.num_beings).astype(np.float32)
        
        # Shape properties
        self.shape_complexities = np.random.uniform(2, 5, self.num_beings).astype(np.float32)
        self.shape_phases = np.random.uniform(0, 2*np.pi, self.num_beings).astype(np.float32)
        self.shape_evolution_rates = np.random.uniform(0.1, 0.3, self.num_beings).astype(np.float32)
        
        # Behavior properties
        self.target_behaviors = np.random.randint(0, 4, self.num_beings).astype(np.int32)  # 0=wander, 1=seek, 2=mimic, 3=avoid
        self.target_entities = np.full(self.num_beings, -1, dtype=np.int32)  # -1 means no target
        self.last_behavior_changes = np.full(self.num_beings, time.time(), dtype=np.float64)
        self.behavior_durations = np.random.uniform(5, 15, self.num_beings).astype(np.float32)
        
        # Tentacle data - store as list of arrays since count varies
        self.max_tentacles = 6
        self.tentacle_counts = np.random.randint(2, 5, self.num_beings).astype(np.int32)
        self.tentacle_angles = []
        self.tentacle_velocities = []
        self.tentacle_rotation_phases = []
        self.tentacle_rotation_rates = []
        self.tentacle_lengths = []
        self.tentacle_wave_rates = []
        self.tentacle_wave_phases = []
        
        for i in range(self.num_beings):
            n = self.tentacle_counts[i]
            self.tentacle_angles.append(np.random.uniform(0, 2*np.pi, n).astype(np.float32))
            self.tentacle_velocities.append(np.random.uniform(-0.08, 0.08, n).astype(np.float32))
            self.tentacle_rotation_phases.append(np.random.uniform(0, 2*np.pi, n).astype(np.float32))
            self.tentacle_rotation_rates.append(np.random.uniform(0.15, 0.3, n).astype(np.float32))
            self.tentacle_lengths.append(np.random.uniform(10, 20, n).astype(np.float32))
            self.tentacle_wave_rates.append(np.random.uniform(0.3, 1.0, n).astype(np.float32))
            self.tentacle_wave_phases.append(np.random.uniform(0, 2*np.pi, n).astype(np.float32))
        
        # Communication pulses - store as list since pulses come and go
        self.color_pulses = [[] for _ in range(self.num_beings)]
        
        # Blorps (balls sent between beings)
        self.blorps = [[] for _ in range(self.num_beings)]
        self.next_blorp_time = time.time() + np.random.uniform(4, 10)
        
        # Spawn beings (for render compatibility - will sync from arrays)
        for i in range(self.num_beings):
            being = Being(viewport.width, viewport.height)
            self.beings.append(being)
            self._sync_being_from_arrays(i)
        
        # Uniforms buffer
        self.instance_VBO = None
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Fullscreen quad vertices
        layout(location = 0) in vec2 position;
        
        uniform float depthValue;  // Depth layer for this effect
        
        out vec2 fragCoord;
        
        void main() {
            // Convert from [-1, 1] to screen coordinates
            fragCoord = position;
            
            // Set depth for proper 3D layering
            // depthValue is in range 0-100, map to 0.0-1.0
            float depth = depthValue / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(position, depth, 1.0);
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
        uniform float fadeAlpha;
        uniform float wrapWidth;  // Viewport width for wrapping
        
        // Being data (max 6 beings)
        uniform int numBeings;
        uniform vec2 beingPositions[6];
        uniform float beingSizes[6];
        uniform float beingHues[6];
        uniform float beingComplexities[6];
        uniform float beingPhases[6];
        
        // Tentacle data (max 6 tentacles per being)
        uniform int beingTentacleCounts[6];
        uniform vec4 tentacleData[36];
        
        // Pulse data (max 2 pulses per being)
        uniform int beingPulseCounts[6];
        uniform vec4 pulseData[12];
        
        // Blorp data (max 2 blorps per being)
        uniform int beingBlorpCounts[6];
        uniform vec4 blorpPositions[12];  // x, y, size, alpha
        uniform vec4 blorpColors[12];     // hue, saturation, value, unused
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        // Get wrapped distance for X axis
        float getWrappedDx(float dx) {
            float wrapped_left = dx + wrapWidth;
            float wrapped_right = dx - wrapWidth;
            if (abs(wrapped_left) < abs(dx)) return wrapped_left;
            if (abs(wrapped_right) < abs(dx)) return wrapped_right;
            return dx;
        }
        
        // Metaball density function with horizontal wrapping - OPTIMIZED
        float metaball(vec2 pos, vec2 center, float sizeInv) {
            float dx = getWrappedDx(pos.x - center.x);
            float dy = pos.y - center.y;
            float distSq = dx * dx + dy * dy;
            return exp(-distSq * sizeInv);
        }
        
        // Simplified tentacle rendering with fewer samples
        float renderTentacle(vec2 screenPos, vec2 center, float baseAngle, float tentacleLength, 
                            float waveRate, float wavePhase, float baseSize) {
            if (tentacleLength < 1.0) return 0.0;
            
            float density = 0.0;
            float baseSizeInv = 1.0 / (2.0 * baseSize * baseSize);
            int segments = min(int(tentacleLength * 0.5), 10);  // Reduced from 16 to 10
            
            vec2 prevPos = center;
            float invSegments = 1.0 / float(segments);
            
            for (int s = 1; s <= segments; s++) {
                float ratio = float(s) * invSegments;
                float segmentLength = tentacleLength * ratio;
                
                // Simplified wave motion
                float waveFactor = sin(time * waveRate + wavePhase + ratio * 3.14159);
                float angle = baseAngle + waveFactor * 0.8;
                
                vec2 segmentPos = center + vec2(cos(angle), sin(angle)) * segmentLength;
                
                // Fast distance to segment approximation
                vec2 relPos = screenPos - (prevPos + segmentPos) * 0.5;
                float dx = getWrappedDx(relPos.x);
                float dist = length(vec2(dx, relPos.y));
                
                float taperFactor = 1.0 - ratio * 0.6;
                float segmentSizeInv = baseSizeInv / (taperFactor * taperFactor);
                float segmentDensity = exp(-dist * dist * segmentSizeInv);
                segmentDensity *= (1.0 + (1.0 - ratio) * 0.3) * 0.8;
                density += segmentDensity;
                
                prevPos = segmentPos;
            }
            
            return min(density, 1.0);
        }
        
                void main() {
            // Convert fragCoord from [-1, 1] to screen space [0, resolution]
            vec2 screenPos = (fragCoord + 1.0) * 0.5 * resolution;
            
            vec3 accumulatedColor = vec3(0.0);
            float accumulatedAlpha = 0.0;
            
            for (int i = 0; i < numBeings; i++) {
                vec2 center = beingPositions[i];
                float size = beingSizes[i];
                float hue = beingHues[i];
                float complexity = beingComplexities[i];
                float phase = beingPhases[i];
                
                // === COMPUTE WRAPPED DISTANCE ONCE ===
                float dx = getWrappedDx(screenPos.x - center.x);
                float dy = screenPos.y - center.y;
                float distFromCenter = length(vec2(dx, dy));
                float normalizedDist = distFromCenter / (size * 2.5);
                float sizeInv = 1.0 / (2.0 * size * size);
                
                                // === GLOW LAYER (Optimized) ===
                float glowSize = size * 2.5;
                float glowSizeInv = 1.0 / (2.0 * glowSize * glowSize);
                float glowDensity = metaball(screenPos, center, glowSizeInv);
                
                // === MAIN BODY LAYER ===
                float density = metaball(screenPos, center, sizeInv);
                
                // Add lobes (single pass for both glow and body)
                int nLobes = min(int(complexity), 5);
                float glowOffsetFactor = size * 0.7;
                float glowLobeSize = size * 2.0;
                float glowLobeSizeInv = 1.0 / (2.0 * glowLobeSize * glowLobeSize);
                
                for (int j = 0; j < nLobes; j++) {
                    float glowAngle = phase + float(j) * 6.28318 / complexity;
                    vec2 glowLobePos = center + vec2(cos(glowAngle), sin(glowAngle)) * glowOffsetFactor;
                    float dx_glow = getWrappedDx(screenPos.x - glowLobePos.x);
                    float dy_glow = screenPos.y - glowLobePos.y;
                    float distSq_glow = dx_glow * dx_glow + dy_glow * dy_glow;
                    glowDensity += 0.5 * exp(-distSq_glow * glowLobeSizeInv);
                }
                
                float offsetFactor = size * 0.7;
                float lobeSize = size * 0.6;
                float lobeSizeInv = 1.0 / (2.0 * lobeSize * lobeSize);
                
                for (int j = 0; j < nLobes; j++) {
                    float angle = phase + float(j) * 6.28318 / complexity;
                    vec2 lobePos = center + vec2(cos(angle), sin(angle)) * offsetFactor;
                    
                    // Main body lobes
                    float dx_lobe = getWrappedDx(screenPos.x - lobePos.x);
                    float dy_lobe = screenPos.y - lobePos.y;
                    float distSq = dx_lobe * dx_lobe + dy_lobe * dy_lobe;
                    density += 0.7 * exp(-distSq * lobeSizeInv);
                }
                
                // Add tentacles to main body
                int tentacleCount = min(beingTentacleCounts[i], 6);
                int tentacleBaseIdx = i * 6;
                float tentacleSize = size * 0.7;
                
                for (int t = 0; t < tentacleCount; t++) {
                    vec4 tentacle = tentacleData[tentacleBaseIdx + t];
                    float tentacleDensity = renderTentacle(
                        screenPos, center, tentacle.x, tentacle.y, 
                        tentacle.z, tentacle.w, tentacleSize
                    );
                    density += tentacleDensity;
                    
                    // Add tentacle glow samples
                    for (int s = 0; s < 3; s++) {
                        float ratio = float(s) * 0.5;
                        if (tentacle.y * ratio < 0.1) continue;
                        float waveFactor = sin(time * tentacle.z + tentacle.w + ratio * 3.14159);
                        float angle = tentacle.x + waveFactor * 0.8;
                        vec2 tentaclePos = center + vec2(cos(angle), sin(angle)) * (tentacle.y * ratio);
                        float tentacleGlowSize = size * 1.5 * (1.0 - ratio * 0.5);
                        float tentacleGlowSizeInv = 1.0 / (2.0 * tentacleGlowSize * tentacleGlowSize);
                        float dx_tent = getWrappedDx(screenPos.x - tentaclePos.x);
                        float dy_tent = screenPos.y - tentaclePos.y;
                        float distSq_tent = dx_tent * dx_tent + dy_tent * dy_tent;
                        glowDensity += 0.3 * exp(-distSq_tent * tentacleGlowSizeInv);
                    }
                }
                
                if (glowDensity > 0.01) {
                    float glowNormalizedDensity = min(glowDensity * 0.8, 1.0);
                    float glowSaturation = 0.7 - glowNormalizedDensity * 0.3;
                    float glowValue = 0.25 + glowNormalizedDensity * 0.4;
                    float glowAlpha = glowNormalizedDensity * 0.35;
                    vec3 glowColor = hsv2rgb(vec3(hue, glowSaturation, glowValue));
                    float newGlowAlpha = glowAlpha + accumulatedAlpha * (1.0 - glowAlpha);
                    if (newGlowAlpha > 0.0) {
                        accumulatedColor = (glowColor * glowAlpha + accumulatedColor * accumulatedAlpha * (1.0 - glowAlpha)) / newGlowAlpha;
                        accumulatedAlpha = newGlowAlpha;
                    }
                }
                
                // === SIMPLIFIED PULSE RENDERING ===
                int pulseCount = min(beingPulseCounts[i], 2);
                int pulseBaseIdx = i * 2;
                
                for (int p = 0; p < pulseCount; p++) {
                    vec4 pulse = pulseData[pulseBaseIdx + p];
                    float pulseDist = length(vec2(dx, dy)) - pulse.z;
                    float ringWidth = size * 0.3;
                    float ringFactor = 1.0 - abs(pulseDist) / ringWidth;
                    
                    if (ringFactor > 0.0) {
                        float pulseProgress = pulse.x / pulse.y;
                        float intensity = ringFactor * (1.0 - pulseProgress) * 0.3;
                        density += intensity;
                    }
                }
                
                // === RENDER IF VISIBLE ===
                if (density > 0.05) {
                    float normalizedDensity = min(density, 1.0);
                    
                    // Radial brightness gradient
                    float brightnessGradient = 1.0 - smoothstep(0.0, 1.0, normalizedDist);
                    brightnessGradient = pow(brightnessGradient, 0.7);
                    
                    float saturation = 0.9 - normalizedDensity * 0.3;
                    float value = 0.2 + normalizedDensity * 0.6 + brightnessGradient * 0.5;
                    value = clamp(value, 0.0, 1.0);
                    
                    float alpha = normalizedDensity * 0.4;
                    vec3 color = hsv2rgb(vec3(hue, saturation, value));
                    
                    // Alpha blend
                    float newAlpha = alpha + accumulatedAlpha * (1.0 - alpha);
                    if (newAlpha > 0.0) {
                        accumulatedColor = (color * alpha + accumulatedColor * accumulatedAlpha * (1.0 - alpha)) / newAlpha;
                        accumulatedAlpha = newAlpha;
                    }
                }
            }
            
            // === RENDER BLORPS ===
            for (int i = 0; i < numBeings; i++) {
                int blorpCount = min(beingBlorpCounts[i], 2);
                int blorpBaseIdx = i * 2;
                
                for (int b = 0; b < blorpCount; b++) {
                    vec4 blorpPos = blorpPositions[blorpBaseIdx + b];
                    vec4 blorpCol = blorpColors[blorpBaseIdx + b];
                    
                    if (blorpPos.w > 0.01) {  // Check alpha
                        float dx = getWrappedDx(screenPos.x - blorpPos.x);
                        float dy = screenPos.y - blorpPos.y;
                        float dist = length(vec2(dx, dy));
                        
                        if (dist < blorpPos.z * 2.0) {
                            float density = 1.0 - smoothstep(0.0, blorpPos.z, dist);
                            density = pow(density, 2.0);  // Sharper falloff
                            
                            vec3 blorpColor = hsv2rgb(vec3(blorpCol.x, blorpCol.y, blorpCol.z));
                            float blorpAlpha = density * blorpPos.w * 0.6;
                            
                            float newAlpha = blorpAlpha + accumulatedAlpha * (1.0 - blorpAlpha);
                            if (newAlpha > 0.0) {
                                accumulatedColor = (blorpColor * blorpAlpha + accumulatedColor * accumulatedAlpha * (1.0 - blorpAlpha)) / newAlpha;
                                accumulatedAlpha = newAlpha;
                            }
                        }
                    }
                }
            }
            
            accumulatedAlpha *= fadeAlpha;
            outColor = vec4(accumulatedColor, accumulatedAlpha);
        }
        """

    
    def compile_shader(self):
        """Compile and link fog being shaders"""
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
        """Initialize OpenGL buffers for fullscreen quad"""
        # Fullscreen quad vertices
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
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
    
    def _sync_being_from_arrays(self, i: int):
        """Sync a Being object from vectorized arrays"""
        being = self.beings[i]
        being.position = self.positions[i].copy()
        being.velocity = self.velocities[i].copy()
        being.size = float(self.sizes[i])
        being.base_hue = float(self.base_hues[i])
        being.hue_drift_rate = float(self.hue_drift_rates[i])
        being.hue_drift_phase = float(self.hue_drift_phases[i])
        being.shape_complexity = float(self.shape_complexities[i])
        being.shape_phase = float(self.shape_phases[i])
        being.shape_evolution_rate = float(self.shape_evolution_rates[i])
        being.target_behavior = int(self.target_behaviors[i])
        being.last_behavior_change = float(self.last_behavior_changes[i])
        being.behavior_duration = float(self.behavior_durations[i])
        
        # Sync tentacles
        being.tentacles = int(self.tentacle_counts[i])
        being.tentacle_params = []
        for j in range(being.tentacles):
            being.tentacle_params.append({
                'angle': float(self.tentacle_angles[i][j]),
                'angle_velocity': float(self.tentacle_velocities[i][j]),
                'rotation_phase': float(self.tentacle_rotation_phases[i][j]),
                'rotation_rate': float(self.tentacle_rotation_rates[i][j]),
                'length': float(self.tentacle_lengths[i][j]),
                'wave_rate': float(self.tentacle_wave_rates[i][j]),
                'wave_phase': float(self.tentacle_wave_phases[i][j])
            })
        
        being.color_pulses = self.color_pulses[i]
        being.blorps = self.blorps[i]
    
    def _sync_all_beings(self):
        """Sync all Being objects from vectorized arrays"""
        for i in range(self.num_beings):
            self._sync_being_from_arrays(i)
    
    def update(self, dt: float, state: Dict):
        """Update being positions and behaviors - VECTORIZED"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # === VECTORIZED BEHAVIOR UPDATES ===
        # Check which beings need behavior change
        time_since_change = current_time - self.last_behavior_changes
        needs_change = time_since_change > self.behavior_durations
        
        if np.any(needs_change):
            change_indices = np.where(needs_change)[0]
            self.target_behaviors[change_indices] = np.random.randint(0, 4, len(change_indices))
            self.behavior_durations[change_indices] = np.random.uniform(5, 15, len(change_indices))
            self.last_behavior_changes[change_indices] = current_time
            
            # Assign targets for seek and avoid behaviors
            for idx in change_indices:
                if self.target_behaviors[idx] in [1, 3] and self.num_beings > 1:  # seek or avoid
                    # Choose a different being as target
                    potential_targets = [j for j in range(self.num_beings) if j != idx]
                    self.target_entities[idx] = np.random.choice(potential_targets)
                else:
                    self.target_entities[idx] = -1
        
        # === WANDER BEHAVIOR (0) ===
        wander_mask = self.target_behaviors == 0
        if np.any(wander_mask):
            # Random chance to change direction
            change_dir = np.random.random(self.num_beings) < 0.02
            wander_change = wander_mask & change_dir
            
            if np.any(wander_change):
                change_indices = np.where(wander_change)[0]
                angles = np.random.uniform(0, 2*np.pi, len(change_indices))
                speeds = np.linalg.norm(self.velocities[change_indices], axis=1)
                speeds = np.where(speeds > 0, speeds, np.random.uniform(0.5, 1.5, len(change_indices)))
                
                self.velocities[change_indices, 0] = np.cos(angles) * speeds
                self.velocities[change_indices, 1] = np.sin(angles) * speeds
        
        # === SEEK BEHAVIOR (1) ===
        seek_mask = (self.target_behaviors == 1) & (self.target_entities >= 0)
        if np.any(seek_mask):
            seek_indices = np.where(seek_mask)[0]
            for idx in seek_indices:
                target_idx = self.target_entities[idx]
                direction = self.positions[target_idx] - self.positions[idx]
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    direction = direction / distance
                    target_velocity = direction * np.random.uniform(0.5, 1.5)
                    self.velocities[idx] += (target_velocity - self.velocities[idx]) * 0.1
                    
                    speed = np.linalg.norm(self.velocities[idx])
                    if speed > 2.0:
                        self.velocities[idx] = self.velocities[idx] / speed * 2.0
        
        # === MIMIC BEHAVIOR (2) ===
        mimic_mask = self.target_behaviors == 2
        if np.any(mimic_mask):
            self.shape_complexities[mimic_mask] = 3 + np.sin(current_time * 0.2) * 2
        
        # === AVOID BEHAVIOR (3) ===
        avoid_mask = (self.target_behaviors == 3) & (self.target_entities >= 0)
        if np.any(avoid_mask):
            avoid_indices = np.where(avoid_mask)[0]
            for idx in avoid_indices:
                target_idx = self.target_entities[idx]
                
                # Calculate wrapped distance
                dx = self.positions[idx, 0] - self.positions[target_idx, 0]
                # Check for wrapped distance
                if abs(dx - self.viewport.width) < abs(dx):
                    dx = dx - self.viewport.width
                elif abs(dx + self.viewport.width) < abs(dx):
                    dx = dx + self.viewport.width
                
                dy = self.positions[idx, 1] - self.positions[target_idx, 1]
                direction = np.array([dx, dy])
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    direction = direction / distance
                    # Move away from target
                    away_velocity = direction * np.random.uniform(0.7, 2.0)
                    self.velocities[idx] += (away_velocity - self.velocities[idx]) * 0.15
                    
                    speed = np.linalg.norm(self.velocities[idx])
                    if speed > 2.5:
                        self.velocities[idx] = self.velocities[idx] / speed * 2.5
        
        # === VECTORIZED POSITION UPDATES ===
        self.positions += self.velocities * dt
        
        # === VECTORIZED BOUNDARY CHECKS ===
        padding = 10
        
        # Horizontal wrapping
        self.positions[:, 0] = np.where(self.positions[:, 0] < 0,
                                       self.positions[:, 0] + self.viewport.width,
                                       self.positions[:, 0])
        self.positions[:, 0] = np.where(self.positions[:, 0] > self.viewport.width,
                                       self.positions[:, 0] - self.viewport.width,
                                       self.positions[:, 0])
        
        # Vertical boundaries with bounce
        # Bottom boundary
        bottom_hit = self.positions[:, 1] < padding
        self.positions[bottom_hit, 1] = padding
        self.velocities[bottom_hit, 1] = np.abs(self.velocities[bottom_hit, 1]) * 0.8
        
        # Top boundary
        top_hit = self.positions[:, 1] > self.viewport.height - padding
        self.positions[top_hit, 1] = self.viewport.height - padding
        self.velocities[top_hit, 1] = -np.abs(self.velocities[top_hit, 1]) * 0.8
        
        # === VECTORIZED APPEARANCE UPDATES ===
        self.shape_phases += self.shape_evolution_rates * dt
        self.hue_drift_phases += self.hue_drift_rates * dt
        
        # === VECTORIZED TENTACLE UPDATES ===
        for i in range(self.num_beings):
            n = self.tentacle_counts[i]
            if n == 0:
                continue
            
            # Smooth base rotation using velocity
            self.tentacle_angles[i] += self.tentacle_velocities[i] * dt
            
            # Add organic oscillation
            oscillation = np.sin(current_time * self.tentacle_rotation_rates[i] + 
                               self.tentacle_rotation_phases[i])
            self.tentacle_angles[i] += oscillation * 0.05 * dt
            
            # Keep angle in 0-2π range
            self.tentacle_angles[i] = self.tentacle_angles[i] % (2 * np.pi)
            
            # Occasionally adjust rotation velocity smoothly
            adjust_vel = np.random.random(n) < 0.003
            if np.any(adjust_vel):
                target_velocities = np.random.uniform(-0.08, 0.08, n)
                self.tentacle_velocities[i] += (target_velocities - self.tentacle_velocities[i]) * 0.05 * adjust_vel
            
            # Occasionally adjust wave parameters
            adjust_wave = np.random.random(n) < 0.008
            if np.any(adjust_wave):
                target_wave_rates = np.random.uniform(0.3, 1.0, n)
                self.tentacle_wave_rates[i] += (target_wave_rates - self.tentacle_wave_rates[i]) * 0.03 * adjust_wave
                self.tentacle_wave_rates[i] = np.clip(self.tentacle_wave_rates[i], 0.2, 1.5)
            
            # Occasionally adjust length
            adjust_len = np.random.random(n) < 0.004
            if np.any(adjust_len):
                target_lengths = np.random.uniform(12, 25, n)
                self.tentacle_lengths[i] += (target_lengths - self.tentacle_lengths[i]) * 0.02 * adjust_len
        
        # === UPDATE PULSES ===
        for i in range(self.num_beings):
            remaining_pulses = []
            for pulse in self.color_pulses[i]:
                pulse['age'] += dt
                if pulse['age'] < pulse['duration']:
                    remaining_pulses.append(pulse)
            self.color_pulses[i] = remaining_pulses
        
        # === UPDATE BLORPS ===
        for i in range(self.num_beings):
            remaining_blorps = []
            for blorp in self.blorps[i]:
                blorp['age'] += dt
                if blorp['age'] < blorp['total_duration']:
                    remaining_blorps.append(blorp)
            self.blorps[i] = remaining_blorps
        
        # === CHECK FOR COMMUNICATION EVENTS ===
        if current_time >= self.next_communication and self.num_beings > 1:
            sender_idx, receiver_idx = np.random.choice(self.num_beings, 2, replace=False)
            
            pulse = {
                'age': 0.0,
                'duration': np.random.uniform(1.0, 3.0),
                'hue': float(self.base_hues[sender_idx])
            }
            self.color_pulses[sender_idx].append(pulse)
            
            self.next_communication = current_time + np.random.uniform(3, 8)
            
            if np.random.random() < 0.3:
                self.target_behaviors[receiver_idx] = np.random.randint(0, 4)
                self.target_entities[receiver_idx] = sender_idx
                self.last_behavior_changes[receiver_idx] = current_time
        
        # === CHECK FOR BLORP EVENTS ===
        if current_time >= self.next_blorp_time and self.num_beings > 1:
            sender_idx, receiver_idx = np.random.choice(self.num_beings, 2, replace=False)
            
            # Calculate wrapped distance for blorp travel time
            dx = self.positions[receiver_idx, 0] - self.positions[sender_idx, 0]
            if abs(dx - self.viewport.width) < abs(dx):
                dx = dx - self.viewport.width
            elif abs(dx + self.viewport.width) < abs(dx):
                dx = dx + self.viewport.width
            dy = self.positions[receiver_idx, 1] - self.positions[sender_idx, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            travel_time = max(distance / 50.0, 0.5)  # 50 pixels/second base speed
            fade_time = 0.3  # Fade in/out duration
            
            blorp = {
                'age': 0.0,
                'fade_time': fade_time,
                'travel_time': travel_time,
                'total_duration': fade_time * 2 + travel_time,
                'start_pos': self.positions[sender_idx].copy(),
                'target_idx': receiver_idx,
                'hue': float(self.base_hues[sender_idx]),
                'size': np.random.uniform(2.0, 4.0)
            }
            self.blorps[sender_idx].append(blorp)
            
            self.next_blorp_time = current_time + np.random.uniform(4, 10)
        
        # Sync Being objects for rendering
        self._sync_all_beings()
    

    def render(self, state: Dict):
        """Render fog beings using shader"""
        if not self.enabled or not self.shader:
            return
        
        # No depth state modifications - use global state
        # Depth test and alpha blending are always enabled
        glUseProgram(self.shader)
        
                # Set uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "time")
        if loc != -1:
            glUniform1f(loc, time.time())
        
        loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        loc = glGetUniformLocation(self.shader, "depthValue")
        if loc != -1:
            glUniform1f(loc, self.depth_layer)
        
        loc = glGetUniformLocation(self.shader, "wrapWidth")
        if loc != -1:
            glUniform1f(loc, float(self.viewport.width))
        
        loc = glGetUniformLocation(self.shader, "numBeings")
        if loc != -1:
            glUniform1i(loc, len(self.beings))
        
        # Upload being data
        positions = np.array([b.position for b in self.beings], dtype=np.float32)
        sizes = np.array([b.size for b in self.beings], dtype=np.float32)
        hues = np.array([(b.base_hue + 0.1 * np.sin(b.hue_drift_phase)) % 1.0 
                         for b in self.beings], dtype=np.float32)
        complexities = np.array([b.shape_complexity for b in self.beings], dtype=np.float32)
        phases = np.array([b.shape_phase for b in self.beings], dtype=np.float32)
        
        loc = glGetUniformLocation(self.shader, "beingPositions")
        if loc != -1:
            glUniform2fv(loc, len(self.beings), positions.flatten())
        
        loc = glGetUniformLocation(self.shader, "beingSizes")
        if loc != -1:
            glUniform1fv(loc, len(self.beings), sizes)
        
        loc = glGetUniformLocation(self.shader, "beingHues")
        if loc != -1:
            glUniform1fv(loc, len(self.beings), hues)
        
        loc = glGetUniformLocation(self.shader, "beingComplexities")
        if loc != -1:
            glUniform1fv(loc, len(self.beings), complexities)
        
        loc = glGetUniformLocation(self.shader, "beingPhases")
        if loc != -1:
            glUniform1fv(loc, len(self.beings), phases)
        
        # Upload tentacle data - increased to 6 tentacles per being
        tentacle_counts = np.array([len(b.tentacle_params) for b in self.beings], dtype=np.int32)
        tentacle_data = np.zeros((36, 4), dtype=np.float32)  # Changed from 24 to 36
        
        for i, being in enumerate(self.beings):
            for j, tentacle in enumerate(being.tentacle_params[:6]):  # Changed from 4 to 6
                idx = i * 6 + j  # Changed from 4 to 6
                tentacle_data[idx] = [
                    tentacle['angle'],
                    tentacle['length'],
                    tentacle['wave_rate'],
                    tentacle['wave_phase']
                ]
        
        loc = glGetUniformLocation(self.shader, "beingTentacleCounts")
        if loc != -1:
            glUniform1iv(loc, len(self.beings), tentacle_counts)
        
        loc = glGetUniformLocation(self.shader, "tentacleData")
        if loc != -1:
            glUniform4fv(loc, 36, tentacle_data.flatten())  # Changed from 24 to 36
        
        # Upload pulse data
        pulse_counts = np.array([len(b.color_pulses) for b in self.beings], dtype=np.int32)
        pulse_data = np.zeros((12, 4), dtype=np.float32)
        
        current_time = time.time()
        for i, being in enumerate(self.beings):
            for j, pulse in enumerate(being.color_pulses[:2]):
                idx = i * 2 + j
                max_radius = being.size * 2.5
                pulse_progress = pulse['age'] / pulse['duration']
                pulse_radius = max_radius * pulse_progress
                
                pulse_data[idx] = [
                    pulse['age'],
                    pulse['duration'],
                    pulse_radius,
                    pulse['hue']
                ]
        
        loc = glGetUniformLocation(self.shader, "beingPulseCounts")
        if loc != -1:
            glUniform1iv(loc, len(self.beings), pulse_counts)
        
        loc = glGetUniformLocation(self.shader, "pulseData")
        if loc != -1:
            glUniform4fv(loc, 12, pulse_data.flatten())
        
        # Upload blorp data
        blorp_counts = np.array([len(b.blorps) for b in self.beings], dtype=np.int32)
        blorp_positions = np.zeros((12, 4), dtype=np.float32)
        blorp_colors = np.zeros((12, 4), dtype=np.float32)
        
        for i, being in enumerate(self.beings):
            for j, blorp in enumerate(being.blorps[:2]):
                idx = i * 2 + j
                age = blorp['age']
                fade_time = blorp['fade_time']
                travel_time = blorp['travel_time']
                total_duration = blorp['total_duration']
                
                # Calculate alpha based on fade in/out
                if age < fade_time:
                    alpha = age / fade_time
                elif age > fade_time + travel_time:
                    alpha = (total_duration - age) / fade_time
                else:
                    alpha = 1.0
                alpha = np.clip(alpha, 0, 1)
                
                # Calculate position (interpolate from start to target)
                if age < fade_time:
                    progress = 0.0
                elif age > fade_time + travel_time:
                    progress = 1.0
                else:
                    progress = (age - fade_time) / travel_time
                
                start_pos = blorp['start_pos']
                target_idx = blorp['target_idx']
                target_pos = self.beings[target_idx].position
                
                # Handle wrapped interpolation
                dx = target_pos[0] - start_pos[0]
                if abs(dx - self.viewport.width) < abs(dx):
                    dx = dx - self.viewport.width
                elif abs(dx + self.viewport.width) < abs(dx):
                    dx = dx + self.viewport.width
                
                current_x = start_pos[0] + dx * progress
                # Wrap current position
                if current_x < 0:
                    current_x += self.viewport.width
                elif current_x > self.viewport.width:
                    current_x -= self.viewport.width
                
                current_y = start_pos[1] + (target_pos[1] - start_pos[1]) * progress
                
                blorp_positions[idx] = [
                    current_x,
                    current_y,
                    blorp['size'],
                    alpha
                ]
                
                blorp_colors[idx] = [
                    blorp['hue'],
                    0.9,  # High saturation
                    0.8,  # Bright
                    0.0   # Unused
                ]
        
        loc = glGetUniformLocation(self.shader, "beingBlorpCounts")
        if loc != -1:
            glUniform1iv(loc, len(self.beings), blorp_counts)
        
        loc = glGetUniformLocation(self.shader, "blorpPositions")
        if loc != -1:
            glUniform4fv(loc, 12, blorp_positions.flatten())
        
        loc = glGetUniformLocation(self.shader, "blorpColors")
        if loc != -1:
            glUniform4fv(loc, 12, blorp_colors.flatten())
        
                # Render fullscreen quad
        glBindVertexArray(self.VAO)
        
        # Blending is already enabled globally - no need to toggle
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
