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

def shader_chromatic_fog_beings(state, outstate, num_beings=4):
    """
    Shader-based chromatic fog beings effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_chromatic_fog_beings, num_beings=5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        num_beings: Number of fog beings to spawn (3-6 recommended)
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
                num_beings=num_beings
            )
            state['fog_effect'] = fog_effect
            state['start_time'] = time.time()
            print(f"✓ Initialized {num_beings} fog beings for frame {frame_id}")
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
        
        self.target_behavior = np.random.randint(0, 3)  # 0=wander, 1=seek, 2=mimic
        self.target_entity = None
        self.last_behavior_change = time.time()
        self.behavior_duration = np.random.uniform(5, 15)
        
        self.tentacles = np.random.randint(2, 5)  # More tentacles, min 2
        self.tentacle_params = []
        
        for _ in range(self.tentacles):
            self.tentacle_params.append({
                'angle': np.random.uniform(0, 2 * np.pi),
                'length': np.random.uniform(10,20),  # Much longer tentacles
                'wave_rate': np.random.uniform(0.5, 2.0),
                'wave_phase': np.random.uniform(0, 2 * np.pi)
            })
        
        self.color_pulses = []  # Active communication pulses



class ChromaticFogBeingsEffect(ShaderEffect):
    """GPU-based chromatic fog beings using metaball rendering"""
    
    def __init__(self, viewport, num_beings: int = 4):
        super().__init__(viewport)
        self.num_beings = np.clip(num_beings, 1, 4)
        self.beings: List[Being] = []
        self.fade_factor = 0.0
        self.next_communication = time.time() + np.random.uniform(3, 8)
        
        # Spawn beings
        for _ in range(self.num_beings):
            self.beings.append(Being(viewport.width, viewport.height))
        
        # Uniforms buffer
        self.instance_VBO = None
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        // Fullscreen quad vertices
        layout(location = 0) in vec2 position;
        
        out vec2 fragCoord;
        
        void main() {
            // Convert from [-1, 1] to screen coordinates
            fragCoord = position;
            gl_Position = vec4(position, 0.0, 1.0);
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
        
        // Being data (max 6 beings)
        uniform int numBeings;
        uniform vec2 beingPositions[6];
        uniform float beingSizes[6];
        uniform float beingHues[6];
        uniform float beingComplexities[6];
        uniform float beingPhases[6];
        
        // Tentacle data (max 6 tentacles per being)
        uniform int beingTentacleCounts[6];
        uniform vec4 tentacleData[36];  // [angle, length, wave_rate, wave_phase] * 6 beings * 6 tentacles
        
        // Pulse data (max 2 pulses per being)
        uniform int beingPulseCounts[6];
        uniform vec4 pulseData[12];  // [age, duration, radius, hue] * 6 beings * 2 pulses
        
        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        // Metaball density function
        float metaball(vec2 pos, vec2 center, float size) {
            float dist = length(pos - center);
            return exp(-dist * dist / (2.0 * size * size));
        }
        
        // Distance from point to line segment
        float distanceToSegment(vec2 p, vec2 a, vec2 b) {
            vec2 pa = p - a;
            vec2 ba = b - a;
            float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            return length(pa - ba * h);
        }
        
        // Smooth tentacle rendering using line segments
        float renderTentacle(vec2 screenPos, vec2 center, float baseAngle, float length, 
                            float waveRate, float wavePhase, float baseSize) {
            if (length < 1.0) return 0.0;
            
            float density = 0.0;
            int segments = min(int(length * 0.8), 16);  // Scale segments with length
            
            vec2 prevPos = center;
            
            for (int s = 0; s <= segments; s++) {
                float ratio = float(s) / float(segments);
                float segmentLength = length * ratio;
                
                // Create smooth wave motion along tentacle
                float waveOffset = ratio * 2.0;  // Wave travels along tentacle
                float waveFactor = sin(time * waveRate + wavePhase + waveOffset * 3.14159);
                float undulation = ratio * sin(time * 0.3) * 0.3;
                float angle = baseAngle + undulation + waveFactor * 0.8;
                
                vec2 segmentPos = center + vec2(cos(angle), sin(angle)) * segmentLength;
                
                // Draw line segment from previous position to current
                if (s > 0) {
                    float dist = distanceToSegment(screenPos, prevPos, segmentPos);
                    
                    // Gradual taper - thicker base, thinner tip
                    float taperFactor = 1.0 - ratio * 0.6;
                    float segmentSize = baseSize * taperFactor;
                    
                    // Create soft falloff for tentacle thickness
                    float segmentDensity = exp(-dist * dist / (2.0 * segmentSize * segmentSize));
                    
                    // Add extra brightness to base segments
                    float baseBrightness = 1.0 + (1.0 - ratio) * 0.3;
                    density += segmentDensity * baseBrightness * 0.8;
                }
                
                prevPos = segmentPos;
            }
            
            return density;
        }
        
        void main() {
            // Convert fragCoord from [-1, 1] to screen space [0, resolution]
            vec2 screenPos = (fragCoord + 1.0) * 0.5 * resolution;
            
            // Accumulate density from all beings
            float totalDensity = 0.0;
            vec3 accumulatedColor = vec3(0.0);
            float accumulatedAlpha = 0.0;
            
            for (int i = 0; i < numBeings; i++) {
                vec2 center = beingPositions[i];
                float size = beingSizes[i];
                float hue = beingHues[i];
                float complexity = beingComplexities[i];
                float phase = beingPhases[i];
                
                // Calculate distance from center for brightness gradient
                float distFromCenter = length(screenPos - center);
                float normalizedDist = distFromCenter / (size * 2.5);  // Normalize to being size
                
                // Base density
                float density = metaball(screenPos, center, size);
                
                // Add lobes for organic shape
                int nLobes = int(complexity);
                float offsetFactor = size * 0.7;
                float lobeSize = size * 0.6;
                
                for (int j = 0; j < nLobes && j < 5; j++) {
                    float angle = phase + float(j) * 6.28318 / complexity;
                    vec2 lobePos = center + vec2(cos(angle), sin(angle)) * offsetFactor;
                    density += 0.7 * metaball(screenPos, lobePos, lobeSize);
                }
                
                // Add tentacles using improved line segment rendering
                int tentacleCount = beingTentacleCounts[i];
                int tentacleBaseIdx = i * 6;
                
                for (int t = 0; t < tentacleCount && t < 6; t++) {
                    vec4 tentacle = tentacleData[tentacleBaseIdx + t];
                    float baseAngle = tentacle.x;
                    float length = tentacle.y;
                    float waveRate = tentacle.z;
                    float wavePhase = tentacle.w;
                    
                    // Thicker tentacles - use 70% of body size for base thickness
                    float tentacleSize = size * 0.7;
                    float tentacleDensity = renderTentacle(
                        screenPos, center, baseAngle, length, 
                        waveRate, wavePhase, tentacleSize
                    );
                    
                    density += tentacleDensity;
                }
                
                // Add communication pulses
                int pulseCount = beingPulseCounts[i];
                int pulseBaseIdx = i * 2;
                
                for (int p = 0; p < pulseCount && p < 2; p++) {
                    vec4 pulse = pulseData[pulseBaseIdx + p];
                    float pulseAge = pulse.x;
                    float pulseDuration = pulse.y;
                    float pulseRadius = pulse.z;
                    
                    if (pulseAge < pulseDuration) {
                        float dist = length(screenPos - center);
                        float ringWidth = size * 0.5;
                        float ringInner = pulseRadius - ringWidth * 0.5;
                        float ringOuter = pulseRadius + ringWidth * 0.5;
                        
                        if (dist >= ringInner && dist <= ringOuter) {
                            float normalizedDist = (dist - ringInner) / ringWidth;
                            float pulseProgress = pulseAge / pulseDuration;
                            float ringIntensity = sin(normalizedDist * 3.14159) * (1.0 - pulseProgress);
                            density += ringIntensity * 0.3;
                        }
                    }
                }
                
                // Normalize density for this being
                if (density > 0.05) {
                    float normalizedDensity = min(density, 1.0);
                    
                    // Create radial brightness gradient - brighter in center, darker at edges
                    float brightnessGradient = 1.0 - smoothstep(0.0, 1.0, normalizedDist);
                    brightnessGradient = pow(brightnessGradient, 0.7);  // Adjust falloff curve
                    
                    // Calculate color with radial brightness
                    float saturation = 0.9 - normalizedDensity * 0.3;
                    float baseValue = 0.2 + normalizedDensity * 0.6;
                    
                    // Add radial brightness boost (brighter in center)
                    float value = baseValue + brightnessGradient * 0.5;
                    value = clamp(value, 0.0, 1.0);
                    
                    float alpha = normalizedDensity * 0.4;
                    
                    vec3 color = hsv2rgb(vec3(hue, saturation, value));
                    
                    // Alpha blend with accumulated color
                    float newAlpha = alpha + accumulatedAlpha * (1.0 - alpha);
                    if (newAlpha > 0.0) {
                        accumulatedColor = (color * alpha + accumulatedColor * accumulatedAlpha * (1.0 - alpha)) / newAlpha;
                        accumulatedAlpha = newAlpha;
                    }
                }
            }
            
            // Apply fade factor
            accumulatedAlpha *= fadeAlpha;
            
            // Output final color
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
    
    def update(self, dt: float, state: Dict):
        """Update being positions and behaviors"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Update each being
        for being in self.beings:
            # Update behavior if needed
            if current_time - being.last_behavior_change > being.behavior_duration:
                being.target_behavior = np.random.randint(0, 3)
                being.behavior_duration = np.random.uniform(5, 15)
                being.last_behavior_change = current_time
                
                if being.target_behavior == 1 and len(self.beings) > 1:
                    potential_targets = [b for b in self.beings if b is not being]
                    being.target_entity = np.random.choice(potential_targets)
                else:
                    being.target_entity = None
            
            # Apply behavior
            if being.target_behavior == 0:  # Wander
                if np.random.random() < 0.02:
                    angle = np.random.uniform(0, 2 * np.pi)
                    speed = np.linalg.norm(being.velocity) or np.random.uniform(0.5, 1.5)
                    being.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
            
            elif being.target_behavior == 1 and being.target_entity:  # Seek
                direction = being.target_entity.position - being.position
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    direction = direction / distance
                    target_velocity = direction * np.random.uniform(0.5, 1.5)
                    being.velocity += (target_velocity - being.velocity) * 0.1
                    
                    speed = np.linalg.norm(being.velocity)
                    if speed > 2.0:
                        being.velocity = being.velocity / speed * 2.0
            
            elif being.target_behavior == 2:  # Mimic
                being.shape_complexity = 3 + np.sin(current_time * 0.2) * 2
            
            # Update position
            being.position += being.velocity * dt
            
            # Boundary checks
            padding = 10
            if being.position[0] < padding:
                being.position[0] = padding
                being.velocity[0] = abs(being.velocity[0]) * 0.8
            elif being.position[0] > self.viewport.width - padding:
                being.position[0] = self.viewport.width - padding
                being.velocity[0] = -abs(being.velocity[0]) * 0.8
            
            if being.position[1] < padding:
                being.position[1] = padding
                being.velocity[1] = abs(being.velocity[1]) * 0.8
            elif being.position[1] > self.viewport.height - padding:
                being.position[1] = self.viewport.height - padding
                being.velocity[1] = -abs(being.velocity[1]) * 0.8
            
            # Update appearance
            being.shape_phase += being.shape_evolution_rate * dt
            being.hue_drift_phase += being.hue_drift_rate * dt
            
            # Update tentacle angles - make them rotate slowly around the body
            for tentacle in being.tentacle_params:
                # Slowly rotate the base angle
                tentacle['angle'] += np.random.uniform(-0.05, 0.05) * dt
                
                # Add some organic drift to the rotation
                tentacle['angle'] += np.sin(current_time * 0.3 + tentacle['wave_phase']) * 0.2 * dt
                
                # Keep angle in 0-2π range
                tentacle['angle'] = tentacle['angle'] % (2 * np.pi)
                
                # Occasionally adjust wave parameters for variety
                if np.random.random() < 0.01:
                    tentacle['wave_rate'] += np.random.uniform(-0.1, 0.1)
                    tentacle['wave_rate'] = np.clip(tentacle['wave_rate'], 0.3, 2.5)
                
                # Occasionally adjust length slightly
                if np.random.random() < 0.005:
                    tentacle['length'] += np.random.uniform(-2, 2)
                    tentacle['length'] = np.clip(tentacle['length'], 10, 40)
            
            # Update pulses
            remaining_pulses = []
            for pulse in being.color_pulses:
                pulse['age'] += dt
                if pulse['age'] < pulse['duration']:
                    remaining_pulses.append(pulse)
            being.color_pulses = remaining_pulses
        
        # Check for communication events
        if current_time >= self.next_communication and len(self.beings) > 1:
            sender, receiver = np.random.choice(self.beings, 2, replace=False)
            
            pulse = {
                'age': 0.0,
                'duration': np.random.uniform(1.0, 3.0),
                'hue': sender.base_hue
            }
            sender.color_pulses.append(pulse)
            
            self.next_communication = current_time + np.random.uniform(3, 8)
            
            if np.random.random() < 0.3:
                receiver.target_behavior = np.random.randint(0, 3)
                receiver.target_entity = sender
                receiver.last_behavior_change = current_time

    
    def render(self, state: Dict):
        """Render fog beings using shader"""
        if not self.enabled or not self.shader:
            return
        
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
        
        # Render fullscreen quad
        glBindVertexArray(self.VAO)
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)
