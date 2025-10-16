"""
Complete cactus effect - rendering + event integration
Everything needed for dancing cactuses in one place!
"""
import numpy as np
import time
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import sys
sys.path.append('..')
from corefunctions.shader_effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_cactus(state, outstate, intensity=1.0):
    """
    Shader-based cactus effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_cactus, intensity=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        intensity: Effect intensity multiplier
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
    
    # Initialize cactus effect on first call
    if state['count'] == 0:
        print(f"Initializing cactus effect for frame {frame_id}")
        
        try:
            cactus_effect = viewport.add_effect(
                CactusEffect,
                intensity=intensity
            )
            state['cactus_effect'] = cactus_effect
            state['start_time'] = time.time()
            state['duration'] = state.get('duration', 60)  # Store duration for fade calculation
            print(f"✓ Initialized shader cactus for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize cactus: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update effect parameters
    if 'cactus_effect' in state:
        effect = state['cactus_effect']
        current_time = time.time()
        elapsed = current_time - state['start_time']
        
        # Calculate fade with 4 second fade in/out
        fade_duration = 4.0
        total_duration = state.get('duration', 60)
        
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
        if 'cactus_effect' in state:
            print(f"Cleaning up cactus effect for frame {frame_id}")
            viewport.effects.remove(state['cactus_effect'])
            state['cactus_effect'].cleanup()
            print(f"✓ Cleaned up shader cactus for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class CactusEffect(ShaderEffect):
    """GPU-based cactus effect using fragment shaders"""
    
    def __init__(self, viewport, intensity: float = 1.0):
        super().__init__(viewport)
        self.intensity = intensity
        self.fade_factor = 0.0
        
        # Initialize random state
        self.rng = np.random.RandomState(int(time.time()))
                # Add depth positioning
        self.cactus_depth = .45  # Middle depth (0=front, 1=back)
        
        # Wind parameters
        self.wind_direction = self.rng.uniform(0, 2*np.pi)
        self.wind_strength = 0.0
        self.wind_target = self.rng.uniform(0.3, 0.8)
        self.wind_change_time = time.time()
        self.wind_change_interval = self.rng.uniform(3, 7)
        
        # Cactus parameters
        self.cactus_x = 0.5  # Normalized (0-1)
        self.cactus_y = 0.83  # Ground level
        self.cactus_size = 0.0625  # Base width
        self.cactus_height = 0.7  # Height
        self.sway_phase = 0.0
        self.color_variation = self.rng.uniform(-0.03, 0.03)
        
        # Arms
        self.left_arm_height = 0.45
        self.left_arm_out = 0.1
        self.left_arm_up = 0.117
        self.left_sway_phase = self.rng.uniform(0, 2*np.pi)
        
        self.right_arm_height = 0.35
        self.right_arm_out = 0.117
        self.right_arm_up = 0.133
        self.right_sway_phase = self.rng.uniform(0, 2*np.pi)
        
        # Eye parameters
        self.eye_y = 0.305  # Face position
        self.eye_target_x = 0.0
        self.eye_target_y = 0.0
        self.eye_current_x = 0.0
        self.eye_current_y = 0.0
        self.eye_movement_time = time.time()
        self.eye_movement_interval = self.rng.uniform(2.0, 4.0)
        
        # Eye blinking
        self.eye_blink_state = 'open'
        self.eye_blink_start = time.time()
        self.eye_blink_progress = 0.0
        self.eye_close_duration = 0.15
        self.eye_open_duration = 0.2
        self.eye_closed_duration = 0.1
        self.eye_open_interval = self.rng.uniform(5, 10)
        
        # Pupil blink
        self.pupil_blink = False
        self.pupil_blink_time = time.time()
        self.pupil_blink_interval = self.rng.uniform(3, 8)
        
        # Decorations (rocks and plants)
        self.decorations = []
        num_decorations = self.rng.randint(5, 10)
        for _ in range(num_decorations):
            self.decorations.append({
                'type': self.rng.randint(0, 2),  # 0=rock, 1=plant
                'x': self.rng.uniform(0.1, 0.9),
                'y': self.rng.uniform(0.83, 0.97),
                'size': self.rng.uniform(0.008, 0.025)
            })
        
        # Texture seeds
        self.body_texture_seed = self.rng.uniform(0, 100)
        self.left_arm_seed = self.rng.uniform(0, 100)
        self.right_arm_seed = self.rng.uniform(0, 100)
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        out vec2 fragCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            // Convert from clip space (-1,1) to texture coords (0,1)
            fragCoord = position * 0.5 + 0.5;
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
        uniform float fade;
        
        // Cactus parameters
        uniform vec2 cactusPos;
        uniform float cactusSize;
        uniform float cactusHeight;
        uniform float swayPhase;
        uniform float colorVar;
        uniform float cactusDepth;  // Base depth position of cactus

        
        // Arms
        uniform float leftArmHeight;
        uniform float leftArmOut;
        uniform float leftArmUp;
        uniform float leftSwayPhase;
        uniform float rightArmHeight;
        uniform float rightArmOut;
        uniform float rightArmUp;
        uniform float rightSwayPhase;
        
        // Wind
        uniform float windStrength;
        
        // Eye
        uniform float eyeY;
        uniform vec2 eyePos;
        uniform float eyeBlinkProgress;
        uniform float pupilBlink;
        
        // Texture seeds
        uniform float bodyTextureSeed;
        uniform float leftArmSeed;
        uniform float rightArmSeed;
        
        // Decorations (up to 10)
        uniform int numDecorations;
        uniform vec4 decorations[10];  // x, y, size, type
        
        // Helper functions
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        float sdEllipse(vec2 p, vec2 ab) {
            p = abs(p);
            if(p.x > p.y) { p = p.yx; ab = ab.yx; }
            float l = ab.y*ab.y - ab.x*ab.x;
            float m = ab.x*p.x/l; float m2 = m*m;
            float n = ab.y*p.y/l; float n2 = n*n;
            float c = (m2+n2-1.0)/3.0; float c3 = c*c*c;
            float q = c3 + m2*n2*2.0;
            float d = c3 + m2*n2;
            float g = m + m*n2;
            float co;
            if(d < 0.0) {
                float h = acos(q/c3)/3.0;
                float s = cos(h);
                float t = sin(h)*sqrt(3.0);
                float rx = sqrt(-c*(s + t + 2.0) + m2);
                float ry = sqrt(-c*(s - t + 2.0) + m2);
                co = (ry+sign(l)*rx+abs(g)/(rx*ry)- m)/2.0;
            } else {
                float h = 2.0*m*n*sqrt(d);
                float s = sign(q+h)*pow(abs(q+h), 1.0/3.0);
                float u = sign(q-h)*pow(abs(q-h), 1.0/3.0);
                float rx = -s - u - c*4.0 + 2.0*m2;
                float ry = (s - u)*sqrt(3.0);
                float rm = sqrt(rx*rx + ry*ry);
                co = (ry/sqrt(rm-rx)+2.0*g/rm-m)/2.0;
            }
            vec2 r = ab * vec2(co, sqrt(1.0-co*co));
            return length(r-p) * sign(p.y-r.y);
        }
        
        void main() {
            vec2 uv = fragCoord;
            vec2 p = vec2(uv.x, 1.0 - uv.y);
            
            // Start with transparent and far depth
            vec4 color = vec4(0.0);
            float depth = 1.0;  // Far plane
            bool hasPixel = false;
            
            // Draw ground (furthest back)
            if (p.y > 0.83) {
                vec3 groundColor = hsv2rgb(vec3(0.08, 0.5, 0.3));
                color = vec4(groundColor, fade);
                depth = 0.95;  // Near back of scene
                hasPixel = true;
            }
            
            // Draw decorations (on ground, slightly forward)
            for (int i = 0; i < numDecorations && i < 10; i++) {
                vec2 decPos = decorations[i].xy;
                float decSize = decorations[i].z;
                float decType = decorations[i].w;
                
                float dist = length(p - decPos);
                if (dist < decSize) {
                    float decorationDepth = 0.9 - (decPos.x * 0.05);  // Vary by x position
                    
                    if (decorationDepth < depth) {
                        if (decType < 0.5) {
                            // Rock
                            vec3 rockColor = hsv2rgb(vec3(0.0, 0.1, 0.4));
                            color = vec4(rockColor, fade);
                        } else {
                            // Plant
                            vec3 plantColor = hsv2rgb(vec3(0.3, 0.6, 0.3));
                            color = vec4(plantColor, fade);
                        }
                        depth = decorationDepth;
                        hasPixel = true;
                    }
                }
            }
            
            // Calculate sway
            float sway = sin(swayPhase) * windStrength * 0.05;
            
            // Draw cactus body (depth varies with height and sway)
            vec2 bodyCenter = cactusPos;
            float height = cactusHeight;
            
            for (float i = 0.0; i < 1.0; i += 0.025) {
                float yOffset = -height * i;
                float heightRatio = i;
                float localSway = sway * heightRatio * heightRatio;
                
                vec2 segCenter = vec2(bodyCenter.x + localSway, bodyCenter.y + yOffset);
                float localWidth = cactusSize * (0.8 + 0.2 * (1.0 - heightRatio));
                
                vec2 diff = p - segCenter;
                float distSq = (diff.x * diff.x) / (localWidth * localWidth) + 
                              (diff.y * diff.y) / 0.0004;
                
                if (distSq < 1.0) {
                    // Depth based on height and horizontal position
                    // Higher parts are further forward, sway affects depth
                    float bodyDepth = cactusDepth - heightRatio * 0.15 + localSway * 0.1;
                    
                    if (bodyDepth < depth) {
                        float noise = hash(p * 20.0 + bodyTextureSeed) * 0.08;
                        float sinPattern = sin(p.x * resolution.x * 0.8 + 123.0) * 0.05;
                        float cosPattern = cos(p.y * resolution.y * 0.5 + 369.0) * 0.04;
                        float texture = noise + sinPattern + cosPattern;
                        
                        vec3 bodyColor = hsv2rgb(vec3(
                            0.33 + colorVar + texture * 0.1,
                            0.6 + texture * 0.2,
                            0.4 + texture * 0.3
                        ));
                        color = vec4(bodyColor, fade);
                        depth = bodyDepth;
                        hasPixel = true;
                    }
                }
            }
            
            // Draw left arm (extends outward, so depth varies)
            {
                float armHeight = leftArmHeight * height;
                vec2 armBase = vec2(
                    bodyCenter.x - cactusSize * 0.45 + sway * (armHeight/height) * (armHeight/height),
                    bodyCenter.y - armHeight
                );
                float armSway = sin(leftSwayPhase) * windStrength * 0.03;
                float baseArmDepth = cactusDepth - (armHeight/height) * 0.15;
                
                // Horizontal segment (extends left, gets further away)
                for (float t = 0.0; t < 1.0; t += 0.2) {
                    vec2 segPos = vec2(
                        armBase.x - leftArmOut * t + armSway * t,
                        armBase.y
                    );
                    float thickness = cactusSize * 0.6 * (1.0 - 0.2 * t);
                    
                    vec2 diff = p - segPos;
                    float distSq = (diff.x * diff.x) / (thickness * thickness) + 
                                  (diff.y * diff.y) / 0.0004;
                    
                    if (distSq < 1.0) {
                        // Arm extends back as it goes out
                        float armDepth = baseArmDepth + t * 0.1 + armSway * 0.05;
                        
                        if (armDepth < depth) {
                            float noise = hash(p * 20.0 + leftArmSeed) * 0.08;
                            vec3 armColor = hsv2rgb(vec3(0.33 + colorVar + noise * 0.1, 0.6, 0.4));
                            color = vec4(armColor, fade);
                            depth = armDepth;
                            hasPixel = true;
                        }
                    }
                }
                
                // Vertical segment (going up, comes forward)
                vec2 elbow = vec2(armBase.x - leftArmOut + armSway, armBase.y);
                for (float t = 0.0; t < 1.0; t += 0.143) {
                    vec2 segPos = vec2(
                        elbow.x + armSway * 0.2 * (1.0 + t),
                        elbow.y - leftArmUp * t
                    );
                    float thickness = cactusSize * 0.6 * (1.0 - 0.3 * t);
                    
                    vec2 diff = p - segPos;
                    float distSq = (diff.x * diff.x) / (thickness * thickness) + 
                                  (diff.y * diff.y) / 0.0004;
                    
                    if (distSq < 1.0) {
                        // Going up brings it forward
                        float armDepth = baseArmDepth + 0.1 - t * 0.08;
                        
                        if (armDepth < depth) {
                            float noise = hash(p * 20.0 + leftArmSeed) * 0.08;
                            vec3 armColor = hsv2rgb(vec3(0.33 + colorVar + noise * 0.1, 0.6, 0.4));
                            color = vec4(armColor, fade);
                            depth = armDepth;
                            hasPixel = true;
                        }
                    }
                }
            }
            
            // Draw right arm (similar depth logic)
            {
                float armHeight = rightArmHeight * height;
                vec2 armBase = vec2(
                    bodyCenter.x + cactusSize * 0.45 + sway * (armHeight/height) * (armHeight/height),
                    bodyCenter.y - armHeight
                );
                float armSway = sin(rightSwayPhase) * windStrength * 0.03;
                float baseArmDepth = cactusDepth - (armHeight/height) * 0.15;
                
                // Horizontal segment
                for (float t = 0.0; t < 1.0; t += 0.2) {
                    vec2 segPos = vec2(
                        armBase.x + rightArmOut * t + armSway * t,
                        armBase.y
                    );
                    float thickness = cactusSize * 0.6 * (1.0 - 0.2 * t);
                    
                    vec2 diff = p - segPos;
                    float distSq = (diff.x * diff.x) / (thickness * thickness) + 
                                  (diff.y * diff.y) / 0.0004;
                    
                    if (distSq < 1.0) {
                        float armDepth = baseArmDepth + t * 0.1 + armSway * 0.05;
                        
                        if (armDepth < depth) {
                            float noise = hash(p * 20.0 + rightArmSeed) * 0.08;
                            vec3 armColor = hsv2rgb(vec3(0.33 + colorVar + noise * 0.1, 0.6, 0.4));
                            color = vec4(armColor, fade);
                            depth = armDepth;
                            hasPixel = true;
                        }
                    }
                }
                
                // Vertical segment
                vec2 elbow = vec2(armBase.x + rightArmOut + armSway, armBase.y);
                for (float t = 0.0; t < 1.0; t += 0.143) {
                    vec2 segPos = vec2(
                        elbow.x + armSway * 0.2 * (1.0 + t),
                        elbow.y - rightArmUp * t
                    );
                    float thickness = cactusSize * 0.6 * (1.0 - 0.3 * t);
                    
                    vec2 diff = p - segPos;
                    float distSq = (diff.x * diff.x) / (thickness * thickness) + 
                                  (diff.y * diff.y) / 0.0004;
                    
                    if (distSq < 1.0) {
                        float armDepth = baseArmDepth + 0.1 - t * 0.08;
                        
                        if (armDepth < depth) {
                            float noise = hash(p * 20.0 + rightArmSeed) * 0.08;
                            vec3 armColor = hsv2rgb(vec3(0.33 + colorVar + noise * 0.1, 0.6, 0.4));
                            color = vec4(armColor, fade);
                            depth = armDepth;
                            hasPixel = true;
                        }
                    }
                }
            }
            
            // Draw eye (at front of cactus face)
            vec2 faceCenter = vec2(
                bodyCenter.x + sway * (eyeY/height) * (eyeY/height),
                eyeY
            );
            
            float faceHeightRatio = eyeY / height;
            float faceDepth = cactusDepth - faceHeightRatio * 0.15 - 0.05;  // Face is at front
            
            float eyeRadiusX = cactusSize * 1.2 * 0.75;
            float eyeRadiusY = cactusSize * 1.0 * 0.75 * (1.0 - 0.9 * eyeBlinkProgress);
            
            float eyeDist = sdEllipse(p - faceCenter, vec2(eyeRadiusX, eyeRadiusY));
            
            if (eyeDist < 0.0 && faceDepth < depth) {
                // Eye white
                vec3 eyeWhite = hsv2rgb(vec3(0.6, 0.5, 0.1));
                color = vec4(eyeWhite, fade * 0.8);
                depth = faceDepth;
                hasPixel = true;
                
                // Iris (slightly forward)
                if (eyeBlinkProgress < 0.7) {
                    float irisVisibility = 1.0 - (eyeBlinkProgress / 0.7);
                    
                    vec2 irisCenter = faceCenter + vec2(
                        eyePos.x * eyeRadiusX * 0.8,
                        eyePos.y * eyeRadiusY * 0.6
                    );
                    
                    float irisRadius = cactusSize * 0.9 * 0.75 * irisVisibility;
                    float irisDist = length(p - irisCenter);
                    
                    if (irisDist < irisRadius) {
                        float irisDepth = faceDepth - 0.01;
                        
                        vec2 irisDir = (p - irisCenter) / irisRadius;
                        float pattern = sin(atan(irisDir.y, irisDir.x) * 8.0) * 0.1 + 
                                      length(irisDir) * 0.2;
                        vec3 irisColor = hsv2rgb(vec3(0.55 + pattern, 0.7, 0.6));
                        color = vec4(irisColor, fade * irisVisibility);
                        depth = irisDepth;
                        
                        // Pupil (furthest forward)
                        float pupilSize = 0.6 + sin(time * 1.5) * 0.1;
                        if (pupilBlink > 0.5) pupilSize -= 0.3;
                        pupilSize = clamp(pupilSize, 0.3, 1.0);
                        
                        float pupilRadius = irisRadius * pupilSize * 0.7;
                        if (irisDist < pupilRadius) {
                            color = vec4(0.0, 0.0, 0.0, fade * irisVisibility);
                            depth = irisDepth - 0.01;
                        }
                        
                        // Highlight (at surface)
                        if (eyeBlinkProgress < 0.5) {
                            vec2 highlightOffset = vec2(-pupilRadius * 0.5, -pupilRadius * 0.5);
                            vec2 highlightCenter = irisCenter + highlightOffset;
                            float highlightRadius = cactusSize * 0.15 * 0.75 * (1.0 - eyeBlinkProgress);
                            float highlightDist = length(p - highlightCenter);
                            
                            if (highlightDist < highlightRadius) {
                                float intensity = 1.0 - (highlightDist / highlightRadius);
                                intensity *= (1.0 - eyeBlinkProgress / 0.5);
                                color = vec4(1.0, 1.0, 1.0, fade * intensity);
                                depth = irisDepth - 0.015;
                            }
                        }
                    }
                }
            }
            
            // Eyelid line when blinking (at surface)
            if (eyeBlinkProgress > 0.9 && abs(p.y - faceCenter.y) < 0.005) {
                if (abs(p.x - faceCenter.x) < eyeRadiusX) {
                    if (faceDepth - 0.02 < depth) {
                        color = vec4(0.0, 0.0, 0.0, fade * 0.8);
                        depth = faceDepth - 0.02;
                        hasPixel = true;
                    }
                }
            }
            
            // Only write depth if we actually drew something
            if (hasPixel) {
                gl_FragDepth = depth;
            } else {
                gl_FragDepth = 1.0;  // Far plane for transparent pixels
            }
            
            outColor = color;
        }
        """


    def compile_shader(self):
        """Compile and link cactus shaders"""
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
        """Initialize OpenGL buffers for full-screen quad"""
        # Full-screen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

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
        """Update cactus animation parameters"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Update wind
        if current_time - self.wind_change_time > self.wind_change_interval:
            self.wind_target = self.rng.uniform(0.3, 0.8)
            self.wind_change_time = current_time
            self.wind_change_interval = self.rng.uniform(3, 7)
        
        self.wind_strength += (self.wind_target - self.wind_strength) * dt * 0.5
        
        # Update sway phases
        self.sway_phase += dt * 0.8
        self.left_sway_phase += dt * 0.5
        self.right_sway_phase += dt * 0.5
        
        # Update eye blinking animation
        blink_elapsed = current_time - self.eye_blink_start
        
        if self.eye_blink_state == 'open':
            if blink_elapsed > self.eye_open_interval:
                self.eye_blink_state = 'closing'
                self.eye_blink_start = current_time
                self.eye_blink_progress = 0.0
        
        elif self.eye_blink_state == 'closing':
            self.eye_blink_progress = blink_elapsed / self.eye_close_duration
            if self.eye_blink_progress >= 1.0:
                self.eye_blink_state = 'closed'
                self.eye_blink_start = current_time
                self.eye_blink_progress = 1.0
        
        elif self.eye_blink_state == 'closed':
            if blink_elapsed > self.eye_closed_duration:
                self.eye_blink_state = 'opening'
                self.eye_blink_start = current_time
        
        elif self.eye_blink_state == 'opening':
            open_progress = blink_elapsed / self.eye_open_duration
            self.eye_blink_progress = 1.0 - open_progress
            if self.eye_blink_progress <= 0.0:
                self.eye_blink_state = 'open'
                self.eye_blink_start = current_time
                self.eye_blink_progress = 0.0
                self.eye_open_interval = self.rng.uniform(5, 10)
        
        # Update eye movement
        if current_time - self.eye_movement_time > self.eye_movement_interval:
            angle = self.rng.random() * 2 * np.pi
            r = self.rng.random() * 1.1
            self.eye_target_x = r * np.cos(angle) * 1.2
            self.eye_target_y = r * np.sin(angle)
            self.eye_movement_time = current_time
            self.eye_movement_interval = self.rng.uniform(2.0, 4.0)
        
        # Smooth eye movement
        dx = self.eye_target_x - self.eye_current_x
        dy = self.eye_target_y - self.eye_current_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0.001:
            movement_speed = 2.0
            move_amount = min(distance, movement_speed * dt)
            if distance > 0:
                self.eye_current_x += (dx / distance) * move_amount
                self.eye_current_y += (dy / distance) * move_amount
        
        # Update pupil blink
        if self.pupil_blink:
            if current_time - self.pupil_blink_time > 0.2:
                self.pupil_blink = False
                self.pupil_blink_time = current_time
        elif current_time - self.pupil_blink_time > self.pupil_blink_interval:
            self.pupil_blink = True
            self.pupil_blink_time = current_time
            self.pupil_blink_interval = self.rng.uniform(3, 8)

    def render(self, state: Dict):
        """Render the cactus using shaders"""
        if not self.enabled or not self.shader:
            return
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
        glUseProgram(self.shader)
        
        # Set uniforms
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "time")
        if loc != -1:
            glUniform1f(loc, time.time())
        
        loc = glGetUniformLocation(self.shader, "fade")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        # Cactus depth
        loc = glGetUniformLocation(self.shader, "cactusDepth")
        if loc != -1:
            glUniform1f(loc, self.cactus_depth)

        # Cactus parameters
        loc = glGetUniformLocation(self.shader, "cactusPos")
        if loc != -1:
            glUniform2f(loc, self.cactus_x, self.cactus_y)
        
        loc = glGetUniformLocation(self.shader, "cactusSize")
        if loc != -1:
            glUniform1f(loc, self.cactus_size)
        
        loc = glGetUniformLocation(self.shader, "cactusHeight")
        if loc != -1:
            glUniform1f(loc, self.cactus_height)
        
        loc = glGetUniformLocation(self.shader, "swayPhase")
        if loc != -1:
            glUniform1f(loc, self.sway_phase)
        
        loc = glGetUniformLocation(self.shader, "colorVar")
        if loc != -1:
            glUniform1f(loc, self.color_variation)
        
        # Arms
        loc = glGetUniformLocation(self.shader, "leftArmHeight")
        if loc != -1:
            glUniform1f(loc, self.left_arm_height)
        
        loc = glGetUniformLocation(self.shader, "leftArmOut")
        if loc != -1:
            glUniform1f(loc, self.left_arm_out)
        
        loc = glGetUniformLocation(self.shader, "leftArmUp")
        if loc != -1:
            glUniform1f(loc, self.left_arm_up)
        
        loc = glGetUniformLocation(self.shader, "leftSwayPhase")
        if loc != -1:
            glUniform1f(loc, self.left_sway_phase)
        
        loc = glGetUniformLocation(self.shader, "rightArmHeight")
        if loc != -1:
            glUniform1f(loc, self.right_arm_height)
        
        loc = glGetUniformLocation(self.shader, "rightArmOut")
        if loc != -1:
            glUniform1f(loc, self.right_arm_out)
        
        loc = glGetUniformLocation(self.shader, "rightArmUp")
        if loc != -1:
            glUniform1f(loc, self.right_arm_up)
        
        loc = glGetUniformLocation(self.shader, "rightSwayPhase")
        if loc != -1:
            glUniform1f(loc, self.right_sway_phase)
        
        # Wind
        loc = glGetUniformLocation(self.shader, "windStrength")
        if loc != -1:
            glUniform1f(loc, self.wind_strength)
        
        # Eye
        loc = glGetUniformLocation(self.shader, "eyeY")
        if loc != -1:
            glUniform1f(loc, self.eye_y)
        
        loc = glGetUniformLocation(self.shader, "eyePos")
        if loc != -1:
            glUniform2f(loc, self.eye_current_x, self.eye_current_y)
        
        loc = glGetUniformLocation(self.shader, "eyeBlinkProgress")
        if loc != -1:
            glUniform1f(loc, self.eye_blink_progress)
        
        loc = glGetUniformLocation(self.shader, "pupilBlink")
        if loc != -1:
            glUniform1f(loc, 1.0 if self.pupil_blink else 0.0)
        
        # Texture seeds
        loc = glGetUniformLocation(self.shader, "bodyTextureSeed")
        if loc != -1:
            glUniform1f(loc, self.body_texture_seed)
        
        loc = glGetUniformLocation(self.shader, "leftArmSeed")
        if loc != -1:
            glUniform1f(loc, self.left_arm_seed)
        
        loc = glGetUniformLocation(self.shader, "rightArmSeed")
        if loc != -1:
            glUniform1f(loc, self.right_arm_seed)
        
        # Decorations
        loc = glGetUniformLocation(self.shader, "numDecorations")
        if loc != -1:
            glUniform1i(loc, len(self.decorations))
        
        # Set decoration array
        for i, dec in enumerate(self.decorations[:10]):  # Max 10
            loc = glGetUniformLocation(self.shader, f"decorations[{i}]")
            if loc != -1:
                glUniform4f(loc, dec['x'], dec['y'], dec['size'], float(dec['type']))
        
        # Draw full-screen quad
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Disable blending after rendering
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
