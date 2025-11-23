"""
Complete eye effect - rendering + event integration using shaders
Shader-based implementation following rain.py template
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import time
from corefunctions.shader_effects.base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_eye(state, outstate, num_eyes=6, scale=0.075):
    """
    Shader-based eye effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_eye, num_eyes=5, scale=0.2, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        num_eyes: Number of eyes to render (default 6)
        scale: Size of each eye (default 0.075)
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
    
    # Initialize eye effect on first call
    if state['count'] == 0:
        print(f"Initializing shader eye effect for frame {frame_id} with {num_eyes} eyes")
        
        try:
            eye_effect = viewport.add_effect(EyeEffect, num_eyes=num_eyes, scale=scale)
            state['eye_effect'] = eye_effect
            state['start_time'] = time.time()
            print(f"✓ Initialized shader eye for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize eye: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update eye parameters from global state
    if 'eye_effect' in state:
        current_time = time.time()
        elapsed_time = current_time - state['start_time']
        total_duration = state.get('duration', 30)
        
        # Calculate fade factor
        fade_duration = 5.0
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        fade_factor = np.clip(fade_factor, 0, 1)
        
        state['eye_effect'].fade_factor = fade_factor
        # Update shared parameters from global state
        state['eye_effect'].movement_interval = outstate.get('eye_movement_interval', 3.0)
        state['eye_effect'].movement_speed = outstate.get('eye_movement_speed', 2.0)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'eye_effect' in state:
            print(f"Cleaning up eye effect for frame {frame_id}")
            viewport.effects.remove(state['eye_effect'])
            state['eye_effect'].cleanup()
            print(f"✓ Cleaned up shader eye for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class EyeEffect(ShaderEffect):
    """GPU-based eye effect using instanced rendering"""
    
    def __init__(self, viewport, num_eyes=6, scale=0.075):
        super().__init__(viewport)
        
        # Eye properties
        self.num_eyes = num_eyes
        self.scale = scale  # Size multiplier for all eyes
        self.fade_factor = 0.0
        self.instance_VBO = None
        self.wrap_margin = 200  # Distance from edge to create duplicates (larger for eye size)
        
        # Vectorized eye data (all stored as numpy arrays)
        self.positions = None  # x, y, z positions
        self.scales = None  # Individual scale multipliers
        self.pupil_sizes = None  # Individual pupil sizes
        self.iris_offsets = None  # x, y iris movement offsets
        
        # Movement parameters (shared by all eyes)
        self.movement_interval = 3.0
        self.movement_speed = 2.0
        self.blink_interval = 7.0
        
        # Movement state (per eye)
        self.start_time = time.time()
        self.last_movement_times = None
        self.target_offsets = None
        
        self._initialize_eyes()
        
    def _initialize_eyes(self):
        """Initialize all eye data as numpy arrays"""
        n = self.num_eyes
        
        # Random positions across the screen
        self.positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n),  # x
            np.random.uniform(0, self.viewport.height, n),  # y
            np.random.uniform(10, 60, n)  # z (depth: 10-60, mid-range)
        ])
        
        # Individual scale variations (0.8 to 1.2x base scale)
        self.scales = np.random.uniform(0.8, 1.2, n)
        
        # Initial pupil sizes
        self.pupil_sizes = np.ones(n)
        
        # Iris offsets (movement)
        self.iris_offsets = np.zeros((n, 2))
        
        # Movement targets
        self.target_offsets = np.zeros((n, 2))
        
        # Last movement times
        self.last_movement_times = np.zeros(n) + self.start_time
        
        # Rotation angles (in radians)
        self.rotations = np.random.uniform(0, 2 * np.pi, n)
        self.rotation_speeds = np.random.uniform(-0.2, 0.2, n)  # rad/sec
        
        # Depth movement (z-axis)
        self.target_depths = self.positions[:, 2].copy()
        self.depth_speeds = np.random.uniform(5, 15, n)  # pixels/sec
        self.last_depth_change_times = np.zeros(n) + self.start_time
        self.depth_change_intervals = np.random.uniform(3, 8, n)  # seconds
        
        # Size pulsing
        self.base_scales = self.scales.copy()
        self.scale_phases = np.random.uniform(0, 2 * np.pi, n)  # Random phase offsets
        self.scale_speeds = np.random.uniform(0.3, 0.8, n)  # Speed variation
        
        # Position movement targets and speeds (independent per eye)
        self.target_positions = self.positions.copy()
        self.position_speeds = np.random.uniform(20, 60, n)  # pixels/sec
        self.last_position_change_times = np.zeros(n) + self.start_time
        self.position_change_intervals = np.random.uniform(4, 10, n)  # seconds
        
        # Iris movement intervals (different per eye)
        self.iris_movement_intervals = np.random.uniform(2, 5, n)
        
        # Blink state (per eye)
        self.blink_amounts = np.zeros(n)  # 0 = open, 1 = closed
        self.next_blink_times = np.zeros(n) + self.start_time + np.random.uniform(2, 8, n)
        self.blink_speeds = np.random.uniform(8, 12, n)  # Blinks per second (speed)
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Base quad vertices
        layout(location = 1) in vec3 offset;    // x, y, z (instance)
        layout(location = 2) in float eyeScale; // Individual scale (instance)
        layout(location = 3) in vec2 irisOffset; // Iris movement (instance)
        layout(location = 4) in float pupilSize; // Pupil size (instance)
        layout(location = 5) in float rotation; // Rotation angle (instance)
        layout(location = 6) in float blinkAmount; // Blink state 0-1 (instance)
        
        out vec2 fragCoord;  // Pass to fragment shader
        out vec2 fragIrisOffset;
        out float fragPupilSize;
        out float fragBlinkAmount;
        
        uniform vec2 resolution;
        uniform float globalScale;  // Global scale multiplier
        
        void main() {
            // Calculate eye size in pixels
            float eyeSize = globalScale * eyeScale * min(resolution.x, resolution.y);
            
            // Apply rotation to quad vertices
            float cosR = cos(rotation);
            float sinR = sin(rotation);
            vec2 rotatedPos = vec2(
                position.x * cosR - position.y * sinR,
                position.x * sinR + position.y * cosR
            );
            
            // Scale the rotated quad vertices
            vec2 scaledPos = rotatedPos * eyeSize;
            
            // Position in screen space (pixels)
            vec2 screenPos = scaledPos + offset.xy;
            
            // Convert to clip space
            vec2 clipPos = (screenPos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;  // Flip Y for screen coords
            
            // Map depth (z=0 near, z=100 far)
            float depth = offset.z / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Pass data to fragment shader
            fragCoord = position;  // Quad space (-1 to 1)
            fragIrisOffset = irisOffset;
            fragPupilSize = pupilSize;
            fragBlinkAmount = blinkAmount;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragCoord;  // Quad space (-1 to 1)
        in vec2 fragIrisOffset;
        in float fragPupilSize;
        in float fragBlinkAmount;
        
        out vec4 outColor;
        
        uniform float fadeAlpha;
        uniform float time;
        
        // Convert HSVA to RGBA
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            // UV in quad space (already -1 to 1 from vertex shader)
            vec2 uv = fragCoord;
            
            // Apply blink effect - squash vertically
            float blinkSquash = 1.0 - (fragBlinkAmount * 0.95);  // Squash to 5% height when fully closed
            uv.y /= blinkSquash;
            
            // Eye dimensions (ellipse) - smaller white part
            float eyeWidth = 1.0;
            float eyeHeight = 0.7;
            
            // Distance from center (normalized to ellipse)
            float dist = length(vec2(uv.x / eyeWidth, uv.y / eyeHeight));
            
            // Discard pixels outside the eye
            if (dist > 1.0) {
                discard;
            }
            
            // Sclera (white of eye) - reduced opacity
            vec3 color = vec3(0.95, 0.95, 0.98);
            float alpha = fadeAlpha * 0.5;
            
            // Calculate distortion based on horizontal position
            float h_stretch = 1.0;
            float v_stretch = 1.0 + (0.4 * abs(fragIrisOffset.x));
            
            // Iris position
            vec2 irisCenter = fragIrisOffset * vec2(0.6, 0.4);  // Scale movement
            vec2 toIris = uv - irisCenter;
            float irisDist = length(vec2(toIris.x / h_stretch, toIris.y / v_stretch));
            
            float irisRadius = 0.4;
            
            if (irisDist < irisRadius) {
                // Iris pattern
                float angle = atan(toIris.y / v_stretch, toIris.x / h_stretch);
                float distRatio = irisDist / irisRadius;
                float pattern = (sin(angle * 8.0) * 0.1) + (distRatio * 0.2);
                
                // Use HSV for iris color
                vec3 hsv = vec3(0.55 + pattern, 0.7, 0.5);
                color = hsv2rgb(hsv);
                alpha = fadeAlpha * 0.8;
            }
            
            // Pupil with breathing and blink variation
            float breathingVar = sin(time * 1.5) * 0.1;
            float blinkPhase = mod(time, 7.0) / 7.0;
            float blinkVar = 0.0;
            if (blinkPhase < 0.1) {
                blinkVar = -0.3 * sin(blinkPhase * 31.416);  // ~10*pi
            }
            
            float currentPupilSize = clamp(fragPupilSize + breathingVar + blinkVar, 0.3, 1.0);
            float pupilRadius = 0.2 * currentPupilSize;
            
            vec2 toPupil = uv - irisCenter;
            float pupilDist = length(vec2(toPupil.x / h_stretch, toPupil.y / v_stretch));
            
            if (pupilDist < pupilRadius) {
                // Black pupil
                color = vec3(0.0, 0.0, 0.0);
                alpha = fadeAlpha * 0.9;
                
                // Highlight
                vec2 highlightPos = irisCenter + vec2(-pupilRadius * 0.5, -pupilRadius * 0.5);
                vec2 toHighlight = uv - highlightPos;
                float highlightDist = length(vec2(toHighlight.x / h_stretch, toHighlight.y / v_stretch));
                float highlightRadius = 0.08;
                
                if (highlightDist < highlightRadius) {
                    float intensity = 1.0 - (highlightDist / highlightRadius);
                    color = mix(color, vec3(1.0), intensity);
                    alpha = fadeAlpha * intensity;
                }
            }
            
            outColor = vec4(color, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link eye shaders"""
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
        """Initialize OpenGL buffers for instanced eye rendering"""
        # Base quad vertices (centered at origin, wider to accommodate ellipse)
        # Eye ellipse is 1.0 wide x 0.7 tall, so make quad 1.0 wide
        vertices = np.array([
            -1.0, -0.7,  # Bottom left
             1.0, -0.7,  # Bottom right
             1.0,  0.7,  # Top right
            -1.0,  0.7   # Top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO (depth testing is globally enabled)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer (base quad)
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Instance buffer (will be updated each frame)
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        self.VBOs.append(self.instance_VBO)
        
        # Allocate instance buffer (positions, scales, iris offsets, pupil sizes, rotation, blink)
        # Layout: vec3 offset, float scale, vec2 irisOffset, float pupilSize, float rotation, float blink = 9 floats per instance
        instance_data = np.zeros((self.num_eyes, 9), dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Setup instance attributes
        stride = 9 * 4  # 9 floats * 4 bytes
        
        # Offset (vec3) - location 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)  # Advance once per instance
        
        # Scale (float) - location 2
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Iris offset (vec2) - location 3
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Pupil size (float) - location 4
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Rotation (float) - location 5
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
        # Blink amount (float) - location 6
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)


    def update(self, dt: float, state: Dict):
        """Update eye movement for all eyes (vectorized)"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Update rotations (continuous, different speeds per eye)
        self.rotations += self.rotation_speeds * dt
        self.rotations = np.fmod(self.rotations, 2 * np.pi)
        
        # Update scale pulsing (different phase and speed per eye)
        self.scale_phases += self.scale_speeds * dt
        pulse_factors = 1.0 + 0.15 * np.sin(self.scale_phases)  # Pulse between 0.85x and 1.15x
        self.scales = self.base_scales * pulse_factors
        
        # Update blinks (independent per eye)
        needs_blink = current_time >= self.next_blink_times
        
        if np.any(needs_blink):
            # Schedule next blink (random interval between 3-10 seconds)
            self.next_blink_times[needs_blink] = current_time + np.random.uniform(3, 10, np.sum(needs_blink))
        
        # Start new blinks
        self.blink_amounts[needs_blink] = 0.01  # Start closing
        
        # Animate blink open/close
        currently_blinking = self.blink_amounts > 0
        
        # Continue existing blinks
        if np.any(currently_blinking):
            # Calculate blink deltas for all eyes
            blink_deltas = self.blink_speeds * dt
            
            # Eyes closing (0 to 1)
            closing_mask = currently_blinking & (self.blink_amounts < 1.0)
            if np.any(closing_mask):
                self.blink_amounts[closing_mask] += blink_deltas[closing_mask]
                self.blink_amounts[closing_mask] = np.minimum(self.blink_amounts[closing_mask], 1.0)
            
            # Eyes opening (1 to 0)
            opening_mask = currently_blinking & (self.blink_amounts >= 1.0)
            if np.any(opening_mask):
                self.blink_amounts[opening_mask] += blink_deltas[opening_mask]
                # When blink completes (reaches 2.0), reset to 0
                completed = self.blink_amounts >= 2.0
                self.blink_amounts[completed] = 0.0
        
        # Position movement (independent per eye)
        time_since_position_change = current_time - self.last_position_change_times
        needs_new_position = time_since_position_change > self.position_change_intervals
        
        if np.any(needs_new_position):
            # Generate new random positions for eyes that need them
            new_x = np.random.uniform(0, self.viewport.width, self.num_eyes)
            new_y = np.random.uniform(0, self.viewport.height, self.num_eyes)
            new_positions = np.column_stack([new_x, new_y, self.target_depths])
            
            self.target_positions[needs_new_position] = new_positions[needs_new_position]
            self.last_position_change_times[needs_new_position] = current_time
        
        # Smoothly move positions toward targets
        position_deltas = self.target_positions - self.positions
        position_deltas[:, 2] = 0  # Don't move Z through position movement
        position_distances = np.linalg.norm(position_deltas, axis=1)
        position_moving_mask = position_distances > 0.5
        
        if np.any(position_moving_mask):
            move_amounts = np.minimum(position_distances[position_moving_mask], self.position_speeds[position_moving_mask] * dt)
            directions = position_deltas[position_moving_mask] / position_distances[position_moving_mask, np.newaxis]
            self.positions[position_moving_mask] += directions * move_amounts[:, np.newaxis]
        
        # Depth movement (independent per eye)
        time_since_depth_change = current_time - self.last_depth_change_times
        needs_new_depth = time_since_depth_change > self.depth_change_intervals
        
        if np.any(needs_new_depth):
            # Generate new random depths
            self.target_depths[needs_new_depth] = np.random.uniform(10, 60, np.sum(needs_new_depth))
            self.last_depth_change_times[needs_new_depth] = current_time
        
        # Smoothly move depths toward targets
        depth_deltas = self.target_depths - self.positions[:, 2]
        depth_distances = np.abs(depth_deltas)
        depth_moving_mask = depth_distances > 0.1
        
        if np.any(depth_moving_mask):
            depth_move_amounts = np.minimum(depth_distances[depth_moving_mask], self.depth_speeds[depth_moving_mask] * dt)
            depth_directions = np.sign(depth_deltas[depth_moving_mask])
            self.positions[depth_moving_mask, 2] += depth_directions * depth_move_amounts
        
        # Iris movement (independent intervals per eye)
        time_since_iris_movement = current_time - self.last_movement_times
        needs_new_iris_target = time_since_iris_movement > self.iris_movement_intervals
        
        if np.any(needs_new_iris_target):
            # Generate new random iris targets
            angles = np.random.random(self.num_eyes) * 2 * np.pi
            max_radius = 0.7
            radii = np.random.random(self.num_eyes) * max_radius
            
            new_targets = np.column_stack([
                radii * np.cos(angles) * 1.5,
                radii * np.sin(angles)
            ])
            
            self.target_offsets[needs_new_iris_target] = new_targets[needs_new_iris_target]
            self.last_movement_times[needs_new_iris_target] = current_time
        
        # Smoothly move iris offsets toward targets
        iris_deltas = self.target_offsets - self.iris_offsets
        iris_distances = np.linalg.norm(iris_deltas, axis=1)
        iris_moving_mask = iris_distances > 0.001
        
        if np.any(iris_moving_mask):
            move_amounts = np.minimum(iris_distances[iris_moving_mask], self.movement_speed * dt)
            directions = iris_deltas[iris_moving_mask] / iris_distances[iris_moving_mask, np.newaxis]
            self.iris_offsets[iris_moving_mask] += directions * move_amounts[:, np.newaxis]

    def render(self, state: Dict):
        """Render all eyes using instanced rendering with horizontal wrapping"""
        if not self.enabled or not self.shader:
            return
        
        # Blend state is globally enabled - don't toggle it
        glUseProgram(self.shader)
        
        # Set uniforms
        current_time = time.time() - self.start_time
        
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "globalScale")
        if loc != -1:
            glUniform1f(loc, self.scale)
        
        loc = glGetUniformLocation(self.shader, "fadeAlpha")
        if loc != -1:
            glUniform1f(loc, self.fade_factor)
        
        loc = glGetUniformLocation(self.shader, "time")
        if loc != -1:
            glUniform1f(loc, current_time)
        
        # Horizontal wrapping: detect eyes near edges
        left_edge_mask = self.positions[:, 0] < self.wrap_margin
        right_edge_mask = self.positions[:, 0] > (self.viewport.width - self.wrap_margin)
        
        duplicate_positions_left = []
        duplicate_positions_right = []
        duplicate_indices_left = []
        duplicate_indices_right = []
        
        if np.any(left_edge_mask):
            # Eyes near left edge need duplicates on the right
            left_indices = np.where(left_edge_mask)[0]
            duplicate_pos = self.positions[left_indices].copy()
            duplicate_pos[:, 0] += self.viewport.width  # Shift to right side
            duplicate_positions_right.append(duplicate_pos)
            duplicate_indices_right.append(left_indices)
        
        if np.any(right_edge_mask):
            # Eyes near right edge need duplicates on the left
            right_indices = np.where(right_edge_mask)[0]
            duplicate_pos = self.positions[right_indices].copy()
            duplicate_pos[:, 0] -= self.viewport.width  # Shift to left side
            duplicate_positions_left.append(duplicate_pos)
            duplicate_indices_left.append(right_indices)
        
        # Combine primary eyes with duplicates
        all_positions = [self.positions]
        all_indices = [np.arange(len(self.positions))]
        
        if duplicate_positions_right:
            all_positions.extend(duplicate_positions_right)
            all_indices.extend(duplicate_indices_right)
        
        if duplicate_positions_left:
            all_positions.extend(duplicate_positions_left)
            all_indices.extend(duplicate_indices_left)
        
        combined_positions = np.vstack(all_positions)
        combined_indices = np.concatenate(all_indices)
        
        # Get attributes for all eyes (primary + duplicates reference the same attributes)
        combined_scales = self.scales[combined_indices]
        combined_iris_offsets = self.iris_offsets[combined_indices]
        combined_pupil_sizes = self.pupil_sizes[combined_indices]
        combined_rotations = self.rotations[combined_indices]
        combined_blink_amounts = self.blink_amounts[combined_indices]
        
        # Calculate smooth blink curve (0->1->0 from linear 0->2)
        smooth_blink_amounts = np.where(combined_blink_amounts <= 1.0,
                                        combined_blink_amounts,  # First half: 0 to 1
                                        2.0 - combined_blink_amounts)  # Second half: 1 to 0
        smooth_blink_amounts = np.clip(smooth_blink_amounts, 0.0, 1.0)
        
        # Build instance data: positions (3), scale (1), iris offset (2), pupil size (1), rotation (1), blink (1) = 9 floats
        instance_data = np.column_stack([
            combined_positions,  # x, y, z (3 floats)
            combined_scales,  # scale (1 float)
            combined_iris_offsets,  # iris x, y (2 floats)
            combined_pupil_sizes,  # pupil size (1 float)
            combined_rotations,  # rotation (1 float)
            smooth_blink_amounts  # blink amount (1 float)
        ]).astype(np.float32)
        
        # Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Draw instanced eyes
        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(combined_positions))
        glBindVertexArray(0)
        
        glUseProgram(0)
