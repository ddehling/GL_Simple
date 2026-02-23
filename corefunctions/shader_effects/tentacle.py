"""
Tentacle shader effect - Large animated tentacle descending from top
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_tentacle(state, outstate, depth=50.0, intensity=1.0, speed=1.0,
                    width_ratio=0.15, extend_duration=10.0, withdraw_duration=10.0):
    """
    Shader-based tentacle effect compatible with EventScheduler
    
    Creates a large organic tentacle that extends from the top of the screen
    over the first 10 seconds, wiggles around during the middle portion,
    and withdraws in the last 10 seconds.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_tentacle, width_ratio=0.2, 
                               extend_duration=10, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        depth: Z-depth of tentacle (0=near, 100=far, default: 50)
        intensity: Base brightness multiplier (default: 1.0)
        speed: Animation speed multiplier (default: 1.0)
        width_ratio: Width of tentacle as proportion of viewport width (default: 0.15)
        extend_duration: Time in seconds for tentacle to fully extend (default: 10.0)
        withdraw_duration: Time in seconds for tentacle to withdraw (default: 10.0)
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
    
    # Initialize on first call
    if state['count'] == 0:
        print(f"Initializing tentacle for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                Tentacle,
                depth=depth,
                intensity=intensity,
                speed=speed,
                width_ratio=width_ratio,
                extend_duration=extend_duration,
                withdraw_duration=withdraw_duration
            )
            state['effect'] = effect
            print(f"✓ Initialized shader tentacle for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize tentacle: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update animation state
    if 'effect' in state:
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        
        # Calculate extension factor (0 = fully retracted, 1 = fully extended)
        if elapsed_time < extend_duration:
            # Extending phase - ease out for smooth arrival
            t = elapsed_time / extend_duration
            extension = t * t * (3.0 - 2.0 * t)  # Smoothstep
        elif elapsed_time > (total_duration - withdraw_duration):
            # Withdrawing phase - ease in for smooth departure
            t = (total_duration - elapsed_time) / withdraw_duration
            extension = t * t * (3.0 - 2.0 * t)  # Smoothstep
        else:
            # Fully extended, wiggling phase
            extension = 1.0
        
        state['effect'].extension_factor = extension
        
        # Update from global state (optional)
        state['effect'].base_intensity = outstate.get('tentacle_intensity', intensity)
        state['effect'].base_speed = outstate.get('tentacle_speed', speed)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up tentacle for frame {frame_id}")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"✓ Cleaned up shader tentacle for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class Tentacle(ShaderEffect):
    """
    Organic tentacle effect descending from top of screen
    
    Creates an animated tentacle with organic movement patterns,
    extending and withdrawing based on timing.
    """
    
    def __init__(self, viewport, depth: float = 50.0, intensity: float = 1.0, 
                 speed: float = 1.0, width_ratio: float = 0.15,
                 extend_duration: float = 10.0, withdraw_duration: float = 10.0):
        super().__init__(viewport)
        self.depth = depth
        self.base_intensity = intensity
        self.base_speed = speed
        self.width_ratio = width_ratio
        self.width = int(viewport.width * width_ratio)
        self.extend_duration = extend_duration
        self.withdraw_duration = withdraw_duration
        self.time = 0.0
        self.extension_factor = 0.0  # 0 = retracted, 1 = extended
        
        # Randomly choose color scheme
        self.color_scheme = np.random.randint(0, 5)  # 5 different color schemes
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.EBO = None
        
        # Mesh resolution (higher = smoother organic movement)
        self.segments_x = 40   # Horizontal segments (width)
        self.segments_y = 80   # Vertical segments (length)
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize mesh data for tentacle"""
        # Create a vertical strip mesh for the tentacle
        # Position it at center-top of viewport, starting above screen
        viewport_width = self.viewport.width
        viewport_height = self.viewport.height
        
        # Center X position
        center_x = viewport_width / 2.0
        
        # Generate vertex positions for a vertical grid
        # Start from negative Y (above screen) to positive Y (at bottom of screen)
        vertices = []
        for y in range(self.segments_y + 1):
            # Normalize y position: 0 at top, 1 at bottom
            norm_y = y / self.segments_y
            # Start above screen (-0.3 * height) and extend down to 0.8 * height
            y_pos = -viewport_height * 0.3 + norm_y * viewport_height * 1.1
            
            for x in range(self.segments_x + 1):
                # X spreads from -width/2 to +width/2 relative to center
                x_offset = (x / self.segments_x - 0.5) * self.width
                x_pos = center_x + x_offset
                vertices.append([x_pos, y_pos])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Generate indices for triangle strip rendering
        indices = []
        for y in range(self.segments_y):
            for x in range(self.segments_x + 1):
                # Two vertices per column (current row and next row)
                indices.append(y * (self.segments_x + 1) + x)
                indices.append((y + 1) * (self.segments_x + 1) + x)
            # Degenerate triangles to connect strips
            if y < self.segments_y - 1:
                indices.append((y + 1) * (self.segments_x + 1) + self.segments_x)
                indices.append((y + 1) * (self.segments_x + 1))
        
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = len(self.indices)
    
    def compile_shader(self):
        """Compile and link tentacle shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            return shader
        except Exception as e:
            print(f"Tentacle shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        
        uniform vec2 resolution;
        uniform float depth;
        uniform float time;
        uniform float speed;
        uniform float extensionFactor;
        uniform float tentacleWidth;
        
        out vec2 vTexCoord;
        out float vWiggle;
        out float vEdgeDist;
        out float vNormY;
        out float vVisibility;
        out vec2 vWorldPos;
        
        void main() {
            vec2 pos = position;
            float centerX = resolution.x * 0.5;
            
            // Normalize Y relative to the full tentacle range
            // Map from [-0.3*height, 0.8*height] to [0, 1]
            float fullRange = resolution.y * 1.1;
            float yOffset = resolution.y * 0.3;
            float normY = (pos.y + yOffset) / fullRange;
            normY = clamp(normY, 0.0, 1.0);
            
            // Normalize X position (-0.5 to 0.5 across width)
            float normX = (pos.x - centerX) / tentacleWidth;
            
            // Calculate visibility based on extension factor
            // Smooth transition at the extension boundary
            float visibilityTransition = 0.02;  // Smooth edge width
            vVisibility = 1.0 - smoothstep(extensionFactor - visibilityTransition, 
                                          extensionFactor + visibilityTransition, 
                                          normY);
            
            // Create organic wiggling motion - multiple frequencies
            float wiggle1 = sin(normY * 6.28318 * 3.0 + time * speed * 1.2) * 0.08;
            float wiggle2 = sin(normY * 6.28318 * 5.0 - time * speed * 0.8) * 0.05;
            float wiggle3 = cos(normY * 6.28318 * 2.0 + time * speed * 0.6) * 0.06;
            
            // Wiggle amplitude increases toward the tip (higher normY)
            float wiggleAmount = (wiggle1 + wiggle2 + wiggle3) * normY * normY;
            wiggleAmount *= tentacleWidth * 3.0;
            
            // Add slight swaying motion
            float sway = sin(time * speed * 0.5) * normY * tentacleWidth * 0.3;
            
            pos.x += wiggleAmount + sway;
            
            // Add organic bulging along the length - more bulge in middle
            float bulge = sin(normY * 6.28318 * 4.0 + time * speed) * 0.1;
            float widthScale = 1.0 + bulge * (1.0 - abs(normX * 2.0));
            pos.x = centerX + (pos.x - centerX) * widthScale;
            
            // Taper toward the tip (gets thinner as normY increases)
            float taper = 1.0 - normY * 0.3;
            pos.x = centerX + (pos.x - centerX) * taper;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Standard depth mapping
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, mappedDepth, 1.0);
            
            // Pass data to fragment shader
            vTexCoord = vec2(normX + 0.5, normY);
            vWiggle = (wiggle1 + wiggle2) * 0.5 + 0.5;
            vEdgeDist = 1.0 - abs(normX * 2.0);
            vNormY = normY;
            vWorldPos = pos;  // Pass world position for texture coordinates
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vTexCoord;
        in float vWiggle;
        in float vEdgeDist;
        in float vNormY;
        in float vVisibility;
        in vec2 vWorldPos;
        
        uniform float time;
        uniform float intensity;
        uniform float speed;
        uniform vec2 resolution;
        uniform float tentacleWidth;
        uniform int colorScheme;
        
        out vec4 outColor;
        
        // Organic tentacle color palette - multiple schemes
        vec3 getTentacleColor(float y, float wiggle, float edgeDist, int scheme) {
            vec3 color1, color2, color3;
            
            if (scheme == 0) {
                // Purple-red scheme (original)
                color1 = vec3(0.4, 0.15, 0.3);   // Dark purple-red at base
                color2 = vec3(0.35, 0.2, 0.5);   // Purple middle
                color3 = vec3(0.5, 0.25, 0.6);   // Lighter purple at tip
            } else if (scheme == 1) {
                // Deep sea blue-green scheme
                color1 = vec3(0.15, 0.25, 0.35);  // Dark blue-green at base
                color2 = vec3(0.2, 0.35, 0.45);   // Teal middle
                color3 = vec3(0.25, 0.45, 0.5);   // Lighter cyan at tip
            } else if (scheme == 2) {
                // Blood red scheme
                color1 = vec3(0.35, 0.1, 0.15);   // Dark crimson at base
                color2 = vec3(0.45, 0.15, 0.2);   // Red middle
                color3 = vec3(0.55, 0.2, 0.25);   // Brighter red at tip
            } else if (scheme == 3) {
                // Toxic green scheme
                color1 = vec3(0.2, 0.3, 0.15);    // Dark olive at base
                color2 = vec3(0.25, 0.4, 0.2);    // Green middle
                color3 = vec3(0.3, 0.5, 0.25);    // Bright green at tip
            } else {
                // Abyssal black-purple scheme
                color1 = vec3(0.15, 0.1, 0.2);    // Near black at base
                color2 = vec3(0.25, 0.15, 0.3);   // Dark purple middle
                color3 = vec3(0.35, 0.2, 0.4);    // Purple at tip
            }
            
            // Blend based on position along tentacle
            vec3 baseColor;
            if (y < 0.5) {
                baseColor = mix(color1, color2, y * 2.0);
            } else {
                baseColor = mix(color2, color3, (y - 0.5) * 2.0);
            }
            
            // Add variation based on wiggle
            baseColor = mix(baseColor, color3, wiggle * 0.2);
            
            return baseColor;
        }
        
        void main() {
            float x = vTexCoord.x;
            float y = vNormY;
            
            // Use world position for texture coordinates that move with the tentacle
            vec2 worldUV = vWorldPos / vec2(tentacleWidth, resolution.y);
            
            // Create organic skin texture using world coordinates
            float skinPattern1 = sin(worldUV.y * 6.28318 * 20.0 + time * speed * 0.2) * 0.5 + 0.5;
            float skinPattern2 = sin(worldUV.x * 6.28318 * 8.0 + worldUV.y * 6.28318 * 12.0) * 0.5 + 0.5;
            float skinPattern = skinPattern1 * skinPattern2;
            skinPattern = pow(skinPattern, 1.3);
            
            // Create muscle fiber texture running lengthwise
            float fibers = 0.0;
            for (float i = -2.0; i <= 2.0; i += 1.0) {
                float fiberX = x + i * 0.15;
                float fiberPattern = sin(worldUV.y * 6.28318 * 15.0 + fiberX * 20.0) * 0.5 + 0.5;
                fiberPattern *= exp(-abs(i) * 0.3);  // Fade outer fibers
                fibers += fiberPattern;
            }
            fibers = fibers * 0.15;
            
            // Create prominent sucker pattern using world coordinates
            float suckerRows = 8.0;
            float suckersPerRow = 10.0;
            float suckers = 0.0;
            float suckerHighlight = 0.0;
            
            // Only show suckers on the inner/center portion (underside)
            if (vEdgeDist > 0.2) {
                for (float row = 0.0; row < suckerRows; row += 1.0) {
                    // Position suckers based on world Y coordinate (moves with tentacle)
                    float rowWorldY = row * 80.0;  // Space suckers every 80 pixels in world space
                    float yDist = abs(vWorldPos.y - rowWorldY) / 60.0;
                    
                    for (float col = 0.0; col < suckersPerRow; col += 1.0) {
                        float colX = (col + 0.5) / suckersPerRow;
                        float xDist = abs(x - colX);
                        
                        // Create circular sucker shape
                        float dist = sqrt(xDist * xDist * 4.0 + yDist * yDist);
                        float suckerShape = smoothstep(0.25, 0.15, dist);
                        
                        // Inner ring (darker depression)
                        float suckerDepth = smoothstep(0.18, 0.08, dist) * 0.5;
                        
                        // Outer rim highlight
                        float rimDist = abs(dist - 0.2);
                        float rim = smoothstep(0.08, 0.02, rimDist) * 0.3;
                        
                        suckers = max(suckers, suckerShape - suckerDepth);
                        suckerHighlight = max(suckerHighlight, rim);
                    }
                }
            }
            
            // Create horizontal muscle/segment rings that wrap around
            // These are based on world Y so they move with the tentacle
            float segmentSpacing = 40.0;  // World space pixels between segments
            float segmentPhase = mod(vWorldPos.y, segmentSpacing) / segmentSpacing;
            float segments = smoothstep(0.45, 0.5, segmentPhase) * smoothstep(0.55, 0.5, segmentPhase);
            segments *= 0.4;
            
            // Add raised ridges between segments
            float ridges = smoothstep(0.15, 0.2, segmentPhase) * smoothstep(0.35, 0.3, segmentPhase);
            ridges += smoothstep(0.65, 0.7, segmentPhase) * smoothstep(0.85, 0.8, segmentPhase);
            ridges *= 0.2;
            
            // Create internal vein/blood vessel structure
            float veins = 0.0;
            for (float i = 0.0; i < 4.0; i += 1.0) {
                float veinX = 0.15 + i * 0.25;
                float veinPath = veinX + sin(worldUV.y * 6.28318 * 0.5) * 0.1;
                float veinDist = abs(x - veinPath);
                float veinPattern = exp(-veinDist * 25.0);
                
                // Pulsing veins
                float pulse = sin(worldUV.y * 6.28318 * 2.0 - time * speed * 2.0 + i * 1.5) * 0.5 + 0.5;
                veins += veinPattern * pulse;
            }
            veins *= 0.12 * vEdgeDist;
            
            // Add subtle chromatophore-like spots (color cells)
            float chromSpots = 0.0;
            for (float i = 0.0; i < 12.0; i += 1.0) {
                vec2 spotCenter = vec2(
                    fract(sin(i * 12.34) * 43758.5) * 0.8 + 0.1,
                    i * 70.0
                );
                vec2 spotDist = vec2(
                    abs(x - spotCenter.x),
                    abs(vWorldPos.y - spotCenter.y) / 50.0
                );
                float dist = length(spotDist);
                float spotIntensity = smoothstep(0.4, 0.1, dist);
                float spotPulse = sin(time * speed * 0.8 + i * 2.0) * 0.5 + 0.5;
                chromSpots += spotIntensity * spotPulse * 0.15;
            }
            
            // Soft edge falloff for organic appearance
            float edgeFalloff = pow(vEdgeDist, 1.2);
            
            // Get base tentacle color based on color scheme
            vec3 color = getTentacleColor(y, vWiggle, vEdgeDist, colorScheme);
            
            // Apply muscle fibers (slight darkening in valleys)
            color *= (1.0 - fibers * 0.3);
            
            // Darken segment grooves
            color *= (1.0 - segments);
            
            // Lighten ridges
            color += ridges * vec3(0.15, 0.1, 0.15);
            
            // Darken suckers (they're indentations)
            color *= (1.0 - suckers * 0.5);
            
            // Add highlights to sucker rims
            color += suckerHighlight * vec3(0.2, 0.15, 0.2);
            
            // Apply skin texture
            color *= (0.85 + skinPattern * 0.15);
            
            // Add vein color (reddish-purple)
            color += veins * vec3(0.4, 0.1, 0.3);
            
            // Add chromatophore color variation (darker purples)
            color = mix(color, color * vec3(0.7, 0.6, 0.9), chromSpots);
            
            // Apply overall brightness
            float brightness = intensity * (0.8 + vWiggle * 0.2);
            
            // Calculate alpha with edge falloff
            float alpha = edgeFalloff * 1.0;
            alpha *= smoothstep(0.0, 0.03, y);  // Soft fade at top
            alpha *= smoothstep(1.0, 0.97, y);  // Soft fade at bottom
            
            // Make center more opaque (tentacle is solid)
            alpha = mix(alpha, 1.0, vEdgeDist * 0.6);
            
            // Apply visibility (hide parts beyond extension point)
            alpha *= vVisibility;
            
            alpha = clamp(alpha, 0.0, 1.0);
            
            outColor = vec4(color * brightness, alpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Generate VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Generate and bind position VBO
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Generate and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        if not self.enabled:
            return
        
        # Update animation time
        self.time += dt
    
    def render(self, state: Dict):
        """Render the tentacle effect"""
        if not self.enabled:
            return
        
        glUseProgram(self.shader)
        
        # Set uniforms
        resolution_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(resolution_loc, float(self.viewport.width), float(self.viewport.height))
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        intensity_loc = glGetUniformLocation(self.shader, "intensity")
        glUniform1f(intensity_loc, self.base_intensity)
        
        speed_loc = glGetUniformLocation(self.shader, "speed")
        glUniform1f(speed_loc, self.base_speed)
        
        extension_loc = glGetUniformLocation(self.shader, "extensionFactor")
        glUniform1f(extension_loc, self.extension_factor)
        
        width_loc = glGetUniformLocation(self.shader, "tentacleWidth")
        glUniform1f(width_loc, float(self.width))
        
        color_scheme_loc = glGetUniformLocation(self.shader, "colorScheme")
        glUniform1i(color_scheme_loc, self.color_scheme)
        
        # Render mesh
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLE_STRIP, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.position_VBO is not None:
            glDeleteBuffers(1, [self.position_VBO])
        if self.EBO is not None:
            glDeleteBuffers(1, [self.EBO])
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
