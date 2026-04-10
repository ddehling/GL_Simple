"""
Kelp shader effect - Swaying kelp forest in ocean waters
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_kelp(state, outstate, density=1.0, depth=60.0, sway_intensity=1.0):
    """
    Shader-based kelp effect compatible with EventScheduler
    
    Creates swaying kelp growing from the ocean floor, responding to
    water currents and ocean conditions.
    
    Usage:
        scheduler.schedule_event(0, 60, shader_kelp, density=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        density: Kelp density multiplier (0.0-2.0, default: 1.0)
        depth: Z-depth of kelp (0=near, 100=far, default: 60)
        sway_intensity: How much the kelp sways (default: 1.0)
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
        print(f"Initializing kelp for frame {frame_id}")
        
        try:
            viewport.add_effect(KelpEffect, density=density, depth=depth, sway_intensity=sway_intensity)
            # Get the effect that was just added
            state['effect'] = viewport.effects[-1]
            # Set initial fade to 0 so it starts invisible
            state['effect'].fade_factor = 0.0
            print(f"Kelp effect initialized successfully")
        except Exception as e:
            print(f"ERROR initializing kelp effect: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update from global state - read ocean weather parameters
    if 'effect' in state:
        # Use ocean weather set parameter names
        state['effect'].target_density = outstate.get('kelp_density', density)
        state['effect'].wave_speed = outstate.get('wave_speed', 0.5)
        state['effect'].current_strength = outstate.get('wind', 0.0)  # Use wind as current
        state['effect'].tide_level = outstate.get('tide_level', 0.5)
        state['effect'].season = outstate.get('season', 0.5)
        
        # Implement fade in/out - ALWAYS update this every frame
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)
        fade_duration = 2.0  # 2 second fade
        
        if elapsed_time < fade_duration:
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            fade_factor = 1.0
        
        state['effect'].fade_factor = float(np.clip(fade_factor, 0, 1))
    
    # Cleanup on close
    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"Kelp effect cleaned up for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class KelpEffect(ShaderEffect):
    """
    Kelp forest effect with organic swaying motion
    
    Creates individual kelp strands that grow from the ocean floor and
    sway naturally with water currents.
    """
    
    def __init__(self, viewport, density: float = 1.0, depth: float = 60.0,
                 sway_intensity: float = 1.0):
        super().__init__(viewport)
        self.fan = FanCoords(viewport.width, viewport.height)
        self.depth = depth
        self.base_density = density
        self.kelp_density = density
        self.sway_intensity = sway_intensity
        
        # Ocean parameters (updated from global state)
        self.wave_speed = 0.5
        self.current_strength = 0.0
        self.smoothed_current_strength = 0.0
        self.tide_level = 0.5
        self.season = 0.5
        self.fade_factor = 0.0  # Start invisible, fade in
        self.time = 0.0
        # Sway phase integrated CPU-side so sudden wave_speed changes
        # (weather state transitions) never snap the kelp to a new pose.
        self.sway_phase = 0.0
        self.smoothed_wave_speed = 0.5
        
        # Smoothed density for gradual transitions
        self.target_density = density
        self.smoothed_density = 0.0  # Start at 0 so kelp fades in
        
        # Buffer objects
        self.VAO = None
        self.position_VBO = None
        self.kelp_data_VBO = None
        self.EBO = None
        
        # Kelp strand parameters
        self.max_kelp_strands = 400  # Increased for more variety
        self.segments_per_strand = 20  # Segments in each kelp strand
        
        self._initialize_kelp_data()
    
    def _initialize_kelp_data(self):
        """Initialize kelp strand positions and properties"""
        width = self.viewport.width
        height = self.viewport.height
        
        # Generate kelp strand base positions (randomly distributed)
        np.random.seed(42)  # Consistent positions
        self.kelp_positions = np.column_stack([
            np.random.uniform(0, width, self.max_kelp_strands),  # x position
            np.zeros(self.max_kelp_strands),  # y position (ocean floor, will be at bottom)
        ]).astype(np.float32)
        
        # Kelp properties: [height, base_width, sway_phase, sway_frequency]
        self.kelp_properties = np.column_stack([
            np.random.uniform(height * 0.3, height * 0.8, self.max_kelp_strands),  # height
            np.random.uniform(2.0, 6.0, self.max_kelp_strands),  # base width
            np.random.uniform(0, 2 * np.pi, self.max_kelp_strands),  # sway phase offset
            np.random.uniform(0.8, 1.2, self.max_kelp_strands),  # sway frequency multiplier
        ]).astype(np.float32)
        
        # Create mesh for kelp strands (each strand is a ribbon)
        vertices = []
        kelp_data = []
        indices = []
        
        for strand_id in range(self.max_kelp_strands):
            base_vertex_id = len(vertices)
            
            for segment in range(self.segments_per_strand + 1):
                # Normalized position along strand (0 = base, 1 = tip)
                t = segment / self.segments_per_strand
                
                # Create two vertices (left and right edge of ribbon)
                for side in [-0.5, 0.5]:
                    # Position will be calculated in vertex shader
                    vertices.append([0, 0])  # Placeholder, calculated in shader
                    
                    # Kelp data: [strand_id, segment_t, side, unused]
                    kelp_data.append([strand_id, t, side, 0])
            
            # Generate triangle strip indices for this strand
            for segment in range(self.segments_per_strand):
                base_idx = base_vertex_id + segment * 2
                # Two triangles per segment
                indices.extend([
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx + 1, base_idx + 3, base_idx + 2
                ])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.kelp_data = np.array(kelp_data, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = len(self.indices)
    
    def compile_shader(self):
        """Compile and link kelp shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            return shaders.compileProgram(
                shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;

        layout(location = 0) in vec2 position;
        layout(location = 1) in vec4 kelpData;  // [strand_id, segment_t, side, unused]

        uniform vec2 resolution;
        uniform float depth;
        uniform float time;
        uniform int maxStrands;
        uniform sampler2D kelpPositions;  // Texture with kelp base positions
        uniform sampler2D kelpProperties; // Texture with kelp properties
        uniform float waveSpeed;   // smoothed sway rate (for frond waves etc.)
        uniform float swayPhase;   // CPU-integrated sway phase, jump-free
        uniform float currentStrength;
        uniform float swayIntensity;
        uniform float tideLevel;
        uniform float kelpDensity;
        """ + FAN_COORDS_UNIFORMS + FAN_COORDS_GLSL + """
        
        out vec2 vTexCoord;
        out float vSegmentT;
        out float vStrandId;
        out vec3 vColor;
        out float vStrandAlpha;  // Per-strand fade
        out float vKelpType;  // 0=ribbon, 1=broad, 2=thin
        
        // Hash function for noise - improved to avoid patterns
        float hash(vec2 p) {
            p = fract(p * vec2(443.897, 441.423));
            p += dot(p, p + vec2(19.19, 23.41));
            return fract(sin(p.x * p.y) * 43758.5453);
        }
        
        // Multi-hash for better variety
        float hash2(float n) {
            return fract(sin(n) * 43758.5453123);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        void main() {
            int strandId = int(kelpData.x);
            float segmentT = kelpData.y;
            float side = kelpData.z;
            
            // Calculate per-strand visibility based on density
            // Each strand has a threshold value (0-1) that determines when it appears
            float strandFloat = float(strandId);
            float strandThreshold = hash(vec2(strandFloat, 0.0));
            
            // Smooth fade in/out as density crosses this strand's threshold
            float fadeWidth = 0.1;  // How gradually strands fade (10% of density range)
            // Ensure strands are completely invisible at density=0
            float fadeStart = max(0.0, strandThreshold - fadeWidth);
            float fadeEnd = strandThreshold + fadeWidth;
            float strandAlpha = smoothstep(fadeStart, fadeEnd, kelpDensity);
            
            // Cull completely invisible strands for performance
            if (strandAlpha < 0.01 || kelpDensity < 0.001) {
                gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
                return;
            }
            
            // Fetch kelp base position and properties from uniforms
            // In actual implementation, we'll pass these as uniforms per-strand
            // For now, use simple indexing
            // (strandFloat already defined above)
            
            // These would come from textures in a more optimized version
            // For simplicity, calculate here
            vec2 basePos = vec2(
                hash2(strandFloat * 0.1) * resolution.x,  // Better distribution
                resolution.y  // Start at bottom of screen
            );
            
            // Determine kelp type (3 types: ribbon=0, broad=1, thin=2)
            int kelpType = int(mod(strandFloat, 3.0));
            
            // Wide kelp height range: tallest strands can reach ~75% of the
            // screen (well past the midpoint), shortest stay near the floor.
            // Multiplying two independent hashes biases the distribution
            // toward short strands so towering ones are special and rare.
            float maxKelpHeight = resolution.y * 0.75;
            float heightVariation = hash2(strandFloat * 0.7) * hash2(strandFloat * 1.3 + 7.7);
            float kelpHeight = maxKelpHeight * (0.08 + heightVariation * 0.92);
            
            // Width in PHYSICAL FEET (will be converted to pixels via local arc width).
            // Range is deliberately wide so some strands are reads-at-a-distance
            // thick while others remain thin wisps.
            float baseWidthFt;
            if (kelpType == 0) {
                baseWidthFt = 0.35 + hash2(strandFloat * 1.1) * 0.90;  // Ribbon kelp
            } else if (kelpType == 1) {
                baseWidthFt = 0.55 + hash2(strandFloat * 1.2) * 1.30;  // Broad kelp
            } else {
                baseWidthFt = 0.18 + hash2(strandFloat * 1.3) * 0.45;  // Thin kelp
            }

            float swayFreq = 0.6 + hash2(strandFloat * 2.7) * 0.8;

            // Apply tide level to base position (kelp floor moves with tide).
            // Tide is a radial offset of ~1 ft, converted to pixels via radial pixel height.
            float tideOffset = (tideLevel - 0.5) * 1.0 / fan_radial_height_ft();
            basePos.y += tideOffset;

            // Calculate position along kelp strand (grow upward = subtract from y)
            vec2 pos = basePos;
            pos.y -= segmentT * kelpHeight;

            // Local arc width in feet at this segment's buffer-v.
            // Used to convert physical-feet magnitudes (sway, blade width) into pixels
            // so the strand looks the same physical size everywhere on the fan.
            float localV = clamp(pos.y / resolution.y, 0.0, 1.0);
            float arcWFt = fan_arc_width_ft(vec2(0.0, localV));
            float ftToPx = 1.0 / arcWFt;
            
            // Sway motion (increases toward tip) - varies by kelp type
            float swayAmount = segmentT * segmentT;  // Quadratic - more sway at tip
            // swayPhase is integrated CPU-side so the kelp never snaps to a
            // new pose when waveSpeed changes between weather states.
            // per-strand phase offset keeps each strand out of sync.
            float strandPhaseOffset = hash2(strandFloat * 2.1) * 6.28318;
            float swayTime = swayPhase * swayFreq + strandPhaseOffset;
            
            // Multi-layer sway for organic motion - intensity varies by type
            float swayMultiplier = 1.0;
            if (kelpType == 0) {
                swayMultiplier = 1.2;  // Ribbon: more sway (has float/bulb)
            } else if (kelpType == 1) {
                swayMultiplier = 0.7;  // Broad: less sway (heavier fronds)
            } else {
                swayMultiplier = 1.5;  // Thin: most sway (lightest)
            }
            
            // Sway magnitudes in PHYSICAL FEET, converted to pixels via local arc width.
            // 0.6 ft / 0.32 ft / 0.2 ft give comparable visual amplitude to the old
            // pixel-space numbers (15/8/5 px) at the mid-radius row.
            float sway1 = sin(swayTime) * swayAmount * 1.8 * swayMultiplier * ftToPx;
            float sway2 = sin(swayTime * 1.7 + 1.3) * swayAmount * 1.0 * swayMultiplier * ftToPx;
            float sway3 = noise(vec2(swayTime * 0.5, strandFloat * 0.1)) * swayAmount * 0.6 * swayMultiplier * ftToPx;

            // Add horizontal undulation for frond effect (broad kelp)
            if (kelpType == 1) {
                float frondWave = sin(segmentT * 3.14159 * 3.0 + swayTime * 2.0);
                sway1 += frondWave * segmentT * 0.32 * ftToPx;
            }

            pos.x += (sway1 + sway2 + sway3) * swayIntensity;

            // Current drift: 0.4 ft of horizontal lean per unit of currentStrength.
            pos.x += currentStrength * segmentT * 0.4 * ftToPx;
            
            // Width varies along strand based on type - create bulbs and fronds.
            // widthProfile is dimensionless; base width is in feet, converted to px below.
            float widthProfile;
            if (kelpType == 0) {
                // Ribbon kelp: bulb near top (like bull kelp)
                float bulbPosition = 0.7 + hash2(strandFloat * 5.5) * 0.2;  // Bulb at 70-90%
                float bulbSize = 2.0 + hash2(strandFloat * 6.6) * 1.5;
                float distToBulb = abs(segmentT - bulbPosition);
                float bulge = exp(-distToBulb * distToBulb * 50.0) * bulbSize;
                widthProfile = (1.0 - segmentT * 0.4) + bulge;
            } else if (kelpType == 1) {
                // Broad kelp: multiple fronds along length
                float frondWave = sin(segmentT * 3.14159 * 3.0 + strandFloat) * 0.5 + 0.5;
                float frondPattern = pow(frondWave, 0.5);
                widthProfile = (1.0 + frondPattern * 1.5) * (1.0 - pow(segmentT, 2.0) * 0.6);
            } else {
                // Thin kelp: irregular width with occasional bulges
                float irregularity = noise(vec2(segmentT * 5.0, strandFloat * 0.3));
                widthProfile = (1.0 - segmentT * 0.85) * (0.7 + irregularity * 0.6);
            }

            // Convert physical-feet width to local pixels via arc width at this row.
            float widthPx = baseWidthFt * widthProfile * ftToPx;
            pos.x += side * widthPx;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Depth mapping
            float mappedDepth = depth / 100.0;
            mappedDepth = clamp(mappedDepth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, mappedDepth, 1.0);
            
            // Pass data to fragment shader
            vTexCoord = pos / resolution;
            vSegmentT = segmentT;
            vStrandId = strandFloat;
            vStrandAlpha = strandAlpha;  // Pass per-strand fade
            vKelpType = float(kelpType);
            
            // Color variation based on strand and type
            float colorVariation = hash2(strandFloat * 3.3);
            vec3 baseColor, tipColor;
            
            // Determine color palette based on strand ID (not just type)
            float colorChoice = hash2(strandFloat * 7.7);
            
            if (colorChoice < 0.33) {
                // Green kelp (33% of kelp)
                if (kelpType == 0) {
                    baseColor = vec3(0.15, 0.4 + colorVariation * 0.15, 0.2);
                    tipColor = vec3(0.3, 0.6 + colorVariation * 0.1, 0.35);
                } else if (kelpType == 1) {
                    baseColor = vec3(0.12, 0.35 + colorVariation * 0.1, 0.25);
                    tipColor = vec3(0.22, 0.5 + colorVariation * 0.1, 0.4);
                } else {
                    baseColor = vec3(0.2, 0.45 + colorVariation * 0.15, 0.15);
                    tipColor = vec3(0.35, 0.65 + colorVariation * 0.1, 0.3);
                }
            } else if (colorChoice < 0.66) {
                // Red/brown kelp (33% of kelp)
                if (kelpType == 0) {
                    baseColor = vec3(0.4 + colorVariation * 0.1, 0.15, 0.1);
                    tipColor = vec3(0.6 + colorVariation * 0.1, 0.25, 0.15);
                } else if (kelpType == 1) {
                    baseColor = vec3(0.35 + colorVariation * 0.1, 0.12, 0.08);
                    tipColor = vec3(0.55 + colorVariation * 0.1, 0.2, 0.12);
                } else {
                    baseColor = vec3(0.45 + colorVariation * 0.15, 0.2, 0.12);
                    tipColor = vec3(0.65 + colorVariation * 0.1, 0.3, 0.18);
                }
            } else {
                // Yellow/golden kelp (33% of kelp)
                if (kelpType == 0) {
                    baseColor = vec3(0.4 + colorVariation * 0.1, 0.35 + colorVariation * 0.1, 0.1);
                    tipColor = vec3(0.6 + colorVariation * 0.1, 0.55 + colorVariation * 0.1, 0.2);
                } else if (kelpType == 1) {
                    baseColor = vec3(0.35 + colorVariation * 0.1, 0.3 + colorVariation * 0.1, 0.08);
                    tipColor = vec3(0.55 + colorVariation * 0.1, 0.5 + colorVariation * 0.1, 0.15);
                } else {
                    baseColor = vec3(0.45 + colorVariation * 0.15, 0.4 + colorVariation * 0.15, 0.15);
                    tipColor = vec3(0.7 + colorVariation * 0.1, 0.6 + colorVariation * 0.1, 0.25);
                }
            }
            
            vColor = mix(baseColor, tipColor, segmentT);
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 vTexCoord;
        in float vSegmentT;
        in float vStrandId;
        in vec3 vColor;
        in float vStrandAlpha;  // Per-strand fade
        in float vKelpType;  // 0=ribbon, 1=broad, 2=thin
        
        uniform float time;
        uniform float fadeAlpha;
        uniform float season;
        
        out vec4 outColor;
        
        // Hash for noise
        float hash(vec2 p) {
            p = fract(p * vec2(123.45, 678.90));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        void main() {
            vec3 color = vColor;
            int kelpType = int(vKelpType);
            
            // Type-specific texture
            float textureNoise;
            if (kelpType == 0) {
                // Ribbon kelp: smooth vertical striations
                textureNoise = noise(vTexCoord * vec2(80.0, 30.0) + vec2(vStrandId * 0.1, time * 0.1));
            } else if (kelpType == 1) {
                // Broad kelp: wider, softer texture
                textureNoise = noise(vTexCoord * vec2(40.0, 20.0) + vec2(vStrandId * 0.15, time * 0.08));
            } else {
                // Thin kelp: fine texture
                textureNoise = noise(vTexCoord * vec2(120.0, 40.0) + vec2(vStrandId * 0.08, time * 0.12));
            }
            color *= 0.85 + textureNoise * 0.3;  // Less darkening
            
            // Subtle shine on kelp surface - varies by type
            float shineIntensity = kelpType == 1 ? 0.4 : 0.25;  // Broad kelp shinier
            float shine = noise(vec2(vTexCoord.x * 30.0, vSegmentT * 10.0 + time * 0.5));
            shine = pow(shine, 3.0) * shineIntensity;
            color += vec3(shine);
            
            // Gentle lighting gradient - less dramatic
            float lighting = 0.8 + vSegmentT * 0.2;
            color *= lighting;
            
            // Season affects color slightly
            float seasonalBrightness = 0.9 + sin(season * 6.28318) * 0.1;
            color *= seasonalBrightness;
            
            // Alpha - fully opaque when at full density/fade
            float alpha = 0.7 + vSegmentT * 0.3;  // 0.7 to 1.0 base range
            
            // Apply per-strand fade (for density changes)
            alpha *= vStrandAlpha;
            
            // Apply global fade factor
            alpha *= fadeAlpha;
            
            // Clamp to valid range
            alpha = clamp(alpha, 0.0, 1.0);
            
            outColor = vec4(color, alpha);
        }
        """
    
    def setup_buffers(self):
        """Initialize OpenGL buffers"""
        # Generate VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Position VBO
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # Kelp data VBO
        self.kelp_data_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.kelp_data_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.kelp_data.nbytes, self.kelp_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        # EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

        # Upload static fan-coordinate uniforms once.
        glUseProgram(self.shader)
        self.fan.set_uniforms(self.shader)
        glUseProgram(0)
    
    def update(self, dt: float, state: Dict):
        """Update effect state each frame"""
        self.time += dt

        # Smooth density transitions so kelp appears/disappears gradually.
        smoothing_speed = 0.5
        self.smoothed_density += (self.target_density - self.smoothed_density) * smoothing_speed * dt
        self.kelp_density = self.smoothed_density

        # Smooth the sway rate. The shader uses sway_phase DIRECTLY (not
        # time * waveSpeed), so a weather state transition to a different
        # wave_speed changes how fast the phase advances but never snaps
        # the kelp to a new pose.
        rate_smooth = 0.25
        self.smoothed_wave_speed += (self.wave_speed - self.smoothed_wave_speed) * rate_smooth * dt
        self.smoothed_current_strength += (self.current_strength - self.smoothed_current_strength) * rate_smooth * dt

        # Floor the effective sway rate so "calm" states don't go completely
        # motionless. The quietest ocean states have wave_speed 0.1-0.15,
        # which at the shader's sway frequency means one oscillation per
        # minute -- visually indistinguishable from frozen kelp. Clamp so
        # every state has readable motion.
        MIN_SWAY_RATE = 0.4
        effective_sway = max(self.smoothed_wave_speed, MIN_SWAY_RATE)
        self.sway_phase += effective_sway * dt
        # Wrap so float32 GPU uniform doesn't lose precision overnight.
        self.sway_phase %= 6283.2  # ~1000 × 2π, preserves all sin() behavior
    
    def render(self, state: Dict):
        """Render kelp strands"""
        if self.shader is None or self.VAO is None:
            return
        
        # For transparent objects, don't write to depth buffer
        # This allows proper blending with objects at any depth
        glDepthMask(GL_FALSE)
        
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniform2f(glGetUniformLocation(self.shader, b"resolution"),
                    self.viewport.width, self.viewport.height)
        glUniform1f(glGetUniformLocation(self.shader, b"depth"), self.depth)
        glUniform1f(glGetUniformLocation(self.shader, b"time"), self.time)
        glUniform1i(glGetUniformLocation(self.shader, b"maxStrands"), self.max_kelp_strands)
        glUniform1f(glGetUniformLocation(self.shader, b"waveSpeed"), self.smoothed_wave_speed)
        glUniform1f(glGetUniformLocation(self.shader, b"swayPhase"), self.sway_phase)
        glUniform1f(glGetUniformLocation(self.shader, b"currentStrength"), self.smoothed_current_strength)
        glUniform1f(glGetUniformLocation(self.shader, b"swayIntensity"), self.sway_intensity)
        glUniform1f(glGetUniformLocation(self.shader, b"tideLevel"), self.tide_level)
        glUniform1f(glGetUniformLocation(self.shader, b"kelpDensity"), self.kelp_density)
        glUniform1f(glGetUniformLocation(self.shader, b"fadeAlpha"), self.fade_factor)
        glUniform1f(glGetUniformLocation(self.shader, b"season"), self.season)
        
        # Render kelp
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glUseProgram(0)
        
        # Restore depth writes
        glDepthMask(GL_TRUE)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
        if self.position_VBO is not None:
            glDeleteBuffers(1, [self.position_VBO])
        if self.kelp_data_VBO is not None:
            glDeleteBuffers(1, [self.kelp_data_VBO])
        if self.EBO is not None:
            glDeleteBuffers(1, [self.EBO])
        if self.shader is not None:
            glDeleteProgram(self.shader)
