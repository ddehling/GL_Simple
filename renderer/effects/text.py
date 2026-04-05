"""
Text shader effect - GPU-accelerated text rendering with animations
Supports multiple display modes, color schemes, and letter animations
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List
from .base import ShaderEffect
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_text(state, outstate, text="Hello,World",
                display_mode="all_lines", color_mode="solid",
                animation_mode="none", fade_duration=2.0,
                font_size=48, color=(1.0, 1.0, 1.0),
                line_spacing=1.2, letter_spacing=1.0,
                sequential_delay=2.0, typewriter_speed=10.0,
                depth=25.0, x_pos=0.5, y_pos=0.5,
                auto_scale=True, scale_factor=0.9,
                text_rotation=0.0, height_scale=1.0):
    """
    Text rendering shader effect compatible with EventScheduler
    
    Display Modes:
        - 'all_lines': Show all lines at once (comma-separated)
        - 'sequential': Show lines one at a time with delays
        - 'typewriter': Character-by-character reveal
        - 'fade_lines': Lines fade in/out sequentially
    
    Color Modes:
        - 'solid': Single color (from color parameter)
        - 'rainbow': Horizontal rainbow gradient
        - 'rainbow_vertical': Vertical rainbow gradient
        - 'wave': Animated color wave
        - 'pulse': Pulsing brightness
        - 'fire': Fire-like colors (red/orange/yellow)
        - 'ocean': Ocean-like colors (blue/cyan/teal)
    
    Animation Modes:
        - 'none': Static text
        - 'wave': Vertical wave motion
        - 'scale_pulse': Letters pulse in size
        - 'rotate': Letters rotate
        - 'float': Letters float up/down
        - 'shake': Subtle shake effect
    
    Usage:
        scheduler.schedule_event(0, 60, shader_text, 
                               text="Line1,Line2,Line3",
                               display_mode="sequential",
                               color_mode="rainbow",
                               animation_mode="wave",
                               frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict
        text: Comma-delimited text string
        display_mode: How to display text ('all_lines', 'sequential', 'typewriter', 'fade_lines')
        color_mode: Color scheme ('solid', 'rainbow', 'wave', 'pulse', 'fire', 'ocean')
        animation_mode: Letter animation ('none', 'wave', 'scale_pulse', 'rotate', 'float', 'shake')
        fade_duration: Fade in/out duration in seconds
        font_size: Font size in pixels
        color: Base color as (r, g, b) tuple (0-1 range)
        line_spacing: Spacing between lines (1.0 = normal)
        letter_spacing: Spacing between letters (1.0 = normal)
        sequential_delay: Delay between lines in sequential mode (seconds)
        typewriter_speed: Characters per second in typewriter mode
        depth: Z-depth for 3D ordering (0=near, 100=far)
        x_pos: Horizontal position (0=left, 0.5=center, 1=right)
        y_pos: Vertical position (0=top, 0.5=center, 1=bottom)
        auto_scale: Automatically scale font to fit viewport (default True)
        scale_factor: Fraction of viewport to use for auto-scaling (0-1, default 0.9)
        text_rotation: Rotation angle in degrees (0=normal, 90=vertical, 180=upside-down, 270/-90=vertical-flipped)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing text effect for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                TextEffect,
                text=text,
                display_mode=display_mode,
                color_mode=color_mode,
                animation_mode=animation_mode,
                font_size=font_size,
                base_color=color,
                line_spacing=line_spacing,
                letter_spacing=letter_spacing,
                sequential_delay=sequential_delay,
                typewriter_speed=typewriter_speed,
                depth=depth,
                x_pos=x_pos,
                y_pos=y_pos,
                auto_scale=auto_scale,
                scale_factor=scale_factor,
                text_rotation=text_rotation,
                height_scale=height_scale
            )
            state['text_effect'] = effect
            print(f"✓ Initialized shader text for frame {frame_id}")
        except Exception as e:
            import traceback
            print(f"ERROR initializing text effect: {e}")
            traceback.print_exc()
            return
    
    # Update effect from global state
    if 'text_effect' in state:
        effect = state['text_effect']
        
        # Update text if changed in outstate
        new_text = outstate.get('text', text)
        if new_text != effect.text:
            effect.set_text(new_text)
        
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
        
        effect.fade_factor = np.clip(fade_factor, 0, 1)
        effect.elapsed_time = elapsed_time
    
    # Cleanup on close
    if state['count'] == -1:
        if 'text_effect' in state:
            effect = state['text_effect']
            if effect in viewport.effects:
                viewport.effects.remove(effect)
            effect.cleanup()
            del state['text_effect']
            print(f"✓ Cleaned up shader text for frame {frame_id}")


# ============================================================================
# Text Effect Class
# ============================================================================

class TextEffect(ShaderEffect):
    """GPU-based text rendering with multiple display and animation modes"""
    
    def __init__(self, viewport, text: str = "Hello,World",
                 display_mode: str = "all_lines",
                 color_mode: str = "solid",
                 animation_mode: str = "none",
                 font_size: int = 48,
                 base_color: tuple = (1.0, 1.0, 1.0),
                 line_spacing: float = 1.2,
                 letter_spacing: float = 1.0,
                 sequential_delay: float = 2.0,
                 typewriter_speed: float = 10.0,
                 depth: float = 25.0,
                 x_pos: float = 0.5,
                 y_pos: float = 0.5,
                 auto_scale: bool = True,
                 scale_factor: float = 0.9,
                 text_rotation: float = 0.0,
                 height_scale: float = 1.0):
        super().__init__(viewport)

        self.text = text
        self.display_mode = display_mode
        self.color_mode = color_mode
        self.animation_mode = animation_mode
        self.font_size = font_size
        self.base_color = np.array(base_color, dtype=np.float32)
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self.sequential_delay = sequential_delay
        self.typewriter_speed = typewriter_speed
        self.depth = depth
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.auto_scale = auto_scale
        self.scale_factor = scale_factor
        self.height_scale = height_scale
        self.text_rotation = text_rotation
        self.text_rotation_rad = np.radians(text_rotation)  # Convert to radians for shader
        
        self.fade_factor = 0.0
        self.elapsed_time = 0.0
        self.time = 0.0
        
        self.instance_VBO = None
        self.font_texture = None
        
        # Character data (populated by _generate_text)
        self.char_positions = np.array([], dtype=np.float32)  # [x, y] per char
        self.char_sizes = np.array([], dtype=np.float32)      # [w, h] per char
        self.char_uvs = np.array([], dtype=np.float32)        # [u0, v0, u1, v1] per char
        self.char_indices = np.array([], dtype=np.int32)      # Character index for animations
        self.line_indices = np.array([], dtype=np.int32)      # Line index per char
        
        # Font atlas
        self._generate_font_atlas()
        self._generate_text()
    
    def _generate_font_atlas(self):
        """Generate a texture atlas containing ASCII characters"""
        # Create a simple 16x6 grid of ASCII characters (32-127)
        chars_per_row = 16
        chars_per_col = 6  # 96 printable ASCII chars
        char_width = 64
        char_height = 64
        
        atlas_width = chars_per_row * char_width
        atlas_height = chars_per_col * char_height
        
        # Create image
        img = Image.new('RGBA', (atlas_width, atlas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to load a bold font, fallback to regular, then default
        try:
            font = ImageFont.truetype("arialbd.ttf", int(self.font_size * 0.8))
        except:
            try:
                font = ImageFont.truetype("arial.ttf", int(self.font_size * 0.8))
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(self.font_size * 0.8))
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", int(self.font_size * 0.8))
                    except:
                        font = ImageFont.load_default()
        
        # Draw characters with tight padding and per-glyph width
        self.char_map = {}    # char -> (u0, v0, u1, v1)
        self.char_widths = {} # char -> width ratio (0-1, fraction of cell used)
        pad = 1
        for i in range(96):  # ASCII 32-127
            char = chr(32 + i)
            row = i // chars_per_row
            col = i % chars_per_row

            x = col * char_width
            y = row * char_height

            # Measure actual glyph width
            try:
                bbox = font.getbbox(char)
                glyph_w = bbox[2] - bbox[0] + pad * 2
            except Exception:
                glyph_w = char_width // 2
            glyph_w = max(glyph_w, char_width // 4)  # minimum width
            glyph_w = min(glyph_w, char_width)        # cap at cell

            # Draw at left edge
            draw.text((x + pad, y + pad),
                     char, font=font, fill=(255, 255, 255, 255))

            # Tight UV: only map the portion of the cell the glyph uses
            u0 = col / chars_per_row
            v0 = row / chars_per_col
            u1 = (col + glyph_w / char_width) / chars_per_row
            v1 = (row + 1) / chars_per_col
            width_ratio = glyph_w / char_width

            self.char_map[char] = (u0, v0, u1, v1)
            self.char_widths[char] = width_ratio
        
        # Convert to OpenGL texture
        img_data = np.array(img, dtype=np.uint8)
        
        self.font_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.font_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlas_width, atlas_height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Store character dimensions
        self.char_width = self.font_size * self.letter_spacing
        self.char_height = self.font_size
    
    def _generate_text(self):
        """Generate character positions and UVs from text"""
        lines = self.text.split(',')
        
        # Auto-scale font size to fit viewport if enabled
        if self.auto_scale and len(lines) > 0:
            max_line_length = max(len(line) for line in lines)
            num_lines = len(lines)
            
            # Account for rotation when calculating available space
            # For 90/270 degree rotations, swap width and height constraints
            rotation_normalized = abs(self.text_rotation % 360)
            
            if 80 <= rotation_normalized <= 100 or 260 <= rotation_normalized <= 280:
                # Nearly 90 or 270 degrees - swap dimensions
                effective_width = self.viewport.height
                effective_height = self.viewport.width
            elif 170 <= rotation_normalized <= 190:
                # Nearly 180 degrees - keep dimensions
                effective_width = self.viewport.width
                effective_height = self.viewport.height
            else:
                # For other angles, calculate rotated bounding box
                angle_rad = np.radians(self.text_rotation)
                cos_a = abs(np.cos(angle_rad))
                sin_a = abs(np.sin(angle_rad))
                
                # Rotated bounding box dimensions (simplified)
                # Available space is reduced by rotation
                effective_width = self.viewport.width * cos_a + self.viewport.height * sin_a
                effective_height = self.viewport.width * sin_a + self.viewport.height * cos_a
                
                # Scale down to ensure fit
                effective_width *= 0.7
                effective_height *= 0.7
            
            # Calculate line widths using per-glyph widths
            def _line_width_ratio(line_text):
                """Sum of glyph width ratios for a line."""
                return sum(self.char_widths.get(c, 0.6) for c in line_text)

            max_line_ratio = max(_line_width_ratio(l) for l in lines)

            # Font size that fits width and height
            width_based_size = (effective_width * self.scale_factor) / max(max_line_ratio * self.letter_spacing, 0.1)
            height_based_size = (effective_height * self.scale_factor) / (num_lines * self.line_spacing)

            self.font_size = min(width_based_size, height_based_size)
            self.char_height = self.font_size * self.height_scale

        all_positions = []
        all_sizes = []
        all_uvs = []
        all_char_indices = []
        all_line_indices = []

        # Calculate total text block size for centering
        def _line_pixel_width(line_text):
            return sum(self.char_widths.get(c, 0.6) * self.font_size * self.letter_spacing for c in line_text)

        max_line_width = max(_line_pixel_width(l) for l in lines)
        total_height = len(lines) * self.char_height * self.line_spacing

        start_x = self.viewport.width * self.x_pos - max_line_width / 2
        start_y = self.viewport.height * self.y_pos - total_height / 2

        char_idx = 0
        for line_idx, line in enumerate(lines):
            line_width = _line_pixel_width(line)
            x = start_x + (max_line_width - line_width) / 2
            y = start_y + line_idx * self.char_height * self.line_spacing

            for char in line:
                if char in self.char_map:
                    cw = self.char_widths.get(char, 0.6) * self.font_size * self.letter_spacing

                    all_positions.append([x, y])
                    all_sizes.append([cw, self.char_height])
                    all_uvs.append(self.char_map[char])
                    all_char_indices.append(char_idx)
                    all_line_indices.append(line_idx)

                    x += cw
                    char_idx += 1
        
        if len(all_positions) > 0:
            self.char_positions = np.array(all_positions, dtype=np.float32)
            self.char_sizes = np.array(all_sizes, dtype=np.float32)
            self.char_uvs = np.array(all_uvs, dtype=np.float32)
            self.char_indices = np.array(all_char_indices, dtype=np.int32)
            self.line_indices = np.array(all_line_indices, dtype=np.int32)
            self.num_chars = len(all_positions)
            self.num_lines = len(lines)
        else:
            self.num_chars = 0
            self.num_lines = 0
    
    def set_text(self, text: str, rotation: float = None):
        """Update text and regenerate character data"""
        if text != self.text:
            self.text = text
            self._generate_text()
        if rotation is not None and rotation != self.text_rotation:
            self.text_rotation = rotation
            self.text_rotation_rad = np.radians(rotation)
    
    def _get_char_visibility(self) -> np.ndarray:
        """Calculate visibility for each character based on display mode"""
        if self.num_chars == 0:
            return np.array([])
        
        visibility = np.ones(self.num_chars, dtype=np.float32)
        
        if self.display_mode == "all_lines":
            # All characters visible
            pass
        
        elif self.display_mode == "sequential":
            # Show one line at a time
            current_line = int(self.elapsed_time / self.sequential_delay) % self.num_lines
            visibility = (self.line_indices == current_line).astype(np.float32)
        
        elif self.display_mode == "fade_lines":
            # Lines fade in/out sequentially
            current_line = int(self.elapsed_time / self.sequential_delay) % self.num_lines
            fade_progress = (self.elapsed_time % self.sequential_delay) / self.sequential_delay
            
            for line_idx in range(self.num_lines):
                if line_idx == current_line:
                    # Fade in
                    line_mask = self.line_indices == line_idx
                    visibility[line_mask] = min(1.0, fade_progress * 2.0)
                elif line_idx == (current_line - 1) % self.num_lines:
                    # Fade out
                    line_mask = self.line_indices == line_idx
                    visibility[line_mask] = max(0.0, 1.0 - fade_progress * 2.0)
                else:
                    line_mask = self.line_indices == line_idx
                    visibility[line_mask] = 0.0
        
        elif self.display_mode == "typewriter":
            # Reveal characters one by one
            chars_revealed = int(self.elapsed_time * self.typewriter_speed)
            visibility = (self.char_indices < chars_revealed).astype(np.float32)
        
        return visibility
    
    def _get_char_colors(self) -> np.ndarray:
        """Calculate colors for each character based on color mode"""
        if self.num_chars == 0:
            return np.array([])
        
        colors = np.tile(self.base_color, (self.num_chars, 1))
        
        if self.color_mode == "rainbow":
            # Horizontal rainbow gradient
            hue = self.char_indices / float(max(1, self.num_chars - 1))
            colors = self._hsv_to_rgb(hue, 1.0, 1.0)
        
        elif self.color_mode == "rainbow_vertical":
            # Vertical rainbow gradient by line
            hue = self.line_indices / float(max(1, self.num_lines - 1))
            colors = self._hsv_to_rgb(hue, 1.0, 1.0)
        
        elif self.color_mode == "wave":
            # Animated wave of colors
            phase = self.char_indices * 0.2 + self.time * 2.0
            hue = (np.sin(phase) * 0.5 + 0.5)
            colors = self._hsv_to_rgb(hue, 1.0, 1.0)
        
        elif self.color_mode == "pulse":
            # Pulsing brightness
            brightness = np.sin(self.time * 3.0) * 0.3 + 0.7
            colors = self.base_color * brightness
        
        elif self.color_mode == "fire":
            # Fire colors (red/orange/yellow)
            noise = np.sin(self.char_indices * 0.5 + self.time * 5.0) * 0.5 + 0.5
            hue = noise * 0.1  # 0.0-0.1 range (red to yellow)
            colors = self._hsv_to_rgb(hue, 1.0, 1.0)
        
        elif self.color_mode == "ocean":
            # Ocean colors (blue/cyan/teal)
            noise = np.sin(self.char_indices * 0.3 + self.time * 2.0) * 0.5 + 0.5
            hue = 0.5 + noise * 0.15  # 0.5-0.65 range (cyan to blue)
            colors = self._hsv_to_rgb(hue, 0.8, 1.0)
        
        return colors.astype(np.float32)
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB (vectorized)"""
        # h, s, v can be scalars or arrays
        h = np.atleast_1d(h)
        s = np.atleast_1d(s) if not isinstance(s, (int, float)) else np.full_like(h, s)
        v = np.atleast_1d(v) if not isinstance(v, (int, float)) else np.full_like(h, v)
        
        c = v * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = v - c
        
        rgb = np.zeros((len(h), 3))
        
        for i in range(len(h)):
            hi = int(h[i] * 6) % 6
            if hi == 0:
                rgb[i] = [c[i], x[i], 0]
            elif hi == 1:
                rgb[i] = [x[i], c[i], 0]
            elif hi == 2:
                rgb[i] = [0, c[i], x[i]]
            elif hi == 3:
                rgb[i] = [0, x[i], c[i]]
            elif hi == 4:
                rgb[i] = [x[i], 0, c[i]]
            else:
                rgb[i] = [c[i], 0, x[i]]
        
        return rgb + m[:, np.newaxis]
    
    def _get_char_transforms(self) -> tuple:
        """Calculate position/scale/rotation offsets based on animation mode"""
        if self.num_chars == 0:
            return np.zeros((0, 2)), np.ones(0), np.zeros(0)
        
        pos_offset = np.zeros((self.num_chars, 2), dtype=np.float32)
        scale = np.ones(self.num_chars, dtype=np.float32)
        rotation = np.zeros(self.num_chars, dtype=np.float32)
        
        if self.animation_mode == "wave":
            # Vertical wave motion
            phase = self.char_indices * 0.3 + self.time * 3.0
            pos_offset[:, 1] = np.sin(phase) * 10.0
        
        elif self.animation_mode == "scale_pulse":
            # Pulsing size
            phase = self.char_indices * 0.2 + self.time * 4.0
            scale = 1.0 + np.sin(phase) * 0.2
        
        elif self.animation_mode == "rotate":
            # Rotation
            rotation = np.sin(self.time * 2.0 + self.char_indices * 0.1) * 0.3
        
        elif self.animation_mode == "float":
            # Float up and down
            phase = self.char_indices * 0.15 + self.time * 1.5
            pos_offset[:, 1] = np.sin(phase) * 15.0
        
        elif self.animation_mode == "shake":
            # Subtle shake
            shake_x = np.sin(self.time * 20.0 + self.char_indices) * 2.0
            shake_y = np.cos(self.time * 25.0 + self.char_indices) * 2.0
            pos_offset[:, 0] = shake_x
            pos_offset[:, 1] = shake_y
        
        return pos_offset, scale, rotation
    
    def update(self, dt: float, state: Dict):
        """Update animation time"""
        if not self.enabled:
            return
        self.time += dt
    
    def render(self, state: Dict):
        """Render text characters"""
        if not self.enabled or self.num_chars == 0:
            return
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Bind font texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.font_texture)
        font_loc = glGetUniformLocation(self.shader, "fontTexture")
        glUniform1i(font_loc, 0)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        rotation_loc = glGetUniformLocation(self.shader, "textRotation")
        glUniform1f(rotation_loc, self.text_rotation_rad)
        
        # Calculate per-character properties
        visibility = self._get_char_visibility()
        colors = self._get_char_colors()
        pos_offset, scale, rotation = self._get_char_transforms()
        
        if len(visibility) == 0:
            glBindVertexArray(0)
            glUseProgram(0)
            return
        
        # Apply transforms to positions
        transformed_positions = self.char_positions + pos_offset
        transformed_sizes = self.char_sizes * scale[:, np.newaxis]
        
        # Build instance data
        # Layout: [x, y, w, h, u0, v0, u1, v1, r, g, b, visibility, rotation]
        instance_data = np.column_stack([
            transformed_positions,
            transformed_sizes,
            self.char_uvs,
            colors,
            visibility,
            rotation
        ]).astype(np.float32)
        
        # Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Draw instanced quads
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_chars)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;      // Quad vertices (0 to 1)
        layout(location = 1) in vec2 charPos;       // Character position
        layout(location = 2) in vec2 charSize;      // Character size
        layout(location = 3) in vec4 charUV;        // UV coords (u0, v0, u1, v1)
        layout(location = 4) in vec3 charColor;     // Color
        layout(location = 5) in float charVisible;  // Visibility (0-1)
        layout(location = 6) in float charRotation; // Rotation in radians
        
        out vec2 fragUV;
        out vec3 fragColor;
        out float fragAlpha;
        
        uniform vec2 resolution;
        uniform float fadeAlpha;
        uniform float depth;
        uniform float textRotation;  // Global text rotation in radians
        
        void main() {
            // Apply per-character rotation to quad vertices
            float c = cos(charRotation);
            float s = sin(charRotation);
            vec2 rotated = vec2(
                position.x * c - position.y * s,
                position.x * s + position.y * c
            );
            
            // Scale and translate to character position
            vec2 scaled = rotated * charSize;
            vec2 pos = scaled + charPos;
            
            // Apply global text rotation around viewport center
            vec2 center = resolution * 0.5;
            vec2 posRelativeToCenter = pos - center;
            
            float textCos = cos(textRotation);
            float textSin = sin(textRotation);
            vec2 rotatedPos = vec2(
                posRelativeToCenter.x * textCos - posRelativeToCenter.y * textSin,
                posRelativeToCenter.x * textSin + posRelativeToCenter.y * textCos
            );
            
            pos = rotatedPos + center;
            
            // Convert to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Fixed depth for text (from uniform, default 25/100 = 0.25)
            float depthValue = 25.0 / 100.0;
            depthValue = clamp(depthValue, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depthValue, 1.0);
            
            // Interpolate UV coordinates
            fragUV = mix(charUV.xy, charUV.zw, position);
            
            // Pass color and visibility
            fragColor = charColor;
            fragAlpha = charVisible * fadeAlpha;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragUV;
        in vec3 fragColor;
        in float fragAlpha;
        
        out vec4 outColor;
        
        uniform sampler2D fontTexture;
        
        void main() {
            // Sample font texture
            vec4 texColor = texture(fontTexture, fragUV);
            
            // Use texture alpha as mask
            float alpha = texColor.a * fragAlpha;
            
            if (alpha < 0.01) {
                discard;
            }
            
            // Apply color
            outColor = vec4(fragColor, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link text shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vertex = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            fragment = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vertex, fragment)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced text rendering"""
        # Quad vertices (0 to 1 for easy UV interpolation)
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
        
        # Vertex buffer (quad)
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
        
        # Instance buffer
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        self.VBOs.append(self.instance_VBO)
        
        # Instance attributes: charPos(2), charSize(2), charUV(4), charColor(3), charVisible(1), charRotation(1)
        # Total: 13 floats per instance
        stride = 4 * 13
        
        # charPos (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # charSize (location 2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # charUV (location 3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # charColor (location 4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # charVisible (location 5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(44))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
        # charRotation (location 6)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(48))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)
        
        glBindVertexArray(0)
    
    def cleanup(self):
        """Clean up font texture and buffers"""
        if self.font_texture is not None:
            glDeleteTextures([self.font_texture])
            self.font_texture = None
        super().cleanup()
