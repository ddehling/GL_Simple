import ctypes

import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
from typing import List, Tuple, Dict, Optional
import platform
from renderer.effects.base import ShaderEffect
from renderer.fan_geometry import FanGeometry
# Detect platform
IS_RASPBERRY_PI = platform.machine() in ['aarch64', 'armv7l', 'armv8']

class ShaderRenderer:
    """GPU-based renderer with visible OpenGL window (single viewport only)"""
    def __init__(self, frame_dimensions: List[Tuple[int, int]], padding=20, headless=False, magnification=1):
        self.frame_dimensions = frame_dimensions
        self.num_frames = len(frame_dimensions)
        self.headless = headless
        self.window = None
        self.viewports = []
        self.ctx_initialized = False
        
        # Initialize GLFW first to detect monitor size
        self.init_glfw()
        
        # Get monitor dimensions for autoscaling
        base_width, base_height = frame_dimensions[0]
        
        # Auto-calculate magnification if set to 0 or None
        if magnification is None or magnification == 0:
            monitor_height = self.get_monitor_height()
            # Calculate maximum magnification that fits monitor height
            # Leave some space for window decorations/taskbar (subtract ~100 pixels)
            available_height = monitor_height - 100 if not headless else monitor_height
            calculated_mag = max(1, int(available_height / base_height))
            self.magnification = calculated_mag
            print(f"[ShaderRenderer] Auto-calculated magnification: {self.magnification}x (monitor height: {monitor_height}px, available: {available_height}px)")
        else:
            self.magnification = max(1, int(magnification))  # Ensure integer >= 1
        
        # Use only the first frame dimension for window size, scaled by magnification
        self.window_width = base_width * self.magnification
        self.window_height = base_height * self.magnification
        
        # Double-check that window height doesn't exceed monitor
        if not headless:
            monitor_height = self.get_monitor_height()
            if self.window_height > monitor_height - 100:
                # Recalculate magnification to fit
                old_mag = self.magnification
                self.magnification = max(1, int((monitor_height - 100) / base_height))
                self.window_width = base_width * self.magnification
                self.window_height = base_height * self.magnification
                print(f"[ShaderRenderer] Adjusted magnification from {old_mag}x to {self.magnification}x to fit monitor")
        
        if self.magnification > 1:
            print(f"[ShaderRenderer] Window: {self.window_width}x{self.window_height} ({self.magnification}x magnification of {base_width}x{base_height})")
        else:
            print(f"[ShaderRenderer] Window: {self.window_width}x{self.window_height} (native size)")
        
        self.create_window()
        
        
    def init_glfw(self):
        """Initialize GLFW with OpenGL ES 3.1"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Make window non-resizable
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        
        if IS_RASPBERRY_PI:
            print("[ShaderRenderer] Configuring for Raspberry Pi (OpenGL ES 3.1 + EGL)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        else:
            print("[ShaderRenderer] Configuring for Desktop (OpenGL ES 3.1)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    
    def get_monitor_height(self):
        """Get the logical height of the primary monitor (in screen coordinates)"""
        try:
            monitor = glfw.get_primary_monitor()
            if monitor:
                video_mode = glfw.get_video_mode(monitor)
                if video_mode:
                    physical_height = video_mode.size[1]
                    # On HiDPI displays (Wayland/Linux), video mode may return physical
                    # pixels while windows are sized in logical coords. Divide by content
                    # scale to get the usable logical height.
                    _, yscale = glfw.get_monitor_content_scale(monitor)
                    logical_height = int(physical_height / yscale)
                    if yscale != 1.0:
                        print(f"[ShaderRenderer] Monitor content scale: {yscale}x (physical: {physical_height}px, logical: {logical_height}px)")
                    return logical_height
        except Exception as e:
            print(f"[ShaderRenderer] Warning: Could not detect monitor size: {e}")

        # Fallback to a reasonable default if detection fails
        return 1080
        
    def create_window(self):
        """Create a visible OpenGL window"""
        self.window = glfw.create_window(self.window_width, self.window_height,
                                        "LED Renderer", None, None)
        if not self.window:
            raise RuntimeError("Failed to create OpenGL window")

        glfw.make_context_current(self.window)

        # Get actual framebuffer size — differs from window size on HiDPI displays
        # (e.g. Wayland + GNOME with content scaling on Linux)
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)
        if self.fb_width != self.window_width or self.fb_height != self.window_height:
            print(f"[ShaderRenderer] HiDPI framebuffer: window={self.window_width}x{self.window_height}, "
                  f"framebuffer={self.fb_width}x{self.fb_height}")

        # Disable window resizing at runtime (redundant with hint, but ensures it)
        glfw.set_window_attrib(self.window, glfw.RESIZABLE, glfw.FALSE)

        # OpenGL setup for depth-based rendering
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)  # ENABLE depth writes for proper occlusion
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_SCISSOR_TEST)

        # Use actual framebuffer size for the GL viewport (not logical window size)
        glViewport(0, 0, self.fb_width, self.fb_height)
        
        version = glGetString(GL_VERSION)
        if version:
            print(f"[ShaderRenderer] OpenGL Version: {version.decode()}")
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
        if glsl_version:
            print(f"[ShaderRenderer] GLSL Version: {glsl_version.decode()}")

        # Store original window dimensions for toggling back from fan mode
        self._orig_window_width = self.window_width
        self._orig_window_height = self.window_height
        self._fan_window_width = 900
        self._fan_window_height = 500

        # Keyboard callback — F=flat/fan, D=continuous/dots, ESC=quit
        renderer_self = self
        def _key_callback(window, key, scancode, action, mods):
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_F:
                renderer_self._toggle_fan_mode()
            elif key == glfw.KEY_D:
                for vp in renderer_self.viewports:
                    vp.dot_mode = not vp.dot_mode
                renderer_self._update_window_title()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(self.window, _key_callback)

        print(f"[ShaderRenderer] Window created: {self.window_width}x{self.window_height}")
        print(f"[ShaderRenderer] Keys: F=flat/fan, D=continuous/dots, ESC=quit")
        self.ctx_initialized = True
        
    def create_viewport(self, frame_id: int) -> 'ShaderViewport':
        """Create a viewport for a specific frame"""
        if frame_id >= self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}")
            
        width, height = self.frame_dimensions[frame_id]
        
        # Only viewport 0 is displayed in the window
        if frame_id == 0:
            # Use framebuffer size for GL operations, not logical window size
            display_width = self.fb_width
            display_height = self.fb_height
            x_offset = 0
            y_offset = 0
        else:
            # Other viewports are offscreen only
            display_width = 0
            display_height = 0
            x_offset = 0
            y_offset = 0
        
        if not self.headless:
            print(f"[ShaderRenderer] Creating viewport {frame_id}:")
            print(f"[ShaderRenderer]   Framebuffer (LED): {width}x{height}")
            if frame_id == 0:
                print(f"[ShaderRenderer]   Display: {display_width}x{display_height} (full window)")
            else:
                print(f"[ShaderRenderer]   Display: offscreen only")
        else:
            print(f"[ShaderRenderer] Creating viewport {frame_id}: {width}x{height} (headless)")
        
        viewport = ShaderViewport(frame_id, width, height, 
                                 x_offset, y_offset, 
                                 display_width, display_height,
                                 self.window, headless=self.headless)
        viewport.init_framebuffer()
        self.viewports.append(viewport)
        return viewport


    
    def get_viewport(self, frame_id: int) -> Optional['ShaderViewport']:
        """Get viewport by frame_id"""
        for vp in self.viewports:
            if vp.frame_id == frame_id:
                return vp
        return None
    
    def _toggle_fan_mode(self):
        """Toggle fan view and resize the window accordingly."""
        entering_fan = not any(vp.fan_mode for vp in self.viewports)

        if entering_fan:
            new_w, new_h = self._fan_window_width, self._fan_window_height
        else:
            new_w, new_h = self._orig_window_width, self._orig_window_height

        glfw.set_window_size(self.window, new_w, new_h)
        self.window_width = new_w
        self.window_height = new_h
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)

        for vp in self.viewports:
            if vp.display_width > 0:
                vp.display_width = self.fb_width
                vp.display_height = self.fb_height
                # Force re-init of fan mesh/dots with new aspect ratio
                if vp._fan_initialized:
                    vp._fan_initialized = False
                if vp._dot_initialized:
                    vp._dot_initialized = False
            vp.toggle_fan_mode()
        self._update_window_title()

    def _update_window_title(self):
        vp = self.viewports[0] if self.viewports else None
        if not vp:
            return
        view = 'Fan' if vp.fan_mode else 'Flat'
        style = 'Dots' if vp.dot_mode else 'Continuous'
        title = f"LED Renderer — {view} {style} [F=view, D=dots]"
        glfw.set_window_title(self.window, title)

    def poll_events(self):
        """Poll GLFW events"""
        glfw.poll_events()
    
    def should_close(self):
        """Check if window should close"""
        return glfw.window_should_close(self.window)
    
    def swap_buffers(self):
        """Swap window buffers (skip in headless mode)"""
        if not self.headless:
            glfw.swap_buffers(self.window)
    
    def clear_window(self):
        """Clear the entire window"""
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.fb_width, self.fb_height)
        glScissor(0, 0, self.fb_width, self.fb_height)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
    
    def cleanup(self):
        """Clean up resources"""
        for vp in self.viewports:
            vp.cleanup()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        
    def sync_gpu(self):
        """Wait for all GPU operations to complete"""
        glfw.make_context_current(self.window)
        glFinish()


class ShaderViewport:
    """Individual viewport with shader effect pipeline and framebuffer for LED output."""

    def __init__(self, frame_id: int, width: int, height: int,
                 window_x: int, window_y: int,
                 display_width: int, display_height: int,
                 glfw_window, headless=False):
        self.frame_id = frame_id
        self.width = width        # Render at native LED resolution
        self.height = height
        self.led_width = width    # Aliases for fan/dot code
        self.led_height = height
        self.window_x = window_x
        self.window_y = window_y
        self.display_width = display_width
        self.display_height = display_height
        self.glfw_window = glfw_window
        self.headless = headless
        self.effects = []

        # Framebuffer for LED output
        self.fbo = None
        self.color_texture = None
        self.depth_texture = None

        # View modes
        self.fan_mode = False
        self.dot_mode = False
        self._fan_initialized = False
        self._fan_shader = None
        self._fan_vao = None
        self._fan_vbo = None
        self._fan_ebo = None
        self._fan_index_count = 0
        self._fan_tex_loc = None

        # Shared geometry helper (created on first use, rebuilt on aspect change)
        self._geometry = None

        # Instanced LED dot rendering
        self._dot_initialized = False
        self._dot_shader = None
        self._dot_vao = None
        self._dot_quad_vbo = None
        self._dot_inst_vbo = None
        self._dot_n_instances = 0
        self._dot_tex_loc = None
        self._dot_radius_loc = None
        

    def _ensure_geometry(self):
        """Create or rebuild FanGeometry for the current display aspect."""
        aspect = self.display_width / max(self.display_height, 1)
        if self._geometry is None or abs(self._geometry.aspect - aspect) > 0.01:
            self._geometry = FanGeometry(self.led_width, self.led_height, aspect)

    def init_framebuffer(self):
        """Create framebuffer for offscreen rendering (for LED output)"""
        glfw.make_context_current(self.glfw_window)
        
        # Create color texture
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create depth texture
        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 
                     self.width, self.height,
                     0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                              GL_TEXTURE_2D, self.color_texture, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_TEXTURE_2D, self.depth_texture, 0)
        
        # Check framebuffer completeness
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        print(f"[ShaderViewport] Framebuffer created: {self.width}x{self.height} (frame {self.frame_id})")

    
    def add_effect(self, effect_class, **params):
        """Add a shader effect to the rendering pipeline"""
        glfw.make_context_current(self.glfw_window)
        
        effect = effect_class(self, **params)
        effect.init()
        self.effects.append(effect)
        print(f"[ShaderViewport] Added effect: {effect.__class__.__name__} (frame {self.frame_id})")
        return effect
    
    def clear(self):
        """Clear the viewport in both window and framebuffer"""
        glfw.make_context_current(self.glfw_window)
        
        # Enable depth writes for clearing
        glDepthMask(GL_TRUE)
        
        # Clear framebuffer (including depth!)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Clear window viewport region (only for frame 0 and if not headless)
        if not self.headless and self.display_width > 0:
            glViewport(self.window_x, self.window_y, self.display_width, self.display_height)
            glScissor(self.window_x, self.window_y, self.display_width, self.display_height)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        

    
    def update(self, dt: float, state: Dict):
        """Update all effects (sorted by priority)"""
        sorted_effects = sorted(self.effects, key=lambda e: getattr(e, 'render_priority', 0))
        for effect in sorted_effects:
            if effect.enabled:
                effect.update(dt, state)
    
    def render(self, state: Dict):
        # 1. Render effects ONCE to framebuffer (LED resolution)
        # Sort by render_priority so post-processing effects render last
        sorted_effects = sorted(self.effects, key=lambda e: getattr(e, 'render_priority', 0))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)  # Set viewport to framebuffer size
        glScissor(0, 0, self.width, self.height)
        for effect in sorted_effects:
            if effect.enabled:
                effect.render(state)
        
        # 2. Copy/blit framebuffer to window (if visible)
        if not self.headless and self.display_width > 0:
            self._blit_framebuffer_to_window()

    def _blit_framebuffer_to_window(self):
        """Copy framebuffer contents to window with proper scaling."""
        if self.dot_mode:
            # Instanced dots — works in both flat and fan mode
            if not self._dot_initialized:
                self._init_dot_view()
            self._render_dots_to_window()
            return

        if self.fan_mode:
            # Fan continuous
            if not self._fan_initialized:
                self._init_fan_view()
            self._render_fan_to_window()
            return

        # Flat continuous — simple blit
        glDisable(GL_SCISSOR_TEST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glViewport(0, 0, self.display_width, self.display_height)
        glBlitFramebuffer(
            0, 0, self.width, self.height,
            0, 0, self.display_width, self.display_height,
            GL_COLOR_BUFFER_BIT, GL_NEAREST,
        )
        glEnable(GL_SCISSOR_TEST)

    # ------------------------------------------------------------------
    # Fan view preview
    # ------------------------------------------------------------------

    _FAN_VERT = """#version 310 es
precision highp float;
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

    _FAN_FRAG = """#version 310 es
precision highp float;
in vec2 vTexCoord;
uniform sampler2D uTexture;
out vec4 outColor;
void main() {
    outColor = texture(uTexture, vTexCoord);
}
"""

    def _init_fan_view(self):
        """Build the fan mesh and compile the shader (called once on first toggle)."""
        glfw.make_context_current(self.glfw_window)
        self._ensure_geometry()

        verts_2d, indices_np = self._geometry.compute_fan_mesh()
        verts_np = verts_2d.flatten()
        self._fan_index_count = len(indices_np)

        # Compile shader
        vert = shaders.compileShader(self._FAN_VERT, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self._FAN_FRAG, GL_FRAGMENT_SHADER)
        self._fan_shader = shaders.compileProgram(vert, frag)
        self._fan_tex_loc = glGetUniformLocation(self._fan_shader, "uTexture")

        # VAO / VBO / EBO
        self._fan_vao = glGenVertexArrays(1)
        glBindVertexArray(self._fan_vao)

        self._fan_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._fan_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts_np.nbytes, verts_np, GL_STATIC_DRAW)

        # position: location 0, 2 floats, stride 16, offset 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        # texcoord: location 1, 2 floats, stride 16, offset 8
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

        self._fan_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._fan_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_np.nbytes, indices_np, GL_STATIC_DRAW)

        glBindVertexArray(0)
        self._fan_initialized = True
        print(f"[ShaderViewport] Fan mesh: {self.led_width} strips x {self.led_height} LEDs, "
              f"{self._fan_index_count // 3} triangles")

    def _render_fan_to_window(self):
        """Draw the fan mesh textured with the FBO contents."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.display_width, self.display_height)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)

        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._fan_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glUniform1i(self._fan_tex_loc, 0)

        glBindVertexArray(self._fan_vao)
        glDrawElements(GL_TRIANGLES, self._fan_index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_SCISSOR_TEST)

    def toggle_fan_mode(self):
        self.fan_mode = not self.fan_mode
        print(f"[ShaderViewport] Fan view: {'ON' if self.fan_mode else 'OFF'}")

    # ------------------------------------------------------------------
    # Instanced LED dot rendering
    # ------------------------------------------------------------------

    _DOT_VERT = """#version 310 es
precision highp float;
layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec2 iPos;
layout(location = 2) in vec2 iTexCoord;
out vec2 vQuad;
flat out vec2 vUV;
uniform float uRadius;
void main() {
    vec2 world = iPos + aQuad * uRadius;
    gl_Position = vec4(world, 0.0, 1.0);
    vQuad = aQuad;
    vUV = iTexCoord;
}
"""

    _DOT_FRAG = """#version 310 es
precision highp float;
in vec2 vQuad;
flat in vec2 vUV;
uniform sampler2D uTexture;
out vec4 outColor;
void main() {
    float d = length(vQuad);
    if (d > 1.0) discard;
    float alpha = smoothstep(1.0, 0.7, d);
    vec4 color = texture(uTexture, vUV);
    outColor = vec4(color.rgb * alpha, alpha);
}
"""

    def _init_dot_view(self):
        """Build instanced LED dot geometry for the current view mode."""
        glfw.make_context_current(self.glfw_window)
        self._ensure_geometry()

        if self.fan_mode:
            inst_2d, self._dot_radius = self._geometry.compute_fan_dots()
        else:
            inst_2d, self._dot_radius = self._geometry.compute_flat_dots()

        inst_np = inst_2d.flatten()
        self._dot_n_instances = self.led_width * self.led_height

        # Compile shader (only once — reuse across re-inits)
        if self._dot_shader is None:
            vert = shaders.compileShader(self._DOT_VERT, GL_VERTEX_SHADER)
            frag = shaders.compileShader(self._DOT_FRAG, GL_FRAGMENT_SHADER)
            self._dot_shader = shaders.compileProgram(vert, frag)
            self._dot_tex_loc = glGetUniformLocation(self._dot_shader, "uTexture")
            self._dot_radius_loc = glGetUniformLocation(self._dot_shader, "uRadius")

        # Unit quad (6 vertices, 2 triangles)
        quad = np.array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1],
                        dtype=np.float32)

        # Clean up old buffers if re-initializing
        if self._dot_vao:
            glDeleteVertexArrays(1, [self._dot_vao])
        if self._dot_quad_vbo:
            glDeleteBuffers(1, [self._dot_quad_vbo])
        if self._dot_inst_vbo:
            glDeleteBuffers(1, [self._dot_inst_vbo])

        self._dot_vao = glGenVertexArrays(1)
        glBindVertexArray(self._dot_vao)

        self._dot_quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._dot_quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self._dot_inst_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._dot_inst_vbo)
        glBufferData(GL_ARRAY_BUFFER, inst_np.nbytes, inst_np, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)
        self._dot_initialized = True

        mode = 'fan' if self.fan_mode else 'flat'
        print(f"[ShaderViewport] LED dots ({mode}): {self._dot_n_instances} instances, "
              f"radius={self._dot_radius:.4f}")

    def _render_dots_to_window(self):
        """Draw instanced LED dots textured with FBO contents."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.display_width, self.display_height)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.02, 0.02, 0.02, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._dot_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glUniform1i(self._dot_tex_loc, 0)
        glUniform1f(self._dot_radius_loc, self._dot_radius)

        glBindVertexArray(self._dot_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self._dot_n_instances)
        glBindVertexArray(0)

        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_SCISSOR_TEST)

    def get_frame(self) -> np.ndarray:
        """Read framebuffer into numpy array for LED output"""
        glfw.make_context_current(self.glfw_window)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        pixels = glReadPixels(0, 0, self.width, self.height, 
                            GL_RGBA, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.height, self.width, 4)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Flip Y axis and drop alpha
        frame = np.flip(frame, axis=0)
        return frame[:, :, :3]
    
    def cleanup(self):
        """Clean up resources"""
        glfw.make_context_current(self.glfw_window)
        for effect in self.effects:
            effect.cleanup()
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.color_texture:
            glDeleteTextures([self.color_texture])
        if self.depth_texture:
            glDeleteTextures([self.depth_texture])
        # Fan view resources
        if self._fan_vao:
            glDeleteVertexArrays(1, [self._fan_vao])
        if self._fan_vbo:
            glDeleteBuffers(1, [self._fan_vbo])
        if self._fan_ebo:
            glDeleteBuffers(1, [self._fan_ebo])
        if self._fan_shader:
            glDeleteProgram(self._fan_shader)
        # Instanced dot resources
        if self._dot_vao:
            glDeleteVertexArrays(1, [self._dot_vao])
        if self._dot_quad_vbo:
            glDeleteBuffers(1, [self._dot_quad_vbo])
        if self._dot_inst_vbo:
            glDeleteBuffers(1, [self._dot_inst_vbo])
        if self._dot_shader:
            glDeleteProgram(self._dot_shader)