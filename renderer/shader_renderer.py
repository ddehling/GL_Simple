"""OpenGL renderer for the LED semicircle display.

Rendering pipeline (per frame):
  1. Effects write to a 128x300 FBO at native LED resolution
  2. get_frame() reads the FBO for DMX output (sACN to physical LEDs)
  3. The FBO is also displayed in the desktop window in one of four
     view modes (toggled with F and D keys):

       Flat Smooth  -- magnified pixel blit (default)
       Flat LED     -- instanced circles, one per physical LED
       Fan Smooth   -- semicircular mesh simulating the physical fan layout
       Fan LED      -- instanced circles arranged in the fan semicircle

The same FBO data is streamed as PNG to the web preview at /preview.

Classes:
  ShaderRenderer  -- owns the GLFW window and manages viewports
  ShaderViewport  -- FBO + effect pipeline + display mode delegation
"""

import glfw
from OpenGL.GL import *
import numpy as np
from typing import List, Tuple, Dict, Optional
from renderer.fan_geometry import FanGeometry
from renderer.display_modes import FlatSmooth, FanSmooth, LEDDots


class ShaderRenderer:
    """Owns the GLFW window, creates viewports, and handles keyboard input."""

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
            available_height = monitor_height - 100 if not headless else monitor_height
            calculated_mag = max(1, int(available_height / base_height))
            self.magnification = calculated_mag
            print(f"[ShaderRenderer] Auto-calculated magnification: {self.magnification}x "
                  f"(monitor height: {monitor_height}px, available: {available_height}px)")
        else:
            self.magnification = max(1, int(magnification))

        # Use only the first frame dimension for window size, scaled by magnification
        self.window_width = base_width * self.magnification
        self.window_height = base_height * self.magnification

        # Double-check that window height doesn't exceed monitor
        if not headless:
            monitor_height = self.get_monitor_height()
            if self.window_height > monitor_height - 100:
                old_mag = self.magnification
                self.magnification = max(1, int((monitor_height - 100) / base_height))
                self.window_width = base_width * self.magnification
                self.window_height = base_height * self.magnification
                print(f"[ShaderRenderer] Adjusted magnification from {old_mag}x to {self.magnification}x to fit monitor")

        if self.magnification > 1:
            print(f"[ShaderRenderer] Window: {self.window_width}x{self.window_height} "
                  f"({self.magnification}x magnification of {base_width}x{base_height})")
        else:
            print(f"[ShaderRenderer] Window: {self.window_width}x{self.window_height} (native size)")

        self.create_window()

    def init_glfw(self):
        """Initialize GLFW with OpenGL ES 3.1"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        print("[ShaderRenderer] Configuring for OpenGL ES 3.1")

    def get_monitor_height(self):
        """Get the logical height of the primary monitor (in screen coordinates)"""
        try:
            monitor = glfw.get_primary_monitor()
            if monitor:
                video_mode = glfw.get_video_mode(monitor)
                if video_mode:
                    physical_height = video_mode.size[1]
                    _, yscale = glfw.get_monitor_content_scale(monitor)
                    logical_height = int(physical_height / yscale)
                    if yscale != 1.0:
                        print(f"[ShaderRenderer] Monitor content scale: {yscale}x "
                              f"(physical: {physical_height}px, logical: {logical_height}px)")
                    return logical_height
        except Exception as e:
            print(f"[ShaderRenderer] Warning: Could not detect monitor size: {e}")
        return 1080

    def create_window(self):
        """Create a visible OpenGL window"""
        self.window = glfw.create_window(self.window_width, self.window_height,
                                         "LED Renderer", None, None)
        if not self.window:
            raise RuntimeError("Failed to create OpenGL window")

        glfw.make_context_current(self.window)

        # Get actual framebuffer size -- differs from window size on HiDPI displays
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)
        if self.fb_width != self.window_width or self.fb_height != self.window_height:
            print(f"[ShaderRenderer] HiDPI framebuffer: window={self.window_width}x{self.window_height}, "
                  f"framebuffer={self.fb_width}x{self.fb_height}")

        glfw.set_window_attrib(self.window, glfw.RESIZABLE, glfw.FALSE)

        # OpenGL setup for depth-based rendering
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_SCISSOR_TEST)
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
        # Fan window sized to fill most of the monitor
        monitor_h = self.get_monitor_height()
        self._fan_window_height = monitor_h - 100
        self._fan_window_width = int(self._fan_window_height * 1.8)

        # Keyboard callback -- F=flat/fan, D=smooth/LED, ESC=quit
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

        # --- Mouse state for pan/zoom ---
        self._dragging = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Scroll callback -- zoom centered on cursor
        def _scroll_callback(window, xoffset, yoffset):
            for vp in renderer_self.viewports:
                if not vp.fan_mode:
                    continue
                # Get cursor in clip space [-1, 1]
                mx, my = glfw.get_cursor_pos(window)
                w = renderer_self.window_width or 1
                h = renderer_self.window_height or 1
                clip_x = (mx / w) * 2.0 - 1.0
                clip_y = 1.0 - (my / h) * 2.0  # flip Y

                old_zoom = vp.zoom
                new_zoom = max(0.5, min(10.0, old_zoom * (1.15 ** yoffset)))
                # Adjust pan so the point under cursor stays fixed
                # clip_pos = world_pos * zoom + pan
                # world_pos = (clip_pos - pan) / zoom  (same before and after)
                # (clip_x - pan_x) / old_zoom == (clip_x - new_pan_x) / new_zoom
                # new_pan_x = clip_x - (clip_x - pan_x) * new_zoom / old_zoom
                vp.pan_x = clip_x - (clip_x - vp.pan_x) * new_zoom / old_zoom
                vp.pan_y = clip_y - (clip_y - vp.pan_y) * new_zoom / old_zoom
                vp.zoom = new_zoom

        glfw.set_scroll_callback(self.window, _scroll_callback)

        # Mouse button callback -- start/stop drag for panning
        def _mouse_button_callback(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    renderer_self._dragging = True
                    renderer_self._last_mouse_x, renderer_self._last_mouse_y = glfw.get_cursor_pos(window)
                elif action == glfw.RELEASE:
                    renderer_self._dragging = False
            elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
                # Middle click resets zoom and pan
                for vp in renderer_self.viewports:
                    vp.zoom = 1.0
                    vp.pan_x = 0.0
                    vp.pan_y = 0.0

        glfw.set_mouse_button_callback(self.window, _mouse_button_callback)

        # Cursor position callback -- drag to pan
        def _cursor_pos_callback(window, xpos, ypos):
            if not renderer_self._dragging:
                return
            dx = xpos - renderer_self._last_mouse_x
            dy = ypos - renderer_self._last_mouse_y
            renderer_self._last_mouse_x = xpos
            renderer_self._last_mouse_y = ypos

            w = renderer_self.window_width or 1
            h = renderer_self.window_height or 1
            # Convert pixel delta to clip space delta
            clip_dx = (dx / w) * 2.0
            clip_dy = -(dy / h) * 2.0  # flip Y

            for vp in renderer_self.viewports:
                if vp.fan_mode:
                    vp.pan_x += clip_dx
                    vp.pan_y += clip_dy

        glfw.set_cursor_pos_callback(self.window, _cursor_pos_callback)

        print(f"[ShaderRenderer] Window created: {self.window_width}x{self.window_height}")
        print(f"[ShaderRenderer] Keys: F=flat/fan, D=smooth/LED, ESC=quit, Scroll=zoom, Drag=pan, MClick=reset")
        self.ctx_initialized = True

    def create_viewport(self, frame_id: int) -> 'ShaderViewport':
        """Create a viewport for a specific frame"""
        if frame_id >= self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}")

        width, height = self.frame_dimensions[frame_id]

        # Only viewport 0 is displayed in the window
        if frame_id == 0:
            display_width = self.fb_width
            display_height = self.fb_height
            x_offset = 0
            y_offset = 0
        else:
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
                vp.mark_display_modes_dirty()
            vp.toggle_fan_mode()
        self._update_window_title()

    def _update_window_title(self):
        vp = self.viewports[0] if self.viewports else None
        if not vp:
            return
        view = 'Fan' if vp.fan_mode else 'Flat'
        style = 'LED' if vp.dot_mode else 'Smooth'
        title = f"LED Renderer \u2014 {view} {style} [F=view, D=style]"
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
    """One LED panel's render target and display pipeline.

    Owns a 128x300 FBO where shader effects draw each frame.  The FBO
    is read by get_frame() for DMX output, and displayed in the GLFW
    window via one of four view modes (flat/fan x smooth/LED).

    Display mode rendering is delegated to DisplayMode subclasses
    in renderer/display_modes/.
    """

    def __init__(self, frame_id: int, width: int, height: int,
                 window_x: int, window_y: int,
                 display_width: int, display_height: int,
                 glfw_window, headless=False):
        self.frame_id = frame_id
        self.width = width            # FBO width  (= number of LED strips, e.g. 128)
        self.height = height          # FBO height (= LEDs per strip, e.g. 300)
        self.led_width = width        # Alias used by FanGeometry
        self.led_height = height
        self.window_x = window_x      # Position in the GLFW window (always 0)
        self.window_y = window_y
        self.display_width = display_width    # Window pixel size (may differ from FBO)
        self.display_height = display_height
        self.glfw_window = glfw_window
        self.headless = headless
        self.effects = []

        # --- FBO (render target at native LED resolution) ---
        self.fbo = None
        self.color_texture = None
        self.depth_texture = None

        # --- Display mode state ---
        self.fan_mode = False    # False = flat, True = fan (semicircle)
        self.dot_mode = False    # False = smooth, True = LED (instanced circles)
        self.zoom = 1.0          # View zoom level (scroll wheel)
        self.pan_x = 0.0         # Pan offset in clip space
        self.pan_y = 0.0

        # --- Shared geometry (FanGeometry, rebuilt when aspect changes) ---
        self._geometry = None

        # --- Display mode instances ---
        self._display_modes = {
            ('flat', 'smooth'): FlatSmooth(),
            ('fan', 'smooth'):  FanSmooth(),
            ('flat', 'led'):    LEDDots(fan=False),
            ('fan', 'led'):     LEDDots(fan=True),
        }

    @property
    def _current_mode_key(self):
        view = 'fan' if self.fan_mode else 'flat'
        style = 'led' if self.dot_mode else 'smooth'
        return (view, style)

    def _ensure_geometry(self):
        """Create or rebuild FanGeometry for the current display aspect."""
        aspect = self.display_width / max(self.display_height, 1)
        if self._geometry is None or abs(self._geometry.aspect - aspect) > 0.01:
            self._geometry = FanGeometry(self.led_width, self.led_height, aspect)

    def mark_display_modes_dirty(self):
        """Force all display modes to re-initialize (e.g. after aspect change)."""
        for mode in self._display_modes.values():
            mode.mark_dirty()

    def toggle_fan_mode(self):
        self.fan_mode = not self.fan_mode
        print(f"[ShaderViewport] Fan view: {'ON' if self.fan_mode else 'OFF'}")

    # ------------------------------------------------------------------
    # Framebuffer setup
    # ------------------------------------------------------------------

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

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        print(f"[ShaderViewport] Framebuffer created: {self.width}x{self.height} (frame {self.frame_id})")

    # ------------------------------------------------------------------
    # Effect pipeline
    # ------------------------------------------------------------------

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
        glDepthMask(GL_TRUE)

        # Clear FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Clear window region
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
        """Render effects to FBO, then display in window."""
        # 1. Render effects to FBO at native LED resolution
        sorted_effects = sorted(self.effects, key=lambda e: getattr(e, 'render_priority', 0))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        for effect in sorted_effects:
            if effect.enabled:
                effect.render(state)

        # 2. Display FBO in window via the active display mode
        if not self.headless and self.display_width > 0:
            mode = self._display_modes[self._current_mode_key]
            if not mode._initialized:
                glfw.make_context_current(self.glfw_window)
                mode.init(self)
            mode.render(self)

    # ------------------------------------------------------------------
    # Frame readback for DMX output
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

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
        for mode in self._display_modes.values():
            mode.cleanup()
