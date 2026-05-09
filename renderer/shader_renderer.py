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
  ShaderViewport  -- window-display wrapper that holds a GroupCanvas
                     (the actual FBO + effect pipeline lives in GroupCanvas)
"""

import os
import OpenGL
import glfw
from OpenGL.GL import *
from typing import List, Tuple, Dict, Optional
from renderer.fan_geometry import FanGeometry
from renderer.display_modes import FlatSmooth, FanSmooth, LEDDots
from renderer.group_canvas import GroupCanvas

# EGL for headless GPU rendering (no display server required)
_egl_available = False
try:
    from OpenGL import EGL as _EGL
    _egl_available = True
except (ImportError, AttributeError, OSError):
    _EGL = None

# Disable PyOpenGL's per-call error checking AFTER EGL import (setting this
# via env var breaks EGL on some PyOpenGL versions). ~2-10x perf improvement.
OpenGL.ERROR_CHECKING = False


class ShaderRenderer:
    """Owns the GLFW window, creates viewports, and handles keyboard input.

    ``frame_dimensions`` and ``group_ids`` are index-aligned: ``frame_id=i``
    selects the canvas for ``group_ids[i]`` with size
    ``frame_dimensions[i]``. Phase-6 multi-canvas projects pass N groups;
    Fan-style single-canvas projects pass N=1 with id ``"main"``.
    """

    def __init__(self, frame_dimensions: List[Tuple[int, int]], padding=20, headless=False, magnification=1, group_ids: Optional[List[str]] = None):
        self.frame_dimensions = frame_dimensions
        self.num_frames = len(frame_dimensions)
        if group_ids is None:
            group_ids = [f"group{i}" if i > 0 else "main" for i in range(self.num_frames)]
        if len(group_ids) != self.num_frames:
            raise ValueError(
                f"group_ids length ({len(group_ids)}) must match "
                f"frame_dimensions length ({self.num_frames})"
            )
        self.group_ids = list(group_ids)
        # Lookup used by the event dispatcher to translate `{"group": "..."}`
        # metadata into the integer frame_id the existing effects API uses.
        self.group_to_frame_id = {gid: i for i, gid in enumerate(self.group_ids)}
        self.headless = headless
        self.window = None
        self.viewports = []
        self.ctx_initialized = False
        self._use_egl = False
        self._egl_display = None
        self._egl_surface = None
        self._egl_context = None

        # Initialize GL context — try GLFW first, fall back to EGL for headless
        if headless:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=glfw.GLFWError)
                try:
                    self.init_glfw()
                except RuntimeError:
                    self._init_egl_headless()
        else:
            self.init_glfw()

        # Get monitor dimensions for autoscaling
        base_width, base_height = frame_dimensions[0]

        # Auto-calculate magnification if set to 0 or None
        if magnification is None or magnification == 0:
            if self._use_egl:
                self.magnification = 1
            else:
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
        if not headless and not self._use_egl:
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

        if self._use_egl:
            self._setup_headless_gl()
        else:
            self.create_window()

    def init_glfw(self):
        """Initialize GLFW with OpenGL ES 3.1"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        if self.headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        print("[ShaderRenderer] Configuring for OpenGL ES 3.1")

    def _init_egl_headless(self):
        """Create a headless GPU context via EGL (no display server needed)."""
        if not _egl_available:
            raise RuntimeError(
                "Failed to initialize GLFW (no display server) and EGL is not "
                "available. Install libEGL (e.g. apt install libegl1-mesa) or "
                "run with a display server (e.g. xvfb-run)."
            )

        EGL = _EGL
        # Use surfaceless platform — works on Mesa without any display server
        os.environ.setdefault('EGL_PLATFORM', 'surfaceless')

        display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if display == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("EGL: no display available")

        major, minor = ctypes.c_long(), ctypes.c_long()
        if not EGL.eglInitialize(display, ctypes.pointer(major), ctypes.pointer(minor)):
            raise RuntimeError("EGL: initialization failed")

        # ES3 renderable bit (0x0040) — consistent across EGL versions
        EGL_OPENGL_ES3_BIT = 0x0040
        config_attribs = (EGL.EGLint * 13)(
            EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RED_SIZE, 8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE, 8,
            EGL.EGL_DEPTH_SIZE, 16,
            EGL.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL.EGL_NONE,
        )
        config = (EGL.EGLConfig * 1)()
        num_configs = ctypes.c_long()
        EGL.eglChooseConfig(display, config_attribs, config, 1,
                            ctypes.pointer(num_configs))
        if num_configs.value == 0:
            raise RuntimeError("EGL: no config supporting OpenGL ES 3.x found")

        EGL.eglBindAPI(EGL.EGL_OPENGL_ES_API)

        context_attribs = (EGL.EGLint * 5)(
            EGL.EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL.EGL_CONTEXT_MINOR_VERSION, 1,
            EGL.EGL_NONE,
        )
        context = EGL.eglCreateContext(display, config[0],
                                       EGL.EGL_NO_CONTEXT, context_attribs)
        if context == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("EGL: failed to create OpenGL ES 3.1 context")

        # Surfaceless — no surface needed, all rendering goes to FBOs
        if not EGL.eglMakeCurrent(display, EGL.EGL_NO_SURFACE,
                                  EGL.EGL_NO_SURFACE, context):
            raise RuntimeError("EGL: failed to make context current")

        self._use_egl = True
        self._egl_display = display
        self._egl_surface = None
        self._egl_context = context

        # PyOpenGL defaults to GLX for context detection on Linux.
        # Patch both the platform instance and module-level reference
        # so that GL calls (e.g. glVertexAttribPointer) can find our context.
        from OpenGL import platform as _gl_platform
        _egl_get_ctx = lambda: EGL.eglGetCurrentContext()
        _gl_platform.PLATFORM.GetCurrentContext = _egl_get_ctx
        _gl_platform.GetCurrentContext = _egl_get_ctx

        print(f"[ShaderRenderer] Headless EGL context created (EGL {major.value}.{minor.value})")

    def _make_current(self):
        """Activate the GL context (no-op for EGL — always current)."""
        if self._use_egl:
            return
        if self.window:
            glfw.make_context_current(self.window)

    def _setup_headless_gl(self):
        """Set up OpenGL state for headless EGL rendering."""
        base_width, base_height = self.frame_dimensions[0]
        self.fb_width = base_width
        self.fb_height = base_height

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_SCISSOR_TEST)

        version = glGetString(GL_VERSION)
        if version:
            print(f"[ShaderRenderer] OpenGL Version: {version.decode()}")
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
        if glsl_version:
            print(f"[ShaderRenderer] GLSL Version: {glsl_version.decode()}")

        self.ctx_initialized = True
        print(f"[ShaderRenderer] Headless rendering ready ({base_width}x{base_height})")

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
        glfw.swap_interval(0)  # Disable V-Sync; frame rate is governed by the main loop

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

        # Framebuffer size callback -- handles async resize on Linux/Wayland
        renderer_self = self

        def _framebuffer_size_callback(window, width, height):
            renderer_self.fb_width = width
            renderer_self.fb_height = height
            for vp in renderer_self.viewports:
                if vp.display_width > 0:
                    vp.display_width = width
                    vp.display_height = height
                    vp.mark_display_modes_dirty()

        glfw.set_framebuffer_size_callback(self.window, _framebuffer_size_callback)

        # Keyboard callback -- F=flat/fan, D=smooth/LED, ESC=quit
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
                                  self.window, headless=self.headless,
                                  group_id=self.group_ids[frame_id])
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

        for vp in self.viewports:
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
        if not self._use_egl:
            glfw.poll_events()

    def should_close(self):
        """Check if window should close"""
        if self._use_egl:
            return False
        return glfw.window_should_close(self.window)

    def swap_buffers(self):
        """Swap window buffers (skip in headless mode)"""
        if not self.headless:
            glfw.swap_buffers(self.window)

    def clear_window(self):
        """Clear the entire window (no-op in headless/EGL mode)"""
        if self._use_egl:
            return
        self._make_current()
        glViewport(0, 0, self.fb_width, self.fb_height)
        glScissor(0, 0, self.fb_width, self.fb_height)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

    def cleanup(self):
        """Clean up resources"""
        for vp in self.viewports:
            vp.cleanup()
        if self._use_egl:
            EGL = _EGL
            if self._egl_display:
                EGL.eglMakeCurrent(self._egl_display, EGL.EGL_NO_SURFACE,
                                   EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
                if self._egl_context:
                    EGL.eglDestroyContext(self._egl_display, self._egl_context)
                EGL.eglTerminate(self._egl_display)
        else:
            if self.window:
                glfw.destroy_window(self.window)
            glfw.terminate()

    def sync_gpu(self):
        """Wait for all GPU operations to complete"""
        self._make_current()
        glFinish()


class ShaderViewport:
    """Window-display wrapper around one GroupCanvas.

    The FBO + effect pipeline + readback live on ``self.canvas`` (a
    GroupCanvas). ShaderViewport adds the GLFW window-display layer:
    flat/fan view modes, smooth/LED render styles, zoom and pan.

    Effects, display modes, the render pipeline, and the web controller
    all read attributes like ``width``, ``height``, ``fbo``,
    ``color_texture``, ``depth_texture``, ``effects``, and ``flip_x``.
    Those are forwarded to the canvas via properties, so callers don't
    need to know whether they hold the canvas or its window wrapper.
    """

    def __init__(self, frame_id: int, width: int, height: int,
                 window_x: int, window_y: int,
                 display_width: int, display_height: int,
                 glfw_window, headless=False, group_id: str = "main"):
        self.frame_id = frame_id
        self.canvas = GroupCanvas(
            group_id=group_id,
            width=width,
            height=height,
            glfw_window=glfw_window,
            frame_id=frame_id,
        )
        self.window_x = window_x      # Position in the GLFW window (always 0)
        self.window_y = window_y
        self.display_width = display_width    # Window pixel size (may differ from FBO)
        self.display_height = display_height
        self.glfw_window = glfw_window
        self.headless = headless

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

    # --- forwarded canvas attributes (kept on the wrapper for API stability) ---
    @property
    def width(self) -> int:
        return self.canvas.width

    @property
    def height(self) -> int:
        return self.canvas.height

    @property
    def led_width(self) -> int:
        return self.canvas.led_width

    @property
    def led_height(self) -> int:
        return self.canvas.led_height

    @property
    def fbo(self):
        return self.canvas.fbo

    @property
    def color_texture(self):
        return self.canvas.color_texture

    @property
    def depth_texture(self):
        return self.canvas.depth_texture

    @property
    def effects(self) -> list:
        return self.canvas.effects

    @property
    def flip_x(self) -> bool:
        return self.canvas.flip_x

    @flip_x.setter
    def flip_x(self, value: bool) -> None:
        self.canvas.flip_x = bool(value)

    def _make_current(self):
        """Activate GL context (no-op when window is None, i.e. EGL headless)."""
        if self.glfw_window is not None:
            glfw.make_context_current(self.glfw_window)

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
    # FBO / effect operations — delegated to the canvas.
    # ------------------------------------------------------------------

    def init_framebuffer(self):
        self.canvas.init_framebuffer()

    def add_effect(self, effect_class, **params):
        return self.canvas.add_effect(effect_class, **params)

    def clear(self):
        """Clear both the FBO (delegated) and the window region."""
        self.canvas.clear()
        if not self.headless and self.display_width > 0:
            self._make_current()
            glViewport(self.window_x, self.window_y, self.display_width, self.display_height)
            glScissor(self.window_x, self.window_y, self.display_width, self.display_height)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def update(self, dt: float, state: Dict):
        self.canvas.update(dt, state)

    def render(self, state: Dict):
        """Render effects to FBO (via canvas), then display in window."""
        self.canvas.render(state)

        if not self.headless and self.display_width > 0:
            mode = self._display_modes[self._current_mode_key]
            if not mode._initialized:
                self._make_current()
                mode.init(self)
            mode.render(self)

    def get_frame(self):
        return self.canvas.get_frame()

    def cleanup(self):
        self.canvas.cleanup()
        for mode in self._display_modes.values():
            mode.cleanup()

    def resize_canvas(self, new_width: int, new_height: int):
        """Resize the underlying GroupCanvas (used by project-swap).

        The window dims (display_width/height) stay the same; only the
        FBO render target changes resolution. Display modes are marked
        dirty so they re-init against the new aspect on the next render.
        """
        self.canvas.resize(new_width, new_height)
        self._geometry = None
        self.mark_display_modes_dirty()
