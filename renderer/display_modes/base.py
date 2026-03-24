"""Base class for display modes (flat/fan × smooth/LED)."""


class DisplayMode:
    """Abstract display mode that renders the FBO to the GLFW window.

    Subclasses implement init/render/cleanup.  The viewport is passed
    as a parameter (not stored) so modes remain stateless w.r.t. the
    viewport and can be shared or swapped freely.
    """

    def __init__(self):
        self._initialized = False

    def init(self, viewport):
        """Compile shaders and build GL buffers.  Called lazily on first render."""
        raise NotImplementedError

    def render(self, viewport):
        """Draw the FBO contents to the default framebuffer (window)."""
        raise NotImplementedError

    def cleanup(self):
        """Delete owned GL resources."""
        pass

    def mark_dirty(self):
        """Force re-initialization on next render (e.g. after aspect change)."""
        self._initialized = False
