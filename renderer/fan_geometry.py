"""Pure-numpy fan/polar geometry for the LED semicircle display.

Computes vertex positions, instance data, and dot radii for all four
display modes (flat/fan × continuous/dots).  No OpenGL dependency —
the arrays are consumed by ShaderViewport for GL upload and by the
web preview client via ``to_json()``.
"""

import math
import numpy as np


class FanGeometry:
    """Precomputes geometry for a semicircular LED fan.

    Parameters
    ----------
    num_strips : int
        Number of LED strips (columns in the framebuffer, e.g. 128).
    num_leds : int
        Number of LEDs per strip (rows in the framebuffer, e.g. 300).
    display_aspect : float
        Width / height of the display window (used to keep the
        semicircle visually circular).
    inner_r, outer_r : float
        Normalised inner / outer radii of the fan.  Default 0.05 / 0.95.
    """

    def __init__(self, num_strips: int, num_leds: int, display_aspect: float,
                 inner_r: float = 0.05, outer_r: float = 0.95):
        self.num_strips = num_strips
        self.num_leds = num_leds
        self.inner_r = inner_r
        self.outer_r = outer_r

        # Aspect correction so the semicircle stays circular
        self.aspect = display_aspect
        self.a_half = max(display_aspect * 0.5, 0.01)

        # Scale so widest point fits in clip-space [-1, 1]
        max_x = outer_r / self.a_half
        self.fit_scale = min(1.0, 0.98 / max_x)

        # Per-strip angles (pi … 0, left-to-right)
        n = max(num_strips - 1, 1)
        self.thetas = np.array([math.pi - (i / n) * math.pi
                                for i in range(num_strips)], dtype=np.float64)
        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _polar_to_clip(self, r: float, cos_t: float, sin_t: float):
        """Convert (r, cos θ, sin θ) to clip-space (x, y)."""
        x = r * cos_t / self.a_half * self.fit_scale
        y = (r * sin_t * 2.0 - 0.9) * self.fit_scale
        return x, y

    # ------------------------------------------------------------------
    # Public geometry builders
    # ------------------------------------------------------------------

    def compute_fan_mesh(self):
        """Fan continuous mesh: textured semicircle.

        Returns
        -------
        verts : np.ndarray, float32, shape (N, 4)
            Each row is (x, y, u, v).
        indices : np.ndarray, uint32
            Triangle indices.
        """
        ns = self.num_strips
        nl = self.num_leds
        stride = nl + 1  # radial vertices per strip (includes both endpoints)

        verts = np.empty((ns * stride, 4), dtype=np.float32)
        for i in range(ns):
            u = i / max(ns - 1, 1)
            cos_t = self.cos_thetas[i]
            sin_t = self.sin_thetas[i]
            for j in range(stride):
                t = j / nl
                r = self.inner_r + t * (self.outer_r - self.inner_r)
                x, y = self._polar_to_clip(r, cos_t, sin_t)
                idx = i * stride + j
                verts[idx] = (x, y, u, t)

        # Two triangles per quad
        indices = []
        for i in range(ns - 1):
            for j in range(nl):
                tl = i * stride + j
                tr = (i + 1) * stride + j
                bl = tl + 1
                br = tr + 1
                indices.extend([tl, bl, tr, bl, br, tr])
        indices = np.array(indices, dtype=np.uint32)

        return verts, indices

    def compute_fan_dots(self):
        """Instance data for fan dot mode.

        Returns
        -------
        instances : np.ndarray, float32, shape (N, 4)
            Each row is (clip_x, clip_y, u, v).
        dot_radius : float
            Suggested dot radius in clip-space units.
        """
        ns = self.num_strips
        nl = self.num_leds
        instances = np.empty((ns * nl, 4), dtype=np.float32)
        idx = 0
        for i in range(ns):
            u = i / max(ns - 1, 1)
            cos_t = self.cos_thetas[i]
            sin_t = self.sin_thetas[i]
            for j in range(nl):
                t = (j + 0.5) / nl
                r = self.inner_r + t * (self.outer_r - self.inner_r)
                x, y = self._polar_to_clip(r, cos_t, sin_t)
                instances[idx] = (x, y, u, t)
                idx += 1

        # Dot radius from average spacing
        avg_r = (self.inner_r + self.outer_r) / 2.0
        arc_per_strip = avg_r * math.pi / ns
        radial_per_led = (self.outer_r - self.inner_r) / nl
        avg_spacing = min(arc_per_strip / self.a_half * self.fit_scale,
                          radial_per_led * 2.0 * self.fit_scale)
        dot_radius = avg_spacing * 0.45

        return instances, dot_radius

    def compute_flat_dots(self):
        """Instance data for flat dot mode.

        Returns
        -------
        instances : np.ndarray, float32, shape (N, 4)
            Each row is (clip_x, clip_y, u, v).
        dot_radius : float
            Suggested dot radius in clip-space units.
        """
        ns = self.num_strips
        nl = self.num_leds
        instances = np.empty((ns * nl, 4), dtype=np.float32)
        idx = 0
        for i in range(ns):
            u = (i + 0.5) / ns
            x = u * 2.0 - 1.0
            for j in range(nl):
                v = (j + 0.5) / nl
                y = v * 2.0 - 1.0
                instances[idx] = (x, y, u, v)
                idx += 1

        dot_radius = (1.0 / max(ns, nl)) * 0.9
        return instances, dot_radius

    # ------------------------------------------------------------------
    # Serialisation for the web preview client
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Return all geometry as JSON-serialisable Python objects.

        The web client fetches this once and builds WebGL buffers from it.
        """
        fan_verts, fan_indices = self.compute_fan_mesh()
        fan_dots, fan_dot_r = self.compute_fan_dots()
        flat_dots, flat_dot_r = self.compute_flat_dots()

        return {
            "num_strips": self.num_strips,
            "num_leds": self.num_leds,
            "fan_mesh": {
                "verts": fan_verts.flatten().tolist(),
                "indices": fan_indices.tolist(),
            },
            "fan_dots": {
                "instances": fan_dots.flatten().tolist(),
                "dot_radius": fan_dot_r,
            },
            "flat_dots": {
                "instances": flat_dots.flatten().tolist(),
                "dot_radius": flat_dot_r,
            },
        }
