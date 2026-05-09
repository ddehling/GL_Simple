"""FanGeometryProvider — adapter for the existing FanGeometry to the
project-level GeometryProvider interface.

The Fan piece's preview is a direct PNG of the single group canvas plus
a polar overlay rendered client-side from FanGeometry's serialized
geometry. So ``make_composite_frame`` is a passthrough and ``to_json``
delegates to FanGeometry plus a ``type: fan`` discriminator.

Construction parameters (all optional; defaults reproduce the existing
runtime values):

  num_strips    : int   — group canvas width (default 128)
  num_leds      : int   — group canvas height (default 300)
  display_aspect: float — preview window aspect (default 900/500 = 1.8)
  inner_r, outer_r — passed through to FanGeometry; defaults match the
                     physical installation.
"""
from __future__ import annotations

import numpy as np

from core.geometry.base import GeometryProvider
from renderer.fan_geometry import FanGeometry


class FanGeometryProvider(GeometryProvider):
    def __init__(self, num_strips: int = 128, num_leds: int = 300,
                 display_aspect: float = 900 / 500,
                 inner_r: float | None = None, outer_r: float = 0.95,
                 group_id: str = "main"):
        self._fan = FanGeometry(
            num_strips=num_strips,
            num_leds=num_leds,
            display_aspect=display_aspect,
            inner_r=inner_r,
            outer_r=outer_r,
        )
        self._group_id = group_id
        self._json_cache: dict | None = None

    def to_json(self) -> dict:
        if self._json_cache is None:
            data = self._fan.to_json()
            data["type"] = "fan"
            self._json_cache = data
        return self._json_cache

    def composite_canvas_size(self) -> tuple[int, int]:
        # Matches the canvas the layout editor and led_positions
        # use to map polar coords to pixels.
        return (1024, 600)

    def make_composite_frame(self, frames) -> np.ndarray:
        # Single-group project: pull the named group's frame, with a
        # generic fallback in case the project renames "main".
        if isinstance(frames, dict):
            if self._group_id in frames:
                return frames[self._group_id]
            return next(iter(frames.values()))
        # Legacy list form (pre-Phase 6); kept for any caller not yet ported.
        return frames[0]

    def led_positions(self, group_id: str, strip_idx: int,
                      length: int) -> np.ndarray:
        """Per-LED (x, y) positions in composite-canvas pixels.

        Fan strips are radial lines: column ``strip_idx`` sits at angle
        ``θ = π − strip_idx/(num_strips-1) · π`` and runs from
        ``inner_r`` (close to center) to ``outer_r`` (far from center).
        We adopt the same composite-canvas coordinate system the
        layout editor uses (1024×600, center at (512, 560)) so the
        positions are interpretable as physical pixels.

        LED chain order matches ``core/strip.py``'s ``direction``
        convention: ``down`` walks high row → low row, ``up`` walks
        low → high. Since FanGeometry maps texture v=0 to inner_r
        and the FBO readout is np.flipud'd, "down" runs OUTER→INNER
        and "up" runs INNER→OUTER. We don't have the strip's
        ``direction`` here (it's not on StripBinding) so we return
        the polyline in OUTER→INNER order; callers that care about
        chain alignment can flip it via the strip's direction."""
        import math
        n_cols = self._fan.num_strips
        if not (0 <= strip_idx < n_cols):
            return np.full((length, 2), np.nan, dtype=np.float32)

        # Composite canvas dims — matches the layout editor's Fan
        # rendering. Could be made configurable later if the web
        # preview uses different dims.
        canvas_w, canvas_h = 1024, 600
        cx, cy = canvas_w / 2.0, canvas_h - 40.0
        outer_r_px = min(canvas_w * 0.46, canvas_h * 0.92)
        inner_r_px = outer_r_px * (self._fan.inner_r / self._fan.outer_r)

        theta = math.pi - (strip_idx / max(n_cols - 1, 1)) * math.pi
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Sample radii from outer_r to inner_r — matches "down"
        # direction's chain order on the FBO. ``length`` LEDs evenly
        # along the radial line.
        if length <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if length == 1:
            r = (outer_r_px + inner_r_px) / 2
            return np.array([[cx + r * cos_t, cy - r * sin_t]],
                            dtype=np.float32)
        ts = np.linspace(0.0, 1.0, length, dtype=np.float32)
        rs = outer_r_px + ts * (inner_r_px - outer_r_px)
        xs = cx + rs * cos_t
        ys = cy - rs * sin_t
        return np.stack([xs, ys], axis=1).astype(np.float32)
