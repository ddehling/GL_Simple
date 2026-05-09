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

    def make_composite_frame(self, frames) -> np.ndarray:
        # Single-group project: pull the named group's frame, with a
        # generic fallback in case the project renames "main".
        if isinstance(frames, dict):
            if self._group_id in frames:
                return frames[self._group_id]
            return next(iter(frames.values()))
        # Legacy list form (pre-Phase 6); kept for any caller not yet ported.
        return frames[0]
