"""GeometryProvider — interface for project-specific physical-layout geometry.

Two responsibilities:

1. ``to_json()`` — serializable geometry data for the WebGL preview client.
   The dict must include a top-level ``"type"`` key so the JS branches on
   the right rendering path. Existing types: ``"fan"`` (semicircular mesh),
   future: ``"multi_object"`` (free-standing objects laid out in 2-D).

2. ``make_composite_frame(frames)`` — produce the single RGB image that
   gets PNG-encoded and streamed to ``/preview``. For Fan this is just
   the FBO frame. For multi-object pieces this rasterizes per-strip rows
   along their physical polylines onto a composite preview canvas.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class GeometryProvider:
    def to_json(self) -> dict:
        raise NotImplementedError

    def make_composite_frame(self, frames) -> np.ndarray:
        """Return a single (H, W, 3) uint8 RGB array for PNG encoding.

        ``frames`` is a dict ``{group_id: frame_rgb}`` produced by
        ``RenderPipeline._render_shader``. Single-group projects (Fan)
        return the only frame; multi-group projects rasterize each
        strip's row from its group canvas onto a composite preview.
        """
        raise NotImplementedError

    def led_positions(self, group_id: str, strip_idx: int,
                      length: int) -> np.ndarray:
        """Return real-world (x, y) positions for each LED of this
        strip, in chain order.

        Shape: ``(length, 2)`` float32. Coordinates are in the
        composite-canvas pixel space the project uses for its
        ``geometry`` block.

        Default implementation returns an array of ``NaN`` so the
        rendering engine can detect "no physical layout known" and
        fall back to FBO coordinates if it cares. Concrete providers
        (``FanGeometryProvider``, ``MultiObjectGeometryProvider``)
        override this with their actual layout math.
        """
        return np.full((length, 2), np.nan, dtype=np.float32)

    def composite_canvas_size(self) -> tuple[int, int]:
        """Return the ``(width, height)`` in pixels of the composite
        canvas this geometry is authored against.

        Used by the rendering engine to derive a normalized
        coordinate space (each axis in ``[-0.5, +0.5]``, origin at
        canvas center) that's independent of the project's pixel
        dimensions — saner than raw pixels for pattern math.
        Default is a 1024×768 placeholder; concrete providers
        override.
        """
        return (1024, 768)
