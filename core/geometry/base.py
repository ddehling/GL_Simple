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
