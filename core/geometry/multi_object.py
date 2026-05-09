"""MultiObjectGeometryProvider — preview rasterizer for free-standing-object pieces.

For pieces like Weight of Light, the physical layout is N free-standing
objects (1 center + 8 peripheral, etc.) each carrying several LED strips
laid out along physical paths (polylines). The web preview shows them
all in one composite canvas; this provider does the per-strip rasterization
on the CPU at the preview's 15 Hz rate.

YAML schema (one file at the path named by ``project.yaml``'s
``geometry.file:``):

    canvas: { width: 1024, height: 768 }
    objects:
      - { id: 0, name: center, x: 512, y: 416 }
      ...
    strips:
      # one entry per (group, strip_idx) pair that exists in any receiver
      - group: trunk
        strip_idx: 0           # = row in the trunk group's canvas
        length: 144            # number of LEDs along this strip
        polyline: [[512, 426], [512, 282]]   # px coords in composite canvas

The strip's source pixels are read from ``frames[group][strip_idx, :length]``
at composite time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from core.geometry.base import GeometryProvider


def _resample_polyline(polyline: np.ndarray, n: int) -> np.ndarray:
    """Return ``n`` evenly-spaced (x, y) points along ``polyline``.

    polyline: (M, 2) float; n: number of samples (== strip length).
    Cumulative-arclength parameterization so multi-segment polylines
    (Phase 6+) are sampled by physical distance, not vertex count.
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if polyline.shape[0] < 2:
        return np.tile(polyline[:1], (n, 1)).astype(np.float32)

    seg = np.diff(polyline, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return np.tile(polyline[:1], (n, 1)).astype(np.float32)

    targets = np.linspace(0.0, total, n)
    seg_idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg_len) - 1)
    local = (targets - cum[seg_idx]) / np.maximum(seg_len[seg_idx], 1e-9)
    pts = polyline[seg_idx] + local[:, None] * seg[seg_idx]
    return pts.astype(np.float32)


class MultiObjectGeometryProvider(GeometryProvider):
    def __init__(self, geometry_yaml_path: str | Path,
                 dot_radius: int = 2, gain: float = 1.0):
        path = Path(geometry_yaml_path)
        with open(path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}

        canvas = spec.get("canvas", {"width": 1024, "height": 768})
        self.canvas_w = int(canvas["width"])
        self.canvas_h = int(canvas["height"])
        self.objects = spec.get("objects", []) or []
        self.dot_radius = int(dot_radius)
        self.gain = float(gain)

        # Pre-resample every strip's polyline to its declared length so the
        # per-frame composite is just gather-and-paint (no geometry math).
        self._strips: list[dict] = []
        for s in spec.get("strips", []) or []:
            polyline = np.asarray(s["polyline"], dtype=np.float32)
            length = int(s.get("length", polyline.shape[0]))
            if length == polyline.shape[0] and length > 2:
                samples = polyline
            else:
                samples = _resample_polyline(polyline, length)
            self._strips.append({
                "group": s["group"],
                "strip_idx": int(s["strip_idx"]),
                "samples": samples,   # (length, 2) float
                "length": length,
            })
        # Round once so the per-frame paint is integer indexing only.
        for s in self._strips:
            s["px"] = np.round(s["samples"]).astype(np.int32)

        self._json_cache: dict | None = None

    # ------------------------------------------------------------------
    # GeometryProvider interface
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        if self._json_cache is None:
            self._json_cache = {
                "type": "multi_object",
                # Aspect dims for preview.js (which still keys off these
                # names from the Fan path): treat the composite canvas as
                # the "raster" the JS should display.
                "num_strips": self.canvas_w,
                "num_leds": self.canvas_h,
                "canvas": {"width": self.canvas_w, "height": self.canvas_h},
                "objects": self.objects,
                "strips": [
                    {
                        "group": s["group"],
                        "strip_idx": s["strip_idx"],
                        "polyline": s["samples"].tolist(),
                    }
                    for s in self._strips
                ],
            }
        return self._json_cache

    def make_composite_frame(self, frames) -> np.ndarray:
        """For each strip, sample its row from the strip's group canvas and
        paint the LEDs along the strip's polyline onto the composite."""
        composite = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        if not frames:
            return composite

        # Phase-6 dict form is the supported shape; legacy list form is
        # tolerated for any caller still passing a list-of-frames.
        if not isinstance(frames, dict):
            frames = {"main": frames[0]} if frames else {}

        r = self.dot_radius
        cw, ch = self.canvas_w, self.canvas_h

        for s in self._strips:
            group_frame = frames.get(s["group"])
            if group_frame is None:
                continue
            atlas_h, atlas_w = group_frame.shape[:2]
            row = s["strip_idx"]
            if row < 0 or row >= atlas_h:
                continue
            length = min(s["length"], atlas_w)
            if length <= 0:
                continue
            colors = group_frame[row, :length, :]      # (length, 3) uint8
            if self.gain != 1.0:
                colors = np.clip(colors.astype(np.int32) * self.gain, 0, 255).astype(np.uint8)
            xs = s["px"][:length, 0]
            ys = s["px"][:length, 1]
            # Stamp a (2r+1) square dot per LED. Vectorize the box by
            # painting the center pixel and r neighbors in each direction.
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    px = np.clip(xs + dx, 0, cw - 1)
                    py = np.clip(ys + dy, 0, ch - 1)
                    composite[py, px] = colors

        return composite
