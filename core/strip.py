"""StripBinding — a single LED strip's mapping into a group canvas.

A strip is a 1-D run of LEDs that lives in one group's 2-D canvas. The
strip carries an Nx2 array of (row, col) indices that index into the
canvas; the DMX sender concatenates these per-strip indices to produce
a flat per-receiver pixel-extraction plan.

YAML schema (a strip entry under a receiver's ``strips:`` list is one of):

  # A column of the group canvas (Fan's serpentine convention).
  # ``start`` is the canvas-axis position of LED 0 (the data-in end).
  # The chain walks from ``start`` toward the opposite end of the
  # canvas axis in the chosen ``direction``. Default depends on
  # direction so the un-shifted strip occupies [0, length-1]:
  #   direction=down → start defaults to length-1 (LED 0 at high row)
  #   direction=up   → start defaults to 0        (LED 0 at low row)
  # Fan's existing YAML omits ``start`` and relies on these defaults.
  - { group, strip_idx, kind: column, col, length, direction: down|up, start? }

  # A row of the group canvas (Weight of Light's convention).
  # ``row`` defaults to ``strip_idx`` if omitted. ``start`` semantics
  # match column kind but on the column axis:
  #   direction=right → start defaults to 0        (LED 0 at low col)
  #   direction=left  → start defaults to length-1 (LED 0 at high col)
  - { group, strip_idx, kind: row, row, length, direction: right|left, start? }

  # Verbose escape hatch — explicit Nx2 indices (used by the legacy
  # HS/VS file paths and by the legacy_to_strips fallback).
  - { group, strip_idx, kind: raw, indices: [[r0,c0], [r1,c1], ...] }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np


@dataclass
class StripBinding:
    group_id: str
    strip_idx: int
    pixel_indices: np.ndarray  # (N, 2) int32, [row, col] into group canvas
    # Project-level identifier of the physical box (receiver) this strip
    # is wired to. Effects can read this from ``state['strips_by_group']``
    # to drive per-object behaviours (e.g. light up object 3's leaves
    # when its sensor fires). ``-1`` means "no associated object".
    # Populated by Stories_OGL._build_receivers after the YAML loader
    # returns the strips, since that's where the receiver context lives.
    object_id: int = -1

    def __post_init__(self):
        if self.pixel_indices.dtype != np.int32:
            self.pixel_indices = self.pixel_indices.astype(np.int32)
        if self.pixel_indices.ndim != 2 or self.pixel_indices.shape[1] != 2:
            raise ValueError(
                f"StripBinding.pixel_indices must be (N, 2); got {self.pixel_indices.shape}"
            )

    @property
    def length(self) -> int:
        return int(self.pixel_indices.shape[0])


# ---------------------------------------------------------------------------
# Builders for the legacy Fan addressing modes
# ---------------------------------------------------------------------------

def strips_from_column_rect(columns: int, height: int, column_offset: int,
                            group_id: str = "main",
                            start_strip_idx: int = 0) -> List[StripBinding]:
    """Reproduce ``make_indices_V_rect_alternate`` as one StripBinding per column.

    Convention (matches lib/dmx_sender.py:make_indices_V_rect_alternate):
      - even x within the block goes top-to-bottom (descending y)
      - odd x  within the block goes bottom-to-top (ascending y)
    """
    strips: List[StripBinding] = []
    for x in range(columns):
        col = x + column_offset
        if x % 2 == 0:
            rows = np.arange(height - 1, -1, -1, dtype=np.int32)
        else:
            rows = np.arange(0, height, dtype=np.int32)
        cols = np.full(height, col, dtype=np.int32)
        idx = np.stack([rows, cols], axis=1)
        strips.append(StripBinding(
            group_id=group_id,
            strip_idx=start_strip_idx + x,
            pixel_indices=idx,
        ))
    return strips


def strips_from_hs_lines(lines: Iterable[Iterable[float]],
                         group_id: str = "main",
                         start_strip_idx: int = 0) -> List[StripBinding]:
    """One strip per line of an HS file: each line is ``[row, start_col, ±count]``."""
    strips: List[StripBinding] = []
    for i, line in enumerate(lines):
        row, start_col, count = line
        row = int(row)
        start_col = int(start_col)
        count_i = int(count)
        if count_i > 0:
            cols = np.arange(start_col, start_col + count_i, dtype=np.int32)
        else:
            # Match make_indicesHS exactly:  start - count - 1 - m  for m in 0..-count-1
            n = -count_i
            cols = np.array(
                [start_col - count_i - 1 - m for m in range(n)],
                dtype=np.int32,
            )
        rows = np.full(cols.shape[0], row, dtype=np.int32)
        idx = np.stack([rows, cols], axis=1)
        strips.append(StripBinding(
            group_id=group_id,
            strip_idx=start_strip_idx + i,
            pixel_indices=idx,
        ))
    return strips


def strips_from_vs_lines(lines: Iterable[Iterable[float]],
                         group_id: str = "main",
                         start_strip_idx: int = 0) -> List[StripBinding]:
    """One strip per line of a VS file: each line is ``[col, start_row, ±count]``."""
    strips: List[StripBinding] = []
    for i, line in enumerate(lines):
        col, start_row, count = line
        col = int(col)
        start_row = int(start_row)
        count_i = int(count)
        if count_i > 0:
            rows = np.arange(start_row, start_row + count_i, dtype=np.int32)
        else:
            n = -count_i
            rows = np.array(
                [start_row - count_i - 1 - m for m in range(n)],
                dtype=np.int32,
            )
        cols = np.full(rows.shape[0], col, dtype=np.int32)
        idx = np.stack([rows, cols], axis=1)
        strips.append(StripBinding(
            group_id=group_id,
            strip_idx=start_strip_idx + i,
            pixel_indices=idx,
        ))
    return strips


# ---------------------------------------------------------------------------
# YAML loader (one entry under a receiver's ``strips:`` list)
# ---------------------------------------------------------------------------

def strip_from_yaml(entry: dict) -> StripBinding:
    """Build a single StripBinding from one YAML strip entry."""
    if "kind" not in entry:
        raise ValueError(f"Strip entry missing 'kind': {entry}")
    kind = entry["kind"]
    group = entry.get("group", "main")
    strip_idx = int(entry.get("strip_idx", 0))

    if kind == "column":
        col = int(entry["col"])
        length = int(entry["length"])
        direction = entry.get("direction", "down")
        # ``start`` = canvas-axis position of LED 0 (the data-in end).
        # Default chosen so the un-shifted strip occupies [0, length-1]:
        # for ``down`` the chain runs high→low so LED 0 sits at the
        # high end (length-1); for ``up`` the chain runs low→high so
        # LED 0 sits at the low end (0).
        if direction == "down":
            start = int(entry.get("start", length - 1))
            rows = np.arange(start, start - length, -1, dtype=np.int32)
        elif direction == "up":
            start = int(entry.get("start", 0))
            rows = np.arange(start, start + length, dtype=np.int32)
        else:
            raise ValueError(f"Unknown direction for column strip: {direction!r}")
        cols = np.full(length, col, dtype=np.int32)
        idx = np.stack([rows, cols], axis=1)

    elif kind == "row":
        # Default ``row`` to ``strip_idx`` for the multi-group convention
        # where each strip occupies its own row in the group canvas.
        row = int(entry.get("row", strip_idx))
        length = int(entry["length"])
        direction = entry.get("direction", "right")
        # Same direction-aware default as column: LED 0 at the natural
        # data-in end of the row.
        if direction == "right":
            start = int(entry.get("start", 0))
            cols = np.arange(start, start + length, dtype=np.int32)
        elif direction == "left":
            start = int(entry.get("start", length - 1))
            cols = np.arange(start, start - length, -1, dtype=np.int32)
        else:
            raise ValueError(f"Unknown direction for row strip: {direction!r}")
        rows = np.full(length, row, dtype=np.int32)
        idx = np.stack([rows, cols], axis=1)

    elif kind == "raw":
        idx = np.asarray(entry["indices"], dtype=np.int32)

    else:
        raise ValueError(f"Unknown strip kind: {kind!r}")

    return StripBinding(group_id=group, strip_idx=strip_idx, pixel_indices=idx)


def strips_from_yaml_list(entries: Iterable[dict]) -> List[StripBinding]:
    return [strip_from_yaml(e) for e in entries]


# ---------------------------------------------------------------------------
# Concat to flat addressing array (legacy hot-path bridge)
# ---------------------------------------------------------------------------

def addressing_array_from_strips(strips: Iterable[StripBinding]) -> np.ndarray:
    """Concatenate strips' pixel_indices in strip-order; returns Nx2 int."""
    arrs = [s.pixel_indices for s in strips]
    if not arrs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.concatenate(arrs, axis=0).astype(int)
