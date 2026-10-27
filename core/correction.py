"""Per-receiver output correction: gain and gamma.

Two knobs a box can carry in ``project.yaml`` to sit differently from the
rest of the piece. Both are properties of the physical node — its LED
batch, its diffuser, its driver, where it stands in the room — so they
live on the receiver rather than in the show's global controls::

    receivers:
    - host: ethernode-north.local
      protocol: ddp
      object_id: 1
      gain: 1.35        # brightness emphasis; 1.0 = no emphasis, 0 mutes
      gamma: 2.4        # this box's exponent; 0 = follow the system slider

``gain`` scales this box's brightness relative to the piece. Weight of
Light uses it to push the central sculpture forward. It is applied
BEFORE the brightness limiter on purpose: the limiter then sees the true
total, so a boosted node takes its extra brightness out of the piece's
shared power budget instead of on top of it.

``gamma`` overrides the web control panel's Gamma slider — the *system*
gamma the pipeline otherwise applies to every canvas. The field carries
its own sentinel: anything **below 0.01** (the default, ``0``) means
"follow the slider"; any larger value is used verbatim as that box's
exponent. There is no separate enable flag, so the layout editor can
present the whole knob as one cell that reads ``System`` at 0.

How it runs
-----------
Canvases come off the PBO as uint8, so gamma is a 256-entry lookup
table, not a per-pixel ``pow``. Everything expensive is therefore
precomputed and only rebuilt when an input actually changes:

  * At project load, ``build_correction_plans`` resolves each receiver's
    knobs to a region of its group's canvas, read off the
    ``receiver_idx`` metadata atlas — so a correction lands on exactly
    the pixels that box drives, whether its strips are rows (Weight of
    Light), columns (Fan) or arbitrary polylines. A region that turns
    out to be a contiguous rectangle (the normal case: a box owns a run
    of rows or columns) is stored as slices, which are far cheaper to
    write than a boolean mask.
  * The tables themselves are built once per system-gamma value and
    cached on the plan. Moving the slider rebuilds k tables of 256
    entries — microseconds — not a per-pixel pass. Each table folds that
    box's gain in with its gamma, so the whole correction for a pixel is
    one lookup rather than an exponentiation plus a multiply.
  * Per frame, ``CorrectionPlan.apply`` does one table lookup over the
    canvas and then overwrites each corrected region. Pixels inside a
    region are looked up twice and the first result discarded; that
    costs less than the slicing needed to avoid it (measured), and a
    lookup is cheap enough that it does not matter.

A group with no corrections at all keeps no plan and takes a single
whole-canvas lookup.
"""
from __future__ import annotations

import math

import numpy as np

# Sentinel: a receiver gamma this small means "use the system slider".
SYSTEM = 0.0
MIN_OVERRIDE = 0.01
# A gain of exactly this is a no-op and is never given a region.
NEUTRAL_GAIN = 1.0

# Normalized 0..1 ramp every table is built from.
_RAMP = np.arange(256, dtype=np.float64) / 255.0


def parse_gamma(value) -> float:
    """Normalize a ``project.yaml`` receiver ``gamma:`` value.

    Returns the exponent, or ``SYSTEM`` (0.0) for the default / anything
    unusable — a missing key, a non-number, NaN, a negative, or a value
    under the 0.01 threshold. Bad input reads as "System" rather than
    raising: a typo in one receiver shouldn't stop the show from booting.
    """
    try:
        g = float(value)
    except (TypeError, ValueError):
        return SYSTEM
    if not math.isfinite(g) or g < MIN_OVERRIDE:
        return SYSTEM
    return g


def parse_gain(value) -> float:
    """Normalize a ``project.yaml`` receiver ``gain:`` value.

    Returns the multiplier, or ``NEUTRAL_GAIN`` (1.0) for the default /
    anything unusable. ``0`` is legitimate — it mutes the box — but a
    negative or non-finite gain is a typo, and reading it as "no
    emphasis" is the safe interpretation.
    """
    try:
        g = float(value)
    except (TypeError, ValueError):
        return NEUTRAL_GAIN
    if not math.isfinite(g) or g < 0.0:
        return NEUTRAL_GAIN
    return g


def make_table(gamma: float, gain: float = NEUTRAL_GAIN) -> np.ndarray:
    """The 256-entry correction table for one (gamma, gain) pair.

    ``table[v]`` is the corrected output for input byte ``v``, gain
    included. Values may exceed 255 when gain > 1 — the same as the old
    multiply produced, and the brightness limiter downstream is what
    brings the frame back into range.
    """
    return np.power(_RAMP, gamma) * 255.0 * gain


def _as_region(mask: np.ndarray):
    """Reduce a boolean mask to slices when it is a solid rectangle.

    A box almost always owns a contiguous run of canvas rows (Weight of
    Light) or columns (Fan), and writing ``out[r0:r1, c0:c1]`` is several
    times cheaper than a boolean scatter. Anything irregular (polyline
    strips that skip pixels) keeps its mask.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return mask
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    if int(mask.sum()) == (r1 - r0) * (c1 - c0):
        return (slice(r0, r1), slice(c0, c1))
    return mask


class CorrectionPlan:
    """One group canvas's baked-in per-receiver corrections.

    Holds the regions (resolved once at project load) and the lookup
    tables (rebuilt only when the system gamma changes). One instance per
    group; the render pipeline calls ``apply`` on it every frame.
    """

    def __init__(self, shape, regions):
        # (height, width) the regions were resolved against. A canvas
        # resized under a live plan (project swap mid flight) must not
        # silently apply the old regions to the new canvas.
        self.shape = tuple(shape)
        # Each entry: (region, gamma, gain, pixel_count). ``gamma`` is
        # SYSTEM when this box only sets gain.
        self.regions = list(regions)
        self._system_gamma = None
        self._base = None
        self._tables = []

    def _rebuild(self, system_gamma: float) -> None:
        self._base = make_table(system_gamma)
        self._tables = [
            self._base * gain if gamma == SYSTEM else make_table(gamma, gain)
            for _, gamma, gain, _ in self.regions
        ]
        self._system_gamma = system_gamma

    def apply(self, frame_rgb: np.ndarray, system_gamma: float) -> np.ndarray:
        """Correct one canvas. Returns a fresh array; input untouched."""
        if system_gamma != self._system_gamma:
            self._rebuild(system_gamma)
        out = self._base[frame_rgb]
        if frame_rgb.shape[:2] != self.shape:
            # Stale plan — every pixel keeps the system correction rather
            # than us writing a region into the wrong place.
            return out
        for (region, _, _, _), table in zip(self.regions, self._tables):
            out[region] = table[frame_rgb[region]]
        return out

    def tables(self, frame_shape, system_gamma: float):
        """(base table, [(region, table), ...]) for one canvas - what
        ``apply`` looks up, without doing the lookup (see fused_tables)."""
        if system_gamma != self._system_gamma:
            self._rebuild(system_gamma)
        if tuple(frame_shape[:2]) != self.shape:
            return self._base, []                     # stale plan: system correction only
        return self._base, [(region, table) for (region, _, _, _), table
                            in zip(self.regions, self._tables)]

    def summary(self) -> list:
        """(pixels, gamma, gain) per region, for boot logging."""
        return [(px, gamma, gain) for _, gamma, gain, px in self.regions]


def apply_correction(frame_rgb: np.ndarray, system_gamma: float,
                     plan: CorrectionPlan = None) -> np.ndarray:
    """Correct one canvas with an optional per-receiver plan.

    Groups without a plan (the common case) take a single whole-canvas
    table lookup, which is what the old scalar ``pow`` path did — only
    without recomputing the exponent for every pixel of every frame.
    """
    if plan is not None:
        return plan.apply(frame_rgb, system_gamma)
    return make_table(system_gamma)[frame_rgb]


# ---------------------------------------------------------------------------
# Fused correction + brightness limiting (the render pipeline's hot path)
# ---------------------------------------------------------------------------
#
# The pipeline used to materialize the corrected frame in float64 (one table
# lookup), sum it per channel for the brightness limiter, divide it, scale it
# by the web brightness, clip it and cast it to uint8 - five full-frame
# passes, ~1 ms per Fan frame. Every one of those steps is a function of the
# input BYTE alone, so:
#   * the limiter's total is sum_v count[v] * table[v] - a 256-bin histogram
#     of the raw frame dotted with the table, per channel and per region;
#   * the output is ONE uint8 table lookup, the table built by the same
#     float64 operations, in the same order, the frame used to go through.
# Per-pixel results are therefore the very same bytes. Only the limiter's
# total is summed in a different order (pairwise over pixels before, over
# bins now), which can move it in the last bit - checked over long frame
# sequences in the session that made this change: no output byte moved.

_SYSTEM_TABLES: dict = {}


def system_table(gamma: float) -> np.ndarray:
    """make_table(gamma), cached: the slider changes rarely, frames do not."""
    t = _SYSTEM_TABLES.get(gamma)
    if t is None:
        if len(_SYSTEM_TABLES) > 64:
            _SYSTEM_TABLES.clear()
        t = _SYSTEM_TABLES[gamma] = make_table(gamma)
    return t


def fused_tables(frame_shape, system_gamma: float, plan: CorrectionPlan = None):
    """(base, regions): the tables ``apply_correction`` would look up."""
    if plan is not None:
        return plan.tables(frame_shape, system_gamma)
    return system_table(system_gamma), []


try:                                   # compiled loops when Numba is there (it is a requirement)
    from lib.pixel_extract import apply_lut_u8 as _apply_lut_u8
    from lib.pixel_extract import table_channel_sums as _table_channel_sums
    # Compile (or load the on-disk cache) at import - boot - never on the
    # render thread's first frame.
    _w = np.zeros((1, 3), dtype=np.uint8)
    _table_channel_sums(_w, np.zeros(256), np.empty(3))
    _apply_lut_u8(_w.reshape(-1), np.zeros(256, dtype=np.uint8), np.empty(3, dtype=np.uint8))
    del _w
except Exception:                      # pragma: no cover - numpy fallback below
    _apply_lut_u8 = _table_channel_sums = None


def corrected_channel_sums(frame_rgb: np.ndarray, base: np.ndarray, regions) -> list:
    """Per-channel sums of the corrected frame, read off the raw one.
    Regions are receiver ownership masks, disjoint by construction."""
    if _table_channel_sums is not None and frame_rgb.flags.c_contiguous and frame_rgb.shape[-1] == 3:
        out3 = np.empty(3)
        _table_channel_sums(frame_rgb.reshape(-1, 3), np.ascontiguousarray(base, dtype=np.float64), out3)
        sums = [float(out3[0]), float(out3[1]), float(out3[2])]
    else:
        sums = [float(np.bincount(frame_rgb[..., c].ravel(), minlength=256) @ base)
                for c in range(3)]
    for region, table in regions:
        px = frame_rgb[region]
        delta = table - base
        for c in range(3):
            sums[c] += float(np.bincount(px[..., c].ravel(), minlength=256) @ delta)
    return sums


def output_lut(table: np.ndarray, divisor=None, web_brightness: float = 1.0) -> np.ndarray:
    """The uint8 lookup for one table: exactly the float ops the frame used to
    take (divide when limiting, scale when dimmed, clip, cast)."""
    x = table
    if divisor is not None:
        x = x / divisor
    if web_brightness < 1.0:
        x = x * max(web_brightness, 0.0)
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_output_luts(frame_rgb: np.ndarray, base_lut: np.ndarray, region_luts) -> np.ndarray:
    if _apply_lut_u8 is not None and frame_rgb.flags.c_contiguous:
        out = np.empty_like(frame_rgb)
        _apply_lut_u8(frame_rgb.reshape(-1), base_lut, out.reshape(-1))
    else:
        out = base_lut[frame_rgb]
    for region, lut in region_luts:
        out[region] = lut[frame_rgb[region]]
    return out


def build_correction_plans(receivers, group_metadata) -> dict:
    """Resolve every receiver's gain/gamma to per-group correction plans.

    ``receivers`` is the project's FULL receiver list (the same list the
    metadata atlas was built from, so list position matches the atlas's
    ``receiver_idx`` values). ``group_metadata`` is the per-group atlas
    dict built by ``Stories_OGL._publish_strip_metadata``.

    Returns ``{group_id: CorrectionPlan}``, omitting groups where no
    receiver corrects anything.
    """
    plans: dict = {}
    for gid, atlas in (group_metadata or {}).items():
        owner = (atlas or {}).get("receiver_idx")
        if owner is None:
            continue
        regions = []
        for rx_idx, rx in enumerate(receivers):
            if not isinstance(rx, dict):
                continue
            gamma = parse_gamma(rx.get("gamma", SYSTEM))
            gain = parse_gain(rx.get("gain", NEUTRAL_GAIN))
            if gamma == SYSTEM and gain == NEUTRAL_GAIN:
                continue
            mask = (owner == rx_idx)
            count = int(mask.sum())
            if count == 0:
                continue
            regions.append((_as_region(mask), gamma, gain, count))
        if regions:
            plans[gid] = CorrectionPlan(owner.shape, regions)
    return plans
