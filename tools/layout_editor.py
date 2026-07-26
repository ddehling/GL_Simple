"""Visual layout editor for ``multi_object``-geometry projects.

Edits the four entities that drive how a project's strips map to physical
objects + their canvases:

  * Objects   — labeled positions on the composite preview canvas
  * Groups    — rendering surfaces (id, width, height) shared across strips
  * Strips    — (group, strip_idx, length, polyline, receiver) tuples
  * Receivers — physical hardware addresses (host/ip + protocol)

Run::

    python tools/layout_editor.py

The editor opens a project picker (defaults to the first project whose
``geometry.type`` is ``multi_object``). Edits in the GUI are the source
of truth — saving writes directly to ``projects/<id>/geometry.yaml`` and
``projects/<id>/project.yaml``.

v1 capabilities (this revision):
  * Project picker + auto-detect multi_object projects
  * 4 fully-editable tables with add/remove
  * Live preview canvas: objects as labeled dots, strip polylines colored
    by group; click on canvas → highlight in table; click in table → highlight on canvas
  * Drag objects on canvas to reposition (edits are applied immediately)
  * Save → writes both YAML files, preserving fields the editor doesn't
    manage (weather sets, hooks, startup_*, etc.)

Planned for follow-up:
  * Drag polyline endpoints/midpoints
  * Insert/delete polyline vertices
  * Undo/redo via QUndoStack
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QPointF, QRectF, QTimer,
    QProcess, QProcessEnvironment, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction, QBrush, QColor, QFont, QPainter, QPen, QPainterPath,
    QImage, QPixmap, QKeySequence, QShortcut, QUndoCommand, QUndoStack,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QTableView, QPushButton, QComboBox, QLabel, QToolBar,
    QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsSimpleTextItem,
    QGraphicsPixmapItem,
    QHeaderView, QStyledItemDelegate, QStatusBar, QFrame, QPlainTextEdit,
    QMenu, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QSpinBox,
    QLineEdit,
)

from core.project import list_projects
from lib.emulator_broadcaster import (
    decode_message, STAGE_RAW, STAGE_CORRECTED,
)


# ---------------------------------------------------------------------------
# Emulator client: TCP receive thread + thread-safe latest-frame caches
# ---------------------------------------------------------------------------

class EmulatorClient:
    """Reads length-prefixed frames from the engine's localhost TCP feed
    and parks the most recent ``raw`` and ``corrected`` frame per group
    in two dicts. Designed for a Qt timer to poll without blocking.

    The connect attempt is retried in the background until the engine's
    listener is up, so the editor can press Launch then connect — the
    engine will be a few hundred ms behind the editor's connect attempts.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = int(port)
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._raw: dict[str, np.ndarray] = {}
        self._corrected: dict[str, np.ndarray] = {}
        self._last_seq: int = 0
        self._connected = False

    # ----- lifecycle -----
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="EmulatorClient", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ----- snapshot accessors (called from Qt thread) -----
    def is_connected(self) -> bool:
        return self._connected

    def snapshot(self) -> tuple[dict[str, np.ndarray],
                                dict[str, np.ndarray], int]:
        """Return (raw_frames, corrected_frames, last_seq). Returned
        arrays are *not* copied — treat them as read-only. Safe to call
        from the Qt thread; the receive thread only ever replaces the
        whole dict entry under the lock."""
        with self._lock:
            return dict(self._raw), dict(self._corrected), self._last_seq

    # ----- internals -----
    def _run(self):
        while not self._stop.is_set():
            if not self._connect_with_retry():
                return
            try:
                self._read_loop()
            except OSError:
                pass
            self._connected = False
            with self._lock:
                self._raw.clear()
                self._corrected.clear()

    def _connect_with_retry(self) -> bool:
        # Keep retrying every ``delay`` seconds until either the
        # connect succeeds or stop() is called. The earlier 6-second
        # deadline gave up too fast on cold Python launches (imports
        # + GL context + mDNS can easily run 8–10 s before the
        # engine's emulator broadcaster binds), and once the deadline
        # was spent the client thread exited permanently, leaving the
        # editor's status stuck on "Engine: starting…" with no
        # recovery path. An indefinite retry instead just waits.
        delay = 0.2
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect((self.host, self.port))
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                self._connected = True
                return True
            except OSError:
                try:
                    s.close()
                except OSError:
                    pass
                self._stop.wait(delay)
        return False

    def _read_exact(self, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _read_loop(self):
        while not self._stop.is_set():
            hdr = self._read_exact(4)
            if hdr is None:
                return
            (msg_len,) = struct.unpack(">L", hdr)
            payload = self._read_exact(msg_len)
            if payload is None:
                return
            msg = decode_message(payload)
            if msg is None:
                continue
            with self._lock:
                if msg["stage"] == STAGE_RAW:
                    self._raw[msg["group_id"]] = msg["frame"]
                elif msg["stage"] == STAGE_CORRECTED:
                    self._corrected[msg["group_id"]] = msg["frame"]
                self._last_seq = msg["seq"]


# ---------------------------------------------------------------------------
# Emulator overlay: per-LED canvas positions + FBO sample indices,
# numpy-painted QImage updates per frame.
# ---------------------------------------------------------------------------

class _ArcParamDialog(QDialog):
    """Edit the (center, radius, start_deg, end_deg, segments) of an
    arc-shaped strip. Live-updates the strip's polyline preview when
    the dialog's Apply button is hit."""

    def __init__(self, strip, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Arc parameters")
        layout = QFormLayout(self)
        self._cx = QSpinBox(); self._cx.setRange(-99999, 99999)
        self._cy = QSpinBox(); self._cy.setRange(-99999, 99999)
        self._cx.setValue(int(strip.arc_center[0]))
        self._cy.setValue(int(strip.arc_center[1]))
        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.1, 99999.0)
        self._radius.setDecimals(1)
        self._radius.setValue(float(strip.arc_radius))
        self._start = QDoubleSpinBox()
        self._start.setRange(-720.0, 720.0)
        self._start.setDecimals(1)
        self._start.setValue(float(strip.arc_start_deg))
        self._end = QDoubleSpinBox()
        self._end.setRange(-720.0, 720.0)
        self._end.setDecimals(1)
        self._end.setValue(float(strip.arc_end_deg))
        self._segments = QSpinBox()
        self._segments.setRange(2, 256)
        self._segments.setValue(int(strip.arc_segments))
        layout.addRow("Center X (canvas px)", self._cx)
        layout.addRow("Center Y (canvas px)", self._cy)
        layout.addRow("Radius (px)", self._radius)
        layout.addRow("Start angle (° CCW from +x)", self._start)
        layout.addRow("End angle (° CCW from +x)", self._end)
        layout.addRow("Polyline segments", self._segments)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def apply_to(self, strip):
        strip.arc_center = (self._cx.value(), self._cy.value())
        strip.arc_radius = float(self._radius.value())
        strip.arc_start_deg = float(self._start.value())
        strip.arc_end_deg = float(self._end.value())
        strip.arc_segments = int(self._segments.value())


def _arc_to_polyline(center: tuple, radius: float,
                     start_deg: float, end_deg: float,
                     segments: int) -> list:
    """Sample a circular arc into an ``(segments+1)``-vertex polyline
    (canvas coords). Angles are degrees, measured CCW from +x. The
    polyline runs from ``start_deg`` toward ``end_deg`` so direction
    is preserved (a reversed angle range yields a reversed walk)."""
    import math
    cx, cy = center
    n = max(int(segments), 1)
    pts = []
    for i in range(n + 1):
        t = i / n
        ang_deg = start_deg + t * (end_deg - start_deg)
        ang = math.radians(ang_deg)
        # Canvas y grows down — negate sin so positive angle goes up
        # in screen-correct orientation (matches the Fan polyline
        # derivation in load_doc).
        x = cx + radius * math.cos(ang)
        y = cy - radius * math.sin(ang)
        pts.append([int(round(x)), int(round(y))])
    return pts


def _resample_polyline(polyline: list, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` evenly-spaced points along a multi-vertex polyline.
    Returns (xs, ys) as int arrays, in chain order (matching polyline
    order — caller is responsible for orienting it correctly)."""
    if n <= 0 or not polyline:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    if n == 1:
        return (np.array([int(polyline[0][0])], dtype=np.int32),
                np.array([int(polyline[0][1])], dtype=np.int32))
    pts = np.asarray(polyline, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_lens = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1] if cum[-1] > 1e-9 else 1.0
    targets = np.linspace(0.0, total, n)
    seg_idx = np.clip(np.searchsorted(cum, targets, side="right") - 1,
                      0, len(seg_lens) - 1)
    base = cum[seg_idx]
    span = np.maximum(seg_lens[seg_idx], 1e-9)
    t = (targets - base) / span
    xs = pts[seg_idx, 0] + t * seg[seg_idx, 0]
    ys = pts[seg_idx, 1] + t * seg[seg_idx, 1]
    return xs.astype(np.int32), ys.astype(np.int32)


def _strip_fbo_indices(s: 'StripSpec') -> tuple[np.ndarray, np.ndarray]:
    """Per-LED (rows, cols) indices into the strip's group canvas, in
    chain order. Mirrors core/strip.py's index logic so the editor
    emulator samples the same pixels the runtime would extract."""
    L = max(int(s.length), 0)
    if L == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    start = int(s.start)
    if s.kind == "row":
        if s.direction == "right":
            cols = np.arange(start, start + L, dtype=np.int32)
        else:  # left
            cols = np.arange(start, start - L, -1, dtype=np.int32)
        rows = np.full(L, int(s.row), dtype=np.int32)
    elif s.kind == "column":
        if s.direction == "down":
            rows = np.arange(start, start - L, -1, dtype=np.int32)
        else:  # up
            rows = np.arange(start, start + L, dtype=np.int32)
        cols = np.full(L, int(s.col), dtype=np.int32)
    else:  # raw — polyline carries indices directly; no fbo sampling
        rows = np.zeros(L, dtype=np.int32)
        cols = np.zeros(L, dtype=np.int32)
    return rows, cols


class EmulatorOverlay:
    """Drives the live LED preview on top of an existing LayoutScene
    (physical canvas) and GroupCanvasScene (group canvases).

    Setup phase: pre-compute per-strip arrays of (canvas_x, canvas_y)
    LED positions and (fbo_row, fbo_col) sample indices. Allocates two
    QGraphicsPixmapItems that overlay each scene's existing items.

    Tick phase: snapshot the EmulatorClient's current frames; for each
    strip sample its LEDs from the corrected frame and paint them into
    the physical-canvas image; for each group blit the raw frame into
    the group-canvas image (nearest-neighbour upscaled to display size).
    """

    # Half-width of each LED dot on the physical-canvas image, in
    # canvas pixels. 0 = single pixel; 1 = 3×3 ish; 2 = 5×5 ish.
    # Implemented via a single per-frame cv2.dilate after writing
    # one pixel per LED — far cheaper than expanding stamps into
    # the index arrays (which costs ~10× more pixel writes per
    # tick on Fan's 40k-LED canvas).
    LED_RADIUS = 2

    # Perceptual gamma applied to the on-screen physical-canvas
    # blit. The corrected frame coming over the wire is already
    # gamma-pre-corrected for the LEDs (~1/2.0) AND scaled down by
    # the engine's brightness_limit (0.2 on WoL), so a shader's
    # 10 %-bright output lands at ~16/255 going to the strip. LEDs
    # at 16/255 are visibly lit on hardware (LED response is
    # non-linear + human eyes are logarithmic), but a monitor pixel
    # at 16/255 is virtually black. This power < 1 boosts the dim
    # region disproportionately on screen so the preview matches
    # what the eye sees from the strip rather than what the byte
    # value is. 0.45 ≈ inverse of a typical 2.2 display gamma.
    DISPLAY_GAMMA = 0.45

    def __init__(self, doc: 'LayoutDoc',
                 phys_scene: 'LayoutScene',
                 group_scene: 'GroupCanvasScene'):
        self.doc = doc
        self.phys_scene = phys_scene
        self.group_scene = group_scene

        # Per-strip precomputed sampling tables (chain-ordered).
        # Each list entry is (group_id, fbo_rows, fbo_cols, canvas_xs,
        # canvas_ys). Strips with no length get empty arrays and are
        # silently skipped each tick.
        self._strip_tables: list[tuple] = []
        self._build_strip_tables()

        # Physical-canvas overlay: one numpy buffer + one QPixmapItem.
        cw = max(int(doc.canvas_w), 1)
        ch = max(int(doc.canvas_h), 1)
        self._phys_buf = np.zeros((ch, cw, 3), dtype=np.uint8)
        self._phys_pixmap_item = QGraphicsPixmapItem()
        self._phys_pixmap_item.setZValue(7)   # above strips (z=5), below handles (z=20)
        # Disable smoothing so single-pixel LEDs stay crisp on zoom.
        self._phys_pixmap_item.setTransformationMode(
            Qt.TransformationMode.FastTransformation
        )
        phys_scene.addItem(self._phys_pixmap_item)

        # Group-canvas overlays: one QPixmapItem per group, sized to the
        # group's display rect. Rebuild whenever the doc/canvas changes.
        self._group_pixmap_items: dict[str, QGraphicsPixmapItem] = {}
        self._allocate_group_pixmaps()

        # Visibility starts hidden; show() is called when the emulator
        # is connected and we have at least one frame in hand.
        self.set_visible(False)

    # ----- precompute -----
    def _build_strip_tables(self):
        # Cache canvas dimensions so we can pre-clip canvas-side
        # destinations once instead of per tick.
        ch = max(int(self.doc.canvas_h), 1)
        cw = max(int(self.doc.canvas_w), 1)
        group_dims = {g.id: (g.height, g.width) for g in self.doc.groups}

        for s in self.doc.strips:
            L = max(int(s.length), 0)
            if L == 0:
                self._strip_tables.append((s.group, None, None, None, None))
                continue
            rows, cols = _strip_fbo_indices(s)
            if s.polyline and len(s.polyline) >= 2:
                xs, ys = _resample_polyline(s.polyline, L)
            elif s.kind in ("row", "column"):
                if s.kind == "row":
                    a_start = int(s.start)
                    step = 1 if s.direction == "right" else -1
                    xs = np.arange(L, dtype=np.int32) * step + a_start
                    ys = np.full(L, int(s.row), dtype=np.int32)
                else:
                    a_start = int(s.start)
                    step = -1 if s.direction == "down" else 1
                    ys = np.arange(L, dtype=np.int32) * step + a_start
                    xs = np.full(L, int(s.col), dtype=np.int32)
            else:
                xs = np.zeros(L, dtype=np.int32)
                ys = np.zeros(L, dtype=np.int32)

            # Clip everything once at setup so the per-tick path is
            # pure indexing — no boolean masks, no bounds compares.
            gh, gw = group_dims.get(s.group, (0, 0))
            ok = (
                (rows >= 0) & (rows < gh) & (cols >= 0) & (cols < gw)
                & (xs >= 0) & (xs < cw) & (ys >= 0) & (ys < ch)
            )
            if not ok.all():
                rows = rows[ok]; cols = cols[ok]
                xs = xs[ok]; ys = ys[ok]
            self._strip_tables.append((s.group, rows, cols, xs, ys))

        # Pre-build the morphological kernel for the per-frame dilate
        # that fattens single-pixel LED writes into visible dots. A
        # circular kernel reads more like an LED than the default
        # square block. Lazily import cv2 so the editor still loads
        # if OpenCV is unavailable (in which case dilate is skipped).
        self._dilate_kernel = None
        if self.LED_RADIUS > 0:
            try:
                import cv2
                size = 2 * self.LED_RADIUS + 1
                self._dilate_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (size, size)
                )
            except ImportError:
                pass

        # Display-gamma LUT: 256-entry map from byte → boosted byte
        # so the per-tick path is one fancy-indexed read, no float
        # math. ``DISPLAY_GAMMA`` is set < 1 to brighten the dim end
        # without crushing the bright end. cv2.LUT (when available)
        # is faster than numpy fancy indexing on big buffers.
        x = np.arange(256, dtype=np.float32) / 255.0
        boosted = np.clip(np.power(x, self.DISPLAY_GAMMA), 0.0, 1.0)
        self._gamma_lut = (boosted * 255.0).astype(np.uint8)

    def _allocate_group_pixmaps(self):
        # Place one pixmap item per group at its origin in the group
        # scene. Sized to the display rect; nearest-neighbour scaling
        # so the FBO grid is visible.
        for gid, origin in self.group_scene._group_origin.items():
            ox, oy, sx, sy = origin
            g = next((g for g in self.doc.groups if g.id == gid), None)
            if g is None or g.width <= 0 or g.height <= 0:
                continue
            item = QGraphicsPixmapItem()
            item.setPos(ox, oy)
            # We'll set the pixmap each tick; pre-set a transform so the
            # native-resolution pixmap fills the group's display rect.
            item.setTransform(item.transform().scale(sx, sy))
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            item.setZValue(2)   # above the group canvas rect, below strip lines
            self.group_scene.addItem(item)
            self._group_pixmap_items[gid] = item

    # ----- per-frame -----
    def update_frame(self,
                     raw_frames: dict[str, np.ndarray],
                     corrected_frames: dict[str, np.ndarray]) -> None:
        # Physical canvas: paint LEDs from corrected frames. Hot path
        # — runs every emulator tick on the GUI thread. Bounds were
        # pre-clipped in _build_strip_tables, so per-tick work is just
        # fancy-indexed reads from the source frame and writes to the
        # destination buffer.
        if corrected_frames:
            self._phys_buf.fill(0)
            for (gid, rows, cols, xs, ys) in self._strip_tables:
                if rows is None or rows.size == 0:
                    continue
                frame = corrected_frames.get(gid)
                if frame is None:
                    continue
                self._phys_buf[ys, xs] = frame[rows, cols]
            # Fatten each single-pixel LED into a visible dot via a
            # single SIMD-accelerated dilate. Far cheaper than
            # expanding every LED into a stamp during the index
            # writes (~1 ms for a 1024×600 buffer vs ~35 ms for
            # 25×-expanded writes on Fan).
            if self._dilate_kernel is not None:
                import cv2
                cv2.dilate(self._phys_buf, self._dilate_kernel,
                           dst=self._phys_buf)
            # Display-gamma boost. Done AFTER dilate so the dilated
            # neighbourhood inherits the LUT-mapped brightness and
            # the preview matches what the eye sees from a real LED
            # rather than the literal byte value going to the strip.
            # Applied via a 256-entry LUT — single indexed read per
            # pixel, well under a millisecond on Fan-sized canvases.
            try:
                import cv2
                disp = cv2.LUT(self._phys_buf, self._gamma_lut)
            except ImportError:
                disp = self._gamma_lut[self._phys_buf]
            self._phys_pixmap_item.setPixmap(_pixmap_from_rgb(disp))

        # Group canvases: blit the raw FBO directly.
        for gid, item in self._group_pixmap_items.items():
            frame = raw_frames.get(gid)
            if frame is None:
                continue
            item.setPixmap(_pixmap_from_rgb(frame))

    def set_visible(self, on: bool):
        self._phys_pixmap_item.setVisible(on)
        for item in self._group_pixmap_items.values():
            item.setVisible(on)

    def teardown(self):
        # Items may already be C++-deleted if the scene was cleared
        # underneath us (e.g. a doc edit triggered scene.rebuild()).
        # In that case ``.scene()`` raises RuntimeError; treat it as a
        # successful no-op cleanup.
        try:
            if self._phys_pixmap_item.scene() is self.phys_scene:
                self.phys_scene.removeItem(self._phys_pixmap_item)
        except RuntimeError:
            pass
        for item in list(self._group_pixmap_items.values()):
            try:
                if item.scene() is self.group_scene:
                    self.group_scene.removeItem(item)
            except RuntimeError:
                pass
        self._group_pixmap_items.clear()


def _pixmap_from_rgb(arr: np.ndarray) -> QPixmap:
    """Wrap a (H, W, 3) uint8 RGB numpy array as a QPixmap. Copies the
    bytes into the QImage's lifetime (don't hold the array reference)."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img)


# ---------------------------------------------------------------------------
# Data model: typed dataclasses + load/save round-trip
# ---------------------------------------------------------------------------

@dataclass
class BoxSpec:
    """One physical hardware unit + its spatial position.

    A "box" merges what the underlying YAML stores as two separate things:

      * ``geometry.yaml``'s ``objects`` entry  — gives the box a name and
        an (x, y) position on the composite preview canvas. Only present
        for ``multi_object`` projects (Fan has no objects, just receivers).
      * ``project.yaml``'s ``receivers`` entry — gives the box a network
        address (``ip`` and/or ``host``) and a protocol. Every project
        has receivers; ``object_id`` cross-references a matching object
        in geometry.yaml when one exists.

    The editor presents these as a single row because in practice they
    pair 1:1 (each sculpture has one driver box). When they don't pair
    (Fan: receivers without objects) the spatial fields just stay zero.
    """
    object_id: int = 0           # link to the matching geometry.yaml object
    name: str = ""               # display name (empty for receiver-only rows)
    x: int = 0                   # composite-canvas position; 0 when no object
    y: int = 0
    ip: Optional[str] = None
    host: Optional[str] = None
    protocol: str = "ddp"

    def label(self) -> str:
        addr = self.host or self.ip or "—"
        if self.name:
            return f"{self.name} ({addr})"
        if self.object_id >= 0:
            return f"obj{self.object_id}: {addr}"
        return addr


@dataclass
class GroupSpec:
    id: str
    width: int
    height: int


@dataclass
class StripSpec:
    """One LED strip — its slot in the wire stream and its footprint on
    a group canvas.

    Three independent indices keep the wire side and the canvas side
    decoupled. They were conflated in an earlier revision; on Fan,
    each receiver counts ``strip_idx`` 0..31 *locally* while ``col``
    spans 0..127 *globally*, so they must stay separate.

      * ``strip_idx`` — wire-order index within this strip's receiver
        (which DMX universe slot, 0-indexed per receiver).
      * ``row`` (kind=row only) — canvas row this strip renders from.
        Defaults to ``strip_idx`` (matches the multi_object convention
        where strip_idx == row index per group).
      * ``col`` (kind=column only) — canvas column this strip renders
        from. Required by the YAML schema; on Fan this is the strip's
        global column 0..127 across all four receivers.
      * ``start`` — canvas-axis position of LED 0 (the strip's
        data-in end). The chain walks ``length`` cells from there in
        the chosen ``direction``. Direction-aware defaults keep
        un-shifted strips occupying [0, length-1]:
            direction=right or up → default start = 0
            direction=left  or down → default start = length - 1
        These defaults match Fan's existing YAML which omits ``start``.
      * ``length`` — number of LEDs.
      * ``direction`` — right/left for row, down/up for column. The
        chain walks from ``start`` toward the opposite end of the axis.

    ``polyline`` is populated for kind=raw and for multi_object row
    strips (from geometry.yaml); for Fan-style column strips it's
    derived at display time from FanGeometry rather than stored.
    """
    group: str
    strip_idx: int = 0           # wire order within receiver
    kind: str = "row"            # row | column | raw
    row: int = 0                 # canvas row (kind=row); defaults to strip_idx
    col: int = 0                 # canvas column (kind=column); required by schema
    start: int = 0               # offset along the canvas axis
    length: int = 0
    direction: str = "right"
    polyline: list = field(default_factory=list)
    receiver_idx: int = -1
    # Optional shape hint used by the editor to drive interactive
    # editing of multi_object polylines. ``"polyline"`` (default) =
    # arbitrary N-vertex path, edited by dragging per-vertex handles.
    # ``"arc"`` = polyline derived from (center, radius, start_deg,
    # end_deg, segments); editing the arc parameters regenerates the
    # polyline. The runtime never reads these — at load/save time the
    # editor regenerates the polyline from the arc params, and writes
    # the polyline to YAML alongside the params for round-trip safety.
    shape: str = "polyline"
    arc_center: tuple = (0, 0)        # (x, y) canvas coords
    arc_radius: float = 0.0
    arc_start_deg: float = 0.0
    arc_end_deg: float = 0.0
    arc_segments: int = 16            # polyline resolution

    @property
    def canvas_pos(self) -> int:
        """Index of this strip in its group canvas — row for row-kind,
        col for column-kind. ``-1`` for kind=raw (no single-axis index)."""
        if self.kind == "row":
            return self.row
        if self.kind == "column":
            return self.col
        return -1

    def set_canvas_pos(self, value: int) -> None:
        if self.kind == "row":
            self.row = value
        elif self.kind == "column":
            self.col = value

    def default_start(self) -> int:
        """Direction-aware default for ``start`` (LED 0's canvas-axis
        position). Used both as a fill-in when the YAML omits ``start``
        and as the threshold for "is this value at the default?" when
        deciding whether to emit ``start`` in save_doc."""
        if self.direction in ("down", "left"):
            return max(int(self.length) - 1, 0)
        return 0   # up, right

    def chain_end_pos(self) -> int:
        """Canvas-axis position of the LAST LED in the chain (the
        opposite end from ``start``)."""
        L = max(int(self.length) - 1, 0)
        if self.direction in ("down", "left"):
            return self.start - L
        return self.start + L

    def axis_range(self) -> tuple[int, int]:
        """(low, high) canvas-axis indices the strip occupies.
        Independent of direction — useful for drawing the bounding
        line on the canvas."""
        a = self.start
        b = self.chain_end_pos()
        return (min(a, b), max(a, b))


@dataclass
class LayoutDoc:
    """In-memory representation of a project's layout. Round-trips with
    the YAML files via ``load_doc`` / ``save_doc``. Keeps the original
    raw dicts so non-editor-managed fields (weather sets, hooks, etc.)
    survive a save.

    ``geometry_type`` is the active project's ``geometry.type``
    (``"fan"``, ``"multi_object"``, etc.). The editor uses this to
    decide whether to round-trip ``geometry.yaml`` (multi_object) and
    which canvas dimensions to display.
    """
    project_id: str
    geometry_type: str = "fan"
    canvas_w: int = 1024
    canvas_h: int = 768
    groups: List[GroupSpec] = field(default_factory=list)
    strips: List[StripSpec] = field(default_factory=list)
    boxes: List[BoxSpec] = field(default_factory=list)
    _raw_project: dict = field(default_factory=dict)
    _raw_geometry: dict = field(default_factory=dict)
    _raw_geometry_path: Optional[Path] = None


def find_editable_projects() -> list[dict]:
    """Return [{id, display_name, geometry_type}] for every project with
    a project.yaml. The editor adapts to whatever ``geometry.type`` each
    one declares; multi_object projects get full polyline editing while
    fan-style projects fall back to a group-canvas view of column strips.
    """
    out = []
    for p in list_projects():
        try:
            with open(ROOT / "projects" / p["id"] / "project.yaml", "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            gtype = (raw.get("geometry") or {}).get("type", "fan")
            entry = dict(p)
            entry["geometry_type"] = gtype
            out.append(entry)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# New-project scaffolding — invoked by the "+ New" button in the toolbar
# ---------------------------------------------------------------------------

import re as _re

# Project-id slug: lowercase letters + digits + underscore, must start
# with a letter. Used as the directory name AND as a Python package
# component, so it has to be import-safe.
_SLUG_RE = _re.compile(r"^[a-z][a-z0-9_]*$")


def _slug_problem(slug: str, existing_ids: set[str]) -> str | None:
    """Return None if slug is valid + unused; otherwise a one-line
    explanation suitable for showing under the input field."""
    if not slug:
        return "Project id is required."
    if not _SLUG_RE.match(slug):
        return ("Use lowercase letters, digits, and underscores. "
                "Must start with a letter.")
    if slug in existing_ids:
        return f"A project with id {slug!r} already exists."
    return None


# Skeleton file templates — kept here (next to the only callsite) so
# the editor is self-contained. If a project file's shape changes
# across the codebase, this is the only place that mirrors it; the
# scaffold is intentionally minimal so subsequent customisation is
# obvious. Format strings use ``{id}`` / ``{name}`` / ``{cw}`` /
# ``{ch}`` placeholders; ``str.format`` is fine because none of the
# templates need raw braces.

_INIT_PY = ""

_WEATHER_PARAMS_PY = '''"""{name} — weather machinery.

A fresh project starts with a single CLEAR state and a single
"default" weather set. The scaffolded state and set both include the
fields needed to enable per-state ambient sounds, per-set narratives,
and per-set random sound pools — leave the values as ``None`` until
you drop matching media into ``projects/{id}/media/sounds/`` and
author the corresponding scripts / pool directories.

Author shaders in ``projects/{id}/shaders/`` and register them in
``event_map.py``, then add their names to ``background_events`` below
to make them fire when this set is active. ``narrative_player`` and
``sound_pool`` are auto-inherited from ``core.default_events`` —
adding them to ``background_events`` and giving the set a
``narrative_script`` / ``sound_pool_dir`` is enough.
"""
from enum import Enum

# Schema bits are shared across every project.
from lib.weather_params import (  # noqa: F401  explicit re-export
    DEFAULT_WEATHER_PARAMS,
    PARAMETER_DEFINITIONS,
    GLOBAL_PARAMETERS,
    AVAILABLE_BACKGROUND_EVENTS,
)


class WeatherState(Enum):
    """States this project uses. Add new members here and a matching
    preset below, then list the state's value in a WEATHER_SETS entry
    to make it reachable at runtime."""
    CLEAR = "clear"


WEATHER_PRESETS = {{
    WeatherState.CLEAR: {{
        "Switch_rate": 0.0,             # never auto-transition
        "transition_duration": 1.0,
        "possible_transitions": ["clear"],
        "transition_weights": [1.0],
        # Per-state ambient background sound. ``None`` keeps silent;
        # set to a filename under ``projects/{id}/media/sounds/`` to
        # surface a picker in the weather editor.
        "ambient_sound": None,
    }},
}}


WEATHER_SETS = {{
    "default": {{
        "name": "Default",
        "description": "Starter set — add background_events as you author them.",
        "states": [WeatherState.CLEAR.value],
        "season_speed": 0.0,
        "transition_speed": 1.0,
        "season_extremity": 0.0,
        "allowed_parameters": [],
        "random_events": [],
        "random_event_rate": 0.0,
        # Set-level audio hooks. Both are inert unless the matching
        # event is also listed in ``background_events``:
        #   - ``narrative_script``: path (relative to
        #     ``projects/{id}/media/``) to a ``script.json`` authored
        #     in the narrative editor. Pair with
        #     ``"narrative_player"`` in ``background_events``.
        #   - ``sound_pool_dir``: directory (relative to
        #     ``projects/{id}/media/``) of clips for the random
        #     ambient sound pool. Pair with ``"sound_pool"`` in
        #     ``background_events``.
        #   - ``sound_pool_crossfade``: seconds of overlap between
        #     pool clips. 0 = clips play one at a time with a gap
        #     (default); above 0 = gapless stream, each clip
        #     crossfading into the next.
        "narrative_script": None,
        "sound_pool_dir": None,
        "sound_pool_crossfade": 0.0,
        "background_events": [],
    }},
}}


DEFAULT_WEATHER_SET = "default"
'''

_EVENT_MAP_PY = '''"""{name} — event registry.

Each entry: (effect_func, params_dict, meta_dict).

  * ``effect_func`` is a ``shader_*`` function from either the shared
    library (``renderer.effects.shader_*``) or this project's local
    ``projects/{id}/shaders/`` directory (auto-imported into ``fx``
    by ``core.shader_loader`` at project boot).
  * ``params_dict`` is passed as keyword args to the effect function.
  * ``meta_dict`` carries ``{{"group": "<group_id>"}}`` so the
    scheduler dispatches the event onto the right canvas. Omit it
    (or use ``{{}}``) to fall through to frame_id 0.

Example::

    "my_glow": (fx.shader_my_glow, {{"speed": 0.5}}, {{"group": "main"}}),

Note: ``narrative_player`` and ``sound_pool`` are auto-inherited from
``core.default_events.DEFAULT_EVENT_MAP`` — you do NOT need to
re-register them here. Just add their names to a weather set's
``background_events`` list. Override only if this project needs
different params (e.g. a non-default ``node_delay``).
"""
from renderer import effects as fx  # noqa: F401  used by EVENT_MAP entries


EVENT_MAP = {{
    # Add your project-specific events here. Universal events
    # (narrative_player, sound_pool) come from core.default_events
    # automatically.
}}
'''

_SHADERS_INIT_PY = '''"""{name} project-local shaders.

Any ``shader_*`` function or ``*Effect`` class defined in a module in
this directory is auto-imported into ``renderer.effects`` when this
project is active (see ``core.shader_loader``). Reference them in
``event_map.py`` as ``fx.shader_<your_name>``.
"""
'''

_GEOMETRY_YAML_MULTI = """canvas:
  width: {cw}
  height: {ch}
objects: []
strips: []
"""


def _project_yaml_template(spec: dict) -> dict:
    """Build the project.yaml dict for a new project. Returned as a
    dict so the caller can dump with the same yaml settings as the
    rest of the editor uses."""
    pid = spec["id"]
    name = spec["display_name"]
    geometry_type = spec["geometry_type"]

    out: dict = {
        "id": pid,
        "display_name": name,
        "brightness_limit": 0.4,
        "target_fps": 40,
        "weather_sets_module": f"projects.{pid}.weather_params",
        "event_map_module":   f"projects.{pid}.event_map",
        # One starter group; operator adds more in the editor or by
        # editing this file. Multi_object projects typically end up
        # with 2-3 groups (sky / ground / etc.).
        "groups": [
            {"id": "main", "width": 300, "height": 1},
        ],
        # Off by default — the random-events hook is opt-in per
        # project. Flip to true and declare ``hooks.random_events``
        # below when authoring probability-driven scheduling.
        "enable_random_events": False,
    }
    if geometry_type == "multi_object":
        out["geometry"] = {
            "type": "multi_object",
            "file": f"projects/{pid}/geometry.yaml",
        }
    else:
        # Fan-style — sensible defaults for a small dome / fan canvas.
        # Operator overrides via the geometry block in project.yaml.
        out["geometry"] = {
            "type": "fan",
            "inner_r_ft": 4.0,
            "outer_r_ft": 20.6,
        }
    # No receivers configured yet — operator adds them via this editor
    # or by editing project.yaml directly. The empty list keeps the
    # engine's startup happy (no hardware to send to, but the boot
    # path doesn't blow up).
    out["receivers"] = []
    return out


def scaffold_new_project(spec: dict) -> Path:
    """Write a fresh project skeleton to ``projects/<id>/`` and return
    the new project's root path. ``spec`` is the dict produced by
    ``_NewProjectDialog.result_spec()``.

    Files written:
      * ``__init__.py`` — empty (Python package marker)
      * ``project.yaml`` — manifest with one starter group + empty receivers
      * ``weather_params.py`` — single CLEAR state + "default" empty set
      * ``event_map.py`` — empty ``EVENT_MAP`` with usage docstring
      * ``shaders/__init__.py`` — project-local shader package
      * ``geometry.yaml`` — only for ``geometry_type == "multi_object"``
      * ``media/sounds/`` — empty directory for per-project audio

    Raises FileExistsError if ``projects/<id>/`` already exists; the
    caller (the dialog) validates against this with ``_slug_problem``
    before invoking, but the guard here is a belt-and-suspenders
    against a TOCTOU race.
    """
    pid = spec["id"]
    name = spec["display_name"]
    geometry_type = spec["geometry_type"]
    cw = int(spec.get("canvas_w", 1280))
    ch = int(spec.get("canvas_h", 1024))

    root = ROOT / "projects" / pid
    if root.exists():
        raise FileExistsError(f"Project directory already exists: {root}")

    # Create all directories first so write failures don't leave a
    # half-built project (the subsequent file writes still might,
    # but at least the layout is consistent).
    root.mkdir(parents=True, exist_ok=False)
    (root / "shaders").mkdir(exist_ok=False)
    (root / "media" / "sounds").mkdir(parents=True, exist_ok=False)
    (root / "media" / "images").mkdir(parents=True, exist_ok=False)

    # __init__.py
    (root / "__init__.py").write_text(_INIT_PY, encoding="utf-8")
    (root / "shaders" / "__init__.py").write_text(
        _SHADERS_INIT_PY.format(name=name), encoding="utf-8"
    )

    # project.yaml — use the same yaml settings the rest of the editor
    # uses (sort_keys=False, flow style off) so re-saves stay diff-clean.
    with open(root / "project.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_project_yaml_template(spec), f,
                       sort_keys=False, default_flow_style=False)

    # weather_params.py + event_map.py
    (root / "weather_params.py").write_text(
        _WEATHER_PARAMS_PY.format(id=pid, name=name), encoding="utf-8"
    )
    (root / "event_map.py").write_text(
        _EVENT_MAP_PY.format(id=pid, name=name), encoding="utf-8"
    )

    # geometry.yaml only when the project declared multi_object
    if geometry_type == "multi_object":
        (root / "geometry.yaml").write_text(
            _GEOMETRY_YAML_MULTI.format(cw=cw, ch=ch), encoding="utf-8"
        )

    return root


class _NewProjectDialog(QDialog):
    """Modal dialog for the layout-editor's "+ New Project" button.

    Asks for project id (slug, live-validated against existing ids),
    display name, geometry type (fan vs multi_object), and — for
    multi_object only — initial canvas dimensions. The Create button
    stays disabled until the slug passes validation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self._existing_ids = {p["id"] for p in find_editable_projects()}

        layout = QFormLayout(self)

        # Project id — slug, live-validated.
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("e.g. glimmering_shell")
        self._id_edit.textChanged.connect(self._on_id_changed)
        layout.addRow("Project id", self._id_edit)

        # Live status under the id field — green check vs red explanation.
        self._id_status = QLabel("Project id is required.")
        self._id_status.setStyleSheet("color: #cc6666;")
        layout.addRow("", self._id_status)

        # Display name (free text).
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Glimmering Shell")
        layout.addRow("Display name", self._name_edit)

        # Canvas size. New projects are always multi_object — the only
        # other geometry today (``fan``) is tied to one specific physical
        # piece, so there's no meaningful choice to surface.
        self._canvas_w = QSpinBox()
        self._canvas_w.setRange(1, 16384)
        self._canvas_w.setValue(1280)
        self._canvas_h = QSpinBox()
        self._canvas_h.setRange(1, 16384)
        self._canvas_h.setValue(1024)
        size_row = QHBoxLayout()
        size_row.addWidget(self._canvas_w)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self._canvas_h)
        size_row.addStretch(1)
        size_widget = QWidget()
        size_widget.setLayout(size_row)
        layout.addRow("Canvas size (px)", size_widget)

        # OK / Cancel.
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok).setText("Create")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addRow(self._buttons)

        # Disable Create until slug is valid.
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def _on_id_changed(self, text: str) -> None:
        # Lowercase-normalize as the user types so they don't have to
        # remember the case rule, but only intercept when they typed a
        # capital letter — otherwise the cursor would jump on every
        # keystroke and selection would be lost.
        if text != text.lower():
            cursor = self._id_edit.cursorPosition()
            self._id_edit.blockSignals(True)
            self._id_edit.setText(text.lower())
            self._id_edit.blockSignals(False)
            self._id_edit.setCursorPosition(cursor)
            text = text.lower()
        problem = _slug_problem(text, self._existing_ids)
        if problem is None:
            self._id_status.setText(f"✓ Will create projects/{text}/")
            self._id_status.setStyleSheet("color: #66cc66;")
            self._buttons.button(
                QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self._id_status.setText(problem)
            self._id_status.setStyleSheet("color: #cc6666;")
            self._buttons.button(
                QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def result_spec(self) -> dict:
        """Return the validated form values as a dict. Caller passes
        this to ``scaffold_new_project``."""
        name = self._name_edit.text().strip()
        pid = self._id_edit.text().strip()
        return {
            "id": pid,
            # Display name falls back to a Title-Case version of the
            # slug if the operator left it empty.
            "display_name": name or pid.replace("_", " ").title(),
            "geometry_type": "multi_object",
            "canvas_w": int(self._canvas_w.value()),
            "canvas_h": int(self._canvas_h.value()),
        }


def load_doc(project_id: str) -> LayoutDoc:
    """Load both YAML files for a project into a LayoutDoc.

    The strip list is rebuilt from the receivers' embedded ``strips``
    blocks in ``project.yaml`` (the runtime source of truth — every
    strip belongs to exactly one receiver). For ``multi_object``
    projects, polylines from ``geometry.yaml`` are merged into the
    matching ``(group, strip_idx)`` entries; other geometry types
    (``fan``) skip the merge since there's no per-strip polyline data.
    """
    proj_dir = ROOT / "projects" / project_id
    with open(proj_dir / "project.yaml", "r", encoding="utf-8") as f:
        raw_p = yaml.safe_load(f) or {}
    geo_block = raw_p.get("geometry") or {}
    geometry_type = str(geo_block.get("type", "fan"))
    geo_file_rel = geo_block.get("file")
    geo_path = (ROOT / geo_file_rel) if geo_file_rel else (proj_dir / "geometry.yaml")
    raw_g: dict = {}
    if geometry_type == "multi_object" and geo_path.exists():
        with open(geo_path, "r", encoding="utf-8") as f:
            raw_g = yaml.safe_load(f) or {}

    doc = LayoutDoc(
        project_id=project_id,
        geometry_type=geometry_type,
        _raw_project=raw_p,
        _raw_geometry=raw_g,
        _raw_geometry_path=geo_path,
    )

    # Groups (always from project.yaml). Default to a single 'main' group
    # sized from the legacy display block if none declared, matching
    # core.project.load_project's fallback.
    raw_groups = raw_p.get("groups") or []
    if raw_groups:
        for g in raw_groups:
            doc.groups.append(GroupSpec(
                id=str(g.get("id", "")),
                width=int(g.get("width", 0)),
                height=int(g.get("height", 0)),
            ))
    else:
        disp = raw_p.get("display") or {}
        doc.groups.append(GroupSpec(
            id="main",
            width=int(disp.get("width", 128)),
            height=int(disp.get("height", 300)),
        ))

    # Canvas dimensions for the editor's preview viewport. multi_object
    # projects bring their own composite canvas; otherwise fall back to
    # the first group's surface (Fan: 128 × 300).
    if geometry_type == "multi_object" and raw_g.get("canvas"):
        canvas = raw_g["canvas"]
        doc.canvas_w = int(canvas.get("width", 1024))
        doc.canvas_h = int(canvas.get("height", 768))
    elif doc.groups:
        doc.canvas_w = doc.groups[0].width
        doc.canvas_h = doc.groups[0].height
    else:
        doc.canvas_w, doc.canvas_h = 1024, 768

    # Boxes: merge geometry.yaml objects with project.yaml receivers into
    # one row per physical hardware unit. Receivers drive the iteration
    # order so a strip's receiver_idx (built below) maps directly to the
    # corresponding box index — no remapping needed.
    raw_objects = raw_g.get("objects") or []
    raw_receivers = raw_p.get("receivers") or []
    obj_by_id: dict[int, dict] = {
        int(o["id"]): o for o in raw_objects if isinstance(o, dict) and "id" in o
    }
    seen_object_ids: set[int] = set()
    for rx in raw_receivers:
        if not isinstance(rx, dict):
            continue
        obj_id = int(rx.get("object_id", -1))
        obj = obj_by_id.get(obj_id, {})
        if obj:
            seen_object_ids.add(obj_id)
        doc.boxes.append(BoxSpec(
            object_id=obj_id,
            name=str(obj.get("name", "")),
            x=int(obj.get("x", 0)),
            y=int(obj.get("y", 0)),
            ip=rx.get("ip"),
            host=rx.get("host"),
            protocol=str(rx.get("protocol", "ddp")),
        ))
    # Passive objects (multi_object only): objects in geometry.yaml that have
    # no matching receiver — append them as receiver-less boxes so they
    # still appear on the canvas + are editable.
    for obj_id, obj in obj_by_id.items():
        if obj_id in seen_object_ids:
            continue
        doc.boxes.append(BoxSpec(
            object_id=obj_id,
            name=str(obj.get("name", "")),
            x=int(obj.get("x", 0)),
            y=int(obj.get("y", 0)),
            ip=None,
            host=None,
            protocol="ddp",
        ))

    # Strips: walk every receiver's strips block; create a StripSpec per entry.
    # This is the canonical source of strips — every strip belongs to one
    # receiver. multi_object polylines from geometry.yaml are merged in next.
    for rx_idx, rx in enumerate(raw_receivers):
        if not isinstance(rx, dict):
            continue
        for s in (rx.get("strips") or []):
            kind = str(s.get("kind", "row"))
            strip_idx = int(s.get("strip_idx", 0))
            length = int(s.get("length", 0))
            direction = str(s.get("direction",
                                  "right" if kind == "row" else "down"))
            # Direction-aware ``start`` default mirrors core/strip.py:
            # for chain-reverses-axis directions (down, left), LED 0
            # sits at length-1 by default; for low-to-high directions
            # (up, right), LED 0 sits at 0.
            if "start" in s:
                start = int(s["start"])
            elif direction in ("down", "left"):
                start = max(length - 1, 0)
            else:
                start = 0
            shape = str(s.get("shape", "polyline"))
            arc_center = tuple(s.get("arc_center", (0, 0)) or (0, 0))
            spec = StripSpec(
                group=str(s.get("group", "")),
                strip_idx=strip_idx,
                kind=kind,
                row=int(s.get("row", strip_idx)) if kind == "row" else 0,
                col=int(s.get("col", 0)) if kind == "column" else 0,
                start=start,
                length=length,
                direction=direction,
                polyline=[],
                receiver_idx=rx_idx,
                shape=shape,
                arc_center=arc_center,
                arc_radius=float(s.get("arc_radius", 0.0)),
                arc_start_deg=float(s.get("arc_start_deg", 0.0)),
                arc_end_deg=float(s.get("arc_end_deg", 0.0)),
                arc_segments=int(s.get("arc_segments", 16)),
            )
            doc.strips.append(spec)

    # Fan: synthesize per-strip polylines from FanGeometry so the
    # physical canvas shows the semicircular layout instead of a stack
    # of vertical lines. Polylines are derived (not edited) and replaced
    # whenever the project is reloaded — they're not written back to
    # YAML on save (the FanGeometry parameters are the source of truth).
    if geometry_type == "fan":
        fan_block = geo_block  # {type: fan, ...optional inner_r/outer_r}
        # Canvas size for the editor preview — wide enough to render the
        # full semicircle comfortably. The runtime uses normalized clip
        # space; the editor uses pixels.
        doc.canvas_w = 1024
        doc.canvas_h = 600
        cx, cy = doc.canvas_w / 2, doc.canvas_h - 40   # baseline near bottom
        outer_r_px = min(doc.canvas_w * 0.46, doc.canvas_h * 0.92)
        # Match FanGeometry's defaults: inner = outer * (4ft / 20.6ft)
        # unless the project overrides it.
        outer_r_norm = float(fan_block.get("outer_r", 0.95))
        inner_r_norm = fan_block.get("inner_r")
        if inner_r_norm is None:
            inner_r_norm = outer_r_norm * (4.0 / 20.6)
        inner_r_px = outer_r_px * (inner_r_norm / outer_r_norm)

        # n_cols = canvas width of the (single) main group, since Fan
        # convention is one strip per column on the group canvas.
        if doc.groups:
            n_cols = max(1, doc.groups[0].width)
        else:
            n_cols = 128

        import math as _m
        for spec in doc.strips:
            if spec.kind != "column":
                continue
            i = int(spec.col)
            if not (0 <= i < n_cols):
                continue
            theta = _m.pi - (i / max(n_cols - 1, 1)) * _m.pi
            x_inner = cx + inner_r_px * _m.cos(theta)
            y_inner = cy - inner_r_px * _m.sin(theta)   # canvas y grows down
            x_outer = cx + outer_r_px * _m.cos(theta)
            y_outer = cy - outer_r_px * _m.sin(theta)
            # Chain-align the polyline so polyline[0] is LED 0's position.
            #
            # In the FBO, FanGeometry maps texture v=0 (=GL y=0, the
            # bottom row) to inner_r. BUT the broadcaster sees the
            # frame *after* GroupCanvas.get_frame() does ``np.flipud``,
            # so in the array we receive, **row 0 is the OUTER end and
            # row 299 is the INNER end** — opposite of the bare GL
            # convention. Combine that with core/strip.py:
            #   direction=down → LED 0 at row=length-1 → INNER
            #   direction=up   → LED 0 at row=0        → OUTER
            inner = [int(round(x_inner)), int(round(y_inner))]
            outer = [int(round(x_outer)), int(round(y_outer))]
            if spec.direction == "down":
                spec.polyline = [inner, outer]
            else:   # up
                spec.polyline = [outer, inner]

    # multi_object: merge per-strip polylines from geometry.yaml by (group, strip_idx).
    # Polylines are queued per-key in declared order rather than dict-overwritten:
    # if an earlier buggy save produced duplicate (group, strip_idx) entries (a
    # symptom of per-receiver strip_idx renumbering on multi_object, where
    # strip_idx must be unique per-group), the duplicates are still recoverable
    # so long as geometry.yaml's strip ordering matches the receiver-sorted
    # doc.strips order. Each matching strip pulls the next queued polyline.
    if geometry_type == "multi_object":
        poly_queue: dict[tuple[str, int], list[list]] = {}
        length_queue: dict[tuple[str, int], list[int]] = {}
        for s in (raw_g.get("strips") or []):
            key = (str(s.get("group", "")), int(s.get("strip_idx", 0)))
            poly_queue.setdefault(key, []).append(
                [list(map(int, p)) for p in (s.get("polyline") or [])]
            )
            length_queue.setdefault(key, []).append(int(s.get("length", 0)))
        for spec in doc.strips:
            key = (spec.group, spec.strip_idx)
            q = poly_queue.get(key)
            if q:
                spec.polyline = q.pop(0)
            lq = length_queue.get(key)
            if lq:
                length = lq.pop(0)
                if spec.length == 0 and length:
                    spec.length = length

    # Regenerate arc-shaped polylines from their stored params so the
    # editor canvas reflects the latest arc settings even if the
    # cached polyline in geometry.yaml went stale.
    for spec in doc.strips:
        if spec.shape == "arc" and spec.arc_radius > 0:
            spec.polyline = _arc_to_polyline(
                spec.arc_center, spec.arc_radius,
                spec.arc_start_deg, spec.arc_end_deg, spec.arc_segments,
            )

    return doc


def save_doc(doc: LayoutDoc) -> None:
    """Round-trip the LayoutDoc back to its YAML files.

    For ``multi_object`` projects: writes both ``geometry.yaml`` (canvas
    + objects + strip polylines) AND ``project.yaml`` (groups +
    receivers with strip assignments).

    For other geometry types (e.g. ``fan``): writes only ``project.yaml``
    — Fan-style projects don't carry a per-strip polyline file. The
    original ``geometry`` block in project.yaml is preserved untouched.
    """
    proj_dir = ROOT / "projects" / doc.project_id

    # Split each Box back into its YAML-shaped halves:
    #   * an ``objects`` entry (when the box represents a real positioned
    #     object) → goes into geometry.yaml for multi_object projects
    #   * a ``receivers`` entry (when the box has any hardware addressing)
    #     → goes into project.yaml regardless of geometry type
    objects_yaml = []
    receivers_yaml_proto: list[dict] = []   # entries without strips field; filled in below
    box_to_rx_idx: dict[int, int] = {}      # box_idx -> receiver index in receivers_yaml_proto
    for box_idx, b in enumerate(doc.boxes):
        # The box represents an object whenever it has a name OR a non-zero
        # position OR (it's a multi_object project, where every box is an
        # object by convention).
        is_object = (
            doc.geometry_type == "multi_object"
            or bool(b.name)
            or b.x != 0 or b.y != 0
        )
        if is_object and doc.geometry_type == "multi_object":
            objects_yaml.append({
                "id": b.object_id,
                "name": b.name,
                "x": b.x,
                "y": b.y,
            })
        # Hardware side: any box with ip OR host becomes a receiver. Boxes
        # with neither (passive objects) are object-only.
        is_receiver = bool(b.ip) or bool(b.host)
        if is_receiver:
            entry: dict = {}
            if b.ip:
                entry["ip"] = b.ip
            if b.host:
                entry["host"] = b.host
            entry["protocol"] = b.protocol
            entry["object_id"] = b.object_id
            entry["strips"] = []  # filled in next
            box_to_rx_idx[box_idx] = len(receivers_yaml_proto)
            receivers_yaml_proto.append(entry)

    # Re-nest strips under their assigned receiver. Each strip's `kind`
    # determines which fields get written so the round-trip preserves the
    # source format. Strips on a box that isn't a receiver get dropped
    # silently (they're inactive — passive objects don't have wire output).
    for s in doc.strips:
        rx_box_idx = s.receiver_idx
        if not (0 <= rx_box_idx < len(doc.boxes)):
            continue
        rx_yaml_idx = box_to_rx_idx.get(rx_box_idx)
        if rx_yaml_idx is None:
            continue  # box is object-only (no hardware), skip
        strip_entry: dict = {
            "group": s.group,
            "strip_idx": s.strip_idx,
            "kind": s.kind,
            "length": s.length,
            "direction": s.direction,
        }
        if s.kind == "column":
            # ``col`` is required by core/strip.py's column schema.
            strip_entry["col"] = s.col
        elif s.kind == "row":
            # ``row`` defaults to strip_idx in core/strip.py — only emit
            # when explicitly different so the YAML stays terse.
            if s.row != s.strip_idx:
                strip_entry["row"] = s.row
        # Only emit ``start`` when it differs from the direction-aware
        # default. Keeps Fan's existing YAML byte-identical when no one
        # has overridden start.
        if s.start != s.default_start():
            strip_entry["start"] = s.start
        # Arc shape metadata — only emitted when shape != "polyline"
        # so polyline-shaped strips stay terse on disk.
        if s.shape == "arc":
            strip_entry["shape"] = "arc"
            strip_entry["arc_center"] = [int(s.arc_center[0]),
                                         int(s.arc_center[1])]
            strip_entry["arc_radius"] = float(s.arc_radius)
            strip_entry["arc_start_deg"] = float(s.arc_start_deg)
            strip_entry["arc_end_deg"] = float(s.arc_end_deg)
            strip_entry["arc_segments"] = int(s.arc_segments)
        receivers_yaml_proto[rx_yaml_idx]["strips"].append(strip_entry)

    # ---- geometry.yaml (multi_object only) ----
    if doc.geometry_type == "multi_object":
        geometry = {
            "canvas": {"width": doc.canvas_w, "height": doc.canvas_h},
            "objects": objects_yaml,
            "strips": [
                {
                    "group": s.group,
                    "strip_idx": s.strip_idx,
                    "length": s.length,
                    "polyline": [[int(p[0]), int(p[1])] for p in s.polyline],
                }
                for s in doc.strips
                if s.polyline
            ],
        }
        geo_path = doc._raw_geometry_path or (proj_dir / "geometry.yaml")
        with open(geo_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(geometry, f, sort_keys=False, default_flow_style=False)

    # ---- project.yaml (always) ----
    proj = deepcopy(doc._raw_project)
    proj["groups"] = [
        {"id": g.id, "width": g.width, "height": g.height}
        for g in doc.groups
    ]
    proj["receivers"] = receivers_yaml_proto

    with open(proj_dir / "project.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(proj, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Color palette: stable color per group id
# ---------------------------------------------------------------------------

_GROUP_COLORS = [
    QColor("#6cc2ff"),  # blue
    QColor("#7be59a"),  # green
    QColor("#f7c46c"),  # amber
    QColor("#ff8a8a"),  # red
    QColor("#c89bff"),  # purple
    QColor("#5be5da"),  # teal
    QColor("#ffd66c"),  # yellow
    QColor("#ff7eb9"),  # pink
]


def color_for_group(group_id: str, all_groups: list[GroupSpec]) -> QColor:
    ids = [g.id for g in all_groups]
    if group_id in ids:
        return _GROUP_COLORS[ids.index(group_id) % len(_GROUP_COLORS)]
    return QColor("#aaaaaa")


# Per-receiver shading for the Strips table — gives a quick visual
# grouping of strips wired to the same physical box. Hand-tuned
# palette: each color is dim enough (luminance ≈ 0.12-0.18) to leave
# the text foreground white and read with good contrast, while still
# carrying a distinguishable hue. Hue order spaces adjacent receivers
# at ~30° intervals across the wheel to maximize differentiation.
_RECEIVER_SHADES = [
    QColor("#7a2828"),   #  0  red
    QColor("#7a4f28"),   #  1  orange
    QColor("#7a7a28"),   #  2  yellow
    QColor("#4f7a28"),   #  3  lime
    QColor("#287a28"),   #  4  green
    QColor("#287a4f"),   #  5  teal-green
    QColor("#287a7a"),   #  6  teal
    QColor("#284f7a"),   #  7  cyan-blue
    QColor("#28287a"),   #  8  blue
    QColor("#4f287a"),   #  9  violet
    QColor("#7a287a"),   # 10  magenta
    QColor("#7a284f"),   # 11  pink
]
_RECEIVER_UNASSIGNED = QColor("#2a2a2a")    # dark grey


def shade_for_receiver(receiver_idx: int) -> QColor:
    if receiver_idx < 0:
        return _RECEIVER_UNASSIGNED
    return _RECEIVER_SHADES[receiver_idx % len(_RECEIVER_SHADES)]


# ---------------------------------------------------------------------------
# Qt models for the four tabs
# ---------------------------------------------------------------------------

class _BaseModel(QAbstractTableModel):
    """Common machinery: emit a 'changed' signal so the canvas refreshes."""
    layout_changed = pyqtSignal()

    def __init__(self, doc: LayoutDoc):
        super().__init__()
        self.doc = doc

    def emit_layout_changed(self):
        self.layout_changed.emit()


class BoxesModel(_BaseModel):
    """Unified table of physical hardware units.

    Each row represents one box: a network endpoint (ip/host + protocol)
    that may also have a spatial position (name/x/y) when the project
    has a multi_object geometry. For Fan-style projects, the position
    columns are inert (not editable, not displayed in the canvas) but
    still present so the table schema is uniform across projects.
    """
    HEADERS = ["object_id", "name", "x", "y", "ip", "host", "protocol"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.doc.boxes)

    def columnCount(self, parent=QModelIndex()):
        return 7

    def headerData(self, section, orient, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orient == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def _is_position_meaningful(self) -> bool:
        return self.doc.geometry_type == "multi_object"

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        b = self.doc.boxes[index.row()]
        c = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if c == 0: return b.object_id
            if c == 1: return b.name
            if c == 2:
                return b.x if self._is_position_meaningful() else "—"
            if c == 3:
                return b.y if self._is_position_meaningful() else "—"
            if c == 4: return b.ip or ""
            if c == 5: return b.host or ""
            if c == 6: return b.protocol
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        b = self.doc.boxes[index.row()]
        c = index.column()
        try:
            if c == 0:
                b.object_id = int(value)
            elif c == 1:
                b.name = str(value)
            elif c == 2:
                if not self._is_position_meaningful():
                    return False
                b.x = int(value)
            elif c == 3:
                if not self._is_position_meaningful():
                    return False
                b.y = int(value)
            elif c == 4:
                b.ip = str(value) or None
            elif c == 5:
                b.host = str(value) or None
            elif c == 6:
                v = str(value).lower()
                if v not in ("sacn", "ddp"):
                    return False
                b.protocol = v
        except (TypeError, ValueError):
            return False
        self.dataChanged.emit(index, index)
        self.emit_layout_changed()
        return True

    def flags(self, index):
        c = index.column()
        # Position columns inert when the project has no spatial layout
        if c in (2, 3) and not self._is_position_meaningful():
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable)

    def add_row(self) -> int:
        next_id = max([b.object_id for b in self.doc.boxes], default=-1) + 1
        if self._is_position_meaningful():
            cx, cy = self.doc.canvas_w // 2, self.doc.canvas_h // 2
        else:
            cx = cy = 0
        new = BoxSpec(
            object_id=next_id,
            name=f"obj{next_id}" if self._is_position_meaningful() else "",
            x=cx, y=cy,
            host=f"ethernode-obj{next_id}.local",
        )
        self.beginInsertRows(QModelIndex(), len(self.doc.boxes), len(self.doc.boxes))
        self.doc.boxes.append(new)
        self.endInsertRows()
        self.emit_layout_changed()
        return len(self.doc.boxes) - 1

    def remove_rows(self, rows: list[int]):
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.doc.boxes):
                # Strips that referenced this box become unassigned;
                # higher box indices shift down by one.
                for s in self.doc.strips:
                    if s.receiver_idx == row:
                        s.receiver_idx = -1
                    elif s.receiver_idx > row:
                        s.receiver_idx -= 1
                self.beginRemoveRows(QModelIndex(), row, row)
                self.doc.boxes.pop(row)
                self.endRemoveRows()
        self.emit_layout_changed()


class GroupsModel(_BaseModel):
    HEADERS = ["id", "width", "height"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.doc.groups)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, section, orient, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orient == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole
        ):
            return None
        g = self.doc.groups[index.row()]
        return [g.id, g.width, g.height][index.column()]

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        g = self.doc.groups[index.row()]
        c = index.column()
        try:
            if c == 0: g.id = str(value)
            elif c == 1: g.width = int(value)
            elif c == 2: g.height = int(value)
        except (TypeError, ValueError):
            return False
        self.dataChanged.emit(index, index)
        self.emit_layout_changed()
        return True

    def flags(self, index):
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable)

    def add_row(self) -> int:
        new_id = "group" + str(len(self.doc.groups))
        new = GroupSpec(id=new_id, width=128, height=10)
        self.beginInsertRows(QModelIndex(), len(self.doc.groups), len(self.doc.groups))
        self.doc.groups.append(new)
        self.endInsertRows()
        self.emit_layout_changed()
        return len(self.doc.groups) - 1

    def remove_rows(self, rows: list[int]):
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.doc.groups):
                self.beginRemoveRows(QModelIndex(), row, row)
                self.doc.groups.pop(row)
                self.endRemoveRows()
        self.emit_layout_changed()


class StripsModel(_BaseModel):
    """Strips table.

    Columns separate the *wire* side (``strip_idx``: which slot of the
    receiver's universe this LED chain occupies) from the *canvas*
    side (``canvas_pos``: which row or column of the group canvas the
    pixels are pulled from). On Fan they're independent: each receiver
    counts ``strip_idx`` 0..31 locally while ``col`` runs 0..127
    globally across all four receivers. ``start`` is the offset along
    the canvas axis where the strip begins (default 0 → starts at the
    edge); combined with ``length`` it lets a strip occupy a sub-range
    rather than the entire row/column.
    """
    HEADERS = ["group", "strip_idx", "kind", "canvas_pos", "start",
               "length", "direction", "receiver", "polyline (n pts)"]
    HEADER_TOOLTIPS = [
        "The group canvas this strip renders from.",
        "Wire-order slot in this strip's receiver "
        "(0..N-1 per receiver — independent of canvas position).",
        "row → one row of the FBO; column → one column; raw → explicit polyline.",
        "Canvas row (kind=row) or column (kind=column) the strip pulls pixels from. "
        "On Fan: 0..127 global column index. On WoL: 0..N row in the group canvas.",
        "Canvas-axis position of LED 0 (the data-in end of the strip). "
        "Direction-aware default: 0 for right/up, length-1 for left/down. "
        "Chain walks ``length`` cells from start in ``direction``.",
        "Number of LEDs in the strip.",
        "Pixel-walk direction. row: right/left. column: down/up.",
        "Which physical box (receiver) drives this strip.",
        "Polyline on the composite preview canvas (multi_object). "
        "Click strip on canvas, drag white dots to reshape.",
    ]
    # Column indices (kept as constants so the layout is searchable).
    COL_GROUP = 0
    COL_STRIP_IDX = 1
    COL_KIND = 2
    COL_CANVAS_POS = 3
    COL_START = 4
    COL_LENGTH = 5
    COL_DIRECTION = 6
    COL_RECEIVER = 7
    COL_POLYLINE = 8

    def rowCount(self, parent=QModelIndex()):
        return len(self.doc.strips)

    def columnCount(self, parent=QModelIndex()):
        return 9

    def headerData(self, section, orient, role=Qt.ItemDataRole.DisplayRole):
        if orient != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.HEADER_TOOLTIPS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        s = self.doc.strips[index.row()]
        c = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if c == self.COL_GROUP: return s.group
            if c == self.COL_STRIP_IDX: return s.strip_idx
            if c == self.COL_KIND: return s.kind
            if c == self.COL_CANVAS_POS:
                if s.kind == "raw":
                    return "—"
                return s.canvas_pos
            if c == self.COL_START: return s.start
            if c == self.COL_LENGTH: return s.length
            if c == self.COL_DIRECTION: return s.direction
            if c == self.COL_RECEIVER:
                if 0 <= s.receiver_idx < len(self.doc.boxes):
                    return self.doc.boxes[s.receiver_idx].label()
                return "(unassigned)"
            if c == self.COL_POLYLINE:
                pts = len(s.polyline)
                if pts >= 2:
                    return f"{pts} pts: {s.polyline[0]} -> {s.polyline[-1]}"
                if pts == 1:
                    return f"1 pt: {s.polyline[0]}"
                return "—"
        if role == Qt.ItemDataRole.BackgroundRole:
            if c == self.COL_GROUP:
                return QBrush(color_for_group(s.group, self.doc.groups))
            # Every other column gets a per-receiver shade so strips
            # on the same physical box read together at a glance.
            return QBrush(shade_for_receiver(s.receiver_idx))
        if role == Qt.ItemDataRole.ForegroundRole:
            # Force text color explicitly so contrast doesn't depend
            # on the OS theme. Group column has a LIGHT pastel BG →
            # dark text; other columns have DIM saturated BG → white.
            if c == self.COL_GROUP:
                return QBrush(QColor("#101010"))
            return QBrush(QColor("#f0f0f0"))
        if role == Qt.ItemDataRole.ToolTipRole and c == self.COL_CANVAS_POS:
            return ("Canvas row (kind=row)" if s.kind == "row"
                    else "Canvas column (kind=column)" if s.kind == "column"
                    else "n/a — raw kind uses an explicit polyline")
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        s = self.doc.strips[index.row()]
        c = index.column()
        try:
            if c == self.COL_GROUP:
                s.group = str(value)
            elif c == self.COL_STRIP_IDX:
                # Wire-order slot. If this strip's row was tracking
                # strip_idx (the multi_object default), keep it tracking
                # so the user doesn't have to edit two cells at once.
                old_strip_idx = s.strip_idx
                new_strip_idx = int(value)
                if s.kind == "row" and s.row == old_strip_idx:
                    s.row = new_strip_idx
                s.strip_idx = new_strip_idx
            elif c == self.COL_KIND:
                v = str(value).lower()
                if v not in ("row", "column", "raw"):
                    return False
                old_kind = s.kind
                old_default = s.default_start()
                s.kind = v
                # Initialize the new kind's canvas index from the old one
                # so toggling preserves the strip's visual position.
                if v == "row" and old_kind != "row":
                    s.row = s.row or s.strip_idx
                    if s.direction in ("down", "up"):
                        s.direction = "right"
                elif v == "column" and old_kind != "column":
                    s.col = s.col or s.strip_idx
                    if s.direction in ("right", "left"):
                        s.direction = "down"
                # If start was at the old default, slide it to the new
                # default so the user doesn't have to re-edit it.
                if s.start == old_default:
                    s.start = s.default_start()
            elif c == self.COL_CANVAS_POS:
                if s.kind == "raw":
                    return False
                s.set_canvas_pos(int(value))
            elif c == self.COL_START:
                s.start = int(value)   # allow negative for direction-aware shifts
            elif c == self.COL_LENGTH:
                old_default = s.default_start()
                s.length = max(0, int(value))
                # If start was at the old direction-aware default, keep
                # it at the new default (which depends on length-1 for
                # down/left). Otherwise leave it alone.
                if s.start == old_default:
                    s.start = s.default_start()
            elif c == self.COL_DIRECTION:
                v = str(value).lower()
                if v not in ("right", "left", "down", "up"):
                    return False
                old_default = s.default_start()
                s.direction = v
                if s.start == old_default:
                    s.start = s.default_start()
            elif c == self.COL_RECEIVER:
                v = str(value)
                if v in ("(unassigned)", "", "-1"):
                    s.receiver_idx = -1
                else:
                    for i, b in enumerate(self.doc.boxes):
                        if b.label() == v or str(i) == v:
                            s.receiver_idx = i
                            break
                    else:
                        return False
                # Re-sort the strips list by (receiver_idx, strip_idx)
                # so this strip slots in next to its receiver siblings.
                # Defer one event loop tick — the QComboBox delegate
                # is still finishing this commit, and a synchronous
                # beginResetModel mid-commit trips Qt assertions on
                # some versions.
                target_strip = s
                QTimer.singleShot(
                    0, lambda ts=target_strip: self._resort_after_receiver_change(ts)
                )
            else:
                return False
        except (TypeError, ValueError):
            return False
        self.dataChanged.emit(index, index)
        self.emit_layout_changed()
        return True

    def _resort_after_receiver_change(self, target_strip):
        """Re-sort strips by receiver after a single strip's
        receiver_idx changed, then renumber strip_idx within each
        receiver so the moved strip picks up the next free wire-order
        slot in its new receiver. Re-selects the moved strip's new
        row so the operator's cursor follows it."""
        self.beginResetModel()
        self._sort_strips_by_receiver()
        self._renumber_strip_idx_per_receiver()
        self.endResetModel()
        self.emit_layout_changed()
        try:
            new_row = self.doc.strips.index(target_strip)
        except ValueError:
            return
        view = getattr(self, "_view", None)
        if view is not None:
            view.selectRow(new_row)

    def flags(self, index):
        c = index.column()
        if c == self.COL_POLYLINE:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if c == self.COL_CANVAS_POS:
            s = self.doc.strips[index.row()]
            base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if s.kind != "raw":
                base = base | Qt.ItemFlag.ItemIsEditable
            return base
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable)

    def add_row(self) -> int:
        group_id = self.doc.groups[0].id if self.doc.groups else "main"
        # Pick a kind that matches the project's existing strips so a new
        # row blends in (Fan: column; multi_object: row).
        existing_kinds = {s.kind for s in self.doc.strips}
        default_kind = "column" if existing_kinds == {"column"} else "row"
        # Next strip_idx within the chosen group
        existing = [s.strip_idx for s in self.doc.strips if s.group == group_id]
        new_idx = max(existing, default=-1) + 1
        cx, cy = self.doc.canvas_w // 2, self.doc.canvas_h // 2
        new = StripSpec(
            group=group_id,
            strip_idx=new_idx,
            length=64,
            polyline=[[cx - 32, cy], [cx + 32, cy]] if default_kind == "row" else [],
            receiver_idx=-1,
            direction="down" if default_kind == "column" else "right",
            kind=default_kind,
            col=new_idx if default_kind == "column" else 0,
        )
        # Append + sort by (receiver_idx, strip_idx) + renumber so
        # the new row slots in next to its receiver siblings AND
        # picks up the next available ``strip_idx`` in that receiver
        # (rather than the strip-counter heuristic above, which can
        # collide when receivers have been re-ordered or trimmed).
        self.beginResetModel()
        self.doc.strips.append(new)
        self._sort_strips_by_receiver()
        self._renumber_strip_idx_per_receiver()
        self.endResetModel()
        self.emit_layout_changed()
        return self.doc.strips.index(new)

    def _sort_strips_by_receiver(self):
        """Stable-sort doc.strips by (receiver_idx, strip_idx, group).
        Unassigned strips (receiver_idx == -1) sort to the end via a
        sentinel, matching their visual treatment (near-black shade)."""
        SENTINEL = 10 ** 9
        self.doc.strips.sort(
            key=lambda s: (
                s.receiver_idx if s.receiver_idx >= 0 else SENTINEL,
                s.strip_idx,
                s.group,
            )
        )

    def _renumber_strip_idx(self):
        """Reassign ``strip_idx`` to a dense 0..N-1 sequence in the
        dimension that the active geometry type considers the
        identity domain.

        - **multi_object**: ``strip_idx`` is the canvas-row identifier
          for kind=row strips and is keyed-on by ``geometry.yaml``
          (one polyline per ``(group, strip_idx)`` pair). It MUST be
          unique within each group, so we renumber per-group. Walking
          ``doc.strips`` in current order means the table's
          (receiver-sorted) ordering stays visually grouped.
        - **other (Fan-style)**: ``strip_idx`` is the wire slot within
          a receiver — it doesn't appear in geometry.yaml and the
          canvas-axis position lives in ``col``. Renumber per-receiver
          so each receiver's strip_idx values are dense 0..K-1.

        Default canvas-pos coupling (``s.row == s.strip_idx`` for
        kind=row, ``s.col == s.strip_idx`` for kind=column) is carried
        across the renumber so default strips don't get stranded on a
        stale canvas index. Hand-edited overrides are preserved.
        """
        is_multi = (self.doc.geometry_type == "multi_object")
        counters: dict = {}
        for s in self.doc.strips:
            bucket = s.group if is_multi else s.receiver_idx
            next_idx = counters.get(bucket, 0)
            old_idx = s.strip_idx
            if old_idx != next_idx:
                # Canvas-pos coupling only applies under multi_object,
                # where row/col default to strip_idx. Under Fan-style
                # geometry, col is the *physical* canvas column,
                # decoupled from per-receiver strip_idx — touching it
                # here would silently corrupt installations where col
                # happens to numerically match strip_idx.
                if is_multi:
                    if s.kind == "row" and s.row == old_idx:
                        s.row = next_idx
                    elif s.kind == "column" and s.col == old_idx:
                        s.col = next_idx
                s.strip_idx = next_idx
            counters[bucket] = next_idx + 1

    # Backwards-compat alias — older call sites referenced the
    # per-receiver name from when that was the only mode.
    _renumber_strip_idx_per_receiver = _renumber_strip_idx

    def remove_rows(self, rows: list[int]):
        # Pop the rows top-down (unchanged), then renumber and reset
        # the model so the ``strip_idx`` column reflects the new
        # dense wire-order numbering.
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.doc.strips):
                self.beginRemoveRows(QModelIndex(), row, row)
                self.doc.strips.pop(row)
                self.endRemoveRows()
        # Ordering preserved by the pops; just close gaps left in
        # strip_idx by the removed rows.
        self.beginResetModel()
        self._renumber_strip_idx_per_receiver()
        self.endResetModel()
        self.emit_layout_changed()

    def fix_strip_indexes_and_canvas_pos(self):
        """Re-sort strips by receiver, then densely renumber
        ``strip_idx`` according to the active geometry type:

        - **multi_object**: per-group 0..N-1. Forces ``row``/``col`` to
          match ``strip_idx`` for kind=row/column strips so the
          canvas-row identity stays coupled (the runtime samples
          ``group_frame[strip_idx, …]`` and geometry.yaml is keyed by
          ``(group, strip_idx)`` — duplicates corrupt polylines).
        - **other (Fan-style)**: per-receiver 0..N-1. Leaves ``col``
          alone — Fan's canvas column is the *physical* column index
          across all receivers, decoupled from the per-receiver wire
          slot.

        Idempotent."""
        self.beginResetModel()
        self._sort_strips_by_receiver()
        self._renumber_strip_idx()
        if self.doc.geometry_type == "multi_object":
            for s in self.doc.strips:
                if s.kind == "row":
                    s.row = s.strip_idx
                elif s.kind == "column":
                    s.col = s.strip_idx
        self.endResetModel()
        self.emit_layout_changed()


# ---------------------------------------------------------------------------
# Editing delegates: dropdowns for enum-like cells in the Strips table
# ---------------------------------------------------------------------------

class _ChoiceDelegate(QStyledItemDelegate):
    """QStyledItemDelegate that swaps the editor for a QComboBox of fixed
    choices. Used for Strips.direction (right/left/down/up) and
    Strips.kind (row/column/raw)."""
    def __init__(self, choices: list[str], parent=None):
        super().__init__(parent)
        self._choices = list(choices)

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self._choices)
        return combo

    def setEditorData(self, editor: QComboBox, index):
        current = str(index.model().data(index, Qt.ItemDataRole.EditRole))
        i = editor.findText(current)
        if i >= 0:
            editor.setCurrentIndex(i)

    def setModelData(self, editor: QComboBox, model, index):
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class _GroupDelegate(QStyledItemDelegate):
    """Group-id dropdown sourced live from the *currently active* doc.

    The delegate stores a zero-arg callable that returns the live doc —
    not the doc itself — so the dropdown stays correct after a project
    switch (which replaces ``EditorWindow.doc`` wholesale)."""
    def __init__(self, doc_provider, parent=None):
        super().__init__(parent)
        self._doc_provider = doc_provider

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        for g in self._doc_provider().groups:
            combo.addItem(g.id)
        # Allow typing a new group id too — handy for projects in flight.
        combo.setEditable(True)
        return combo

    def setEditorData(self, editor: QComboBox, index):
        current = str(index.model().data(index, Qt.ItemDataRole.EditRole))
        i = editor.findText(current)
        if i >= 0:
            editor.setCurrentIndex(i)
        else:
            editor.setEditText(current)

    def setModelData(self, editor: QComboBox, model, index):
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class _ReceiverDelegate(QStyledItemDelegate):
    """Receiver dropdown sourced live from the *currently active* doc.
    Stores the box's label string (StripsModel.setData resolves label →
    idx). Uses a doc-provider callable so a project switch is reflected
    immediately."""
    def __init__(self, doc_provider, parent=None):
        super().__init__(parent)
        self._doc_provider = doc_provider

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItem("(unassigned)")
        for b in self._doc_provider().boxes:
            combo.addItem(b.label())
        return combo

    def setEditorData(self, editor: QComboBox, index):
        current = str(index.model().data(index, Qt.ItemDataRole.EditRole))
        i = editor.findText(current)
        if i >= 0:
            editor.setCurrentIndex(i)

    def setModelData(self, editor: QComboBox, model, index):
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


# ---------------------------------------------------------------------------
# Canvas: QGraphicsScene/View showing the layout, with draggable objects
# ---------------------------------------------------------------------------

class _ObjectItem(QGraphicsEllipseItem):
    """Draggable dot for one box's spatial position (multi_object only).

    Visual radius stays small (so dense layouts don't smush together)
    but the click ``shape()`` is widened to a larger invisible disc —
    grab anywhere within ``HIT_RADIUS`` to drag the object."""

    RADIUS = 9
    HIT_RADIUS = 16

    def __init__(self, obj_idx: int, obj: BoxSpec, on_moved):
        super().__init__(-self.RADIUS, -self.RADIUS, 2 * self.RADIUS, 2 * self.RADIUS)
        self.obj_idx = obj_idx
        self._on_moved = on_moved
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setBrush(QBrush(QColor("#ffd66c")))
        self.setPen(QPen(QColor("#000000"), 1.5))
        self.setZValue(10)
        self.setPos(obj.x, obj.y)
        self.setToolTip(
            f"{obj.name or 'obj'+str(obj.object_id)} "
            f"(id={obj.object_id})  ({obj.x},{obj.y})"
        )

    def shape(self):
        # Widen the click/drag hit area beyond the visible dot so
        # rubber-band selection doesn't steal the click when the
        # operator's aim is slightly off. The visible ellipse stays
        # small (RADIUS=9); only the interaction zone is bigger.
        path = QPainterPath()
        path.addEllipse(-self.HIT_RADIUS, -self.HIT_RADIUS,
                        2 * self.HIT_RADIUS, 2 * self.HIT_RADIUS)
        return path

    def paint(self, painter, option, widget=None):
        # Custom paint: thick white outline + bigger fill when this
        # object is selected. Qt's default selected-item rendering
        # (a faint dashed rect) is hard to see on the dark canvas
        # and made multi-select status invisible to the operator.
        if self.isSelected():
            painter.setBrush(QBrush(QColor("#ffd66c")))
            painter.setPen(QPen(QColor("#ffffff"), 3))
            r = self.RADIUS + 2
            painter.drawEllipse(-r, -r, 2 * r, 2 * r)
        else:
            painter.setBrush(QBrush(QColor("#ffd66c")))
            painter.setPen(QPen(QColor("#000000"), 1.5))
            r = self.RADIUS
            painter.drawEllipse(-r, -r, 2 * r, 2 * r)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_moved(self.obj_idx, int(value.x()), int(value.y()))
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # Force a repaint so the highlight border swaps in/out.
            self.update()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)):
            scene.object_drag_started.emit(self.obj_idx)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)):
            scene.object_drag_finished.emit(self.obj_idx)
        super().mouseReleaseEvent(event)


class _VertexHandle(QGraphicsEllipseItem):
    """Draggable handle at a polyline vertex (multi_object strips only).

    Color codes the role:
      * green = chain start (LED 0)
      * red   = chain end   (last LED)
      * white = interior vertex
    Start/end also get a slightly larger radius so they're easy to
    spot on dense paths. Hidden by default; the scene reveals only
    the handles for the currently-selected strip.
    """
    RADIUS_INTERIOR = 5
    RADIUS_ENDPOINT = 7

    COLOR_START    = QColor("#7be59a")   # green
    COLOR_END      = QColor("#ff8a8a")   # red
    COLOR_INTERIOR = QColor("#ffffff")

    def __init__(self, strip_idx: int, vertex_idx: int, x: int, y: int,
                 on_moved, role: str = "interior"):
        # ``role`` is "start" / "end" / "interior" — picks color + size.
        if role in ("start", "end"):
            r = self.RADIUS_ENDPOINT
        else:
            r = self.RADIUS_INTERIOR
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.strip_idx = strip_idx
        self.vertex_idx = vertex_idx
        self.role = role
        self._on_moved = on_moved
        self._suppress_callback = True
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        if role == "start":
            self.setBrush(QBrush(self.COLOR_START))
            label = "chain start (LED 0)"
        elif role == "end":
            self.setBrush(QBrush(self.COLOR_END))
            label = "chain end (last LED)"
        else:
            self.setBrush(QBrush(self.COLOR_INTERIOR))
            label = f"vertex {vertex_idx}"
        self.setPen(QPen(QColor("#000000"), 1))
        self.setZValue(20)
        self.setVisible(False)
        self.setPos(x, y)
        self.setToolTip(f"{label}  ({x}, {y}) — drag to reshape")
        self._suppress_callback = False

    def set_pos_silent(self, x: int, y: int):
        """Reposition without firing the on_moved callback."""
        self._suppress_callback = True
        try:
            self.setPos(x, y)
        finally:
            self._suppress_callback = False

    def itemChange(self, change, value):
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
                and not self._suppress_callback):
            self._on_moved(self.strip_idx, self.vertex_idx,
                           int(value.x()), int(value.y()))
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)):
            scene.vertex_drag_started.emit(self.strip_idx)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)):
            scene.vertex_drag_finished.emit(self.strip_idx)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        scene = self.scene()
        if isinstance(scene, LayoutScene):
            scene.vertex_context_requested.emit(
                self.strip_idx, self.vertex_idx,
                event.screenPos().x(), event.screenPos().y(),
            )
        event.accept()


class _ArcHandle(QGraphicsEllipseItem):
    """Direct-manipulation handle for an arc-shaped strip.

    Roles map to the four interactive points on an arc:
      * ``center`` — yellow; drag to pan the whole arc
      * ``start``  — white; drag to change ``arc_start_deg`` (snaps to
                    the current radius circle so the arc stays
                    well-defined)
      * ``end``    — white; drag to change ``arc_end_deg``
      * ``radius`` — teal; sits at the arc's midpoint, drag radially
                    to change ``arc_radius``

    Hidden by default; revealed when the strip is the highlighted
    selection (same gating as polyline vertex handles)."""

    # Bigger than vertex-handle endpoints (R=7) so arc controls stand
    # out and are easier to grab without precision aim.
    RADIUS = 9
    HIT_RADIUS = 16
    # Match the polyline-vertex palette so start/end are recognisable
    # whether the strip is shape=polyline or shape=arc.
    COLORS = {
        "center": QColor("#ffd66c"),   # yellow — pan
        "start":  QColor("#7be59a"),   # green — chain start (LED 0)
        "end":    QColor("#ff8a8a"),   # red — chain end (last LED)
        "radius": QColor("#5be5da"),   # teal — radius
    }

    def __init__(self, strip_idx: int, role: str,
                 x: int, y: int, on_moved):
        super().__init__(-self.RADIUS, -self.RADIUS,
                         2 * self.RADIUS, 2 * self.RADIUS)
        self.strip_idx = strip_idx
        self.role = role
        self._on_moved = on_moved
        self._suppress_callback = True
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setBrush(QBrush(self.COLORS.get(role, QColor("#ffffff"))))
        self.setPen(QPen(QColor("#000000"), 1))
        self.setZValue(21)         # above polyline vertex handles
        self.setVisible(False)
        self.setPos(x, y)
        self.setToolTip(f"arc {role} — drag to reshape")
        self._suppress_callback = False

    def set_pos_silent(self, x: int, y: int):
        self._suppress_callback = True
        try:
            self.setPos(x, y)
        finally:
            self._suppress_callback = False

    def itemChange(self, change, value):
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
                and not self._suppress_callback):
            self._on_moved(self.strip_idx, self.role,
                           int(value.x()), int(value.y()))
        return super().itemChange(change, value)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(-self.HIT_RADIUS, -self.HIT_RADIUS,
                        2 * self.HIT_RADIUS, 2 * self.HIT_RADIUS)
        return path

    def mousePressEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)):
            scene.vertex_drag_started.emit(self.strip_idx)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)):
            scene.vertex_drag_finished.emit(self.strip_idx)
        super().mouseReleaseEvent(event)


def _strip_path_points(strip: StripSpec) -> list:
    """Derive the canvas-space points to draw this strip on the
    *physical* layout scene.

    For multi_object row strips and raw strips, ``polyline`` is the
    composite-canvas path (filled by load_doc). For other cases the
    strip is drawn on its group canvas's local axes between the
    direction-aware chain-start (``strip.start``) and chain-end
    (``strip.chain_end_pos()``) positions, ordered chain-start →
    chain-end so the polyline is data-flow-aligned. The Fan project
    replaces these column-line stubs with FanGeometry-derived
    polylines at load time."""
    if strip.polyline:
        return list(strip.polyline)
    length = max(int(strip.length), 0)
    if length <= 0:
        return []
    a_start = int(strip.start)
    a_end = int(strip.chain_end_pos())
    if strip.kind == "column":
        return [[int(strip.col), a_start], [int(strip.col), a_end]]
    if strip.kind == "row":
        return [[a_start, int(strip.row)], [a_end, int(strip.row)]]
    return []


class _StripItem(QGraphicsPathItem):
    """Strip path on the physical-layout scene. Click to select (the
    parent ``LayoutScene`` emits ``selection_changed_strip`` so the
    editor switches to the Strips tab + reveals this strip's vertex
    handles)."""
    def __init__(self, strip_idx: int, strip: StripSpec, color: QColor):
        super().__init__()
        self.strip_idx = strip_idx
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        # Strip-translation drag state. Initialized to "no drag in
        # progress" so the move handler is safe even if a synthetic
        # move arrives before any press.
        self._press_scene_pos = None
        self._dragging = False
        self._drag_emitted_started = False
        # Use a thicker invisible "hit" area so close-packed strips
        # (Fan: 32 columns 1px apart) are still individually clickable.
        self.set_points(_strip_path_points(strip))
        pen = QPen(color, 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.setZValue(5)
        if strip.kind == "row":
            pos_str = f"row={strip.row}"
        elif strip.kind == "column":
            pos_str = f"col={strip.col}"
        else:
            pos_str = "raw"
        self.setToolTip(
            f"{strip.group} strip_idx={strip.strip_idx}  kind={strip.kind}  "
            f"{pos_str}  start={strip.start}  len={strip.length}  "
            f"dir={strip.direction}\n"
            f"(click to select, then drag the white dots to reshape)"
        )

    DRAG_THRESHOLD = 4   # scene px before a press becomes a translation drag

    def mousePressEvent(self, event):
        scene = self.scene()
        if isinstance(scene, LayoutScene):
            scene.selection_changed_strip.emit(self.strip_idx)
        # Capture potential strip-translation. The drag only commits
        # to a translation after the cursor moves past DRAG_THRESHOLD,
        # so a quick click-release doesn't shift the strip by a pixel.
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_scene_pos = event.scenePos()
            self._dragging = False
            self._drag_emitted_started = False
        else:
            self._press_scene_pos = None
        # If this strip is already part of a multi-selection AND the
        # user pressed without a modifier, skip Qt's default selection
        # logic (which would clear the others and select only this).
        # Preserving the multi-selection lets the subsequent drag move
        # the whole group together.
        if (event.button() == Qt.MouseButton.LeftButton
                and self.isSelected()
                and isinstance(scene, LayoutScene)
                and not (event.modifiers() & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.ShiftModifier))):
            multi = sum(
                1 for it in scene.selectedItems()
                if isinstance(it, _StripItem)
            ) > 1
            if multi:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_scene_pos is None:
            super().mouseMoveEvent(event)
            return
        scene = self.scene()
        if not isinstance(scene, LayoutScene):
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._press_scene_pos
        if not self._dragging:
            if abs(delta.x()) + abs(delta.y()) < self.DRAG_THRESHOLD:
                super().mouseMoveEvent(event)
                return
            # Crossed the threshold — start the translation drag.
            self._dragging = True
            scene.strip_translate_started.emit(self.strip_idx)
            self._drag_emitted_started = True
        scene.strip_translated.emit(
            self.strip_idx, int(delta.x()), int(delta.y())
        )
        # Don't call super() while translating; we don't want Qt's
        # default selection-rect / pan behaviour to interfere.
        event.accept()

    def mouseReleaseEvent(self, event):
        scene = self.scene()
        if (event.button() == Qt.MouseButton.LeftButton
                and isinstance(scene, LayoutScene)
                and self._drag_emitted_started):
            scene.strip_translate_finished.emit(self.strip_idx)
        self._press_scene_pos = None
        self._dragging = False
        self._drag_emitted_started = False
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        scene = self.scene()
        if isinstance(scene, LayoutScene):
            # Pass the click point in scene coords so the editor can
            # insert a vertex at the right place along the polyline.
            scene.strip_context_requested.emit(
                self.strip_idx,
                int(event.scenePos().x()), int(event.scenePos().y()),
                event.screenPos().x(), event.screenPos().y(),
            )
        event.accept()

    def shape(self):
        # Widen the click target so close-packed strips remain selectable.
        from PyQt6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(8.0)
        return stroker.createStroke(self.path())

    def set_points(self, points: list):
        path = QPainterPath()
        if points:
            path.moveTo(points[0][0], points[0][1])
            for p in points[1:]:
                path.lineTo(p[0], p[1])
        self.setPath(path)

    # Backwards-compat alias for callers that still say set_polyline()
    set_polyline = set_points


class LayoutScene(QGraphicsScene):
    object_moved = pyqtSignal(int, int, int)   # (obj_idx, x, y)
    vertex_moved = pyqtSignal(int, int, int, int)  # (strip_idx, vertex_idx, x, y)
    selection_changed_obj = pyqtSignal(int)    # row index, -1 = none
    selection_changed_strip = pyqtSignal(int)  # emitted on strip click
    # Right-click on a strip path: (strip_idx, scene_x, scene_y, screen_x, screen_y).
    # The first two are where the click landed (used to insert a vertex
    # at that position along the polyline); the last two are the global
    # screen coords for popping a QMenu.
    strip_context_requested = pyqtSignal(int, int, int, int, int)
    # Right-click on a vertex handle: (strip_idx, vertex_idx, screen_x, screen_y).
    vertex_context_requested = pyqtSignal(int, int, int, int)
    # Drag lifecycle for the undo stack — pre-state captured on
    # ``_started``, command pushed on ``_finished``.
    vertex_drag_started = pyqtSignal(int)        # strip_idx
    vertex_drag_finished = pyqtSignal(int)
    object_drag_started = pyqtSignal(int)        # obj_idx
    object_drag_finished = pyqtSignal(int)
    # Strip-translation drag: user clicks and drags the strip's path
    # (not a handle) to translate every vertex by the same delta.
    strip_translate_started = pyqtSignal(int)        # strip_idx
    strip_translated = pyqtSignal(int, int, int)     # strip_idx, dx, dy (scene)
    strip_translate_finished = pyqtSignal(int)

    arc_handle_moved = pyqtSignal(int, str, int, int)
    # ^^ (strip_idx, role, x, y) where role is "center" / "start" /
    # "end" / "radius". Editor recomputes the strip's arc params
    # from the new handle position, regenerates the polyline, and
    # asks the scene to reposition the OTHER arc handles.

    def __init__(self):
        super().__init__()
        self._objects: list[_ObjectItem] = []
        self._strips: list[_StripItem] = []
        self._labels: list[QGraphicsSimpleTextItem] = []
        # Per-strip vertex handles (polyline shape) and per-strip arc
        # handles (arc shape). Mutually exclusive — a given strip has
        # one populated and the other empty depending on its shape.
        self._handles: list[list[_VertexHandle]] = []
        self._arc_handles: list[dict[str, _ArcHandle]] = []

    def rebuild(self, doc: LayoutDoc):
        # Remove everything
        self.clear()
        self._objects = []
        self._strips = []
        self._labels = []
        self._handles = []
        self._arc_handles = []

        # Canvas frame
        frame = self.addRect(0, 0, doc.canvas_w, doc.canvas_h)
        frame.setPen(QPen(QColor("#444"), 2, Qt.PenStyle.DashLine))
        frame.setBrush(QBrush(QColor("#1a1a1a")))
        frame.setZValue(-100)

        # Strips (drawn under objects)
        polylines_editable = (doc.geometry_type == "multi_object")
        for i, s in enumerate(doc.strips):
            color = color_for_group(s.group, doc.groups)
            item = _StripItem(i, s, color)
            self.addItem(item)
            self._strips.append(item)

            # Vertex handles — only for strips with editable polylines
            # (multi_object only). Fan polylines are derived from
            # FanGeometry and not editable from the canvas; you change
            # them by editing FanGeometry's inner_r / outer_r in the
            # project.yaml ``geometry`` block.
            #
            # Arc-shaped strips get a different handle set (center,
            # endpoints, radius) instead of per-vertex handles, so the
            # operator can manipulate the arc directly without
            # accidentally converting it back to a polyline.
            arc_handles: dict[str, _ArcHandle] = {}
            if (polylines_editable and s.shape == "arc"
                    and s.arc_radius > 0):
                import math as _m
                cx, cy = s.arc_center
                sa = _m.radians(s.arc_start_deg)
                ea = _m.radians(s.arc_end_deg)
                ma = (sa + ea) / 2.0
                r = s.arc_radius
                # canvas y grows down → negate sin (matches _arc_to_polyline)
                spec_pts = {
                    "center": (cx, cy),
                    "start":  (cx + r * _m.cos(sa), cy - r * _m.sin(sa)),
                    "end":    (cx + r * _m.cos(ea), cy - r * _m.sin(ea)),
                    "radius": (cx + r * _m.cos(ma), cy - r * _m.sin(ma)),
                }
                for role, (hx, hy) in spec_pts.items():
                    h = _ArcHandle(i, role, int(hx), int(hy),
                                   self._on_arc_handle_moved)
                    self.addItem(h)
                    arc_handles[role] = h
            self._arc_handles.append(arc_handles)
            handles: list[_VertexHandle] = []
            if polylines_editable and not arc_handles and s.polyline:
                last = len(s.polyline) - 1
                for vi, (vx, vy) in enumerate(s.polyline):
                    if vi == 0:
                        role = "start"
                    elif vi == last:
                        role = "end"
                    else:
                        role = "interior"
                    h = _VertexHandle(i, vi, int(vx), int(vy),
                                      self._on_vertex_moved, role=role)
                    self.addItem(h)
                    handles.append(h)
            self._handles.append(handles)

        # Box dots (only meaningful when the project has spatial layout —
        # multi_object — Fan-style projects render only strips on the canvas).
        if doc.geometry_type == "multi_object":
            for i, b in enumerate(doc.boxes):
                item = _ObjectItem(i, b, self._on_object_moved)
                self.addItem(item)
                self._objects.append(item)

                label = QGraphicsSimpleTextItem(b.name or f"obj{b.object_id}")
                font = QFont()
                font.setPointSize(9)
                label.setFont(font)
                label.setBrush(QBrush(QColor("#cccccc")))
                label.setPos(b.x + _ObjectItem.RADIUS + 2, b.y + _ObjectItem.RADIUS - 4)
                label.setZValue(11)
                self.addItem(label)
                self._labels.append(label)

        self.setSceneRect(QRectF(-50, -50, doc.canvas_w + 100, doc.canvas_h + 100))

    def update_strip_polyline(self, strip_idx: int, polyline: list,
                              skip_handle_at: int = -1):
        """Redraw a strip's polyline + reposition its vertex handles.

        ``skip_handle_at`` is the index of a handle the user is currently
        dragging — moving that handle from inside its own itemChange
        callback would re-emit the change signal and recurse. Pass it
        when called from the drag path; pass -1 (default) when called
        from any other code path.
        """
        if not (0 <= strip_idx < len(self._strips)):
            return
        self._strips[strip_idx].set_points(polyline)
        handles = self._handles[strip_idx] if strip_idx < len(self._handles) else []
        for vi, (vx, vy) in enumerate(polyline):
            if vi >= len(handles) or vi == skip_handle_at:
                continue
            handles[vi].set_pos_silent(int(vx), int(vy))
            handles[vi].setToolTip(f"vertex {vi} ({int(vx)}, {int(vy)}) — drag to reshape")

    def update_strip_color(self, strip_idx: int, color: QColor):
        if 0 <= strip_idx < len(self._strips):
            pen = self._strips[strip_idx].pen()
            pen.setColor(color)
            self._strips[strip_idx].setPen(pen)

    def update_object(self, obj_idx: int, box: BoxSpec):
        if 0 <= obj_idx < len(self._objects):
            item = self._objects[obj_idx]
            item.setPos(box.x, box.y)
            item.setToolTip(
                f"{box.name or 'obj'+str(box.object_id)} "
                f"(id={box.object_id})  ({box.x},{box.y})"
            )
        if 0 <= obj_idx < len(self._labels):
            self._labels[obj_idx].setText(box.name or f"obj{box.object_id}")
            self._labels[obj_idx].setPos(
                box.x + _ObjectItem.RADIUS + 2, box.y + _ObjectItem.RADIUS - 4
            )

    def highlight_object(self, obj_idx: int):
        for i, item in enumerate(self._objects):
            pen = QPen(QColor("#000000"), 1.5)
            if i == obj_idx:
                pen = QPen(QColor("#ffffff"), 3)
            item.setPen(pen)

    def highlight_strip(self, strip_idx: int):
        self.highlight_strips({strip_idx} if strip_idx >= 0 else set())

    def highlight_strips(self, indices):
        """Multi-select variant: bumps the pen width on every strip in
        ``indices`` AND sets the underlying ``QGraphicsItem`` selection
        state so the editor's group-drag capture (which reads from
        ``scene.selectedItems()``) sees the multi-selection. Vertex /
        arc handles only show when exactly one strip is selected —
        multi-select would flood the canvas with handles and the
        operator can't drag-edit shape across multiple strips at once
        anyway."""
        indices = set(indices) if indices else set()
        # Block selectionChanged emissions while we sync N items, then
        # fire one consolidated update so trackers (objects + strips)
        # refresh exactly once.
        self.blockSignals(True)
        try:
            for i, item in enumerate(self._strips):
                pen = item.pen()
                pen.setWidthF(4.5 if i in indices else 2.5)
                item.setPen(pen)
                want = (i in indices)
                if item.isSelected() != want:
                    item.setSelected(want)
        finally:
            self.blockSignals(False)
        self.selectionChanged.emit()
        single = next(iter(indices)) if len(indices) == 1 else -1
        for i, handles in enumerate(self._handles):
            visible = (i == single)
            for h in handles:
                h.setVisible(visible)
        for i, handles in enumerate(self._arc_handles):
            visible = (i == single)
            for h in handles.values():
                h.setVisible(visible)

    def update_arc_handles(self, strip_idx: int,
                           cx: float, cy: float,
                           radius: float,
                           start_deg: float, end_deg: float,
                           except_role: str = "") -> None:
        """Reposition the arc's three derived handles (everything
        except the one currently being dragged). Called by the editor
        after recomputing arc params from a handle drag."""
        if not (0 <= strip_idx < len(self._arc_handles)):
            return
        import math as _m
        sa = _m.radians(start_deg)
        ea = _m.radians(end_deg)
        ma = (sa + ea) / 2.0
        positions = {
            "center": (cx, cy),
            "start":  (cx + radius * _m.cos(sa), cy - radius * _m.sin(sa)),
            "end":    (cx + radius * _m.cos(ea), cy - radius * _m.sin(ea)),
            "radius": (cx + radius * _m.cos(ma), cy - radius * _m.sin(ma)),
        }
        handles = self._arc_handles[strip_idx]
        for role, (hx, hy) in positions.items():
            if role == except_role:
                continue
            h = handles.get(role)
            if h is not None:
                h.set_pos_silent(int(hx), int(hy))

    def _on_object_moved(self, obj_idx: int, x: int, y: int):
        self.object_moved.emit(obj_idx, x, y)

    def _on_vertex_moved(self, strip_idx: int, vertex_idx: int, x: int, y: int):
        self.vertex_moved.emit(strip_idx, vertex_idx, x, y)

    def _on_arc_handle_moved(self, strip_idx: int, role: str, x: int, y: int):
        self.arc_handle_moved.emit(strip_idx, role, x, y)


class GroupCanvasScene(QGraphicsScene):
    """Secondary canvas showing each group as its own rendering surface.

    The main canvas above shows physical layout (composite world); this
    panel shows the *logical* canvases that effects render to. For each
    group, draws a labeled rectangle at the group's (width, height) and
    overlays each strip's footprint on that surface:

      * ``row``    strip → horizontal segment at row ``strip_idx``,
                   columns 0..length-1
      * ``column`` strip → vertical segment at column ``col``,
                   rows 0..length-1
      * ``raw``    strip → drawn as the polyline coords (legacy fallback)

    Groups are stacked vertically with labels above each. Clicking a
    strip in this panel emits ``selection_changed_strip`` so the main
    canvas + Strips table can sync to the same selection.
    """

    GROUP_GAP = 50        # vertical gap between stacked group rects
    LABEL_HEIGHT = 22
    # Each group canvas is drawn at a UNIFORM per-group scale so its
    # FBO aspect ratio is preserved. Three constraints layered on top
    # of each other:
    #   - smaller axis at least MIN_DISPLAY_DIM px (a 9-row trunk
    #     canvas stays readable)
    #   - adjacent strip lines at least MIN_STRIP_SPACING px apart
    #     (Fan's 32 strips on 128 px native = 4 px stride get scaled
    #     up to ~9 px stride uniformly — preserves aspect AND avoids
    #     the "jammed up" look)
    #   - larger axis at most MAX_DISPLAY_DIM px (panel can scroll
    #     for the rare case where strip density forces this floor
    #     above the cap)
    MIN_DISPLAY_DIM = 80
    MAX_DISPLAY_DIM = 800
    # Hard absolute cap on the larger display axis. Even when strip
    # density demands more scaling than this allows, we settle so the
    # group canvas panel doesn't become a kilometre-tall scrollable
    # column. Wheel-zoom remains for inspection.
    HARD_CAP = 1500
    MIN_STRIP_SPACING = 9
    TICK_PT = 7

    selection_changed_strip = pyqtSignal(int)  # strip_idx, -1 = clear

    def __init__(self):
        super().__init__()
        # Map (group_id) → (origin_x, origin_y, scale) on this scene.
        # ``scale`` is the per-group multiplier from FBO pixels to scene
        # pixels — needed so strip paths align with their group's rect.
        self._group_origin: dict[str, tuple[float, float, float]] = {}
        self._strip_items: dict[int, QGraphicsPathItem] = {}

    def _group_display_scale(self, g: GroupSpec,
                             strips: list) -> tuple[float, float]:
        """Per-group uniform display scale, returned as (sx, sy) where
        sx == sy so the FBO aspect ratio is preserved.

        Three lower-bound constraints, all in scale-factor units:
          * strip-density floor — adjacent strip lines at least
            MIN_STRIP_SPACING px apart on either axis (Fan: 128 cols
            on 128px native = 1px stride → scale ≥ MIN_STRIP_SPACING)
          * smaller-axis floor — tiny canvases stay readable
          * native 1.0 — never down-scale a small canvas
        Plus an upper-bound *target* (MAX_DISPLAY_DIM): used only when
        none of the floors require a bigger scale. Density wins when
        they conflict; the panel's QGraphicsView handles scroll and
        wheel-zoom for the resulting taller-than-screen canvases."""
        if g.width <= 0 or g.height <= 0:
            return 1.0, 1.0

        smaller = min(g.width, g.height)
        larger = max(g.width, g.height)

        cols = {int(s.col) for s in strips
                if s.group == g.id and s.kind == "column"}
        rows = {int(s.row) for s in strips
                if s.group == g.id and s.kind == "row"}
        strip_floor = 1.0
        if len(cols) >= 2:
            native = g.width / max(len(cols), 1)
            if native < self.MIN_STRIP_SPACING:
                strip_floor = max(strip_floor, self.MIN_STRIP_SPACING / native)
        if len(rows) >= 2:
            native = g.height / max(len(rows), 1)
            if native < self.MIN_STRIP_SPACING:
                strip_floor = max(strip_floor, self.MIN_STRIP_SPACING / native)

        small_floor = max(self.MIN_DISPLAY_DIM / smaller, 1.0)

        floor = max(strip_floor, small_floor, 1.0)

        # If neither floor demanded growth, pull the canvas up to a
        # comfortable display size (at most MAX_DISPLAY_DIM on the
        # larger axis). This is what makes a 60×8 WoL ambient canvas
        # display at ~600 px wide instead of 60 px.
        if floor < 1.5:
            comfort = self.MAX_DISPLAY_DIM / larger
            floor = max(floor, min(comfort, self.MAX_DISPLAY_DIM / smaller))

        # Hard cap on the larger axis. Caps Fan's 128-strip canvas at
        # ``HARD_CAP / 300`` ≈ 5× scale instead of 9× — strip stride
        # ~5 px instead of ~9 px, but the panel stays a reasonable
        # height. User can wheel-zoom to inspect dense regions.
        floor = min(floor, self.HARD_CAP / larger)

        # Round to integer scale when ≥ 2 — cleaner pixel rendering.
        if floor >= 2.0:
            floor = float(round(floor))
        return floor, floor

    def rebuild(self, doc: LayoutDoc):
        self.clear()
        self._group_origin = {}
        self._strip_items = {}

        if not doc.groups:
            return

        cursor_y = 0.0
        max_w = 0.0
        tick_font = QFont()
        tick_font.setPointSize(self.TICK_PT)

        for g in doc.groups:
            color = color_for_group(g.id, doc.groups)
            sx, sy = self._group_display_scale(g, doc.strips)
            disp_w = g.width * sx
            disp_h = g.height * sy

            # Header label
            scale_note = ""
            if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                if abs(sx - sy) < 0.01:
                    scale_note = f"   (×{sx:.1f} display)"
                else:
                    scale_note = f"   (×{sx:.1f}h ×{sy:.1f}v display)"
            label = QGraphicsSimpleTextItem(
                f"{g.id}   {g.width} × {g.height} px{scale_note}"
            )
            f = QFont()
            f.setPointSize(10)
            f.setBold(True)
            label.setFont(f)
            label.setBrush(QBrush(color))
            label.setPos(0, cursor_y)
            label.setZValue(2)
            self.addItem(label)
            cursor_y += self.LABEL_HEIGHT

            # Group canvas rect (the FBO surface, displayed scaled)
            rect = self.addRect(0, cursor_y, disp_w, disp_h)
            rect.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
            rect.setBrush(QBrush(QColor("#181818")))
            rect.setZValue(0)
            self._group_origin[g.id] = (0.0, cursor_y, sx, sy)

            # Tick labels along the top edge: each col index (for column
            # strips) or each row index (for row strips). Pick the axis
            # with the most strips on this group; show evenly-spaced
            # subset if there are many.
            strips_here = [s for s in doc.strips if s.group == g.id]
            row_count = sum(1 for s in strips_here if s.kind == "row")
            col_count = sum(1 for s in strips_here if s.kind == "column")
            if col_count >= row_count and col_count > 0:
                # Column ticks above the rect
                cols = sorted({int(s.col) for s in strips_here if s.kind == "column"})
                step = max(1, len(cols) // 16)
                for ci, col in enumerate(cols):
                    if ci % step != 0 and ci != len(cols) - 1:
                        continue
                    t = QGraphicsSimpleTextItem(str(col))
                    t.setFont(tick_font)
                    t.setBrush(QBrush(QColor("#888")))
                    t.setPos((col + 0.5) * sx - 5, cursor_y - 11)
                    t.setZValue(1)
                    self.addItem(t)
            elif row_count > 0:
                # Row ticks to the left of the rect
                rows = sorted({int(s.row) for s in strips_here if s.kind == "row"})
                step = max(1, len(rows) // 16)
                for ri, row in enumerate(rows):
                    if ri % step != 0 and ri != len(rows) - 1:
                        continue
                    t = QGraphicsSimpleTextItem(str(row))
                    t.setFont(tick_font)
                    t.setBrush(QBrush(QColor("#888")))
                    t.setPos(-16, cursor_y + (row + 0.5) * sy - 5)
                    t.setZValue(1)
                    self.addItem(t)

            cursor_y += disp_h + self.GROUP_GAP
            max_w = max(max_w, disp_w)

        # Strips: draw each strip's footprint on its group's canvas (scaled)
        for i, s in enumerate(doc.strips):
            origin = self._group_origin.get(s.group)
            if origin is None:
                continue   # strip references a non-existent group
            ox, oy, sx, sy = origin
            color = color_for_group(s.group, doc.groups)

            path = QPainterPath()
            length = max(int(s.length), 0)
            # Canvas range the strip actually occupies, derived from
            # direction-aware ``start`` and ``length``. ``a_start`` is
            # the chain-start (LED 0) end; ``a_end`` is the chain-end
            # (last LED) end.
            a_start = int(s.start)
            a_end = int(s.chain_end_pos())
            arrow = None
            if s.kind == "row" and length > 0:
                y = oy + (s.row + 0.5) * sy
                xs = ox + a_start * sx
                xe = ox + a_end * sx
                path.moveTo(xs, y)
                path.lineTo(xe, y)
                arrow = (xe, y, 1.0 if a_end >= a_start else -1.0, 0.0)
            elif s.kind == "column" and length > 0:
                x = ox + (s.col + 0.5) * sx
                ys = oy + a_start * sy
                ye = oy + a_end * sy
                path.moveTo(x, ys)
                path.lineTo(x, ye)
                arrow = (x, ye, 0.0, 1.0 if a_end >= a_start else -1.0)
            elif s.kind == "raw" and s.polyline:
                p0 = s.polyline[0]
                path.moveTo(ox + p0[0] * sx, oy + p0[1] * sy)
                for p in s.polyline[1:]:
                    path.lineTo(ox + p[0] * sx, oy + p[1] * sy)

            item = _GroupStripItem(i, path, color, s, arrow=arrow)
            self.addItem(item)
            self._strip_items[i] = item

        self.setSceneRect(QRectF(-30, -10, max_w + 60, cursor_y + 20))

    def highlight_strip(self, strip_idx: int):
        self.highlight_strips({strip_idx} if strip_idx >= 0 else set())

    def highlight_strips(self, indices):
        indices = set(indices) if indices else set()
        for idx, item in self._strip_items.items():
            if isinstance(item, _GroupStripItem):
                item.set_highlighted(idx in indices)
            else:
                pen = item.pen()
                pen.setWidthF(4.5 if idx in indices else 2.0)
                item.setPen(pen)

    def _on_strip_clicked(self, strip_idx: int):
        self.selection_changed_strip.emit(strip_idx)


class _GroupStripItem(QGraphicsPathItem):
    """Strip footprint on the group-canvas scene; clickable for selection.

    Carries a widened invisible hit area (via ``shape()``) so close-packed
    strip lines (e.g. Fan's 32 columns) stay individually pickable even
    when the visible stroke is thin. ``arrow`` (optional) draws a small
    filled triangle at the chain *end* (last LED) pointing in the
    direction the chain walks, giving an at-a-glance read on which way
    the LED data is flowing."""

    def __init__(self, strip_idx: int, path: QPainterPath, color: QColor,
                 strip: StripSpec, arrow: tuple | None = None):
        super().__init__(path)
        self.strip_idx = strip_idx
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._base_color = color
        self._base_width = 2.0
        self._hovered = False
        self._highlighted = False
        # Arrow: (head_x, head_y, dx, dy) in scene coords, or None.
        # head is the chain *end* position; (dx, dy) is unit-ish direction.
        self._arrow = arrow
        self._apply_pen()
        self.setZValue(5)
        if strip.kind == "row":
            pos_str = f"row={strip.row}"
        elif strip.kind == "column":
            pos_str = f"col={strip.col}"
        else:
            pos_str = "raw"
        self.setToolTip(
            f"{strip.group} strip_idx={strip.strip_idx}  kind={strip.kind}  "
            f"{pos_str}  start={strip.start}  len={strip.length}  "
            f"dir={strip.direction}\n"
            f"(arrow points to last LED; opposite end is data-in)"
        )

    def _apply_pen(self):
        if self._highlighted:
            color = QColor("#ffffff")
            width = 4.5
        elif self._hovered:
            color = self._base_color.lighter(160)
            width = 3.5
        else:
            color = self._base_color
            width = self._base_width
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def set_highlighted(self, on: bool):
        self._highlighted = on
        self._apply_pen()

    def hoverEnterEvent(self, event):
        self._hovered = True
        self._apply_pen()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._apply_pen()
        super().hoverLeaveEvent(event)

    def shape(self):
        from PyQt6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(8.0)
        return stroker.createStroke(self.path())

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self._arrow is None:
            return
        from PyQt6.QtGui import QPolygonF
        from PyQt6.QtCore import QPointF
        import math
        hx, hy, dx, dy = self._arrow
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        px, py = -uy, ux            # perpendicular for the wings
        # Slightly larger when highlighted so the selected strip's
        # direction stands out.
        size = 9.0 if self._highlighted else 7.0
        wing = size * 0.45
        # Tip is just past the line's end so the head doesn't bury
        # itself in the stroke.
        tip = QPointF(hx + ux * size * 0.55, hy + uy * size * 0.55)
        base_l = QPointF(hx - ux * size * 0.45 + px * wing,
                         hy - uy * size * 0.45 + py * wing)
        base_r = QPointF(hx - ux * size * 0.45 - px * wing,
                         hy - uy * size * 0.45 - py * wing)
        poly = QPolygonF([tip, base_l, base_r])
        painter.setBrush(self.pen().color())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(poly)

    def mousePressEvent(self, event):
        scene = self.scene()
        if isinstance(scene, GroupCanvasScene):
            scene._on_strip_clicked(self.strip_idx)
        super().mousePressEvent(event)


class LayoutView(QGraphicsView):
    """Canvas viewport with mouse-wheel zoom and middle-button pan."""

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#101010")))
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # Make the rubber-band rectangle clearly visible against the
        # dark background — the default thin grey line is barely
        # perceptible and led to operators thinking band-select wasn't
        # working.
        self.setStyleSheet(
            "QGraphicsView { selection-background-color: #4a90d9; }"
        )
        self.setRubberBandSelectionMode(
            Qt.ItemSelectionMode.IntersectsItemShape
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Undo commands
# ---------------------------------------------------------------------------

class _StripPolylineCmd(QUndoCommand):
    """Reversible polyline edit on a single strip. Captures the full
    polyline + shape (and the arc params if applicable) before/after,
    so undo restores both the geometry and the arc-derived flag.

    Used for: vertex drags (debounced via mouse-press/release on the
    handle), insert vertex, delete vertex, arc-dialog apply,
    convert-to-arc, convert-to-polyline. Anything that mutates a
    StripSpec's polyline-side fields routes through this command."""

    def __init__(self, editor, strip_idx, before, after, label,
                 from_drag: bool = False):
        super().__init__(label)
        self._editor = editor
        self._strip_idx = strip_idx
        self._before = before
        self._after = after
        # ``from_drag``: was this command pushed from a vertex/arc
        # drag-finished event? Drag paths already keep the canvas in
        # sync incrementally (update_strip_polyline / update_arc_handles)
        # so no rebuild is needed on the initial push — and crucially
        # rebuilding from inside the caller's mouseReleaseEvent
        # destroys the handle whose event is still on the stack, then
        # Qt crashes when control returns. Other paths (insert vertex,
        # delete vertex, arc dialog) DO need a rebuild to reflect the
        # structural change in the polyline; they pass ``False`` and
        # get a synchronous rebuild.
        self._initial = True
        self._from_drag = from_drag

    def _apply(self, snapshot: dict):
        if not (0 <= self._strip_idx < len(self._editor.doc.strips)):
            return
        s = self._editor.doc.strips[self._strip_idx]
        s.polyline = [list(p) for p in snapshot["polyline"]]
        s.shape = snapshot["shape"]
        s.arc_center = tuple(snapshot["arc_center"])
        s.arc_radius = float(snapshot["arc_radius"])
        s.arc_start_deg = float(snapshot["arc_start_deg"])
        s.arc_end_deg = float(snapshot["arc_end_deg"])
        s.arc_segments = int(snapshot["arc_segments"])
        self._editor._rebuild_scenes()
        self._editor._mark_dirty()

    def undo(self):
        self._apply(self._before)

    def redo(self):
        if self._initial:
            self._initial = False
            if self._from_drag:
                # Canvas already in sync from the live drag updates.
                # Nothing to draw; just register that the doc changed.
                self._editor._mark_dirty()
            else:
                # Polyline structure changed; rebuild now (we're not
                # inside an item event handler, so it's safe).
                self._apply(self._after)
            return
        self._apply(self._after)


def _strip_snapshot(s) -> dict:
    """Capture the polyline-side fields of a StripSpec for undo."""
    return {
        "polyline": [list(p) for p in s.polyline],
        "shape": s.shape,
        "arc_center": tuple(s.arc_center),
        "arc_radius": s.arc_radius,
        "arc_start_deg": s.arc_start_deg,
        "arc_end_deg": s.arc_end_deg,
        "arc_segments": s.arc_segments,
    }


class _ObjectMoveCmd(QUndoCommand):
    """Reversible drag of one box's (x, y) on the physical canvas.
    Pushed on mouseRelease so it covers the whole drag as a single
    undo step rather than one per pixel."""

    def __init__(self, editor, box_idx, before_xy, after_xy):
        super().__init__(f"Move object {box_idx}")
        self._editor = editor
        self._box_idx = box_idx
        self._before = before_xy
        self._after = after_xy
        # See _StripPolylineCmd — drags keep the canvas in sync
        # incrementally so no rebuild is needed on the initial push.
        self._initial = True

    def _apply(self, xy):
        if not (0 <= self._box_idx < len(self._editor.doc.boxes)):
            return
        b = self._editor.doc.boxes[self._box_idx]
        b.x, b.y = int(xy[0]), int(xy[1])
        self._editor.scene.update_object(self._box_idx, b)
        tl = self._editor.boxes_m.index(self._box_idx, 2)
        br = self._editor.boxes_m.index(self._box_idx, 3)
        self._editor.boxes_m.dataChanged.emit(tl, br)
        self._editor._mark_dirty()

    def undo(self):
        self._apply(self._before)

    def redo(self):
        if self._initial:
            self._initial = False
            self._editor._mark_dirty()
            return
        self._apply(self._after)


class _MultiObjectMoveCmd(QUndoCommand):
    """Reversible drag of *multiple* boxes simultaneously. Used when
    the operator rubber-band-selects several objects on the physical
    canvas and drags one of them — Qt moves the rest as a group, and
    we capture the before/after for every selected object so undo
    restores them all in one step."""

    def __init__(self, editor, moves: list):
        # moves = [(box_idx, before_xy, after_xy), ...]
        super().__init__(f"Move {len(moves)} objects")
        self._editor = editor
        self._moves = moves
        self._initial = True

    def _apply_to(self, key: str):
        # key is "before" or "after"; pick that field of each move
        idx = 1 if key == "before" else 2
        for m in self._moves:
            box_idx, _b, _a = m[0], m[1], m[2]
            xy = m[idx]
            if 0 <= box_idx < len(self._editor.doc.boxes):
                b = self._editor.doc.boxes[box_idx]
                b.x, b.y = int(xy[0]), int(xy[1])
                self._editor.scene.update_object(box_idx, b)
                tl = self._editor.boxes_m.index(box_idx, 2)
                br = self._editor.boxes_m.index(box_idx, 3)
                self._editor.boxes_m.dataChanged.emit(tl, br)
        self._editor._mark_dirty()

    def undo(self):
        self._apply_to("before")

    def redo(self):
        if self._initial:
            self._initial = False
            self._editor._mark_dirty()
            return
        self._apply_to("after")


class _GroupTranslateCmd(QUndoCommand):
    """Reversible drag of a *mixed* selection — any combination of
    objects and strips that were band-selected together. Each object
    contributes a (before, after) (x, y) pair; each strip contributes
    a full polyline+arc snapshot pair. Undo restores everything.

    Used when the operator rubber-bands a region containing more than
    one item type and drags one of them; with the editor owning the
    group-translate semantics, all selected items follow the primary
    by the same delta and undo restores them in one step."""

    def __init__(self, editor, obj_moves: list, strip_moves: list, label: str):
        # obj_moves   = [(box_idx, before_xy, after_xy), ...]
        # strip_moves = [(strip_idx, before_snap, after_snap), ...]
        super().__init__(label)
        self._editor = editor
        self._obj_moves = obj_moves
        self._strip_moves = strip_moves
        self._initial = True

    def _restore_objects(self, key: str):
        for m in self._obj_moves:
            box_idx = m[0]
            xy = m[1] if key == "before" else m[2]
            if not (0 <= box_idx < len(self._editor.doc.boxes)):
                continue
            b = self._editor.doc.boxes[box_idx]
            b.x, b.y = int(xy[0]), int(xy[1])
            self._editor.scene.update_object(box_idx, b)
            tl = self._editor.boxes_m.index(box_idx, 2)
            br = self._editor.boxes_m.index(box_idx, 3)
            self._editor.boxes_m.dataChanged.emit(tl, br)

    def _restore_strips(self, key: str):
        for m in self._strip_moves:
            strip_idx = m[0]
            snap = m[1] if key == "before" else m[2]
            if not (0 <= strip_idx < len(self._editor.doc.strips)):
                continue
            s = self._editor.doc.strips[strip_idx]
            s.polyline = [list(p) for p in snap["polyline"]]
            s.shape = snap["shape"]
            s.arc_center = tuple(snap["arc_center"])
            s.arc_radius = float(snap["arc_radius"])
            s.arc_start_deg = float(snap["arc_start_deg"])
            s.arc_end_deg = float(snap["arc_end_deg"])
            s.arc_segments = int(snap["arc_segments"])

    def undo(self):
        self._restore_objects("before")
        self._restore_strips("before")
        self._editor._rebuild_scenes()
        self._editor._mark_dirty()

    def redo(self):
        if self._initial:
            self._initial = False
            # Doc is already in after-state from the live drag.
            self._editor._mark_dirty()
            return
        self._restore_objects("after")
        self._restore_strips("after")
        self._editor._rebuild_scenes()
        self._editor._mark_dirty()


class EditorWindow(QMainWindow):
    def __init__(self, project_id: str):
        super().__init__()
        self.setWindowTitle(f"Layout Editor — {project_id}")
        self.resize(1400, 900)

        self.doc = load_doc(project_id)
        self._dirty = False

        # Undo stack — covers canvas-side ops (vertex drag, insert /
        # delete vertex, arc edits, object drags). Cell edits in the
        # tables aren't routed through it yet; that'd require wrapping
        # every model setData and is out of scope for this pass.
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(200)
        # Drag-state buffers populated by mousePress on _VertexHandle
        # and _ObjectItem; consumed on mouseRelease to construct an
        # undo command spanning the full drag.
        #
        # ``_drag_strip_pre`` holds vertex-drag pre-snapshots; the
        # group-drag buffers below collect simultaneously-dragged
        # objects + strips so a single rubber-band-then-drag moves
        # everything in the selection regardless of item type.
        self._drag_strip_pre: dict[int, dict] = {}
        self._drag_object_pre: dict[int, tuple] = {}
        # Group-drag participants captured at press time. ``_pre`` maps
        # box_idx → (x, y); ``_strip_pre`` maps strip_idx → full
        # polyline/arc snapshot. Both populated when group-drag starts
        # via either an _ObjectItem or _StripItem press.
        self._group_obj_pre: dict[int, tuple] = {}
        self._group_strip_pre: dict[int, dict] = {}
        self._drag_primary_obj_idx: int = -1
        self._drag_primary_strip_idx: int = -1
        # Authoritative record of which object/strip indices are
        # currently selected on the canvas. Maintained by listening
        # to ``scene.selectionChanged``.
        self._selected_obj_idxs: set[int] = set()
        self._selected_strip_idxs: set[int] = set()

        # Resolve the engine's web API port from config.yaml so REST
        # calls (project swap, weather state polling) reach the right
        # server. Falls back to the class-level default if config is
        # missing or malformed.
        try:
            with open(ROOT / "config.yaml", "r", encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f) or {}
            _port = (_cfg.get("web") or {}).get("port", 5000)
            self.WEB_API_BASE = f"http://127.0.0.1:{int(_port)}"
        except Exception:
            pass   # keep class default

        # Models. The Boxes model is the unified Object+Receiver pair —
        # each row represents one physical hardware unit (with optional
        # spatial position for multi_object projects).
        self.boxes_m = BoxesModel(self.doc)
        self.groups_m = GroupsModel(self.doc)
        self.strips_m = StripsModel(self.doc)

        # Wire layout-changed signals to canvas + dirty tracking
        for m in (self.boxes_m, self.groups_m, self.strips_m):
            m.layout_changed.connect(self._mark_dirty)
            m.layout_changed.connect(self._refresh_canvas)

        # Main physical-layout scene + view (composite-canvas world)
        self.scene = LayoutScene()
        self.scene.rebuild(self.doc)
        self.scene.object_moved.connect(self._on_object_moved_on_canvas)
        self.scene.vertex_moved.connect(self._on_vertex_moved_on_canvas)
        self.scene.selection_changed_strip.connect(
            self._on_strip_clicked_on_canvas
        )
        self.scene.strip_context_requested.connect(
            self._on_strip_context_menu)
        self.scene.vertex_context_requested.connect(
            self._on_vertex_context_menu)
        self.scene.vertex_drag_started.connect(self._on_vertex_drag_started)
        self.scene.vertex_drag_finished.connect(self._on_vertex_drag_finished)
        self.scene.object_drag_started.connect(self._on_object_drag_started)
        self.scene.object_drag_finished.connect(self._on_object_drag_finished)
        self.scene.arc_handle_moved.connect(self._on_arc_handle_moved)
        self.scene.strip_translate_started.connect(
            self._on_strip_translate_started)
        self.scene.strip_translated.connect(self._on_strip_translated)
        self.scene.strip_translate_finished.connect(
            self._on_strip_translate_finished)
        # Track scene selection independently of any specific event
        # callback so we can answer "which objects are selected?"
        # reliably at any point during a press / drag sequence.
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.view = LayoutView(self.scene)

        # Secondary group-canvas scene + view (logical rendering surfaces)
        self.group_scene = GroupCanvasScene()
        self.group_scene.rebuild(self.doc)
        self.group_scene.selection_changed_strip.connect(
            self._on_strip_clicked_on_canvas
        )
        self.group_view = LayoutView(self.group_scene)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_tab(self.boxes_m, with_add_remove=True), "Boxes")
        self.tabs.addTab(self._make_tab(self.groups_m, with_add_remove=True), "Groups")
        self.tabs.addTab(self._make_tab(self.strips_m, with_add_remove=True), "Strips")
        self.tabs.currentChanged.connect(self._tab_changed)

        # Layout: project picker on top, splitter (tabs | view) below
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Project:"))
        self.project_picker = QComboBox()
        for p in find_editable_projects():
            label = f"{p['display_name']}  ({p.get('geometry_type', '?')})"
            self.project_picker.addItem(label, p["id"])
        # Select active
        for i in range(self.project_picker.count()):
            if self.project_picker.itemData(i) == project_id:
                self.project_picker.setCurrentIndex(i)
                break
        self.project_picker.currentIndexChanged.connect(self._switch_project)
        top.addWidget(self.project_picker, 1)

        new_btn = QPushButton("+ New")
        new_btn.setToolTip("Scaffold a new project under projects/<id>/")
        new_btn.clicked.connect(self._on_new_project_clicked)
        top.addWidget(new_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        save_btn.setShortcut("Ctrl+S")
        top.addWidget(save_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        top.addWidget(reload_btn)

        outer.addLayout(top)

        # Emulator toolbar — Launch/Stop, weather set/state dropdowns,
        # connection indicator. Only enabled while the engine is running.
        self._build_emulator_toolbar(outer)

        # Right side: tabs for the two canvases so each gets the full
        # panel area when active. Stacked-splitter mode was cramped on
        # Fan (300-row group canvas) and on multi-object physical
        # layouts where the geometry needs room.
        self.canvas_tabs = QTabWidget()
        self.canvas_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.canvas_tabs.addTab(self.view, "Physical layout")
        self.canvas_tabs.addTab(self.group_view, "Group canvases")

        # Engine log panel (collapsed unless engine is running)
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(2000)
        f = QFont("Consolas", 8)
        self.log_panel.setFont(f)
        self.log_panel.setVisible(False)

        right_col = QSplitter(Qt.Orientation.Vertical)
        right_col.addWidget(self.canvas_tabs)
        right_col.addWidget(self.log_panel)
        right_col.setSizes([900, 0])
        self._right_col = right_col

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 880])
        outer.addWidget(splitter, 1)

        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_status()

        # Use QUndoStack's own factory actions for Undo / Redo. They
        # auto-enable/disable based on the stack's state and update
        # their menu text to reflect the next action ("Undo Move
        # vertex", etc.) so the operator can confirm something will
        # happen before pressing the shortcut.
        # NB: ``QKeySequence.StandardKey.Undo`` resolves to ``Ctrl+Z`` on
        # Windows/Linux and ``⌘Z`` on macOS, so we don't combine it with
        # an explicit ``Ctrl+Z`` — that yields duplicate registrations
        # and Qt logs "Ambiguous shortcut overload" each keypress. Use
        # the StandardKey alone for the platform-native binding, and
        # add only the *non-overlapping* extras explicitly.
        self.act_undo = self.undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.act_redo = self.undo_stack.createRedoAction(self, "&Redo")
        # Redo gets the platform default (Ctrl+Y on Win/Linux, ⇧⌘Z on
        # mac) plus Ctrl+Shift+Z which is widely expected on every
        # platform and doesn't collide with anything the standard
        # binding already covers on Windows.
        self.act_redo.setShortcuts([
            QKeySequence.StandardKey.Redo,
            QKeySequence("Ctrl+Shift+Z"),
        ])
        self.act_redo.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        # Adding to the window AND to a menubar makes the shortcuts
        # active regardless of which child widget currently has focus
        # — without this, a focused QTableView eats Ctrl+Z to do its
        # own in-cell editing undo, never reaching our stack.
        self.addAction(self.act_undo)
        self.addAction(self.act_redo)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)

        # Emulator runtime state. Started/stopped by toolbar buttons.
        self._engine_proc: Optional[QProcess] = None
        self._emu_client: Optional[EmulatorClient] = None
        self._emu_overlay: Optional[EmulatorOverlay] = None
        self._emu_port: int = 0
        # Sequence the overlay last painted, so the timer skips any
        # tick where the engine hasn't published a new frame.
        self._emu_last_seq: int = -1
        # 20 Hz frame poll. The render loop emits ~30 Hz but every
        # numpy-paint of Fan's 40k LEDs costs a few ms on the Qt main
        # thread; polling slower keeps the UI responsive without
        # noticeable visual lag (snapshot returns the *latest* frame).
        self._emu_timer = QTimer(self)
        self._emu_timer.setInterval(50)
        self._emu_timer.timeout.connect(self._on_emu_tick)
        # Slower (1 Hz) polling for connection state + weather sync.
        self._emu_state_timer = QTimer(self)
        self._emu_state_timer.setInterval(1000)
        self._emu_state_timer.timeout.connect(self._on_emu_state_poll)

        # Engine stdout buffer + flush timer. Engine logs can fire
        # many readyRead events per second on boot; appendPlainText
        # on the GUI thread per event is what stalled the UI in
        # earlier runs. We accumulate raw bytes here and flush them
        # to the log panel at 5 Hz so the GUI sees one repaint per
        # 200 ms regardless of engine verbosity.
        self._log_buffer = bytearray()
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(200)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)

        # Worker-thread → GUI-thread signal bridges (urllib runs off
        # the main thread to keep the UI responsive).
        self.weather_info_received.connect(self._on_weather_info)
        self.post_failed.connect(self._append_log)

    def _build_emulator_toolbar(self, outer_layout: QVBoxLayout):
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)

        self.run_btn = QPushButton("▶ Run engine")
        self.run_btn.clicked.connect(self._toggle_engine)
        bar.addWidget(self.run_btn)

        self._engine_status = QLabel("Engine: stopped")
        self._engine_status.setStyleSheet("color: #888; padding: 0 8px;")
        bar.addWidget(self._engine_status)

        bar.addWidget(QLabel("Weather set:"))
        self.weather_set_picker = QComboBox()
        self.weather_set_picker.setEnabled(False)
        self.weather_set_picker.setMinimumWidth(140)
        self.weather_set_picker.activated.connect(self._on_pick_weather_set)
        bar.addWidget(self.weather_set_picker)

        bar.addWidget(QLabel("State:"))
        self.weather_state_picker = QComboBox()
        self.weather_state_picker.setEnabled(False)
        self.weather_state_picker.setMinimumWidth(180)
        self.weather_state_picker.activated.connect(self._on_pick_weather_state)
        bar.addWidget(self.weather_state_picker)

        self.live_preview_chk = QPushButton("Live preview")
        self.live_preview_chk.setCheckable(True)
        self.live_preview_chk.setChecked(True)
        self.live_preview_chk.toggled.connect(self._on_toggle_live_preview)
        self.live_preview_chk.setEnabled(False)
        bar.addWidget(self.live_preview_chk)

        bar.addStretch(1)

        self.show_log_btn = QPushButton("Show log")
        self.show_log_btn.setCheckable(True)
        self.show_log_btn.toggled.connect(self._on_toggle_log)
        bar.addWidget(self.show_log_btn)

        outer_layout.addLayout(bar)

    def _wrap_panel(self, title: str, view: QWidget) -> QWidget:
        """Add a small italic header above a QGraphicsView."""
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QLabel(title)
        f = QFont()
        f.setPointSize(9)
        f.setItalic(True)
        header.setFont(f)
        header.setStyleSheet("color: #888; padding: 2px 4px;")
        layout.addWidget(header, 0)
        layout.addWidget(view, 1)
        return wrap

    def _make_tab(self, model: _BaseModel, with_add_remove: bool) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)

        view = QTableView()
        view.setModel(model)
        view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        view.setEditTriggers(
            QTableView.EditTrigger.DoubleClicked
            | QTableView.EditTrigger.SelectedClicked
            | QTableView.EditTrigger.EditKeyPressed
        )
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        view.horizontalHeader().setStretchLastSection(True)
        view.setAlternatingRowColors(True)
        view.selectionModel().selectionChanged.connect(
            lambda *_: self._on_table_selection_changed(model, view)
        )

        # Per-column dropdown delegates for enum-like / cross-referenced cells.
        # Doc-cross-referencing delegates take a callable so they always
        # read self.doc *now*, not whatever doc happened to be active when
        # the delegate was constructed (matters across project switches).
        if isinstance(model, StripsModel):
            view.setItemDelegateForColumn(StripsModel.COL_GROUP, _GroupDelegate(lambda: self.doc, view))
            view.setItemDelegateForColumn(StripsModel.COL_KIND,
                _ChoiceDelegate(["row", "column", "raw"], view))
            view.setItemDelegateForColumn(StripsModel.COL_DIRECTION,
                _ChoiceDelegate(["right", "left", "down", "up"], view))
            view.setItemDelegateForColumn(StripsModel.COL_RECEIVER,
                _ReceiverDelegate(lambda: self.doc, view))
            # Tighten column widths for fields that hold short content
            # (single integers / 2-letter direction codes / one-word
            # kinds) so the wider ``group``, ``receiver``, and
            # ``polyline`` columns get more visual breathing room.
            view.horizontalHeader().setStretchLastSection(False)
            tight = {
                StripsModel.COL_STRIP_IDX: 60,
                StripsModel.COL_KIND:      60,
                StripsModel.COL_CANVAS_POS: 80,
                StripsModel.COL_START:     56,
                StripsModel.COL_LENGTH:    62,
                StripsModel.COL_DIRECTION: 70,
            }
            for col, width in tight.items():
                view.setColumnWidth(col, width)
            # Group + receiver get more room; polyline summary stretches
            view.setColumnWidth(StripsModel.COL_GROUP, 78)
            view.setColumnWidth(StripsModel.COL_RECEIVER, 180)
            view.horizontalHeader().setSectionResizeMode(
                StripsModel.COL_POLYLINE,
                QHeaderView.ResizeMode.Stretch,
            )
        elif isinstance(model, BoxesModel):
            view.setItemDelegateForColumn(6, _ChoiceDelegate(["sacn", "ddp"], view))  # protocol

        layout.addWidget(view, 1)

        # Stash the view on the model so we can find it later
        model._view = view  # type: ignore[attr-defined]

        if with_add_remove:
            row = QHBoxLayout()
            add = QPushButton("+ Add")
            add.clicked.connect(lambda: self._add_row(model, view))
            rem = QPushButton("− Remove selected")
            rem.clicked.connect(lambda: self._remove_selected(model, view))
            row.addWidget(add)
            row.addWidget(rem)
            if isinstance(model, StripsModel):
                fix = QPushButton("Fix indexes & canvas pos")
                fix.setToolTip(
                    "Sort by receiver, renumber strip_idx densely per "
                    "receiver, and reassign each group's row/col to "
                    "0..N-1 in receiver/wire order."
                )
                fix.clicked.connect(
                    lambda _=False, m=model: m.fix_strip_indexes_and_canvas_pos()
                )
                row.addWidget(fix)
            row.addStretch(1)
            layout.addLayout(row)

        return wrap

    # ----- editing helpers -----
    def _add_row(self, model: _BaseModel, view: QTableView):
        new_row = model.add_row()
        view.selectRow(new_row)

    def _remove_selected(self, model: _BaseModel, view: QTableView):
        rows = sorted({i.row() for i in view.selectionModel().selectedRows()})
        if rows:
            model.remove_rows(rows)

    def _on_table_selection_changed(self, model: _BaseModel, view: QTableView):
        rows = view.selectionModel().selectedRows()
        if isinstance(model, BoxesModel):
            row = rows[0].row() if rows else -1
            self.scene.highlight_object(row)
            self.scene.highlight_strip(-1)
            self.group_scene.highlight_strip(-1)
        elif isinstance(model, StripsModel):
            rows_set = {r.row() for r in rows}
            self.scene.highlight_strips(rows_set)
            self.scene.highlight_object(-1)
            self.group_scene.highlight_strips(rows_set)
        else:
            self.scene.highlight_object(-1)
            self.scene.highlight_strip(-1)
            self.group_scene.highlight_strip(-1)

    def _on_strip_clicked_on_canvas(self, strip_idx: int):
        """User clicked a strip on either canvas. Switch to the Strips
        tab and select that row — which in turn highlights it on both
        canvases and reveals its polyline vertex handles.

        If the clicked strip is already in the table's multi-selection,
        leave the selection alone so the subsequent drag can translate
        the whole group together."""
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        self.tabs.setCurrentIndex(2)   # Boxes(0), Groups(1), Strips(2)
        view = getattr(self.strips_m, "_view", None)
        if view is None:
            return
        selected_rows = {i.row() for i in view.selectionModel().selectedRows()}
        if strip_idx in selected_rows:
            return
        view.selectRow(strip_idx)

    # ----- canvas → table sync -----
    def _on_object_moved_on_canvas(self, obj_idx: int, x: int, y: int):
        if not (0 <= obj_idx < len(self.doc.boxes)):
            return
        self.doc.boxes[obj_idx].x = x
        self.doc.boxes[obj_idx].y = y
        tl = self.boxes_m.index(obj_idx, 2)
        br = self.boxes_m.index(obj_idx, 3)
        self.boxes_m.dataChanged.emit(tl, br)
        self.scene.update_object(obj_idx, self.doc.boxes[obj_idx])

        # Group drag: if the primary object moves and we have a
        # multi-item group captured (objects + strips), propagate the
        # delta to all other group participants.
        if (self._drag_primary_obj_idx == obj_idx
                and (len(self._group_obj_pre) + len(self._group_strip_pre)) > 1):
            primary_pre = self._group_obj_pre.get(obj_idx)
            if primary_pre is not None:
                dx = x - primary_pre[0]
                dy = y - primary_pre[1]
                self._apply_group_delta(dx, dy, skip_obj_idx=obj_idx)
        self._mark_dirty()

    def _on_vertex_moved_on_canvas(self, strip_idx: int, vertex_idx: int,
                                   x: int, y: int):
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        s = self.doc.strips[strip_idx]
        if not (0 <= vertex_idx < len(s.polyline)):
            return
        s.polyline[vertex_idx] = [x, y]
        # Dragging a vertex on an arc-shaped strip means the user is
        # explicitly reshaping it as a polyline; flip the shape so
        # subsequent loads keep their hand-edits intact.
        if s.shape == "arc":
            s.shape = "polyline"
        # Redraw the strip's path. Pass the vertex_idx so the scene
        # doesn't reposition the handle the user is actively dragging.
        self.scene.update_strip_polyline(strip_idx, s.polyline,
                                         skip_handle_at=vertex_idx)
        # Refresh the polyline summary cell in the strips table.
        idx = self.strips_m.index(strip_idx, StripsModel.COL_POLYLINE)
        self.strips_m.dataChanged.emit(idx, idx)
        self._mark_dirty()

    def _on_arc_handle_moved(self, strip_idx: int, role: str,
                             x: int, y: int):
        """Recompute arc params from the dragged handle's new position
        and regenerate the strip's polyline. Other handles get
        repositioned to track the new shape; the dragged handle is
        skipped to avoid feedback loops."""
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        s = self.doc.strips[strip_idx]
        if s.shape != "arc":
            return
        import math
        cx, cy = s.arc_center
        if role == "center":
            # Pan: shift center, keep radius and angles.
            s.arc_center = (int(x), int(y))
        elif role == "start":
            # New start angle (canvas y inverted), radius unchanged.
            dx = x - cx
            dy = y - cy
            s.arc_start_deg = math.degrees(math.atan2(-dy, dx))
        elif role == "end":
            dx = x - cx
            dy = y - cy
            s.arc_end_deg = math.degrees(math.atan2(-dy, dx))
        elif role == "radius":
            # New radius from drag distance to center (min 1 to avoid
            # the arc collapsing to a point and breaking handle math).
            r = math.hypot(x - cx, y - cy)
            s.arc_radius = max(1.0, r)

        s.polyline = _arc_to_polyline(
            s.arc_center, s.arc_radius,
            s.arc_start_deg, s.arc_end_deg, s.arc_segments,
        )
        # Update strip path on canvas; skip vertex-handle reposition
        # entirely (arc strips have no vertex handles).
        self.scene._strips[strip_idx].set_points(s.polyline)
        # Reposition the other arc handles so they track the new shape.
        self.scene.update_arc_handles(
            strip_idx,
            float(s.arc_center[0]), float(s.arc_center[1]),
            float(s.arc_radius),
            float(s.arc_start_deg), float(s.arc_end_deg),
            except_role=role,
        )
        # Refresh polyline summary cell.
        idx = self.strips_m.index(strip_idx, StripsModel.COL_POLYLINE)
        self.strips_m.dataChanged.emit(idx, idx)
        self._mark_dirty()

    # ----- drag undo: capture pre-state on press, push command on release -----
    def _on_vertex_drag_started(self, strip_idx: int):
        if 0 <= strip_idx < len(self.doc.strips):
            self._drag_strip_pre[strip_idx] = _strip_snapshot(
                self.doc.strips[strip_idx]
            )

    def _on_vertex_drag_finished(self, strip_idx: int):
        before = self._drag_strip_pre.pop(strip_idx, None)
        if before is None or not (0 <= strip_idx < len(self.doc.strips)):
            return
        after = _strip_snapshot(self.doc.strips[strip_idx])
        # Skip no-op drags (click without movement).
        if before["polyline"] == after["polyline"] and \
                before["shape"] == after["shape"]:
            return
        # ``from_drag`` skips the redundant scene rebuild at push time
        # — the live drag updates already kept the canvas in sync.
        self.undo_stack.push(_StripPolylineCmd(
            self, strip_idx, before, after, "Move vertex", from_drag=True))

    def _on_scene_selection_changed(self):
        """Refresh selection trackers (objects + strips) whenever the
        canvas selection changes."""
        obj_idxs = set()
        strip_idxs = set()
        for item in self.scene.selectedItems():
            if isinstance(item, _ObjectItem):
                obj_idxs.add(item.obj_idx)
            elif isinstance(item, _StripItem):
                strip_idxs.add(item.strip_idx)
        self._selected_obj_idxs = obj_idxs
        self._selected_strip_idxs = strip_idxs

    def _capture_group_drag(self, primary_kind: str, primary_idx: int):
        """Snapshot every selected object + strip's pre-drag state so
        a press anywhere in the selection drags everything together.

        Called from both ``_on_object_drag_started`` (press on an
        object dot) and ``_on_strip_translate_started`` (press on a
        strip path body). Uses the union of the tracker (updated via
        selectionChanged) and a live scene query so neither path is
        missed by signal-ordering quirks across Qt versions."""
        live_objs = {item.obj_idx for item in self.scene.selectedItems()
                     if isinstance(item, _ObjectItem)}
        live_strips = {item.strip_idx for item in self.scene.selectedItems()
                       if isinstance(item, _StripItem)}
        all_objs = self._selected_obj_idxs | live_objs
        all_strips = self._selected_strip_idxs | live_strips

        # If the primary is in the multi-selection, the drag is a
        # group-drag — capture everyone. Otherwise, solo drag.
        primary_in_group = (
            (primary_kind == "object" and primary_idx in all_objs)
            or (primary_kind == "strip" and primary_idx in all_strips)
        )
        total_selected = len(all_objs) + len(all_strips)
        if primary_in_group and total_selected > 1:
            obj_participants = all_objs
            strip_participants = all_strips
        else:
            obj_participants = {primary_idx} if primary_kind == "object" else set()
            strip_participants = {primary_idx} if primary_kind == "strip" else set()

        self._group_obj_pre = {}
        for i in obj_participants:
            if 0 <= i < len(self.doc.boxes):
                b = self.doc.boxes[i]
                self._group_obj_pre[i] = (b.x, b.y)
        self._group_strip_pre = {}
        for i in strip_participants:
            if 0 <= i < len(self.doc.strips):
                self._group_strip_pre[i] = _strip_snapshot(self.doc.strips[i])

        if primary_kind == "object":
            self._drag_primary_obj_idx = primary_idx
            self._drag_primary_strip_idx = -1
        else:
            self._drag_primary_strip_idx = primary_idx
            self._drag_primary_obj_idx = -1

        n = len(self._group_obj_pre) + len(self._group_strip_pre)
        if n > 1:
            obj_n = len(self._group_obj_pre)
            strip_n = len(self._group_strip_pre)
            parts = []
            if obj_n: parts.append(f"{obj_n} object{'s' if obj_n != 1 else ''}")
            if strip_n: parts.append(f"{strip_n} strip{'s' if strip_n != 1 else ''}")
            self.status.showMessage(f"Group drag: {' + '.join(parts)}", 3000)

    def _apply_group_delta(self, dx: int, dy: int,
                           skip_obj_idx: int = -1,
                           skip_strip_idx: int = -1):
        """Apply ``(dx, dy)`` to every captured group-drag participant
        EXCEPT the primary (whose move triggered this propagation).
        Computes each participant's target as ``pre + delta`` so live
        drags track the cursor without rounding drift."""
        for i, pre_xy in self._group_obj_pre.items():
            if i == skip_obj_idx:
                continue
            if not (0 <= i < len(self.doc.boxes)):
                continue
            target_x = int(pre_xy[0] + dx)
            target_y = int(pre_xy[1] + dy)
            b = self.doc.boxes[i]
            if (b.x, b.y) == (target_x, target_y):
                continue
            b.x = target_x
            b.y = target_y
            if i < len(self.scene._objects):
                item = self.scene._objects[i]
                item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                    False,
                )
                item.setPos(target_x, target_y)
                item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                    True,
                )
            self.scene.update_object(i, b)
            tl_i = self.boxes_m.index(i, 2)
            br_i = self.boxes_m.index(i, 3)
            self.boxes_m.dataChanged.emit(tl_i, br_i)

        for i, pre in self._group_strip_pre.items():
            if i == skip_strip_idx:
                continue
            if not (0 <= i < len(self.doc.strips)):
                continue
            s = self.doc.strips[i]
            s.polyline = [
                [int(p[0]) + dx, int(p[1]) + dy] for p in pre["polyline"]
            ]
            if s.shape == "arc":
                s.arc_center = (
                    int(pre["arc_center"][0]) + dx,
                    int(pre["arc_center"][1]) + dy,
                )
            self.scene.update_strip_polyline(i, s.polyline)
            if s.shape == "arc":
                self.scene.update_arc_handles(
                    i,
                    float(s.arc_center[0]), float(s.arc_center[1]),
                    float(s.arc_radius),
                    float(s.arc_start_deg), float(s.arc_end_deg),
                )

    def _on_object_drag_started(self, obj_idx: int):
        self._capture_group_drag("object", obj_idx)
        # Backwards compat: keep the old _drag_object_pre populated
        # so legacy paths (single-object drag undo) still work.
        self._drag_object_pre = dict(self._group_obj_pre)

    # ----- strip-line translation (drag the path body, not a handle) -----
    def _on_strip_translate_started(self, strip_idx: int):
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        # Capture the entire selection (mixed types) so dragging this
        # strip translates any other selected strips and objects too.
        self._capture_group_drag("strip", strip_idx)

    def _on_strip_translated(self, strip_idx: int, dx: int, dy: int):
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        # Apply delta to the primary first, then propagate.
        primary_pre = self._group_strip_pre.get(strip_idx)
        if primary_pre is None:
            return
        s = self.doc.strips[strip_idx]
        s.polyline = [
            [int(p[0]) + dx, int(p[1]) + dy] for p in primary_pre["polyline"]
        ]
        if s.shape == "arc":
            s.arc_center = (
                int(primary_pre["arc_center"][0]) + dx,
                int(primary_pre["arc_center"][1]) + dy,
            )
        self.scene.update_strip_polyline(strip_idx, s.polyline)
        if s.shape == "arc":
            self.scene.update_arc_handles(
                strip_idx,
                float(s.arc_center[0]), float(s.arc_center[1]),
                float(s.arc_radius),
                float(s.arc_start_deg), float(s.arc_end_deg),
            )
        # Propagate to the other group participants (other strips +
        # any selected objects) — same delta from each one's pre.
        if (len(self._group_obj_pre) + len(self._group_strip_pre)) > 1:
            self._apply_group_delta(dx, dy, skip_strip_idx=strip_idx)

    def _on_strip_translate_finished(self, strip_idx: int):
        self._finalize_group_drag(f"Move strip {strip_idx}")
        self._drag_primary_strip_idx = -1

    def _on_object_drag_finished(self, obj_idx: int):
        self._finalize_group_drag("Move group")
        self._drag_object_pre = {}
        self._drag_primary_obj_idx = -1

    def _finalize_group_drag(self, default_label: str):
        """Build before/after move lists for every captured object +
        strip, then push the right undo command.

        Pushed command:
          * 1 object only      → ``_ObjectMoveCmd``
          * many objects only  → ``_MultiObjectMoveCmd``
          * any strips involved or mixed types → ``_GroupTranslateCmd``
        """
        obj_pre = self._group_obj_pre
        strip_pre = self._group_strip_pre
        self._group_obj_pre = {}
        self._group_strip_pre = {}
        if not obj_pre and not strip_pre:
            return
        obj_moves: list = []
        for i, before_xy in obj_pre.items():
            if not (0 <= i < len(self.doc.boxes)):
                continue
            b = self.doc.boxes[i]
            after_xy = (b.x, b.y)
            if before_xy != after_xy:
                obj_moves.append((i, before_xy, after_xy))
        strip_moves: list = []
        for i, before_snap in strip_pre.items():
            if not (0 <= i < len(self.doc.strips)):
                continue
            after_snap = _strip_snapshot(self.doc.strips[i])
            if before_snap["polyline"] != after_snap["polyline"] or \
                    before_snap.get("arc_center") != after_snap.get("arc_center"):
                strip_moves.append((i, before_snap, after_snap))
        if not obj_moves and not strip_moves:
            return
        if strip_moves:
            # Mixed group — single unified undo command.
            label = (
                f"Move group ({len(obj_moves)} obj, {len(strip_moves)} strip)"
                if obj_moves else
                f"Move {len(strip_moves)} strip{'s' if len(strip_moves) != 1 else ''}"
            )
            self.undo_stack.push(_GroupTranslateCmd(
                self, obj_moves, strip_moves, label
            ))
        elif len(obj_moves) == 1:
            i, b_xy, a_xy = obj_moves[0]
            self.undo_stack.push(_ObjectMoveCmd(self, i, b_xy, a_xy))
        else:
            self.undo_stack.push(_MultiObjectMoveCmd(self, obj_moves))

    # ----- right-click on strip / vertex (canvas) -----
    def _on_strip_context_menu(self, strip_idx: int,
                               scene_x: int, scene_y: int,
                               screen_x: int, screen_y: int):
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        s = self.doc.strips[strip_idx]
        is_multi_object = (self.doc.geometry_type == "multi_object")
        menu = QMenu(self)
        # Add-vertex options extend the polyline at one of its
        # endpoints; the user adjusts that vertex (and any existing
        # ones) by dragging. Splitting a segment in the middle is
        # rarely what's wanted when building a path, so we don't
        # offer it here.
        if is_multi_object and s.shape == "polyline":
            act_add_end = menu.addAction("Add vertex at end (extend chain)")
            act_add_start = menu.addAction("Add vertex at start")
            act_split = menu.addAction("Split segment at click")
        else:
            act_add_end = act_add_start = act_split = None
        if is_multi_object:
            if s.shape == "arc":
                act_arc = menu.addAction("Edit arc parameters…")
                act_to_poly = menu.addAction("Convert to polyline")
            else:
                act_arc = menu.addAction("Convert to arc…")
                act_to_poly = None
        else:
            act_arc = act_to_poly = None
        chosen = menu.exec(QPointF(screen_x, screen_y).toPoint())
        if chosen is None:
            return
        if chosen is act_add_end:
            self._extend_polyline(strip_idx, scene_x, scene_y, at_end=True)
        elif chosen is act_add_start:
            self._extend_polyline(strip_idx, scene_x, scene_y, at_end=False)
        elif chosen is act_split:
            self._split_segment_on_strip(strip_idx, scene_x, scene_y)
        elif chosen is act_arc:
            self._show_arc_dialog(strip_idx)
        elif chosen is act_to_poly:
            before = _strip_snapshot(s)
            s.shape = "polyline"
            after = _strip_snapshot(s)
            self.undo_stack.push(_StripPolylineCmd(
                self, strip_idx, before, after, "Convert to polyline"))

    def _on_vertex_context_menu(self, strip_idx: int, vertex_idx: int,
                                screen_x: int, screen_y: int):
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        s = self.doc.strips[strip_idx]
        if s.shape == "arc":
            return   # vertices on an arc are derived; no per-vertex ops
        if len(s.polyline) <= 2:
            return   # need at least 2 vertices remaining
        menu = QMenu(self)
        act_del = menu.addAction(f"Delete vertex {vertex_idx}")
        chosen = menu.exec(QPointF(screen_x, screen_y).toPoint())
        if chosen is act_del:
            self._delete_vertex_on_strip(strip_idx, vertex_idx)

    def _extend_polyline(self, strip_idx: int,
                         scene_x: int, scene_y: int,
                         at_end: bool):
        """Add a new vertex at one end of the strip's polyline,
        extending the path with a new segment. ``at_end=True``
        appends after the last vertex (chain-end side); ``False``
        prepends before the first (chain-start side). The new
        vertex is placed at the click position; the user can then
        drag any vertex to fine-tune."""
        s = self.doc.strips[strip_idx]
        before = _strip_snapshot(s)
        new_vertex = [int(scene_x), int(scene_y)]
        if at_end:
            s.polyline.append(new_vertex)
            label = "Add vertex (end)"
        else:
            s.polyline.insert(0, new_vertex)
            label = "Add vertex (start)"
        after = _strip_snapshot(s)
        self.undo_stack.push(_StripPolylineCmd(
            self, strip_idx, before, after, label))

    def _split_segment_on_strip(self, strip_idx: int,
                                scene_x: int, scene_y: int):
        """Insert a vertex *into* the existing path at the segment
        nearest the click — useful when the operator wants to refine
        an existing curve rather than extend it."""
        s = self.doc.strips[strip_idx]
        if len(s.polyline) < 2:
            return
        before = _strip_snapshot(s)
        best = (float("inf"), 1)
        for i in range(len(s.polyline) - 1):
            ax, ay = s.polyline[i]
            bx, by = s.polyline[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = max(dx * dx + dy * dy, 1e-6)
            t = ((scene_x - ax) * dx + (scene_y - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            fx = ax + t * dx
            fy = ay + t * dy
            d = (scene_x - fx) ** 2 + (scene_y - fy) ** 2
            if d < best[0]:
                best = (d, i + 1)
        s.polyline.insert(best[1], [int(scene_x), int(scene_y)])
        after = _strip_snapshot(s)
        self.undo_stack.push(_StripPolylineCmd(
            self, strip_idx, before, after, "Split segment"))

    def _delete_vertex_on_strip(self, strip_idx: int, vertex_idx: int):
        s = self.doc.strips[strip_idx]
        if not (0 <= vertex_idx < len(s.polyline)):
            return
        if len(s.polyline) <= 2:
            return
        before = _strip_snapshot(s)
        s.polyline.pop(vertex_idx)
        after = _strip_snapshot(s)
        self.undo_stack.push(_StripPolylineCmd(
            self, strip_idx, before, after, "Delete vertex"))

    def _show_arc_dialog(self, strip_idx: int):
        s = self.doc.strips[strip_idx]
        # Seed defaults from the current polyline endpoints if the
        # strip isn't already an arc — center at midpoint, radius
        # half the chord, end-angle a half-circle from start.
        if s.shape != "arc":
            if len(s.polyline) >= 2:
                p0 = s.polyline[0]; p1 = s.polyline[-1]
                cx = (p0[0] + p1[0]) // 2
                cy = (p0[1] + p1[1]) // 2
                import math
                r = max(20.0, math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 2)
                start_deg = 180.0
                end_deg = 0.0
                segments = max(8, len(s.polyline))
            else:
                cx = self.doc.canvas_w // 2
                cy = self.doc.canvas_h // 2
                r = 100.0
                start_deg = 0.0
                end_deg = 180.0
                segments = 16
            s.arc_center = (cx, cy)
            s.arc_radius = r
            s.arc_start_deg = start_deg
            s.arc_end_deg = end_deg
            s.arc_segments = segments

        before = _strip_snapshot(s)
        dlg = _ArcParamDialog(s, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # Apply the dialog's values back to the strip and regenerate
        # its polyline from scratch.
        dlg.apply_to(s)
        s.shape = "arc"
        s.polyline = _arc_to_polyline(
            s.arc_center, s.arc_radius,
            s.arc_start_deg, s.arc_end_deg, s.arc_segments,
        )
        after = _strip_snapshot(s)
        self.undo_stack.push(_StripPolylineCmd(
            self, strip_idx, before, after, "Edit arc"))
        # Reveal the arc handles right away — without this the operator
        # has to find the strip's path on canvas and click it before
        # the handles appear, which is fiddly on small arcs.
        self._select_strip_in_table(strip_idx)

    def _select_strip_in_table(self, strip_idx: int):
        """Switch to the Strips tab and select that strip's row,
        which trips ``_on_table_selection_changed`` and reveals the
        strip's handles on both canvases."""
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        self.tabs.setCurrentIndex(2)
        view = getattr(self.strips_m, "_view", None)
        if view is not None:
            view.selectRow(strip_idx)

    # ----- project switching -----
    def _on_new_project_clicked(self) -> None:
        """Pop the new-project dialog, scaffold on accept, then switch
        the picker (and editor) onto the freshly-created project. Logs
        a 'now do these next' checklist into the engine log panel so the
        operator knows what's still hand-edit territory after scaffold."""
        dlg = _NewProjectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        spec = dlg.result_spec()
        try:
            root = scaffold_new_project(spec)
        except FileExistsError as e:
            QMessageBox.warning(self, "Project exists", str(e))
            return
        except OSError as e:
            QMessageBox.critical(
                self, "Failed to create project",
                f"Couldn't write project files:\n{e}",
            )
            return

        new_id = spec["id"]

        # Refresh picker from disk so the freshly-scaffolded project
        # shows up. Block signals while we rebuild so the index churn
        # doesn't trigger _switch_project mid-rebuild.
        self.project_picker.blockSignals(True)
        self.project_picker.clear()
        target_index = 0
        for i, p in enumerate(find_editable_projects()):
            label = f"{p['display_name']}  ({p.get('geometry_type', '?')})"
            self.project_picker.addItem(label, p["id"])
            if p["id"] == new_id:
                target_index = i
        self.project_picker.setCurrentIndex(target_index)
        self.project_picker.blockSignals(False)

        # Now actually load the new project into the editor.
        self._reload(project_id=new_id)
        if self._engine_proc is not None:
            self._post_json_bg("/api/project/change", {"project_id": new_id})

        # Surface a next-steps checklist in the log panel — scaffold
        # writes empty receivers/event_map/weather_sets, so the operator
        # still has hand-edits to do before the project actually runs.
        self.log_panel.appendPlainText(
            f"[scaffold] Created projects/{new_id}/ at {root}\n"
            f"[scaffold] Next steps:\n"
            f"  1. Add receivers (Boxes tab) with their IPs/protocols\n"
            f"  2. Add groups (Groups tab) sized to your physical strips\n"
            f"  3. Add strips (Strips tab) and bind them to receivers\n"
            f"  4. Edit projects/{new_id}/event_map.py to register effects\n"
            f"  5. Edit projects/{new_id}/weather_params.py to define states\n"
            f"  6. Drop ambient audio into projects/{new_id}/media/sounds/\n"
        )
        self.log_panel.setVisible(True)
        self._right_col.setSizes([550, 200])

    def _switch_project(self, idx: int):
        new_id = self.project_picker.itemData(idx)
        if new_id == self.doc.project_id:
            return
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved edits in the current project. "
                "Save before switching?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                # Restore picker to current project
                for i in range(self.project_picker.count()):
                    if self.project_picker.itemData(i) == self.doc.project_id:
                        self.project_picker.blockSignals(True)
                        self.project_picker.setCurrentIndex(i)
                        self.project_picker.blockSignals(False)
                        break
                return
            if ans == QMessageBox.StandardButton.Save:
                self.save()
        self._reload(project_id=new_id)
        # If the engine is running, ask it to swap too — otherwise the
        # editor would be showing a layout that doesn't match the
        # frames the engine is broadcasting (different group ids).
        if self._engine_proc is not None:
            self._post_json_bg("/api/project/change", {"project_id": new_id})

    def _reload(self, project_id: Optional[str] = None):
        pid = project_id or self.doc.project_id
        self.doc = load_doc(pid)
        self.boxes_m.doc = self.doc
        self.groups_m.doc = self.doc
        self.strips_m.doc = self.doc
        for m in (self.boxes_m, self.groups_m, self.strips_m):
            m.beginResetModel()
            m.endResetModel()
        self.setWindowTitle(f"Layout Editor — {pid}")
        self._rebuild_scenes()    # overlay-safe rebuild
        self._dirty = False
        self._update_status()

    # ----- canvas refresh -----
    def _refresh_canvas(self):
        self._rebuild_scenes()

    def _rebuild_scenes(self):
        """Rebuild both scenes safely while the emulator overlay may be
        active. ``QGraphicsScene.clear()`` destroys *every* item the
        scene owns, including the overlay's pixmap items — leaving the
        Python wrappers pointing at deleted C++ objects, so the next
        ``setPixmap()`` raises RuntimeError. Drop the overlay first,
        then rebuild, then construct a fresh overlay against the new
        scene items. Used by every code path that calls scene.rebuild()
        (model edits, Reload, project switch)."""
        overlay_alive = self._emu_overlay is not None
        overlay_visible = False
        if overlay_alive:
            try:
                overlay_visible = self._emu_overlay._phys_pixmap_item.isVisible()
            except RuntimeError:
                overlay_visible = self.live_preview_chk.isChecked()
            self._emu_overlay = None
            # Reset seq so the next tick definitely repaints rather
            # than thinking "same frame, skip" against the new overlay.
            self._emu_last_seq = -1

        self.scene.rebuild(self.doc)
        self.group_scene.rebuild(self.doc)

        if overlay_alive:
            self._emu_overlay = EmulatorOverlay(
                self.doc, self.scene, self.group_scene
            )
            self._emu_overlay.set_visible(overlay_visible)
            self._set_strip_visuals_visible(not overlay_visible)
            self._set_object_visuals_visible(not overlay_visible)

        # Restore the highlighted strip's handles after the rebuild.
        # The Strips-table selection survives (it lives outside the
        # QGraphicsScene), but the highlight state inside the scene
        # was wiped along with the items. Without this re-application,
        # an undo or any model edit drops the visible handles even
        # though the row is still selected — which looked to the
        # operator like "undo doesn't work" or "I can't edit this
        # strip after clicking another and clicking back".
        self._restore_strip_highlight()

    def _restore_strip_highlight(self):
        view = getattr(self.strips_m, "_view", None)
        if view is None:
            return
        rows = view.selectionModel().selectedRows()
        if not rows:
            return
        rows_set = {r.row() for r in rows}
        self.scene.highlight_strips(rows_set)
        self.group_scene.highlight_strips(rows_set)

    # ----- save -----
    def save(self):
        try:
            save_doc(self.doc)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{e}")
            return
        self._dirty = False
        self._update_status(saved=True)

    def _mark_dirty(self):
        self._dirty = True
        self._update_status()

    def _update_status(self, saved: bool = False):
        marker = "  •  unsaved changes" if self._dirty else ""
        if saved:
            marker = "  •  saved"
        d = self.doc
        self.status.showMessage(
            f"{d.project_id} ({d.geometry_type}): {len(d.boxes)} boxes, "
            f"{len(d.groups)} groups, {len(d.strips)} strips{marker}"
        )

    # ----- tab change → refresh selection highlights -----
    def _tab_changed(self, _idx: int):
        # Wipe canvas highlights when leaving a selection-bearing tab
        self.scene.highlight_object(-1)
        self.scene.highlight_strip(-1)
        self.group_scene.highlight_strip(-1)

    # ----- close handling -----
    def closeEvent(self, event):
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved edits. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.StandardButton.Save:
                self.save()
        # Tear down the engine subprocess + emulator client cleanly so
        # we don't leak a child process when the editor closes.
        self._stop_engine()
        event.accept()

    # ======================================================================
    # Emulator runtime: launch engine subprocess, listen for frames,
    # wire weather dropdowns to the engine's REST API.
    # ======================================================================

    EMULATOR_PORT = 58741   # localhost only; not exposed to network
    # Web API base — port read from config.yaml at __init__ so the
    # editor follows whatever port the engine's Flask server actually
    # binds (the runtime default is 80; some configs use 5000).
    WEB_API_BASE = "http://127.0.0.1:5000"

    def _toggle_engine(self):
        if self._engine_proc is not None:
            self._stop_engine()
        else:
            self._start_engine()

    def _start_engine(self):
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved layout",
                "You have unsaved edits in the layout. The engine reads "
                "the YAML files on disk — save before launching?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                return
            if ans == QMessageBox.StandardButton.Save:
                self.save()

        # Tear down the existing scene's strip-line visuals so they
        # don't draw on top of the LED overlay; rebuild later when we
        # stop or the user toggles live preview off.
        self._emu_overlay = EmulatorOverlay(
            self.doc, self.scene, self.group_scene
        )
        self._set_strip_visuals_visible(False)

        self._emu_port = self.EMULATOR_PORT
        # Spawn the engine. QProcess merges stderr+stdout for the log.
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProgram(sys.executable)
        proc.setArguments([
            "-X", "utf8",
            str(ROOT / "Stories_OGL.py"),
            "--project", self.doc.project_id,
            "--emulator-port", str(self._emu_port),
            # Engine watches our PID and exits when we disappear, so a
            # crashed editor doesn't leave an orphan engine running.
            "--parent-pid", str(os.getpid()),
        ])
        proc.setWorkingDirectory(str(ROOT))
        # Force UTF-8 stdio so unicode characters in engine prints
        # don't crash the engine when its output is piped to us
        # (Windows defaults to cp1252 in piped mode). PyQt6 only
        # supports the QProcessEnvironment-based API.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_engine_stdout)
        proc.finished.connect(self._on_engine_finished)
        proc.errorOccurred.connect(self._on_engine_error)
        self.log_panel.clear()
        self._append_log(
            f"$ {sys.executable} Stories_OGL.py --project {self.doc.project_id} "
            f"--emulator-port {self._emu_port}\n"
        )
        proc.start()
        self._engine_proc = proc

        # Start the TCP listener. It retries the connect for ~6 s while
        # the engine boots, then settles into reading frames.
        self._emu_client = EmulatorClient(port=self._emu_port)
        self._emu_client.start()

        self._emu_timer.start()
        self._emu_state_timer.start()
        self._log_flush_timer.start()

        self.run_btn.setText("■ Stop engine")
        self._engine_status.setText("Engine: starting…")
        self._engine_status.setStyleSheet("color: #f7c46c; padding: 0 8px;")
        self.live_preview_chk.setEnabled(True)

    def _stop_engine(self):
        self._emu_timer.stop()
        self._emu_state_timer.stop()
        self._log_flush_timer.stop()
        self._flush_log_buffer()   # drain anything still pending

        if self._emu_client is not None:
            self._emu_client.stop()
            self._emu_client = None

        if self._emu_overlay is not None:
            self._emu_overlay.teardown()
            self._emu_overlay = None
            self._set_strip_visuals_visible(True)

        if self._engine_proc is not None:
            proc = self._engine_proc
            self._engine_proc = None
            try:
                proc.terminate()
                if not proc.waitForFinished(2000):
                    proc.kill()
                    proc.waitForFinished(1000)
            except Exception:
                pass

        self.run_btn.setText("▶ Run engine")
        self._engine_status.setText("Engine: stopped")
        self._engine_status.setStyleSheet("color: #888; padding: 0 8px;")
        self.live_preview_chk.setEnabled(False)
        self.weather_set_picker.setEnabled(False)
        self.weather_state_picker.setEnabled(False)
        self.weather_set_picker.clear()
        self.weather_state_picker.clear()

    # ----- engine process I/O -----
    def _on_engine_stdout(self):
        # Hot path — fires per kernel readyRead, which can be many
        # times per second on a chatty engine boot. Just append the
        # raw bytes; the slow 5 Hz flush timer turns them into log
        # entries on its own schedule.
        if self._engine_proc is None:
            return
        data = bytes(self._engine_proc.readAllStandardOutput())
        if data:
            self._log_buffer.extend(data)

    def _flush_log_buffer(self):
        if not self._log_buffer:
            return
        try:
            text = self._log_buffer.decode("utf-8", errors="replace")
        except Exception:
            text = repr(bytes(self._log_buffer))
        self._log_buffer.clear()
        # appendPlainText splits on its own; trim the trailing newline
        # so we don't get a phantom blank line at the bottom each flush.
        self.log_panel.appendPlainText(text.rstrip("\n"))

    def _on_engine_finished(self, _code, _status):
        # Mirror _stop_engine but skip the QProcess.terminate path
        # since the process has already finished.
        self._engine_proc = None
        self._emu_timer.stop()
        self._emu_state_timer.stop()
        self._log_flush_timer.stop()
        self._flush_log_buffer()
        self.log_panel.appendPlainText("[engine exited]")
        if self._emu_client is not None:
            self._emu_client.stop()
            self._emu_client = None
        if self._emu_overlay is not None:
            self._emu_overlay.teardown()
            self._emu_overlay = None
            self._set_strip_visuals_visible(True)
        self.run_btn.setText("▶ Run engine")
        self._engine_status.setText("Engine: stopped")
        self._engine_status.setStyleSheet("color: #888; padding: 0 8px;")
        self.live_preview_chk.setEnabled(False)
        self.weather_set_picker.setEnabled(False)
        self.weather_state_picker.setEnabled(False)

    def _on_engine_error(self, err):
        self._append_log(f"\n[QProcess error: {err}]\n")

    def _append_log(self, text: str):
        self.log_panel.appendPlainText(text.rstrip("\n"))

    def _on_toggle_log(self, on: bool):
        self.log_panel.setVisible(on)
        if on:
            self._right_col.setSizes([550, 200])
            self.show_log_btn.setText("Hide log")
        else:
            self._right_col.setSizes([700, 0])
            self.show_log_btn.setText("Show log")

    # ----- frame poll (~20 Hz) -----
    def _on_emu_tick(self):
        if self._emu_client is None or self._emu_overlay is None:
            return

        # Surface the actual connection state in the status field so
        # the user can tell which step in the launch chain is stalled.
        # Order of states the editor passes through during a launch:
        #   "Engine: starting…"  — set by _start_engine before client
        #                          first sees the engine's TCP listener
        #   "Engine: connecting…" — TCP retry loop running, listener
        #                          not yet bound (engine still in
        #                          imports / GL init)
        #   "Engine: waiting for frames…" — TCP connected but no
        #                          frame has arrived yet
        #   "Engine: connected"   — frames flowing
        if not self._emu_client.is_connected():
            cur = self._engine_status.text()
            if cur not in ("Engine: connecting…", "Engine: stopped"):
                self._engine_status.setText("Engine: connecting…")
                self._engine_status.setStyleSheet(
                    "color: #f7c46c; padding: 0 8px;")
            return

        raw, corrected, seq = self._emu_client.snapshot()
        if not raw and not corrected:
            # TCP up, no frames yet. Distinct from "starting…" so the
            # operator can see the chain is one step further along.
            cur = self._engine_status.text()
            if cur not in ("Engine: waiting for frames…", "Engine: connected"):
                self._engine_status.setText("Engine: waiting for frames…")
                self._engine_status.setStyleSheet(
                    "color: #f7c46c; padding: 0 8px;")
            return
        # Skip if the engine hasn't published a new frame since the
        # last paint — saves the per-tick numpy + pixmap cost.
        if seq == self._emu_last_seq:
            return
        if self._engine_status.text() != "Engine: connected":
            self._engine_status.setText("Engine: connected")
            self._engine_status.setStyleSheet("color: #7be59a; padding: 0 8px;")
            on = self.live_preview_chk.isChecked()
            self._emu_overlay.set_visible(on)
            # Sync schematic chrome visibility too — engine just came
            # up, so if the user has live preview ticked we need the
            # strip lines + object dots/labels out of the way.
            self._set_strip_visuals_visible(not on)
            self._set_object_visuals_visible(not on)
        try:
            self._emu_overlay.update_frame(raw, corrected)
        except RuntimeError:
            # Overlay's pixmap items got destroyed underneath us between
            # the None-check above and this call (e.g. a model edit
            # racing the timer). Rebuild and let the next tick paint.
            self._emu_overlay = None
            self._rebuild_scenes()
            return
        self._emu_last_seq = seq

    def _on_toggle_live_preview(self, on: bool):
        if self._emu_overlay is not None:
            self._emu_overlay.set_visible(on)
            self._set_strip_visuals_visible(not on)
            self._set_object_visuals_visible(not on)

    def _set_strip_visuals_visible(self, on: bool):
        """Hide the schematic strip lines + arrows on both scenes
        when the LED overlay is active so colors aren't fighting the
        group-color schematic."""
        for item in self.scene._strips:
            item.setVisible(on)
        for item in self.group_scene._strip_items.values():
            item.setVisible(on)

    def _set_object_visuals_visible(self, on: bool):
        """Hide the labeled object dots + their name labels on the
        physical-layout scene when LED overlay is active. The dots
        otherwise sit on top of the LED render and obscure live
        colors, plus the labels add visual noise that has no place
        in a "show" view of the piece."""
        for item in self.scene._objects:
            item.setVisible(on)
        for label in self.scene._labels:
            label.setVisible(on)

    # ----- weather control via REST API (engine's web_controller) -----
    # All urllib calls run on a daemon worker thread so the Qt main
    # thread never blocks on network I/O. Results come back through
    # signals so dropdown updates stay on the GUI thread.
    weather_info_received = pyqtSignal(dict)
    post_failed = pyqtSignal(str)

    def _on_emu_state_poll(self):
        """Background-fetch project/weather state every second so the
        dropdowns stay in sync. Skipped until the engine's emulator
        feed is connected — the web server typically isn't up before
        that, and a tight loop of refused connections just adds
        latency."""
        if self._emu_client is None or not self._emu_client.is_connected():
            return
        threading.Thread(
            target=self._fetch_weather_info_bg, name="WeatherFetch",
            daemon=True,
        ).start()

    def _fetch_weather_info_bg(self):
        try:
            import urllib.request
            import json
            with urllib.request.urlopen(
                f"{self.WEB_API_BASE}/api/weather_set/info", timeout=0.5
            ) as resp:
                info = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return   # web server not up yet; quietly retry next tick
        # Marshal back to the GUI thread.
        self.weather_info_received.emit(info)

    def _on_weather_info(self, info: dict):
        sets = info.get("available_sets", [])
        cur_set = info.get("current_set", "")
        states = info.get("available_weather_states", [])
        cur_state = info.get("current_weather", "")

        existing_sets = [self.weather_set_picker.itemText(i)
                         for i in range(self.weather_set_picker.count())]
        if existing_sets != sets:
            self.weather_set_picker.blockSignals(True)
            self.weather_set_picker.clear()
            self.weather_set_picker.addItems(sets)
            self.weather_set_picker.blockSignals(False)
        idx = self.weather_set_picker.findText(cur_set)
        if idx >= 0 and self.weather_set_picker.currentIndex() != idx:
            self.weather_set_picker.blockSignals(True)
            self.weather_set_picker.setCurrentIndex(idx)
            self.weather_set_picker.blockSignals(False)

        existing_states = [self.weather_state_picker.itemText(i)
                           for i in range(self.weather_state_picker.count())]
        if existing_states != states:
            self.weather_state_picker.blockSignals(True)
            self.weather_state_picker.clear()
            self.weather_state_picker.addItems(states)
            self.weather_state_picker.blockSignals(False)
        idx = self.weather_state_picker.findText(cur_state)
        if idx >= 0 and self.weather_state_picker.currentIndex() != idx:
            self.weather_state_picker.blockSignals(True)
            self.weather_state_picker.setCurrentIndex(idx)
            self.weather_state_picker.blockSignals(False)

        self.weather_set_picker.setEnabled(True)
        self.weather_state_picker.setEnabled(True)

    def _on_pick_weather_set(self, _idx: int):
        name = self.weather_set_picker.currentText()
        if name:
            self._post_json_bg("/api/weather_set/change", {"set_name": name})

    def _on_pick_weather_state(self, _idx: int):
        name = self.weather_state_picker.currentText()
        if name:
            self._post_json_bg("/api/weather_state/change", {"state_name": name})

    def _post_json_bg(self, path: str, payload: dict):
        """Fire-and-forget POST on a worker thread. Failures surface
        through the ``post_failed`` signal so they reach the log panel
        without blocking the GUI."""
        threading.Thread(
            target=self._do_post_json, args=(path, payload),
            name=f"POST {path}", daemon=True,
        ).start()

    def _do_post_json(self, path: str, payload: dict):
        try:
            import urllib.request
            import json
            req = urllib.request.Request(
                f"{self.WEB_API_BASE}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                resp.read()
        except Exception as e:
            self.post_failed.emit(f"[POST {path} failed: {e}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    projects = find_editable_projects()
    if not projects:
        print("No projects found under projects/.")
        return 1

    # Default to whichever project is currently active in config.yaml so
    # the editor opens on what the operator was last running.
    initial = projects[0]["id"]
    try:
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        active = cfg.get("project")
        if active and any(p["id"] == active for p in projects):
            initial = active
    except Exception:
        pass

    app = QApplication(sys.argv)
    win = EditorWindow(initial)
    win.show()

    # Clean Ctrl+C handling. Without this, SIGINT lands inside whatever
    # Python frame happened to be active (often the QProcess readyRead
    # callback) and unwinds the editor without going through closeEvent
    # — so the engine subprocess stays alive and threads aren't joined.
    # The wakeup timer is needed because Qt's event loop sleeps in the
    # OS waitqueue and won't notice Python signals otherwise.
    import signal
    def _on_sigint(*_):
        win.close()
        app.quit()
    signal.signal(signal.SIGINT, _on_sigint)
    sigint_wakeup = QTimer()
    sigint_wakeup.setInterval(200)
    sigint_wakeup.timeout.connect(lambda: None)
    sigint_wakeup.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
