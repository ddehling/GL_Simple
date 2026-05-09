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
``projects/<id>/project.yaml``. The Phase-9 ``build_wol_layout.py``
script remains a one-shot bootstrapper for new projects, but the editor
no longer defers to it for ongoing edits.

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

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QPointF, QRectF, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction, QBrush, QColor, QFont, QPainter, QPen, QPainterPath,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QTableView, QPushButton, QComboBox, QLabel, QToolBar,
    QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsSimpleTextItem,
    QHeaderView, QStyledItemDelegate, QStatusBar, QFrame,
)

from core.project import list_projects


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
            spec.polyline = [
                [int(round(x_inner)), int(round(y_inner))],
                [int(round(x_outer)), int(round(y_outer))],
            ]

    # multi_object: merge per-strip polylines from geometry.yaml by (group, strip_idx)
    if geometry_type == "multi_object":
        polylines: dict[tuple[str, int], list] = {}
        lengths: dict[tuple[str, int], int] = {}
        for s in (raw_g.get("strips") or []):
            key = (str(s.get("group", "")), int(s.get("strip_idx", 0)))
            polylines[key] = [list(map(int, p)) for p in (s.get("polyline") or [])]
            lengths[key] = int(s.get("length", 0))
        for spec in doc.strips:
            key = (spec.group, spec.strip_idx)
            if key in polylines:
                spec.polyline = polylines[key]
            # geometry.yaml's length wins if project.yaml didn't carry one
            if spec.length == 0 and key in lengths:
                spec.length = lengths[key]

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
            host=f"wol-obj{next_id}.local",
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
        if role == Qt.ItemDataRole.BackgroundRole and c == self.COL_GROUP:
            return QBrush(color_for_group(s.group, self.doc.groups))
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
            else:
                return False
        except (TypeError, ValueError):
            return False
        self.dataChanged.emit(index, index)
        self.emit_layout_changed()
        return True

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
        self.beginInsertRows(QModelIndex(), len(self.doc.strips), len(self.doc.strips))
        self.doc.strips.append(new)
        self.endInsertRows()
        self.emit_layout_changed()
        return len(self.doc.strips) - 1

    def remove_rows(self, rows: list[int]):
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.doc.strips):
                self.beginRemoveRows(QModelIndex(), row, row)
                self.doc.strips.pop(row)
                self.endRemoveRows()
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
    """Draggable dot for one box's spatial position (multi_object only)."""
    RADIUS = 7

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

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_moved(self.obj_idx, int(value.x()), int(value.y()))
        return super().itemChange(change, value)


class _VertexHandle(QGraphicsEllipseItem):
    """Draggable handle at a polyline vertex (multi_object strips only).

    Hidden by default; the scene reveals only the handles for the
    currently-selected strip so the canvas doesn't drown in tiny dots.
    """
    RADIUS = 5

    def __init__(self, strip_idx: int, vertex_idx: int, x: int, y: int, on_moved):
        super().__init__(-self.RADIUS, -self.RADIUS,
                         2 * self.RADIUS, 2 * self.RADIUS)
        self.strip_idx = strip_idx
        self.vertex_idx = vertex_idx
        self._on_moved = on_moved
        # When True, programmatic setPos() calls won't fire the on_moved
        # callback. QGraphicsEllipseItem isn't a QObject so blockSignals()
        # is not available — we gate the callback ourselves instead.
        self._suppress_callback = True
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#000000"), 1))
        self.setZValue(20)         # above strips and object dots
        self.setVisible(False)     # revealed by LayoutScene.highlight_strip
        self.setPos(x, y)
        self.setToolTip(f"vertex {vertex_idx} ({x}, {y}) — drag to reshape")
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

    def mousePressEvent(self, event):
        scene = self.scene()
        if isinstance(scene, LayoutScene):
            scene.selection_changed_strip.emit(self.strip_idx)
        super().mousePressEvent(event)

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

    def __init__(self):
        super().__init__()
        self._objects: list[_ObjectItem] = []
        self._strips: list[_StripItem] = []
        self._labels: list[QGraphicsSimpleTextItem] = []
        # vertex handles per strip; each entry is the list of handles for
        # that strip in polyline order. Empty for strips without polylines.
        self._handles: list[list[_VertexHandle]] = []

    def rebuild(self, doc: LayoutDoc):
        # Remove everything
        self.clear()
        self._objects = []
        self._strips = []
        self._labels = []
        self._handles = []

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
            handles: list[_VertexHandle] = []
            if polylines_editable and s.polyline:
                for vi, (vx, vy) in enumerate(s.polyline):
                    h = _VertexHandle(i, vi, int(vx), int(vy), self._on_vertex_moved)
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
        for i, item in enumerate(self._strips):
            pen = item.pen()
            pen.setWidthF(4.5 if i == strip_idx else 2.5)
            item.setPen(pen)
        # Show vertex handles for the highlighted strip only.
        for i, handles in enumerate(self._handles):
            visible = (i == strip_idx)
            for h in handles:
                h.setVisible(visible)

    def _on_object_moved(self, obj_idx: int, x: int, y: int):
        self.object_moved.emit(obj_idx, x, y)

    def _on_vertex_moved(self, strip_idx: int, vertex_idx: int, x: int, y: int):
        self.vertex_moved.emit(strip_idx, vertex_idx, x, y)


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
    # Minimum pixels between adjacent strip lines on the panel — drives
    # the per-group display scale so close-packed strips (Fan: 32 columns
    # on a 128-wide FBO = 4 px apart natively) remain individually
    # clickable and labellable.
    MIN_STRIP_SPACING = 9
    MIN_DISPLAY_DIM = 90    # also enforce a minimum on the smaller axis
    MAX_DISPLAY_DIM = 800   # cap on the larger axis so huge canvases fit
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
        """Per-group (sx, sy) display scale.

        Non-uniform on purpose: the group canvas is an FBO, not a
        physical surface, so distorting its aspect ratio for display is
        fine. The X scale is sized so adjacent column strips have at
        least ``MIN_STRIP_SPACING`` px between them; Y likewise for
        rows. Both axes are clamped to a ``[MIN_DISPLAY_DIM,
        MAX_DISPLAY_DIM]`` band so tiny canvases stay readable and
        massive ones still fit in the panel."""
        if g.width <= 0 or g.height <= 0:
            return 1.0, 1.0

        cols = sorted({int(s.col)
                       for s in strips
                       if s.group == g.id and s.kind == "column"})
        rows = sorted({int(s.row)
                       for s in strips
                       if s.group == g.id and s.kind == "row"})

        sx = 1.0
        if len(cols) >= 2:
            native = (cols[-1] - cols[0]) / max(1, len(cols) - 1)
            if native < self.MIN_STRIP_SPACING:
                sx = self.MIN_STRIP_SPACING / max(native, 1e-3)
        sy = 1.0
        if len(rows) >= 2:
            native = (rows[-1] - rows[0]) / max(1, len(rows) - 1)
            if native < self.MIN_STRIP_SPACING:
                sy = self.MIN_STRIP_SPACING / max(native, 1e-3)

        # Clamp each axis to [MIN_DISPLAY_DIM, MAX_DISPLAY_DIM].
        def clamp(scale, native_dim):
            disp = scale * native_dim
            if disp < self.MIN_DISPLAY_DIM:
                return self.MIN_DISPLAY_DIM / native_dim
            if disp > self.MAX_DISPLAY_DIM:
                return self.MAX_DISPLAY_DIM / native_dim
            return scale
        sx = clamp(sx, g.width)
        sy = clamp(sy, g.height)
        return sx, sy

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
        for idx, item in self._strip_items.items():
            if isinstance(item, _GroupStripItem):
                item.set_highlighted(idx == strip_idx)
            else:
                pen = item.pen()
                pen.setWidthF(4.5 if idx == strip_idx else 2.0)
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

class EditorWindow(QMainWindow):
    def __init__(self, project_id: str):
        super().__init__()
        self.setWindowTitle(f"Layout Editor — {project_id}")
        self.resize(1400, 900)

        self.doc = load_doc(project_id)
        self._dirty = False

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

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        save_btn.setShortcut("Ctrl+S")
        top.addWidget(save_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        top.addWidget(reload_btn)

        outer.addLayout(top)

        # Right side: physical canvas above, group-canvas panel below.
        right_split = QSplitter(Qt.Orientation.Vertical)
        # Wrap each view with a tiny header label so it's clear which is which.
        right_split.addWidget(self._wrap_panel("Physical layout", self.view))
        right_split.addWidget(self._wrap_panel(
            "Group canvases (rendering surfaces)", self.group_view))
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 0)
        right_split.setSizes([600, 300])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 880])
        outer.addWidget(splitter, 1)

        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_status()

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
        row = rows[0].row() if rows else -1
        if isinstance(model, BoxesModel):
            self.scene.highlight_object(row)
            self.scene.highlight_strip(-1)
            self.group_scene.highlight_strip(-1)
        elif isinstance(model, StripsModel):
            self.scene.highlight_strip(row)
            self.scene.highlight_object(-1)
            self.group_scene.highlight_strip(row)
        else:
            self.scene.highlight_object(-1)
            self.scene.highlight_strip(-1)
            self.group_scene.highlight_strip(-1)

    def _on_strip_clicked_on_canvas(self, strip_idx: int):
        """User clicked a strip on either canvas. Switch to the Strips
        tab and select that row — which in turn highlights it on both
        canvases and reveals its polyline vertex handles."""
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        self.tabs.setCurrentIndex(2)   # Boxes(0), Groups(1), Strips(2)
        view = getattr(self.strips_m, "_view", None)
        if view is not None:
            view.selectRow(strip_idx)

    # ----- canvas → table sync -----
    def _on_object_moved_on_canvas(self, obj_idx: int, x: int, y: int):
        if 0 <= obj_idx < len(self.doc.boxes):
            self.doc.boxes[obj_idx].x = x
            self.doc.boxes[obj_idx].y = y
            # Notify the boxes model (the x/y columns are 2 and 3)
            tl = self.boxes_m.index(obj_idx, 2)
            br = self.boxes_m.index(obj_idx, 3)
            self.boxes_m.dataChanged.emit(tl, br)
            # Reflect tooltip + label on canvas
            self.scene.update_object(obj_idx, self.doc.boxes[obj_idx])
            self._mark_dirty()

    def _on_vertex_moved_on_canvas(self, strip_idx: int, vertex_idx: int,
                                   x: int, y: int):
        if not (0 <= strip_idx < len(self.doc.strips)):
            return
        s = self.doc.strips[strip_idx]
        if not (0 <= vertex_idx < len(s.polyline)):
            return
        s.polyline[vertex_idx] = [x, y]
        # Redraw the strip's path. Pass the vertex_idx so the scene
        # doesn't reposition the handle the user is actively dragging.
        self.scene.update_strip_polyline(strip_idx, s.polyline,
                                         skip_handle_at=vertex_idx)
        # Refresh the polyline summary cell in the strips table.
        idx = self.strips_m.index(strip_idx, StripsModel.COL_POLYLINE)
        self.strips_m.dataChanged.emit(idx, idx)
        self._mark_dirty()

    # ----- project switching -----
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
        self.scene.rebuild(self.doc)
        self.group_scene.rebuild(self.doc)
        self._dirty = False
        self._update_status()

    # ----- canvas refresh -----
    def _refresh_canvas(self):
        self.scene.rebuild(self.doc)
        self.group_scene.rebuild(self.doc)

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
        event.accept()


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
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
