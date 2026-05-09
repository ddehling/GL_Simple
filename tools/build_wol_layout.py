"""One-shot generator for projects/weight_of_light/{geometry,project}.yaml.

Produces a Phase-5 single-canvas layout: 9 objects (1 center + 8 peripheral
arranged clockwise from north), each with a trunk + 2 leaves (and an
ambient on peripherals). The single 144x35 canvas is the atlas; row
indices are pre-baked here so the generated YAML is ready to load.

Re-run after changing layout constants. The committed YAML files are
authoritative — this script is just a deterministic source-of-truth tool.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml


CANVAS_W, CANVAS_H = 1280, 1024
CENTER = (640, 512)
RING_R = 320
TRUNK_LEN = 144
LEAF_LEN = 96
AMBIENT_LEN = 60
LEAF_OFFSET_DEG = 45

PERIPHERAL_NAMES = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]


def load_receiver_addresses(proj_dir: Path) -> dict:
    """Read the optional per-object hardware-address override file.

    Returns a dict ``{object_name: {key: value}}`` where each entry is
    spread directly onto the receiver entry in project.yaml. Common keys:
    ``ip`` (literal dotted-quad), ``host`` (mDNS hostname), ``protocol``
    (``"sacn"`` or ``"ddp"``; default ``"ddp"``).

    Missing file → empty dict; build script falls back to placeholder
    ``wol-{name}.local`` hostnames so the layout is still loadable
    (mDNS misses are non-fatal at runtime).
    """
    path = proj_dir / "receiver_addresses.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _rot(vec, deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    x, y = vec
    return (x * c - y * s, x * s + y * c)


def _line(start, direction, length):
    sx, sy = start
    dx, dy = direction
    end = (sx + length * dx, sy + length * dy)
    return [[int(round(sx)), int(round(sy))],
            [int(round(end[0])), int(round(end[1]))]]


def build_objects():
    objs = [{"id": 0, "name": "center", "x": CENTER[0], "y": CENTER[1]}]
    for i in range(8):
        angle = -math.pi / 2 + i * (math.pi / 4)
        x = CENTER[0] + RING_R * math.cos(angle)
        y = CENTER[1] + RING_R * math.sin(angle)
        objs.append({"id": i + 1, "name": PERIPHERAL_NAMES[i],
                     "x": int(round(x)), "y": int(round(y))})
    return objs


def outward_dir(obj):
    if obj["id"] == 0:
        return (0.0, -1.0)
    dx = obj["x"] - CENTER[0]
    dy = obj["y"] - CENTER[1]
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def build_strips_and_receivers(objects, addresses: dict | None = None):
    """Returns (strips_for_geometry, receivers_for_project).

    Phase-6 convention: each group has its own canvas, and each strip
    occupies row ``strip_idx`` of its group canvas. ``row`` is omitted
    from receiver strip entries because ``kind: row`` defaults it to
    ``strip_idx`` (see core/strip.py).

    ``addresses`` is the dict returned by ``load_receiver_addresses``;
    each entry overrides the default ``host: wol-{name}.local`` for that
    object so real hardware can replace placeholders without re-running
    or editing the build script.
    """
    addresses = addresses or {}
    strips: list[dict] = []
    receivers: list[dict] = []

    for obj in objects:
        oid = obj["id"]
        base = (obj["x"], obj["y"])
        trunk_dir = outward_dir(obj)
        leaf_a_dir = _rot(trunk_dir, -LEAF_OFFSET_DEG)
        leaf_b_dir = _rot(trunk_dir, LEAF_OFFSET_DEG)
        ambient_dir = (-trunk_dir[0], -trunk_dir[1])

        rx_strips: list[dict] = []

        strips.append({
            "group": "trunk", "strip_idx": oid,
            "length": TRUNK_LEN,
            "polyline": _line(base, trunk_dir, TRUNK_LEN),
        })
        rx_strips.append({
            "group": "trunk", "strip_idx": oid,
            "kind": "row", "length": TRUNK_LEN, "direction": "right",
        })

        leaf_a_idx = 2 * oid
        leaf_b_idx = 2 * oid + 1
        for leaf_idx, ldir in ((leaf_a_idx, leaf_a_dir),
                               (leaf_b_idx, leaf_b_dir)):
            strips.append({
                "group": "leaves", "strip_idx": leaf_idx,
                "length": LEAF_LEN,
                "polyline": _line(base, ldir, LEAF_LEN),
            })
            rx_strips.append({
                "group": "leaves", "strip_idx": leaf_idx,
                "kind": "row", "length": LEAF_LEN, "direction": "right",
            })

        if oid != 0:
            ambient_idx = oid - 1
            strips.append({
                "group": "ambient", "strip_idx": ambient_idx,
                "length": AMBIENT_LEN,
                "polyline": _line(base, ambient_dir, AMBIENT_LEN),
            })
            rx_strips.append({
                "group": "ambient", "strip_idx": ambient_idx,
                "kind": "row", "length": AMBIENT_LEN, "direction": "right",
            })

        # Per-object address override. Default to a placeholder mDNS
        # hostname; mDNS misses are non-fatal at runtime so an unconfigured
        # WoL still boots cleanly with zero active receivers.
        addr_entry = addresses.get(obj["name"]) or {}
        if not isinstance(addr_entry, dict):
            addr_entry = {}
        rx_entry: dict = {}
        if "ip" in addr_entry:
            rx_entry["ip"] = addr_entry["ip"]
        if "host" in addr_entry:
            rx_entry["host"] = addr_entry["host"]
        if not rx_entry:
            rx_entry["host"] = f"wol-{obj['name']}.local"
        rx_entry["protocol"] = addr_entry.get("protocol", "ddp")
        rx_entry["object_id"] = oid
        rx_entry["strips"] = rx_strips
        receivers.append(rx_entry)

    return strips, receivers


def main():
    root = Path(__file__).resolve().parent.parent
    proj_dir = root / "projects" / "weight_of_light"

    objects = build_objects()
    addresses = load_receiver_addresses(proj_dir)
    if addresses:
        print(f"[build_wol_layout] Loaded {len(addresses)} address overrides "
              f"from receiver_addresses.yaml")
    strips, receivers = build_strips_and_receivers(objects, addresses)

    geometry = {
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "objects": objects,
        "strips": strips,
    }
    geometry_path = proj_dir / "geometry.yaml"
    with open(geometry_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(geometry, f, sort_keys=False, default_flow_style=False)
    print(f"[build_wol_layout] Wrote {geometry_path} "
          f"({len(objects)} objects, {len(strips)} strips)")

    project = {
        "id": "weight_of_light",
        "display_name": "Weight of Light",
        "weather_sets_module": "projects.weight_of_light.weather_params",
        "event_map_module": "projects.weight_of_light.event_map",
        # Phase 6: one canvas per logical group. Strips reference rows
        # within their group canvas (strip_idx == row).
        "groups": [
            {"id": "trunk",   "width": TRUNK_LEN,   "height": 9},
            {"id": "leaves",  "width": LEAF_LEN,    "height": 18},
            {"id": "ambient", "width": AMBIENT_LEN, "height": 8},
        ],
        # Disable the global Fan-flavored random_events loop on WoL (no
        # shader_meteor / shader_aurora / shader_sandstorm leaking onto
        # WoL canvases). The WoL weather set has its own random_events
        # list (currently empty) for any project-specific surprises.
        "enable_random_events": False,
        "geometry": {
            "type": "multi_object",
            "file": str(Path("projects/weight_of_light/geometry.yaml")).replace("\\", "/"),
        },
        "receivers": receivers,
    }
    project_path = proj_dir / "project.yaml"
    with open(project_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(project, f, sort_keys=False, default_flow_style=False)
    print(f"[build_wol_layout] Wrote {project_path} "
          f"({len(receivers)} receivers, "
          f"{sum(len(r['strips']) for r in receivers)} total strips)")


if __name__ == "__main__":
    main()
