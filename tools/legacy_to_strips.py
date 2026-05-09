"""Convert legacy ``config.yaml`` DMX receivers to strip form for project.yaml.

Reads the active project's machine-local ``config.yaml`` and emits a
strip-based ``receivers:`` block to stdout (or, with --write, merges it
into ``projects/<project>/project.yaml``).

The converter validates byte-identical output: it rebuilds the addressing
array from the generated strips and asserts elementwise equality with the
legacy ``make_indices_*`` function output. If the assertion fails the run
exits non-zero with a diff so the discrepancy can be tracked down before
hardware is exercised.

Usage::

    python tools/legacy_to_strips.py            # print to stdout
    python tools/legacy_to_strips.py --write    # also patch projects/fan/project.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lib.dmx_sender as imdmx  # noqa: E402
from core.project import load_project  # noqa: E402
from core.strip import (  # noqa: E402
    StripBinding,
    addressing_array_from_strips,
    strips_from_column_rect,
    strips_from_hs_lines,
    strips_from_vs_lines,
)


def _height_from_config(cfg: dict) -> int:
    return int(cfg.get("display", {}).get("height", 300))


def convert_receiver(rx: dict, height: int, group_id: str = "main") -> tuple[list[StripBinding], np.ndarray]:
    """Return (strips, legacy_addressing_array) for one config.yaml receiver.

    Mirrors the receiver-building logic in ``Stories_OGL.py``: legacy
    receivers can use either ``columns/column_offset`` (column-rect) or
    ``addressing: {mode: hs|vs, file: ...}`` (CSV per-strip).
    """
    if "addressing" in rx:
        mode = rx["addressing"]["mode"]
        filepath = rx["addressing"]["file"]
        if mode == "hs":
            legacy = imdmx.make_indicesHS(filepath)
            lines = np.loadtxt(filepath, delimiter=",").tolist()
            strips = strips_from_hs_lines(lines, group_id=group_id)
        elif mode == "vs":
            legacy = imdmx.make_indicesVS(filepath)
            lines = np.loadtxt(filepath, delimiter=",").tolist()
            strips = strips_from_vs_lines(lines, group_id=group_id)
        else:
            raise ValueError(f"Unknown addressing mode: {mode}")
        return strips, legacy

    columns = int(rx["columns"])
    column_offset = int(rx["column_offset"])
    legacy = imdmx.make_indices_V_rect_alternate(columns, height, column_offset)
    strips = strips_from_column_rect(
        columns=columns, height=height, column_offset=column_offset,
        group_id=group_id,
    )
    return strips, legacy


def strip_to_yaml_dict(s: StripBinding) -> dict:
    """Compact YAML form for column-style strips (the only kind Fan emits)."""
    rows = s.pixel_indices[:, 0]
    cols = s.pixel_indices[:, 1]
    # Heuristic: if all cols are equal, this is a column strip.
    if cols.size > 0 and np.all(cols == cols[0]):
        col = int(cols[0])
        length = int(rows.size)
        descending = rows.size >= 2 and rows[0] > rows[1]
        return {
            "group": s.group_id,
            "strip_idx": s.strip_idx,
            "kind": "column",
            "col": col,
            "length": length,
            "direction": "down" if descending else "up",
        }
    if rows.size > 0 and np.all(rows == rows[0]):
        row = int(rows[0])
        length = int(cols.size)
        descending = cols.size >= 2 and cols[0] > cols[1]
        return {
            "group": s.group_id,
            "strip_idx": s.strip_idx,
            "kind": "row",
            "row": row,
            "length": length,
            "direction": "left" if descending else "right",
        }
    # Fallback: explicit indices.
    return {
        "group": s.group_id,
        "strip_idx": s.strip_idx,
        "kind": "raw",
        "indices": s.pixel_indices.tolist(),
    }


def build_yaml_receivers(cfg: dict) -> list[dict]:
    height = _height_from_config(cfg)
    out: list[dict] = []
    for rx in cfg["dmx"]["receivers"]:
        if not isinstance(rx, dict):
            continue
        strips, legacy = convert_receiver(rx, height)
        rebuilt = addressing_array_from_strips(strips)
        if not np.array_equal(rebuilt, legacy):
            diff = np.where(np.any(rebuilt != legacy, axis=1))[0]
            sys.exit(
                f"[legacy_to_strips] FAIL: addressing mismatch on receiver "
                f"{rx.get('ip') or rx.get('host')!r}: "
                f"first divergence at row {int(diff[0])}: "
                f"new={rebuilt[diff[0]].tolist()} legacy={legacy[diff[0]].tolist()}"
            )

        entry: dict = {}
        for key in ("ip", "host", "protocol"):
            if key in rx:
                entry[key] = rx[key]
        entry["strips"] = [strip_to_yaml_dict(s) for s in strips]
        out.append(entry)
        print(
            f"[legacy_to_strips] OK {entry.get('host', entry.get('ip'))}: "
            f"{len(strips)} strips, {rebuilt.shape[0]} pixels — byte-identical"
        )
    return out


def load_machine_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None,
                        help="Project id (defaults to config.yaml's `project:`)")
    parser.add_argument("--write", action="store_true",
                        help="Patch projects/<project>/project.yaml in place")
    args = parser.parse_args()

    cfg = load_machine_config()
    project_id = args.project or cfg.get("project", "fan")
    project = load_project(project_id)

    print(f"[legacy_to_strips] Active project: {project.id}")
    print(f"[legacy_to_strips] Source: config.yaml dmx.receivers "
          f"({len(cfg['dmx']['receivers'])} entries)")

    receivers_yaml = build_yaml_receivers(cfg)

    if not args.write:
        print()
        print("# --- copy this into projects/<project>/project.yaml ---")
        print(yaml.safe_dump({"receivers": receivers_yaml},
                              sort_keys=False, default_flow_style=False))
        return 0

    proj_yaml_path = project.root / "project.yaml"
    with open(proj_yaml_path, "r", encoding="utf-8") as f:
        existing = yaml.safe_load(f) or {}
    existing["receivers"] = receivers_yaml
    with open(proj_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, sort_keys=False, default_flow_style=False)
    print(f"[legacy_to_strips] Wrote {proj_yaml_path} "
          f"({len(receivers_yaml)} receivers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
