#!/usr/bin/env python3
"""Offline wire-level check of the DMX output path.

Answers "is the right pixel reaching the right LED on the right box?"
without any hardware: it builds the real ``SACNPixelSender`` from a
project's ``project.yaml`` receivers, swaps its UDP socket for a
recorder, pushes known frames through, then decodes every captured
datagram back into (receiver, protocol, universe/offset, LED index)
and compares against an expectation derived INDEPENDENTLY from the YAML
(this file re-implements the strip addressing rules from the schema doc
rather than calling ``core.strip``, so a bug in the loader shows up as a
mismatch instead of cancelling out).

What it verifies
  1. Config sanity  — strips land inside their group canvas; reports
     canvas pixels claimed by more than one receiver, and coverage.
  2. Pixel mapping  — every wire byte equals the frame pixel the YAML
     says that LED should show, in wire order.
  3. sACN framing   — E1.31 header decodes, per-receiver universe
     numbering, destination ip:5568, 510-byte payloads, tail padding,
     sequence increments frame to frame.
  4. DDP framing    — contiguous byte offsets, <=1440 payloads, PUSH on
     the final chunk only, destination ip:4048, sequence cycles 1..15.
  5. Numba parity   — the Numba fast path and the numpy fallback emit
     byte-identical packets.

Usage:
    python tools/tests/_dmx_wire_test.py                 # every project
    python tools/tests/_dmx_wire_test.py --project fan
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lib.dmx_sender as imdmx  # noqa: E402

SACN_PORT = 5568
DDP_PORT = 4048
SACN_HEADER = 126
ACN_ID = b'ASC-E1.17\x00\x00\x00'


# ---------------------------------------------------------------------------
# Independent re-implementation of the strip addressing schema
# (docstring of core/strip.py). Deliberately NOT importing core.strip.
# ---------------------------------------------------------------------------

def expected_indices(entry: dict) -> list[tuple[int, int]]:
    """(row, col) per LED, in chain order, for one YAML strip entry."""
    kind = entry["kind"]
    if kind == "column":
        length = int(entry["length"])
        col = int(entry["col"])
        direction = entry.get("direction", "down")
        if direction == "down":
            start = int(entry.get("start", length - 1))
            rows = [start - i for i in range(length)]
        elif direction == "up":
            start = int(entry.get("start", 0))
            rows = [start + i for i in range(length)]
        else:
            raise ValueError(f"bad column direction {direction!r}")
        return [(r, col) for r in rows]

    if kind == "row":
        length = int(entry["length"])
        row = int(entry.get("row", entry.get("strip_idx", 0)))
        direction = entry.get("direction", "right")
        if direction == "right":
            start = int(entry.get("start", 0))
            cols = [start + i for i in range(length)]
        elif direction == "left":
            start = int(entry.get("start", length - 1))
            cols = [start - i for i in range(length)]
        else:
            raise ValueError(f"bad row direction {direction!r}")
        return [(row, c) for c in cols]

    if kind == "raw":
        return [(int(r), int(c)) for r, c in entry["indices"]]

    raise ValueError(f"unknown strip kind {kind!r}")


# ---------------------------------------------------------------------------
# Socket recorder
# ---------------------------------------------------------------------------

class RecordingSocket:
    """Stands in for the sender's UDP socket; keeps every datagram."""

    def __init__(self):
        self.packets: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data, addr):
        self.packets.append((bytes(data), addr))
        return len(data)

    def close(self):
        pass


def parse_sacn(data: bytes) -> dict:
    if len(data) < SACN_HEADER:
        raise ValueError(f"short sACN packet: {len(data)} bytes")
    if data[4:16] != ACN_ID:
        raise ValueError("missing ACN packet identifier")
    return {
        "root_len": struct.unpack('>H', data[16:18])[0] & 0x0FFF,
        "framing_len": struct.unpack('>H', data[38:40])[0] & 0x0FFF,
        "dmp_len": struct.unpack('>H', data[115:117])[0] & 0x0FFF,
        "source": data[44:108].split(b'\x00')[0].decode('utf-8', 'replace'),
        "priority": data[108],
        "sequence": data[111],
        "universe": struct.unpack('>H', data[113:115])[0],
        "prop_count": struct.unpack('>H', data[123:125])[0],
        "start_code": data[125],
        "dmx": data[SACN_HEADER:],
        "total_len": len(data),
    }


def parse_ddp(data: bytes) -> dict:
    if len(data) < 10:
        raise ValueError(f"short DDP packet: {len(data)} bytes")
    flags1, seq, dtype, dest_id, offset, length = struct.unpack('>BBBBIH', data[:10])
    return {
        "version": (flags1 >> 6) & 0x03,
        "push": bool(flags1 & 0x01),
        "sequence": seq & 0x0F,
        "data_type": dtype,
        "dest_id": dest_id,
        "offset": offset,
        "length": length,
        "payload": data[10:],
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Result:
    def __init__(self):
        self.checks = 0
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, cond, msg):
        self.checks += 1
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def note(self, msg):
        self.notes.append(msg)


def load_project(project_id: str) -> dict:
    path = ROOT / "projects" / project_id / "project.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_plan(proj: dict, res: Result):
    """Expected wire layout per receiver, straight from YAML.

    Returns (plan, group_shapes) where plan is a list of dicts:
      {label, ip, protocol, strips: [ {group, indices} ], pixel_count}
    """
    group_shapes = {}
    for g in proj.get("groups", []):
        group_shapes[g["id"]] = (int(g["height"]), int(g["width"]))

    plan = []
    for rx_idx, rx in enumerate(proj.get("receivers") or []):
        strips = []
        for s in rx.get("strips") or []:
            gid = s.get("group", "main")
            idx = expected_indices(s)
            strips.append({"group": gid, "indices": idx,
                           "strip_idx": s.get("strip_idx")})
            h, w = group_shapes.get(gid, (None, None))
            if h is None:
                res.ok(False, f"receiver {rx_idx} strip references unknown group {gid!r}")
                continue
            bad = [(r, c) for r, c in idx if not (0 <= r < h and 0 <= c < w)]
            res.ok(not bad,
                   f"receiver {rx_idx} strip {s.get('strip_idx')} has "
                   f"{len(bad)} LED(s) outside group {gid} canvas "
                   f"{h}x{w} (e.g. {bad[:3]})")
        plan.append({
            "label": rx.get("host") or rx.get("ip") or f"receiver-{rx_idx}",
            # Offline: no mDNS, so synthesize a unique destination per
            # receiver. Routing correctness is checked as "each receiver's
            # packets all went to its own address, and no other's".
            "ip": rx.get("ip") or f"10.255.0.{rx_idx + 1}",
            "protocol": (rx.get("protocol") or "sacn").lower(),
            "strips": strips,
            "pixel_count": sum(len(s["indices"]) for s in strips),
            "declared_host": rx.get("host"),
            "object_id": rx.get("object_id"),
        })
    return plan, group_shapes


def coverage_report(plan, group_shapes, res: Result):
    owners = defaultdict(list)
    for rx in plan:
        for s in rx["strips"]:
            for rc in s["indices"]:
                owners[(s["group"], rc)].append(rx["label"])

    dupes = {k: v for k, v in owners.items() if len(set(v)) > 1}
    if dupes:
        by_pair = defaultdict(int)
        for k, v in dupes.items():
            by_pair[tuple(sorted(set(v)))] += 1
        for pair, n in sorted(by_pair.items(), key=lambda kv: -kv[1]):
            res.note(f"OVERLAP: {n} canvas pixel(s) sent to BOTH "
                     f"{' and '.join(pair)} (they will show the same content)")

    for gid, (h, w) in group_shapes.items():
        claimed = sum(1 for (g, _rc) in owners if g == gid)
        total = h * w
        pct = 100.0 * claimed / total if total else 0.0
        res.note(f"group {gid}: {claimed}/{total} canvas pixels driven ({pct:.1f}%)")


def make_frames(group_shapes, seed):
    rng = np.random.default_rng(seed)
    return {gid: rng.integers(1, 256, size=(h, w, 3), dtype=np.uint8)
            for gid, (h, w) in group_shapes.items()}


def build_sender(proj, plan, res: Result):
    """Real SACNPixelSender over the project's real strips, with the
    socket swapped for a recorder. Mirrors engine/render_pipeline.py's
    construction flags exactly."""
    from core.strip import strips_from_yaml_list

    receivers = []
    for rx_yaml, p in zip(proj["receivers"], plan):
        receivers.append({
            "ip": p["ip"],
            "protocol": p["protocol"],
            "strips": strips_from_yaml_list(rx_yaml["strips"]),
        })

    sender = imdmx.SACNPixelSender(
        receivers, skip_network=False,
        use_raw_udp=True, per_receiver_universe=True,
        bind_ip="",
    )
    if getattr(sender, "udp_socket", None) is not None:
        sender.udp_socket.close()
    rec = RecordingSocket()
    sender.udp_socket = rec
    return sender, rec


def reassemble(rec: RecordingSocket, plan, res: Result, frame_no: int):
    """Decode captured packets into {receiver_label: bytes} plus framing checks."""
    by_ip = defaultdict(list)
    for data, (ip, port) in rec.packets:
        by_ip[ip].append((port, data))

    known_ips = {p["ip"] for p in plan}
    stray = set(by_ip) - known_ips
    res.ok(not stray, f"frame {frame_no}: packets sent to unexpected address(es) {stray}")

    buffers = {}
    for p in plan:
        label, ip, proto = p["label"], p["ip"], p["protocol"]
        pkts = by_ip.get(ip, [])
        if not res.ok(pkts, f"frame {frame_no}: {label} received NO packets"):
            continue
        expected_bytes = p["pixel_count"] * 3

        if proto == "sacn":
            ports = {port for port, _ in pkts}
            res.ok(ports == {SACN_PORT},
                   f"frame {frame_no}: {label} sACN sent to port(s) {ports}, expected {SACN_PORT}")
            n_uni = math.ceil(p["pixel_count"] / 170)
            res.ok(len(pkts) == n_uni,
                   f"frame {frame_no}: {label} sent {len(pkts)} packets, "
                   f"expected {n_uni} universes")
            seen = {}
            for _port, data in pkts:
                pkt = parse_sacn(data)
                res.ok(pkt["priority"] == 100,
                       f"{label} u{pkt['universe']}: priority {pkt['priority']} != 100")
                res.ok(pkt["start_code"] == 0,
                       f"{label} u{pkt['universe']}: DMX start code {pkt['start_code']} != 0")
                res.ok(len(pkt["dmx"]) == 510,
                       f"{label} u{pkt['universe']}: {len(pkt['dmx'])} DMX bytes, expected 510")
                res.ok(pkt["universe"] not in seen,
                       f"{label}: universe {pkt['universe']} sent twice in one frame")
                seen[pkt["universe"]] = pkt
            # per_receiver_universe=True -> every receiver restarts at 1
            res.ok(sorted(seen) == list(range(1, n_uni + 1)),
                   f"frame {frame_no}: {label} universes {sorted(seen)[:4]}... "
                   f"!= 1..{n_uni}")
            buf = bytearray()
            for u in sorted(seen):
                buf += seen[u]["dmx"]
            # trailing padding beyond the real pixel data must be zero
            tail = bytes(buf[expected_bytes:])
            res.ok(all(b == 0 for b in tail),
                   f"frame {frame_no}: {label} has {sum(1 for b in tail if b)} "
                   f"non-zero byte(s) in universe padding")
            buffers[label] = bytes(buf[:expected_bytes])
            p["_seqs"] = {u: seen[u]["sequence"] for u in seen}

        else:  # ddp
            ports = {port for port, _ in pkts}
            res.ok(ports == {DDP_PORT},
                   f"frame {frame_no}: {label} DDP sent to port(s) {ports}, expected {DDP_PORT}")
            parsed = [parse_ddp(d) for _port, d in pkts]
            offset = 0
            buf = bytearray()
            seqs = set()
            for i, pk in enumerate(parsed):
                res.ok(pk["version"] == 1, f"{label} chunk {i}: DDP version {pk['version']} != 1")
                res.ok(pk["data_type"] == 0x01, f"{label} chunk {i}: data type {pk['data_type']:#x} != 0x01 (RGB)")
                res.ok(pk["dest_id"] == 1, f"{label} chunk {i}: dest id {pk['dest_id']} != 1")
                res.ok(pk["offset"] == offset,
                       f"{label} chunk {i}: byte offset {pk['offset']}, expected {offset}")
                res.ok(pk["length"] == len(pk["payload"]),
                       f"{label} chunk {i}: header length {pk['length']} != payload {len(pk['payload'])}")
                res.ok(pk["length"] <= imdmx.SACNPixelSender.DDP_MAX_PAYLOAD,
                       f"{label} chunk {i}: payload {pk['length']} > MTU-safe "
                       f"{imdmx.SACNPixelSender.DDP_MAX_PAYLOAD}")
                last = (i == len(parsed) - 1)
                res.ok(pk["push"] == last,
                       f"{label} chunk {i}: PUSH={pk['push']} but last={last}")
                seqs.add(pk["sequence"])
                buf += pk["payload"]
                offset += pk["length"]
            res.ok(len(seqs) == 1,
                   f"frame {frame_no}: {label} used {len(seqs)} DDP sequence values "
                   f"within one frame ({sorted(seqs)}) — should be 1")
            res.ok(len(buf) == expected_bytes,
                   f"frame {frame_no}: {label} DDP carried {len(buf)} bytes, "
                   f"expected {expected_bytes}")
            buffers[label] = bytes(buf[:expected_bytes])
            p["_seqs"] = {0: next(iter(seqs))} if len(seqs) == 1 else {}

    return buffers


def framing_report(rec: RecordingSocket, res: Result):
    """E1.31 header self-consistency, reported as notes rather than
    failures: the rig's receivers accept these packets today, but a
    stricter implementation may not, so the deltas are worth surfacing."""
    for data, (_ip, port) in rec.packets:
        if port != SACN_PORT:
            continue
        pkt = parse_sacn(data)
        n = len(data)
        # Each PDU's length field must cover from its own first octet to
        # the end of the packet (E1.31 5.4 / 6.2 / 7.2).
        for name, got, want in (("root", pkt["root_len"], n - 16),
                                ("framing", pkt["framing_len"], n - 38),
                                ("DMP", pkt["dmp_len"], n - 115)):
            if got != want:
                res.note(f"E1.31 {name} PDU length field = {got}, but the "
                         f"{n}-byte packet requires {want} (informational — "
                         f"the rig's receivers ignore it)")
        slots_present = len(pkt["dmx"]) + 1   # + START code
        if pkt["prop_count"] != slots_present:
            res.note(f"E1.31 DMP property count = {pkt['prop_count']} but only "
                     f"{slots_present} slots are transmitted "
                     f"({len(pkt['dmx'])} data + START); a receiver that trusts "
                     f"the count reads 2 bytes past the packet")
        if data[22:38] == b'\x00' * 16:
            res.note("E1.31 CID is all zeros; the spec wants a unique non-zero "
                     "UUID per source (matters only if two sources ever share "
                     "a universe)")
        return  # one representative packet is enough


def check_mapping(buffers, plan, frames, res: Result, frame_no: int):
    for p in plan:
        buf = buffers.get(p["label"])
        if buf is None:
            continue
        cursor = 0
        mismatches = 0
        first_bad = None
        for s in p["strips"]:
            src = frames[s["group"]]
            for led, (r, c) in enumerate(s["indices"]):
                got = buf[cursor:cursor + 3]
                want = bytes(src[r, c][:3])
                if got != want:
                    mismatches += 1
                    if first_bad is None:
                        first_bad = (s["strip_idx"], led, (r, c), tuple(want), tuple(got))
                cursor += 3
        res.ok(mismatches == 0,
               f"frame {frame_no}: {p['label']} — {mismatches} LED(s) carry the wrong "
               f"pixel; first at strip {first_bad[0]} LED {first_bad[1]} "
               f"(canvas {first_bad[2]}): expected RGB {first_bad[3]}, got {first_bad[4]}"
               if first_bad else "")


def run_project(project_id: str, res: Result):
    print(f"\n{'=' * 72}\n  PROJECT: {project_id}\n{'=' * 72}")
    proj = load_project(project_id)
    plan, group_shapes = build_plan(proj, res)

    for p in plan:
        n_uni = math.ceil(p["pixel_count"] / 170)
        wire = (f"{n_uni} universes" if p["protocol"] == "sacn"
                else f"{math.ceil(p['pixel_count'] * 3 / imdmx.SACNPixelSender.DDP_MAX_PAYLOAD)} DDP chunks")
        print(f"  {p['label']:<28} {p['protocol']:<5} obj={p['object_id']} "
              f"{len(p['strips']):>2} strips {p['pixel_count']:>5} px -> {wire}")
    coverage_report(plan, group_shapes, res)

    sender, rec = build_sender(proj, plan, res)
    try:
        # --- two consecutive distinct frames: mapping + sequence advance ---
        prev_seqs = None
        for frame_no in (1, 2):
            rec.packets.clear()
            frames = make_frames(group_shapes, seed=frame_no)
            sender.send(frames)
            buffers = reassemble(rec, plan, res, frame_no)
            check_mapping(buffers, plan, frames, res, frame_no)

            seqs = {p["label"]: p.get("_seqs", {}) for p in plan}
            if prev_seqs is not None:
                for p in plan:
                    a = prev_seqs.get(p["label"]) or {}
                    b = seqs.get(p["label"]) or {}
                    if not a or not b:
                        continue
                    if p["protocol"] == "sacn":
                        bad = [u for u in a if b.get(u) != (a[u] + 1) % 256]
                        res.ok(not bad,
                               f"{p['label']}: sACN sequence did not advance on "
                               f"{len(bad)} universe(s) between frames")
                    else:
                        prev = a[0]
                        want = (prev % 15) + 1
                        res.ok(b.get(0) == want,
                               f"{p['label']}: DDP sequence {prev} -> {b.get(0)}, expected {want}")
            prev_seqs = seqs
            if frame_no == 1:
                framing_report(rec, res)
                total = sum(len(d) for d, _ in rec.packets)
                print(f"  wire: {len(rec.packets)} packets / {total:,} bytes per frame "
                      f"({len(rec.packets) * 40} pkt/s, {total * 40 / 1e6:.1f} MB/s at 40 FPS)")

        # --- numba fast path vs numpy fallback must agree byte-for-byte ---
        rec.packets.clear()
        frames = make_frames(group_shapes, seed=7)
        sender.send(frames)
        fast = [(d, a) for d, a in rec.packets]

        saved = imdmx.NUMBA_AVAILABLE
        imdmx.NUMBA_AVAILABLE = False
        try:
            rec.packets.clear()
            sender.send(frames)
            slow = [(d, a) for d, a in rec.packets]
        finally:
            imdmx.NUMBA_AVAILABLE = saved

        res.ok(len(fast) == len(slow),
               f"numba path emitted {len(fast)} packets, numpy fallback {len(slow)}")
        # strip the sequence byte (it advances per frame) before comparing
        def _strip_seq(pair):
            data, addr = pair
            if addr[1] == SACN_PORT:
                b = bytearray(data); b[111] = 0; return bytes(b), addr
            b = bytearray(data); b[1] = 0; return bytes(b), addr
        diff = sum(1 for f, s in zip(fast, slow) if _strip_seq(f) != _strip_seq(s))
        res.ok(diff == 0,
               f"numba and numpy extraction disagree on {diff} packet(s)")
        if not saved:
            res.note("numba NOT importable here — only the numpy fallback was exercised")
    finally:
        sender.udp_socket = None
        sender.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", action="append",
                    help="project id (repeatable); default = all under projects/")
    args = ap.parse_args()

    ids = args.project
    if not ids:
        ids = sorted(p.parent.name for p in (ROOT / "projects").glob("*/project.yaml"))
    if not ids:
        print("No projects found under projects/ — nothing to check.")
        return 1

    res = Result()
    for pid in ids:
        run_project(pid, res)

    print(f"\n{'=' * 72}")
    if res.notes:
        print("  NOTES")
        for n in res.notes:
            print(f"    - {n}")
    print(f"\n  {res.checks} checks run, {len(res.failures)} failed")
    for f in res.failures:
        print(f"    FAIL: {f}")
    print("=" * 72)
    return 1 if res.failures else 0


if __name__ == "__main__":
    sys.exit(main())
