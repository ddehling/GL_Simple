"""Weight of Light — 9-box hardware emulator.

Fakes the array of buttons + 24 GHz range sensors for all nine WoL shards by
sending the same OSC the real EtherNode boxes send, to a *running* GL_Simple
instance. Lets you test the participatory features (collective-press
lightning, presence-driven theme reactivity, presence_glow) without any
hardware.

    # 1. run the app (ideally in the Elements set so the theme shaders react)
    python Stories_OGL.py --project weight_of_light
    # 2. run this emulator (same machine: defaults work; a Pi: --host <ip>)
    python tools/wol_box_emulator.py

GUI: a ring of the nine shards (laid out from geometry.yaml).
  * Click a shard            -> press its lightning button (button 3).
  * "Press ALL" / pick a few -> simultaneous presses = a collective storm
                                (the more shards within ~1 s, the bigger).
  * Per-shard distance slider -> that shard's "person": left = closer (more
                                presence), far right = nobody. Drives
                                presence_weight -> theme reactivity + glow.
  * "Walk in" / "Approach all" -> animates a slider inward to test approach.

OSC it emits per box (matches projects/weight_of_light/radar.py + button_router):
  /en/<host>/button/<idx>            <1>
  /en/<host>/radar/model             "ld2450"
  /en/<host>/radar                   <target_count 0/1>
  /en/<host>/radar/d                 <closest distance, cm>
  /en/<host>/radar/t0                [x_mm, y_mm, speed_cm_s, res_mm]

Run ``--selftest`` to exercise the config-load + OSC message construction
without opening a window (used in CI / headless dev).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config: discover the nine boxes (host stem + ring position) from the project
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absent sentinel: a distance slider at/above this means "nobody here".
ABSENT_CM = 400.0
NEAR_CM = 30.0
RADAR_HZ = 5.0            # how often we re-emit each box's sensor state
RES_MM = 50              # LD2450 resolution field (cosmetic)


@dataclass
class Box:
    oid: int
    stem: str            # OSC host stem, e.g. "ethernode-north"
    name: str            # display name, e.g. "north"
    nx: float            # ring position, normalized 0..1 (x)
    ny: float            # ring position, normalized 0..1 (y)


def load_boxes(project_id: str = "weight_of_light"):
    """Return (boxes, canvas_w, canvas_h). Host stems come from the project's
    receivers; ring positions from geometry.yaml. Falls back to a generated
    ring if either file is missing."""
    import yaml
    proj = os.path.join(REPO_ROOT, "projects", project_id)
    stems: dict[int, str] = {}
    names: dict[int, str] = {}
    pos: dict[int, tuple] = {}
    cw, ch = 1280.0, 1024.0

    try:
        with open(os.path.join(proj, "project.yaml"), encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        for r in raw.get("receivers", []) or []:
            if not isinstance(r, dict):
                continue
            try:
                oid = int(r.get("object_id", -1))
            except (TypeError, ValueError):
                continue
            host = str(r.get("host", "") or "")
            if oid >= 0 and host:
                stems[oid] = host[:-6] if host.endswith(".local") else host
    except OSError:
        pass

    try:
        with open(os.path.join(proj, "geometry.yaml"), encoding="utf-8") as fh:
            geom = yaml.safe_load(fh) or {}
        canvas = geom.get("canvas") or {}
        cw = float(canvas.get("width", cw))
        ch = float(canvas.get("height", ch))
        for o in geom.get("objects", []) or []:
            oid = int(o["id"])
            names[oid] = str(o.get("name", oid))
            pos[oid] = (float(o.get("x", cw / 2)), float(o.get("y", ch / 2)))
    except (OSError, KeyError):
        pass

    # Fallback: if no receivers were found, assume 9 ethernode boxes.
    if not stems:
        ring = ["central", "north", "northeast", "east", "southeast",
                "south", "southwest", "west", "northwest"]
        stems = {i: f"ethernode-{n}" for i, n in enumerate(ring)}
        names = {i: n for i, n in enumerate(ring)}

    boxes = []
    n = max(len(stems), 1)
    for k, oid in enumerate(sorted(stems)):
        if oid in pos:
            x, y = pos[oid]
            nx, ny = x / cw, y / ch
        else:
            # generated ring, center reserved for oid 0
            if oid == 0:
                nx, ny = 0.5, 0.5
            else:
                ang = 2 * math.pi * (k - 1) / max(n - 1, 1) - math.pi / 2
                nx, ny = 0.5 + 0.4 * math.cos(ang), 0.5 + 0.4 * math.sin(ang)
        boxes.append(Box(oid, stems[oid], names.get(oid, str(oid)), nx, ny))
    return boxes, cw, ch


# ---------------------------------------------------------------------------
# OSC message construction (pure — testable without a socket)
# ---------------------------------------------------------------------------

def button_message(stem: str, btn: int = 3):
    return (f"/en/{stem}/button/{btn}", 1)


def spread_indices(total: int, n: int):
    """n indices spread evenly around `total` items (deduped) — so "Press N"
    fires N shards spaced around the ring rather than clumped together."""
    n = max(1, min(int(n), int(total)))
    idx = sorted({int(round(i * total / n)) % total for i in range(n)})
    # Rounding collisions can drop one or two; backfill from the front.
    k = 0
    while len(idx) < n and k < total:
        if k not in idx:
            idx.append(k)
        k += 1
    return sorted(idx)[:n]


def present_messages(stem: str, cm: float, speed_cm_s: float = 0.0,
                     az_norm: float = 0.0, half_fov_deg: float = 60.0):
    """OSC for 'one person at <cm> and azimuth az_norm' on this box (LD2450).

    ``az_norm`` in -1..+1 maps across the sensor's +/- half_fov: -1 = hard
    left, 0 = straight ahead, +1 = hard right. We place the target at that
    bearing so the app's direction logic (atan2(x, y) in radar/presence_glow)
    actually sees an angle — without this every target reads dead-centre and
    the glow never moves."""
    az = max(-1.0, min(1.0, az_norm)) * math.radians(half_fov_deg)
    r_mm = cm * 10.0
    x_mm = int(round(math.sin(az) * r_mm))      # lateral
    y_mm = int(round(math.cos(az) * r_mm))      # forward
    return [
        (f"/en/{stem}/radar/model", "ld2450"),
        (f"/en/{stem}/radar", 1),
        (f"/en/{stem}/radar/d", int(round(cm))),
        (f"/en/{stem}/radar/t0",
         [x_mm, y_mm, int(round(speed_cm_s)), RES_MM]),
    ]


def absent_messages(stem: str):
    """OSC for 'nobody here' (keeps the sensor online but target_count 0)."""
    return [
        (f"/en/{stem}/radar/model", "ld2450"),
        (f"/en/{stem}/radar", 0),
        (f"/en/{stem}/radar/t0", [0, 0, 0, 0]),
    ]


class OscBackend:
    """Thin wrapper over python-osc's UDP client. Swappable for a recorder
    in --selftest."""
    def __init__(self, host: str, port: int):
        from pythonosc import udp_client
        self.client = udp_client.SimpleUDPClient(host, port)
        self.host = host
        self.port = port

    def send(self, address, value):
        self.client.send_message(address, value)


class RecordingBackend:
    def __init__(self):
        self.sent = []

    def send(self, address, value):
        self.sent.append((address, value))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui(boxes, backend, button_idx: int):
    import tkinter as tk

    NODE_R = 30
    CANVAS = 520
    MARGIN = 60

    root = tk.Tk()
    root.title("WoL 9-box emulator")

    # --- top: global controls + connection info ---
    top = tk.Frame(root, padx=8, pady=6)
    top.pack(fill="x")
    conn = getattr(backend, "host", "?"), getattr(backend, "port", "?")
    tk.Label(top, text=f"-> {conn[0]}:{conn[1]}   button {button_idx}",
             fg="#888").pack(side="left")
    # Press exactly N shards (spread around the ring) so you can compare
    # storm sizes — e.g. 2 vs 5 vs 9 — and see the exponential escalation.
    tk.Label(top, text="   N:").pack(side="left")
    nvar = tk.IntVar(value=2)
    tk.Spinbox(top, from_=1, to=len(boxes), width=3,
               textvariable=nvar).pack(side="left")
    tk.Button(top, text="Press N",
              command=lambda: press_n(nvar.get())).pack(side="left", padx=4)
    tk.Button(top, text="Press ALL",
              command=lambda: press_n(len(boxes))).pack(side="left", padx=4)
    tk.Button(top, text="Clear ALL sensors",
              command=lambda: clear_all()).pack(side="left", padx=6)
    tk.Button(top, text="Approach ALL",
              command=lambda: approach_all()).pack(side="left", padx=6)

    body = tk.Frame(root)
    body.pack(fill="both", expand=True)

    # --- left: ring canvas ---
    canvas = tk.Canvas(body, width=CANVAS, height=CANVAS, bg="#101014",
                       highlightthickness=0)
    canvas.grid(row=0, column=0, padx=6, pady=6)

    # --- right: per-box rows (distance sliders + walk) ---
    panel = tk.Frame(body, padx=6, pady=6)
    panel.grid(row=0, column=1, sticky="n")
    tk.Label(panel, text="distance (cm) — far right = nobody",
             fg="#888").grid(row=0, column=0, columnspan=3, sticky="w")

    sliders: dict[int, tk.Scale] = {}
    angles: dict[int, tk.Scale] = {}
    node_ids: dict[int, int] = {}
    prev_cm: dict[int, float] = {}
    walk_jobs: dict[int, str] = {}

    def px(b):
        return MARGIN + b.nx * (CANVAS - 2 * MARGIN)

    def py(b):
        return MARGIN + b.ny * (CANVAS - 2 * MARGIN)

    def press(oid):
        stem = box_by_oid[oid].stem
        addr, val = button_message(stem, button_idx)
        backend.send(addr, val)
        flash(oid)
        log(f"press {box_by_oid[oid].name}")

    def press_n(n):
        """Press exactly N shards (spread around the ring) at once — for
        comparing storm sizes. The app's rolling press-memory accumulates
        over a few seconds, so leave ~4 s between tests for a clean read."""
        ordered = sorted(boxes, key=lambda z: z.oid)
        chosen = [ordered[i] for i in spread_indices(len(ordered), n)]
        for b in chosen:
            addr, val = button_message(b.stem, button_idx)
            backend.send(addr, val)
            flash(b.oid)
        names = ", ".join(b.name for b in chosen)
        log(f"PRESS {len(chosen)} ({names}) -> storm  (wait ~4s before next test)")

    def clear_all():
        for oid, s in sliders.items():
            cancel_walk(oid)
            s.set(ABSENT_CM)
        log("cleared all sensors")

    def approach(oid, target=NEAR_CM + 20, step=14, ms=120):
        cancel_walk(oid)

        def stepfn():
            s = sliders[oid]
            v = s.get()
            if v <= target:
                walk_jobs.pop(oid, None)
                return
            s.set(max(target, v - step))
            walk_jobs[oid] = root.after(ms, stepfn)
        # start from absent so it's a real walk-in
        if sliders[oid].get() < ABSENT_CM - 1:
            sliders[oid].set(ABSENT_CM)
        stepfn()
        log(f"approach {box_by_oid[oid].name}")

    def approach_all():
        for b in boxes:
            approach(b.oid)

    def cancel_walk(oid):
        j = walk_jobs.pop(oid, None)
        if j is not None:
            root.after_cancel(j)

    def flash(oid):
        nid = node_ids.get(oid)
        if nid is None:
            return
        canvas.itemconfig(nid, fill="#ffffff")
        root.after(160, lambda: refresh_node(oid))

    box_by_oid = {b.oid: b for b in boxes}

    # draw nodes + build slider rows
    for i, b in enumerate(sorted(boxes, key=lambda z: z.oid)):
        x, y = px(b), py(b)
        nid = canvas.create_oval(x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R,
                                 fill="#23323a", outline="#3a4a52", width=2,
                                 tags=(f"node{b.oid}",))
        canvas.create_text(x, y - 6, text=b.name, fill="#cfe", font=("", 8))
        tid = canvas.create_text(x, y + 8, text="", fill="#9fb", font=("", 7))
        node_ids[b.oid] = nid
        canvas.tag_bind(f"node{b.oid}", "<Button-1>",
                        lambda e, o=b.oid: press(o))
        # store the distance-readout text id on the node for refresh
        b_dist_text = tid

        row = tk.Frame(panel)
        row.grid(row=i + 1, column=0, columnspan=3, sticky="w", pady=1)
        tk.Label(row, text=f"{b.oid} {b.name:<10}", width=12, anchor="w",
                 font=("", 8)).pack(side="left")
        s = tk.Scale(row, from_=NEAR_CM, to=ABSENT_CM, orient="horizontal",
                     length=150, showvalue=True, resolution=5)
        s.set(ABSENT_CM)
        s.pack(side="left")
        sliders[b.oid] = s
        prev_cm[b.oid] = ABSENT_CM
        # Direction: -1 = hard left, 0 = ahead, +1 = hard right.
        ang = tk.Scale(row, from_=-1.0, to=1.0, orient="horizontal",
                       length=80, showvalue=False, resolution=0.1,
                       label="L<dir>R", font=("", 6))
        ang.set(0.0)
        ang.pack(side="left", padx=(4, 0))
        angles[b.oid] = ang
        tk.Button(row, text="press", font=("", 7),
                  command=lambda o=b.oid: press(o)).pack(side="left", padx=2)
        tk.Button(row, text="walk", font=("", 7),
                  command=lambda o=b.oid: approach(o)).pack(side="left")
        b._dist_text = b_dist_text   # type: ignore[attr-defined]

    logvar = tk.StringVar(value="ready")
    tk.Label(root, textvariable=logvar, anchor="w", fg="#8ab",
             padx=8).pack(fill="x")

    def log(msg):
        logvar.set(msg)

    def presence_color(cm):
        if cm >= ABSENT_CM:
            return "#23323a", ""
        # closeness 0..1 (near=1)
        c = max(0.0, min(1.0, (ABSENT_CM - cm) / (ABSENT_CM - NEAR_CM)))
        g = int(60 + 180 * c)
        r = int(20 + 40 * c)
        bl = int(40 + 30 * c)
        return f"#{r:02x}{g:02x}{bl:02x}", f"{int(cm)}cm"

    def refresh_node(oid):
        cm = sliders[oid].get()
        fill, label = presence_color(cm)
        canvas.itemconfig(node_ids[oid], fill=fill)
        canvas.itemconfig(box_by_oid[oid]._dist_text, text=label)

    # --- periodic sender: emit each box's sensor state ~RADAR_HZ ---
    period_ms = int(1000 / RADAR_HZ)

    def tick():
        now = time.time()
        dt = max(1e-3, period_ms / 1000.0)
        for b in boxes:
            cm = float(sliders[b.oid].get())
            if cm < ABSENT_CM:
                speed = max(0.0, (prev_cm[b.oid] - cm) / dt)  # +ve approaching
                az = float(angles[b.oid].get())
                for addr, val in present_messages(b.stem, cm, speed, az_norm=az):
                    backend.send(addr, val)
            else:
                for addr, val in absent_messages(b.stem):
                    backend.send(addr, val)
            prev_cm[b.oid] = cm
            refresh_node(b.oid)
        root.after(period_ms, tick)

    for b in boxes:
        refresh_node(b.oid)
    root.after(period_ms, tick)
    root.mainloop()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="OSC target (the running app). Default 127.0.0.1")
    ap.add_argument("--port", type=int, default=9001,
                    help="OSC port (config.yaml osc.port). Default 9001")
    ap.add_argument("--project", default="weight_of_light")
    ap.add_argument("--button", type=int, default=3,
                    help="Button index to press (3 = lightning). Default 3")
    ap.add_argument("--selftest", action="store_true",
                    help="Headless: verify config-load + message construction")
    args = ap.parse_args()

    boxes, cw, ch = load_boxes(args.project)
    if not boxes:
        sys.exit("no boxes found")

    if args.selftest:
        rec = RecordingBackend()
        print(f"loaded {len(boxes)} boxes:")
        for b in boxes:
            print(f"  oid={b.oid} stem={b.stem!r} name={b.name!r} "
                  f"pos=({b.nx:.2f},{b.ny:.2f})")
        # button
        a, v = button_message(boxes[0].stem, args.button)
        rec.send(a, v)
        # present at 80cm + absent
        for a, v in present_messages(boxes[0].stem, 80.0, speed_cm_s=25):
            rec.send(a, v)
        for a, v in absent_messages(boxes[1].stem):
            rec.send(a, v)
        print(f"\nsample messages ({len(rec.sent)}):")
        for a, v in rec.sent:
            print(f"  {a}  {v}")
        # sanity assertions
        assert all(b.stem for b in boxes), "every box needs a host stem"
        assert button_message("x", 3) == ("/en/x/button/3", 1)
        pm = present_messages("x", 80.0)
        assert (f"/en/x/radar", 1) in pm and (f"/en/x/radar/d", 80) in pm
        assert (f"/en/x/radar", 0) in absent_messages("x")
        # "Press N" subset selection: distinct, correct count, spread.
        nb = len(boxes)
        print("\nPress-N spread (of "f"{nb} boxes):")
        for n in (1, 2, 3, 5, nb):
            idx = spread_indices(nb, n)
            print(f"  N={n}: indices {idx}")
            assert len(idx) == min(n, nb), f"spread_indices({nb},{n}) -> {idx}"
            assert len(set(idx)) == len(idx), "duplicate indices"
        print("\nSELFTEST OK")
        return

    try:
        backend = OscBackend(args.host, args.port)
    except Exception as e:
        sys.exit(f"could not create OSC client: {e}")
    try:
        run_gui(boxes, backend, args.button)
    except Exception as e:
        sys.exit(f"GUI failed (no display?): {e}")


if __name__ == "__main__":
    main()
