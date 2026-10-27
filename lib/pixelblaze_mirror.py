"""Pixelblaze mirror — stream the live show output to any Pixelblaze seen on the LAN.

Pixelblazes announce themselves with a UDP beacon on port 1889 about once a
second. PixelblazeMirror listens for those beacons (plus any static
``hosts:`` from config.yaml) and, for each device it finds:

  1. opens its websocket API (ws://<ip>:81), reads ``pixelCount`` and the
     currently running pattern, and fetches its pixel map
     (http://<ip>/pixelmap.dat) if one was saved in the Mapper tab;
  2. looks for the receiver pattern (default name "GL_Simple Mirror",
     source in tools/hardware/pixelblaze_mirror.js) and makes it active —
     NOT saved to flash, so a reboot puts the device back on its own pattern;
  3. streams the frame the DMX receivers get (post gamma + brightness
     limiter) via ``setVars``, one packed number per pixel.

A device WITHOUT the receiver pattern is left alone: it gets one log line
telling you to paste the pattern in, and is re-checked every 30 s.

Mapping from the show canvas (H x W) to the device's pixels:
  Then SamplePlan thins that to <= ``max_samples`` per frame (the device's
  parse + per-LED cost is the limit, not the network) and the pattern
  fills the gaps, nearest-sample or blended (``interpolate``).

  - 2D/3D map  -> each pixel takes the max (``strip_reduce``) over the
                  canvas cell at its (x, y) map position (z is ignored)
  - 1D / none  -> the strip runs across the canvas columns (``strip_axis``),
                  each pixel taking the max (``strip_reduce``) down its column

Packing: a Pixelblaze number is 16.16 fixed point, so r + g/256 + b/65536
carries three bytes exactly (integer part r, fraction g then b).

On close() every device we switched goes back to the pattern it was running.
"""

import json
import socket
import struct
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from scipy import ndimage

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover - reported at start()
    websocket = None

BEACON_PORT = 1889
BEACON_PACKET = 42
WS_PORT = 81
HTTP_PORT = 80

# Binary websocket frames: byte 0 = message type, byte 1 = continuation flags.
_MSG_PROGRAM_LIST = 7
_FRAME_LAST = 4


# ----------------------------------------------------------------------
# One websocket conversation with one Pixelblaze
# ----------------------------------------------------------------------

class PixelblazeLink:
    """Thin synchronous client for the bits of the Pixelblaze API we use."""

    def __init__(self, ip: str, timeout: float = 4.0):
        self.ip = ip
        self.timeout = timeout
        self.ws = None

    def connect(self):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed")
        self.ws = websocket.create_connection(
            f"ws://{self.ip}:{WS_PORT}", timeout=self.timeout)
        # No preview frames: we never read them and they'd fill the socket.
        self.send_json({"sendUpdates": False})

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def send_json(self, obj):
        self.ws.send(json.dumps(obj, separators=(",", ":")))

    def get_config(self) -> tuple[dict, dict]:
        """(settings, activeProgram). The device answers getConfig with a
        settings JSON, an activeProgram JSON and (with an expander) a binary
        frame, in no fixed order."""
        self.send_json({"getConfig": True})
        settings, active = {}, {}
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline and not (settings and active):
            msg = self._recv()
            if not isinstance(msg, str):
                continue
            try:
                obj = json.loads(msg)
            except ValueError:
                continue
            if "activeProgram" in obj:
                active = obj.get("activeProgram") or {}
            elif "pixelCount" in obj:
                settings = obj
        if not settings:
            raise TimeoutError("no getConfig reply")
        return settings, active

    def list_programs(self) -> dict:
        """{pattern_id: name} for every pattern saved on the device."""
        self.send_json({"listPrograms": True})
        blob = b""
        deadline = time.monotonic() + 3 * self.timeout
        while time.monotonic() < deadline:
            msg = self._recv()
            if isinstance(msg, (bytes, bytearray)) and len(msg) >= 2 \
                    and msg[0] == _MSG_PROGRAM_LIST:
                blob += bytes(msg[2:])
                if msg[1] & _FRAME_LAST:
                    break
        programs = {}
        for line in blob.decode("utf-8", "replace").split("\n"):
            parts = line.split("\t")
            if len(parts) == 2:
                programs[parts[0]] = parts[1]
        return programs

    def set_active_program(self, program_id: str):
        self.send_json({"activeProgramId": program_id, "save": False})

    def get_map(self, pixel_count: int):
        """(N, dims) float array in 0..1 from the Mapper tab, or None when
        there is no map or it is stale (saved for a different pixel count)."""
        try:
            with urllib.request.urlopen(f"http://{self.ip}:{HTTP_PORT}/pixelmap.dat",
                                        timeout=self.timeout) as r:
                data = r.read()
        except (urllib.error.URLError, OSError):
            return None
        if len(data) < 12:
            return None
        version, dims, size = struct.unpack("<3I", data[:12])
        if version not in (1, 2) or dims not in (1, 2, 3):
            return None
        dtype = np.uint8 if version == 1 else np.dtype("<u2")
        values = np.frombuffer(data[12:12 + size], dtype=dtype)
        if values.size != pixel_count * dims:
            return None
        return values.reshape(pixel_count, dims).astype(np.float32) / \
            float(np.iinfo(dtype).max)

    def _recv(self):
        try:
            return self.ws.recv()
        except websocket.WebSocketTimeoutException:
            return None


def pack_pixels(rgb: np.ndarray) -> str:
    """(N, 3) uint8 -> comma-joined r + g/256 + b/65536 values.

    A quarter step (1/262144) is added before printing to 6 decimals, so the
    device lands on the right 16.16 value whether its parser rounds or
    truncates. Black is sent as a bare 0 to keep big frames small."""
    rgb = rgb.astype(np.uint32)
    word = rgb[:, 0] * 65536 + rgb[:, 1] * 256 + rgb[:, 2]
    packed = (word + 0.25) / 65536.0
    return ",".join("%.6f" % v if w else "0"
                    for v, w in zip(packed.tolist(), word.tolist()))


# The receiver pattern's sample array size. It also keeps a frame inside
# one websocket message: the device drops a message much past 16 KB
# (firmware 3.70: 15 KB fine, 24 KB resets the connection) and 1000 fully
# lit samples pack to ~11 KB.
MAX_SAMPLES = 1000

PING = '{"ping":true}'


def find_segment(coords, n: int) -> int:
    """Pixels per segment, from big jumps between consecutive pixels in the
    map: the largest length that puts every jump on a segment boundary
    (the Tree's strands run 30 px then 270 px -> 30). The whole device when
    there's no map, no jumps, or only a uselessly short common length."""
    if coords is None or coords.shape[1] < 2 or n < 2:
        return n
    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    breaks = np.flatnonzero(d > max(10 * float(np.median(d)), 0.2)) + 1
    if not len(breaks):
        return n
    lengths = np.diff(np.concatenate(([0], breaks, [n])))
    seg = int(np.gcd.reduce(lengths))
    return seg if seg >= 8 else n


class SamplePlan:
    """Which pixels to send. ``k`` samples per ``seg``-pixel segment, spread
    evenly from its first pixel to its last; the pattern interpolates
    between them. Each sample carries the max over the pixels nearest it,
    so a lone lit pixel between two samples still shows."""

    def __init__(self, n: int, seg: int, max_samples: int = 600,
                 interpolate: bool = False):
        self.interp = int(bool(interpolate))
        seg = max(1, min(int(seg), n))
        n_seg = -(-n // seg)
        budget = max(1, min(int(max_samples), MAX_SAMPLES))
        stride = 1
        while True:
            k = seg if stride == 1 else -(-(seg - 1) // stride) + 1
            if n_seg * k <= budget or k <= 2:
                break
            stride += 1
        self.n, self.seg, self.k = n, seg, k
        if k == seg:
            local = [[j] for j in range(seg)]
        else:
            step = (seg - 1) / (k - 1)
            local = [list(range(max(0, int(np.ceil((j - 0.5) * step))),
                                min(seg - 1, int(np.floor((j + 0.5) * step))) + 1))
                     for j in range(k)]
        width = max(len(w) for w in local)
        local = np.array([w + [w[-1]] * (width - len(w)) for w in local])
        idx = (np.arange(n_seg)[:, None, None] * seg + local[None]).reshape(-1, width)
        self.idx = np.minimum(idx, n - 1)[:MAX_SAMPLES]

    def message(self, rgb: np.ndarray, seq: int) -> str:
        samples = rgb[self.idx].max(axis=1)
        return ('{"setVars":{"seg":%d,"k":%d,"interp":%d,"seq":%d,"p0":[%s]}}'
                % (self.seg, self.k, self.interp, seq, pack_pixels(samples)))


# ----------------------------------------------------------------------
# Canvas -> device pixel sampling
# ----------------------------------------------------------------------

def build_sampler(coords, n: int, h: int, w: int,
                  strip_axis: str = "columns", strip_reduce: str = "max",
                  flip_y: bool = False):
    """Return frame (h, w, 3) -> (n, 3) for a device with ``n`` pixels.

    ``coords`` is the device's map ((n, dims) in 0..1) or None."""
    if coords is not None and coords.shape[1] >= 2:
        y = coords[:, 1]
        if flip_y:
            y = 1.0 - y
        # Each pixel stands for a cell of the canvas, not a point: 8 strands
        # mapped across 128 columns cover 16 columns each, and a point
        # sample would miss whatever lit between them. Cell size per axis
        # = canvas size / distinct map positions on that axis; the cells
        # tile the canvas edge to edge (the outer cells' centres sit half a
        # cell in from the edges).
        x = coords[:, 0]
        bw = max(1, round(w / len(np.unique(np.rint(x * (w - 1))))))
        bh = max(1, round(h / len(np.unique(np.rint(y * (h - 1))))))
        cols = np.clip(np.rint(x * (w - bw) + (bw - 1) / 2), 0, w - 1).astype(np.intp)
        rows = np.clip(np.rint(y * (h - bh) + (bh - 1) / 2), 0, h - 1).astype(np.intp)
        if bw == 1 and bh == 1:
            return lambda frame: frame[rows, cols]
        pool = (ndimage.uniform_filter if strip_reduce == "mean"
                else ndimage.maximum_filter)
        size = (bh, bw, 1)
        return lambda frame: pool(frame, size=size, mode="nearest")[rows, cols]

    # Strip: a 1D map gives each pixel's position, otherwise evenly spaced.
    if coords is not None:
        pos = coords[:, 0]
    else:
        pos = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    across_cols = strip_axis != "rows"
    length = w if across_cols else h
    idx = np.clip(np.rint(pos * (length - 1)), 0, length - 1).astype(np.intp)
    reduce_axis = 0 if across_cols else 1
    if strip_reduce == "mean":
        def sample(frame):
            return frame.mean(axis=reduce_axis).astype(np.uint8)[idx]
    else:
        def sample(frame):
            return frame.max(axis=reduce_axis)[idx]
    return sample


# ----------------------------------------------------------------------
# One device: connect, take over, stream, reconnect
# ----------------------------------------------------------------------

class _Device:
    RETRY_S = 10.0
    NO_PATTERN_RETRY_S = 30.0

    def __init__(self, mirror: "PixelblazeMirror", ip: str, host: str = None):
        self.mirror = mirror
        # A static hosts: entry is resolved on THIS device's thread at every
        # attempt (see _session), never on the caller's - a slow or failing
        # resolver must not stall the show's boot, and a box that appears
        # later must still be found.
        self.host = host
        self.ip = ip or host
        self.name = host or ip
        self.status = "connecting"
        self.pixel_count = 0
        self._prior_program = None   # what it ran before we switched it
        self._link = None
        self._acks = 0
        self._ack_cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"pixelblaze-{ip}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._ack_cv:
            self._ack_cv.notify_all()   # a send waiting on its ack
        self._thread.join(timeout=3.0)

    def _set_status(self, status: str):
        if status != self.status:
            self.status = status
            print(f"[Pixelblaze] {self.name} ({self.ip}): {status}")

    def _run(self):
        while not self._stop.is_set():
            wait = self.RETRY_S
            try:
                wait = self._session()
            except Exception as e:
                self._set_status(f"offline ({e.__class__.__name__}: {e})")
            finally:
                self._release()
            self._stop.wait(wait)

    def _session(self) -> float:
        """One connection's worth of work. Returns the seconds to wait
        before trying again."""
        cfg = self.mirror.cfg
        if self.host:
            self.ip = socket.gethostbyname(self.host)   # raises -> "offline", retried
        link = self._link = PixelblazeLink(self.ip)
        link.connect()
        settings, active = link.get_config()
        self.name = settings.get("name") or self.ip
        self.pixel_count = int(settings.get("pixelCount") or 0)
        if self.pixel_count <= 0:
            self._set_status("reports 0 pixels, skipping")
            return self.NO_PATTERN_RETRY_S

        wanted = str(cfg.get("pattern", "GL_Simple Mirror")).strip().lower()
        programs = link.list_programs()
        mirror_id = next((pid for pid, name in programs.items()
                          if name.strip().lower() == wanted), None)
        if mirror_id is None:
            self._set_status(
                f"no '{cfg.get('pattern', 'GL_Simple Mirror')}' pattern — paste "
                f"tools/hardware/pixelblaze_mirror.js into its editor and save "
                f"it under that name")
            return self.NO_PATTERN_RETRY_S

        current = active.get("activeProgramId")
        if current != mirror_id:
            if self._prior_program is None:
                self._prior_program = current
            link.set_active_program(mirror_id)

        coords = link.get_map(self.pixel_count)
        shape = ("strip" if coords is None or coords.shape[1] == 1
                 else f"{coords.shape[1]}D map")
        plan = SamplePlan(self.pixel_count,
                          int(cfg.get("segment") or 0)
                          or find_segment(coords, self.pixel_count),
                          int(cfg.get("max_samples", 600)),
                          interpolate=bool(cfg.get("interpolate", False)))
        n_seg = -(-self.pixel_count // plan.seg)
        self._set_status(f"mirroring, {self.pixel_count} px ({shape}), "
                         f"{n_seg} x {plan.k} samples")

        # One frame in flight: each frame is followed by a ping, and the next
        # frame (always the newest) goes only once the device has acked it.
        # Sending at our own fps instead let a slower device fall seconds
        # behind (the Tree: 1.8 s of frames queued in TCP buffers). The ping
        # MUST be its own message: in the same message as setVars the device
        # acks but drops the vars. The device seems to take one message per
        # render cycle, so the ping costs a cycle (Tree: ~185 ms per frame);
        # pinging every 2nd-4th frame bought <2 fps for 2-6x the lag.
        # The drain thread reads the acks (and discards the stats).
        link.ws.settimeout(5.0)
        drain = threading.Thread(target=self._drain, args=(link,), daemon=True)
        drain.start()

        interval = 1.0 / max(1.0, float(cfg.get("fps", 30)))
        gain = float(cfg.get("gain", 1.0))
        sampler, sampler_key = None, None
        last_frame, seq = None, 0
        next_t = time.monotonic()
        # 10 s health window: how long each frame waited for its ack (the
        # network + device) against everything else (our side: sampling,
        # packing, a busy app holding the GIL). Logged only when slow.
        win_t, win_rtts, win_frames = time.monotonic(), [], 0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - win_t >= 10.0:
                if win_frames:
                    self._report_health(now - win_t, win_frames, win_rtts)
                win_t, win_rtts, win_frames = now, [], 0
            frame = self.mirror.latest_frame()
            if frame is not None and frame is not last_frame:
                h, w = frame.shape[:2]
                if (h, w) != sampler_key:
                    sampler = build_sampler(
                        coords, self.pixel_count, h, w,
                        strip_axis=cfg.get("strip_axis", "columns"),
                        strip_reduce=cfg.get("strip_reduce", "max"),
                        flip_y=bool(cfg.get("flip_y", False)))
                    sampler_key = (h, w)
                rgb = sampler(frame)
                if gain != 1.0:
                    rgb = np.clip(rgb * gain, 0, 255).astype(np.uint8)
                seq = (seq + 1) % 30000
                with self._ack_cv:
                    acked = self._acks
                    sent_t = time.monotonic()
                    link.ws.send(plan.message(rgb, seq))
                    link.ws.send(PING)
                    if not self._ack_cv.wait_for(
                            lambda: self._acks > acked or self._stop.is_set(),
                            timeout=5.0):
                        raise TimeoutError("no ack for 5 s")
                    win_rtts.append(time.monotonic() - sent_t)
                    win_frames += 1
                last_frame = frame
            if not drain.is_alive():
                raise ConnectionError("websocket closed")
            next_t = max(next_t + interval, time.monotonic())
            self._stop.wait(next_t - time.monotonic())
        return 0.0

    def _report_health(self, span: float, frames: int, rtts: list):
        fps = frames / span
        rtt_med = float(np.median(rtts)) * 1000
        rtt_max = max(rtts) * 1000
        ours = max(0.0, span - sum(rtts)) / frames * 1000
        self.health = (f"{fps:.1f} fps, ack median {rtt_med:.0f} ms / "
                       f"max {rtt_max:.0f} ms, our side {ours:.0f} ms per frame")
        slow = fps < 4.0 or rtt_max > 1000.0
        if slow or getattr(self, "_was_slow", False):
            print(f"[Pixelblaze] {self.name}: {'SLOW ' if slow else 'ok again '}"
                  f"{self.health}")
        self._was_slow = slow

    def _drain(self, link: PixelblazeLink):
        while link.ws is not None and not self._stop.is_set():
            try:
                msg = link.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                return
            if isinstance(msg, str) and '"ack"' in msg:
                with self._ack_cv:
                    self._acks += 1
                    self._ack_cv.notify_all()

    def _release(self):
        """Close the socket; on shutdown, give the device its pattern back."""
        link, self._link = self._link, None
        if link is None:
            return
        if self._stop.is_set() and self._prior_program and link.ws is not None:
            try:
                link.set_active_program(self._prior_program)
                print(f"[Pixelblaze] {self.name}: restored its own pattern")
            except Exception:
                pass
        link.close()


# ----------------------------------------------------------------------
# The service: beacon listener + one _Device per Pixelblaze
# ----------------------------------------------------------------------

class PixelblazeMirror:
    """Frame sink for RenderPipeline: ``send(corrected_frames)`` each frame."""

    def __init__(self, cfg: dict, bind_ip: str = ""):
        self.cfg = dict(cfg or {})
        self.bind_ip = bind_ip
        self._latest = None
        self._devices: dict[str, _Device] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._beacon_thread = None

    def start(self):
        if websocket is None:
            print("[Pixelblaze] websocket-client not installed "
                  "(pip install websocket-client) — mirror disabled")
            return
        for host in self.cfg.get("hosts") or []:
            try:
                self._ensure_device(None, host=str(host))
            except Exception as e:                       # never let a bad entry stop the show
                print(f"[Pixelblaze] host {host!r} skipped: {e}")
        if self.cfg.get("discover", True):
            self._beacon_thread = threading.Thread(
                target=self._listen_beacons, daemon=True,
                name="pixelblaze-beacons")
            self._beacon_thread.start()
        print("[Pixelblaze] mirror started (listening for beacons on "
              f"UDP {BEACON_PORT})" if self.cfg.get("discover", True)
              else "[Pixelblaze] mirror started (static hosts only)")

    def send(self, frames: dict):
        group = self.cfg.get("group")
        frame = frames.get(group) if group else None
        if frame is None and frames:
            frame = next(iter(frames.values()))
        self._latest = frame

    def latest_frame(self):
        return self._latest

    def status(self) -> list[dict]:
        with self._lock:
            return [{"ip": d.ip, "name": d.name, "pixels": d.pixel_count,
                     "status": d.status, "health": getattr(d, "health", "")}
                    for d in self._devices.values()]

    def close(self):
        self._stop.set()
        with self._lock:
            devices = list(self._devices.values())
        for d in devices:
            d.stop()
        # The beacon listener wakes at least once a second (its recv timeout)
        # and exits on _stop; wait for it so UDP 1889 is free again when
        # close() returns (a restart or a second mirror can bind it).
        t = self._beacon_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.5)

    def _ensure_device(self, ip, host: str = None):
        key = f"host:{host}" if host else ip
        with self._lock:
            if self._stop.is_set() or key in self._devices:
                return
            # A beacon from a box we already mirror as a static host (its
            # resolved ip) must not start a second stream to it.
            if not host and any(d.ip == ip for d in self._devices.values()):
                return
            self._devices[key] = _Device(self, ip, host=host)

    def _listen_beacons(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # ALL interfaces, not dmx.bind_ip: beacons are broadcasts, and on
            # Linux a UDP socket bound to a unicast address never receives
            # broadcast datagrams - pinning DMX to the wired NIC silently
            # turned discovery off. (bind_ip still pins nothing outbound here:
            # device traffic is TCP to the device's own address.)
            sock.bind(("", BEACON_PORT))
        except OSError as e:
            print(f"[Pixelblaze] can't listen on UDP {BEACON_PORT} ({e}); "
                  f"only static hosts will be mirrored")
            return
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, (ip, _port) = sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) >= 12 and \
                    struct.unpack("<L", data[:4])[0] == BEACON_PACKET:
                self._ensure_device(ip)
        sock.close()


def discover(seconds: float = 3.0, bind_ip: str = "") -> set[str]:
    """IPs of every Pixelblaze beaconing within ``seconds``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_ip or "", BEACON_PORT))
    sock.settimeout(0.25)
    found = set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            data, (ip, _port) = sock.recvfrom(64)
        except socket.timeout:
            continue
        if len(data) >= 12 and struct.unpack("<L", data[:4])[0] == BEACON_PACKET:
            found.add(ip)
    sock.close()
    return found
