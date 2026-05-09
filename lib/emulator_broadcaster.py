"""EmulatorBroadcaster — localhost TCP feed of per-frame group canvases
for the layout editor's live preview.

The render loop calls :meth:`publish` once per frame with two dicts:

  * ``raw_frames``       — pre-correction FBOs as the shaders rendered
                           them (group canvases display these so you see
                           "what is actually being generated")
  * ``corrected_frames`` — post-gamma + post-brightness frames (physical
                           layout displays these so the LEDs match what
                           goes out the wire)

Both are ``dict[group_id_str -> np.ndarray (H, W, 3) uint8]``.

Only listens when an explicit port is set (the editor passes it via
``--emulator-port``). With no port the broadcaster is a no-op so a
normal CLI run is unaffected.

Wire format (length-prefixed messages over a TCP stream)::

    [4B msg_len BE][payload]

    payload:
        [1B stage]              0 = raw, 1 = corrected
        [1B group_id_len][group_id_bytes]
        [2B width BE][2B height BE]
        [4B seq BE]
        [width * height * 3 bytes RGB rows top-to-bottom]

Multiple groups per frame are emitted as separate consecutive
messages. The editor reads as many as available and uses whatever
arrives — a missed frame is fine, the next one supersedes it.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Dict, Optional

import numpy as np

# Stage constants exposed for the editor to import.
STAGE_RAW = 0
STAGE_CORRECTED = 1

# Header layout: stage(1) + gid_len(1) + width(2) + height(2) + seq(4)
_HDR_FMT = ">BBHHL"
_HDR_LEN = struct.calcsize(_HDR_FMT)


class EmulatorBroadcaster:
    """Thread-backed TCP listener that pushes frames to a single client.

    Designed for the layout editor: the editor opens the editor first
    (calls "Launch"), then spawns the engine with this port, so by the
    time the engine starts publishing the editor is already accepting.
    A single connected client at a time is plenty — if the editor
    disconnects, the broadcaster drops the connection and waits for a
    new one.
    """

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._port = int(port)
        self._host = host
        self._seq = 0
        self._lock = threading.Lock()
        self._client: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Bind the listener and start the accept loop. Returns False on
        failure (port already in use, OS rejected). Caller can decide
        whether to log and continue without emulation."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._host, self._port))
            sock.listen(1)
            sock.settimeout(0.5)   # short accept timeout so stop() is responsive
        except OSError as e:
            print(f"[Emulator] Failed to bind {self._host}:{self._port}: {e}")
            return False
        self._listener = sock
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="EmulatorAccept", daemon=True
        )
        self._accept_thread.start()
        print(f"[Emulator] Listening on {self._host}:{self._port}")
        return True

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

    def _accept_loop(self):
        # Single-client model — replace any existing client when a new
        # one connects (editor reconnect after restart, etc.).
        while not self._stop.is_set() and self._listener is not None:
            try:
                conn, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return  # listener closed by stop()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[Emulator] Client connected from {addr}")
            with self._lock:
                if self._client is not None:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                self._client = conn

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self,
                raw_frames: Dict[str, np.ndarray],
                corrected_frames: Dict[str, np.ndarray]) -> None:
        """Send all groups' raw + corrected frames to the connected
        client (if any). Failures drop the client; the next accept will
        pick up a reconnection."""
        with self._lock:
            client = self._client
        if client is None:
            return
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        seq = self._seq

        try:
            for gid, frame in raw_frames.items():
                self._send_frame(client, STAGE_RAW, gid, frame, seq)
            for gid, frame in corrected_frames.items():
                self._send_frame(client, STAGE_CORRECTED, gid, frame, seq)
        except OSError as e:
            print(f"[Emulator] Client send failed ({e}); dropping connection")
            with self._lock:
                if self._client is client:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                    self._client = None

    def _send_frame(self, sock: socket.socket, stage: int, gid: str,
                    frame: np.ndarray, seq: int):
        if frame is None:
            return
        # Strip alpha if present, ensure uint8, ensure C-contiguous.
        if frame.ndim != 3:
            return
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        h, w = frame.shape[:2]
        gid_bytes = gid.encode("utf-8")[:255]
        header = struct.pack(_HDR_FMT, stage, len(gid_bytes), w, h, seq)
        body = gid_bytes + frame.tobytes()
        msg = header + body
        # Length-prefixed framing.
        sock.sendall(struct.pack(">L", len(msg)) + msg)


# ---------------------------------------------------------------------------
# Decoder helper — used by the editor side
# ---------------------------------------------------------------------------

def decode_message(payload: bytes) -> Optional[dict]:
    """Decode a length-prefixed payload (without the 4-byte length
    prefix) into ``{stage, group_id, width, height, seq, frame}``.
    Returns ``None`` if the buffer is malformed."""
    if len(payload) < _HDR_LEN:
        return None
    stage, gid_len, width, height, seq = struct.unpack(_HDR_FMT, payload[:_HDR_LEN])
    body_off = _HDR_LEN
    if len(payload) < body_off + gid_len + width * height * 3:
        return None
    gid = payload[body_off:body_off + gid_len].decode("utf-8", errors="replace")
    body_off += gid_len
    n = width * height * 3
    frame = np.frombuffer(payload[body_off:body_off + n], dtype=np.uint8)
    frame = frame.reshape((height, width, 3))
    return {
        "stage": stage,
        "group_id": gid,
        "width": width,
        "height": height,
        "seq": seq,
        "frame": frame,
    }
