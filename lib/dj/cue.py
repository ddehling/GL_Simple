"""THE CUE OUTPUT: a second audio device for auditioning in headphones (docs/DJ_MORE_THAN_A_MIXER.md 1).

Named by `dj.cue_device` in config.yaml (a fragment of the playback device's name, case-insensitive), or the
environment variable DJ_CUE_DEVICE. Without one nothing here plays: the floor is never used for a preview
while the Director runs. Never opened by a test.
"""
import os
import threading

import numpy as np


def cue_device_name():
    name = os.environ.get("DJ_CUE_DEVICE", "")
    if name:
        return name
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(((cfg.get("dj") or {}).get("cue_device")) or "")
    except Exception:
        return ""


def playback_devices():
    """[(name, id)] of the machine's playback devices, or [] when miniaudio cannot enumerate them."""
    try:
        import miniaudio
        return [(d["name"], d["id"]) for d in miniaudio.Devices().get_playbacks()]
    except Exception:
        return []


def find_playback(fragment):
    frag = (fragment or "").strip().lower()
    if not frag:
        return None
    for name, dev_id in playback_devices():
        if frag in name.lower():
            return name, dev_id
    return None


class CuePlayer:
    """Plays one buffer at a time on the cue device; a new play stops the last. `available` says whether the
    named device was found; `why` says what is missing."""

    def __init__(self, name=None):
        self.name = name if name is not None else cue_device_name()
        self._found = find_playback(self.name)
        self._dev = None
        self._lock = threading.Lock()
        self.playing = False

    @property
    def available(self):
        return self._found is not None

    @property
    def why(self):
        if not self.name:
            return "no cue output: set dj.cue_device in config.yaml to a second playback device (e.g. your headphones)"
        if self._found is None:
            names = ", ".join(n for n, _ in playback_devices()[:6])
            return f"no playback device matches '{self.name}' (found: {names})"
        return ""

    def play(self, samples, rate):
        """`samples`: (n, 2) float32 in -1..1. Returns (ok, words)."""
        if self._found is None:
            return False, self.why
        import miniaudio
        buf = np.ascontiguousarray(np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0))
        if buf.ndim == 1:
            buf = np.stack([buf, buf], axis=1)
        self.stop()
        with self._lock:
            try:
                dev = miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2,
                                               sample_rate=int(rate), buffersize_msec=100, device_id=self._found[1])
            except Exception as e:  # noqa: BLE001
                return False, f"the cue device would not open: {e}"

            def feed():
                pos = 0
                required = yield b""
                while pos < len(buf):
                    n = int(required)
                    chunk = buf[pos:pos + n]
                    pos += n
                    if len(chunk) < n:
                        chunk = np.concatenate([chunk, np.zeros((n - len(chunk), 2), dtype=np.float32)])
                    required = yield chunk.tobytes()
                self.playing = False
                # (the device is closed by stop() or the next play; miniaudio pads silence after the generator ends)
                while True:
                    required = yield np.zeros((int(required), 2), dtype=np.float32).tobytes()
            gen = feed()
            next(gen)
            dev.start(gen)
            self._dev = dev
            self.playing = True
        return True, f"on {self._found[0]}"

    def stop(self):
        with self._lock:
            dev, self._dev = self._dev, None
            self.playing = False
        if dev is not None:
            try:
                dev.stop()
                dev.close()
            except Exception:
                pass
