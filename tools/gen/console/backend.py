"""Where the console's state comes from and where its actions go.

  LocalBackend   GenSystem in THIS process: audio on this machine through
                 lib/audio_engine (miniaudio), or headless (the console
                 pumps the rack in wall-clock time - a dry run with no
                 sound device; also what the gate uses).
  RemoteBackend  the running show (or gen_server) over HTTP:
                 GET /api/gen/status, POST /api/gen/action - the same
                 whitelist the web page uses, so the two surfaces can never
                 disagree about what an action means.

Both expose status() -> gen_info dict and act(action, value)."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

from lib.gen import RATE
from lib.gen.actions import apply_gen_action, idle_info, sanitize_gen_action


class LocalBackend:
    def __init__(self, cfg=None, audio=True):
        self.cfg = dict(cfg or {})
        self.cfg.setdefault("log_dir", "logs")
        self.audio = audio
        self.engine = None
        self.gen = None
        self.error = ""
        self._pump_clock = None
        self._lock = threading.Lock()

    def _ensure_engine(self):
        if not self.audio or self.engine is not None:
            return
        try:
            from lib.audio_engine import AudioEngine
            self.engine = AudioEngine(sample_rate=RATE)
            self.engine.start()
        except Exception as e:  # noqa: BLE001
            self.error = f"audio engine unavailable ({e}); running headless"
            self.audio = False

    def start(self):
        if self.gen is not None and self.gen.active:
            return
        from lib.gen.system import GenSystem
        self._ensure_engine()
        c = self.cfg
        self.gen = GenSystem(engine=self.engine, style=c.get("style", "groove"), bpm=c.get("bpm"),
                             key=c.get("key", "8A"), seed=c.get("seed"), soundfont=c.get("soundfont"),
                             fluid_slots=c.get("fluid_slots", ""), set_length_s=float(c.get("set_length_s", 10800)),
                             energy_bias=float(c.get("energy_bias", 0.0)), density=float(c.get("density", 1.0)),
                             swing=c.get("swing"), master=float(c.get("master", 0.8)), muted=c.get("muted", ""),
                             log_dir=c.get("log_dir", "logs"), threaded=self.engine is not None)
        if not self.gen.start():
            self.error = self.gen.last_error
            self.gen = None
            return
        self.error = ""
        if c.get("pattern"):
            self.gen.set_pattern(c["pattern"])
        self._pump_clock = time.time()

    def stop(self):
        if self.gen is not None:
            self.gen.stop()
            self.gen = None

    def act(self, action, value=None):
        pair = sanitize_gen_action({"action": action, "value": value})
        if pair is None:
            return False
        with self._lock:
            apply_gen_action(self.gen, self.cfg, pair[0], pair[1], start_fn=self.start, stop_fn=self.stop)
        return True

    def pump(self, seconds=None):
        """Headless: advance the rack by wall-clock time (or `seconds`)."""
        if self.gen is None or self.engine is not None:
            return
        now = time.time()
        if seconds is None:
            seconds = min(2.0, now - (self._pump_clock or now))
        self._pump_clock = now
        n = int(seconds * RATE)
        while n > 0 and self.gen is not None:
            blk = self.gen.rack.read(min(2048, n))
            if blk is None:
                self.gen = None
                break
            n -= blk.shape[0]
            self.gen.step()

    def status(self):
        if self.gen is not None and not self.gen.active:
            self.gen = None
        if self.gen is None:
            info = idle_info(self.cfg, self.error)
        else:
            self.pump()
            info = self.gen.status()
        info["available"] = True
        info["backend"] = "local" + ("" if self.engine is not None else " (headless)")
        return info

    def audio_tap(self, n):
        """Last n stereo samples the rack rendered, or None when idle."""
        if self.gen is None or self.gen.rack is None:
            return None
        return self.gen.rack.recent(n)

    def close(self):
        self.stop()
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            pass


class RemoteBackend:
    def __init__(self, base_url, timeout=2.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.error = ""

    def status(self):
        try:
            with urllib.request.urlopen(self.base + "/api/gen/status", timeout=self.timeout) as r:
                info = json.loads(r.read().decode("utf-8")) or {}
            self.error = ""
            info["backend"] = "remote " + self.base
            return info
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            return {"available": False, "active": False, "error": f"show unreachable at {self.base}: {self.error}", "backend": "remote"}

    def act(self, action, value=None):
        data = json.dumps({"action": action, "value": value}).encode("utf-8")
        req = urllib.request.Request(self.base + "/api/gen/action", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status == 200
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            return False

    def start(self):
        self.act("start")

    def stop(self):
        self.act("stop")

    def audio_tap(self, n):
        return None          # the show does not stream audio to the console

    def close(self):
        pass
