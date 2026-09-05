"""Standalone generative-music server: the /gen control page WITHOUT the
show app. Runs GenSystem on a real AudioEngine (this machine's speakers)
or headless into a WAV, and serves the show's own web controller so the
page, the socket events and the HTTP action twin are the exact code the
show uses (web/web_controller.py, lib/gen/actions.py).

    python tools/gen/gen_server.py                       # speakers + http://localhost:5000/gen
    python tools/gen/gen_server.py --port 8080 --style ambient --fluid-slots keys,pad
    python tools/gen/gen_server.py --wav night.wav --minutes 20   # headless: no device, real page

Every control on the page is applied through lib/gen/actions.py at the
same 5 Hz bridge cadence Stories_OGL uses.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.actions import apply_gen_action, idle_info   # noqa: E402
from lib.gen.system import GenSystem                      # noqa: E402


class _Host:
    """The bridge Stories_OGL provides in the show, reduced to what the
    generator needs: an engine, a cfg dict, start/stop with takeover."""

    def __init__(self, engine, cfg, autostart):
        self.engine = engine
        self.cfg = cfg
        self.gen = None
        self.error = ""
        self._autostart = autostart

    def start(self):
        if self.gen is not None and self.gen.active:
            return
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
        else:
            self.error = ""

    def stop(self):
        if self.gen is not None:
            self.gen.stop()
            self.gen = None

    def tick(self, web):
        with web._dict_lock:
            actions = web.control_dict.pop("request_gen_actions", [])
        for action, arg in actions:
            try:
                apply_gen_action(self.gen, self.cfg, action, arg, start_fn=self.start, stop_fn=self.stop)
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] action {action} failed: {e}")
        if self.gen is not None and not self.gen.active:
            self.gen = None
        info = self.gen.status() if self.gen is not None else idle_info(self.cfg, self.error)
        info["available"] = True
        web.set("gen_info", info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--style", default="groove")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--key", default="8A")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fluid-slots", default="")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--set-length", type=float, default=10800.0, help="arc length, seconds")
    ap.add_argument("--wav", help="headless: render to this WAV instead of playing")
    ap.add_argument("--minutes", type=float, default=10.0, help="headless render length")
    ap.add_argument("--autostart", action="store_true", help="start playing immediately")
    args = ap.parse_args()

    cfg = {"style": args.style, "bpm": args.bpm, "key": args.key, "seed": args.seed,
           "fluid_slots": args.fluid_slots, "soundfont": args.soundfont, "set_length_s": args.set_length,
           "master": 0.8, "log_dir": "logs"}
    from web.web_controller import WebController
    web = WebController(control_dict={}, port=args.port, service_name="lucifera-gen")
    web.start(threaded=True)
    print(f"[GEN] control page: http://localhost:{args.port}/gen")

    if args.wav:
        # Headless: hand-pump the generator like the offline DJ renders; the
        # page still drives it live while it renders.
        host = _Host(None, cfg, True)
        host.start()
        if host.gen is None:
            print(f"[GEN] start failed: {host.error}")
            return 1
        total = int(args.minutes * 60 * RATE)
        out, n = [], 0
        last_tick = 0.0
        while n < total and host.gen is not None:
            b = host.gen.rack.read(2048)
            if b is None:
                break
            out.append(b); n += b.shape[0]
            host.gen.step()
            if time.time() - last_tick > 0.2:
                host.tick(web); last_tick = time.time()
            time.sleep(0.0)               # yield so the socket thread emits
        mix = np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)
        import soundfile as sf
        sf.write(args.wav, np.clip(mix, -1, 1), RATE, subtype="PCM_16")
        print(f"wrote {args.wav} ({len(mix) / RATE:.0f}s)")
        host.stop()
        return 0

    from lib.audio_engine import AudioEngine
    engine = AudioEngine(sample_rate=RATE)
    engine.start()
    host = _Host(engine, cfg, args.autostart)
    if args.autostart:
        host.start()
    try:
        while True:
            host.tick(web)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        host.stop()
        try:
            engine.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
