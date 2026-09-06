"""Gate for the generative SUBSYSTEM + its frontend contract.

  1. GenSystem offline (no engine, hand-pumped): starts, keeps the composer
     ahead of the rack, status() carries what the page renders, every
     steering call lands (style, bpm, key, energy, density, swing, mute,
     hold, reseed, master, set_length), movements drift the key, END plays
     the outro and stops with a fade, outstate_keys() publishes truth.
  2. ACTIONS: lib/gen/actions.py rejects garbage, clamps numbers, and
     apply_gen_action drives the same system through the whitelist.
  3. WEB: the show's WebController serves /gen, gates /api/gen/active,
     queues POST /api/gen/action into request_gen_actions (the show's 5 Hz
     bridge drains it), and ships gen_info in the socket snapshot path.
  4. SUPERVISION: a composer failure is survived (reseed, keep playing).

Usage: python tools/tests/_gen_system_test.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.actions import (GEN_ACTIONS, apply_gen_action, idle_info,  # noqa: E402
                             sanitize_gen_action)
from lib.gen.system import GenSystem                      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def pump(g, seconds, out=None):
    for _ in range(int(seconds * RATE / 2048)):
        b = g.rack.read(2048)
        if b is None:
            return False
        if out is not None:
            out.append(b)
        g.step()
    return True


def main():
    logdir = tempfile.mkdtemp(prefix="gen_test_")
    print("== system offline")
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=4, set_length_s=90,
                  threaded=False, log_dir=logdir)
    check(g.start(), "start()")
    out = []
    pump(g, 12, out)
    s = g.status()
    for k in ("state", "style", "bpm", "key", "camelot", "section", "bar", "beat", "bar_phase",
              "energy", "chords", "chord_now", "layers", "slots", "muted", "arc_progress",
              "movement", "log", "notes", "peak", "lead_s", "styles"):
        check(k in s, f"status has {k}")
    check(s["lead_s"] >= 4.0, f"composer stays ahead ({s['lead_s']} s)")
    check(s["notes"] > 50 and 1 <= s["beat"] <= 4, f"playing: {s['notes']} notes, beat {s['beat']}")
    check(os.listdir(logdir) and os.path.getsize(os.path.join(logdir, os.listdir(logdir)[0])) > 100, "night log written")
    ok = g.outstate_keys()
    check(ok.get("gen_active") and "gen_beat_phase" in ok and "gen_energy" in ok, f"outstate keys {sorted(ok)[:4]}...")

    print("== steering")
    g.set_energy_bias(0.3); g.set_density(0.6); g.set_swing(0.2); g.set_master(0.5)
    g.set_mute("hat", True); g.set_hold(True)
    pump(g, 9, out)
    s = g.status()
    check(abs(s["energy_bias"] - 0.3) < 1e-9 and abs(s["density"] - 0.6) < 1e-9, "energy/density landed")
    check(abs(s["swing"] - 0.2) < 1e-9 and abs(s["master"] - 0.5) < 1e-9, "swing/master landed")
    check("hat" in s["muted"] and "hat" in g.rack.muted, "mute landed (composer + rack)")
    check(s["state"] == "hold", "hold shows in state")
    held = s["section"]
    pump(g, 40, out)
    check(g.status()["section"] == held, f"hold keeps section {held!r}")
    g.set_hold(False); g.set_style("downtempo"); g.set_bpm(96); g.set_key("9A"); g.reseed(77)
    pump(g, 12, out)
    s = g.status()
    check(s["style"] == "downtempo" and abs(s["bpm"] - 96) < 1e-9, f"style/bpm switched: {s['style']} {s['bpm']}")
    check(s["camelot"] == "9A", f"key switched: {s['key']} {s['camelot']}")
    check(s["seed"] == 77, "reseed landed")
    g.set_set_length(60)
    pump(g, 75, out)
    s = g.status()
    check(s["movement"] >= 1 and s["camelot"] != "9A", f"movement {s['movement']} drifted key to {s['camelot']}")

    print("== end of set")
    g.request_end()
    ran = pump(g, 240, out)
    check(not ran and g.rack.done and not g.active, "END: outro, fade, done, inactive")
    mix = np.concatenate(out)
    check(np.isfinite(mix).all() and float(np.abs(mix).max()) < 1.0, f"{len(mix) / RATE:.0f}s rendered clean")
    check(g.outstate_keys() == {"gen_active": False}, "outstate goes inactive")

    print("== actions")
    check(sanitize_gen_action({"action": "nope"}) is None, "unknown action rejected")
    check(sanitize_gen_action({"action": "style", "value": "polka"}) is None, "unknown style rejected")
    check(sanitize_gen_action({"action": "bpm", "value": "999"}) == ("bpm", 180.0), "bpm clamped")
    check(sanitize_gen_action({"action": "energy", "value": -3}) == ("energy", -0.5), "energy clamped")
    check(sanitize_gen_action({"action": "key", "value": "13A"}) is None, "bad key rejected")
    check(sanitize_gen_action({"action": "key", "value": "9a"}) == ("key", "9A"), "key normalised")
    check(sanitize_gen_action({"action": "mute", "value": {"slot": "drums", "on": 1}}) is None, "bad slot rejected")
    check(sanitize_gen_action({"action": "mute", "value": {"slot": "hat", "on": 1}}) == ("mute", {"slot": "hat", "on": True}), "mute ok")
    check(sanitize_gen_action({"action": "fluid", "value": "keys,bogus,pad"}) == ("fluid", "keys,pad"), "fluid slots filtered")
    cfg = {"style": "groove"}
    apply_gen_action(None, cfg, "style", "ambient")
    apply_gen_action(None, cfg, "mute", {"slot": "hat", "on": True})
    apply_gen_action(None, cfg, "bpm", 70.0)
    check(cfg["style"] == "ambient" and cfg["muted"] == "hat" and cfg["bpm"] == 70.0, "idle steering arms cfg")
    info = idle_info(cfg)
    check(info["style"] == "ambient" and "hat" in info["muted"] and info["styles"], "idle_info reflects armed cfg")
    started = []
    g2 = GenSystem(engine=None, style=cfg["style"], bpm=cfg["bpm"], key="8A", seed=1, threaded=False, log_dir=logdir)
    apply_gen_action(None, cfg, "start", None, start_fn=lambda: started.append(g2.start()))
    check(started == [True], "start action -> host start hook")
    apply_gen_action(g2, cfg, "density", 0.3)
    apply_gen_action(g2, cfg, "hold", True)
    pump(g2, 5)
    check(abs(g2.status()["density"] - 0.3) < 1e-9 and g2.status()["state"] == "hold", "live actions reach the system")
    apply_gen_action(g2, cfg, "stop", None, stop_fn=g2.stop)
    check(not g2.active, "stop action")

    print("== supervision")
    g3 = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=2, threaded=False, log_dir=logdir)
    g3.start()
    pump(g3, 3)
    orig = g3.composer.next_phrase
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("synthetic composer failure")
        return orig()
    g3.composer.next_phrase = boom
    pump(g3, 12)
    check(g3.active and g3.status()["lead_s"] > 2 and "synthetic" in g3.last_error, "composer failure survived (reseed + continue)")
    g3.stop()

    print("== web")
    try:
        from web.web_controller import WebController
        web = WebController(control_dict={}, port=0, service_name="test")
        client = web.app.test_client()
        r = client.get("/gen")
        check(r.status_code == 200 and b"Lucifera Gen" in r.data and b"/static/js/gen/index.js" in r.data, "/gen renders the page shell")
        r = client.get("/api/gen/active")
        check(r.status_code == 200 and r.get_json()["available"] is False, "/api/gen/active idle before the show reports")
        web.set("gen_info", idle_info({"style": "groove"}))
        r = client.get("/api/gen/active")
        check(r.get_json()["available"] is True and r.get_json()["active"] is False, "/api/gen/active reflects gen_info")
        r = client.post("/api/gen/action", json={"action": "bpm", "value": 130})
        r2 = client.post("/api/gen/action", json={"action": "bogus"})
        q = web.control_dict.get("request_gen_actions")
        check(r.status_code == 200 and r2.status_code == 400 and q == [("bpm", 130.0)], f"POST /api/gen/action queues sanitized actions: {q}")
        from lib.interaction import _REQUIRES_GATES
        check(_REQUIRES_GATES["gen"]({"gen_info": {"available": True}}) and not _REQUIRES_GATES["gen"]({}), "interaction 'gen' gate")
        src = open(os.path.join(os.path.dirname(__file__), "..", "..", "web", "web_controller.py")).read()
        check(src.count("\"gen\": self.control_dict.get('gen_info')") == 2, "gen_info ships in both socket snapshots")
    except Exception as e:  # noqa: BLE001
        check(False, f"web controller test errored: {type(e).__name__}: {e}")

    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
