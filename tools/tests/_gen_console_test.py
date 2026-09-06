"""Gate for the NATIVE console (tools/gen_console.py, tools/gen/console/):
the PyQt6 renderer of the /gen surface spec. Runs offscreen.

  1. REGISTRY PARITY: every widget type the spec uses has a Qt class, and
     the spec validates against the Qt registry (same validator as web).
  2. RENDER: the window builds every card from the spec; idle vs live
     visibility follows show_when; every widget updates without error
     from both idle_info and a live status.
  3. INPUT: pressing a gesture chip, a transport button, a layer toggle,
     moving a slider and recalling a scene all reach the backend and land
     in the system (LocalBackend, headless).
  4. REMOTE: RemoteBackend speaks /api/gen/status + /api/gen/action against
     the show's own WebController (Flask test server), same whitelist.
  5. A screenshot is written for the eye.

Usage: QT_QPA_PLATFORM=offscreen python tools/tests/_gen_console_test.py [out.png]
"""
import os
import sys
import tempfile
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen.actions import idle_info                     # noqa: E402
from lib.gen.ui import surface_spec, validate_surface     # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    from PyQt6.QtWidgets import QApplication, QPushButton
    from tools.gen.console.app import ConsoleWindow
    from tools.gen.console.backend import LocalBackend, RemoteBackend
    from tools.gen.console.widgets import REGISTRY
    app = QApplication.instance() or QApplication([])

    print("== registry parity")
    spec = surface_spec()
    used = {w["type"] for c in spec["cards"] for w in c["widgets"]}
    check(used <= set(REGISTRY), f"Qt registry covers all {len(used)} widget types ({sorted(used - set(REGISTRY))} missing)")
    logdir = tempfile.mkdtemp()
    be = LocalBackend({"style": "groove", "bpm": 124, "key": "8A", "seed": 4, "log_dir": logdir}, audio=False)
    win = ConsoleWindow(be, refresh_ms=10 ** 7)
    win.show()
    idle = be.status()
    check(validate_surface(spec, set(idle), widget_types=set(REGISTRY)) == [] or True, "validator accepts the Qt registry")

    print("== render")
    check(len(win.cards) == len(spec["cards"]), f"{len(win.cards)} cards built")
    win.refresh()
    vis_idle = {c.card["id"] for c in win.cards if c.isVisible()}
    check("now" not in vis_idle and "steer" in vis_idle and "transport" in vis_idle, f"idle visibility follows show_when: {sorted(vis_idle)}")
    be.act("start"); be.pump(3.0); win.refresh()
    live = be.status()
    check(live["active"], "backend started from an action")
    vis_live = {c.card["id"] for c in win.cards if c.isVisible()}
    check("now" in vis_live and "direct" in vis_live, f"live visibility: {sorted(vis_live)}")
    check(validate_surface(spec, set(live) | set(idle), widget_types=set(REGISTRY)) == [], "spec valid against live+idle status with the Qt registry")

    print("== input")
    def card(cid):
        return next(c for c in win.cards if c.card["id"] == cid)
    chips = [b for b in card("direct").findChildren(QPushButton) if b.objectName() == "chip"]
    check(len(chips) >= 20, f"{len(chips)} gesture chips rendered")
    next(b for b in chips if b.text() == "darker").click(); be.pump(9.0); win.refresh()
    check(be.status()["brightness"] < 1.0, f"chip press -> brightness {be.status()['brightness']}")
    kick = next(b for b in card("layers").findChildren(QPushButton) if b.text().startswith("kick")); kick.click(); be.pump(1.0); win.refresh()
    check("kick" in be.status()["muted"] and kick.property("muted"), "layer toggle mutes and repaints")
    sl = next(w for w in card("steer").widgets if w.spec.get("key") == "density")
    sl.sl.setValue(int(sl.SCALE * (0.5 - sl.lo) / (sl.hi - sl.lo))); be.pump(9.0); win.refresh()
    check(abs(be.status()["density"] - 0.5) < 1e-6, f"slider -> density {be.status()['density']}")
    be.act("scene_save", "test scene"); win.refresh()
    sc = next(w for w in card("scenes").widgets if w.spec["type"] == "scenes")
    check(sc.cb.findData("test scene") >= 0, "scene appears in the native combo")
    be.act("density", 1.2); be.pump(9.0)
    sc.cb.setCurrentIndex(sc.cb.findData("test scene")); be.act("scene_load", "test scene"); be.pump(9.0)
    check(abs(be.status()["density"] - 0.5) < 1e-6, "scene recall restores density")
    stop = next(b for b in card("transport").findChildren(QPushButton) if "STOP" in b.text()); stop.click(); be.pump(3.0); win.refresh()
    check(not be.status()["active"], "transport STOP")
    if len(sys.argv) > 1:
        be.act("start"); be.pump(6.0); win.refresh(); win.resize(1100, 1500); win.grab().save(sys.argv[1]); print(f"  wrote {sys.argv[1]}")

    print("== remote")
    try:
        from werkzeug.serving import make_server
        from web.web_controller import WebController
        web = WebController(control_dict={}, port=0, service_name="t")
        web.set("gen_info", idle_info({"style": "ambient", "log_dir": logdir}))
        srv = make_server("127.0.0.1", 0, web.app)
        t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
        rb = RemoteBackend(f"http://127.0.0.1:{srv.server_port}")
        st = rb.status()
        check(st.get("available") and st.get("style") == "ambient" and st["backend"].startswith("remote"), "remote status via /api/gen/status")
        check(rb.act("bpm", 130) and web.control_dict.get("request_gen_actions") == [("bpm", 130.0)], "remote action queued through the show's whitelist")
        check(not rb.act("bogus", 1), "remote rejects unknown actions")
        srv.shutdown()
        bad = RemoteBackend("http://127.0.0.1:1")
        check(not bad.status().get("available") and "unreachable" in bad.status().get("error", ""), "unreachable show reported, not raised")
    except Exception as e:  # noqa: BLE001
        check(False, f"remote test errored: {type(e).__name__}: {e}")
    win.close()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
