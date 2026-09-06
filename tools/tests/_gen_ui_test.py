"""Gate for the /gen SURFACE (lib/gen/ui.py + web/static/js/gen/): the
spec-driven, extensible operator UI.

  1. SPEC: every card id unique; every widget type registered on the
     client (scanned from the JS modules); every action whitelisted; every
     status key referenced exists both idle and live.
  2. CLIENT: every JS widget module registers at least one type, index.js
     imports every widget module, the shell template loads the module entry
     and no longer carries an inline renderer.
  3. SCENES: save -> listing -> recall applies the steering (style, params,
     mutes, slot patterns) -> delete; persisted on disk.
  4. ROUTES: /api/gen/surface serves the spec; /gen serves the shell.
  5. EXTENSIBILITY: a spec that names an unregistered widget, an
     unknown action, or a missing key is REJECTED by the validator.

Usage: python tools/tests/_gen_ui_test.py
"""
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.actions import GEN_ACTIONS, idle_info, sanitize_gen_action   # noqa: E402
from lib.gen.system import GenSystem                      # noqa: E402
from lib.gen.ui import registered_widget_types, surface_spec, validate_surface   # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def pump(g, seconds):
    for _ in range(int(seconds * RATE / 2048)):
        if g.rack.read(2048) is None:
            return
        g.step()


def main():
    print("== spec")
    logdir = tempfile.mkdtemp()
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=4, threaded=False, log_dir=logdir)
    g.start(); pump(g, 3)
    live_keys = set(g.status())
    idle_keys = set(idle_info({"style": "groove", "log_dir": logdir}))
    spec = surface_spec()
    check(len(spec["cards"]) >= 8 and all(c.get("id") for c in spec["cards"]), f"{len(spec['cards'])} cards")
    probs = validate_surface(spec, live_keys | idle_keys)
    check(not probs, f"spec valid against live+idle status: {probs}")
    types = registered_widget_types()
    used = {w["type"] for c in spec["cards"] for w in c["widgets"]}
    check(used <= types, f"all {len(used)} used widget types registered ({sorted(used - types)} missing)")
    check(len(types - used) == 0, f"no orphan widget types ({sorted(types - used)})")
    # keys the spec reads must be present in BOTH modes for widgets on always-visible cards
    for c in spec["cards"]:
        if c.get("show_when") in (None, "always"):
            for w in c["widgets"]:
                for k in (w.get("key"), w.get("options_key"), w.get("items_key")):
                    if k:
                        check(k in idle_keys and k in live_keys, f"{c['id']}/{w['type']}: key {k!r} present idle and live")

    print("== client")
    wd = os.path.join(ROOT, "web", "static", "js", "gen", "widgets")
    mods = [f for f in os.listdir(wd) if f.endswith(".js")]
    index = open(os.path.join(ROOT, "web", "static", "js", "gen", "index.js"), encoding="utf-8").read()
    for m in mods:
        src = open(os.path.join(wd, m), encoding="utf-8").read()
        check(re.search(r"register\(\s*['\"]", src) is not None, f"{m} registers a widget")
        check(f"./widgets/{m}" in index, f"index.js imports {m}")
    shell = open(os.path.join(ROOT, "web", "templates", "gen_panel.html"), encoding="utf-8").read()
    check('type="module" src="/static/js/gen/index.js"' in shell and 'id="surface"' in shell, "shell loads the module entry")
    check("socket.on(" not in shell and "gen_action" not in shell, "shell carries no inline renderer")

    print("== scenes")
    g.set_density(0.4); g.set_brightness(1.3); g.set_mute("hat", True); pump(g, 9)
    g.scene_save("dark hats off")
    lst = g.status()["scenes"]
    check([s["name"] for s in lst] == ["dark hats off"] and lst[0]["style"] == "groove", f"saved + listed {lst}")
    g.set_density(1.2); g.set_brightness(0.8); g.set_mute("hat", False); pump(g, 9)
    done = g.scene_load("dark hats off"); pump(g, 9)
    s = g.status()
    check(abs(s["density"] - 0.4) < 1e-6 and abs(s["brightness"] - 1.3) < 1e-6 and "hat" in s["muted"], f"recall restored density/brightness/mutes: {done}")
    check(os.path.exists(os.path.join(logdir, "gen_scenes.json")), "scenes persisted")
    check(sanitize_gen_action({"action": "scene_save", "value": "x" * 50}) is None and sanitize_gen_action({"action": "scene_load", "value": " night "}) == ("scene_load", "night"), "scene actions sanitized")
    g.scene_delete("dark hats off")
    check(not g.status()["scenes"], "deleted")
    g.stop()

    print("== routes")
    try:
        from web.web_controller import WebController
        web = WebController(control_dict={}, port=0, service_name="test")
        client = web.app.test_client()
        r = client.get("/api/gen/surface")
        check(r.status_code == 200 and r.get_json()["version"] == spec["version"] and len(r.get_json()["cards"]) == len(spec["cards"]), "/api/gen/surface serves the spec")
        r = client.get("/gen")
        check(r.status_code == 200 and b"/static/js/gen/index.js" in r.data, "/gen serves the shell")
        r = client.get("/static/js/gen/index.js")
        check(r.status_code == 200 and b"widgets/basic.js" in r.data, "module entry served as a static file")
        r = client.get("/static/css/gen.css")
        check(r.status_code == 200 and b".surface" in r.data, "gen.css served")
    except Exception as e:  # noqa: BLE001
        check(False, f"web routes errored: {type(e).__name__}: {e}")

    print("== extensibility guard")
    bad = {"version": 1, "cards": [
        {"id": "x", "widgets": [{"type": "hologram"}, {"type": "slider", "key": "nope", "action": "launch"}]},
        {"id": "x", "widgets": []}]}
    probs = validate_surface(bad, live_keys)
    check(any("hologram" in p for p in probs) and any("launch" in p for p in probs) and any("nope" in p for p in probs) and any("duplicate" in p for p in probs),
          f"validator rejects unknown widget/action/key/duplicate id: {len(probs)} problems")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
