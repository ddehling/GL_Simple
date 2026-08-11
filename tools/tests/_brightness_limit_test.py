#!/usr/bin/env python3
"""Offline check of the brightness-limit ceiling, end to end.

Each piece declares its own PSU budget in ``project.yaml``
(``brightness_limit:`` — Fan 0.1, Weight of Light 0.4). That number is a
hardware guard: below it the supplies are fine, above it they are
over-drawn. It has to reach the limiter without being quietly reduced,
and it must not be exceedable from any control surface.

    project.yaml -> web_controller.global_modifiers["brightness_limit"]
                 -> state["brightness_limit"]   (send_variables, every frame)
                 -> RenderPipeline._apply_brightness_limiting (clamped again)

Two ways it used to break, both silent:
  * the seeding call wrote to ``control_dict`` instead of
    ``global_modifiers``, so nothing read it — the web default (0.1)
    overwrote the project's ceiling on the first frame, and WoL ran at a
    quarter of its budget;
  * the slider's max was one hardcoded number for every rig, so it
    could not represent a project asking for more.

The fix must not trade one failure for a worse one: widening the slider
for every project would let a mis-drag over-draw a tighter rig. So the
slider's max is rewritten per project, and the render path clamps again
against the project value regardless of what any UI said.

Neither failure is visible in the preview (the limiter runs downstream of
it), so this gate exists to catch them by measurement instead.

Usage:
    python tools/tests/_brightness_limit_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(cond: bool, label: str) -> bool:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)
    return cond


def project_limits() -> dict:
    out = {}
    for path in sorted((ROOT / "projects").glob("*/project.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        bl = raw.get("brightness_limit")
        if bl is not None:
            out[path.parent.name] = float(bl)
    return out


def make_controller():
    """A real WebController. Constructing one builds the Flask app and the
    SocketIO wrapper but starts no server and binds no port, so the real
    schema and dicts are available offline."""
    from web.web_controller import WebController
    return WebController(port=5099)


def test_schema():
    print("\n[1] the slider's resting range, before any project loads")
    wc = make_controller()
    schema = wc.global_modifier_schema["brightness_limit"]
    limits = project_limits()
    print(f"      projects declare: "
          f"{', '.join(f'{k}={v}' for k, v in limits.items())}")

    check(schema["max"] <= 0.1,
          f"the built-in max ({schema['max']}) is the conservative fallback "
          f"for a project that declares no budget — never a permissive one")
    check(schema["min"] == 0.0, "the slider can always be taken to zero")


def test_ceiling_per_project():
    print("\n[2] the slider is re-scaled to the active project's budget")
    limits = project_limits()

    ok_max = ok_seed = ok_dict = True
    for pid, bl in limits.items():
        wc = make_controller()               # fresh: simulates booting that piece
        wc.set_project_ceiling("brightness_limit", bl)
        schema = wc.global_modifier_schema["brightness_limit"]
        # The guard: the slider cannot represent a value above the budget.
        ok_max &= (schema["max"] == bl and schema["default"] == bl)
        # The seed: this is the read send_variables performs every frame.
        ok_seed &= (wc.global_modifiers.get("brightness_limit") == bl)
        # ...and it must NOT land in control_dict, which nothing reads.
        ok_dict &= ("brightness_limit" not in wc.control_dict)
    check(ok_max, f"each project's slider max IS its declared budget "
                  f"({limits}) — a mis-drag stops at the safe value")
    check(ok_seed, "the piece boots at its full budget rather than the web "
                   "default")
    check(ok_dict, "the ceiling goes to global_modifiers, never to "
                   "control_dict where nothing reads it")

    # Swapping to a tighter rig must pull the slider down with it.
    wc = make_controller()
    wc.set_project_ceiling("brightness_limit", 0.4)      # WoL
    wc.set_project_ceiling("brightness_limit", 0.1)      # ...swap to Fan
    schema = wc.global_modifier_schema["brightness_limit"]
    check(schema["max"] == 0.1
          and wc.global_modifiers["brightness_limit"] == 0.1,
          "swapping from a generous rig to a tighter one lowers both the "
          "slider and its ceiling — no value left parked above the new budget")


def test_overshoot_is_refused():
    print("\n[3] nothing can push the setpoint past the budget")
    wc = make_controller()
    wc.set_project_ceiling("brightness_limit", 0.4)      # WoL

    for attempt in (0.5, 1.0, 99.0, float("inf")):
        wc.set_global_modifier("brightness_limit", attempt)
        if wc.global_modifiers["brightness_limit"] != 0.4:
            check(False, f"a write of {attempt} was clamped to the budget")
            break
    else:
        check(True, "writes of 0.5 / 1.0 / 99 / inf all clamp back to the "
                    "0.4 budget")

    wc.set_global_modifier("brightness_limit", 0.25)
    check(wc.global_modifiers["brightness_limit"] == 0.25,
          "taking the piece DOWN is always allowed")

    check(wc.set_global_modifier("no_such_slider", 1.0) is False,
          "an unknown modifier name is rejected rather than silently added")

    # The render path clamps again, so a value that reached state by some
    # other route than the slider still cannot raise the ceiling.
    pipe = (ROOT / "engine" / "render_pipeline.py").read_text(encoding="utf-8")
    clamped = ("setpoint = min(" in pipe
               and "float(self.brightness_setpoint)," in pipe)
    check(clamped, "the limiter clamps its setpoint against the project "
                   "ceiling independently of the web layer")


def test_wiring():
    print("\n[4] the wiring the failure hid behind")
    src = (ROOT / "Stories_OGL.py").read_text(encoding="utf-8")

    # send_variables must keep reading the slider into state every frame —
    # that read is what makes a stale seed fatal.
    check('state["brightness_limit"] = brightness_limit_mod' in src,
          "send_variables still copies the slider into state each frame")

    # Both the boot path and the swap path must re-scale the slider, and
    # neither may use set() (which writes control_dict).
    seeds = src.count('set_project_ceiling(\n                    "brightness_limit"')
    check(seeds == 2, f"both boot and project-swap re-scale the slider "
                      f"(found {seeds} of 2)")
    check('set("brightness_limit"' not in src,
          "nothing writes brightness_limit into control_dict any more")

    pipe = (ROOT / "engine" / "render_pipeline.py").read_text(encoding="utf-8")
    check('self.state.get("brightness_limit"' in pipe,
          "the limiter reads its setpoint from state (not a frozen copy)")


def main() -> int:
    print("=" * 70)
    print("BRIGHTNESS LIMIT: project ceiling -> slider -> limiter")
    print("=" * 70)
    test_schema()
    test_ceiling_per_project()
    test_overshoot_is_refused()
    test_wiring()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
