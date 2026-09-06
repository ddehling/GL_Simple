"""Gate for the DIRECTOR layer (lib/gen/director.py, lib/gen/feedback.py):
interaction above code.

  1. GESTURES: every entry in the vocabulary validates, and the musical
     ones land - darker moves brightness, strip-to-drums mutes pitched
     slots and holds, build requests the section, wind-down ramps over
     bars, modulate moves the key a fifth.
  2. RAMPS interpolate phrase by phrase and finish at the target.
  3. SLOT PATTERNS (needs a Strudel engine, else that part SKIPs): a
     pattern for one slot replaces only that slot; the rules keep the rest;
     bad code is refused before it reaches the rack.
  4. INTENT VALIDATION: unknown keys/slots/sections dropped with warnings,
     numbers clamped, patterns sandboxed.
  5. LANGUAGE DIRECTOR with an injected transport: JSON reply -> intent ->
     applied; junk reply -> error surfaced, nothing changed; runs on a
     worker thread and reports busy/last.
  6. TASTE: thumbs record snapshots; 'more like this' nudges toward liked
     ground; memory persists to disk and reloads.
  7. ACTIONS: gesture/ask/feedback/brightness/section sanitize and apply.

Usage: python tools/tests/_gen_director_test.py
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.actions import apply_gen_action, sanitize_gen_action   # noqa: E402
from lib.gen.director import (GESTURES, LLMDirector, apply_intent,  # noqa: E402
                              gesture_intent, validate_intent)
from lib.gen.feedback import PreferenceMemory             # noqa: E402
from lib.gen.system import GenSystem                      # noqa: E402

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


def fresh(seed=4, logdir=None):
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=seed, threaded=False,
                  log_dir=logdir or tempfile.mkdtemp())
    g.start()
    pump(g, 3)
    return g


def main():
    print("== gestures")
    for name in GESTURES:
        intent, warn = validate_intent(gesture_intent(name))
        check(intent and not warn, f"gesture {name} validates {warn or ''}")
    g = fresh()
    g.gesture("darker"); pump(g, 9)
    s = g.status()
    check(s["brightness"] < 1.0 and s["energy_bias"] < 0, f"darker: brightness {s['brightness']} energy_bias {s['energy_bias']}")
    ev = [e for p in list(g._phrases)[-1:] for e in p.events if e.slot == "bass"]
    check(ev and all(e.params.get("cutoff") and e.params["cutoff"] < 900 for e in ev), "brightness reaches the bass notes' cutoff")
    g.gesture("strip_drums"); pump(g, 9)
    s = g.status()
    check(set(s["muted"]) >= {"bass", "lead", "pad", "arp", "keys"} and s["state"] == "hold", "strip to drums mutes pitched slots and holds")
    g.gesture("open_up"); pump(g, 9)
    check(not g.status()["muted"], "open it up unmutes everything")
    g.gesture("build"); pump(g, 1)
    check(g.status()["section_requested"] == "build" and g.status()["section_bars_left"] <= 4, "build requests the section soon")
    pump(g, 10)
    check(g.status()["section_composed"] == "build", f"next composed phrase is {g.status()['section_composed']}")
    pump(g, 12)
    check(g.status()["section"] == "build", f"heard section became {g.status()['section']}")
    k0 = g.status()["camelot"]
    g.gesture("modulate_up"); pump(g, 9)
    check(g.status()["camelot"] != k0, f"modulate up: {k0} -> {g.status()['camelot']}")

    print("== ramps")
    g.add_ramp("density", 0.4, 16); pump(g, 1)
    vals = []
    for _ in range(6):
        pump(g, 8)
        vals.append(g.status()["density"])
    check(all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])) and abs(vals[-1] - 0.4) < 1e-6 and not g.status()["ramps"],
          f"density ramps down phrase by phrase to 0.4: {[round(v, 2) for v in vals]}")

    print("== slot patterns")
    from lib.gen.composer.strudel import available
    if available()[0]:
        g.set_slot_pattern("arp", 'note("0 2 4 7 9 7 4 2").scale("A4:minor").s("arp")'); pump(g, 12)
        p = list(g._phrases)[-1]
        slots = {e.slot for e in p.events}
        arps = [e for e in p.events if e.slot == "arp"]
        check("arp" in g.status()["pattern_slots"] and len(arps) == 32 and len(slots) > 3,
              f"arp from the pattern (32 notes/phrase), rules keep {sorted(slots - {'arp'})}")
        g.set_slot_pattern("lead", "nope(("); pump(g, 2)
        check(g.status()["error"].startswith("pattern[lead]") and "lead" not in g.status()["pattern_slots"], "bad slot pattern refused")
        g.clear_slot_pattern("arp"); pump(g, 9)
        check(not g.status()["pattern_slots"], "slot pattern cleared")
    else:
        print(f"  SKIP slot patterns: {available()[1]}")

    print("== intent validation")
    raw = {"say": "ok", "set": {"density": 9, "foo": 1, "key": "9a"}, "nudge": {"bpm": "x"}, "section": "chorus",
           "layers": {"mute": ["hat", "drums"]}, "patterns": {"tuba": "s(\"bd\")"}, "ramp": {"swing": {"to": 2, "bars": 8}},
           "hold": 1, "junk": True}
    intent, warn = validate_intent(raw)
    check(intent["set"]["density"] == 1.5 and intent["set"]["key"] == "9A" and "foo" not in intent["set"], "clamps and drops unknown params")
    check("section" not in intent and intent["layers"]["mute"] == ["hat"] and "patterns" not in intent, "unknown section/slot dropped")
    check(intent["ramp"]["swing"]["to"] == 0.33 and intent["hold"] is True and "junk" not in intent, "ramp clamped, bool coerced, junk dropped")
    check(len(warn) >= 4, f"{len(warn)} warnings: {warn}")
    if available()[0]:
        from lib.gen.composer.strudel import open_engine
        sb = open_engine()
        i2, w2 = validate_intent({"patterns": {"arp": 'note("0 4 7").scale("A4:minor").s("arp")', "lead": 's("bd*4")', "bass": "nope(("}}, sandbox=sb)
        check(list(i2.get("patterns", {})) == ["arp"] and len(w2) == 2, f"sandbox keeps only patterns that produce events for their slot: {w2}")
        sb.stop()

    print("== language director")
    replies = {"answer": json.dumps({"say": "darker, and a breakdown next", "nudge": {"brightness": -0.3}, "section": "break", "layers": {"mute": ["arp"]}})}
    def transport(system, prompt):
        check("Intent" in system and "Operator says" in prompt and '"style"' in prompt, "prompt carries state + schema")
        return "Sure!\n```json\n" + replies["answer"] + "\n```"
    b0 = g.status()["brightness"]
    check(g.ask("make it darker then break it down", transport=transport), "ask accepted")
    for _ in range(100):
        if not g._director_busy:
            break
        time.sleep(0.05)
    pump(g, 20)
    d = g.status()["director"]
    s = g.status()
    check(not d["busy"] and d["last"].get("say") == "darker, and a breakdown next", f"director replied: {d['last'].get('say')!r}")
    hist = [h[1] for h in g.composer.form.history[-3:]]
    check(s["brightness"] < b0 and "arp" in s["muted"] and "break" in hist, f"intent applied (brightness {s['brightness']}, muted {s['muted']}, sections {hist})")
    check(d["log"] and d["log"][-1]["kind"] == "ask", "director log records the exchange")
    replies["answer"] = "I don't know what you mean"
    g.ask("???", transport=transport)
    for _ in range(100):
        if not g._director_busy:
            break
        time.sleep(0.05)
    check("error" in g.status()["director"]["last"] and g.status()["director"]["last"]["error"], "junk reply -> error surfaced, system alive")
    check(g.active, "system still playing")
    dd = LLMDirector(transport=lambda s_, p_: "{}")
    check(dd.available and dd.mode == "injected", "injected transport mode")

    print("== taste")
    logdir = tempfile.mkdtemp()
    g2 = fresh(seed=9, logdir=logdir)
    g2.set_density(1.2); g2.set_brightness(1.4); pump(g2, 9)
    g2.feedback(True); g2.feedback(True)
    g2.set_density(0.3); g2.set_brightness(0.5); pump(g2, 9)
    g2.feedback(False)
    nud = g2.prefs.nudges("groove", {"density": 0.3, "brightness": 0.5, "energy": 0.5, "swing": 0.08})
    check(nud.get("density", 0) > 0 and nud.get("brightness", 0) > 0, f"nudges pull toward liked ground: {nud}")
    g2.gesture("more_like_this"); pump(g2, 9)
    check(g2.status()["density"] > 0.3 and g2.status()["brightness"] > 0.5, "'more like this' moved the parameters")
    pm = PreferenceMemory(os.path.join(logdir, "gen_prefs.json"))
    check(pm.counts() == {"up": 3, "down": 1}, f"memory persisted and reloaded {pm.counts()}")
    g2.stop()

    print("== actions")
    check(sanitize_gen_action({"action": "gesture", "value": "darker"}) == ("gesture", "darker"), "gesture action")
    check(sanitize_gen_action({"action": "gesture", "value": "explode"}) is None, "unknown gesture rejected")
    check(sanitize_gen_action({"action": "ask", "value": "  hi "}) == ("ask", "hi"), "ask trimmed")
    check(sanitize_gen_action({"action": "ask", "value": "x" * 3000}) is None, "ask too long rejected")
    check(sanitize_gen_action({"action": "brightness", "value": 5}) == ("brightness", 1.6), "brightness clamped")
    check(sanitize_gen_action({"action": "section", "value": "drop"}) == ("section", "drop") and sanitize_gen_action({"action": "section", "value": "chorus"}) is None, "section validated")
    cfg = {}
    apply_gen_action(g, cfg, "gesture", "sparser"); apply_gen_action(g, cfg, "brightness", 1.3); apply_gen_action(g, cfg, "feedback", False)
    pump(g, 9)
    check(abs(g.status()["brightness"] - 1.3) < 1e-6 and g.status()["taste"]["down"] >= 1, "actions reach the system")
    g.stop()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
