"""The generative subsystem's operator actions: ONE whitelist, ONE
sanitizer, ONE applier - shared by the show's web controller (socket +
HTTP twin), Stories_OGL's 5 Hz bridge, and the standalone gen_server, so
a control can never be accepted by one path and rejected by another.
Same discipline as the DJ's queue_dj_action / _apply_dj_controls."""
from __future__ import annotations

from lib.gen.composer.styles import STYLES
from lib.gen.events import SLOTS

GEN_ACTIONS = {
    "start", "stop", "end",            # transport (end = play the outro, then stop)
    "style", "bpm", "key",             # what
    "energy", "density", "swing",      # how much
    "hold", "reseed", "mute",          # form / identity / layers
    "master", "set_length", "fluid",   # level / arc / SoundFont slots
    "pattern", "pattern_clear",        # Strudel code (replaces the rule composer's notes)
    "gesture", "ask", "feedback",      # the director: vocabulary, language, taste
    "brightness", "section",           # timbre lever; next-section request
    "scene_save", "scene_load", "scene_delete",   # named steering snapshots
    "humanize", "lane", "automation",  # feel amount; a mix lane {lane,to,ramp_s}; form automation on/off
    "script",                          # follow a SongScript file (lib/gen/script.py) from the next phrase
}

_KEYS = [f"{n}{ab}" for n in range(1, 13) for ab in ("A", "B")]


def sanitize_gen_action(data):
    """dict -> (action, arg) or None. Clamps every numeric, rejects every
    unknown string. The browser sends ids/values only, never names."""
    data = data or {}
    action = data.get("action")
    if action not in GEN_ACTIONS:
        return None
    arg = data.get("value")
    try:
        if action == "style":
            if arg not in STYLES:
                return None
        elif action == "bpm":
            arg = max(50.0, min(180.0, float(arg)))
        elif action == "key":
            if not isinstance(arg, str) or arg.upper() not in _KEYS:
                return None
            arg = arg.upper()
        elif action == "energy":
            arg = max(-0.5, min(0.5, float(arg)))
        elif action in ("density", "master"):
            arg = max(0.0, min(1.5 if action == "density" else 1.0, float(arg)))
        elif action == "swing":
            arg = max(0.0, min(0.33, float(arg)))
        elif action == "hold":
            arg = bool(arg)
        elif action == "reseed":
            arg = None if arg in (None, "") else int(arg)
        elif action == "mute":
            # {"slot": "hat", "on": true}
            if not isinstance(arg, dict) or arg.get("slot") not in SLOTS:
                return None
            arg = {"slot": arg["slot"], "on": bool(arg.get("on"))}
        elif action == "set_length":
            arg = max(600.0, min(43200.0, float(arg)))       # 10 min .. 12 h
        elif action == "pattern":
            if not isinstance(arg, str) or not arg.strip() or len(arg) > 20000:
                return None
        elif action == "gesture":
            from lib.gen.director import GESTURES
            if arg not in GESTURES:
                return None
        elif action == "ask":
            if not isinstance(arg, str) or not arg.strip() or len(arg) > 2000:
                return None
            arg = arg.strip()
        elif action == "feedback":
            arg = bool(arg)
        elif action == "brightness":
            arg = max(0.4, min(1.6, float(arg)))
        elif action == "humanize":
            arg = max(0.0, min(1.5, float(arg)))
        elif action == "script":
            import os as _os
            if not isinstance(arg, str) or not arg.lower().endswith((".yaml", ".yml", ".json")) or not _os.path.exists(arg):
                return None
        elif action == "automation":
            arg = bool(arg)
        elif action == "lane":
            from lib.gen.synth.rack import LANES
            if not isinstance(arg, dict) or arg.get("lane") not in LANES:
                return None
            lo, hi = {"hp": (10.0, 4000.0), "lp": (200.0, 20000.0), "duck": (0.0, 0.9),
                      "verb": (0.0, 3.0), "delay_fb": (0.0, 0.85), "gain": (0.0, 2.0)}[arg["lane"]]
            arg = {"lane": arg["lane"], "to": max(lo, min(hi, float(arg.get("to", lo)))),
                   "ramp_s": max(0.0, min(60.0, float(arg.get("ramp_s", 0.0))))}
        elif action in ("scene_save", "scene_load", "scene_delete"):
            if not isinstance(arg, str) or not arg.strip() or len(arg) > 40:
                return None
            arg = arg.strip()
        elif action == "section":
            if arg not in ("intro", "groove", "build", "drop", "break", "outro", "flow", "swell", "calm"):
                return None
        elif action == "fluid":
            # comma list of slots to render with SoundFonts ("" = none)
            if not isinstance(arg, str) or len(arg) > 120:
                return None
            arg = ",".join(s for s in arg.split(",") if s in SLOTS)
        else:
            arg = None
    except (TypeError, ValueError):
        return None
    return action, arg


def apply_gen_action(system, cfg, action, arg, start_fn=None, stop_fn=None):
    """Apply one sanitized action. `system` may be None (idle): steering
    then lands in `cfg` and is applied at the next start (the DJ's
    'armed idle steering' contract). start_fn/stop_fn are the host's
    soundtrack-takeover hooks (Stories_OGL) or plain system.start/stop."""
    if action == "start":
        if start_fn:
            start_fn()
        elif system is not None:
            system.start()
        return
    if action == "stop":
        if stop_fn:
            stop_fn()
        elif system is not None:
            system.stop()
        return
    live = system is not None and system.active
    if action == "style":
        cfg["style"] = arg
        if live:
            system.set_style(arg)
    elif action == "bpm":
        cfg["bpm"] = arg
        if live:
            system.set_bpm(arg)
    elif action == "key":
        cfg["key"] = arg
        if live:
            system.set_key(arg)
    elif action == "energy":
        cfg["energy_bias"] = arg
        if live:
            system.set_energy_bias(arg)
    elif action == "density":
        cfg["density"] = arg
        if live:
            system.set_density(arg)
    elif action == "swing":
        cfg["swing"] = arg
        if live:
            system.set_swing(arg)
    elif action == "master":
        cfg["master"] = arg
        if live:
            system.set_master(arg)
    elif action == "set_length":
        cfg["set_length_s"] = arg
        if live:
            system.set_set_length(arg)
    elif action == "fluid":
        cfg["fluid_slots"] = arg          # takes effect at next start
    elif action == "hold":
        if live:
            system.set_hold(arg)
    elif action == "reseed":
        if live:
            system.reseed(arg)
    elif action == "mute":
        muted = set((cfg.get("muted") or "").split(",")) - {""}
        (muted.add if arg["on"] else muted.discard)(arg["slot"])
        cfg["muted"] = ",".join(sorted(muted))
        if live:
            system.set_mute(arg["slot"], arg["on"])
    elif action == "end":
        if live:
            system.request_end()
    elif action == "pattern":
        cfg["pattern"] = arg               # re-applied at the next start
        if live:
            system.set_pattern(arg)
    elif action == "pattern_clear":
        cfg.pop("pattern", None)
        if live:
            system.clear_pattern()
    elif action == "gesture":
        if live:
            system.gesture(arg)
    elif action == "ask":
        if live:
            system.ask(arg)
    elif action == "feedback":
        if live:
            system.feedback(arg)
    elif action == "humanize":
        if system is not None:
            system.set_humanize(arg)
        cfg["humanize"] = arg
    elif action == "script":
        if system is not None:
            system.load_script(arg)
        cfg["script"] = arg
    elif action == "automation":
        if system is not None:
            system.set_automation(arg)
        cfg["automation"] = arg
    elif action == "lane":
        if system is not None:
            system.set_lane(arg["lane"], arg["to"], arg["ramp_s"])
    elif action == "brightness":
        cfg["brightness"] = arg
        if live:
            system.set_brightness(arg)
    elif action == "section":
        if live:
            system.request_section(arg)
    elif action == "scene_save":
        if live:
            system.scene_save(arg)
    elif action == "scene_load":
        if live:
            system.scene_load(arg)
    elif action == "scene_delete":
        if live:
            system.scene_delete(arg)


def _scene_listing(cfg):
    try:
        from lib.gen.scenes import SceneStore
        import os
        return SceneStore(os.path.join(cfg.get("log_dir", "logs"), "gen_scenes.json")).listing()
    except Exception:
        return []


def idle_info(cfg, error=""):
    """Status published while the subsystem is available but not running,
    so the page offers the whole steering surface before START."""
    return {
        "available": True, "active": False, "state": "idle",
        "style": cfg.get("style", "groove"),
        "styles": [{"id": k, "label": v["label"], "bpm": list(v["bpm"])} for k, v in STYLES.items()],
        "bpm": cfg.get("bpm") or STYLES.get(cfg.get("style", "groove"), STYLES["groove"])["bpm"][0],
        "key": cfg.get("key", "8A"),
        "camelot": cfg.get("key", "8A"),      # same field live; the page's key select reads it
        "energy_bias": cfg.get("energy_bias", 0.0),
        "density": cfg.get("density", 1.0),
        "swing": cfg.get("swing"),
        "master": cfg.get("master", 0.8),
        "set_length_s": cfg.get("set_length_s", 3 * 3600.0),
        "muted": sorted(set((cfg.get("muted") or "").split(",")) - {""}),
        "fluid_slots": cfg.get("fluid_slots", ""),
        "pattern": cfg.get("pattern"),
        "brightness": cfg.get("brightness", 1.0),
        "scenes": _scene_listing(cfg),
        "gestures": [{"id": k, "label": v["label"]} for k, v in __import__("lib.gen.director", fromlist=["GESTURES"]).GESTURES.items()],
        "slots": list(STYLES.get(cfg.get("style", "groove"), STYLES["groove"])["slots"].keys()),
        "error": error,
    }
