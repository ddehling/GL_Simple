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


def idle_info(cfg, error=""):
    """Status published while the subsystem is available but not running,
    so the page offers the whole steering surface before START."""
    return {
        "available": True, "active": False, "state": "idle",
        "style": cfg.get("style", "groove"),
        "styles": [{"id": k, "label": v["label"], "bpm": list(v["bpm"])} for k, v in STYLES.items()],
        "bpm": cfg.get("bpm") or STYLES.get(cfg.get("style", "groove"), STYLES["groove"])["bpm"][0],
        "key": cfg.get("key", "8A"),
        "energy_bias": cfg.get("energy_bias", 0.0),
        "density": cfg.get("density", 1.0),
        "swing": cfg.get("swing"),
        "master": cfg.get("master", 0.8),
        "set_length_s": cfg.get("set_length_s", 3 * 3600.0),
        "muted": sorted(set((cfg.get("muted") or "").split(",")) - {""}),
        "fluid_slots": cfg.get("fluid_slots", ""),
        "slots": list(STYLES.get(cfg.get("style", "groove"), STYLES["groove"])["slots"].keys()),
        "error": error,
    }
