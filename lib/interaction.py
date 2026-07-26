"""Per-weather-set interaction panels.

The web UI carries ONE generic "interaction" nav slot whose content is
owned by whichever weather set is currently live. A set that declares no
panel gets no tab at all — the slot simply disappears, which is the
default for every set.

Two kinds of panel exist:

``page`` panels
    The set points the tab at a bespoke, hand-written page (the Club /
    DJ set does this: ``{"label": "DJ", "page": "/dj"}``). Nothing here
    renders it; the tab just links there.

``sections`` panels
    The set describes its controls declaratively and
    ``web/templates/interaction_panel.html`` renders them, themed with
    the set's own colors. This is the path for "give this set a few
    buttons that fire events".

Where panels live
-----------------
Panels are project content, not engine content, so they live in the
project repo and are wired through the ``hooks:`` block of
``projects/<id>/project.yaml``::

    hooks:
      interaction: projects.fan.interaction

That module exposes ``INTERACTION_PANELS = {set_id: panel_spec}``.
See ``docs/INTERACTION_PANELS.md`` for the full schema.

Trust model
-----------
The browser never names an event / param / state. It sends a control
*id* and (for sliders and selects) a value; the server looks the control
up in the ACTIVE set's normalized panel and derives the real action from
the spec. A stale or hostile client can therefore only fire actions the
active set already published, with values clamped to the declared range.
"""
from __future__ import annotations

import time


# Control types the generic renderer knows how to draw.
CONTROL_TYPES = ("button", "slider", "toggle", "select")

# Action kinds a control may bind to.
#   event  -> schedule a named event from the project's event map
#   state  -> request a weather-state transition inside the current set
#   param  -> write (or release) a web parameter override
#   signal -> publish a named value into outstate['interaction'] for shaders
ACTION_KINDS = ("event", "state", "param", "signal")

# ``requires`` gates: a panel is only offered when its gate passes.
# Keyed by the string a project writes in the spec.
_REQUIRES_GATES = {
    "dj": lambda cd: bool((cd.get("dj_info") or {}).get("available")),
}

_DEFAULT_THEME = {
    "accent": "#5ac8fa",
    "bg": "#101014",
    "panel": "#1a1a20",
    "text": "#e8e8ee",
}


def _warn(set_id, msg):
    print(f"[Interaction] {set_id}: {msg}")


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_control(set_id, raw, cid):
    """Return a normalized control dict, or None if the spec is unusable.

    Bad controls are dropped with a warning rather than raising — a typo
    in one button must not cost the operator the whole panel mid-show.
    """
    if not isinstance(raw, dict):
        _warn(set_id, f"control {cid} is not a dict; skipped")
        return None

    ctype = str(raw.get("type", "button")).lower()
    if ctype not in CONTROL_TYPES:
        _warn(set_id, f"control {cid}: unknown type {ctype!r}; skipped")
        return None

    action = str(raw.get("action", "event")).lower()
    if action not in ACTION_KINDS:
        _warn(set_id, f"control {cid}: unknown action {action!r}; skipped")
        return None

    ctrl = {
        "id": cid,
        "type": ctype,
        "action": action,
        "label": str(raw.get("label", cid)),
        "icon": str(raw.get("icon", "")) or None,
        "hint": str(raw.get("hint", "")) or None,
        "color": str(raw.get("color", "")) or None,
    }

    if action == "event":
        event = raw.get("event")
        if not isinstance(event, str) or not event:
            _warn(set_id, f"control {cid}: action 'event' needs an 'event' name; skipped")
            return None
        ctrl["event"] = event
        ctrl["duration"] = _as_float(raw.get("duration"), None)
        ctrl["frame_id"] = int(raw.get("frame_id", 0) or 0)
        # A per-control cooldown keeps a mashed button from stacking
        # twenty copies of the same effect.
        ctrl["cooldown"] = max(0.0, _as_float(raw.get("cooldown"), 0.0) or 0.0)

    elif action == "state":
        if ctype == "select":
            options = _normalize_options(set_id, raw, cid)
            if not options:
                return None
            ctrl["options"] = options
        else:
            state = raw.get("state")
            if not isinstance(state, str) or not state:
                _warn(set_id, f"control {cid}: action 'state' needs a 'state' name; skipped")
                return None
            ctrl["state"] = state

    elif action == "param":
        param = raw.get("param")
        if not isinstance(param, str) or not param:
            _warn(set_id, f"control {cid}: action 'param' needs a 'param' name; skipped")
            return None
        ctrl["param"] = param
        if ctype == "slider":
            ctrl["min"] = _as_float(raw.get("min"), 0.0)
            ctrl["max"] = _as_float(raw.get("max"), 1.0)
            ctrl["step"] = _as_float(raw.get("step"), 0.01)
            ctrl["default"] = _as_float(raw.get("default"), ctrl["min"])
        elif ctype == "toggle":
            ctrl["on"] = _as_float(raw.get("on"), 1.0)
            ctrl["off"] = _as_float(raw.get("off"), 0.0)
        elif ctype == "select":
            options = _normalize_options(set_id, raw, cid, numeric=True)
            if not options:
                return None
            ctrl["options"] = options
        else:  # button
            # value None means "release the override and hand the
            # parameter back to the weather system".
            ctrl["value"] = _as_float(raw.get("value"), None)
            ctrl["release"] = "value" not in raw or raw.get("value") is None

    elif action == "signal":
        key = raw.get("signal")
        if not isinstance(key, str) or not key:
            _warn(set_id, f"control {cid}: action 'signal' needs a 'signal' key; skipped")
            return None
        ctrl["signal"] = key
        if ctype == "slider":
            ctrl["min"] = _as_float(raw.get("min"), 0.0)
            ctrl["max"] = _as_float(raw.get("max"), 1.0)
            ctrl["step"] = _as_float(raw.get("step"), 0.01)
            ctrl["default"] = _as_float(raw.get("default"), ctrl["min"])
        elif ctype == "toggle":
            ctrl["on"] = _as_float(raw.get("on"), 1.0)
            ctrl["off"] = _as_float(raw.get("off"), 0.0)
        elif ctype == "select":
            options = _normalize_options(set_id, raw, cid, numeric=True)
            if not options:
                return None
            ctrl["options"] = options
        else:  # button — a momentary press shaders can detect by timestamp
            ctrl["value"] = _as_float(raw.get("value"), 1.0)
            ctrl["cooldown"] = max(0.0, _as_float(raw.get("cooldown"), 0.0) or 0.0)

    return ctrl


def _normalize_options(set_id, raw, cid, numeric=False):
    """Normalize a select control's ``options`` into [{label, value}]."""
    options = []
    for opt in raw.get("options") or []:
        if isinstance(opt, dict):
            value = opt.get("value")
            label = str(opt.get("label", value))
        else:
            value = opt
            label = str(opt)
        if numeric:
            value = _as_float(value, None)
            if value is None:
                continue
        else:
            if not isinstance(value, str) or not value:
                continue
        options.append({"label": label, "value": value})
    if not options:
        _warn(set_id, f"control {cid}: select has no usable options; skipped")
        return None
    return options


def normalize_panel(set_id, raw):
    """Validate + normalize one panel spec. Returns None if unusable."""
    if not isinstance(raw, dict):
        _warn(set_id, "panel spec is not a dict; ignored")
        return None

    page = raw.get("page")
    theme = dict(_DEFAULT_THEME)
    for key, value in (raw.get("theme") or {}).items():
        if key in _DEFAULT_THEME and isinstance(value, str):
            theme[key] = value

    panel = {
        "set": set_id,
        "label": str(raw.get("label", set_id.replace("_", " ").title())),
        "title": str(raw.get("title", raw.get("label", set_id))),
        "blurb": str(raw.get("blurb", "")) or None,
        "requires": raw.get("requires") if isinstance(raw.get("requires"), str) else None,
        "theme": theme,
        "page": str(page) if isinstance(page, str) and page else None,
        "sections": [],
    }

    if panel["requires"] and panel["requires"] not in _REQUIRES_GATES:
        _warn(set_id, f"unknown requires gate {panel['requires']!r}; panel will always show")
        panel["requires"] = None

    if panel["page"]:
        # Bespoke page — no controls to normalize.
        return panel

    index = {}
    counter = 0
    for s_idx, raw_section in enumerate(raw.get("sections") or []):
        if not isinstance(raw_section, dict):
            continue
        controls = []
        for raw_ctrl in raw_section.get("controls") or []:
            cid = f"c{counter}"
            counter += 1
            ctrl = _normalize_control(set_id, raw_ctrl, cid)
            if ctrl is None:
                continue
            controls.append(ctrl)
            index[cid] = ctrl
        if not controls:
            continue
        panel["sections"].append({
            "title": str(raw_section.get("title", "")) or None,
            "note": str(raw_section.get("note", "")) or None,
            "layout": str(raw_section.get("layout", "grid")).lower(),
            "controls": controls,
        })

    if not panel["sections"]:
        _warn(set_id, "panel has no usable controls and no 'page'; ignored")
        return None

    panel["_index"] = index
    return panel


def normalize_panels(raw_panels):
    """Normalize a project's ``INTERACTION_PANELS`` mapping."""
    out = {}
    if not isinstance(raw_panels, dict):
        return out
    for set_id, raw in raw_panels.items():
        panel = normalize_panel(str(set_id), raw)
        if panel is not None:
            out[str(set_id)] = panel
    return out


def load_project_panels(project):
    """Return normalized panels for a project, via its ``interaction`` hook.

    Never raises: a broken panel module costs the interaction tab, not
    the show.
    """
    try:
        hook = project.load_hook("interaction")
    except Exception as e:  # pragma: no cover — load_hook already guards
        print(f"[Interaction] hook load failed: {e}")
        return {}
    if hook is None:
        return {}
    try:
        return normalize_panels(getattr(hook, "INTERACTION_PANELS", None) or {})
    except Exception as e:
        print(f"[Interaction] panel normalization failed: {e}")
        return {}


def panel_for_set(panels, set_id, control_dict):
    """Return the panel for ``set_id`` if it exists and its gate passes."""
    panel = (panels or {}).get(set_id)
    if panel is None:
        return None
    gate = _REQUIRES_GATES.get(panel.get("requires") or "")
    if gate is not None:
        try:
            if not gate(control_dict or {}):
                return None
        except Exception:
            return None
    return panel


def public_panel(panel):
    """Strip internals so the panel can be handed to the browser."""
    return {k: v for k, v in panel.items() if not k.startswith("_")}


def resolve_action(panel, control_id, value):
    """Turn a client (control_id, value) pair into an executable action.

    Returns a dict the render thread can act on, or None when the
    control doesn't exist in this panel (stale tab, wrong set) or the
    value is unusable. The event / state / param names always come from
    the SPEC, never from the client.
    """
    ctrl = (panel.get("_index") or {}).get(control_id)
    if ctrl is None:
        return None

    action = ctrl["action"]
    ctype = ctrl["type"]

    if action == "event":
        return {
            "kind": "event",
            "event": ctrl["event"],
            "duration": ctrl.get("duration"),
            "frame_id": ctrl.get("frame_id", 0),
            "cooldown": ctrl.get("cooldown", 0.0),
            "control": control_id,
        }

    if action == "state":
        if ctype == "select":
            state = _pick_option(ctrl, value, numeric=False)
            if state is None:
                return None
        else:
            state = ctrl["state"]
        return {"kind": "state", "state": state, "control": control_id}

    if action == "param":
        resolved = _resolve_value(ctrl, value)
        if resolved is _INVALID:
            return None
        return {"kind": "param", "param": ctrl["param"],
                "value": resolved, "control": control_id}

    if action == "signal":
        resolved = _resolve_value(ctrl, value)
        if resolved is _INVALID or resolved is None:
            return None
        return {"kind": "signal", "key": ctrl["signal"], "value": resolved,
                "cooldown": ctrl.get("cooldown", 0.0), "control": control_id}

    return None


class _Invalid:
    pass


_INVALID = _Invalid()


def _pick_option(ctrl, value, numeric):
    """Return the option value matching ``value``, or None."""
    if numeric:
        value = _as_float(value, None)
        if value is None:
            return None
        for opt in ctrl["options"]:
            if abs(opt["value"] - value) < 1e-9:
                return opt["value"]
        return None
    value = value if isinstance(value, str) else None
    for opt in ctrl["options"]:
        if opt["value"] == value:
            return opt["value"]
    return None


def _resolve_value(ctrl, value):
    """Clamp / validate an incoming value against a control's declared range."""
    ctype = ctrl["type"]
    if ctype == "slider":
        v = _as_float(value, None)
        if v is None:
            return _INVALID
        lo, hi = ctrl["min"], ctrl["max"]
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(hi, v))
    if ctype == "toggle":
        return ctrl["on"] if bool(value) else ctrl["off"]
    if ctype == "select":
        picked = _pick_option(ctrl, value, numeric=True)
        return _INVALID if picked is None else picked
    # button
    if ctrl["action"] == "param" and ctrl.get("release"):
        return None  # release the override
    return ctrl.get("value", 1.0)


def make_signal(value, previous=None):
    """Build the outstate record a signal control publishes.

    Shaders read ``outstate['interaction'][key]`` and get
    ``{"value": float, "t": wall-clock seconds, "count": presses}``.
    A momentary press is detected with ``now - rec['t'] < window``;
    a slider/toggle is read straight off ``rec['value']``.
    """
    count = int((previous or {}).get("count", 0)) + 1
    return {"value": float(value), "t": time.time(), "count": count}
