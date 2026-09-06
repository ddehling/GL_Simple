"""The /gen SURFACE: a declarative description of the operator UI.

The page does not know what the generative system is; it renders this
spec with a small widget registry (web/static/js/gen/). Every widget
reads its live value from the `gen_info` status dict by `key` and sends
operator input through an `action` from lib/gen/actions.py. So:

  * add a control   -> one entry here (and, if new, one action)
  * add a widget    -> one JS module that registers its type
  * add a card      -> one dict here
  * another surface -> another spec (a native console can render the same
                       spec; docs/GENERATIVE_UI.md)

validate_surface() is the gate: every action must be whitelisted, every
key must exist in status, every widget type must be registered.
"""
from __future__ import annotations

import os
import re

from lib.gen.actions import GEN_ACTIONS

SURFACE_VERSION = 2

# show_when: "always" | "live" | "idle"
SURFACE = {
    "version": SURFACE_VERSION,
    "title": "Lucifera Gen",
    "tab": "Gen",
    "cards": [
        {"id": "banner", "kind": "banner", "widgets": [{"type": "banner"}]},
        {"id": "transport", "kind": "transport", "sticky": True, "widgets": [
            {"type": "buttons", "items": [
                {"label": "▶ START", "action": "start", "style": "go", "show_when": "idle"},
                {"label": "■ STOP", "action": "stop", "style": "stop", "show_when": "live"},
                {"label": "⏏ END SET", "action": "end", "style": "alt", "show_when": "live", "confirm": "Play the outro and stop?"},
                {"label": "HOLD", "action": "hold", "toggle_key": "state", "toggle_value": "hold", "show_when": "live"},
                {"label": "🎲 NEW IDEAS", "action": "reseed", "show_when": "live"},
            ]},
        ]},
        {"id": "now", "title": "Now", "col": 1, "show_when": "live", "widgets": [
            {"type": "headline", "key": "section", "sub_keys": ["bar", "beat"], "arrow_key": "section_requested"},
            {"type": "keyline", "keys": ["key", "camelot", "bpm", "chord_now", "lead_op"]},
            {"type": "beats", "key": "beat"},
            {"type": "chords", "key": "chords", "phase_key": "phrase_phase"},
            {"type": "countdown", "key": "drop_eta", "label": "drop in", "hot_below": 8},
            {"type": "meter", "label": "section", "done_key": "section_bars_left", "total_key": "section_bars",
             "inverse": True, "right_keys": ["section_bars_left", "section_next"], "right_format": "{0} bars left · next: {1}", "palette": "section"},
            {"type": "meter", "key": "energy", "label": "energy", "palette": "energy", "right_keys": ["energy", "energy_bias"], "right_format": "{0:.2f} (bias {1:+.2f})"},
            {"type": "meter", "key": "arc_progress", "label": "arc", "palette": "arc", "right_keys": ["movement"], "right_format": "movement {0}"},
        ]},
        {"id": "direct", "title": "Direct", "col": 1, "show_when": "live", "widgets": [
            {"type": "chips", "items_key": "gestures", "action": "gesture", "id_field": "id", "label_field": "label", "flash": True},
            {"type": "ask", "action": "ask", "placeholder": "tell the director… e.g. darker and slower over ten minutes, then a bright arp for the drop",
             "status_key": "director"},
            {"type": "director_log", "key": "director", "limit": 6},
            {"type": "buttons", "items": [
                {"label": "👍 more like this", "action": "gesture", "value": "more_like_this"},
                {"label": "👎 not this", "action": "feedback", "value": False},
            ], "trailing_key": "taste", "trailing_format": "taste: {up} up · {down} down"},
        ]},
        {"id": "steer", "title": "Steering", "col": 2, "foldable": True, "widgets": [
            {"type": "choice", "key": "style", "options_key": "styles", "action": "style", "id_field": "id",
             "label_field": "label", "sub_format": "{label} · {bpm[0]}–{bpm[1]} bpm"},
            {"type": "slider", "key": "bpm", "action": "bpm", "label": "tempo", "min": 50, "max": 180, "step": 0.5, "decimals": 1},
            {"type": "select", "key": "camelot", "idle_key": "key", "action": "key", "label": "key", "options": "camelot", "trailing_key": "mode"},
            {"type": "slider", "key": "energy_bias", "action": "energy", "label": "energy bias", "min": -0.5, "max": 0.5, "step": 0.01, "decimals": 2, "signed": True},
            {"type": "slider", "key": "density", "action": "density", "label": "density", "min": 0, "max": 1.5, "step": 0.01, "decimals": 2},
            {"type": "slider", "key": "swing", "action": "swing", "label": "swing", "min": 0, "max": 0.33, "step": 0.005, "decimals": 3},
            {"type": "slider", "key": "brightness", "action": "brightness", "label": "brightness", "min": 0.4, "max": 1.6, "step": 0.01, "decimals": 2},
            {"type": "slider", "key": "master", "action": "master", "label": "level", "min": 0, "max": 1, "step": 0.01, "decimals": 2},
            {"type": "select", "key": "set_length_s", "action": "set_length", "label": "arc length", "options": [
                {"id": 1800, "label": "30 min"}, {"id": 3600, "label": "1 h"}, {"id": 7200, "label": "2 h"},
                {"id": 10800, "label": "3 h"}, {"id": 21600, "label": "6 h"}, {"id": 43200, "label": "12 h"}]},
            {"type": "text", "key": "fluid_slots", "action": "fluid", "label": "SoundFont", "placeholder": "slots on SoundFonts, e.g. keys,pad (next start)"},
        ]},
        {"id": "layers", "title": "Layers", "col": 2, "hint": "tap to mute", "foldable": True, "widgets": [
            {"type": "toggles", "items_key": "slots", "on_key": "layers", "off_key": "muted", "badge_key": "fluid_slots",
             "badge": "SF", "action": "mute", "value_format": {"slot": "$item", "on": "$next"}},
        ]},
        {"id": "scenes", "title": "Scenes", "col": 2, "foldable": True, "widgets": [
            {"type": "scenes", "key": "scenes", "actions": {"save": "scene_save", "load": "scene_load", "delete": "scene_delete"}},
        ]},
        {"id": "pattern", "title": "Pattern", "col": 2, "hint": "Strudel · one cycle = one bar", "foldable": True, "folded": True, "advanced": True, "widgets": [
            {"type": "code", "key": "pattern", "action": "pattern", "clear_action": "pattern_clear", "status_key": "pattern_error",
             "available_key": "pattern_available", "engine_key": "pattern_engine", "slots_key": "pattern_slots",
             "placeholder": "stack(\n  s(\"bd*4, [~ cp]*2, hh(5,8)\"),\n  note(\"<0 3 5 7>(3,8)\").scale(\"A:minor\").s(\"bass\").lpf(800),\n  note(\"0 2 4 [6 7]\").scale(\"A2:minor\").s(\"lead\").gain(energy)\n)",
             "help": "Globals: energy, section, bar, key, bpm, phrase, chords. Sounds map to slots: bd sd/cp hh oh rim, bass lead pad arp keys. Ctrl+Enter evaluates."},
        ]},
        {"id": "timeline", "title": "Phrases", "col": 1, "foldable": True, "show_when": "live", "widgets": [
            {"type": "phrase_log", "key": "log", "limit": 14},
        ]},
        {"id": "health", "title": "Health", "col": 2, "foldable": True, "folded": True, "show_when": "live", "widgets": [
            {"type": "kv", "items": [
                {"label": "uptime", "key": "uptime_s", "format": "duration"},
                {"label": "heard", "key": "heard_s", "format": "{0:.1f} s"},
                {"label": "composed ahead", "key": "lead_s", "format": "{0} s"},
                {"label": "notes", "key": "notes"}, {"label": "peak", "key": "peak"},
                {"label": "render errors", "key": "render_errors"},
                {"label": "motifs", "key": "motifs"}, {"label": "seed", "key": "seed"},
                {"label": "pattern engine", "key": "pattern_engine"},
                {"label": "SoundFont slots", "key": "fluid_slots", "format": "list"},
                {"label": "ramps", "key": "ramps", "format": "json"},
            ]},
        ]},
    ],
}


def surface_spec():
    return SURFACE


# -- validation (the gate) ----------------------------------------------------
_JS_WIDGETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "web", "static", "js", "gen", "widgets")


def registered_widget_types():
    """Widget types the client registers (scanned from the JS modules)."""
    types = set()
    if not os.path.isdir(_JS_WIDGETS_DIR):
        return types
    for fn in os.listdir(_JS_WIDGETS_DIR):
        if fn.endswith(".js"):
            with open(os.path.join(_JS_WIDGETS_DIR, fn), encoding="utf-8") as fh:
                types |= set(re.findall(r"""register\(\s*['"]([a-z_]+)['"]""", fh.read()))
    return types


def _widget_keys(w):
    keys = []
    for k in ("key", "items_key", "options_key", "on_key", "off_key", "badge_key", "status_key", "available_key",
              "engine_key", "slots_key", "done_key", "total_key", "phase_key", "arrow_key", "trailing_key", "idle_key", "toggle_key"):
        if isinstance(w.get(k), str):
            keys.append(w[k])
    for k in ("keys", "sub_keys", "right_keys"):
        keys += [x for x in (w.get(k) or []) if isinstance(x, str)]
    for it in w.get("items") or []:
        if isinstance(it, dict) and isinstance(it.get("key"), str):
            keys.append(it["key"])
    return keys


def validate_surface(spec, status_keys, actions=GEN_ACTIONS, widget_types=None):
    """Return a list of problems (empty = valid)."""
    problems = []
    widget_types = registered_widget_types() if widget_types is None else set(widget_types)
    ids = set()
    for card in spec.get("cards", []):
        cid = card.get("id")
        if not cid or cid in ids:
            problems.append(f"card id missing/duplicate: {cid!r}")
        ids.add(cid)
        for w in card.get("widgets", []):
            t = w.get("type")
            if t not in widget_types:
                problems.append(f"{cid}: widget type {t!r} not registered on the client")
            acts = [w.get("action"), w.get("clear_action")] + list((w.get("actions") or {}).values())
            acts += [it.get("action") for it in (w.get("items") or []) if isinstance(it, dict)]
            for a in acts:
                if a and a not in actions:
                    problems.append(f"{cid}/{t}: action {a!r} not whitelisted")
            for k in _widget_keys(w):
                if k not in status_keys:
                    problems.append(f"{cid}/{t}: key {k!r} not in status")
    return problems
