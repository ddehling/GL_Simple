# Interaction Panels

The web UI carries **one** interaction nav slot, and whichever weather
set is currently live owns it. A set that declares nothing gets no tab —
that is the default, and most sets should stay that way. The tab exists
so a specific set can put something in a guest's or an operator's hands:
buttons that fire events, a slider that opens the water up, a picker that
dives somewhere.

The Club / DJ set uses this mechanism to keep its existing bespoke DJ
page: its panel is a pointer at `/dj`, so the tab reads "DJ" and only
exists while the club set is running.

## Where panels live

Panels are project content, so they live in the project repo and hang off
the `hooks:` block of `projects/<id>/project.yaml`:

```yaml
hooks:
  random_events: projects.fan.random_events
  interaction: projects.fan.interaction
```

That module exposes one dict keyed by weather-set id:

```python
INTERACTION_PANELS = {
    "ocean": { ... },
}
```

A project with no `interaction` hook has no interaction tab on any set.
Panels are re-read on project swap.

## Panel shapes

### 1. Point at a bespoke page

For a set that deserves a hand-written page (the club set's DJ console):

```python
"club": {
    "label": "DJ",          # nav tab text
    "page": "/dj",          # where the tab goes
    "requires": "dj",       # optional extra gate (see below)
    "theme": {"accent": "#ff4fa3"},
},
```

Nothing is rendered generically — the tab is just a link, live only while
that set is.

### 2. Declare controls

Everything else: describe the controls and the shared renderer
(`web/templates/interaction_panel.html`) draws them, themed with the
set's own colors.

```python
"ocean": {
    "label": "Ocean",                  # nav tab text
    "title": "The Deep Listens",       # page heading
    "blurb": "Call something up out of the dark.",
    "theme": {"accent": "#3fd8e8", "bg": "#03121a",
              "panel": "#09232e", "text": "#dff6fb"},
    "sections": [
        {
            "title": "Call a creature",
            "note": "One swims past.",
            "layout": "grid",          # grid | list | compact
            "controls": [ ... ],
        },
    ],
},
```

## Controls

Every control has a `type` (default `button`) and an `action` (default
`event`), plus `label`, and optional `icon`, `hint`, `color`.

| type | what it draws |
|------|---------------|
| `button` | a big touch target; fires once per press |
| `slider` | a range with a live readout (`min`, `max`, `step`, `default`) |
| `toggle` | an on/off pad (`on`, `off` values) |
| `select` | a dropdown over `options` |

| action | what it does | keys |
|--------|--------------|------|
| `event` | schedules a named event from the project's event map | `event`, optional `duration` (defaults to the set's `random_event_duration`), `frame_id`, `cooldown` |
| `state` | transitions to a weather state **inside the current set** | `state`, or `options` for a `select` |
| `param` | writes a web parameter override; a `button` with no `value` releases it | `param`, `value` |
| `signal` | publishes a value into `outstate['interaction']` for shaders | `signal`, `value`, `cooldown` |

Examples:

```python
# fire an event, at most one every 20 s
{"label": "Whale", "icon": "🐋", "event": "whale", "cooldown": 20},

# drive a weather parameter
{"type": "slider", "action": "param", "param": "fog", "label": "Clarity",
 "min": 0.0, "max": 1.0, "step": 0.02, "default": 0.3},

# hand the parameter back to the weather system
{"type": "button", "action": "param", "param": "fog", "label": "Release"},

# jump somewhere in this set
{"type": "select", "action": "state", "label": "Dive to", "options": [
    {"label": "Kelp forest", "value": "ocean_kelp_forest"},
    {"label": "The abyss",   "value": "ocean_abyss"},
]},

# something only a shader cares about
{"type": "button", "action": "signal", "signal": "lantern", "label": "Light it"},
```

### Reading a signal from a shader

Signal controls publish into outstate:

```python
rec = outstate.get('interaction', {}).get('lantern')
if rec:
    fresh = time.time() - rec['t'] < 2.0     # a press in the last 2 s
    level = rec['value']                     # slider / toggle value
    presses = rec['count']
```

`t` is wall-clock seconds, `count` increments on every press. A momentary
button reads as "was `t` recent?"; a slider or toggle reads straight off
`value`.

## Gates

`requires` hides a panel even while its set is live:

| gate | passes when |
|------|-------------|
| `dj` | the autonomous DJ subsystem is available |

Add new gates to `_REQUIRES_GATES` in `lib/interaction.py`.

## Trust model

The browser never names an event, state or parameter. It sends a control
*id* plus (for sliders and selects) a value; the server looks the control
up in the **live** set's panel and derives the action from the spec, with
values clamped to the declared range. A tab left open through a set
change therefore stops working rather than firing the wrong set's events.

## Gotchas

- **Event dedup is per (effect, frame_id).** `EventScheduler` skips an
  event whose effect is already running on that frame. Several buttons
  that all schedule the same effect class (e.g. the ocean megafauna
  events, which share one effect) can only put one thing on screen at a
  time — a press during another one is silently dropped. Give each button
  its own effect, or accept the one-at-a-time reading.
- **`state` actions respect set membership**, not the state-lock toggle:
  a state outside the live set is refused and logged.
- **Parameter overrides persist** after the panel is closed, exactly like
  the ones set from the Controls tab. Offer a release button for anything
  a guest can push far.
- A malformed control is dropped with a `[Interaction]` warning at load;
  the rest of the panel still renders. A panel with nothing usable and no
  `page` is ignored entirely.

## Files

| file | role |
|------|------|
| `lib/interaction.py` | spec normalization, gating, action resolution |
| `web/web_controller.py` | `/interaction`, `/api/interaction/info`, `/api/interaction/spec`, the `interaction_action` socket handler |
| `Stories_OGL.py` | `_publish_interaction_panels`, `_apply_interaction_controls` (render thread) |
| `web/templates/interaction_panel.html` | the generic page |
| `web/static/js/interaction_panel.js` | the renderer |
| `web/static/js/interaction_tab.js` | the nav slot on every page |
| `web/static/css/interaction.css` | theme-variable-driven styling |
| `projects/<id>/interaction.py` | the panels themselves |
