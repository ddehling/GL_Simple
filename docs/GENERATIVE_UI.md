# The generative surface — one spec, native console first

The operator UI for the generative system is a declarative *surface spec*
(`lib/gen/ui.py`) rendered by a widget registry. Two renderers exist and must
stay in parity (the gates enforce it):

| Renderer | Entry | Where it runs | Talks to |
|---|---|---|---|
| **Native console (PyQt6)** — the primary surface | `python tools/gen_console.py` | a laptop / the show box desktop | this machine's audio (`LocalBackend`), or the show over HTTP (`--remote URL`: `GET /api/gen/status`, `POST /api/gen/action`) |
| Web page | `/gen` in the show's panel, or `tools/gen/gen_server.py` | any phone/tablet on the venue Wi-Fi | the show's socket |

`tools/gen/console/widgets.py` is the native registry (`@register('type')`, class
with `build()`/`update_state()`), `tools/gen/console/app.py` the window (two column
stacks, foldable cards, 10 Hz refresh, Space/Esc/Ctrl+Q). Adding a widget type means
one class there **and** one module in the web registry, or the console gate fails;
adding a control or a card is still one spec entry that both renderers pick up.

## The web renderer

The `/gen` page is a shell rendering the same spec through a browser-side
widget registry. The DJ page (`/dj`) is a 1,200-line monolith; this one is built so
that growing it never means editing a monolith.

```
lib/gen/ui.py            SURFACE spec: cards -> widgets {type, key, action, ...}
   |  GET /api/gen/surface (web/web_controller.py)
   v
web/static/js/gen/index.js      imports every widget module, boots app.js
web/static/js/gen/app.js        fetches the spec, builds cards, feeds state to widgets
web/static/js/gen/registry.js   register(type, {create, update})
web/static/js/gen/widgets/*.js  one module per family of widgets
web/static/js/gen/actions.js    emit(action, value) -> socket 'gen_action'
web/static/js/gen/store.js      state, fmt(), CAMELOT, el()
web/static/css/gen.css          layout (1 column / 2 columns >= 900 px) + widget styles
web/templates/gen_panel.html    a shell: header, tabs, <div id="surface">
```

Live data flows one way: the show publishes `gen_info` (the `GenSystem.status()`
dict, or `idle_info()` while idle) at 5 Hz over the existing socket; every widget
reads its value from that dict by `key`. Operator input flows one way: every widget
sends `{action, value}` through `gen_action`; the server whitelists and clamps it
(`lib/gen/actions.py`) and the conductor applies it at the next phrase boundary.
Nothing in the browser names an engine, a synth, or a composer.

## The contract

**Card**: `{id, title?, hint?, kind?: "card"|"banner"|"transport", col?: 1|2,
show_when?: "always"|"live"|"idle", foldable?, folded?, advanced?, sticky?, widgets: [...]}`

**Widget**: `{type, key?, action?, ...type-specific}`. Conventions:

| Field | Meaning |
|---|---|
| `key` | status field the widget displays (`*_key` variants for secondary fields) |
| `action` | whitelisted action the widget sends (`actions` map for multi-action widgets) |
| `items_key` / `options_key` | status field holding a list to render (chips, choice) |
| `show_when` (on button items) | `always` / `live` / `idle` |
| `*_format` | tiny Python-style format: `{0:.2f}`, `{1:+.2f}`, `{name}`, `{bpm[0]}` |

Widget types today: `banner buttons headline keyline beats chords countdown meter kv`
(basic.js), `chips choice slider select text toggles` (controls.js), `ask director_log
scenes` (director.js), `code phrase_log` (pattern.js).

## How to extend it

**Add a control that uses an existing widget** — one dict in `lib/gen/ui.py`:
```python
{"type": "slider", "key": "swing", "action": "swing", "label": "swing", "min": 0, "max": 0.33, "step": 0.005}
```
If the value is new, add the status field in `GenSystem.status()` (and `idle_info()`
if the card is visible idle) and the action in `lib/gen/actions.py` (`GEN_ACTIONS`,
`sanitize_gen_action`, `apply_gen_action`). Run `tools/tests/_gen_ui_test.py`: it fails
if a key or action is missing.

**Add a widget type** — one module under `web/static/js/gen/widgets/` (or a new
function in an existing one):
```js
import { register } from '../registry.js';
import { el } from '../store.js';
register('vu', {
    create(spec, ctx) { return el('div', 'meter'); },          // build once
    update(el_, state, spec, ctx) { /* paint from state[spec.key] */ },  // 5 Hz
});
```
Import the module from `index.js` if it is new. The validator scans the widgets
folder for `register('name')`, so a spec cannot name a widget the client lacks.

**Add a card** — one dict in `SURFACE["cards"]`. `col` places it in the two-column
layout; `show_when: "live"` hides it while idle; `foldable`/`folded` persist per card.

**Another surface** — the spec is renderer-agnostic JSON. A native console
(PyQt6, like the DJ planner) or a MIDI mapping can consume the same spec and speak
the same actions (`POST /api/gen/action` is the HTTP twin of the socket event).

## Gates
- `tools/tests/_gen_console_test.py` — native console: registry parity with the spec,
  idle/live rendering, chip/slider/toggle/scene/transport input reaching the system,
  remote mode against the show's own web controller; writes a screenshot when given a path.
- `tools/tests/_gen_ui_test.py` — web: spec validity (types, actions, keys, ids), client
  module wiring, scenes round trip, routes, and that the validator rejects bad specs.
- Browser smoke (Playwright) is run by hand against `tools/gen/gen_server.py`.
