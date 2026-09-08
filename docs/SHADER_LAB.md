# Shader Lab — live programming the LEDs in plain language

Open **`/shaderlab`** in the web control panel, type a description
("slow purple waves that pulse with the bass"), and the pattern goes live
on the LEDs and in the page's preview within seconds. Follow-ups refine
the same pattern ("slower", "more orange", "add sparkles on the beat").
A collapsible code editor below the chat shows the generated GLSL and
compiles hand edits through the same pipeline.

## How it works

- **Six engine-level live slots** (`renderer/effects/live_shader.py`,
  `NUM_SLOTS`, one `live_shader` event managing six layered
  `LiveShaderEffect` instances) are scheduled as an implicit background
  event in **every** weather set of **every** project. Slot 1 draws first
  (bottom layer); higher slots composite on top via straight-alpha
  blending, so several patterns run at once (e.g. a background wash in
  slot 1, objects layered above). Each slot renders nothing until a source
  is loaded, has its own conversation/session, intensity + knobs, and
  status; the `/shaderlab` page's slot cards select which slot the chat,
  editor, and library Play target.
- The `/shaderlab` page sends the description to Claude (model
  `claude-opus-5`, via the Claude Code CLI or the API — see "Enabling AI
  generation") on a web worker thread; the generated
  `mainImage` body is validated, queued through `control_dict`
  (`request_live_shader`, latest-wins), and hot-compiled **on the render
  thread** — side-compiled into a fresh program and swapped only on
  success. A typo can never blank the LEDs; the previous pattern keeps
  rendering and the driver's error log (line numbers match the user code)
  is fed back to the model for up to **2 automatic repair rounds** before
  the error is shown. Validator rejections (banned keyword, missing code
  block) get their own 2 automatic rewrite rounds *before* compiling, so
  the model fixes its own contract violations instead of surfacing them.
- Because the source lives in `outstate['live_shader_source']`, the
  pattern **survives weather-set changes and project swaps** — the
  respawned wrapper recompiles it automatically (same trick as
  `narrative_player`).

## Enabling AI generation

Same two transports as the DJ planner copilot and the narrative editor,
tried in this order:

1. **Claude Code CLI** (`claude` on PATH or in the usual install spots) —
   uses your existing Claude Code session, **no API key needed**. If you
   run Claude Code on this machine, the lab just works.
2. **anthropic SDK** with an explicitly resolved credential: the key saved
   from the DJ planner's copilot panel (`~/.gl_simple_copilot.json`),
   `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or the `ant` CLI's OAuth
   token (`ant auth login`).

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # only needed without the CLI
```

If neither resolves, the page shows an "editor mode only" banner — the
code editor, compile pipeline, and saved patterns still work.

## What user/LLM code looks like

Only a Shadertoy-style entry point — the engine owns `#version`,
`precision`, all uniform declarations, and `main()`:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;   // canvas is 128 x 300 (tall)
    ...
    fragColor = vec4(rgb, alpha);        // STRAIGHT alpha (HARD RULE 4)
}
```

### Uniform contract (provided every frame; never declare these)

| Uniform | Meaning |
|---|---|
| `float iTime` | seconds (CPU-accumulated) |
| `vec2 iResolution` | (128, 300) — tall portrait fan |
| `float iBass / iMid / iHigh` | smoothed band energy (~1.0 = recent average) |
| `float iBassPunch / iHighPunch` | transient punch envelopes |
| `float iBeat` | 1.0 on each beat, decaying (`beat_decay`) |
| `float iEnergy` | overall music energy |
| `float iDrop / iBuild` | drop spike / build-up level |
| `float iBPM / iPhrase` | tempo / phrase position 0..1 |
| `float iSeason / iWind / iRain` | environment |
| `float iKnob0..iKnob3` | live performance sliders 0..1 (rest at 0.5) — see below |
| `float iFade` | engine-applied spawn fade (already multiplied into alpha) |
| `float iIntensity` | engine-applied web "Intensity" slider (multiplied into alpha) |

### Performance knobs

A pattern can expose up to four live sliders by declaring labels in its
first line — `// KNOBS: speed | sparkle amount` — and reading
`iKnob0..iKnob3` (label order = knob order). The `/shaderlab` Now-playing
card renders a slider per declared knob, plus an always-present Intensity
fader. Values flow web → `control_dict['live_shader_params']` →
`outstate` (~5 Hz) and are NOT saved with patterns; the declaration
comment is, so a saved pattern keeps its knobs. The system prompt asks
the model to wire natural parameters to knobs on its own.

### Built-in GLSL stdlib

The wrapper header ships a curated helper library so generated patterns
don't re-derive the same boilerplate every time (that boilerplate was
~half the output tokens — and output tokens are what you wait for):
noise (`hash11/hash21/vnoise/fbm`), color (`hsv2rgb`, IQ `palette`),
shapes (`sdCircle/sdSegment/sdBox`, `glow`), `rot2`, and `sphereNormal`
for fake-3D balls. The validator rejects redefinitions by name
(`validate_glsl` / `STDLIB_NAMES`), the gate keeps the ban-list and the
header in sync, and hand-written editor code can call all of them too.

### Diff-based refinements

Follow-ups ("slower", "more orange") reply with ```edit fences containing
`<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` pairs instead of regenerating
the whole shader — several times fewer output tokens, so refinements land
in seconds. The server applies them against the conversation's current
code (each SEARCH must match exactly once), re-validates the merged
result, and canonicalizes the chat history to the merged shader so the
next edit targets real code. A failed merge automatically re-asks for the
full shader through the same repair loop.

### Refining saved patterns and hand edits

Any successful editor/library compile *seeds the conversation* with the
compiled code (a synthetic turn), so "slower, more purple" right after
loading a saved pattern refines THAT pattern — the library is a set of
starting points, not dead ends.

### Model toggle

The prompt row has an Opus (best) / Sonnet (fast) / Haiku (fastest)
selector, remembered per browser. On the SDK transport the reply streams
into the status line as it's written; on the CLI transport the status
line shows elapsed time instead.

### Parameterized pattern bank (instant, no LLM)

`web/shader_patterns.py` holds ~a dozen hand-tuned GLSL templates
(waves, orb, rain, fire, plasma, sparkles, rays, rings, aurora, chase,
breathe), each with typed clamped params (speed plus
per-template extras), an audio-reactivity choice, and a PALETTE — one
hue ('custom'), a duotone parsed from "blue and purple ..." ('duo'), or
named schemes (rainbow, sunset, neon, ice, pastel, candy) — painted
along each template's color coordinate and slidable live via the third
knob (color shift). Templates are layered compositions (foam on wave
crests, embers over flames, splash rings under rain, star backdrops
behind aurora...), all pre-validated
against the fan geometry, alpha rules, and contrast playbook, and all
declaring the standard `speed | brightness | color shift` knobs.

Two ways in:
- **Instant match**: a clear chat request ("slow blue waves", "green
  sparkles that pulse with the bass") is matched locally — colors, speed
  words, and audio words parse into params — and compiles in
  milliseconds with **no model call**. The matcher is conservative: any
  meaningful word it doesn't recognize sends the request to normal
  codegen instead, so novel ideas still get the full model.
- **Built-in chips**: the "Built-in patterns" row plays any template
  with defaults into the selected slot.

Either way the instantiated GLSL is normal user code: it seeds the
slot's conversation (chat refinements work on it), lands in the editor,
and can be saved to the library.

### Layout helpers (engine-provided GLSL functions; never redeclare)

The canvas is a polar fan folded into a rectangle: column = strip angle
across a 180° arc, row = distance along the strip from 4 ft (hub) to
20.6 ft (rim). The wrapper header defines helpers (kept in sync with
`renderer/fan_geometry.py`) so patterns can draw in true physical space:

| Helper | Meaning |
|---|---|
| `vec2 fanUV(fragCoord)` | physical position, x −1..1 / y 0..1 — distances and circles are TRUE here |
| `vec2 fanPos(fragCoord)` | the same in feet (rim = 20.6) |
| `float fanAngle(fragCoord)` | strip angle in radians, π (left) .. 0 (right) |
| `float fanRadius(fragCoord)` | feet from the hub, 4.0 .. 20.6 |

Objects/motion ("a ball", "rain falling", "ripples from a point") should
use `fanUV`; strip-native looks (rays, expanding arcs, per-strip chases)
use raw `fragCoord / iResolution`. The system prompt teaches the model
this rule; hand-written editor code gets the same helpers.

**Preview caveat:** the `/shaderlab` page previews the raw 128×300 canvas,
so `fanUV`-space patterns look bent there — open `/preview` and switch to
a Fan mode to see the true physical layout.

### Restrictions (enforced by `web/shader_lab.py::validate_glsl`)

No `while`/`do` (GPU-hang guard; `for` with small constant bounds only),
no `uniform`/`layout` declarations, no textures/samplers, no
`gl_FragDepth`, no preprocessor `#version/#include/#extension/#pragma`
(`#define` is fine), 20k char cap. Comments are stripped before the scan —
a `// textured look` comment doesn't trip the keyword ban.

## Seeing and driving what's running

The page is a wish-driven stage — nobody picks slots:

- **The wish box** routes free-form requests server-side
  (`web/shader_lab.py::route_wish`): a new subject lands on a free layer
  (16 available; the oldest is replaced when full), "make the waves
  slower" refines the layer it names, "a bit slower" follows the layer
  you last touched, "remove the fire" takes a layer off, "clear
  everything" empties the stage. Refinements keep their layer's
  conversation; new layers start fresh ones.
- **The pattern palette** — colorful tappable tiles, one per built-in
  template — tosses patterns onto the stage instantly (no LLM).
- **"On the lights now"** shows one tile per ACTIVE layer (empty slots
  are invisible): level fader always; tap a tile to focus it — revealing
  its knobs, and for built-in layers a row of **palette dots** (instant
  recolor), a **dice** (shuffle colors/pace via
  `shader_patterns.shuffle_params`), and an **audio-cycle** button —
  all direct manipulation, zero model calls (`shaderlab_restyle`).
  Focusing also aims the chat and editor at that layer ("talking to
  ..."); unfocus to wish freely again. ✕ removes a layer.
- The **Advanced drawer** holds the GLSL editor and the saved-pattern
  library; the engine panel (with the Blank-stage toggle) sits under the
  live preview.

## HTTP API (for scripts / future file-watch mode)

```
GET  /api/shaderlab/info                  -> {llm_available, num_slots,
                                              saved_patterns: [{name, prompt,
                                              created}], slots, params, status}
POST /api/shaderlab/compile {code, slot?} -> {ok, error, seq}  (synchronous;
                                              slot defaults to 0)
```

Saved patterns live in `config/shader_lab_library.json`.

## If a shader hangs the GPU

`while` is banned, but a huge constant-bound `for` loop can still stall
the driver. On Windows, TDR resets the GPU after ~2 s (screen blink, app
usually survives); if the app wedges, kill and relaunch `Stories_OGL.py`.
The offending pattern is NOT auto-reloaded on restart (the slot starts
empty), so the show comes back clean.

## Threading contract (for maintainers)

Web/worker threads never touch GL; the render thread never calls the
Anthropic API. Flow: socket event → validate on web thread →
`control_dict['request_live_shader']` under `_dict_lock` + `live_shader_dirty`
flag → `Stories_OGL._apply_live_shader_controls()` (every-frame fast path,
bypasses the 5 Hz throttle) moves it into `outstate` → next frame the
`shader_live_shader` wrapper calls `effect.set_source()` (side-compile,
swap-or-keep-old, **`_uniform_cache.clear()`**) → status flows back via
`control_dict['live_shader_status']` → worker polls it
(`wait_for_compile`) and reports over the socket.
