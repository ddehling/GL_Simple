# Generative Note-Level Music for the Club / DJ System — Options & Plan

Status: **Phase 0 spike landed 2026-09-05** (see §0.1). Plan rewritten 2026-09-04/05 after the operator clarified the goal:
**a system that generatively composes and plays notes** — a live algorithmic composer
driving a synthesizer — not a system that renders finished tracks for the DJ to mix.
Operator decisions folded in (2026-09-05): **both analog synthesis and SoundFonts from the
start; its own subsystem and interface, separate from the club and modelled on how the DJ
is wired; must run for long periods (hours to all night); SuperCollider explored in §3.1
as a candidate analog backend, to be decided by spike.** (The earlier track-rendering direction is summarised in §9 and set aside.)

## 0.1 Phase 0 status (2026-09-05)

Standalone, no engine changes. Both analog backends and the SoundFont path exist and
are gated:

| Piece | Where | Gate |
|---|---|---|
| Composer (theory, styles, form, harmony, rhythm, melody + motif memory) | `lib/gen/composer/`, `lib/gen/theory.py`, `lib/gen/events.py` | `tools/tests/_gen_composer_test.py` |
| Synth rack: numba/numpy analog voices, delay/reverb/sidechain, FluidSynth SoundFont slots, AudioEngine track protocol | `lib/gen/synth/` | `tools/tests/_gen_synth_test.py` |
| SuperCollider backend: SynthDefs in Python (supriya), NRT render, realtime scheduler | `lib/gen/backends/sc.py` | `tools/tests/_gen_sc_test.py` (SKIPs without scsynth) |
| Player CLI | `tools/gen/gen_player.py` | — |
| **Strudel patterns** (C4): Node bridge + composer source; Pattern card on `/gen`; `--strudel` in the player | `tools/gen/strudel/`, `lib/gen/composer/strudel.py`, `media/patterns/example.js` | `tools/tests/_gen_strudel_test.py` |
| **Director** (C5): gestures, language, taste; Direct card on `/gen`; brightness / ramps / section requests / slot patterns | `lib/gen/director.py`, `lib/gen/feedback.py` | `tools/tests/_gen_director_test.py` |
| **Native console** (2026-09-06, the primary surface — operator does not want a web UI here): PyQt6 window rendering the same spec with native widgets; in-process audio or `--remote` control of the show (`/api/gen/status` + `/api/gen/action`); Space/Esc shortcuts | `tools/gen_console.py`, `tools/gen/console/` | `tools/tests/_gen_console_test.py` |
| **Spec-driven surface** (2026-09-06): `/gen` is a shell that renders a declarative spec (cards → widgets with `key`/`action`) through a widget registry; two-column layout on wide screens; **scenes** (named steering snapshots); extension guide in `docs/GENERATIVE_UI.md` | `lib/gen/ui.py`, `lib/gen/scenes.py`, `web/static/js/gen/`, `web/static/css/gen.css`, `web/templates/gen_panel.html` | `tools/tests/_gen_ui_test.py` |
| **Frontend (Phase 3, landed 2026-09-05):** `GenSystem` conductor (composes ahead on its own thread, steering queue, movements, end-of-set, supervision, night log), shared action whitelist, `/gen` page + `gen_action` socket + `POST /api/gen/action`, Gen nav tab, Stories_OGL soundtrack-takeover bridge, `gen:` config, standalone server | `lib/gen/system.py`, `lib/gen/actions.py`, `web/templates/gen_panel.html`, `web/static/js/gen_tab.js`, `web/web_controller.py`, `Stories_OGL.py` (`_apply_gen_controls/_gen_start/_gen_stop`), `config.yaml`, `tools/gen/gen_server.py` | `tools/tests/_gen_system_test.py` |

```bash
pip install numba supriya pyfluidsynth            # numba is in requirements.txt already
sudo apt install supercollider-server sc3-plugins fluidsynth fluid-soundfont-gm   # optional backends
pip install mini-racer                            # Strudel patterns in-process (no node); bundle is committed
python tools/gen/gen_player.py --wav out.wav --minutes 3 --style groove --bpm 124 --key 8A --seed 1 --log
python tools/gen/gen_player.py --wav out.wav --style downtempo --fluid-slots keys,pad
python tools/gen/gen_player.py --wav out_sc.wav --backend sc-nrt          # same notes through scsynth
python tools/gen/gen_player.py --live --minutes 10                        # speakers (miniaudio)
python tools/tests/_gen_composer_test.py && python tools/tests/_gen_synth_test.py && python tools/tests/_gen_sc_test.py
python tools/tests/_gen_system_test.py                                     # conductor + frontend contract
python tools/gen/gen_server.py --port 5000 [--fluid-slots keys,pad]        # /gen page on this machine's speakers, no show app
python tools/gen/gen_server.py --wav night.wav --minutes 20                # same page, headless render
```
In the show: `gen.enabled: true` in `config.yaml` makes the **Gen** tab appear; START on
`/gen` takes the soundtrack exactly as the DJ does (ambient silenced, oneshots muted,
analyzer on the internal tap) and STOP hands it back. A project may also point a
set's interaction panel at it: `{"label": "Generative", "page": "/gen", "requires": "gen"}`.

Measured in the dev container (4 cores, no GPU): numpy rack renders at ~17x realtime
steady state; scsynth NRT at ~49x; FluidSynth adds negligible cost. Kick onsets land on
the composer's sample grid on both backends (43/43 and 35/35). **Next: the operator
listening session** (styles × backends), then Phase 1 decisions from §5.

## 0. What we are building

A **generative music subsystem** (`GenSystem`), a sibling of the DJ rather than a part of
it: an autonomous composer that decides notes (drums, bass, chords, melody, texture) a
few bars ahead, a sample-accurate scheduler on its own sample clock, and a synth rack
(analog-style voices + SoundFont voices) that renders audio inside the existing
`AudioEngine`. It has its own weather set and its own control page (like the club set has
`/dj`), runs for hours unattended, is steered by an arc plus the show's published state,
and publishes exact beat/bar/phrase/drop truth to the visuals. Optional later: DJ
hand-offs so the two subsystems can trade the floor, and ambient generative beds for
other weather sets.

It honours the operator verdicts on record in `docs/DJ_README.md` (245-289, 386-408):
it is never a layer pasted over a record; when it plays, **it is the music**.

---

## 1. Constraints and seams (from the code survey)

| Constraint / seam | Where | Consequence for this design |
|---|---|---|
| Show box: Intel N150, 4 cores, no GPU | `config.yaml` `dj:`, `bin/ensure_cpu_performance.sh` | Note-level composition is negligible CPU. Synthesis must be vectorised (numpy) or JIT (numba, already in `requirements.txt` and used by `lib/pixel_extract.py`). No neural audio models on-box. |
| 44100 Hz stereo float32; every `lib/dj/*` hardcodes `RATE` | `Stories_OGL.py:2860-2868` | Synth renders natively at 44.1 kHz. |
| Render-ahead thread is the deadline; 43% of a core measured during a keylock blend; pattern is pre-render off-thread | `lib/audio_engine.py:457-485`, `RING_TARGET_MS=1200` `:59` | Synthesis runs on its own worker thread into a ring; the deck's `read()` is a memcpy. numba `nogil` kernels release the GIL. |
| Track protocol `read(n)->(n,2)`, `.done`, `fade_out`, `attach_track` | `lib/audio_engine.py:755` | Peer-track mode for non-club sets. |
| `DJSubmix` decks a/b/c, sample-accurate automation in 256-frame sub-blocks, PLL sync, `duck`, `fx_play`, telemetry | `lib/dj/submix.py:46,:59,:105-130,:195,:521` | Home for the generative deck `"g"`. Its note scheduler uses the same `at`-stamped event idiom. |
| `Deck` grid/phase surface: `beat_phase:336`, `phase_snap:349`, `_fetch:401`, `read:492` | `lib/dj/deck.py` | `GenDeck` subclasses `Deck`, serves audio from its ring, reports phase from its own clock (exact). |
| Brain hand-off planning: `plan_transition:3044`, `best_pair:2470` (needs `mix_ins`/`mix_outs`), `build_events:4645`; a/b flip hardcoded at `system.py:1703,:1952` | `lib/dj/brain.py`, `system.py` | A virtual `GenTrack` TrackInfo (exact grid, `bpm_conf=1`, rolling sections, phrase-aligned mix points) lets the real brain plan record→gen and gen→record seams. The a/b flip must be generalised so `active_deck` may be `"g"`. |
| Steering signals: `arc_target:668`, `bpm_target:611`, `_arc_base:651`, `theme.prefer_tags/bpm_range`, `key_center`, `music_mood`, weather params | `lib/dj/system.py`, `themes.py`, `Stories_OGL.py:2322-2399` | The composer's inputs. |
| Visual truth: `live_beat:747`, `outstate_keys:791`, `DJVisualCoupler` | `lib/dj/system.py`, `lib/dj/vis.py` | The composer *knows* every drop before it happens → `dj_next_drop_eta`, engineered-drop stamps (`_dj_drop_wall/_dj_drop_hard`, `system.py:830`). |
| Internal analyzer tap | `Stories_OGL.py:2319` | Spectrum for shaders is free. |
| Web `/dj` `DJ_ACTIONS`, `POST /api/dj/action`, declarative `INTERACTION_PANELS` | `web/web_controller.py:1150`, `lib/interaction.py` | Operator controls: style preset, density/brightness/complexity, take/give the floor. |
| Ambient beds per weather state: `play_ambient` / `_restore_state_ambient` | `Stories_OGL.py:1911-1945,:2950` | A set may declare `generative:` instead of `ambient_sound`; the generator becomes the bed. |
| `python-osc` in requirements; `loopback` analyzer source exists | `requirements.txt`, `lib/audio_analyzer.py:26,:122` | Makes a SuperCollider backend *possible* later (§3, S3) — optional, see §7. |
| Offline verification culture: hand-pumped engine e2e, `_dj_night_sim`, `seamverify`, Seam Lab ratings | `tools/tests/_dj_brain_test.py:301`, `docs/DJ_VERIFICATION.md` | Every phase ends with a rendered gate and a listening session. |

---

## 2. The composer (what notes) — options

The composer works **phrase-first**: it plans 4–8 bars ahead (harmony, section, density
targets), then realises each bar into note events just before it is needed. All variants
share one `Phrase` data model so they can be swapped or combined.

### C1. Rule- and probability-based algorithmic composition (recommended core)
- **Harmony**: key from the DJ's Camelot / `key_center`; per-style chord grammars (Markov
  over functions, e.g. i–VI–III–VII loops for club, modal drones for ambient); voice-leading
  by nearest-tone; occasional modulation staged at phrase boundaries (and only when the DJ
  is not about to mix a record in — the seam plan fixes the key).
- **Rhythm**: per-voice Euclidean / probability grids (kick, clap, hats, perc), swing,
  ghost notes, fills at phrase ends; density and syncopation scale with the energy target.
- **Bass / melody**: constrained random walks and Markov chains over scale degrees with a
  **motif memory** (state, repeat, vary, transpose, invert) so the night has identity;
  call-and-response between voices; register and velocity ride energy.
- **Form**: section state machine (intro → groove → build → break → drop → outro) driven
  by the arc, with 16/32-bar phrase discipline and an engineered drop the visuals can
  count down to.
- **Style presets** per theme (`groove`, `peak_heavy`, `chill_evening`, `wind_down`) and
  per weather set (forest, ocean, spooky…) as parameter bundles, not code.
- Cost: microseconds per bar. Fully deterministic under a seed → hermetic tests.
- Libraries: hand-roll the core (the sample-clock scheduler must be ours); optionally
  borrow pattern classes from `isobar` (pure Python, MIT) and theory helpers from
  `music21` (already in `requirements-dj-mood.txt`, heavy — only for offline tooling).

### C2. Neural symbolic "phrase proposer" (later, optional)
- Small MIDI transformers exist that run on CPU: Anticipatory Music Transformer
  (`stanford-crfm/music-small-ar-inter-100k`, ~112M, Lakh MIDI, Apache-2.0), or the 2026
  125M piano-autocomplete class of models (~100 notes/s on a phone). Electronic/dance
  coverage in their training data is thin; treat output as *proposals* the rule engine
  filters (key, range, density) rather than the composer itself.
- Runs on the show box CPU for sparse parts (a 112M model at a few tokens per note is
  feasible ahead of time) or on the optional GPU box (§6).

### C3. LLM phrase director (later, optional)
- Once per phrase (every 8–16 bars, seconds of latency is fine) an LLM proposes a
  high-level plan: section, chord loop, motif operations, energy curve. Precedent: the
  planner's Set Copilot runs through the `claude` CLI with no API key. Not in the audio
  path; degrades to C1 silently when unavailable.

### C4. Strudel patterns (explored and landed 2026-09-05)
**What Strudel is.** The JavaScript port of TidalCycles (AGPL-3.0, active on Codeberg
`uzu/strudel`, npm `@strudel/*` 1.2.6): a *pattern language* where `s("bd*4, [~ cp]*2,
hh(5,8)")` is a bar of drums, `note("<0 3 5 7>(3,8)").scale("A:minor")` a Euclidean bass
line, and functions like `every`, `off`, `sometimesBy`, `jux` transform patterns
algebraically. It is browser-first (Web Audio via superdough, Web MIDI, an OSC bridge to
SuperDirt), but its **pattern engine is pure JS and runs headless under Node**.

**Verified here.** `@strudel/core` + `mini` + `tonal` + `transpiler` evaluate code and
query cycles into events (haps: begin/end fractions + a control dict) in Node 22: 64
cycles ≈ 1500 events in ~190 ms, deterministic across queries (random ops are seeded by
cycle). One packaging wart: core imports a browser-only class from `@kabelsalat/web`
at module load, whose Node entry lacks the export; a two-line local shim package
(`tools/gen/strudel/shim/`) fixes it without patching node_modules.

**Can it run in Python? Yes (verified 2026-09-06).** The four packages bundled by
esbuild into one browser-style script (`tools/gen/strudel/dist/strudel.bundle.js`,
~0.9 MB, committed; rebuild with `build_bundle.sh`) run inside the Python process on an
embedded V8 (`pip install mini-racer`, a manylinux wheel) with a few shims (`console`,
`performance`, timers). Same numbers as Node: 64 bars ≈ 1150 events in ~150 ms, context
signals and error reporting intact. **The show box needs no Node at all.** The Node
sidecar (`bridge.mjs`) stays as the fallback and development path; `open_engine()` picks
V8 first. Vortex, the official Python *port* of Tidal, is stalled ("free puppies") and
could not even be fetched here (Codeberg only); not a candidate.

**How it fits: Strudel as a composer, not as an audio engine.** We do not use its browser
synth, MIDI or OSC. Either engine speaks the same tiny interface (eval / query / ping). `lib/gen/composer/strudel.py` translates haps
to `NoteEvent`s on the rack's sample clock — **one cycle == one bar** at the composer's
tempo — mapping `s` to slots (bd→kick, cp→snare, hh→hat, oh→ohat, rim→perc, bass/lead/
pad/arp/keys), `note`/`n` to MIDI, `gain`/`velocity` to velocity, `lpf`/`cutoff` to the
voice cutoff, `legato`/`clip` to gate length. The rule composer's **form, harmony,
energy and arc keep running underneath** and are handed to the pattern as globals
(`energy`, `section`, `bar`, `key`, `bpm`, `phrase`, `chords`), so a pattern can be
written once and *breathe with the night*: `.gain(energy)`, `.lpf(600 + energy*900)`.
Everything downstream — SoundFont slots, the SuperCollider backend, the engine mount,
the visuals' ground truth — is untouched.

**Operator surface.** The `/gen` page has a Pattern card (textarea, EVAL applies at the
next phrase boundary, CLEAR returns to the autonomous composer; Ctrl+Enter). Bad code is
reported on the card and never interrupts playback (the previous pattern keeps playing).
`gen_player.py --strudel file.js` renders a pattern file; `media/patterns/example.js`
is a starting point. Gate: `tools/tests/_gen_strudel_test.py` (runs on whichever engine
is available; SKIPs with neither).

**Verdict.** Strudel is the best answer to "how does the operator *author* generative
material" — far more expressive per line than preset knobs, with a large public corpus
of patterns to borrow — while the rule composer remains the always-on autonomous core
and the long-run/arc machinery. Not adopted: Strudel's own scheduler/audio (browser tab
must stay open; bypasses the engine), the OSC→SuperDirt route (a second synth stack),
and Vortex (the Python port, self-described "free puppies", stalled).

### C5. The director: interaction above code (landed 2026-09-06)
The operator's stated preference is to interact **higher than code**. `lib/gen/director.py`
gives three inputs that all resolve to one **Intent** (a small dict: `set`/`nudge`/`ramp`
parameters, `section`, `hold`, `layers` mute/unmute, per-slot Strudel `patterns`, `reseed`,
`end`, `like`) applied through the same phrase-boundary steering queue:

- **Gestures** — a curated vocabulary with musical meaning (`darker`, `brighter`, `open it
  up`, `strip to drums`, `bring a melody`, `sparser/denser`, `more swing`, `slower/faster`,
  `build to a drop`, `breakdown`, `back to groove`, `modulate up/down`, `keep this going`,
  `let it move`, `wind down`, `more like this`, `new ideas`). Chips on the `/gen` page. No
  dependencies. A new **brightness** lever (filter-cutoff multiplier on every pitched patch)
  makes "darker/brighter" real; **ramps** interpolate a parameter over N bars; **section
  requests** steer the form; **slot patterns** let one slot take Strudel notes while the
  rules keep the rest.
- **Language** — free text to an LLM director (Claude through the `claude` CLI when present,
  else the `anthropic` SDK; the DJ planner's copilot precedent). The prompt carries the live
  state, the schema and a Strudel cheat sheet; the reply is JSON only, **validated**
  (unknown keys dropped, numbers clamped, sections/slots checked) and any proposed pattern is
  **sandboxed** in a scratch Strudel engine and must produce events for its slot before it can
  reach the rack. Runs on a worker thread; the page shows busy / reply / what changed.
  Degrades cleanly: gestures and autonomy never depend on it.
- **Taste** — 👍/👎 record snapshots (style, section, key, layers, energy, density, swing,
  brightness) to `logs/gen_prefs.json`; `more like this` records a like and nudges the
  parameters toward the liked centroid and away from disliked ground. A nudge, never a lock.

Gate: `tools/tests/_gen_director_test.py` (director tested with an injected transport; the
real model path is exercised only when `claude` or a key is present).

**Recommendation:** C1 is the product; C5 is how the operator talks to it; C4 is the
director's (and the curious operator's) language for new material; C2/C3 remain optional.

---

## 3. The synth engine (how it sounds) — options

All backends implement one `Voice` interface: `note_on/off(pitch, vel, at_sample)`,
`set_param(name, value, ramp)`, `render(n) -> (n,2)`.

| | Backend | Fit | Verdict |
|---|---|---|---|
| **S1** | **In-process numba/numpy synth**: subtractive (multi-osc saw/square/sine → SVF/ladder-style filter → ADSR), 2-op FM, 808-style drum synthesis (kick pitch-sweep, noise snare/hat), sample player for one-shots; FX: tempo delay, FDN reverb, sidechain (`duck` already exists), master soft-clip | Everything stays inside `AudioEngine`: limiter, ring, analyzer tap, DJ hand-offs, no new process or system package. Rough budget: 16 voices × ~50 ops/sample × 44.1 k ≈ 35 M ops/s, a few % of a core in numba; block-vectorised numpy is also viable with the 1.2 s ring. | **Core.** Sound design is our effort, but that is also where the character comes from. |
| **S2** | **FluidSynth + SoundFonts** via `pyfluidsynth` (`Synth` without `start()`, `get_samples(n)` → numpy, 44.1 kHz default) | Instant huge palette (pianos, strings, mallets, GM kits, synth SF2s), block-rendered into our protocol. Needs `libfluidsynth` (apt) + `.sf2` assets in `media/soundfonts/`. Low CPU. | **Optional instrument backend** behind the same `Voice` interface — strongest for ambient / organic sets, adequate for club leads. |
| S3 | **SuperCollider `scsynth`** driven over OSC (`python-osc` present; `supriya`/`sc3nb` optional) | Best synth vocabulary, tiny CPU, runs on Pi-class hardware. But audio leaves our engine: no limiter/ring coupling, DJ crossfades only via the analyzer `loopback` source, a second process to supervise (JACK/PipeWire). | Later "pro" backend only if S1's palette proves limiting. |
| S4 | VST3 hosting (Surge XT / Vital / Dexed via `pedalboard` or DawDreamer) | Excellent sounds, but Python hosts render in blocks with awkward state/tail semantics for continuous play, and pull in JUCE-sized deps. | Not first. |
| S5 | MIDI out to hardware synths | Repo deliberately keeps MIDI out of the show (`Stories_OGL.py:417`); new hardware, `mido`/`rtmidi`. | Not recommended. |

**Decision (operator):** S1 **and** S2 from the start, one `Voice` interface, backend
chosen per instrument slot in the style preset (e.g. drums + bass + lead on S1, keys /
pads / mallets / strings on S2). S2 needs `libfluidsynth` on the show box (apt package;
the install scripts in `bin/` gain one line) and `.sf2` assets under `media/soundfonts/`
(start with a GM bank such as FluidR3 or GeneralUser GS, both free; add specialised banks
per palette). Pin the FluidSynth output rate to 44100 and render on the rack's worker
thread, never in the engine callback.

**SuperCollider** is explored in depth in §3.1 below (operator asked to explore it,
2026-09-05). Short version: it is a serious candidate for the *analog* backend, in place
of or alongside the numba voices, if a second process plus PipeWire routing is an
acceptable operating cost on the show box. It does not replace FluidSynth for SoundFonts.

### 3.1 SuperCollider, explored

**What it is.** SuperCollider has two halves. `scsynth` is a real-time synthesis
*server*: a tree of nodes (synths in groups) reading and writing audio/control buses,
where every instrument is a compiled `SynthDef` graph of unit generators (about 300 core
UGens; ~200 more in `sc3-plugins`: Moog-style ladder filters, `JPverb`/`Greyhole`
reverbs, `DFM1`, granular, physical models). It is controlled *only* by OSC messages, and
OSC bundles carry timestamps, so scheduling is sample-accurate however late the client
is. `supernova` is a multi-threaded variant. `sclang` is SuperCollider's own language and
client; **we would not use it** — the Python binding below compiles SynthDefs itself, so
the show box needs only the server package. NRT mode (`scsynth -N score.osc … out.wav`)
renders an OSC score to a file with no audio device, which gives hermetic tests.

**Python binding: `supriya`** (26.9b0, Sept 2026; Python 3.10–3.14; MIT; actively
maintained; POSIX/macOS/Windows). It boots and supervises `scsynth`, compiles SynthDefs
from Python UGen classes, allocates nodes/buses/buffers, sends timestamped bundles, has
tempo/meter-aware clocks and patterns, and renders non-realtime scores to WAV. The
alternatives are the wrong shape: `sc3nb` leans on `sclang`; FoxDot/Renardo are
live-coding environments with their own SynthDef packs and an `sclang` dependency; raw
`python-osc` would reinvent supriya's server/node bookkeeping.

**How its audio gets into our engine — three routings.**

| | Routing | Keeps limiter / ring / internal tap / DJ crossfade? | Notes |
|---|---|---|---|
| R1 | `scsynth` plays straight to the sound card next to our engine; PipeWire mixes them | No | Simplest. Visuals only via the analyzer's `loopback` source. Two masters on one device; not recommended. |
| **R2** | `scsynth` → PipeWire **null sink**; our engine **captures the sink's monitor** and mounts it as the `GenRack` source track (`attach_track`) | **Yes** | Everything downstream is identical to the in-process rack. Capture via miniaudio's capture API, or the `parec` subprocess the analyzer already uses (`lib/audio_analyzer.py:242`). Added latency ≈ SC block (64 samples) + PipeWire quantum + capture ≈ 10–40 ms, in front of our 1.2 s ring; the composer schedules bars ahead, so it is invisible. Fallback without PipeWire: ALSA `snd-aloop`. Windows: WASAPI loopback of a virtual cable (VB-Cable). **Recommended if SC is chosen.** |
| R3 | NRT render per phrase to buffers, played by our engine | Yes | No live parameter changes (a phrase is fixed once rendered); good for tests, not for the live rack. |

**Clock and truth.** `scsynth` has its own clock. We schedule every note with a
timestamped bundle at `now + latency` (SuperCollider's standard practice, 100–200 ms);
the composer already works phrases ahead. Visual ground truth = the intended note time
plus the measured capture latency, auto-calibrated at start (send a click, detect it in
the captured stream, store the offset; re-check hourly). Under PipeWire both processes
clock off the same device, so no long-run drift between "SC time" and engine time; the
engine's ring lead (`render_lead_frames`) is subtracted exactly as for the DJ.

**Ops on the show box.** `apt install supercollider-server sc3-plugins` (server + core
plugins, no Qt IDE; both are in Debian/Ubuntu and Arch). Launch as a supervised
subprocess of `GenSystem`: `pw-jack scsynth -u 57110 -B 127.0.0.1 -i 0 -o 2 -z 64` with
`/status` heartbeats, restart-on-death, and a silence fallback while restarting (same
supervision the in-process worker would need, §4.2). Pin the SC version (3.13/3.14) in
the install scripts (`bin/linux-install.sh` already drives apt). Windows: the SC
installer provides `scsynth` on PortAudio; one line in `bin/windows-install.ps1` plus the
virtual-cable step for R2. Raspberry Pi is supported by SC upstream.

**CPU.** `scsynth` renders dozens of voices in ~1–3 % of one core on N150-class hardware,
in its own process — zero GIL interaction with the engine or the DJ. PipeWire adds ~1–2 %.

**What it buys over the in-process numba rack (S1).**
- Hundreds of proven, well-documented UGens: sound design is a week, not a season.
  Per-sample feedback and modulation (FM matrices, resonators, physical models) that
  block-based numpy cannot do cheaply.
- New instruments are a Python function (a `@synthdef`), hot-loaded into the server while
  the show runs, never touching engine threads.
- Hermetic NRT rendering for tests; a large corpus of example instruments to borrow from;
  the same server runs on a Pi.
**What it costs.**
- A second process and a PipeWire routing to install, pin and supervise (R2); one more
  thing to go wrong at 2 a.m. Debugging spans two processes.
- SoundFonts are *not* an SC feature: FluidSynth (S2) stays in-process either way
  (or runs as a second captured process through the same null sink).
- Windows needs a virtual cable for R2; the club box is Linux, so this matters only if the
  generative set must also run on Windows dev machines.

**Decision framework.** SC wins if sound-design leverage and instrument variety matter
more than a minimal process count; the numba rack wins if "one process, one venv" is the
higher value. Each side is about a day to spike, so the recommendation is to run both
Phase 0 spikes and choose by ear and by ops feel:

- **Spike SC-0** (standalone, no engine changes): install `supercollider-server` +
  `sc3-plugins` on the show box; `supriya` boots `scsynth` under `pw-jack` into a null
  sink; the Phase 0 composer sends a 3-minute set through four SynthDefs (kick, hats,
  ladder bass, pad + `JPverb`) as timestamped bundles; capture the monitor with `parec` to
  WAV; measure round-trip latency and CPU; render the same score through NRT and confirm
  it matches. Listen next to the numba spike output.
- If SC is chosen: S3 becomes the analog backend behind the same `Voice` interface
  (`note_on` → `s_new` in a timestamped bundle, `note_off` → gate, `set_param` → `n_set`),
  `GenRack` becomes "capture track + FluidSynth in-process", and §4.2 supervision covers
  the `scsynth` child. Everything else in the plan is unchanged.

---

## 4. Architecture

Mirror the DJ's wiring exactly, so every seam is one already proven in this codebase.

| DJ (exists) | Generative (new) |
|---|---|
| `lib/dj/system.py DJSystem` — conductor on its own planner thread, `step()`, `status()`, `outstate_keys()` | `lib/gen/system.py GenSystem` — conductor on its own thread; composer runs here |
| `lib/dj/submix.py DJSubmix` — one `attach_track` object, sample clock, `post_many` automation | `lib/gen/rack.py GenRack` — one `attach_track("gen_rack")` object; note scheduler + synth rack + FX; worker thread renders ~2 bars ahead into a ring; `read()` is a memcpy |
| `Stories_OGL._dj_start/_dj_stop` (`:2854/:2932`) — soundtrack takeover, `oneshots_muted`, analyzer → `internal` | `_gen_start/_gen_stop` — same takeover contract (stop ambient, mute oneshots, analyzer `internal`, restore on stop) |
| `Stories_OGL._apply_dj_controls` (`:2541`) — 5 Hz bridge, `dj_info` at `:2705` | `_apply_gen_controls` — 5 Hz bridge, `gen_info` |
| `web_controller.py DJ_ACTIONS` (`:1150`), `queue_dj_action`, `dj_action` socket, `POST /api/dj/action`, `/dj` page | `GEN_ACTIONS`, `queue_gen_action`, `gen_action`, `POST /api/gen/action`, `/gen` page (`web/templates/gen_panel.html`) |
| club set: `INTERACTION_PANELS` `{"label": "DJ", "page": "/dj", "requires": "dj"}` (`lib/interaction.py:12,:56`) | a new weather set in the fan project, e.g. `generative`, with `{"label": "Generative", "page": "/gen", "requires": "gen"}`; add the `gen` gate to `_REQUIRES_GATES` |
| `lib/dj/vis.py DJVisualCoupler` + `live_beat` | `lib/gen/vis.py GenVisualCoupler` — same reactive keys (`audio_energy`, `build_level`, `drop`, beat pulses, `bar_phase`, `phrase_phase`), exact from the scheduler |
| `tools/dj/dj_player.py --live/--wav` | `tools/gen/gen_player.py --live/--wav/--minutes/--seed/--style` |
| `tools/tests/_dj_*` gates, `_dj_night_sim` | `tools/tests/_gen_*` gates, `_gen_long_run_sim` |

```
  arc (set length / all-night) + outstate (weather, season, key_center, music_mood)
  + operator controls (style, density, brightness, complexity, tempo, key, hold, reseed)
        │
        ▼
  GenSystem (own thread)  ── Composer: form → harmony → rhythm → bass/melody/motif memory
        │  Phrase = note events {at_sample, slot, pitch, vel, dur, params}, 4–8 bars ahead
        ▼
  GenRack (attach_track peer of dj_submix)
     NoteScheduler (sample clock, 256-frame sub-blocks)
     SynthRack worker thread → ring:  slots → Voice(S1 numba analog | S2 FluidSynth) → FX → soft-clip
        │  (n,2) float32 @ 44.1k
        ▼
  AudioEngine mixer → limiter → speakers → internal tap → shaders
        │
  GenSystem.outstate_keys() → GenVisualCoupler → club-style director in the generative set
```

Why a sibling and not a deck inside the DJ: the operator wants a separate instrument with
its own page; the DJ's brain, decks and seam vocabulary are built around *records* with
analysed grids; and the DJ is off while the generative set plays (same as the club set vs
every other set today). The one place the two meet is the optional hand-off in Phase 6,
which can then be done through the DJ's own `plan_transition` with a virtual `GenTrack`
(exact grid, `bpm_conf=1.0`) and a `GenDeck` view of the same rack — the rack's audio is
the same either way.

### 4.1 Control surface

- **Phase 3 (web, matches the DJ):** `/gen` page with start/stop, style preset, tempo, key
  /mode lock, energy nudge and arc waypoints (reuse the `/dj` arc-strip widget), density /
  brightness / complexity / swing sliders, section hold, "reseed", motif freeze, per-slot
  mute and backend (analog ↔ SoundFont), 👍/👎 on the current phrase. State goes down as
  `gen_info` at 5 Hz over the existing socket, like `dj_info`. Coarse controls at 5 Hz are
  fine for an autonomous instrument; this is the cheapest route and stays remote-friendly.
- **"More performant than a web interface", two candidates for later:**
  1. **Native desktop console** — a PyQt6 app (`tools/gen_console.py`; the DJ planner is
     already PyQt6) driving the running show over the same `POST /api/gen/action` +
     socket contract, so nothing in the engine changes. Gives real faders/knobs, ms-level
     feedback (a local OSC channel via `python-osc` can replace the socket if 5 Hz is too
     coarse), scopes and a phrase timeline.
  2. **Physical MIDI controller** — the nanoKONTROL2 driver exists (`lib/midi_controller.py`,
     input only, `register_callback:317`). The show deliberately takes no MIDI
     (`Stories_OGL.py:417-422`, an autonomy decision for the club); the generative set
     would be the first justified exception, and it should be scoped to that set. This is
     an operator decision to record when Phase 3 lands.
  Recommendation: build the web page first (it is also the API), then decide between 1
  and 2 from live use.

### 4.2 Long-run operation (hours to all night)

Requirements that follow from "runs for long periods", each with the mechanism:
- **Macro-form over hours**: an arc like the DJ's themes (`themes.py`) but for movements —
  palette rotation, tonal-centre drift by fifths/relative modes at movement boundaries,
  tempo drift within a band, density waves. Prevents the "same loop for four hours" fatigue.
- **Motif memory with decay**: motifs are reused and varied for identity, then retired;
  a bounded store (LRU) so memory is flat over the night.
- **Seeded and reseedable**: a night seed plus per-phrase sub-seeds; "reseed" is a control;
  every event log line carries the seed so a moment can be reproduced offline.
- **Numerical hygiene**: int64 sample clock (no float drift), filter and delay state flushed
  of denormals, envelopes clamped, FluidSynth voice count capped, periodic soft reset of
  reverb tails at silent bars.
- **Supervision**: the rack worker is a supervised thread — if the ring underruns or the
  worker dies, the rack emits silence with a fade, restarts the worker, logs it, and
  `gen_info` shows it; a continuity watchdog like the DJ's (`system.py:2574`) ensures the
  composer always has the next phrase queued.
- **Flat resource profile**: no per-note allocations on the audio path (preallocated
  voice pools), bounded event queues, log rotation (`logs/gen_*.jsonl` like `logs/dj_*.jsonl`).
- **Offline soak**: `_gen_long_run_sim.py` renders an 8-hour night at faster than real
  time through the hand-pumped engine and asserts no NaN/clip, bounded memory, phrase
  discipline, and macro-form coverage; `gen_review.py` reads the night log the way
  `dj_review.py` does.

Code layout: `lib/gen/` (`system.py`, `rack.py`, `scheduler.py`, `composer/` {form,
harmony, rhythm, melody, motif, styles}, `synth/` {voices_numba, voices_fluid, fx},
`vis.py`, `presets/*.yaml`), `tools/gen/gen_player.py`, `tools/gen/gen_review.py`,
`tools/gen_console.py` (later), `web/templates/gen_panel.html`, `media/soundfonts/`,
`tools/tests/_gen_*`, `docs/GENERATIVE_MUSIC.md`. The generative *weather set* (states,
palette, director shader, `INTERACTION_PANELS`) is project content in the fan repo.

---

## 5. Phased roadmap

**Phase 0 — Spike (no engine changes).** `tools/gen/gen_player.py --wav out.wav --minutes 3
--bpm 124 --key 8A --style groove --seed 1`: rule composer + S1 voices (kick, hats, clap,
sub bass, lead) + S2 SoundFont keys/pad + delay/reverb. Listen. Measure CPU on the N150.
Confirms the `Voice` interface, the FluidSynth block-render path at 44.1 k, and whether
the sound is worth the rest of the plan. **Run Spike SC-0 (§3.1) alongside it** and pick
the analog backend (numba S1 vs SuperCollider S3) by ear and ops feel before Phase 1.

**Phase 1 — Engine.** `Voice` interface, S1 voices, S2 FluidSynth voices, FX,
NoteScheduler, SynthRack worker + ring, `GenRack` track protocol, seeds, supervision.
Gates: `_gen_synth_test.py` (seed → WAV, kick-to-grid alignment via
`seamverify.measured_kick_alignment`, no NaN/clip, CPU per block under budget),
`_gen_rack_test.py` (underrun → fade + restart).

**Phase 2 — Composer v1.** Form / harmony / rhythm / melody / motif memory; three style
presets (club groove, downtempo, ambient); energy/arc steering; engineered drops; macro-form
over hours. Gates: per-preset offline renders with phrase/drop timing assertions; the
8-hour soak (`_gen_long_run_sim.py`); and the operator listening session, which is the
real gate.

**Phase 3 — Show integration + web page.** `GenSystem` in `Stories_OGL` (`_gen_start/_stop`,
5 Hz bridge, `gen_info`), `GEN_ACTIONS` + `/gen` page, `gen` gate in `lib/interaction.py`,
`GenVisualCoupler`, night log + `gen_review.py`. The generative weather set (states,
palette, director) lands in the fan project repo. Gate: hand-pumped e2e modelled on
`_dj_brain_test.e2e_test`, plus a live evening.

**Phase 4 — Control surface v2.** From live use, choose the PyQt6 console and/or the
nanoKONTROL2 mapping (§4.1); either talks to the engine through the Phase 3 API.

**Phase 5 — Quality loop and seasoning.** "Phrase Lab" rating treadmill (Seam Lab pattern)
per preset; tune style parameters from verdicts; more palettes and SoundFont banks; then
optional C2 (neural phrase proposer), C3 (LLM phrase director), S3 (SuperCollider).

**Phase 6 (optional) — DJ hand-offs and ambient beds.** `GenTrack` + `GenDeck` view of the
rack for record↔generative seams through the DJ brain; `generative:` per weather state for
other sets (weather → composer mapping) as an ambient bed via the same rack.

---

## 6. Hardware

**No GPU is needed for the core plan.** Composition is trivial; S1/S2 synthesis fits the
N150 with numba, on a worker thread separate from the DJ's render thread.

A GPU box only matters if Phase 5's neural phrase proposer is adopted. If so, the earlier
recommendation stands: an SFF x86 Linux desktop with a low-profile NVIDIA card (RTX A2000
12 GB at 70 W, or RTX 4060 LP) — standard Ubuntu + CUDA, PyTorch and JAX both first-class.
For a ~112M symbolic model a Jetson Orin Nano Super (7–25 W, ~$249, Ubuntu/JetPack,
PyTorch) is also sufficient and cheaper, at the cost of an aarch64 toolchain.

---

## 7. Decisions recorded (2026-09-05)

| Question | Decision |
|---|---|
| Palette | Both: analog-style synthesis (S1) **and** SoundFont instruments (S2) from the start. |
| Scope / mount | A **separate subsystem and interface**, modelled on the DJ (own set, own `/gen` page, own conductor), not a deck inside the DJ. DJ hand-offs and ambient beds for other sets are optional later phases. |
| Interface | Web page first (it doubles as the API). Later, a more performant surface: PyQt6 console and/or a scoped nanoKONTROL2 mapping — decide from live use. |
| Duration | Must run for long periods → §4.2 long-run requirements are in scope from Phase 1, soak test from Phase 2. |
| SuperCollider | **Under exploration (2026-09-05, §3.1).** Viable as the analog backend via R2 (null-sink capture into the engine) with `supriya`; decide between it and the numba rack by running both Phase 0 spikes. FluidSynth stays for SoundFonts either way. |

## 8. Remaining open questions (non-blocking)
1. Which SoundFont banks to license/ship first (GM bank for coverage, plus one or two
   specialised banks per palette)?
2. Should the generative set also be allowed while the club set is active (Phase 6), or
   strictly one-or-the-other like every other set today?
3. Log retention and whether thumbed-up phrases should be exported as MIDI for reuse.

---

## 9. Set aside: track-level generation (previous draft)

Researched and documented before the clarification: commissioning finished 4–6 min tracks
from offline neural models (Stable Audio Open 1.5, ACE-Step, Magenta RealTime 2) on a GPU
box and ingesting them as first-class library rows with ground-truth grids (DB v16
`provenance`, hot-add via a 10 s poll, `Brain.add_tracks`); and a stream deck for Google
Lyria RealTime / Magenta RT2 station mode. It remains a valid complement (the ingest and
`StreamDeck` designs share the `GenDeck` seam work) but it is not what the operator wants
as the product. Details are in git history of this file.
