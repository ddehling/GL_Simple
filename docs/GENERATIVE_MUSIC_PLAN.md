# Generative Note-Level Music for the Club / DJ System — Options & Plan

Status: PLAN ONLY (no code). Rewritten 2026-09-04 after the operator clarified the goal:
**a system that generatively composes and plays notes** — a live algorithmic composer
driving a synthesizer — not a system that renders finished tracks for the DJ to mix.
(The earlier track-rendering direction is summarised in §8 and set aside.)

## 0. What we are building

A **generative deck**: an autonomous composer that decides notes (drums, bass, chords,
melody, texture) a few bars ahead, a sample-accurate scheduler that plays them on the
DJ's clock, and a synth engine that turns them into audio inside the existing
`AudioEngine`. It is steered by the signals the show already publishes (night arc,
energy target, key, mood, weather state), it can take the floor from the DJ and hand it
back at a planned seam, and it publishes exact beat/bar/phrase/drop truth to the visuals.
The same engine can play ambient generative music for the non-club weather sets.

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
| `python-osc` in requirements; `loopback` analyzer source exists | `requirements.txt`, `lib/audio_analyzer.py:26,:122` | Makes a SuperCollider backend *possible* later (§3, S3). |
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

**Recommendation:** C1 is the product; C2/C3 are seasoning after the engine exists.

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

**Recommendation:** S1 core + S2 optional, one `Voice` interface, backend chosen per
instrument slot in the style preset.

---

## 4. Architecture

```
  outstate: arc phase/heat, bpm_target, key (camelot / key_center), music_mood, weather,
            operator controls (style, density, brightness, complexity, floor)
        │
        ▼
  Composer (control thread, phrase-first)         lib/dj/gen/composer/
    form state machine → harmony → rhythm grids → bass/melody/motif memory
    emits Phrase(bar_events[...]) 4–8 bars ahead; knows every drop in advance
        │  note events {at_sample, voice, pitch, vel, dur, params}
        ▼
  NoteScheduler (sample clock, 256-frame sub-blocks, `at`-stamped like submix automation)
        │
        ▼
  SynthRack (worker thread → ~2-bar ring)          lib/dj/gen/synth/
    Voice S1 numba/numpy | Voice S2 FluidSynth ; FX chain ; sidechain from kick
        │  (n,2) float32 @ 44.1k
        ▼
  GenDeck(Deck) = DJSubmix.decks["g"]  — sync MASTER, exact grid, gain/EQ/filter automation
        │                              (or: peer attach_track for non-club sets)
        ▼
  AudioEngine mixer → limiter → speakers → internal tap → shaders
        │
  GenTrack (virtual TrackInfo) → DJSystem.live_beat / outstate_keys → club director
```

Key design points:
- **Two mount modes, one engine.** Club: `GenDeck` inside `DJSubmix` so the brain plans
  seams (`plan_transition` with `GenTrack`: exact grid, `bpm_conf=1.0`, rolling sections,
  `mix_ins`/`mix_outs` regenerated at the next phrase boundary) and records slave to it
  (a ring cannot be reseeked, so `_apply("sync")` refuses `"g"` as slave). Non-club: peer
  `attach_track("generative")` with `is_ambient=True` so the existing ambient volume and
  crossfade rules apply and `play_ambient` is not called for that state.
- **Take / give the floor.** Record→gen: `plan_transition(cur=record, cand=GenTrack)`; the
  composer commissions its own tempo/key from the live `out_bpm` and the record's Camelot
  and starts on the record's downbeat grid. Gen→record: composer schedules an outro at
  the next phrase, `GenTrack.mix_outs` advertises it, normal arming follows. Continuity
  watchdog and a pre-decoded escape record remain mandatory.
- **Visual truth is exact.** Beat/bar/phrase phase come from the scheduler clock (no
  measurement, unlike a stream); `dj_next_drop_eta`, `build_level` and hard-drop stamps
  come from the composer's plan. The club director gets choreography a human VJ cannot.
- **Autonomous first.** No performer input; operator controls are sparse and coarse.

Code layout: `lib/dj/gen/` (`composer/` {form, harmony, rhythm, melody, motif, styles},
`scheduler.py`, `synth/` {voices_numba, voices_fluid, fx, rack}, `deck.py` (GenDeck,
GenTrack), `presets/*.yaml`), `tools/dj/gen_player.py` (standalone `--wav`/`--live`,
mirrors `dj_player.py`), `tools/tests/_dj_gen_*_test.py`, `media/soundfonts/` (optional),
`docs/GENERATIVE_MUSIC.md` (user doc). Engine deltas: `DJSubmix.decks["g"]`, sync refusal,
a/b flip generalisation in `system.py`, ambient bypass in `Stories_OGL.transition_to_weather`.

---

## 5. Phased roadmap

**Phase 0 — Spike (no engine changes).** `tools/dj/gen_player.py --wav out.wav --minutes 2
--bpm 126 --key 8A --style groove`: rule composer + S1 voices (kick, hats, clap, sub bass,
one lead, one pad) + delay/reverb. Listen. Measure CPU on the N150. Decide the initial
voice set and whether S2 is wanted from the start.

**Phase 1 — Engine.** `Voice` interface, S1 voices, FX, NoteScheduler, SynthRack worker +
ring, deterministic seeds. Gate `_dj_gen_synth_test.py`: seed → WAV, kick-to-grid
alignment via `seamverify.measured_kick_alignment`, no NaN/clip, CPU per block under
budget.

**Phase 2 — Composer v1 (club).** Form/harmony/rhythm/melody/motif; style presets for the
four themes; energy/arc steering; engineered drops. Gate: 30-minute offline render per
preset, phrase-boundary and drop timing assertions, plus an operator listening session
(this is the ear-gate; instruments do not judge musicality).

**Phase 3 — Club integration.** `GenDeck`, `GenTrack`, sync-master rule, a/b
generalisation, record↔gen seams through the real brain, outstate truth, `/dj` controls
(gen on/off, style, density, brightness, complexity, take/give floor), night sim
`--generative`, `dj_review` rows. Gate `_dj_gen_handoff_test.py` (hand-pumped engine e2e,
modelled on `_dj_brain_test.e2e_test`).

**Phase 4 — Non-club sets.** `generative:` per weather state in project config; weather →
composer mapping (rain → density, wind → filter motion, fog → reverb, season → mode);
`INTERACTION_PANELS` sliders; S2 SoundFont voices for organic palettes.

**Phase 5 — Quality loop and seasoning.** "Phrase Lab" rating treadmill (Seam Lab
pattern) that renders phrases per preset and logs verdicts; tune style parameters from
verdicts; motif memory across the night; then optional C2 (neural phrase proposer),
C3 (LLM phrase director), S3 (SuperCollider backend).

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

## 7. Open decisions for the operator
1. **Palette**: analog-style synthesis only (S1), or add SoundFont instruments (S2) from
   the start? (S2 needs `libfluidsynth` on the show box and `.sf2` assets.)
2. **Scope**: club deck first (Phases 1–3), or ambient generative for all weather sets
   first (Phases 1, 2-lite, 4)? Both share Phases 0–1.
3. **How much floor**: should the generative deck be an occasional guest between records
   (bridges, wind-down, empty library) or able to run the whole night?
4. **Appetite for SuperCollider** as a later backend (second process, richer sounds).

---

## 8. Set aside: track-level generation (previous draft)

Researched and documented before the clarification: commissioning finished 4–6 min tracks
from offline neural models (Stable Audio Open 1.5, ACE-Step, Magenta RealTime 2) on a GPU
box and ingesting them as first-class library rows with ground-truth grids (DB v16
`provenance`, hot-add via a 10 s poll, `Brain.add_tracks`); and a stream deck for Google
Lyria RealTime / Magenta RT2 station mode. It remains a valid complement (the ingest and
`StreamDeck` designs share the `GenDeck` seam work) but it is not what the operator wants
as the product. Details are in git history of this file.
