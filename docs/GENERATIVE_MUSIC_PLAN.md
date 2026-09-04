# Continuous Generative Music for the Club / DJ System — Options & Plan

Status: PLAN ONLY (no code). Written 2026-09-04 after a survey of the audio engine,
the DJ subsystem, the operator verdicts recorded in `docs/DJ_README.md`, and the
2026 generative-music landscape.

## 0. Context

The club set plays a music *library*: `lib/dj/` scans real tracks, beat-matches
them, and mounts the whole mix as one `AudioEngine` track. The goal is a
**continuous generative music** capability that never runs out, is steered by the
signals the DJ already publishes (night arc, energy, key, mood), and drives the
visuals the way the DJ does.

One design decision dominates everything else, and it comes from the operator's
own ear, on record in `docs/DJ_README.md`:

- the deck-C loop layer was shelved because "the bed loop isn't clean... not
  particularly impressive" (README 245-289);
- synthesized one-shots over tracks "read as a cheap sample pasted on the song"
  (README 386-398);
- "a crowd moment must change the music; everything else is a wet fart"
  (README 399-408).

**Therefore generative material must be first-class music the DJ selects and mixes
at seams — never a layer pasted over records.** The plan below is built on that.

---

## 1. What the codebase already gives us

### Hard constraints
| Constraint | Where | Consequence |
|---|---|---|
| Show box = Intel N150, 4 cores, **no GPU** | `config.yaml` `dj:` block, `bin/ensure_cpu_performance.sh` | No neural generation on the show box. Numpy/scipy DSP is fine (DJ FX run at ~2% CPU). |
| **44100 Hz** mandatory while the DJ runs | `Stories_OGL.py:2860-2868`; every `lib/dj/*` hardcodes `RATE=44100` | All generated audio must arrive at (or be resampled to) 44.1 kHz stereo float32. |
| `numpy==2.2.4`, `scipy==1.16.2` hard-pinned; no torch/librosa/mido in base | `requirements.txt`; torch stacks live in `requirements-dj-*.txt` and separate installs (`lib/dj/analyze.py:45-84`) | Neural generation runs on **another machine or a separate venv process**; the show process never imports torch. Hand-off is by files. |
| Audio render thread is the deadline; measured 43% of a core during a keylock blend | `lib/audio_engine.py:457-485` | Never generate or resample full tracks in-process while music plays. Pre-render off-thread / in a nice-19 subprocess (`lib/dj/preflight.py:116-137`). |
| Show is autonomous; MIDI deliberately not wired in | `Stories_OGL.py:417-422` | Generation is self-driving from outstate; operator control via the web panel only. |
| SQLite is WAL and must be written locally, never over a network mount | `lib/dj/db.py:195`, `analyze.py:62-67` | A remote GPU box ships WAV + manifest; a show-side ingest process writes the DB. |

### Integration seams to reuse
| Seam | File | Use |
|---|---|---|
| Track protocol `read(n)->(n,2) f32`, `.done`, `fade_out`, mounted by `AudioEngine.attach_track` | `lib/audio_engine.py:755`, mixer `:838-925`, 1.2 s render-ahead ring `:59-62` | Anything that produces blocks is "just a track". |
| `DJSubmix` — decks a/b/c, sample-accurate automation (`post_many`), PLL sync, `fx_play`, `duck`, `mix_gain`, telemetry | `lib/dj/submix.py:46` | Home for a stream deck (Path 2). |
| Non-library audio through the *real* brain: Beatport preview "ghost" tracks; fully synthetic tracks with hermetic grids run end-to-end | `tools/dj/planner/discover.py`; `tools/tests/_dj_brain_test.py:264 synth_structured`, `e2e_test:301` | **The established pattern for Path 1.** |
| Brain contract: `TrackInfo` (`brain.py:262`), `load_library:590`, `choose_next:2287`, `best_pair:2470`, `plan_transition:3044`, `build_events:4645` | `lib/dj/brain.py` | What a generated track must satisfy to get the full transition-style menu. |
| Commission signals: `DJSystem.arc_target:668`, `bpm_target:611`, `_arc_base:651`, `_maybe_horizon:1416` (arc at +300 s steps), `theme.prefer_tags/bpm_range` | `lib/dj/system.py`, `themes.py` | Tell the generator *what to make next*. |
| Ground truth → visuals: `live_beat:747`, `outstate_keys:791`, `DJVisualCoupler` | `lib/dj/system.py`, `lib/dj/vis.py` | Generated tracks carry exact grids, so visuals get better-than-analyzed truth. |
| Internal analyzer tap (`set_monitor_tap`, `audio_source: internal`) | `Stories_OGL.py:2319`, `lib/audio_analyzer.py:443` | Spectrum for shaders comes free. |
| Web `/dj` actions whitelist `DJ_ACTIONS`, `POST /api/dj/action`; declarative `INTERACTION_PANELS` | `web/web_controller.py:1150`, `lib/interaction.py` | Operator controls without new plumbing. |
| Offline verification: hand-pumped engine e2e, `_dj_night_sim.py`, `seamverify.measured_kick_alignment`, `_dj_quality_test.render_seam`, Seam Lab ratings | `tools/tests/`, `lib/dj/seamverify.py:237`, `docs/DJ_VERIFICATION.md` | Every phase ends with a rendered gate. |

---

## 2. Generation options

| Option | On N150? | Beat-locked to DJ | Sound quality | Autonomy | Risk | Role |
|---|---|---|---|---|---|---|
| **A. Commissioned offline neural tracks** (Stable Audio Open 1.5 / ACE-Step / Magenta RT2 offline) on a GPU box, ingested as first-class library tracks | ✅ (gen elsewhere) | ✅ exact — ground-truth grid | ●●●● | ✅ | low-medium | **Primary** |
| **B. Google Lyria RealTime** cloud stream (48k PCM, steerable bpm/scale/density/brightness; bpm/scale change = hard cut via `reset_context`; experimental, currently free) | ✅ (network) | ⚠ measured, not given | ●●●● | ✅ | medium (experimental API, venue internet) | Station mode |
| **C. Magenta RealTime 2** live stream (Apache-2.0, 230M/2.4B; real-time officially on Apple Silicon; ~2× real-time `mrt2_small` on NVIDIA/Linux via community JAX port) | ❌ needs 2nd box | ⚠ measured | ●●●● | ✅ | high | Local alternative to B |
| **D. Procedural numpy/scipy synthesis** in-process (pattern grammars + synth voices) | ✅ | ✅ exact | ●●○ | ✅ | low | CPU fallback + hermetic test fixture |
| E. SuperCollider/Pure Data via OSC + loopback | ✅ | ⚠ via loopback | ●●● | ✅ | medium (2nd process, JACK, bypasses engine) | not recommended |
| F. Loop-layer beds (deck C) with generated percussion | ✅ | ✅ | — | ✅ | **rejected by operator verdict** | only if real DJ tools land in `media/loops/` first |

Why A is primary: it needs **no real-time generation at all**. The next pick is locked
~20 s into the current record (`system.py:1577-1587`), so a generator only has to stay
two horizon slots (~10–15 min) ahead of the arc; 10–60 s neural rendering fits with
margin. The DJ then mixes generated tracks with its proven machinery, and because the
generator *knows* its BPM, grid, downbeats, key and structure, every confidence gate
that limits real records (`bpm_conf`, `downbeat_conf`, phrase, mix points) is cleared
by ground truth. Quality is judged in the same Seam Lab / night-census flow as records.

---

## 3. Recommended architecture

Two delivery paths behind one commission/ingest contract, plus a procedural fallback.

```
  DJSystem (night arc, bpm_target, key of horizon tail, prefer_tags, dryness)
        │  commission JSON  (target bpm, camelot set, energy, duration, structure template)
        ▼
  lib/dj/gen/commission.py  ── files ──▶  GPU box worker  (Stable Audio / ACE-Step / MRT2 offline)
                                                │  WAV (44.1k) + manifest (ground-truth grid, key,
                                                │  sections, mix points, model, commission id)
                                                ▼
  lib/dj/gen/ingest.py  (show box, nice-19 subprocess): verify tempo/key against the audio,
      measure loudness/kick_offset/energy/chroma/rhythm/beatpower, write DB row
      (provenance='generated', auto_tag "generated", unique title) — atomic rename first
        │
        ▼
  DJSystem.step 10 s poll → Brain.add_tracks(...)  (hot-add; no restart)
        │
        ▼
  Brain selects / plans / mixes it like any record → DJSubmix → AudioEngine → speakers
                                                    └─▶ live_beat / outstate → club director
```

Path 2 (station mode, Lyria RT or MRT2 stream) adds a `StreamDeck(Deck)` inside
`DJSubmix` (`decks["s"]`), fed by a feeder thread that resamples 48k→44.1k off the
render thread into a ~10 s ring; always sync **master** (a ring cannot be reseeked);
represented to the brain as a virtual `StationTrack` TrackInfo (synthetic grid, rolling
groove section, growing energy curve) so `plan_transition` plans record→station and
station→record hand-offs with the normal blend/bass-swap/filter-sweep vocabulary. Stream
health watchdog + a pre-decoded escape record are mandatory.

### 3.1 Path 1 details (from the design memo; file:line cited)

**Write from ground truth:** `bpm`, `beat_grid` (one segment, `period_s=60/bpm`,
`first_beat_s`), `bpm_conf=1.0`, `downbeat_offset`, `downbeat_conf=1.0`,
`phrase_beats/_start_s/_conf`, `camelot/key_pc/key_mode`, section boundaries + kinds
(intro/build/groove/breakdown/drop/outro), `structure` labels (`db.set_structure:417`),
`mix_points` (in = end of intro, out = start of outro, `style_hint: blend`), `duration_s`,
`axes.vocal=0` with `vocal_src="ground_truth"` (survives rescans via `scan.py:275`).

**Still measure from the audio** (generators do not obey prompts exactly):
`loudness_gain_db` (`features.py:1143`), `kick_offset_s` (`:1089`, with the true grid),
`energy_curve`/`band_curve` (`:1158/:1170`), `spectral`, `chroma` (`:628`) — and **reject**
if `chroma_key_compat` < ~0.8 or `verify_tempo_window` (`:191`) deviates > 0.5 %,
otherwise the live tempo write-back (`system.py:1248-1277`, `db.py:467-481`) would
silently "correct" the track and drop its beat-power record; per-section stats
(`build_sections:730`), `rhythm_signature` (`:164`), `classify_axes` (`:990`), and the
`beat_power.json` entry (`beatpower.py:325/412/482`, picked up live by mtime cache).

**Schema v16** (`db.py:198-266` ALTER-if-missing pattern): `tracks.provenance TEXT`
(NULL = library, `'generated'`), `tracks.gen_meta TEXT` JSON (model, commission id,
targets, created_at, expires_at). `bpm_source` is the wrong home (overwritten by
`set_verified_tempo`). Mirror as `auto_tags` `"generated"` so the web music-type chips
filter it with zero new code; expose `TrackInfo.is_generated`.

**Traps:** titles must be unique (`ckey` at `brain.py:1271-1285` collapses same-title
tracks into one "song" and the no-repeat wall blocks them all); set `content_hash` from
`scan.quick_hash`; `scan_library` needs a skip for generated rows (or re-ingest from the
manifest) so `--force` never replaces ground truth with estimates; write
`analysis_version = features.ANALYSIS_VERSION`; never DELETE rows (play_history cascades,
`db.py:123`) — set `missing=1`/`excluded=1` on expiry; promote thumbed-up generated tracks
(`seam_feedback.up=1`) to permanent by clearing `expires_at`.

**Hot-add (no runtime reload exists today):** producer writes WAV to a temp name and
renames; ingest writes the row; `DJSystem.step` polls `max(id) WHERE provenance='generated'`
on the existing 10 s `_refresh_tags` cadence (`system.py:1281`); hydrate with the one-track
constructor pattern (`preflight._load_track`, `preflight.py:194-202`); new `Brain.add_tracks`
appends to `library`, extends `ckey`, ranks the newcomer by bisecting existing sorted arrays
(`brain.py:606-667`), stamps `has_stems`; leaves `adapt_theme`/`norepeat_n` frozen; DJSystem
adds to `_by_id` and clears `_horizon` so `_maybe_horizon` reconsiders.

**Commission contract:** `{target_bpm: bpm_target at slot k, camelot: neighbours of the
horizon tail's key (compat ≥ 0.9), energy: arc_at(k), duration: 240–360 s, structure:
32-bar drums-only intro / groove / breakdown / drop / 32-bar outro, tags: theme.prefer_tags,
persona/flavor}`; keep ≥ 2 finished tracks ahead; raise urgency on `_horizon_dry`
(`system.py:1465`) / small `eligible_pool_size` (`brain.py:2258`).

### 3.2 Path 2 details (station mode)
`StreamDeck` overrides `load` (accept ring), `_fetch` (`deck.py:401`), `source_time_s`
(`:319`); keeps `samples` non-None so `read`'s guard passes (`:494`); `cue/set_loop/jump_cut/
brake/phase_snap/nudge` become logged no-ops; `_apply("sync")` (`submix.py:195`) refuses a
StreamDeck as slave; rate 1.0 with varispeed passthrough (no keylock cost). Visual truth:
`StationTrack` gets a synthetic grid once beat phase is **measured** on the first ~20 s
(`verify_tempo_window` / `estimate_beat_grid:401`), re-measured after every
`reset_context`; until then `bpm_conf=0` so `live_beat` honestly returns None. Commanded
density/brightness jumps stamp `_dj_drop_wall/_dj_drop_hard` so `DJVisualCoupler` fires an
engineered drop. Lyria bpm/scale are commissioned from the live `out_bpm` before going on
air and held fixed while audible; records stretch to the stream. `system.py` needs the a/b
flip generalised (`:1703`, `:1952`) so `active_deck` may be `"s"`, a station slot length in
`_draw_exit` (`:2937`), and a stream-health watchdog that calls `_emergency_handoff`
(`:2646`) to a pre-decoded escape record.

### 3.3 Code layout
- `lib/dj/gen/` — `spec.py` (commission + manifest schema), `commission.py`, `ingest.py`,
  `procedural.py` (grow `synth_structured` into intro/groove/break/drop/outro with kick,
  key-rooted bass, in-key pad — both the CPU fallback and the hermetic fixture),
  `station.py` (StreamDeck, StationTrack, feeder/resampler, `FakeStreamSource` replaying a
  48k WAV), `backends/lyria.py`, `backends/remote.py` (imported only when
  `dj.generative` config enables them).
- Brain/system/db deltas kept minimal: `Brain.add_tracks`, `TrackInfo.is_generated`,
  DB v16, ingest poll, a/b generalisation.
- `tools/dj/dj_gen.py` (`commission`, `ingest`, `procedural --n`, `sweep`) modelled on
  `dj_scan.py`. Worker box: `tools/dj/gen_worker/` (its own venv/requirements, torch).
- Tests: `tools/tests/_dj_gen_ingest_test.py`, `_dj_gen_hotadd_test.py`,
  `_dj_station_test.py`; `_dj_night_sim.py --generated K`; Seam Lab ratings filtered by
  provenance (per `docs/DJ_VERIFICATION.md` rule 2).
- `docs/GENERATIVE_MUSIC.md` (user doc, written with Phase 2).

---

## 4. Hardware: a low-power Linux GPU box for the worker

The worker does **offline** generation for Path 1 (and optionally MRT2 streaming for
Path 2). It lives on the venue LAN and ships files to the show box.

| Option | Power | Cost (approx.) | Software path | Verdict |
|---|---|---|---|---|
| **SFF x86 + low-profile NVIDIA card** — RTX A2000 12 GB (70 W, single-slot, no aux power) or RTX 3050 6 GB LP (70 W) or RTX 4060 LP (115 W); host = used Dell OptiPlex / Lenovo ThinkCentre SFF or a 1-slot mini-ITX build | ~20 W idle, 120–170 W load | $150–250 host + $180 (3050) / $400–500 (A2000, used) / $300 (4060 LP) | Standard Ubuntu + CUDA; PyTorch **and** JAX both first-class. The only path the MRT2 community CUDA port and every batch model actually exercise. 12 GB runs `mrt2_base` offline, Stable Audio Open 1.5, ACE-Step. | **Recommended.** A2000 12 GB if VRAM matters; 4060 LP if new and cheap matters. |
| **NVIDIA Jetson Orin Nano Super** dev kit | 7–25 W | ~$249 | Ubuntu (JetPack), PyTorch supported; aarch64 — JAX/CUDA is *not* a supported target, 8 GB shared with the OS | Lowest power and cheapest, and fine for slow batch generation with PyTorch models. Not for MRT2 real time; VRAM-tight for 2.4B models. Choose only if power is the overriding constraint. |
| Mini-PC with mobile RTX 4060/4070 (laptop-GPU class) | 100–150 W | $800–1200 | Ubuntu works on most; verify vendor Linux support before buying | Compact, more expensive per FLOP than option 1. |
| Mac mini M4 | 5–30 W | ~$599 | macOS only — but the **officially supported** MRT2 real-time path | Not Linux. Mention only because it is the cheapest way to get MRT2 streaming in real time if Linux is negotiable. |

Notes: Path 1 does not need real-time performance; a 70 W card rendering a 5-minute
track in 30–90 s keeps the DJ's horizon full with hours of slack. Cloud (Lyria RT) needs
no box at all but needs reliable venue internet and accepts experimental-API risk.

---

## 5. Phased roadmap

**Phase 0 — Spikes (1–2 sessions, no engine changes)**
- Extend `synth_structured` into a procedural 4-minute track with a manifest; run it
  through `e2e_test` and a rendered seam; listen. Establishes the manifest schema and the
  ingest contract with a zero-cost generator.
- On any GPU machine (or Colab), generate three commissioned tracks with Stable Audio
  Open 1.5 and/or ACE-Step at a fixed BPM/key; measure how far they miss the commission
  (tempo, key, structure) to size the ingest rejection thresholds.
- If a Gemini key is available: standalone Lyria RT client; measure latency, jitter,
  and the hard cut on `reset_context`; decide whether station mode is worth Phase 4.

**Phase 1 — Ingest + hot-add (engine)**
- DB v16, `lib/dj/gen/spec.py` + `ingest.py`, `Brain.add_tracks`, DJSystem poll, scan skip,
  expiry sweep, `dj_gen.py ingest/sweep`. Gates: `_dj_gen_ingest_test`, `_dj_gen_hotadd_test`.
- Deliverable: drop a WAV + manifest into `music/generated/` while the DJ plays and hear
  the brain mix it in with a synced style.

**Phase 2 — Commissioning + worker**
- `commission.py` reading arc/bpm/key/tags/dryness; `gen_worker` on the GPU box
  (file-based job queue in the music root, nice-19, resumable, like `preflight.launch_shadow`);
  procedural backend as always-on fallback; web `/dj` chip "generated: only / avoid / off"
  (free via the auto_tag) and a status line ("2 generated ahead, next in 3:10").
- Gates: night census with `--generated K`; Seam Lab session rating generated↔library seams.
- Deliverable: a night that never runs dry and stays on-arc even with a thin library.

**Phase 3 — Quality loop**
- Feed Seam Lab / thumbs verdicts back into commissions (which model, prompt vocabulary,
  structure templates win); promote thumbed-up tracks to permanent; `dj_review` provenance
  breakdown; planner Library tab provenance column/filter.

**Phase 4 — Station mode (stream)**
- `StreamDeck`/`StationTrack`, feeder + resampler, a/b generalisation, watchdog + escape
  record, `FakeStreamSource` gate `_dj_station_test`. Lyria backend first (no hardware),
  MRT2-on-LAN backend second, same interface. Operator controls: station on/off, prompt
  presets per theme, density/brightness sliders.

**Phase 5 (optional)**
- Real DJ-tool loops in `media/loops/` and, only then, revisit deck-C beds carved
  (HP 200–300 Hz + sidechain) per the README's revival notes. Generated percussion is a
  candidate source *only* after real tools have proven the mechanism.

Each phase ends with an offline rendered gate plus a live night reviewed via
`dj_review.py`, matching `docs/DJ_VERIFICATION.md`.

---

## 6. Open decisions for the operator
1. **Worker hardware**: buy the SFF + low-profile NVIDIA box (recommended) vs Jetson vs
   cloud-only (Lyria RT)? Phases 1 and 0-procedural need none of them.
2. **Cloud OK?** Is a Gemini API dependency (and venue internet) acceptable for station
   mode, or must everything be local?
3. **Genre envelope** for commissions: the club set's themes (`groove`, `peak_heavy`,
   `chill_evening`, `wind_down`) map to prompt vocabularies — confirm the target styles.
4. **Retention**: expire generated tracks after N nights unless thumbed up (proposed), or
   keep everything?
