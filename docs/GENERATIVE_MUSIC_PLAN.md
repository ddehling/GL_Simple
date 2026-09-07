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
| **Visuals coupling + soak** (2026-09-06): `GenVisualCoupler` maps the composer's ground truth (energy, drop foreknowledge, drop stamps, beat/bar/phrase, pulse floor only with a kick) onto the shaders' reactive keys, applied in `Stories_OGL` after the DJ coupler; `_gen_long_run_sim.py` is the all-night offline soak with operator moves (its first run caught a style-swap crash in the rack, now fixed and hardened) | `lib/gen/vis.py`, `Stories_OGL.py`, `tools/tests/_gen_long_run_sim.py` | `tools/tests/_gen_vis_test.py`, soak run |
| **Soak result** (2026-09-06, 4 h groove/ambient/downtempo with a move every 10 min, arc 40 min): 22x realtime, 972 phrases, 0 errors, 0 bad blocks, peak 0.93, 6 movements as expected, keys 8A→9A→9B→7B→6B, all 8 sections visited; RSS 101 MB → 331 MB in the first hour (JIT + Strudel engine warm-up) then **flat at 341 MB** for the remaining 3 h; pending notes ≤ 457, active voices ≤ 37, motif memory ≤ 12 | `tools/tests/_gen_long_run_sim.py --hours 4 --set-length 2400` | — |
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

Listening session 1 (2026-09-06, operator verdict: "a terrible MIDI keyboard") led to a
sound-design pass: hats/perc/arp/lead had sat 25-35 dB under the kick with the pad owning
the midrange, the image was mono (L/R correlation 0.996), the voices were static presets
and the composer played exact grid steps at flat velocities. Now: rebalanced slot gains,
detuned oscillators split L/R plus a per-note pan spread (correlation 0.90), per-note
pitch drift / filter LFOs / key-tracked cutoff / late vibrato, a distinct `keys` voice,
per-slot micro-timing + strums + 303-style bass slides + metric accents and articulation
(`Composer.humanize`), and a peak limiter instead of a hard tanh on the master. Notes are
now synthesised at `schedule()` time on the conductor thread, so the audio thread only
mixes (render-thread max 1.7 ms per 23 ms block, was 90 ms; a JIT warm-up runs at start
so the first note no longer stalls the ring).

Listening session 2 pass (2026-09-06, "do them all"), gated by `tools/tests/_gen_music_test.py`:
- **Rack**: 909 kick (oscillator-driven, choked on retrigger) + 909 clap, tom/rim/ride/shaker
  voices; trapezoidal-integrator SVF with input drive (`dsp.svf_tpt`, resonance up to 0.45 fs)
  replaces the Chamberlin; FM (2-op), supersaw and one-shot sample voices; transition material
  (`fx` slot: riser, reverse cymbal, impact, sweep) scheduled from the form's drop foreknowledge;
  per-bus chain (drum compressor, bass saturation, music high-pass, master high shelf) ahead of the
  limiter; every pitched voice parameter overridable from the style patch (`_Subtractive.TUNABLE`);
  **auto gain staging** (`SynthRack.calibrate`, targets in `rack.SLOT_TARGET_DB`) so "gain" means the
  same loudness whichever voice class sits in a slot.
- **Composer**: borrowed chords, suspensions, bass pedals and slow harmonic rhythm per section
  (`harmony.py`, chords are now `(degree, label, spec)`); a **theme** made in the build and restated
  on the drop, retired on a key change; bass avoids the kick's steps and hats/shaker thin under a
  busy bass; keys answer the lead in its gaps; per-slot groove templates (`style["feel"]`: swing
  multiplier + push).
- **System**: `tools/gen/listen.py` renders style x section excerpts (and full sets) with mix
  numbers and `--compare` deltas for A/B between passes. Render thread unchanged at <= 1.5 ms per
  23 ms block. Groove drop after the pass: L/R correlation 0.86, crest 13 dB, 8-16 kHz within 6 dB
  of the low end (was 21 dB under before session 1).
- Not done on this box: the long-run soak (`_gen_long_run_sim.py` imports the Linux-only `resource`).

Listening session 3 pass (2026-09-06, "do them all", round 2), gated by `tools/tests/_gen_mix_test.py`:
- **Mix automation lanes** (`rack.LANES`: hp, lp, duck, verb, delay_fb): the composer writes `auto`
  events per section (`composer.AUTO`, style overrides in `style["auto"]`) - the high-pass climbs
  through a build and snaps open on the drop, the low-pass closes breaks and outros, sidechain depth
  and reverb amount follow the section; the director/console/MIDI can move lanes too (`lane` action).
- **Space**: modulated 8-line FDN reverb behind diffusion allpasses (`fx.FDNReverb`, per-style
  `reverb_decay`) replaces Freeverb; a stereo chorus send (`send_chorus`).
- **Layers**: `patch["layers"]` stacks patches under one note with per-layer gain and hp/lp
  crossover (groove: sub + top bass, body + click kick; dnb: sub + reese; hiphop: 808 + sampled click).
- **Karplus-Strong** pluck (`ks` voice) for guitar/harp/kalimba colour (hiphop lead/arp, ambient arp).
- **One-shot library**: `oneshots:<name>` sample refs resolved through `media/oneshots/manifest.json`
  (or the project's `media/gen/oneshots/`); `tools/gen/oneshots.py scan|bootstrap|list` (the
  bootstrap renders a starter set from the rack's own voices).
- **Loudness + true-peak**: K-weighted-ish meter with a slow makeup gain toward `style["target_lufs"]`
  (+-8 dB), a lookahead limiter (64 samples) that never exceeds 0.95 - the tanh safety clip is gone.
- **Polyphony**: per-slot caps with oldest-note stealing (`rack.POLY`), total cap 96.
- **Melody with harmonic function**: strong beats snap to chord tones, the note before a strong beat
  approaches by scale step, a contour (arch/rise/fall/wave) is drawn with each motif; pad voicings
  are voice-led (least movement from the previous voicing).
- **Rhythmic language**: `style["drums"]` four | broken | breakbeat (a break library) | halftime,
  `halftime_in` sections, a shared fill library (kick/snare/toms play the same fill), 12-step perc
  cells rolling 3-against-4, builds fill the hats.
- **Style morphing**: `rack.set_style(..., morph=)` glides slot gains over 8 bars on a style change.
- **Four new styles**: techno, trance, dnb, hiphop (7 total).
- **Taste loop**: liked sections weigh more in the form grammar (`PreferenceMemory.section_bias`,
  `Form.taste`), a thumbs-up boosts the motif that was playing (`Melody.like`).
- **MIDI surface**: console plugin `tools/gen/console/plugins/midi.py` maps the nanoKONTROL2 to
  steering, mutes, gestures, feedback, lanes and transport through the action whitelist
  (new actions: `humanize`, `lane`, `automation`).
- Numbers (groove drop, 64 s): render thread max 3.0 ms per 23 ms block; L/R correlation 0.81;
  breaks 0.2-0.7; every style holds -12 to -14 LUFS-ish (ambient -20 target) with peak 0.95.
  `tools/gen/listen.py --sections drop,break,flow` is the per-pass A/B.

Listening session 4 pass (2026-09-06, real instruments + melodies + timeline), gated by
`tools/tests/_gen_vst_test.py` and `tools/tests/_gen_melody_test.py`:
- **Hosted instruments and effects** (`lib/gen/synth/plugins.py`, VST3 through pedalboard, verified
  on Windows with Dexed at ~50-80x realtime, deterministic): a slot patch `{"voice": "vst",
  "plugin": "vst:<name>", "program"/"preset"/"params"}`; the rack renders a hosted slot PER PHRASE
  (all its notes in one call plus a release tail - instruments are stateful and pedalboard does not
  carry state between calls) on the conductor thread, latency-compensated, calibrated like any
  voice; a style's `"vst"` overlay switches slots to plugins only where the binary exists (the
  analog patch stays as the fallback), `"bus_fx"` hosts effects per bus. Manifest
  `media/plugins/plugins.json` (binaries per machine, gitignored); `tools/gen/plugins.py
  scan|list|test|programs`. Groove uses Dexed for keys/lead/pad when present. Pi stays analog.
- **Melodies**: three sources, best first - AUTHORED hooks (`lib/gen/composer/hooks.py`: the Claude
  CLI writes 2-bar hooks + answer phrases for the style/key in a background thread, validated,
  cached in `logs/gen_hooks.json`; GEN_HOOKS=0 turns it off, the gates run with it off), the CORPUS
  model (`tools/gen/melody_corpus.py` builds `lib/gen/composer/data/melody_model.json` from
  music21's public-domain scores: rhythm cells, an order-2 interval model by metric strength,
  cadence/start tables, contours; `melody_model.py` samples motifs with leap resolution and
  cadences), else the old walk. Development operators (sequence, fragment, augment, retrograde
  beside repeat/transpose/vary/invert), question-answer phrases (the authored answer or a
  developed one), section-final cadence on the tonic (held), a climax an octave up on the last
  phrase of a build; a like boosts the motif that played. Status reports the melody source.
- **Timeline** (console tab `tools/gen/console/plugins/timeline.py`, data `GenSystem.timeline()`
  riding in status for the remote console): what has played, what is composed ahead in the rack,
  and what the form knows beyond that (rest of the section hatched, the drop it counts down to,
  the most likely next sections with their weights, the energy arc), chords per bar, theme
  statements, key changes, automation moves, a phrase table.

Song analysis (2026-09-06, "ingest a song, recreate it in our language, score it"), gated by
`tools/tests/_gen_analysis_test.py`:
- **The song description language** is the system's own steering, written down: a SongScript
  (`lib/gen/script.py`, YAML) = style / bpm / key / seed + a sequence of sections with length and
  levers (energy, density, brightness, swing, layers, chords, lanes, key/bpm changes, hook), plus
  optional MATERIAL from the source (kit one-shots, placed vocal phrases). `Composer.load_script`
  follows one (scripted Form, plain scripted chords in Harmony, the hook as the theme, lanes into
  the automation), `script.render` plays one offline, `script.to_actions` compiles it to the
  whitelisted actions (style, key, bpm, section, energy, density, swing, brightness, mute, lane,
  end) - the literal command list that regenerates the song. New action `script` (a file path)
  makes the running show follow it.
- **Ingest** (`lib/gen/analysis/ingest.py`, on the DJ analysis in `lib/dj/features.py`): tempo,
  downbeats -> bar grid, key (KS estimate refined over all 24 keys by diatonic-triad fit),
  sections (DJ kinds -> generator sections, drop = loud after a break/build), energy / density /
  brightness / swing levers, layers from band shares, chords per bar from a per-bar FFT chroma
  with the bass note weighted as the root, style from tempo + kick/snare step pattern. Also the
  per-bar FEATURE track (energy dB, band shares, low/high onset density, chroma).
- **Reuse** (`lib/gen/analysis/reuse.py`, optional: torch + demucs, CUDA on this box): stems ->
  a drum kit of one-shots cut from the drum stem (the recreation's kick/snare/hat play them),
  vocal phrases chopped from the vocal stem and placed on the bar grid (a `vox` slot), the melodic
  stem transcribed (basic-pitch, else librosa pyin) into the most repeated two-bar cell = the hook.
- **Score** (`lib/gen/analysis/score.py`): per 4-bar window energy / spectrum / rhythm / harmony
  (0..100), global = local mean + structure (energy envelopes correlate) + tempo + key; the song
  against itself is 100; the faithful synth-only recreation of the example script scores ~78, a
  deliberately wrong one ~54.
- **Surfaces**: console tab Analysis (`tools/gen/console/plugins/analysis.py`: ingest with an
  optional stem reuse, editable section table, the command list, recreate, score strip with
  original vs recreation energy and a score bar per phrase, save, play to the show) and the CLI
  `tools/gen/analyze.py ingest|recreate|score|all|play [--reuse]` writing logs/analysis/<name>/.
- Limits: melody transcription is only as good as the stem + basic-pitch; chord detection on
  dense mixes is ~60% right per bar (the score uses chroma directly, not the chord labels);
  sections come from the DJ novelty segmentation (a long plateau is one section).

Analysis round 2 (2026-09-06, "how can we improve things" -> "do them all"):
- **Closed loop**: `lib/gen/analysis/tune.py` hill-climbs each section's levers (energy, density,
  brightness, swing, lp/hp lanes, layer toggles) against its local score, rendering only that
  section (first 16 bars); CLI `analyze.py tune`, tab button Tune; accepted moves are reported.
- **Scoring**: a 32-band spectral profile per bar and a TIMBRE term (profile correlation) in the
  local score; DTW alignment of the two energy envelopes (+-8 bars) before windows are compared.
  Faithful synth-only recreation of the example: 78 -> 83.
- **Chords from the bass**: with stems, the bass stem is transcribed (librosa pyin 30-300 Hz); its
  pitch class per bar roots the chord reader, and its per-phrase cell (16th onsets + degree
  offsets) becomes the section's scripted bass line (`section["bass"]`, Melody.bass_override).
- **More reuse**: the melodic stem is sliced at its strongest onsets into pitched tones
  (`script["bank"]`; keys/arp become multisample players), vocal phrases are time-stretched to a
  changed tempo (librosa, constant pitch; `bpm_src`), basic-pitch installed with `--no-deps` on
  the ONNX runtime (numpy 2 kept) for polyphonic hook transcription.
- **Learning**: `analyze.py batch <folder>` ingests a library; `analyze.py learn` derives
  per-style presets (tempo range, swing, section energies/lengths, layers, progressions) into
  `lib/gen/composer/data/learned_styles.json`, applied by `get_style` (GEN_LEARNED=0 to skip);
  scored recreations feed the taste memory (`PreferenceMemory.record_scores`).
- **Engine**: `listen.py --analysis <folder>` prints original vs recreation numbers; the system asks
  for a hook written over THIS build's chords when a build starts; a web `timeline_strip` widget
  (Log tab) draws the song strip from status, mirrored by a Qt widget of the same name.
- **The beat itself** (operator: "no evidence of the beat structure"): per bar and per section the
  onsets of the kick / snare / hat bands are folded onto the 16th grid (`ingest.bar_patterns`,
  `section_pattern`); each section carries `drums` (hits with velocities) and `drums_grid` (the
  evidence), the Analysis tab draws the selected section's grid under the score strip with the
  grid facts (tempo, beat length, first downbeat, swing, kind), the script describes it as
  `kick:x...x...x...x...`, the recreation's kit plays exactly those hits (`Drums.override`), and
  the rhythm term of the score compares the patterns, not just onset counts (example 83 -> 88).
- **Play and compare** (operator): the Analysis tab plays the original and the recreation
  (`tools/gen/console/player.py`, sounddevice, independent of the show engine) with A/B switching
  that keeps the position, seek by clicking, Space to pause; a `CompareView` stacks
  high-resolution log-frequency spectrograms (`lib/gen/analysis/spectro.py`: 256 log bins
  30 Hz-16 kHz, 86 fps, cached as .spec.npz) on one time axis with the recreation shifted so its
  bar 0 sits under the original's first downbeat, bar ticks, section markers, the play cursor,
  wheel zoom and drag scroll.
- **Source loops = the recreation that matches** (operator: "the songs just don't match at all"): a
  generator playing its own synths over the song's key, chords and beat never sounds like the
  record. With stems, every section now gets a representative 4-bar loop per stem (the phrase
  nearest the section's median energy, `reuse.section_loops`), the script carries them
  (`section["loops"]`) and a `fidelity` dial: 0 generator only, 0.5 drum + bass loops under
  generated melodic layers, 1 every loop (drums / bass / other / vocals on `loop_*` slots straight
  to the fx bus, the generator only adds transitions and the hook; master shelf and loudness
  target off). Ananta Groove: synth-only 82 -> stems 85 -> loops 89.6, spectrum within 2-3 dB of the
  original. The Analysis tab has the "source material" slider; the tab ticks "reuse stems" by
  default when demucs is installed.
- **Notes as samples, played PROGRAMMATICALLY** (operator: "extract the notes to use as samples",
  then "not sheet-music transcription - generate the pattern with our mechanisms so we can adjust on
  the fly"): the melodic stem is transcribed (basic-pitch) and every note's audio is cut into a
  multisample keyed by pitch (`reuse.note_samples`, up to 24 pitches; the bass stem the same way
  via pyin); the transcribed lines are kept as EVIDENCE only. What plays is generated: the line's
  two-bar cells become `script["motifs"]` (chord-relative degrees, recurrence counts) that seed the
  composer's motif memory and theme (`Composer.load_script`), so the lead DEVELOPS the song's cells
  (repeat / transpose / vary / invert / sequence / fragment / augment / retrograde) with harmonic
  function on the scripted chords; the bass stem's cells become `script["bass_cells"]`, the library
  the bass generator draws from (kick avoidance, slides, energy thinning intact); the section's
  drum grid is a TEMPLATE the kit varies (strong hits always, weaker by strength x energy x density);
  lead / keys / arp / bass play through the song's own note samples. Density and energy thin the
  lead by a hard cap (120 -> 44 -> 27 notes per minute). Fidelity default is now 0 (programmatic);
  the slider adds the source loops as a reference up to 1.0.
- **The ceiling, measured** (Ananta Groove, 212 bars, the scorer as it was then): self = 100;
  all-loops reference 91.1; drum + bass loops under generated layers 88.7; fully programmatic 85.1.
  Per-term loss programmatic vs loops (points of global): rhythm 1.9, harmony 1.4, energy 1.3,
  timbre 0.4, spectrum 0.3. Even the loops only reached 72 on the rhythm term, timbre was solved by
  the note samples (94 vs 98), harmony lost where the chord reader handed the generator wrong
  chords, energy lost the within-section dynamics. Two of those three diagnoses turned out to be
  partly the scorer's (next item).
- **Closing the gap (2026-09-06, after the ceiling diagnostics)** - three script fields and two
  scorer corrections:
  - `chords` is now one entry PER BAR of the section (cycled when shorter), each an int degree or
    `{"deg", "third": maj|min, "sus": 2|4}` (text form `5M`, `6s4`); the harmony spells an altered
    third / a suspension (`Harmony.scripted`). The reader (`ingest.read_chords`) picks the root by
    a soft vote (harmonic chroma fit + bass chroma at the root + transcribed bass note + a hold
    preference) and checks the quality on that root; with stems the harmonic and bass stems supply
    the chroma (`reuse.stem_chroma`). On the synthetic gate song: 38% of bars right (old mix reader)
    -> 59% (stems + quality); note the gate's beat tracker puts the first downbeat a bar in, so the
    test compares with a +-1 bar shift.
  - `drums_phrases`: one template per 4-bar phrase (`ingest.phrase_templates`) plus `fill` - the
    phrase's last bar when its kick/snare differ from the phrase mean; the kit plays the fill
    template on that bar (`Drums.bar`).
  - `dyn`: dB per bar relative to the section's mean, written bar by bar to a new `gain` mix lane
    (after loudness normalisation, before the limiter - the normaliser holds the long-term level and
    must not undo phrasing); a louder bar also nudges the energy lever (+6 dB ~ +0.12).
    `level`: the section's level vs the song's mean; `script.render` CALIBRATES when levels are
    present - a first pass measures the recreation's own section levels and the difference is
    written to `trim_db` (saved with the script, so the next render is one pass).
  - Scorer corrections found on the way: (1) the bar chroma was a single long FFT from 55 Hz with
    1/f weighting, so on a dance record the KICK's fundamental read as the tonic on every bar - key
    detection said A# minor for a song that sits on G#, and the harmony term rewarded playing the
    kick's pitch; it is now the MEDIAN over 186 ms frames from 80 Hz (transients drop out;
    `ingest.bar_chroma`), and the key comes out G# major. (2) The rhythm term's density half counted
    onsets above a whole-track percentile, i.e. WHERE the loudest onsets fell, so even the song's
    own loops scored 0.35 on it; it now counts active 16th steps (kick; snare+hat) per second from
    the folded patterns (`ingest._busyness`). (3) `energy_db` is 10 log10 of a mean AMPLITUDE (the
    DJ bands are sqrt(power)), i.e. half-decibels; `dyn` / `level` / `trim_db` are real dB
    (`DB_PER_UNIT`). Numbers before and after are therefore not comparable; everything below is
    re-measured with the corrected scorer.
- **The ceiling, re-measured** (Ananta Groove, 212 bars, corrected scorer, GEN_STRUCTURE=0, three
  seeds each): self = 100; all-loops reference 92.0 (rhythm 83.5, harmony 98.9, energy 84.5); drum
  + bass loops under generated layers 91.9; programmatic with the old script (section levers, a
  4-bar chord loop) 87.1 (energy 82.6, rhythm 74.4, harmony 86.6, structure 88); programmatic with
  per-bar chords / drum templates / dynamics + level calibration 90.7 (energy 89.3, rhythm 75.4,
  harmony 87.3, structure 96). The programmatic version now beats the loops on ENERGY (the
  calibration; the loops are re-levelled by the rack's master chain) and is 1.3 global points from
  the all-loops reference. What remains: harmony 87 vs 99 (about 1.9 global points: one root per
  bar on a drone record whose colour is in the melodic stem; the generator's own voicings), rhythm
  75 vs 84 (about 0.8: the kit's own hats/percussion around the template, fills), spectrum 88 vs 90.
  The earlier ablation showed the three fields are worth ~0 alone under the old scorer on this
  song and 3.6 points together with the corrected one, most of it the dynamics + calibration.
- **Sound identification (2026-09-06, operator: "an order of magnitude better in note and sound
  identification")**. What was wrong, measured on Ananta: the "melody" was 799 basic-pitch notes that were
  the DRONE's harmonics at every octave (G#/D#), so the motifs were flat "-7 -7 -7" cells; the plucked
  onsets (1277, the actual tune) it mostly missed; the kit was three one-shots for a record with 2600
  percussion onsets; and - found on the way - the rack's plugin overlay replaced the lead / keys / pad
  patches with Dexed AFTER the script's material was applied, so the recreation never played the
  song's melodic samples at all. New module `lib/gen/analysis/sounds.py`:
  - Drums: every onset of the drum stem gets a timbre vector (attack + body mel spectra, decay,
    centroid), clustered (k-means, merge near-identical, drop tiny); templates that are a SUM of two
    sounds (a kick under a clap makes its own cluster) are explained by the others with NNLS and keep
    only their residual (or merge when they are the same kind of sound); fixed-template NMF over the
    whole stem gives each sound's activation per 6 ms, peak-picked and folded onto the 16th grid ->
    per bar, a strength grid PER SOUND (`drum_grids`). Each sound gets its most isolated hit as a
    one-shot, its level relative to the loudest (`kit_db` - sample slots are not calibrated by the
    rack, so the balance must come from the song), and a drum SLOT (kick = the low-band sound that
    plays the most, snare = the mid sound on the backbeat, hat = the busiest high one, the rest by
    register). Joint NMF refinement of the templates was tried and rejected: templates drift into the
    loudest band. Unsupervised NMF was tried: a kick's pitch sweep splits into four components.
  - The script carries the beat per identified sound: `drums` / `drums_phrases` for every slot the
    kit has, plus `drums_bars` - the identified beat bar by bar; the kit plays it as a template
    (hits above 0.4 of the sound's max always, weaker ones by strength) thinned or thickened only by
    the STEERING relative to the section's own levers (`Drums.steer`), so the song's fills and
    variation are there and the operator can still push it. Layers follow which sounds play.
  - Melodic: the "other" stem is split (HPSS) into the sustained part and the plucked part. Plucked
    onsets get a pitch by harmonic summation on the spectrum after the onset MINUS the spectrum
    before it (the drone subtracted; pyin on the transient part could not hear the tone), and a
    timbre vector; clustered -> up to three instruments, each with a multisample from its own cleanest
    hit per pitch and a note line -> lead (motifs), keys (`bank_keys`), arp. The sustained part's
    steadiest two bars become the pad sample, pitched by the chord ROOT sounding there (pyin heard
    the fifth) and played as a DRONE (one note on the root, the texture itself, `pad.drone`).
  - Mix: `mix_db` per-stem trims and the vocal chops on their own bus.
  - THE STRICT MEASURE (`score.stem_fidelity`): the phrase-level score said 88-91 for recreations
    the ear called bad. The recreation is now rendered with its four buses as exact stems
    (`render(stems=True)`, `SynthRack.capture`) and each is compared with the source's demucs stem on
    a 64-band log-mel spectrogram BEAT BY BEAT: mean |dB| of the beat spectra, the level difference
    (fed back into `mix_db`; `analyze.py score` / the tab's Score do it) and the activity correlation.
    Ananta, drums / bass / other / vocals, mean |dB| (activity r): before 8.8 / 4.1 (-6 dB level) /
    7.7 (+11 dB, and that was Dexed) / 1.5; after 6.4 (0.62) / 3.7 (0.59) / 8.5 (0.47, at the right
    level, playing the song's plucks) / 1.0 (0.90). The drum identification CEILING - the stem
    resynthesised from the identified sounds on the grid - is 3.9-4.1 dB (r 0.64-0.68): one exemplar
    per sound, no velocity layers, 16th quantisation. The generator path loses ~2.3 dB more.
  - Not solved: the melodic stem at 8.5 dB. The plucks are right but the LINES are generated from
    motifs (that is the brief); the pad is one texture; the second instrument plays chords not its
    line. Next: an exemplar per (pitch, dynamics), the keys slot following its own identified line
    as a second motif memory, and per-sound NMF for the plucks too.
- **"Absolutely unworking garbage" (operator, same evening) - what it actually was.** The tab's
  flows all complete (driven offscreen: ingest, recreate, score, play); the SOUND was broken, by
  four things the stem numbers had not shown:
  1. The recreation's mix peaked at 3.4 BEFORE the limiter: the rack's loudness normaliser pushed
     the (quiet) sample material up by its full +8 dB clamp and the peak limiter flattened every
     drum hit. Material scripts now run with no normaliser and no master shelf; `level_ref_db` (the
     source's RMS) and a measured `master_db` from the calibration pass put the recreation at the
     record's level as far as the peaks allow (at most 2 dB into the limiter; -13.4 vs -11.8 dBFS).
  2. The form's own transition FX: an 8-bar white-noise RISER into every drop, at the analog style's
     level, over material 10 dB quieter - 28,000 sample-to-sample jumps in one ten-second window.
     `script.fx` (the analyser writes False) keeps the form's risers / impacts / sweeps out of a
     recreation. Mix discontinuities: 67,634 -> 12 (the record itself: 4,616).
  3. Every kit one-shot carried the next two or three hits and every pluck sample several notes:
     isolation was measured against the same SOUND's onsets, not all onsets in the stem. Cuts now
     end before the next hit of any sound. Drums 6.4 -> 4.8 dB (activity 0.70).
  4. The 4-second pad texture went silent after 4 s of every held chord (the sample voice plays a
     file once); `loop` patches loop the body with crossfades for the note's length.
  Ananta after all four: drums 5.1 / bass 3.7 / other 9.6 / vocals 1.0 dB. One more measurement
  lesson: the per-stem numbers and the phrase score both missed a ten-second noise blast; a
  discontinuity count (|sample-to-sample jump| > 0.4) against the source's own count catches that
  class of defect in seconds, and the recreation is now checked that way in the gate.
- **Hosted instruments off the main thread**: pedalboard refuses `reset=True` outside the main
  thread; instruments render with `reset=False` and the previous batch's tail is flushed with
  silence first (the console error "Plugin ... must be reloaded on the main thread" was this).
- **allin1** (the DJ planner's structure model) does NOT run in this venv: it needs natten < 0.15,
  which has no build for torch 2.11 on Windows (the planner runs it under WSL). `ingest` tries it
  and falls back to the DJ segmentation silently (GEN_STRUCTURE=0 skips the attempt).

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
