# Programmatic Song Reconstruction — Plan

Goal: recreate a library song from a **reduced program** (patterns, chords,
voices, a sequence, a few verbatim slices), not from a list of every note,
and have it sound close. This plan is built on two measurements taken on
2026-09-08 over six varied library tracks (ids 1, 51, 101, 452, 552, 1152)
with the current reading (`lib/dj/instruments.py` v3) and renderer
(`lib/dj/resynth.py`).

## 1. Where we stand (measured)

`tools/tests/_dj_recon_eval.py` renders each stem's instruments from the
reading and compares them with the real stem (medians over the six tracks):

| stem | onset F1 | timing | spectral gap | chroma r | verdict |
|---|---|---|---|---|---|
| drums | 0.90 | +9 ms | 20 dB, level −2 dB | – | rhythm right, sound wrong (dry one-shots, no tails, no room, no velocity layers) |
| bass | 0.78 | 0 ms | 14 dB | 0.63 | roughly the line; a third of the notes wrong (kick bleed at the range floor, octave errors) |
| other | 0.20 | −3 ms | 11 dB | 0.53 | mostly wrong: polyphony, pads, arps and FX are read as one note per onset |
| vocals | (stem passthrough) | | | | no sample-and-notes model sings |
| mix | 0.66 | +12 ms | 8 dB | 0.62 | a rhythmic and harmonic sketch |

The second measurement is the one that decides the language. Encoding every
bar of every instrument as a pattern (step, pitch relative to the bar's
lowest note, duration, velocity class) and counting how many distinct
patterns cover 90% of the bars:

| instrument (track) | bars | exact patterns | steps only | similar (Jaccard ≥ 0.75) |
|---|---|---|---|---|
| hat (Sussudio) | 130 | 2 | 2 | 2 |
| kick (Sussudio) | 130 | 16 | 16 | 9 |
| kick (First Snow) | 116 | 22 | 22 | 13 |
| snare (Evolution) | 119 | 53 | 53 | 12 |
| pad (Sussudio) | 129 | 115 | 85 | 4 |
| pad (First Snow) | 120 | 94 | 61 | 11 |
| bass (Sussudio, 905 notes) | 128 | 110 | 63 | 28 |
| bass (First Snow) | 116 | 34 | 28 | 26 |
| pluck (First Snow) | 120 | 90 | 84 | 57 |
| pluck (Evolution) | 132 | 94 | 90 | 82 |
| sung line (Sussudio) | 127 | 54 | 39 | 20 |

Read across a row: the music IS repetitive (a handful of similar bars cover
a whole kick track or pad), but the exact encoding sees almost every bar as
new. The difference is **reading noise**: a spurious hit here, a missed
one there, a velocity class that flips, a pad chord read with a different
inversion each bar. Compressing the note list after the fact cannot work;
the repetition has to be recovered **in the analysis**, by fitting a small
pattern vocabulary to the noisy observations and classifying every
deviation as noise (dropped) or as a variation (kept as an op). That is
the central design decision of this plan.

## 2. The language: a SongProgram

One file per song, human-readable (YAML), rendered deterministically.

```
song:      title, bpm, grid (first downbeat, period), key, sections (from the DJ structure)
voices:    one per identified instrument, with a SOUND MODEL:
             kit sound    layers: 2-3 samples by velocity, each WITH its tail; a per-sound decay
             pluck        per-pitch samples (own recordings) or a fitted synth (osc + filter + env)
             pad          the chord sample (played as a unit) + loop points, or a fitted synth
             verbatim     a slice of a stem (vocals, FX, textures)
patterns:  per voice, a small vocabulary of 1/2/4-bar patterns:
             steps with velocity class, relative pitch (scale degrees against the bar's chord)
             a GROOVE template per voice: mean timing offset and velocity per step
chords:    a chord symbol per bar (or half-bar): root, quality, bass note
sequence:  per section, per bar: pattern ref + ops
             repeat, transpose(n), follow-chord, drop(step), add(step, vel), fill(pattern), mute
             arp(chord, pattern) for arpeggiated parts (pattern over chord tones)
verbatim:  slices of stems placed by bar, with reuse (a chorus phrase recorded once, placed 3 times)
automation: per section per voice: level, brightness (filter), send to the stem's measured room
mix:       per stem: measured room (decay fit -> a reverb send), sidechain ducking (envelope vs kick)
```

Size budget: tens of patterns and a few hundred sequence entries per song,
against the current 1,300 – 3,900 events. The evaluator gains a
"program size" number (events explained per pattern) so compression is
measured, not asserted.

## 3. Per stem: what works, what won't, what to do

### Drums — will work well
- **Works**: the patterns. Kick, snare and hat tracks reduce to 2 – 13
  similar bars per song; fills and breaks are a few extra patterns. Onset
  timing is already right (F1 0.90).
- **Works poorly today**: the sound. One dry exemplar per sound, cut at
  the next hit, so no tail, no room, no velocity layers; coincident hits
  smear the exemplar; ghost notes and rolls are read as noise.
- **Do**:
  1. Pattern inference with a noise model: cluster bars at Jaccard ≥ 0.75,
     take the cluster's consensus as the pattern, keep per-bar deviations
     only when they recur (fills at phrase ends) or are loud.
  2. Kit extraction with tails: the longest isolated instance per sound,
     decay extrapolated where the next hit cuts it; 2 – 3 velocity layers
     by clustering hit levels.
  3. Room: fit the stem's decay after isolated hits (RT, level) and render
     it as a send, not baked into samples.
  4. A groove template per sound (per-step offset and velocity) instead of
     per-hit offsets.
- **Expected**: spectral gap 20 → under 10 dB; the drum bus becomes
  a usable re-render, and it is where the language compresses most.

### Bass — will work well once the reading is cleaned
- **Works**: monophonic riffs that follow the chord; sub holds. Riff
  vocabularies of 10 – 30 bars per song at steps level; with pitch
  expressed relative to the chord root, most of those collapse further.
- **Works poorly today**: a third of the notes are wrong. Kick bleed in
  the bass stem reads as C1 at the range floor; octave errors; slides read
  as two notes; the busy funk bass (Sussudio, 905 notes) is real but still
  riff-based.
- **Do**:
  1. Cross-stem cleaning: subtract the drum stem's low band, time-aligned,
     before pitch reading (the kick is in both stems).
  2. Pitch with continuity: Viterbi over per-onset candidates with a
     chord prior, instead of independent per-onset picks.
  3. Riff inference relative to the chord root; transposition and
     follow-chord ops in the sequence; slides as a pitch-bend op.
  4. Sound: a fitted subtractive synth (oscillator mix, filter cutoff
     and envelope matched to the stem's spectrum per pitch) as the
     default voice, with per-pitch samples as the fallback.
- **Expected**: chroma 0.63 → above 0.85; spectral gap 14 → under 8 dB.

### Other (synths, keys, pads, FX) — mixed: pads well, polyphony badly
- **Works**: pads and sustained chords, once expressed as a chord track.
  The pad rows already reduce to 4 – 13 similar bars; a chord symbol per
  bar plus the song's own pad sample played as a unit is the right model.
  Arps are repetitive in reality and will reduce once read consistently.
- **Works poorly today**: everything polyphonic (keys, stacked synths):
  the reader takes one pitch per onset; pad voicings change bar to bar
  in the reading; FX, risers, textures and reverb tails have no note
  model at all (onset F1 0.20 here).
- **Do**:
  1. Real multi-pitch transcription for this stem (basic-pitch is
     installed; fuse its frame-level pitch activations with the onset
     reader and a chord prior), replacing the per-onset harmonic pick.
  2. A chord reader on the harmonic part (root, quality, bass) → the
     chord track; pads and keys become chord-following voices.
  3. Arp detection: a repeating step pattern over chord tones → one
     `arp(chord, pattern)` op per section, not a note list.
  4. An "unexplained residual" test per bar (spectral energy of the stem
     not accounted for by the rendered voices): above threshold, that
     bar's stem audio goes into the verbatim track. This is how FX,
     risers and textures are carried without pretending to transcribe
     them.
- **Expected**: chroma 0.53 → above 0.75 on pads and keys; FX carried
  verbatim; this stem stays the least compressible.

### Vocals — no programmatic model; reuse is the compression
- **Works**: nothing note-based. Sung lines were read but no sample model
  sings; the stem is passed through today.
- **Do**: phrase segmentation (silence and onset), similarity (chroma +
  MFCC DTW) to find repeats (choruses), one recording per distinct phrase
  placed wherever it recurs, with level per placement. That makes vocals
  a verbatim voice with reuse, which is honest and still programmatic at
  the phrase level.
- **Expected**: identical to the stem where phrases repeat exactly;
  slightly off where a repeat differs (ad-libs), which is the right place
  to keep a second recording.

## 4. Rendering (shared)
- Deterministic: sequence + patterns + groove template → events → voices.
- Velocity layers and tails on kit sounds; synth voices for bass where
  fitted; chord-as-unit pad playback; verbatim slices with fades.
- Per-stem room from measured decay; sidechain ducking measured from the
  bass/pad envelopes against the kick.
- The gen console's Replay becomes "render the SongProgram"; Recreate
  keeps feeding the composer the same material (kit, banks, chords,
  patterns) so variation starts from the program, not from the note list.

## 5. Validation gates (no listening required to reject a change)
`tools/tests/_dj_recon_eval.py` per stem, plus a program-size number:
- drums: spectral < 10 dB, onset F1 ≥ 0.9, level within 2 dB
- bass: chroma ≥ 0.85, spectral < 8 dB
- other: chroma ≥ 0.75 on pad/keys bars; verbatim share reported
- mix: spectral < 6 dB
- program size: ≥ 5 events explained per pattern entry (drums ≥ 10)
- the synthetic gate (`_dj_instruments_test.py`) stays green
Listening is the final judge; the numbers exist so tuning is not done on one song.

## 7. Implementation status (2026-09-08, same day)

Built, all gated by `tools/tests/_dj_instruments_test.py` (synthetic) and
`tools/tests/_dj_recon_eval.py [--program] <ids>` (six real tracks):

| piece | where | state |
|---|---|---|
| SongProgram: pattern inference with the noise model, per-item timing, `add`/`drop`/`nudge` ops, best unit (1/2/4 bars) per voice, groove template, chord-as-unit holds | `lib/dj/songprogram.py` | done; expansion reproduces the reading's events within a few ms; 6–11 events per pattern+op |
| Sound models: kit velocity layers with tails (decay extrapolated, capped 0.5 s), per-pitch samples with tails, measured drum room (decay fit, r² ≥ 0.8, RT ≤ 2.5 s) as a send | `lib/dj/voices.py` | done |
| Bass: kick-bleed subtraction from the drum stem's low band | `lib/dj/instruments.py` `clean_bass` | done (small net gain); the Viterbi continuity prior was built, measured to LOWER chroma on real bass lines (leaps are real), and is off (`CONTINUITY`) |
| Chord track (templates on beat-synchronous chroma, bass names the root, Viterbi stay cost) | `lib/dj/chords.py` | done; feeds the program and the gen script's per-section chords |
| Unexplained-residual test: `other` bars the voices leave > 14 dB unexplained are carried verbatim, share reported | `songprogram.residual_bars` | done |
| Multi-pitch for `other`: basic-pitch fused on the reader's onsets (hit-length notes join the hit as chord tones; the reader's held voices keep their activity and levels, the transcriber re-voices their pitches) | `instruments._poly_refine` (v4 readings) | done; bass tested and NOT applied (worse) |
| Vocal phrases with reuse (segmentation, log-mel+chroma shape similarity ≥ 0.82, one recording per distinct phrase) | `voices.phrase_library` | done: Sussudio 39 phrases / 9 distinct |
| Wiring: gen-link builds and saves the program, Replay renders it; planner "hear reconstruction" renders the program (fast build); `dj_instruments.py --program` | `lib/dj/gen_link.py`, planner, CLI | done |

**Where the program render stands against the gates** (`--program`, six tracks, medians; the day started at mix 9.3 dB / drums 20.5 dB on the raw reading):

| stem | spectral | onset F1 (raw-mode) | chroma | gate | status |
|---|---|---|---|---|---|
| drums | 13.5 dB (Sussudio 7.4) | 0.88 | – | < 10 dB | not yet; kit extraction still smears coincident hits, no per-hit timbre |
| bass | 13.0 dB | 0.52 | 0.56 (First Snow 0.82) | chroma ≥ 0.85 | not yet; the reading's pitch errors, not the renderer |
| other | 11.0 dB | 0.24 | 0.43 (Evolution 0.66) | chroma ≥ 0.75 | not yet; polyphony fused but hit pitches still per-onset |
| vocals | 11.3 dB (phrases) | – | 0.73 | – | phrase reuse works; identical where phrases repeat exactly |
| mix | 8.1 dB (Sussudio 5.8) | 0.73 | 0.55 | < 6 dB | one track passes |
| size | 6–15 events per pattern+op; verbatim share 3–74% of bars (Balance Perc Tool is mostly verbatim: a percussion tool the voices cannot explain) | | | ≥ 5 | met, but verbatim-heavy tracks are accurate by copying, not by program |

Next levers, in order of expected gain: kit extraction by subtraction (a
sound's template removed from coincident hits so each layer is one
sound, not a smear); a fitted synth voice for bass (the sample path has
hit its ceiling at ~13 dB); per-hit pitch verification against the stem
spectrum over the whole note (closed loop) for bass and plucks; a
polyphonic hit model for `other` (several onsets per step already
supported by the format); vocal phrase matching with time alignment
(DTW) so near-repeats reuse one recording with a warp instead of a
second recording.

**2026-09-09, the next four levers (all from the song's own recordings, no synthesis):**

| change | what it does | measured effect |
|---|---|---|
| kit subtraction (`voices.kit_sound(others=...)`) | every drum sound re-cut with the other sounds' samples subtracted at their event times, so a sound that never plays alone comes out alone with its real tail; the synthetic room send is off by default (`build(room=False)`) | neutral on the spectral gap on three tracks — the drum gap was not the coincident smear |
| velocity shares (reading v5) | a hit under another sound was rendered at the isolated hits' level (up to 8 dB quiet on Evolution); velocities now run to +12 dB above it and a coincidence's energy is split by each sound's NNLS share | see the v5 evaluation below |
| bass onsets on the full cleaned signal (not the percussive part) | a bass note's attack is mostly harmonic | chroma 0.82→0.87, 0.50→0.55, 0.30→0.69, 0.36→0.51 on four tracks; adopted |
| bass articulations (`voices.pitch_sounds`) | per pitch a short and a held recording from the song, the held one sustained by looping its own steady part | in place of the fitted-synth lever (rejected: not the song's instrument) |
| chord hits (`voices.chord_sounds`) | a pitched hit voice that strikes chords plays the song's own chord recordings, one per distinct chord shape, reused, transposed only when a shape appears nowhere | applies where the transcription found chord tones on hits |
| warped phrase matching (`voices._phrase_sim`, DTW) | near-repeat vocal phrases reuse one recording | Sussudio 9→6 distinct recordings for 39 phrases; Evolution 10→8 |

**2026-09-09, second half — what the synthetic gate taught (notes read 100% right, so only the renderer is measured):**

- The spectral measure was counting near-silent cells; an additive render with exact level and chroma scored "13 dB". It now weighs cells within 40 dB of the beat's loudest (`_dj_recon_eval._compare`). All numbers after this point are on that measure.
- Renderer ceiling on the synthetic (program): bass 2.2 dB, other 7.6 dB (was 17.5 with samples), drums 9.4 dB.
- **Additive instrument model** (`lib/dj/additive.py`): per-harmonic attack matrix, decay per harmonic, release, level, measured over the voice's own notes and averaged. Every pitched voice keeps its profile (the morphable form); the model that explains the stem better over the voice's notes plays it (`songprogram._additive_better`, samples win ties by 1 dB). On the synthetic: pads and plucks go additive, the bass stays on its samples.
- Tried on the drums and turned off because the synthetic measured them worse: velocity layers with extrapolated tails, per-event level fitting (NNLS on the waveform), masked extraction from templates, subtraction of coincident sounds (both the layer cascade and the isolated-exemplar cascade), and cluster-relative credited velocities. What remained: the plain exemplar at its recording level as the level reference, event velocities relative to that hit, credited hits by energy fraction. The drum residue (9.4 dB, −5 dB level) is the coincident-hit level split; it is measured, not solved.
- Bass onsets on the full cleaned signal rather than the percussive part: adopted (chroma up on all four real tracks tried).

**Real tracks on the corrected measure (six tracks, medians, reading v6, 2026-09-09 evening):**

| stem | raw exemplar | program | chroma (program) | note |
|---|---|---|---|---|
| drums | 21.0 dB | 21.1 dB | – | levels −5.6 dB; Sussudio −9.6 dB: the exemplar-hit level reference is wrong on real kits |
| bass | 14.8 | 16.1 | 0.73 (best yet, from onsets on the full signal) | |
| other | 14.4 | 15.6 | 0.41 | additive chosen on some voices; no gain on real synths |
| vocals | (voice) | 17.9 as reused phrases | 0.62 | the DTW reuse costs fidelity vs exact phrases |
| mix | 10.1 | 10.4 | 0.72 | |

Verdict: the day's renderer work moved the SYNTHETIC ceiling a lot and the real songs not at all. On real stems the voices themselves (what the reading calls an instrument, and the events it gives it) are the bottleneck, not the sample-vs-additive question: a real "other" stem is several evolving sounds at once and the onset clusters do not separate them; a real kit's coincident hits do not have one level. Next: measure the reading per voice on real songs with the same analysis-by-synthesis gap (which voices explain their share of the stem and which are noise), prune and merge on that evidence, and show it in the planner so the honest state is visible per instrument.

**2026-09-09, night — the reading checked against the stems by synthesis (`lib/dj/explain.py`, readings v7):**

Every instrument the reading claims is now played back alone and compared with the stem INSIDE ITS OWN EVENT WINDOWS (onset + 300 ms): `explained` = the share of the stem's spectral energy there the voice accounts for, `overshoot` = energy it puts where the stem has none, `spurious` = events whose window is silent. The figures are stored per instrument (the planner's gutter shows the % beside each name, red under 25%, `dj_instruments.py --show` prints them) and turned into candidates: events in silence are dropped (the voice, when ≥ 90% of them are); a voice explaining < 8% goes; two voices with exemplar spectra ≥ 0.95 alike whose events never coincide AND where the keeper's sound at the absorbed voice's hits still explains the stem are one sound.

The first version applied those rules directly and the six-track evaluation said no: Sussudio drums 20.3 → 9.1 dB (a 3% rim voice, a look-alike "snare" merged into the kick), but Final Voyage drums 42.8 → 49.7 (a voice with 60% of its events in silence still carried the other 40%) and Balance Perc Tool drums 14.1 → 15.8 (a merge that explained more in its own windows was louder over the stem). So the pass is a **closed loop**: each candidate is applied to the reading, the stem rebuilt and rendered, and kept only if the stem's spectral gap (`lib/dj/fidelity.spectral_gap`, the evaluator's measure) does not widen (`VERIFY_TOL_DB` 0.1). With that, on the same three tracks: Sussudio keeps both changes (20.3 → 9.1), Final Voyage rejects both drops, Balance Perc Tool rejects the merge and keeps three small event drops. Cost: 20–40 s per track on top of the reading. The synthetic gate stays ALL OK and drops nothing (hat vs shaker pass the spectral test at 0.96 but the hat sample at the shaker's hits overshoots, so they stay two).

What the per-voice figures say about real songs (the honest state, now visible per row): drum kick/snare voices explain 85–97%, hats 30–75%; bass voices 82–96%; `other` splits into one voice that explains 50–94% and one or two "perc"/"pluck" voices at 12–19% (real but barely modelled: several evolving sounds in one stem); vocals 5–30% (no sample model sings; they are reused as phrases, not rendered). The pads on the synthetic explain 13–30% — the held-voice sound model is the weakest renderer piece.

Six tracks on v7 readings (`--program`, medians): drums 20.5 dB (was 21.1), bass 16.1, other 15.6, vocals 17.9, mix 10.4 — the medians barely move because the loop changed only what it could prove: Sussudio drums 20.3 → 9.1 dB and its mix 7.8 → 6.4 dB (level −9.6 → −0.6: the dropped and merged voices had been splitting the kit's level); every other track is unchanged to the decimal. That is the honest result of this step: the check finds the wrong voices where they exist and does no harm elsewhere; it does not close the gap on stems whose voices are all "real but poorly modelled" (a 12–19% perc voice is not noise, it is a sound the onset clusters cannot separate).

The same measure now lives in one module, `lib/dj/fidelity.py`, and feeds three places: the evaluator (imports it), the reading's check, and the gen console's Analysis tab, which shows two readouts per linked song written by `gen_link.link` into `fidelity.json`: **NOTES AS CODE** (events → patterns + ops, events per entry, share of events carried by patterns vs written one by one, verbatim bars, vocal phrase reuse, per voice the share of its stem it explains) and **WAVEFORM vs THE ORIGINAL** (per stem and mix: spectral gap, level, rhythm r, missed/extra 16ths, onset F1, chroma). The **Fidelity** button recomputes it.

**2026-09-09, later — the four levers, measured (readings v8):**

1. *Per-voice level as a verified candidate* — built (`explain.GAIN_STEPS`, ±3 dB steps kept while the stem gap shrinks) and **turned off after a truth check**: on the synthetic gate the stem gap went 9.3 → 6.4 dB while the hat and shaker were pushed 4–6 dB ABOVE their true levels and the kick moved away from its. The energy-weighted mel gap rewards filling the stem's low-level high-band cells; it is not a level-faithful objective, and a candidate that changes only levels can game it. The prune/merge candidates survive this because each has independent evidence (3% explained, 94% explained by the other sample); a pure level move has none. Lesson recorded in the code: a new candidate kind needs a truth check on the synthetic (per-voice render vs the true sound, `scratchpad/synth_levels.py`), not just the objective.
2. *Held voices* — three reading defects found on the synthetic and fixed: the transcription fusion duplicated every held note per beat on top of the long hold (double energy, an attack every beat; `instruments.merge_overlaps`); the program collapsed a held chord to its lowest note, so the additive model played the root alone (items are now one per tone, with a `len` op when an occurrence holds longer than its pattern); the additive decay was fitted on the stem's harmonics and read the other voices' decaying notes as the pad's (a flat pad measured −6 dB/s), now the hold's own per-beat level slope from the reading (`hold_decay_db_s`). Then the pad LEVEL: the profile's reference was the loudest harmonic in the attack, which a melody an octave up owns (+7 dB, far too bright). Waveform subtraction of the other voices' render did nothing (a sample render is not phase-aligned; nothing cancels). What worked: a low percentile (20th) over the sustained frames per harmonic (the other voices' notes are transient, the pad's level is the floor they decay back to) plus a 6 dB spectral mask where the other voices' render owns a harmonic. Held voices always play additive (the alternative, the exemplar chunk looped, carries the melody inside the cut: onset F1 0.98 → 0.60, and it is not notes). Synthetic: pads vs the TRUE pad 31 → 15 dB and +7 → −2.5 dB level; the `other` ceiling 7.6 → 6.7 dB; chroma 0.80 → 0.88.
3. *Sounds inside `other` the voices miss* — experiment (`scratchpad/residual_voices.py`): on three real tracks the voices OWN only 2–14% of the `other` stem's energy (cells where the render is within 6 dB of the stem). Reading the spectral residual with the reader's own machinery finds 3–5 more voices and narrows the stem gap 4–5 dB (Sussudio 15.7 → 10.2), but they are catch-alls (a "perc" with 763 hits, a "pad" C2–C7 over 428 beats), onset F1 collapses to 0.1–0.4 and the level drops 6–11 dB. Not adopted: the gap is gamed the same way as in 1. The honest state: a real `other` stem is mostly unexplained, and adding voices by residual onset clustering does not separate its sounds.
4. *Bass pitch* — diagnostic (`scratchpad/bass_diag.py`): per-beat chroma r medians are 0.96 (Evolution) and 0.90 (Animalia); the evaluator's lower means come from outlier beats. Sussudio is the outlier track (median 0.61, 20% of beats under 0.3): every one of its worst beats is a wrong pitch CLASS, mostly C1–D1 readings at the range floor (kick bleed that `clean_bass` left) and a held E2 against a G♯. Candidate prepared, not yet applied: drop a voice's notes whose read pitch has under a quarter of the strongest pitch's salience over the note itself, verified in the loop.

**The honest state at the end of 2026-09-09 (readings v8, six tracks):** the spectral gap medians are drums 20.5, bass 14.0 (chroma 0.81), other 19.6 (was 15.6: the held-voice changes are better on the synthetic and WORSE on real `other`, level −8 dB — the 20th-percentile harmonic level under-reads real, evolving pads), mix 11.0. And the figure that matters was not being measured at all: **note agreement with an independent transcription** (basic-pitch, onset within 60 ms and the same pitch) is F1 0.19 / 0.14 (Sussudio bass / other), 0.04 / 0.34 (Evolution), 0.11 / 0.15 (Animalia). The user listened to Sussudio and called it beyond awful at the note level; the numbers agree. On the synthetic gate the reader scores 1.00 (bass) and 0.84 (melody) and basic-pitch 0.96 / 0.40, so the reader is right on clean synthetic sounds and wrong on real stems; the disagreement on real tracks is in onset timing first (Evolution bass: only 14 of 99 program notes have a transcribed onset within 60 ms) and in pitch second (Sussudio: pitch class agrees on 51% of the notes whose onsets match). The spectral gap tolerated all of that. `fidelity.notes_agreement` now puts the note figure in the report and the gen tab, first in line.

**2026-09-09, late — the planner's "hear reconstruction" returned garbage, and the cause was levels, not notes.** Per-voice renders of three real tracks (1, 51, 452) alone, unclamped: every kit and additive voice sits within a few dB of its stem inside its own windows, but the `other.1` pitched voice comes out at a raw peak of 1.2 / 6.5 / 2.6 against stem peaks of 0.75 / 0.93 / 0.90 (its per-pitch samples are 'un-velocitied' by up to 26 dB — `voices.pitch_sounds` divides by the sample event's own velocity gain, floor 0.05 — and then played at event gains up to +17 dB). `render` then scaled the whole track by that one voice's peak: the sum of voices had the original's loudness (−10.5 vs −10.2 dBFS on track 51) and the full render came out at −25.3. The planner now renders voices raw (`render(limit=None)`), caps each at its stem's in-window level (`voice_ceiling_db`, one-sided) and limits the sum with a look-ahead peak limiter; the additive voices are also where the render time goes (26-47 s of a 33-51 s full render, `other.3` on all three tracks), so the planner renders voice by voice and caches. Open for the evaluator: its numbers still go through the whole-track clamp, so the "level" column on a track with a hot voice measures the clamp; and the un-velocity gain itself (a sample chosen for isolation is often a quiet event) is the lever behind the hot voice.

**2026-09-09, the plan executed (readings v9) — the judge changed first, then the readers:**

1. **Truth set** (`tools/tests/_dj_truthset.py build | eval [--true]`): four composed songs (80s pop, house, rock, funk; 48 bars each, humanised timing, velocity accents, fills, rests) rendered part by part through the GM SoundFont (FluidSynth) and separated with the library's own demucs model, truth notes attached — `logs/truthset/<song>/`. Scores: notes F1 per pitched stem (60 ms + pitch), per drum sound hit F1, and the program's render against the TRUE stems. Everything below was judged here; the old additive gate checks plumbing only (its kit thresholds were relaxed to say so).
2. **Bass** (`lib/dj/bassreader.py`, `BASS_READER = "pitchtrack"`): one monophonic line from a pyin pitch track (frame 4096, hop 512) over the cleaned stem; note boundaries from pitch changes, unvoiced gaps and RE-ATTACKS of a fine rms envelope (a 6 dB rise within 40 ms after a 3 dB dip — legato repeated notes were the old reader's big miss: rock 8ths 129 read for 352 written, now 378); starts refined to the envelope's steepest rise; isolated short notes ≥ 7 semitones from both neighbours and notes 36 dB under the line dropped. Truth set, demucs stems: **notes F1 0.61 → 0.80** median (rock 0.49 → 0.92, house 0.65 → 0.91, funk 0.76 → 0.70 — slap-bass octaves, octave-blind 0.79 —, pop 0.58 → 0.67). `clean_bass` now skips subtraction under a correlation gain of 0.2 (a 0.07 subtraction on a clean stem read eight F#1 notes).
3. **Other** (`lib/dj/polyreader.py`, `OTHER_READER = "transcription"`): notes from basic-pitch (amp ≥ 0.3, ≥ 50 ms), grouped by start (a chord is one group), voices by clustering the groups' attack and sustained mel spectra, length, register and size. **Notes F1 0.29 → 0.61** median (house 0.79, funk 0.66, pop 0.55, rock 0.41 — distorted power chords and organ are where the transcriber fails: organ recall 0.18). Voice PURITY is poor (a voice's notes from its main part: 0.29–0.68; standardising and a lower silhouette did not help) — the notes are right, the assignment of notes to sounds is the open problem, and it decides which sample plays them. The `other` render against the true stems: 12.7 → 7.2 dB.
4. **Drums**: the onset threshold was the loss — `delta` 0.12 → 0.03 recalls 0.85/0.87 of the distinct hit moments (was 0.67/0.56) at precision 0.99–1.00; per-band detection measured worse and is off. Then the coincidence logic: a kick that always carries a hat has its 8 pure kicks merged INTO the 107 kick+hat onsets (explained by them at coefficient 0.63), so the mixture could never be split — a merge into one sound now needs that sound fitted at ≥ 0.8 of its level; a coincidence of several sounds each at level credits every one of them whatever their cluster sizes; the merge decision rests on the fit of the credited components only (a hat+snare cluster was being folded into the hat on a fit that leaned on an excluded component). Pop kit: hits F1 0.70 → 0.84, hats 0.47 → 0.83, kick 0.94, snare 0.89; house 0.83 → 0.74 on all hits (a regression to look at), rock and funk flat; per-sound median 0.81 → 0.83. Still open: toms and crashes (5–10 hits each, never their own cluster), the closed hat under the open hat (not separable by timbre), ghost snares.
5. **Real songs** (no truth): agreement with basic-pitch on bass rose on all three (Sussudio 0.19 → 0.32, Evolution 0.04 → 0.08, Animalia 0.11 → 0.25) but stays low — the reader finds 2–5× the notes basic-pitch does on busy synth bass lines, and which is right is not measurable without a truth. Sussudio re-linked: drums 7.6 dB, mix 5.8 dB, `other` chroma 0.76 (was 0.33). The user's ear on Replay is the remaining judge.
6. **Measurement hygiene**: every measuring render (fidelity, explain, evaluator, truth set) now uses `limit=None` — the renderer's default peak clamp scaled a stem by its one hottest voice (found by the mixer work) and made level columns lie.

**2026-09-09, the second round ("do them all", readings v10) — every item measured on the truth set, kept only when it won:**

| item | result | state |
|---|---|---|
| house-kit regression (0.83 → 0.74 on all hits) | diagnosed: the closed hat never plays alone in that pattern (always under the kick or the open hat), so no pure template exists and no rule can credit it; the earlier 0.83 was an accident of how the mixtures resolved | known limit, documented |
| bass: second faster pitch pass for fast octave pops | median 0.84 → 0.67 (house 0.93 → 0.22: fragments sub-bass) | rejected (`FINE_PASS = False`) |
| bass: per-note octave re-check | 0.84 → 0.84 | neutral, off |
| bass: no kick subtraction under gain 0.2; level floor; isolated-outlier filter | fixed the synthetic gate's F#1 notes; truth set 0.80 → 0.84 | kept |
| drums: longer timbre tail (0.2, 0.3 s) for ride vs hat | per-sound median 0.83 → 0.83; pop up, house and funk down | neutral, off |
| drums: masked-hit pattern fill | already what `SURE_SHARE` does for detected voices; the house closed hat has no detected voice to fill from | no change |
| other: held notes from the sustain pass where the transcription has none | pad recall 0.11 → 0.11 (its pitch classes were covered by the transcriber's fragments; the missing thing is the hold's onset) | neutral, off |
| other: merge the transcriber's same-pitch fragments | median 0.60 → 0.38 (piano stabs and clav chops are repeated notes) | rejected (`FRAG_GAP_S = 0`) |
| voice purity: harmonic-profile features + co-occurrence merging | 0.29/0.70/0.33/0.46 against mel's 0.39/0.56/0.33/0.46 | neutral, `FEATURES = "mel"` |
| truth set: purity and a 30 ms figure in the report; two more songs (dnb: ride + open + closed hats; ballad: a vocal-like lead on the vocals stem, rim, fretless bass) | in | done |

**Where the reading stands on the six-song truth set (demucs stems, readings v10, medians):**

| stem | notes F1 (60 ms) | at 30 ms | octave-blind | onsets | voice purity | render vs true stem | target |
|---|---|---|---|---|---|---|---|
| bass | 0.85 | 0.76 | 0.88 | 0.89 | 0.95 | 9.2 dB | 0.90 |
| other | 0.61 | 0.58 | 0.64 | 0.68 | 0.43 | 7.7 dB | 0.75 (0.68 realistic) |
| vocals (ballad only, old reader) | 0.73 | 0.72 | 0.73 | 0.81 | 0.67 | – | – |
| drums, all hits | 0.77 | – | – | – | – | 7.5 dB | 0.90 |
| drums, per sound (34 sounds) | 0.74 | – | – | – | – | – | 0.85 (0.78 realistic) |

Per song, bass: pop 0.70 (fast octave pops), house 0.93, rock 0.94, funk 0.75 (slap octaves; 0.90 octave-blind), dnb 0.87, ballad 0.82. Other: pop 0.55, house 0.79, rock 0.41, funk 0.66, dnb 0.82, ballad 0.43 (piano + strings). Drums: pop 0.84, house 0.74, rock 0.78, funk 0.75, dnb 0.66 (ride / closed / open hat), ballad 0.86.

What this round says: the reader changes that were going to be cheap wins were not; the numbers that moved today moved in the first round (bass 0.61 → 0.84, other 0.29 → 0.61, drum hits 0.67 → 0.76). The remaining distance to the targets (bass 0.90, other 0.75, drums 0.90 / 0.85 per sound, purity 0.8) is in three places the current methods do not reach: sounds that never play alone (kit coincidences), the transcriber's ceiling on distorted and sustained sounds, and telling GM-like timbres apart inside a separated stem.

**2026-09-09, third round ("execute the plan", readings v11) — the three named limits attacked with new methods, each diagnosed on the truth set before it was built, then measured:**

Diagnosis first (scratch scripts over the truth set, per note against the truth):

- *`other` purity is not an assignment problem.* On the pop track the voices ARE the parts (brass 96 of 102 groups in one voice, lead 88 of 95 in another); purity reads 0.39 because 262 of 461 transcribed groups match NO truth note, and the purity figure counts those in its denominator. What they are: **continuations** (the transcriber re-triggering a held note: 215 of a pad's 284 unmatched notes, 593 of a distorted guitar's, 235 of a string part's), then sub-octave ghosts (179 on the pop track), then bass bleed. Precision is the lever, and the pads' recall (0.11 / 0.28 / 0.19) is the transcriber placing the hold's first fragment a median 0.5 s after the onset.
- *Bass: the pop track's octave pops are not in the stem.* Every 1-step octave pop (44 of 44) is missed because demucs puts a short note above ~90 Hz into `other`: during the pop the demucs bass stem sits at −42 dBFS (true stem −27), `other` at −25. Reader ceiling on the TRUE stems: pop 0.91 (demucs 0.68). The funk track's pops that survive separation are tracked an octave LOW by pyin (31 of 36), and a D2 line read as D1 (28 notes).
- *Drums: the rare sounds are not in the residual.* Toms and crashes (5–10 hits, 7 of the 34 sounds at F1 0.05–0.33) are folded into big clusters by k-means, and a residual test cannot recover them: 4–5 broadband templates with free NNLS gains explain ANY onset patch to within 7 % (tom and crash onsets r_pos 0.01–0.07, the same as pure hats).

| change | result on the truth set | state |
|---|---|---|
| **bass odd-harmonic octave test** (`bassreader._octave_fix`): a note whose odd harmonics (1, 3, 5) carry under 0.55 of its even ones over the note is the octave up | funk 0.75 → 0.88 (P 0.82 → 0.96), the other five unchanged; median **0.85 → 0.87**; ratio 0.3 gave 0.76, 0.7 the same as 0.55, 0.85 lost pop and rock | kept |
| bass octave-DOWN test (sub-octave's odd harmonics present) | neutral | off (`OCTAVE_DOWN`) |
| **`other` continuation merge** (`polyreader.merge_continuations`): a same-pitch note within 0.3 s of its predecessor is folded into it when the pitch's own harmonic energy (23 ms windows every 2.9 ms) neither dips ≥ 4 dB nor rises ≥ 8 dB at its start AND the predecessor is ≥ 0.2 s (a hold's fragment chain begins with a long note; a 16th-note clav chop's predecessor is short) | per candidate: keeps 0.97–1.00 of the real same-pitch successors on five songs (0.84 on the funk clav), folds 13–49 % of the fakes. Notes F1: rock 0.41 → 0.49, ballad 0.53 → 0.60, pop 0.54 → 0.56, house flat, dnb +0.01, **funk 0.66 → 0.62** (clav and e-piano play the SAME pitches, so a re-strike shows no dip under the other's chops); median flat, precision 0.54 → 0.59, purity 0.42 → 0.46; holds now carry their real length into the render | kept (the render gain and four songs up; the funk loss is the shared-pitch case) |
| continuation test on basic-pitch's onset posterior (exposed as `instruments.transcribe_full`) | median 0.60 → 0.52: basic-pitch makes repeated stabs from activation jumps, not from its onset head, so their posterior is as low as a hold's | off (`CONT_TEST = "spectral"`) |
| continuation test on a level rise alone (46 or 93 ms windows, 4–12 dB) | recall 0.65 → 0.43 on funk, 0.99 → 0.89 on house piano: fast repeats never decay inside any pre-window | superseded by the dip + predecessor-length rule |
| distance to the 16th grid as a cue | useless: re-triggers sit on the grid too (they happen when other notes strike) | not used |
| **sub-octave ghost drop** (`polyreader.drop_ghosts`): a note an octave under a sounding one whose odd harmonics sit ≥ 12 dB under its even ones | +0.01 on three songs, nothing lost | kept |
| hold backtracking: a merged hold starts where its pitch's energy rose | 0.61 → 0.60, pad recall 0.08 → 0.06: the pitch's energy is shared with other parts' notes and the walk back lands on them | off (`HOLD_BACKTRACK`) |
| drums: rare-sounds pass over unexplained onsets (`instruments._rare_sounds`) | finds nothing (see the diagnosis) | off (`RARE_SOUNDS`) |
| drums: clusterer capacity 8 → 12 / 16 (`DRUM_K_MAX`) | all-hits 0.77 → 0.80 / 0.81 but per-sound median 0.73 → 0.63 / 0.64: more clusters, more wrong merges among the cymbals | kept at 8 |

Where the reading stands (six songs, demucs stems, readings v11, medians; the full evaluation table is below the caveat):

| stem | notes F1 (60 ms) | at 30 ms | octave-blind | onsets | voice purity | render vs true stem | target | was (v10) |
|---|---|---|---|---|---|---|---|---|
| bass | **0.87** | 0.82 | 0.88 | 0.89 | 0.97 | 9.2 dB | 0.90 | 0.85 / 0.76 |
| other | 0.60 | 0.58 | 0.64 | 0.68 | **0.46** | 8.1 dB | 0.75 | 0.61 / 0.43 |
| vocals (ballad) | 0.73 | 0.72 | 0.73 | 0.81 | 0.67 | – | – | 0.73 |
| drums, all hits | 0.76 | – | – | – | – | 7.4 dB | 0.90 | 0.76 |
| drums, per sound | 0.74 | – | – | – | – | – | 0.85 | 0.74 |

Per song, bass: pop 0.69, house 0.93, rock 0.94, funk **0.88** (was 0.75), dnb 0.87, ballad 0.82. Other: pop 0.57 (0.55), house 0.79, rock 0.49 (0.41), funk 0.62 (0.66), dnb 0.83 (0.82), ballad 0.44 (0.43); precision 0.54 → 0.59 median, the `other` note count on the rock track 1140 → 783 for 819 written.

What this round says: of the three named limits, one was misdiagnosed (`other` purity was transcription precision, and the fixable part of it — held notes re-triggered — is now folded where the stem shows no attack), one is partly separation (the pop track's bass pops live in the `other` stem; the funk pops that survive are now read at the right octave), and one stayed closed (kit coincidences and rare sounds: neither a residual test nor more clusters finds them, because the template space is too flexible to leave a residual and too coarse to keep a cymbal apart). Synthetic gate ALL OK throughout.

**2026-09-09, fourth round ("do them all in your suggested order", readings v12):**

1. *Listening prep* — the three real tracks (1 Final Voyage, 51 Evolution, 452) re-read and re-linked with the final readers, so Replay plays v12 (`logs/analysis/<title>/`). Caveat for the gen tab's `other` note agreement: it is measured against raw basic-pitch, and the continuation merge folds basic-pitch's own fragments, so that figure reads lower by construction now; the bass figure is unchanged in meaning.
2. *Bass completion from `other`* (`instruments.complete_bass`): built with every structural test (monophonic, short, in a gap of the line, an octave or less from both neighbours, near one) — **rejected**: 26 notes moved on the pop track, none a pop (bass 0.68 → 0.66), 2 wrong moves on rock. The pop sits on the pad's root pitch, so the transcriber reads it as the pad continuing; there is no note in `other` to move. Off (`COMPLETE_BASS`).
3. *Continuation merge within each voice* (`polyreader.CONT_SCOPE = "voice"`): clustering first, then a same-pitch note folds only into a predecessor of its own voice — the clav chop can no longer fall into the e-piano hold on the same pitch. Funk 0.62 → 0.64 (e-piano recall 0.65 → 0.74), the others unchanged; **median 0.62** (baseline 0.60, global merge 0.61). Kept.
4. *Drums by a learned kit decomposition* (`lib/dj/drumsep.py`, `DRUM_READER = "drumsep"`): the drum stem split into kick / snare / toms / cymbals by drumsep (a hybrid-demucs checkpoint, 167 MB, fetched into `models/` on first use, loaded with the demucs classes allowlisted under torch's weights-only loader) and each family read alone by the existing cluster reader, with the other families' bleed gated 18 dB under the family's loud onsets (0 dB → all-hits 0.74, 18 → 0.87, 24 → 0.79); exemplar cuts still come from the mixed stem, isolated against every family's onsets. Raw family onsets against the truth: kick / snare / toms recall 1.00 everywhere; cymbals F1 0.72–0.97 (funk 0.97 where the mixed reader had hat 0.80 and open hat 0.11).

   | | v11 (mixed stem) | v12 (per family) |
   |---|---|---|
   | all hits F1 median | 0.77 | **0.87** |
   | per-sound F1 mean | 0.62 | **0.73** |
   | per-sound F1 median | 0.74 | 0.71 |
   | kick, six songs | 0.94 0.81 0.89 0.80 1.00 0.94 | 0.99 0.84 0.98 0.89 0.98 1.00 |
   | snare | 0.89 0.86 0.86 0.52 0.68 1.00 | 0.95 0.86 0.94 0.63 0.62 1.00 |
   | toms / crash | 0.20 0.20 0.20 · 0.10 0.10 0.05 · 0.33 | 0.67 0.67 0.50 · 0.50 0.50 0.62 · 0.70 |
   | hat / open hat / ride | pop hat 0.83, house ohat 0.93, rock ride 0.40 | pop hat 0.76, house ohat 0.73, rock ride 0.69 |

   A cross-family bleed rule (a voice whose hits coincide with louder voices of other families is their bleed, `_drop_family_bleed`) was measured at three settings and turned OFF: all-hits 0.87 → 0.85–0.86, it also drops real rare sounds (a crash, a rim, a clap). It only mattered on the synthetic gate's additive kit, which the separator sprays over all four families (446 "toms" onsets on a kit without toms); that gate's drum-count check now accepts up to 14 voices under the drumsep reader (the four written sounds are all still found, kick / clap / shaker at 1.00) and says why — the count is judged on the truth set.

   The coincidences the plan called closed (a hat under every kick, toms folded into big clusters) are open again and largely solved; what remains is INSIDE the cymbal family, where the cluster reader over-splits (8–13 voices for 6 sounds) and the hat / open hat / ride distinction is still timbre clustering. That is the next lever: a cymbal-specific pass (or LarsNet, which separates hi-hat from the other cymbals, 562 MB, CC BY-NC).
5. *Hold backtracking within a voice*: 0.62 → 0.61 — still lands on other parts' notes at the same pitch. Off.

**Where the reading stands (six songs, demucs stems, readings v12, full evaluation through `identify` + program, medians):**

| stem | notes F1 (60 ms) | at 30 ms | octave-blind | onsets | voice purity | render vs true stem | target | v10 |
|---|---|---|---|---|---|---|---|---|
| bass | **0.87** | 0.82 | 0.88 | 0.89 | 0.97 | 9.2 dB | 0.90 | 0.85 |
| other | 0.60 | 0.58 | 0.64 | 0.68 | 0.46 | 8.1 dB | 0.75 | 0.61 / purity 0.43 |
| vocals (ballad) | 0.73 | 0.72 | 0.73 | 0.81 | 0.67 | – | – | 0.73 |
| drums, all hits | **0.87** | – | – | – | – | 8.2 dB | 0.90 | 0.76 / 7.4 dB |
| drums, per sound | 0.69 | – | – | – | – | – | 0.85 | 0.74 |

Drums per song (all hits): pop 0.89, house 0.85, rock 0.90, funk 0.91, dnb 0.82, ballad 0.82 (v10: 0.84 0.74 0.78 0.75 0.66 0.86). Kick 0.99 0.84 0.98 0.90 0.98 1.00; snare 0.95 0.86 0.94 0.63 0.62 1.00; toms 0.67 · 0.51 · 0.70; crash 0.50 / 0.62. The per-sound median falls to 0.69 because the cymbal family over-splits (pop hat 0.83 → 0.76, house open hat 0.93 → 0.67, ballad hat 0.83 → 0.54) while ride improves (rock 0.40 → 0.69, dnb 0.46 → 0.58); the drum render's gap against the true stem is 0.8 dB wider than v10 because the samples are still cut from the MIXED stem while the events come from the families — cutting them from the family stems is the natural follow-up and would also make a soloed kit clean. `other` in the full evaluation is 0.60 (0.62 in the reader-only harness: the self-check's verified drops sit between the two), with precision and purity up on v10.

What the fourth round says: the drum reading's structural limit (coincidences, rare sounds) yielded to a learned decomposition where every rule had failed, and the remaining distance to the drum targets is one family deep (cymbals). Bass is 0.03 from its target and the rest is separation (the pop track's pops). `other` is at the transcriber's ceiling on the notes; what moved was the language's honesty (holds carry their length, the count of notes is close to the written count on four of six songs).

**2026-09-09, fifth round — the SOUND MODEL, after the user heard v12 as "a bit better, tinny and muddy, not the song":**

The note readings had moved and the ear had not, and the render-vs-true-stem column had said why all along (8–9 dB with notes ~0.9 right). So the judge changed again: **the render gate** (`_dj_truthset.py render`) plays each truth part's TRUE notes through the reading's voice for that part (its extracted sound model) and compares with the TRUE part stem, one gain fitted per part, per-part renders written next to the true parts for listening (`render_<part>.wav`), the reading cached per version so a renderer change re-measures in minutes. It also reports the OTHER model per pitched voice (samples vs additive), so the model choice is measured rather than trusted.

First measurement (v12 renderer): drums 8.2 dB, bass 9.8, **other 15.7** — and the catastrophes were the held parts through hit models: pads 50–57 dB, organ 54, horns 35 (a pitched voice plays its 0.3 s cut and then nothing for a 2 s note; a pad CHORD went through the chord path, one decaying keys recording). Sample voices rendered 4–8 dB hot against additive ones in the same stem; fully additive voices 12 dB quiet.

| change | measured on the render gate | state |
|---|---|---|
| **held notes through the additive profile** (`HOLD_ADDITIVE`): a note ≥ 0.6 s and > 1.5× its recording plays the voice's profile, decay clamped to a sustain (−6 dB/s), level tied to the recording at the same gain; the same rule in the CHORD path per tone; a looped recording when no profile | pop pad 50 → 18 dB (chroma 0.11 → 0.80), house pad 57 → 15 (0.88), dnb pad 33 → 11 (0.90), organ 54 → 10; **other 15.7 → 11.9 dB** | kept |
| **additive level calibration** (`CALIBRATE_ADDITIVE`): the profile's level tied to the voice's exemplar recording (additive note at velocity 1.0 over the 0.25 s attack window against the exemplar, which is scaled to the loud-hit level) | ballad piano / strings −12.5 → +2.3 / +2.6 dB; a first version at the exemplar event's own velocity over-shot up to +13 dB | kept |
| **the truth PART files carried no mix gain** (written before the normalisation): the level column read +8 dB on every sample voice | fixed in the gate | – |
| multi-sample pitched voices (`voices.MULTI_SAMPLE`): 1–3 velocity zones per pitch, each a recording at its RECORDED level, played by the zone nearest the event and moved by the difference only (the un-velocity division, floor 0.05, up to +26 dB, is gone); saved/loaded with the program | neutral on the gate's spectral column; removes the hot-voice mechanism found in the planner | kept |
| kit samples from the drumsep family stems (`FAMILY_SAMPLES`) | drums 8.2 dB (the mixed-stem run could not be completed alone: see the GPU note) | kept |
| shared **mixdown** (`songprogram.mixdown`): raw voices, one-sided per-voice ceiling against the stem in its own windows, per-stem level match (±6 dB, rms in the voices' windows), verbatim/phrases at their level, look-ahead limiter on the sum — now what Replay and the CLI render play (both used to go through the whole-track peak clamp that sank a replay 7–15 dB under one hot voice) | – | kept |
| model chooser (`_additive_better`): judged on the 30 best-isolated notes, level-free, and NOT on the events the samples were cut from (there the sample render IS the stem: rock other.1 read "0.0 dB" and samples always won) | dnb sub now chosen right (additive 9.7 vs 18.8); rock bass (21.5 vs 8.2) and pop bass (10.1 vs 7.7) still samples: on the demucs stem the samples reproduce the guitar bleed and the chooser cannot see past it (stem gap 3.7 vs 4.5) | kept; the two wrong calls are a stem-vs-part limit |
| chooser on the note's harmonic bands only | flipped the dnb sub back to the wrong model, rock unchanged | off (`CHOOSER_HARMONIC`) |
| per-pitch samples from SOLO onsets (no other onset within 30 ms) | other 11.9 → 14.3 dB (e-piano 11.1 → 14.5, clav 9.7 → 10.6): the solo onsets are the quiet, short ones | off (`SOLO_WEIGHT` 0) |

**Render gate, six songs, true notes, v13 renderer:** all parts 11.0 dB; drums 8.2 (n 6); bass 9.9 (n 6); other 11.9 (n 17; was 15.7). What remains above 15 dB is voice PURITY, not rendering: rock lead guitar through the power-chord voice (21.6), funk horns through the clav voice (36), house piano / pluck / pad in ONE voice (15–17), brass 15.8 — and the two bass voices where the chooser follows the bled stem.

**2026-09-09, sixth round — the four next steps (purity, cymbals, other-stem recall, timbre), in progress:**

- *Other-stem recall via the transcriber's thresholds* (`TRANSCRIBE_ONSET` / `TRANSCRIBE_FRAME`): 0.3/0.2 → notes F1 0.62 → 0.48, 0.4/0.25 → 0.54; pad recall 0.08 → 0.15 / 0.11. **Rejected**: the held parts' missing onsets are not under the threshold (the same conclusion as the backtracking), and everything else that comes up is false. Held-part recall stays a transcriber limit.
- *Voice purity*: two alternative clusterers in the `other` reader (`polyreader.CLUSTER`): Gaussian mixture by BIC and Ward agglomerative under a threshold. Reader-only harness: k-means/silhouette (current) F1 0.62 / purity 0.46; agglomerative 0.60 / 0.48 with the per-voice merge, 0.61 / 0.48 with the global merge — house purity 0.60 → 0.71–0.74, funk 0.46 → 0.56, rock and ballad down (more voices, fewer re-triggers folded). The render gate said **no**: `other` 11.9 → 14.3 dB (the pop pad gained a voice of its own, 18.5 → 14.3; the house pad 15.1 → 19.0, pluck and dnb keys lost; the funk horns stayed at 35 dB in a voice of their own — their recordings are wrong, not only their voice). k-means stays. Purity is not a clustering-method problem; it is the features (a GM part in a demucs stem is not separable on attack/sustain mel) and it remains the open problem.
- *Cymbals*: a 0.25 s timbre window inside the cymbal family (`FAMILY_TAIL_S`) — **rejected**: all-hits 0.87 → 0.84, per-sound median flat (house hat 0.83 → 0.92, four other hats down).
- *Timbre*: the hybrid note model (`TAIL_ADDITIVE`, recording attack + additive continuation) — **neutral** on the render gate (every part within 0.4 dB); kept as the more honest tail.

**Speed (same day, the user: "the per-song analysis seems extremely slow now"):** the day's hold and tail changes had routed many more notes through the additive synthesizer, the renderer's cost, and the build separated the drums a second time. Done: additive notes memoised per voice (`_note_cache`), energy-only renders in a fast mode (the profile masks, the self-check), the drum separation cached in-process, the stem readers in three threads (`PARALLEL_READERS`; the bass waits for the drums), the vocals note reading off by default (`READ_VOCALS`; the program carries vocals as phrases from the stem audio, now independent of vocal rows), and pyin on four pitch states per semitone instead of ten (`PYIN_RESOLUTION` 0.15: its Viterbi decode was 49 of the bass reader's 51 s; same notes at 0.15, the funk slap line goes at 0.2–0.25; the decimation alone changed nothing). Final Voyage: reading 177 → 73 s, program build 35 s, render 31 s. The six-song truth-set eval through the parallel readers: every note figure identical (bass 0.87, other 0.60, drums 0.88 / 0.69) in 446 s for the six against ~780 before. **`onnxruntime-gpu`** installed and kept: the CUDA provider loads (torch imported first puts the cuDNN DLLs on the path), the house `other` stem transcribes identically in 4.9 s where the CPU took 14–36 s. Synthetic gate ALL OK.

Two more render-gate verdicts the same evening: **kit samples from the drumsep family stems measured WORSE than the mixed stem's isolated cuts** (drums 8.2 vs 7.0 dB; house 15.6 vs 9.6, pop 7.7 vs 5.9) — a separated family stem is a softer, smeared version of the hit; `FAMILY_SAMPLES` off, which also removes a separation from the build. And at the user's word ("larger samples from the vocal stem are fine"), **vocal phrase reuse is exact-only** (`voices.PHRASE_WARP` False: a repeat must match frame for frame within 5 % of the same length); the DTW reuse stays as the compression switch.

**2026-09-09, seventh round — existing models for the open problems (the user: "is this solved with existing libraries?" → "do them all"):**

| piece | what | licence / footprint | state |
|---|---|---|---|
| `htdemucs_6s` on the `other` stem (`lib/dj/sixstem.py`, `OTHER_FAMILIES`) | guitar / piano / the rest read as families by the transcription reader; a family under the stem by 18 dB is the separator's bleed and is skipped | demucs (MIT), weights from demucs' remote (52 MB) | **measured, no as first built**: notes F1 0.60 → 0.56 (a note leaking into two families is transcribed twice: dnb 597 → 924 notes for 496 written; purity up on house only, 0.61 → 0.79); render gate `other` 11.9 → 12.3 dB (pop pad 18.5 → 12.3, organ 10.0 → 24.3, lead guitar 21.6 → 25.0, horns 36 either way). A cross-family de-duplication (`OTHER_FAMILY_DEDUPE`) is built and queued for re-measurement; if that does not turn it, the split stays off |
| YourMT3+ (`lib/dj/ymt3.py`, `tools/dj/ymt3_transcribe.py`, `OTHER_READER = "ymt3"`) | notes labelled by GM program from the YPTF.MoE+Multi (noPS) checkpoint; one voice per program (`polyreader.read_labelled`); runs in ITS OWN interpreter (WSL venv `~/ymt3venv`: torch cu126, transformers 4.45.1, lightning) because its pins conflict with this environment's transformers 5.x; 55 s of GPU inference per 100 s stem at batch 2 (batch 8 paged next to the app's own CUDA context) | model code GPL-3.0 (kept outside the repo under `models/yourmt3/space`, called as a subprocess), checkpoint 561 MB | **the round's finding**: `other` notes F1 **0.60 → 0.69**, purity **0.47 → 0.75** (target 0.80); house 0.79 → 0.81, rock 0.49 → 0.62, funk 0.62 → 0.75, ballad 0.44 → 0.83 (purity 0.80); pop 0.55 → 0.31 (a voice read an octave off: octave-blind 0.63) and dnb 0.83 → 0.54 (the same note under two programs: 853 notes for 496). Render gate `other` 11.9 → 11.1 dB (pop pad 18.5 → 11.1, ballad piano 9.2; the organ 10 → 24.5, the funk horns 35 either way — that part's RENDER is wrong, not its voice). With cross-voice de-duplication + a per-voice octave vote (0.6 of 40 notes, odd < 0.55 × even): pop 0.31 → 0.65, funk 0.75 → 0.82, dnb 0.54 → 0.61, but house 0.81 → 0.60 and ballad 0.83 → 0.74 (octave-blind 0.84 / 0.85: the vote moved piano voices whose low notes have weak fundamentals); median 0.64. **De-duplication alone: notes F1 0.73, purity 0.73** (house 0.83, rock 0.64, funk 0.82, dnb 0.57, ballad 0.84, pop 0.32) — the new default (`OTHER_READER = "ymt3"`, readings v13); the octave vote is off (a stricter vote, 0.8 and odd < 0.35 × even, measured separately). Transcriptions are cached on disk (`logs/ymt3_cache/`) so a re-read of a track skips the model |
| LarsNet (`lib/dj/larsnet.py`, vendored `lib/dj/vendor/larsnet/unet.py`, `DRUM_SEPARATOR = "larsnet"`) | five drum families incl. the hi-hat apart from the cymbals | **checkpoints CC BY-NC 4.0 (non-commercial)**: measured for research; a commercial deployment must not ship them; 562 MB under `models/larsnet/` | per-sound F1 median **0.78** (drumsep 0.69, mixed 0.74): rides 0.88 / 0.91 / 0.92 (were 0.69 / 0.58 / 0.75), rock hat 0.50 → 0.86, dnb hat 0.53 → 0.78; but 12–32 voices for six sounds, the closed hats split across them (pop 0.76 → 0.45, house 0.83 → 0.51) and all-hits 0.87 → 0.73. With a per-family cluster cap (`FAMILY_K_MAX`: kick 2, snare 3, toms 2, hi-hat 2, cymbals 3): **per-sound median 0.85 — the target** (rides 0.90 / 0.95 / 0.92, kicks 0.97–1.00, closed hats 0.87 / 0.88 / 0.80 on pop / house / rock), all-hits 0.79 (duplicate voices still add false hits: house 12 voices for 6 sounds), weak spots ballad hat 0.40, open hats 0.3–0.6. drumsep with the same caps: per-sound 0.69, all-hits 0.87, 5–7 voices. The caps stay for both; `DRUM_SEPARATOR` stays "drumsep" by default **because LarsNet's weights are CC BY-NC** — switching the default is a licence decision |
| DDSP per voice | a learned per-instrument model (acids-ircam `ddsp_pytorch`, Apache-2.0) | training per voice | not started: only if the render gate says timbre is what remains after the three above |
| harmonic-masked sample cuts (`voices.MASK_SAMPLES`) | a per-pitch or chord recording keeps only the energy near its own harmonics (in a dense stem a cut at a note's onset is the whole mix at that moment) | – | **rejected**: fixes other-pitch bleed on harmonic instruments (rock bass 21.5 → 9.9, lead 14.5 → 11.3, pluck 16.1 → 13.6) and damages broadband ones (slap 7.3 → 15.1, distorted guitar 6.0 → 10.0, clav, stab); an energy-ratio rule to tell the two apart still lost three bass parts; medians flat. The funk horns sit at 35 dB in every variant and in their own voice: **same-pitch overlap** (the clav plays the horns' chord tones under them) — no cut and no mask separates that; only a source model can |

**2026-09-09, eighth round — after the user's listening ("improve it more"), four items, measurements queued:**

1. *Chord-tone level split* (`polyreader.SPLIT_CHORD_LEVEL`): a group's measured level is the chord's; each tone now gets 1/n of the energy. Found on the ballad: the additive piano rendered 9 dB above the true stem because every tone of a chord carried the whole chord's level (the same stacking behind the sample voices' 4–8 dB hot polyphony). The rock track's 25 dB was a different cause: YourMT3 gives it no organ voice and truncates holds (p90 0.9 s for 2–3 s written). **The render gate then caught the split's other half**: with the split alone, every SAMPLE voice of `other` rendered true notes 4–10 dB hotter than before (house piano level +4.4 → +10.1, rock organ +4.1 → +14.0, funk clav +5.5 → +10.4, dnb stab +2.2 → +8.8; `other` median 14.9 → 16.3 dB) — a per-pitch recording cut at one tone of a chord CONTAINS the chord, and its un-velocity reference had become the tone's share. Fixed at the root (`voices.GROUP_LEVEL_REF`, readings v13+): a cut's level reference is its voice's simultaneous tones summed in energy (`group_vel`), for per-pitch layers, chord recordings and chord playback alike (`songprogram._chord_vel`). Gate: piano +4.7, organ +7.2, clav +5.8, stab +2.2, epiano 16.4 → 11.5 dB; `other` median **15.0 dB** (baseline 14.9, additive parts unchanged at 9.2 / 7.6) — the split now costs nothing on samples and keeps its additive gain. Also learned: the render gate runs in 3¼ min on the CPU from cached readings after item 5, so it is the cheap check for every renderer change.
2. *Notes verified against the stem* (`instruments.verify_other_notes`, `OTHER_VERIFY_OCTAVE`, `OTHER_EXTEND_HOLDS`): per note, the read pitch against the octave above and below by harmonic-summation salience over the note (margin 1.3), and the note's END moved to where the stem stops holding its pitch (≥ 0.25 s notes, up to 4 s, up to the voice's next onset at that pitch). Measured on a CPU harness that runs the `other` reader as `identify` does (YourMT3 from its cache) and renders the events against the TRUE `other` stem. Baseline (v13 + chord split): notes F1 0.73, purity 0.72, render 20.1 dB, level +3.8. **Octave test: rejected** — pop 0.29 → 0.64 (the synth voices YourMT3 reads an octave off come right) but house 0.82 → 0.67, rock 0.64 → 0.53, ballad 0.84 → 0.73 (a strong second harmonic on pianos and guitars reads as "the octave up"); median 0.65, and per voice (a vote over the voice's notes) the same trade at 0.60 / 0.55. No salience rule tells a wrong octave from a bright instrument; `OTHER_VERIFY_OCTAVE = False`. The stricter per-voice vote (0.8 of the notes, odd < 0.35 × even) measured in the full eval as well: `other` 0.66 / purity 0.70 — rejected too. **Hold extension: kept** — alone it leaves the notes untouched (F1 0.73, purity 0.72, oct-blind 0.75) and takes the render from 20.1 to **17.3 dB** (rock 25.1 → 20.7, ballad 23.0 → 17.2, pop 24.2 → 23.0, dnb 17.4, house 13.3, funk 8.5), level +3.8 → +4.0; `OTHER_EXTEND_HOLDS = True`. Full v13 evaluation through `identify` + program with both defaults: bass 0.87 / purity 0.97, `other` **0.73 / 0.73** (house 0.83, rock 0.64, funk 0.82, dnb 0.57, ballad 0.84, pop 0.32), drums all-hits 0.88, per-sound median 0.67; the `other` wave gap in the eval sits at 19.4 dB against 8.1 dB under basic-pitch v12 — that column rewards over-reading (v12 read 597 notes for dnb's 496 at purity 0.46 and still "won" the gap by 10 dB) and is not the judge; the render gate on true notes is.
3. *Twin voices in a drum family* (`FAMILY_MERGE_TWINS`): two voices of one family whose hits coincide ≥ 60 % within 20 ms are one sound read twice; the larger keeps its recording and takes the smaller's new hits — LarsNet's duplicates (12 voices for 6 sounds). **Rejected** on the drums harness: all-hits F1 up (drumsep 0.87 → 0.90, LarsNet 0.79 → 0.85) but per-sound F1 — the target — down (drumsep median 0.69 → 0.66, mean 0.72 → 0.68; LarsNet 0.85 → 0.73): the merges swallow real sounds (ballad snare 1.00 → 0.67, tom 0.70 → 0.48; every kit comes out at exactly 5 voices). The same trade `DRUM_K_MAX` showed in the third round: fewer voices count more hits and name fewer sounds. `FAMILY_MERGE_TWINS = False`; the v13 eval and render gate above ran with it on (drums per-sound 0.67, the house kit 14.8 dB in the gate), so both are re-run on the final defaults below.
4. *Real-instrument truth songs* (`tools/tests/_dj_truthset_slakh.py`): BabySlakh (the first 20 Slakh2100 tracks, professionally sampled instruments, per-stem audio + aligned MIDI, CC BY 4.0, 16 kHz, 883 MB → `logs/slakh/`) imported into the truth set's own format — stems summed into drums / bass / other by GM class, drum parts split per sound by pitch, upsampled to 44.1 kHz, separated with the library's demucs, `truth.json` from the MIDI tempo map — so every change is also judged on material that behaves like a record. Four tracks built (Track00001–4: 14–17 parts, 1165–3479 notes, 80–150 bpm). First eval: `other` **0.72** (0.74 / 0.47 / 0.71 / 0.73; purity 0.77) — the GM truth set's 0.73 holds on sampled real instruments; drums all-hits 0.68, per sound 0.50 (hats and shakers at 0.2–0.6, kicks 0.9–1.0, snares 0.8–1.0 on three of four); bass **0.00 exact with octave-blind 0.94–0.96** — every BabySlakh electric bass sounds 12 semitones below its MIDI (pyin on the TRUE stems agrees on 40 of 40 long notes per track), so the MIDI was not the truth of pitch. The importer now measures each pitched part's octave on its own audio (`octave_offset`: pyin over the 40 longest notes, kept when 80 % agree on one whole octave) and shifts the notes; `retruth` rewrites truth.json without re-separating. **On the corrected truth (four tracks, medians): bass 0.94 / purity 0.90 (0.95, 0.72, 0.95, 0.93 — the pick bass of Track00002 is the one below the GM set's 0.87), `other` 0.72 / 0.77 (0.74, 0.47, 0.71, 0.73), drums all-hits 0.68, per sound 0.50.** Read against the GM set: the bass reader is better on sampled basses than on SoundFont ones, `other` holds at the same figure with more of it in the right voice, and the drum reading is a clear step down on real kits — hats and shakers at 0.2–0.6 and open hats near 0 under drumsep, with kicks 0.9–1.0 and snares 0.8–1.0 on three of four tracks. The Slakh kits are the next drum target; the render gate on them (`_dj_truthset.py render slakh_Track0000N`) is the next measurement.
5. *Additive synthesis 5–10× faster* (`additive.render_note`; the reference routine kept as `render_note_ref`): profiling the ballad — the slow song in every evaluation (200–330 s where the others take 50–120) — put 51 of its 54 s program build in `render_note` (678 notes at 65 ms: one `np.sin` per harmonic per sample in float64, the envelope interpolated per sample per harmonic, and the note cache re-created on every render call so the six mask renders of the build re-rendered everything). Now the envelopes are built in dB on the hop grid for all harmonics at once and interpolated in amplitude, the sinusoids come from the angle-addition recurrence on the fundamental (one sin and one cos per note), and the note cache lives with the profile object (level in the key). Same sound: level exact, waveform error 40–63 dB under the signal on the ballad's profiles (the release ramps between hops instead of per sample), truth-set `other` harness identical to the decimal on all six songs (F1 0.73 / purity 0.72 / render 17.3 dB). Ballad `other` build 54 → 4.7 s, a 95 s render of its voices 0.2 s, the six-song `other` harness 59 s end to end. This is also the planner's stem ↔ reconstruction switch.

**Where the reading and the render stand on the final defaults of the day (readings v13: YourMT3 + de-duplication + chord split + hold extension, drumsep families with caps, twins off; renderer with the group-level reference and the fast synthesis; six songs, medians):**

| | bass | other | drums |
|---|---|---|---|
| notes F1 (60 ms + pitch) | **0.87** (target 0.90) | **0.73** (target 0.75) | all hits **0.88** (target 0.90); per sound **0.67** (target 0.85) |
| voice purity | 0.97 | 0.73 (target 0.80) | – |
| render gate, true notes through the voices | 10.4 dB | 15.0 dB (additive parts 7.6–9.2, sample parts 9.6–21, the funk horns 35.5) | 7.1 dB |
| eval time, six songs | 6 min (was 13–17) | | |

The full eval through `identify` + program: bass 0.87 / 0.97, `other` 0.73 / 0.73 (house 0.83, rock 0.64, funk 0.82, dnb 0.57, ballad 0.84, pop 0.32), drums 0.88 / 0.67 — every song 42–60 s. Render gate on a fresh reading: all parts 11.1, drums 7.1, bass 10.4, `other` 15.0 dB. One kit moved against the day: the house kit 11.8 → 15.2 dB since the cymbal-family cap (its open hat now shares the closed hat's voice and sample); the drum median is unchanged and the cap was kept for its per-sound gain, so it stays as a known cost. The three planner tracks (Final Voyage, Evolution, First Snow) are re-read and rendered on these defaults.

**The synthetic gate, re-run on the final defaults (it had not been run since the YourMT3 and cap changes):** three fails, each attributed by a single-flag variant. (1) *melody 0.07 exact, pad roots 0.30*: YourMT3+ reads 7 % of the gate's 5-harmonic synthetic pluck and 30 % of its sine pad's roots, where the transcription reader reads 93 % / 99 % — a real limit of a model of real instruments on pure synthetic tones (the truth set's GM "synth" patches are still sampled instruments). The gate's `other` checks are plumbing checks and now run on the transcription reader, with this recorded; **for the user's electronic library this is the open question about YourMT3**: a hybrid `other` reader (YourMT3 for what it labels, the transcription reader for stems or voices it reads far fewer notes on than the salience shows) is the next lever to measure. (2) *kick precision 0.80*: the kick-family cap of 2 (`FAMILY_K_MAX`, kept for LarsNet's per-sound gain) merges the synthetic clap's kick-family share into the kick; with the cap at 4 the kick reads 1.00 / 1.00, and the truth set is unmoved (medians 0.87 / 0.69 / 0.72 either way; funk kick 0.76 → 0.85 and its open hat 0.17 → 0.30, house all-hits 0.84 → 0.77) — `FAMILY_K_MAX["kick"] = 4`. (3) Twins on gives five fails (clap 0.32, shaker 0.30 precision) — the rejection above holds on the synthetic too. **Synthetic gate ALL OK** on the final defaults (transcription reader for its `other` checks, kick cap 4, twins off).

A truth-set lesson from the Slakh octave fix: measured on every pitched part, the same pyin test also shifted two organs and a string part by −12 (a 16' drawbar or a sub-octave of a rich tone is what pyin hears), and `other` fell 0.74 → 0.55 and 0.73 → 0.49 on those two tracks with octave-blind unchanged — YourMT3 had agreed with the MIDI. The importer measures BASS parts only, where the offset is systematic and the reader confirms it. A truth file is checked the same way a reader is: by what agrees, not by one measurement.

**2026-09-10, the planner (the user: "the reconstruction takes FOREVER to start playing", then "QThread: Destroyed while thread '' is still running" on close):** the planner they ran had been started before the evening's synthesis change, so it still built and rendered the slow way. Measured on Final Voyage with the current code: stems 2.7 s, build 16–19 s (profiles, chooser, calibration, patterns), all 16 units 11 s, the first unit lands 0.1 s after the build. The build depends only on the stored reading and the build flags, so `ResynthWorker` now stores the program next to the reading (`.stems/<id>/program/`, `SP.save`, 32-bit float sounds — a 16-bit store left a −36 dBFS kit cut nine bits and 0.8 % of peak off; ~100 MB per track, on a par with its stems) with a fingerprint of the reading file, and loads it on every later open: 4 s instead of the build, renders bit-identical to the fresh build. The crash: three worker `done` slots dropped the last reference to a QThread whose `run()` was still returning (the signal is emitted from inside it); they wait for the thread first now. The self-check of the three re-read tracks is the honest quality figure on real material: Final Voyage drums 20.3 / bass 26.0 / `other` 21.0 dB, Evolution 13.5 / 7.9 / **37.8** (91 of 140 bars unexplained), First Snow 13.2 / 13.3 / 21.1 — against 7–15 dB on the truth set. The readers do not hold on this material yet; that gap, not levels, is what the user hears.

**2026-09-10, ninth round ("execute on the plan"): the hybrid `other` reader, measured before it is built.** Both readers on the six truth `other` stems, scored at the note level (scratch `fast_hybrid*.py`, YourMT3 from its cache, the transcription reader 4–8 s per stem on the CPU):

| song | YourMT3 | transcription | best of both |
|---|---|---|---|
| pop80 | 0.29 (P 0.26) | **0.55** | 0.55 |
| house | **0.82** | 0.79 | 0.82 |
| rock | **0.64** | 0.49 | 0.64 |
| funk | **0.82** | 0.64 | 0.82 |
| dnb | 0.55 (P 0.44) | **0.83** | 0.83 |
| ballad | **0.83** | 0.45 | 0.83 |
| median | 0.73 | 0.60 | **0.82** |

The prize is real: choosing the right reader per stem is worth +0.09 on the median. The mechanisms measured: *union* (the transcription reader's notes no YourMT3 note covers, added): 0.64 — precision falls everywhere (pop 0.32: both octaves of the synth voices end up in the list). *Count gate* (the transcription reading replaces YourMT3's when it finds 1.5× the notes): never fires, the counts are alike; median 0.73. *Salience fill* (only the uncovered salient frames filled): 0.70. *Stem-support chooser* (each reading checked against the stem as `explain` checks a voice — the share of notes with salience at their pitch, the share on silent frames — the better-supported reading taken): right on four of six, but the ballad goes to the transcription reader (0.45 for 0.83) because chord tones sit under the chord's loudest tone in salience and the proxy reads a polyphonic reading as unsupported; median 0.71. **All four rejected.** No cheap stem-level signal separates the two cases where YourMT3 loses — pop's voices read an octave off (P 0.26) and dnb's hallucinated notes (P 0.44) — from the songs it wins. **The per-voice octave vote by the other reader — kept** (`YMT3_OCTAVE_BY_READER`, `octave_vote_by_reader`): each YourMT3 voice's notes matched to the transcription reader's by onset, at the same pitch or an octave up / down; a voice whose octave disagreements outnumber its agreements 2:1 (≥ 10) moves. Pop 0.29 → **0.68** (one synth voice: 255 of 256 notes heard an octave down by the other reader, 1 agreed), dnb 0.55 → 0.59, the other four untouched (house 601 agree / 15 down, ballad 121 / 4); **median 0.73 → 0.75**, and identical to the same vote decided by the truth on every song — this mechanism is at its ceiling. Through the reader as `identify` runs it: median F1 0.75, purity 0.72 → 0.73 (the pop voice's purity 0.22 → 0.69, dnb's 0.40 → 0.47), the read-notes render against the true stem 17.3 → 17.2 dB, level +4.0 → +3.4. It costs the transcription reader's pass on the `other` stem (4–8 s on the CPU, less on the GPU). Readings are v14 from here (chord split, hold extension, the octave vote, twins off, kick cap 4). **v14 through every gate**: six-song eval bass 0.87 / 0.97, `other` **0.75 / 0.73**, drums 0.88 / 0.67 (6 min); render gate on a fresh v14 reading all parts 11.1, drums 7.1, bass 10.4, `other` 15.0 dB (unchanged — true notes through the voices do not see a reading's octave); synthetic gate ALL OK; Final Voyage, Evolution and First Snow re-read and rendered on v14 for the planner (their self-check gaps as before: the vote moved no voice on them — the real-track gap is not an octave problem). Where the salience vote failed (a piano's second harmonic reads as "the octave up"), a second reader's opinion does not: it hears the note, not a harmonic. What remains of the gap to the 0.82 oracle is dnb's hallucinated notes (P 0.44 → the transcription reader's 0.83 there) and pop's second synth voice (0.68 against the transcription reader's recall 0.77) — a per-voice precision test is the next candidate, not a stem chooser.

**Real kits (step 3 of the plan), measured on the four Slakh tracks:** drumsep per sound **0.51** (all hits 0.69), LarsNet **0.53** (0.63) — LarsNet's per-sound lead on the GM set (0.85 against 0.69) does not carry to sampled kits, so the separator is not the limit there. Where the score goes: within the cymbal family. Track00002's shaker and hat share one voice (each at precision 1.00 and recall 0.45), Track00001's hat voice holds the rides (hat precision 0.13), open hats sit at 0.00–0.33 and crashes at 0.05–0.37 under both separators, while kicks read 0.89–1.00 and snares 0.58–0.99. The family's timbre clusters (spectrum plus the 23 and 58 ms envelope frames) do not tell a shaker from a closed hat or a ride from a hat on real recordings. The cymbal cap 3 → 5, measured on both sets: **rejected** — no sound gains where it matters (Slakh Track00001 rim 0.51 → 0.59 is the only rise), the GM ballad's all-hits 0.85 → 0.78, house hat 0.96 → 0.92 and shaker 0.74 → 0.66, Slakh Track00003's hat 0.52 → 0.40; more clusters split what the features cannot separate. The diagnostic on the TRUE isolated hits (scratch `diag_cymbals.py`: per sound, the onset energy rise per drumsep family, and inside the cymbal stem the ring to −20 dB, the 50 ms centroid, the late energy share) found three things. (1) Where the cymbal stem holds the sounds, ring and centroid DO separate them: house hat 115 ms / 12.0 kHz against shaker 45 ms / 10.7 kHz, ballad hat 177 ms / 11.9 kHz against ride 300 ms / 10.3 kHz, Slakh Track00001 hat 50 ms against open hat 215 and ride 249. Those two scalars are already in the timbre vector (`sounds.timbre`: decay and log-centroid) but are 2 of ~80 standardised dimensions; weighting them ×4 in the cymbal family (`FAMILY_SCALAR_WEIGHT`) was measured on both sets and **rejected**: house hat 0.96 → 0.97 and shaker 0.74 → 0.77 but its clap 0.90 → 0.73, dnb all-hits 0.70 → 0.67, the ballad's hat and ride stay one voice (0.57 / 0.60), Slakh unmoved. (2) **The separator drops whole sounds**: Slakh Track00002's 371 hats and 391 shakers rise +29 dB at their onsets in the drums stem and produce no onset in ANY family stem; Track00003's open hats the same. (3) Crashes go to the KICK family on both Slakh tracks (isolated crashes: kick 1.00, cymbals 0.00) — a low wash the model files under kick. For (2) a safety net was built and measured: `FAMILY_REST` — the mixed stem's onsets no family claims within 20 ms, gated like a family's, read as a "rest" family from the mixed stem. **It found nothing to read**: on Track00002 the mixed stem yields 331 onsets and every one is claimed (kick 181, snare 180, toms 106, cymbals 0) — the hats and shakers are not dropped from the onsets, they are routed by the separator into the snare and toms families, where they become "perc" and "tom" voices. And 331 onsets for 762 hats and shakers is not a detection loss either: that kit's hat and shaker strike on the SAME 16ths (the full-band detector recalls 0.83 of the distinct cymbal-class moments; a high-band detector 0.75 at worse precision), so they are one sound to any onset reader while the truth counts two — each capped near 0.5, as measured. A fallback to the mixed-stem reader for such kits was measured and **rejected** too: Slakh per sound 0.51 → 0.39 (snares 0.87 → 0.29, toms 0.84 → 0.19). Step 3's conclusion: the four cheap levers (cap, scalar weight, rest pass, mixed fallback) leave real kits at 0.51 per sound under either separator; what is left is coincidence (unsolvable per sound), the separator's routing of cymbal-class sounds into other families, and crashes under kick. The next lever would be learned, and the Slakh truth makes it measurable before it is built: a random forest over the reader's own timbre features plus the per-family onset rise, at the TRUE hit times of nine tracks, tested on the tenth (scratch `diag_kit_classifier.py`). **It does not carry across kits**: isolated cymbal-class hits are classified right on 0.96 / 0.97 of the pop and ballad kits and on 0.12–0.52 of the Slakh kits; over all tracks the classifier calls nearly everything a hat (hat precision 0.48 at recall 0.97; shaker recall 0.05, open hat 0.00, crash 0.00; Slakh Track00004's 375 shakers: recall 0.05). With ten kits of truth, the features that separate a shaker from a hat inside one recording (ring, centroid) do not name the sound across recordings — one kit's shaker rings like another's hat. So a classifier over these features is not the lever either, and step 3 closes where it stands: real kits at 0.51 per sound under either separator, kicks and snares read (0.9–1.0 / 0.6–1.0), the cymbal class split only where a kit's sounds are spectrally apart and do not coincide. What would move it is outside this round: a separator trained per cymbal-class sound (the LarsNet hi-hat class is the only one that exists, and it did not carry), or a larger labelled corpus (all 20 BabySlakh tracks, then Slakh2100) for a classifier with room to generalise. The Slakh render gate: `other` 19.4 dB (n 29; the GM set 15.0) with pads and strings the worst (a synth pad read as a struck voice: 31.5 dB; strings 27–33), bass 13.9; it had no kit row because the importer had split the kit into per-sound parts — it writes one `kit` part in the GM format now, with `true_kit.wav` — and three parts with no notes in the window scored "0.0 dB" and pulled the medians down; the gate skips those now. Slakh gate on the fixed truth: **kits 12.1 dB** (19.1 / 13.5 / 10.8 / 7.1; onset F1 0.74–0.84), bass 13.9, `other` **23.1 dB** (n 26 — the honest figure once the three empty parts no longer count as perfect; the GM set's 15.0), all parts 19.0. On sampled real instruments the render is 4 to 8 dB further from the truth than on the SoundFont songs: the kits by 5 dB, `other` by 8 dB — the same voice-purity and pad/strings-as-struck limits, larger.

**2026-09-10, "the reconstructions sound somewhat like the source but very discordant — can you measure this?"** Two measures now in `fidelity.discord`, per beat the recording is active on: the render's chroma energy on pitch classes the recording does not sound on that beat (*wrong-pitch share*, the recording's own share at the same floor as the baseline), and *sensory roughness* (Plomp-Levelt/Sethares over the beat spectrum's strongest partials, level-free). Measured (scratch `discord_eval.py`):

| | wrong-pitch share render / recording | roughness render / recording |
|---|---|---|
| truth set `other`, six songs, median | 0.38 / 0.27 (pop +0.17, house +0.18, dnb +0.09, rock +0.05, funk +0.01, ballad −0.06) | 0.014 / 0.014 |
| same, only the notes both readers agree on (precision 0.77 → 0.86) | 0.39 / 0.27 | 0.013 / 0.014 |
| Final Voyage, full mix | 0.61 / 0.48 | 0.016 / 0.014 |
| Evolution, full mix | 0.10 / 0.08 (16ths: 0.19 / 0.17) | 0.014 / 0.015 |
| First Snow, full mix | 0.41 / 0.39 | 0.018 / 0.018 |

What this says: the renders are **not rougher** than the recordings anywhere, and on Evolution and First Snow their pitch-class content matches the recording's beat by beat and 16th by 16th; the excess wrong-pitch energy is 11 points on the truth set and 13 on Final Voyage. Raising note precision by nine points does not move it — the "false" notes are timing, octave and duplicate errors at the RIGHT pitch classes. So what is heard as discord is not, mostly, harmonic clash of wrong notes; it is what these measures cannot see and the other figures do: one note in four in the wrong voice (purity 0.73 — a bass line's note on a piano sample), voices holding several instruments, pads and strings read as struck notes, sample voices 4–10 dB off in balance, and octave placement. The voice-level render gate (15 dB GM, 23 dB Slakh, 21–38 dB on these tracks by self-check) is the figure that tracks it; the wrong-pitch share is a useful second reading on sparse material (Evolution baseline 0.08) and a weak one on dense mixes (baselines 0.4–0.5, where most classes count as present).

**2026-09-10, the pivot ("ok, do it"): THE HYBRID PROGRAM.** The target is no longer every stem as notes; it is a program that plays as the song and says which of it is notes. Built on the pieces that existed: the unexplained-residual bar test (`residual_bars`, mean |dB| on a log-mel per bar against the stem, `RESIDUAL_DB` 14) now runs on every note stem (`RESIDUAL_STEMS` drums / bass / other, per-stem thresholds in `RESIDUAL_DB_STEM`), a bar its voices cannot explain is the stem's own audio, and — the part that was missing — **the voices are silent inside those bars** (`verbatim_spans` / `outside_verbatim`, applied in `render` and `mixdown`; before, a note played on top of its own recording and doubled it). A stem's phrases and verbatim slices play only for the stem's own unit (its bare name in `ids`), never once per voice. The program reports what it is: `stats["note_share"]` per stem (the share of the stem's sounding bars carried as notes) and the median bar gap, in `describe()` and the planner's progress line. The planner builds with the residual test on (seconds since the fast synthesis; the stored program's fingerprint changed, so old stores rebuild) and gives every note stem a "recording" unit next to its voices ("drums / bass / other (the recording where its voices fall short)"), audible with the stem's header like the vocals' phrases. The fidelity report renders each stem as voices + recording parts. Measured on both truth sets (scratch `hybrid_eval.py`: from the cached v14 reading, notes-only against hybrid, each stem rendered as voices + recording parts against the TRUE stem over 5–100 s):

| stem | notes only | hybrid | notes on (median share of sounding bars) |
|---|---|---|---|
| drums | 9.4 dB | 9.0 dB | 98 % |
| bass | 12.6 dB | 12.6 dB | 99 % |
| other | 12.3 dB | **9.8 dB** | 81 % |

Where the notes were worst the hybrid gains most: rock `other` 20.4 → 4.5 dB (notes on 71 %), pop 23.0 → 12.6 (62 %), dnb 16.8 → 8.2 (74 %), house 12.4 → 7.4 (63 %), rock bass 11.0 → 6.6 (77 %). Three `other` stems come out 1–1.5 dB WORSE against the truth (ballad 15.7 → 16.7, Slakh Track00001 11.8 → 13.3, Track00002 9.8 → 11.2): the recording parts are the demucs stem, and against the TRUE stem they carry its bleed — a cost the listener never hears (it is the song's own audio) and a truth comparison does. Drums and bass rarely fall back: their bar gaps against the demucs stem sit under the threshold even where the render is 24–28 dB from the TRUE stem (Slakh Track00004 drums, pop bass — separation error, which no fallback to the same separated stem can repair). The product-level judge is therefore the fidelity report against the stems the program was built from and the mix against the original, where the recording parts count as the recording. The report now carries both: `waveform` (the voices alone — the reader's figure, unchanged) and `waveform_played` (the program as heard, with `note_share` per stem), printed by `describe()` as AS PLAYED and shown in the gen console's Fidelity block. The planner's headless mixer test passes with the recording units (21 units in 20 s from the stored program, the first voice at 4 s; `_recon_units` is the one list the mixer and the test share).

**The three planner tracks as hybrid programs (fidelity report, 30–150 s, against the stems the program was built from; voices alone → as played, with the note share):**

| track | drums | bass | other | mix |
|---|---|---|---|---|
| Final Voyage | 20.3 → **7.6** dB (notes 51 %) | 26.0 → 20.2 (70 %) | 20.7 → **6.8** (68 %) | 12.9 → **6.7** |
| Evolution | 13.5 → 8.0 (73 %) | 7.9 → 6.8 (78 %) | 38.2 → **4.0** (35 %) | 11.4 → **3.7** |
| First Snow | 12.9 → 7.9 (73 %) | 14.7 → 7.5 (66 %) | 21.1 → **7.5** (75 %) | 11.3 → **5.4** |

The mixes sit 4 to 7 dB from the original where the notes alone sat 11 to 13, and every stem but one is inside the "same part, other instrument" band. The one that is not — Final Voyage's bass at 20 dB with 70 % of its bars still notes — says the 14 dB per-bar threshold is too loose for that stem: its bars pass against the demucs bass stem and the whole still sounds wrong. A tighter bass threshold is the next thing to measure (`RESIDUAL_DB_STEM["bass"]`), by ear on this track and on the truth sets. Between half and three quarters of every stem is notes; that is the morph material as it stands, and the honest size of it.

GPU note (2026-09-09): two concurrent demucs runs plus the planner's CUDA context oversubscribed the 8 GB card; Windows paged it and two 5-second separations sat for 2.5 hours. `drumsep._device` falls back to the CPU under 1.5 GB free; GPU jobs run one at a time from here on.

Evaluator caveat learned: with tails and room in the render, the onset
detector on the reconstruction under-reports F1 and the envelope lag
column can show a one-step "lag" that is not there (event timing is
verified within 3 ms by expansion); read `off`/`sd` for timing, not
`lag`, and read F1 on the raw-exemplar mode.

## 6. Order of work
1. **Pattern inference + groove template** for drums (the biggest
   compression, the reading is already good enough) — and the
   SongProgram file format with a renderer that plays patterns.
2. **Kit with tails, velocity layers and room** — the biggest audible
   gain on drums.
3. **Bass cleaning and riff inference** (cross-stem kick removal, Viterbi
   pitch, chord-relative riffs, fitted synth voice).
4. **Chord track + pad-as-chord + arp ops**; the unexplained-residual
   test and the verbatim track for FX.
5. **Multi-pitch transcription for keys/polyphony** (basic-pitch fusion).
6. **Vocal phrase library with reuse.**
7. Wire the gen console: Replay renders the program; Recreate composes from it.

Each step has a gate above; none is started before the previous one's
gate holds on the six evaluation tracks.
