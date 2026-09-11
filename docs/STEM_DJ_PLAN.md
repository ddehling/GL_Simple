# The stem-level DJ system — plan (2026-09-10)

## Where this comes from

Two days of measured work on reconstructing songs at the note level (readers per
instrument, a SongProgram renderer, truth sets with known notes, render and discord
gates) ended where the numbers said it would: kick, snare and bass lines read well
(0.87–0.94 notes F1), vocals reuse well as phrases, and everything polyphonic in the
`other` stem sits at the ceiling of current transcription (0.75 F1, one note in four in
the wrong voice) and does not render as the song. The user's verdict on the best of it:
"an angry child slamming piano keys". That work is archived, complete with its plan and
every rejected lever, on `claude/generative-music-integration`
(`docs/RECONSTRUCTION_PLAN.md` there). Nothing of it is needed here.

## The goal

High-level DJ software: the technical work — mixing, blending, morphing, looping — is
done by the system; the performance is chosen by the user from sliders and buttons. The
unit of material is the **stem** (drums / bass / other / vocals per track, separated by
demucs), never the note. Morphing between songs is a per-stem handover, not a
reconstruction.

## What exists (on `main`) and stays the foundation

- Per-track analysis: beat grid, downbeats, sections, key, energy, vocal regions
  (`lib/dj/analyze.py`, `dj_structure.py`, `dj_chroma.py`, `dj_rhythm.py`).
- Stems on disk per track (`dj_stems.py`, `lib/dj/stems.py`).
- Decks, submix and seam styles with their measured gates (`lib/dj/deck.py`,
  `submix.py`, `brain.py`, `system.py`); the campaign harness and ear-verdict data.
- A shelved loop layer (`lib/dj/looplayer.py`, `tools/dj/planner/layerlab.py`).
- The planner (`tools/dj_planner.py`): library, analysis view with stem lanes, set
  builder, mix timeline, seam lab, discover, nights.
- The live tab with interaction panels, and the nanoKONTROL2 MIDI integration.

## Phases — each has a gate before the next starts

1. **Stem decks on one clock.** Any two tracks, four stems each, time-stretched to a
   master tempo and pitch-shifted within a key window, per-stem gain / EQ / filter,
   every action quantised to bars. *Gate:* offline renders of arbitrary pairs at the
   master tempo with per-stem faders; beat alignment by the existing flam meter; no
   audible stretch artefacts within the allowed range (to be fixed by ear, expected
   ±8 %).
2. **Morph as a per-stem handover.** One morph control from A to B drives an ordered
   handover — drums first, then bass, then the rest, vocals last, or any order the user
   sets — each a bar-quantised crossfade or cut under a harmonic guard (key and chord
   compatibility; pitch-shift or hold when incompatible) and energy matching from the
   submix. *Gate:* the campaign harness invariants, the key-clash meter, and ear
   verdicts on twenty pairs.
3. **Loops and cues from structure.** Un-shelve the loop layer: bar-accurate stem loops
   cut from sections (A's drum loop under B's intro), one-shot cues (drop, break,
   echo-out), all quantised. *Gate:* the flam meter on loop seams; twenty judged renders.
4. **The performance surface.** Macro sliders (energy, morph, vocal presence, texture),
   buttons (drop, break, loop 4 / 8, next), on the nanoKONTROL2 and the live tab. The
   system chooses the technique from the state; the user chooses the moment. *Gate:* a
   night's log through the arm pre-flight harness.
5. **Autonomy on top.** The brain plans sets from these primitives; the sliders bias it.

## Rules carried over (they were earned)

- One change at a time, measured on a library-wide sample, never one track; keep only
  measured wins; record every rejection with its numbers.
- The user's ear outranks the statistics; fence a verdict when the execution is suspect.
- Nothing heavy or audible runs while the user is working; one GPU job at a time.
