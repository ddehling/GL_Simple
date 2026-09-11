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

## Status

**2026-09-10, Phase 1 read against the engine:** it already exists. `lib/dj/deck.py` is a stem deck —
stems attached to the track, per-stem gains with ramps summed before the stretcher, keylock engines
(Rubber Band R3, WSOLA, phase vocoder, varispeed), key shift, loop with an equal-power seam, brake,
three-band EQ, sweep filter, echo; `lib/dj/submix.py` schedules sample-accurate events on three decks
(A, B, and C the loop layer) with a beat PLL, transactions and a duck. What Phase 1 lacked is
per-stem EQ/filter, which no phase needs yet. So Phase 2 starts at once as a **transition style**,
`stem_morph` in `lib/dj/brain.py`: the blend is one bar-quantised handover per stem in
`MORPH_ORDER` (drums, bass, other, vocals; `plan["morph_order"]` per seam), B's stem coming in as
A's goes out over `swap_beats`, both decks EQ-flat (the stems are the carve), the bass handover the
point of no return, the vocals always sequential and the melodic stem sequential on off-key pairs
(the harmonic guard from the existing Camelot fit). It goes through every existing gate (stems on
both sides, grid confidence, tempo wall), the offline renderer (`lib/dj/audition.render_seam`) and
the Seam Lab's style pin, so it is measured with the same instruments as every other style. The deck
now keeps an in-flight stem ramp's destination when the next ramp arrives (it froze it before).

**First renders of `stem_morph` (2026-09-10, four compatible pairs through the offline renderer, per-bar
level across the blend relative to A's last full bar before it):**

| pair | entry | level across the blend |
|---|---|---|
| Side by Side → Mirador | intro 30.5 s → groove 61.5 s | flat, worst bar −1.8 dB |
| Side by Side → Birds Mind | 62.5 s → groove 109 s (a build at 94 s first: one bar −7.1 dB) | flat, worst −1.9 dB |
| Natural Cause → Zula | already in a body | +1 dB (the reference bar is a dip in A's outro) |
| Side by Side → A Walk in the Deer Park | 24.7 s → groove 239 s | flat, worst −1.8 dB |

Two things learned and kept: (1) **a morph enters B's body, not its intro** — the mix-in point every
blend uses put the room 12 dB down by the blend's end, so the plan places B's first groove (a groove
before a build: the build's bar dipped 7 dB) at the drum handover, one bar into the blend, when B keeps
120 s of runway; (2) the handover schedule itself (drums 4, bass 12, other 20, vocals 28 beats of 32;
vocals sequential) renders as scheduled. Measured and left neutral: a lean of both decks through the
middle (`MORPH_LEAN`) — built on a misread reference bar, null on the clean pairs. Renders in
`logs/dj_morph_<a>_<b>.wav`; the Seam Lab pins the style by name for the ear.

**Phase 3, the stem stage (2026-09-10, built after the user's verdict on the morph: "generally satisfied
by the stem mixing, but this is just a mix"):** `lib/dj/stage.py` + the planner's **Stage** tab
(`tools/dj/planner/stage.py`). Up to four LANES, each one stem of one track looping a bar-aligned
section (4 / 8 / 16 bars or the whole section) on its own deck, all on one clock: the first lane in
is the master (its tempo and key), every later lane is stretched to it (inside the deck's 0.90–1.10
wall, else refused), key-shifted toward its key within ±3 semitones with a harmonic guard (a melodic
stem whose best fit stays under 0.55 Camelot compatibility is refused unless allowed), brought in and
taken out on the master's next bar, and held on the master's kicks by the submix PLL — which now
runs one session per slave (`DJSubmix._syncs`; the proven one-slave body runs unchanged once per
slave, the two sync gates still ALL PASS). When the master leaves, the next live lane takes the clock
and the others re-sync to it. The stage does the technical work; the tab exposes only track, stem,
section, bars, level, IN / OUT. Real-time through the same AudioEngine + DJSubmix a night runs.
Gate (`tools/tests/_dj_stage_test.py`, three stems of three tracks through the real mixer): lock
median 0.037 and 0.015 beat with p95 0.045 / 0.055 (n 266 / 224), re-lock after the handover median
0.038, no bar-to-bar hole while lanes hold, peak 0.82 — **ALL OK**. Deck: `set_stem_gains` keeps an
in-flight ramp's destination for stems the new call does not name; submix: a `pitch` command.

**The stage as played (2026-09-10, after the user tried the four-lane tab: "nothing is loading, and it's
amazingly overly complicated. I don't want to have to find multiple stem tracks across different songs"):**
two defects, one of design. The tab had copied the library when it was built, before the planner loads it,
so its track box was empty in the real window (and my headless test had handed it a loaded library);
it reads the planner's library when used now. And four lanes from four songs is not how anyone plays
this. The Stage tab is now the **pair stage** (`lib/dj/pairstage.py`): song A is the clock and the key,
song B runs beat-locked to it on the submix's proven one-slave path (stretched inside the wall,
key-shifted toward A within ±3 semitones), and each of the four stem lanes is an **A / B / off** switch
that flips on the next bar with a one-beat crossfade — any mixture of the two songs is a state you hold.
**MORPH → B / → A** hands the lanes over in the default order at 4 / 8 / 16 beats apart (the morph style,
live); **loop A / loop B** hold a song on 4 / 8 / 16 bars of where it is; Play starts both songs at their
bodies and opens the device itself. The clock line shows B's grid lock against A in ms (the PLL's own
error, not the wide audible meter). Driven headless through the real engine with the library arriving
after the tab was built: load, Play, lanes, MORPH both ways, loop, stop — all as scheduled. The N-lane
engine (`lib/dj/stage.py`, gated) stays as the general form for when more than two songs are wanted.

**Phase 4, the performance surface (2026-09-10, after the user on the pair stage: "it lets me pick 2 songs
and morph from one to another? That's not what I want, this is way too simple" — then: "I want theme
control, and I want the ability to trigger mixes, mix speed, and type"):** the target is the SYSTEM as the
DJ — it chooses, mixes, morphs, loops — and the user steers. Both hand-operated stage tabs were the wrong
layer and are unregistered (their engine code stays: the system's stem vocabulary). The **Perform** tab
(`tools/dj/planner/perform.py`) runs the autonomous `DJSystem` on the real engine inside the planner and
exposes: **theme** (the brain's theme, live), **mix type** (a style pinned for every seam through the same
gates — `DJSystem.set_mix_style`; auto = the dice), **mix speed** (short ×0.5 / normal / long ×2 / marathon
×3 on every overlapped style's blend length, whole phrases — `set_mix_speed`; cuts and fades keep theirs),
**MIX NOW** (the next transition now), **HOLD** (one more phrase), **REROLL** (another next track), **DROP /
NEXT DROP** (the night's moments), **ABORT MIX**, and an **energy** lean. The readout: state, what plays,
what comes next with the planned style and blend length, seconds to the blend, the last seam's verdict.
`stem_morph` sits in the mix-type menu, so the stem morph is one setting away on a real set. Proven on
one, headless and silent (the engine's mixer pulled by hand — a test must never open the device; one did,
and played in the user's room): the system started on 1073 tracks, MIX NOW with the pin set gave
`stem_morph` at 16 beats (32 × the short setting), armed, executed, and the seam's own verdict was clean
with a maximum phase error of 0.01 beat. Then the user: "the perform tab needs way more information" —
the readout now carries the playing track's map (sections by kind, energy, playhead, the planned exit and
the blend window), the next track's map with its entry and window, the seam as planned (style, beats,
seconds to the blend, exit and entry times, tempo and key shift, pair score, whether the pin was honoured
and why not, the length scaling, the morph schedule, the "why" chips), the night's arc with the current
position and target, the horizon, the history with verdicts, the system's own event log as it happens,
and the engine line (both decks, sync bias and audible error, resnaps and nudges, level, errors).

## Rules carried over (they were earned)

- One change at a time, measured on a library-wide sample, never one track; keep only
  measured wins; record every rejection with its numbers.
- The user's ear outranks the statistics; fence a verdict when the execution is suspect.
- Nothing heavy or audible runs while the user is working; one GPU job at a time.
