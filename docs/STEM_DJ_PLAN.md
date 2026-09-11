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

**Remix mode (2026-09-10, after the user on the Perform tab: "this is still just a fancy automixer"):** the
automixer plays whole songs one seam at a time; a stem system can do what no automixer does — play PARTS
of several songs together and keep changing them. `lib/dj/remix.py` (`RemixConductor`) runs two or three
songs live on the submix's three decks, all on one clock (the first song's tempo; the PLL holds the others,
one session per slave) and shifted toward one key (±3 st). The four stem lanes — drums, bass, other, vocals
— are each assigned to one live song. Every phrase (4 / 8 / 16 / 32 bars, the mix-speed box) the conductor
makes one MOVE on the master's next bar with a two-beat crossfade: a lane crosses to another live song, a
lane rests for a phrase, a new song (the brain's pick against the master, to the theme and the energy
target) enters through one lane, a song that holds no lane leaves. The brain's pick is decoded in the
background and STAGED — started beat-locked at its body with every lane closed — so the PLL has settled
before it is heard. Rules: a tonal lane (bass / other / vocals) only crosses to a song whose shifted key
fits the songs on the other tonal lanes (Camelot ≥ 0.55; drums are free); a new song takes its lane from
the OLDEST song; a song's last lane cannot be taken before it has been heard three phrases; a song under
40 s of body loops its last 8 bars and is evicted after two phrases of looping (its lanes move on a bar
apart; if it was the clock, the song holding most lanes takes it and the others re-sync); a looping song
gives lanes but never takes one; a resting lane comes back at the next move. Controls: **blend** (0 =
every move hands a lane to the newest song, so song follows song as a morph, two songs live; 1 = free
recombination across three), **vocals** (how freely the vocal lane crosses), **theme**, **energy** (the
brain's arc target), **MIX NOW** (a move on the next bar), **HOLD** (freeze the lane map; the runway rules
still apply). The Perform tab has a mode box (Automix / Remix); in Remix the three maps are the decks (song,
lanes held, master / loop / leaving, PLL lock in ms), the lane line shows which song each lane plays, the
lists show what is decoding or staged, entries and exits, and every move.
Gate `tools/tests/_dj_remix_test.py --music D:/Devel/music` (headless, the mixer pulled by hand, ~2× real
time so decodes arrive as they do live; 300 s rendered at a move every 4 bars): 3 songs live, 32 lane
moves, 2+ songs heard 88 % of the time, 6 songs entered lane by lane and 5 left, the harmonic guard never
violated at any block, slave locks median 0.014 / 0.020 beat (p95 < 0.05, n 2498 / 1709), the clock
handed over once (re-lock median 0.016), peak 0.98, no dead bar, HOLD froze the map, NEXT moved on the
next bar — **ALL OK**. Two defects the first run found: loop expiry was counted on the master's bar index,
which wraps inside a loop (now a monotonic bar count), and new songs took lanes from the newest song
(now the oldest). A second seed after the minimum-stay rule: 11 songs entered and 9 left in 300 s (a move
every 4 bars; songs stay the three phrases and go), the clock handed over five times, locks median 0.010 /
0.018 / 0.021 beat, guard clean, no dead bar — ALL OK; it also showed one pick that looped two seconds after
entering (its body starts near its end), so a pick now needs 90 s of body from its entry point. The tab,
headless and silent in Remix mode: mode switch hides the seam controls and
shows blend / vocals, Start builds the conductor on 1073 stem-bearing tracks, five songs came and went in
two minutes, HOLD and MIX NOW reach the conductor. Not yet heard by the user.

**A pinned cut is a cut (2026-09-10, user: "I don't like how cut at drop uses some shitty echo effect
instead of just cutting"):** the cut itself was always a 40 ms gain drop with no echo; what played was
something else. With `cut_at_drop` pinned on 80 random pairs only 29 were honoured — the pin was refused
by `kick_offset>28ms` (26), `cut_drop_shape` (17), `anti_streak` (8), tempo / meter (9) — and a refused pin
fell back to the WHOLE dice menu (long_blend, long_fade, bass_swap, echo_out…), so a night pinned to cuts
played blends and, when the dice said so, the echo throw; and half the honoured cuts wound the platter down
first (the brake coin flip). Now: `cut_at_drop` never brakes and a pinned cut never brakes (the coin flip
survives only on a dice-rolled `phrase_cut`); a pin is exempt from `anti_streak` (the pin IS the request);
a pinned cut crosses the `kick_offset>28ms` bar (earned for the dice — kick delta sorts cut verdicts 68 vs
59 % good — but a pin asked for a slam, and a long blend is further from that than a 28 ms flam in the
run-in; the waiver is shown in the readout as "crossed kick_offset>28ms"); and a refused pin falls back
inside its family before the dice (`_PIN_FAMILY`: cuts → cuts, stem styles → stem styles, echo_out →
long_fade). Same 80 pairs after: 54 honoured, every refusal structural (no drop shape in B 15, tempo clash
6, off-meter 3, beatless 1), the compiled events of the honoured cuts carry no brake and no echo, A leaves on
a 0.04 s ramp. The family fallback rarely engages for cuts because `phrase_cut` shares the structural bars.

**Phases 4 and 5 in Remix mode (2026-09-11, "do phase 4 and 5 and we'll evaluate this evening"):**
*Performing.* Three buttons on the conductor, each a system behaviour on the master's next bar: **DROP**
(every lane to the newest song at once over half a beat — the morph as a cut; a staged song not yet heard
drops in whole), **BREAK** (every lane but the most melodic one held — vocals, else other, else bass —
rests four bars and they all come back on the bar; no song leaves during a break), **LOOP 4 / 8** (every
live song loops that many bars from its own next downbeat, so the whole combination holds; the same
button again releases). HOLD and MIX NOW as before. *The nanoKONTROL2* drives the Perform tab in both
modes, polled from the readout timer (no thread, no device other than MIDI): faders 1–3 = energy lean,
blend, vocals; knob 1 = change rate / mix speed; PLAY = MIX NOW, STOP = HOLD, REC = DROP, CYCLE = BREAK,
MARKER ◀ ▶ = LOOP 4 / 8, TRACK ◀ ▶ = BAD / GOOD. Faders move the on-screen sliders, so the screen is the
truth. *Autonomy.* The energy target is no longer a number you set: it is the theme's arc over a
90-minute cycle (all-night themes over the night) plus the lean, as in the automixer, and the readout
shows the arc position. And the conductor LEARNS: **GOOD / BAD** rate the last rated kind of move (entry,
cross, consolidating cross, rest, return, drop, break); each verdict is stored as `seam_feedback` with
style `remix:<kind>:<lane>` between the songs involved, and every conductor starts from the record — a
Laplace rate per (kind, lane), (ups + 1) / (n + 2), scales that kind's chance (0.5 = as set, 1.0 = twice
as often, 0 = never): the rest chance, the vocal lane's freedom, the consolidating-versus-free split,
the order lanes are tried for crosses and entries. A handful of verdicts nudges; it does not dictate.
*Left out on purpose:* Remix mode on the show's live web page. That page is DJSystem inside
Stories_OGL (visual outstate, analyzer routing, ambient takeover, a queued action channel through the
web controller) and nobody has heard the mode yet; it stays in the planner until the ear verdict, then
the same conductor mounts on the show's engine.
Gate (`_dj_remix_test.py`, extended): LOOP 8 loops every song and releases, BREAK rests three lanes and
restores them where they were, DROP puts every lane on one song, BAD on the break and GOOD on the drop
move the weights (0.50 → 0.33 / 0.75) with the library write stubbed — plus the earlier invariants. The
tab, headless: the real nanoKONTROL2 connected; simulated fader / knob / transport events moved blend,
energy, mix speed, forced a move and armed LOOP 4.

## Rules carried over (they were earned)

- One change at a time, measured on a library-wide sample, never one track; keep only
  measured wins; record every rejection with its numbers.
- The user's ear outranks the statistics; fence a verdict when the execution is suspect.
- Nothing heavy or audible runs while the user is working; one GPU job at a time.
