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

**The big additions (2026-09-11, "what big additions can we make?" → "do them"):**
*Sound.* STRUCTURE: a staged song is cued so its landmark (the groove after its first build, else its
body) lands on the move that lets it in (`ready_k`); a planned move waits up to two bars for a section
boundary of the master; the vocal lane only crosses into a song that is SINGING there and leaves one whose
singing stops; a breakdown of the song holding most lanes may become a break (learned, once per song).
Singing is MEASURED, not trusted: at decode the vocals stem's RMS per section is the song's singing map
(a section sings above a quarter of the song's loudest singing section and above −45 dBFS; an instrumental
never sings) — the ML vocalness covers 755 of 1220 tracks and a song with unknown vocals kept alone on the
vocal lane by an automatic break was three bars of dead air in the gate before this. HYGIENE: songs
holding neither the bass nor the drums lane lose their lows (EQ low 0.25 under 200 Hz), the `other` lane
sits at 0.8 under another song's singing, every lane is trimmed toward the first song's stem levels (±6
dB). SHAPES: moves are planned a bar early; a song giving its last lane leaves through a low-pass sweep to
260 Hz or a one-beat stutter, a lane that rests throws a dotted-eighth echo (deck FX are per deck, so a
shape runs only when the song holds exactly the lane that moves); an echo or sweep rings out before the
deck stops. Anti ping-pong: a lane that moved within two phrases is tried last. The opener comes from the
theme's tempo window (a 70 bpm half-time read once opened a session nothing could join).
*Capability.* TEMPO JOURNEY: a span slider (0–6 %, knob 2) lets the clock travel with the arc, at most
0.5 % per bar, every deck re-rated together so the PLL never absorbs more than its 1.2 % window, no step
past a song's wall (gate: 121.9 → 126.3 bpm with locks holding). SNAPSHOTS: SAVE remembers the lane map;
RECALL brings it back, re-decoding and staging songs that left (a live song outside the snapshot gives
its lanes up to free a deck), then crosses the lanes on one bar; abandoned after four phrases if the
songs cannot come back. RECORDING: REC taps the submix to logs/remix_*.wav and writes the move log and
snapshots beside it as JSON. NATURAL-LANGUAGE STEERING: `tools/dj/planner/perform_copilot.py`
(PerformCopilot on SetCopilot's transports: Claude Code CLI first, no key) with four tools — get_state,
steer, act, rate — behind a SAY line in the Perform tab; the copilot reads the live state and queues
control changes through a bridge the tab drains on its timer, so nothing touches a widget or the engine
off the GUI thread (headless, scripted: "wilder, more vocals, then drop it" set blend 0.95, vocals 0.8,
4 bars, dropped, rated).
*Product.* REMIX ON THE SHOW: Stories_OGL `_remix_start` / `_remix_stop` (the automixer's soundtrack
takeover, the conductor on the show's engine, the analyzer on the mix, the outstate keys in the
automixer's vocabulary — arc, energy target, next-move ETA as the blend ETA, DROP / entry / recall stamped
as drops, a BREAK as the moment hole); web actions `remix_start`, `remix_blend/vocals/lean/tempo/change`,
`remix_act`, `remix_rate` validated and clamped in the web controller (driven through the HTTP twin: bad
actions 400, values clamped); the DJ page's REMIX button and Remix panel (lanes, songs with section /
singing / lock, MOVE / HOLD / DROP / BREAK / LOOP 4 / 8 / SAVE / RECALL, sliders, GOOD / BAD, the moves).
The automixer's page stays idle underneath (`active` false, `remix_active` true), so its renderer is
untouched. Not yet run on the show machine.
*Left out:* the remix partner graph (an offline campaign over song pairs) and a fourth deck for a loop bed
— both wait for the ear verdict on the basic mode.
Gate (`_dj_remix_test.py`, now also structure / hygiene / shapes / snapshots / tempo / recording): seed 7
ALL OK — vocal lane on a silent section 4 % of blocks (transients the next move fixes), lows still open 0 %
of 1759 blocks, shapes echo / filter / stutter, 11 songs staged on landmarks, tempo 118.0 → 122.1, RECALL
3/4, recording 2 s + JSON; seed 11 exposed the unknown-vocals break (fixed by the measured singing map).

**THE INSTRUMENT (2026-09-11, user: "this is still just a fancy automixer… I have almost no information,
control, or ways to steer the system. It feels like I'm calling into a radio station and asking for a type
of song"):** the diagnosis was right - every decision sat in the system and the operator's inputs were
biases on probabilities. The Perform tab is now an instrument the system executes for: **the pad grid**
(lanes down, songs across; a cell puts that lane on that song on the next bar through the same crossfade,
shapes, hygiene and harmonic guard; a lit cell is where the lane is; a grey cell says why it is refused -
"key clash with the other of X", "not singing there now"; a rest column; each column heads with the song,
its tempo and key fit, its section, singing, lock, a **fader**, LOOP 4 and **OUT**); **the crate** (songs
that fit the clock now, ranked - tempo inside the wall, key fit with the shift it would take, energy
against the arc - with the reasons; search; STAGE decodes one onto the free fourth deck, beat-locked and
key-fitted, silent until you give it a lane); **autopilot as an amount** (0 = only you move lanes, 100 =
the conductor moves every phrase; with it off nothing is auto-staged either); **songs** = a saved setlist
as the pool (Remix: the crate and the conductor pick only from it; Automix: the night's pool); verdicts on
the move itself in the feed (👍 / 👎 per row, clearable); the stage-and-deck layout (dark, large grouped
buttons, sliders with values, a phrase bar counting down to the next move, colour per song across grid
and strips, keyboard shortcuts). The nanoKONTROL2 became the instrument's surface: channel strips 1-4 =
decks A-D (fader = that song's level, S / M / R = drums / bass / other to that song; S of strips 5-8 =
vocals to A-D), faders 5-8 = energy, blend, vocals, auto. Conductor API behind it: `set_auto`,
`candidates`, `stage_track`, `why_not`, `assign`, `eject`, `song_loop`, `song_gain`, `rate_id`,
`set_pool`; DECKS is four now. Headless: autopilot 0 for 20 s produced no conductor move; a crate pick
staged on B; drums and bass put on it from the grid; the vocal cell refused with "not singing there now";
a fader; OUT handed the clock over; autopilot 100 resumed the conductor.
*Where the user wants this to go (2026-09-11):* "finer resolution spectral information, entry and exit
points, the ability to create paths from entry to exit points via drag and drop, the ability to see
multiple songs' info at once, and the start of a way to play parts of many songs at once" - an
ARRANGEMENT surface: songs as rows with stem-level spectral detail, their entry and exit points, paths
drawn between them that the conductor then plays as parts of many songs. Next.

**THE TIMELINE (2026-09-11, user: "the interface is not what I've been asking for" → "single ongoing
timeline. but I don't want to have to set mix volumes for many stems"):** the interface the user asked for
is the music itself as the surface. `tools/dj/planner/timeline.py` + `lib/dj/timeline.py`: ONE continuous
run of bars with four lane tracks (drums, bass, other, vocals). A CLIP is a span of one song's stem placed
at a bar, drawn with that stem's real spectrogram (48 log bands at 0.1 s, computed once from the stems and
cached beside them) in the song's colour; one clip per lane at a time (placing ends what was there), so
levels, beat lock, key shift and crossfades are the conductor's and never the operator's; a clip runs
until the next clip on its lane unless given an end. Drag to move, drag the right edge to end,
double-click to end / run on, Delete, Ctrl+wheel zoom, click the ruler to set where PLAY starts; the
playhead keeps running (ongoing: place the next thing when you want a change). The MATERIAL below: the
setlist (or a library search) and the SONG VIEWER - four spectrogram rows in song time, the 4-bar grid,
sections, the analyser's entry (green) and exit (red) points, the song's live position; drag a stem row
onto a lane, or the title bar for all four stems; SEND places the song at the playhead. The player
(TimelinePlayer on a RemixConductor at autopilot 0) stages each clip's song ahead so its song time lands
exactly on its bar (conductor: `stage_for_bar`, `start(cue_s, lanes)`, `jump`, `assign(force)`), puts
lanes on decks at the bar, re-cues a live song for a clip elsewhere in it, ejects songs no clip needs.
Headless: whole song A at 0, B's drums+bass at 8, C's vocals at 16, A's other ending at 24 - the lane map
followed bar by bar, songs staged ahead, A left when nothing needed it; spectrograms 5.7 s per song the
first time, cached after. Not yet heard by the user. Open next: entry / exit handles the operator drags
and adds; clip trimming at the left edge; a lane's own fader only as an exception; the offline moment
finder feeding the material; autopilot drawing ghost clips ahead of the playhead.

**The live layer on the Timeline (2026-09-12, user: "once again this is getting way too complicated for
spontaneous live shows"):** placing clips by hand is preparation, not performance. Live, the SYSTEM writes
the timeline and you approve or override in a glance: the player's **autopilot** (0–100 %, the `auto`
slider) plans the conductor's next move for each coming phrase two phrases ahead of the playhead and
writes it as a **ghost clip** (dashed, "auto ·") - a new song in through one lane (the brain's pick,
from its landmark), a lane crossing toward the newest song or freely, a rest - which plays like any clip
unless you delete or move it; a bar you placed something on yourself is left alone; an EMPTY timeline
with the autopilot up opens on the conductor's own pick. Four **gestures** write clips the same way:
NEXT SONG (the chosen song in the way the conductor would - a lane a phrase in the morph order from the
next phrase), DROP (every lane to the chosen or newest song on the next bar), BREAK (every lane but the
most melodic rests four bars), HOLD (no new plans; what is placed still plays); keys N / D / B / H.
Precedence on a lane is now IMPLICIT (the latest-starting clip that covers a bar is heard; an earlier clip
resumes when a bounded one ends; dragging a clip away leaves nothing trimmed - the first drag crashed on a
clip object the move had replaced, and the trim-on-place rule left holes), the canvas draws the heard
segments, and RESTS are clips of silence. Headless: an empty timeline at autopilot 100 opened on a pick,
planned "Kirghiz in through drums at bar 4, Look Of Today in through bass at 8, drums → Look Of Today at
12, Need You Now in through other at 16…" and the plan became the music; DROP put every lane on one song;
BREAK rested three lanes and they came back; HOLD stopped new plans. Not yet heard by the user.

**THE DIRECTOR (2026-09-12) - the agreed surface.** After "I can't even tell what you're trying to build,
we need to basically start from the beginning", the design restarted from the show: parties and events;
the autoDJ preserved as is; "a semi-active system where I control high level behavior. What kind of
songs, maybe the next few songs (not always though), what kind of mixing, and potential high level
performances: are multiple songs being mixed, are we hard cutting, are we amping up the baseline, are we
playing songs for short or long bits, are we looping things. I don't want to deal with arranging things
like individual stems or their timing or volume." Agreed on paper first: six dials and song selectors.
`lib/dj/director.py` owns one engine at a time - the autoDJ (`DJSystem`, behaviour untouched) or the
stem conductor (`RemixConductor`) - and translates: **songs** (theme, a playlist as the pool, UP NEXT
honoured when it can be), **mixing** auto/blend/cut/morph (seam-family pins; the conductor's crossfade
0.5/2/4 beats), **layers** one/two/three (which engine; the conductor's blend 0.3/0.85), **energy**
cool/hold/amp (the arc lean ±0.25; a 3 % tempo journey when amped and layered), **pace** short/normal/
long (`DJSystem.set_pace` 0.6/1/1.5 on the drawn play length, still capped; the conductor's change rate
4/8/16 bars), **loops** off/some/lots (`set_loop_bias`: loop_in / loop_roll_exit / loop_build ×1/×3/×8
through the brain's style multipliers; layered, the clock holds four bars now and then). Moments NEXT /
HOLD / DROP / BREAK, a verdict. Changing LAYERS between one and two/three HANDS THE PLAYING SONG ACROSS:
the other engine opens on the same song at the same song time on a bar of the song (the conductor holds
its decoded opener for `open_pending`; the autoDJ takes a prepared opener via `set_opener` and its live
thread is spawned after that first step) and the two master buses crossfade over a beat. The tab
(`tools/dj/planner/director.py`): left in words - NOW, NEXT, THE DIRECTOR INTENDS, WHAT JUST HAPPENED;
right the six dials as segmented buttons, UP NEXT, the four moments, GOOD / BAD. The Perform (grid /
crate) and Timeline tabs are off the tab bar; their code stays as engine tools. Headless: a queued song
became the autoDJ's next; LAYERS two handed the playing song to the conductor in 12 s (lanes then crossed
to two more songs); cut / short / lots / amp applied live; LAYERS one handed the same song back at the same
position in 10 s and the autoDJ kept playing with the pins set. Not yet heard by the user.
*The picture (same day, "there needs to be some kind of timeline view with a visual representation of
what is going on"):* the Director keeps its run as a Timeline and the tab draws it read-only above the
words - the four lanes across bars with what played as spectrogram clips (one-song mode: each record
fills the lanes from the bar it started; layered: every lane change is a clip from that song's position,
rests dark), the playhead on the Director's bar clock across both engines, and the plan ahead as dashed
ghosts (the autoDJ's next song at the seam's projected bar; staged songs on their entry lane).

**The Director at the controls (2026-09-12, the user's first hours on it):** "it shouldn't be taking so
long to switch directions. I'm not trying to steer a container ship" → any dial turn in one-song mode
brings the next seam forward once the record has played 45 s (pace re-draws the record's hold at once);
layered, a dial turn forces a move on the next bar; the handover song is kept WARM in the other engine so
switching layers is a bar or two (2–4 s measured, 7–9 s back); the conductor keeps one song staged and
ready. "It isn't clear why the DJ is doing things, or how I can steer it" → WHY (the next song's reason,
the seam and why, a refused dial named with what plays instead; layered, the last move's reason and why a
song is staged) and WHAT YOUR DIALS ARE DOING RIGHT NOW (per dial, the concrete effect this moment); a
pinned cut is BINDING (a refused drop cut becomes a phrase cut through anything but a missing grid; 80
pairs: 55 drop cuts + 14 phrase cuts, 11 fades for tempo clash / off-meter / beatless). "We probably need
several more controls" → TEMPO (slower / hold / faster), SEAMS (quick / normal / long mixes), VOCALS (none /
some / lots), VARIETY (close / varied / wild) - ten dials, each with its one-line caption. "It isn't clear
what these do" → the moments say what they do in the mode we are in ("NEXT SONG - move on at the next
phrase" / "NEXT MOVE - the conductor's next lane move on the next bar"; "STAY"; "DROP - build and land on
this song's drop" / "every part to the newest song"; "BREAK - strip to one part for four bars"; "THAT WAS
GOOD - the DJ does more of it"). "The song search shows nothing when there is no text, and it doesn't
show if songs are likely to be rejected" → the list shows the whole pool with an empty box, ranked by fit
to what is playing, each row ✓ mixable from here or ✗ would be rejected with the reason (tempo out of
reach, key clash, loose grid, no stems), refreshed as the song changes. "Looping is bad" → eight-bar
holds on grooves (lots: also breakdowns) at phrase boundaries, released on the bar; one-song mode biases
only the clean loop entry. Open: the right column is tall (ten dials with captions); the remaining fade
cases under a pinned cut; the user's verdict on the handover's sound.

**Same evening: "the system keeps reusing songs" and "get rid of the fucking stutter cut".** Repeats:
every handover built a fresh engine whose brain knew nothing of the other's plays, and the conductor never
logged its plays to the history the autoDJ seeds from - now the Director keeps ONE played list for the
night, seeds every new brain from it, the conductor refuses anything on it and logs its plays to
play_history, and the song list marks "✗ played N min ago". The stutter: the DROP moment's build was the
dying deck's loop shrinking 1 → ½ → ¼ beat under a synthesized snare roll, then an impact sample on the
landing - removed; the build is the music itself (the high-pass sweep and a push on the dying track), the
one-beat hole, and the next track's drop cold. (The loop-roll seam styles were already retired on the same
verdict in August; the loops dial in one-song mode therefore has almost nothing to bias - open.)

**Later the same evening: the layout, the play, and a philosophy of play.** "The button and text layout is
bad… messed up fullscreen / doesn't show everything", "too much explanatory text that could be in
tooltips", "the interface rolls off the screen while large parts of it are taken up by nearly pointless
reporting windows", "buttons are way bigger than they need to be" (then: "I'm not asking for less buttons,
I'm asking for the button size to be smaller") → the tab is compact: no scroll areas, every caption a
tooltip (the dial's live effect is the tooltip's first line), dial buttons 22 px in two columns, the
picture on top, NOW / NEXT / lanes / WHY and the song list + UP NEXT on the left, dials / MOOD / moments /
verdict on the right; fits 1900×1000 without clipping. "The DJ needs a philosophy of musical play that isn't
just schizophrenically jamming everything together at random" → the conductor's ARRANGEMENT policy: at any
moment one BED (drums + bass of one song, moved together) and one or two VOICES (other / vocals of other
songs) over it; a voice that has been heard two phrases may take the bed (its drums + bass slide under, the
old bed's song leaves after a phrase, its vocals come back "the bed is whole again"); housekeeping rests a
vocal lane whose song stops singing; the old free policy stays behind `policy="free"`. The move log now reads
as a story (arrives as a voice → the bed passes → leaves → vocals return); gate ALL OK: 0 dead bars, guard
0 violations, low carve 0 %, locks p95 ≤ 0.06 beat. "I need feedback when I hit the bad button that it
actually does things and can improve based on it" → GOOD / BAD print under the buttons what was rated, what
the DJ learned (the move-kind weight, 0.50 = neutral) and what it did - BAD also acts at once (one song: the
seam comes now; layered: a different move on the next bar). "I need more options, like bass boost, moment
level, and things about how the music will be played, not just how it transitions" → five play dials: BASS
(flat / boost / heavy) and TONE (dark / neutral / bright) on a new mix-bus EQ in the submix (bit-exact bypass
while flat), LEVEL (quiet / normal / loud) on the bus gain, MOMENTS (rare / some / lots: the DJ's own
double-drops into the next song once per record, and layered its own breaks and drops onto a heard voice),
FX (none / some / lots: filter sweeps and echoes on lane moves; none = clean crosses only). "If we're playing
multiple of the same stem layer from different songs simultaneously, I'd like an indication" → ⚠ DOUBLED in
the state line whenever two songs' stems sound on one lane (a crossfade in flight, or a blend). "Probably
need more moods" → 36 MOOD chips from the library's own tags (≥ 5 songs each); lit chips restrict both
engines and the song list. Fourteen dials now; the buttons follow the Director's dials, so a dial turned by a
script or the copilot shows. Open: the user's ear on all of it; whether fourteen dials is already too many
for a live show; the MOMENTS dial's auto-drop rate.

**A philosophy of play (2026-09-12, evening, `/goal improve the philosophy of play`):** researched how
DJs choose and mix (measured mixes: Kim et al. 2020 on 1,557 mixes - tempo barely touched, key almost never
transposed, transitions in multiples of 32 beats, ~4 min per track; Kell & Tzanetakis 2013 on 114 Essential
Mixes - timbral continuity, tempo and loudness held, key not significant, evenly sized steps beat
nearest-neighbour ordering; practice literature - phrase mixing, the payoff before the exit, 2.5 / 3.5 / 5
minute plays, waves not constant peaks, one or two big tricks a set, texture contrast, follow what works).
Written up with sources and a principle → mechanism → dial table in `docs/DJ_PHILOSOPHY_OF_PLAY.md`. Code:
the conductor's arrangement follows the songs' own structure - a voice takes the bed AT ITS OWN DROP when
one comes within a phrase (`_drop_in_bars`), by five phrases regardless; a new bed SETTLES three phrases
before the next voice arrives, nothing arrives while the bed builds to its drop, a breakdown of the bed opens
the door early; with MOMENTS on and mixing auto / cut, a bed change is a four-bar BREAKDOWN first (the old
bed's drums + bass out) and the new drums and bass land as a cut (`_bed_to`, `strip_before_bed`); the
conductor says why a phrase passed without a move (`wait_why`) and reports the arrangement (bed, settle
left, each voice's drop ETA). The Director spaces its own moments (8 / 4 minutes since the last moment, yours
or its own) and at "some" makes them only while the arc is warm; GOOD / BAD now also teach SONG choice (the
rated song's tags as prefer / avoid leans on both brains, halving per verdict); the intent line reads the
arrangement in words ("bed: X for 12 bars, settles 12 more"; "voice Y takes the bed at its drop in ~6 bars";
one song: "arc building (target 0.62)"). Also fixed a live crash the user hit: a song given a voice lane
outside the policy (an evict, a recall) had no `voice_since` and the subtraction killed the Director thread.
Gate: strips on, checks for bed changes at the voice's drop and with a breakdown first. Open: the settle /
voice / strip numbers are first guesses from the literature - the user's ear at a party decides them.
*"We need arc info and control" (same evening):* the ARC STRIP under the top row - the night's plan as a
curve (the ENERGY lean on it), the songs that played as dots at their energy, the playhead with the target,
quarter ticks in minutes, and the words ("ARC waves · building · 32 of 90 min · target 0.62 · last song 0.58
· peak (0.94) in 22 min"); click on it = "we are here" (both engines' set clocks move: `set_arc_progress`),
drag up / down = bend the plan there (13 waypoints, neighbours follow; the ARC dial becomes "yours"). Two
dials: ARC (theme / steady / build / waves / down / yours - a chosen shape is sampled over the theme's energy
floor and swing and handed to both engines as waypoints; the conductor got `_arc_base` / waypoints / a length
/ a progress jump of its own) and LENGTH (45 m / 90 m / 3 h / night → `set_set_length` and the conductor's
`arc_len_s`). The autoDJ's waypoint cap went 8 → 16 for the 13-point curves. Also from the gate: a voice
lane is never taken from a voice that has not had its turn (new voices were displacing the previous one
before it could take the bed); a voice that can never take the bed (bass clash) fades out after its phrases;
a BREAK keeps the stem that is measured loudest over its span (a kept `other` that fell silent gave dead
bars), refuses when nothing is loud enough, and your BREAK ends a running auto break; auto breaks never come
while loops hold, within two phrases of a DROP, or while a bed change is in flight; the strip before a bed
lands only when the voice is audible through it.

**The measurement work (2026-09-12, night): what is inside the parts.** "Is the system aware of the song
components?" - it knew the parts a DJ works with (four stems, section labels, the grid, a whole-track key
and chroma, a vocal curve) but not the content inside them: which part is the hook, what chords sound in
each section. `lib/dj/measure.py` reads both from the stems on disk (no GPU, ~4 s a track, resumable, run
with `dj_scan.py --measure`; stored in tracks.axes and preserved across rescans): SECTION CHROMA - a
12-bin profile per section from bass + other + vocals, silence gated out - and the HOOK - per-bar loudness
and a chroma + timbre signature of the vocal stem, four-bar windows scored by loudness × repetition (self-
similarity), an ML "chorus" label as a bonus, the ML vocal curve to tell bleed from singing (a demucs
fraction: 0.1 is a sung chorus, instrumentals read 0.0), every occurrence listed with the FIRST as the
payoff. Checked on three songs: a vocal house track's hook landed on its ML chorus (63 s, again at 171-192
s), two instrumentals got none. Consumers, all evidence-gated: the conductor's tonal guard and `why_not`
compare the sections that actually overlap ("the chords clash right now… keys agree, the sections do
not"); a voice takes the bed at its HOOK or its drop, whichever comes first (`_payoff_in_bars`); the
brain's seam score refines the key term with the chroma at the planned out / in points; the song list says
"hook at 1:03 ×5". Gate ALL OK with the guard live (4 bed changes, 2 at the payoff, 0 dead bars). The
library pass is running as this is written.

**Uncompleted parts of the plan, continued (same night):** Phase 4's surface - the nanoKONTROL2 drives
the Director (faders 1-8: energy, tempo, pace, seams, vocals, variety, bass, level, each fader's travel cut
into the dial's options; knobs 1-8: mixing, layers, loops, moments, fx, tone, arc, length; PLAY / STOP /
REC / CYCLE = NEXT / STAY / DROP / BREAK; TRACK < > = BAD / GOOD; MARKER < > = arc back / ahead 10 %) -
and the Director on the show's web page (★ DIRECTOR next to REMIX: the arc strip (tap = we are here),
NOW / NEXT / intent / lanes / WHY, the four moments, GOOD / BAD with the verdict line, every dial as a
button row; `director_*` actions through the one queued channel; the visuals get the arc heat and the
playing engine's outstate). Neither has been run on the show machine or with the controller plugged in.
The loops dial in one-song mode now reaches the autoDJ's shelved LOOP LAYER (Phase 3's un-shelving, in
the dial's terms): once per record at most, on a groove, after the record's payoff (first drop or hook)
and with a minute of runway, a drum loop cut from another record rides under the playing one on deck C
(some = a third of records, lots = most); the tooltip and WHAT YOUR DIALS ARE DOING say when one is
riding. Never heard by the user with real DJ ears - the layer was shelved for exactly that reason, so the
dial's default stays off. Still open: the remaining fade cases under a pinned cut, the settle / voice /
strip numbers, and the user's ear on all of it.

## Rules carried over (they were earned)

- One change at a time, measured on a library-wide sample, never one track; keep only
  measured wins; record every rejection with its numbers.
- The user's ear outranks the statistics; fence a verdict when the execution is suspect.
- Nothing heavy or audible runs while the user is working; one GPU job at a time.
