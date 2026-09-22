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
s), two instrumentals got none. The library pass ran the same night: 1073 of 1073 stem-bearing tracks
measured, 588 with a hook (the rest read as instrumentals). Consumers, all evidence-gated: the conductor's tonal guard and `why_not`
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
dial's default stays off. Still open: the remaining fade cases under a pinned cut (80-pair probe unchanged:
55 drop cuts, 14 phrase cuts, 11 fades - 6 tempo clash, 5 off-meter), the settle / voice / strip numbers,
and the user's ear on all of it.

**The user's ear, first verdict on the philosophy (2026-09-12, late): "too busy, too happy to have large
sections of vocal rest, and some clashing."** Busy → settle 3 → 4 phrases, a voice 2 → 3 phrases before it
may take the bed (6 at most), the random "breath" rests removed (a phrase with nothing to do is a phrase
of music). Vocal rests → the cause was the singing test: it read the vocal stem per SECTION (a verse under a
quarter of the chorus's level counted as silence, so the lane rested for whole verses) and the lane only
came back at the next phrase move; now `_singing` reads the stem's own half-second envelope two bars ahead
(singing above 0.12 of its loud level, silent below 0.05, the section map and the ML curve only as
fallbacks), and `_vocal_tick` keeps the vocal lane every BAR: rest when the singing stops for two bars,
back on the bed the bar it sings again. Clashes → a song ARRIVING on a tonal lane must fit the room's keys
at 0.68 (what already plays keeps the 0.55 tolerance), staged songs are shifted two semitones at most (three
sounded wrong), the conductor's pick refuses songs that cannot be shifted to fit, and a melody under
another song's singer steps back to 0.6 (was 0.8). The gate gained the opposite vocal measure: the lane
resting while the bed sings (< 15 % of singing blocks). The section-chroma guard bites more as the
measurement pass covers the library (it finished the same night: every stem-bearing track measured).

**The picture and the picker, second pass (2026-09-12, late night).** "The timeline view keeps weirdly
morphing" - two causes: a live clip that grows a pixel a tick was re-rendered at a new scale every tick
(now spectrograms render on a fixed 8-bar lattice from the clip's start and only more of it is revealed),
and the plan ghost was removed and re-added every tick at a jittering bar and, layered, on whatever lane the
free policy would give (now one steady ghost on the voice lane at the bar the settling allows, kept while
the plan is the same song within a bar). "It also can't make up its mind" - every dial turn and every
GOOD / BAD re-sent all steering to the autoDJ, and flavor / persona / arc / pin each make it drop its NEXT
and pick again; now steering reaches an engine only when it changed (`_send`), GOOD never replans, a dial
turn nudges the next phrase decision instead of forcing a move, dial-forced seams come at most every two
minutes after 90 s of play, and an arc drag applies a second after the hand stops. "More info on the
timeline: zoom, song section info, stem volume info, where songs morph / mix, how the system is building
across songs" → zoom − / + / now / follow buttons (Ctrl+wheel still zooms; a hand scroll turns follow off);
a SECTION STRIP along each clip (intro / groove / build / breakdown / outro by colour, the drop as a white
tick, the measured hook as a gold line); a LEVEL METER per lane, the level heard each bar; and the STORY
ROW above the lanes: what happened, positioned in time (a song opens, arrives as a voice, the bed passes at
its hook, fades out, a DROP, a breakdown → new bed as a band over drums + bass, the autoDJ's seam as a band
with its style and length) and what is planned, dashed ("bed settled: a voice may arrive", "X may take the
bed at its hook", "seam: long_blend 32 beats → Y"). "The next song picker should be based, at least in
part, on the last song in the next song list" → with songs queued the list ranks after the last queued
song (the header says so) and queued songs are marked. "The song picker presents a wall of songs… would
help if we could add more steering to it and more ways of knowing what the song is" → a filter row
(instrumental / vocal, calmer / same / hotter and slower / same / faster than the reference, hook only,
unheard tonight on by default) and a sort (fit, calm → hot, hot → calm, bpm, title, hook); two-line rows -
who it is, then its character (tempo and key, an energy glyph, sings or instrumental, dark / even / bright,
genre, year, its mood tags, the hook's time, its length) and the verdict; the search matches tags, genre
and year too; the tooltip adds the song's shape (its builds, breakdowns and drops with times) and when it
last played. Also: the engine starts after the Director is ready and the standby engine waits 30 s (an
underrun at START in the user's log). "Prefer zoom to pan on the scroll; more song structure info" → the
wheel zooms around the pointer (Ctrl / Shift + wheel pans), the vocals lane carries the singing map (green)
under its section strip, and a SONG MAP sits under NOW and under NEXT: the whole record's sections by
colour and energy, its drops, its hook in gold, where it sings, and the playhead - the shape of the song at
hand without touching the picture's zoom (the tooltip lists the sections with times). "I want a few
controls that will orchestrate over the next few songs" → PROGRAMS (NEXT SONGS row above the moments; on
the web page too): BUILD (energy up song by song, the tempo leaning faster, a big moment on the last), COOL,
PEAK (amped, short, moments and fx on, bass up - for these songs only), VOCALS UP (instrumentals → singers),
BREATHE (long plays, long seams, nothing added), BRIGHTEN, DARKEN - over 2, 3 or 4 songs; a program steps
the dials it names at every song change (layered: every bed change), the other dials stay yours, the
baseline returns when it ends or on stop; the intent line says "BUILD song 2 of 3 (next: energy amp, tempo
faster)", the row under it lists the steps with the current one marked, and the story row carries the
coming steps as dashed marks where the next songs will start.

**Performance (2026-09-12, night): "we need to make some big performance improvements."** Two underruns
in the user's log ("the render thread cannot keep up"), one at START and one at the conductor's arrival,
against a 1.2 s render-ahead ring - so something held the GIL for over a second. Measured, not guessed:
steady-state calls (status, rank, the picture's record, the conductor's step) are all under 10 ms; the
autoDJ alone costs ~3 ms per 100 ms block; layered, the mixer costs 35-42 ms per 100 ms block with spikes
past real time (R3 keylock on each slave ~14 % of real time, the deck EQ 28 % of the mixer, the PLL small).
The one-off hogs: `load_library` 1.5 s (run again by every DJSystem.start - at START and at every handover
back to one song), the conductor's pick re-running the brain's full scoring up to eight times (0.35-0.8 s
each), a song's decode (~4.3 s of PyAV for four stems + 0.4 s for the mix) plus 0.9 s of numpy over the
stems, all in threads of the show process. Done: DJSystem takes the Director's already-loaded library
(`library=`); the pick applies its cheap gates (tempo wall, runway, key fit) to the whole library first and
scores ONCE; the stem measurements are one vectorized pass (0.57 s, identical numbers); the deck EQ takes a
one-cascade shortcut when only the low or only the high band moves (the carve, BASS, TONE); decoding moved
to a helper PROCESS (`lib/dj/stemload.py`: a warm spawn worker started at Director start, 90 s timeout and
an in-process fallback that also fires when a spawn cannot import `__main__` and hangs) for the conductor's
songs, the autoDJ's decodes and the handover pre-decodes. The tab starts the engine after the Director and
the standby engine waits 30 s. Still to do if underruns persist: R2 ("faster") keylock for the conductor's
slaves under load, or varispeed for slaves within ±1 %, and the Spectro computation's placement.
Also this night: "why does the Director keep changing its mind about when to start a new song?" - the
record's exit is fixed once per record, but the seam's START moves with the style each re-plan picks (a
64-beat blend begins a minute before a cut) and re-plans follow every steering change; NEXT now anchors on
"this record ends in N s" with the seam's style, length and start as the detail. And the picker rows went
back to ONE line (tempo/key, an energy word, sings or inst, two moods, the hook; a ✗ row adds its reason) -
"flooding me with details" - with everything else in the tooltip; "hotter" / "calmer" mean a step (0.04-0.30
from the reference) and fall back to "same" with a note when nothing fits, because an open-ended "hotter"
against a reference that climbs with the queue dried the list up in three picks.
*Later that night, live:* "the Director is really doing some weird stuff… it just fades things in and out
for no reason. It left the bass track empty for minutes." A hole, not a tuning: only the `other` lane had a
"come back" rule; a drums or bass lane emptied by a break whose song left before the break ended, an
eviction with no target or a landing that never came stayed empty for good. Now THE BED IS NEVER EMPTY
(the first housekeeping rule: an empty drums / bass lane goes back to the bed at the next move, or the song
holding most lanes takes the bed), a break restores to whatever is the bed now, a bed change in flight
cannot freeze the room past four phrases, and the four-bar strip is only for MIXING cut. "Things seem a lot
less loud when multiple songs are playing" - every deck is loudness-compensated, but a bed with resting
voice lanes plus a carved, ducked voice measures several dB under a whole record: the Director now walks
the conductor's bus gain up (never down, at most +6 dB, slowly) until the layered output measures like the
autoDJ's did. Also found: the steering cache was keyed by the engine's id(), which a new engine after a
handover can inherit - it now lives on the engine. And an interference of my own making: the headless
smokes and gates run on the same machine, log to the same dj_*.jsonl and write to the same play_history as
the user's live session; while the user listens, runs go at idle priority, one at a time, and the log
cannot tell the sessions apart (open: a session id in the log rows).
"The button layout has become unreadable again" → the NEXT SONGS row had put eleven buttons on one line,
and word-wrapped labels drove the tab's MINIMUM height to 977 px in a 1600-wide window, so on a 1080 screen
Qt squeezed everything: programs on two rows of four, moods at three rows with a "+N more…" chip, the dial
grid tighter, the picture's lanes 34 px, the arc strip 48, the wrapped labels capped. "Parts of a song were
playing longer than they were displayed… I can't tell if multiple parts of different songs are going in the
same stem" → the picture now takes each lane's clip from what is SOUNDING (deck telemetry: stem gain × deck
gain), so a bed change is drawn when the new drums and bass land, not when it was decided; and the level
meter under each lane stacks every song sounding on that stem in its own colour - a crossfade, the autoDJ's
blend or a doubled lane read as two colours.
"Certain songs just play more frequently than others… 'Need You Now' will almost always play in the first
few songs of a set" - measured: 1,296 plays in 30 days over 384 of 1,073 songs, a top twelve with 16-28
plays each; from 80 random playing songs the picker gave 40 distinct winners, two of them taking a quarter of
the picks (Need You Now 10, Kirghiz 9); the opener choice gave 16 distinct openers over 30 seeds. Why: the
night's no-repeat forgets at dawn, and the evidence gates (verified grids, beat power, blendability in and
out) leave a small elite that satisfies everything at the arc's energy, so the same few win every night -
Need You Now's terms were energy 1.00 at the target, blend 1.00, exit_chain 1.00 while the runners-up lost
0.1-0.45 on energy. Fix: CROSS-NIGHT FRESHNESS - `db.play_counts` (plays that ENDED normally after 45 s, so
headless runs and stopped sessions do not count), `brain.load_play_counts` at every start, and a soft lean
`s_worn = 1 / (1 + 0.12 × plays_30d)` (floor 0.4) in both the pick and the opener. And the history was
polluted: my gates and smokes logged plays all day (the gate's seed-7 opener sat at the top of the 30-day
list); headless runs now set DJ_NO_PLAY_LOG and log nothing.
"I don't see how we can make it do big moments where a DJ really amps up the bass with a bassline from one
song while another plays… it doesn't seem responsive enough and tactical enough" → three pieces. (1) THE
TACTICAL MOVE: `RemixConductor.bring(lane, deck, boost, when)` - this song's stem in on the NEXT BAR (or at
the phrase), +30 % loud for a phrase (`lane_boost`, the peak guard keeps the bus honest), the previous
holder's stem gone in a beat; the guard still refuses a clash, the arrangement's waits do not apply;
`release(lane)` sends it back to the bed. (2) THE BASS MOMENT in the arrangement: with MOMENTS on (mixing
auto / cut) a bed change is bass-first - at the voice's payoff its bassline takes over, loud, the old bed's
bass gone, and its drums follow at the next phrase (the classic bass swap: the room feels the new low end
before it knows the song changed). (3) THE READY ROW under UP NEXT: every song on a deck with four stem
buttons (bass / drums / melody / vocal), lit where it holds the lane, greyed with the reason where the guard
refuses, LOUD and PHRASE toggles; right-click any song in the list = stage it and bring a chosen stem the
moment it is ready (`Director.stem_in` / `stem_out`, a pending stem move fires when the decode lands). The
gate brings a live song's bass in loud and releases it, and counts bass-first bed changes.

**2026-09-12, late - "the interface isn't great. things bounce around. it isn't really clear how this creates
what I want. it seems a pale shadow. go read more on DJ theory about how moments are created" / "it's a
painful mishmash".** Building stopped. Read Solberg 2014 (the break routine: uplifters, the drum-roll effect,
large frequency changes, removal and reintroduction of bass and bass drum, contrasting breakdown, the
micro-vacuum before the drop, layers changing after 2/4/8/16/32 bars, builds of 16-32 s) and Solberg &
Dibben 2019 (expectancy set up, delayed, gratified; the delayed downbeat at the drop is what listeners name
unprompted), plus the practice pages (phrase mixing, bass swap on the 32, kill-and-slam, acapella bridge).
Diagnosis: the conductor makes one guarded lane move per phrase forever - busy and shapeless at once - and
every control is a knob on that machinery; none of them IS a moment (a composed 16-32-bar shape: take away →
build → hole → payoff on the one). Written for a yes before any code: `docs/DJ_MOMENTS_DESIGN.md` - five
composed moments (SLAM, BREAK ROUTINE, SING-ALONG, TAKEOVER, ACAPELLA BRIDGE) as bar-offset scripts the
conductor runs with the arrangement paused, landing on the measured drop/hook, armed from ONE row of five
buttons with a countdown and the choreography drawn on the picture; READY / LOUD / PHRASE / FX / the program
grid leave the main surface. Open questions for the user: loop tightening and a bus high-pass sweep as build
tools (they touch the "no stutter builds" rule). Done meanwhile, mechanical: the READY row no longer
rebuilds when the guard's bar-by-bar opinion flips (in-place text/enabled/tooltip; rebuild only when a song's
holdings change) and the SONGS list re-ranks only when its reference (record, last UP NEXT) changes or every
30 s, keeping its scroll - the two visible sources of "things bounce around".

**2026-09-12, "yes, do it" - THE MOMENTS built as designed.** Conductor (`lib/dj/remix.py`): `MOMENTS`
(slam 2 / break 16 / sing 2 / takeover 1 / bridge 8 bars of lead), `arm_moment(kind, auto)` chooses the
landing - the target song's drop or hook within MOMENT_LOOK_PHRASES (4) phrases via `_payoff_in_bars`, else
`_phrase_ahead` (the next 8-bar boundary of its section grid at least the lead away) - and stores a script
of (bar offset from THE ONE, action); `_moment_step` (in `step()` after the boost step) fires each action
when the bar before it begins, posted at the next bar's clock, and returns True so the arrangement, the vocal
rule, the auto break and the phrase move all PAUSE while a moment is armed or running; `moment_go` re-anchors
the one to the next bar + lead; `moment_cancel` undoes a running one on the bar; `moment_status` carries the
countdown and the choreography marks. The scripts: SLAM = bed's bass + drums out (1 beat) two bars before,
`_m_hole` (every held lane hard-muted) on the last beat, `_m_payoff` on the one (every held lane back at its
level, bass + drums MOMENT_BOOST 1.25 via `lane_boost` for a phrase, filter off, loop cleared). BREAK = out
at -16 (2 beats; refused when no melody/vocal is held or the one left is measured silent over the take-away),
`build` at -8 (drums back, a HIGH-PASS SWEEP on the bed deck 50 -> 420 Hz over 7.5 bars when MOMENT_SWEEP),
`tighten` at -4 (the bed deck LOOPS 2 bars -> 1 -> half, posted at absolute clocks because a looping master
stalls the bar count; the hole for the last 2 beats keeping the vocal; the one posted then too, with a
`jump` back to where the song would have been); SING = bed out at hook + 0 for two bars, back loud at
hook + 2; TAKEOVER = old bass out on the last beat, the other song's bass in loud on the one (lanes moved,
`_m_roles` later), its drums cross at + change_bars (`bed_to` logged); BRIDGE = bed's drums/bass/other out
over 4 bars from -8 under its vocal, the other song's drums under at -4, a one-beat hole keeping the vocal,
its bass + drums (+ other when the guard allows) slam on the one and it is the bed. After a moment:
`phrase_bars = 0`, `_drop_k = bar_n` (a phrase of plain play; the auto break keeps its distance). Director:
`arm_moment` / `moment_go` / `moment_cancel` / `moment_status` (one song: SLAM = the autoDJ's DROP, the rest
refuse with the reason); `_auto_moments` layered now arms these same five (bridge/takeover when a voice has
been heard two phrases, else sing, break, slam) instead of the DROP-onto-voice; the intent line and
`status["moment"]`; the picture draws the choreography as gold dashed marks (kind "moment") from the one
backwards. Tab: ONE ROW of five checkable buttons (press = arm, the button counts down `SLAM in 6`, press
again = go now, right-click = cancel, a line under the row for refusals); DROP/BREAK buttons, the READY grid,
LOUD/PHRASE, the FX dial and the 2×4 program grid are OFF the surface (one PROGRAM box); the right-click
song menu still brings one stem. nanoKONTROL2 REC/CYCLE = SLAM/BREAK. Show page: the five + ✕. Gate:
SLAM and BREAK armed (landing within four phrases + lead), take-away and the one in the log, the bed's drums
measured out then back in the deck telemetry; TAKEOVER when a second song is there (bass then drums on it).
First live minutes: "the music shut off totally for many seconds" (two candidates: my smoke's decode helper
running at normal priority beside the restarted planner, or an auto BREAK whose take-away left an unmeasured
melody alone - guards for both: a silence watchdog undoes a running moment after a bar and a half of nothing,
BREAK refuses an unmeasured melody, the DJ's first moment waits the gap from START, headless runs write no
log and take no controller), then "the break that just happened was painful in its execution" → MOMENT_TIGHTEN
and MOMENT_SWEEP OFF (the ear's verdict on stutter builds stands; the whole-deck high-pass thinned the melody
and vocal too); the BREAK's build is now the drums back thin (the deck's low shelf at the carve) with their
lows returning over eight bars, the bass stem waiting for the one. Which part hurt is still to be named.

**"what are the current problems and how can we improve things?" → "do them all" (2026-09-12, late).**
(1) The ear session's tools: BAD on a moment asks WHICH of five words (`MOMENT_WORDS`: hole / thin / long /
beat / place) and `RemixConductor.tune_moment` moves ONE number of that moment (`MOMENT_TUNE_DEFAULT` per
kind: hole_beats, out_bars, take_bars, build_low, fade_bars, boost, payoff_only; the hole halves to none, the
take-away shortens, the build's low shelf fills, `beat`/`place` make it land only on a measured drop or hook),
persisted per library in `<music>/.dj_moment_tune.json` (never by gates); the scripts read the numbers.
(2) `Director.rate(up, word)` + `rate_target()`; the tab's BAD opens the five-word menu when the last rateable
move is a moment. (3) CALM: `rc.calm_phrases` (the Director sets 2) - the arrangement moves at most once per
two phrases and not within a phrase after a moment (`_calm_until`); NEXT and an empty bed lane exempt;
`moves_10min` measured in the arrangement status and said in the intent line. (4) TAKEOVER / BRIDGE with a
CHOSEN song: `arm_moment(kind, other=deck)`, `Director.moment_with(track_id, kind)` stages if needed and arms
when ready; right-click a song → "TAKEOVER with this" / "BRIDGE with this". (5) `lib/dj/session.py`:
SESSION_ID on every jsonl row (the autoDJ's, the conductor's moves, the Director's events, refusals), nothing
written under DJ_NO_PLAY_LOG; the DROPOUT detector in `Director._dropout_step` (bus under -48 dBFS for more
than a bar → one event + log row with mode, handover, armed moment, last events and moves; its end with the
length; the state line shows it for three minutes). (6) Refusal statistics: `moment_stats` per kind (armed,
refused by reason key) in status and each moment button's tooltip, refusals logged. Gate: WORDS, CHOSEN SONG
refused and counted, CALM (≤ one arrangement move per two phrases over six). Not run yet: the user was live.

**2026-09-13 morning - "the system just seems unable to make up its mind when planning. it keeps changing
the length of how long something will play. also it has long chunks of time when very little is going on. I
just slammed bad several times during one".** The log (session 670b43): 86 s of one song alone after the
handover (the settle counted from zero although the record had played two minutes), a voice waiting 40 s
for its payoff, then five BADs in six seconds that landed on the voice arrival, on the takeover the first BAD
had just forced, on a two-minute-old vocal rest, then on nothing - teaching LESS of exactly what was wanted.
Every countdown was recomputed each tick (the next-move ETA reached zero, calm said wait, it jumped back to
8; the staged ghost moved with it; a voice flipped between "at its drop in N" and "may take the bed in M").
Built: (1) THE PLAN - `RemixConductor.plan`, one next event {what: voice_in | bed_to | voice_out, deck, bar,
why, payoff}, decided ONCE in `_plan_check` (called once a bar and at every arrangement tick) with the calm,
the settle, the song's readiness and its payoff search (from the first allowed bar to the limit) baked into
the bar, the bed's build skipped at decision time; kept until done or stale; `_arrangement_move` executes ON
that bar (the step makes the plan's bar due even inside a phrase; bed_to within the phrase before it via
`_bed_to`'s landing) and moves it only through `_plan_move`, which logs a `plan` move ("plan moved: … - why");
NEXT/forced, DROP, a moment's end clear it. Status: `arrangement.plan` with words; the Director's intent line
starts with "next: … at bar K (N bars) - why", the layered NEXT ETA counts to the plan's bar, the picture
draws ONE ghost mark at it. (2) BAD semantics: `rate_last(up, max_age_s)` - a verdict is about the last move
only within two phrases; later or with nothing to rate a BAD is about THIS STRETCH (pushes on, blames no
move); moves made because of a BAD carry `unrateable_until` (a phrase) so the correction is heard before it
can be judged. (3) The handover backdates `bed_since` by the bars the record had already played (`_open_first`
from the cue), so layering the playing song does not restart the settle. Gate: PLAN (plans decided ≥ 2, every
bar change logged as a plan move, few plans move). Not run: the user was live.

**"I need a more robust way of messaging what is not working so we can fix problems later without just
hurting specific mixes or song tags" (2026-09-13).** FLAGS: `Director.flag(kind, text)` with `FLAG_KINDS`
(dead / busy / still / clash / seam / song / plan / timing / vocals / level / ui / other) writes a SNAPSHOT
of the room to `logs/dj_flags.jsonl` (via `session.log_row(name=)`; also a short row in the day's log): mode,
theme, pool, dials, arc, the engine's clock and bus rms/peak, every song on a deck (id, title, tempo, key,
shift, rate, position, section, lanes held, stem gains, deck gain, low EQ, loop, entered/leaving/was_bed,
voice_since/bed_since), the lane map, boosts, the plan, wait_why, moves/10min, the armed moment and the
moment tune, the last ten conductor moves, the last six Director events, the last verdict and dropout; in
one-song mode the autoDJ's current/next/plan/history. A ⚑ mark in the story row; `status["flags"]`. NOTHING
is learned or moved. Tab: FLAG button beside GOOD/BAD (menu of kinds + "with a note…"); nanoKONTROL2 SET =
a plain flag. `tools/dj/dj_flags.py` prints the flags as blocks (or `--summary`, `--kind`, `--session`,
`--last`, `--json`) - enough to re-render the combination offline from the stems.

**"have we made too many judgements into the record recently that might be causing learned problems?"
(2026-09-13).** Measured (read-only): 48 verdicts since 09-11, 45 BAD, 42 of them conductor lane moves
(return:vocals 9, voice_in 7, rest:vocals 6, bed_to 9, drop 3 …). The conductor's own weights from them are
almost inert (the arrangement policy never consults `_w` for those kinds; only the auto break got a third
rarer) - BUT `db.seam_feedback_rows` and `pair_stats` fed EVERY row, remix ones included, into the autoDJ's
class and exact-pair memory (a vocal rest is a self-pair; a voice arrival marked that pair class rough). And
play_history: 668 rows on 09-12, 120 ended, an unknown share of them my gates' before DJ_NO_PLAY_LOG existed
- freshness leaned those songs away for 30 days. Done ("do it"): `seam_feedback_rows` / `pair_stats` exclude
`remix:%` styles and `fenced%` sources for good; `db.fence_feedback(since)` marked the 44 conductor verdicts
since 09-11 `fenced:user` (kept, read by nothing); `FRESH_SINCE` = 2026-09-13 - play_counts never looks
before the first clean day. BAD's tag learning was session-only all along; no moment-tune file exists yet.

**The first eleven flags (2026-09-13, session c7bcc2) and what they taught.** busy at 15 s (the vocal lane
opened on a silent passage, rested, came back; a voice arrived at bar 9 because the backdated settle was
nearly spent); clash / dead / wrong song within 40 s of a bass-first bed change that landed on a drop whose
groove ended 30 s later (the room fell into the new bed's build); wrong song: a voice arrived and looped
its last 8 bars 16 s later (cued for an entry that came a minute after the cue assumed); level: melody
trimmed to 0.5 and vocal boosted to 2.0 against the FIRST song's levels; dead: the bed's own breakdown with
nothing anticipating it; seam / clash / timing: ONE decision - a bass-first change into a song looping its
last bars - evicted mid-change, drums still followed, roles tangled, a returning vocal at x2 over an
unducked melody, a loop wrapping under the PLL. Built ("ok"): `_on_way_out` (looping, evicted, leaving, or
< MIN_RUNWAY_S of body) - never takes the bed, never a lane (bring refuses, `_second_song` skips), an
in-flight bass-first change aborts (bass back to the drums' song); `_drop_in_bars(min_groove_bars)`: a bed
change lands on a drop only when the groove after it lasts two phrases; `_bed_dies_in_bars`: the plan takes
the bed a bar before the current bed's breakdown / outro / body end, and a voice arrives a phrase before it
when nothing is heard yet; `_recue_for`: the plan re-cues the staged song so its landmark lands ON the
arrival bar (nobody hears the jump), and an arrival that would leave < 90 s is cued back a phrase; the
`other` duck only under a vocal that IS singing and `_reduck` on every vocal rest/return; stem levels
against the LIBRARY's median (`stem_rms` remembered in each track's axes as songs mount, `_library_ref_levels`
once eight are known, the scanner keeps the key) instead of the first song's; the handover opens the vocal
lane only where the song sings and keeps two phrases of the bed alone. Tab: GOOD FLAG and BAD FLAG are two
buttons. Gate: WAY OUT (no bed change into a looping song), re-cue count printed. Not run: the user was live.
Open: the two chord-clash flags (11:42, 11:44) await an offline re-render and a section-fit measurement.

**Nine more flags (session cb76c0, 11:50-11:55, the build before ff80d1f).** seam + still: a voice on
`other` at 1.38 (trimmed toward the first song) sat alone over the bed for 96 s waiting out VOICE_MAX (6
phrases, no payoff in sight); seam / clash x2 / "sounds were buzzing": the vocal lane FLAPPED rest→return
within 4 s four times (two-bar reads on a sparse vocal), a voice's section changed 20-40 s after a clean
arrival (breakdown / build chords against the bed), and a bass-first change "at its drop" that landed on
Need You Now's 8-second groove (266-274 s, between a breakdown and a build) - the honest cause of the early
takeover; plan: the stale "takes the bed at bar 130" stayed with a negative countdown while the drums were
still to follow; seam at -9.6 dBFS: the loudness matcher had walked the bus to x2 during a quiet stretch
and the next full arrival came in 6 dB hot; clash: the BED was looping its last 8 bars with the voice's
takeover planned 74 s away. Built: VOICE_MAX_PHRASES 4; VOCAL_REST_BARS 4 both ways (`_vocal_gone(bars)`,
return needs `_sings_through` 4 bars and 4 bars rested); `_clash_tick` per bar - a voice whose pair fit
with the bed falls under CLASH_BELOW over the next two bars is held at CLASH_DUCK 0.5 (`duck` / `unduck`
moves, `_clash_duck` in `_lane_gain`) until the sections agree again; honest bed-change text ("at its drop"
only with a payoff, else "at the phrase, after its turn as a voice"); the plan shows a landing in flight
(`what: land` at `_land_k`); the loudness matcher caps at +4 dB, releases at once and holds during moments
and breaks; `measure_track` now stores `stem_rms` (stereo, the conductor's own `stem_levels`) so `--measure`
fills the library's level reference offline; nine more flag kinds (bass, energy, runout, moment, long,
queue, control; good_bass, good_calm, good_level). Gate: VOCAL LANE flaps ≤ 1, duck count printed. Not run:
the user was live. The 8-second-groove drop is already excluded by ff80d1f's min_groove_bars.

**Five more (session 2b6b2b, 11:57-12:01) and the clash measurement (2026-09-13 noon).** Two of the five
exposed a bug in the new plan: a bed change decided "before the breakdown" (bar 109, bar 172) was still held
back by the executor's three-phrase rule, so both came a phrase late and the bed sat in its breakdown with a
melody on top → `plan_now` waives VOICE_PHRASES when the plan's bar is within the phrase. "vocals wrong"
twice: Birds Mind's bass and vocal at x2.0 against the first song's levels (the library reference now
fills: `measure_levels` pass), and Need You Now's vocal missed its own hook while only its bass had taken
over → the vocal lane's return considers the bass holder (the incoming bed) as well as the drums'. THE
CLASH FLAGS (nine, then ten): at every one the stored section-chroma fit was 0.76-0.97 (same Camelot key
in all) - far above CLASH_BELOW 0.55. A bar-level probe from the stems (the bed's sounding bass+vocals vs
the voice's `other`, 16 bars around each flag, per-bar chroma cosine) scored the flagged alignment below
its shifted controls in only 2 of 9, and the baselines killed the measure: self pairs (a song against its
own melody 32 bars later) median 0.62, random same-key pairs 0.51, random ANY-key pairs 0.60, flagged 0.49.
Neither chroma measure hears what the ear called a clash. What the ten share is structural: 10/10 had
another song's FULL `other` stem over the bed, 8/10 while the bed's own vocal was on, levels 0.45-1.4 on
the melody and up to 2.0 on the vocal. Conclusion for the philosophy (awaiting the user's yes): "a melody
over a singer" is the clash - arrivals should come through the VOCAL (an acapella over an instrumental
bed), through the BASS (the swap the user asked for) or through `other` only over an instrumental bed and
at a texture level; the per-bar clash watch stays (it is cheap and honest) but will rarely fire. Gate after
all of today's changes: ALL OK (the PLAN check now keys plans by a sequence number, not id()).

**"yes" (2026-09-13) - THE ARRIVAL RULE.** In `_arrangement_move` section 3: a song arrives (1) through
its VOCAL when it sings through the coming phrase and the bed does not (`vocal_freedom` > 0.1, the lane
free, the guard), (2) else through its BASS - `_bed_to(d, at, None, None, bass_first=True)`: the bass
swap as an arrival, its bassline takes over under the bed's drums (logged `voice_in`/bass "as it arrives"),
the drums follow at `_land_k` - when the guard allows its bass and no change is in flight, (3) else through
its MELODY only over an instrumental bed, at OTHER_VOICE 0.5 (a voice's `other` is always a texture in
`_lane_gain`), (4) else it waits with the reason ("the bed is singing - no melody over a singer, and its bass
would clash"). `_bed_to(bass_first=)` overrides the MOMENTS dial for arrivals. Philosophy principle 8 and the
guide amended. Gate: ARRIVALS by lane, melody-over-a-singer must be 0.
The first order (bass swap before melody) made every arrival a bed change and no song was ever a voice;
the order is now vocal → melody over an instrumental bed → bass swap when the bed sings. Nine gate runs
then found, one per run: a bed's BREAKDOWN counted as its death (`_bed_dies_in_bars`) and chained a change
every phrase → outro / body end only; a re-cued song arrived inside its re-lock → no re-cue within two bars,
the re-snap a beat after the cue, and STAGED DECKS SLEEP at gain 0 (`_stage_song`, `_wake` on the first
lane; the PLL re-snaps only a near-silent deck - a deck at gain 1 with every stem closed slewed 0.3 beat
for fifteen seconds); the clock passes WITH THE DRUMS (`_pass_clock` at the follow) instead of when the old
master leaves; the plan's `land` entry made a move due every step and each no-op `_one_move` queued a
hygiene pass at the landing with the pre-landing lane map that REOPENED the lows a second after the carve
(found with an EQ event trace in the gate) → `land` triggers nothing, hygiene only after a move; and the
audio PLL chased melodic pattern offset on every slave (their sounding stems never have kicks) → the
conductor's sync events are GRID-ONLY (`audio_pll: False`). Low carve 0 %, locks p95 ≤ 0.08 on the bed decks.
The last lock failure was a MIS-TRACKED GRID SEGMENT: Elephante's grid reads 66 bpm from 140 to 160 s inside
a 116 bpm song, and 327 of 1073 stem tracks (30 %) carry such a segment (mostly 10-30 s, breakdowns and
intros); the deck's beat phase there is meaningless, the PLL saturated its trim both ways and the new bed's
tempo wobbled 1.2 % for the whole segment. `submix._odd_segment` (period off the song's dominant segment by
> 3 %) HOLDS the PLL while either deck is inside one - this touches the shared PLL, as a guard only; the
conductor's plan lands no bed change and no arrival inside one (`_odd_grid_bars`); the gate's lock metric
skips those samples (it read the same meaningless phase). The clock passes when the old master LEAVES, as
before (`_pass_clock` exists but is not called: passing it at the follow re-synced the audible old bed).

**"will future songs have this analysis done as part of the DJ planner analysis pipeline?" (2026-09-13).**
They did not: the stem measurements (section chroma, hook, stem levels) lived only behind `dj_scan.py
--measure`; a live layered play filled `stem_rms` per mounted song, nothing filled the rest. Now:
`dj_stems.py` measures every track it renders right after the render; the planner's Stems button runs a
"stem measurements" stage when the render finishes; `lib/dj/analyze.build_stages` (the one stage list of
"Analyze all" and the headless `dj_analyze.py`) has a "stem measurements" stage right after the stems
stage (skips fast when nothing is unmeasured). The 900-track level pass for the existing library ran at
13:38 (below-normal priority, resumable). Done 16:01: 1073/1073, medians drums 0.195 / bass 0.129 / other
0.100 / vocals 0.032 (two NaN rows from empty body windows stored as silent; `measure_track` guards it).

**"will a song be balanced within itself if only its stems are playing?" (2026-09-13).** It was not: the
per-lane trims pulled each stem toward the library median independently, so a lone record through the
conductor could differ from itself by up to 6 dB between stems. Now `_trim`: the song carrying the DRUMS
(the bed, or a song alone) has ONE factor for all its stems (its drums brought to the library's drum level,
geometric mean of the lanes when the drums are unmeasured) - it sounds like the record; a stem sounding
against another song's stems (a voice's melody / vocal, a swapped bassline) is trimmed per lane toward the
library's level for that lane. `_cross` updates the lane map before computing the incoming stem's level;
`_hygiene` re-posts every held stem's level after a move (a song that just became the bed switches from
lane trims to its own balance); the opener is posted at the library's level through `_lane_gain`. Gate:
LONE SONG - while one song holds every lane its four stem gains are equal. Open: the vocal median (0.032)
is taken over a half-instrumental library; a real singer trims down toward a mostly silent lane - the
median should be over songs that sing.

**Eight flags from session 137820 (16:08-16:15, the arrival-rule build).** Five were LANDINGS IN THE
WRONG SECTION: a bassline took over as its song entered its breakdown, an arrival came six seconds before
the bed's build, The Long Journey arrived in its own build under Ananta's build (two builds at once) - the
runway rule only guarded drop landings. Now `_groove_through(deck, at, bars)` / `_groove_start_bars`: the
plan's bed_to bar moves to the incoming song's next two-phrase groove when it would land in an intro, build
or breakdown; an arrival requires BOTH songs to groove through the coming phrase (else the plan moves to
the next groove, logged). Two were the TEXTURE DEFEATED BY THE TRIM (a melody "held low" at 0.73, a swapped
bassline at 1.61): a stem set against another song is never trimmed above unity before its role level, and
every stem gain is capped at 1.6. The vocal at x2 again → the VOCAL REFERENCE is the median over songs that
sing (vocal stem ≥ a fifth of the drums' median). A "filter out" exit was flagged as a bad seam → FX
default "none" (plain crossfades; the dial is off the surface, the controller's knob still reaches it).

**The gate on the afternoon's dozen commits (2026-09-13 17:00).** First run: the boost-step crash the user
hit live (a three-field boost unpacked as two - after any stem brought in loud the conductor's step raised
every tick and nothing moved: the true shape of "my direction was ignored" all afternoon), fixed before the
run. Three fails then: (1) a slave crossing its MIS-TRACKED GRID SEGMENT drifted half a beat - the PLL hold
had DECAYED its trim to zero, and the trim was the learned tempo: the hold now freezes the trim; (2) a boost's
restore posted during a running moment's take-away re-opened the muted drums: no restore while a moment has
started (its payoff re-posts everything); (3) the LONE SONG check read a boost's restore ramp - it skips
three bars after a boost ends. Telemetry now carries `rate_trim` and `odd_seg` per deck. ALL OK: every
deck's lock p95 ≤ 0.08, lone songs equal on 100 % of blocks, no dead bars.

**Twenty-one flags from three sessions (2026-09-13 16:54-18:14) and "I feel we just keep regressing".** Four
mechanisms explained all of them, two of them consequences of the day's own builds. (A) Every clash flag
(five) plus the low-end flag and a seam were ONE SONG'S MELODY OVER ANOTHER'S BASSLINE, reached from the
other side: the arrival rule had made the bass swap the usual arrival, and the swap left the old bed's
melody over the new bass (at 0.5 or full). (B) Five dead/seam flags were the conductor trusting SECTION
LABELS where the stems disagree: The Long Journey's "groove at 0 s" is a near-silent intro and its
bassline "took over" thirty seconds after a handover with nothing in it; the DJ's BREAK measured its melody
against the melody's own quiet scale and left rms 0.02; Lonely Souls took the bed where its drums and bass
are silent (rms 0.007). (C) Five plan/still flags were one loop: a plan that moves re-cues the song so its
landmark lands on the new bar, the landmark fails the same check, the plan moves a phrase - Lonely Souls
eight times, Wild Window five. (D) An evicted bed handed its drums and melody to a sleeping staged song in
its intro (the 17:31 cascade: vocals, clash, bass). The user's yes to all four:

1. HARMONY TRAVELS TOGETHER. A song's bass and melody are one thing, its drums another, a vocal is free.
   Every bed change moves bass and melody as a pair: the bass swap (under a singing bed) brings the new
   melody with the new bass and takes the old melody out with the old bass; the drum swap (under an
   instrumental bed, new) brings the new drums under the old harmony and the bass and melody follow at the
   phrase; TAKEOVER and BRIDGE carry the melody with the bass. The "melody held low" arrival is gone (it was
   flagged clash at 0.5 too). `_lane_gain`: the bass holder's melody is full; a melody on a song holding
   neither drums nor bass is the texture. The clash tick judges voices against the BASS holder. Gate:
   HARMONY (melody and bass on one song, two bars of crossing allowed) and ARRIVALS (no melody arrivals).
2. STEM-BACKED POSITIONS. `_stem_carries` (25th percentile of the drums' and bass's envelopes ≥ 0.35 of
   their loud level), `_track_steady` (labels, sampled every bar a quarter bar inside the ends - the
   three-point read passed at 16 bars what it failed at 8), `_steady_at` / `_steady_from` (both). The body
   start and the landmark are the first place that carries for two phrases; a bed change (plan (b), the
   arrival paths) lands only where the new song carries; the BREAK needs the bed's drums playing in a
   groove through the take-away and what stays at ≥ 0.25 of the library's drum level in ABSOLUTE terms
   (else it tries the next three phrases, then refuses: "nothing to take away", "too quiet there").
3. THE ARRIVAL BAR DECIDED ONCE FOR BOTH SONGS (`_arrival_bar`): from the settle bar, the first bar where
   the bed grooves through the change and the arriving song, cued to its landmark, is steady and carries;
   a song arriving through its vocal is cued to where it SINGS (`_sung_landmark`, plan `sing`). A re-cue
   that cannot place the landmark (the song has fewer bars before it than the wait) is DEFERRED
   (`_cue_later`, posted from step, dropped when the plan moves or the song enters) instead of "as early
   as it goes". A bed change after a handover waits the full settle from the open (`open_k`): the handed
   song was taken apart 30 s after the switch.
4. EVICTION hands lanes only to songs that have ENTERED; the lanes a bed change still owes go to the song
   that owes them (the follow completes there); the melody goes with the bass or rests.
Flag snapshots now carry the re-cue log and the in-flight follow. Gate: five runs to green. Run one found a
real hole (a RECALL had put the bass back on the old bed mid-flight; the follow moved only the drums and the
room sat with the bass on one song and the melody on another for thirty bars - a follow now ends the bed
WHOLE, and a departing voice hands any bed lane it still holds to the bed) and three text mismatches in the
gate itself; run three, with the HARMONY check listing its stretches, found that the old harmony kept its
melody a few bars past its bass at every follow (`_other_free_for` did not count the song giving up the bass
as free to give); run four's last fail was the CALM count treating a bed change's follow and the vocal lane's
bar-by-bar rests as separate moves. ALL OK: harmony split for 0.0 bars of the run, arrivals vocal 2 / bass 6 /
melody 0, no dead bars, every deck's lock p95 ≤ 0.08.

**Live on the harmony build (2026-09-13 19:31-19:42), eight flags.** "The plan is totally unstable" / "songs
come in for a few phrases then disappear again, replaced by the song they just replaced": (1) a bed change
was ABANDONED at the follow because the arriving song had under 90 s of body by then (its landmark was
allowed with 90 s left; the change ate into it) - a follow now abandons only a song looping, evicted or
leaving, and every chosen position leaves the runway plus two phrases; (2) a song that arrived through its
vocal left 24 s later at the first pause (the vocal came back only on the bed) - a voice keeps its deck for
its phrases and the vocal returns to it first; (3) "not carrying there" re-cued the song to the same
landmark at every move - a move caused by where the song is no longer re-cues; (4) the DJ's own TAKEOVER
armed in the middle of a drum swap, chose the harmony already playing as its "other song", and handed the
bed back to it - no moment arms while a bed change is in flight, and a two-song moment's other song is
never the bass holder; (5) the DRUM SWAP (new that evening): its first three were flagged seam, dead and
clash within three minutes (one arrived eight bars before the harmony's breakdown; the bed check now covers
both phrases of a change) - OFF behind `DRUM_SWAP`; every non-vocal arrival is the bass swap. Every plan
decision now goes to the night log as a `plan` row. Then four more (19:45-19:46, still on the old process):
(6) 'still' - the plan for Antrea moved three times in two minutes, one bed obstacle at a time (a build,
then a breakdown after a short groove, then a build): the arrival-time re-read now runs the same joint
search as plan time (`_arrival_bar`, twelve phrases) and moves ONCE to the first bar where both songs
carry; a VOCAL may arrive over the bed's breakdown (room for a voice), a bed change may not; (7) 'clash' +
'seam' - the old bed's vocal carried over a new harmony two keys away (fit 0.48), the clash tick ducked it
to half and the ear still heard it: at every bed change the old vocal goes out with its harmony unless it
fits the new one (`_vocal_fits`, pair fit ≥ CLASH_BELOW here and two bars on); (8) 'seam' on a bass swap
at rms 0.09 - the harmony's melody ducked to half under the old song's vocal (by design) over a quiet
groove; noted, not changed. Gate on the four commits since the last green: three runs. Run one - the
gate's own script (its HARMONY check counted the scripted BRING, the operator's override; its BREAK test
did not retry after the new "in flight" refusal). Run two - one real hole: the strip before a bed lands
measured only the voice's stem on its own scale and a vocal's gap left one dead bar (it now measures what
stays in the room, the voice's lanes and the old melody, against the library's drum level, like the
BREAK); two script-level (an old bed with nothing left now leaves on the bar its harmony passes, so the
loop test counted a departing song; a chosen song off a deck was refused for the in-flight reason first).
ALL OK: harmony split 0.0 bars, arrivals vocal 1 / bass 9 / melody 0, no dead bars.

**THE RECORD PLAYS (2026-09-13 evening, docs/DJ_THE_RECORD_PLAYS.md, the user's yes twice).** "How can we make
the system plan well and play tastefully? we keep fighting this overly garbage mixing" → the log's numbers
(1.5-2.8 min per song, 17-35 lane moves per ten minutes, against a DJ's 4-5 min, one 16-64-bar seam each)
→ "what is the gap?" → the unit of play was the lane, not the record; principle 8 and the drop-landmark
were mine, not the literature's; we mixed middles in beats, thirty times an hour, unheard. Built as
`lib/dj/record.py` (a mixin, `policy = "record"`, the default): one record whole for `play_min_s` (PACE
150/210/300 s) and never past its OUT; one seam per record, planned when the next record is staged - A's
OUT at a section boundary where its drums carry through the seam (`_seam_a_bar`), B cued so its landmark
lands on the completion (`_seam_b_start`, a shorter seam when its drums carry from later, the deferred cue);
the seam's steps as bar offsets (`_seam_steps`: drums first with the darkening high shelf and the harmony
swap on the one; the bass swap; the acapella lead-in; the breakdown and cut), fired like a moment; NEXT
moves the seam to the next phrase, HOLD and a moment move it a phrase; the old vocal carries over only at a
fit ≥ 0.65; no auto break within two phrases of a seam; `_trim` one factor for the drums holder AND the
bass holder. The arrangement policy stays in the file behind `policy = "arrangement"`, unused. Gate
rewritten around it (SEAM NOW, RECORD, SEAMS with the landmark error, PLAY LENGTH): ALL OK on the second
run (the first fell over a name clash with the recorder) - four seams (drums 2, bass 2), harmony split
0.6 bars, no dead bars, no move outside a seam, lock p95 ≤ 0.07. Step 2 (the rehearsal render) and step 3
(the dice on the order, layers three) are next.

**THE RECORD CAMPAIGN (2026-09-13 night, "can you do significant testing and have solid metrics?").**
`tools/tests/_dj_record_campaign.py`: unattended twelve-minute layered nights rendered headless at 2x,
three at a time, both policies on the same seeds, measured per SEAM with the instruments the ear exam
validated (docs/DJ_VERIFICATION.md: isolated-kick flam, dead air min/median, lurch against the pair's own
steps, double bass) plus harmony split, two drums, the level step across the seam and A's sections under
it; per RUN the pace, lane moves, songs that came and went, dead bars, clipping, the beat lock. Report with
Wilson 95 % intervals against the autoDJ's 4,667-seam baseline. Campaign v1 (8 + 8 runs, 22 record seams,
24 arrangement seams): pace 3.5 vs 4.0 min per record (the record's by construction), lane moves 17 vs 24
per ten minutes, dead bars 1 vs 11, lock p95 0.075 vs 0.098 - but per seam the record policy was NOT
better: dead air 13.6 % [5-33] vs 20.8 %, lurch 31.8 % [16-53] vs 29.2 %, both worse than the autoDJ's
7.3 % / 14.2 %. The seam breakdown said why: the mix-out search put A's section boundary at the seam's
START and fell back to "before A runs out" for most records, landing five seams of twenty-two in A's build
or breakdown (dead 0.05, lurch 17 dB), and two records stepped +10 dB at the swap with both at the
library's drum level (a bass- and pad-heavy record). Fixed: the seam COMPLETES at A's section boundary (the
new record takes over as A's breakdown or outro would begin) with a scored search for the steadiest stretch
otherwise; B's pre-landmark search asks only that its drums carry (the bars before a drop are its build);
`_trim`'s one factor is the geometric mean of the drums-anchored and the total-anchored factor (level
continuity across records). The kick-flam instrument measured only 9 of 22 seams and is not validated for
stem seams (after the drum swap A's low band is its bassline, not a kick): the beat lock (median 2-7 ms)
is the honest sync number there. Campaign v2 (record only, after those fixes): dead bars 0, level step
worst 10.2 → 7.4 dB, but dead air still 13.6 % and lurch 36 % - the seam ANATOMY (the 0.5 s level series
per seam, now captured by the campaign) named the remaining classes: (a) the next record's drums
INTERMITTENT under the old harmony (its pre-drop pattern on and off every bar: the 25th-percentile carry
test let an eight-bar dropout through a 32-bar window) - the carry test now also needs the two-second mean
never under half the line; (b) the darkening (high shelf 0.35) opened a 10 dB hole under the new drums and a
13 dB lurch at the swap - now 0.7; (c) seventeen of thirty seams fell to "before it runs out" because a
record too short for its minimum play plus a seam had no valid bar - the DJ now mixes out up to four
phrases before the minimum rather than into the outro; (d) one record's re-cue lost to its own staging cue
on the same bar, it arrived 72 bars past its landmark, ran out inside its seam, was evicted, and its harmony
still took the bed (the melody and the bass on different songs for the next 35 bars) - the re-cue waits a
bar after staging, a PRE-FLIGHT at the seam's first bar moves the seam and cues again when B is more than a
bar from its place, and a running seam whose B runs out is ABANDONED (its lanes back to A). Campaign v3
(8 runs, 25 seams): dead air 8.0 % [2-25] (the autoDJ's 7.3 %), dead bars 0 of 2730, harmony split 0,
two drums 0, level step median −1.0 dB (worst 9.3), placement "into A's section" 13 / "at the phrase" 9 /
steady 7 / runs-out 2, pace 3.4 min per record, 18 lane moves per ten minutes, lock p95 0.071. Lurch stays
at 28 % [14-48] - twice the autoDJ - in two classes the anatomy shows: the new record's drop arriving as a
9-13 dB rise on the one after the old harmony has thinned in its last bars (a musical event the instrument
counts), and a thin stretch where the new drums alone carry under a fading old harmony. The next lever is
the ear's, then the rehearsal (step 2), which would hear both before the room does. Run it:
`python tools/tests/_dj_record_campaign.py --campaign --policies record --seeds 1-8 --seconds 720 --parallel 3`,
then `--report`; results in logs/record_campaign/, earlier campaigns archived as logs/record_campaign_v*/.

**The first live set on the record policy (2026-09-14 17:32-17:52, "terrible set, lots of flags", eleven).**
The night log and the re-cue log in the flags told it in order. (1) The record opened without its vocal lane
(the opener's "vocals only where singing" rule) - the record plays WHOLE now: all four lanes are its from the
first bar, the vocal tick never rests the record's own vocal, and `_room_never_empty` gives any lane with
nobody back to the record every bar. (2) Birds Mind became the record with 70 s of body left and its own
seam began 16 s later ("before it runs out"): the pick gate now needs the minimum play plus the runway after
a candidate's body start. (3) Blinding Lights' vocal rode across two seams over two harmonies it never fitted
(three clash flags): at a seam's first step any vocal or melody belonging to neither record goes out. (4)
The auto break fired on the new record's own breakdown eleven seconds after it took over (drums alone, rms
0.03): no auto break in the record policy - the record's breakdown is its arrangement. (5) THE SILENCE:
Funk Is Love's seam start was computed at its very first bar (a bass swap from a bassless intro: the carry
test asked for drums, the bass swap brings the bass first), every re-cue went to 0.0 s, the pre-flight found
it a phrase out of place and moved the seam a phrase at a time - eleven times - until Grandma Dance ran out
and was evicted with no entered song to take its lanes; every lane rested and the room was silent for twenty
seconds. Now: B's seam start is never under two bars in and is found where the stems the seam brings FIRST
carry (bass for the bass swap and the cut); a record looping its last bars fires its seam NOW as a cut
whatever B's place; an eviction with nobody entered enters the staged song whole; and the room-never-empty
rule enters a staged song whole when no record is left. Gate ALL OK on those; the fourth campaign is
reported below.

**"Even when it was working it was pretty boring… basically an automixer with slightly longer overlapped
transitions. It also basically refused instructions I gave it" (2026-09-14).** Both true. The log: Beam
Me Up's vocal asked for twice and refused "running out (0 s of body left)" - a staged deck's cursor sat at
the song's end before its cue landed and a rule for a playing song was applied; NEXT answered "the seam is
already running" and did nothing. And the record policy IS an automixer with stem seams; the events that
justify layered mode had been pushed out to a few an hour because they clashed. The user's "ok, do it" on
two things: (1) COMMANDS ARE ORDERS - `bring` refuses only a song that is not on a deck (a staged song is
cued to where its part sounds whatever its cursor says; a key clash is said, not obeyed), NEXT completes a
running seam on the next bar / starts a planned one at the next phrase / with nothing staged stages the
DJ's pick and seams as soon as it is ready, UP NEXT takes the DJ's silent staged song's deck when none is
free and replaces the seam's next record, your TAKEOVER or BRIDGE takes the room from a seam in flight, a
stale "NEXT: …" no longer sits in the WHY panel; (2) THE PAIRINGS (docs/DJ_PAIRINGS.md): `tools/dj/
dj_pairings.py` computes once, from the stems, per record the songs whose vocal sits over its instrumental
grooves (VOCAL OVER: key fit ≥ 0.68 over ≤ 2 semitones, tempo ±6 %, a sung stretch ≥ 8 bars, the hook a
bonus, the record instrumental there) and whose bass and melody swap under its drums (BASS UNDER: fit, the
wall, the bass carrying 16 bars from the landmark, the minimum play after it) - `--envelopes` (the 2 Hz
stem envelopes to .stems/<id>/env.npz, parallel, resumable, ~50 min once), `--pairs` (arithmetic on the
per-song measures → .stems/pairings.json), `--show`. The conductor's `_event_step` arms one event per
record at MOMENTS some (two at lots): a VOCAL OVER becomes the new ACAPELLA moment (the other song staged
with `stage_for_bar` so its sung stretch begins on a phrase where the record is instrumental; in for two to
four phrases; out; it leaves unless it is UP NEXT), a BASS UNDER the TAKEOVER moment at its drop (it is then
the record). The SONGS list stars the pairings of the record playing ("its vocal sits over this record
(0.81)"); the Song parts window's words say which vocals sit over a song and where its own parts sit.
Gate: the EVENTS check (skipped without the pairings file). The envelope pass ran once (1204 stem tracks,
47 min, two workers); the first pairing pass left 906 records without an instrumental groove - a sample of
sixty said why: the silent line (0.05) called a separated vocal stem's constant bleed floor "singing" (26),
the strict carry test refused fills (19), and 12 bars would have done where 16 did not (15) - so the quiet
test asks for little over the SINGING line (mean ≥ 0.12 under 15 %, p90 under 0.3), a record's own groove
is not held to the two-second minimum, and 12 bars suffice: 460 of 1205 records now have a vocal that sits
over them, 1194 a bassline that swaps under them. Twelve gate runs followed the orders and events build,
each finding one thing: a user SLAM or BREAK mid-seam had abandoned the seam (three in a row) - now
refused, only a two-song moment with another song takes the room; a user TAKEOVER with the seam's own next
record hands over instead of abandoning; a record looping its last bars is not evicted while its next
record is on its way; the events' lead is four phrases (a decode is real seconds - at two, all four
planned events came too late) and a late-ready event moves to the next phrase; the gate's dead bar is
under -45 dBFS (a record's own quiet breakdown at library level read -40 to -45); a deck whose stored bpm
was 1.5 % off its grid sat at the PLL's 1.2 % trim cap and drifted half a beat in a minute, twice - the cap
is 2 % and the run-in tempo calibration accepts snaps up to 60 s apart (a slow drift never resnapped inside
12 s); a deck that reads past its song's end is a stale read, not a song running out. Campaign v5 with
events on: one BASS UNDER per record a phrase in halved the pace (2.5 min per record) and six dead bars
came back - the bass event now sits at the end of the record's minimum play, as its exit, and the campaign
records where each dead bar falls. Campaign v6: 36 records, 10 events planned, 8 armed (3 acapellas), pace
3.2 min per record, 10 lane moves per ten minutes, dead bars 0, dead air 4.3 % [1-21] (under the autoDJ's
7.3 %), double bass 0, harmony split 0, lock p95 0.078, the landmark within a bar 9/11 - and lurch 43 %
[26-63], still two to three times the autoDJ's, in the same two classes (the new record's drums thin or
intermittent under the old harmony; the drop's rise on the one). That is the next lever, after the ear.

**"What improvements can we make in the meantime?" - "all" (2026-09-16).** Seven, built together. (1) The
lurch classes: B's leading stems are judged BAR BY BAR (every bar's mean over 0.7 of the line) when its
start is chosen, and its landmark lands one bar AFTER the step that brings its harmony (`_seam_lm_off`:
the drop is B's own build resolving inside its harmony, not a step in the mix). (2) THE REHEARSAL
(lib/dj/rehearse.py): two phrases before a seam the conductor renders it from the two decks' loaded stems
at bar granularity (mono, decimated, in a thread) with both records' level factors and judges it with the
campaign's instruments - a hole (a half second under 0.15 of the median) or a lurch (a step over 6 dB or
over the records' own steps by 1.5 dB); a failing seam moves a phrase (twice), then swaps its order, then
falls back to the breakdown and cut; `rehearsal_log`, "rehearsed: clean (…)" in the log, the campaign
counts them. (3) The events: a pending event now steps BEFORE the seam check (a takeover event drops its
seam - stepped after it, it never armed and blocked every seam for its timeout); it is dropped when the
room changes under it (the record is another song, or its own song became the record through a DROP) or
when it came late three times; the acapella takes the vocal lane from a departing record's carrying vocal
(only a vocal YOU brought refuses it). (4) The pairing scores carry taste: energy closeness (the library
rank, 0.3 per unit of distance) and a shared genre (× 1.08), each candidate with its energy - version 2
of pairings.json, same coverage (460 / 1194). (5) The dials in layered mode: VOCALS chooses the event's
kind (none / a vocal first / vocals only, two at lots), ENERGY re-ranks the list toward a calmer or hotter
song, PROGRAM steers the count through MOMENTS. (6) The campaign's kick-flam column is the beat lock p95
(the flam instrument is not validated for stem seams; it stays as a note). (7) Housekeeping: the
analysis pipeline has a `pairings` stage after the stem measurements (envelopes for new stems, the lists
rebuilt only when stale); the Song parts window draws the record's instrumental grooves in blue; `--nightly`
runs the eight-seed campaign into a dated folder and appends one line per run and seam to
logs/record_campaign_nightly.log. Seven gate runs then found, one each: B's placement fallback returned
the landmark ITSELF when nothing carried before it (Ruhi: the drop on the seam's first bar under the old
harmony, then its silent breakdown eight bars after the swap) - now a LATER landmark where the leading
stems carry the whole lead and the song is steady after, else the cut (`_seam_b_cut`, which asks only that
B carry from its landmark); the pre-flight read a stale position a bar after a re-cue ("-27 s from its
place") - it now waits a phrase for a cue to settle; a record's OWN silence (every stem out for a bar:
Ruhi's breakdown at 252 s) is logged (`silence_log`) and the gate and campaign count those bars apart
from holes; the SLAM's "loud" on drums and bass was never taken back when a moment was running at its end
(the record played on at 1.25 × for a minute) - the boost entry now outlives the moment and restores
after it. Campaign v7 (8 seeds × 12 min, play 210 s) then said two things: the pace had fallen to 2.45 min
per record with the shortest at 50 s - the takeover events (30 planned, 27 armed) ARMED the moment their
song was ready and landed at the next drop ahead, not at the event's bar at the end of the record's
minimum play; and the rehearsal failed half the drums-first seams with "a hole at bar 16": the bar before
B's drop is B's own silent bar, and the harmony swap a bar before the drop had put it in the room alone.
Fixes: the DJ's takeover arms two phrases before its bar and lands ON it; the swap lands on the drop when
B's harmony is out in the bar before it (at plan time from the envelopes, and as the rehearsal's first
answer to a hole at the swap bar); the rehearsal renders the record playing on ALONE as well and judges
the seam against that (A's own 9 dB break is A's, not the seam's). Five more gate runs, one thing each: a
NEXT during a moment's two-phrase countdown was swallowed (a not-yet-running moment now gives way to
NEXT); a SAVE inside a seam's breakdown strip saved two lanes (it remembers the record whole); a RECALL
is the operator's override for the gate's harmony rule; a NEXT-forced or emergency completion counts as a
moved seam for the landmark check; a DROP drops any seam in flight (its next step had pulled a lane away
from the dropped-to song); a cue posted within the last bar is trusted by the pre-flight, not re-read.
Gate ALL OK. Campaign v8: 34 records, pace 3.09 min per record (v6 3.2), 8.3 lane moves per ten minutes,
dead bars 0, dead air 0 % [0-26] of 11 seams, LURCH 18 % [5-48] (v6 43 % [26-63], v7 54 % [29-77]; n is
small, the intervals overlap - the ear decides), double bass 0, harmony split 0, lock p95 0.077, events 19
planned / 16 armed, rehearsals 20: 16 clean, 4 moved (two "an 11 dB step at bar 1 against the record's
own 9 dB", two "a 7 dB step at bar 14"). The landmark stat in v8 read +28 bars for the one bass swap: the
measure took the done bar for the landmark's bar - fixed after the campaign (measured where the landmark
was due), not re-run.

**"Commands don't work" - "a control system needs to respond to controls" (2026-09-16, evening).** The
first command of the night failed: Maaya's vocal, asked for by right-click, was staged and then refused
"not singing there now" - the cue-to-where-it-sings code tried the hook first, the hook was 66 s before
the end (no runway), and that branch returned instead of falling through to the sung stretch at 3:50;
then the seam took Maaya as the next record and its DRUMS came in. Fixed (the fall-through; and with
nowhere that sounds the stem comes in anyway, said). The user's question - what have you built, have you
met the goals - answered honestly: no. The plan agreed ("ok, do it"): (1) no new features; (2) THE
CONTROL CONTRACT, enforced by `tools/tests/_dj_control_gate.py` - every control acts within a bar and a
bit or says why in the story row from a short allowed list, no third outcome, no exception swallowed;
the gate drives the Director headless, paced at 1.5 x real time (a run at 5 x starved the decks), and
presses random controls against random songs in whatever state the machine is in, a third of the vocal
brings from songs with a late hook; (3) a night with the user at the controls as the acceptance test,
every flag answered the same night and turned into a case; (4) taste only after that. Six rounds of the
control gate (90 presses a round, three seeds) found and fixed: a TAKEOVER with the song already playing
said "not ready on a deck"; NEXT during a running moment, or with the seam already due within the phrase,
said nothing; a refused BREAK was logged as "BREAK"; a playing song's quiet stem was still refused; THE
CLOCK WAS LOST - the master left while every other song was leaving too, nothing re-took it, and every
control answered "layer first" for four minutes with the room silent (a room with a song on a deck now
always has a clock); a song whose cue had not landed yet was looped "21 s of body left" from its old
position. Three more rounds over seeds 4-15 (450 presses): a TAKEOVER whose other song had left the
decks raised a KeyError at its step (dropped and said now); a SLAM on top of your plain BREAK took the
last drums out and the room went silent until the watchdog undid it (said: "a BREAK is running"); a
chosen TAKEOVER / BRIDGE, and the plain button, were still refused on a chord clash (said, done); an
undone TAKEOVER left the bass and melody on the silent song (a cancel puts moved lanes back); a staged
song whose "drop" had a silent bassline took the bass lane into a silent room (cued to where its bass
carries, on the one); SING with no hook in reach is an honest refusal. Last two rounds: 270 presses, 0
broken, 0 dead bars; the remix gate green after. Next: the listening night.

**2026-09-17/18, the user's second evening and what it showed.** "The system still doesn't make solid
plans, it just shows potential new songs continuously pushed back": the log said the next song changed
six times in three minutes, all from the machine's reaction to the user's own commands - a vocal brought
in made its song UP NEXT and the seam's next record, it left when its part ended and the plan flipped
back, the next part's staging threw the DJ's staged next record off its deck. Fixed: A PART IS A PART
(`stage_track(as_next=False)`, `part_only`; only UP NEXT makes a song the next record; a part's staging
takes the event's song, a finished part or the DJ's spare pick, never the seam's next record); the
picture's marks anchored once per plan (the wall-clock bar and the conductor's bar count drift); a seam
blocked by a moment moves once, to after it. Then the displays the user asked for: the right-click menu
trimmed to the pair (a fit line; the vocal only when it sings; the bassline; TAKEOVER; BRIDGE only while
the record sings; drums and melody under "more"; ★ and bold for what fits; the tempo wall greys), MIXES
WITH for the highlighted song (after it as the next record with how it would arrive - the pairing file,
version 3, carries per song whether it sings, whether its bass carries, and its seam-ability - whose
vocal sits over it, whose bassline swaps under it, where its own parts sit), the key fit and tempo step
in the SONGS line, PLAYING (the songs on the decks; the record's row highlighted at each change so MIXES
WITH has the playing song as context; eject by right-click or Delete), PARTS as a tab of the right column,
and THE PLAN - "fundamentally, the auto system just doesn't display its plan. this is an ongoing problem
you have not resolved" - an ordered list of what comes next with the time from now (NOW with the record's
minimum play and next section, MOMENT, EVENT with its song's state, SEAM with how, why there, the drop and
the rehearsal, the RECORD after it, UP NEXT; a row says why when there is nothing and "MOVED: …" for
twenty seconds after a move), as the third tab (DIALS | PARTS | PLAN) so it does not squeeze the lists.
Gates after all of it: the control gate found a looping record's strip leaving a silent vocal stem and a
record leaving inside its own strip (the emergency cut fires now; a strip keeps the bass when nothing
else would sound), and a NEXT forcing a bass swap onto the record's breakdown with the next record's
intro bass (a NEXT that lands in a build or a breakdown becomes the cut). Campaign v9 = v8 (pace 3.10,
dead bars 0, dead air 0 %, lurch 27 % [10-57] n=11, events 19/16, rehearsals 15 clean / 5 moved). Both
gates green at the end. Step 3 of the plan, the user's hour at the controls, is still the open item.

**2026-09-19, "what features is this system currently lacking?" → "do the planning horizon and floor
work".** The gap list given (and two of its items withdrawn on reading the code: the deck already transposes
independently of tempo, and the Director already has its nanoKONTROL2 mapping - a gap list is checked
against the code before it is presented). The user's order lifted the feature freeze before the listening
night; noted, built. docs/DJ_THE_SET.md is the page. What was true: the layered engine planned one seam
ahead and nothing after; `base_bpm` was the opener's tempo all night (the clock could not leave it by more
than 6 %, so a set that opened at 118 never reached 128); a queued song outside the wall was popped from
UP NEXT a tick later with a line in the log; a song without stems had no path to the decks during a night.
Built: (1) THE CLOCK FOLLOWS THE RECORD (`remix.py`: `follow_clock`, `_clock_follows` at both clock
handovers - `base_bpm` and the session key become the new record's; `_tempo_step` walks the clock there at
half a percent a bar, every deck re-rated, never past the wall for any deck). (2) `lib/dj/setplan.py`: the
pair score (key over the deck's shifts at the pick's own 0.68, tempo closeness to the step the TEMPO dial
asks for, the arc's energy at the projected time, the pairing file's arrival word, kin), `chain` (a seeded
beam of width four, five deep, after the pinned queue), `route` (a chain of at most three records to a far
song, each hop making headway in tempo), `hops_by_tempo` (the cheap "via N" for SONGS). (3) The Director:
`horizon`, pruned every two seconds and on a trigger, extended on a worker thread (a plan is 150-350 ms;
the Director's thread steps the conductor), the head is the conductor's pick (`pick_hook`, with "wait"
while the clock travels toward where the head fits); `queue` returns a route or an honest refusal and
`_feed_up_next` lets a route's next record wait for the clock instead of popping it; `skip_in_set`;
`adopt_library`; THEN, CLOCK and REQUEST rows in the plan; "via N records" in `rank`. (4) The tab: a menu on
THE PLAN's rows, the walled menu's route, "make it playable" (the planner's one-slot GPU queue with the
fast model → a pairings pass → the library re-read on a worker and adopted), the drop box (the music
folder swept every twenty seconds; `dj_scan.py --add` analyzes one file). First smoke over the library:
a chain in ~200 ms; before the TEMPO lean every hop was the same key at +1 % (the score had no reason to
move) - the lean gives each step a target; the route search needed a headway rule (each hop must close a
fifth of the wall toward the destination) to find two-hop routes at all; half-time records (65 against 124)
stay unreachable, honestly. Five rounds of the control gate (seeds 31-44, 415 presses; the new presses
`UP NEXT far` and `not tonight` among them) found, in order: the churn check's false alarm (a head the
conductor had picked and was decoding was on neither the decks nor the queue - a pick on its way is pinned);
a hop that reached the decks while the search ran (the result is filtered against the room before it is
applied); a chain honestly empty after a 147 bpm queue with no row saying so (a THEN row now says "nothing
fits after X tonight"); a finished result waiting behind the two-second throttle (applied at once); and two
things of the machine's own the gate had not hit before: a BRIDGE checked the vocal at the arm bar, not at
its landing twelve bars on (the room went silent under a vocal that had stopped - now checked where it
lands), and the breakdown strip took a TAKEOVER song's bassline as if it were the record's (harmony travels
together: its melody went too, the song left, one silent vocal for four bars - the strip takes only what the
record holds; another song's bassline carries the breakdown; nothing to carry it = no strip, the cut on the
bar). Last two rounds: 180 presses, 0 broken, 0 dead bars; the remix gate green three times. Campaign v10
(8 seeds × 12 min, play 210 s, the clock following): pace 3.05 min per record (v9 3.10), 8.8 lane moves per
ten minutes, dead bars 0, dead air 0 % [0-22] of 14 seams, lurch 28.6 % [12-55] (v9 27 %), double bass 0,
harmony split 0, lock p95 0.078, events 20 planned / 16 armed, rehearsals 17 clean / 2 moved; the clock
travelled 17 times, a median span of 1.5 bpm per run and 6.9 at most (the brain's picks sit near the clock;
the span grows when the chain or the TEMPO dial asks for it). Nothing regressed. Not run: the planner tab's
new code (the PLAN menu, the walled menu's route, "make it playable", the drop box) was compiled and
imported, not clicked - the Director side of it ran in the gate; the stems path was not exercised (no gate
runs demucs); `dj_scan.py --add` was smoked on a throwaway folder. The user's hour at the controls remains
the open item, and the first thing to watch there is whether the THEN rows read as a plan.

**2026-09-19, 17:36, the user at the controls: "the planner is not producing a stable output. it is still
pushing back the start of songs note by note."** The session's log (sid 9504ff): the seam into Blinding
Lights was planned at bar 5; one bar later the DJ planned its TAKEOVER event (Glam Trap's bassline at the
end of the record's minimum play) - the event IS the record's exit, so the seam was dropped, as designed;
for the next eighty bars there was no seam to anchor to, so the staged song's ghost in the picture was
re-placed at "now plus a phrase" bar by bar (the 09-17 anchor fix covered only a planned seam), and THE PLAN
said "Blinding Lights is staged · its seam is being placed (a bar or two)" for three minutes; then Glam
Trap took over and Blinding Lights' seam appeared a minute later than first shown. Fixed: with a takeover
event pending the picture ghosts the EVENT's song at the event's bar (anchored) and THE PLAN carries a
RECORD row for it ("Glam Trap plays (the takeover is this record's exit) · min play until … · Blinding
Lights follows it, its seam planned then"); THE SET pins the event's song as the next record ahead of the
staged one; a staged song with no seam and no event is ghosted at a phrase boundary and re-placed a phrase
at a time, never bar by bar. Code and compile only - the planner was up; the gate is pending.

**The six flags of 17:38-17:41 (the same session), read from the log.** One story: Glam Trap took over at
its drop with three minutes of body (the pairing file asks 90 s after the landmark); the seam into Blinding
Lights was then rehearsed and failed on marginal steps three times in thirty seconds (9 dB against 8.5, 8
against 5.5, 13 against 9.5), each failure moving it a phrase later ('plan': "could not make up its mind";
and the start of the next song pushed back again); Glam Trap ran out and looped its last eight bars
('runout'); Blinding Lights, staged since the first minute, landed 274 s into itself as "a short record"
and had its own seam ninety seconds later; the same rehearsal pattern moved that seam twice more and
Blinding Lights looped its end too. Meanwhile the user's vocal of Next To You rode into Blinding Lights'
harmony unjudged ('clash', 'vocals muddy') over a bed the matcher held at 0.74 under a loud vocal ('thin');
and the user's vocal of Just Don't was cleared at the next seam's first step as a "stray" belonging to
neither record - an unlogged note - and the record's vocal came back "singing again" ('the system overrode
my vocals choice'). A 2.4 s dropout at 17:37:01, thirty seconds after Glam Trap took the clock with every
lane, is not explained by the log (the record's own silence is the likely class; the day log does not carry
silence_log). Fixed: a BASS UNDER candidate needs runway for the minimum play and a seam (play_min + 45 s),
not 90 s; a failed rehearsal changes the seam IN PLACE first - the other order, then the breakdown and cut -
and moves it a phrase later only when the record has the body for it; a part the user brought is never a
stray (it stays through the seam, said); a vocal the user holds is judged against the incoming harmony when
the seam is planned and a clash is said (it stays: their order). Not changed: the loudness matcher's bed
level under a loud vocal (one flag, no number yet).

**18:08, the next session (32105f): "it still takes my new vocals and just dumps them and returns to the
original vocals. why is that? ... like, i'm supposed to be in charge here."** The log: the vocal of You Are
My High in at bar 36, cued to its hook; at bar 54 its singing paused, the lane rested and the room-never-
empty rule put the record's vocal back the same bar ("the record plays whole"); the record's vocal never
rests, so the user's song never got the lane back and left four phrases later. The rule was written for the
record's own vocal and never told a vocal the user chose apart. Now YOUR VOCAL HOLDS THE LANE
(`_vocal_hold`, set by `bring` on the vocal lane): when its singing pauses the lane rests in silence and
says for how long ("pause: the lane is yours, silent until it sings again, in about 12 bars"); only that
song's vocal comes back on it; the record's vocal returns when the song has sung its last (said), when you
take the vocal off, DROP, eject, or bring another; its song stays on its deck while it will sing again
within its body; a seam's vocal step waits behind it. Code and compile only - the planner was up; the gate
is pending.

**The seven flags of 18:11-18:14 (session 32105f, three layers) and what the log said underneath them.**
The conductor's bars took 2.4-3.0 s against a real 1.97 for the whole session (25 logged bars in 70 s at one
point); every earlier night ran at 2.00 (measured per minute from the remix_move rows: 09-16, 09-17, 09-18
all 1.95-2.05 except the decoding minute at a start). The decks' own time advanced at 65-85 % of the wall
clock: THE ENGINE WAS BEHIND REAL TIME, and nothing said so - it read as 'dead air', 'nothing happening for
too long', 'seam', 'clash'. Suspects: the chain's worker (a burst of pure Python next to the audio producer,
re-run on every staging because the decks were in its trigger), a fourth deck of RubberBand, and today's
extra per-tick work. Done: the worker yields the GIL every 64 candidates, skips songs outside the wall
before any arithmetic, and no longer seeds a Random per candidate (a third of a plan's time); the decks'
comings and goings are no longer a trigger (the result is checked against the room when applied); and a
REAL-TIME SLIP MONITOR in the Director (`_slip_step`: the clock song's time against the wall, its rate
allowed for, ten-second windows) says "⚠ the engine is behind real time: the music ran at 78 %" in the story
row once a minute and logs a `slip` row - the next such night is named at once, not inferred from bar
counts. Not proven: which of the suspects it was; the measure is in place for the next session, and the
first test is the same set at LAYERS two. Also from the log: the seam "before Carte Blanche runs out" was
placed as a 32-bar morph with 48 s of body left, so the record looped its last eight bars through most of
it ('clash' at 18:14) - a seam placed before A runs out now shrinks to what fits (eight bars, else the
cut); the user's TAKEOVER with Carte Blanche read 'clash' and 'seam' at once - Kigelia's vocal (7A) over
Carte Blanche's harmony (7A, whole-trim 1.6 against Kigelia's 0.86): a level step by design ("loud"), not
a key clash, left as it is; 'the low end wrong' at 1.6 on Carte Blanche's bass is the same whole-trim, one
flag, no change yet.

**"I feel that we don't necessarily have good transitions now. they seem pretty abrupt at times, which is
something we worked a lot with on the dj side" (18:45).** The log agreed: nine record changes that day, seven
a bassline and melody on the one, loud. Three causes: every record's exit was a TAKEOVER event (MOMENTS
some), the seam steps crossed in beats (drums 4, harmony 2, a cut half a beat) while the autoDJ's exits
fade over 16-32 beats with the EQ hand-over, and the rehearsal's first fallback turned morphs into bass
swaps. The one-pager (four points) got "ok, do it all": (1) the DJ's exit events are seams only under MIXING
blend / morph, and at unity under auto / cut - loud stays for the moments the user presses; (2) the seam
steps cross for as long as the SEAMS dial says (quick = the old switches, normal = two bars a step, long = a
phrase for the drums and four bars for the harmony), the harmony step placed so its cross completes on the
bar before B's landmark, the bass handed over under a low shelf (the old harmony's lows down to the carve as
the new one's come up); (3) the rehearsal keeps the shape (move a phrase with body, else the cut - no order
swap); (4) the strip and the cut untouched. docs/DJ_THE_RECORD_PLAYS.md "How the steps cross". Code and
compile only - the planner was up. Gates when it came down: control gate 60 presses ALL OK, 0 dead bars;
the remix gate green three times in a row (its log now reads "its bassline and melody cross in over 2 bars");
campaign v11 (MIXING auto, SEAMS normal): pace 2.99 min per record, dead bars 0, dead air 0 % of 13 seams,
LURCH 15.4 % [4-42] (v10 28.6 %, v9 27 %), double bass 7.7 % (1 of 13: the two basslines overlap inside a
cross now - the low shelf keeps it under most of the time; watch it), harmony split 0, lock p95 0.078,
rehearsals 19 clean / 1 moved (v10: 17 / 2), events 19 planned / 16 armed (auto keeps the takeover exits, at
unity now). The ear decides; the numbers moved the right way.

**"takeover commands take forever to implement" - "once again, this is the system not listening to
directions" (19:50).** The log of the two evening sessions: a TAKEOVER on a song whose bassline the user had
already brought in was refused "its bassline is already the room's harmony" (three times - the takeover never
finished); a TAKEOVER pressed while the DJ's own takeover was armed became "TAKEOVER now" WITH THE DJ's SONG
(the same-kind-again = go-now rule never looked at which song); and a user's TAKEOVER landed on the song's
drop within four phrases (up to a minute of waiting after an eight-second staging). Fixed: a takeover whose
bassline is in finishes the job (its drums follow on the next bar, it is the record, said); a moment armed
with a different song is cancelled and yours armed; your two-song moments look one phrase ahead for a payoff,
else the next phrase, and the NOW / PHRASE toggle applies to them too (NOW = the next bar plus the lead).
Code and compile only - the planner was up; the control gate (it presses TAKEOVER with, now with both
timings) is pending.

**The flags of 19:44-19:52 (session d4b2fd, the first with the crosses).** 'great transition' on the first
two-bar cross (Zephyr → Tarantula). Then three flags in one minute: 'nothing happening for too long' after
Fellaheen played four minutes whole at VOCALS lots with no event (349 vocal candidates in its list, none
staged - the event planner set "nothing usable" silently; the likely reason is no free deck, with the user's
Beam Me Up and the seam's Nagar on the others); 'bad transition' and 'clash' at Beam Me Up → Nagar: Beam Me
Up, brought in as parts (vocal, then bass) and made UP NEXT, became the record where it was playing (5:30
in) - "a short record" of 55 s - so Nagar's bass swap came under the user's held vocal (7B over 8B, fit
0.9) and, the moment the seam completed, the clock travelled to Nagar's tempo, +4.4 % over nine bars,
as that vocal ended (on a second reading the vocal had rested and Beam Me Up had left the moment the travel
began: the ramp ran on Nagar alone; the 'clash' is the swap itself). Tried and dropped, both against the remix gate's tempo
check: a hold "only while the record is alone" (it stopped the arc's journey too) and a quarter-percent
step (0.75 % in the gate's ten bars against the 1 % it asks) - the half-percent step stands, and whether a
4 % travel over nine bars is audible on its own is an ear question, open. Fixed: the event planner says why a
record got no event ("no event for X: <the last reason>"). Open from the same run: one dead bar in one of
six remix runs inside the scripted BRIDGE's beat of nothing (the bed's vocal was at 0 in that bar too) - the
BRIDGE-hole class, one in six.

**"the system basically lies about its transitions" (20:33, session 88cae1).** The log: at 20:30:08 the plan
said "Dunes in at bar 155: a 8-bar morph, drums first"; at 20:30:11 the DJ planned its exit event - a
takeover with Collecting Stories at bar 64 - which drops the seam by design, through a private note, not a
story-row line; the user queued Collecting Stories UP NEXT at 20:30:32 expecting a seam; at 20:32:51 it
arrived as a takeover (bass and melody on the one, drums eleven seconds later), flagged 'bad transition'.
MIXING was auto, which the afternoon's rule still allowed takeover exits under. Fixed: takeover exits only
under MIXING cut - under auto, blend or morph the records change through seams and the plan shown is the plan
played; a seam dropped for an event is said in the story row and logged (`plan / exit`); the event's words
say what it is ("TAKEOVER at its drop: X's bassline and melody under these drums, its drums a phrase later -
it becomes the record"). Gates when the planner came down: control gate 60 presses ALL OK, 0 dead bars; the
remix gate then found, with takeover exits gone from its auto night, a NEXT-forced completion of a bass swap
that fired only the drums - the harmony step had not come yet, so the new record took the drums while the
old one kept the bass: 12.8 bars of split harmony. A forced completion now fires every step that still moves
a lane, a bar apart in order (harmony first, drums the bar after); the darkening and the strip are skipped.
The gate's EVENTS check accepts a night where every record without an event said why (`event_none`: singing
records have no vocal candidates, and under auto there are no takeover exits any more). Remix gate green
after (harmony split 0.0 bars).

**2026-09-20 13:31, 'clash' + 'seam' (session 86ac2f).** Three records in two and a half minutes: God Sigma
became the record at its THIRD drop (3:14 of 7:35 - its first groove's bass dipped in one bar, the body moved
a section later, and the landmark is "the drop after the body"), had 1:40 of play left before the rule's
minimum-play-plus-runway, so Fiji's seam came early; Fiji landed deep the same way; Mahoya followed 26 s
after. God Sigma's own vocal (7B) carried over Fiji's harmony (7B) for 17 s because the Camelot fit said it
could: 'clash'. Fixed: the next record's landmark is its EARLIEST drop that leaves the minimum play, a
runway and a seam (`_record_landmark`: grooves after a build / breakdown / intro from the first non-intro
section on), else its body when that does, else the old landmark; and at a bed change the old record's vocal
always goes with its harmony (twice in one evening a same-key vocal over the new harmony was flagged; a
vocal the user holds stays). Code and compile only - the planner was up; the gates and a campaign are
pending (the landmark change moves every seam's placement: the campaign's pace and lurch must be re-read).

**13:45-13:48, 'volume keeps fluctuating', 'vocals keep coming in and out', 'vocals wrong', 'still bouncing in
and out from starlit'.** One cause, mine: Starlit (the user's vocal, brought in at 13:38) ran out of body at
13:40 and went into its last-eight-bars loop; the vocal hold keeps a song on its deck "while it will sing
again", and inside a loop it sings again every pass - the lane paused and returned every eight bars from
13:42 to 13:48, the melody re-ducked each time (the level "fluctuating"). Fixed: a looping song has sung its
last - the hold ends ("X has run out: its vocal rests, the record's vocal returns"), the lane does not come
back to it, its deck is not kept, and an evicted song's hold is cleared. The 13:32 'dead' (Mahoya alone at
1:56 with every stem at 0.74) is not explained by the log: no slip, no move; the record's own level is the
likely class.

**"what can we do to make it more than a playlist mixer?" → "do all 4 things" (2026-09-20).** The page:
docs/DJ_MORE_THAN_A_MIXER.md. Built, in the order 2-4-3-1: (2) THE DJ'S IDEAS AS ROWS - a planned event is an
IDEA row in THE PLAN (do it / skip; it passes, said, two phrases before its bar; MOMENTS lots = the DJ takes
its own; a takeover idea drops the seam only when taken); (4) BUILD (the eight bars before the record's drop
loop once more, the bass pulled on the second pass, released into the drop on the one; press again =
release now) and AGAIN (the current section once more from its start on the phrase) as moment kinds with
their buttons; (3) HALF-TIME / DOUBLE-TIME PARTS - a deck's beat multiple for the PLL, the rate the clock over
the tempo times the multiple, parts only (UP NEXT on such a song refused with the reason, the clock never
theirs); (1) PAIRS, the fourth tab of the right column - the best pairs across the library ranked, coloured
by lane, filtered to the clock and to the highlighted song; HEAR renders eight bars offline
(`lib/dj/preview.py`: the decode in the helper process, RubberBand offline for the stretch and the shift,
peak 0.9) and plays them on the CUE OUTPUT (`lib/dj/cue.py`: a second playback device named by
`dj.cue_device` in config.yaml - this machine has a Headphones (Elgato) and a Logitech headset - never the
floor); DROP IT queues the record and plants the pair as an idea already taken (`plant_idea`, placed by the
event planner whatever MOMENTS and MIXING say). Gates: control (IDEA do it / skip, BUILD, AGAIN, half-time
parts pressed), remix, the preview smoke (`_dj_preview_smoke.py`, a WAV, no device) - pending while the
planner is up. Gates after: preview smoke 16.85 s at peak 0.900 (rendered in 13 s), control gate 60 presses
ALL OK with IDEA do it / skip and AGAIN pressed (BUILD not drawn by the dice), remix gate green. Then OUTPUTS
(a button: the MAIN output and the CUE output from the machine's playback devices, per machine in
logs/dj_outputs.json), and DROP IT renamed PLAY THIS PAIR with the action spelled out ("very ambiguous
phrasing").

**"it isn't clear the pairs system is working" (14:24, session fe5985).** Three pairs taken, none fired: each
record went UP NEXT and became the record, then "your pair waits: no phrase of the record inside one of its
grooves in the window" for the rest of its play. The pair names a groove ("the vocal of X over Y from
5:12") but the record was cued to its first drop near 1:30 and its seam came four minutes later - the
groove was never inside the play (the 0:29 pair's groove was before the record's start). Fixed: a pair you
chose decides where its record starts - five phrases before the pair's groove, on a downbeat
(`_planted_start`, ahead of the landmark rule); the PAIRS row says so ("the record would start at 3:54"); a
pair whose record is past its groove is dropped and said ("your pair missed its groove") instead of waiting
forever; and each row says how the pair's record follows the record playing now ("into it after Beam Me
Up: key 0.55 (a clash)") - the 14:14 'seam' + 'clash' flags were a pair's record queued into a clash the
list never mentioned. Code and compile only - the planner was up.

**"the pairs system is still not working?" (14:34, 14:46) - "plans have to be rock solid. we can't have any of this
nonsense".** Round two from the log: the pair set the record's start but the seam's carry check walked it back
77 bars. Round three: the start held, but the seam OUT of the record was placed thirteen bars in, before the
pair's groove ("a short record"), and the "no room" branch dropped the pair silently; the DJ's idea had died
at a 150 s timeout meant for events. Then THE PAIR SMOKE (`tools/tests/_dj_pair_smoke.py`: the path driven
end to end headless, failing on the first thing that does not happen) replaced the post-mortems and found two
more in two runs: a pair on a record's FIRST groove (nothing carries before it) entered at the groove itself
after a drums-first seam from the intro, and its event window - four phrases for a decode - closed before the
groove; and under MOMENTS rare the event planner returned before looking at a pending event at all. Fixed,
all of it: the seam out of a pair's record comes after the pair's groove; a pair the record cannot hold is
said and dropped; ideas have no timeout but their bar; the pair's other song is STAGED when the pair is
taken, so its event needs one phrase, not four; a pair on the record's first groove enters as a cut on the
groove, whole (`pair_record_start`, shared with the PAIRS list's "the record would start at"); a pending pair
arms whatever MOMENTS says; and PLAY THIS PAIR refuses an impossible pair on the spot (no stems, outside the
wall, a groove under ten bars, no deck) - a plan you take happens or is refused when you take it. Pair smoke
green on two seeds (the cut on the groove at 1:03, the pair planned five bars later, armed, the vocal over the
record for 16 bars, no dead bars); control gate 60 presses ALL OK; remix gate green. Not changed: a part-song made UP NEXT becomes the record
where it plays (it cannot be re-cued while its parts sound); the seam text already says "a short record".

**"impossible pairs should not be displayed in the first place" → "are there any other systems that can
attempt to queue up invalid mixes?" → "do them all" (2026-09-20).** PAIRS first: every refusal of PLAY THIS
PAIR applied to the list (the wall, ten bars of groove, a deck for the pair, the record playing not past its
groove, no song that is already the record); the "fits the clock" toggle went. Then the audit of every other
offer against its take, six gaps, all closed: (1) UP NEXT dropped a queued song when the decks were full -
the wait matched an old wording of staging's refusal (`"no free deck - eject one first"` against `"no free
deck - the decks hold …"`); it waits now and says so. (2) A song without stems could be put UP NEXT in
layered mode and was popped at staging a minute later; refused at queue time, and the menu no longer offers
it. (3) THE SET's chain and the live pick asked different questions (key against the predecessor at the
predecessor's clock vs key against the room at today's clock): a vetoed head fell to the brain silently and
the THEN rows were not what played; `pick_veto_why` is the pick's gate in one place, and a head that fails it
leaves the plan with the reason in the story row before the brain picks. (4) The DJ's IDEA was placed on a
groove the pairing file called instrumental and refused at its bar "is singing here" nine times on the 18th;
the placement now asks the arm's own measure (the vocal stem's envelope over eight bars). (5) MIXES WITH
listed the pairing file unfiltered; it applies PAIRS' rules (played, on the decks, queued, ten bars, the last
minute, twins). (6) TAKEOVER / BRIDGE with a song refuse at once for what can be known before staging (the
record itself, no stems, the wall). Not changed: the moment buttons refuse at the press with the reason
(that meets the rule); the autoDJ's "would fade" songs stay queueable by design.

**"the second part of the pair hasn't played" → "wasn't this problem supposed to be resolved?" → "stop chipping
away at the margin of filter tuning and address the actual issues" → docs/DJ_ORDERS.md → "do it all"
(2026-09-20, evening).** The Resurrection pair (its vocal over a first groove after a 62 s intro with drums and
no bass): the PAIRS list said "starts at 0:31" from section geometry, the cut placement wanted drums AND bass
carrying and fell forward onto the groove, the event wanted a phrase boundary with a phrase of margin and the
one that fit was inside the margin. Three models of one fact; the DJ's taste rules judging an order; the
display rendering state, not the plan; and a smoke that only ever took the best-scoring pair. The one-pager
(DJ_ORDERS.md) and the build the same evening: an order is a SCORE - `lib/dj/orders.py` plans a pair once from
the record's real stems with the conductor's own judge (three doors: drums first five phrases early; a cut at
the latest place before the groove where its drums carry; on the groove with the vocal four bars in - else
refused by name), and the seam, the event and the arm READ it: `_seam_b_cut` / `_seam_b_start` / `_plan_seam`
search nothing for a scored record, the event bar is geometry from the score's time and where the record is
(no phrase grid, no window), a seam in the way moves past the pair, the arm lets a scored acapella through
where the record's stem sings (said). Every end is said (done / refused / failed + reason), a refused pair's
record leaves UP NEXT, and THE PLAN's PAIR rows and the picture come from the order object itself - from the
moment the pair is taken, not from the moment its record plays. The pair census (`_dj_pair_census.py`) runs
that planner over the offered pairs offline and writes `.stems/pair_plans.json`; the list hides what it refused
and says the entry it decided: top 40 → 40 feasible, 0 refused. Verified: Resurrection replayed through the
score (planned the bar it staged: "enters whole at 0:46, a cut, its groove 8 bars on"; the vocal 12 bars on
the groove), the best pair (16 bars), control gate seeds 5-6 with the new PLAY THIS PAIR press and the
end-of-run orders check (60 presses ALL OK), remix gate green. Not yet: UP NEXT and TAKEOVER / BRIDGE as full
scores; the census by the pairings pass.

**2026-09-21, the morning at the controls (five reports, each answered from the log within the hour):**
(1) "the plan does not match the timeline" - the next-record ghost drew the first staged song, not the seam's
B (two staged: the DJ's pick and the pair's record); the PLAYING list did not redraw on a state word; a 0:00
groove said 0:00 for an entry heard at 0:12. (2) "the vocals from another song last a pathetic amount of
time" - 8 and 16 bars live, then "ACAPELLA over: the vocal goes" and its song off the deck: the acapella
script's two-to-four-phrases-and-gone was the DJ's rule for its own idea, applied to the user's choice; a
vocal YOU chose (a pair, an idea taken) now holds the lane under the vocal hold's rules until it has sung its
last or you take it out - the smoke measures the hold (48 bars and still holding). (3) "why can't this stuff
run when the planner is playing?" → "stutters are understandable. It's just when the system decides to simply
not play anything that I criticise" - the gates run at IDLE priority for the whole process tree
(`tools/tests/_low_priority.py`, first import; the 09-12 helper ran at normal priority because it was spawned
before the drop) and may run beside the live planner, one at a time. (4) "the system refused a suggested
pair" - Raw's groove at 0:00, 16 bars: the list said "enters at 0:00" from the census of the night before;
the morning's planner (a cut is heard six bars in at the earliest) refused it - the census was stale against
its own planner; verdicts now carry `PLAN_VERSION`, and the list's own geometry applies the cut floor for a
groove at the record's very start. (5) "the system takes a long time to start a new song ... when it starts
up" - 8 to 16 s from START to the first note across twelve sessions: the Director built at START (1.5-7 s),
the decode helper spawned on the first decode (2-3 s; the warm-up was placed after it), the opener's decode
(5-8 s). READY BEFORE START: all three while the window sits open (`Director.prepare`, off the GUI thread);
START attaches the engine and plays within a bar; the start smoke measures 7.4 s → 0.6 s. Verified after all
five: Resurrection and the best pair (the picture checked: PAIR row from the take, the vocals mark and the
other song's clip from the staging; the ghost's identity against the seam), control gate seeds 7-8 ALL OK,
remix gate green. Lessons written to memory: a census is only as true as the planner that wrote it; replay
the failing case by name; the user holds SILENCE BY DECISION against the system, not a stutter from load.
**The evening's flags (same day):** "the vocals in the pair are too faint to hear mostly" - the chosen vocal
at unity (never trimmed up against another song) under a bed at 0.86, and the hold resting the lane in every
four-bar pause with a one-beat fade back on the first detected beat (eight rests in three minutes, the head
of every phrase lost): a chosen vocal is now brought to the library's vocal level and its lane stays through
its pauses. "bad transition / seam" - the score's drums-first door tried one place (five phrases before the
groove) and fell to the cut; it tries five, four, six, three, seven, eight (PLAN_VERSION 3). Verified: both
named pairs (the vocal held 48 bars, 0 rests, lane gain 1.6), remix gate, census 60/60. "you need to not play
the same songs repeatedly after a restart" - the night's memory (`Director.played`) lived in the process: the
opener, the chain and the lists forgot the evening at every START. Now every song that becomes the record is
a 'played' row in the night log and `_seed_night` reads the last eight hours at construction (older rows by
their "takes the clock" / "opens on every lane" titles); headless runs neither write nor read it. The real log
gave twelve songs for tonight at once. Test: `_dj_night_memory.py`. **The evening's second flags:** "dead air"
- a 14.7 s silence on the bus at 19:54 with every deck at gain, no slip row, two decodes in flight for a pair
(cause unproven: the dropout row now carries memory, the decks' cursors and the decodes in flight, and a stall
inside one ten-second window is said and logged at once). "bad transition / seam" + "there was basically no
fade between songs, just a bland switch" - a 16-bar morph rehearsed a 10 dB step at its harmony swap, moved
twice and fell back to the cut: A LEVEL STEP IS A LEVEL, NOT A SHAPE - the judge now reports the step's sign,
the arriving stems are trimmed by that much at the cross and ease back over a phrase (`step_trim`, read by
`_lane_gain` so no re-post lands back on the step), and the morph stays a morph; the cut fallback is for holes
only. The gate then found (seed 9) two dead bars in a NEXT-forced cut's strip: the record's vocal "sang" by
its own measure but was near silent in the room; the seam's strip now applies the BREAK's absolute rule
(what stays must reach BREAK_LEFT_MIN of the library's drum level) - seed 9 clean after. Verified: remix gate,
Resurrection, control gate seed 9. **Three more "bad transition / seam" flags (20:15-20:27):** the middle one
was a thirty-second record - Medes' bass carries nowhere the seam could use before 6:09 of its 7:33, the bass
swap put it in there, and its seam out was forced "before its minimum play" into its quiet tail (the room at
-28 dB). The pick's runway rule reads the section map, the placement reads the stems, and where they
disagreed the placement won silently. Now such a placement makes the song UNFIT as the next record tonight
(said, logged, skipped by the picker, the chain and the seam; a user's pick plays as a short record, said). The
other two (a build's drums under the record; a bass swap into a record's groove) have no measurable cause in
the log and wait for the user's word.

**"what is all this building to? seems like we're making a bunch of weak instructions with no feedback and
rough transitions" → "the transitions are really rough right now. fast, no blending, no artistry" → "yes"
(2026-09-21, night): THE BLEND (docs/DJ_THE_BLEND.md).** The honest answer: the plumbing was true by the gates
and nothing the gates measure hears a seam; every seam rule is a proxy, and at the default SEAMS setting a
"16-bar morph" was three two-bar switches a phrase apart with the outgoing record gone in a beat and a half,
linear ramps dipping the bus in every cross, and any rehearsal verdict collapsing the shape to the cut. The
page and the build the same night: a seam is 32 bars (64 as a blend), every cross a phrase and equal power
(the deck's stem ramp: sin in, cos out), the harmony lead and landmark offset following the cross, A's vocal
leaving over the cross, B's leading stems entering only from a groove of its own that carries, the placement
ladder 32 → 16 → 8 → the cut; the rehearsal models the real crosses, renders every rehearsed seam as AUDIO
(logs/seams/), a 'seam' flag keeps the sound beside it, and HEAR THE SEAM plays the render on the cue output
from the PLAN's SEAM row - the headphone cue promised on 09-13, delivered. The campaign after: the level
step across a seam median -0.3 dB (was -3.3), double bass 0 % (was 7.7), lurch 25 % (2 of 8, the drums'
cross), minutes per record 4.4 (was 3.0), one song alone 354 s (was 192).

**"We are not making a system that really does anything interesting. It slowly and ponderously plays songs
with slight overlap. This is not a good system" → docs/DJ_THE_COMPOSER.md → "Yes" (2026-09-22).** The honest
answer: two rules I wrote to end the clashes and the churn - "the record plays whole" and "one idea per
record, only if you say do it" - ended the play; the blend made the seams smooth and the campaign showed the
DJ's own events at zero. Built the same night: THE COMPOSER (`lib/dj/composer.py`, pure) - every record gets
a score of MOMENTS-many events from its own pairing lists and section map when it is staged as the next
record (or when it is the record already): a guest vocal on a groove (held), a GUEST BASS (a new moment: the
other record's bassline and melody under its drums for two phrases, then its own back), a BUILD into its
own drop, and the exit as a TAKEOVER when the next record is in its lists; each is an ORDER (a record now
holds several; `_planted[rid]` is the current one), shown as SCORE rows the moment the record is staged,
fired on its bar or ended with a reason; records with two or more events move on at seven tenths of the
minimum play; only YOUR pair drives where a record starts. The composer census over the library: at MOMENTS
some 27 % of records get two events, 45 % one, 28 % none - the pairing lists are the limit (260 vocal
candidates), not the composer. The compose smoke (twelve rendered minutes, MOMENTS some): five scores, a
guest vocal, two guest basslines and a build sounded, one build let go with its reason, no dead bars; remix
gate and the Resurrection pair green; control gate seed 12 clean but for one dead bar in the old
moment-abandons-seam → "the room had nobody" handoff (eight seconds at gain zero before the staged record
was put on whole - not the composer's, still open).

## Rules carried over (they were earned)

- One change at a time, measured on a library-wide sample, never one track; keep only
  measured wins; record every rejection with its numbers.
- The user's ear outranks the statistics; fence a verdict when the execution is suspect.
- Nothing heavy or audible runs while the user is working; one GPU job at a time.
