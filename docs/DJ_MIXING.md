# What mixing is (and what the engine does instead)

*2026-10-24, after the record mode's first night: "that transition was terrible", "go look up what
actual mixing is. It's not just cutting the last song off for a beat then starting another."
This page is the look-up. Nothing here is built until you say yes to section 4.*

## 1. The one idea

**The overlap is the transition.** A DJ mix is two beat-matched records playing *together* for a
whole phrase or more, with the DJ deciding bar by bar which one owns the low end, the mids, the
highs, and the volume. The moment the second record "takes over" is a point inside that overlap,
placed on a phrase boundary, not the start of it. A cut (one record stops, the next starts) is one
tool among several and the sources agree it is the exception: energy shifts, genre pivots, section
changes. For a room in a groove it is the wrong tool.

Numbers at this library's tempo (116 BPM, one bar = 2.07 s):

| | bars | seconds |
|---|---|---|
| one phrase | 8 | 17 |
| short blend (decisive) | 16 | 33 |
| standard house blend | 32 | 66 |
| long blend | 64 | 132 |
| tonight's record-mode cut | 0 overlap, one beat of silence | 0.5 |

## 2. The rules every source repeats

- **Beat-matched.** Kicks locked, and they stay locked for the whole overlap (the drill is 64 bars
  without drift). Real DJs do this with the pitch fader, a few percent of rate, no time-stretch.
- **Phrase-aligned.** Tracks move in 8/16/32-bar blocks. The incoming track starts on the *first
  bar of a phrase* of the outgoing one, and it is started so that its own structure lands where
  you want it: the forum arithmetic is "A's outro starts at bar 32, B's intro is 8 bars, so B
  starts at A's bar 24 and B's body lands exactly on A's outro". Mixing mid-phrase "feels off even
  when beat-matched".
- **Where in the records.** The incoming track's *intro* goes under the outgoing track's *outro or
  breakdown*: the places built with room for a second element. You mix out of a record from about
  two-thirds in, with a few phrases left, never out of its peak.
- **One bassline at a time.** Both low ends together is mud. The incoming track enters with its
  bass killed; the bass changes hands once, on a phrase boundary.
- **EQ handover is top-down.** Highs first (about 4 bars), then mids (about 4 bars), then the bass
  swap (8–16 bars, or one clean move on the phrase). Cut on the other deck rather than boost.
- **Vocals never over vocals.** And a vocal is checked by ear; the waveform does not show it.
- **Energy decides the shape.** Warm-up and release: long blends, breakdowns, filters, echoes.
  Peak: shorter, decisive swaps. A set has one main peak about two-thirds through, not a plateau.

## 3. The shapes, as you would hear them

Ordered by how often a progressive / organic house set uses them.

1. **Long blend** (the workhorse). B's intro starts, inaudible, on a phrase of A's last third with
   its bass killed. B's fader comes up over a phrase. Highs trade over 4 bars, mids over 4, then
   the bass swaps on the next phrase boundary. A's fader comes down over the following phrase.
   32 bars standard, 64 when both records give you the room. Same or adjacent key, similar energy.
2. **EQ swap** (decisive). Same overlap and placement as the long blend, but every move is instant
   and on a boundary: highs at a 4-bar line, bass slammed on the phrase change, mids at the next
   4-bar line, A taken out sharply at the next phrase. For building toward a climax.
3. **Breakdown swap** (this genre's own). A goes into its breakdown; B's intro starts under it;
   B's beat drops exactly as A's breakdown resolves. Needs 8+ bars of open breakdown in A and a
   build in B that lands. High key compatibility.
4. **Filter sweep.** A low-passes out (highs go first, mids, kick last); B high-passes in (highs
   first, lows last), over a phrase or two. For records of different texture.
5. **Layering off a hot cue.** B's vocal breakdown is dropped, full volume, bass killed, on the bar
   where A goes instrumental; A's mids dip to make room; bass swaps on the next phrase; A sits a
   third down and is killed a phrase later. Mix out of A early, about halfway. Energy up.
6. **Loop and build.** A 4- or 8-beat loop of A under a rising filter, reverb, noise, for up to
   16 bars; on the one, release and let B hit. For a peak moment, rare.
7. **Echo out.** On the last beat of a phrase A's echo is thrown and A is killed; the tail rings
   2–4 bars while B comes in underneath. Section changes only; "a set full of echo throws tires."
8. **Cut.** A killed on a phrase downbeat, B starts whole on the same beat. Energy shifts and
   pivots. Not for a room in a groove.

## 4. Where we are, honestly

**Record mode (last night).** It fixes the right thing: one record, original mix, untouched. But it
has exactly one transition, and it is the bare cut: A's gain to zero one beat before the one, a
beat of nothing, B whole at its landmark. No overlap, no EQ, no phrase placement in A (the cut
comes at the first phrase boundary after the 3:30 minimum, wherever A is: Zephyr was in its peak
groove with ninety seconds left). None of section 3 exists in it.

**The autoDJ (the old engine).** It does know the shapes: a style menu with blends of 64, 32 and
16 beats, EQ staging for the bass, breakdown swap, filter sweep, echo out, cut at drop. Measured
against section 1 it runs short (64 *beats* is 16 bars, half a standard house blend; last night's
filter sweep blended in over 3.3 s), its lengths get halved by flam and groove gates, and dice pick
the shape. The layered Director on top of it then moved the music for a dozen other reasons.

## 5. What I propose to build (say yes or no to each)

**A. Record mode's transition becomes a mix.** The record still plays whole and untouched. When
it has played its minimum, the next record is started on a phrase boundary of *this* record's last
third, at the point that makes the next record's first body phrase land on a boundary of this one.
It enters with its bass killed, highs then mids over 4 bars each, bass swapped on the next phrase
boundary, this record faded out over the following phrase. 32 bars by default; 64 when both
records' outro and intro allow it; 16 only when the room needs a decisive move. Two decks, three
EQ bands and a fader each, on the original mixes. No stems, no dice: given the same two records,
the same mix, every time, and the plan says the bar numbers before it happens.

**B. Beat-matching by rate, not by stretching.** The incoming record is pitched to the playing one
within a few percent, the way a pitch fader does it, and stays there. No Rubber Band in the
record path. (The rescope said "own tempo"; a real blend cannot be done without beat-matching, so
this is the one place I would change that.)

**C. Breakdown swap when the records offer it.** If this record has a breakdown of 8+ bars in
its last third and the next record's build fits under it, that is the mix instead of A.

**D. Cut and echo out stay, but only on your order.** NEXT, or a mode you turn on for a peak.

**E. Not in this step.** Stems, figures, moments, loops, the pads. First the room has to hear two
whole records mixed properly, every time. Then we talk about what gets added on top.

If A to D are yes, the next page is the plan: which code is reused (the autoDJ's EQ staging and
phrase math), what is new (placement from the records' sections, the top-down EQ schedule), and
the gate (every rendered seam measured for overlap length, one bassline at a time, phrase
alignment, and your ear on five of them before it goes live).

## 6. Status

**2026-10-24, the decision:** "do A and C for now. definitely NOT B." Keylock (Rubber Band) stays;
beat-matching is the stretcher and the PLL, never varispeed. D and E were not answered: the cut
remains only as the fallback and NEXT; nothing else was added.

**Built (2026-10-24, `lib/dj/recordplay.py`, uncommitted).** The record player's cut is now the
blend of section 5 A, planned the moment the next record is decoded, in the record's own bars:

| bar | the record (A) | the next record (B) |
|---|---|---|
| K (a phrase boundary at or after the minimum play, not into a build or breakdown) | plays on | starts at rate bpm_A / bpm_B, synced to A; bass killed, mids and highs at 0.8; fader 0 → 1 over L/2 bars (constant power) |
| K + L/2 | highs to 0.5 over L/4 bars | highs to full over L/4 bars |
| K + 3L/4 | mids to 0.6 over L/4 bars | mids to full over L/4 bars |
| S = K + L | **the bass swap**: low band to 0 over two beats | low band to full over two beats; B is the record |
| S … E = S + T | fader 1 → 0 over T bars (constant power), then stops | plays whole; glides back to its own tempo over 8 s |

L, T = 16, 8 (standard, 24 bars of two records together), 32, 16 (long: B's intro is 32 bars or more
and A has the room), 8, 4 (short: NEXT, or A has little room). B is cued so its landmark lands on
the swap bar when its intro is long enough, else from its first downbeat. STAY moves the whole plan
a phrase later while it has not started; NEXT under way brings the swap to the next bar + 2 with the
short tail. The breakdown swap (5 C) is planned first when A has an 8+ bar breakdown resolving at or
after the minimum play and B has a build into its landmark: B in 4 bars before the breakdown, bass
killed, full by the breakdown; A's mids dip; on the bar where A's breakdown resolves (its own drop
within 16 bars, else the breakdown's end) B's bass opens and A is out in one beat. The cut survives
only when a record runs out before any blend fits or the tempo gap is beyond 8 %.

**Gated (`tools/tests/_dj_record_gate.py`, rewritten for mixing; seeds 1-2, nine rendered minutes
each, play_min 60 s): ALL OK.** 10 of 10 transitions were blends (standard, long and short all
exercised); overlap within 0.2 bars of the plan every time; the two low bands were never both open;
every start within 85 ms of the planned clock and on the record's beat (phase within 0.03); the beat
lock held through the last 60 % of every overlap (max 0.075 beats, p95 under 0.07); no dead air; every
NEXT and STAY acted; no errors. The breakdown swap has its own smoke (`_dj_breakdown_smoke.py`).

**The breakdown swap smoke (`tools/tests/_dj_breakdown_smoke.py`): OK** on a library pair (I Love You...
I'll Kill You, 8-bar breakdown at 4:25 → Steamed Shrimp Break, 120 → 116 bpm): planned 3:34 ahead; under
the breakdown both records audible with the next one's bass closed; two beats after the drop the next
record's bass is open (1.00) and the old record's gain is 0.00. The smoke needs a breakdown 150 s or more
into the record because the first decode of a run takes tens of seconds (the pair it tried first had its
breakdown at 82 s and the player rightly fell back to the blend).

**Heard (2026-10-24).** "The seams were ok on the three recorded seams, and so far listening live it's
been fine." The blend shape stands.

**The LED system (2026-10-24, "I do want something that works in the LED system as well").** In
Stories_OGL the Director's visuals feed read only the autoDJ or the conductor, so in record mode the
lights got nothing from the decks. Now `RecordPlayer.live_beat()` (bpm, beat phase, bar and phrase
phase counted from where the record opened, the section's bass share and drive) and
`RecordPlayer.outstate_keys()` (dj_energy from each sounding record's drive × its energy curve, gain-
weighted; dj_blend_eta to the next record entering; dj_swap_eta to the bass swap / the drop / the one;
drop stamps for the record's own drops and for every swap, HARD for a breakdown swap and a cut;
dj_moment_hole through a cut's hole; dj_style = blend | breakdown | cut) publish the same vocabulary
DJSystem does; `Director.live_beat()` / `outstate_keys()` delegate to whichever engine plays, and
Stories_OGL reads them for the beat floor and the club director's pre-arm. The record gate checks the
feed (a live beat on 90 %+ of playing blocks, the bpm an audible record's, energy in 0.12..1 and moving,
the swap countdown reaching zero, a drop stamp per swap). Seeds 1-2 with these rows: ALL OK (a live beat on
every playing block, bpm always an audible record's, energy 0.12-1.00, six swap countdowns per seed each
reaching zero, a drop stamp per swap).

**The first live try in Stories_OGL crashed (09:43).** `club_director` does `float(outstate.get('dj_arc_heat', 0.5))`
and the value was None: RECORD mode's Director has no arc engine, its status carried `target: None`, Stories_OGL
copied it into the state, and the autoDJ's own outstate (which used to overwrite it with a number) no longer
applied. The render loop died with the lights mid-frame and the web page stopped answering. Fixed on both
ends: Stories_OGL publishes 0.0 / 0.5 for a missing arc, and the club director skips the arc lean when the
value is None (projects/fan, its own repo). Replayed offline through a swap countdown: the club director
pre-arms and glides on the record player's dj_swap_eta.

**The web page (second live try, 09:49): "the director web controls do nothing and I can select no theme or
list".** The controls did reach the Director (its event log showed the press: "program PEAK: not in RECORD
mode yet"; a STAY sent over the page's own socket path acted), but in RECORD mode fifteen of the sixteen
dials, every moment and every program are refused or no-ops by design, the page showed the refusal only as
the tail of one small line, and it hides the theme / setlist panel whenever the Director runs. Changed:
in RECORD mode the Director panel shows THE PLAN rows (NOW / BLEND / NEXT / UP NEXT with bar numbers), only
the dials that act (layers, pace, bass, tone, level), a SONGS pool select over the setlists, the theme
buttons (read at the next START), and the story (the player's last six moves) so every press visibly
answers; the arc, moments and programs are hidden. Stories_OGL routes `theme`, `setlist` / `setlist_pool`
and `next_id` to the Director too (set_theme, set_pool, queue) and publishes the pool and the setlist names
in the DJ info. The "choose a set" complaint is the same panel (the setlist).

**Then (10:00-10:30): "the buttons do not highlight" (a generic theme re-light loop un-lit the dials every
tick; and Flask's template cache kept serving the old page after the fix - `TEMPLATES_AUTO_RELOAD` is now on),
"way too much text" (plan rows, intent and why removed from record mode; two story lines), "the visualizer is
awful and low info", "if I select a pool I want it selected". The record-mode picture is now two deck strips
drawn from each record's analysis: low / mid / high silhouettes, the sections coloured under it, phrase ticks
every 8 bars from the cue, the drops, the playhead, the plan on both decks (the lead, THE BASS SWAP / DROP,
the tail; on the next record its cue and landmark), a fader and a bass lamp per deck; above it the state chip,
the countdown and the bar-in-phrase dots. `Director.deck_maps()` supplies the strips (sections, 240-point
curves, cue, landmark, drops; ~7 KB per record, sent as a heavy delta key). The pool select applies on pick.

**LAYERS while running (11:40, "make the engine change when I press the button").** Record mode has no handover
into or out of the layered engines, so a LAYERS press used to say "takes effect at the next START". Now
`Director.set_dial("layers", …)` schedules `_relaunch()` on the Director's own thread whenever the engine kind
changes to or from record: the engine fades out over a second, its submix is released, the engine the dial names
is built and started, and the loop carries on. one <-> layered keep their warm handover (the playing song is
carried across). `tools/tests/_dj_engine_switch_smoke.py` (paced to real time: a decode is wall seconds): record
-> two first sound after 10 s, two -> one after 8 s, one -> record after 4 s, no errors, OK.

**The picture for every engine (11:55, "the other systems are back to the text heavy no visual interface").**
`Director._engine_decks()` / `_picture()` / `deck_maps()` now read whichever engine plays: the record player
(role, cue, the blend plan), the conductor (master, lanes held per deck, the seam as the plan, an armed moment
as a mark), the autoDJ (the active deck, the armed blend from blend_at / swap_at / the plan's beats). The page
draws the strips in every Director mode - two or three (the conductor's third deck) - with D B M V lane lamps
in the layered modes, hides the intent and why text everywhere, keeps the arc, the moments and the programs
only for the layered engines, and offers the songs pool and the theme in every mode. The switch smoke checks
the picture on each engine (record, decks, maps, JSON-clean): OK.

**The clicks (12:30, "I feel I've heard a couple of clicks … definitely some clicks").** Hunted by measurement,
not by ear: `tools/tests/_dj_record_clicks.py` renders a paced record session and compares the largest sample jump
within 80 ms of every scheduled event with the music's own jumps; two DSP unit tests feed a real record through
the pieces in isolation. The stretcher's tempo glide and its bypass flip at the end: clean (old, held, reference
all alike). The three-band EQ: a 0.18 full-scale step where the music's jumps were 0.03, exactly at the end of
the bass-swap ramp on the incoming deck - `lib/dj/eq.py` crossfaded its bypass <-> active switch over ONE block
(6-23 ms) although the LR4 band sum is an allpass whose sub-bass sits half a cycle from the raw signal, and it
switched between its one-band shortcuts and the full split with no crossfade at all. Fix: every topology change
(bypass / low shortcut / high shortcut / full) crossfades over 80 ms across blocks, a cascade the new topology
needs is warmed on 60 ms of input history first, and each cascade runs once per block even when two topologies
share it (the first version of the fix ran a shared one twice - 4x jumps, caught by the same unit test).
Measured: the incoming deck's bass-swap end 6.5x -> 1.5x of the music's own jumps, the outgoing deck's highs ->
mids handover 3.1x -> 1.5x. The same EQ serves every engine, so the layered modes gain the fix too. A stretcher
"hold" (no bypass flip) was tried and reverted: measured no effect. A/B on the SAME seeded record session
(`--seed 7`, `--old-eq` = the committed eq.py): the old EQ shows a 5.0x click at a bass swap (0.651 vs the music's
0.130), the fixed EQ none at any event; the record gate on seeds 1-2 with the fixed EQ: ALL OK.

**"Got some really weird looping" (13:15, LAYERS two).** The log names it: the conductor's composed BUILD event -
"the eight bars before the drop, once more", "the bass pulled for the second pass", "released - the drop on the
one" - armed by THE COMPOSER's score and by the DJ's auto moments under MOMENTS = some, twice on the same record,
then an automatic BREAK. MOMENTS now defaults to **rare**, which is zero of everything automatic (auto moments 0,
events per record 0, and `COMPOSE_N["rare"]` is now 0 instead of 1): the layered engines move only on your press.
The planner labels it "none (yours only)".

**START off the render thread (2026-10-25).** The perf session's frame-time profile measured `director_start`
holding the render thread 1157 ms (the library load and the engine start): an LED freeze on every START.
`Stories_OGL._director_start` now builds and starts the Director on a worker thread; `_director_adopt()` (the
render thread's next tick, at the top of `_apply_dj_controls`) takes it over - analyzer to the mix, state ambient
off, dj_active - or restores the room on a failure or a STOP pressed meanwhile. The page shows "THE DIRECTOR is
starting…" with STOP available until it is live.

**The perf session's DJ handover (2026-10-25), all landed:** the autoDJ's start off the render thread the same way
(`_dj_start` worker + `_dj_adopt`, the idle steering applied before start() in the old order, steering changed
during the build applied on adopt, presses during either build queued and replayed); `DJSystem.stop()` no longer
waits on the tempo worker inline (367 ms) and `_verify_in_worker` spawns nothing and falls back to nothing once
STOP has come; `_refresh_tags` (60-95 ms of GIL every 10 s on the brain thread) now skips when its inputs' signature
is unchanged and the character tags are memoized on the track - measured 137 ms -> 0.06 ms unchanged, 9 ms forced,
identical vocabulary; `sys.setswitchinterval` no longer overrides the audio engine's smaller value.

**Attribution of the perf session's control-gate failures on the working tree (2026-10-25):** control gate seeds
1-2 with the tree vs the tree with the committed EQ and the picture block off: the EQ is clear (its only
"failure" was a dead-bar episode inside a MASHUP scene with no sounding part; seed 2 clean both ways), and the
mashup smoke on one seed both ways is ALL OK with 0 dead bars and a lower worst scene-change step with the fixed
EQ (4.3 dB vs 7.9). The THEN-rows error and the PAIR press that "spoke, did not act" happen under the MASHUP,
which keeps no THE SET chain by design; the gate's THEN-rows check now skips the mashup policy. The stem_in-vs-
automation race (a press verified 1.5 bars later overtaken by a scheduled scene or seam move) is at HEAD too.
`/dj-verify` quick over the system.py changes: phase-waiver gate, selection audit, persona sim, setlist gate all
pass.

## 7. Heard 2026-10-26: "the latest music director mix was utterly terrible" (two BADs, 19:46:09 and 19:46:48)

The night log holds only the two BADs and the two "played" rows (the player's moves are not logged), so the
transition was REPLAYED headless and measured (`logs/replay_321_954.wav`, the session's scratchpad
`replay_night.py`): Zephyr (321, 115.9 bpm, 6A, 5:16) into Wild Window (954, 115.9 bpm, 7A, 6:12), pool Prog
House, play_min 3:30. Key fit 0.90, tempo ratio 1.000: neither is the problem. What was:

- **Placement.** The blend started at bar 104 = 3:35 of Zephyr: its final groove (3:16 to the end), its peak,
  with 1:40 left. Section 2's own rule (mix out of the outro or a breakdown, never the peak) is not in the code:
  K is the first phrase boundary after the minimum play that is not a build or a breakdown, full stop.
- **Vocals over vocals, twice.** Wild Window was cued to 0:46 (its landmark 1:18 minus 16 bars), so under the
  lead it played its first sung groove: its vocal stem at 0.85 to 1.06 of its loudest from 0:50 to 1:04 while
  Zephyr ran whole at full. Then Zephyr's loudest sung passage (3:50 to 4:05, vocal 1.0 to 1.5) landed in bars
  9 to 16 of the lead, exactly as Wild Window reached full fader with its highs trading in. The rule "vocals
  never over vocals" exists in section 2 and nowhere in the player (it loads no stem information at all).
- **Two whole records at full for 16 of the 24 bars.** B reaches full fader 8 bars in; A stays at full until the
  swap; A then lingers 8 bars. 50 s of two peak sections together with mild EQ: the simultaneity the rescope
  named as the mud, with whole mixes now.
- **The beat lock breathed.** The telemetry shows B's effective rate swinging 0.987 to 1.012 every couple of
  seconds through the whole overlap on a pair whose tempos are identical: the audio PLL hunting kicks, through
  Rubber Band. Plus or minus 1.2 % of tempo wobble on a whole mix is audible as drift. To be measured on the wav.
- **BAD did nothing.** In record mode a verdict is "noted"; the room had to sit through the whole 24 bars.

What the user heard is consistent with all four at once. Proposed (not built): (1) placement by the records -
K only in A's last third at an outro, a breakdown, or a groove where A's VOCAL ENVELOPE (from the stems on
disk, measured at decode, not played) is quiet for the whole lead; (2) B cued so its vocal is silent until the
swap (its intro proper), the landmark landing on the swap only when that holds; (3) no placement before A runs
out -> the CUT, never a bad blend; (4) B held under A until the swap (fader to ~0.6, "cut, don't boost"), A's
tail 4 bars unless it is in an outro; (5) grid-only sync, or the PLL trim frozen after lock, for pairs within
0.2 % of tempo - measured first; (6) BAD during a blend brings the swap to the next bar and ends A in a beat;
(7) the player's moves written to the night log so a BAD can be read the next day.

**Built (2026-10-26, `lib/dj/recordplay.py`; the user: "at least 2 and 3 were very noticeable").** (1) The vocal
stem's ENVELOPE (half-second RMS scaled to its loud level) is measured at decode for every record, through the same
helper process as the mix; nothing is played from stems. A stem whose loud level is under 6 % of the mix's (-24 dB) is
bleed and reads as silent (`VOX_FLOOR`; without it the breakdown smoke's instrumental pair read as "singing" over its
own intro, and Wild Window - vocal stem at 0.1 % of its mix: an instrumental despite its title - read as singing at
its loudest through tonight's lead; the library's ML vocal axis says 0.0 for both tonight's records, so the envelope
is the only instrument that tells). (2) PLACEMENT BY THE RECORDS (`_find_blend`): the blend starts at the earliest
phrase boundary at or after the minimum play and the record's last stretch (60 % of its length), in a groove or an
outro, with the largest size that fits its room, and NEVER TWO SINGERS: either the next record is cued where its vocal
is silent through the whole overlap (the record may sing over its intro), or the record's vocal is quiet through the
overlap and the next record's until the swap. No such place before the record runs out: the cut on a phrase. The
breakdown swap refuses a next record that sings under the breakdown. (3) The next record sits at 0.6 until the bass
swap, then full with its bass; the record's tail is 4 bars (8 in an outro). (4) BAD during a blend: the swap on the
next bar, the record out in a beat; before one, a NEXT. (5) Every move of the player goes to the night log.

**Tonight's pair replayed with this (Zephyr -> Wild Window, play_min 3:30):** a short blend (the next record was ready
late under three concurrent runs), starting at bar 136 = 4:41 of Zephyr where its vocal is quiet (its sung passage ends
4:05), Wild Window cued at 1:02 so its drop lands on the swap, held at 0.6 under Zephyr for 8 bars, bass swap, Zephyr
out over 4 bars; "both singing" worst 0.00; 24.5 s of overlap instead of 49. The wav: `logs/replay_321_954.wav`.
The beat lock still breathed 0.990-1.003 on this unshifted same-tempo pair; the engine session is on it (deck.py: the
PLL reads the stretcher's rendered position, not the audible one).

## Sources

- [Crossfader: 3 Ways To Mix House Music](https://wearecrossfader.co.uk/blog/3-ways-to-mix-house-music/) (seamless blend, EQ swap, layering, with bar counts)
- [Mixgraph: DJ Transition Techniques](https://www.mixgraph.io/learn/dj-transition-techniques) (long blend, cut, filter sweep, echo out, loop and build, breakdown swap)
- [DJ.Studio: The Anatomy of a Great DJ Mix](https://dj.studio/blog/anatomy-great-dj-mix-structure-energy-flow-transition-logic) (energy phases, transition logic)
- [DJ TechTools forum: phrase matching](https://forum.djtechtools.com/t/question-about-phrase-matching-mixing-to-compliment-a-tracks-musical-structure/68140) (the start-point arithmetic)
- [Digital DJ Pool: DJ EQ mixing](https://digitaldjpool.com/blog/dj-eq-mixing-for-beginners/), [Violet Recording: How to EQ Mix](https://violetrecording.com/how-to-eq-mix-as-a-dj/) (one bassline, cut don't boost)
- [DJ.Studio: phrasing](https://dj.studio/blog/phrasing-dj-mixing), [MixReady: six transition types](https://mixready.art/guides/dj-transition-types), [djtechreviews: transitions](https://djtechreviews.com/guides/tutorials/djing-transitions)
