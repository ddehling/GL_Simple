# The Director, rescoped: whole records and hard figures

*2026-10-23. Written after the user's verdict on six weeks of the Director: "boring", "inconsistent",
"acoustically bad" - "a muddled mess of blurred stem mixes and song chunks that only play for short periods
of time, even when explicitly commanded to. I want something cool and just got a chore to mess with."
This is the review of why, the rescope, and the plan. Nothing here is built until the user says yes.
Companion: the code audit of the same date (performance and structure) in the session log; the sound
performance work running on another agent is coordinated in "What the other agent's work means" below.*

## 1. What went wrong, and why - the mechanism behind each complaint

The complaints are not three bugs. They are three consequences of one design decision taken on
2026-09-10 and never revisited: **the unit of play is a stem lane, and the texture of the night is
continuous recombination** (`lib/dj/remix.py` docstring: "songs enter and leave lane by lane, never whole,
so what comes out is a continuous recombination rather than a sequence of tracks"). Everything after that
(THE RECORD PLAYS, the composer, the orders, the mashup) was a patch on top of a lane mixer.

### Acoustically bad: the mud is structural

- **Every slave deck is time-stretched all the time.** Once a deck is synced, the stretcher's bypass is
  pinned off (`submix.py:260`) even at a rate of 1.000, so two of three songs always go through Rubber
  Band, on separated stems that already carry Demucs bleed. Stretch artifacts on bleed, summed across
  songs, is the "blur".
- **Key shift is a two-point linear interpolator** (`deck.py:481-498`), imaging around -30 dB. Rubber
  Band's own pitch scaling is present and unused.
- **A song's stems no longer sum to the song.** Per-lane trims (+-6 dB), the low carve (x0.25 under
  200 Hz), the melody duck under a singer (x0.6), the clash duck (x0.5), the voice texture level (x0.5),
  the loudness matcher (up to +4 dB on the bus), and the peak guard are all applied between a song's
  stems and the room. The record "plays whole" in name; acoustically it is four processed lanes.
- **Lane moves are two-beat crossfades** (`XFADE_BEATS = 2.0`). A cut that is a crossfade smears every
  transient a cut is for.
- **The render thread runs at ~88 % of real time with three stretched decks** (bench, 2026-10-23). The
  DROPOUT rows in the night logs are this.

### Boring: nothing can be big

- A "drop" in this engine is a one-beat hole followed by **x1.25 (+1.9 dB) on bass and drums**
  (`MOMENT_BOOST`), which the peak guard then trims. Real drops are 8 to 16 bars of take-away (no low
  end, a high-pass sweep, a stripped bed) and then the full record on the one. The take-away is the drop;
  the gain is not. The build mechanisms that would supply it (`MOMENT_SWEEP`, `MOMENT_TIGHTEN`) were
  switched off after one bad live try instead of being fixed on the right stems.
- Contrast needs one thing to stop and another to start. The engine's instinct is the opposite: keep
  everything partly sounding (the old harmony "darkening", the vocal "carrying over when it fits", the
  melody at half as a texture). Half-sounding is the definition of boring.

### Inconsistent, and short chunks even when commanded

- **No single owner of "when does the music change."** Competing agents all move the music: the seam
  planner, the composer's events (two per record at MOMENTS some; "a record with two or more events moves
  on at seven tenths of the minimum play"), the DJ's auto moments, the mashup's scene every 16 bars and
  its rotations, THE SET's chain, the runway loop-and-evict, "the room is never empty", "the clock was
  lost", the rehearsal moving a seam, a dial turn "bringing the next seam within a phrase", and BAD
  pushing the music on. Any one of them shortens a record.
- **A command is an input to a planner, not an order.** A planned seam "moves a phrase (twice at most),
  then swaps its order, then falls back to the breakdown and cut" - four outcomes for one plan. A pair
  "is REFUSED then, by name". An UP NEXT "waits for a deck". An IDEA "passes if not taken". The guide
  documents these as features; to the operator they are the system changing its mind.
- **Dice and learning inside execution.** Random rests, auto-break chances, loop odds, moment rates, move
  weights learned from tonight's verdicts, taste leans. The same button does different things on
  different nights. Variety belongs in *choice*; execution must be deterministic.
- **Silent degradation.** 86 `except Exception` sites in the Director and 53 in the conductor, 52 of them
  pass-only; about 70 attributes exist only on some code paths. A rule that fails every bar becomes a
  different behaviour, not an error.
- **Fifteen dials that lean** plus seven moments, programs, pairs, ideas, orders, flags, requests, the set,
  the arc strip and the mashup pads: the chore. Almost none of them does one big obvious thing.

### What was right and stays

The intent of THE RECORD PLAYS (one song whole, one seam per record), the plan shown before it happens,
"act or say why", the rehearsal (render the seam before it plays and judge it), the pairings file (whose
vocal sits over whose groove), the hook and drop measurements, the night's memory, the flags, and the
headless gates. The autoDJ engine is untouched by this and stays as it is.

## 2. The rescope - one sentence

**The unit of play is a whole record at its own tempo, untouched; the unit of excitement is a FIGURE: a
short, deterministic, composed transition or break, rendered offline from the stems, judged, and then
played as audio, exactly as rendered.**

Everything follows from that:

- **The record is the record.** One deck, the original mix (or the four stems at unity, which must sum to
  it within 0.5 dB - a gate), native tempo, no trims, carve, duck or loudness matcher. It plays at least
  PLAY MIN (3:00 proposed) and nothing but your NEXT shortens that. No events on a record unless you
  pressed one.
- **A figure is a script measured in bars with large contrast,** chosen from a small catalogue, never
  improvised live. The first five:
  1. **CUT ON THE DROP** - A's last two bars stripped to drums, one beat of nothing, B whole at its drop.
     Always possible; the fallback for every other figure.
  2. **BREAKDOWN AND RETURN** (within one record) - bass and drums out for 8 bars, a high-pass sweep on
     the remaining stems over the last 4, one beat of nothing, the whole record back on the one.
  3. **ACAPELLA IN** - B's vocal alone over A's groove for 8 or 16 bars (from the pairings), A strips to a
     beat of nothing, B whole at its drop.
  4. **BASS SWAP** - B's bass and drums under A's melody for 8 bars, then the cut to B whole.
  5. **DOUBLE DROP** - B cued so its drop lands on A's drop bar, A out one bar after.
  The classic 32-bar blend is a sixth, allowed only for key-compatible instrumental sections.
- **Rendered offline, played as audio.** The figure is rendered two phrases ahead from both songs' stems
  (the rehearsal already does this), with the best stretch and pitch quality offline (Rubber Band R3 and
  its own pitch scaling), judged for a hole, a lurch and a minimum contrast, and the rendered audio is
  what plays - sample-exact, no live ramps, no PLL, no cue that "has not executed yet". A figure that fails
  its judge becomes CUT ON THE DROP at the same bar, said once. The plan shown is the plan played.
- **Tempo moves only inside a figure.** B is stretched toward A for the figure's bars and then plays
  natively. That is what the measured DJ mixes do (86 % of tracks under 5 %).
- **Commands are orders.** NEXT: the figure starts at the next phrase (if B is not decoded yet: the cut
  the moment it is). STAY: the figure waits one phrase. DROP: the biggest figure available now.
  BREAK: BREAKDOWN AND RETURN on this record now. UP NEXT: the next record, no conditions beyond
  "it has stems". No "moved", no "refused", no "waits for a deck", no "passes".
- **The surface is six buttons and two pickers:** NEXT, STAY, DROP, BREAK, GOOD, BAD; UP NEXT and the
  figure picker (auto or a named figure). ENERGY and the theme pool stay for picks. Everything else
  (thirteen dials, programs, ideas, orders as a separate system, the mashup pads, loops, FX) leaves the
  surface; most of it leaves the code.
- **No dice in execution. No learning in execution.** GOOD and BAD steer the *choice* of the next figure
  and the next record; they never change how a figure sounds tonight.

### Blends are figures too: the quiet half of the catalogue

The user asked (2026-10-23) how this works with three-band blending across lows, mids and highs. It is a
figure: a scripted, fixed-length transition with hard rules, rendered offline like the rest. The classic
32-bar schedule: B in with its lows cut, highs rising over 8 bars, mids to about -6 dB by bar 16, A
untouched; on the one of bar 16 the bass swap, A's lows to zero and B's to full over one beat, never both;
then A's highs down over 8 bars and mids over 16 while B's mids come to full; A out on the phrase.

Two renderers for the same script: the **EQ blend** on the whole mixes (the authentic sound, the filtered
intro creeping in) and the **stem blend** (B's drums high-passed, then its melody, the bass swap on the one
with the bass stem plus a low shelf on the drums stem for the kick, then A's melody and drums out) - no
band overlap at all. The judge for the quiet half is not the contrast floor but a **flatness floor**: lows
from one song at a time outside the swap beat, bar-to-bar level within 1.5 dB, no hole, and key
compatibility (a blend overlaps harmony for eight bars or more; a cut does not care).

The mud of the old conductor was never EQ blending; it was permanent partial simultaneity with no end bar
and the low end carved to a quarter instead of swapped. A blend is time-boxed and the low end always has
one owner.

The catalogue is therefore two halves. **Loud:** CUT ON THE DROP, DOUBLE DROP, BREAKDOWN AND RETURN,
ACAPELLA IN, BASS SWAP. **Quiet:** EQ BLEND 16 / 32, STEM BLEND 32, FILTER IN. The auto-choice varies the
technique (the philosophy's rule: all blends is as monotonous as all cuts); GOOD and BAD steer that choice.
Riding three EQ knobs live by ear is performing, not directing - a separate two-deck instrument if wanted,
not the Director's job.

### What rendering limits, and render-ahead (the user: "doesn't prerendering narrow what can be played?")

Rendering does not narrow **what** can be played: anything the live engine can schedule the renderer can
render, with better stretch, real pitch shift, lookahead limiting and a judge. It narrows **when you can
change your mind**: a figure rendered whole is committed at render start - no swapping figures inside the
window, no steering a running blend, DROP only instant if the render fits in a bar. (The old system had a
commit horizon too - the seam planned when B was staged, rehearsed two phrases ahead - and then kept
moving it; committing was never the problem, wobbling was.) Three answers:

1. **Single-song figures run live.** BREAKDOWN AND RETURN, SLAM, a filter build, AGAIN touch only the
   record's own stems: no stretch, no second song; one deck, one bar of lead, scheduled sample-exact.
2. **Speculative renders.** When B is staged, its likely figures (the cut, the blend, the acapella when a
   pairing exists) render in the background; NEXT with any of them is instant up to the start bar.
3. **The escape hatch is live.** From any running figure, cut to B whole or back to A on the next bar: a
   gain move on a deck, no render.

**The architecture is render-ahead, not render-whole.** A render thread executes the figure's script one
phrase ahead of the playhead into a buffer. Everything beyond the horizon can still change (swap the
blend's tail, extend the breakdown, pull the swap a phrase); the judge hears the next phrase before the
room does; the floors are checked on the script analytically and on each rendered phrase; the quality is
offline quality (batch stretch, no PLL). Steering latency is about one phrase - what a DJ has anyway.
Whole-figure prerender is the degenerate case. **Entry gate for Phase 2:** the renderer's speed with B
pre-decoded - a 16-bar figure (about 30 s of audio, four stems, Rubber Band R3) must render in well under
a phrase; measure before building, and point the sound performance work there.

### The mashup

Three songs' stems blended live is the purest form of the mud, and it is shelved in this form. The same
idea can come back later as **rendered scenes** (a scene is a figure of three songs, rendered and judged
like any other, cut to on the bar); that is phase 4 and only if the first three phases sound right.

## 3. How "cool" is measured before the user hears it

The ear outranks all of it (docs/DJ_VERIFICATION.md), but each complaint gets a proxy the gates can hold:

| complaint | proxy, measured on the rendered audio | floor |
|---|---|---|
| boring | contrast across the landing bar of every figure: level step (dB) and spectral-centroid step; bars of take-away before the land | >= 6 dB and >= 4 bars of take-away for every figure but the blend |
| mud | seconds per 10 minutes with two songs audible on the same lane outside a figure; a record's stems at unity against its mix | 0 s; within 0.5 dB |
| short chunks | audible span of every record | >= PLAY MIN unless NEXT was pressed - a hard assert in the gate |
| inconsistent | same seed and plan -> identical render hash; every command acts at the said bar; plan rows shown == what played | 100 %, no "moved" outcome allowed in the control gate |
| acoustically bad | the existing hole and lurch judge on every figure; every figure saved as wav for the ear (logs/seams) | 0 holes, no lurch over 6 dB; a listening session per phase |

## 4. The plan - a gate and a listening session before each next phase

**Phase 0 - the bar (half a day).** Render ten figures from the current system to wav (the rehearsal
renderer) and put them next to three transitions from real DJ mixes the user likes. The user grades them.
This fixes what "cool" means in this library before anything is built.

**Phase 1 - the record is the record (1 to 2 days).** One deck, native tempo, no processing, PLAY MIN
hard, no events, no auto moments, no loudness matcher, mashup and composer off the live path. Gate: stems
sum to the mix; min-play assert; the control gate with the shortened allowed-reason list (not on a deck,
still decoding, nothing armed). Listening: three records, three cuts. This alone answers "short chunks"
and most of "mud".

**Phase 2 - the figures (3 to 5 days).** The offline figure renderer and judge (grown from
`lib/dj/rehearse.py`), played as audio by a two-deck player (record A, the rendered figure, record B).
CUT ON THE DROP first, then BREAKDOWN AND RETURN, ACAPELLA IN, BASS SWAP, DOUBLE DROP. Gate: the
contrast floors, determinism hash, plan == played; a campaign of a hundred rendered figures across the
library with the metrics table (as `_dj_record_campaign.py` does today). Listening: one session per
figure, the user's words recorded with the render.

**Phase 3 - the surface (1 to 2 days).** The Director tab reduced to the six buttons, two pickers, UP
NEXT, THE PLAN and the picture. The dead code (arrangement and free policies, flags that are off, perform-
only paths, programs, loops, ideas) deleted, not hidden. nanoKONTROL2: PLAY / STOP / REC / CYCLE = NEXT /
STAY / DROP / BREAK. Gate: the control gate; the tab's tick under 5 ms.

**Phase 4 - only if asked:** auto choice of figures from the arc and the pairings (the DJ picks the
figure, you still press NEXT or let PLAY MIN run out); rendered mashup scenes; the long blend.

## 4a. Phase 1 implementation contract (2026-10-24)

Phase 1 is a NEW small engine beside the conductor, not a policy inside it: `lib/dj/recordplay.py`,
class `RecordPlayer`. The Director gets a third mode, `"record"`, selected by `layers = "record"` (the new
default). The conductor and the autoDJ are untouched; they stay selectable until Phase 3.

**RecordPlayer (owned by this session).**
- `RecordPlayer(db, music_root, library, theme="groove", seed=None)`; `.submix = DJSubmix(deck_names=("a","b","c"))`
  (deck c is reserved for Phase 2's figures); `.brain` = a `Brain` over the library for picks.
- Plays the ORIGINAL MIX of a record (decoded with `stemload.decode_mix` in the helper process), at rate 1.0,
  deck EQ flat, no filter, no echo, no stem gains, no sync, no PLL. The only per-record gain is the library's
  loudness trim `track.gain_db` at load (a DJ's trim knob, not processing). Stems are not loaded in Phase 1.
- Attributes: `play_min_s` (default 180.0), `hole_beats` (default 1.0), `pick_hook` (callable returning a
  track or None; the Director supplies UP NEXT), `avoid_ids` (set, never picked), `running`, `record` (deck
  name of the record playing or None), `songs` (`{deck: Rec}` with `track`, `state` in decoding / ready /
  playing / leaving, `cue_t`, `open_clock`), `cut` (None or `{"seq", "b", "clock", "bar", "why", "state"}`,
  decided ONCE; it moves only on NEXT, STAY or when the record runs out), `moves` (list of `{t, kind, text,
  from_id, to_id}`; kinds: open, pick, decoding, ready, cut_plan, cut_move, cut, next, stay, ended, error),
  `wait_why` (one sentence why nothing is planned), `last_error`.
- Methods: `start(first_track=None, threaded=False) -> bool`, `stop(fade_s=1.0)`, `step()` (call every
  0.2 s; the Director's thread), `next() -> (ok, msg)`, `stay() -> (ok, msg)`, `deck_of(track_id)`,
  `eject(track_id) -> msg`, `bar_n` (property: whole bars of the record since it opened), `bar_s()` (seconds
  per bar of the record), `status() -> dict`.
- `status()` returns `{"record": {"title","artist","track_id","time_s","duration_s","bpm","camelot",
  "played_s","min_left_s","deck"} | None, "next": {"title","artist","track_id","state","deck"} | None,
  "cut": {"in_s","in_bars","bar","why","seq","state"} | None, "bar": int, "bar_s": float, "wait_why": str,
  "moves": [(hms, text)...], "error": str | None}`.
- Behaviour. On start the first record opens on deck a from its first downbeat. As soon as a record opens
  the next is picked (`pick_hook()` first, else the brain, never `avoid_ids`) and decoded onto a free deck;
  it sits loaded, cued to its LANDMARK (the groove after its first build within 90 s of the body, else the
  body's first groove), gain 0, not started. When the record has played `play_min_s` and the next is ready,
  the cut is planned once: the first phrase boundary of the record (8 bars from its opening downbeat) at
  least two bars ahead. The cut is CUT ON THE DROP without the strip: at T - hole the record's gain goes to
  0 (10 ms ramp); at T the next deck starts at gain 1 on its landmark downbeat; the old deck stops two
  seconds later and is unloaded. The events are posted when planned, tagged with a `txn`, and cancelled and
  re-posted when the cut moves. `next()`: the cut comes at the next bar + 2 (phrase alignment waived); the
  next not ready → `(False, "still decoding: the cut comes the moment it is ready")` and it does. `stay()`:
  the cut moves one phrase later, said; none planned → `(False, "nothing planned yet: the record plays on")`.
  A record whose deck finishes with no next ready: silence, a move `ended` with the reason. Every refusal
  is one of: "still decoding", "nothing planned yet", "is the record playing", "not on a deck", "no such
  song", "press START".
- No dice, no learning, no envelopes, no trims. Deterministic given the same picks.

**Director integration (a subagent).** `lib/dj/director.py`: `DIALS["layers"] = ("record","one","two",
"three","mashup")`, `DEFAULTS["layers"] = "record"`; `self.player`, `self.mode == "record"`; `start()` builds
the player when `layers == "record"` and attaches `"dj_record"`; `prepare()` returns None in record mode;
`set_dial("layers", …)` while running to or from "record" says "takes effect at the next START" and
`_handover` returns early in record mode; `step()` runs `player.step()` and the Director's own steps that
apply (dropout watch via `player.submix`; night memory via `_note_played` when a record opens; `rank()` uses
the record as its reference; the picture `_record()` draws the record on all four lanes and the cut as the
ghost; `status()` fills now / next / intent / plan_rows (NOW, CUT, NEXT, UP NEXT rows) from
`player.status()`; `next()` and `hold()` route to the player; `arm_moment` and the stem moves answer
"not in RECORD mode yet (Phase 2)"; `rate()` records the verdict and the song's tags only). Skipped in record
mode: loudness match, auto moments, loops, the set chain, slip watch, warm handovers, pending stems and
moments. `pick_hook` = the UP NEXT head (popped when it is on a deck), else None. The tab
(`tools/dj/planner/director.py`): `DIAL_LABELS["layers"]` gains `"record"` with a tooltip; the moments row
reads the mode. Existing gates keep passing (they set `layers` explicitly).

**The gate (a subagent): `tools/tests/_dj_record_gate.py`.** Headless, the mixer pulled by hand as the control
gate does, `play_min_s` 60 s for speed, three or more records, NEXT and STAY pressed at random states. It
fails on: a record's audible span under `play_min_s` unless NEXT was pressed on it; two decks over gain 0.1
at once outside a cut's own two beats; a hole at a cut shorter than 0.5 or longer than 1.5 beats; dead air
over 1.5 s anywhere else; a NEXT or STAY that neither acted within 3 bars nor said one of the allowed
reasons; `last_error` set; the next deck's first audible downbeat more than 50 ms from the planned clock.
Plus `tools/tests/_dj_stems_unity.py`: for N library records with stems, the four stems summed against the
mix over the body, in dB, failing over 0.5 dB (the Phase 2 precondition).

**Phase 1 status (2026-10-24, early).** `lib/dj/recordplay.py` written and smoked headless standalone (play_min
40 s, 3x real time): seven records in a row, every cut planned once at the record's phrase boundary 2-4 bars
after its minimum, fired on the one, the next record picked and decoded within 3 s of each open, no errors.
`tools/tests/_dj_stems_unity.py` on 12 random records: 11 within 0.5 dB (worst of those 0.45), one at -1.81 dB
("I Wish It Would Rain Down", aligned residual -6.6 dB - a bad separation, to be excluded from figures). **Finding
for Phase 2:** the stems lag the decoded mix by exactly 1024 samples (23 ms) on every record measured; a figure
that switches between the mix and the stems must shift the stems by that, or play stems only.

**Phase 1 built and gated (2026-10-24).** The Director has the third mode (`layers = record`, now the default;
the tab has the button); `tools/tests/_dj_record_gate.py` seeds 1-2, nine rendered minutes each, play_min 60 s:
**ALL OK** - 9 + 10 records heard, every un-NEXTed record over its minimum, 0.0 s of two decks at once, 0 dead-air
gaps, every hole 0.92-0.97 beats, every new record's first downbeat within 6 ms of the planned clock, 11 presses
all acted or said an allowed reason, no errors. The layered control gate (seed 16) still ALL OK. Uncommitted:
`lib/dj/recordplay.py`, `lib/dj/director.py`, `tools/dj/planner/director.py`, the two gates, the Phase 0 script, this
doc. **Next: the Phase 1 listening session** - restart the planner, START with LAYERS record, three records, three
cuts; then Phase 2's entry gate (the renderer's speed).

Known limits of Phase 1 (by design or deferred): the theme cannot change while a record plays; a song queued after
the next record is already decoded waits one more record; the planned-cut ghost sits a bar from the real clip (the
Director's wall-clock bar against the record's own bars, as the seam ghost did); the opener is the most tempo-central
unplayed track (no dice), so the night's memory is what rotates it.

## 5. Coordination with the sound performance pass (another session, 2026-10-23)

What that session has in the working tree (read, not changed, by this plan): `lib/dj/sosfast.py` (scipy's
sosfilt recurrence under Numba, GIL released, bit-identical), `eq.py` and `fx.py` on it; `submix.py`
renders a pass of up to 1024 frames while no automation event falls inside it (events still land on the
256-frame grid; the PLL's time constants scaled by the pass length); `deck.py` drops back to 256-frame
steps while any ramp, brake or filter sweep is moving. Golden renders (`golden_rb.npz`, `golden_vari.npz`)
check the audio is unchanged. This is item 1 of the code audit done the right way, and it is complementary
to this plan: the autoDJ, the single-song live figures and the render-ahead renderer all run on this
submix, so every millisecond it saves is theirs too.

Division of labour from here:

- **That session owns the engine floor:** the submix, decks, EQ, stretch engines, the decode helper. Its next
  two wins that serve this plan: (a) an **offline-quality mode on the deck** (Rubber Band without the
  realtime option, its own pitch scaling instead of the linear-interp key shift) selectable per submix
  instance - the renderer uses it, the live autoDJ keeps realtime mode; (b) the **zero-copy decode handoff**
  (shared memory or memory-mapped float16 stems) so a song's arrival no longer holds the GIL. Not worth its
  time under this plan: making live three-deck Rubber Band cheaper for its own sake.
- **This plan owns the policy layer:** `lib/dj/director.py`, `remix.py`, `record.py`, `mashup.py`, the
  renderer and judge, the Director tab. It does not touch the engine files above until that session commits.
- **Sequencing:** their pass commits first (its own commit, its own gates: the sync gates, the golden
  renders, the remix and control gates). Then the bench and the Director profile are re-run here as the
  baseline for Phase 1. Phase 2's entry gate (the renderer's speed) is measured on their committed engine.
- **From that session, 2026-10-23 21:30 (its pass verified, awaiting the user's word to commit):** sosfast
  bit-identical on 12/12 golden arrays; the 1024-frame passes are byte-identical to the old code at 256 and
  audibly equal or better at 1024 (kick alignment measured from each deck's output in 16 variants), while
  the grid-lock TELEMETRY reads worse for pitched slaves under Rubber Band (14 vs 6 ms median, a reporting
  alias against the stretcher fifo, not the audio) - do not read `max_err_beats` as a regression. Census of
  336 paired seams: flam 29→29, lurch 37→35, dead air 17→17, double bass 7→6, clip 0→0. Speed in one process:
  a blend 10.0 → 5.3 ms per 46 ms block, a mashup 14.6-18.2 → 10.0-10.5 ms. **Two cautions for Phase 2's
  renderer:** `rb_stretch.py` sets the time ratio once per 1024-frame Rubber Band block, and the deck's
  256-frame step fallback (`Deck._moving_in_steps`) is what keeps stem, rate, EQ and filter ramps sounding
  as tuned - the offline-quality mode must keep that stepping or replace it deliberately and re-listen.
- **Committed 2026-10-24 as 98fad9e** (the LED pass c735d9d before it touches no DJ file). Bench re-run here on
  that commit, four decks, 2048-frame blocks (46.4 ms): bypass 6.0 → 2.6 ms; stems + EQ carve 18.4 → 4.5 ms;
  three decks stretched and synced with stems and EQ **40.8 → 24.3 ms (88 → 52 % of real time)**; with a
  -2 st key shift on three decks 25.5 ms; one stretched deck 8.9 → 6.2 ms. This is the Phase 1 baseline.
- **Shared instruments, unchanged by either side:** `tools/tests/_dj_control_gate.py`, `_dj_mashup_smoke.py`,
  `_dj_remix_gate`, the seam renderer in `lib/dj/rehearse.py` and the measurement code (`measure.py`,
  `audition.py`) stay on scipy as the reference.

## 6. Decisions (2026-10-23: "yes to all")

1. Perfect execution with a one-phrase steering horizon over zero-latency live tweaking: **yes.**
2. Figure order: CUT ON THE DROP, BREAKDOWN AND RETURN, ACAPELLA IN, EQ BLEND 32, BASS SWAP, DOUBLE DROP,
   STEM BLEND 32: **yes.**
3. Mashup shelved now, rendered scenes later: **yes.**
4. Minimum play 3:00, hard, NEXT the only exception: **yes.**
5. Phase 0 first: **yes.** `tools/tests/_dj_phase0_baseline.py` captures the current Director's bus for
   28 rendered minutes (MIXING rotated per record, SLAM / BREAK / SING pressed now and then) and cuts every
   transition and moment into a clip in `logs/phase0_baseline/` with `index.md`; the rehearsal's offline
   renders of the same seams go to `logs/phase0_baseline/rehearsal/`. The user puts three real-mix
   transitions in the same folder and grades the clips against them.

The questions as they were asked:

1. Offline-rendered figures mean a figure is fixed two phrases before it plays and cannot be tweaked
   mid-figure. Perfect execution over live tweakability - yes?
2. Which three figures matter most? Proposed order: CUT ON THE DROP, BREAKDOWN AND RETURN, ACAPELLA IN.
3. The mashup: shelved now, back later as rendered scenes - yes?
4. PLAY MIN 3:00, hard, NEXT the only exception - yes?
5. Phase 0 first (grade ten renders against real mixes), or straight to Phase 1?
