# The Director: how to run a night

*The planner's Director tab (`tools/dj/planner/director.py`, engine in `lib/dj/director.py`). Written
2026-09-12 after "I don't really understand how to use this?". The philosophy behind it is in
`DJ_PHILOSOPHY_OF_PLAY.md`; the build history in `STEM_DJ_PLAN.md`.*

## Start

Pick a **theme** (what kind of songs, the shape of the night) and, if you want, a **playlist** as the pool.
Press **START**. The autoDJ plays one record at a time exactly as it always has; every dial only leans it.
Press **STOP** to end the night.

## The one big switch: LAYERS

- **one** - the autoDJ. One record, seams between records.
- **two / three** - the playing song is handed to the conductor, which plays it as **the record**: whole,
  all four stems, its own balance, for at least the PACE dial's minimum (2:30 / 3:30 / 5:00) and never past
  its natural out point. When the next record is staged, ONE **seam** is planned and shown: the bar of the
  record (its mix-out point: a section boundary where its drums still carry), the bar of the next (cued so
  its drop lands on the bar the seam completes), the seam's length (MIXING: cut = a 4-bar breakdown then a
  cut; morph = 16 bars; blend = 32; auto = the dice) and its order. The seam is the stem morph over
  phrases: **drums first** by default (the new drums under the old harmony, the old harmony darkening over
  the last four bars, then the bassline and melody swap on the one), the **bass swap** for variety (the
  new bassline and melody under the old drums, the drums follow), or an **acapella lead-in** (the new vocal
  two phrases before its drums, when the record is instrumental there). **Harmony travels together**: a
  song's bass and melody are one thing; a melody never sounds over another song's bassline. The old
  record's vocal carries over only when it fits the new harmony. Between seams nothing moves but the
  vocal lane following its own singing. (Read docs/DJ_THE_RECORD_PLAYS.md for the whole design.)
  **THE PLAN** (under the maps) is what will happen next, in order, each row with the time from now: NOW
  (the record, how far in, its minimum play, its next section), a MOMENT armed or running, the EVENT
  pending and whether its song is staged, the SEAM (which record comes in, how, why there, whether its
  drop lands on the swap, the rehearsal's verdict), the RECORD after it with its minimum play, and UP
  NEXT. When there is nothing to show the row says why ("no next record staged yet: the DJ stages its
  pick while this record plays toward its minimum"). A row changes only when the plan moves, and then
  carries the reason for twenty seconds ("MOVED: rehearsed a hole at bar 16"). The picture shows the
  same seam as a gold bracket with its steps.
  **THE SET** continues the plan past your queue (docs/DJ_THE_SET.md): up to five `THEN` rows, the records
  the DJ means to play after UP NEXT, each with its projected time, tempo step, key fit, energy word and how
  it would arrive ("Antrea · 124 (+2 %) · 7B 1.00 · same energy · drums first"). The DJ's own next pick IS
  the first of them. A row changes only when it must (the song was played, its predecessor changed, the
  wall, you said "not tonight") and the story row says why; a dial turn keeps the hops that still fit.
  Right-click a THEN row: put it UP NEXT, not tonight, show its parts. A `CLOCK` row says where the tempo
  goes ("122 now → 126 by 12:40 → 128 by 25:00"): the clock follows each record to its own tempo, half a
  percent a bar, so the set can travel; the TEMPO dial leans each step (faster = about +4 % a record).
  `REQUEST` rows at the top are songs being made playable tonight (see "The floor" below).

Switching back to **one** hands the playing song back to the autoDJ at the same position.

## The dials (right column)

Three kinds, hover any heading for the detail and its live effect.

- **What songs.** ARC (the night's energy plan: the theme's own, steady, a build, waves, a wind-down, or
  yours), LENGTH (how long the arc runs), ENERGY and TEMPO (leans on it), VOCALS (instrumentals to
  singers), VARIETY (how far the next song may roam), and the **MOOD** chips (only songs carrying a lit
  tag may play; none lit = everything; "+N more…" opens the rest).
- **How it mixes.** MIXING (auto / long blends / hard cuts / stem morphs; layered: the seam's kind - the DJ
  changes records through seams; only under cut may it exit a record through a takeover), PACE (how long records run:
  layered, the record's minimum play), SEAMS (how long the mixes themselves are: layered, the seam's length
  scaled by a half or double AND how long each step crosses - quick = a switch, normal = two bars a step, long
  = a phrase for the drums and four bars for the harmony, the bass handed over under a low shelf), LOOPS.
  The DJ's own seams and events land at unity; loud is for what you press.
- **How it plays.** BASS, TONE, LEVEL on the mix bus; MOMENTS (how often the DJ arms its own moments, the
  same five as yours: rare / some / lots). FX is off the surface (the moments carry their own shapes); the
  controller's knob 5 still reaches it.

A dial acts within a phrase; it never re-plans the next song unless the dial changes what songs fit.

## The arc strip (under the top row)

The plan as a curve, the songs that played as dots at their energy, the playhead, the peak ahead.
**Click** = "we are here" (both engines jump to that point of the plan). **Drag up or down** = bend the
plan there (ARC becomes "yours").

## The picture

Four lanes are the four stems. Clips are what is actually sounding on each lane, with the song's
**sections** along the top (intro / groove / build / breakdown / outro, the drop as a white tick, the
measured **hook** in gold, singing in green on the vocals lane). The **meter** under each lane stacks every
song sounding on that stem in its own colour - two colours = two songs on one stem. The **story row**
above the lanes says what happened where and what is planned (dashed). **Wheel** zooms, **Ctrl+wheel**
pans, **− / + / now / follow** on the right of the top row. Under NOW and NEXT a **song map** shows the
whole record's shape.

## The moments (the row of five)

A moment is a composed shape over 1 to 16 bars, the way the literature describes the break routine: take
away, build, a hole with the downbeat delayed, then bass drum and bass back **on the one** of a phrase,
louder. Press a moment to **arm** it: the system picks the landing (the target song's drop or hook within
four phrases, else its next phrase far enough away), the picture draws the choreography in **gold** across
the lanes, and the button counts down in bars. Press the armed one again = **go now** (it starts on the next
bar). Right-click = cancel (a running one is undone on the bar). Nothing else moves while a moment is armed
or running; afterwards the arrangement waits a phrase.

- **SLAM** (2 bars) - bass and drums out for the last two bars, a beat of nothing, everything back louder.
  One song at a time this button reads **DROP**: the autoDJ's build into the next song's drop.
- **BREAK** (16 bars) - drums and bass out for eight bars while the melody or vocal stays; the drums come
  back thin and their lows return over eight; half a bar of nothing; everything back louder on the one.
  (The loop-tightening and high-pass-sweep builds exist behind two switches in the conductor and are off:
  the first live BREAK with them was "painful in its execution".)
- **SING** (at the hook) - bass and drums out for two bars, the vocal alone for the room; back louder on the
  third bar. Needs a song singing with a hook ahead.
- **TAKEOVER** - the other song's bassline and melody take over on the one, loud, at its drop or hook; its
  drums follow at the phrase. Needs a second song (a voice, or one staged: put one UP NEXT).
- **BRIDGE** - the bed fades out under this song's vocal, the other song's drums come in under it, a beat of
  nothing, its bass, melody and drums slam on the one; the vocal carries over. Needs the bed singing and a
  second song.

A BREAK is refused where it would take nothing away (the bed thin or breaking down itself) or where what
stays would be too quiet in the room; the line under the row says which.

The **MOMENTS** dial (rare / some / lots) makes the DJ arm these same five by itself, spaced as before; you
see the same countdown and can cancel or replace it. Refusals say why on the line under the row.

**The events (layered).** At MOMENTS some, every record gets ONE event; at lots, two; at rare, none. They
come from the **pairings** computed once from the stems for every record in the pool (`python
tools/dj/dj_pairings.py --all`, and the planner's analysis stage keeps them current): the other songs whose
vocal sits over this record's instrumental groove, and the songs whose bassline swaps under its drums. The
DJ stages the song, and the event arms: an **ACAPELLA** (another record's vocal over the groove for two to
four phrases, then gone) or a **TAKEOVER** with that song at its drop, after which it is the record. The
intent line says "event in 40 bars: the vocal of X over this groove (fit 0.81)"; the picture draws it in
gold; you can cancel it like any moment. **VOCALS** chooses the kind (none: no vocal events; some: a vocal
first when one fits, else the bass swap; lots: vocals only, two of them at MOMENTS lots), **ENERGY** leans
the choice of song (cool: a calmer one, amp: a hotter one), and a PROGRAM steers the count through MOMENTS
as before. See docs/DJ_PAIRINGS.md.

**The rehearsal (layered).** Two phrases before a seam, the DJ renders it in the headphones: the seam
from the two decks' stems, exactly as scheduled, judged for a hole (a half second under 0.15 of the
median) and a lurch (a step over 6 dB, or over the records' own steps by 1.5 dB). A seam that fails moves a
phrase (twice at most), then swaps its order (drums first ↔ bass swap), then falls back to the breakdown
and cut. The log says "rehearsed: clean (…)" or "the seam moved: … - rehearsed: a 10 dB step at bar 2".

**Your instructions are orders (2026-09-14).** A song you ask for is refused only when it is not on a deck.
Right-click a part in: a staged song is cued to where that part sounds, whatever its cursor says; a key
clash is said in the story row, not obeyed. NEXT completes a running seam on the next bar, starts a planned
one at the next phrase, and with nothing staged it stages the DJ's pick and seams as soon as it is ready.
UP NEXT is the next record, ahead of the DJ's own pick, and takes the DJ's silent staged song's deck when
none is free. TAKEOVER or BRIDGE with a song of yours takes the room from a seam in flight.

- **NEXT** - move on at the next phrase (layered: the planned seam starts at the next phrase; with no next
  record staged yet, nothing).
- **STAY** - one more phrase of this (layered: a seam due in that phrase moves a phrase later).
- **👍 GOOD / 👎 BAD** - what to do more or less of. The line underneath says what was rated, what it
  learned (seam-style and move weights, and the song's tags for the next picks) and what it did. BAD acts
  at once. **BAD on a moment asks which of five things hurt** - the hole was too long / it sounded thin /
  it went on too long / the return was off the beat / it landed in the wrong place - and moves ONE number
  of that moment (the hole halves, the take-away shortens, the build keeps more lows, or it lands only on a
  measured drop or hook from then on). The numbers are kept per library and show in the button's tooltip,
  with how often that moment was armed or refused tonight and why.

Between moments the arrangement is calm: at most one lane move per two phrases and none within a phrase
after a moment. The intent line counts the lane moves of the last ten minutes.

- **GOOD FLAG / BAD FLAG** - "this worked", or "this is not working", said for LATER. GOOD FLAG: that
  moment landed, a great transition, these songs sit well together, a good song choice, the energy building
  right, the low end right, the vocals sat perfectly, this stretch was right as it was, the level right, the
  plan read right, or a note. BAD FLAG: dead air, too busy, nothing happening, a clash, a bad seam, a wrong
  song, the plan changed, off the beat, vocals wrong, the level, the low end, the energy, a song ran out, a
  moment did not land, too long on one thing, UP NEXT ignored, a control did the wrong thing, the screen,
  or a note. Either way the whole room is written down with it: every song on a deck with its position, lanes and gains, the plan, the armed moment, the dials, the
  bus level, the last moves. It is marked in the story row. **Nothing is learned and nothing moves.** BAD is
  the verdict that changes the music and the weights; FLAG is the message to the fix, and a good flag is a
  case to protect when a rule changes. Review them with `python tools/dj/dj_flags.py` (`--summary`,
  `--good`, `--bad`, `--kind dead`, `--last 5`). SET on the nanoKONTROL2 raises a plain problem flag.

**The plan is decided once.** Layered, THE PLAN reads "NOW Believe · 2:10 played · min play in 1:20 ·
its drop in 0:40" and "1:40 SEAM Antrea comes in · a 16-bar morph, drums first · at Believe's outro · its
drop lands on the swap · rehearsed: clean". That bar does not drift:
it moves only when you press NEXT (the seam at the next phrase), STAY (a phrase later), when UP NEXT
replaces the next record, or when a moment holds the room at its bar; then the story row says "the seam
moved" and why. A BAD is about the last move only while that move is fresh (two phrases); later it means
"this stretch": it pushes the music on and blames no move, and the move it causes cannot be rated until
it has been heard.

## Steering the songs (left column)

The **SONGS** list is the pool ranked by fit to what is playing, or, once you have queued something, to the
**last song in UP NEXT** (a chain). ✓ fits from here, ✗ would be rejected with the reason. **Double-click**
puts a song UP NEXT. The **filter row** narrows the list: instrumental / vocal, a step calmer / same /
hotter, slower / same / faster, songs with a hook, unheard tonight; the sort box orders it. Hover a row
for the song's shape and when it last played. A fitting row carries how it fits in the line: the key fit
and the tempo step; ★ marks a pairing with the record playing. **The colour is the mix**, in the picture's
lane colours: a song coloured like the vocal lane would come as a vocal over the record; like the bass lane,
its bassline under it or a bass swap as the next record; like the drums lane, drums first as the next
record. Plain = the cut, or nothing known. The header carries the legend; the tooltip says it in words.

**MIXES WITH** (under UP NEXT) follows the highlighted song and says what goes with it, from the stems:
after it as the next record (key, tempo, energy, and how it would arrive - drums first, the bass swap, or
the cut), whose vocal sits over its instrumental grooves, whose bassline swaps under its drums, and where
its own vocal and bassline sit. Double-click a row to find that song in SONGS; right-click for the same
commands. Its rows wear the same lane colours as SONGS: orange = drums first, blue = a bass swap or a
bassline under, rose = a vocal over; plain = the cut. Like PAIRS, it lists only what would play: no song
played tonight, on the decks or already queued as the next record, no vocal over a groove under ten bars
or in the song's last minute.

**Nothing is offered that would be refused.** The same rule runs through every list and menu: a song
without stems is not offered UP NEXT (only "make it playable"); TAKEOVER and BRIDGE refuse at once for what
can be known before staging (the record itself, no stems, the wall); a queued song whose deck is not free
WAITS in UP NEXT and says so, it is not dropped; the DJ's IDEA is placed only where the record's vocal
stem is quiet by the same measure the arm uses; and THE SET's next hop is asked the pick's own questions
when the pick comes - a hop that fails them leaves the plan, said in the story row, before the brain picks.

**The right-click menu is trimmed to the pair.** Its first line says how the song fits the record playing
(keys and fit, tempo, energy). Then only what makes sense: its vocal in (when it sings), its bassline in,
TAKEOVER with it, BRIDGE with it only while the record sings; drums and melody wait under "more parts…"
(the drum swap is off by design, a melody alone is a texture). What fits is ★ and bold with the fit. A
song outside the tempo wall gets the reason, its parts window, and - when a chain of records reaches it -
"put it UP NEXT via 2 records: A (125), B (131)": the intermediates go into UP NEXT ahead of it, marked
"(via)", and the clock travels there record by record. In SONGS such a song reads "via 2 records" instead
of grey. When nothing reaches it the menu says so.

## PAIRS: the mashup finder (the fourth tab of the right column)

The best pairs across the whole library, from the pairing file, ranked: "the vocal of X over Y from 2:10
(fit 0.91)" in the vocal lane's colour, "X's bassline under Y's drums, at its drop 1:30 (fit 0.88)" in the
bass lane's. The list holds only pairs that would PLAY if taken now: the record inside the tempo wall of
the clock playing, a groove of at least ten bars, a free deck for what the pair needs, the record playing
not yet past its groove, no song that is already the record. Everything PLAY THIS PAIR would refuse is
left out. **with the highlighted song** keeps the pairs that involve the song highlighted in SONGS or UP NEXT.
**Double-click = HEAR**: eight bars rendered offline from both records' stems, the vocal or bassline
stretched to the record's tempo and shifted by the pair's semitones, played on the CUE OUTPUT. **OUTPUTS**
(the button in the top row) chooses both devices: the MAIN output, where the floor plays (system default
unless you pick one; it takes effect at the next START), and the CUE output for the preview (at once). Kept
per machine in logs/dj_outputs.json; `dj.cue_device` in config.yaml still works as a fallback. Without a cue
output the tab says so; a preview never plays on the floor while the Director runs. **Right-click → PLAY THIS PAIR**:
an ORDER (docs/DJ_ORDERS.md). The pair's record goes UP NEXT (unless it is the record now) and its other
song is staged at once. The moment the record's stems are on a deck the pair is PLANNED once, from those
stems: where the record enters (drums first five phrases before the groove; or whole, a cut, at the latest
place before the groove where its drums carry; or on the groove itself with the vocal four bars in), and how
many bars the vocal gets. A pair that does not fit is REFUSED then, by name, and its record leaves UP NEXT.
From then on the seam into the record, the event and the arm read that score; nothing is searched again. THE
PLAN carries a PAIR row from the moment you take it (planning → planned with its times → the EVENT row when it
is armed → done, refused or failed with the reason, kept a minute), and the picture a dashed vocals mark at
its bar with the other song's dashed clip once it is on a deck. Where the pair census has run
(`tools/tests/_dj_pair_census.py`), the list already says the entry the planner decided and hides what it
refused. A song played tonight shows grey.

## IDEA rows and the moments row

The DJ's events are proposals: an **IDEA** row in THE PLAN ("the vocal of X over this groove for two phrases
(fit 0.9)"), right-click **do it** or **skip**; not taken two phrases before its bar, it passes, said. MOMENTS
lots = the DJ takes its own ideas; some = it proposes; rare = it has none. Two more buttons in the moments row
edit the record itself: **BUILD** loops the eight bars before the record's next drop once more with the bass
pulled on the second pass and releases into the drop on the one (press again = release now; it needs a drop
eight bars away at least within four phrases); **AGAIN** brings the record's current section back once, from
its start on the phrase. A song outside the tempo wall may still bring its PARTS in at half or double time
(the menu says "half-time part"); it is never the record.

## The floor: a request during the night

A song in the library **without stems** cannot be layered, and its menu offers **"make it playable"**: its
stems render on the GPU (the fast model, about a minute), its pairing lists are built, and the Director
learns the song without a restart. A file **dropped into the music folder** while the night runs is found
within twenty seconds and goes the same way (analysis first). Each request is a `REQUEST` row at the top
of THE PLAN with its stage ("separating (40 %)", "pairings running"); when it is done the row says
"is ready · put it UP NEXT?" for a minute (right-click it), and the song is in SONGS as a ✓. One job at a
time: the GPU is shared with the player. A request is not a promise of a fit: it still has to pass the
wall and the key like any other song.

## Seeing a song's parts: PARTS

**PARTS** (in the filter row) opens the **Song parts** window. It follows whatever you highlight in SONGS or
UP NEXT: the song's four stems as spectrogram rows over its whole length, the sections along the top, the
drops as dashed lines, the hook in gold, singing in green, its instrumental grooves in blue under the MELODY
row (twelve bars and more with no singing: where another song's vocal can sit - the pairing finder's), and
its live position when it is on a deck. Each row ends in a line of words: how much of the body that part
sounds in, its longest silence, its loudest half minute, and the pairings (which vocals sit over it, which
basslines swap under it, where its own parts sit). That is how you know a bassline is worth bringing in,
and where. Right-click a row for the same
commands as the lists: that part in now, TAKEOVER or BRIDGE with the song, UP NEXT. A song without rendered
stems says so.

## Over the next few songs: PROGRAM

The **PROGRAM** box: pick 2, 3 or 4 songs, then **BUILD** (energy up song by song, a big moment on the
last), **COOL**, **PEAK**, **VOCALS UP**, **BREATHE** (long plays, nothing added), **BRIGHTEN** or
**DARKEN**. The program steps the dials it names at every song change (layered: every bed change) and hands
them back when it ends; choose **none**, or **stop**, to end it early. The intent line says where it stands.

## How songs are balanced (layered)

A song keeps its own balance: the record is scaled as a whole so its drums sit at the library's drum level
- it sounds like the record - and so is each side of a seam (the song holding the drums and the song
holding the bass and melody each keep their own balance while they overlap). Only a stem set against
another song's stems on its own is trimmed toward the library's level for that lane: a vocal over the
record, a part you brought in by hand. On top of that come the roles: a melody on a song holding neither
drums nor bass plays at half as a texture; a melody under another song's singing steps back further; a
voice whose chords clash with the harmony here is held down until they agree; and bass and drums come back
a quarter louder for a phrase after a moment. Songs holding neither drums nor bass have their lows carved so two
songs never share the low end. The layered bus is then trimmed up to plus four decibels toward the loudness
the autoDJ produced, and LEVEL, BASS and TONE sit on the bus last.

## One stem of one song (layered)

**Right-click any song in the SONGS list**: its bassline, drums, melody or vocal comes in under what plays
on the next bar, loud for a phrase (a song not on a deck is staged first and comes in the moment it is
ready). The guard still refuses a key clash. The same menu offers **TAKEOVER with this** and **BRIDGE with
this**: the song is staged if it has to be and the moment arms with it the moment it is ready. With MOMENTS on, the conductor's own bed changes are bass-first:
the new song's bassline and melody take over at its drop or hook, loud, and its drums follow a phrase later.

## The controller (nanoKONTROL2)

Faders 1-8: ENERGY, TEMPO, PACE, SEAMS, VOCALS, VARIETY, BASS, LEVEL. Knobs 1-8: MIXING, LAYERS, LOOPS,
MOMENTS, FX, TONE, ARC, LENGTH. PLAY / STOP = NEXT / STAY; REC = SLAM (again = go now); CYCLE = BREAK
(again = go now). TRACK ◀ ▶ = BAD / GOOD. MARKER ◀ ▶ = arc back / ahead 10 %.

## When it does something odd

The **story row** names the move at that bar and the **WHY** panel names the rule behind the last move or
the last wait. A sentence from either is a complete bug report. Silence on the bus for more than a bar is a
**DROPOUT**: the state line shows it for three minutes with what was happening (a moment, a handover), and
the night's log (`logs/dj_YYYYMMDD.jsonl`, every row tagged with the session id) holds the same with the
last three moves.
