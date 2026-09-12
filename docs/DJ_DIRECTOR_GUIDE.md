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
- **two / three** - the playing song is handed to the conductor, which plays PARTS of songs: one song is
  the **bed** (its drums and bass), other songs are **voices** over it (their melody or vocals). A voice
  arrives once the bed has settled, is heard a few phrases, and takes the bed at its own drop or hook;
  the old bed song fades out. The **intent line** says what it is doing and waiting for; the **WHY**
  lines say why; the picture shows it.

Switching back to **one** hands the playing song back to the autoDJ at the same position.

## The dials (right column)

Three kinds, hover any heading for the detail and its live effect.

- **What songs.** ARC (the night's energy plan: the theme's own, steady, a build, waves, a wind-down, or
  yours), LENGTH (how long the arc runs), ENERGY and TEMPO (leans on it), VOCALS (instrumentals to
  singers), VARIETY (how far the next song may roam), and the **MOOD** chips (only songs carrying a lit
  tag may play; none lit = everything; "+N more…" opens the rest).
- **How it mixes.** MIXING (auto / long blends / hard cuts / stem morphs), PACE (how long records or
  phrases run), SEAMS (how long the mixes themselves are), LOOPS.
- **How it plays.** BASS, TONE, LEVEL on the mix bus; MOMENTS (how often the DJ makes its own drops,
  breaks and bass-first bed changes: rare / some / lots); FX.

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

## The moments (the four big buttons)

- **NEXT** - move on at the next phrase (layered: the next lane move, waits waived).
- **STAY** - one more phrase of this.
- **DROP** - one song: build and land on the next song's drop; layered: every part to the newest song.
- **BREAK** - layered: strip to one part for four bars, then all back.
- **👍 GOOD / 👎 BAD** - what to do more or less of. The line underneath says what was rated, what it
  learned (seam-style and move weights, and the song's tags for the next picks) and what it did. BAD acts
  at once.

## Steering the songs (left column)

The **SONGS** list is the pool ranked by fit to what is playing, or, once you have queued something, to the
**last song in UP NEXT** (a chain). ✓ fits from here, ✗ would be rejected with the reason. **Double-click**
puts a song UP NEXT. The **filter row** narrows the list: instrumental / vocal, a step calmer / same /
hotter, slower / same / faster, songs with a hook, unheard tonight; the sort box orders it. Hover a row
for the song's shape and when it last played.

## Over the next few songs: PROGRAMS

**NEXT SONGS**: pick 2, 3 or 4 songs, then **BUILD** (energy up song by song, a big moment on the last),
**COOL**, **PEAK**, **VOCALS UP**, **BREATHE** (long plays, nothing added), **BRIGHTEN** or **DARKEN**.
The program steps the dials it names at every song change (layered: every bed change) and hands them
back when it ends; press the lit program again, or **stop**, to end it early. The intent line says
where it stands.

## Big moments on purpose: READY (layered)

Under UP NEXT, **READY** lists every song on a deck with four stem buttons: **bass / drums / melody /
vocal**. Lit = that song holds the lane. Press an unlit one and **that song's stem comes in on the next
bar**, loud for a phrase when **LOUD** is on (or at the next phrase with **PHRASE** on); press a lit one
and it goes back to the bed. Greyed = the guard refuses (hover for the reason: a key clash, not singing
there). **Right-click any song in the SONGS list** to stage it and bring a chosen stem the moment it is
ready. This is "the bassline of that song under this one, now".

With MOMENTS on, the conductor's own bed changes are bass-first: the new song's bassline takes over at its
drop or hook, loud, and its drums follow a phrase later.

## The controller (nanoKONTROL2)

Faders 1-8: ENERGY, TEMPO, PACE, SEAMS, VOCALS, VARIETY, BASS, LEVEL. Knobs 1-8: MIXING, LAYERS, LOOPS,
MOMENTS, FX, TONE, ARC, LENGTH. PLAY / STOP / REC / CYCLE = NEXT / STAY / DROP / BREAK. TRACK ◀ ▶ = BAD /
GOOD. MARKER ◀ ▶ = arc back / ahead 10 %.

## When it does something odd

The **story row** names the move at that bar and the **WHY** panel names the rule behind the last move or
the last wait. A sentence from either is a complete bug report.
