# More than a playlist mixer

*2026-09-20. "ok, the system is starting to be okish. what can we do to make it more than a playlist mixer?"
Four things, agreed on paper ("do all 4 things"). A playlist mixer picks the next song and gets from A to B;
these use what only this system has - the stems and the offline pairing analysis - to make music out of two
records, and they all obey the evening's rules: the machine proposes, the user decides; loud and sudden are
for what the user presses; nothing happens that was not shown.*

## 1. The mashup finder with a headphone preview

The pairing file knows, per record, which vocal sits over which groove and which bassline swaps under which
drums, at bar precision. Today it feeds the DJ's events and the ★ rows. It becomes the main proposition:

- **PAIRS**, a fourth tab of the right column: the best pairs across the whole library, ranked by their
  score, holding only what would play if taken now (inside the wall, a groove of ten bars, a deck for it,
  the record playing not past its groove - every refusal of PLAY THIS PAIR applied to the list first) and,
  on request, only pairs with the song highlighted in SONGS. Each row reads "the vocal of X over Y from 2:10 (fit 0.91)" or "Y's bassline under X's drums at
  its drop (fit 0.88)", in the lane colours.
- **HEAR**: eight bars rendered offline - Y's groove stems with X's vocal stem stretched to Y's tempo and
  shifted by the pair's semitones (`lib/dj/preview.py`, the decode helper process and RubberBand offline) -
  and played on the CUE OUTPUT: a second audio device named by `dj.cue_device` in config.yaml. Without one
  the button says so; it never plays a preview on the floor while the Director runs.
- **PLAY THIS PAIR**: Y goes UP NEXT and the pair becomes an accepted IDEA (below) that fires when Y is the record (it was "DROP IT" for a day: "very ambiguous phrasing").

## 2. The DJ's ideas as rows you accept

The DJ's events stop firing on their own. What `_event_step` plans becomes an **IDEA** row in THE PLAN:
"IDEA  1:20  the vocal of X over this groove for two phrases (fit 0.9)" or "IDEA  at its drop  Y's bassline
under these drums - it becomes the record". Right-click: **do it** / **skip**. An idea not taken by the
time it would arm passes ("idea passed: not taken"), and the story row says so. MOMENTS lots = the DJ takes
its own ideas (tonight's behaviour); some = it proposes; rare = it has none. A pair you chose in PAIRS is an
idea already taken. Nothing happens that you did not see and let stand.

## 3. Half-time and double-time parts

A third of the library is outside the wall at any clock. A 64 bpm record's vocal or bassline under a 128
groove is a classic move, and the beat grid makes it exact: the deck gets a **beat multiple** (2 = its
beat is two of the clock's, 0.5 the other way), its beat phase for the PLL is taken at that multiple, and
its rate is the clock over its tempo times the multiple. Only PARTS may play at a multiple - a record sets
the clock and plays at its own tempo. The menu says "half-time: its bassline under these drums"; the tempo
wall's grey lifts for the parts of such a song; the SONGS "via" word stays for it as a record.

## 4. Edits of the record itself

Two buttons in the moments row, composed edits of the playing record, said in THE PLAN a phrase ahead:

- **BUILD**: the eight bars before the record's next drop loop once more with the bass pulled on the second
  pass, then release into the drop on the one - the drop lands eight bars later than written, earned. Armed
  like SLAM: it needs a drop of the record within four phrases and eight bars away at least; else it says
  so. Pressing it again while armed = the loop releases on the next bar.
- **AGAIN**: the record's current section once more - a jump back to the section's start on the next phrase
  boundary, once; the story row says "AGAIN: X's breakdown once more from 2:40".

## The gates

The control gate presses IDEA do-it / skip, BUILD, AGAIN, and a half-time part; the remix gate's scripted
night keeps its checks (no dead bars, harmony together, the record plays whole); the preview renderer is
smoked offline on one pair (a WAV of eight bars, its peak and length checked) - it never opens a device in a
test.

## Build order

2 (ideas), 4 (BUILD / AGAIN), 3 (half-time parts), 1 (PAIRS, the preview, the cue device).
