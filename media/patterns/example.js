// Example Strudel pattern for lib/gen (python tools/gen/gen_player.py --strudel media/patterns/example.js)
// One cycle == one bar. Globals from the composer: energy (0..1), section, bar, key, bpm, phrase, chords.
stack(
  s("bd*4, [~ cp]*2, hh(5,8,<0 2>)").gain(0.9),
  s("oh").struct("~ ~ x ~").gain(0.6),
  note("<0 3 5 7>(3,8)").scale("A1:minor").s("bass").lpf(600 + energy * 900),
  note("0 2 4 [6 7]").scale("A3:minor").off(1/8, x => x.add(7)).s("lead").gain(0.4 + energy * 0.4).sometimesBy(0.3, x => x.gain(0.2)),
  note("<[0,2,4] [3,5,7]>").scale("A2:minor").s("pad").gain(0.5)
)
