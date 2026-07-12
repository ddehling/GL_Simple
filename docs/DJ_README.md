# Autonomous DJ Subsystem

Scans a music library offline, understands each track's structure, and
mixes real DJ sets all night — beat-matched, time-stretched (constant
pitch), EQ-blended, loop-rolled — fully autonomously or following a
preplanned setlist. The mix feeds the analyzer's `internal` source, so
every audio-reactive shader dances to what the DJ is actually playing,
and the DJ's planned arc drives the club set's night phase.

## Quick start

```bash
# 1. Put music in <repo_parent>/music (or pass --dir anywhere below)
# 2. Analyze the library (incremental - re-runs only touch new files):
python tools/dj_scan.py
#    add --refine-grids to also re-run tracks whose beat-grid confidence
#    sits below 0.75 (vote-weighted confidence can promote them back into
#    the precision transition styles)

# 3a. Hear it standalone:
python tools/dj_player.py --live --theme groove
# 3b. ...or in the show: open the web panel -> DJ tab -> START THE DJ.

# Plan a set for tonight (native desktop app; audition every seam):
python tools/dj_planner.py
```

## Pieces

| Piece | What it does |
|---|---|
| `tools/dj_scan.py` | Incremental scanner → `music/dj_library.sqlite3` (BPM + ms-accurate beat grid, downbeats, Camelot key, structure sections with busyness/vocalness, loops, mix points, loudness, live-pipeline cross-check) |
| `lib/dj/` | `features` analysis · `db` library · `stretch` WSOLA · `eq` LR4 3-band · `deck`/`submix` playback (ONE engine track, sample-accurate automation, sync PLL) · `brain` selection + transition planning · `themes` arcs · `setlist` compiler · `system` conductor |
| `tools/dj_player.py` | Standalone: `--live`, `--wav out.wav --minutes N`, `--audition A B` (render one seam), `--file X --rate r`, `--setlist NAME` |
| MusicBrainz enrichment | Built into the planner's **Library tab** — the "Enrich (MusicBrainz)" button pulls genre, release year/era, label and canonical identity (free, live, no key) for every track that lacks it, in a background thread with live progress; genres + decade fold into `TrackInfo.all_tags` and appear in the tag browser as they land, steering selection, flavor, and the Set Copilot. Stored per track (DB v9), incremental + resumable, ~1 track/sec. (Spotify audio-features and AcousticBrainz APIs are both dead as of 2024/2022 — MusicBrainz is the durable open source; local acoustic descriptors are the **Mood (ML)** pass below.) `tools/dj_enrich.py` remains as an optional CLI (`--limit/--force/--stats`) but the GUI needs no scripts. |
| Mood (ML) descriptors | Built into the planner's **Library tab** — the "Mood (ML)" button runs the [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) model (PyTorch, same torch+CUDA stack as the vocal pass) over each track and stores real **valence/arousal** (0..1) + **mood tags** (dark, party, epic, melancholic…). `character.py` PREFERS these over its heuristic derivation, so `danceable`/`dark`/`uplifting` tags, valence steering and mood vocabulary in the tag browser + Set Copilot all become ML-grounded. This is the local-acoustic-descriptor path Essentia can't provide on Windows (no wheels); Music2Emo runs natively. Stored per track (DB v10), incremental + resumable, ~3-5 s/track on GPU. Needs a Music2Emotion clone + `requirements-dj-mood.txt` (see that file); `tools/dj_mood.py` (`--limit/--force/--stats/--model-dir`) is the optional CLI. The scanner also now keeps each file's embedded **genre** ID3 tag (free, `file_genre`) as another `all_tags` source. **The live autoDJ + set generation actively USE this** once the library is ≥80% mood-scored: valence continuity at seams (don't cut dark→bubbly), an arousal-blended energy arc, and per-theme danceability targets (`Theme.dance_target`) + ML-mood `prefer_tags`. Below 80% coverage the steering stays OFF (a partially-scored library would bias the scored few against the unscored majority), so it's all-or-nothing. |
| `tools/dj_beatport.py` | Beatport discovery CLI: `login` (paste token / `--pkce`), `search "…" --bpm 118-126 --fit`, `fit <id> --deep` (analyze the preview clip), `wish add/list/open`. Public v4 API = search + metadata + preview audio + your library; NO cart API, so buying is a browser click on the track page. See `lib/dj/beatport.py` for the auth story |
| `tools/dj_planner.py` | PyQt6 planner, five tabs — **Library** (scan button w/ live populate + **"Rescan all"** to force full re-analysis, search, user tags, auto-classification tags, multi-select add, **"🚫 Do not use"** flag — right-click or button — that greys a track out and removes it from EVERYTHING that auto-selects: set generation, the Copilot, and the live autoDJ; the track stays in the browser, "Allow" clears it, DB v11), **Analysis** (zoomable waveform to beat level w/ sections + vocal regions, play/scrub, user IN/OUT/INTEREST cues that override the analyzer), **Set** (the v3 set creator: arc strip showing energy-vs-theme + bpm path + seam quality while you build; beam-search Optimize order; anchor timing solver Auto-fill that actually lands timed anchors; right-click repair — slot alternatives + insert-bridge; per-seam fade-risk/blend-floor/groove-offset/pair-memory badges; report card; seam audition; and the **Set Copilot** — a conversational Claude tool-loop that searches the library, edits the set, and runs the planning ops, every change visible and one-click revertible; runs through your Claude Code session via the `claude` CLI with NO API key when it's installed — same as the narrative editor — otherwise falls back to the `anthropic` SDK with a key you paste in the panel; **▶ Play set** plays the whole compiled set right there without switching to the Mix tab), **Mix** (DJ-style timeline w/ overlap, beat ticks, real gain/EQ envelopes; play the whole set, jump tracks/seams), **Discover** (Beatport: sign in with your Beatport **username + password** — the app does the OAuth automatically, no token hunting; password goes only to Beatport's server, only the token is stored. Then search live results with per-result fit vs your set's last track + library; preview audio; ♥ wishlist; open-on-Beatport to buy; "add to set" ghosts a track from its analyzed preview so you can audition the seam before purchasing) |
| Web `/dj` tab | LIVE control only: start/stop, now/next + blend countdown, theme, energy nudge, autopilot, skip, ABORT MIX (recalls an armed transition before its point of no return), setlist picker |

## Config (`config.yaml`)

```yaml
dj:
  enabled: true        # availability only - never auto-plays on boot
  music_dir: ""        # empty = <repo_parent>/music
  theme: groove        # chill_evening / groove / peak_heavy / wind_down / all_night
  night_hours: 6.0     # all_night arc length
  stretch_max: 1.08
```

## How it stays musical

- Track selection couples SONG choice to MIX quality: tempo fit (≤8%
  stretch, half/double-time reads), Camelot compatibility, energy vs the
  theme's arc, recency — AND section-pair mixability: transitions land on
  detected structure boundaries, and two busy/vocal sections never blend
  over each other.
- Selection is TRANSITION-AWARE: candidates whose best seam would be
  forced to a long_fade (loose grid, beatless seam, vocal-over-vocal)
  lean down, as do pairs whose groove offsets differ enough to flam.
  Pair scoring walks the 2 Hz energy curves through the seam so a blend
  never lands in (or hands over into) a near-silent stretch. Result on
  the real library: long_fade share 55% → 38%.
- Every finished seam is SELF-ASSESSED from its own measurements (worst
  audible grid flam, level holes); a measured train-wreck is stored as a
  gentle auto thumbs-down in cross-night pair memory (half an operator
  vote) — the DJ improves nightly with nobody touching a button.
- Transition/technique repertoire (all beat-matched via a sync SNAP at
  launch — the incoming deck's phase is instantly aligned to the playing
  track, then a PLL holds it, ±1.2% authority):
  - `long_blend` / `bass_swap` — staged-EQ blends (highs→mids→bass swap)
  - `cut_at_drop` — hard cut on the incoming track's drop
  - `loop_roll_exit` — shrinking loop-roll outro
  - `bassline_layer` — isolate A's groove as a looping bed, ride B's
    melody/vocals over it beat-locked for ~16 bars, then hand the low over
  - `double_drop` — align both tracks' drop onsets for a peak hit (A ducks
    so it doesn't clip)
  - `loop_build` — stutter a shrinking loop into A's drop to build tension,
    release exactly on the drop as B slams in
  - `long_fade` — fallback for low-confidence grids
  Styles are gated by per-track analysis confidence and theme weights.
- Incoming deck launches bar-aligned from the DB grid, stretched to the
  running tempo, then a PLL trims ±0.3% on measured beat-phase error;
  after the handover the new track glides back to its natural tempo.
- An armed transition can be RECALLED (web ABORT MIX, or a skip during
  the armed window) up to its point of no return — every style stamps
  the decisive clock (bass swap / cut / drop) into its plan; past it,
  finishing sounds better than any rescue.
- A CONTINUITY WATCHDOG guarantees the music never simply runs out: if
  nothing is armed ~20s before the current track ends (persistent
  "no compatible next", stuck decode, dead planner), it force-picks
  (ignoring tempo gates if it must), buys time with a safety loop over
  the last phrase, and hands off with a clock-domain fade.
- Audit everything before the night: planner seam audition, or
  `dj_player --audition "trackA" "trackB"`.

## Tests (all self-checking, `ALL PASS` gates)

`_dj_features_test` · `_dj_stretch_test` · `_dj_mix_test` ·
`_dj_brain_test` (incl. full autonomous end-to-end through the hand-pumped
engine judged by the live signals pipeline) · `_dj_setlist_test` ·
`_dj_rescue_test` (abort/skip-while-armed/watchdog through the offline
DJSystem) · `_dj_quality_test` / `_dj_theory_test` / `_dj_soak_test`
(real-library audio invariants, DJ-practice conformance, full-night soak) ·
`_dj_enrich_test` (MusicBrainz + DB v9) · `_dj_mood_test` (Music2Emo mood-pass
wiring + DB v10 — canned blobs, no torch) · `_dj_exclude_test` (do-not-use
flag DB v11 + boundary filter + save-set invariant).

## Gotchas (hard-won)

- The live `BeatDetector` quantizes BPM to integer 40 fps lags (±2.5%);
  measure tempo precision with `features.estimate_beat_grid`, never the
  live detector.
- Analyzer chroma is A-origin: `c_origin[j] = a_origin[(j+3)%12]`.
- Spectral-flux onsets LEAD the true transient by ~28 ms with our 4096
  framing — `features.ONSET_LATENCY_S` compensates; don't remove it.
- WSOLA legitimately duplicates/skips the odd transient beyond ±5%
  stretch; the brain prefers small ratios for a reason.
- Windows can't decode m4a/aac via miniaudio — PyAV fallback handles it.
