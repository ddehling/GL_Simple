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
| `tools/dj_planner.py` | PyQt6 set planner: structure strips, anchors + autofill, compiled plan with warnings, in-app seam audition |
| Web `/dj` tab | LIVE control only: start/stop, now/next + blend countdown, theme, energy nudge, autopilot, skip, setlist picker |

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
- Styles (`long_blend`, `bass_swap`, `cut_at_drop`, `loop_roll_exit`,
  fallback `long_fade`) are gated by per-track analysis confidence.
- Incoming deck launches bar-aligned from the DB grid, stretched to the
  running tempo, then a PLL trims ±0.3% on measured beat-phase error;
  after the handover the new track glides back to its natural tempo.
- Audit everything before the night: planner seam audition, or
  `dj_player --audition "trackA" "trackB"`.

## Tests (all self-checking, `ALL PASS` gates)

`_dj_features_test` · `_dj_stretch_test` · `_dj_mix_test` ·
`_dj_brain_test` (incl. full autonomous end-to-end through the hand-pumped
engine judged by the live signals pipeline) · `_dj_setlist_test`.

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
