---
name: dj-verify
description: Run the DJ verification stack (gates, quality audit, night census) with the validated instruments and interpretation guardrails. Use after changing DJ engine code (brain/submix/deck/beatpower/system) or when asked to check DJ mixing quality. `/dj-verify` = quick gates; `/dj-verify full` adds the rendered quality gate and the 1,008-seam night census.
---

# DJ verification

First **read `docs/DJ_VERIFICATION.md`** — it names which instruments
are ear-validated and which are known overreaders, the exact bars, and
the census baseline to compare against. Do not interpret any number
before reading it.

Ground rules for every run on this machine:
- Repo root `d:\Devel\GL_Simple`, plain `python` (no venv).
- **Always `--music D:/Devel/music`** — some defaults point at an
  empty Desktop DB.
- Each `_*.py` under `tools/tests/` is its own gate, run directly
  (no pytest). Long runs go in the background; renders are CPU-heavy,
  so don't stack them on top of each other.

## Quick (default, ~10 min)

Run in this order, report PASS/FAIL per gate:

```
python tools/tests/_dj_phasewaiver_test.py
python tools/tests/_dj_quality_test.py --audit-only --music D:/Devel/music
python tools/tests/_dj_persona_sim.py --music D:/Devel/music
python tools/tests/_dj_setlist_test.py --music D:/Devel/music
```

## Full (`full` argument, ~1.5 h — renders audio offline)

Quick suite first, then:

1. **Rendered quality gate** (~15 min):
   `python tools/tests/_dj_quality_test.py --music D:/Devel/music`
2. **Night census** (~55 min): clear old shards
   (`logs/night_sim/w*.jsonl*`), launch 4 detached workers
   (`--worker N --nights 3 --seams 84 --music D:/Devel/music`,
   N=0..3), watch for the 4 `.done` files, then
   `python tools/tests/_dj_night_sim.py --report`.
3. Compare the census against the baseline table in
   `docs/DJ_VERIFICATION.md`. Actionable rows ONLY: `KICK FLAM >45ms
   (isolated-kick)`, `level lurch`, `dead air`, `double bass`,
   `clipping`. The `env-xcorr` and `audible-meter` rows are retained
   overread comparisons — never act on them.

## If an instrument was added or changed

Before believing it anywhere: fresh ear-anchor run
(`del logs/ear_anchor.jsonl`, then
`python tools/tests/_dj_ear_anchor.py --music D:/Devel/music
--limit 80`, ~15 min). It must order the operator's verdicts
(good < passable < bad). An instrument that fails stays a diagnostic,
never a gate — two instruments already failed this exam inverted.

## Reporting

Lead with what changed vs the documented baseline, per believable
axis. A regression in a believable axis blocks; movement in the
overread comparison rows is noise. If a gate fails on machinery
unrelated to the change being verified, say so explicitly rather than
absorbing it.
