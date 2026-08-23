"""Gate for lib/dj/preflight - the shadow arm-time render pre-flight.

Three things must hold before the shadow data can be believed:
  1. PARITY: the worker's lean render measures the same kick flam the
     quality harness measures for the same plan (same instrument, same
     submix, same events - a fidelity gap here invalidates every shadow
     log line; two calibrations died that way, see DJ_VERIFICATION.md).
  2. SUBPROCESS: the real worker entry (python -m lib.dj.preflight)
     runs the job end-to-end at idle priority and writes its result.
  3. FAIL-OPEN: a broken job produces an error result, never a raise.

Usage:
    python tools/tests/_dj_preflight_test.py --music D:/Devel/music
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from lib.dj.themes import get_theme
from lib.dj import preflight

import tools.tests._dj_quality_test as Q
from tools.tests._dj_quality_test import force_style

MUSIC = "D:/Devel/music"
if "--music" in sys.argv:
    MUSIC = sys.argv[sys.argv.index("--music") + 1]
Q.MUSIC = MUSIC

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def iter_seams(library, style="long_blend"):
    """(cur, cand, meta, plan) tuples where `style` is legal - mirrors
    render_seam's own planning exactly (force_style pin, seed 7,
    after_s 0.45) so the harness and the worker see the same plan."""
    theme = force_style(get_theme("groove"), style)
    for cur in library[::7]:
        brain = Brain(library, theme, seed=7)
        brain.note_played(cur)
        cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
        if cand is None:
            continue
        plan = brain.plan_transition(cur, cand, meta,
                                     after_s=cur.duration_s * 0.45,
                                     force_style=style)
        if plan["style"] == style:
            yield cur, cand, meta, plan


def main():
    db = LibraryDB(MUSIC)
    library = load_library(db)
    print(f"library: {len(library)} tracks")

    # --- 1. parity: worker measurement vs harness measurement --------
    # Walk seams until the instrument MEASURES one (a both-decline
    # "parity" proves nothing) - cap the walk so the gate stays bounded.
    cur = cand = meta = plan = res = None
    t_worker = 0.0
    for i, (c1, c2, mt, pl) in enumerate(iter_seams(library)):
        t0 = time.time()
        r = preflight.render_and_measure(MUSIC, c1.id, c2.id,
                                         dict(pl), "groove")
        t_worker = time.time() - t0
        cur, cand, meta, plan, res = c1, c2, mt, pl, r
        if r.get("flam_med_ms") is not None or i >= 4:
            break
    check("test seam found", cur is not None,
          f"{cur and cur.title} -> {cand and cand.title}")
    if cur is None:
        sys.exit(1)
    check("worker measures without error", "error" not in res,
          res.get("error") or
          f"flam={res.get('flam_med_ms')}ms decode={res.get('decode_s')}s "
          f"render={res.get('render_s')}s")
    check("instrument yields a number on some seam",
          res.get("flam_med_ms") is not None,
          f"flam={res.get('flam_med_ms')} after {i + 1} seam(s)")

    m = Q.render_seam(library, cur, "long_blend", pair=(cand, meta),
                      gap_policy=False)
    ki = (m or {}).get("kick_iso") or {}
    h_flam = ki.get("flam_med_ms")
    w_flam = res.get("flam_med_ms")
    if h_flam is None and w_flam is None:
        check("parity with harness instrument", True,
              "both decline (same verdict)")
    elif h_flam is None or w_flam is None:
        check("parity with harness instrument", False,
              f"harness={h_flam} worker={w_flam} - one declined")
    else:
        check("parity with harness instrument", abs(h_flam - w_flam) <= 5.0,
              f"harness={h_flam}ms worker={w_flam}ms "
              f"(delta {abs(h_flam - w_flam):.1f}ms)")

    # --- 2. real subprocess entry, idle priority ---------------------
    tmp = tempfile.mkdtemp(prefix="dj_pf_test_")
    job_p, res_p = os.path.join(tmp, "j.pkl"), os.path.join(tmp, "r.json")
    with open(job_p, "wb") as f:
        pickle.dump({"music_root": MUSIC, "a_id": cur.id,
                     "b_id": cand.id, "plan": dict(plan),
                     "theme": "groove"}, f)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    kw = {"cwd": repo}
    if sys.platform == "win32":
        kw["creationflags"] = 0x40          # IDLE_PRIORITY_CLASS
    else:
        kw["preexec_fn"] = lambda: os.nice(19)
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m", "lib.dj.preflight",
                         job_p, res_p], timeout=300, **kw).returncode
    t_sub = time.time() - t0
    sub_res = {}
    if os.path.exists(res_p):
        with open(res_p, encoding="utf-8") as f:
            sub_res = json.load(f)
    check("subprocess worker completes", rc == 0 and sub_res,
          f"rc={rc} flam={sub_res.get('flam_med_ms')}ms "
          f"wall={t_sub:.1f}s")
    if sub_res.get("flam_med_ms") is not None and w_flam is not None:
        check("subprocess repeats in-process result",
              abs(sub_res["flam_med_ms"] - w_flam) <= 5.0,
              f"{sub_res['flam_med_ms']} vs {w_flam}")

    # --- 3. fail-open ------------------------------------------------
    bad = preflight.render_and_measure(MUSIC, -999, cand.id,
                                       dict(plan), "groove")
    check("fail-open on broken job", "error" in bad,
          bad.get("error", "no error field"))

    print(f"\n  deploy telemetry preview: worker wall {t_worker:.1f}s "
          f"in-process / {t_sub:.1f}s as idle subprocess "
          f"(decode {res.get('decode_s')}s of it); arm leads are "
          f"60-110s.")
    if failures:
        print(f"\nFAILURES: {failures}")
        sys.exit(1)
    print("\npreflight gate: all clear")


if __name__ == "__main__":
    main()
