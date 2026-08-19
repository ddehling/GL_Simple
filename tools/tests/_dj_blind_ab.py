"""BLIND A/B for the exit-life fix: same seam, two exits, hidden order.

The fix moves WHERE A leaves on ~21% of pairs. Rating seams one at a
time against a 61% baseline would need ~40 per arm to resolve that;
comparing two versions of the SAME pair cancels the material, so ~15
pairs decides it. This plays both, in random order, without telling you
which is which, and logs your pick.

  python tools/tests/_dj_blind_ab.py --music D:/Devel/music [--n 15]

Per seam: [1] play first  [2] play second  [a]/[b] pick  [s] same
          [k] skip  [q] quit and score

Verdicts land in logs/blind_ab.jsonl with the mapping, so the tally is
computed after the fact and nothing on screen reveals the arm while you
are listening. Resumable: seams already judged are skipped.

DECIDE THE BAR BEFORE LISTENING. On 15 decided pairs: >=11 for the new
exit means it is audible and stays; 8-10 means no detectable effect
(metric-only); <=7 means it hurts and should be reverted.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tools.tests._dj_quality_test as Q
from lib.dj import exitvariants as ev
from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from tools.dj.planner.player import TrackPlayer
from lib.dj.themes import get_theme

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "logs", "blind_ab.jsonl")
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 15


def render_at(lib, a, b, pair):
    """Render this exact seam with the anchors PINNED, at whatever style
    the engine plans FOR THOSE ANCHORS.

    plan_transition honours a caller-supplied pair when the retry flag
    is set (that is how the exit-retry re-plans), which is the only way
    to render one pair at two different exits and change nothing else.

    The style is NOT forced to match across the two arms, and that is
    deliberate: pinning the old dying anchor makes the gates refuse the
    blend and divert to a fade, so forcing a common style simply fails
    to render. The style change IS part of what the fix does and part of
    what the ear will hear, so each arm is rendered as it would actually
    have been played.
    """
    brain = Brain(lib, get_theme("groove"), seed=7)
    s, meta = brain.score(a, b, 0.6, a.bpm, bpm_target=None)
    if meta is None:
        return None
    meta = dict(meta, pair=dict(pair), _exit_retry=True)
    style = brain.plan_transition(a, b, dict(meta),
                                  after_s=a.duration_s * 0.45)["style"]
    m = Q.render_seam(lib, a, style, pair=(b, meta), wav=True)
    if m is None or not m.get("wav"):
        return None
    # render_seam returns metrics, not audio, and writes its WAV to one
    # filename PER STYLE - so the second variant would overwrite the
    # first. Read it back immediately and carry the array on the dict.
    from scipy.io import wavfile
    sr, data = wavfile.read(m["wav"])
    m["_audio"] = (data.astype("float32") / 32767.0)
    return m


def main():
    lib = load_library(LibraryDB(Q.MUSIC))
    by_id = {t.id: t for t in lib}
    brain = Brain(lib, get_theme("groove"), seed=7)
    cur_fn = ev.build_best_pair(ev.VARIANTS["current"][0])
    nol_fn = ev.build_best_pair(ev.VARIANTS["nolife"][0])

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["a_id"], r["b_id"]))
            except Exception:
                pass

    rng = random.Random(818)
    elig = [t for t in lib if t.duration_s > 150]
    work = []
    tries = 0
    while len(work) < N and tries < 6000:
        tries += 1
        a, b = rng.choice(elig), rng.choice(elig)
        if a.id == b.id or (a.id, b.id) in done:
            continue
        aft = a.duration_s * 0.45
        try:
            p_new = cur_fn(brain, a, b, after_s=aft)
            p_old = nol_fn(brain, a, b, after_s=aft)
        except Exception:
            continue
        if not p_new or not p_old:
            continue
        if abs(p_new["out_s"] - p_old["out_s"]) <= 1.0:
            continue                       # both engines agree - no signal
        if brain._exit_life(a, p_old["out_s"]) >= 0.6:
            continue                       # old exit was already alive
        # best_pair alone does not say the pair is PLAYABLE - score()
        # is what refuses tempo-impossible partners, and without its
        # meta there is nothing to render. Check before queueing, or
        # the run dies on the first unscoreable pair.
        try:
            _s, _m = Brain(lib, get_theme("groove"),
                           seed=7).score(a, b, 0.6, a.bpm, bpm_target=None)
        except Exception:
            continue
        if _m is None:
            continue
        work.append((a, b, p_old, p_new))
    print(f"{len(work)} seams where the fix moved the exit\n"
          f"logging to {OUT}\n")

    player = TrackPlayer()
    n_new = n_old = n_same = 0
    for i, (a, b, p_old, p_new) in enumerate(work, 1):
        print(f"[{i}/{len(work)}] {a.title[:34]} -> {b.title[:30]}")
        print("  rendering both versions...", flush=True)
        brain2 = Brain(lib, get_theme("groove"), seed=7)
        s, meta = brain2.score(a, b, 0.6, a.bpm, bpm_target=None)
        if meta is None:
            print("  (pair refused, skipping)\n")
            continue
        m_old = render_at(lib, a, b, p_old)
        m_new = render_at(lib, a, b, p_new)
        if not m_old or not m_new:
            print("  (render failed, skipping)\n")
            continue
        # BLIND: which arm plays first is a coin flip, and nothing
        # printed below distinguishes them.
        first_is_new = rng.random() < 0.5
        slots = {"1": m_new if first_is_new else m_old,
                 "2": m_old if first_is_new else m_new}
        style = f"{m_old.get('style')}/{m_new.get('style')}"
        print(f"  two versions ready "
              f"([1]/[2] play, [a]/[b] pick better, [s] same, "
              f"[k] skip, [q] quit)")
        pick = None
        while pick is None:
            try:
                c = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                c = "q"
            if c in ("1", "2"):
                player.load(slots[c]["_audio"])
                player.play()
                print(f"    playing {c} ... (press enter to stop)")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    pass
                player.pause()
            elif c in ("a", "1b"):
                pick = "1"
            elif c == "b":
                pick = "2"
            elif c == "s":
                pick = "same"
            elif c == "k":
                pick = "skip"
            elif c == "q":
                pick = "quit"
            else:
                print("    [1]/[2] play, [a]/[b] pick, [s] same, "
                      "[k] skip, [q] quit")
        if pick == "quit":
            break
        if pick == "skip":
            print()
            continue
        if pick == "same":
            chose = "same"
            n_same += 1
        else:
            chose = ("new" if (pick == "1") == first_is_new else "old")
            if chose == "new":
                n_new += 1
            else:
                n_old += 1
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "t": time.time(), "a_id": a.id, "b_id": b.id,
                "a": a.title[:44], "b": b.title[:44], "style": style,
                "old_out_s": round(p_old["out_s"], 2),
                "new_out_s": round(p_new["out_s"], 2),
                "old_life": round(brain._exit_life(a, p_old["out_s"]), 3),
                "new_life": round(brain._exit_life(a, p_new["out_s"]), 3),
                "first_played": "new" if first_is_new else "old",
                "picked": chose,
                "old_lurch": m_old.get("lurch_db"),
                "new_lurch": m_new.get("lurch_db"),
                "old_floor": m_old.get("rms_min_ratio"),
                "new_floor": m_new.get("rms_min_ratio")}) + "\n")
        print(f"  logged.\n")

    dec = n_new + n_old
    print(f"\n=== {dec} decided ({n_same} 'same') ===")
    print(f"  new exit preferred: {n_new}")
    print(f"  old exit preferred: {n_old}")
    if dec >= 10:
        share = n_new / dec
        if share >= 11 / 15:
            print("  -> the fix is AUDIBLE. Keep it.")
        elif share > 7 / 15:
            print("  -> no detectable effect by ear (metric-only).")
        else:
            print("  -> the fix sounds WORSE. Revert it.")
    else:
        print("  -> too few decided verdicts to conclude "
              "(the bar was 15).")


if __name__ == "__main__":
    main()
