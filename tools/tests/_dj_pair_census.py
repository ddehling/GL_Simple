"""The pair census (docs/DJ_ORDERS.md): every pair the PAIRS list would offer, planned by the SAME planner that
executes a pair (lib/dj/orders.plan_pair_entry) against the record's real stems - no audio, decode only.

    python tools/tests/_dj_pair_census.py --music D:/Devel/music [--n 60] [--all] [--record <title prefix>]

Prints feasible / refused with the reasons, and writes .stems/pair_plans.json - the PAIRS list hides what the
planner refused and says the entry it decided. Re-run after new stems or a new pairings file. Records are decoded
one at a time (whole stems in memory: one job at a time, the GPU rule's cousin).
"""
import gc
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.tests._low_priority import PRIORITY  # noqa: E402  (idle priority: it may run beside the live planner)
os.environ.setdefault("DJ_NO_PLAY_LOG", "1")
os.environ.setdefault("DJ_NO_MIDI", "1")


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    n = arg("--n", 60, int)
    want_r = arg("--record", "", str).lower()
    from lib.dj import director as D
    from lib.dj.orders import plan_pair_entry, PLAN_VERSION
    from lib.dj.remix import RemixConductor
    from lib.dj.stemload import decode_song
    d = D.Director(music, engine=None, theme="groove")
    pairs = [p for p in d.pairs(n=(100000 if "--all" in sys.argv else n)) if p["kind"] == "vocal"]
    if want_r:
        pairs = [p for p in pairs if p["r"].title.lower().startswith(want_r)]
    if not pairs:
        print("no vocal pairs offered")
        return 1
    path = os.path.join(music, ".stems", "pair_plans.json")
    try:
        plans = json.load(open(path, encoding="utf-8"))
    except Exception:
        plans = {}
    by_rec = {}
    for p in pairs:
        by_rec.setdefault(p["r"].id, []).append(p)
    print(f"{len(pairs)} vocal pairs over {len(by_rec)} records")
    t0 = time.time()
    n_ok = n_ref = 0
    reasons = {}
    for i, (rid, ps) in enumerate(by_rec.items()):
        R = ps[0]["r"]
        try:
            _samples, _stems, rms = decode_song(d.db.abs(R.path), music, R.id, 0.0, R.sections)
            env = (rms or {}).pop("_env", None)
        except Exception as e:  # noqa: BLE001
            print(f"  {R.title[:30]}: decode failed ({type(e).__name__}: {e})")
            env = None
        for p in ps:
            bar_r = 4.0 * float(R.period_s)
            if env is None:
                plan = {"ok": False, "why": "its stems could not be decoded", "entry": None, "entry_t": None, "event_t": None, "bars": None}
            else:
                plan = plan_pair_entry(R, env, p["r_start"], p["r_end"], 8, bar_r, RemixConductor._stem_carries,
                                       s_start=p.get("s_start"), s_end=p.get("s_end"), s_rate=p.get("s_rate") or 1.0)
            key = f"{R.id}:{int(round(p['r_start']))}"
            plans[key] = {"ok": bool(plan["ok"]), "why": plan["why"][:160], "entry": plan.get("entry"), "entry_t": plan.get("entry_t"),
                          "event_t": plan.get("event_t"), "bars": plan.get("bars"), "t": time.time(), "record": R.title[:40], "v": PLAN_VERSION}
            if plan["ok"]:
                n_ok += 1
            else:
                n_ref += 1
                k = plan["why"].split(":")[-1].strip()[:60] if "drums carry nowhere" not in plan["why"] else "drums carry nowhere before the groove"
                reasons[k] = reasons.get(k, 0) + 1
            mark = "ok " if plan["ok"] else "REF"
            print(f"  {mark} {R.title[:28]:28s} {p['s'].title[:24]:24s} groove {int(p['r_start']) // 60}:{int(p['r_start']) % 60:02d}  {plan['why'][:90]}")
        del _samples, _stems, rms, env
        gc.collect()
        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(by_rec)} records, {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(plans, open(tmp, "w", encoding="utf-8"), indent=0)
    os.replace(tmp, path)
    print(f"\n== the pair census: {n_ok} feasible, {n_ref} refused of {n_ok + n_ref} in {(time.time() - t0) / 60:.1f} min → {path}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {v:3d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
