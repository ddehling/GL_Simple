"""THE PLAN GATE - a planned night plays as planned (docs/DJ_SETS.md section 7, phase 1 (a) and (b); 2026-10-28).

Drives the Director headless in RECORD mode with a set planner theme (a preset), the planner in its own process as live,
no audio device (the mixer pulled by hand, or the engine's own render-ahead ring drained by a thread in the device's
place). Two parts per seed:

THE NIGHT (fast-forward: the mixer pulled as fast as --pace allows; decodes are wall seconds) - one planned night of
--minutes, nobody pressing anything:
  ORDER         every record that played was the plan's slot at that moment (100 %; an alternate taken, a gap the DJ
                filled or a record the plan did not offer is a miss); also against the first plan, record by record
  RELAXED       blends the player relaxed on a plan pick (the target is none; reported)
  CUTS          zero cuts on plan-to-plan seams (other cuts reported apart)
  CHAPTERS      each chapter arrives within one record of its planned time: the record playing when the night's clock
                crosses a chapter's start is the chapter's first planned record or the one before or after it
  LENGTH        the night's last planned record ends within a record of the night's length (reported)
  DEAD AIR      no silence over 1.5 s outside the first record's wait and the end (the record gate's measure)
  THEN ROWS     after every plan the subprocess sent, THE PLAN's THEN rows are its records in its order (the answer as
                the planner sent it, minus what has since played or left)
  ORDER         (2026-10-29) from each plan's answer on, what played is that answer's order; nobody pressing, the only
                re-plan allowed is the end extension
  LIGHTS        the club's dj_arc_phase (Stories_OGL._dj_arc_outstate) is the plan's progress, rising; dj_arc_heat the
                chapter's energy
  CLOSED        Director.close_planner ends the planner's process
  ERRORS        Director / player / plan errors stay None

THE REFUSAL (fast-forward like the night) - decision 1 forced: the player's strict placement is made to refuse one planned
pair, so the planned record cannot be placed after the record before it. The slot is the seed's: the (seed+1)-th from the
third on that carries an alternate (2026-10-29: one fixed pair on every seed made seeds 2-3 add nothing):
  ALTERNATE     the slot's alternate plays, as the plan's pick, and its row says "picked again: ..."
  RE-SEARCH     slots +2 on are re-searched once it is kept, and followed; 0 relaxed blends; at most PLAN_PROBES probes
                (counted by the gate's own hook: the player's decode probes it refused, not the player's capped counter)
THE ALTERNATE REFUSED (the same slot, its first alternate refused too - the chain past one probe):
  ALTERNATE     the second alternate plays as the plan's pick ("picked again"); with none left, the brain fills the gap
                with a MIXABLE record: the player's own seam into it a blend or a breakdown swap, not relaxed (2026-10-29;
                the forced refusal is the plan's - the brain's picks and its pre-check meet the real rules)
  RE-SEARCH     as above, from the replacement once it is kept; at most PLAN_PROBES probes; 0 relaxed blends
EVERY ALTERNATE REFUSED (the same slot, all its alternates refused - the plan's gap, which the presets' 2-3 alternates
never reach on their own): the brain's MIXABLE gap record plays, its row says "the DJ filled a gap" (the probes spent,
when the slot carries more candidates than PLAN_PROBES), the re-search follows from it once it is kept (a gap's
trigger, never the "picked again" of the last alternate refused); at most PLAN_PROBES probes
FEWER ALTERNATES (2026-10-29: the slot in the applied plan trimmed to fewer alternates than PLAN_PROBES, all refused):
as above, the row saying "nor its alternates" - the plan's own end of a slot, run against the real player

THE CONTROLS (real time, the engine's producer thread and ring as live, a thread drains it like the device callback)
- --presses NEXT / BAD / UP NEXT at random states while the plan is followed, then BAD / UP NEXT until 20 presses have been
answered by a re-plan:
  TRIGGERS      every change of the plan has a trigger; a change of the planned order with none is BROKEN
  SLOT +1       the next record, once kept on its deck, is the record that plays next (nothing here ejects)
  PRESSES       every BAD and UP NEXT re-planned (a change whose trigger names it); every press fired; 20 re-plans
  UP NEXT       rejoined or re-searched (reported); a slot skipped for your record never comes back
  BAD           the slots kept are the old order up to the first that carries one of the BAD's tags (or breaks)
  UNDERRUNS     zero ring underruns; the ring's lowest depth (the callback's margin) reported, under 100 ms fails
  RE-PLAN COST  each answer's wall time; then a re-plan of 90 min and of 6 h sent while the night plays, timed
  DEAD AIR / ERRORS as above

    python tools/tests/_dj_plan_gate.py --music D:/Devel/music [--seeds 1-2] [--preset "Desert Night"] [--minutes 90]
        [--pace 8] [--presses 20] [--log-dir DIR] [--skip-night] [--skip-refusal] [--skip-controls]

Runs at normal priority (the controls measure the audio margin the way the show runs, beside the planner's
below-normal child). Exit 0 = ALL OK, 1 = FAIL.
"""
import math
import os
import random
import sys
import tempfile
import threading
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410                                       # 0.1 s
WIN = RATE // 100                                  # 10 ms
SIL = 0.004
AUDIBLE_GAIN = 0.1


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def seeds_of(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def silent_runs(rms):
    s = np.concatenate([[0], (rms < SIL).astype(np.int8), [0]])
    d = np.diff(s)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def dead_air(rms, first_win):
    """Silences over 1.5 s after the first record sounded and before the end (s, length)."""
    out = []
    for s0, e0 in silent_runs(rms):
        if e0 - s0 <= 150 or s0 <= first_win or e0 >= len(rms):
            continue
        out.append((s0 * 0.01, (e0 - s0) * 0.01))
    return out


def make_director(music, log_dir):
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track, played_log_dir=log_dir)
    d.plan_log_dir = log_dir
    return engine, d


def then_ids(d):
    """THE PLAN's THEN rows (the planned records, not the rows a change took out)."""
    try:
        rows = d._plan_rows()
    except Exception:  # noqa: BLE001
        return None
    return [r["id"] for r in rows if r["kind"] == "THEN" and not str(r["key"][0]).startswith("then-x")]


REPLANS = 20                                       # forced re-plans the controls part waits for


def press_replans(pl):
    """The plan's changes answering a press (BAD, UP NEXT, not tonight), after the first plan."""
    return [c for c in pl.changes[1:] if any(w in str(c["trigger"]) for w in ("BAD on", "your UP NEXT", "not tonight"))]


def then_want(pl, ch):
    """What THE PLAN's THEN rows must show after change `ch`: THE PLANNER'S OWN ANSWER (`sent`, as the subprocess sent it,
    its fixed prefix cut off) minus the records that have since left it - played, on a deck, taken by an alternate, or
    dropped after the change (2026-10-29: the check compared NightPlan's rows with NightPlan's rows before)."""
    gone = {r["id"] for r in pl.rows if r["state"] != "planned"} | {r.get("alt_of") for r in pl.rows if r.get("alt_of")}
    gone |= {x["id"] for x in pl.played} | {x["id"] for x in pl.dropped if float(x.get("t") or 0) >= ch["t"] - 1e-6}
    # (cut at the records on the decks, not at the answer's whole fixed prefix: a rejoin fixes your queued records too,
    # and those are THEN rows - "your record")
    return [i for i in ch["sent"][len(ch.get("fixed") or []):] if i not in gone][:8]


def words_of(t):
    """A record's words as the planner reads them (setplanner _Fact.words): tags, genres, moods, lower case."""
    return ({str(x).lower() for x in (getattr(t, "all_tags", None) or [])} | set(getattr(t, "genre_set", None) or ())
            | {str(m).lower() for m in (getattr(t, "ml_moods", None) or [])})


def order_misses(pl):
    """THE PLANNED ORDER IS WHAT PLAYED, 100 % (DJ_SETS.md section 7, phase 1): from the moment each plan was applied,
    the records that played until the next plan are that plan's order, record by record. Returns the misses."""
    played = [x["id"] for x in pl.played]
    out = []
    chs = list(pl.changes)
    for i, c in enumerate(chs):
        n0 = c["n_played"]
        n1 = chs[i + 1]["n_played"] if i + 1 < len(chs) else len(played)
        got, want = played[n0:n1], c["order"][:max(0, n1 - n0)]
        if got != want:
            out.append(f"plan #{i + 1} ({c['trigger'][:30]}): played {got[:4]} vs planned {want[:4]}")
    return out


def watch_cut(p, last, seen):
    cut = dict(p.cut) if p.cut else None
    sig = (cut.get("seq"), cut.get("clock"), cut.get("swap_clock"), cut.get("state")) if cut else None
    if sig != last[0]:
        last[0] = sig
        if cut:
            cut["_old_tid"] = p.songs[cut["a"]].track.id if cut.get("a") in p.songs else None
            cut["_new_tid"] = p.songs[cut["b"]].track.id if cut.get("b") in p.songs else None
            seen.append(cut)


def transitions(played, cuts):
    """[(a, b, the last plan seen for that pair)] for every record change."""
    out = []
    for a, b in zip(played, played[1:]):
        plan = None
        for c in reversed(cuts):
            if c.get("_old_tid") == a["id"] and c.get("_new_tid") == b["id"]:
                plan = c
                break
        out.append((a, b, plan))
    return out


# ---- THE NIGHT ------------------------------------------------------------------------------------------------------------
def run_night(music, seed, preset, minutes, pace, log_dir, refuse=0):
    """refuse: 0 = the night; 1 = the refusal (the seed's slot refused); 2 = the alternate refused (the slot and its
    first alternate refused); 3 = every alternate refused (the slot and all its alternates: the brain's gap); 4 = as 3,
    the slot trimmed to fewer alternates than PLAN_PROBES ("nor its alternates")."""
    from Stories_OGL import EnvironmentalSystem
    from lib.dj.recordplay import PLAN_PROBES
    arc_of = EnvironmentalSystem._dj_arc_outstate
    engine, d = make_director(music, log_dir)
    t0 = time.time()
    ok, msg = d.set_theme_plan(preset, length_min=minutes, seed=seed)
    if not ok:
        return {"fatal": f"set_theme_plan: {msg}"}
    if not d.plan_wait(300.0) or not d.plan.rows:
        return {"fatal": f"no plan in 300 s: {d.plan.error if d.plan else '?'}"}
    pl = d.plan
    plan0 = [dict(r) for r in pl.rows]
    cost0 = dict(pl.costs[-1]) if pl.costs else {}
    print(f"seed {seed}: {pl.name} planned in {time.time() - t0:.1f} s ({len(plan0)} records, chapters "
          + ", ".join(f"{c['name']} {c['start_s'] / 60:.0f}'" for c in pl.chapters) + ")", flush=True)
    if not d.start(threaded=False):
        return {"fatal": f"did not start: {d.last_error}"}
    p = d.player
    forced = None
    if refuse:
        # DECISION 1, FORCED (2026-10-29): the player's strict placement refuses one planned pair - the seed's slot (the
        # (seed+1)-th from the third on that carries an alternate; one fixed slot made every seed the same pair), after
        # the record before it - so the plan's own path runs: the slot's alternate plays ("picked again"), slots +2 on
        # are re-searched once it is kept, no relaxed blend, at most PLAN_PROBES probes. refuse=2 refuses its first
        # alternate too: the second alternate, or with none the brain's gap record, and the re-search from it
        ks = [k for k in range(2, len(plan0)) if plan0[k].get("alts")]
        if not ks:
            return {"fatal": "no planned slot with an alternate to force"}
        k = ks[seed % len(ks)]
        alt_rows = list(plan0[k]["alts"])
        if refuse >= 4:
            # FEWER ALTERNATES THAN PROBES (2026-10-29): the presets carry 2-3, so with all refused the probes run out first
            # and "nor its alternates" never ran against the real player - the applied plan's slot keeps fewer, so the
            # slot's candidates are used up while probes are left
            alt_rows = alt_rows[:max(1, PLAN_PROBES - 2)]
            live = next((r for r in pl.rows if r["id"] == plan0[k]["id"] and r["state"] == "planned"), None)
            if live is None:
                return {"fatal": f"slot {k + 1} is not in the applied plan to trim"}
            live["alts"] = [dict(x) for x in alt_rows]
        alts = [int(x["id"]) for x in alt_rows]
        a_id, b_id = plan0[k - 1]["id"], plan0[k]["id"]
        refused = [b_id] + (alts if refuse >= 3 else alts[:1] if refuse >= 2 else [])
        orig = p._find_blend_strict
        hits = []

        def strict(rec, nxt, _orig=orig):
            if rec.track.id == a_id and nxt.track.id in refused:
                tid = int(nxt.track.id)
                # WHOSE PICK (2026-10-29): the forced case is decision 1's - only the PLAN's picks of these records are refused
                # (the plan's pre-check probes a Rec on deck "p"; a record on a deck is the plan's when the player says so).
                # The brain's own pre-check (deck "q") and the brain's picks meet the real rules: refusing them made the
                # brain's gap record look unmixable to the gate when the rules had never been asked
                whose = "plan" if nxt.deck == "p" else ("brain" if nxt.deck == "q" else p._src.get(tid, "brain"))
                if whose != "plan":
                    return _orig(rec, nxt)
                # WHO ASKED: "pre" the plan's pre-check, "player" the decode probe (its own thread), "planning" the blend's
                # placement or status()'s projection (counted apart: how often status() is polled must not move a count)
                who = "pre" if nxt.deck == "p" else ("player" if threading.current_thread().name.startswith("record-decode") else "planning")
                hits.append((time.time(), tid, who, whose))
                p._no_blend_why = "refused by the gate (decision 1's forced case)"
                return None
            return _orig(rec, nxt)
        p._find_blend_strict = strict
        forced = {"k": k, "a": a_id, "b": b_id, "b_title": plan0[k]["title"], "alts": alts, "refused": refused,
                  "alt_titles": [str(x.get("title"))[:24] for x in alt_rows], "refuse": refuse,
                  "hits": hits, "max_plan_repicks": 0, "probes": PLAN_PROBES}
        print(f"seed {seed}: forcing a refusal of slot {k + 1} ({plan0[k]['title'][:30]}"
              + (" and every alternate" if refuse >= 3 else f" and its first alternate {forced['alt_titles'][0]}" if refuse >= 2 else "")
              + f") after {plan0[k - 1]['title'][:30]}; {len(alts)} alternates" + (" (trimmed)" if refuse >= 4 else "")
              + f": {', '.join(forced['alt_titles'])}", flush=True)
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    rms10, errors, cuts, lights, then_checks = [], [], [], [], []
    err_seen = {}
    cut_last = [None]
    n_changes = len(pl.changes)
    first_nb, t_play0 = None, None
    end_s = max(float(plan0[-1].get("end_s") or 0.0), 60.0 * minutes)
    nb = 0
    t_wall0 = time.time()
    while True:
        x = np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2)
        nb += 1
        rms10.append(np.sqrt(np.mean(x[:BLOCK].astype(np.float64).reshape(10, -1) ** 2, axis=1)))
        d.step()
        watch_cut(p, cut_last, cuts)
        for key, obj in (("d", d), ("p", p), ("plan", pl)):
            e = getattr(obj, "last_error", None) if key != "plan" else obj.error
            if e and e != err_seen.get(key):
                err_seen[key] = e
                errors.append((round(pl.night_s(), 1), key, str(e)))
        if forced is not None:
            forced["max_plan_repicks"] = max(forced["max_plan_repicks"], p._plan_repicks)
            if len(pl.played) >= forced["k"] + 3:
                break                                  # (the alternate played and the re-searched plan has been followed)
        if len(pl.changes) != n_changes:
            n_changes = len(pl.changes)
            ch = pl.changes[-1]
            want = then_want(pl, ch)
            got = then_ids(d)
            then_checks.append((round(pl.night_s()), ch["trigger"], got == want, got, want))
        if first_nb is None:
            if p.record is not None and ((p.submix.telemetry or {}).get("decks") or {}).get(p.record, {}).get("playing"):
                first_nb, t_play0 = nb, time.time()
                print(f"seed {seed}: the first record plays {time.time() - t_wall0:.1f} s after START ({p.songs[p.record].track.title})", flush=True)
                got = then_ids(d)
                want = then_want(pl, pl.changes[0])
                then_checks.append((0, "the first plan", got == want, got, want))
            elif time.time() - t_wall0 > 180.0:
                d.stop(0.1)
                return {"fatal": f"no first record in 180 s: {d.last_error} / {p.last_error}"}
            else:
                time.sleep(0.02)
            continue
        if nb % 20 == 0:
            try:
                dst = d.status()
                lk = arc_of(d, dst)
                lights.append((pl.night_s(), dst.get("plan_progress"), dst.get("plan_heat"), lk.get("dj_arc_phase"), lk.get("dj_arc_heat"),
                               (dst.get("set_plan") or {}).get("chapter")))
            except Exception as e:  # noqa: BLE001
                errors.append((round(pl.night_s(), 1), "status", f"{type(e).__name__}: {e}"))
        lag = (nb - first_nb) * BLOCK / RATE / pace - (time.time() - t_play0)
        if lag > 0.05:
            time.sleep(min(lag, 0.2))
        if nb % 3000 == 0:
            print(f"seed {seed} [{pl.night_s() / 60:5.1f} min] {len(pl.played)} records, {len(pl.changes)} plans, "
                  f"err {d.last_error} / {p.last_error} / {pl.error}", flush=True)
        if pl.night_s() >= end_s:
            break
    out = {"d": d, "p": p, "pl": pl, "plan0": plan0, "cost0": cost0, "rms": np.concatenate(rms10), "first_win": first_nb * 10,
           "cuts": cuts, "errors": errors, "lights": lights, "then": then_checks, "end_s": end_s, "minutes": minutes,
           "wall_s": time.time() - t_wall0, "forced": forced}
    d.stop(0.1)
    # the planner's child closes with the Director (Director.close_planner), not when the GC gets round to it
    proc = getattr(d._planner, "_proc", None) if d._planner is not None else None
    d.close_planner()
    t_c = time.time()
    while proc is not None and proc.poll() is None and time.time() - t_c < 5.0:
        time.sleep(0.05)
    out["child_closed"] = (proc is None or proc.poll() is not None, round(time.time() - t_c, 2))
    return out


def analyse_night(R):
    rows = []

    def add(check, value, ok, note=""):
        rows.append((check, value, ok, note))
    pl, plan0 = R["pl"], R["plan0"]
    add("plan", f"{len(plan0)} records", len(plan0) >= max(3, int(R["minutes"] / 8.0)),
        f"{pl.name}, seed {pl.seed}: first plan in {R['cost0'].get('wall_s')} s wall (search {R['cost0'].get('search_s')} s, "
        f"table {R['cost0'].get('table_s')} s, load {R['cost0'].get('load_s')} s)")
    played = list(pl.played)
    # ORDER: each record the plan's slot at that moment
    miss = [x for x in played if x.get("src") != "plan" or x.get("alt_of")]
    add("order (followed)", f"{len(played) - len(miss)} of {len(played)}", not miss and len(played) >= 3,
        "every record that played was the plan's slot" + ("" if not miss else "; missed: " + "; ".join(
            f"{str(x.get('title'))[:22]} ({x.get('src')}{', alternate of ' + str(x.get('alt_of')) if x.get('alt_of') else ''}: {str(x.get('why') or '')[:50]})" for x in miss[:4])))
    # THE PLANNED ORDER IS WHAT PLAYED, 100 % (2026-10-29: a re-plan used to waive this): record by record against the
    # first plan's answer, and from each later plan's answer on against that one; with nobody pressing anything the only
    # re-plan allowed is the end extension (the night running on in its last chapter's character)
    ids0 = pl.changes[0]["order"] if pl.changes else [r["id"] for r in plan0]
    same = 0
    for k, x in enumerate(played):
        if k < len(ids0) and ids0[k] == x["id"]:
            same += 1
        else:
            break
    n_first = min(len(played), len(ids0), (pl.changes[1]["n_played"] if len(pl.changes) > 1 else len(played)))
    add("order (first plan)", f"{min(same, n_first)} of {n_first}", same >= n_first, "record by record against the answer to the plan made before START")
    miss = order_misses(pl)
    add("order (every plan)", f"{len(miss)} misses", not miss, "from each plan's answer on, what played is its order" + ("; " + "; ".join(miss[:2]) if miss else ""))
    replans = [c for c in pl.changes[1:]]
    other = [c for c in replans if not c.get("extend")]
    add("re-plans", f"{len(replans)} ({len(other)} not the end extension)", not other,
        "nobody pressed anything: only the end extension may re-plan" + ("; " + "; ".join(c["trigger"][:50] for c in other[:3]) if other else ""))
    ok_c, dt_c = R.get("child_closed", (True, 0))
    add("planner closed", f"{'yes' if ok_c else 'NO'} in {dt_c} s", ok_c, "Director.close_planner ends the planner's process")
    # RELAXED / CUTS
    tr = transitions(played, R["cuts"])
    relaxed = [(a, b) for a, b, c in tr if c is not None and c.get("relaxed") and c.get("src_b") == "plan"]
    add("relaxed on plan picks", str(len(relaxed)), len(relaxed) <= 1,
        "the target is none" + ("; " + "; ".join(f"{a['title'][:18]} -> {b['title'][:18]}" for a, b in relaxed[:3]) if relaxed else ""))
    kinds = [(c or {}).get("kind", "?") for _a, _b, c in tr]
    pp_cuts = [(a, b) for (a, b, c) in tr if (c or {}).get("kind") == "cut" and a.get("src") == "plan" and b.get("src") == "plan"]
    add("cuts plan-to-plan", str(len(pp_cuts)), not pp_cuts,
        f"{len(tr)} seams: " + ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds)))
        + ("; " + "; ".join(f"{a['title'][:18]} -> {b['title'][:18]}" for a, b in pp_cuts[:3]) if pp_cuts else ""))
    other_cuts = sum(1 for (a, b, c) in tr if (c or {}).get("kind") == "cut") - len(pp_cuts)
    add("other cuts", str(other_cuts), True, "cuts on a seam that is not plan-to-plan (reported)")
    unplanned = [k for k in kinds if k == "?"]
    add("seams seen", f"{len(tr) - len(unplanned)} of {len(tr)}", not unplanned, "a plan of the player's seen for every record change")
    # CHAPTERS
    starts = [x.get("start_s") for x in played]
    for ci, c in enumerate(pl.chapters):
        b0 = float(c["start_s"])
        if b0 >= R["end_s"] - 1.0:
            continue
        pf = next((k for k, r in enumerate(plan0) if r.get("chapter") == ci), None)
        j = None
        for k, s in enumerate(starts):
            if s is not None and s <= b0 + 1e-6:
                j = k
        if pf is None or j is None:
            add(f"chapter {c['name'][:14]}", "unseen", pf is None, f"no planned record in it" if pf is None else "no record played at its start")
            continue
        dt = (starts[pf] - float(plan0[pf]["start_s"])) if pf < len(starts) and starts[pf] is not None else None
        add(f"chapter {c['name'][:14]}", f"record {j + 1} vs {pf + 1} planned", abs(j - pf) <= 1,
            f"at {b0 / 60:.1f}' the record playing is #{j + 1} ({str(played[j]['title'])[:20]}); its first planned record #{pf + 1}"
            + (f" started {dt:+.0f} s off its planned time" if dt is not None else ""))
    last_end = float(plan0[-1].get("end_s") or 0.0)
    add("night length", f"{last_end / 60:.1f} of {R['minutes']:.0f} min", True, "the first plan's last record's end (reported)")
    # DEAD AIR
    dead = dead_air(R["rms"], R["first_win"])
    add("dead air", f"{len(dead)} gaps", not dead, "; ".join(f"{t:.1f}s for {l:.1f}s" for t, l in dead[:4]))
    # THEN ROWS
    bad_then = [x for x in R["then"] if not x[2]]
    add("THEN rows", f"{len(R['then']) - len(bad_then)} of {len(R['then'])}", not bad_then and bool(R["then"]),
        "THE PLAN's THEN rows = the plan the subprocess sent" + ("; " + "; ".join(f"at {x[0]}s ({x[1][:30]}): {x[3]} vs {x[4]}" for x in bad_then[:2]) if bad_then else ""))
    # LIGHTS
    lt = [x for x in R["lights"] if x[1] is not None]
    if lt:
        bad_phase = sum(1 for (_ns, prog, _h, ph, _hh, _c) in lt if ph is None or abs(float(ph) - min(0.999, max(0.0, float(prog)))) > 1e-6)
        back = sum(1 for a, b in zip(lt, lt[1:]) if b[3] is not None and a[3] is not None and b[3] < a[3] - 1e-9)
        heats = [x[4] for x in lt]
        add("lights phase", f"{lt[0][3]:.3f} -> {lt[-1][3]:.3f}", bad_phase == 0 and back == 0 and lt[-1][3] >= 0.85,
            f"dj_arc_phase = the plan's progress ({bad_phase} off, {back} backwards, {len(lt)} readings)")
        add("lights heat", f"{min(heats):.2f}-{max(heats):.2f}", all(0.0 <= h <= 1.0 for h in heats) and (max(heats) - min(heats) > 0.05),
            "dj_arc_heat = the chapter's energy, moving with the story: " + ", ".join(sorted({str(x[5]) for x in lt})))
    else:
        add("lights", "no readings", False, "status() carried no plan_progress")
    # ERRORS
    for t, who, text in R["errors"]:
        add("error", f"{who} at {t:.0f}s", False, text[:140])
    if not R["errors"]:
        add("errors", "none", True, "Director, player and plan errors stayed None")
    add("render", f"{R['wall_s'] / 60:.1f} min wall", True, f"{R['end_s'] / 60:.0f} min of night")
    return rows


def analyse_refusal(R):
    """Decision 1, forced: a planned record the strict blend refuses plays its alternate; with its first alternate refused
    too, the next alternate, or the brain's mixable gap record."""
    rows = []

    def add(check, value, ok, note=""):
        rows.append((check, value, ok, note))
    pl, fx = R["pl"], R["forced"]
    played = list(pl.played)
    k = fx["k"]
    refused = list(fx.get("refused") or [fx["b"]])
    hit_ids = [h[1] for h in fx["hits"]]
    add("refused", f"{len(fx['hits'])} probes", bool(fx["hits"]) and all(i in hit_ids for i in refused[:1]),
        f"the player's strict placement refused {fx['b_title'][:30]} after the record before it"
        + (f" ({hit_ids.count(refused[0])}x), then its first alternate {fx['alt_titles'][0]} ({hit_ids.count(refused[1])}x)" if len(refused) > 1 else ""))
    x = next((y for y in played if y.get("alt_of") == fx["b"]), played[k] if k < len(played) else None)
    k = played.index(x) if x is not None else k
    prev = played[k - 1] if 0 < k <= len(played) else None
    alts_left = [i for i in fx["alts"] if i not in refused]
    if len(refused) > 1:
        # THE CHAIN PAST ONE PROBE (2026-10-29): the slot and its first alternate both refused. The first alternate must
        # have been offered and refused (else the chain did not run: two probes, not one), then either the next alternate
        # plays as the plan's pick, or - none left, or all of them refused by the rules - the brain fills the gap with a
        # record that MIXES with the one playing (the plan's gap is never a licence to cut)
        add("first alt refused", f"{hit_ids.count(refused[1])} probes", refused[1] in hit_ids,
            f"{fx['alt_titles'][0]} was offered after the slot's refusal and refused in turn")
        plan_took = (x is not None and x.get("src") == "plan" and x.get("alt_of") == fx["b"] and x["id"] in alts_left)
        gap = x is not None and x.get("src") == "brain" and not x.get("alt_of")
        if plan_took or not gap:
            idx = fx["alts"].index(x["id"]) if x is not None and x["id"] in fx["alts"] else None
            add("next alternate played", (str(x.get("title"))[:30] if x else "nothing"),
                plan_took and fx["b"] not in {y["id"] for y in played} and not any(i in {y["id"] for y in played} for i in refused),
                f"slot {k + 1}: " + (f"alternate #{idx + 1} of {len(fx['alts'])}, as (track, 'plan')" if plan_took else
                                     f"got {x and (x.get('src'), x.get('alt_of'), x['id'])}, alternates left {alts_left}"))
            why = str((x or {}).get("why") or "")
            add("THEN row says", why[:40], why.startswith("picked again"), "\"picked again: X could not blend after Y, Z instead\"")
        else:
            from lib.dj.recordplay import KEY_BLEND, MAX_TEMPO_GAP, RecordPlayer
            ta = R["d"].track(prev["id"]) if prev else None
            tb = R["d"].track(x["id"])
            tgap = abs(float(ta.bpm) / max(float(tb.bpm), 1e-6) - 1.0) if ta is not None and tb is not None else None
            fit = RecordPlayer.key_fit(ta, tb) if ta is not None and tb is not None else None
            # MIXABLE = THE SEAM THE PLAYER PLANNED (2026-10-29): a blend or a breakdown swap, not relaxed, on the brain's
            # pick - judged from the player's own plan, not from the gate's forced list (the brain may pick a record the
            # plan was refused: the forcing is the plan's, the rules decide the brain's)
            seam = next((c for a_, b_, c in transitions(played, R["cuts"]) if b_["id"] == x["id"] and prev is not None and a_["id"] == prev["id"]), None)
            mix = (prev is not None and prev["id"] == fx["a"] and seam is not None and seam.get("kind") in ("blend", "breakdown")
                   and not seam.get("relaxed") and seam.get("src_b") == "brain")
            add("gap filled, mixable", str(x.get("title"))[:30], mix,
                f"slot {k + 1}: the brain's record after {str((prev or {}).get('title'))[:22]}: "
                + (f"{seam.get('kind')} {seam.get('size')}, relaxed {bool(seam.get('relaxed'))}, src_b {seam.get('src_b')}" if seam else "no seam seen")
                + "; tempo gap " + (f"{tgap * 100:.1f} %" if tgap is not None else "?") + f" (<= {MAX_TEMPO_GAP * 100:.0f} %), key fit "
                + (f"{fit:.2f}" if fit is not None else "unknown") + f" (>= {KEY_BLEND}); alternates left {len(alts_left)}")
            why = str(x.get("why") or "")
            # (nothing of the slot left: "nor its alternates"; the alternates outnumbering the probes: "probes are spent" -
            # which of the two is the player's own count, 2026-10-29: the probes it ran and the gate's hook refused)
            n_player = sum(1 for h in fx["hits"] if h[2] == "player" and h[3] == "plan")
            words ="probes are spent" if n_player >= fx["probes"] else "nor its alternates"
            add("THEN row says", why[:40], why.startswith("the DJ filled a gap") and words in why,
                f"\"the DJ filled a gap (could not blend after Y{', nor its alternates' if words.startswith('nor') else ': the plans probes are spent'})\""
                f" - {n_player} player probes refused, {len(fx['alts'])} alternates")
    else:
        took = (x is not None and x.get("src") == "plan" and x.get("alt_of") == fx["b"] and x["id"] in fx["alts"]
                and fx["b"] not in {y["id"] for y in played})
        add("alternate played", (str(x.get("title"))[:30] if x else "nothing"), took,
            f"slot {k + 1}: one of its alternates, as (track, 'plan')" + ("" if took else f"; got {x and (x.get('src'), x.get('alt_of'), x['id'])}"))
        why = str((x or {}).get("why") or "")
        add("THEN row says", why[:40], why.startswith("picked again"), "\"picked again: X could not blend after Y, Z instead\"")
    # (a gap: the trigger is the gap's - a "picked again" trigger left over from the last alternate refused named a record
    # that never played, 2026-10-29)
    want = ("picked again",) if (x or {}).get("src") == "plan" else ("could not blend after", "the DJ filled a gap")
    re = [c for c in pl.changes[1:] if str(c["trigger"]).startswith(want)]
    add("re-search from it", f"{len(re)} re-plans", len(re) >= 1, "slots +2 on re-searched once the alternate is kept"
        + (f": {re[0]['trigger'][:60]}" if re else ""))
    # ...naming what plays (2026-10-29): the trigger is the "new: ..." line every later THEN row carries, so a refused
    # record named as "Z instead" tells the room a record that never played took the slot
    trig = str(re[0]["trigger"]) if re else ""
    named = [t for t in fx["alt_titles"] if f"{t} instead" in trig and fx["alts"][fx["alt_titles"].index(t)] in refused]
    says = (f"{str(x.get('title'))[:24]} instead" in trig) if (x or {}).get("src") == "plan" else not trig.startswith("picked again")
    add("re-search names it", trig[:32], bool(re) and not named and says,
        "the trigger names the record that took the slot" + (f"; it names {named} (refused, never played)" if named else "")
        + ("" if says or named else f"; {str((x or {}).get('title'))[:24]} played"))
    # ...and asked once the replacement is KEPT (on its deck, or playing): the answer's fixed prefix holds it
    kept = [c for c in re if x is not None and x["id"] in (c.get("fixed") or [])]
    add("re-search kept it", f"{len(kept)} of {len(re)}", bool(kept),
        "the re-search's fixed records (the decks when it answered) hold the replacement"
        + (f": {kept[0].get('fixed')}" if kept else (f"; fixed {[c.get('fixed') for c in re[:2]]}" if re else "")))
    after = played[k + 1:]
    add("then followed", f"{sum(1 for y in after if y.get('src') == 'plan')} of {len(after)}", bool(after) and all(y.get("src") == "plan" for y in after),
        "the records after it are the re-searched plan's")
    tr = transitions(played, R["cuts"])
    relaxed = [(a, b) for a, b, c in tr if c is not None and c.get("relaxed") and c.get("src_b") == "plan"]
    add("relaxed on plan picks", str(len(relaxed)), not relaxed, "decision 1: the alternate, never the relaxed blend")
    # PROBES, COUNTED BY THE GATE (2026-10-29: the player's own _plan_repicks is capped by the player, so "at most N" of it
    # could not fail): the player's decode probes the gate's hook refused after the record before the slot, until the slot
    # was given up (the alternate kept, or the gap)
    n_player = sum(1 for h in fx["hits"] if h[2] == "player" and h[3] == "plan")
    n_pre = sum(1 for h in fx["hits"] if h[2] == "pre")
    n_planning = sum(1 for h in fx["hits"] if h[2] == "planning")
    add("probes", f"{n_player} refused", n_player <= fx["probes"],
        f"the plan's own budget is {fx['probes']}; {n_pre} refusals of the plan's pre-check (never offered); {n_planning} planning "
        f"refusals (a plan pick's blend placement or status()'s projection, not probes); the player's counter peaked at {fx['max_plan_repicks']}")
    for t, who, text in R["errors"]:
        add("error", f"{who} at {t:.0f}s", False, text[:140])
    return rows


# ---- THE CONTROLS ---------------------------------------------------------------------------------------------------------
def run_controls(music, seed, preset, presses, log_dir, minutes=90.0):
    engine, d = make_director(music, log_dir)
    rng = random.Random(5000 + seed)
    ok, msg = d.set_theme_plan(preset, length_min=minutes, seed=seed + 100)
    if not ok or not d.plan_wait(300.0) or not d.plan.rows:
        return {"fatal": f"no plan: {msg} / {d.plan.error if d.plan else '?'}"}
    pl = d.plan
    # THE ENGINE AS LIVE: its own render-ahead producer and ring; a thread in the device callback's place drains it in real
    # time (the device is never opened)
    engine._ring = np.zeros((engine._ring_capacity, 2), dtype=np.float32)
    engine._ring_w = engine._ring_r = 0
    engine._ring_stop.clear()
    if not d.start(threaded=True):
        return {"fatal": f"did not start: {d.last_error}"}
    p = d.player
    engine._producer = threading.Thread(target=engine._render_ahead, daemon=True, name="audio-render")
    engine._producer.start()
    t_pf = time.time()
    while time.time() - t_pf < 5.0:
        with engine._ring_lock:
            if engine._ring_w - engine._ring_r >= engine._ring_target:
                break
        time.sleep(0.01)
    cons = {"rms": [], "stop": False, "late_ms": 0.0, "n": 0}

    def consumer():
        feed = engine._device_feed()
        next(feed)
        t0 = time.perf_counter()
        pulled = 0
        while not cons["stop"]:
            due = int((time.perf_counter() - t0) * RATE)
            if due - pulled >= WIN:
                late = (due - pulled - WIN) / RATE * 1000.0
                cons["late_ms"] = max(cons["late_ms"], late)
                while due - pulled >= WIN:
                    out = np.frombuffer(feed.send(WIN), dtype=np.float32)
                    cons["rms"].append(float(np.sqrt(np.mean(out.astype(np.float64) ** 2))))
                    pulled += WIN
                    cons["n"] += 1
            time.sleep(0.002)
    th = threading.Thread(target=consumer, daemon=True, name="device-stand-in")
    th.start()
    errors, cuts, broken, then_checks, press_log, nexts_kept = [], [], [], [], [], []
    err_seen = {}
    cut_last = [None]
    kinds = ["next"] * 7 + ["bad"] * 7 + ["queue"] * 6
    rng.shuffle(kinds)
    kinds = (kinds * (presses // len(kinds) + 1))[:presses]
    t_wall0 = time.time()
    first_t = None
    next_press = None
    n_changes = len(pl.changes)
    n_dropped = len(pl.dropped)
    fut_last = [r["id"] for r in pl.rows if r["state"] == "planned"]
    kept_next, rec_last = None, None
    starved_at_replan = []
    while True:
        time.sleep(0.1)
        now = time.time()
        rec = p.songs.get(p.record) if p.record else None
        watch_cut(p, cut_last, cuts)
        for key, obj in (("d", d), ("p", p), ("plan", pl)):
            e = getattr(obj, "last_error", None) if key != "plan" else obj.error
            if e and e != err_seen.get(key):
                err_seen[key] = e
                errors.append((round(pl.night_s(), 1), key, str(e)))
        if rec is not None and first_t is None and ((p.submix.telemetry or {}).get("decks") or {}).get(p.record, {}).get("playing"):
            first_t = now
            next_press = now + 40.0
            print(f"seed {seed} controls: the first record plays ({rec.track.title}); {presses} presses to come", flush=True)
        # SLOT +1: a next record kept on its deck is the one that plays next
        try:
            nxt = p._next_rec()
            c = p.cut
            if nxt is None and c is not None and c.get("state") == "started":
                nxt = p.songs.get(c["b"])
            kept = nxt is not None and (nxt.state in ("ready", "playing") or nxt.deck in p._pending)
            rid = rec.track.id if rec is not None else None
            if rid != rec_last:
                if rec_last is not None and kept_next is not None and rid != kept_next:
                    broken.append(f"at {pl.night_s():.0f}s the record became {d.title(rid)[:24]}, not the kept next {d.title(kept_next)[:24]}")
                rec_last, kept_next = rid, None
            if kept:
                if kept_next is not None and nxt.track.id != kept_next:
                    broken.append(f"at {pl.night_s():.0f}s slot +1 changed from {d.title(kept_next)[:24]} to {nxt.track.title[:24]} with no eject")
                kept_next = nxt.track.id
                if not nexts_kept or nexts_kept[-1] != kept_next:
                    nexts_kept.append(kept_next)
        except Exception:  # noqa: BLE001
            pass
        # TRIGGERS: every change of the planned order has one
        fut = [r["id"] for r in list(pl.rows) if r["state"] == "planned"]
        if len(pl.changes) != n_changes:
            for ch in pl.changes[n_changes:]:
                if not str(ch.get("trigger") or "").strip():
                    broken.append(f"a plan change with no trigger (seq {ch.get('seq')})")
                if pl.costs:
                    starved_at_replan.append((ch["trigger"], engine._cb_starved))
            time.sleep(0.25)
            n_changes = len(pl.changes)
            ch = pl.changes[-1]                       # (the newest answer: one applied during the wait is the one to show)
            want = then_want(pl, ch)
            got = then_ids(d)
            then_checks.append((round(pl.night_s()), ch["trigger"], got == want, got, want))
            fut = [r["id"] for r in list(pl.rows) if r["state"] == "planned"]
        elif fut != fut_last:
            explained = (len(pl.dropped) != n_dropped or fut == fut_last[1:] or
                         (fut[1:] == fut_last[1:] and any(str(r.get("why") or "").startswith("picked again") for r in pl.rows)) or
                         fut_last[1:] == fut or fut[:-1] == fut_last[1:])
            if not explained:
                broken.append(f"at {pl.night_s():.0f}s the planned order changed with no trigger: {fut_last[:5]} -> {fut[:5]}")
        fut_last, n_dropped = fut, len(pl.dropped)
        # PRESSES at random states
        if first_t is not None and kinds and now >= next_press:
            kind = kinds.pop(0)
            state = ("blend under way" if (p.cut or {}).get("state") == "started" else
                     ("blend planned" if p.cut else ("next decoding" if p._decoding else "playing")))
            before = len(pl.changes)
            if kind == "next":
                ret = d.next()
            elif kind == "bad":
                ret = d.rate(False)
                ret = (ret or {}).get("did")
            else:
                planned = [r["id"] for r in pl.rows if r["state"] == "planned"]
                played = {int(t) for t, _w in d.played} | {s.track.id for s in p.songs.values()}
                if planned[2:] and rng.random() < 0.7:
                    tid = rng.choice(planned[2:min(len(planned), 7)])
                else:
                    pool = [i for i in (pl.crate_ids or []) if i not in played and i not in planned]
                    tid = rng.choice(pool) if pool else (planned[-1] if planned else None)
                ret = d.queue(tid) if tid is not None else (False, "nothing to queue")
            press_log.append({"kind": kind, "t": now, "night_s": round(pl.night_s()), "state": state, "ret": str(ret)[:80], "changes0": before})
            print(f"seed {seed} [{pl.night_s() / 60:5.1f} min] {kind.upper():5s} ({state}): {str(ret)[:70]}", flush=True)
            next_press = now + rng.uniform(15.0, 45.0)
        # TWENTY FORCED RE-PLANS (DJ_SETS.md section 7: the margin is measured "during 20 forced re-plans"; 2026-10-29: NEXT
        # only re-times, so the press mix alone gave 13-18): BAD / UP NEXT go on until 20 presses have been answered by a
        # re-plan (at most 30 presses more)
        if not kinds and next_press is not None and now >= next_press and len(press_log) < presses + 30 \
                and len(press_replans(pl)) < REPLANS:
            kinds.append(rng.choice(["bad", "queue"]))
        if not kinds and next_press is not None and now >= next_press + 30.0:
            break
        if now - t_wall0 > 3600.0:
            errors.append((round(pl.night_s(), 1), "gate", "the controls ran over an hour"))
            break
    # RE-PLAN COST while the night plays: 90 min and 6 h from the decks as they are
    cost = []
    for L in (90.0, 360.0):
        req, _com, _q, _ext = pl._req()
        req.update({"op": "replan", "length_min": L})
        req.pop("queued_id", None)
        req.pop("plan_ids", None)
        s0 = engine._cb_starved
        t0 = time.time()
        resp = d._planner_proc().request(req)
        cost.append({"length_min": L, "wall_s": round(time.time() - t0, 2), "ok": resp.get("ok"), "n": len(resp.get("slots") or []),
                     "timing": resp.get("timing"), "starved": engine._cb_starved - s0, "error": resp.get("error")})
    time.sleep(1.0)
    stats = engine.callback_stats()
    cons["stop"] = True
    th.join(timeout=2.0)
    d.stop(0.1)
    d.close_planner()
    engine._ring_stop.set()
    return {"d": d, "p": p, "pl": pl, "errors": errors, "cuts": cuts, "broken": broken, "then": then_checks, "presses": press_log,
            "stats": stats, "late_ms": cons["late_ms"], "rms": np.array(cons["rms"]), "cost": cost, "wall_s": time.time() - t_wall0,
            "first_win": 0, "starved_at": starved_at_replan, "nexts_kept": nexts_kept}


def analyse_controls(R):
    rows = []

    def add(check, value, ok, note=""):
        rows.append((check, value, ok, note))
    pl = R["pl"]
    changes = pl.changes[1:]
    add("plan changes", str(len(changes)), True, "; ".join(f"{c['trigger'][:34]}" for c in changes[:6]) + (" ..." if len(changes) > 6 else ""))
    add("triggers", f"{len(R['broken'])} broken", not R["broken"], "; ".join(R["broken"][:3]))
    ret = [x for x in pl.retimes]
    add("re-times", str(len(ret)), True, "re-anchored on the player's clock: " + ", ".join(f"{x['trigger']} {x['delta_s']:+.0f}s" for x in ret[:6]))
    # PRESSES: every BAD / UP NEXT re-planned (a change whose trigger names it)
    for j, pr in enumerate(R["presses"]):
        name = f"{pr['kind'].upper()} #{j + 1} at {pr['night_s'] / 60:.1f}'"
        if pr["kind"] == "next":
            add(name, "fired", True, f"[{pr['state']}] {pr['ret'][:60]} (re-time only)")
            continue
        word = "BAD on" if pr["kind"] == "bad" else "your UP NEXT"
        hit = [c for c in pl.changes if c["t"] >= pr["t"] - 0.5 and word in c["trigger"]]
        tags_none = pr["kind"] == "bad" and "re-scored" not in pr["ret"]
        add(name, "re-planned" if hit else "no change", bool(hit) or tags_none,
            f"[{pr['state']}] {pr['ret'][:50]}" + (f" -> {hit[0]['trigger'][:50]} in {hit[0]['timing']['wall_s']} s" if hit else ""))
    add("presses fired", str(len(R["presses"])), len(R["presses"]) >= 20 or len(R["presses"]) == int(arg("--presses", 20, int)), "")
    rp = press_replans(pl)
    add("forced re-plans", f"{len(rp)} answered", len(rp) >= REPLANS,
        f"re-plans answering a press while the ring was drained (the margin below is measured through them; target {REPLANS})")
    # UP NEXT: rejoined or re-searched; the slots skipped for your record never come back (decision 3)
    rj = [c for c in pl.changes if c.get("op") == "rejoin"]
    add("UP NEXT rejoins", f"{sum(1 for c in rj if c.get('rejoined'))} rejoined, {sum(1 for c in rj if c.get('rejoined') is False)} re-searched", True,
        f"{sum(len(c.get('skipped_user') or []) for c in rj)} slots skipped for your records (reported)")
    back = []
    for i, c in enumerate(pl.changes):
        for s in c.get("skipped_user") or []:
            for later in pl.changes[i + 1:]:
                if s in later["sent"][later["sent_fixed"]:]:
                    back.append(f"{R['d'].title(s)[:20]} (skipped at seq {c['seq']}) back at seq {later['seq']}")
                    break
    add("skipped stay out", f"{len(back)} came back", not back, "; ".join(back[:3]))
    # BAD: the slots that still fit and do not carry its tags stay - the kept order is the old one, up to the first slot
    # that breaks; none of the kept slots carries one of the BAD's tags; a stop "carries ..." is a slot that does
    bad_rows, bad_ok = [], True
    for c in pl.changes:
        if c.get("op") != "replan" or not c.get("bad_tags") or not c.get("plan_ids") or c.get("kept") is None:
            continue
        tags = set(c["bad_tags"])
        gone = set(c.get("gone") or [])
        old = [i for i in c["plan_ids"] if i not in gone]
        kept = list(c["kept"])
        new = c["sent"][c["sent_fixed"]:]
        tagged_kept = [i for i in kept if tags & words_of(R["d"].track(i))]
        stop = c.get("kept_stop") or {}
        stop_ok = not str(stop.get("why") or "").startswith("carries") or bool(tags & words_of(R["d"].track(stop.get("id"))))
        ok = kept == old[:len(kept)] and new[:len(kept)] == kept and not tagged_kept and stop_ok \
            and (stop.get("id") is None or stop.get("id") == (old[len(kept)] if len(kept) < len(old) else None)
                 or "length" in str(stop.get("why")))     # (records past the night's end are trimmed after the stop)
        bad_ok &= ok
        bad_rows.append(f"kept {len(kept)} of {len(old)}, stop: {str(stop.get('why') or 'none')[:24]}" + ("" if ok else " BROKEN"))
    add("BAD keeps", f"{len(bad_rows)} BAD re-plans", bad_ok, "; ".join(bad_rows[:5]) or "no BAD re-plan without a queue in the way")
    # SLOT +1
    add("slot +1", f"{len(R['nexts_kept'])} kept nexts", not [b for b in R["broken"] if "slot +1" in b or "kept next" in b],
        "a next record kept on its deck is always the one that plays next (no eject here)")
    # THEN rows
    bad_then = [x for x in R["then"] if not x[2]]
    add("THEN rows", f"{len(R['then']) - len(bad_then)} of {len(R['then'])}", not bad_then,
        "" if not bad_then else "; ".join(f"at {x[0]}s ({x[1][:30]}): {x[3]} vs {x[4]}" for x in bad_then[:2]))
    # RELAXED / CUTS
    played = list(pl.played)
    tr = transitions(played, R["cuts"])
    relaxed = [(a, b) for a, b, c in tr if c is not None and c.get("relaxed") and c.get("src_b") == "plan"]
    add("relaxed on plan picks", str(len(relaxed)), True, "reported (presses move the blends)" + ("; " + "; ".join(f"{a['title'][:16]} -> {b['title'][:16]}" for a, b in relaxed[:3]) if relaxed else ""))
    kinds = [(c or {}).get("kind", "?") for _a, _b, c in tr]
    add("seams", f"{len(tr)}", True, ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds))) + " (cuts here are NEXT / BAD orders)")
    srcs = [x.get("src") for x in played]
    add("records", f"{len(played)}", len(played) >= 5, ", ".join(f"{s} {srcs.count(s)}" for s in sorted(set(map(str, srcs)))))
    # UNDERRUNS
    st = R["stats"]
    add("underruns", f"{st['starved']} of {st['callbacks']}", st["starved"] == 0, f"{st['short_frames']} frames lost; the device stand-in's worst lateness {R['late_ms']:.0f} ms")
    add("callback margin", f"{st['min_depth_ms']:.0f} ms", st["min_depth_ms"] >= 100.0, "the ring's lowest depth through the presses and re-plans (target 400 ms)")
    # RE-PLAN COST
    walls = [c["timing"]["wall_s"] for c in pl.changes if c.get("timing")]
    if walls:
        add("re-plan answers", f"median {sorted(walls)[len(walls) // 2]:.1f} s, max {max(walls):.1f} s", True, f"{len(walls)} answers (wall, sent -> received)")
    for c in R["cost"]:
        tm = c.get("timing") or {}
        add(f"re-plan {c['length_min'] / 60:.1f} h", f"{c['wall_s']:.1f} s wall", bool(c.get("ok")) and c["starved"] == 0,
            f"search {tm.get('search_s')} s, table {tm.get('table_s')} s, total {tm.get('total_s')} s; {c['n']} slots; "
            f"{c['starved']} underruns while it ran" + (f"; {c['error']}" if c.get("error") else ""))
    dead = dead_air(R["rms"], 300)
    add("dead air", f"{len(dead)} gaps", not dead, "; ".join(f"{t:.1f}s for {l:.1f}s" for t, l in dead[:4]))
    for t, who, text in R["errors"]:
        add("error", f"{who} at {t:.0f}s", False, text[:140])
    if not R["errors"]:
        add("errors", "none", True, "Director, player and plan errors stayed None")
    add("run", f"{R['wall_s'] / 60:.1f} min wall", True, "")
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    seeds = seeds_of(arg("--seeds", "1-2"))
    preset = arg("--preset", "Desert Night")
    minutes = arg("--minutes", 90.0, float)
    pace = arg("--pace", 8.0, float)
    presses = arg("--presses", 20, int)
    log_dir = arg("--log-dir", None) or tempfile.mkdtemp(prefix="dj_plan_gate_")
    print(f"the plan gate: {preset}, {minutes:.0f} min, seeds {seeds}, log dir {log_dir}", flush=True)
    all_rows = []
    for s in seeds:
        parts = []
        if "--skip-night" not in sys.argv:
            parts.append(("night", run_night, analyse_night, (music, s, preset, minutes, pace, log_dir)))
        if "--skip-refusal" not in sys.argv:
            parts.append(("refusal", run_night, analyse_refusal, (music, s, preset, minutes, pace, log_dir, 1)))
            # (its rows read "A ...": the alternate refused too)
            parts.append(("alternate refused", run_night, analyse_refusal, (music, s, preset, minutes, pace, log_dir, 2)))
            # (rows "E ...": every alternate refused, the brain's gap)
            parts.append(("every alternate refused", run_night, analyse_refusal, (music, s, preset, minutes, pace, log_dir, 3)))
            # (rows "F ...": fewer alternates than probes, all refused - "nor its alternates" against the real player)
            parts.append(("fewer alternates", run_night, analyse_refusal, (music, s, preset, minutes, pace, log_dir, 4)))
        if "--skip-controls" not in sys.argv:
            parts.append(("controls", run_controls, analyse_controls, (music, s, preset, presses, log_dir)))
        for label, fn, an, a in parts:
            print(f"== seed {s}: {label}", flush=True)
            try:
                R = fn(*a)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                all_rows.append((s, (f"{label} run", "EXCEPTION", False, f"{type(e).__name__}: {e}")))
                continue
            if "fatal" in R:
                print(R["fatal"])
                all_rows.append((s, (f"{label} run", "fatal", False, R["fatal"])))
                continue
            try:
                rows = an(R)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                rows = [(f"{label} analysis", "EXCEPTION", False, f"{type(e).__name__}: {e}")]
            all_rows += [(s, (f"{label[0].upper()} {r[0]}",) + tuple(r[1:])) for r in rows]
            pl = R["pl"]
            print(f"\n-- seed {s} {label}: what played --")
            for x in pl.played:
                print(f"  {(x.get('start_s') or 0) / 60:6.1f}'  [{x.get('src')}] {str(x.get('title'))[:40]}"
                      + (f"  (planned at {x['plan_start_s'] / 60:.1f}')" if x.get("plan_start_s") is not None else "") + (f"  - {x['why']}" if x.get("why") else ""))
            if "--trail" in sys.argv:
                print(f"-- seed {s} {label}: the Director's plan words and the player's picks --")
                for h, m in R["d"].events:
                    if "plan" in m or "night" in m:
                        print(f"  {h}  {m[:150]}")
                for o in R["pl"].offers:
                    print(f"  offer at {o['night_s'] / 60:.1f}': after {o['after'][:20]} the slot {str(o['slot'])[:20]} -> {o['given'][:20]} (probe {o['probe']})")
                for m in R["p"].moves:
                    if m.get("kind") in ("pick", "decoding", "open", "swap"):
                        print(f"  {time.strftime('%H:%M:%S', time.localtime(m['t']))}  [{m['kind']}] {m['text'][:140]}")
            print(f"-- seed {s} {label}: the plan's changes --")
            for c in pl.changes:
                print(f"  {c['night_s'] / 60:6.1f}'  [{c['op']}] {c['trigger'][:90]}  ({len(c['after'])} ahead, {len(c['skipped'])} out, {c['timing']['wall_s']} s)")
            del R
    print(f"\n{'seed':>4s} {'check':26s} {'value':32s} {'':5s} note")
    bad = 0
    for s, (check, value, ok, note) in all_rows:
        bad += 0 if ok else 1
        print(f"{s:4d} {check[:26]:26s} {str(value)[:32]:32s} {'PASS' if ok else 'FAIL':5s} {note}")
    try:
        with open(os.path.join(log_dir, "dj_plan.jsonl"), encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        print(f"\n{n} set_plan / set_skip rows in {os.path.join(log_dir, 'dj_plan.jsonl')}")
    except OSError:
        pass
    print("\nALL OK" if bad == 0 else f"\nFAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
