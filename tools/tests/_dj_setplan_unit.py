"""THE SET PLANNER, unit-checked on the real library (lib/dj/setplanner.py, docs/DJ_SETS.md phase 1).

Plans every preset at 45 / 90 / 180 minutes (in-process: no audio, no Director) and checks the plan against the
record player itself:

  NO REPEATS    no record twice, no two versions of one song
  SPACING       no artist twice within the theme's spacing (6 records)
  THE VETO      every planned seam: tempo within MAX_TEMPO_GAP, keys at or over KEY_BLEND (RecordPlayer.key_fit),
                and THE PLAYER'S STRICT PLACEMENT takes it (the real _find_blend_strict on a probe player whose
                telemetry says the record is where the plan says) - no planned record is ever "picked again"
  THE CLOCK     on that probe the real _plan_breakdown / _plan_blend start the next record at the song time and
                from the cue the plan says (the planner's clock is the player's clock); starts add up
  CHAPTERS      chapter indices never go back
  LENGTH        the night ends within one record of its length (the last record's own play)
  ALTERNATES    2-3 per planned slot, each fitting both neighbours by the same strict placement; a THIN crate (under
                20 blendable pairs per record) may have fewer, but only said (the plan's warning) and only where no
                crate record left out passes both seams by the real player
  RE-PLAN       a re-plan from a state (played, the record playing + slot +1 fixed, an exclusion, BAD's tags) keeps
                the fixed prefix as it is and never brings back what played or was excluded
  BAD           a re-plan with the previous plan's slots and BAD's tags keeps exactly the slots before the first one
                that carries a tag, in order, and searches on from there
  REJOIN        after an UP NEXT the plan rejoins at the first slot that connects (the skipped ones listed, never
                back), or re-searches from the user's record when none does
  LANDED PAST   (2026-10-29) Prophecy Theme / Neon Dawn: the player's cut lands past their silent start; a re-plan told
                the moved cue and the phrase origin times the record's way out as the player does
  SUBPROCESS    the same request through PlannerProcess gives the same night
  VOCALS        the stems' envelope as the planner reads it agrees with the player's vocal_envelope on sampled records

and prints the plan time at 90 minutes and 6 hours.

    python tools/tests/_dj_setplan_unit.py --music D:/Devel/music [--presets all|"Desert Night,Deep Space"] [--quick]

Exit 0 = ALL OK, 1 = FAIL.
"""
import os
import random
import sys
import threading
import time

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
os.environ["DJ_DECODE_INPROC"] = "1"           # (the vocal check decodes a few stems here; no helper pool)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.tests._low_priority import PRIORITY  # noqa: E402,F401  (idle priority: it may run beside the planner)


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


FAILS = []


def check(ok, what):
    if not ok:
        FAILS.append(what)
        print("  FAIL", what, flush=True)
    return ok


class _Tel:
    """The probe player's submix: telemetry saying deck a is at `time_s`, nothing else (no audio)."""
    def __init__(self):
        self.telemetry = {"clock": 0, "decks": {}}
        self.clock = 0

    def post(self, *_a, **_k):
        pass

    def post_many(self, *_a, **_k):
        pass


def probe_player(play_min):
    from lib.dj.recordplay import RecordPlayer
    rp = RecordPlayer.__new__(RecordPlayer)
    rp.play_min_s = float(play_min)
    rp.songs, rp.record, rp.submix = {}, None, _Tel()
    rp.cut, rp._seq, rp.moves, rp._lock = None, 0, [], threading.Lock()
    rp._next_asap, rp.wait_why, rp._settled_clock, rp.hole_beats = False, "", 0, 1.0
    return rp


def player_seam(rp, fa, cue_a, fb, phrase=None):
    """What the REAL player does with record b after record a opened at cue_a, planned PLAN_AT_S into a: (strict
    placement found?, cut dict) - _plan_breakdown first, then _plan_blend, as RecordPlayer.step does. `phrase`: where a's
    phrases count from when a cut landed it past a silent start (Rec.phrase_t)."""
    from lib.dj.recordplay import Rec
    from lib.dj.setplanner import PLAN_AT_S
    t_now = min(cue_a + PLAN_AT_S, max(cue_a, fa.dur - 1.0))
    ra = Rec("a", fa.t)
    ra.vox, ra.cue_t, ra.state, ra.phrase_t = fa.vox, cue_a, "playing", phrase
    rb = Rec("b", fb.t)
    rb.vox, rb.state = fb.vox, "ready"
    rp.submix.telemetry = {"clock": 0, "decks": {"a": {"time_s": t_now, "rate": 1.0}}}
    rp.songs, rp.record, rp.cut = {"a": ra, "b": rb}, "a", None
    ra.open_bar = rp._bar_index(fa.t, cue_a if phrase is None else phrase)
    strict = rp._find_blend_strict(ra, rb) is not None
    if not rp._plan_breakdown(rb):
        rp._plan_blend(rb, why="probe")
    return strict, rp.cut


def ak_of(f):
    return f.ak


def check_plan(P, rp, resp, L_s, label, n_fixed=0, check_clock=True, spacing=6, banned=()):
    from lib.dj.recordplay import KEY_BLEND, MAX_TEMPO_GAP
    F = P.facts
    slots = resp["slots"]
    ok = True
    ok &= check(resp.get("ok") and len(slots) >= 2, f"{label}: a plan ({resp.get('error')})")
    if len(slots) < 2:
        return False
    ids = [s["id"] for s in slots]
    ok &= check(len(set(ids)) == len(ids), f"{label}: no repeats")
    roots = [F[i].root for i in ids]
    ok &= check(len(set(roots)) == len(roots), f"{label}: one version of a song")
    aks = [F[i].ak for i in ids]
    bad_sp = [(k, aks[k]) for k in range(len(aks)) if aks[k] and aks[k] in aks[max(0, k - spacing):k] and k >= n_fixed]
    ok &= check(not bad_sp, f"{label}: artist spacing {bad_sp[:3]}")
    chs = [s["chapter"] for s in slots]
    ok &= check(all(chs[k] <= chs[k + 1] for k in range(len(chs) - 1)), f"{label}: chapters in order {chs}")
    for k in range(len(slots) - 1):
        if abs(slots[k]["start_s"] + slots[k]["play_s"] - slots[k + 1]["start_s"]) > 0.2:
            ok &= check(False, f"{label}: starts add up at slot {k}")
            break
    end, last = slots[-1]["end_s"], slots[-1]["play_s"]
    ok &= check(resp.get("finished") and L_s - 1.0 <= end <= L_s + last + 1.0,
                f"{label}: length {end / 60:.1f} min for {L_s / 60:.0f} (last record {last / 60:.1f} min)")
    n_clock = n_bad_clock = 0
    relaxed_said = any("relaxed blend" in w["text"] for w in resp.get("warnings") or [])
    end_said = any("plays to its end" in w["text"] for w in resp.get("warnings") or [])
    for k in range(max(1, n_fixed), len(slots)):
        a, b = F[ids[k - 1]], F[ids[k]]
        sm = slots[k]["in"]
        fit = RecordPlayerKeyFit(a, b)
        gap = abs(a.bpm / b.bpm - 1.0)
        # the one relaxed seam allowed: right after the fixed records, when nothing blends strictly after them (and said)
        relaxed_ok = k == n_fixed and n_fixed > 0 and ((sm["kind"] == "relaxed" and relaxed_said) or (sm["kind"] == "end" and end_said))
        if not check(gap <= MAX_TEMPO_GAP + 1e-9 and (fit is None or fit >= KEY_BLEND) and (sm["kind"] in ("blend", "breakdown") or relaxed_ok),
                     f"{label}: seam {k} {a.title[:20]} -> {b.title[:20]} veto (gap {gap:.3f} fit {fit} kind {sm['kind']})"):
            ok = False
            continue
        if not check_clock:
            continue
        strict, cut = player_seam(rp, a, slots[k - 1]["cue_s"], b)
        if relaxed_ok:
            ok &= check(not strict, f"{label}: seam {k} {a.title[:20]} -> {b.title[:20]} planned relaxed although the player places it strictly")
        else:
            ok &= check(strict, f"{label}: seam {k} {a.title[:20]} -> {b.title[:20]}: the player's strict placement refuses it")
        n_clock += 1
        want = {"relaxed": "blend", "end": "cut"}.get(sm["kind"], sm["kind"])   # (the player's names for its fallbacks)
        if cut is None or cut["kind"] != want or abs(cut["song_t"] - sm["blend_at_s"]) > 0.06 or abs(cut["cue_b"] - sm["cue_s"]) > 0.006:
            n_bad_clock += 1
            ok &= check(False, f"{label}: seam {k} {a.title[:20]} -> {b.title[:20]}: the player's clock "
                               f"{None if cut is None else (cut['kind'], round(cut['song_t'], 1), round(cut['cue_b'], 2))} vs the plan's "
                               f"{(sm['kind'], sm['blend_at_s'], sm['cue_s'])}")
    # alternates: 2-3 on planned slots, each fitting both neighbours by the player's own strict placement
    few = 0
    for k in range(n_fixed, len(slots)):
        alts = slots[k]["alts"]
        if len(alts) < 2:
            few += 1
        for al in alts[:2]:
            c = F[al["id"]]
            if c.id in ids:
                ok &= check(False, f"{label}: alternate {c.title[:20]} is already in the plan")
                continue
            if k > 0 and check_clock:
                strict_in, cut_in = player_seam(rp, F[ids[k - 1]], slots[k - 1]["cue_s"], c)
                fallback = (slots[k]["in"] or {}).get("kind") in ("relaxed", "end")     # (the slot itself comes in on the player's fallback)
                ok &= check(strict_in or fallback, f"{label}: alternate {c.title[:20]} for slot {k} does not blend after {F[ids[k - 1]].title[:20]}")
                if k + 1 < len(slots) and cut_in is not None:
                    strict_out, _c = player_seam(rp, c, cut_in["cue_b"], F[ids[k + 1]])
                    ok &= check(strict_out, f"{label}: alternate {c.title[:20]} for slot {k} does not blend into {F[ids[k + 1]].title[:20]}")
    # the alternates are every crate record that passes both seams, so a slot with fewer than two is the crate's limit.
    # A night of up to 3 h must have them on all but ALTS_FEW_SHARE of its slots - UNLESS the crate is thin (under
    # THIN_PAIRS blendable pairs per record; 2026-10-29: Slow Morning holds 311 records but 13 pairs each, the other
    # presets 70-90): then the plan must SAY so (its thin-crate warning) and the shortfall must be real - no crate record
    # left out passes both seams by the real player's strict placement. A 6 h night that has used up much of its crate is
    # only reported.
    n_planned = max(1, len(slots) - n_fixed)
    per = (resp.get("pairs") or 0) / max(1, resp.get("crate") or 1)
    if check_clock:
        from lib.dj.setplanner import ALTS_FEW_SHARE
        if few > ALTS_FEW_SHARE * n_planned:
            said = any(w.get("alts") is not None for w in resp.get("warnings") or [])
            ok &= check(per < THIN_PAIRS and said, f"{label}: {few} of {n_planned} planned slots with fewer than 2 alternates "
                                                   f"({per:.0f} pairs per record; thin-crate warning {'given' if said else 'MISSING'})")
            missed = missed_alternates(P, rp, resp, n_fixed, spacing, banned)
            ok &= check(not missed, f"{label}: alternates the planner missed: " + "; ".join(missed[:3]))
            print(f"  ({label}: {few} of {n_planned} planned slots with fewer than 2 alternates - a thin crate, {per:.0f} pairs per record: "
                  f"said, and every crate record checked by the player: {'none' if not missed else len(missed)} missed)", flush=True)
    elif few:
        print(f"  ({label}: {few} of {n_planned} planned slots with fewer than 2 alternates - the crate is used up)")
    return ok, n_clock, n_bad_clock


THIN_PAIRS = 20.0         # blendable pairs per crate record under which a crate is thin (Slow Morning 13; the others 70-90)


def missed_alternates(P, rp, resp, n_fixed, spacing, banned=()):
    """Every crate record the planner did NOT list as an alternate for a slot with fewer than two, checked with the REAL
    player: within the tempo limit and the key veto of both neighbours, the variety rules kept, and both seams placed by
    _find_blend_strict. Any that passes is an alternate the planner missed."""
    from lib.dj.recordplay import KEY_BLEND, MAX_TEMPO_GAP
    F = P.facts
    slots = resp["slots"]
    ids = [s["id"] for s in slots]
    crate = [int(i) for i in resp.get("crate_ids") or []]
    out = []

    def vetoed(a, b):
        fit = RecordPlayerKeyFit(a, b)
        return abs(a.bpm / b.bpm - 1.0) > MAX_TEMPO_GAP + 1e-9 or (fit is not None and fit < KEY_BLEND)
    for k in range(max(1, n_fixed), len(slots)):
        if len(slots[k]["alts"]) >= 2 or (slots[k]["in"] or {}).get("kind") in ("relaxed", "end"):
            continue
        prev, nxt = F[ids[k - 1]], (F[ids[k + 1]] if k + 1 < len(ids) else None)
        listed = {a["id"] for a in slots[k]["alts"]}
        roots = {F[i].root for j, i in enumerate(ids) if j != k}
        near = {F[i].ak for i in ids[max(0, k - spacing):k] + ids[k + 1:k + 1 + spacing] if F[i].ak}
        for j in crate:
            c = F.get(j)
            if c is None or j in ids or j in listed or j in banned or c.root in roots or (c.ak and c.ak in near):
                continue
            if vetoed(prev, c) or (nxt is not None and vetoed(c, nxt)):
                continue
            strict_in, cut_in = player_seam(rp, prev, slots[k - 1]["cue_s"], c)
            if not strict_in or cut_in is None:
                continue
            if nxt is not None and not player_seam(rp, c, cut_in["cue_b"], nxt)[0]:
                continue
            out.append(f"slot {k}: {c.title[:20]} fits between {prev.title[:16]} and {nxt.title[:16] if nxt else '(the end)'}")
    return out


def RecordPlayerKeyFit(a, b):
    from lib.dj.recordplay import RecordPlayer
    return RecordPlayer.key_fit(a.t, b.t)


def main():
    music = arg("--music", "D:/Devel/music")
    quick = "--quick" in sys.argv
    from lib.dj import setplanner as SP
    names = SP.preset_names()
    sel = arg("--presets", "all")
    if sel != "all":
        names = [n.strip() for n in sel.split(",") if n.strip()]
    lengths = (45, 90) if quick else (45, 90, 180)
    t0 = time.time()
    P = SP.Planner(music, log=print)
    P._load()
    pm = SP.DEFAULT_PLAY_MIN_S
    rp = probe_player(pm)
    F = P.facts
    print(f"library: {len(F)} records, loaded in {P.load_s:.1f} s", flush=True)

    # ---- every preset at every length ------------------------------------------------------------------------------
    plans = {}
    times = []
    clock_n = clock_bad = 0
    for name in names:
        for L in lengths:
            r = P.handle({"op": "plan", "preset": name, "length_min": L})
            tm = r.get("timing") or {}
            n = len(r.get("slots") or [])
            print(f"{name:13s} {L:4d} min: {n:3d} records, crate {r.get('crate')}, pairs {r.get('pairs')}, ends {(r.get('end_s') or 0) / 60:6.1f} min, "
                  f"table {tm.get('table_s', 0):.2f} s, search {tm.get('search_s', 0):.2f} s, warnings {len(r.get('warnings') or [])}"
                  f"{'' if r.get('ok') else '  ERROR ' + str(r.get('error'))}", flush=True)
            res = check_plan(P, rp, r, L * 60.0, f"{name} {L}")
            if isinstance(res, tuple):
                clock_n += res[1]
                clock_bad += res[2]
            plans[(name, L)] = r
            times.append((name, L, tm.get("table_s", 0), tm.get("search_s", 0)))
    print(f"the player's clock: {clock_n - clock_bad} of {clock_n} planned seams start where the real player starts them", flush=True)

    # ---- the plan time at 90 minutes and 6 hours -------------------------------------------------------------------------
    for name in names[:2] if quick else names:
        r = P.handle({"op": "plan", "preset": name, "length_min": 360})
        tm = r.get("timing") or {}
        print(f"PLAN TIME {name:13s}  90 min: search {next((t[3] for t in times if t[0] == name and t[1] == 90), float('nan')):.2f} s | "
              f"6 h: search {tm.get('search_s', 0):.2f} s, {len(r.get('slots') or [])} records, ends {(r.get('end_s') or 0) / 3600:.2f} h", flush=True)
        check_plan(P, rp, r, 360 * 60.0, f"{name} 360", check_clock=False)

    # ---- re-plan from a state ----------------------------------------------------------------------------------------------
    base = plans[(names[0], 90)]
    s = base["slots"]
    played = [s[0]["id"]]
    fixed = [{"id": s[1]["id"], "cue_s": s[1]["cue_s"], "start_s": s[1]["start_s"]}, {"id": s[2]["id"]}]
    excl = [s[4]["id"]]
    bad_tags = sorted(F[s[5]["id"]].words)[:4]
    r = P.handle({"op": "replan", "preset": names[0], "length_min": 90, "played_ids": played, "fixed": fixed,
                  "exclude_ids": excl, "neg_leans": {w: 0.5 for w in bad_tags}, "seed": 7})
    ids = [x["id"] for x in r.get("slots") or []]
    check(r.get("ok"), f"replan: {r.get('error')}")
    check(ids[:2] == [s[1]["id"], s[2]["id"]] and r["slots"][0]["fixed"] and r["slots"][1]["fixed"] and not r["slots"][2]["fixed"],
          f"replan keeps the fixed prefix: {ids[:3]}")
    check(abs(r["slots"][0]["start_s"] - s[1]["start_s"]) < 0.2 and abs(r["slots"][1]["start_s"] - s[2]["start_s"]) < 0.2,
          "replan keeps the night's clock at the fixed records")
    check(not (set(played) | set(excl)) & set(ids), "replan never brings back what played or was excluded")
    check(r["slots"][0]["alts"] == [] and r["slots"][1]["alts"] == [], "fixed slots carry no alternates")
    check_plan(P, rp, r, 90 * 60.0, "replan", n_fixed=2, banned=set(played) | set(excl))
    # ---- BAD: the previous plan's slots that still fit and carry none of its tags stay (DJ_SETS.md section 3) ------------
    old_future = [x["id"] for x in r["slots"][2:]]
    fixed2 = [{"id": r["slots"][0]["id"], "cue_s": r["slots"][0]["cue_s"], "start_s": r["slots"][0]["start_s"]}, {"id": r["slots"][1]["id"]}]
    # tags carried by the 4th slot of the old future and not by the three before it: the kept order must stop there
    stop_k = 3
    tagged = sorted(F[old_future[stop_k]].words - set().union(*(F[i].words for i in old_future[:stop_k])))[:2] if len(old_future) > stop_k else []
    if tagged:
        rb = P.handle({"op": "replan", "preset": names[0], "length_min": 90, "played_ids": played, "fixed": fixed2, "exclude_ids": excl,
                       "neg_leans": {w: 1.0 for w in tagged}, "bad_tags": tagged, "plan_ids": old_future, "seed": 7})
        nb = [x["id"] for x in rb.get("slots") or []]
        kept = rb.get("kept_ids") or []
        check(rb.get("ok") and nb[:2] == [fixed2[0]["id"], fixed2[1]["id"]], f"BAD replan keeps the fixed prefix ({rb.get('error')})")
        check(kept == old_future[:stop_k] and nb[2:2 + stop_k] == old_future[:stop_k],
              f"BAD keeps exactly the slots before the first that carries {tagged}: kept {kept} of {old_future[:stop_k + 1]}")
        check((rb.get("kept_stop") or {}).get("id") == old_future[stop_k] and old_future[stop_k] not in nb[2:],
              f"BAD stops at the tagged slot and re-searches from there ({rb.get('kept_stop')})")
        check_plan(P, rp, rb, 90 * 60.0, "BAD replan", n_fixed=2, banned=set(played) | set(excl))
        print(f"BAD replan: kept {len(kept)} slots, stopped at {(rb.get('kept_stop') or {}).get('why')}, {len(nb)} records", flush=True)
    print(f"replan: {len(ids)} records, {ids[:2]} kept, search {(r.get('timing') or {}).get('search_s')} s", flush=True)

    # ---- rejoin after an UP NEXT ---------------------------------------------------------------------------------------------
    rnd = random.Random(5)
    plan_ids = [x["id"] for x in s[2:]]
    fixed = [{"id": s[0]["id"], "cue_s": s[0]["cue_s"], "start_s": s[0]["start_s"]}, {"id": s[1]["id"]}]
    in_plan = {x["id"] for x in s}
    pool = [f for f in F.values() if f.id not in in_plan and abs(f.bpm / F[s[2]["id"]].bpm - 1) < 0.05 and P._geo_of(pm, f).why is None]
    rnd.shuffle(pool)
    # a record too short to be a record: the player refuses it, the plan goes on without it
    short = next((f for f in F.values() if f.id not in in_plan and P._geo_of(pm, f).why is not None), None)
    if short is not None:
        r = P.handle({"op": "rejoin", "preset": names[0], "length_min": 90, "fixed": fixed, "queued_id": short.id, "plan_ids": plan_ids})
        check(r.get("ok") and r.get("queued_refused") and short.id not in [x["id"] for x in r["slots"]]
              and [x["id"] for x in r["slots"]][:2] == [s[0]["id"], s[1]["id"]], f"rejoin refuses {short.title[:20]} (it cannot be the record)")
    got_rejoin = got_research = got_skip = False
    for q in pool[:80]:
        r = P.handle({"op": "rejoin", "preset": names[0], "length_min": 90, "fixed": fixed, "queued_id": q.id, "plan_ids": plan_ids})
        if not check(r.get("ok"), f"rejoin {q.title[:20]}: {r.get('error')}"):
            break
        ids = [x["id"] for x in r["slots"]]
        check(ids[:3] == [s[0]["id"], s[1]["id"], q.id] and r["slots"][2]["user"] and r["slots"][2]["fixed"], f"rejoin keeps the prefix and the user's record: {ids[:3]}")
        check(len(set(ids)) == len(ids), "rejoin: no repeats")
        if r["rejoined"]:
            at = r["rejoined_at"]
            skipped = [x["id"] for x in r["skipped"]]
            check(ids[3] == plan_ids[at], f"rejoin at the first slot that connects ({ids[3]} vs {plan_ids[at]})")
            check(skipped == plan_ids[:at] and all(x["why"] == SP.SKIP_WHY for x in r["skipped"]), "rejoin lists the skipped slots")
            check(not set(skipped) & set(ids), "skipped slots do not come back")
            # nothing earlier connects: each skipped slot fails the strict seam from the user's record
            for sid in skipped:
                sm = P.handle({"op": "seam", "a": q.id, "b": sid, "cue_s": r["slots"][2]["cue_s"], "play_min_s": pm})
                check(sm.get("seam") is None, f"rejoin skipped {F[sid].title[:20]} although it connects")
            # the kept chain is the old plan's order
            kept = [i for i in ids[3:] if i in plan_ids]
            check(kept == [i for i in plan_ids if i in kept], "rejoin keeps the plan's order")
            check_plan(P, rp, r, 90 * 60.0, f"rejoin {q.title[:16]}", n_fixed=3)
            if not got_rejoin or (skipped and not got_skip):
                print(f"rejoin: {q.title[:30]} rejoins at slot {at + 2} ({len(skipped)} skipped), {len(ids)} records, ends {r['end_s'] / 60:.1f} min", flush=True)
            got_rejoin = True
            got_skip = got_skip or bool(skipped)
        else:
            if not got_research or not r["finished"]:
                print(f"rejoin: {q.title[:30]} connects to no slot: the rest re-searched, {len(ids)} records"
                      + ("" if r["finished"] else f" (a dead end: {r['warnings'][0]['text'][:80]})"), flush=True)
            if r["finished"]:
                check_plan(P, rp, r, 90 * 60.0, f"rejoin-research {q.title[:16]}", n_fixed=3)
                got_research = True
            else:
                # nothing strict follows the user's record at all: the prefix stands, the warning says the DJ picks next
                check(len(ids) == 3 and any(w["bad"] for w in r["warnings"]), f"rejoin dead end after {q.title[:20]}: prefix kept and said")
        if got_rejoin and got_research and got_skip:
            break
    if not got_research:
        # a record far from every planned tempo connects to nothing: the rest is searched from it
        far = max((f for f in F.values() if f.id not in in_plan and P._geo_of(pm, f).why is None),
                  key=lambda f: min(abs(f.bpm / F[i].bpm - 1) for i in plan_ids))
        r = P.handle({"op": "rejoin", "preset": names[0], "length_min": 90, "fixed": fixed, "queued_id": far.id, "plan_ids": plan_ids})
        check(r.get("ok") and not r["rejoined"] and [x["id"] for x in r["slots"]][:3] == [s[0]["id"], s[1]["id"], far.id],
              f"rejoin from a record that connects to nothing ({far.title[:20]}, {far.bpm:.0f} bpm) re-searches from it")
        got_research = True
        print(f"rejoin: {far.title[:30]} ({far.bpm:.0f} bpm) connects to no slot: the rest re-searched, {len(r.get('slots') or [])} records", flush=True)
    # an UP NEXT record with no steady phrase after its minimum play (nothing blends after it by the rules): the plan takes
    # the player's own fallback for that one seam, says so, and is strict again after it
    tb = next(iter(P._tables.values()))
    stuck = [f for f in pool if not P._row(tb, f, {"play_min": pm})]
    for q in stuck[:2]:
        r = P.handle({"op": "rejoin", "preset": names[0], "length_min": 90, "fixed": fixed, "queued_id": q.id, "plan_ids": plan_ids})
        k3 = (r["slots"][3]["in"] or {}).get("kind") if len(r.get("slots") or []) > 3 else None
        if r.get("ok") and not r["finished"] and len(r["slots"]) == 3:
            # a true dead end: not even the player's fallbacks reach a crate record (tempo or keys) - said, nothing invented
            near = {F[x["id"]].ak for x in s[:2]} | {q.ak}
            loose = [j for j, _s in P._loose_row(tb, q) if j not in {x["id"] for x in s[:2]} and F[j].ak not in near and F[j].root != q.root]
            check(any("Nothing in the crate blends after" in w["text"] for w in r["warnings"]) and not loose,
                  f"after {q.title[:20]} the dead end is real and said ({len(loose)} records within tempo and keys)")
            print(f"rejoin: after {q.title[:30]} nothing in the crate is within tempo and keys: the DJ picks next (said)", flush=True)
            continue
        check(r.get("ok") and r["finished"] and k3 in ("relaxed", "end", "blend", "breakdown")
              and (k3 in ("blend", "breakdown") or any(("relaxed blend" in w["text"] or "plays to its end" in w["text"]) for w in r["warnings"])),
              f"after {q.title[:20]} (nothing strict follows it) the plan goes on ({k3})")
        check_plan(P, rp, r, 90 * 60.0, f"stuck {q.title[:16]}", n_fixed=3)
        print(f"rejoin: after {q.title[:30]} nothing blends by the rules: the next comes in by '{k3}', {len(r['slots'])} records", flush=True)
    check(got_rejoin, "rejoin: some UP NEXT record rejoined the plan")
    check(got_skip, "rejoin: some UP NEXT record rejoined past a slot it could not blend into")

    # ---- A CUT LANDED PAST A SILENT START (the user's pick, 2026-10-29) --------------------------------------------------------
    # Prophecy Theme and Neon Dawn open near-silent at their landmark: the player's cut lands them on the first downbeat it
    # hears (cut_landing, on the mix envelope it measures at decode) and counts their phrases from the landmark. A re-plan
    # from one of them as the record playing (the Director sends the cue and the phrase origin) times its way out as the
    # player does. (The planner itself has no mix envelope: a seam IT plans to its end lands at the landmark - the player
    # lands later and the next slot's label still counts as placed from that record, NightPlan._placed_here.)
    from lib.dj.recordplay import Rec, mix_envelope
    from lib.dj.stemload import decode_mix
    for tid in (355, 73):
        f = F.get(tid)
        if f is None or P._geo_of(pm, f).why is not None:
            print(f"landed past: #{tid} is not a record in this library at play_min {pm:.0f} (skipped)", flush=True)
            continue
        r0 = Rec("b", f.t)
        r0.mix_env = mix_envelope(decode_mix(P._db.abs(f.t.path)), f.t.gain_db)
        land, lm0 = rp.cut_landing(r0)
        check(land > lm0 + 0.5, f"landed past: {f.title[:24]} lands past its silent start ({lm0:.2f} -> {land:.2f} s)")
        if land <= lm0:
            continue
        rr = P.handle({"op": "replan", "preset": names[0], "length_min": 45, "seed": 3,
                       "fixed": [{"id": tid, "cue_s": land, "phrase_s": lm0, "start_s": 0.0}]})
        if not rr.get("ok") or len(rr.get("slots") or []) < 2:
            print(f"landed past: nothing in {names[0]}'s crate follows {f.title[:24]} ({rr.get('error')}) - its way out not compared", flush=True)
            continue
        nb_, sm = F[rr["slots"][1]["id"]], rr["slots"][1]["in"]
        _strict, cut = player_seam(rp, f, land, nb_, phrase=lm0)
        check(cut is not None and abs(cut["song_t"] - sm["blend_at_s"]) <= 0.06 and abs(cut["cue_b"] - sm["cue_s"]) <= 0.006,
              f"landed past: {f.title[:24]} -> {nb_.title[:20]}: the planner times its way out as the player does "
              f"({None if cut is None else (cut['kind'], round(cut['song_t'], 2), round(cut['cue_b'], 2))} vs {(sm['kind'], sm['blend_at_s'], sm['cue_s'])})")
        r_np = P.handle({"op": "replan", "preset": names[0], "length_min": 45, "seed": 3,
                         "fixed": [{"id": tid, "cue_s": land, "start_s": 0.0}]})
        alt = ((r_np.get("slots") or [None, None])[1] or {}).get("in") or {}
        print(f"landed past: {f.title[:24]} cue {lm0:.2f} -> {land:.2f} s; its way out at {sm['blend_at_s']} s with the phrase origin, "
              f"{alt.get('blend_at_s')} s if the planner counted from the moved cue", flush=True)

    # ---- the subprocess gives the same night ----------------------------------------------------------------------------------
    req = {"op": "plan", "preset": names[0], "length_min": 45}
    pp = SP.PlannerProcess(music, log=print)
    t1 = time.time()
    a = pp.request({"op": "ping"}, timeout=120)
    b = pp.request(req, timeout=600)
    t_sub = time.time() - t1
    check(a.get("ok") and b.get("ok"), f"subprocess answers ({a.get('error')} / {b.get('error')})")
    # the same library, the same night: compared on equal library signatures (a keys rescan running beside this test
    # rewrites the DB between the two loads - then both read again, a few times)
    same = None
    for _try in range(4):
        mine = P.handle(dict(req, reload=True))
        if mine.get("lib_sig") == b.get("lib_sig"):
            same = [x["id"] for x in b.get("slots") or []] == [x["id"] for x in mine.get("slots") or []]
            break
        b = pp.request(dict(req, reload=True), timeout=600)
    pp.close()
    if same is None:
        print("subprocess: the library kept changing under the test (a rescan writing the DB?): the comparison was not possible", flush=True)
    else:
        check(same, "the subprocess plans the same night as in-process on the same library")
    print(f"subprocess: first answer in {t_sub:.1f} s (cold child: library load + pair table), {len(b.get('slots') or [])} records", flush=True)

    # ---- the vocal envelope the planner reads vs the player's ---------------------------------------------------------------
    if not quick:
        import numpy as np
        from lib.dj.recordplay import vocal_envelope, mix_level, VOX_QUIET
        from lib.dj.stemload import decode_mix
        sample = [f for f in F.values() if f.vox is not None and f.vox_exact]
        random.Random(11).shuffle(sample)
        agree = 0
        for f in sample[:6]:
            real = vocal_envelope(music, f.id, mix_rms=mix_level(decode_mix(P._db.abs(f.t.path))))
            if real is None:
                continue
            n = min(len(real), len(f.vox))
            frac = float(np.mean((np.asarray(real[:n], dtype=np.float32) >= VOX_QUIET) == (f.vox[:n] >= VOX_QUIET)))
            agree += frac >= 0.98
            check(frac >= 0.98, f"vocals: {f.title[:24]} sings where the player says it does ({frac:.3f})")
        print(f"vocals: {agree} of {min(6, len(sample))} sampled records agree with vocal_envelope; "
              f"{sum(1 for f in F.values() if not f.vox_exact)} vocal floors still approximate (run --warm-vox)", flush=True)

    print(f"\n{len(FAILS)} failures in {time.time() - t0:.0f} s")
    if FAILS:
        for x in FAILS[:30]:
            print("  -", x)
        print("FAIL")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
