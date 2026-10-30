"""THE NIGHT AS A PLAN, in the show (docs/DJ_SETS.md sections 3 and 7, phase 1; 2026-10-28).

The set planner (lib/dj/setplanner.py) turns a theme into an ordered night; this is the Director's side of it - the
plan as it plays, `Director.plan`, beside `self.horizon` and under the same rule (horizon's rule 2): A SLOT CHANGES
ONLY WHEN IT BECOMES IMPOSSIBLE OR YOU CHANGE SOMETHING. A GOOD never re-plans.

  the rows       every record of the night in order, each with its state - played, now, next, planned, skipped - and,
                 when it moved, why. The record playing and the next one (slot +1) are the player's: it decodes the next
                 record the moment a record opens and decides the blend once it is ready, so every re-plan keeps both
                 and changes slot +2 onward. That is also why a re-plan can never cause a cut.
  the hook       Director._record_pick asks `pick()` for the next record when UP NEXT is empty: the next planned slot,
                 pre-checked against the record playing (tempo, the key veto, the vocal room from the stems' envelope
                 on disk) so most refusals never reach a decode; refused by the player's probe, the slot's alternate
                 (PLAN_PROBES of them at most); nothing left: None, and the brain fills the gap steered by the plan's
                 energy and tempo at that point of the night (`steer()`), from the theme's crate.
  the re-routes  section 3's table: NEXT / STAY re-time (the night re-anchored on the player's clock); BAD a negative
                 lean, the slots that still fit and carry none of its tags kept, the rest re-searched; UP NEXT a rejoin
                 (the slots skipped for it stay out for the night); eject / not tonight a skip (and a set_skip row);
                 "cannot be the record" pops the slot; the plan used up is extended in the last chapter's character;
                 LENGTH / ENERGY / the arc / PACE re-plan the rest. Every change is kept with its trigger (`changes`)
                 and written as a set_plan row.
  the clock      the night's seconds are the PLAYER'S clock (the submix's rendered samples from the night's first record),
                 never wall time: a headless gate that renders fast and the live show read the same night.

The search runs in the planner's own process (PlannerProcess): the show only sends ids and receives slot lists, on a
worker thread; answers are applied on the Director's thread at its next step. Nothing heavy is imported here.
"""
import json
import os
import threading
import time

from lib.dj.submix import RATE

EXTEND_MIN = 45.0         # the plan used up at the night's end: this much more, in the last chapter's character
STALE_RETRIES = 4         # an answer whose fixed records no longer match the decks is asked again this often in a row
THEN_ROWS = 8             # THEN rows shown after UP NEXT
ORDER_WINDOW_S = 45.0     # a re-time within this long of NEXT / STAY / BAD is that press's
# THE PLANNER FAILING (2026-10-29): a failed answer was asked again on every Director tick (5 Hz) - a thread, a child
# respawn and a night-log row each. Automatic requests now back off (15 s, doubling, 5 min at most); after this many
# failures in a row the planner is DOWN - said once, the DJ picks record by record, and only something you do (a theme,
# a dial, UP NEXT, BAD, not tonight) asks it again
FAIL_BACKOFF_S = (15.0, 300.0)
PLANNER_DOWN_AFTER = 3


def _mmss(x):
    try:
        x = max(0, int(round(float(x))))
    except (TypeError, ValueError):
        return "?"
    return f"{x // 60}:{x % 60:02d}"


def _hm(x):
    x = max(0, int(round(float(x or 0))))
    return f"{x // 3600}:{(x % 3600) // 60:02d}"


def resolve_theme(theme):
    """A theme from a preset's name, a theme JSON file (the builder's shape), or a dict. Returns (theme dict, source)
    or (None, why)."""
    from lib.dj.setplanner import get_preset, preset_names
    if isinstance(theme, dict):
        return json.loads(json.dumps(theme)), "theme"
    name = str(theme or "").strip()
    if not name:
        return None, "no theme"
    if os.path.isfile(name):
        try:
            with open(name, "r", encoding="utf-8") as fh:
                th = json.load(fh)
        except (OSError, ValueError) as e:
            return None, f"{name}: {type(e).__name__}: {e}"
        if isinstance(th, dict) and isinstance(th.get("theme"), dict):
            th = th["theme"]                          # (a saved request: its theme)
        if not isinstance(th, dict) or not th.get("chapters"):
            return None, f"{name}: not a theme (no chapters)"
        th.setdefault("name", os.path.splitext(os.path.basename(name))[0])
        return th, name
    th = get_preset(name)
    if th is None:
        return None, f"no preset called {name!r} (the presets: {', '.join(preset_names())})"
    return th, name


class NightPlan:
    def __init__(self, director, theme, length_min, seed=None, source=None):
        self.d = director
        self.theme = json.loads(json.dumps(theme))   # THE SNAPSHOT: what was picked; the dials lean it (effective_theme)
        self.name = str(self.theme.get("name") or "a theme")
        self.source = source
        self.length_min = float(length_min)
        # the night seed: the builder's noise reshuffles near-ties; a new night, a new seed unless one is given
        self.seed = int(seed) if seed else int(self.theme.get("night") or (2 + int(time.time()) % 9973))
        self.t_wall = time.time()
        self.clock0 = None                # the submix clock where the night's first record opened (minus offset_s)
        self.offset_s = 0.0               # the night's seconds before this player (a STOP and START go on from here)
        self.rows = []
        self.dropped = []                 # rows a change took out: {id, title, why, change, t}
        self.chapters, self.warnings, self.arc = [], [], []
        self.crate_ids = []
        self.lib_sig = None
        self.finished = True
        self.dead_end = False             # the last answer had nothing to follow the fixed records: the brain carries on
        self.skips = set()                # not tonight / ejected
        self._skip_why = {}               # id -> "ejected" | "not tonight" (the row's words when it leaves)
        self.skipped_for_user = set()     # slots a rejoin skipped for your UP NEXT: they do not come back (decision 3)
        self._failed_ids = set()          # records whose file would not decode (the player's "decode" errors)
        self.neg_leans = {}               # BAD: {word: weight}, fading by half with each new BAD
        self._bad_tags = None             # the last BAD's own tags, for the request that keeps the slots without them
        self.changes = []                 # every change of the plan: {t, seq, trigger, op, before, after, skipped}
        self.retimes = []                 # every re-anchor: {t, trigger, delta_s}
        self.costs = []                   # every answer: {op, trigger, wall_s, total_s, search_s, table_s, length_min, n}
        self.played = []                  # the records that became the record, in order: {id, src, row, start_s, plan_start_s}
        self.offers = []                  # what the hook handed the player: {night_s, after, slot, given, probe}
        self.pending = None
        self.want = []                    # triggers waiting for the request in flight
        self.error = None
        self.replan_on_commit = None      # a trigger: re-search slot +2 on once the new slot +1 is kept
        self.gap_why = None
        self._spent_on = None             # (record id, open clock) whose slot the spent probes dropped (once per record)
        self.drift_s = 0.0
        self._inbox = []
        self._posted = []
        self._lock = threading.Lock()
        self._seq = 0
        self._stale = 0
        self._now_id = None
        self._next_kept = None
        self._cut_sig = None
        self._last_order = None
        self._moves_seen = 0
        self._extend_next = False
        self._extend_want = False
        self._want_user = False           # a trigger waiting in `want` is something you did (it goes through a back-off)
        self._fail_n = 0                  # the planner's failed answers in a row
        self._fail_t = 0.0
        self.planner_down = False
        self._vox = {}
        self._vox_cache = None
        self._reload_next = False

    # -- the night's clock (the player's) ---------------------------------------------------------------------------------
    def night_s(self):
        p = self.d.player
        if p is None or self.clock0 is None:
            return self.offset_s
        return max(0.0, (p._clock() - self.clock0) / RATE)

    def night_of_clock(self, c):
        if c is None or self.clock0 is None:
            return None
        return (int(c) - self.clock0) / RATE

    def length_s(self):
        return 60.0 * self.length_min

    def target_at(self, t):
        """(energy, bpm) the story asks for at night second t: the chapter's courses, held at the last chapter's end
        past the night's length."""
        chs = self.chapters
        if not chs:
            return 0.6, None
        c = chs[-1]
        for x in chs:
            if t < float(x.get("end_s") or 0.0):
                c = x
                break
        s0, s1 = float(c.get("start_s") or 0.0), float(c.get("end_s") or 0.0)
        u = min(max((t - s0) / max(s1 - s0, 1e-6), 0.0), 1.0)
        e = c.get("energy") or [0.6, 0.6]
        b = c.get("tempo_bpm") or [None, None]
        en = float(e[0]) + (float(e[1]) - float(e[0])) * u
        bp = (float(b[0]) + (float(b[1]) - float(b[0])) * u) if (b and b[0] is not None and b[1] is not None) else None
        return max(0.0, min(1.0, en)), bp

    def chapter_at(self, t):
        for i, x in enumerate(self.chapters):
            if t < float(x.get("end_s") or 0.0):
                return i
        return len(self.chapters) - 1 if self.chapters else None

    # -- what is asked of the planner -------------------------------------------------------------------------------------
    def effective_theme(self):
        """The snapshot with the dials on it: ENERGY leans every chapter's course. The ARC dial does not reach a planned
        night: its STORY is the arc (2026-10-29 - an ARC shape turned on the web page or a MIDI knob silently replaced
        every chapter's energy the user had set in the NIGHT tab)."""
        from lib.dj.director import ENERGY_LEAN
        th = json.loads(json.dumps(self.theme))
        chs = th.get("chapters") or []
        lean = ENERGY_LEAN.get(self.d.dials.get("energy"), 0.0)
        if lean:
            for c in chs:
                c["energy"] = [round(max(0.0, min(1.0, float(e) + lean)), 3) for e in (c.get("energy") or [0.5, 0.5])]
        return th

    @staticmethod
    def _ext_theme(th):
        """The plan used up at the night's end: one chapter in the last chapter's character (its end values held)."""
        chs = th.get("chapters") or []
        last = dict(chs[-1]) if chs else {"name": "the night", "leans": [], "dims": {}}
        e = list(last.get("energy") or [0.6, 0.6])
        q = list(last.get("tempo") or [0.5, 0.5])
        last.update({"share": 1.0, "energy": [e[-1], e[-1]], "tempo": [q[-1], q[-1]],
                     "dims": {k: [v[-1], v[-1]] for k, v in (last.get("dims") or {}).items() if v}})
        out = dict(th)
        out["chapters"] = [last]
        return out

    def committed(self):
        """The records the player has committed, in order: the record playing (its cue, its start on the night's clock)
        and the next one on a deck (its cue and start once the blend is decided). Before the first record opens: the
        opener on its deck, if any."""
        p = self.d.player
        out = []
        if p is None:
            return out
        rec = p.songs.get(p.record) if p.record else None
        if rec is None:
            # (a snapshot of the decks: the decode thread adds and drops entries while this runs - 2026-10-29)
            op = next((r for r in list(p.songs.values()) if r.state in ("decoding", "ready")), None)
            if op is not None:
                out.append({"id": int(op.track.id), "start_s": self.offset_s})
            return out
        it = {"id": int(rec.track.id), "cue_s": float(rec.cue_t)}
        if getattr(rec, "phrase_t", None) is not None:
            it["phrase_s"] = float(rec.phrase_t)      # (a cut landed it past its silent start: its phrases count from here)
        st = self.night_of_clock(rec.open_clock)
        if st is not None:
            it["start_s"] = st
        out.append(it)
        nxt = self._next_of(p)
        if nxt is not None:
            it = {"id": int(nxt.track.id)}
            c = p.cut
            if c is not None and c.get("b") == nxt.deck:
                it["cue_s"] = float(c["cue_b"])
                if c.get("lm0") is not None and abs(float(c["lm0"]) - float(c["cue_b"])) > 1e-6:
                    it["phrase_s"] = float(c["lm0"])
                st = self.night_of_clock(c["clock"])
                if st is not None:
                    it["start_s"] = st
            out.append(it)
        return out

    @staticmethod
    def _next_of(p):
        nxt = p._next_rec()
        c = p.cut
        if nxt is None and c is not None and c.get("state") == "started":
            nxt = p.songs.get(c["b"])
        return nxt

    def _req(self, op=None):
        d = self.d
        p = d.player
        from lib.dj.director import PLAY_MIN_S
        com = self.committed()
        com_ids = [c["id"] for c in com]
        th = self.effective_theme()
        length_min = self.length_min
        extend = self._extend_next
        self._extend_next = False
        if extend:
            th = self._ext_theme(th)
            end_known = max([float(r.get("end_s") or 0.0) for r in self.rows if r["state"] in ("now", "next")] + [self.night_s()])
            length_min = min(480.0, (max(self.length_s(), end_known) + EXTEND_MIN * 60.0) / 60.0)
        queued = [int(t) for t in d.up_next if int(t) not in com_ids]
        req = {"theme": th, "length_min": length_min, "seed": self.seed, "neg_leans": dict(self.neg_leans),
               "play_min_s": float(p.play_min_s if p is not None else PLAY_MIN_S[d.dials["pace"]]),
               "played_ids": [int(t) for t, _w in d.played if int(t) not in com_ids],
               # (the slots skipped for your record stay out for the night - decision 3 - and so do files that would
               # not decode)
               "exclude_ids": sorted({int(x) for x in self.skips | set(d.set_skips) | self.skipped_for_user | self._failed_ids
                                      | set(getattr(p, "_failed", None) or ())} - set(com_ids)),
               "user_ids": sorted(set(queued) | {r["id"] for r in self.rows if r.get("src") == "user"})}
        bad, self._bad_tags = self._bad_tags, None
        if bad:
            req["bad_tags"] = list(bad)
        if self._reload_next:
            req["reload"] = True
            self._reload_next = False
        if com:
            req["fixed"] = com + [{"id": t} for t in queued[:-1]]
            req["elapsed_s"] = float(com[0].get("start_s", self.night_s()))
        else:
            req["fixed"] = [{"id": t} for t in queued[:-1]]
            req["elapsed_s"] = float(self.offset_s)
        plan_ids = [r["id"] for r in self.rows if r["state"] == "planned" and not r.get("user") and r["id"] not in queued]
        if queued:
            req["op"] = "rejoin"
            req["queued_id"] = queued[-1]
            req["plan_ids"] = plan_ids
        else:
            req["op"] = op or ("replan" if req["fixed"] else "plan")
            if bad and req["op"] == "replan" and plan_ids:
                # BAD: the slots that still fit and do not carry its tags stay (DJ_SETS.md section 3); the planner keeps the
                # order up to the first slot that breaks and searches on from there
                req["plan_ids"] = plan_ids
        return req, com_ids, queued, extend

    def _held(self):
        """Automatic requests wait while the planner is failing: a back-off after each failed answer, and not at all
        once it is down (only something you do asks it again)."""
        if self.planner_down:
            return True
        if self._fail_n <= 0:
            return False
        wait = min(FAIL_BACKOFF_S[1], FAIL_BACKOFF_S[0] * 2 ** (self._fail_n - 1))
        return time.time() - self._fail_t < wait

    def _keep_bad(self, pend):
        """A BAD's request that was not applied (stale, failed): its tags wait for the request that asks again, so that
        one still keeps the slots without them."""
        if pend.get("bad_tags"):
            self._bad_tags = sorted(set(self._bad_tags or []) | set(pend["bad_tags"]))

    def _failed(self, trigger, why):
        """A request that came to nothing (no answer, an error, a dead child): counted for the back-off; said once, and
        once more when the planner is down - never once per tick."""
        self._fail_n += 1
        self._fail_t = time.time()
        self.error = why
        if self._fail_n == 1:
            self.d._event(f"the plan could not be made ({trigger[:40]}): {why[:80]} - asked again in {FAIL_BACKOFF_S[0]:.0f} s")
        if self._fail_n >= PLANNER_DOWN_AFTER and not self.planner_down:
            self.planner_down = True
            self.d._event(f"the planner is not available ({why[:60]}): the DJ picks record by record "
                          "(a theme, a dial, UP NEXT or BAD asks it again)")

    def request(self, trigger, extend=False, user=False):
        """Ask the planner, off-thread. One request in flight; a trigger that comes meanwhile waits for it (the newest
        state is what is sent then). `user`: something you did - it goes through a back-off and wakes a planner that is
        down; an automatic trigger waits them out."""
        with self._lock:
            if user and self.planner_down:
                self.planner_down = False
                self.d._event(f"the planner is asked again ({trigger[:40]})")
            if self.pending is not None or (not user and self._held()):
                if trigger not in self.want:
                    self.want.append(trigger)
                if extend:
                    self._extend_want = True
                if user:
                    self._want_user = True
                return False
            if extend:
                self._extend_next = True
            req, com_ids, queued, ext = self._req()
            self._seq += 1
            req["seq"] = self._seq
            self.pending = {"seq": self._seq, "op": req["op"], "trigger": trigger, "t": time.time(), "com": com_ids,
                            "queued": queued, "length_min": req["length_min"], "extend": ext,
                            "plan_ids": list(req.get("plan_ids") or []), "bad_tags": list(req.get("bad_tags") or []),
                            "gone": sorted(set(req["played_ids"]) | set(req["exclude_ids"]) | set(com_ids))}
        try:
            self.d._planner_proc().submit(req, self._on_answer)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                pend, self.pending = self.pending, None
            self._keep_bad(pend or {})
            self._failed(trigger, f"the planner: {type(e).__name__}: {e}")
            return False
        return True

    def _on_answer(self, resp):
        """(the request's worker thread) kept for the Director's thread"""
        with self._lock:
            self._inbox.append(resp)

    # -- answers -------------------------------------------------------------------------------------------------------------
    def _row_of(self, s, n_fixed_idx, prev):
        r = {k: s.get(k) for k in ("id", "title", "artist", "bpm", "camelot", "energy", "sung", "chapter", "chapter_name",
                                   "start_s", "play_s", "end_s", "cue_s", "target", "score", "matched", "in", "alts")}
        r["id"] = int(r["id"])
        r["user"] = bool(s.get("user"))
        r["fixed"] = bool(s.get("fixed"))
        r["plan_start_s"] = s.get("start_s")
        r["planned_title"] = s.get("title")
        r["prev_id"] = int(prev["id"]) if prev is not None else None
        r["prev_cue"] = prev.get("cue_s") if prev is not None else None
        r["state"] = "planned"
        r["why"] = None
        r["src"] = "user" if r["user"] else "plan"
        r["alts"] = list(s.get("alts") or [])
        return r

    def _apply(self, resp):
        pend = self.pending
        if pend is None or resp.get("seq") != pend["seq"]:
            return                                    # (an answer to a request already given up)
        self.pending = None
        tm = resp.get("timing") or {}
        cost = {"t": time.time(), "op": pend["op"], "trigger": pend["trigger"], "wall_s": round(time.time() - pend["t"], 3),
                "total_s": tm.get("total_s"), "search_s": tm.get("search_s"), "table_s": tm.get("table_s"), "load_s": tm.get("load_s"),
                "length_min": pend["length_min"], "n": len(resp.get("slots") or []), "ok": bool(resp.get("ok"))}
        self.costs.append(cost)
        del self.costs[:-200]
        if not resp.get("ok"):
            self._keep_bad(pend)
            self._failed(pend["trigger"], str(resp.get("error") or "the planner did not answer"))
            return
        self.error = None
        self._fail_n = 0
        self.planner_down = False
        sent = [int(s["id"]) for s in resp.get("slots") or []]                  # (the answer as the planner sent it)
        sent_fixed = sum(1 for s in resp.get("slots") or [] if s.get("fixed"))
        slots = list(resp.get("slots") or [])
        com_now = [c["id"] for c in self.committed()]
        played_ids = {r["id"] for r in self.rows if r["state"] == "played"}
        while slots and int(slots[0]["id"]) in played_ids and int(slots[0]["id"]) not in com_now:
            slots.pop(0)                              # (a swap while it was searched: that record has played)
        ids = [int(s["id"]) for s in slots]
        if ids[:len(com_now)] != com_now:
            # THE DECKS MOVED while it was planned (a pick, a swap, an eject): the answer does not start from what plays -
            # asked again from the decks as they are now, never applied over them
            self._stale += 1
            self._keep_bad(pend)
            if self._stale <= STALE_RETRIES:
                self.want.insert(0, pend["trigger"] if "(again" in pend["trigger"] else pend["trigger"] + " (again: the decks moved while it was planned)")
            else:
                # (counted as a failure, so the automatic re-asks back off instead of chasing the decks every tick)
                self._stale = 0
                self._failed(pend["trigger"], "the plan kept missing the decks - the DJ picks until something changes")
            return
        self._stale = 0
        n_com = len(com_now)
        first = not self.changes
        kept = [r for r in self.rows if r["state"] == "played" or (r["state"] in ("now", "next") and r["id"] in com_now)]
        old_future = [r for r in self.rows if r not in kept and r["state"] in ("planned", "next")]
        for r in old_future:
            r["state"] = "planned"                    # (a slot handed over and refused is the plan's again)
        old_ids = [r["id"] for r in old_future]
        old_by_id = {r["id"]: r for r in old_future}
        by_kept = {r["id"]: r for r in kept if r["state"] in ("now", "next")}
        new_rows = list(kept)
        prev = None
        for k, s in enumerate(slots):
            sid = int(s["id"])
            if k < n_com:
                r = by_kept.get(sid)
                fresh = self._row_of(s, k, prev)
                if r is None:
                    fresh["state"] = "now" if k == 0 and self.d.player is not None and self.d.player.record else "next"
                    new_rows.append(fresh)
                else:
                    for key in ("chapter", "chapter_name", "play_s", "end_s", "target", "score", "matched"):
                        r[key] = fresh[key]
                    if r.get("start_s") is None:
                        r["start_s"] = fresh["start_s"]
                prev = s
                continue
            r = self._row_of(s, k, prev)
            if not first:
                o = old_by_id.get(sid)
                r["why"] = (o.get("why") if o is not None else f"new: {pend['trigger'][:60]}")
                if o is not None and o.get("planned_title"):
                    r["planned_title"] = o["planned_title"]
            new_rows.append(r)
            prev = s
        if pend["extend"]:
            ci = len(self.chapters) - 1 if self.chapters else 0
            for r in new_rows[len(kept):]:
                if r["state"] == "planned":
                    r["chapter"] = ci
                    r["chapter_name"] = (self.chapters[ci]["name"] if self.chapters else r.get("chapter_name"))
                    r["why"] = r["why"] or "the night runs on in its last chapter's character"
        new_future = [r["id"] for r in new_rows if r["state"] == "planned"]
        skipped_ids = {int(x["id"]) for x in resp.get("skipped") or []}
        self.skipped_for_user |= skipped_ids                 # (decision 3: skipped for your record, they do not come back)
        dropped = []
        for o in old_future:
            if o["id"] not in new_future:
                why = "skipped for your record" if o["id"] in skipped_ids else (o.get("why") if o["state"] == "skipped" else f"dropped: {pend['trigger'][:60]}")
                dropped.append({"id": o["id"], "title": o.get("title"), "why": why, "t": time.time(), "change": pend["seq"]})
        self.rows = new_rows
        if dropped:
            self.dropped = (self.dropped + dropped)[-60:]
        if not pend["extend"] or not self.chapters:
            self.chapters = list(resp.get("chapters") or [])
            self.arc = list(resp.get("arc") or [])
        self.warnings = list(resp.get("warnings") or [])
        if resp.get("keys"):
            self.keys_words = str(resp["keys"])       # (the planner's own words for the theme's keys mode)
        if resp.get("crate_ids"):
            self.crate_ids = [int(i) for i in resp["crate_ids"]]
        self.finished = bool(resp.get("finished", True))
        self.dead_end = not new_future and not self.finished
        self.drift_s = 0.0
        sig = resp.get("lib_sig")
        if sig is not None and self.lib_sig is not None and sig != self.lib_sig:
            self.d._event("the plan: the library changed under it (keys or tempos rescanned): read again on the next re-plan")
        self.lib_sig = sig
        if resp.get("queued_refused"):
            self.d._event(f"the plan: your record cannot be the record ({resp['queued_refused']}): the plan goes on without it")
        played_now = {x["id"] for x in self.played}
        # (kept with the change for the gates: the planner's own answer - `sent`, its fixed prefix first - and from it the
        # order the night should now follow; what the request asked to keep and what it kept)
        ch = {"t": time.time(), "seq": pend["seq"], "trigger": pend["trigger"], "op": pend["op"], "extend": pend["extend"],
              "before": old_ids, "after": new_future, "skipped": [x["id"] for x in dropped],
              "rejoined": resp.get("rejoined"), "timing": cost, "night_s": round(self.night_s(), 1),
              "sent": sent, "sent_fixed": sent_fixed, "order": [i for i in sent if i not in played_now], "n_played": len(self.played),
              "skipped_user": sorted(skipped_ids), "plan_ids": pend.get("plan_ids") or [], "bad_tags": pend.get("bad_tags") or [],
              "gone": pend.get("gone") or [], "kept": resp.get("kept_ids"), "kept_stop": resp.get("kept_stop"), "fixed": com_now}
        self.changes.append(ch)
        del self.changes[:-200]
        self._log({"event": "set_plan", "theme": self.name, "trigger": pend["trigger"][:160], "op": pend["op"], "seq": pend["seq"],
                   "seed": self.seed, "length_min": pend["length_min"], "night_s": round(self.night_s(), 1),
                   "fixed": com_now, "ids": new_future, "skipped": [x["id"] for x in dropped],
                   "finished": self.finished, "timing": {k: cost.get(k) for k in ("wall_s", "total_s", "search_s", "table_s")}})
        n_ahead = len(new_future)
        self.d._event((f"the night is planned: {self.name}, {n_ahead} records, {len(self.chapters)} chapters"
                       if first else f"the plan ({pend['trigger'][:50]}): {n_ahead} records ahead")
                      + (f", {len(dropped)} out" if dropped else "") + f" [{cost['wall_s']:.1f} s]")

    # -- the hook: the next record of the plan ----------------------------------------------------------------------------
    def _first_planned(self):
        return next((r for r in self.rows if r["state"] == "planned"), None)

    def opener_id(self):
        r = self._first_planned()
        return r["id"] if r else None

    def pick(self, p=None):
        """Director._record_pick's plan part for player `p` (the one asking: a player prepared before START is not the
        Director's yet): (track, 'plan'[, probe]) or None (the brain fills the gap)."""
        p = p if p is not None else self.d.player
        if p is None:
            return None
        rec = p.songs.get(p.record) if p.record else None
        self._unhand(p, rec)                          # (a slot the probe just refused is the plan's again: its alternate next)
        busy = p._busy_ids()
        avoid = set(p.avoid_ids or ()) | self.skips | set(self.d.set_skips)
        failed = self._failed_ids | set(getattr(p, "_failed", None) or ())
        taken = {r["id"] for r in self.rows if r["state"] in ("played", "now", "next")}
        from lib.dj.recordplay import PLAN_PROBES
        if rec is None:
            # the opener: the plan's first record (no probe: nothing plays before it)
            for r in self.rows:
                if r["state"] != "planned":
                    continue
                t = self.d.track(r["id"])
                if t is None or r["id"] in busy or r["id"] in avoid or r["id"] in taken or r["id"] in failed:
                    self._drop(r, "not available to open the night")
                    continue
                r["state"] = "next"
                return (t, "plan")
            self.gap_why = "nothing planned to open the night"
            return None
        spent = p._plan_repicks >= PLAN_PROBES
        r = self._first_planned()
        while r is not None:
            if r.get("user") and r["id"] not in self.d.up_next:
                self._drop(r, "taken off UP NEXT")
                if "your UP NEXT changed" not in self.want:
                    self.want.append("your UP NEXT changed")
            elif r["id"] in self.skips or r["id"] in self.d.set_skips:
                self._drop(r, self._skip_why.get(r["id"], "not tonight"))     # (ejected or not tonight: never offered)
            else:
                break
            r = self._first_planned()
        if r is None:
            self.gap_why = "the plan is used up" if not self.pending else "the plan is being made"
            self._need_more()
            return None
        if spent:
            # THE PROBES SPENT (2026-10-29) end the slot like "nor its alternates": its row - still holding the last
            # alternate refused - goes, and the re-search trigger is the gap's. Left, the trigger and the later THEN rows
            # named a refused record "instead", and the brain's gap record never got its "the DJ filled a gap" row
            # (once per record: a gap pick of the brain's refused in turn asks the hook again, and the next slot is not
            # this one's to lose)
            why = f"could not blend after {rec.track.title[:22]}: the plan's {PLAN_PROBES} probes are spent"
            mark = (int(rec.track.id), rec.open_clock)
            if self._spent_on != mark:
                self._spent_on = mark
                if not r.get("user"):
                    self._drop(r, f"{why}, the DJ filled a gap")
                self.replan_on_commit = f"{why}, the DJ filled a gap"
            self.gap_why = why
            return None
        cands = [(r["id"], None)] + [(int(a["id"]), a) for a in (r.get("alts") or [])]
        for cid, alt in cands:
            if cid in busy or cid in avoid or cid in taken or cid in p._unfit or cid in failed:
                continue
            t = self.d.track(cid)
            if t is None:
                continue
            kind = ((alt if alt is not None else r).get("in") or {}).get("kind")
            why = self._precheck(p, rec, t, r, kind)
            if why:
                self.d._event(f"the plan: {t.title[:24]} would not blend after {rec.track.title[:22]} ({why}): not offered")
                continue
            if cid != r["id"]:
                orig = r.get("planned_title") or r.get("title") or "?"
                reason = "would not decode" if r["id"] in failed else f"could not blend after {rec.track.title[:22]}"
                r["alt_of"] = r.get("alt_of") or r["id"]
                r.update({"id": cid, "title": t.title, "artist": t.artist or "", "bpm": round(float(t.bpm), 1), "camelot": t.camelot or "",
                          "energy": (alt or {}).get("energy"), "in": (alt or {}).get("in")})
                r["why"] = f"picked again: {orig[:24]} {reason}, {t.title[:24]} instead"
                self.replan_on_commit = r["why"]   # slots +2 on are re-searched from what plays (DJ_SETS.md section 3)
            r["state"] = "next"
            # THE ONE UNPROBED SEAM is the planner's relaxed / end label - and only after the very record and cue it was
            # placed from (2026-10-29): once that record has left (refused, ejected), the label says nothing about the
            # record playing, so the slot is probed like any other and a refusal moves on to its alternate
            probe = not (kind in ("relaxed", "end") and self._placed_here(r, rec))
            self.offers.append({"night_s": round(self.night_s(), 1), "after": rec.track.title, "slot": r.get("planned_title"),
                                "given": t.title, "probe": probe})
            del self.offers[:-100]
            return (t, "plan", probe)
        # nothing of this slot can follow the record playing: the brain fills the gap, the rest is re-searched from it
        why = f"could not blend after {rec.track.title[:22]}, nor its alternates"
        self._drop(r, f"{why}: the DJ filled a gap")
        self.gap_why = why                            # (the gap record's row: "the DJ filled a gap (<this>)", said once)
        self.replan_on_commit = f"{why}: the DJ filled a gap"
        return None

    @staticmethod
    def _placed_here(r, rec):
        """Did the planner place this slot's seam from the record playing, at the cue it opened at?"""
        pc = r.get("prev_cue")
        if r.get("prev_id") != rec.track.id or pc is None:
            return False
        # (or at the landmark a cut landed it past: the planner, with no mix envelope, places a record it brings in whole at
        # its landmark - 2026-10-29)
        ph = getattr(rec, "phrase_t", None)
        return abs(float(pc) - float(rec.cue_t)) < 0.05 or (ph is not None and abs(float(pc) - float(ph)) < 0.05)

    def _precheck(self, p, rec, t, r, kind=None):
        """The pair facts against the record playing, BEFORE a decode (DJ_SETS.md section 3, the third hook fix): the
        tempo (the stretcher's limit), the key veto (the player's own fit, pass/fail), its fit as a record, and - unless
        the planner placed this seam from this very record and cue - the vocal room, by the player's own strict placement
        with the vocal envelope the stems keep on disk. `kind`: the seam's kind as planned (relaxed / end are honoured
        only from that record). None = offer it; else why not."""
        from lib.dj.recordplay import MAX_TEMPO_GAP, KEY_BLEND, RecordPlayer, Rec
        gap = abs(float(rec.track.bpm) / max(float(t.bpm), 1e-6) - 1.0)
        if gap > MAX_TEMPO_GAP:
            return f"a tempo gap of {gap * 100:.0f} %"
        fit = RecordPlayer.key_fit(rec.track, t)
        if fit is not None and fit < KEY_BLEND:
            return f"keys {rec.track.camelot}→{t.camelot} fit {fit:.2f}"
        why = p.fits_as_record(t)
        if why:
            return why
        if self._placed_here(r, rec):
            # the planner placed this seam from this record at this cue (strict, or the one relaxed / end seam it
            # labelled, which pick() hands over without the strict probe)
            return None
        probe = Rec("p", t)
        probe.vox, probe.state = self._vox_of(t), "ready"
        try:
            ok = p._find_blend_strict(rec, probe) is not None
        except Exception:
            ok = True
        return None if ok else (getattr(p, "_no_blend_why", "") or "no placement")[:80]

    def _vox_of(self, t):
        if t.id in self._vox:
            return self._vox[t.id]
        v = None
        try:
            from lib.dj import setplanner as SP
            if self._vox_cache is None:
                self._vox_cache = SP._load_vox_cache(self.d.music_root)
            v, _exact = SP._vox_measure(self.d.music_root, t, self._vox_cache)
        except Exception:
            v = None
        self._vox[t.id] = v
        return v

    def _drop(self, r, why):
        r["state"] = "skipped"
        r["why"] = why
        self.dropped = (self.dropped + [{"id": r["id"], "title": r.get("title"), "why": why, "t": time.time(),
                                         "change": self._seq}])[-60:]
        self.rows = [x for x in self.rows if x is not r]

    def _need_more(self):
        """The plan has nothing after the next record: a re-search fills the night while it lasts; past its end the night
        runs on in its last chapter's character; the brain carries on only when the crate has nothing left."""
        if self.dead_end or self.pending is not None or self._held():
            return
        end_known = max([float(r.get("end_s") or 0.0) for r in self.rows if r["state"] in ("now", "next")] + [self.night_s()])
        if end_known + 60.0 < self.length_s() and not any(c.get("op") == "replan" and c["trigger"].startswith("the plan is used up")
                                                            and time.time() - c["t"] < 30.0 for c in self.changes):
            self.request("the plan is used up early: a re-search fills the night")
        else:
            self.request("the night runs on: more in its last chapter's character", extend=True)

    # -- the tick (the Director's thread) -----------------------------------------------------------------------------------
    def post(self, fn, *a):
        """A press from another thread (the tab, the web): run at the Director's next step while it runs, else now."""
        if self.d.running and self.d.player is not None:
            with self._lock:
                self._posted.append((fn, a))
        else:
            fn(*a)

    def poll(self):
        """Apply answers and send what waits (no player needed): before START and on every step."""
        with self._lock:
            inbox, self._inbox = self._inbox, []
            posted, self._posted = self._posted, []
        for fn, a in posted:
            try:
                fn(*a)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
        for resp in inbox:
            try:
                self._apply(resp)
            except Exception as e:  # noqa: BLE001
                import traceback
                self.error = f"apply: {type(e).__name__}: {e}"
                traceback.print_exc()
        if self.pending is None and self.want and (self._want_user or not self._held()):
            ext, user = self._extend_want, self._want_user
            self._extend_want = self._want_user = False
            trig, self.want = "; ".join(self.want), []
            self.request(trig, extend=ext, user=user)
        elif self.pending is not None and time.time() - self.pending["t"] > 330.0:
            trig = self.pending["trigger"]
            self._keep_bad(self.pending)
            self.pending = None
            self._failed(trig, "no answer in 5 min: given up")

    def step(self):
        self.poll()
        p = self.d.player
        if p is None:
            return
        rec = p.songs.get(p.record) if p.record else None
        if self.clock0 is None and rec is not None and rec.open_clock is not None:
            self.clock0 = int(rec.open_clock) - int(round(self.offset_s * RATE))
        now_id = int(rec.track.id) if rec is not None else None
        if now_id is not None and now_id != self._now_id:
            self._on_now(rec)
            self._now_id = now_id
        nxt = self._next_of(p)
        self._sync_next(p, rec, nxt)
        self._read_moves(p)
        kept = nxt is not None and (nxt.state in ("ready", "playing") or nxt.deck in p._pending)
        if kept and int(nxt.track.id) != self._next_kept:
            self._next_kept = int(nxt.track.id)
            if self.replan_on_commit:
                trig, self.replan_on_commit = self.replan_on_commit, None
                self.request(trig)
        self._reanchor(p, nxt)
        if rec is not None and self._first_planned() is None and not self.dead_end and self.pending is None and not self.want:
            self._need_more()

    def _row_index(self, tid, states):
        for i, r in enumerate(self.rows):
            if r["id"] == tid and r["state"] in states:
                return i
        return None

    def _insert_row(self, t, state, src, why):
        """A record the plan did not offer (a gap the brain filled, your UP NEXT before the plan answered): its own row,
        before the planned ones."""
        i = next((k for k, r in enumerate(self.rows) if r["state"] == "planned"), len(self.rows))
        ns = self.night_s()
        e, b = self.target_at(ns)
        ci = self.chapter_at(ns)
        r = {"id": int(t.id), "title": t.title, "artist": t.artist or "", "bpm": round(float(t.bpm or 0.0), 1), "camelot": t.camelot or "",
             "energy": None, "chapter": ci, "chapter_name": (self.chapters[ci]["name"] if ci is not None and self.chapters else None),
             "start_s": None, "play_s": None, "end_s": None, "cue_s": None, "target": {"energy": round(e, 3), "bpm": b}, "in": None,
             "alts": [], "user": src == "user", "fixed": True, "state": state, "why": why, "src": src, "planned_title": t.title,
             "prev_id": None, "prev_cue": None, "plan_start_s": None}
        self.rows.insert(i, r)
        return r

    def _on_now(self, rec):
        # a new record, a new question (2026-10-29): "nothing follows" was said of the record before - the plan is asked
        # again from this one (step's _need_more) instead of leaving the brain every slot for the rest of the night
        self.dead_end = False
        for r in self.rows:
            if r["state"] == "now":
                r["state"] = "played"
        i = self._row_index(int(rec.track.id), ("next", "planned"))
        st = self.night_of_clock(rec.open_clock)
        if i is None:
            src = getattr(rec, "src", "brain")
            why = "your record" if src == "user" else ("the DJ filled a gap" + (f" ({self.gap_why})" if self.gap_why else ""))
            r = self._insert_row(rec.track, "now", src, why)
        else:
            r = self.rows[i]
            r["state"] = "now"
        r["src"] = getattr(rec, "src", r.get("src"))
        if st is not None:
            r["actual_start_s"] = st
            if r.get("start_s") is None:
                r["start_s"] = st
        self.played.append({"id": int(rec.track.id), "src": r["src"], "title": rec.track.title, "start_s": st,
                            "plan_start_s": r.get("plan_start_s"), "chapter": r.get("chapter"), "why": r.get("why"),
                            "alt_of": r.get("alt_of")})
        self.gap_why = None

    def _unhand(self, p, rec):
        """A slot handed over but not on a deck: refused by the probe (the slot's alternate comes at the next pick), or the
        player took another pick instead - the slot is the plan's again."""
        rid = int(rec.track.id) if rec is not None else None
        for r in list(self.rows):
            if r["state"] == "next" and r["id"] != rid:
                # (snapshots: the decode thread changes these dicts while this runs - 2026-10-29)
                on = any(s.track.id == r["id"] for s in list(p.songs.values())) or any(t.id == r["id"] for t in list(p._decoding.values())) \
                    or any(x[0].id == r["id"] for x in list(p._pending.values()))
                if not on:
                    if r["id"] in self.skips or r["id"] in self.d.set_skips:
                        # EJECTED (or not tonight): off the decks and never the plan's again - put back as 'planned' it was
                        # handed straight back at the next pick (2026-10-29)
                        self._drop(r, self._skip_why.get(r["id"], "not tonight"))
                    else:
                        r["state"] = "planned"

    def _sync_next(self, p, rec, nxt):
        nid = int(nxt.track.id) if nxt is not None else None
        self._unhand(p, rec)
        if nid is None:
            return
        i = self._row_index(nid, ("next", "planned"))
        if i is None:
            if self._row_index(nid, ("now", "played")) is None:
                src = getattr(nxt, "src", "brain")
                why = "your record" if src == "user" else ("the DJ filled a gap" + (f" ({self.gap_why})" if self.gap_why else ""))
                self._insert_row(nxt.track, "next", src, why)
                if src == "brain":
                    # a gap the brain filled is a new fixed record: whatever "nothing follows" said, the plan is asked again
                    # from it once it is kept (2026-10-29)
                    self.dead_end = False
                    if not self.replan_on_commit:
                        self.replan_on_commit = f"the DJ filled a gap with {nxt.track.title[:24]}"
        else:
            r = self.rows[i]
            src = getattr(nxt, "src", r.get("src"))
            if r["state"] == "planned" and src == "brain":
                # THE BRAIN'S GAP RECORD IS A LATER SLOT (2026-10-29: the gate's gap after "nor its alternates" took Afon,
                # planned two slots on): it is the DJ's gap record, said like one, and comes forward to slot +1 - its row
                # had kept the plan's place and no words, so THEN and the night's log read it as the plan's own pick
                r["why"] = "the DJ filled a gap" + (f" ({self.gap_why})" if self.gap_why else "")
                j = next((k for k, x in enumerate(self.rows) if x["state"] == "planned"), i)
                if j < i:
                    self.rows.insert(j, self.rows.pop(i))
                self.dead_end = False
                if not self.replan_on_commit:
                    self.replan_on_commit = f"the DJ filled a gap with {nxt.track.title[:24]}"
            if r["state"] == "planned":
                # (on a deck before the plan offered it: it was the next planned anyway, or your record)
                r["state"] = "next"
            r["src"] = src

    def _read_moves(self, p):
        """The player's own words: a slot it would not take as a record ("cannot be the record") is popped - otherwise the
        plan stalls while the brain silently fills every slot."""
        moves = p.moves or []
        seen = self._moves_seen if self._moves_seen <= len(moves) else 0
        for m in moves[seen:]:
            if isinstance(m, dict) and m.get("kind") == "error" and m.get("to_id") is not None and str(m.get("text")).startswith("decode "):
                # its file would not decode: never offered again tonight - the slot moves on to its alternate at the next
                # pick (pick() skips it), the request leaves it out (2026-10-29)
                self._failed_ids.add(int(m["to_id"]))
                continue
            if not isinstance(m, dict) or m.get("kind") != "pick":
                continue
            tid = m.get("to_id")
            if "cannot be the record" in str(m.get("text")) and tid is not None:
                i = self._row_index(int(tid), ("next", "planned"))
                if i is not None and not self.rows[i].get("user"):
                    r = self.rows[i]
                    why = str(m.get("text")).split("(", 1)[-1].split(")")[0][:60]
                    self._drop(r, f"cannot be the record ({why})")
                    self.replan_on_commit = self.replan_on_commit or f"{r.get('title', '?')[:24]} cannot be the record"
        self._moves_seen = len(moves)

    def _reanchor(self, p, nxt):
        """NEXT / STAY / the player's own decision moved where the next record starts: the rest of the night is re-timed
        on the player's clock (the order stands); past a record's worth of drift since the last plan, re-searched."""
        c = p.cut
        if c is None or nxt is None or c.get("b") != nxt.deck:
            return
        sig = (c.get("seq"), c.get("clock"))
        if sig == self._cut_sig:
            return
        self._cut_sig = sig
        st = self.night_of_clock(c["clock"])
        i = self._row_index(int(nxt.track.id), ("next",))
        if st is None or i is None:
            return
        r = self.rows[i]
        r["actual_start_s"] = st
        planned = r.get("start_s")
        if planned is None:
            r["start_s"] = st
            return
        delta = st - float(planned)
        if abs(delta) < 0.5:
            return
        for x in self.rows[i:]:
            if x["state"] in ("next", "planned"):
                for k in ("start_s", "end_s"):
                    if x.get(k) is not None:
                        x[k] = round(float(x[k]) + delta, 3)
        r["start_s"] = st
        self.drift_s += delta
        lo = self._last_order
        trig = (f"{lo[0]}" if lo and time.time() - lo[1] < ORDER_WINDOW_S else "the player's clock")
        self.retimes.append({"t": time.time(), "trigger": trig, "delta_s": round(delta, 2), "night_s": round(self.night_s(), 1), "id": r["id"]})
        del self.retimes[:-200]
        nxt_p = self._first_planned()
        if nxt_p is not None and abs(self.drift_s) > max(60.0, float(nxt_p.get("play_s") or 0.0)):
            self.drift_s = 0.0
            self.request(f"re-timed by {trig}: the night moved by more than a record")

    # -- what you do ---------------------------------------------------------------------------------------------------------
    def on_order(self, kind):
        self._last_order = (str(kind), time.time())

    def on_bad(self, tid, tags):
        """BAD: the record's tags lean the rest of the night away (fading by half with each new BAD); slots +2 on are
        re-scored. Nothing is trained."""
        for k in list(self.neg_leans):
            self.neg_leans[k] = round(self.neg_leans[k] * 0.5, 3)
            if self.neg_leans[k] < 0.1:
                del self.neg_leans[k]
        tags = [str(x).lower() for x in (tags or []) if x]
        for tag in tags:
            self.neg_leans[tag] = min(2.0, self.neg_leans.get(tag, 0.0) + 1.0)
        # the slots that do not carry THIS BAD's tags and still blend stay; the search goes on from the first that does
        # (a BAD waiting behind another request keeps the union of both records' tags)
        self._bad_tags = sorted(set(self._bad_tags or []) | set(tags)) if tags else self._bad_tags
        self.on_order("BAD")
        title = self.d.title(int(tid))[:24] if tid is not None else "the record"
        self.request(f"BAD on {title}" + (f": less like {', '.join(tags)}" if tags else ""), user=True)

    def skip_now(self, tid, why):
        """The skip, AT ONCE (2026-10-29): Director.eject calls this before the player lets go of the record, so the hook
        (which the Director's next tick runs before the posted on_eject) cannot hand the same record straight back.
        Returns whether it was skipped already."""
        tid = int(tid)
        was = tid in self.skips
        self.skips.add(tid)
        self._skip_why.setdefault(tid, why)
        return was

    def unskip(self, tid):
        """An eject that did not happen (the record was playing, or coming in): the skip skip_now made is taken back."""
        self.skips.discard(int(tid))
        self._skip_why.pop(int(tid), None)

    def on_eject(self, tid):
        tid = int(tid)
        self.skip_now(tid, "ejected")
        self._log({"event": "set_skip", "track_id": tid, "title": self.d.title(tid)[:60], "why": "eject", "night_s": round(self.night_s(), 1)})
        i = self._row_index(tid, ("next", "planned"))
        if i is not None:
            self._drop(self.rows[i], "ejected")
        self.replan_on_commit = f"eject: {self.d.title(tid)[:24]}"

    def on_skip(self, tid):
        tid = int(tid)
        self.skip_now(tid, "not tonight")
        self._log({"event": "set_skip", "track_id": tid, "title": self.d.title(tid)[:60], "why": "not tonight", "night_s": round(self.night_s(), 1)})
        i = self._row_index(tid, ("planned",))
        if i is not None:
            self._drop(self.rows[i], "not tonight")
            self.request(f"not tonight: {self.d.title(tid)[:24]}", user=True)

    def on_queue(self, tid):
        self.request(f"your UP NEXT: {self.d.title(int(tid))[:24]}", user=True)

    def on_dial(self, name, value):
        if name == "arc":
            return                                    # (the STORY is the arc: nothing to re-plan - effective_theme)
        if name == "length":
            from lib.dj.director import ARC_LEN_S
            self.length_min = ARC_LEN_S[value] / 60.0
        self.dead_end = False
        self.request(f"{name}: {value}", user=True)

    def on_stop(self):
        """STOP: the night's clock is kept; what was on a deck goes back to the plan (a START goes on from there)."""
        self.offset_s = self.night_s()
        self.clock0 = None
        for r in self.rows:
            if r["state"] == "now":
                r["state"] = "played"
            elif r["state"] == "next":
                r["state"] = "planned"
        self._now_id = self._next_kept = self._cut_sig = None

    # -- what is shown -------------------------------------------------------------------------------------------------------
    def steer(self):
        """The brain's gap pick: (energy, bpm) the story asks for about a minute ahead, and the crate as its pool (minus
        what played and what was skipped); no crate left -> no pool (the brain's whole library)."""
        e, b = self.target_at(self.night_s() + 60.0)
        played = {int(t) for t, _w in self.d.played}
        pool = set(self.crate_ids) - played - self.skips - set(self.d.set_skips)
        return (e, b, pool or None)

    def status(self):
        ns = self.night_s()
        L = self.length_s()
        heat, _b = self.target_at(ns)
        ci = self.chapter_at(ns)
        keys = ("id", "title", "artist", "bpm", "camelot", "energy", "chapter", "chapter_name", "start_s", "play_s", "end_s",
                "state", "why", "src", "user", "actual_start_s", "plan_start_s")
        slots = []
        for r in list(self.rows):
            s = {k: r.get(k) for k in keys}
            s["in_kind"] = (r.get("in") or {}).get("kind")
            s["n_alts"] = len(r.get("alts") or [])
            slots.append(s)
        last = self.changes[-1]["seq"] if self.changes else None
        for x in self.dropped:
            if x.get("change") == last or time.time() - float(x.get("t") or 0) < 120.0:
                slots.append({"id": x["id"], "title": x.get("title"), "state": "skipped", "why": x.get("why")})
        return {"theme": self.name, "source": self.source, "seed": self.seed, "length_s": L, "elapsed_s": round(ns, 1),
                "plan_progress": (ns / L) if L > 0 else None, "plan_heat": round(heat, 3),
                "chapter": (self.chapters[ci]["name"] if ci is not None and self.chapters else None), "chapter_i": ci,
                "chapters": list(self.chapters), "slots": slots, "warnings": list(self.warnings),
                "pending": (self.pending or {}).get("trigger"), "error": self.error, "finished": self.finished, "dead_end": self.dead_end,
                "planner_down": self.planner_down, "failures": self._fail_n,
                "changes": [{k: c.get(k) for k in ("t", "seq", "trigger", "op", "night_s")} for c in self.changes[-8:]],
                "crate": len(self.crate_ids), "keys": getattr(self, "keys_words", None) or "veto only"}

    def then_rows(self, now_bpm=None):
        """THE PLAN's THEN rows after UP NEXT: [(seconds from now, text, sub, key, id)] - the planned records in order, a
        row that moved says why, and the records the latest change took out."""
        ns = self.night_s()
        out = []
        for r in [x for x in self.rows if x["state"] == "planned"][:THEN_ROWS]:
            t = (float(r["start_s"]) - ns) if r.get("start_s") is not None else None
            bits = []
            if now_bpm and r.get("bpm"):
                bits.append(f"{(float(r['bpm']) / float(now_bpm) - 1.0) * 100:+.0f} %")
            if r.get("energy") is not None:
                bits.append(f"{float(r['energy']):.2f}")
            if r.get("user"):
                bits.append("your record")
            if r.get("why"):
                bits.append(str(r["why"]))
            out.append((t, f"{_hm(r['start_s']) if r.get('start_s') is not None else '?:??'}  {str(r.get('title') or '')[:30]} · {r.get('chapter_name') or ''}",
                        " · ".join(bits), ("then", r["id"], r.get("why")), r["id"]))
        last = self.changes[-1]["seq"] if self.changes else None
        for x in self.dropped:
            if x.get("change") == last or time.time() - float(x.get("t") or 0) < 120.0:
                out.append((None, f"({x.get('why')}) {str(x.get('title') or '')[:30]}", "", ("then-x", x["id"], x.get("why")), x["id"]))
        return out

    def _log(self, row):
        """The night log's set_plan / set_skip rows (silent under DJ_NO_PLAY_LOG), or the gate's own folder."""
        d = getattr(self.d, "plan_log_dir", None)
        if d:
            try:
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "dj_plan.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"t": round(time.time(), 2), **row}) + "\n")
            except OSError:
                pass
            return
        try:
            from lib.dj.session import log_row
            log_row(row)
        except Exception:
            pass
