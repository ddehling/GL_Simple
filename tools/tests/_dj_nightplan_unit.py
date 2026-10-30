"""THE PLANNED NIGHT'S HOOK, unit-checked with stand-ins (lib/dj/nightplan.py + the record player's pick and decode;
docs/DJ_SETS.md section 3; 2026-10-29). No library, no audio, no planner process - the real NightPlan, Director.eject,
RecordPlayer._pick and RecordPlayer._decode on small fakes, one case per finding of the phase-1 review:

  EJECT RACE    an ejected slot +1 is skipped AT ONCE: the hook the Director's next tick runs before the posted on_eject
                never hands it back; on_eject then writes the set_skip row; a refused eject takes the skip back
  EJECT LOADING a record ejected while it decodes (or decoded, not mounted) is dropped where it lands, never on a deck;
                the skip stands and the plan's next slot is offered; with no plan the brain re-picks without it; the
                opener ejected: the step picks the plan's next record as the opener and decodes it
  EJECT / MOUNT an eject that read "still loading" while the step mounted the record and planned its blend: the plan is
                cancelled, the deck unloaded, the playing record untouched (2026-10-29)
  EJECT / PLAN  the step's blend planned (another thread) at the moment the normal eject records the id: it waits for the
                eject's lock, then plans nothing on the ejected deck (2026-10-29)
  BRAIN PRE-CHECK the brain's ranking walked by the strict placement before a decode (deck "q"); none placed in
                BRAIN_PRECHECKS: its best, said, on the relaxed blend, not probed again
  CUT LANDING   a cut lands past a near-silent landmark (the first downbeat heard), never short of the runway; a blend's
                cue is untouched; the moved record's phrases count from its landmark
  GAP SLOT      the brain's gap record is a later planned slot: its row says the DJ filled a gap, moves to slot +1
  PLACED HERE   a relaxed / end label placed from the landmark a cut landed past is still "this record and cue"; the
                planner is told the cue and the phrase origin
  PROBE EJECTED a record ejected while its probe runs is not a refusal: no probe spent, never "picked again"
  PROBES SPENT  the plan's probes spent end the slot: its row (the refused alternate) goes, the trigger is the gap's, once
                per record; the brain's gap record's row says "the DJ filled a gap"
  STALE LABEL   a slot the planner labelled "relaxed" after one record is offered unprobed only after that record and cue;
                after any other record it gets the full pre-check and the strict probe
  DECODE FAIL   a record whose file will not decode is remembered (never handed over again, out of the unprobed set); the
                plan moves to the slot's alternate ("picked again: ... would not decode") instead of offering it forever
  DEAD END      "nothing follows" clears when the record changes: the plan is asked again from the new record
  BACK-OFF      a failing planner is not asked again every tick; after 3 failures it is down (said once) until you act
  SKIPPED       slots a rejoin skipped for your UP NEXT stay out of every later request
  GAP PASSES    the brain's gap pick takes a mixable record from outside the crate before a clashing one inside it

    python tools/tests/_dj_nightplan_unit.py

Exit 0 = ALL OK, 1 = FAIL.
"""
import os
import sys
import tempfile
import threading
import time

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

FAILS = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)
    return ok


class T:
    """A track: what the hook and the pre-check read."""
    def __init__(self, i, bpm=124.0, camelot="8A"):
        self.id, self.title, self.artist, self.bpm, self.camelot = i, f"rec{i}", f"artist{i}", bpm, camelot
        self.path = f"missing/{i}.mp3"
        self.has_stems = False


class R:
    def __init__(self, deck, t, state="playing"):
        self.deck, self.track, self.state, self.cue_t, self.open_clock, self.src = deck, t, state, 12.0, 0, "plan"


class P:
    """The record player's surface NightPlan reads."""
    def __init__(self, now):
        self.songs = {"a": R("a", now)}
        self.record = "a"
        self.cut = None
        self.moves = []
        self._pending, self._decoding = {}, {}
        self.avoid_ids, self._unfit, self._failed, self._ejected, self._no_probe = set(), set(), set(), set(), set()
        self._reopen = False
        self.play_min_s = 210.0
        self._plan_repicks = 0
        self.strict_calls = []
        self.strict_ok = True
        self._c = 0

    def _clock(self):
        self._c += 4410
        return self._c

    def _next_rec(self):
        return next((r for d, r in self.songs.items() if d != self.record and r.state in ("ready", "decoding")), None)

    def _busy_ids(self):
        return {r.track.id for r in self.songs.values()} | {t.id for t in self._decoding.values()} | {x[0].id for x in self._pending.values()}

    def fits_as_record(self, t):
        return None

    @staticmethod
    def _hook_answer(got):
        from lib.dj.recordplay import RecordPlayer
        return RecordPlayer._hook_answer(got)

    @staticmethod
    def key_fit(a, b):
        from lib.dj.recordplay import RecordPlayer
        return RecordPlayer.key_fit(a, b)

    def _find_blend_strict(self, rec, nxt):
        self.strict_calls.append((rec.track.id, nxt.track.id))
        self._no_blend_why = "no placement (the test)"
        return {"K": 0} if self.strict_ok else None

    def _precheck_pick(self, current, t):
        from lib.dj.recordplay import RecordPlayer
        return RecordPlayer._precheck_pick(self, current, t)        # (the real pre-check, on this stand-in's strict placement)

    def _disk_vox(self, t):
        return None

    def deck_of(self, tid):
        return next((d for d, r in self.songs.items() if r.track.id == tid), None)

    def eject(self, tid):
        d = self.deck_of(tid)
        if d is None:
            return "not on a deck"
        if d == self.record:
            return f"rec{tid} is the record playing (press NEXT to move on)"
        self.songs.pop(d, None)
        return f"rec{tid} off the decks"


class Proc:
    """The planner process: answers fail when `fail` is set, else nothing comes back (the test applies answers itself)."""
    def __init__(self):
        self.n, self.fail, self.reqs = 0, False, []

    def submit(self, req, cb):
        self.n += 1
        self.reqs.append(req)
        if self.fail:
            cb({"ok": False, "seq": req["seq"], "error": "the child crashed (the test)"})


class D:
    """The Director's surface NightPlan reads."""
    def __init__(self, p):
        self.player = p
        self.dials = {"energy": "hold", "arc": "theme", "pace": "normal", "length": "90m"}
        self.up_next, self.played, self.set_skips = [], [], set()
        self.running, self.music_root = True, ROOT
        self.plan_log_dir = tempfile.mkdtemp(prefix="dj_nightplan_unit_")
        self.events = []
        self._p = Proc()
        self.plan = None
        self.tracks = {}

    def _event(self, m):
        self.events.append(m)

    def _planner_proc(self):
        return self._p

    def track(self, i):
        return self.tracks.setdefault(int(i), T(int(i)))

    def title(self, i):
        return f"rec{i}"

    def unqueue(self, tid):
        pass

    def arc_curve(self, n):
        return [(0.0, 0.5), (1.0, 0.5)]


def row(i, state="planned", alts=(), kind="blend", prev_id=None, prev_cue=None):
    return {"id": i, "title": f"rec{i}", "planned_title": f"rec{i}", "state": state, "user": False, "src": "plan", "start_s": None,
            "end_s": None, "play_s": 240.0, "chapter": 0, "chapter_name": "Dusk", "in": {"kind": kind}, "why": None,
            "alts": [{"id": a, "energy": 0.5, "in": {"kind": kind}} for a in alts], "prev_id": prev_id, "prev_cue": prev_cue}


def night(now_id=1):
    from lib.dj.nightplan import NightPlan
    from lib.dj.setplanner import get_preset
    p = P(T(now_id))
    d = D(p)
    pl = NightPlan(d, get_preset("Desert Night"), 90)
    d.plan = pl
    pl.chapters = [{"name": "Dusk", "start_s": 0.0, "end_s": 5400.0, "energy": [0.4, 0.6], "tempo_bpm": [118, 124]}]
    pl.changes = [{"t": time.time(), "seq": 1, "trigger": "theme", "op": "plan"}]
    pl._now_id = now_id
    return d, p, pl


def eject_race():
    print("EJECT RACE")
    from lib.dj.director import Director
    d, p, pl = night()
    p.songs["b"] = R("b", d.track(2), state="ready")                 # slot +1 on its deck
    pl.rows = [row(1, "now"), row(2, "next", prev_id=1, prev_cue=12.0), row(3, prev_id=2)]
    msg = Director.eject(d, 2)                                       # the real Director.eject on the stand-in
    check(msg.endswith("off the decks"), f"the eject happened ({msg})")
    got = pl.pick(p)                                                 # THE DIRECTOR'S NEXT TICK: the hook first...
    check(got is not None and got[0].id != 2, f"the hook does not hand the ejected record back (offered {got and got[0].id})")
    pl.poll()                                                        # ...then the posted on_eject
    rows = [r["id"] for r in pl.rows]
    check(2 not in rows and any(x["id"] == 2 and x["why"] == "ejected" for x in pl.dropped), f"the row left as 'ejected' ({rows})")
    try:
        with open(os.path.join(d.plan_log_dir, "dj_plan.jsonl"), encoding="utf-8") as fh:
            logged = [ln for ln in fh if '"set_skip"' in ln and '"track_id": 2' in ln]
    except OSError:
        logged = []
    check(len(logged) == 1, f"one set_skip row for it ({len(logged)})")
    # a refused eject (the record playing) takes the skip back
    msg2 = Director.eject(d, 1)
    check("record playing" in msg2 and 1 not in pl.skips, f"a refused eject leaves no skip ({msg2[:40]}, skips {sorted(pl.skips)})")


def _loader(p, d):
    """The stand-in player with the real decode / eject / mount / pick, a decode that waits on `gate` (no file, no audio)."""
    from lib.dj.recordplay import RecordPlayer

    class DB:
        def abs(self, path):
            return path
    p.db, p.music_root, p._lock, p.last_error = DB(), ROOT, threading.Lock(), None
    p._src, p._user_ids, p._repicks, p.brain, p.steer = {}, set(), 0, None, None
    p._move = lambda kind, text, **k: p.moves.append({"t": time.time(), "kind": kind, "text": text, **k})
    p.eject = lambda tid: RecordPlayer.eject(p, tid)
    p._loading_deck = lambda tid: RecordPlayer._loading_deck(p, tid)
    p.pick_hook = lambda: _director_pick(d, p)


def _bind(p):
    """The real step's own calls on the stand-in (mount, pick, decode, the free deck)."""
    from lib.dj.recordplay import RecordPlayer
    p._mount = lambda dk: RecordPlayer._mount(p, dk)
    p._pick = lambda cur: RecordPlayer._pick(p, cur)
    p._decode = lambda t, dk, opener=False: RecordPlayer._decode(p, t, dk, opener=opener)
    p._free_deck = lambda: RecordPlayer._free_deck(p)


class Sub:
    """The submix: what was posted, in order."""
    def __init__(self):
        self.posts = []

    def post(self, e):
        self.posts.append(e)

    def post_many(self, es):
        self.posts.extend(es)


def _director_pick(d, p):
    from lib.dj.director import Director
    return Director._record_pick(d, p)                               # the real hook: UP NEXT, else the plan, else None


def _wait_decoded(p):
    t0 = time.time()
    while p._decoding and time.time() - t0 < 5.0:
        time.sleep(0.01)


def eject_loading():
    print("EJECT WHILE LOADING")
    import numpy as np
    import lib.dj.stemload as SL
    from lib.dj.director import Director
    from lib.dj.recordplay import RecordPlayer
    gate = threading.Event()
    real = SL.decode_mix
    SL.decode_mix = lambda path: (gate.wait(5.0), np.zeros((2 * 44100, 2), dtype=np.float32))[1]
    try:
        # A PLANNED NIGHT: slot +1 is decoding onto deck B when you eject it
        d, p, pl = night()
        _loader(p, d)
        pl.rows = [row(1, "now"), row(2, "next", prev_id=1, prev_cue=12.0, alts=(4,)), row(3, prev_id=2)]
        RecordPlayer._decode(p, d.track(2), "b")
        msg = Director.eject(d, 2)                                   # the real Director.eject -> the real RecordPlayer.eject
        check(msg == "ejected while loading: rec2 will not play", f"the eject says what happened ({msg})")
        check("ejected while loading: rec2 will not play" in d.events, f"and the event says it ({d.events[-1:]})")
        check(2 in pl.skips and "b" not in p.songs, f"the skip stands, the deck's entry is gone (skips {sorted(pl.skips)})")
        pl.poll()                                                    # the posted on_eject
        check(2 not in [r["id"] for r in pl.rows] and any(x["id"] == 2 and x["why"] == "ejected" for x in pl.dropped),
              f"its row left as 'ejected' ({[r['id'] for r in pl.rows]})")
        got = pl.pick(p)
        check(got is None or got[0].id != 2, f"while it still decodes the hook never hands it back ({got and got[0].id})")
        gate.set()                                                   # the decode finishes...
        _wait_decoded(p)
        # (dropped where it lands: nothing waits for the step's mount, nothing is on a deck)
        check("b" not in p._pending and p.deck_of(2) is None and not p._decoding,
              f"it never lands on a deck (pending {list(p._pending)}, songs {list(p.songs)}, decoding {list(p._decoding)})")
        got = RecordPlayer._pick(p, p.songs["a"].track)              # the step's pick on the free deck
        check(got is not None and got.id == 3, f"the plan's next slot is offered ({got and got.id})")
        p.avoid_ids = set()                                          # (the Director rewrites avoid_ids every tick: the player's own memory holds)
        p.pick_hook = lambda: (d.track(2), "plan")
        got = RecordPlayer._pick(p, p.songs["a"].track)
        check(got is None and "you ejected it tonight" in p.moves[-1]["text"], f"a later plan offer of it is refused ({p.moves[-1]['text'][:60]})")

        # NO PLAN: decoded and waiting to mount when you eject it; the brain re-picks without it
        cur = T(1)                                                   # (the gate stays open: the decode lands at once)

        class Brain:
            def __init__(self):
                self.pool_ids, self.veto_ids, self.library = None, set(), [cur, T(2), T(3)]

            def choose_next(self, current, e, bpm, bpm_target=None):
                for t in self.library:
                    if t.id != current.id and t.id not in self.veto_ids and (self.pool_ids is None or t.id in self.pool_ids):
                        return t, {}
                return None, {}
        p = P(cur)
        d = D(p)
        _loader(p, d)
        p.brain = Brain()
        p.library, p._rate_now = list(p.brain.library), (lambda deck: 1.0)
        RecordPlayer._decode(p, T(2), "b")
        _wait_decoded(p)
        check("b" in p._pending, "the decode landed (waiting for the step's mount)")
        msg = Director.eject(d, 2)
        check(msg == "ejected while loading: rec2 will not play" and "ejected while loading: rec2 will not play" in d.events, f"said truthfully ({msg})")
        check(RecordPlayer._mount(p, "b") is None and "b" not in p._pending and p.deck_of(2) is None, "what had landed is dropped, never mounted")
        got = RecordPlayer._pick(p, cur)
        check(got is not None and got.id == 3, f"the brain re-picks, without it ({got and got.id})")
        # the opener ejected while it loads: nothing plays, so the step picks another opener - the plan's next - and
        # decodes it (2026-10-29: the case used to stop at the flag, never showing a record come of it)
        gate.clear()
        d2, p2, pl2 = night()
        p2.songs, p2.record = {}, None
        pl2._now_id = None
        _loader(p2, d2)
        _bind(p2)
        pl2.rows = [row(5, "next"), row(6), row(7)]
        RecordPlayer._decode(p2, d2.track(5), "a", opener=True)
        msg = Director.eject(d2, 5)
        gate.set()
        _wait_decoded(p2)
        check(msg.startswith("ejected while loading") and p2._reopen and not p2._pending and not p2.songs, f"the opener: dropped ({msg}, reopen {p2._reopen})")
        p2.running = True
        RecordPlayer.step(p2)                                        # the step: no record, the room let go - another opener
        dec = p2._decoding.get("a") or (p2._pending.get("a") or (None,))[0]
        check(not p2._reopen and p2._first is not None and p2._first.id == 6 and dec is not None and dec.id == 6,
              f"the step picks the plan's next record as the opener and decodes it ({p2._first and p2._first.id}, decoding {dec and dec.id})")
        _wait_decoded(p2)
        t6 = d2.track(6)
        t6.grid, t6.gain_db, t6.nearest_downbeat = [], 0.0, (lambda x: 0.0)
        p2.submix = Sub()
        r6 = RecordPlayer._mount(p2, "a")
        check(r6 is not None and r6.track.id == 6 and r6.state == "ready" and p2.deck_of(5) is None,
              f"it lands on deck A, ready to open; the ejected one nowhere ({r6 and r6.track.id}, songs {list(p2.songs)})")
    finally:
        gate.set()
        SL.decode_mix = real


def _next_track(i):
    t = T(i)
    t.grid, t.gain_db, t.nearest_downbeat = [], 0.0, (lambda x: 0.0)
    return t


def eject_mount():
    print("EJECT / MOUNT RACE")
    import numpy as np
    from lib.dj.recordplay import RecordPlayer

    def stand_in():
        p = P(T(1))
        d = D(p)
        _loader(p, d)
        p.submix = Sub()
        p.landmark = lambda t: 0.0
        return p
    # (1) THE STALE READ: eject reads "still loading onto deck B", then - the forced interleaving - the step mounts it
    # and plans the blend on deck B before eject acts. (The lock is re-entrant here only, so the mount runs on this
    # thread inside eject's read: live, _mount and that read share one lock and cannot interleave like this at all.)
    p = stand_in()
    p._lock = threading.RLock()
    nt = _next_track(2)
    p._pending["b"] = (nt, np.zeros((10, 2), np.float32), None, None)
    p.songs["b"] = R("b", nt, state="decoding")

    def stale(tid):
        ld = RecordPlayer._loading_deck(p, tid)
        RecordPlayer._mount(p, "b")
        p.cut = {"a": "a", "b": "b", "state": "planned", "txn": 7, "kind": "blend"}
        return ld
    p._loading_deck = stale
    msg = RecordPlayer.eject(p, 2)
    cmds = [e.get("cmd") for e in p.submix.posts if e.get("cmd") in ("load", "unload")]
    check(p.cut is None and any(e.get("cmd") == "cancel" and e.get("txn") == 7 for e in p.submix.posts),
          f"the blend planned on it is cancelled ({msg}; cut {p.cut})")
    check(cmds[-1:] == ["unload"] and "load" in cmds and p.deck_of(2) is None and 2 in p._ejected,
          f"its deck is unloaded after the load, no deck holds it ({cmds}, songs {list(p.songs)})")
    check(p.record == "a" and p.songs["a"].state == "playing" and not any(r.state == "leaving" for r in p.songs.values()),
          "the playing record is untouched (never sent 'leaving')")
    # (2) LIVE TOO: mounted and planned while its decode thread's entry has not let go yet - the read finds it "decoding"
    p = stand_in()
    nt = _next_track(2)
    p.songs["b"] = R("b", nt, state="ready")
    p._decoding["b"] = nt
    p.cut = {"a": "a", "b": "b", "state": "planned", "txn": 8, "kind": "blend"}
    msg = RecordPlayer.eject(p, 2)
    check(msg.endswith("off the decks") and p.cut is None and [e.get("cmd") for e in p.submix.posts] == ["cancel", "unload"],
          f"ready on its deck: the normal eject, the plan cancelled ({msg}, {[e.get('cmd') for e in p.submix.posts]})")


def eject_plan():
    print("EJECT / PLAN RACE")
    from lib.dj.recordplay import RecordPlayer
    # THE FORCED INTERLEAVING (the review's eject_plan_race, 2026-10-29): the planner tab ejects slot +1 on the GUI thread
    # while the Director's thread plans its blend. The step's _plan_blend is started on another thread at the very moment
    # eject records the id (after its "no plan on this deck" check, before the unload): it must wait for the eject, then
    # see it and plan nothing - unlocked, its events stayed queued on the unloaded deck and the record playing faded out
    p = P(T(1))
    d = D(p)
    _loader(p, d)
    p.submix = Sub()
    p._rate_now = lambda deck: 1.0
    p._tempo_ratio = lambda rec, nxt: 1.0
    p._still_next = lambda n: RecordPlayer._still_next(p, n)          # (the real re-check under the lock)
    nt = _next_track(2)
    nb = p.songs["b"] = R("b", nt, state="ready")
    box = {}

    class EjectSet(set):
        armed = True

        def add(self, x):
            if EjectSet.armed:
                EjectSet.armed = False
                th = threading.Thread(target=lambda: box.__setitem__("planned", RecordPlayer._plan_blend(p, nb, why="the step")), name="director")
                th.start()
                th.join(0.3)
                box["waited"] = th.is_alive()      # (still waiting for the lock eject holds)
                box["th"] = th
            return set.add(self, x)
    p._ejected = EjectSet()
    msg = RecordPlayer.eject(p, 2)
    box["th"].join(5.0)
    cmds = [(e.get("cmd"), e.get("deck"), e.get("txn")) for e in p.submix.posts]
    check(msg.endswith("off the decks") and box.get("waited") is True, f"the step's plan waited for the eject ({msg}; waited {box.get('waited')})")
    check(box.get("planned") is False and p.cut is None and cmds == [("unload", "b", None)],
          f"then it saw the eject and planned nothing on the ejected deck (planned {box.get('planned')}, cut {p.cut}, posts {cmds})")
    check(p.record == "a" and p.songs["a"].state == "playing" and "b" not in p.songs, "the record playing is untouched")
    # the other order: the plan first (posted under the lock), then the eject - it cancels the plan, as before
    p = P(T(1))
    d = D(p)
    _loader(p, d)
    p.submix = Sub()
    nb = p.songs["b"] = R("b", _next_track(2), state="ready")
    p.cut = {"a": "a", "b": "b", "state": "planned", "txn": "blend9", "kind": "blend"}
    msg = RecordPlayer.eject(p, 2)
    check(p.cut is None and [e.get("cmd") for e in p.submix.posts] == ["cancel", "unload"], f"a plan made first is cancelled ({msg})")


def brain_precheck():
    print("BRAIN PRE-CHECK")
    from lib.dj.recordplay import BRAIN_PRECHECKS, RecordPlayer
    # the brain's ranking walked by the strict placement BEFORE a decode (2026-10-29): the seed 3 gap pick spent the
    # brain's two re-picks on decodes the rules refused, and the third came in unprobed on the relaxed blend
    cur = T(1)
    lib = [cur] + [T(i) for i in range(2, 12)]

    class Brain:
        def __init__(self):
            self.pool_ids, self.veto_ids, self.library = None, set(), lib

        def choose_next(self, current, e, bpm, bpm_target=None):
            for t in self.library:
                if t.id != current.id and t.id not in self.veto_ids and (self.pool_ids is None or t.id in self.pool_ids):
                    return t, {}
            return None, {}

    def player(refuse):
        p = P(cur)
        p.library, p.brain, p.pick_hook = list(lib), Brain(), None
        p._src, p.last_error, p.steer = {}, None, None
        p._rate_now = lambda deck: 1.0
        p._move = lambda kind, text, **k: p.moves.append({"kind": kind, "text": text, **k})
        calls = []

        def strict(rec, nxt):
            calls.append((nxt.deck, nxt.track.id))
            p._no_blend_why = "two singers (the test)"
            return None if nxt.track.id in refuse else {"K": 0}
        p._find_blend_strict = strict
        return p, calls
    p, calls = player({2, 3})
    got = RecordPlayer._pick(p, cur)
    check(got is not None and got.id == 4 and [c[1] for c in calls] == [2, 3, 4] and all(c[0] == "q" for c in calls),
          f"the first record the rules place is picked, checked on deck 'q' before any decode ({got and got.id}, {calls})")
    check({2, 3} <= p._unfit and 4 not in p._no_probe and p._src.get(4) == "brain", f"the refused ones are unfit with the record playing ({sorted(p._unfit)})")
    # nothing the brain offers can be placed: its best, said, on the relaxed blend - and not decoded twice to be told so
    p, calls = player(set(range(2, 12)))
    got = RecordPlayer._pick(p, cur)
    check(got is not None and got.id == 2 and len(calls) == BRAIN_PRECHECKS and 2 in p._no_probe
          and "blends by the rules" in p.moves[-1]["text"], f"none placed in {BRAIN_PRECHECKS}: the best one, said ({got and got.id}, "
          f"{len(calls)} checks, {p.moves[-1]['text'][:70] if p.moves else None})")


class LT:
    """A track for the real geometry: 120 bpm (2 s bars) from 0.0, no sections, `dur` seconds."""
    def __init__(self, i, dur=400.0):
        self.id, self.title, self.artist, self.bpm, self.camelot = i, f"rec{i}", "", 120.0, "8A"
        self.duration_s, self.gain_db, self.downbeat_offset, self.period_s = float(dur), 0.0, 0, 0.5
        self.grid = [{"start_s": 0.0, "end_s": float(dur), "period_s": 0.5, "first_beat_s": 0.0}]
        self.sections = []
        self.path = f"missing/{i}.mp3"

    def nearest_downbeat(self, t):
        return max(0.0, round(float(t) / 2.0) * 2.0)


def cut_landing():
    print("CUT LANDING")
    import numpy as np
    from lib.dj.recordplay import RecordPlayer, Rec
    # THE CUT LANDS PAST THE SILENCE (the user's pick, 2026-10-29): only a cut moves; its phrases count from the landmark
    rp = RecordPlayer.__new__(RecordPlayer)
    rp.play_min_s = 210.0
    rp.songs, rp.record, rp.submix = {}, None, Sub()
    rp.submix.telemetry, rp.submix.clock = {"clock": 0, "decks": {"a": {"time_s": 100.0, "rate": 1.0}}}, 0
    rp.cut, rp._seq, rp.moves, rp._lock = None, 0, [], threading.Lock()
    rp._next_asap, rp.wait_why, rp._settled_clock, rp.hole_beats = False, "", 0, 1.0
    rp._repicks = rp._plan_repicks = 0
    rp._unfit, rp.brain = set(), None
    env_quiet = np.full(800, 0.2, dtype=np.float16)
    env_quiet[:8] = 1e-5                                     # its first 4 s near-silent, loud from 4 s
    tb = LT(2)
    r = Rec("b", tb)
    check(rp.cut_landing(r) == (0.0, 0.0), f"no envelope measured: the landmark ({rp.cut_landing(r)})")
    r.mix_env = np.full(800, 0.2, dtype=np.float16)
    check(rp.cut_landing(r) == (0.0, 0.0), f"loud at its landmark: the landmark ({rp.cut_landing(r)})")
    r.mix_env = env_quiet
    check(rp.cut_landing(r) == (4.0, 0.0), f"near-silent for 4 s: the first downbeat whose first beat is heard, 4.0 s ({rp.cut_landing(r)})")
    short = Rec("b", LT(3, dur=233.0))
    short.mix_env = env_quiet[:466]
    check(rp.cut_landing(short) == (0.0, 0.0), f"a landing that would leave under play_min + 20 s: the landmark stays ({rp.cut_landing(short)})")

    def fresh():
        ra = Rec("a", LT(1))
        ra.cue_t, ra.state, ra.open_bar = 0.0, "playing", 0
        rb = Rec("b", tb)
        rb.state, rb.mix_env = "ready", env_quiet
        rp.songs, rp.record, rp.cut = {"a": ra, "b": rb}, "a", None
        rp.submix.posts = []
        return ra, rb
    ra, rb = fresh()
    ok = rp._plan_blend(rb, why="the test")
    check(ok and rp.cut["kind"] == "blend" and rb.cue_t == rp.cut["cue_b"] and rb.phrase_t is None,
          f"a blend's cue is its own (no silence rule): cue {rb.cue_t}, phrase {rb.phrase_t}")
    ra, rb = fresh()
    ok = rp._plan_cut(rb, why="the test")
    cue = [e for e in rp.submix.posts if e.get("cmd") == "cue" and e.get("deck") == "b"]
    check(ok and rp.cut["cue_b"] == 4.0 and rp.cut["lm0"] == 0.0 and rb.cue_t == 4.0 and rb.phrase_t == 0.0 and cue and cue[0]["time_s"] == 4.0,
          f"a cut lands past the silence: cued {cue and cue[0]['time_s']}, phrases from {rb.phrase_t}")
    check("past its silent start" in rp.moves[-1]["text"], f"and says so ({rp.moves[-1]['text'][:80]})")
    RecordPlayer._swapped(rp, rp.cut["clock"])
    rp.submix.telemetry = {"clock": 0, "decks": {"b": {"time_s": 30.0, "rate": 1.0}}}
    g = rp._geometry(rb)
    check(rp.record == "b" and rb.open_bar == 0 and g[4] == 0, f"its phrases count from the landmark, not the moved cue (open bar {rb.open_bar}, geometry {g[4]})")
    ra, rb = fresh()
    rb.phrase_t = 0.0                                        # (a cut planned, then replaced by a blend: the blend's cue is the origin)
    rp._plan_blend(rb, why="the test")
    check(rb.phrase_t is None, "a blend that replaces a planned cut drops the cut's phrase origin")
    # the mount (the opener, and a next record) cues as before, whatever its envelope says: the silence rule is the cut's
    rp._pending, rp._decoding, rp._ejected, rp._no_probe, rp._src = {}, {}, set(), set(), {}
    rp.songs, rp.record = {}, None
    rp._pending["a"] = (LT(4), np.zeros((10, 2), np.float32), None, env_quiet)
    r4 = rp._mount("a")
    check(r4 is not None and r4.cue_t == 0.0 and r4.phrase_t is None, f"the opener's mount: its first downbeat ({r4 and r4.cue_t})")
    r4.state, rp.record = "playing", "a"
    rp._pending["b"] = (LT(5), np.zeros((10, 2), np.float32), None, env_quiet)
    r5 = rp._mount("b")
    check(r5 is not None and r5.cue_t == rp.landmark(r5.track) and r5.phrase_t is None, f"a next record's mount: its landmark ({r5 and r5.cue_t})")


def gap_takes_later_slot():
    print("GAP TAKES A LATER SLOT")
    # the brain's gap record is a slot the plan had later (the plan gate's seed 1 F, 2026-10-29): its row says the DJ
    # filled a gap and comes forward to slot +1, and the plan is asked again from it
    d, p, pl = night()
    pl.rows = [row(1, "now"), row(2), row(3), row(4)]
    pl.gap_why = "could not blend after rec1, nor its alternates"
    p.songs["b"] = R("b", d.track(3), state="ready")
    p.songs["b"].src = "brain"
    pl._sync_next(p, p.songs["a"], p.songs["b"])
    r3 = next(r for r in pl.rows if r["id"] == 3)
    check([r["id"] for r in pl.rows] == [1, 3, 2, 4] and r3["state"] == "next" and r3["src"] == "brain"
          and str(r3["why"]).startswith("the DJ filled a gap (could not blend after rec1"), f"its row ({[r['id'] for r in pl.rows]}, {r3['why']})")
    check(bool(pl.replan_on_commit), f"and the rest is asked again once it is kept ({pl.replan_on_commit})")


def placed_here_landed():
    print("PLACED HERE, LANDED PAST")
    from lib.dj.nightplan import NightPlan
    # a record a cut landed past its silent start (cue 16.0, phrases from its landmark 12.0): the planner, which brings it
    # in whole at the landmark, placed the next seam from cue 12.0 - that is still "from this record and cue"
    d, p, pl = night(now_id=4)
    rec = p.songs["a"]
    rec.cue_t, rec.phrase_t = 16.0, 12.0
    check(NightPlan._placed_here(row(6, kind="end", prev_id=4, prev_cue=12.0), rec), "placed from its landmark: honoured")
    check(NightPlan._placed_here(row(6, prev_id=4, prev_cue=16.0), rec), "placed from its cue: honoured")
    check(not NightPlan._placed_here(row(6, prev_id=4, prev_cue=13.0), rec), "placed from anywhere else: not")
    com = pl.committed()
    check(com and com[0].get("cue_s") == 16.0 and com[0].get("phrase_s") == 12.0, f"the planner is told both ({com[:1]})")


def probe_ejected():
    print("PROBE ON AN EJECTED RECORD")
    import numpy as np
    import lib.dj.stemload as SL
    from lib.dj.recordplay import RecordPlayer
    real = SL.decode_mix
    SL.decode_mix = lambda path: np.zeros((2 * 44100, 2), dtype=np.float32)
    try:
        p = P(T(1))
        d = D(p)
        _loader(p, d)
        p._src = {2: "plan"}

        def strict(rec, nxt):
            RecordPlayer.eject(p, 2)                                 # you eject it while the probe runs
            p._no_blend_why = "the test"
            return None
        p._find_blend_strict = strict
        RecordPlayer._decode(p, T(2), "b")
        _wait_decoded(p)
        check(p._plan_repicks == 0 and p._repicks == 0 and 2 not in p._unfit, f"your eject is not a refusal: no probe spent ({p._plan_repicks}, unfit {p._unfit})")
        check("b" not in p._pending and p.deck_of(2) is None and not any("picked again" in m["text"] for m in p.moves),
              "it never lands, and nothing says it was picked again")
    finally:
        SL.decode_mix = real


def probes_spent():
    print("PROBES SPENT")
    from lib.dj.recordplay import PLAN_PROBES
    d, p, pl = night()
    pl.rows = [row(1, "now"), row(2, prev_id=1, prev_cue=12.0, alts=(4,)), row(3, prev_id=2)]
    # the slot's last alternate was refused: its row holds it, 'planned' again, with its "picked again" trigger
    r2 = pl.rows[1]
    r2.update({"id": 4, "title": "rec4", "alt_of": 2, "why": "picked again: rec2 could not blend after rec1, rec4 instead"})
    pl.replan_on_commit = r2["why"]
    p._plan_repicks, p._unfit = PLAN_PROBES, {2, 4}
    got = pl.pick(p)
    check(got is None and [r["id"] for r in pl.rows] == [1, 3] and any(x["id"] == 4 and "probes are spent" in x["why"] for x in pl.dropped),
          f"the slot's row goes, the refused record with it ({[r['id'] for r in pl.rows]})")
    check(str(pl.replan_on_commit).startswith(f"could not blend after rec1: the plan's {PLAN_PROBES} probes are spent"),
          f"the re-search trigger is the gap's, not the stale 'picked again' ({pl.replan_on_commit})")
    got = pl.pick(p)                                                 # the brain's gap pick refused in turn: asked again
    check(got is None and [r["id"] for r in pl.rows] == [1, 3], f"once per record: the next slot is not this one's to lose ({[r['id'] for r in pl.rows]})")
    p.songs["b"] = R("b", d.track(9), state="ready")                 # the brain's gap record on deck B
    p.songs["b"].src = "brain"
    pl._sync_next(p, p.songs["a"], p.songs["b"])
    r9 = next((r for r in pl.rows if r["id"] == 9), None)
    check(r9 is not None and str(r9["why"]).startswith("the DJ filled a gap (could not blend after rec1") and "probes are spent" in r9["why"],
          f"its row says the DJ filled a gap ({r9 and r9['why']})")


def stale_label():
    print("STALE LABEL")
    d, p, pl = night(now_id=5)                                       # record 5 plays; the label was placed after record 4
    pl.rows = [row(4, "played"), row(5, "now"), row(6, kind="relaxed", prev_id=4, prev_cue=30.0)]
    got = pl.pick(p)
    check(got is not None and got[0].id == 6 and got[2] is True, f"after another record: offered WITH the probe ({got and got[2]})")
    check(p.strict_calls == [(5, 6)], f"and the full pre-check ran the strict placement ({p.strict_calls})")
    # after the very record and cue the planner placed it from: the one unprobed seam
    d, p, pl = night(now_id=4)
    pl.rows = [row(4, "now"), row(6, kind="relaxed", prev_id=4, prev_cue=12.0)]
    got = pl.pick(p)
    check(got is not None and got[2] is False and not p.strict_calls, f"after its own record and cue: unprobed, no pre-check ({got and got[2]}, {p.strict_calls})")
    # after another record the strict placement refuses it: not offered, its alternate is
    d, p, pl = night(now_id=5)
    p.strict_ok = False
    pl.rows = [row(5, "now"), row(6, kind="relaxed", prev_id=4, prev_cue=30.0, alts=(7,))]
    got = pl.pick(p)
    check(got is None and p.strict_calls == [(5, 6), (5, 7)], f"refused by the strict pre-check: neither offered, the DJ picks ({got}, {p.strict_calls})")


def decode_fail():
    print("DECODE FAIL")
    from lib.dj.recordplay import RecordPlayer
    # the player's side: the decode thread fails -> remembered, out of the unprobed set
    p = P(T(1))
    p.songs = {"a": R("a", T(1))}
    p._no_probe, p._src, p._lock, p.last_error = {9}, {9: "plan"}, threading.Lock(), None
    p.music_root = ROOT

    class DB:
        def abs(self, path):
            raise FileNotFoundError(path)
    p.db = DB()
    p._move = lambda kind, text, **k: p.moves.append({"t": time.time(), "kind": kind, "text": text, **k})
    RecordPlayer._decode(p, T(9), "b")
    t0 = time.time()
    while p._decoding and time.time() - t0 < 5.0:
        time.sleep(0.01)
    check(9 in p._failed and 9 not in p._no_probe and "b" not in p.songs, f"a failed decode is remembered ({p._failed}, no_probe {p._no_probe})")
    # ...and the hook path refuses it from then on, whoever offers it
    p.pick_hook = lambda: (T(9), "plan")
    p.brain, p._user_ids, p._repicks = None, set(), 0
    p._move = lambda kind, text, **k: p.moves.append({"t": time.time(), "kind": kind, "text": text, **k})
    got = RecordPlayer._pick(p, T(1))
    check(got is None and "would not decode" in p.moves[-1]["text"], f"the hook's offer of it is refused ({p.moves[-1]['text'][:60]})")
    # the plan's side: the slot moves on to its alternate, once, with the words
    d, p2, pl = night()
    pl.rows = [row(1, "now"), row(9, prev_id=1, prev_cue=12.0, alts=(10,))]
    p2.moves = [{"t": time.time(), "kind": "error", "text": "decode rec9: FileNotFoundError: x", "to_id": 9}]
    pl._read_moves(p2)
    offers = [pl.pick(p2) for _ in range(3)]
    ids = [o[0].id if o else None for o in offers]
    check(ids[0] == 10 and 9 not in ids, f"the alternate, never the record that would not decode (offers {ids})")
    r = next(x for x in pl.rows if x["id"] == 10)
    check(str(r["why"]).startswith("picked again: rec9 would not decode"), f"its row says so ({r['why']})")
    req, *_x = pl._req()
    check(9 in req["exclude_ids"], "and the planner is told to leave it out")


def dead_end():
    print("DEAD END")
    d, p, pl = night(now_id=1)
    pl.rows = [row(1, "now")]
    pl.dead_end = True
    pl.step()
    check(d._p.n == 0, f"nothing is asked while the same record plays after a dead end ({d._p.n})")
    p.songs = {"b": R("b", d.track(2))}                              # the record changed (the brain's gap record)
    p.record = "b"
    pl.step()
    check(not pl.dead_end and d._p.n == 1, f"a new record: the plan is asked again from it (dead_end {pl.dead_end}, {d._p.n} request)")


def backoff():
    print("BACK-OFF")
    d, p, pl = night()
    pl.rows = [row(1, "now")]
    d._p.fail = True
    pl.request("theme", user=True)
    for _ in range(50):                                              # 50 Director steps = 10 s of a live night at 5 Hz
        pl.step()
    said = sum("could not be made" in e for e in d.events)
    check(d._p.n == 1 and said == 1, f"a failed answer is not asked again every tick ({d._p.n} requests, said {said}x)")
    for _ in range(4):
        pl._fail_t -= 400.0                                          # (the back-off waited out)
        for _ in range(3):
            pl.step()
    down = sum("not available" in e for e in d.events)
    check(pl.planner_down and down == 1 and d._p.n == 3, f"after 3 failures the planner is down, said once ({d._p.n} requests, said {down}x)")
    for _ in range(20):
        pl.step()
    check(d._p.n == 3, f"down: nothing automatic asks it ({d._p.n})")
    d._p.fail = False
    pl.on_dial("energy", "lift")
    check(d._p.n == 4 and not pl.planner_down, f"something you do asks it again ({d._p.n}, down {pl.planner_down})")


def skipped():
    print("SKIPPED FOR YOUR RECORD")
    d, p, pl = night()
    pl.rows = [row(1, "now"), row(3), row(4), row(5)]
    d.up_next = [7]
    pl.request("your UP NEXT: rec7", user=True)
    req = d._p.reqs[-1]
    slots = [{"id": 1, "fixed": True}, {"id": 7, "fixed": True, "user": True}, {"id": 5}]
    resp = {"ok": True, "seq": req["seq"], "slots": slots, "skipped": [{"id": 3, "title": "rec3", "why": "skipped for your record"},
                                                                       {"id": 4, "title": "rec4", "why": "skipped for your record"}],
            "rejoined": True, "chapters": pl.chapters, "finished": True}
    pl._apply(resp)
    d.up_next = []
    pl.on_dial("energy", "lift")
    ex = d._p.reqs[-1]["exclude_ids"]
    check(3 in ex and 4 in ex, f"the slots skipped for your record stay out of later requests ({ex})")


def gap_passes():
    print("GAP PASSES")
    from lib.dj.recordplay import RecordPlayer
    cur = T(1, bpm=124.0, camelot="8A")
    clash_in_crate = T(2, bpm=124.0, camelot="3B")                  # in the crate, keys clash with the record playing
    mix_outside = T(3, bpm=123.0, camelot="8A")                      # outside the crate, mixes

    class Brain:
        def __init__(self):
            self.pool_ids, self.veto_ids, self.library = None, set(), [cur, clash_in_crate, mix_outside]

        def choose_next(self, current, e, bpm, bpm_target=None):
            for t in self.library:
                if t.id != current.id and t.id not in self.veto_ids and (self.pool_ids is None or t.id in self.pool_ids):
                    return t, {}
            return None, {}

    p = P(cur)
    p.library = [cur, clash_in_crate, mix_outside]
    p.brain = Brain()
    p.pick_hook = None
    p._src, p.last_error = {}, None
    p._rate_now = lambda deck: 1.0
    p.steer = lambda: (0.5, None, {2})
    p._move = lambda kind, text, **k: p.moves.append({"kind": kind, "text": text})
    got = RecordPlayer._pick(p, cur)
    check(got is not None and got.id == 3, f"a mixable record outside the crate before a clashing one inside it ({got and got.id})")
    check(p.brain.pool_ids is None, "the brain's own pool is back afterwards")
    p.library = [cur, clash_in_crate]
    p.brain.library = [cur, clash_in_crate]
    got = RecordPlayer._pick(p, cur)
    check(got is not None and got.id == 2 and "will be a cut" in p.moves[-1]["text"], "nothing mixable anywhere: the crate's clashing record, said")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for fn in (eject_race, eject_loading, eject_mount, eject_plan, probe_ejected, probes_spent, stale_label, decode_fail, dead_end, backoff,
               skipped, gap_passes, brain_precheck, cut_landing, placed_here_landed, gap_takes_later_slot):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(False, f"{fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(FAILS)} failures")
    print("ALL OK" if not FAILS else "FAIL")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
