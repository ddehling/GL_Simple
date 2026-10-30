"""THE NIGHT ON THE WEB PAGE - the planned night's buttons, strip and story lines through the show's own action path
(docs/DJ_SETS.md section 4 "Web /dj page (phone)" and phase 4's gate; 2026-10-29).

Drives Stories_OGL.EnvironmentalSystem's real DJ bridge (_apply_dj_controls: the action queue, _director_night,
_director_start / _director_adopt / _director_stop, the 5 Hz mirror) on a stand-in show object - a fake web controller
that records every dj_info it is handed, no GLFW, no audio device (the engine's render-ahead ring drained by a thread in
the device's place, as _dj_plan_gate.py's controls do). A thread named "render" ticks the bridge at 5 Hz, as the show's
render loop does; presses go through it.

  BEFORE START  a night pressed with no Director is ARMED (published, lit); LENGTH with none on sets the next press's
                length; clear un-arms; ★ DIRECTOR then plans it in the start worker (never on the render thread), waits
                for it, and opens on the plan's first record, at the armed length
  DURING        (a running record-mode night, a record playing and slot +1 kept on its deck) another night pressed, a
                LENGTH the dial has no button for (120), one it has (180 -> the LENGTH dial), the 6 h night, clear, a
                night again, then NEXT:
                  - every press costs the render thread under 5 ms of its own CPU (the wall span printed beside it)
                  - the request and the answer's apply run on the Director's own thread ("director")
                  - the published night part (dj_night) changes within 1 s of the planner's answer
                  - the plan's now / next rows are the decks' record and slot +1; neither changes under a press; NEXT
                    brings in the kept slot +1
                  - clear: no plan, dj_night / night_tick published as None
  DROPPED       STOP closes the planner's process (off the render thread); a STOP during the start build, and a start
                that fails, close it too; a STOP in the armed night's plan wait gives up at once and never attaches, and
                ★ DIRECTOR pressed again then starts; a night press and STOP on one tick start no planner child after the
                STOP; at the end no planner child is alive
  COST          with the 6 h plan applied: the night's own part of the show tick (the payload builder + its hash) in the
                render thread's own CPU (QueryThreadCycleTime on Windows - a wall span counts the OS descheduling this
                BelowNormal gate, so it is printed, not judged), the whole mirror; the dj_info size per tick before (the
                full set_plan inside `director`, every tick) and
                after (dj_night only when it changes, night_tick every tick)
  PAYLOAD       the page's merged view (the deltas merged like dj_panel.html does) saved at 90 min and at 6 h, for the
                static harness page (--out)

    python tools/tests/_dj_web_night.py [--music D:/Devel/music] [--out DIR] [--preset "Desert Night"] [--other "Deep Space"]

Below normal priority. Prints ALL OK or FAIL (exit 1).
"""
import json
import os
import sys
import threading
import time
import types

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def below_normal():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        except Exception:  # noqa: BLE001
            pass


class FakeWeb:
    """The web controller as the bridge sees it: the action queue and set(); every dj_info kept with its time, and the
    page's merged view (dj_panel.html merges each delta over the last full picture)."""

    def __init__(self):
        self.control_dict = {}
        self._dict_lock = threading.Lock()
        self.published = []                # (t, info)
        self.page = {}
        self.night_pubs = []               # (t, dj_night) every time the key was sent

    def set(self, key, value):
        with self._dict_lock:
            self.control_dict[key] = value
        if key == "dj_info":
            t = time.time()
            self.published.append((t, value))
            del self.published[:-4000]
            self.page.update(value)
            if "dj_night" in value:
                self.night_pubs.append((t, value["dj_night"]))


def make_show(music, engine):
    from Stories_OGL import EnvironmentalSystem as ES
    show = ES.__new__(ES)
    show.web_controller = FakeWeb()
    show.dj_cfg = {"enabled": True, "music_dir": music, "theme": "groove"}
    show.scheduler = types.SimpleNamespace(state={"soundengine": engine})
    show.analyzer = None
    show._dj = None
    show._remix = None
    show._dj_last_error = ""
    show._dj_prev_source = None
    show._dj_pending_setlist = None
    show._dj_pending_arc = []
    show._dj_pending_nudge = 0.0
    show._dj_pending_flavor = {}
    show.active_effects = {}
    far = time.time() + 1e9                # (the idle vocabulary and the disk info: never refreshed - DB walks not under test)
    show._dj_idle_vocab = (far, [], [])
    show._dj_disk_cache = (far, [], music)
    return show


class Render(threading.Thread):
    """The show's render loop, as far as the DJ bridge goes: _apply_dj_controls at 5 Hz; a press is queued for the next
    tick and its cost read back from the bridge's own per-action timing."""

    def __init__(self, show):
        super().__init__(daemon=True, name="render")
        self.show = show
        self.stop = False
        self._q = []
        self._lock = threading.Lock()
        self.ticks = []                    # {t, ms, night_ms, mirror_ms, night_cpu_ms}
        self.errors = []
        self.after = None                  # called with each tick's row, on this thread, outside the bridge's own timing

    def press(self, action, value=None, timeout=5.0):
        ev = threading.Event()
        box = {}
        with self._lock:
            self._q.append((action, value, ev, box))
        ev.wait(timeout)
        return box.get("ms")

    def press_many(self, items, timeout=5.0):
        """Several presses handled in ONE tick, in order (a night press and STOP on the same render tick)."""
        evs = []
        with self._lock:
            for action, value in items:
                ev = threading.Event()
                evs.append(ev)
                self._q.append((action, value, ev, {}))
        for ev in evs:
            ev.wait(timeout)

    def run(self):
        show = self.show
        while not self.stop:
            t0 = time.perf_counter()
            with self._lock:
                q, self._q = self._q, []
            if q:
                with show.web_controller._dict_lock:
                    show.web_controller.control_dict.setdefault("request_dj_actions", []).extend((a, v) for a, v, _e, _b in q)
            try:
                show._apply_dj_controls()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.errors.append(f"{type(e).__name__}: {e}")
            ms = (time.perf_counter() - t0) * 1000.0
            det = getattr(show, "_perf_dj_detail", {}) or {}
            row = {"t": time.time(), "ms": ms, "night_ms": det.get("dj_night"), "mirror_ms": det.get("dj_mirror")}
            if self.after is not None:
                try:
                    self.after(row)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"after: {type(e).__name__}: {e}")
            self.ticks.append(row)
            for a, _v, ev, box in q:
                box["ms"] = det.get(f"dj_act:{a}")
                ev.set()
            time.sleep(max(0.0, 0.2 - (time.perf_counter() - t0)))


def cpu_clock():
    """The calling thread's own CPU time in seconds, at sub-millisecond resolution (2026-10-29: the 5 ms checks were wall
    spans, so a BelowNormal gate descheduled for one 15.6 ms quantum beside the live app failed them; time.thread_time is
    no help on Windows, its GetThreadTimes ticks in those same quanta). Windows: QueryThreadCycleTime, calibrated against
    perf_counter on a busy loop (a descheduled calibration only makes the cost read higher). Elsewhere: thread_time."""
    if sys.platform != "win32":
        return time.thread_time
    import ctypes
    k = ctypes.windll.kernel32
    k.GetCurrentThread.restype = ctypes.c_void_p              # (a 64-bit pseudo-handle: as a C int the call failed, all zeros)
    k.QueryThreadCycleTime.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    k.QueryThreadCycleTime.restype = ctypes.c_int
    tls = threading.local()

    def cycles():
        c = getattr(tls, "c", None)
        if c is None:
            c = tls.c = ctypes.c_ulonglong()
        if not k.QueryThreadCycleTime(k.GetCurrentThread(), ctypes.byref(c)):
            raise OSError("QueryThreadCycleTime failed")
        return c.value
    hz = 0.0
    for _ in range(3):
        c0, t0 = cycles(), time.perf_counter()
        while time.perf_counter() - t0 < 0.05:
            pass
        hz = max(hz, (cycles() - c0) / max(time.perf_counter() - t0, 1e-9))
    if hz < 1e8:
        raise RuntimeError(f"the thread cycle clock reads {hz:.0f} Hz: not usable")
    print(f"  (the CPU clock: QueryThreadCycleTime at {hz / 1e9:.2f} GHz)", flush=True)
    return lambda: cycles() / hz


def wait_for(cond, timeout, step=0.05):
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(step)
    return False


def start_audio(engine):
    """The engine as live: its render-ahead producer and ring, a thread in the device callback's place draining it in
    real time (the device is never opened) - _dj_plan_gate.py's controls."""
    import numpy as np
    from lib.dj.submix import RATE
    engine._ring = np.zeros((engine._ring_capacity, 2), dtype=np.float32)
    engine._ring_w = engine._ring_r = 0
    engine._ring_stop.clear()
    engine._producer = threading.Thread(target=engine._render_ahead, daemon=True, name="audio-render")
    engine._producer.start()
    cons = {"stop": False}
    WIN = RATE // 100

    def consumer():
        feed = engine._device_feed()
        next(feed)
        t0 = time.perf_counter()
        pulled = 0
        while not cons["stop"]:
            due = int((time.perf_counter() - t0) * RATE)
            while due - pulled >= WIN:
                feed.send(WIN)
                pulled += WIN
            time.sleep(0.002)
    threading.Thread(target=consumer, daemon=True, name="device-stand-in").start()
    return cons


def main():
    below_normal()
    music = arg("--music", "D:/Devel/music")
    preset = arg("--preset", "Desert Night")
    other = arg("--other", "Deep Space")
    out_dir = arg("--out", os.path.join(ROOT, "logs", "web_night"))
    os.makedirs(out_dir, exist_ok=True)
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    from lib.dj import nightplan as NP
    from lib.dj import setplanner as SP

    # -- the hooks: which thread did what, when an answer came, every planner child -----------------------------------
    calls = []
    procs = []
    applied = []                                       # (t, the planned ids, length_min) after each applied answer
    _orig = {"stp": D.Director.set_theme_plan, "req": NP.NightPlan.request, "apply": NP.NightPlan._apply,
             "ans": NP.NightPlan._on_answer, "start": SP.PlannerProcess.start, "close": D.Director.close_planner}

    def stp(self, theme, length_min=None, seed=None):
        calls.append(("set_theme_plan", threading.current_thread().name, time.time(), theme))
        return _orig["stp"](self, theme, length_min=length_min, seed=seed)

    def req(self, trigger, extend=False, user=False):
        calls.append(("request", threading.current_thread().name, time.time(), trigger))
        return _orig["req"](self, trigger, extend=extend, user=user)

    def apply(self, resp):
        calls.append(("apply", threading.current_thread().name, time.time(), resp.get("seq")))
        out = _orig["apply"](self, resp)
        applied.append((time.time(), [r["id"] for r in self.rows if r["state"] == "planned"], self.length_min))
        return out

    def ans(self, resp):
        calls.append(("answer", threading.current_thread().name, time.time(), resp.get("seq")))
        return _orig["ans"](self, resp)

    def pstart(self):
        ok = _orig["start"](self)
        if self._proc is not None and self._proc not in procs:
            procs.append(self._proc)
        return ok

    def close(self):
        calls.append(("close_planner", threading.current_thread().name, time.time(), None))
        return _orig["close"](self)
    D.Director.set_theme_plan, NP.NightPlan.request, NP.NightPlan._apply = stp, req, apply
    NP.NightPlan._on_answer, SP.PlannerProcess.start, D.Director.close_planner = ans, pstart, close

    engine = AudioEngine()                             # never opened
    show = make_show(music, engine)
    web = show.web_controller
    stash = {}
    _ni = show._dj_night_info
    cpu = cpu_clock()

    def night_info(d, sp):
        stash["sp"] = sp
        c0 = cpu()
        out = _ni(d, sp)
        stash["ni_cpu"] = (cpu() - c0) * 1000.0
        stash["night"] = out.get("dj_night") if isinstance(out, dict) else None
        return out
    show._dj_night_info = night_info                   # (the set_plan block of the tick, kept for the size measure)
    # THE PRESS'S OWN CPU on the render thread (the 5 ms checks; the wall span is printed beside it)
    cpu_last = {}
    _dn = show._director_night

    def director_night(arg):
        c0 = cpu()
        try:
            return _dn(arg)
        finally:
            cpu_last["director_night"] = cpu_last.get("director_night", 0.0) + (cpu() - c0) * 1000.0
    show._director_night = director_night
    rnd = Render(show)

    def after_tick(row):
        # the night's own part of the tick in CPU: the payload builder (above) + the strip's hash the mirror takes of it
        # (the same json.dumps + hash, re-run here on this thread, outside the bridge's wall timing)
        ni = stash.pop("ni_cpu", None)
        if ni is None:
            return
        c0 = cpu()
        if stash.get("night") is not None:
            hash(json.dumps(stash["night"], separators=(',', ':'), default=str))
        row["night_cpu_ms"] = ni + (cpu() - c0) * 1000.0
    rnd.after = after_tick
    rnd.start()
    audio = start_audio(engine)
    act_ms = []
    wall = [None]

    def press(action, value=None):
        """The press on the next render tick. A night press returns its own CPU on the render thread (wall[0] holds the
        tick's wall span for it); any other press its wall span."""
        cpu_last.pop(action, None)
        ms = rnd.press(action, value)
        wall[0] = ms
        c = cpu_last.get(action)
        act_ms.append((action, json.dumps(value), ms, c))
        return c if action == "director_night" else ms

    def night_now():
        return (web.page.get("night_tick") or {}), web.page.get("dj_night")

    try:
        # == BEFORE START ===============================================================================================
        print("BEFORE START")
        ms = press("director_night", {"op": "plan", "theme": preset, "length_min": 90})
        check("a press with no Director costs the render thread under 5 ms", ms is not None and ms < 5.0, f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
        check("the night is armed", (getattr(show, "_dj_night", None) or {}).get("theme") == preset, f"{getattr(show, '_dj_night', None)}")
        wait_for(lambda: (web.page.get("night_armed") or {}).get("theme") == preset, 2.0)
        check("published: night_armed (the button lit) and the presets", (web.page.get("night_armed") or {}).get("theme") == preset
              and [p["name"] for p in web.page.get("night_presets") or []] == SP.preset_names(), f"{web.page.get('night_armed')}")
        press("director_night", {"op": "length", "length_min": 60})
        wait_for(lambda: web.page.get("night_len") == 60, 2.0)
        check("LENGTH before START sets the armed night's length", (show._dj_night or {}).get("length_min") == 60
              and web.page.get("night_len") == 60, f"{show._dj_night} night_len={web.page.get('night_len')}")
        press("director_night", {"op": "clear"})
        wait_for(lambda: web.page.get("night_armed") is None, 2.0)
        check("clear before START un-arms it", show._dj_night is None and web.page.get("night_armed") is None)
        press("director_night", {"op": "plan", "theme": preset, "length_min": 60})
        n_calls = len(calls)
        ms = press("director_start")
        print(f"  (director_start held the render tick {ms} ms)")
        ms = press("director_night", {"op": "length", "length_min": 60})   # (during the build: queued, replayed on adopt - a no-op)
        check("a press during the start build costs under 5 ms and waits for adopt", ms is not None and ms < 5.0
              and bool(getattr(show, "_director_queued", None)) == bool(getattr(show, "_director_starting", False)),
              f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
        ok = wait_for(lambda: getattr(show, "_director", None) is not None, 300.0, 0.2)
        check("★ DIRECTOR started", ok, f"{show._dj_last_error}")
        if not ok:
            return
        d = show._director
        pl = d.plan
        th = [c for c in calls[n_calls:] if c[0] in ("set_theme_plan", "request", "apply")]
        check("the armed night was planned in the start worker (never the render thread)",
              bool(th) and all(c[1] == "director-start" for c in th), f"{[(c[0], c[1]) for c in th][:6]}")
        check("its first plan came before START, at the armed length", pl is not None and bool(pl.changes) and abs(pl.length_min - 60) < 0.5,
              f"(changes {len(pl.changes) if pl else None}, length {pl.length_min if pl else None})")
        first_slot = pl.changes[0]["sent"][0] if pl and pl.changes and pl.changes[0].get("sent") else None
        ok = wait_for(lambda: d.player is not None and d.player.record is not None
                      and ((d.player.submix.telemetry or {}).get("decks") or {}).get(d.player.record, {}).get("playing"), 180.0, 0.2)
        check("a first record plays", ok, f"{d.last_error}")
        if not ok:
            return
        p = d.player
        rec0 = p.songs[p.record].track.id
        check("the opener is the plan's first record", rec0 == first_slot, f"({d.title(rec0)[:30]} vs {d.title(first_slot)[:30] if first_slot else None})")
        check("the press replayed after START (the plan's own length) re-planned nothing",
              [c["trigger"] for c in pl.changes if c["trigger"].startswith("length")] == [], f"({[c['trigger'] for c in pl.changes]})")

        # == DURING =====================================================================================================
        print("DURING")

        def decks():
            rec = p.songs.get(p.record) if p.record else None
            nxt = NP.NightPlan._next_of(p)
            kept = nxt is not None and (nxt.state in ("ready", "playing") or nxt.deck in p._pending)
            return (rec.track.id if rec else None), (nxt.track.id if (nxt is not None and kept) else None)
        ok = wait_for(lambda: decks()[1] is not None, 120.0, 0.2)
        check("slot +1 is kept on its deck", ok)
        if not ok:
            return
        watch = {"broken": [], "rec": decks()[0], "next": decks()[1], "stop": False, "swaps": []}

        def watcher():
            while not watch["stop"]:
                r, n = decks()
                if r != watch["rec"]:
                    if r != watch["next"]:
                        watch["broken"].append(f"the record became {d.title(r)[:24]}, not the kept next {d.title(watch['next'])[:24] if watch['next'] else None}")
                    watch["swaps"].append((time.time(), watch["rec"], r))
                    watch["rec"], watch["next"] = r, n
                elif n is not None and watch["next"] is not None and n != watch["next"]:
                    watch["broken"].append(f"slot +1 changed from {d.title(watch['next'])[:24]} to {d.title(n)[:24]}")
                    watch["next"] = n
                elif n is not None:
                    watch["next"] = n
                time.sleep(0.05)
        threading.Thread(target=watcher, daemon=True, name="gate-watch").start()

        def press_and_follow(label, value, cond, want_theme=None, want_len=None):
            """One press during the night: its cost, the change it made (applied on the Director's thread), the
            published strip within 1 s of the planner's answer, the decks kept in the plan's now / next rows."""
            n0, c0, pubs0 = len(pl_ref[0].changes) if pl_ref[0] else 0, len(calls), len(web.night_pubs)
            r_before, n_before = decks()
            ms = press("director_night", value)
            check(f"{label}: the press costs the render thread under 5 ms", ms is not None and ms < 5.0, f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
            ok = wait_for(cond, 120.0, 0.05)
            check(f"{label}: the plan answered", ok, f"({d.plan.error if d.plan else 'no plan'})")
            if not ok:
                return None
            new = [c for c in calls[c0:] if c[0] in ("set_theme_plan", "request", "apply")]
            check(f"{label}: set_theme_plan / the request / the apply ran on the Director's thread",
                  bool(new) and all(c[1] == "director" for c in new), f"{sorted({(c[0], c[1]) for c in new})}")
            ans = [c for c in calls[c0:] if c[0] == "answer"]
            t_ans = ans[-1][2] if ans else None
            ids = applied[-1][1] if applied else None   # (the planned order the last applied answer left)
            want = (lambda nt: nt is not None and (want_theme is None or nt.get("source") == want_theme)
                    and (want_len is None or nt.get("length_s") == int(want_len * 60))
                    and [s["id"] for s in nt.get("slots") or [] if s.get("state") == "planned"] == ids)
            ok = wait_for(lambda: any(want(nt) for t, nt in web.night_pubs[pubs0:] if t >= (t_ans or 0)), 3.0)
            t_pub = next((t for t, nt in web.night_pubs[pubs0:] if t >= (t_ans or 0) and want(nt)), None)
            lag = (t_pub - t_ans) if (t_pub and t_ans) else None
            check(f"{label}: the published strip is the answer's plan within 1 s of the answer", ok and lag is not None and 0 <= lag <= 1.0,
                  f"({lag:.2f} s after the answer; {len(ids or [])} planned)" if lag is not None else "(not published)")
            stale = [nt for t, nt in web.night_pubs[pubs0:] if t < (t_ans or 0) and nt is not None and nt.get("source") == want_theme
                     and want_theme is not None and not nt.get("chapters")]
            check(f"{label}: no empty strip before the answer (the night before stays up)", not stale, f"({len(stale)} empty)")
            pl2 = d.plan
            now_rows = [r["id"] for r in pl2.rows if r["state"] == "now"]
            next_rows = [r["id"] for r in pl2.rows if r["state"] == "next"]
            r_after, n_after = decks()
            check(f"{label}: the record playing and slot +1 stayed, and are the plan's now / next",
                  r_after == r_before and n_after == n_before and now_rows == [r_after] and (n_after is None or next_rows == [n_after]),
                  f"(decks {d.title(r_before)[:18]} / {d.title(n_before)[:18] if n_before else None} -> rows {now_rows} / {next_rows})")
            return pl2
        pl_ref = [d.plan]
        pl2 = press_and_follow(f"another night ({other})", {"op": "plan", "theme": other, "length_min": 90},
                               lambda: d.plan is not pl_ref[0] and d.plan is not None and d.plan.source == other and d.plan.changes
                               and d.plan.pending is None, want_theme=other, want_len=90)
        if pl2 is not None:
            with open(os.path.join(out_dir, "night_payload_90.json"), "w", encoding="utf-8") as fh:
                json.dump(web.page, fh, default=str)
        pl_ref[0] = d.plan
        n0 = len(d.plan.changes)
        press_and_follow("LENGTH 120 (no dial button)", {"op": "length", "length_min": 120},
                         lambda: len(d.plan.changes) > n0 and d.plan.pending is None and abs(d.plan.length_min - 120) < 0.5, want_len=120)
        check("LENGTH 120: the change says so", d.plan.changes[-1]["trigger"].startswith("length: 120"), f"({d.plan.changes[-1]['trigger']})")
        n0 = len(d.plan.changes)
        press_and_follow("LENGTH 180 (the 3h dial)", {"op": "length", "length_min": 180},
                         lambda: len(d.plan.changes) > n0 and d.plan.pending is None and abs(d.plan.length_min - 180) < 0.5, want_len=180)
        check("LENGTH 180 went the dial path", d.dials.get("length") == "3h" and "length: 3h" in d.plan.changes[-1]["trigger"],
              f"(dial {d.dials.get('length')}, {d.plan.changes[-1]['trigger']})")
        n0 = len(d.plan.changes)
        press_and_follow("LENGTH 360 (6 h)", {"op": "length", "length_min": 360},
                         lambda: len(d.plan.changes) > n0 and d.plan.pending is None and abs(d.plan.length_min - 360) < 0.5, want_len=360)

        # == COST (the 6 h plan applied) ================================================================================
        print("COST")
        k0 = len(rnd.ticks)
        p0 = len(web.published)
        sizes_before, sizes_after = [], []
        t_end = time.time() + 12.0
        while time.time() < t_end:
            time.sleep(0.2)
        for t, info in web.published[p0:]:
            a = len(json.dumps(info, default=str))
            sizes_after.append(a)
        sp = stash.get("sp") or {}
        sp_len = len(json.dumps(sp, default=str))
        last = web.published[-1][1]
        tick_len = len(json.dumps(last.get("night_tick"), default=str))
        for a in sizes_after:
            sizes_before.append(a - tick_len + sp_len)       # (the old tick: the whole set_plan inside `director`, no night keys)
        nights = [x["night_ms"] for x in rnd.ticks[k0:] if x.get("night_ms") is not None]
        nights_cpu = [x["night_cpu_ms"] for x in rnd.ticks[k0:] if x.get("night_cpu_ms") is not None]
        mirrors = [x["mirror_ms"] for x in rnd.ticks[k0:] if x.get("mirror_ms") is not None]
        n_slots = len((web.page.get("dj_night") or {}).get("slots") or [])
        night_len = len(json.dumps(web.page.get("dj_night"), default=str))

        def pct(xs, q):
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None
        print(f"  6 h plan: {n_slots} slots; the strip (dj_night) {night_len / 1024:.1f} KB, the tick part {tick_len} B, "
              f"set_plan (the old block) {sp_len / 1024:.1f} KB")
        print(f"  the night's part of the tick, wall (info: includes the OS descheduling this BelowNormal gate): median {pct(nights, 0.5)} ms, "
              f"p95 {pct(nights, 0.95)} ms, max {max(nights) if nights else None} ms ({len(nights)} ticks); the whole mirror median "
              f"{pct(mirrors, 0.5)} ms, max {max(mirrors) if mirrors else None} ms")
        print(f"  the night's part of the tick, its own CPU: median {pct(nights_cpu, 0.5) and round(pct(nights_cpu, 0.5), 3)} ms, "
              f"p95 {pct(nights_cpu, 0.95) and round(pct(nights_cpu, 0.95), 3)} ms, max {round(max(nights_cpu), 3) if nights_cpu else None} ms")
        if sizes_after:
            print(f"  dj_info per tick: before {sum(sizes_before) / len(sizes_before) / 1024:.1f} KB (every tick), after "
                  f"{sum(sizes_after) / len(sizes_after) / 1024:.1f} KB average, {min(sizes_after) / 1024:.1f} KB between strip sends, "
                  f"{max(sizes_after) / 1024:.1f} KB on a strip send")
        check("the CPU clock reads the night's work (not zeros)", bool(nights_cpu) and max(nights_cpu) > 0.0, f"(cpu max {max(nights_cpu) if nights_cpu else None})")
        check("6 h: the night's part of the tick under 5 ms of CPU", bool(nights_cpu) and max(nights_cpu) < 5.0,
              f"(cpu max {round(max(nights_cpu), 3) if nights_cpu else None} ms; wall max {max(nights) if nights else None} ms)")
        check("6 h: the strip is not resent every tick", sum(1 for _t, i in web.published[p0:] if "dj_night" in i) < max(2, len(web.published[p0:]) // 2),
              f"({sum(1 for _t, i in web.published[p0:] if 'dj_night' in i)} of {len(web.published[p0:])} ticks)")
        with open(os.path.join(out_dir, "night_payload_6h.json"), "w", encoding="utf-8") as fh:
            json.dump(web.page, fh, default=str)
        with open(os.path.join(out_dir, "night_cost.json"), "w", encoding="utf-8") as fh:
            json.dump({"slots": n_slots, "night_ms": nights, "night_cpu_ms": nights_cpu, "mirror_ms": mirrors, "sizes_after": sizes_after, "sizes_before": sizes_before,
                       "dj_night_bytes": night_len, "tick_bytes": tick_len, "set_plan_bytes": sp_len}, fh)

        # == CLEAR, AGAIN, NEXT =========================================================================================
        print("CLEAR / AGAIN / NEXT")
        c0 = len(calls)
        r_before, n_before = decks()
        ms = press("director_night", {"op": "clear"})
        check("clear: the press costs the render thread under 5 ms", ms is not None and ms < 5.0, f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
        ok = wait_for(lambda: d.plan is None, 5.0)
        cl = [c for c in calls[c0:] if c[0] == "set_theme_plan"]
        check("clear: no plan, done on the Director's thread", ok and bool(cl) and all(c[1] == "director" for c in cl), f"{[(c[1], c[3]) for c in cl]}")
        ok = wait_for(lambda: web.page.get("night_tick") is None and web.page.get("dj_night") is None, 2.0)
        check("clear: dj_night and night_tick published as None within 1 s", ok)
        check("clear: the record playing and slot +1 stayed", decks() == (r_before, n_before))
        pl_ref[0] = None
        press_and_follow(f"{preset} again", {"op": "plan", "theme": preset, "length_min": 90},
                         lambda: d.plan is not None and d.plan.source == preset and d.plan.changes and d.plan.pending is None,
                         want_theme=preset, want_len=90)
        kept = decks()[1]
        ms = press("director_act", "next")
        ok = wait_for(lambda: decks()[0] == kept, 90.0, 0.1)
        check("NEXT: the kept slot +1 is the record", ok, f"({d.title(kept)[:24] if kept else None})")
        time.sleep(1.0)
        watch["stop"] = True
        check("the record playing and slot +1 never changed under a press", not watch["broken"], f"{watch['broken'][:3]}")
        check("no errors on the render thread", not rnd.errors, f"{rnd.errors[:3]}")

        # == DROPPED ====================================================================================================
        print("DROPPED")
        proc = d._planner._proc if d._planner is not None else None
        c0 = len(calls)
        ms = press("stop")
        print(f"  (STOP held the render tick {ms} ms)")
        ok = wait_for(lambda: proc is not None and proc.poll() is not None, 8.0)
        cl = [c for c in calls[c0:] if c[0] == "close_planner"]
        check("STOP: the planner's process is closed, off the render thread", ok and bool(cl) and all(c[1] != "render" for c in cl),
              f"({[c[1] for c in cl]}, exit {proc.poll() if proc else None})")
        audio["stop"] = True
        # (every track the show's engine is handed from here on: a Director that starts attaches "dj_record")
        att = []
        _attach0 = engine.attach_track

        def attach_rec(key, obj):
            att.append((time.time(), key))
            return _attach0(key, obj)
        engine.attach_track = attach_rec
        # a STOP while ★ DIRECTOR is still being built (the night armed: the worker's Director has a planner child) - in the
        # armed night's PLAN WAIT (2026-10-29: the worker waited the minute out, then attached and started the opener)
        n_procs = len(procs)
        press("director_night", {"op": "plan", "theme": preset, "length_min": 45})
        press("director_start")
        ok = wait_for(lambda: len(procs) > n_procs, 60.0, 0.1)
        in_wait = ok and getattr(show, "_director_starting", False) and getattr(show, "_director_pending", None) is None
        t_stop = time.time()
        press("stop")
        wait_for(lambda: not getattr(show, "_director_starting", False), 300.0, 0.2)
        dt_stop = time.time() - t_stop
        ok = ok and wait_for(lambda: all(x.poll() is not None for x in procs[n_procs:]), 8.0)
        check("STOP during the start build: its planner's process is closed", ok and show._director is None,
              f"({[x.poll() for x in procs[n_procs:]]})")
        check("STOP during the plan wait: the build gives up at once and never attaches", in_wait and dt_stop < 8.0
              and not [a for a in att if a[1] == "dj_record"], f"(in the wait {in_wait}, gave up in {dt_stop:.1f} s, attached {[a[1] for a in att]})")
        # ...then ★ DIRECTOR again on the next tick: honoured (the cancel re-arms the build, or a fresh one follows)
        n_procs, n_att = len(procs), len(att)
        press("director_start")
        ok = wait_for(lambda: len(procs) > n_procs, 60.0, 0.1)
        press("stop")
        press("director_start")
        ok2 = wait_for(lambda: getattr(show, "_director", None) is not None, 300.0, 0.2)
        check("STOP during the plan wait, then ★ DIRECTOR again: it starts", ok and ok2 and [a[1] for a in att[n_att:]].count("dj_record") == 1,
              f"(director {show._director is not None}, attached {[a[1] for a in att[n_att:]]}, error {show._dj_last_error!r})")
        # A NIGHT PRESS AND STOP ON THE SAME RENDER TICK (2026-10-29): the Director's step in flight must not start a new
        # planner child after close_planner (each one ~400 MB, reclaimed only by the GC)
        if show._director is not None:
            n_procs = len(procs)
            rnd.press_many([("director_night", {"op": "plan", "theme": other, "length_min": 45}), ("stop", None)])
            ok = wait_for(lambda: all(x.poll() is not None for x in procs), 8.0)
            time.sleep(5.0)
            check("a night press and STOP on one tick: no planner child alive 5 s later, none started after the STOP",
                  ok and all(x.poll() is not None for x in procs) and len(procs) == n_procs,
                  f"({len(procs) - n_procs} started after, {[x.poll() for x in procs]})")
        # a start that fails (the night armed)
        n_procs = len(procs)
        D.Director.start = lambda self, threaded=True: (setattr(self, "last_error", "the gate: start refused") or False)
        press("director_start")
        wait_for(lambda: not getattr(show, "_director_starting", False), 300.0, 0.2)
        ok = wait_for(lambda: all(x.poll() is not None for x in procs[n_procs:]), 8.0)
        check("a start that fails: its planner's process is closed", ok and len(procs) > n_procs and show._director is None,
              f"({len(procs) - n_procs} children, {[x.poll() for x in procs[n_procs:]]}, error {show._dj_last_error!r})")
        ok = wait_for(lambda: all(x.poll() is not None for x in procs), 8.0)
        check("no planner child is left alive", ok, f"({[x.poll() for x in procs]})")
        print("PRESS COSTS (wall ms on the render tick / cpu ms of a night press): " + ", ".join(
            f"{a} {v[:40]} {ms}" + (f"/{c:.3f}" if c is not None else "") for a, v, ms, c in act_ms))
    finally:
        rnd.stop = True
        audio["stop"] = True
        try:
            d0 = getattr(show, "_director", None)
            if d0 is not None:
                d0.stop(0.1)
                _orig["close"](d0)
        except Exception:  # noqa: BLE001
            pass
        for x in procs:
            try:
                if x.poll() is None:
                    x.kill()
            except Exception:  # noqa: BLE001
                pass
    if failures:
        print(f"FAIL ({len(failures)}): {', '.join(failures)}")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
