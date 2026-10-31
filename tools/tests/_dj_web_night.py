"""THE NIGHT ON THE WEB PAGE - the planned night's buttons, strip and story lines through the show's own action path
(docs/DJ_SETS.md section 4 "Web /dj page (phone)" and phase 4's gate; 2026-10-29).

Drives Stories_OGL.EnvironmentalSystem's real DJ bridge (_apply_dj_controls: the action queue, _director_night,
_director_start / _director_adopt / _director_stop, the 5 Hz mirror) on a stand-in show object - a fake web controller
that records every dj_info it is handed, no GLFW, no audio device (the engine's render-ahead ring drained by a thread in
the device's place, as _dj_plan_gate.py's controls do). A thread named "render" ticks the bridge at 5 Hz, as the show's
render loop does; presses go through it.

  VALIDATOR     (2026-10-30, "more like these") director_night op "like" through the real web controller (its HTTP twin,
                a Flask test client, never started): 2 to 5 distinct int ids in the library pass, the length snapped; 1 id,
                6 ids, unknown ids, strings, floats, bools, one id twice, no list, and a library not known yet are refused
  BEFORE START  a night pressed with no Director is ARMED (published, lit); LENGTH with none on sets the next press's
                length; clear un-arms; night_wanted (the page hides the brain's Theme row) false / true armed / false
                cleared; tonight's records (a stopped session's) published and three of them, "like these", ARMED under
                "Like <first> +2" with a press under 1 ms; ★ DIRECTOR then plans a preset in the start worker (never on the
                render thread), waits for it, and opens on the plan's first record, at the armed length
  DURING        (a running record-mode night, a record playing and slot +1 kept on its deck) another night pressed, a
                LENGTH the dial has no button for (120), one it has (180 -> the LENGTH dial), the 6 h night, clear, a
                night again, then NEXT:
                  - every press costs the render thread under 5 ms of its own CPU (the wall span printed beside it)
                  - the request and the answer's apply run on the Director's own thread ("director")
                  - the published night part (dj_night) changes within 1 s of the planner's answer
                  - the plan's now / next rows are the decks' record and slot +1; neither changes under a press; NEXT
                    brings in the kept slot +1
                  - the Theme row stays hidden (night_wanted) on every tick while a press is on its way
                  - LIKE THESE with 3 records played tonight (the record playing + two noted played): the press under 1 ms,
                    the inferred night planned from slot +2 (now / next kept), lit as "like" under its name; it REACHES
                    ITS LENGTH (2026-10-30: "1 records", "runs out at 0:13" - the planner widens the crate until it
                    does) or says why not, every planned record by the rules; a widened crate is kept by the NightPlan
                    and published (dj_night.widened, a story line); the same
                    records again re-plan nothing; an inference that fails publishes its error (the underline) and a
                    story line, the night on and the decks stay; the page holds no alert / confirm / prompt
                  - clear: no plan, dj_night / night_tick / night_like published as None, the Theme row back
  DROPPED       STOP closes the planner's process (off the render thread); a STOP during the start build, and a start
                that fails, close it too; a STOP in the armed night's plan wait gives up at once and never attaches, and
                ★ DIRECTOR pressed again then starts; a night press and STOP on one tick start no planner child after the
                STOP; tonight's records stay offered on the idle page after STOP, a "like these" of three is armed and
                ★ DIRECTOR infers and plans it before its first record (off the render thread), STOP closes its planner;
                at the end no planner child is alive
  COST          with the 6 h plan applied: the night's own part of the show tick (the payload builder + its hash) in the
                render thread's own CPU (QueryThreadCycleTime on Windows - a wall span counts the OS descheduling this
                BelowNormal gate, so it is printed, not judged), the whole mirror; the dj_info size per tick before (the
                full set_plan inside `director`, every tick) and
                after (dj_night only when it changes, night_tick every tick)
  PAYLOAD       the page's merged view (the deltas merged like dj_panel.html does) saved at 90 min and at 6 h, with the
                like night on, and with no night (after clear), for the static harness page (--out)
  SHAPE CARDS   (2026-10-30, the night buttons replaced by cards that draw each night's energy course) night_presets carries
                every preset of setplanner.PRESETS (read generically) with its chapters' name, share and energy course, and
                is a delta key: sent when it changes and in the 3 s full refresh, never every tick (counted over the COST
                window). THE PAGE (headless Edge over the DevTools protocol, the gate's own merged payloads fed to the real
                dj_panel.html, the socket stubbed; skipped when Edge is missing) at 390 and 1280 px: a card per preset with
                its curve and chapter names, the strip swipes (about 2.5 cards in view at 390, several at 1280) and the page
                never scrolls sideways; a tap on the last card emits the plan and lights it at once (pending), the server's
                answer keeps it lit and scrolled into view, a tap on the lit card emits clear and unlights it; the like
                night's card appears, lit, with its inferred chapter's curve; no JS error

    python tools/tests/_dj_web_night.py [--music D:/Devel/music] [--out DIR] [--preset "Desert Night"] [--other "Deep Space"] [--no-page]

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


EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def page_harness(path):
    """web/templates/dj_panel.html as a static page: its CSS from disk, no socket.io; window.__feed(dj) hands the page a
    state as the socket does, every emit is kept in window.__emits, JS errors in window.__errs."""
    src = open(os.path.join(ROOT, "web", "templates", "dj_panel.html"), encoding="utf-8").read()
    for css in ("common.css", "control_panel.css"):
        src = src.replace(f'href="/static/css/{css}"', 'href="file:///' + os.path.join(ROOT, "web", "static", "css", css).replace("\\", "/") + '"')
    for js in ("socket.io.min.js", "socket-client.js", "club_tab.js", "interaction_tab.js"):
        src = src.replace(f'<script src="/static/js/{js}"></script>', "")
    stub = ("<script>window.__errs = []; window.onerror = (m, u, l, c) => { window.__errs.push(m + ' @' + l + ':' + c); };"
            "window.__emits = []; function createSocket() { return { on(ev, fn) { if (ev === 'state_update') window.__feed = (dj) => fn({dj}); },"
            " emit(ev, payload) { window.__emits.push(payload); } }; }</script>")
    hook = "<script>\n        const socket = createSocket();"
    if hook not in src:
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(hook, stub + "\n    " + hook, 1))
    return True


class EdgePage:
    """Headless Edge over the DevTools protocol (no --dump-dom: it prints nothing on this machine): one page at a width."""

    def __init__(self, width, touch=False):
        import subprocess
        import tempfile
        import urllib.request
        import websocket
        self.ud = tempfile.mkdtemp(prefix="edge_gate_")
        port = 9800 + (os.getpid() + width) % 150
        self.proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={self.ud}", "--disable-gpu",
                                      "--hide-scrollbars", "--allow-file-access-from-files", "--no-first-run", f"--window-size={max(width, 500)},1000",
                                      "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tgt = None
        for _ in range(150):
            try:
                tgt = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1)) if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        if tgt is None:
            raise RuntimeError("Edge's DevTools did not answer")
        self.ws = websocket.create_connection(tgt["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
        self.n = 0
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1, mobile=False)
        if touch:
            self.call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result") or {}

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(f"page: {r['exceptionDetails'].get('text')} {(r['exceptionDetails'].get('exception') or {}).get('description', '')[:200]}")
        return (r.get("result") or {}).get("value")

    def close(self):
        """Browser.close over the protocol, then the whole process tree (killing the browser alone left its renderer, GPU
        and utility children running - 300 of them after a few runs)."""
        import shutil
        import subprocess
        try:
            self.call("Browser.close")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(5)
        except Exception:  # noqa: BLE001
            pass
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.proc.kill()
        for _ in range(20):                            # (the profile folder is free once every child is gone)
            shutil.rmtree(self.ud, ignore_errors=True)
            if not os.path.isdir(self.ud):
                break
            time.sleep(0.1)


# the strip as the page shows it (run in the page)
CARDS_JS = r"""(() => {
    const box = document.getElementById('dj-night-btns'), strip = box && box.querySelector('.night-strip');
    if (!strip) return null;
    const sr = strip.getBoundingClientRect();
    const cards = [...strip.querySelectorAll('.night-card[data-night]')];
    const like = strip.querySelector('.like-night');
    const lit = strip.querySelector('.night-card.on:not(.hidden)');
    const inView = (c) => { const r = c.getBoundingClientRect(); return r.left >= sr.left - 1 && r.right <= sr.right + 1; };
    const w = cards.length ? cards[0].getBoundingClientRect().width : 1;
    return {shown: box.offsetParent !== null, overflow: document.documentElement.scrollWidth - innerWidth, scroll_x: scrollX,
            swipes: strip.scrollWidth > strip.clientWidth + 4, in_view: +(strip.clientWidth / (w + 6)).toFixed(2),
            cards: cards.map((c) => ({name: c.dataset.night, on: c.classList.contains('on'), pending: c.classList.contains('pending'),
                                      curve: !!c.querySelector('.nc-curve path.l:not(.dash)'), fill: !!c.querySelector('.nc-curve path.f'),
                                      chs: [...c.querySelectorAll('.nc-chs > span')].map((s) => s.textContent)})),
            like: like ? {shown: !like.classList.contains('hidden'), on: like.classList.contains('on'), label: like.dataset.label || '',
                          curve: !!like.querySelector('.nc-curve path.l:not(.dash)'), chs: [...like.querySelectorAll('.nc-chs > span')].map((s) => s.textContent)} : null,
            lit: lit ? (lit.dataset.night || '~like') : null, lit_in_view: lit ? inView(lit) : null,
            emits: window.__emits.slice(-3), errors: window.__errs.slice()};
})()"""


def page_check(out_dir, p_none, p_on, p_like, presets):
    """THE SHAPE CARDS ON THE PAGE: the gate's own merged payloads fed to the real dj_panel.html in headless Edge, at 390
    and 1280 px. A tap on the last preset's card, the server's answer, a tap on the lit card, the like night."""
    if not os.path.isfile(EDGE):
        print(f"  (SKIP the page check: no Edge at {EDGE})")
        return
    try:
        import websocket  # noqa: F401
    except ImportError:
        print("  (SKIP the page check: websocket-client is not installed)")
        return
    path = os.path.join(out_dir, "cards_page.html")
    if not page_harness(path):
        check("the page check: the page's socket hook is where the harness expects it", False)
        return
    names = [p["name"] for p in presets]
    target = names[-1]                                 # (the last card: it starts out of view, so the scroll is tested too)

    def with_night(p, name):
        q = json.loads(json.dumps(p, default=str))
        for k in ("night_tick", "dj_night"):
            if isinstance(q.get(k), dict):
                q[k]["source"] = q[k]["theme"] = name
        q["night_tick"]["pending"] = None
        return q
    for width in (390, 1280):
        pg = EdgePage(width, touch=width < 500)
        try:
            pg.call("Page.navigate", url="file:///" + path.replace("\\", "/"))
            ok = wait_for(lambda: pg.js("typeof window.__feed === 'function'"), 15.0, 0.1)
            if not ok:
                check(f"{width} px: the page loaded", False)
                continue

            def feed(p):
                pg.js("window.__feed(" + json.dumps(p, default=str) + ")")
                time.sleep(0.35)
                return pg.js(CARDS_JS)
            a = feed(p_none)
            cards = (a or {}).get("cards") or []
            check(f"{width} px: a card per preset, in order, each with its energy curve, its fill and its chapter names",
                  a is not None and a["shown"] and [c["name"] for c in cards] == names and all(c["curve"] and c["fill"] for c in cards)
                  and all(c["chs"] == [ch[0] for ch in p["chapters"]] for c, p in zip(cards, presets)),
                  f"({len(cards)} cards of {len(names)}; {[c['name'] for c in cards if not c['curve']][:3]})")
            want_view = (2.0, 3.2) if width < 500 else (5.0, 99.0)
            check(f"{width} px: one strip that swipes ({a and a['in_view']} cards in view), the page never scrolls sideways",
                  a is not None and a["swipes"] and want_view[0] <= a["in_view"] <= want_view[1] and a["overflow"] <= 0 and a["scroll_x"] == 0,
                  f"(overflow {a and a['overflow']} px)")
            check(f"{width} px: no night: no card lit, no like card", a is not None and a["lit"] is None and not (a["like"] or {}).get("shown"))
            # a tap: the plan emitted, the card lit at once (planning until the answer)
            pg.js(f"[...document.querySelectorAll('#dj-night-btns .night-card[data-night]')].find((c) => c.dataset.night === {json.dumps(target)}).click()")
            b = pg.js(CARDS_JS)
            em = (b["emits"] or [None])[-1] or {}
            tc = next((c for c in b["cards"] if c["name"] == target), {})
            check(f"{width} px: a tap on {target} emits its plan and lights it at once (planning)",
                  em.get("action") == "director_night" and (em.get("value") or {}).get("op") == "plan" and (em.get("value") or {}).get("theme") == target
                  and tc.get("on") and tc.get("pending"), f"({em}, {tc.get('on')}, {tc.get('pending')})")
            feed(with_night(p_on, target))
            wait_for(lambda: pg.js(CARDS_JS)["lit_in_view"], 2.0, 0.1)    # (a smooth scroll to the lit card)
            c = pg.js(CARDS_JS)
            tc = next((x for x in c["cards"] if x["name"] == target), {})
            check(f"{width} px: the answer keeps it lit (not planning), the only lit card, scrolled into view in the strip (not the page)",
                  tc.get("on") and not tc.get("pending") and c["lit"] == target and c["lit_in_view"] and c["scroll_x"] == 0
                  and sum(1 for x in c["cards"] if x["on"]) == 1, f"(lit {c['lit']}, in view {c['lit_in_view']})")
            pg.js(f"[...document.querySelectorAll('#dj-night-btns .night-card[data-night]')].find((c) => c.dataset.night === {json.dumps(target)}).click()")
            d = pg.js(CARDS_JS)
            em = (d["emits"] or [None])[-1] or {}
            check(f"{width} px: a tap on the lit card emits clear and unlights it at once",
                  (em.get("value") or {}).get("op") == "clear" and d["lit"] is None, f"({em}, lit {d['lit']})")
            e = feed(p_none)
            check(f"{width} px: cleared: nothing lit", e["lit"] is None)
            if p_like is not None:
                feed(p_like)
                wait_for(lambda: pg.js(CARDS_JS)["lit_in_view"], 2.0, 0.1)
                f = pg.js(CARDS_JS)
                lk = f["like"] or {}
                want = (p_like.get("night_tick") or {}).get("theme")
                chs = [ch.get("name") for ch in (p_like.get("dj_night") or {}).get("chapters") or []]
                check(f"{width} px: the like night's card appears, lit, with its inferred chapters' curve",
                      lk.get("shown") and lk.get("on") and lk.get("label") == want and lk.get("curve") and lk.get("chs") == chs and f["lit"] == "~like"
                      and f["lit_in_view"], f"({lk}, want {want} {chs})")
            z = pg.js(CARDS_JS)
            check(f"{width} px: no JS error on the page", not z["errors"], f"({z['errors'][:2]})")
        finally:
            pg.close()
    if sys.platform == "win32":
        import subprocess
        left = None
        for _ in range(30):
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "@(Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | Where-Object { $_.CommandLine -match 'edge_gate_' }).Count"],
                               capture_output=True, text=True)
            left = int((r.stdout or "0").strip() or 0)
            if left == 0:
                break
            time.sleep(0.2)
        check("the page check left no Edge process running", left == 0, f"({left} left)")


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
    answers = []                                       # (the NightPlan, the planner's answer) of each apply
    _orig = {"stp": D.Director.set_theme_plan, "req": NP.NightPlan.request, "apply": NP.NightPlan._apply,
             "ans": NP.NightPlan._on_answer, "start": SP.PlannerProcess.start, "close": D.Director.close_planner}

    def stp(self, theme, length_min=None, seed=None, source=None):
        calls.append(("set_theme_plan", threading.current_thread().name, time.time(), theme))
        return _orig["stp"](self, theme, length_min=length_min, seed=seed, source=source)

    def req(self, trigger, extend=False, user=False):
        calls.append(("request", threading.current_thread().name, time.time(), trigger))
        return _orig["req"](self, trigger, extend=extend, user=user)

    def apply(self, resp):
        calls.append(("apply", threading.current_thread().name, time.time(), resp.get("seq")))
        answers.append((self, resp))
        del answers[:-20]
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

    # THE REAL VALIDATOR for "like these" (web_controller.queue_dj_action through its HTTP twin): a Flask test client on a
    # WebController that is never started, the library's ids as the show publishes them
    from web.web_controller import WebController
    from lib.dj.db import LibraryDB
    from lib.dj.themestore import _short
    wc = WebController(control_dict={})
    db = LibraryDB(music)
    lib_rows = [(int(r["id"]), r["title"] or "", r["artist"] or "")
                for r in db.conn.execute("SELECT id, title, artist FROM tracks WHERE error IS NULL AND missing = 0 ORDER BY id")]
    db.close()
    wc.dj_library_ids = frozenset(i for i, _t, _a in lib_rows)
    client = wc.app.test_client()

    def web_like(value):
        """POST director_night through the web controller: (accepted, the sanitized value it queued or None)."""
        with wc._dict_lock:
            wc.control_dict["request_dj_actions"] = []
        r = client.post("/api/dj/action", json={"action": "director_night", "value": value})
        q = wc.control_dict.get("request_dj_actions") or []
        return r.status_code == 200, (q[-1][1] if q else None)

    def like_name(ids, title):
        return "Like " + _short(title(ids[0])) + (f" +{len(ids) - 1}" if len(ids) > 1 else "")

    def wanted_through(t0, t1):
        """night_wanted on every dj_info published in [t0, t1] (the Theme row hidden all along), and how many."""
        xs = [i.get("night_wanted") for t, i in list(web.published) if t0 <= t <= t1]
        return bool(xs) and all(xs), len(xs)

    try:
        # == "LIKE THESE": the web controller refuses bad input ======================================================
        print("LIKE THESE: the validator")
        some = [i for i, _t, _a in lib_rows[:6]]
        ok, v = web_like({"op": "like", "ids": some[:3], "length_min": 95})
        check("3 library ids are accepted, the length snapped", ok and v == {"op": "like", "theme": None, "length_min": 90, "ids": some[:3]}, f"({ok}, {v})")
        for label, bad in (("1 id", some[:1]), ("6 ids", some[:6]), ("unknown ids", [10 ** 9, 10 ** 9 + 1, 10 ** 9 + 2]),
                           ("strings", [str(i) for i in some[:3]]), ("floats", [float(i) for i in some[:3]]), ("bools", [True, False]),
                           ("one id twice", [some[0], some[0]]), ("not a list", some[0]), ("no ids", None)):
            ok, v = web_like({"op": "like", "ids": bad, "length_min": 90})
            check(f"refused: {label}", not ok and v is None, f"({ok}, {v})")
        saved_ids, wc.dj_library_ids = wc.dj_library_ids, None
        ok, v = web_like({"op": "like", "ids": some[:3]})
        check("refused while the library's ids are not known", not ok and v is None)
        wc.dj_library_ids = saved_ids

        # == BEFORE START ===============================================================================================
        print("BEFORE START")
        wait_for(lambda: "night_wanted" in web.page, 2.0)
        check("no night: the Theme row shows (night_wanted false)", web.page.get("night_wanted") is False, f"({web.page.get('night_wanted')})")
        ms = press("director_night", {"op": "plan", "theme": preset, "length_min": 90})
        check("a press with no Director costs the render thread under 5 ms", ms is not None and ms < 5.0, f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
        check("the night is armed", (getattr(show, "_dj_night", None) or {}).get("theme") == preset, f"{getattr(show, '_dj_night', None)}")
        wait_for(lambda: (web.page.get("night_armed") or {}).get("theme") == preset, 2.0)
        check("published: night_armed (the button lit) and the presets", (web.page.get("night_armed") or {}).get("theme") == preset
              and [p["name"] for p in web.page.get("night_presets") or []] == SP.preset_names(), f"{web.page.get('night_armed')}")
        check("armed: the Theme row is hidden (night_wanted)", web.page.get("night_wanted") is True)
        # THE SHAPE CARDS' DATA: every preset (setplanner.PRESETS, however many) with each chapter's name, share and energy
        # course - what the card draws
        pub = web.page.get("night_presets") or []
        bad = []
        for p, q in zip(SP.PRESETS, pub):
            chs = p.get("chapters") or []
            tot = sum(float(c.get("share") or 0) for c in chs) or 1.0
            qc = q.get("chapters") or []
            if q.get("name") != p["name"] or q.get("blurb") != (p.get("blurb") or "") or len(qc) != len(chs) or any(
                    r[0] != c["name"] or abs(r[1] - 100.0 * float(c.get("share") or 0) / tot) > 0.51
                    or abs(r[2] - float(c["energy"][0])) > 0.006 or abs(r[3] - float(c["energy"][-1])) > 0.006 for r, c in zip(qc, chs)):
                bad.append(p["name"])
        check(f"published: night_presets carries all {len(SP.PRESETS)} presets with each chapter's name, share and energy course",
              len(pub) == len(SP.PRESETS) and not bad, f"({len(pub)} published; wrong: {bad[:4]}; {len(json.dumps(pub)) / 1024:.1f} KB)")
        press("director_night", {"op": "length", "length_min": 60})
        wait_for(lambda: web.page.get("night_len") == 60, 2.0)
        check("LENGTH before START sets the armed night's length", (show._dj_night or {}).get("length_min") == 60
              and web.page.get("night_len") == 60, f"{show._dj_night} night_len={web.page.get('night_len')}")
        press("director_night", {"op": "clear"})
        wait_for(lambda: web.page.get("night_armed") is None and web.page.get("night_wanted") is False, 2.0)
        check("clear before START un-arms it", show._dj_night is None and web.page.get("night_armed") is None)
        check("cleared: the Theme row shows again", web.page.get("night_wanted") is False)
        # LIKE THESE before START: tonight's records (a stopped session's list, as _dj_tonight leaves it) offered, three of
        # them through the web controller, ARMED for ★ DIRECTOR under its own name
        tonight = [{"id": i, "title": t[:60], "artist": a[:40], "now": False} for i, t, a in lib_rows[10:14]]
        show._dj_tonight_last = tonight
        show._dj_tonight_titles = {i: t for i, t, _a in lib_rows[10:14]}
        wait_for(lambda: [r["id"] for r in web.page.get("night_tonight") or []] == [r["id"] for r in tonight], 2.0)
        check("idle: tonight's records published (night_tonight)", [r["id"] for r in web.page.get("night_tonight") or []] == [r["id"] for r in tonight])
        ids0 = [r["id"] for r in tonight[:3]]
        ok, v = web_like({"op": "like", "ids": ids0, "length_min": 60})
        ms = press("director_night", v) if ok else None
        want_name = like_name(ids0, lambda i: show._dj_tonight_titles[i])
        check("like these before START: the press costs the render thread under 1 ms", ms is not None and ms < 1.0, f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
        wait_for(lambda: (web.page.get("night_armed") or {}).get("like") == ids0, 2.0)
        armed = web.page.get("night_armed") or {}
        check("like these before START: armed, under its own name", armed.get("like") == ids0 and armed.get("theme") == want_name
              and armed.get("length_min") == 60 and web.page.get("night_wanted") is True, f"({armed})")
        press("director_night", {"op": "clear"})
        wait_for(lambda: web.page.get("night_armed") is None, 2.0)
        check("like these before START: clear un-arms it", show._dj_night is None and web.page.get("night_wanted") is False)
        show._dj_tonight_last = None
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

        def press_and_follow(label, value, cond, want_theme=None, want_len=None, max_ms=5.0):
            """One press during the night: its cost, the change it made (applied on the Director's thread), the
            published strip within 1 s of the planner's answer, the decks kept in the plan's now / next rows, the Theme
            row hidden (night_wanted) on every tick while it was on its way."""
            n0, c0, pubs0 = len(pl_ref[0].changes) if pl_ref[0] else 0, len(calls), len(web.night_pubs)
            r_before, n_before = decks()
            ms = press("director_night", value)
            t_press = time.time()                     # (after the tick that took the press)
            check(f"{label}: the press costs the render thread under {max_ms:g} ms", ms is not None and ms < max_ms, f"(cpu {ms and round(ms, 3)} ms, wall {wall[0]} ms)")
            ok = wait_for(cond, 120.0, 0.05)
            check(f"{label}: the plan answered", ok, f"({d.plan.error if d.plan else 'no plan'})")
            if not ok:
                return None
            hid, n_t = wanted_through(t_press, time.time())
            check(f"{label}: the Theme row stayed hidden while it planned (night_wanted on every tick)", hid, f"({n_t} ticks)")
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
        # THE SHAPE CARDS' DATA is a delta key: only in the mirror's 3 s full refresh (every 15th tick) while it is unchanged
        n_pub = len(web.published[p0:])
        n_pre = sum(1 for _t, i in web.published[p0:] if "night_presets" in i)
        pre_b = len(json.dumps(web.page.get("night_presets"), default=str))
        print(f"  night_presets: {pre_b / 1024:.1f} KB, in {n_pre} of {n_pub} ticks ({n_pub / max(1, n_pre):.0f} ticks apart)")
        check("the shape cards' presets are not sent every tick (the 3 s full refresh only)", n_pub >= 30 and n_pre <= n_pub // 15 + 1,
              f"({n_pre} of {n_pub} ticks)")
        with open(os.path.join(out_dir, "night_payload_6h.json"), "w", encoding="utf-8") as fh:
            json.dump(web.page, fh, default=str)
        with open(os.path.join(out_dir, "night_cost.json"), "w", encoding="utf-8") as fh:
            json.dump({"slots": n_slots, "night_ms": nights, "night_cpu_ms": nights_cpu, "mirror_ms": mirrors, "sizes_after": sizes_after, "sizes_before": sizes_before,
                       "dj_night_bytes": night_len, "tick_bytes": tick_len, "set_plan_bytes": sp_len}, fh)

        # == LIKE THESE during the night ================================================================================
        print("LIKE THESE (during the night)")
        # three records played tonight: the record playing and two the night log would hold (noted as played - records
        # from the plan's later slots, so they blend with tonight's; slot +1 is never one of them)
        later = [r["id"] for r in d.plan.rows if r["state"] == "planned"][1:3]
        for tid in later:
            d._note_played(tid)
        ok = wait_for(lambda: {r["id"] for r in web.page.get("night_tonight") or []} >= set(later) | {decks()[0]}, 3.0)
        tn = web.page.get("night_tonight") or []
        check("tonight's records published, newest first, the record playing marked", ok and [r["id"] for r in tn[:2]] == later[::-1]
              and any(r["now"] and r["id"] == decks()[0] for r in tn), f"({[(r['title'][:16], r['now']) for r in tn][:5]})")
        ids3 = [decks()[0]] + later
        ok, v = web_like({"op": "like", "ids": ids3, "length_min": 90})
        check("like these: the web controller takes 3 played records", ok and v and v.get("ids") == ids3, f"({ok}, {v})")
        want_name = like_name(ids3, d.title)
        pl_ref[0] = d.plan
        pl_like = press_and_follow("like these (3 records)", v,
                                   lambda: d.plan is not pl_ref[0] and d.plan is not None and d.plan.source == "like" and d.plan.changes
                                   and d.plan.pending is None, want_theme="like", want_len=90, max_ms=1.0)
        if pl_like is not None:
            check("like these: the plan is the inferred night (its examples, its name)",
                  [int(i) for i in pl_like.theme.get("examples") or []] == ids3 and pl_like.name == want_name, f"({pl_like.name}, {pl_like.theme.get('examples')})")
            check("like these: the inference ran in the planner's process, then planned on the Director's thread",
                  any(c[0] == "set_theme_plan" and c[1] == "director" and isinstance(c[3], dict) for c in calls), "")
            wait_for(lambda: (web.page.get("night_tick") or {}).get("source") == "like" and not (web.page.get("night_like") or {}).get("pending"), 2.0)
            nt, lk = web.page.get("night_tick") or {}, web.page.get("night_like") or {}
            check("published: the like night lit (source like, its name), the press answered", nt.get("source") == "like"
                  and nt.get("theme") == want_name and lk.get("name") == want_name and lk.get("ids") == ids3 and not lk.get("pending")
                  and not lk.get("error") and web.page.get("night_wanted") is True, f"({nt.get('source')}, {nt.get('theme')}, {lk})")
            # THE NIGHT REACHES ITS LENGTH (2026-10-30: "Like Galapagos +2, 1 records" - the route ran out at 0:13: the
            # examples, played tonight, were counted toward the crate and then banned, and nothing in the crate blended
            # on after slot +1). The planner widens the inferred crate until the night finishes, or the warning says why.
            from lib.dj import themestore as TS
            resp_l = next((r for npl, r in reversed(answers) if npl is pl_like), {})
            L_s = pl_like.length_s()
            fut = [r for r in pl_like.rows if r["state"] == "planned"]
            end_l = max([float(r.get("end_s") or 0.0) for r in pl_like.rows if r["state"] in ("now", "next", "planned")] + [0.0])
            wid = resp_l.get("widened") or []
            tm = resp_l.get("timing") or {}
            need = TS.need_records(pl_like.length_min, float(p.play_min_s))
            print(f"  like these: {len(fut)} records planned after the decks, the night ends {end_l / 60:.1f} of {L_s / 60:.0f} min, finished "
                  f"{pl_like.finished}; crate {resp_l.get('crate')} ({len(wid)} widenings by the planner: {', '.join(w['what'] for w in wid) or '-'}; "
                  f"inferred {', '.join(w['what'] for w in (pl_like.theme.get('inferred') or {}).get('widened') or [] if w.get('by') != 'plan') or '-'}); "
                  f"{tm.get('plans', 1)} plans in {tm.get('widen_s', tm.get('total_s'))} s (need_records {need})", flush=True)
            lad = (pl_like.theme.get("inferred") or {}).get("ladder") or {}
            said = [w["text"] for w in pl_like.warnings if w.get("bad")]
            check("like these from 3 records played tonight: the night reaches its length, or says why not",
                  (pl_like.finished and end_l >= L_s - 1.0 and len(fut) >= 2)
                  or (bool(said) and (int(lad.get("at") or 0) >= len(TS.WIDEN) or float(tm.get("widen_s") or 0) >= SP.WIDEN_BUDGET_S)),
                  f"({len(fut)} planned, ends {end_l / 60:.1f} min; {said[:1]})")
            # (the one seam the planner may take on the player's fallback is the first after the fixed records, and said)
            loose_said = any("relaxed blend" in w["text"] or "plays to its end" in w["text"] for w in pl_like.warnings)
            check("like these: every planned record comes in by the rules (no relaxed seam on a plan pick)",
                  all((r.get("in") or {}).get("kind") in ("blend", "breakdown") or (k == 0 and loose_said) for k, r in enumerate(fut)),
                  f"({[(r.get('in') or {}).get('kind') for r in fut][:6]})")
            if wid:
                check("like these: the NightPlan keeps the widened crate the planner used (later re-plans start from it)",
                      pl_like.theme.get("crate") == (resp_l.get("theme_used") or {}).get("crate") and len(pl_like.crate_ids) == resp_l.get("crate"),
                      f"({pl_like.theme.get('crate')})")
                wait_for(lambda: (web.page.get("dj_night") or {}).get("widened") == [w["what"] for w in wid], 2.0)
                notes = [n["text"] for n in (web.page.get("night_tick") or {}).get("notes") or []]
                check("published: the widened crate (dj_night.widened) and a story line saying it",
                      (web.page.get("dj_night") or {}).get("widened") == [w["what"] for w in wid]
                      and any(n.startswith("A night like these: the crate was widened") for n in notes), f"({notes})")
            with open(os.path.join(out_dir, "night_payload_like.json"), "w", encoding="utf-8") as fh:
                json.dump(web.page, fh, default=str)
        # the same records again while that night is on: a replay, nothing re-planned
        n0 = len(d.plan.changes) if d.plan else 0
        press("director_night", v)
        time.sleep(1.5)
        check("like these: the same records pressed again re-plan nothing", d.plan is not None and len(d.plan.changes) == n0)
        # AN ERROR: the planner's inference fails - an amber underline (night_like.error), a story line, the night on stays
        ids_err = ids3[::-1]
        _req = SP.PlannerProcess.request

        def req_down(self, r, timeout=300.0):
            if r.get("op") == "infer":
                return {"ok": False, "error": "the gate: the planner is down", "seq": r.get("seq")}
            return _req(self, r, timeout=timeout)
        SP.PlannerProcess.request = req_down
        try:
            r_before, n_before = decks()
            pl0 = d.plan
            ok, v2 = web_like({"op": "like", "ids": ids_err, "length_min": 90})
            ms = press("director_night", v2)
            check("like these, failing: the press costs the render thread under 1 ms", ms is not None and ms < 1.0, f"(cpu {ms and round(ms, 3)} ms)")
            ok = wait_for(lambda: ((web.page.get("night_like") or {}).get("error") or "").startswith("the gate"), 5.0)
            lk = web.page.get("night_like") or {}
            notes = [n["text"] for n in (web.page.get("night_tick") or {}).get("notes") or []]
            check("like these, failing: the error is published (the button's underline and tooltip)", ok and lk.get("ids") == ids_err
                  and not lk.get("pending"), f"({lk})")
            check("like these, failing: a story line says it", any(n.startswith("like these: the gate") for n in notes), f"({notes})")
            check("like these, failing: the night on stays, the decks too", d.plan is pl0 and decks() == (r_before, n_before))
        finally:
            SP.PlannerProcess.request = _req
        page_src = open(os.path.join(ROOT, "web", "templates", "dj_panel.html"), encoding="utf-8").read()
        import re
        check("no modal on the page (no alert / confirm / prompt)", not re.search(r"\b(alert|confirm|prompt)\s*\(", page_src))

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
        check("clear: the like press and its error are gone, the Theme row shows again",
              web.page.get("night_like") is None and web.page.get("night_wanted") is False, f"({web.page.get('night_like')}, {web.page.get('night_wanted')})")
        check("clear: the record playing and slot +1 stayed", decks() == (r_before, n_before))
        with open(os.path.join(out_dir, "night_payload_none.json"), "w", encoding="utf-8") as fh:
            json.dump(web.page, fh, default=str)
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
        # A NIGHT LIKE THESE ARMED BEFORE START: the stopped session's records offered on the idle page, three pressed,
        # ★ DIRECTOR infers and plans it in the start worker and opens on its first record
        print("LIKE THESE AT START")
        ok = wait_for(lambda: show._director is None and len(web.page.get("night_tonight") or []) >= 3, 10.0)
        tn = web.page.get("night_tonight") or []
        check("idle after STOP: tonight's records stay offered (nothing marked playing)", ok and not any(r["now"] for r in tn), f"({len(tn)})")
        if ok:
            ids_s = [r["id"] for r in tn[:3]]
            okv, v = web_like({"op": "like", "ids": ids_s, "length_min": 45})
            ms = press("director_night", v) if okv else None
            check("like these armed: the press costs the render thread under 1 ms", ms is not None and ms < 1.0, f"(cpu {ms and round(ms, 3)} ms)")
            n_procs, c0 = len(procs), len(calls)
            press("director_start")
            ok = wait_for(lambda: getattr(show, "_director", None) is not None, 300.0, 0.2)
            d2 = show._director
            pl3 = d2.plan if d2 is not None else None
            th = [c for c in calls[c0:] if c[0] == "set_theme_plan"]
            check("like these at START: inferred and planned off the render thread (the start worker's), before the first record",
                  ok and pl3 is not None and pl3.source == "like" and [int(i) for i in pl3.theme.get("examples") or []] == ids_s
                  and bool(pl3.changes) and bool(th) and all(c[1] in ("director-start", "setplanner-request") for c in th),
                  f"({pl3.source if pl3 else None}, {pl3.theme.get('examples') if pl3 else None}, {[(c[0], c[1]) for c in th][:6]})")
            proc = d2._planner._proc if d2 is not None and d2._planner is not None else None
            press("stop")
            ok = wait_for(lambda: proc is None or proc.poll() is not None, 8.0)
            check("like these at START: STOP closes its planner", ok and show._director is None)
        press("director_night", {"op": "plan", "theme": other, "length_min": 45})   # (the armed night the next check had)
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
        # == SHAPE CARDS ON THE PAGE (the payloads this run saved: no night, a night at 90 min, the like night) ===========
        if "--no-page" not in sys.argv:
            print("SHAPE CARDS (the page, headless Edge)")
            ld = {}
            for k in ("none", "90", "like"):
                fp = os.path.join(out_dir, f"night_payload_{k}.json")
                ld[k] = json.load(open(fp, encoding="utf-8")) if os.path.isfile(fp) else None
            if ld["none"] is None or ld["90"] is None:
                check("the page check has its payloads (no night, a night at 90 min)", False)
            else:
                page_check(out_dir, ld["none"], ld["90"], ld["like"], ld["90"].get("night_presets") or [])
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
