"""The Director: high-level behaviour over the two engines that work.

The autoDJ (lib/dj/system.py: one song at a time, seams through the gates) stays exactly as it is. The
stem conductor (lib/dj/remix.py: parts of two or three songs together) is the other engine. The Director
owns one of them at a time on the show's AudioEngine and translates SIX DIALS into what they do:

    songs    a theme, a playlist as the pool, and an optional UP NEXT list (honoured when it can be)
    mixing   auto | blend | cut | morph         the seam family (pins through the gates) / the crossfade
    layers   one | two | three                  the autoDJ, or the conductor with two / three songs live
    energy   cool | hold | amp                  the arc lean (and, amped, a tempo journey when layered)
    pace     short | normal | long              how long records play / how often the lanes move
    loops    off | some | lots                  the loop styles' odds / loop moves

plus four moments - NEXT, HOLD, DROP, BREAK - and a verdict on what just happened. The operator never
touches songs' parts, timing or levels. Every dial takes effect at the next musical opportunity.

Switching layers between one and two/three HANDS THE PLAYING SONG ACROSS: the other engine opens on the
same song at the same song time on a bar of the song, and the two master buses crossfade over a beat, so
nothing stops. The Director ticks both engines itself (neither runs its own thread here), and nothing in
this file opens a device: the caller attaches the submixes it is told about.
"""
import threading
import time

from lib.dj.submix import RATE

DIALS = {
    "mixing": ("auto", "blend", "cut", "morph"),
    "layers": ("one", "two", "three"),
    "energy": ("cool", "hold", "amp"),
    "tempo": ("slower", "hold", "faster"),
    "pace": ("short", "normal", "long"),
    "seams": ("quick", "normal", "long"),
    "vocals": ("none", "some", "lots"),
    "variety": ("close", "varied", "wild"),
    "loops": ("off", "some", "lots"),
}
DIALS.update({
    # how the music is PLAYED, not only how it transitions
    "bass": ("flat", "boost", "heavy"),
    "tone": ("dark", "neutral", "bright"),
    "level": ("quiet", "normal", "loud"),
    "moments": ("rare", "some", "lots"),
    "fx": ("none", "some", "lots"),
    # the ARC: the night's energy plan - its shape and how long it runs; "yours" = a curve you bent on the strip
    "arc": ("theme", "steady", "build", "waves", "down", "yours"),
    "length": ("45m", "90m", "3h", "night"),
})
DEFAULTS = {"mixing": "auto", "layers": "one", "energy": "hold", "tempo": "hold", "pace": "normal",
            "seams": "normal", "vocals": "some", "variety": "varied", "loops": "off",
            "bass": "flat", "tone": "neutral", "level": "normal", "moments": "some", "fx": "some",
            "arc": "theme", "length": "90m"}
ARC_SHAPE = {"theme": None, "steady": "flat", "build": "rise", "waves": "peak_wave", "down": "wind_down", "yours": "yours"}
ARC_LEN_S = {"45m": 45 * 60.0, "90m": 90 * 60.0, "3h": 3 * 3600.0, "night": 6 * 3600.0}
ARC_POINTS = 13                                                      # a chosen shape is handed to the engines as this many waypoints

# PROGRAMS: orchestration over the next few songs. Each is a function (i, n) -> the dials for song i of n
# (0-based); the dials it does not name keep their baseline; when the program ends the baseline returns.
def _prog_build(i, n):
    f = (i + 1) / n
    d = {"energy": "hold" if f < 0.5 else "amp", "tempo": "hold" if f < 0.67 else "faster", "pace": "normal"}
    if i == n - 1:
        d.update({"moments": "lots", "fx": "lots"})
    return d


def _prog_cool(i, n):
    f = (i + 1) / n
    return {"energy": "hold" if f < 0.34 else "cool", "tempo": "hold" if f < 0.67 else "slower", "moments": "rare", "pace": "long" if f >= 0.67 else "normal"}


def _prog_peak(i, n):
    return {"energy": "amp", "moments": "lots", "fx": "lots", "pace": "short" if n >= 3 else "normal", "bass": "boost"}


def _prog_vocals_up(i, n):
    f = (i + 1) / n
    return {"vocals": "none" if f < 0.34 else ("some" if f < 0.67 else "lots")}


def _prog_breathe(i, n):
    return {"energy": "hold", "pace": "long", "moments": "rare", "loops": "off", "fx": "none", "seams": "long"}


def _prog_brighten(i, n):
    f = (i + 1) / n
    return {"tone": "dark" if f < 0.34 else ("neutral" if f < 0.67 else "bright"), "energy": "hold" if f < 0.67 else "amp"}


def _prog_darken(i, n):
    f = (i + 1) / n
    return {"tone": "bright" if f < 0.34 else ("neutral" if f < 0.67 else "dark"), "energy": "hold" if f < 0.67 else "cool"}


PROGRAMS = {
    "build": ("BUILD", "energy up song by song, the tempo leaning faster, a big moment on the last", _prog_build),
    "cool": ("COOL", "energy down song by song, longer plays, no moments, the tempo easing", _prog_cool),
    "peak": ("PEAK", "amped, short plays, moments and fx on, the bass up - for these songs only", _prog_peak),
    "vocals": ("VOCALS UP", "from instrumentals to singers across these songs", _prog_vocals_up),
    "breathe": ("BREATHE", "long plays, long seams, nothing added: let the records be records", _prog_breathe),
    "brighten": ("BRIGHTEN", "the tone from dark to bright across these songs, warming up at the end", _prog_brighten),
    "darken": ("DARKEN", "the tone from bright to dark across these songs, cooling at the end", _prog_darken),
}
BASS_LOW = {"flat": 1.0, "boost": 1.3, "heavy": 1.6}                # the mix bus EQ, low band (< 200 Hz)
TONE_HIGH = {"dark": 0.7, "neutral": 1.0, "bright": 1.3}             # the mix bus EQ, high band (> 2.5 kHz)
LEVEL_GAIN = {"quiet": 0.6, "normal": 0.85, "loud": 1.0}             # the mix bus gain
MOMENT_RATE = {"rare": 0.0, "some": 0.25, "lots": 0.6}               # the DJ's own drops / breaks, chance per opportunity
MOMENT_GAP_S = {"rare": 1e9, "some": 480.0, "lots": 240.0}           # ...and never closer than this to the last moment (peaks are spaced)
FX_LEVEL = {"none": 0.0, "some": 0.6, "lots": 1.0}                   # shapes on moves; filter / echo seams' odds
TEMPO_LEAN_BPM = {"slower": -6.0, "hold": 0.0, "faster": 6.0}       # the autoDJ's planned journey
TEMPO_LEAN_X = {"slower": -0.03, "hold": 0.0, "faster": 0.03}        # the conductor's clock
SEAM_SPEED = {"quick": 0.5, "normal": 1.0, "long": 2.0}              # the autoDJ's blend length factor / the crossfade
VOCAL_AXIS = {"none": 0.05, "some": 0.5, "lots": 0.9}                # the brain's vocal axis target
VOCAL_FREEDOM = {"none": 0.0, "some": 0.4, "lots": 0.9}              # the conductor's vocal lane
VARIETY_PERSONA = {"close": "purist", "varied": "off", "wild": "crate_digger"}   # how far picks may roam
MIX_PIN = {"auto": None, "blend": "long_blend", "cut": "cut_at_drop", "morph": "stem_morph"}
MIX_CROSS = {"auto": 2.0, "blend": 4.0, "cut": 0.5, "morph": 2.0}
ENERGY_LEAN = {"cool": -0.25, "hold": 0.0, "amp": 0.25}
PACE_X = {"short": 0.6, "normal": 1.0, "long": 1.5}
PACE_BARS = {"short": 4, "normal": 8, "long": 16}
LOOP_LEVEL = {"off": 0, "some": 1, "lots": 2}
LAYER_BLEND = {"two": 0.3, "three": 0.85}
HANDOVER_BEATS = 1.0


class Director:
    def __init__(self, music_root, engine=None, theme="groove", attach=None):
        """`attach(key, submix)` mounts a submix on the audio engine (the tab passes engine.attach_track;
        the gate passes its own). Library and DB are opened here once and shared with both engines."""
        from lib.dj.db import LibraryDB
        from lib.dj import brain as B
        self.music_root = music_root
        self.engine = engine
        self._attach = attach or (engine.attach_track if engine is not None else (lambda k, s: None))
        self.db = LibraryDB(music_root)
        self.library = [t for t in B.load_library(self.db) if not t.excluded]
        self.theme = theme
        self.dials = dict(DEFAULTS)
        self.pool_name = None
        self.tags = []                   # MOOD: only songs carrying one of these may play (empty = all)
        self.up_next = []                # track ids, in order
        self.system = None               # the autoDJ, when layers = one
        self.rc = None                   # the conductor, when layers = two / three
        self.mode = None                 # "one" | "layered"
        self._switch = None              # a handover in progress
        self._hold_until = None
        self._fed = None                 # (current id, queued id) fed to the autoDJ
        self._loop_k = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self.events = []                 # (hms, text) the Director's own log
        self.last_error = None
        # THE PICTURE: what played on each lane, when, and what is planned ahead - kept as a Timeline
        # (lib/dj/timeline.py) so the same canvas draws it: real clips behind the playhead, ghosts ahead
        from lib.dj.timeline import Timeline
        self.tl = Timeline("director")
        self._bar = 0.0                  # the Director's bar clock across engines (display)
        self._bar_wall = None
        self._lane_snap = {}
        self._cur_song = None
        self._ghosts = []                # clip ids of the current plan ahead
        # WARM HANDOVERS: while the autoDJ plays, a conductor waits with the playing song already decoded
        # and pending; while the conductor plays, the clock song's samples are decoded for the autoDJ.
        # Switching LAYERS is then a bar, not a decode.
        self._rc_warm = None
        self._warm_id = None
        self._warm_one = None            # (track_id, samples) for the autoDJ opener
        self._warm_busy = False
        self._loop_hold = None           # (release_bar) while a loop hold runs
        # ONE NIGHT'S MEMORY across both engines: everything either has played (user: "the system keeps
        # reusing songs" - every handover built a fresh brain that knew nothing of the other's plays)
        self.played = []                 # (track_id, when)
        self._noted = set()
        self.last_verdict = None         # what the last GOOD / BAD did (the tab shows it)
        # TASTE: what GOOD / BAD said about SONGS tonight (their tags), fed to both brains as a lean on the
        # next picks - "more like that" / "less like that" - fading as newer verdicts arrive
        self.taste = {"like": {}, "dislike": {}}
        self._last_moment_t = 0.0        # when the last moment (yours or the DJ's) happened: peaks are spaced
        self.arc_custom = None           # your bent curve [(progress, energy)] when the arc dial says "yours"
        self._arc_jump = None
        self._sent = {}                  # per engine: the steering values it already has (see _send)
        self._dial_seam_t = 0.0          # when a dial turn last brought a seam forward
        self.program = None              # {"key", "n", "i", "baseline", "started"} while a program runs over the next songs
        self._song_seq = 0               # song changes seen (one song: a new record; layered: a new bed)
        # THE STORY on the picture: per-lane levels bar by bar, and marks - what happened where (a seam, a
        # song arriving as a voice, the bed passing, a breakdown before it lands) and what is planned ahead
        self._gains = {ln: {} for ln in ("drums", "bass", "other", "vocals")}
        self.marks = []                  # real: {bar, end_bar, lanes, text, kind}
        self.plan_marks = []             # planned (ghost), rebuilt every tick
        self._mark_seen = 0
        self._last_plan = None

    # -- lifecycle ---------------------------------------------------------------------------------------
    def start(self, threaded=True):
        want = "one" if self.dials["layers"] == "one" else "layered"
        try:
            if want == "one":
                self.system = self._new_system()
            else:
                self.rc = self._new_conductor()
                self.rc.start(threaded=False)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return False
        self.mode = want
        self.running = True
        self._t_start = time.time()
        try:
            from lib.dj.stemload import warm
            warm()                                    # the decode helper process: spawned now, not at the first song
        except Exception:
            pass
        self._event(f"started: {self.describe_dials()}")
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True, name="director")
            self._thread.start()
        return True

    def stop(self, fade_s=1.5):
        self.running = False
        if self.system is not None:
            try:
                self.system.stop(fade_s=fade_s)
            except Exception:
                pass
            self.system = None
        if self.rc is not None:
            try:
                self.rc.stop(fade_s=fade_s)
            except Exception:
                pass
            self.rc = None

    def _run(self):
        while self.running:
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                import traceback
                traceback.print_exc()
            time.sleep(0.2)

    # -- engines -------------------------------------------------------------------------------------------
    def _new_system(self, opener=None):
        """The autoDJ as it runs live (its own thread, background decodes). For a handover it is built
        unthreaded so the Director can fire its first step - the opener - on the chosen bar, then its
        thread is spawned and it runs exactly as live from there (see _spawn)."""
        from lib.dj.system import DJSystem
        sysm = DJSystem(self.music_root, engine=None, theme=self.theme, threaded=opener is None, library=self.library)
        if opener is not None:
            sysm.set_opener(*opener)
        sysm._loop_bias = LOOP_LEVEL[self.dials["loops"]]
        self._apply_to_system(sysm)
        if not sysm.start():
            raise RuntimeError(sysm.last_error or "autoDJ failed to start")
        self._seed_brain(sysm.brain)                  # what the conductor played tonight counts here too
        self._attach("dj_submix", sysm.submix)
        self._apply_to_system(sysm)
        return sysm

    @staticmethod
    def _spawn(sysm):
        """Give an unthreaded autoDJ its live thread after its first step."""
        if sysm.threaded:
            return
        sysm.threaded = True
        sysm._thread = threading.Thread(target=sysm._run, daemon=True, name="dj-brain")
        sysm._thread.start()

    def _new_conductor(self, attach=True):
        from lib.dj.remix import RemixConductor
        rc = RemixConductor(self.db, self.music_root, self.library, theme=self.theme)
        if attach:
            self._attach("dj_remix", rc.submix)
        self._apply_to_conductor(rc)
        return rc

    def _send(self, eng, name, value, fn):
        """Hand a steering value to an engine ONLY when it changed since the last time this engine got it.
        The autoDJ drops its planned NEXT and picks again on flavor / persona / arc / pin changes; re-sending
        the same values on every dial turn or verdict made it change its mind constantly (user 2026-09-12:
        "it can't make up its mind")."""
        # the cache lives ON the engine: keyed by id() a new engine after a handover could inherit a freed
        # engine's id and be told it already had everything
        sent = getattr(eng, "_director_sent", None)
        if sent is None:
            sent = {}
            try:
                eng._director_sent = sent
            except Exception:
                sent = self._sent.setdefault(id(eng), {})
        key = repr(value)
        if sent.get(name) == key:
            return False
        sent[name] = key
        fn(value)
        return True

    def _apply_to_system(self, sysm):
        d = self.dials
        self._send(sysm, "mix_style", MIX_PIN[d["mixing"]], sysm.set_mix_style)
        sysm.set_energy_nudge(ENERGY_LEAN[d["energy"]])           # guarded inside (replans only past 0.05)
        self._send(sysm, "bpm_lean", TEMPO_LEAN_BPM[d["tempo"]], sysm.set_bpm_lean)
        self._send(sysm, "pace", PACE_X[d["pace"]], sysm.set_pace)
        self._send(sysm, "mix_speed", SEAM_SPEED[d["seams"]], sysm.set_mix_speed)
        self._send(sysm, "flavor", {"axis_targets": {"vocal": VOCAL_AXIS[d["vocals"]]}, "require_tags": sorted(self.tags),
                                    "prefer_tags": dict(sorted(self.taste["like"].items())),
                                    "avoid_tags": dict(sorted(self.taste["dislike"].items()))}, sysm.set_flavor)
        self._send(sysm, "persona", VARIETY_PERSONA[d["variety"]], sysm.set_persona)
        self._send(sysm, "loop_bias", LOOP_LEVEL[d["loops"]], sysm.set_loop_bias)
        try:
            fx = FX_LEVEL[d["fx"]]
            for k in ("filter_sweep", "echo_out"):
                sysm.brain.style_fb[k] = 0.3 if fx == 0.0 else (1.0 if fx < 1.0 else 2.5)
        except Exception:
            pass
        self._apply_bus(sysm.submix)
        self._send(sysm, "arc", [(round(p, 4), round(e, 4)) for p, e in self._arc_points()], sysm.set_arc_waypoints)
        if abs(ARC_LEN_S[d["length"]] - getattr(sysm, "set_cycle_s", 0)) > 1.0:
            self._send(sysm, "set_len", ARC_LEN_S[d["length"]], sysm.set_set_length)
        if self.pool_name:
            self._send(sysm, "pool", self.pool_name, lambda n: sysm.load_setlist(n, mode="pool"))

    def _apply_to_conductor(self, rc):
        d = self.dials
        rc.avoid_ids = self._played_ids()             # nothing played tonight comes back
        if not getattr(rc, "_seeded", False):
            self._seed_brain(rc.brain)
            rc._seeded = True
        rc.policy = "arrangement"                     # a bed and voices, one change at a time
        try:
            rc.brain.set_require_tags(self.tags)
        except Exception:
            pass
        rc.set_auto(1.0)
        rc.set_cross_beats(MIX_CROSS[d["mixing"]] * SEAM_SPEED[d["seams"]])
        rc.set_blend(LAYER_BLEND.get(d["layers"], 0.3))
        rc.set_energy_lean(ENERGY_LEAN[d["energy"]])
        rc.set_tempo_span(0.03 if d["energy"] == "amp" else 0.0)
        rc.set_tempo_lean(TEMPO_LEAN_X[d["tempo"]])
        rc.set_change_bars(PACE_BARS[d["pace"]])
        rc.set_vocal_freedom(VOCAL_FREEDOM[d["vocals"]])
        rc.fx_level = FX_LEVEL[d["fx"]]
        # the bed changes with a four-bar breakdown first (tension, then the new drums and bass land) when
        # moments are allowed and the mixing is not a blend or a morph (those cross the bed gradually)
        # only when you asked for cuts: the four-bar drop-out before a bed lands read as "fading things out
        # for no reason" under auto mixing
        rc.strip_before_bed = d["moments"] != "rare" and d["mixing"] == "cut"
        rc.set_arc_waypoints(self._arc_points())
        rc.set_arc_length(ARC_LEN_S[d["length"]])
        try:
            rc.brain.flavor["prefer_tags"] = dict(self.taste["like"])
            rc.brain.flavor["avoid_tags"] = dict(self.taste["dislike"])
        except Exception:
            pass
        self._apply_bus(rc.submix)
        try:
            from lib.dj.persona import PERSONAS
            name = VARIETY_PERSONA[d["variety"]]
            rc.brain.persona = PERSONAS.get("neutral" if name == "off" else name, rc.brain.persona)
        except Exception:
            pass
        if self.pool_name:
            from lib.dj.setlist import get_setlist
            sl = get_setlist(self.db, name=self.pool_name)
            rc.set_pool([e["track_id"] for e in (sl or {}).get("entries", [])])
            rc.pool_name = self.pool_name

    # -- the arc: the night's energy plan ------------------------------------------------------------------
    def _theme_obj(self):
        from lib.dj.themes import get_theme
        try:
            if self.system is not None and self.system.brain is not None:
                return self.system.brain.theme
            if self.rc is not None and self.rc.brain is not None:
                return self.rc.brain.theme
        except Exception:
            pass
        return get_theme(self.theme)

    def _arc_points(self):
        """The arc as waypoints for the engines: [] for the theme's own curve; a chosen shape sampled over the
        theme's energy floor and swing; your bent curve as it is."""
        shape = ARC_SHAPE[self.dials["arc"]]
        if shape is None:
            return []
        if shape == "yours":
            return list(self.arc_custom or [])
        import dataclasses
        th = dataclasses.replace(self._theme_obj(), arc=shape)
        return [(i / (ARC_POINTS - 1), th.arc_target(i / (ARC_POINTS - 1))) for i in range(ARC_POINTS)]

    def _arc_engine(self):
        return self.system if self.system is not None else self.rc

    def arc_progress(self):
        eng = self._arc_engine()
        try:
            return float(eng.arc_progress()) if eng is not None else 0.0
        except Exception:
            return 0.0

    def arc_length_s(self):
        return ARC_LEN_S[self.dials["length"]]

    def arc_curve(self, n=48):
        """The plan: (progress, energy) samples of the base curve with the ENERGY dial's lean on it."""
        pts = self._arc_points()
        lean = ENERGY_LEAN[self.dials["energy"]]
        th = self._theme_obj()

        def base(p):
            if pts:
                xs = [x for x, _ in pts]
                ys = [y for _, y in pts]
                if p <= xs[0]:
                    return ys[0]
                if p >= xs[-1]:
                    return ys[-1]
                for i in range(len(xs) - 1):
                    if xs[i] <= p <= xs[i + 1]:
                        f = (p - xs[i]) / max(xs[i + 1] - xs[i], 1e-6)
                        return ys[i] + f * (ys[i + 1] - ys[i])
            return th.arc_target(p)
        return [(i / (n - 1), max(0.0, min(1.0, base(i / (n - 1)) + lean))) for i in range(n)]

    def arc_jump(self, p):
        """'We are here': both engines' set clocks move so the arc reads `p` now (the next pick follows)."""
        p = max(0.0, min(0.999, float(p)))
        if self.system is not None:
            self.system.set_arc_progress(p)
        if self.rc is not None:
            self.rc.set_arc_progress(p)
        self._arc_jump = (p, time.time())
        self._event(f"arc: we are at {100 * p:.0f} % of the {self.dials['length']} arc")

    def arc_bend(self, p, e):
        """Bend the curve: the waypoint nearest `p` (on a 13-point grid of the current curve) moves to energy
        `e`; the arc dial becomes 'yours'."""
        p, e = max(0.0, min(1.0, float(p))), max(0.0, min(1.0, float(e)))
        cur = self.arc_curve(ARC_POINTS)
        lean = ENERGY_LEAN[self.dials["energy"]]
        pts = [(x, max(0.0, min(1.0, y - lean))) for x, y in cur]     # the base, without the lean
        if self.dials["arc"] == "yours" and self.arc_custom:
            pts = list(self.arc_custom)
        i = min(range(len(pts)), key=lambda k: abs(pts[k][0] - p))
        pts[i] = (pts[i][0], max(0.0, min(1.0, e - lean)))
        # neighbours follow a little, so a bend is a hill, not a spike
        for k, w in ((i - 1, 0.5), (i + 1, 0.5)):
            if 0 <= k < len(pts):
                pts[k] = (pts[k][0], max(0.0, min(1.0, pts[k][1] + w * (pts[i][1] - pts[k][1]))))
        self.arc_custom = pts
        if self.dials["arc"] != "yours":
            self.dials["arc"] = "yours"
            self._event("arc: yours (bent on the strip)")
        # the engines get the bent curve once the hand has been still for a second (every mouse move
        # would replan the next song)
        self._arc_dirty = time.time()

    def arc_status(self):
        """The arc for the strip: where we are, the plan, what actually played (progress, energy)."""
        eng = self._arc_engine()
        p = self.arc_progress()
        length = self.arc_length_s()
        target = None
        try:
            target = float(eng.arc_target()) if self.system is not None else float(eng.energy_target())
        except Exception:
            pass
        curve = self.arc_curve()
        # the peak ahead: the highest point of the plan after now
        ahead = [(x, y) for x, y in curve if x >= p]
        peak = max(ahead, key=lambda t: t[1]) if ahead else None
        word = "steady"
        try:
            later = next((y for x, y in curve if x >= min(1.0, p + 0.06)), curve[-1][1])
            now_y = next((y for x, y in curve if x >= p), curve[-1][1])
            word = "building" if later > now_y + 0.03 else ("easing" if later < now_y - 0.03 else ("at the peak" if now_y >= 0.7 else "steady"))
        except Exception:
            pass
        played = []
        by_id = {t.id: t for t in self.library}
        try:
            for tid, when in self.played[-40:]:
                t = by_id.get(tid)
                if t is None:
                    continue
                age = time.time() - when
                pp = p - age / length
                if 0.0 <= pp <= 1.0:
                    played.append((round(pp, 4), round(float(t.energy_proxy()), 3), t.title))
        except Exception:
            pass
        return {"progress": p, "length_s": length, "elapsed_s": p * length, "target": target, "shape": self.dials["arc"],
                "word": word, "curve": curve, "played": played,
                "peak_in_s": ((peak[0] - p) * length if peak else None), "peak_energy": (peak[1] if peak else None),
                "heard": (played[-1][1] if played else None)}

    # -- programs: orchestration over the next few songs ------------------------------------------------------
    def start_program(self, key, n=3):
        """Run `key` over this song and the next n-1 (a bed change counts as a song when layered). The dials
        the program names are stepped at every song change; the others stay yours; the baseline returns at
        the end (or on cancel)."""
        if key not in PROGRAMS:
            return False
        n = int(max(2, min(6, n)))
        if self.program is not None:
            self.cancel_program(quiet=True)
        label, _tip, fn = PROGRAMS[key]
        touched = set()
        for i in range(n):
            touched |= set(fn(i, n))
        self.program = {"key": key, "label": label, "n": n, "i": 0, "baseline": {k: self.dials[k] for k in touched},
                        "started": time.time(), "seq0": self._song_seq}
        self._apply_program_step()
        self._event(f"program {label} over {n} songs")
        return True

    def cancel_program(self, quiet=False):
        p = self.program
        if p is None:
            return
        self.program = None
        for k, v in p["baseline"].items():
            if self.dials.get(k) != v:
                self.dials[k] = v
        if self.system is not None:
            self._apply_to_system(self.system)
        if self.rc is not None:
            self._apply_to_conductor(self.rc)
        if not quiet:
            self._event(f"program {p['label']} {'done' if p['i'] >= p['n'] else 'cancelled'}: your dials are back")

    def _apply_program_step(self):
        p = self.program
        if p is None:
            return
        _label, _tip, fn = PROGRAMS[p["key"]]
        step = fn(p["i"], p["n"])
        changed = []
        for k, v in step.items():
            if self.dials.get(k) != v:
                self.dials[k] = v
                changed.append(f"{k} {v}")
        if changed:
            if self.system is not None:
                self._apply_to_system(self.system)
            if self.rc is not None:
                self._apply_to_conductor(self.rc)
        self._event(f"{p['label']} song {p['i'] + 1} of {p['n']}: " + (", ".join(changed) if changed else "as set"))

    def _song_changed(self):
        """A new record (one song) or a new bed (layered): the program's next step."""
        self._song_seq += 1
        p = self.program
        if p is None:
            return
        p["i"] += 1
        if p["i"] >= p["n"]:
            self.cancel_program()
            return
        self._apply_program_step()

    def program_status(self):
        p = self.program
        if p is None:
            return None
        _label, tip, fn = PROGRAMS[p["key"]]
        steps = []
        for i in range(p["n"]):
            s = fn(i, p["n"])
            steps.append({"i": i, "dials": s, "text": ", ".join(f"{k} {v}" for k, v in s.items()), "now": i == p["i"], "done": i < p["i"]})
        return {"key": p["key"], "label": p["label"], "n": p["n"], "i": p["i"], "steps": steps, "tip": tip}

    # -- the dials ------------------------------------------------------------------------------------------
    def set_dial(self, name, value):
        if name not in DIALS or value not in DIALS[name]:
            return False
        if self.dials.get(name) == value:
            return True
        self.dials[name] = value
        self._event(f"{name}: {value}")
        if self.system is not None:
            self._apply_to_system(self.system)
        if self.rc is not None:
            self._apply_to_conductor(self.rc)
        self._react(name)
        return True

    def _react(self, name):
        """A dial turn is heard within a phrase, not at some seam minutes away ("I'm not trying to steer
        a container ship"). One song: pace re-draws this record's hold now; any other dial brings the
        next seam forward once the record has played 45 s, so the new mixing / energy / loops are heard on
        the next song within a phrase or two. Layered: a move on the next bar."""
        if name == "layers":
            return
        if name == "loops":
            self._loop_k = None                       # a hold may come at the very next phrase boundary
        if self.system is not None and self.system.current is not None:
            if name == "pace":
                self.system.redraw_exit()
            played = self._played_s()
            # heard within a phrase - but not on every touch: once the record has played 90 s, and at most
            # one dial-forced seam every two minutes (three dials in a row skipped three records: "terrible")
            if played >= 90.0 and name in ("mixing", "energy", "loops", "pace") \
                    and time.time() - self._dial_seam_t >= 120.0:
                if name != "pace" or self.dials["pace"] == "short":
                    self.system.request_skip()
                    self._dial_seam_t = time.time()
                    self._event(f"{name}: the next seam comes now")
        if self.rc is not None and self.rc.master is not None and name in ("mixing", "pace", "vocals", "loops", "energy", "moments"):
            self.rc.nudge_move()                      # the next phrase decides with the new dial - no forced move

    def _played_s(self):
        sysm = self.system
        if sysm is None or sysm.current is None:
            return 0.0
        try:
            return (sysm.submix.clock - sysm._started_clock) / RATE
        except Exception:
            return 0.0

    def _warm_step(self):
        """Keep the OTHER engine's opener ready for the playing song."""
        if self._warm_busy or self._switch is not None:
            return
        # not in the first half minute: the standby engine's brain is seconds of GIL, and the start already
        # cost the audio thread one (the user's log: an underrun right after START)
        if time.time() - getattr(self, "_t_start", 0.0) < 30.0:
            return
        if self.system is not None and self.system.current is not None:
            cur = self.system.current
            if self._warm_id == cur.id and self._rc_warm is not None:
                return
            self._warm_busy = True
            track = cur

            def work():
                try:
                    from lib.dj.features import decode_file_stereo
                    from lib.dj.stems import load_stems
                    samples = decode_file_stereo(self.db.abs(track.path))
                    stems = load_stems(self.music_root, track.id, expected_len=len(samples))
                    if not stems:
                        self._warm_id = track.id
                        return
                    rc = self._rc_warm if self._rc_warm is not None else self._new_conductor(attach=False)
                    rc.drop_preload()
                    rc.preload_opener(track, samples, stems)
                    self._rc_warm, self._warm_id = rc, track.id
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"warm: {type(e).__name__}: {e}"
                    self._warm_id = track.id
                finally:
                    self._warm_busy = False
            threading.Thread(target=work, daemon=True, name="director-warm").start()
        elif self.rc is not None and self.rc.master is not None:
            master = self.rc.songs[self.rc.master].track
            if self._warm_one is not None and self._warm_one[0] == master.id:
                return
            self._warm_busy = True

            def work1():
                try:
                    from lib.dj.features import decode_file_stereo
                    self._warm_one = (master.id, decode_file_stereo(self.db.abs(master.path)))
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"warm: {type(e).__name__}: {e}"
                finally:
                    self._warm_busy = False
            threading.Thread(target=work1, daemon=True, name="director-warm").start()

    def set_theme(self, name):
        self.theme = name
        if self.system is not None:
            self.system.set_theme(name)
        if self.rc is not None:
            self.rc.set_theme(name)
        self._event(f"theme: {name}")

    def set_pool(self, name):
        self.pool_name = name or None
        if self.system is not None:
            self.system.load_setlist(name or "", mode="pool")
        if self.rc is not None:
            if name:
                self._apply_to_conductor(self.rc)
            else:
                self.rc.set_pool(None)
        self._event(f"songs: {name or 'whole library'}")

    def queue(self, track_id):
        tid = int(track_id)
        if tid not in self.up_next:
            self.up_next.append(tid)
            self._event(f"up next: {self.title(tid)}")

    def unqueue(self, track_id):
        self.up_next = [t for t in self.up_next if t != int(track_id)]

    def clear_queue(self):
        self.up_next = []

    def set_tags(self, tags):
        """MOOD: a hard tag filter on picks in both engines (empty = everything). Heard on the next pick."""
        self.tags = [str(t) for t in (tags or [])]
        if self.system is not None:
            self._apply_to_system(self.system)
            if self.system.current is not None and self._played_s() >= 45.0:
                self.system.request_reroll()
        if self.rc is not None:
            self._apply_to_conductor(self.rc)
        self._event("mood: " + (", ".join(self.tags) if self.tags else "everything"))

    def describe_dials(self):
        d = self.dials
        return (f"{d['layers']} song{'s' if d['layers'] != 'one' else ''} · {d['mixing']} mixing · energy {d['energy']} · "
                f"{d['pace']} plays · loops {d['loops']}")

    # -- the moments -----------------------------------------------------------------------------------------
    def next(self):
        if self.system is not None:
            self.system.request_skip()
        if self.rc is not None:
            self.rc.next_move()
        self._event("NEXT")

    def hold(self):
        if self.system is not None:
            self.system.request_hold()
        if self.rc is not None:
            self.rc.set_hold(True)
            self._hold_until = self.rc.bar_n + self.rc.change_bars
        self._event("HOLD")

    def drop(self):
        if self.system is not None:
            self.system.moment("drop")
        if self.rc is not None:
            self.rc.drop()
        self._last_moment_t = time.time()
        self._event("DROP")

    def break_(self):
        if self.rc is not None:
            self.rc.break_()
            self._last_moment_t = time.time()
            self._event("BREAK")
            return True
        return False

    # -- taste: what your verdicts say about the SONGS -------------------------------------------------------
    def _learn_song(self, track_id, up):
        """A verdict also teaches SONG choice: the rated song's tags lean the next picks toward (GOOD) or
        away from (BAD) songs like it. Older leans fade by half with every new verdict, so the night follows
        your last few words, not a ledger."""
        t = next((x for x in self.library if x.id == track_id), None) if track_id is not None else None
        if t is None:
            return None
        tags = [str(x) for x in (getattr(t, "all_tags", None) or [])][:3]
        if not tags:
            return None
        for side in ("like", "dislike"):
            for k in list(self.taste[side]):
                self.taste[side][k] = round(self.taste[side][k] * 0.5, 3)
                if self.taste[side][k] < 0.08:
                    del self.taste[side][k]
        side, other = ("like", "dislike") if up else ("dislike", "like")
        for tag in tags:
            self.taste[side][tag] = min(1.0, self.taste[side].get(tag, 0.0) + 0.6)
            self.taste[other].pop(tag, None)
        return tags

    def rate(self, up):
        """GOOD / BAD: say WHAT was rated, what the DJ learns from it, and - on a BAD - change the music
        now (a fresh move layered; the next seam brought forward in one-song mode), so the button is
        visibly consequential."""
        up = bool(up)
        info = {"up": up, "t": time.time(), "what": None, "learn": None, "did": None, "style": None}
        song_id = None
        if self.system is not None:
            h = next((x for x in reversed(self.system._history) if x.get("via") != "start"), None)
            style = h.get("via") if h else None
            self.system.seam_feedback(up)
            info["what"] = f"the mix into {h.get('title')} ({style})" if h else "the last mix"
            info["style"] = style
            info["learn"] = (f"tonight {style} seams are {'favoured' if up else 'held back'} (the style weight rebuilds from your verdicts, "
                             f"and the pair is remembered as {'good' if up else 'rough'})" if style else "stored")
            cur = self.system.current
            song_id = cur.id if cur is not None else None
            if not up and cur is not None and self._played_s() >= 45.0:
                self.system.request_skip()
                info["did"] = "moving on to the next song at the next phrase"
        if self.rc is not None:
            m = self.rc.rate_last(up)
            if m is not None:
                w = self.rc._w(m["kind"], m.get("lane"))
                info["what"] = m["text"]
                info["learn"] = f"moves of that kind ({m['kind'].replace('_', ' ')}{', ' + m['lane'] if m.get('lane') else ''}) now weigh {w:.2f} (0.50 = neutral): {'more' if up else 'less'} of them"
                song_id = m.get("to_id") or m.get("from_id")
            else:
                info["what"] = "nothing left to rate"
            if not up and self.rc.master is not None:
                self.rc.next_move()
                info["did"] = "a different move on the next bar"
        tags = self._learn_song(song_id, up)
        if tags:
            info["learn"] = (info["learn"] or "") + f"; songs tagged {', '.join(tags)} are {'favoured' if up else 'held back'} for the next picks"
            # a BAD reaches the engines now (the next pick must change); a GOOD waits for the next steering
            # change or song - a GOOD must never make the DJ drop the NEXT it had chosen
            if not up:
                if self.system is not None:
                    self._apply_to_system(self.system)
                if self.rc is not None:
                    self._apply_to_conductor(self.rc)
        self.last_verdict = info
        self._event(("GOOD" if up else "BAD") + (f": {info['what'][:50]}" if info.get("what") else ""))
        return info

    # -- the tick ---------------------------------------------------------------------------------------------
    def step(self):
        if not self.running:
            return
        if self.system is not None and not self.system.threaded:
            self.system.step()
        if self.rc is not None:
            self.rc.step()
            for s in self.rc.songs.values():
                if s.staged_at is not None:
                    self._note_played(s.track.id)
            self.rc.avoid_ids = self._played_ids()
            if self._hold_until is not None and self.rc.bar_n >= self._hold_until:
                self.rc.set_hold(False)
                self._hold_until = None
            self._loops_layered()
        try:
            self._loops_one()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"loops: {type(e).__name__}: {e}"
        if getattr(self, "_arc_dirty", None) is not None and time.time() - self._arc_dirty >= 1.0:
            self._arc_dirty = None
            if self.system is not None:
                self._apply_to_system(self.system)
            if self.rc is not None:
                self._apply_to_conductor(self.rc)
        self._feed_up_next()
        self._handover()
        self._warm_step()
        try:
            self._loudness_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"loudness: {type(e).__name__}: {e}"
        try:
            self._auto_moments()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"moments: {type(e).__name__}: {e}"
        try:
            self._record()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"picture: {type(e).__name__}: {e}"

    def _feed_up_next(self):
        """The UP NEXT list: the head goes to the autoDJ as its next track (once per song), or is staged
        for the conductor when a deck is free; popped when it plays."""
        if not self.up_next:
            return
        head = self.up_next[0]
        if self.system is not None and self.system.current is not None:
            cur = self.system.current.id
            if cur == head:
                self.up_next.pop(0)
                self._fed = None
                return
            if self._fed != (cur, head):
                self.system.request_next(head)
                self._fed = (cur, head)
        elif self.rc is not None and self.rc.master is not None:
            d = self.rc.deck_of(head)
            if d is not None and self.rc.songs[d].entered:
                self.up_next.pop(0)
                return
            if d is None:
                ok, msg = self.rc.stage_track(head)
                if not ok and msg not in ("no free deck - eject one first", "already on a deck"):
                    self._event(f"up next {self.title(head)}: {msg}")
                    self.up_next.pop(0)

    def _loops_one(self):
        """The LOOPS dial in one-song mode: the autoDJ's LOOP LAYER - a drum loop cut from another record
        rides under the playing one for a stretch (deck C, never over an armed seam, never from the song it
        plays under). Once per record at most, on a groove, with the record's payoff behind it (its first
        drop or hook heard) and enough runway ahead; some = a third of records, lots = most."""
        level = LOOP_LEVEL[self.dials["loops"]]
        sysm = self.system
        if level <= 0 or sysm is None or sysm.current is None:
            return
        if sysm.state != "playing":
            self._loop_note = f"the autoDJ is {sysm.state}"
            return
        cur = sysm.current
        if getattr(self, "_loop_song", None) != cur.id:
            self._loop_song, self._loop_tried = cur.id, False
        if getattr(sysm, "_layer_txn", None):
            self._loop_note = "a loop is riding now"
            return
        if self._loop_tried:
            return
        played = self._played_s()
        pos = None
        try:
            pos = sysm._pos_s()
        except Exception:
            pass
        if pos is None or played < 30.0:
            self._loop_note = "this record has not played 30 s yet"
            return
        sec = cur.section_at(pos) or {}
        if sec.get("kind") != "groove":
            self._loop_note = f"waiting for a groove (now: {sec.get('kind') or '?'})"
            return
        # the payoff first: past the first drop (a groove after a build / breakdown) or past the hook
        secs = cur.sections or []
        first_drop = next((secs[i]["start_s"] for i in range(1, len(secs))
                           if secs[i].get("kind") == "groove" and secs[i - 1].get("kind") in ("build", "breakdown")), None)
        hook = getattr(cur, "hook", None)
        payoff = min([x for x in (first_drop, hook["start_s"] if hook else None) if x is not None] or [0.0])
        if pos < payoff:
            self._loop_note = f"the record's payoff comes first ({payoff - pos:.0f} s)"
            return
        left = (cur.duration_s or 0.0) - pos
        if left < 60.0:
            self._loop_note = "too close to the record's end"
            return
        self._loop_tried = True
        import random
        if random.random() < {1: 0.35, 2: 0.75}[level]:
            sysm.layer()
            self._loop_note = "a drum loop was asked for on this record"
            self._event(f"loops: a drum loop from another record rides under {cur.title[:30]} for a stretch")
        else:
            self._loop_note = "the dice let this record play plain"

    def _loops_layered(self):
        """The loops dial in layered mode: the clock song holds EIGHT bars of a groove - never a build or a
        breakdown, never while it is singing - starting on a phrase boundary and released on the bar
        (user: four-bar holds on whatever was playing, released on a timer, were "bad")."""
        lvl = LOOP_LEVEL[self.dials["loops"]]
        rc = self.rc
        if rc.master is None:
            return
        if self._loop_hold is not None:
            deck, release_at = self._loop_hold
            if rc.bar_n >= release_at or deck not in rc.songs or rc.songs[deck].leaving:
                if deck in rc.songs:
                    rc.song_loop(deck, None)
                self._loop_hold = None
                self._event("loop released")
            return
        if lvl == 0 or rc.phrase_bars != 0:
            return                                     # only at a phrase boundary
        if self._loop_k is not None and rc.bar_n - self._loop_k < rc.change_bars * (3 if lvl == 1 else 1):
            return
        self._loop_k = rc.bar_n
        song = rc.songs[rc.master]
        d = rc._tel_deck(rc.master)
        t_now = float(d.get("time_s") or 0.0)
        sec = song.track.section_at(t_now) or {}
        kinds_ok = ("groove",) if lvl == 1 else ("groove", "breakdown")
        if sec.get("kind") not in kinds_ok or (lvl == 1 and rc._singing(song, t_now)):
            return
        if rc.rng.random() < (0.5 if lvl == 1 else 0.9):
            if rc.song_loop(rc.master, 8):
                self._loop_hold = (rc.master, rc.bar_n + 8)
                self._event(f"loop: {song.track.title[:28]} holds eight bars")

    # -- the handover between engines --------------------------------------------------------------------------
    def _bar_clock_of(self, sub, track, deck):
        """(clock, song_time) of the next bar of `track` on `deck` of submix `sub`, at least 0.6 s ahead."""
        import math
        tel = sub.telemetry or {}
        d = (tel.get("decks") or {}).get(deck) or {}
        now = int(tel.get("clock", sub.clock))
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        seg = next((s for s in track.grid or [] if s["start_s"] <= time_s <= s["end_s"]), (track.grid or [None])[0])
        if seg is None:
            return now + int(0.6 * RATE), time_s + 0.6 * rate
        bar = 4 * seg["period_s"]
        first_down = seg["first_beat_s"] + track.downbeat_offset * seg["period_s"]
        t_min = time_s + 0.6 * rate
        k = math.ceil((t_min - first_down) / bar - 1e-6)
        t_next = first_down + k * bar
        return now + int((t_next - time_s) / rate * RATE), t_next

    def _handover(self):
        want = "one" if self.dials["layers"] == "one" else "layered"
        if self._switch is None:
            if want == self.mode:
                return
            if want == "layered" and self.system is not None and self.system.current is not None \
                    and self.system.state == "playing":
                cur = self.system.current
                warm = self._rc_warm if (self._rc_warm is not None and self._warm_id == cur.id and not self._warm_busy
                                         and self._rc_warm.decoded()) else None
                rc = warm if warm is not None else self._new_conductor(attach=False)
                self._rc_warm, self._warm_id = None, None
                self._attach("dj_remix", rc.submix)
                rc.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": 0.0})
                rc.start(first_track=cur, threaded=False, cue_s=0.0, lanes=set(("drums", "bass", "other", "vocals")), hold_open=True)
                self._switch = {"to": "layered", "rc": rc, "stage": "decoding", "t": time.time()}
                self._event(f"layers: handing {cur.title} to the conductor" + (" (warm)" if warm is not None else ""))
            elif want == "one" and self.rc is not None and self.rc.master is not None:
                master = self.rc.songs[self.rc.master].track
                self._switch = {"to": "one", "track": master, "stage": "decoding", "t": time.time()}
                if self._warm_one is not None and self._warm_one[0] == master.id:
                    self._switch["samples"], self._switch["stage"] = self._warm_one[1], "ready"
                else:
                    def work():
                        try:
                            from lib.dj.features import decode_file_stereo
                            self._switch["samples"] = decode_file_stereo(self.db.abs(master.path))
                            self._switch["stage"] = "ready"
                        except Exception as e:  # noqa: BLE001
                            self._switch["error"] = str(e)
                    threading.Thread(target=work, daemon=True).start()
                self._event(f"layers: handing {master.title} back to the autoDJ" + (" (warm)" if self._switch["stage"] == "ready" else ""))
            return
        sw = self._switch
        if sw.get("error") or time.time() - sw["t"] > 60:
            self._event(f"handover abandoned: {sw.get('error') or 'timed out'}")
            self._switch = None
            return
        if sw["to"] == "layered":
            rc = sw["rc"]
            if sw["stage"] == "decoding":
                rc.step()
                if not rc.decoded():
                    return
                sysm = self.system
                T_sys, song_t = self._bar_clock_of(sysm.submix, sysm.current, sysm.active_deck)
                T_rc = rc.submix.clock + (T_sys - sysm.submix.clock)
                beat = 60.0 / max(sysm.current.bpm, 60.0)
                rc.open_pending(T_rc, song_t)
                sysm.submix.post({"at": T_sys, "cmd": "mix_gain", "value": 0.0, "ramp_s": HANDOVER_BEATS * beat})
                rc.submix.post({"at": T_rc, "cmd": "mix_gain", "value": self.level_gain, "ramp_s": HANDOVER_BEATS * beat})
                sw["stage"], sw["done_at"] = "crossing", T_sys + int((HANDOVER_BEATS * beat + 0.3) * RATE)
                return
            rc.step()
            if self.system.submix.clock >= sw["done_at"]:
                old = self.system
                self.system = None
                self.rc = rc
                self.mode = "layered"
                self._switch = None
                try:
                    old.stop(fade_s=0.05)
                except Exception:
                    pass
                self._event("layers: the conductor has the room")
        else:
            if sw["stage"] != "ready":
                return
            rc = self.rc
            # the conductor collapses to its clock's song first (every lane to the master, a cut)
            if sw.get("collapsed") is None:
                for ln in ("drums", "bass", "other", "vocals"):
                    if rc.lanes.get(ln) != rc.master:
                        rc.assign(ln, rc.master, force=True)
                sw["collapsed"] = rc.bar_n
                return
            if rc.bar_n < sw["collapsed"] + 2:
                return
            if sw.get("sys") is None:
                T_rc, song_t = self._bar_clock_of(rc.submix, sw["track"], rc.master)
                sysm = self._new_system(opener=(sw["track"], song_t, sw["samples"]))
                sysm.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": 0.0})
                sw["sys"], sw["T_rc"], sw["song_t"] = sysm, T_rc, song_t
                return
            sysm = sw["sys"]
            lead = int(0.05 * RATE)
            if rc.submix.clock < sw["T_rc"] - lead:
                return
            beat = 60.0 / max(sw["track"].bpm, 60.0)
            sysm.step()                                    # the opener starts now, at song time song_t
            self._spawn(sysm)                              # and from here the autoDJ runs as it does live
            sysm.submix.post({"cmd": "mix_gain", "value": self.level_gain, "ramp_s": HANDOVER_BEATS * beat})
            rc.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": HANDOVER_BEATS * beat})
            old = rc
            self.rc = None
            self.system = sysm
            self.mode = "one"
            self._switch = None
            threading.Timer(HANDOVER_BEATS * beat + 0.4, lambda: old.stop(fade_s=0.05)).start()
            self._event("layers: the autoDJ has the room")

    # -- how the music is played: the bus, the DJ's own moments -----------------------------------------------------
    @property
    def level_gain(self):
        return LEVEL_GAIN[self.dials["level"]]

    def _apply_bus(self, sub):
        """BASS / TONE on the mix bus EQ, LEVEL on the bus gain (never while a handover crossfade owns it)."""
        d = self.dials
        sub.post({"cmd": "master_eq", "low": BASS_LOW[d["bass"]], "mid": 1.0, "high": TONE_HIGH[d["tone"]], "ramp_s": 0.5})
        if self._switch is None:
            sub.post({"cmd": "mix_gain", "value": self.level_gain * getattr(self, "_loud_trim", 1.0), "ramp_s": 0.4})

    def _loudness_step(self):
        """LOUDNESS MATCH across the engines. A layered room - a bed with its voice lanes resting, a voice
        with its lows carved and its melody ducked under a singer - measures several dB under a whole record
        even with every deck loudness-compensated (user: "things seem a lot less loud when multiple songs
        are playing"). The autoDJ's output RMS is the night's reference (a slow average while it plays);
        layered, the conductor's bus gain is trimmed UP toward it - never down, at most +6 dB, a slow ramp."""
        now = time.time()
        if now - getattr(self, "_loud_t", 0.0) < 0.5:
            return
        self._loud_t = now
        src = self.system if self.system is not None else self.rc
        if src is None or self._switch is not None:
            return
        try:
            rms = float((src.submix.telemetry or {}).get("rms") or 0.0)
        except Exception:
            return
        if rms < 1e-4:
            return
        # the measurement is the bus OUTPUT (after the gain), so this is a closed loop: the trim walks until
        # the layered output measures like the autoDJ's did
        cur_trim = getattr(self, "_loud_trim", 1.0)
        if self.system is not None:
            self._ref_rms = rms if getattr(self, "_ref_rms", None) is None else 0.97 * self._ref_rms + 0.03 * rms
            if abs(cur_trim - 1.0) > 0.01:
                self._loud_trim = 1.0
                self._apply_bus(self.system.submix)
            return
        ref = getattr(self, "_ref_rms", None)
        if ref is None:
            return
        self._lay_rms = rms if getattr(self, "_lay_rms", None) is None else 0.9 * self._lay_rms + 0.1 * rms
        want = max(1.0, min(2.0, ref / max(self._lay_rms, 1e-6)))
        trim = cur_trim + 0.25 * (want - cur_trim)          # a slow walk, no pumping
        if abs(trim - cur_trim) > 0.02:
            self._loud_trim = trim
            self._apply_bus(self.rc.submix)

    def _auto_moments(self):
        """The MOMENTS dial: the DJ makes its own moments. One song: a double-drop into the next song
        (the nextdrop moment) once per record, when the record has played 60 % and the next has a drop.
        Layered: a BREAK at a breakdown of the bed song (the conductor's auto break, scaled) and now and
        then a DROP onto a voice that has been heard two phrases - the bed and voice become one song."""
        rate = MOMENT_RATE[self.dials["moments"]]
        if rate <= 0.0:
            return
        # PEAKS ARE SPACED AND EARNED: never closer than MOMENT_GAP_S to the last moment (yours or the
        # DJ's), and at "some" only while the arc asks for heat (a peak in the warm-up is a peak wasted)
        spaced = time.time() - self._last_moment_t >= MOMENT_GAP_S[self.dials["moments"]]
        hot = True
        try:
            heat = self.system.arc_target() if self.system is not None else (self.rc.energy_target() if self.rc is not None else 0.5)
            hot = heat >= 0.45 or self.dials["moments"] == "lots" or self.dials["energy"] == "amp"
        except Exception:
            pass
        if self.system is not None and self.system.current is not None and self.system.state == "playing":
            cur = self.system.current
            if getattr(self, "_moment_song", None) != cur.id:
                self._moment_song, self._moment_done = cur.id, False
            if not self._moment_done and cur.duration_s and self._played_s() >= 0.6 * cur.duration_s:
                self._moment_done = True
                import random
                if spaced and hot and random.random() < rate:
                    self.system.moment("nextdrop")
                    self._last_moment_t = time.time()
                    self._event("moment: double-drop into the next song")
        elif self.rc is not None and self.rc.master is not None:
            self.rc.auto_break_scale = rate / 0.5 if spaced else 0.0
            if self.rc.phrase_bars == 0 and getattr(self, "_moment_k", None) != self.rc.bar_n:
                self._moment_k = self.rc.bar_n
                voices = [s for d, s in self.rc.songs.items() if s.entered and not s.leaving and d != self.rc.lanes.get("drums")
                          and s.voice_since is not None and self.rc.bar_n - s.voice_since >= 2 * self.rc.change_bars]
                if voices and spaced and hot and self.rc.rng.random() < rate * 0.35:
                    self.rc.drop()
                    self._last_moment_t = time.time()
                    self._event("moment: DROP onto the voice - bed and voice become one song")

    # -- one night's memory --------------------------------------------------------------------------------------------
    def _note_played(self, tid):
        if tid is None or tid in self._noted:
            return
        self._noted.add(tid)
        self.played.append((int(tid), time.time()))

    def _played_ids(self, hours=3.0):
        cutoff = time.time() - hours * 3600.0
        return {tid for tid, when in self.played if when >= cutoff}

    def _seed_brain(self, brain):
        for tid, when in self.played:
            t = next((x for x in self.library if x.id == tid), None)
            if t is not None:
                try:
                    brain.note_played(t, when=when)
                except Exception:
                    pass

    # -- the picture: the run as a timeline ----------------------------------------------------------------------------
    def _bpm(self):
        if self.system is not None and self.system.current is not None:
            return float(self.system.current.bpm or 120.0)
        if self.rc is not None and self.rc.master_bpm:
            return float(self.rc.master_bpm)
        return 120.0

    def bar(self):
        """The Director's bar clock: advances with wall time at the playing tempo, across engines."""
        now = time.time()
        if self._bar_wall is not None and self.running:
            self._bar += (now - self._bar_wall) * self._bpm() / 240.0
        self._bar_wall = now
        return self._bar

    def _bar_len(self, tid):
        t = next((x for x in self.library if x.id == tid), None)
        return 4 * t.period_s if t is not None else 2.0

    def _record(self):
        """Write what is heard onto the Director's timeline (real clips) and the plan ahead (ghosts)."""
        b = int(self.bar())
        LANES = ("drums", "bass", "other", "vocals")

        def ghost(key, track_id, lanes, at, start_s):
            """The plan ahead as a dashed clip - kept where it is while the plan is the same song within a
            bar of where it was (re-adding it every tick made it hop; user: 'the timeline keeps morphing')."""
            old = getattr(self, "_ghost_key", None)
            if old is not None and old[0] == key[0] and old[1] == key[1] and abs(old[2] - key[2]) <= 1 \
                    and all(any(x.id == cid for x in self.tl.clips) for cid in self._ghosts):
                return
            for cid in self._ghosts:
                c = next((x for x in self.tl.clips if x.id == cid), None)
                if c is not None:
                    self.tl.clips.remove(c)
            self._ghosts = [c.id for c in self.tl.add(track_id, lanes, at, start_s, ghost=True)]
            self._ghost_key = key

        def no_ghost():
            for cid in self._ghosts:
                c = next((x for x in self.tl.clips if x.id == cid), None)
                if c is not None:
                    self.tl.clips.remove(c)
            self._ghosts = []
            self._ghost_key = None

        def mark(text, kind, lanes=None, bar=None, end_bar=None):
            self.marks.append({"bar": int(bar if bar is not None else b), "end_bar": (int(end_bar) if end_bar is not None else None),
                               "lanes": lanes, "text": text, "kind": kind})
            if len(self.marks) > 400:
                del self.marks[:-400]
        self._record_gains(b)
        self.plan_marks = []
        if self.system is not None:
            sysm = self.system
            cur = sysm.current
            if cur is None:
                return
            tel = (sysm.submix.telemetry or {}).get("decks", {}).get(sysm.active_deck) or {}
            pos = float(tel.get("time_s") or 0.0)
            if self._cur_song != cur.id:
                first = self._cur_song is None
                self._cur_song = cur.id
                self._note_played(cur.id)
                if not first:
                    self._song_changed()
                self.tl.add(cur.id, list(LANES), b, max(0.0, pos - (self.bar() - b) * self._bar_len(cur.id)))
                self._lane_snap = {ln: cur.id for ln in LANES}
                # the seam that just happened: its style and length, as a band ending here
                h = next((x for x in reversed(sysm._history) if x.get("via") != "start"), None)
                lp = self._last_plan or {}
                beats = float(lp.get("beats") or 0.0)
                if first or h is None:
                    mark(f"{cur.title[:28]} opens", "open")
                else:
                    mark(f"seam: {h.get('via') or lp.get('style') or '?'}" + (f" {beats:.0f} beats" if beats else "") + f" → {cur.title[:24]}",
                         "seam", lanes=list(LANES), bar=b - int(round(beats / 4.0)) if beats else b, end_bar=b)
                self._last_plan = None
            st = None
            nxt = sysm.next_track
            if nxt is not None:
                try:
                    st = sysm.status()
                except Exception:
                    st = None
                eta = (st or {}).get("blend_in_s")
                plan = (st or {}).get("plan") or {}
                if plan:
                    self._last_plan = dict(plan)
                at = max(b + 1, b + (int(round(eta / self._bar_len(cur.id))) if eta is not None else 24))
                ghost(("one", nxt.id, at), nxt.id, list(LANES), at, float(plan.get("in_s") or 0.0))
                beats = float(plan.get("beats") or 0.0)
                self.plan_marks.append({"bar": at, "end_bar": at + max(1, int(round(beats / 4.0))) if beats else None, "lanes": list(LANES), "ghost": True,
                                        "text": (f"seam: {plan.get('style')} {beats:.0f} beats → {nxt.title[:24]}" if plan else f"next: {nxt.title[:24]} (seam not planned yet)"),
                                        "kind": "seam"})
            else:
                no_ghost()
        elif self.rc is not None:
            rc = self.rc
            if rc.master is None:
                return
            for ln in LANES:
                d = rc.lanes.get(ln)
                s = rc.songs.get(d) if d else None
                tid = s.track.id if s is not None else None
                if self._lane_snap.get(ln, "unset") != tid:
                    self._lane_snap[ln] = tid
                    if tid is None:
                        from lib.dj.timeline import Clip
                        prev = self.tl.active(ln, b - 1)
                        self.tl.clips.append(Clip(prev.track_id if prev else 0, ln, b, 0.0, end_bar=None, rest=True))
                    else:
                        pos = float(rc._tel_deck(d).get("time_s") or 0.0)
                        self.tl.add(tid, [ln], b, pos)
            # what happened: the conductor's moves become marks (bands for a bed change with a breakdown)
            titles = {}
            for m in rc.move_log:
                if m["id"] <= self._mark_seen:
                    continue
                self._mark_seen = m["id"]
                k = m["kind"]
                to_t = self.title(m.get("to_id")) if m.get("to_id") else ""
                fr_t = self.title(m.get("from_id")) if m.get("from_id") else ""
                if k == "voice_in":
                    mark(f"{to_t[:22]} arrives as a voice ({m.get('lane')})", k, lanes=[m.get("lane")])
                elif k == "bed_to":
                    self._song_changed()
                    strip = "drop out for" in (m.get("text") or "")
                    land = (rc._land_k - rc.bar_n) if rc._land_k is not None else 0
                    why = "at its hook" if "at its hook" in m["text"] else ("at its drop" if "at its drop" in m["text"] else "after its phrases")
                    mark(("breakdown, then " if strip else "") + f"the bed → {to_t[:22]} {why}", k, lanes=["drums", "bass"],
                         end_bar=(b + land) if land > 0 else None)
                elif k == "voice_out":
                    mark(f"{fr_t[:22]} fades out", k, lanes=["other", "vocals"])
                elif k == "drop":
                    mark(f"DROP → {to_t[:22]}", k, lanes=list(LANES))
                elif k in ("break", "break_auto"):
                    mark(("BREAK: " if k == "break" else "breakdown: ") + f"{m.get('lane')} alone", k, lanes=[ln for ln in LANES if ln != m.get("lane")])
                elif k in ("rest", "return"):
                    mark(f"{m.get('lane')} {'rest' if k == 'rest' else 'back'}", k, lanes=[m.get("lane")])
                elif k in ("loop", "evict", "recall", "save"):
                    mark(m["text"][:36], k)
            # the plan ahead: the staged song arrives as a VOICE (the arrangement's `other` lane) at the move
            # the bed's settling allows - one steady ghost, not a lane-hopping one; the story row says what
            # each live song is waiting for
            left = max(0, rc.change_bars - 1 - rc.phrase_bars)
            try:
                arr = rc._arrangement_status()
                settle = int(((arr.get("bed") or {}).get("settle_left")) or 0)
                if arr.get("landing_in_bars"):
                    settle = max(settle, int(arr["landing_in_bars"]) + rc.change_bars)
                    self.plan_marks.append({"bar": b, "end_bar": b + int(arr["landing_in_bars"]), "lanes": ["drums", "bass"], "ghost": True,
                                            "text": "the new bed lands", "kind": "bed_to"})
                bed = arr.get("bed") or {}
                if bed and settle > 0 and not arr.get("landing_in_bars"):
                    self.plan_marks.append({"bar": b + settle, "end_bar": None, "lanes": ["other"], "ghost": True,
                                            "text": f"bed {bed['title'][:18]} settled: a voice may arrive", "kind": "voice_in"})
                for v in arr.get("voices") or []:
                    if v.get("leaving"):
                        self.plan_marks.append({"bar": b + rc.change_bars, "end_bar": None, "lanes": [v["lane"]], "ghost": True,
                                                "text": f"{v['title'][:18]} fades out", "kind": "voice_out"})
                    elif v.get("drop_in_bars") is not None and not v.get("may_take_bed_in"):
                        self.plan_marks.append({"bar": b + int(round(v["drop_in_bars"])), "end_bar": None, "lanes": ["drums", "bass"], "ghost": True,
                                                "text": f"{v['title'][:18]} takes the bed at its {v.get('payoff') or 'drop'}", "kind": "bed_to"})
                    elif v.get("may_take_bed_in"):
                        self.plan_marks.append({"bar": b + int(v["may_take_bed_in"]), "end_bar": None, "lanes": ["drums", "bass"], "ghost": True,
                                                "text": f"{v['title'][:18]} may take the bed (its {'hook or ' if v.get('hook') else ''}drop first)", "kind": "bed_to"})
                    else:
                        self.plan_marks.append({"bar": b + int(v.get("must_take_bed_in") or rc.change_bars), "end_bar": None, "lanes": ["drums", "bass"], "ghost": True,
                                                "text": f"{v['title'][:18]} takes the bed by then", "kind": "bed_to"})
            except Exception:
                settle = 0
            if settle > left:
                left = left + rc.change_bars * ((settle - left + rc.change_bars - 1) // rc.change_bars)
            staged = [(d, s) for d, s in rc.songs.items() if s.staged_at is not None and not s.entered and not s.leaving]
            if staged:
                d, s = staged[0]
                at = b + 1 + left
                ghost(("layered", s.track.id, at), s.track.id, ["other"], at,
                      rc._song_time_at(d, rc._next_bar_clock()) + left * self._bar_len(s.track.id))
            else:
                no_ghost()

    def _record_gains(self, b):
        """Per lane, the level actually heard this bar (the holder's stem gain × deck gain; one song: the
        deck gains), kept as the max within the bar - the picture's level meters."""
        try:
            if self.rc is not None:
                decks = (self.rc.submix.telemetry or {}).get("decks") or {}
                for ln in self._gains:
                    g = 0.0
                    for d, t in decks.items():
                        if t.get("playing"):
                            g = max(g, float((t.get("stem_gains") or {}).get(ln, 0.0)) * float(t.get("gain") or 0.0))
                    self._gains[ln][b] = max(self._gains[ln].get(b, 0.0), min(1.0, g))
            elif self.system is not None:
                decks = (self.system.submix.telemetry or {}).get("decks") or {}
                g = 0.0
                for d, t in decks.items():
                    if t.get("playing"):
                        sg = t.get("stem_gains") or {}
                        for ln in self._gains:
                            self._gains[ln][b] = max(self._gains[ln].get(b, 0.0), min(1.0, float(sg.get(ln, 1.0)) * float(t.get("gain") or 0.0)))
            for ln in self._gains:
                if len(self._gains[ln]) > 3000:
                    for k in sorted(self._gains[ln])[:-3000]:
                        del self._gains[ln][k]
        except Exception:
            pass

    def lane_gains(self):
        return self._gains

    def all_marks(self):
        out = list(self.marks) + list(self.plan_marks)
        # the program's coming steps, roughly where the next songs will start (a song ≈ 64 bars; the next
        # one at its ghost when there is one)
        p = self.program
        if p is not None:
            b = int(self._bar)
            nxt = min((c.bar for c in self.tl.clips if c.ghost), default=b + 48)
            _label, _tip, fn = PROGRAMS[p["key"]]
            for j, i in enumerate(range(p["i"] + 1, p["n"])):
                s = fn(i, p["n"])
                out.append({"bar": int(nxt + j * 64), "end_bar": None, "lanes": None, "ghost": True, "kind": "program",
                            "text": f"{p['label']} {i + 1}/{p['n']}: " + ", ".join(f"{k} {v}" for k, v in s.items())})
        return out

    # -- the song list: what fits from HERE, and what would be rejected -------------------------------------------------
    def rank(self, query="", n=40, filters=None, sort="fit"):
        """Songs for the UP NEXT list: the pool (or the library) ranked by fit to the reference song's tempo, key
        and the arc (the reference is the last queued song, else the playing one), each with a verdict - '✓ fits
        …' or '✗ would be rejected: …' - and enough of the song's CHARACTER to know it without recalling it:
        energy, singing, its tags and genre, year, length, its hook, and a one-line shape for the tooltip.
        `filters`: {"voice": "inst"|"vocal", "energy": "calmer"|"same"|"hotter", "tempo": "slower"|"same"|
        "faster", "hook": True, "unplayed": True}; `sort`: fit | energy | energy_desc | bpm | title | hook."""
        import math
        from lib.dj.brain import camelot_compat
        q = (query or "").strip().lower()
        filters = filters or {}
        pool_ids = None
        if self.pool_name:
            try:
                from lib.dj.setlist import get_setlist
                sl = get_setlist(self.db, name=self.pool_name)
                pool_ids = {e["track_id"] for e in (sl or {}).get("entries", [])}
            except Exception:
                pool_ids = None
        cur, bpm, cam, arc = None, None, None, 0.6
        layered = self.rc is not None
        if self.system is not None and self.system.current is not None:
            cur = self.system.current
            bpm, cam = float(cur.bpm or 0), cur.camelot
            try:
                arc = self.system.arc_target()
            except Exception:
                arc = 0.6
        elif self.rc is not None and self.rc.master is not None:
            cur = self.rc.songs[self.rc.master].track
            bpm, cam = float(self.rc.master_bpm or cur.bpm), self.rc.key_centre
            arc = self.rc.energy_target()
        # a chain, not a star: with songs queued, the list ranks against the LAST queued song - that is what
        # the next pick will follow (user 2026-09-12: "the next song picker should be based, at least in
        # part, on what the last song in the next song list is")
        self.rank_ref = None
        if self.up_next:
            ref = next((t for t in self.library if t.id == self.up_next[-1]), None)
            if ref is not None:
                cur, bpm, cam = ref, float(ref.bpm or bpm or 0), ref.camelot or cam
                self.rank_ref = ref.title
        brain = self.system.brain if self.system is not None else (self.rc.brain if self.rc is not None else None)
        smin = getattr(brain, "stretch_min", 0.92) if brain else 0.92
        smax = getattr(brain, "stretch_max", 1.08) if brain else 1.08
        rows = []
        for t in self.library:
            if pool_ids is not None and t.id not in pool_ids:
                continue
            if cur is not None and t.id == cur.id:
                continue
            tags_all = [str(x) for x in (getattr(t, "all_tags", ()) or ())]
            genre = ", ".join(sorted(getattr(t, "genre_set", None) or [])[:2])
            year = getattr(t, "year", None)
            if q:
                hay = " ".join([(t.title or ""), (t.artist or ""), genre, str(year or ""), " ".join(tags_all)]).lower()
                if not all(w in hay for w in q.split()):
                    continue
            if self.tags and not (set(self.tags) & set(tags_all)):
                continue
            # the song's character, for the filters and the words
            try:
                e = float(brain._arc_energy(t)) if brain else t.energy_proxy()
            except Exception:
                e = 0.5
            vocal = (t.axes_rank.get("vocal") if getattr(t, "axes_rank", None) else None)
            if vocal is None:
                vocal = (t.axes or {}).get("vocal")
            sings = (vocal is not None and float(vocal) >= 0.5) or bool(getattr(t, "hook", None))
            fv = filters.get("voice")
            if fv == "inst" and sings:
                continue
            if fv == "vocal" and not sings:
                continue
            fe = filters.get("energy")
            if fe and cur is not None:
                try:
                    e_ref = float(brain._arc_energy(cur)) if brain else cur.energy_proxy()
                except Exception:
                    e_ref = 0.5
                # a STEP, not a leap: hotter / calmer means 0.04 .. 0.30 away from the reference (open-ended,
                # with the reference walking up the queue, the list ran out of hotter songs fast)
                if (fe == "calmer" and not (e_ref - 0.30 <= e <= e_ref - 0.04)) or (fe == "hotter" and not (e_ref + 0.04 <= e <= e_ref + 0.30)) \
                        or (fe == "same" and abs(e - e_ref) > 0.12):
                    continue
            ft = filters.get("tempo")
            if ft and bpm and t.bpm:
                if (ft == "slower" and t.bpm > bpm - 1.5) or (ft == "faster" and t.bpm < bpm + 1.5) or (ft == "same" and abs(t.bpm - bpm) > 2.0):
                    continue
            if filters.get("hook") and not getattr(t, "hook", None):
                continue
            if filters.get("unplayed") and any(tid == t.id for tid, _w in self.played):
                continue
            fit, ok, why = 0.5, True, []
            if bpm:
                ratios = [bpm / max(t.bpm * m, 1e-6) for m in (1.0, 2.0, 0.5)] if t.bpm else [9.0]
                r = min(ratios, key=lambda x: abs(math.log(x)))
                if layered:
                    reach = 0.90 <= (bpm / max(t.bpm, 1e-6)) <= 1.10
                else:
                    reach = smin <= r <= smax
                pct = 100.0 * (r - 1.0)
                why.append(f"tempo {pct:+.1f}%")
                if not reach:
                    ok = False
                    why[-1] += " (out of reach: would fade)" if not layered else " (outside the wall)"
                kc = camelot_compat(cam, t.camelot) if (cam and t.camelot) else None
                if kc is not None:
                    why.append(f"key {t.camelot} fit {kc:.2f}" + (" (clash)" if kc < 0.55 else ""))
                    if kc < 0.55 and not layered:
                        why[-1] += " - a fade or a key shift"
                why.append(f"energy {e:.2f} vs {arc:.2f}")
                fit = (0.45 * (kc if kc is not None else 0.6) + 0.35 * max(0.0, 1.0 - abs(math.log(max(r, 1e-6))) / 0.1)
                       + 0.2 * max(0.0, 1.0 - abs(e - arc) / 0.5)) if reach else 0.0
                if (t.bpm_conf or 0) < 0.5:
                    why.append("loose grid (beat-matching unreliable)")
                    fit *= 0.6
            if layered and not getattr(t, "has_stems", False):
                ok = False
                why.append("no stems: cannot be layered")
            hk = getattr(t, "hook", None)
            if hk:
                why.append(f"hook at {int(hk['start_s']) // 60}:{int(hk['start_s']) % 60:02d}" + (f" ×{len(hk.get('starts') or [])}" if len(hk.get("starts") or []) > 1 else ""))
            when = next((w for tid, w in reversed(self.played) if tid == t.id), None)
            if when is not None:
                ok = False
                why.insert(0, f"played {max(1, int((time.time() - when) / 60))} min ago")
            if t.id in self.up_next:
                ok = False
                why.insert(0, "already queued")
            # words for a song you do not remember: its moods and genre, when it is from, how long, its shape
            mood_words = [x.replace("_", " ") for x in tags_all if not x.isdigit() and not x.endswith("s") or x in ("drums", "vocals")][:5]
            vw = None
            mv = getattr(t, "ml_valence", None)
            if mv is not None:
                vw = "dark" if mv < 0.4 else ("bright" if mv > 0.6 else "even")
            shape = []
            secs = t.sections or []
            for i, s in enumerate(secs):
                k = s.get("kind")
                if k in ("build", "breakdown") or (k == "groove" and i > 0 and secs[i - 1].get("kind") in ("build", "breakdown", "intro")):
                    shape.append(f"{'drop' if k == 'groove' else k} {int(s['start_s']) // 60}:{int(s['start_s']) % 60:02d}")
            rows.append({"id": t.id, "title": t.title, "artist": t.artist or "", "bpm": t.bpm, "camelot": t.camelot,
                         "ok": ok, "fit": fit, "why": ", ".join(why) if why else "",
                         "energy": round(e, 2), "sings": bool(sings), "vocal": (round(float(vocal), 2) if vocal is not None else None),
                         "tags": mood_words, "genre": genre, "year": year, "duration_s": t.duration_s,
                         "hook_s": (hk["start_s"] if hk else None), "valence": vw,
                         "shape": " · ".join(shape[:6]), "last_played_min": (int((time.time() - when) / 60) if when is not None else None)})
        keys = {"fit": lambda x: (not x["ok"], -x["fit"], x["title"].lower()),
                "energy": lambda x: (not x["ok"], x["energy"], x["title"].lower()),
                "energy_desc": lambda x: (not x["ok"], -x["energy"], x["title"].lower()),
                "bpm": lambda x: (not x["ok"], x["bpm"] or 0, x["title"].lower()),
                "title": lambda x: (not x["ok"], x["title"].lower()),
                "hook": lambda x: (not x["ok"], x["hook_s"] is None, x["hook_s"] or 0)}
        rows.sort(key=keys.get(sort, keys["fit"]))
        # a relative filter that leaves almost nothing falls back to "same", and says so
        self.rank_note = None
        n_ok = sum(1 for r in rows if r["ok"])
        if n_ok < 5 and filters.get("energy") in ("calmer", "hotter") and not filters.get("_fallback"):
            f2 = dict(filters, energy="same", _fallback=True)
            rows2 = self.rank(query, n=n, filters=f2, sort=sort)
            self.rank_note = f"nothing {filters['energy']} than {(self.rank_ref or (cur.title if cur is not None else 'the room'))[:22]} fits: showing the same energy"
            return rows2
        if n_ok < 5 and filters.get("tempo") in ("slower", "faster") and not filters.get("_fallback"):
            f2 = dict(filters, tempo="same", _fallback=True)
            rows2 = self.rank(query, n=n, filters=f2, sort=sort)
            self.rank_note = f"nothing {filters['tempo']} than {(self.rank_ref or (cur.title if cur is not None else 'the room'))[:22]} fits: showing the same tempo"
            return rows2
        return rows[:n]

    # -- what the operator sees ---------------------------------------------------------------------------------------
    def title(self, tid):
        t = next((x for x in self.library if x.id == tid), None)
        return t.title if t else f"#{tid}"

    def status(self):
        out = {"mode": self.mode, "dials": dict(self.dials), "dials_menu": {k: list(v) for k, v in DIALS.items()},
               "theme": self.theme, "pool": self.pool_name,
               "up_next": [{"id": t, "title": self.title(t)} for t in self.up_next],
               "switching": None if self._switch is None else self._switch.get("to"),
               "events": list(self.events[-12:]), "error": self.last_error, "now": None, "next": None,
               "intent": "", "songs": [], "recent": []}
        try:
            if self.system is not None:
                st = self.system.status()
                cur, nxt, plan = st.get("current") or {}, st.get("next") or {}, st.get("plan") or {}
                if cur:
                    out["now"] = {"title": cur.get("title"), "artist": cur.get("artist"), "pos_s": cur.get("pos_s"),
                                  "duration_s": cur.get("duration_s"), "bpm": cur.get("bpm"), "camelot": cur.get("camelot")}
                if nxt:
                    # the STABLE anchor is the record's exit (drawn once per record); the seam's start moves
                    # with the style each re-plan picks (a 64-beat blend starts a minute before a cut) - the
                    # user saw that as "changing its mind about when to start a new song"
                    exit_in = None
                    try:
                        ep = getattr(self.system, "_exit_played", None)
                        if ep:
                            exit_in = max(0.0, float(ep) - self._played_s())
                    except Exception:
                        exit_in = None
                    out["next"] = {"title": nxt.get("title"), "artist": nxt.get("artist"), "eta_s": st.get("blend_in_s"),
                                   "exit_in_s": exit_in, "how": plan.get("style"), "beats": plan.get("beats")}
                bits = [f"one song at a time", f"{self.dials['mixing']} mixing" if self.dials["mixing"] != "auto" else "mixing as the dice fall",
                        f"energy {self.dials['energy']}", f"{self.dials['pace']} plays"]
                try:
                    # the arc in words: where the night's energy is heading over the next stretch (the strip's reading)
                    a = self.arc_status()
                    bits.append(f"arc {a['word']} (target {a['target']:.2f}, {a['elapsed_s'] / 60:.0f} of {a['length_s'] / 60:.0f} min)")
                except Exception:
                    pass
                if plan:
                    eta = st.get("blend_in_s")
                    bits.append(f"next seam: {plan.get('style')} " + (f"in {eta:.0f} s" if eta is not None else "when the exit comes"))
                played = self._played_s()
                if played < 45.0:
                    bits.append(f"a dial turn acts once this record has played 45 s ({45 - played:.0f} s more)")
                else:
                    bits.append("a dial turn brings the next seam within a phrase")
                out["intent"] = ", ".join(bits)
                out["recent"] = [f"{time.strftime('%H:%M:%S', time.localtime(e.get('t', 0)))}  {e.get('event')}  "
                                 + ", ".join(f"{k} {v}" for k, v in e.items() if k not in ("t", "clock_s", "event") and not isinstance(v, (dict, list)))[:120]
                                 for e in (st.get("recent_events") or [])[-8:]]
                out["state"] = st.get("state")
                out["error"] = out["error"] or self.system.last_error
            elif self.rc is not None:
                st = self.rc.status()
                songs = st.get("songs") or {}
                lanes = st.get("lanes") or {}
                m = songs.get(st.get("master")) or {}
                if m:
                    out["now"] = {"title": m.get("title"), "artist": m.get("artist"), "pos_s": m.get("time_s"),
                                  "duration_s": m.get("duration_s"), "bpm": st.get("master_bpm"), "camelot": st.get("key_centre")}
                out["songs"] = [{"title": s.get("title"), "lanes": s.get("lanes"), "entered": s.get("entered")} for s in songs.values()]
                staged = [s for s in songs.values() if not s.get("entered")]
                if staged:
                    out["next"] = {"title": staged[0]["title"], "artist": staged[0].get("artist"),
                                   "eta_s": max(0, (st.get("change_bars", 8) - 1 - st.get("phrase_bars", 0))) * 4 * 60.0 / max(st.get("master_bpm") or 120, 1),
                                   "how": "in through one lane"}
                live = [s for s in songs.values() if s.get("entered")]
                bits = [f"{len(live)} song{'s' if len(live) != 1 else ''} layered", f"a move every {st.get('change_bars')} bars",
                        "vocal lane free" if self.rc.vocal_freedom > 0.5 else "vocals cautious", f"energy {self.dials['energy']}"]
                left = st.get("change_bars", 8) - 1 - st.get("phrase_bars", 0)
                bits.append(f"next move in {max(0, left)} bars" + (" (held)" if st.get("hold") else ""))
                # the arrangement's plan, in words: the bed and how long it has to settle, each voice and what it waits for
                arr = st.get("arrangement") or {}
                bed = arr.get("bed")
                if bed:
                    b = f"bed: {bed['title'][:26]}"
                    if bed.get("since_bars") is not None:
                        b += f" for {bed['since_bars']} bars"
                    if arr.get("landing_in_bars") is not None:
                        b += f" (a new bed lands in {arr['landing_in_bars']} bars)"
                    elif bed.get("settle_left"):
                        b += f", settles {bed['settle_left']} more before a new voice"
                    elif bed.get("kind") == "build":
                        b += " (building to its drop)"
                    bits.append(b)
                for v in arr.get("voices") or []:
                    if v.get("leaving"):
                        bits.append(f"{v['title'][:26]} fades out as a voice")
                    elif v.get("may_take_bed_in"):
                        bits.append(f"voice {v['title'][:26]}: heard {v['heard_bars']} bars, may take the bed in {v['may_take_bed_in']}")
                    elif v.get("drop_in_bars") is not None:
                        bits.append(f"voice {v['title'][:26]} takes the bed at its drop in ~{v['drop_in_bars']:.0f} bars")
                    else:
                        bits.append(f"voice {v['title'][:26]} takes the bed by {v['must_take_bed_in']} bars (no drop in sight)")
                out["intent"] = ", ".join(bits)
                out["wait_why"] = arr.get("wait_why")
                out["lanes"] = {ln: (songs.get(d) or {}).get("title") if d else None for ln, d in lanes.items()}
                out["recent"] = [f"{t}  {m}" for t, m in (st.get("moves") or [])[-8:]]
                out["error"] = out["error"] or st.get("error")
        except Exception as e:  # noqa: BLE001
            out["error"] = f"status: {type(e).__name__}: {e}"
        try:
            out["why"], out["in_effect"] = self._explain()
        except Exception as e:  # noqa: BLE001
            out["why"], out["in_effect"] = [f"(explain: {type(e).__name__}: {e})"], []
        # DOUBLED: the same stem from two songs audible at once (a crossfade in flight, or the autoDJ's blend)
        doubled = []
        try:
            if self.rc is not None:
                decks = (self.rc.submix.telemetry or {}).get("decks") or {}
                for ln in ("drums", "bass", "other", "vocals"):
                    n = sum(1 for d, t in decks.items() if t.get("playing") and float((t.get("stem_gains") or {}).get(ln, 0.0)) > 0.15
                            and float(t.get("gain") or 0.0) > 0.1)
                    if n > 1:
                        doubled.append(ln)
            elif self.system is not None:
                decks = (self.system.submix.telemetry or {}).get("decks") or {}
                up = [d for d, t in decks.items() if t.get("playing") and float(t.get("gain") or 0.0) > 0.1]
                if len(up) > 1:
                    style = (self.system.plan or {}).get("style") or ""
                    doubled = ["drums", "bass", "other", "vocals"] if not style.startswith(("stem_", "acapella", "drum_bridge", "bass_swap")) else ["mix"]
        except Exception:
            pass
        out["doubled"] = doubled
        try:
            out["arc"] = self.arc_status()
        except Exception as e:  # noqa: BLE001
            out["arc"] = {"error": f"{type(e).__name__}: {e}"}
        out["program"] = self.program_status()
        out["programs"] = {k: (v[0], v[1]) for k, v in PROGRAMS.items()}
        if out["program"]:
            ps = out["program"]
            nxt_steps = [s for s in ps["steps"] if not s["done"] and not s["now"]]
            out["intent"] = (out.get("intent") or "") + f", {ps['label']} song {ps['i'] + 1} of {ps['n']}" + (f" (next: {nxt_steps[0]['text']})" if nxt_steps else " (last)")
        lv = self.last_verdict
        if lv is not None:
            lv = dict(lv)
            if lv.get("style") and self.system is not None:
                try:
                    lv["weight"] = round(float(self.system.brain.style_fb.get(lv["style"], 1.0)), 2)
                except Exception:
                    pass
        out["last_verdict"] = lv
        return out

    def _explain(self):
        """WHY the DJ is doing what it does, and what each dial is doing right now - in words."""
        d = self.dials
        why, eff = [], []
        if self.system is not None:
            sysm = self.system
            st = sysm.status()
            plan = st.get("plan") or {}
            nxt = st.get("next") or {}
            hz = st.get("horizon") or []
            if nxt:
                reason = next((h.get("why") for h in hz if h.get("title") == nxt.get("title") and h.get("why")), None)
                why.append(f"next song {nxt.get('title')}: " + (reason or "the brain's best fit to the theme, the arc and the playing song's tempo and key"))
            if plan:
                chips = st.get("seam_chips") or []
                line = f"the seam will be {plan.get('style')} ({plan.get('beats')} beats)"
                if d["mixing"] != "auto":
                    if plan.get("pin"):
                        line += f" - your '{d['mixing']}' dial" + (f", allowed past {plan['pin_waived']}" if plan.get("pin_waived") else "")
                    else:
                        line += f" - your '{d['mixing']}' dial was refused: {plan.get('pin_why_not') or 'gated'}"
                if chips:
                    line += "; " + ", ".join(str(c) for c in chips[:4])
                why.append(line)
            elif st.get("state") == "playing":
                why.append("the seam is not planned yet: the brain plans when the record's exit comes into range")
            exit_s = getattr(sysm, "_exit_played", None)
            eff.append(f"mixing {d['mixing']}: " + ("the dice choose" if d['mixing'] == 'auto' else f"every seam pinned to {MIX_PIN[d['mixing']]} (family fallback, a refused cut becomes a phrase cut)"))
            eff.append(f"layers one: the autoDJ, one song at a time")
            eff.append(f"energy {d['energy']}: arc target {st.get('arc_heat', 0):.2f} with lean {ENERGY_LEAN[d['energy']]:+.2f}")
            eff.append(f"arc {d['arc']}: " + ("the theme's own curve" if d['arc'] == 'theme' else ("your bent curve" if d['arc'] == 'yours' else f"a {d['arc']} curve over the theme's energy floor and swing")) + f", {100 * sysm.arc_progress():.0f} % through")
            eff.append(f"length {d['length']}: the arc runs {ARC_LEN_S[d['length']] / 60:.0f} min")
            eff.append(f"tempo {d['tempo']}: picks aim at {sysm.bpm_target():.0f} bpm" + (f" ({TEMPO_LEAN_BPM[d['tempo']]:+.0f})" if d['tempo'] != 'hold' else ""))
            eff.append(f"pace {d['pace']}: this record holds about {exit_s:.0f} s" if exit_s else f"pace {d['pace']}")
            eff.append(f"seams {d['seams']}: blend lengths ×{SEAM_SPEED[d['seams']]:g}")
            eff.append(f"vocals {d['vocals']}: picks lean to vocal presence {VOCAL_AXIS[d['vocals']]:.2f}")
            eff.append(f"variety {d['variety']}: persona {VARIETY_PERSONA[d['variety']] if VARIETY_PERSONA[d['variety']] != 'off' else 'neutral'}")
            riding = bool(getattr(sysm, "_layer_txn", None))
            denied = getattr(sysm, "_layer_denied", None)
            note = getattr(self, "_loop_note", None)
            eff.append(f"loops {d['loops']}: " + ("off" if d['loops'] == 'off' else
                       ("a drum loop from another record is riding under this one now" if riding else
                        f"loop entries ×{ {1: 3, 2: 8}[LOOP_LEVEL[d['loops']]] } as likely; a drum loop from another record may ride under a groove once per record"
                        + (f" - {note}" if note else "")
                        + (f" - the autoDJ refused the last one: {denied[0]}" if denied and time.time() - denied[1] < 120 else ""))))
        elif self.rc is not None:
            rc = self.rc
            last = rc.move_log[-1] if rc.move_log else None
            kinds = {"enter": "there was room for another song, so the brain's pick came in through one lane",
                     "cross_new": "a lane crossed toward the newest song (the layers dial leans that way)",
                     "cross": "a lane recombined freely between the live songs",
                     "rest": "a lane rested for a phrase (a breath)", "return": "a resting lane came back",
                     "leave": "a song held no lane any more, so it left", "loop": "a song ran out of body and looped its last bars",
                     "evict": "a song had looped long enough; its lanes moved on", "clock": "the clock passed to the song holding most lanes",
                     "drop": "DROP: every lane to one song", "break": "BREAK", "break_auto": "the song holding most lanes reached a breakdown, so the conductor broke with it",
                     "manual": "your move", "open": "the opener",
                     "voice_in": "a new song arrives as a VOICE over the bed once the bed has settled (drums + bass stay where they are)",
                     "bed_to": "the voice was heard two phrases and reached its drop (or ran out of phrases), so the BED (drums + bass together) passes to it",
                     "voice_out": "the old bed song, now only a voice, fades out; one song's journey is complete"}
            if last is not None:
                why.append(f"last move - {last['text'][:70]}: {kinds.get(last['kind'], last['kind'])}")
            if rc.wait_why:
                why.append(f"the last phrase passed without a move: {rc.wait_why}")
            for dname, s in rc.songs.items():
                if s.staged_at is not None and not s.entered and not s.leaving:
                    fit = f"key fit {s.compat:.2f}" if s.compat is not None else "key unknown"
                    why.append(f"{s.track.title[:34]} is staged: tempo ×{s.rate:.3f}, {fit}{f' shifted {s.shift:+d}' if s.shift else ''}, "
                               f"enters at the next move ({max(0, rc.change_bars - 1 - rc.phrase_bars)} bars)")
            n_live = len([s for s in rc.songs.values() if s.entered and not s.leaving])
            eff.append(f"mixing {d['mixing']}: lanes cross over {rc.cross_beats:g} beat{'s' if rc.cross_beats != 1 else ''}")
            eff.append(f"layers {d['layers']}: {n_live} heard now, up to {2 if rc.blend < 0.5 else 3}; moves lean {'toward the newest song' if rc.blend < 0.5 else 'to free recombination'}")
            eff.append(f"energy {d['energy']}: target {rc.energy_target():.2f}" + (", tempo may climb with it" if rc.tempo_span else ""))
            eff.append(f"arc {d['arc']}: " + ("the theme's own curve" if d['arc'] == 'theme' else ("your bent curve" if d['arc'] == 'yours' else f"a {d['arc']} curve")) + f", {100 * rc.arc_progress():.0f} % through")
            eff.append(f"length {d['length']}: the arc runs {ARC_LEN_S[d['length']] / 60:.0f} min")
            eff.append(f"tempo {d['tempo']}: clock {rc.master_bpm or 0:.1f} bpm" + (f" heading {TEMPO_LEAN_X[d['tempo']] * 100:+.0f}%" if d['tempo'] != 'hold' else ""))
            eff.append(f"pace {d['pace']}: a move every {rc.change_bars} bars")
            eff.append(f"vocals {d['vocals']}: the vocal lane crosses with freedom {rc.vocal_freedom:.1f}")
            eff.append(f"variety {d['variety']}: picks by persona {getattr(rc.brain.persona, 'name', 'neutral')}")
            eff.append(f"loops {d['loops']}: " + ("holding eight bars now" if self._loop_hold else ("eight-bar holds on grooves at phrase boundaries" if d['loops'] != 'off' else "off")))
        return why, eff

    def _event(self, msg):
        self.events.append((time.strftime("%H:%M:%S"), msg))
        if len(self.events) > 200:
            del self.events[:-200]
