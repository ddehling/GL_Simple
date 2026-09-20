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
import os
import threading
import time

from lib.dj.submix import RATE


def _mmss(x):
    try:
        x = max(0, int(round(float(x))))
    except (TypeError, ValueError):
        return "?"
    return f"{x // 60}:{x % 60:02d}"

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
            "bass": "flat", "tone": "neutral", "level": "normal", "moments": "some", "fx": "none",   # (no shapes on exits by default: a "filter out" sweep was flagged as a bad seam; the dial is off the surface)
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
# FLAGS: "this is not working" - or "this worked" - said for LATER, with the whole context, and no learning
# applied - unlike BAD, which lowers a move's weight, leans the song's tags away and pushes the music on.
# key -> what the operator heard. The good flags are the cases to PROTECT when a rule changes.
FLAG_BAD = {
    "dead": "dead air / too quiet",
    "busy": "too busy / too much going on",
    "still": "nothing happening for too long",
    "clash": "clash (key, chords, two songs fighting)",
    "seam": "bad transition / seam",
    "song": "wrong song choice",
    "plan": "the plan changed / could not make up its mind",
    "timing": "off the beat or the phrase",
    "vocals": "vocals wrong (resting, doubled, cut)",
    "level": "too loud / too quiet / thin",
    "bass": "the low end wrong (muddy, missing, two basslines)",
    "energy": "the energy is wrong for the room (too low / too high)",
    "runout": "a song ran out (looping its end, cut off, faded for no reason)",
    "moment": "a moment did not land (SLAM / BREAK / SING / TAKEOVER / BRIDGE)",
    "long": "too long on one song or one combination",
    "queue": "my UP NEXT / steering was ignored",
    "control": "a button or dial did not do what it says",
    "ui": "the screen (bouncing, unclear, wrong)",
    "other": "something else (say what)",
}
FLAG_GOOD = {
    "good_moment": "that moment landed (keep it exactly like this)",
    "good_seam": "great transition / blend",
    "good_layer": "these songs sit well together",
    "good_song": "good song choice here",
    "good_build": "the energy is building right",
    "good_bass": "the low end is right (that bass swap, that weight)",
    "good_vocals": "the vocals sat perfectly",
    "good_calm": "this stretch was right - nothing more was needed",
    "good_level": "the level and balance are right",
    "good_plan": "the plan read right and happened as said",
    "good_other": "something else worked (say what)",
}
FLAG_KINDS = {**FLAG_BAD, **FLAG_GOOD}
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
PLAY_MIN_S = {"short": 150.0, "normal": 210.0, "long": 300.0}       # the record plays at least this long (quick-mix / club / long blends)
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
        self.last_dropout = None         # the last silence on the bus over a bar: when, what was happening
        self.flags = []                  # the FLAGS raised tonight: {t, kind, label, text} (the snapshots are in logs/dj_flags.jsonl)
        self._quiet_since = None
        self._dropout_open = False
        self._pending_moment = None      # a TAKEOVER / BRIDGE waiting for its chosen song to be staged
        self._queued_on_deck = set()     # UP NEXT ids that have been on a deck (they leave the list when they become the bed or leave)
        # TASTE: what GOOD / BAD said about SONGS tonight (their tags), fed to both brains as a lean on the
        # next picks - "more like that" / "less like that" - fading as newer verdicts arrive
        self.taste = {"like": {}, "dislike": {}}
        self._last_moment_t = time.time()  # when the last moment (yours or the DJ's) happened: peaks are spaced - and the first waits the gap too
        self.arc_custom = None           # your bent curve [(progress, energy)] when the arc dial says "yours"
        self._arc_jump = None
        self._sent = {}                  # per engine: the steering values it already has (see _send)
        # THE SET (docs/DJ_THE_SET.md): the chain of records after the queue, the songs you said "not tonight"
        # to, the intermediates of a route ("via"), and the floor's requests (a song being made playable)
        self.horizon = []                # hops from lib/dj/setplan.chain, in order
        self.set_skips = set()           # track ids: not tonight
        self.via_ids = set()             # UP NEXT entries that are a route's intermediates
        self._hz_sig = None              # the triggers' signature at the last plan
        self._hz_t = 0.0                 # when the chain was last checked
        self._hz_why = ""                # why it was last (re)planned
        self._hz_note = ""               # why it is empty, when it is
        self.requests = {}               # key -> {title, stage, pct, t, ready, id, error}
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
        rc.policy = "record"                          # THE RECORD PLAYS: one song whole, one seam per record (docs/DJ_THE_RECORD_PLAYS.md)
        rc.follow_clock = True                        # THE SET: the clock travels to each record's own tempo (docs/DJ_THE_SET.md rule 3)
        rc.pick_hook = self._horizon_pick             # the DJ's spare pick is the chain's head
        rc.set_play_min(PLAY_MIN_S[d["pace"]])
        rc.set_seam(d["mixing"], SEAM_SPEED[d["seams"]])
        rc.set_event_rate({"rare": 0, "some": 1, "lots": 2}[d["moments"]])   # the EVENTS per record, from the pairing lists (docs/DJ_PAIRINGS.md)
        rc.set_event_pref(vocals=d["vocals"], energy=d["energy"])           # VOCALS: which kind of event; ENERGY: which song
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
        # the bass moment: with MOMENTS on and mixing auto / cut, a bed change is bass-first - the bassline
        # takes over at the payoff, loud, the drums follow at the phrase
        rc.bass_moments = d["moments"] != "rare" and d["mixing"] in ("auto", "cut")
        rc.calm_phrases = 2                            # calm between moments: one arrangement move per two phrases
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
        """UP NEXT. A song outside the tempo wall is a ROUTE, not a refusal (docs/DJ_THE_SET.md rule 4): the
        records that reach it go into the queue ahead of it, marked "via"; when nothing reaches it, said.
        Returns (ok, words)."""
        tid = int(track_id)
        if tid in self.up_next:
            if self.rc is not None:
                self.rc.mark_next(tid)
            return True, "already queued"
        rc = self.rc
        t = self.track(tid)
        if rc is not None and rc.deck_of(tid) is not None:
            okm, why_m = rc.mark_next(tid)
            if not okm:
                self._event(f"up next refused: {why_m}")     # (a half-time part: its parts, never the record)
                return False, why_m
        if rc is not None and rc.master is not None and t is not None and t.bpm:
            clock = self._chain_clock()
            rate = float(clock) / float(t.bpm)
            if not (0.90 <= rate <= 1.10):
                hops = self.route_to(tid)
                pct = 100.0 * (float(t.bpm) / float(clock) - 1.0)
                if hops is None:
                    msg = f"{t.title[:28]}: {float(t.bpm):.0f} bpm is {pct:+.0f} % off the clock and no chain of records reaches it tonight"
                    self._event(msg)
                    return False, msg
                for h in hops:
                    if h["id"] not in self.up_next:
                        self.up_next.append(h["id"])
                        self.via_ids.add(h["id"])
                self._event(f"to reach {t.title[:24]} ({float(t.bpm):.0f} bpm, {pct:+.0f} %): via " + ", ".join(f"{h['title'][:20]} ({h['bpm']:.0f})" for h in hops))
        self.up_next.append(tid)
        self._event(f"up next: {self.title(tid)}")
        if rc is not None:
            rc.mark_next(tid)                         # (a song whose part is in becomes the next record only when you say so)
        return True, "up next"

    def up_next_status(self):
        """The UP NEXT list as rows with each song's state: queued (waiting for a deck), staged (on a deck,
        silent, ready), in play (which of its parts sound now) - a song whose stem you placed is here too,
        until it becomes the bed or leaves."""
        rows = []
        rc = self.rc
        for tid in self.up_next:
            state, lanes = "queued", []
            if rc is not None:
                d = rc.deck_of(tid)
                if d is not None and d in rc.songs:
                    s = rc.songs[d]
                    lanes = s.held(rc.lanes)
                    state = ("leaving" if s.leaving else "in play" if lanes else ("staged" if s.staged_at is not None else "decoding"))
            elif self.system is not None and self.system.next_track is not None and self.system.next_track.id == tid:
                state = "next"
            rows.append({"id": tid, "title": self.title(tid), "state": state, "lanes": lanes})
        return rows

    def unqueue(self, track_id):
        self.up_next = [t for t in self.up_next if t != int(track_id)]
        self.via_ids.discard(int(track_id))

    def track(self, tid):
        return next((x for x in self.library if x.id == int(tid)), None)

    # -- THE SET: the chain of records ahead (docs/DJ_THE_SET.md) ----------------------------------------------
    def _arc_at(self, t_s):
        """The arc's energy target `t_s` seconds from now (the lean included)."""
        rc = self.rc
        if rc is None:
            return 0.6
        try:
            length = float(getattr(rc, "arc_len_s", 0) or 0) or 4 * 3600.0
            p = min(1.0, rc.arc_progress() + max(0.0, float(t_s)) / length)
            return max(0.05, min(0.95, rc._arc_base(p) + rc.energy_lean))
        except Exception:
            return 0.6

    def _chain_pinned(self):
        """The records already decided after the current one, in order: the seam's next record (or the staged
        next record), then UP NEXT."""
        rc = self.rc
        seq = []
        if rc is not None:
            pend = getattr(rc, "_event_pending", None)
            if pend is not None and pend.get("kind") == "takeover" and pend.get("track_id") is not None:
                seq.append(int(pend["track_id"]))     # THE EVENT IS THE EXIT: the takeover's song is the next record
            if rc.seam is not None and rc.songs.get(rc.seam["b"]) is not None:
                seq.append(rc.songs[rc.seam["b"]].track.id)
            else:
                for d, s in sorted(rc.songs.items(), key=lambda kv: kv[1].staged_at or 0):
                    if d != rc.record_deck and not s.leaving and s.staged_at is not None and not getattr(s, "part_only", False) and not s.entered \
                            and s.track.id not in seq:
                        seq.append(s.track.id)
                        break
                else:
                    # the conductor's pick on its way (decoding, or decoded and waiting for its deck): the next
                    # record already decided, though not on a deck yet
                    for t in list(rc._decoding.values()) + [p[0] for p in rc._pending.values()]:
                        if t is not None and t.id not in self.up_next and (rc.record_deck not in rc.songs or t.id != rc.songs[rc.record_deck].track.id):
                            seq.append(t.id)
                            break
        for tid in self.up_next:
            if tid not in seq:
                seq.append(tid)
        return seq

    def _chain_clock(self):
        """The clock the next queued song would face: with the clock following the record, the native tempo of
        the last record decided (the seam's B, the last of UP NEXT), else the clock itself."""
        rc = self.rc
        if rc is None or rc.master is None:
            return 0.0
        if not getattr(rc, "follow_clock", True):
            return float(rc.master_bpm)
        pinned = self._chain_pinned()
        last = self.track(pinned[-1]) if pinned else (rc.songs[rc.record_deck].track if rc.record_deck in rc.songs else None)
        return float(last.bpm) if (last is not None and last.bpm) else float(rc.master_bpm)

    def _horizon_step(self):
        """Keep the chain: every two seconds, drop the hops that became impossible (played, the wall from a new
        predecessor, a key clash, not tonight) and extend to five again; a change of the triggers (the record,
        the queue, a skip, the dials that steer picks) does the same at once. Hops that still fit stay - the
        chain does not churn (rule 2)."""
        rc = self.rc
        if rc is None or rc.master is None or rc.record_deck not in rc.songs:
            if self.horizon:
                self.horizon, self._hz_sig = [], None
            return
        now = time.time()
        d = self.dials
        # (the decks' comings and goings are NOT a trigger: a plan is a burst of Python next to the audio producer, and
        # re-planning on every staging starved it - the result is checked against the room when it is applied instead)
        sig = (rc.songs[rc.record_deck].track.id, tuple(self._chain_pinned()), d["arc"], d["energy"], d["tempo"], d["pace"],
               len(self.set_skips), self.pool_name, tuple(self.tags))
        if sig == self._hz_sig and now - self._hz_t < 2.0 and getattr(self, "_hz_result", None) is None:
            return                                    # (a worker's finished result is applied at once, not throttled)
        why = "the triggers changed" if sig != self._hz_sig else ""
        self._hz_sig, self._hz_t = sig, now
        from lib.dj import setplan
        cur = rc.songs[rc.record_deck].track
        pinned = [p for p in self._chain_pinned() if self.track(p) is not None]
        follow = bool(getattr(rc, "follow_clock", True))
        played = self._played_ids() | {cur.id}
        busy = {s.track.id for s in rc.songs.values()} | {t.id for t in rc._decoding.values()} | {p[0].id for p in rc._pending.values()}
        busy -= set(pinned)
        prev = self.track(pinned[-1]) if pinned else cur
        clock = (float(prev.bpm) if (follow and prev.bpm) else float(rc.master_bpm))
        # prune: the first hop that no longer follows its predecessor ends what is kept
        kept = []
        for h in self.horizon:
            t = self.track(h["id"])
            reason = setplan.hop_ok(prev, t, clock, played=played | set(pinned), avoid=self.set_skips | busy, play_min_s=float(rc.play_min_s))
            if reason is not None:
                if reason == "not tonight" and t is not None and t.id in busy and t.id not in self.set_skips:
                    reason = "on the decks now"
                why = why or f"{h['title'][:20]}: {reason}"
                break
            kept.append(h)
            prev, clock = t, (float(t.bpm) if follow else clock)
        try:
            arr = rc._arrangement_status()
            bar_s = rc._bar_clock() / RATE
            sm = arr.get("seam") or {}
            rec = arr.get("record") or {}
            t0 = max(float(sm.get("in_bars") or 0), float(rec.get("settle_left") or 0)) * bar_s
        except Exception:
            t0 = float(rc.play_min_s)
        # the projected times of the kept hops move with the room; the hops themselves do not
        step_s = float(rc.play_min_s) + setplan.SEAM_S
        for i, h in enumerate(kept):
            h["t"] = t0 + (len(pinned) + i + 1) * step_s
        if len(kept) < len(self.horizon):
            self.horizon = list(kept)
            self._hz_why = why
        if len(kept) >= 5:
            return
        kept_ids = [h["id"] for h in kept]
        # the search runs OFF this thread (a plan is 150-350 ms over a thousand songs; this thread steps the
        # conductor): a worker computes the extension, the next step applies it if the room is still the same
        res = getattr(self, "_hz_result", None)
        if res is not None:
            self._hz_result = None
            r_sig, r_kept, more = res
            if r_sig == sig and r_kept == kept_ids:
                # a song that reached the decks or was played while the search ran is not a hop
                more = [h for h in more if h["id"] not in busy and h["id"] not in played and h["id"] not in self.set_skips]
                new = kept + more
                if not new:
                    last = self.track(pinned[-1]) if pinned else cur
                    self._hz_note = (f"nothing fits after {last.title[:22]} ({float(last.bpm):.0f} bpm, {last.camelot or '?'}) tonight: the wall, the key, "
                                     f"and what has played" if last is not None else "nothing fits tonight")
                else:
                    self._hz_note = ""
                if [h["id"] for h in new] != [h["id"] for h in self.horizon]:
                    self._hz_why = why or self._hz_why or "extended"
                    self.horizon = new
                    if self.horizon and self.mode == "layered":
                        self._event("the set: " + " → ".join(h["title"][:16] for h in self.horizon[:5]) + (f"  ({self._hz_why[:40]})" if self._hz_why else ""))
                return
        if getattr(self, "_hz_busy", False):
            return
        lib = [t for t in self.library if getattr(t, "has_stems", False)]
        if rc.brain is not None and getattr(rc.brain, "pool_ids", None):
            lib = [t for t in lib if t.id in rc.brain.pool_ids]
        try:
            recs = (rc._load_pairings() or {}).get("records") or {}
        except Exception:
            recs = {}
        lean = {"slower": -1.0, "hold": 0.0, "faster": 1.0}.get(d["tempo"], 0.0)
        args = dict(clock_bpm=(float(cur.bpm) if follow else float(rc.master_bpm)), arc_fn=self._arc_at, played=set(played), avoid=set(self.set_skips),
                    busy=set(busy), play_min_s=float(rc.play_min_s), n=5 - len(kept), follow=follow, t0=t0,
                    seed=(int(self._hz_t) // 86400) + 1, pinned=list(pinned) + kept_ids, lean=lean)
        self._hz_busy = True

        def work():
            try:
                more = setplan.chain(cur, lib, recs, **args)
                self._hz_result = (sig, kept_ids, more)
            except Exception as e:  # noqa: BLE001
                self.last_error = f"set: {type(e).__name__}: {e}"
            finally:
                self._hz_busy = False
        threading.Thread(target=work, daemon=True, name="director-set").start()

    def _slip_step(self):
        """REAL TIME: the clock song's own time against the wall clock (its rate allowed for). A session on
        2026-09-19 ran at 65-85 % of real time for eight minutes (2.4-3.0 s per bar against 1.97) and nothing said
        so - the flags read 'dead air', 'nothing happening', 'seam'. Measured over ten-second windows; when two in
        a row fall under 0.92 the story row says it, once a minute, and the log gets a row."""
        rc = self.rc
        now = time.time()
        st = getattr(self, "_slip", None)
        if rc is None or rc.master is None or rc.master not in rc.songs:
            self._slip = None
            return
        try:
            tel = rc._tel_deck(rc.master) or {}
            if not tel.get("playing"):
                return
            t_s = float(tel.get("time_s") or 0.0)
        except Exception:
            return
        s = rc.songs[rc.master]
        if st is None or st["deck"] != rc.master or st["tid"] != s.track.id or s.loop is not None or t_s < st["t_s"]:
            self._slip = {"deck": rc.master, "tid": s.track.id, "wall": now, "t_s": t_s, "low": 0, "said": 0.0, "ratio": None}
            return
        if now - st["wall"] < 10.0:
            return
        expected = (now - st["wall"]) * float(s.rate or 1.0)
        ratio = (t_s - st["t_s"]) / max(expected, 1e-6)
        st.update({"wall": now, "t_s": t_s, "ratio": round(ratio, 3)})
        if ratio < 0.92:
            st["low"] += 1
            if st["low"] >= 2 and now - st["said"] > 60.0:
                st["said"] = now
                self._event(f"⚠ the engine is behind real time: the music ran at {100 * ratio:.0f} % over the last ten seconds (stutters and holes follow) - fewer layers, or close other programs")
                try:
                    from lib.dj.session import log_row
                    log_row({"event": "slip", "ratio": round(ratio, 3), "layers": self.dials.get("layers"), "decks": len(rc.songs)})
                except Exception:
                    pass
        else:
            st["low"] = 0

    def _horizon_pick(self, veto):
        """The conductor's pick hook (RemixConductor._pick_next): the chain's head, or "wait" when it is inside
        the wall where the clock is heading but not yet where the clock is. (track or None, wait)."""
        rc = self.rc
        if rc is None or not self.horizon:
            return None, False
        for h in self.horizon[:2]:
            t = self.track(h["id"])
            if t is None or t.id in self.set_skips:
                continue
            if t.id not in veto:
                return t, False
            # vetoed: by the wall against today's clock while the clock travels toward a place it fits?
            base = float(getattr(rc, "base_bpm", 0) or 0)
            if base and getattr(rc, "follow_clock", True) and t.bpm and 0.90 <= base / float(t.bpm) <= 1.10 \
                    and abs(float(rc.master_bpm) / base - 1.0) > 0.005:
                return None, True
            return None, False
        return None, False

    def route_to(self, track_id):
        """The records that reach `track_id` from the end of the queue (docs/DJ_THE_SET.md rule 4): [] when it
        follows directly, hops when a route exists, None when none does."""
        rc = self.rc
        t = self.track(track_id)
        if rc is None or rc.master is None or t is None or rc.record_deck not in rc.songs:
            return None
        from lib.dj import setplan
        pinned = [p for p in self._chain_pinned() if self.track(p) is not None]
        cur = self.track(pinned[-1]) if pinned else rc.songs[rc.record_deck].track
        follow = bool(getattr(rc, "follow_clock", True))
        clock = float(cur.bpm) if (follow and cur.bpm) else float(rc.master_bpm)
        lib = [x for x in self.library if getattr(x, "has_stems", False)]
        try:
            recs = (rc._load_pairings() or {}).get("records") or {}
        except Exception:
            recs = {}
        busy = {s.track.id for s in rc.songs.values()} | set(pinned)
        return setplan.route(cur, t, lib, recs, clock, played=self._played_ids() | {cur.id}, avoid=self.set_skips, busy=busy,
                             play_min_s=float(rc.play_min_s), follow=follow)

    def via_by_tempo(self):
        """{bpm: hops} by tempo alone from the end of the queue (the SONGS list's 'via N' word)."""
        rc = self.rc
        if rc is None or rc.master is None or not getattr(rc, "follow_clock", True):
            return {}
        from lib.dj import setplan
        return setplan.hops_by_tempo(self._chain_clock(), [t.bpm for t in self.library if getattr(t, "has_stems", False)])

    def skip_in_set(self, track_id):
        """"Not tonight": the song leaves the chain and does not come back to it (a right-click on a THEN row)."""
        tid = int(track_id)
        self.set_skips.add(tid)
        self.horizon = [h for h in self.horizon if h["id"] != tid]
        self._hz_sig = None
        self._event(f"not tonight: {self.title(tid)[:30]}")
        return True

    # -- THE PAIRS: the mashup finder (docs/DJ_MORE_THAN_A_MIXER.md 1) ---------------------------------------------
    def _pairings_file(self):
        import json
        path = os.path.join(self.music_root, ".stems", "pairings.json")
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return None
        if getattr(self, "_pairs_mt", None) != mt:
            try:
                self._pairs_data = json.load(open(path, encoding="utf-8"))
                self._pairs_mt = mt
            except Exception:  # noqa: BLE001
                self._pairs_data = None
        return getattr(self, "_pairs_data", None)

    def pairs(self, n=80, fit_clock=True, with_tid=None):
        """The best pairs across the library, from the pairing file: {kind ("vocal" | "bass"), r (the record), s (the
        other song), score, fit, shift, r_start, s_start, words}. `fit_clock`: the record inside the wall of the
        clock now (when one plays); `with_tid`: only pairs that involve this song. Played songs are left out."""
        p = self._pairings_file()
        if not p:
            return []
        recs = p.get("records") or {}
        rc = self.rc
        clock = float(rc.master_bpm) if (rc is not None and rc.master is not None) else None
        played = self._played_ids()
        by_id = {t.id: t for t in self.library}
        out = []
        for rid, r in recs.items():
            R = by_id.get(int(rid))
            if R is None or not getattr(R, "has_stems", False) or not R.bpm:
                continue
            if fit_clock and clock and not (0.90 <= clock / float(R.bpm) <= 1.10):
                continue
            g0 = float((r.get("grooves") or [[max(30.0, float(r.get("landmark") or 30.0))]])[0][0])
            for kind, key in (("vocal", "vocal_over"), ("bass", "bass_under")):
                for x in r.get(key) or []:
                    S = by_id.get(int(x["id"]))
                    if S is None or not getattr(S, "has_stems", False) or not S.bpm:
                        continue
                    if with_tid is not None and int(with_tid) not in (R.id, S.id):
                        continue
                    if R.id in played and S.id in played:
                        continue
                    if kind == "vocal":
                        r_start, s_start = float((x.get("groove") or [g0])[0]), float((x.get("sung") or [0.0])[0])
                        words = f"the vocal of {S.title[:24]} over {R.title[:24]} from {_mmss(r_start)}"
                    else:
                        r_start, s_start = g0, float(x.get("landmark") or 0.0)
                        words = f"{S.title[:24]}'s bassline under {R.title[:24]}'s drums, at its drop {_mmss(s_start)}"
                    out.append({"kind": kind, "r": R, "s": S, "score": float(x.get("score") or 0.0), "fit": x.get("fit"), "shift": int(x.get("shift") or 0),
                                "r_start": r_start, "s_start": s_start, "words": words + f"  (fit {x.get('fit')})" + (f", shift {int(x.get('shift') or 0):+d}" if x.get("shift") else ""),
                                "played": (R.id in played or S.id in played)})
        out.sort(key=lambda z: (-z["score"], z["r"].title))
        return out[:n]

    def drop_pair(self, pair):
        """DROP IT: the pair's record goes UP NEXT (unless it is the record now) and the pair is planted as an idea
        already taken - it fires when that record plays. Returns a sentence."""
        rc = self.rc
        if rc is None or rc.master is None:
            return "layer first: LAYERS two"
        R, S = pair["r"], pair["s"]
        kind = "acapella" if pair["kind"] == "vocal" else "takeover"
        rec_now = rc.songs[rc.record_deck].track.id if rc.record_deck in rc.songs else None
        words = ""
        if R.id != rec_now:
            ok, msg = self.queue(R.id)
            if not ok:
                return f"the pair's record could not be queued: {msg}"
            words = f"{R.title[:24]} is UP NEXT; "
        rc.plant_idea(R.id, kind, S.id)
        self._event(f"PAIR taken: {pair['words'][:80]}")
        return words + (f"when it plays, {S.title[:24]}'s {'vocal comes over it' if kind == 'acapella' else 'bassline takes over at its drop'} (taken)")

    def idea_take(self):
        """"Do it" on the DJ's IDEA row (docs/DJ_MORE_THAN_A_MIXER.md 2). Returns a sentence."""
        if self.rc is None or self.rc.master is None:
            return "not layered"
        ok, msg = self.rc.idea_take()
        self._event(("IDEA taken: " if ok else "IDEA: ") + msg)
        return msg

    def idea_skip(self):
        """"Skip" on the DJ's IDEA row. Returns a sentence."""
        if self.rc is None or self.rc.master is None:
            return "not layered"
        ok, msg = self.rc.idea_skip()
        self._event(("IDEA skipped: " if ok else "IDEA: ") + msg)
        return msg

    # -- THE FLOOR: requests (docs/DJ_THE_SET.md rule 5) -----------------------------------------------------------
    def note_request(self, key, title, stage, pct=None, ready=False, track_id=None, error=None):
        """A request's state for THE PLAN: 'separating', 'pairings', 'analyzing'; ready → 'is ready' for a
        minute; an error stays a minute too."""
        r = self.requests.get(key) or {"title": title, "t0": time.time()}
        r.update({"title": title or r.get("title"), "stage": stage, "pct": pct, "t": time.time(), "ready": bool(ready),
                  "id": (int(track_id) if track_id is not None else r.get("id")), "error": error})
        self.requests[key] = r
        if ready:
            self._event(f"{(title or key)[:30]} is ready to play")
        elif error:
            self._event(f"{(title or key)[:30]}: {error[:60]}")

    def adopt_library(self, tracks):
        """A fresh hydration of the library (from a worker's own DB handle): new songs join, songs that gained
        stems become layerable - without a restart. Returns (new, gained)."""
        known = {t.id: t for t in self.library}
        new, gained = [], []
        for t in tracks:
            old = known.get(t.id)
            if old is None:
                if getattr(t, "excluded", False):
                    continue
                self.library.append(t)
                new.append(t)
                if getattr(t, "has_stems", False):
                    gained.append(t)
            elif getattr(t, "has_stems", False) and not getattr(old, "has_stems", False):
                old.has_stems = True
                gained.append(old)
        rc = self.rc
        if rc is not None:
            ids = {t.id for t in rc.library}
            for t in gained:
                if t.id not in ids:
                    rc.library.append(t)
                    if rc.brain is not None:
                        rc.brain.library.append(t)
                        try:
                            rc.brain._lib_bpms.append(t.bpm)
                        except Exception:
                            pass
            if gained:
                self._hz_sig = None                   # (the pairing file is re-read by its mtime when it changes)
        if new or gained:
            self._event(f"library: {len(new)} new, {len(gained)} now with stems")
        return new, gained

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
            r = self.rc.break_()
            ok = bool(r[0]) if isinstance(r, tuple) else bool(r)
            self._last_moment_t = time.time()
            # a refused break SAYS so (the story row said "BREAK" and nothing happened - the control gate, 2026-09-16)
            self._event("BREAK" if ok else "BREAK refused: " + ((r[1] if isinstance(r, tuple) and len(r) > 1 else "") or "nothing to take away here (one lane, or the record looping its last bars)"))
            return ok
        return False

    # -- the composed moments: SLAM / BREAK / SING / TAKEOVER / BRIDGE ------------------------------------------------
    def arm_moment(self, kind, auto=False):
        """Arm a composed moment (layered). Pressing the armed one again = go now. One song at a time: the
        first moment is the autoDJ's DROP (a build into the next song's drop); the rest need layers.
        Returns (ok, sentence)."""
        if self.rc is not None and self.rc.master is not None:
            ok, msg = self.rc.arm_moment(kind, auto=auto)
            if ok:
                self._last_moment_t = time.time()
                if not auto:
                    self._event(f"{kind.upper()}: {msg}")
            elif not auto:
                self._event(f"{kind.upper()} refused: {msg}")
            return ok, msg
        if self.system is not None:
            if kind == "slam":
                self.drop()
                return True, "DROP: a short build, then the next song's drop"
            return False, "needs songs layered (LAYERS two or three)"
        return False, "press START first"

    def moment_go(self):
        if self.rc is not None:
            return self.rc.moment_go()
        return False, "nothing armed"

    def moment_cancel(self):
        if self.rc is not None:
            ok, msg = self.rc.moment_cancel()
            if ok:
                self._event(msg)
            return ok, msg
        return False, "nothing armed"

    def moment_status(self):
        return self.rc.moment_status() if self.rc is not None else None

    def moment_with(self, track_id, kind, when="phrase"):
        """TAKEOVER / BRIDGE with THIS song: on a deck and ready = armed now with it as the other song; not on
        a deck = staged first and armed the moment it is ready (layered only). `when`: "phrase" (its payoff
        within the coming phrase, else the phrase) or "bar" (the next bar). Returns a sentence."""
        if kind not in ("takeover", "bridge"):
            return "only TAKEOVER and BRIDGE take a song"
        if self.rc is None or self.rc.master is None:
            return "layer first: LAYERS two (the autoDJ owns its decks in one-song mode)"
        rc = self.rc
        tid = int(track_id)
        self.queue(tid)                               # the song of a TAKEOVER / BRIDGE is UP NEXT too
        deck = next((d for d, s in rc.songs.items() if s.track.id == tid and not s.leaving), None)
        if deck is not None and rc.songs[deck].staged_at is not None:
            ok, msg = rc.arm_moment(kind, other=deck, when=when)
            self._event((f"{kind.upper()} with {self.title(tid)[:24]}: " if ok else f"{kind.upper()} refused: ") + msg)
            return msg
        if deck is None:
            ok, msg = rc.stage_track(tid)
            if not ok:
                self._event(f"{kind.upper()} refused: {msg}")
                return msg
        self._pending_moment = {"track_id": tid, "kind": kind, "t": time.time(), "when": when}
        self._event(f"{kind.upper()} with {self.title(tid)[:24]}: staging, armed as soon as it is ready")
        return f"{self.title(tid)[:24]} is being staged; {kind.upper()} arms the moment it is ready"

    def _pending_moment_step(self):
        p = getattr(self, "_pending_moment", None)
        if p is None or self.rc is None:
            return
        if time.time() - p["t"] > 90.0:
            self._pending_moment = None
            self._event("the staged song never became ready: the moment is dropped")
            return
        deck = next((d for d, s in self.rc.songs.items() if s.track.id == p["track_id"] and s.staged_at is not None and not s.leaving), None)
        if deck is None:
            return
        self._pending_moment = None
        ok, msg = self.rc.arm_moment(p["kind"], other=deck, when=p.get("when", "phrase"))
        self._event((f"{p['kind'].upper()} with {self.title(p['track_id'])[:24]}: " if ok else f"{p['kind'].upper()} refused: ") + msg)

    # -- the dropout detector: silence on the bus is named, with what was happening ------------------------------------
    def _dropout_step(self):
        """The playing engine's bus under -48 dBFS for more than a bar (a planned hole is two beats at most)
        is a DROPOUT: logged once with the mode, the handover state, the armed moment and the last moves, and
        shown in the state line; its end is logged with the length. The next silence names itself."""
        eng = self.rc if self.rc is not None else self.system
        if eng is None or not self.running:
            self._quiet_since = None
            return
        tel = eng.submix.telemetry or {}
        rms = float(tel.get("rms") or 0.0)
        now = time.time()
        bar_s = 240.0 / max(self._bpm(), 60.0)
        # silence only counts once the room has sounded: the two seconds between START and the first song's
        # audio raised a DROPOUT two seconds into a set (2026-09-13)
        if rms >= 0.004:
            self._heard = True
        elif not getattr(self, "_heard", False):
            self._quiet_since = None
            return
        if rms >= 0.004:
            q = getattr(self, "_quiet_since", None)
            if q is not None and getattr(self, "_dropout_open", False):
                self._dropout_open = False
                length = now - q
                self.last_dropout = {**(self.last_dropout or {}), "end_t": now, "length_s": round(length, 1)}
                self._event(f"dropout over: {length:.1f} s of silence")
                from lib.dj.session import log_row
                log_row({"event": "dropout_end", "length_s": round(length, 1)})
            self._quiet_since = None
            return
        if getattr(self, "_quiet_since", None) is None:
            self._quiet_since = now
            return
        if now - self._quiet_since > bar_s and not getattr(self, "_dropout_open", False):
            self._dropout_open = True
            mo = self.moment_status()
            ctx = {"t": now, "mode": self.mode, "switching": (self._switch or {}).get("to") if self._switch else None,
                   "moment": (f"{mo['label']} {'running' if mo['started'] else 'armed'} ({mo['who']}), the one in {mo['in_bars']} bars" if mo else None),
                   "events": [e[1] for e in self.events[-3:]],
                   "moves": ([m for _, m in self.rc.moves[-3:]] if self.rc is not None else [])}
            self.last_dropout = ctx
            self._event(f"DROPOUT: the bus has been silent for {now - self._quiet_since:.1f} s" + (f" during {ctx['moment']}" if ctx["moment"] else "")
                        + (f" while switching to {ctx['switching']}" if ctx["switching"] else ""))
            from lib.dj.session import log_row
            log_row({"event": "dropout", **{k: v for k, v in ctx.items() if k != "t"}})

    # -- tactical stem moves: THIS song's bassline (drums, vocals, melody) in, now, loud -------------------------
    def stem_in(self, track_id, lane, loud=True, when="bar"):
        """Bring one song's `lane` under what plays - on the next bar (or the next phrase), loud for a
        phrase. A song not yet on a deck is staged first and comes in the moment it is ready. Layered
        mode only (one song: the autoDJ owns its decks). Returns a sentence."""
        if self.rc is None or self.rc.master is None:
            return "layer first: LAYERS two (the autoDJ owns its decks in one-song mode)"
        rc = self.rc
        # a PART of a song is a part: it does not make the song UP NEXT or the next record (it did, and the plan's
        # next song changed five times in three minutes - user 2026-09-17); UP NEXT is a separate order
        deck = next((d for d, s in rc.songs.items() if s.track.id == int(track_id) and not s.leaving), None)
        boost = 1.3 if loud else 1.0
        if deck is None:
            ok, msg = rc.stage_track(int(track_id), as_next=False)
            if not ok:
                self._event(f"stem in refused: {msg}")
                return msg
            self._pending_stem = {"track_id": int(track_id), "lane": lane, "boost": boost, "when": when, "t": time.time()}
            self._event(f"{lane} of {self.title(int(track_id))[:24]}: staging, in as soon as it is ready")
            return f"{self.title(int(track_id))[:24]} is being staged; its {lane} comes in the moment it is ready"
        ok, msg = rc.bring(lane, deck, boost=boost, when=when)
        if not ok and msg.startswith("still staging"):
            self._pending_stem = {"track_id": int(track_id), "lane": lane, "boost": boost, "when": when, "t": time.time()}
            return f"{self.title(int(track_id))[:24]}: its {lane} comes in on the next bar"
        self._event(("moment: " if ok else "refused: ") + msg)
        return msg

    def eject(self, track_id):
        """Take a song off the decks (the PLAYING list, user 2026-09-17): its parts go back to the record and it
        leaves a bar later; a staged song just leaves (the seam or the event that counted on it is re-planned).
        The record itself is not ejected - that is NEXT (the seam now) or DROP. Returns a sentence."""
        if self.rc is None or self.rc.master is None:
            return "not layered"
        rc = self.rc
        tid = int(track_id)
        self.unqueue(tid)
        deck = rc.deck_of(tid)
        if deck is None:
            if any(t.id == tid for t in rc._decoding.values()) or any(p[0].id == tid for p in rc._pending.values()):
                rc.avoid_ids = set(rc.avoid_ids or ()) | {tid}
                self._event(f"{self.title(tid)[:24]} is still decoding: it will be dropped when it lands")
                self._eject_when_staged = tid
                return "still decoding: dropped when it lands"
            self._event(f"eject refused: {self.title(tid)[:24]} is not on a deck")
            return "not on a deck"
        if deck == rc.record_deck:
            self._event(f"eject: {self.title(tid)[:24]} is the record playing - NEXT (the seam now) or DROP takes it off")
            return "the record playing: NEXT or DROP"
        ok = rc.eject(deck)
        self._event((f"{self.title(tid)[:24]} off the decks" if ok else f"eject refused: {self.title(tid)[:24]} is already leaving"))
        return "off the decks" if ok else "already leaving"

    def stem_out(self, lane):
        if self.rc is None or self.rc.master is None:
            return "not layered"
        ok, msg = self.rc.release(lane)
        self._event(("moment: " if ok else "refused: ") + msg)
        return msg

    def _pending_stem_step(self):
        p = getattr(self, "_pending_stem", None)
        if p is None or self.rc is None:
            return
        if time.time() - p["t"] > 90.0:
            self._pending_stem = None
            self._event("the staged song never became ready: the stem move is dropped")
            return
        deck = next((d for d, s in self.rc.songs.items() if s.track.id == p["track_id"] and s.staged_at is not None and not s.leaving), None)
        if deck is None:
            return
        ok, msg = self.rc.bring(p["lane"], deck, boost=p["boost"], when=p["when"])
        if not ok and msg.startswith("still staging"):
            return                                    # its staging bar has not passed: again on the next tick
        self._pending_stem = None
        self._event(("moment: " if ok else "refused: ") + msg)

    def ready(self):
        """The songs you can move stems from right now (layered): every live or staged song that is not the
        bed, with what it holds, what each lane would say, and its loud state."""
        out = []
        rc = self.rc
        if rc is None or rc.master is None:
            return out
        bed = rc.lanes.get("drums")
        at = rc._next_bar_clock()
        for d, s in rc.songs.items():
            if s.leaving:
                continue
            held = s.held(rc.lanes)
            lanes = {}
            for ln in ("drums", "bass", "other", "vocals"):
                if rc.lanes.get(ln) == d:
                    lanes[ln] = {"held": True, "why": None, "loud": bool(rc.lane_boost.get(ln) and rc.bar_n < rc.lane_boost[ln][1])}
                else:
                    lanes[ln] = {"held": False, "why": ("still decoding" if s.staged_at is None else rc.why_not(ln, d)), "loud": False}
            out.append({"id": s.track.id, "title": s.track.title, "deck": d, "bed": d == bed, "staged": s.staged_at is not None,
                        "entered": s.entered, "held": held, "lanes": lanes,
                        "hook_in_bars": rc._hook_in_bars(d, at, 64) if s.staged_at is not None else None,
                        "drop_in_bars": rc._drop_in_bars(d, at, 64) if s.staged_at is not None else None})
        out.sort(key=lambda r: (r["bed"], not r["entered"], r["title"]))
        return out

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

    def rate_target(self):
        """What GOOD / BAD would land on right now: {"text", "kind", "moment": kind or None} - the tab asks
        before a BAD, so a BAD on a moment can carry the ear's word."""
        if self.rc is not None:
            from lib.dj.remix import RATED_KINDS, MOMENTS
            m = next((x for x in reversed(self.rc.move_log) if x["kind"] in RATED_KINDS and x.get("fb") is None), None)
            if m is None:
                return None
            moment = None
            if m["kind"] == "moment":
                label = m["text"].split(":")[0].strip()
                moment = next((k for k, v in MOMENTS.items() if v[0] == label), None)
            return {"text": m["text"], "kind": m["kind"], "moment": moment}
        if self.system is not None:
            h = next((x for x in reversed(self.system._history) if x.get("via") != "start"), None)
            return {"text": f"the mix into {h.get('title')}", "kind": "seam", "moment": None} if h else None
        return None

    def rate(self, up, word=None):
        """GOOD / BAD: say WHAT was rated, what the DJ learns from it, and - on a BAD - change the music
        now (a fresh move layered; the next seam brought forward in one-song mode), so the button is
        visibly consequential. `word` (a BAD on a moment): one of the ear's five words - hole / thin / long /
        beat / place - and ONE number of that moment moves (see RemixConductor.tune_moment)."""
        up = bool(up)
        info = {"up": up, "t": time.time(), "what": None, "learn": None, "did": None, "style": None}
        song_id = None
        tuned = None
        if not up and word and self.rc is not None:
            tgt = self.rate_target()
            if tgt and tgt.get("moment"):
                tuned = self.rc.tune_moment(tgt["moment"], word)
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
            # a verdict is about the last move only while that move is fresh (two phrases); later, or with
            # nothing to rate, a BAD means THIS STRETCH: it pushes the music on and blames no move (five BADs
            # in a dead stretch once condemned the voice arrival, the bed change and a two-minute-old vocal rest)
            phrase_s = self.rc.change_bars * 240.0 / max(self._bpm(), 60.0)
            m = self.rc.rate_last(up, max_age_s=2 * phrase_s)
            if m is not None:
                w = self.rc._w(m["kind"], m.get("lane"))
                info["what"] = m["text"]
                info["learn"] = f"moves of that kind ({m['kind'].replace('_', ' ')}{', ' + m['lane'] if m.get('lane') else ''}) now weigh {w:.2f} (0.50 = neutral): {'more' if up else 'less'} of them"
                song_id = m.get("to_id") or m.get("from_id")
            else:
                info["what"] = "this stretch (the last move is older than two phrases, so no move is blamed)" if not up else "nothing recent to rate"
            if tuned:
                info["learn"] = tuned                 # the word moved a number: that is the lesson, not the weight
                info["did"] = "the next one of that moment plays with the new number"
            elif not up and self.rc.master is not None:
                self.rc._verdict_forced_until = time.time() + phrase_s   # the move this BAD makes is heard before it can be judged
                self.rc.next_move()
                info["did"] = "a different move on the next bar" if m is not None else "this stretch is dead: a move now (unrateable until it has been heard)"
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

    # -- FLAGS: "this is not working", kept for later with everything needed to reproduce it -------------------------
    def flag(self, kind, text=""):
        """Raise a flag: what the operator saw (a FLAG_KINDS key, a free note) with a SNAPSHOT of the room -
        mode, dials, the plan, the armed moment, every song on a deck with its position, lanes and gains, the
        bus level, the last moves and events - written to logs/dj_flags.jsonl (and the day's log), marked in
        the story row. Nothing is learned from it and nothing moves: it is a message to the fix, not a
        verdict on a mix or a song. Returns the sentence shown."""
        kind = kind if kind in FLAG_KINDS else "other"
        good = kind in FLAG_GOOD
        snap = {"event": "flag", "kind": kind, "good": good, "label": FLAG_KINDS[kind], "text": (text or "")[:400],
                "mode": self.mode, "theme": self.theme, "pool": self.pool_name, "dials": dict(self.dials),
                "bar": round(self._bar, 1), "events": [f"{h} {m}" for h, m in self.events[-6:]],
                "last_verdict": ({k: v for k, v in self.last_verdict.items() if k in ("up", "what", "learn", "did")} if self.last_verdict else None),
                "last_dropout": self.last_dropout}
        try:
            snap["arc"] = {k: v for k, v in self.arc_status().items() if k in ("word", "target", "elapsed_s", "length_s", "shape")}
        except Exception:
            pass
        try:
            if self.rc is not None and self.rc.master is not None:
                rc = self.rc
                tel = rc.submix.telemetry or {}
                decks = tel.get("decks") or {}
                songs = {}
                for d, s in rc.songs.items():
                    t = decks.get(d) or {}
                    songs[d] = {"id": s.track.id, "title": s.track.title, "artist": s.track.artist, "bpm": s.track.bpm, "camelot": s.track.camelot,
                                "shift": s.shift, "rate": round(float(s.rate), 4), "time_s": t.get("time_s"), "playing": bool(t.get("playing")),
                                "lanes": s.held(rc.lanes), "gains": {k: round(float(v), 2) for k, v in (t.get("stem_gains") or {}).items()},
                                "gain": t.get("gain"), "eq_low": (t.get("eq") or [None])[0], "loop": t.get("loop"),
                                "entered": s.entered, "leaving": s.leaving, "was_bed": s.was_bed, "voice_since": s.voice_since, "bed_since": s.bed_since,
                                "section": (s.track.section_at(float(t.get("time_s") or 0.0)) or {}).get("kind")}
                arr = rc._arrangement_status()
                snap.update({"engine": "conductor", "clock_s": tel.get("clock_s"), "rms": tel.get("rms"), "peak": tel.get("peak"),
                             "master": rc.master, "master_bpm": rc.master_bpm, "key_centre": rc.key_centre, "bar_n": rc.bar_n,
                             "phrase_bars": rc.phrase_bars, "change_bars": rc.change_bars, "calm_phrases": rc.calm_phrases,
                             "lanes": {ln: (rc.songs[d].track.title if d in rc.songs else None) for ln, d in rc.lanes.items()},
                             "lane_boost": {k: list(v) for k, v in rc.lane_boost.items()}, "songs": songs,
                             "plan": arr.get("plan"), "wait_why": arr.get("wait_why"), "moves_10min": arr.get("moves_10min"),
                             "recue": arr.get("recue"), "follow": arr.get("follow"),
                             "moment": rc.moment_status(), "moment_tune": rc.moment_tune, "breaking": rc._break is not None,
                             "user_loop": rc.user_loop_bars, "hold": rc.hold,
                             "moves": [{"hms": m.get("hms"), "kind": m["kind"], "lane": m.get("lane"), "text": m["text"][:140]} for m in rc.move_log[-10:]]})
            elif self.system is not None:
                sysm = self.system
                st = sysm.status()
                tel = sysm.submix.telemetry or {}
                cur = st.get("current") or {}
                snap.update({"engine": "autodj", "clock_s": tel.get("clock_s"), "rms": tel.get("rms"), "peak": tel.get("peak"),
                             "state": st.get("state"), "current": cur, "next": st.get("next"), "plan": st.get("plan"),
                             "blend_in_s": st.get("blend_in_s"), "played_s": self._played_s(),
                             "history": [{k: v for k, v in h.items() if k in ("title", "via", "t", "beats")} for h in sysm._history[-4:]],
                             "recent": (st.get("recent_events") or [])[-8:]})
        except Exception as e:  # noqa: BLE001
            snap["snapshot_error"] = f"{type(e).__name__}: {e}"
        try:
            from lib.dj.session import log_row, FLAGS_FILE
            log_row(snap, name=FLAGS_FILE)
            log_row({"event": "flag", "kind": kind, "text": snap["text"]})
        except Exception:
            pass
        self.flags.append({"t": time.time(), "kind": kind, "good": good, "label": FLAG_KINDS[kind], "text": snap["text"]})
        try:
            self.marks.append({"bar": int(self._bar), "end_bar": None, "lanes": None, "ghost": False, "kind": "flag",
                               "text": ("✓ " if good else "⚑ ") + FLAG_KINDS[kind] + (f": {text[:30]}" if text else "")})
        except Exception:
            pass
        n = sum(1 for f in self.flags if f["kind"] == kind)
        msg = ("kept: " if good else "flagged: ") + FLAG_KINDS[kind] + (f" - {text[:60]}" if text else "") + f" ({n} tonight; nothing learned, nothing moved)"
        self._event(msg)
        return msg

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
            self._pending_stem_step()
            self._pending_moment_step()
            ew = getattr(self, "_eject_when_staged", None)
            if ew is not None:
                dk = self.rc.deck_of(ew)
                if dk is not None:
                    self._eject_when_staged = None
                    if self.rc.songs[dk].staged_at is not None:
                        self.rc.eject(dk)
                        self._event(f"{self.title(ew)[:24]} off the decks (you took it off while it decoded)")
        try:
            self._dropout_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"dropout: {type(e).__name__}: {e}"
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
        try:
            self._horizon_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"set: {type(e).__name__}: {e}"
        try:
            self._slip_step()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"slip: {type(e).__name__}: {e}"
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
            if self.rc is not None:
                self.rc.reserve_pick = False
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
            if d is not None:
                self._queued_on_deck.add(head)
            # a queued song leaves the list when it has BECOME THE BED (or left the decks after being on one) -
            # not when a first part of it sounds: a song whose stem you placed stays listed while its part plays
            if (d is not None and (self.rc.lanes.get("drums") == d or self.rc.songs[d].leaving)) or (d is None and head in self._queued_on_deck):
                self.up_next.pop(0)
                self._queued_on_deck.discard(head)
                self.rc.reserve_pick = False
                return
            if d is None:
                # while your head is not on a deck the conductor stages no pick of its own (a free deck is yours)
                self.rc.reserve_pick = True
                ok, msg = self.rc.stage_track(head)
                if ok:
                    self._event(f"up next {self.title(head)[:30]}: {msg}")
                    self._fed_wait = None
                elif msg not in ("no free deck - eject one first", "already on a deck"):
                    rc, t = self.rc, self.track(head)
                    base = float(getattr(rc, "base_bpm", 0) or 0)
                    # THE CLOCK FOLLOWS THE RECORD: a song outside the wall against today's clock but inside it
                    # where the clock is heading (a route's next record) WAITS; the clock travels half a percent a bar
                    travelling = ("outside the tempo wall" in msg and getattr(rc, "follow_clock", True) and base and t is not None and t.bpm
                                  and 0.90 <= base / float(t.bpm) <= 1.10 and abs(float(rc.master_bpm) / base - 1.0) > 0.005)
                    if travelling:
                        if getattr(self, "_fed_wait", None) != head:
                            self._fed_wait = head
                            self._event(f"up next {self.title(head)[:24]} waits for the clock ({float(rc.master_bpm):.0f} → {base:.0f} bpm)")
                    else:
                        self._event(f"up next {self.title(head)}: {msg}")
                        self.up_next.pop(0)
                        self.via_ids.discard(head)
                        self.rc.reserve_pick = False
            else:
                self.rc.reserve_pick = False

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
        # +4 dB at most; a planned quiet (a moment's take-away, a break) is not a reason to turn up
        if getattr(self.rc, "_moment", None) is not None or getattr(self.rc, "_break", None) is not None:
            return
        want = max(1.0, min(1.6, ref / max(self._lay_rms, 1e-6)))
        # up slowly, DOWN AT ONCE: a quiet stretch had walked the bus to x2 and the next full arrival came in
        # 6 dB hot (flag 'seam' at -9.6 dBFS, 2026-09-13)
        trim = cur_trim + 0.25 * (want - cur_trim) if want > cur_trim else want
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
            rc = self.rc
            rc.auto_break_scale = rate / 0.5 if spaced else 0.0
            # the DJ's own moments are the same five as yours, armed the same way (you see the countdown and
            # can cancel): at a phrase start, spaced and hot, one whose conditions hold - a two-song moment
            # when a voice has been heard two phrases, SING when the bed sings toward a hook, else BREAK, else SLAM
            if rc.phrase_bars == 0 and getattr(self, "_moment_k", None) != rc.bar_n and rc._moment is None:
                self._moment_k = rc.bar_n
                if spaced and hot and rc.rng.random() < rate * 0.5:
                    bed = rc.lanes.get("drums")
                    voices = [s for d, s in rc.songs.items() if s.entered and not s.leaving and d != bed
                              and s.voice_since is not None and rc.bar_n - s.voice_since >= 2 * rc.change_bars]
                    order = (["bridge", "takeover"] if voices else []) + ["sing", "break", "slam"]
                    for kind in order:
                        ok, msg = rc.arm_moment(kind, auto=True)
                        if ok:
                            self._last_moment_t = time.time()
                            self._event(f"the DJ: {msg}")
                            break

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

    def _anchor(self, key, bar_abs):
        """A plan mark's absolute picture bar, decided ONCE per plan (a seam's seq and bar, a moment's one).
        The Director's wall-clock bar and the conductor's bar count drift apart by a bar now and then (the
        PLL trims the tempo, the two tick out of phase); a mark placed at b + in_bars on every tick crept
        along with the drift - "potential new songs continuously pushed back" (user, 2026-09-17). It moves
        only when the plan itself moves (a new key)."""
        anchors = getattr(self, "_plan_anchors", None)
        if anchors is None:
            anchors = self._plan_anchors = {}
        a = anchors.get(key)
        if a is None:
            if len(anchors) > 60:
                anchors.clear()
            anchors[key] = a = int(bar_abs)
        return a

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
            # the clip in charge of a lane is the song SOUNDING loudest on it (telemetry), not the plan: a bed
            # change is drawn when the new drums and bass land, not when it was decided (user: "parts of a
            # song were playing longer than they were displayed")
            snd = getattr(self, "_last_sounding", None) or self._sounding()
            by_tid = {s.track.id: d for d, s in rc.songs.items()}
            for ln in LANES:
                per = snd.get(ln) or {}
                tid = max(per, key=per.get) if per else None
                if tid is not None and per[tid] < 0.15:
                    tid = None
                d = by_tid.get(tid)
                if self._lane_snap.get(ln, "unset") != tid:
                    self._lane_snap[ln] = tid
                    if tid is None:
                        from lib.dj.timeline import Clip
                        prev = self.tl.active(ln, b - 1)
                        self.tl.clips.append(Clip(prev.track_id if prev else 0, ln, b, 0.0, end_bar=None, rest=True))
                    else:
                        pos = float(rc._tel_deck(d).get("time_s") or 0.0) if d else 0.0
                        # ONE ANCHOR PER SONG: a lane heard on a song later takes its (bar, song time) line from the
                        # song's first clip, not from a fresh telemetry read at this tick - four lanes stamped at
                        # four ticks drew the same song's parts up to a bar apart (user 2026-09-19: "why do
                        # different stems of the same song not line up?"). Telemetry wins only when the song has
                        # moved (a cue, a jump, a loop): the derived place is then more than two bars off it.
                        ref = next((c for c in reversed(self.tl.clips) if c.track_id == tid and not c.ghost and not c.rest and b - c.bar < 400), None)
                        if ref is not None:
                            derived = ref.start_s + (b - ref.bar) * self._bar_len(tid)
                            if abs(derived - pos) <= 2.0 * self._bar_len(tid):
                                pos = derived
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
                elif k in ("loop", "evict", "recall", "save", "seam_plan", "seam", "event_plan", "event"):
                    mark(m["text"][:44], k)
            # the plan ahead: the staged song arrives as a VOICE (the arrangement's `other` lane) at the move
            # the bed's settling allows - one steady ghost, not a lane-hopping one; the story row says what
            # each live song is waiting for
            left = max(0, rc.change_bars - 1 - rc.phrase_bars)
            # an armed moment: its choreography, dashed and gold, from the one backwards
            mo = rc.moment_status()
            if mo:
                L = self._anchor(("moment", mo["label"], rc.bar_n + int(mo["in_bars"])), b + mo["in_bars"])
                for mk in mo.get("marks") or []:
                    self.plan_marks.append({"bar": int(L + mk["off"]), "end_bar": (int(L + mk["end_off"]) if mk.get("end_off") is not None else None),
                                            "lanes": mk["lanes"], "ghost": True, "kind": "moment", "text": mk["text"]})
            plan_in = None
            seam_S = None
            arr = {}
            try:
                arr = rc._arrangement_status()
                if arr.get("landing_in_bars"):
                    self.plan_marks.append({"bar": b, "end_bar": b + int(arr["landing_in_bars"]), "lanes": ["drums", "bass"], "ghost": True,
                                            "text": "the new bed lands", "kind": "bed_to"})
                # THE SEAM (the record policy): its steps as a gold bracket from its first bar
                sm = arr.get("seam")
                seam_S = None
                if sm:
                    S = seam_S = self._anchor(("seam", sm.get("seq"), int(sm["bar"])), b + int(sm["in_bars"]))
                    for mk in sm.get("marks") or []:
                        self.plan_marks.append({"bar": int(S + mk["off"]), "end_bar": (int(S + mk["end_off"]) if mk.get("end_off") is not None else None),
                                                "lanes": mk["lanes"], "ghost": True, "kind": "seam", "text": mk["text"]})
                    plan_in = int(sm["in_bars"])
                # ONE mark for THE PLAN (decided once, so it stays put): the next event at its bar
                pl = arr.get("plan")
                if pl and not arr.get("landing_in_bars") and not sm:
                    lanes_of = {"voice_in": ["other"], "bed_to": ["drums", "bass"], "voice_out": ["other", "vocals"], "land": ["drums"]}
                    self.plan_marks.append({"bar": b + int(pl["in_bars"]), "end_bar": None, "lanes": lanes_of.get(pl["what"], list(LANES)), "ghost": True,
                                            "text": f"{pl['title'][:18]} " + {"voice_in": "arrives as a voice", "bed_to": f"takes the bed ({pl.get('payoff') or 'at the phrase'})",
                                                                              "voice_out": "fades out", "land": "drums follow: the bed is its"}.get(pl["what"], pl["what"]),
                                            "kind": pl["what"]})
                    if pl["what"] == "voice_in":
                        plan_in = int(pl["in_bars"])
            except Exception:
                pass
            if plan_in is not None:
                left = max(1, plan_in)
            staged = [(d, s) for d, s in rc.songs.items() if s.staged_at is not None and not s.entered and not s.leaving]
            evp = ((arr.get("record") or {}).get("event") if isinstance(arr, dict) else None) or None
            d_e = rc.deck_of(evp["track_id"]) if (evp and evp.get("kind") == "takeover" and evp.get("track_id") is not None) else None
            if d_e is not None and d_e in rc.songs and not rc.songs[d_e].entered:
                # THE EVENT IS THE EXIT: the takeover's song is the next record, and the seam into the staged one
                # was dropped when the event was planned. Its ghost sits at the event's bar, anchored - the staged
                # song's ghost, placed at "now plus a phrase" every bar, crept along for eighty bars ("pushing back
                # the start of songs note by note", user 2026-09-19)
                s_e = rc.songs[d_e]
                E = self._anchor(("event", int(evp["track_id"]), int(evp["bar"])), b + int(evp["in_bars"]))
                ghost(("event", s_e.track.id, E), s_e.track.id, ["bass", "other"], E,
                      rc._song_time_at(d_e, rc._next_bar_clock()) + max(1, E - b - 1) * self._bar_len(s_e.track.id))
            elif staged:
                d, s = staged[0]
                if seam_S is not None:
                    at = seam_S                                   # the ghost sits where the seam's bracket starts, and stays there
                else:
                    # no seam yet: the ghost waits at a phrase boundary ahead and is re-placed a PHRASE at a time
                    # when the room passes it - never bar by bar
                    want = b + 1 + left
                    at = self._anchor(("staged", s.track.id, int(want // max(1, rc.change_bars))), want)
                    if at <= b:
                        at = self._anchor(("staged", s.track.id, int((b + 1 + left) // max(1, rc.change_bars)) + 1), b + 1 + left)
                sm = (arr.get("seam") if isinstance(arr, dict) else None) or {}
                first_lanes = {"drums": ["drums"], "bass": ["bass", "other"], "vocal": ["vocals"], "cut": ["drums", "bass", "other"]}.get(sm.get("order"), ["drums"])
                ghost(("layered", s.track.id, at), s.track.id, first_lanes, at,
                      rc._song_time_at(d, rc._next_bar_clock()) + max(1, at - b - 1) * self._bar_len(s.track.id))
            else:
                no_ghost()

    def _sounding(self):
        """Per lane, what is SOUNDING now: {lane: {track_id: level}} from the deck telemetry (stem gain × deck
        gain, decks that play) - the truth the picture draws from, not the plan."""
        out = {ln: {} for ln in ("drums", "bass", "other", "vocals")}
        if self.rc is not None:
            decks = (self.rc.submix.telemetry or {}).get("decks") or {}
            by_deck = {d: s.track.id for d, s in self.rc.songs.items()}
            for d, t in decks.items():
                if not t.get("playing"):
                    continue
                tid = by_deck.get(d) or t.get("track_id")
                if tid is None:
                    continue
                sg = t.get("stem_gains") or {}
                dg = float(t.get("gain") or 0.0)
                for ln in out:
                    g = min(1.0, float(sg.get(ln, 0.0)) * dg)
                    if g > 0.02:
                        out[ln][tid] = max(out[ln].get(tid, 0.0), g)
        elif self.system is not None:
            decks = (self.system.submix.telemetry or {}).get("decks") or {}
            for d, t in decks.items():
                if not t.get("playing"):
                    continue
                tid = t.get("track_id")
                if tid is None:
                    continue
                sg = t.get("stem_gains") or {}
                dg = float(t.get("gain") or 0.0)
                for ln in out:
                    g = min(1.0, float(sg.get(ln, 1.0)) * dg)
                    if g > 0.02:
                        out[ln][tid] = max(out[ln].get(tid, 0.0), g)
        return out

    def _record_gains(self, b):
        """Per lane and bar, every song sounding on that stem and its level (the max within the bar) - the
        picture's meters stack them, so two songs on one stem show as two colours (user: "I can't tell if
        multiple parts of different songs are going in the same stem")."""
        try:
            snd = self._sounding()
            self._last_sounding = snd
            for ln, per in snd.items():
                cell = self._gains[ln].setdefault(b, {})
                if not isinstance(cell, dict):
                    cell = self._gains[ln][b] = {}
                for tid, g in per.items():
                    cell[tid] = max(cell.get(tid, 0.0), g)
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
        try:
            via_map = self.via_by_tempo() if layered else {}
        except Exception:
            via_map = {}
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
                reach_via = 0
                if not reach:
                    # THE SET rule 4: a song the clock can reach through other records is "via N", not grey
                    k = via_map.get(float(t.bpm)) if (layered and t.bpm) else None
                    if k and k > 1:
                        reach_via = k - 1
                        why[-1] += f" (via {reach_via} record{'s' if reach_via > 1 else ''})"
                    else:
                        ok = False
                        why[-1] += " (out of reach: would fade)" if not layered else " (outside the wall)"
                kc = camelot_compat(cam, t.camelot) if (cam and t.camelot) else None
                if kc is not None:
                    why.append(f"key {t.camelot} fit {kc:.2f}" + (" (clash)" if kc < 0.55 else ""))
                    if kc < 0.55 and not layered:
                        why[-1] += " - a fade or a key shift"
                why.append(f"energy {e:.2f} vs {arc:.2f}")
                fit = (0.45 * (kc if kc is not None else 0.6) + 0.35 * max(0.0, 1.0 - abs(math.log(max(r, 1e-6))) / 0.1)
                       + 0.2 * max(0.0, 1.0 - abs(e - arc) / 0.5)) if reach else (0.2 / reach_via if reach_via else 0.0)
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

    def _plan_rows(self):
        """THE PLAN as an ordered list (user 2026-09-17: "the auto system just doesn't display its plan"): one row
        per thing to come, in time order, each with the seconds from now and the bar - NOW (the record, how far
        in, its minimum play, its next section), a MOMENT armed or running, the EVENT pending with its song's
        state, the SEAM (the next record, how it comes in, why there, whether its drop lands, the rehearsal),
        the RECORD after it and its minimum play, UP NEXT after that. When there is nothing, the row says why.
        A row that just moved carries the reason for twenty seconds. Rows: {t (s from now or None), bar, kind,
        text, sub, key}."""
        rows = []
        rc, sysm = self.rc, self.system
        now_t = time.time()

        def row(t, kind, text, sub="", key=None, bar=None, tid=None):
            rows.append({"t": (None if t is None else max(0.0, float(t))), "bar": bar, "kind": kind, "text": text, "sub": sub, "key": key or (kind, text[:40]), "id": tid})
        # THE FLOOR's requests first: a song being made playable, with its stage; ready for a minute after
        for k in list(self.requests.keys()):
            r = self.requests[k]
            if (r.get("ready") or r.get("error")) and now_t - float(r.get("t") or 0) > 90.0:
                del self.requests[k]
                continue
            if r.get("error"):
                row(None, "REQUEST", f"{r['title'][:30]}: failed", str(r["error"])[:80], key=("req", k, "err"))
            elif r.get("ready"):
                row(None, "REQUEST", f"{r['title'][:30]} is ready · put it UP NEXT?", "right-click", key=("req", k, "ready"), tid=r.get("id"))
            else:
                row(None, "REQUEST", f"{r['title'][:30]}: {r.get('stage', '')}" + (f" ({float(r['pct']):.0f} %)" if r.get("pct") is not None else ""),
                    "one job at a time: analysis, stems, pairings", key=("req", k, r.get("stage")), tid=r.get("id"))
        if rc is not None and rc.master is not None:
            bar_s = rc._bar_clock() / RATE
            arr = rc._arrangement_status()
            rec, sm = arr.get("record"), arr.get("seam")
            rd = rc.record_deck
            s_rec = rc.songs.get(rd)
            if rec is not None and s_rec is not None:
                played = (rec.get("since_bars") or 0) * bar_s
                left = (rec.get("settle_left") or 0) * bar_s
                bits = [f"{_mmss(played)} played", (f"min play in {_mmss(left)}" if left > 0 else "past its minimum play")]
                try:
                    t_now = float(rc._tel_deck(rd).get("time_s") or 0.0)
                    secs = s_rec.track.sections or []
                    nxt_sec = None
                    for i, sec in enumerate(secs):
                        if float(sec["start_s"]) > t_now + 1.0:
                            k = sec.get("kind")
                            if k in ("build", "breakdown", "outro") or (k == "groove" and i > 0 and secs[i - 1].get("kind") in ("build", "breakdown", "intro")):
                                nxt_sec = ("drop" if k == "groove" else k, float(sec["start_s"]) - t_now)
                                break
                    if nxt_sec:
                        bits.append(f"its {nxt_sec[0]} in {_mmss(nxt_sec[1] / max(float(s_rec.rate or 1.0), 1e-6))}")
                    bits.append(f"ends in {_mmss((float(s_rec.track.duration_s) - t_now) / max(float(s_rec.rate or 1.0), 1e-6))}")
                except Exception:
                    pass
                row(0.0, "NOW", f"{s_rec.track.title[:34]}", " · ".join(bits), key=("now", s_rec.track.id))
            mo = rc.moment_status()
            if mo:
                t_m = max(0, mo["in_bars"]) * bar_s
                row(t_m, "MOMENT", f"{mo['label']} {'running' if mo['started'] else 'armed'} ({mo['who']})",
                    f"the one on {mo['what']}" + (f" of {mo['other'][:22]}" if mo.get("other") else ""), key=("moment", mo["label"], rc.bar_n + int(mo["in_bars"])))
            evp = (rec or {}).get("event")
            if evp:
                d_ev = rc.deck_of(evp["track_id"]) if evp.get("track_id") is not None else None
                state = ("staged, arms two phrases before" if (d_ev is not None and rc.songs[d_ev].staged_at is not None) else "its song is decoding")
                if evp.get("accepted"):
                    row(max(0, evp["in_bars"]) * bar_s, "EVENT", evp["why"][:90], state, key=("event", evp.get("track_id"), evp.get("bar")), bar=evp.get("bar"), tid=evp.get("track_id"))
                else:
                    # THE DJ'S IDEA (docs/DJ_MORE_THAN_A_MIXER.md 2): proposed, not taken - right-click do it / skip;
                    # it passes two phrases before its bar if nobody says
                    passes = max(0, evp["in_bars"] - 2 * rc.change_bars) * bar_s
                    row(max(0, evp["in_bars"]) * bar_s, "IDEA", evp["why"][:90],
                        f"right-click: do it · skip · passes in {_mmss(passes)} if not taken · {state}", key=("idea", evp.get("track_id"), evp.get("bar")), bar=evp.get("bar"), tid=evp.get("track_id"))
            if sm:
                sb = rc.songs.get(rc.seam["b"]) if rc.seam is not None else None
                title_b = sb.track.title[:30] if sb else sm["b"][:30]
                t_seam = max(0, sm["in_bars"]) * bar_s
                where = sm["why"]
                land = ("its drop lands on the swap" if sm.get("on_landmark") else (f"its drop {sm['drop_after_bars']} bars after the seam" if (sm.get("drop_after_bars") or 0) > 0 else "its drop falls inside the seam"))
                reh = sm.get("rehearsed")
                reh_w = ("rehearsed: " + str(reh)[:60]) if reh else (f"rehearsal {_mmss(max(0.0, t_seam - 2 * rc.change_bars * bar_s))} before" if t_seam > 2 * rc.change_bars * bar_s else "rehearsal pending")
                sub = f"{where} · {land} · {reh_w}"
                # a seam that just moved says why, for twenty seconds
                last_pm = next((m for m in reversed(rc.move_log) if m["kind"] == "plan" and "moved" in m["text"] and now_t - m["t"] < 20.0), None)
                if last_pm:
                    sub = "MOVED: " + last_pm["text"].split(" - ")[-1][:70] + " · " + sub
                text = (f"{title_b} comes in · {sm['words'].split(': ', 1)[-1]}" if sm.get("state") != "running" else f"SEAM RUNNING: {sm['words'].split(': ', 1)[-1]}")
                row(t_seam, "SEAM", text, sub, key=("seam", sm.get("seq"), sm.get("bar")), bar=sm.get("bar"))
                t_rec2 = t_seam + int(sm["len"]) * bar_s
                pm = rc._play_min_bars() * bar_s
                row(t_rec2, "RECORD", f"{title_b} plays", f"min play until {_mmss(t_rec2 + pm)} · its seam is planned when its next record is staged",
                    key=("record2", sm["b"]))
            elif rec is not None:
                staged_any = [s for d, s in rc.songs.items() if d != rd and not s.leaving and s.staged_at is not None and not getattr(s, "part_only", False)]
                ev_tid = evp.get("track_id") if (evp and evp.get("kind") == "takeover" and evp.get("accepted")) else None
                if ev_tid is not None:
                    # THE EVENT IS THE EXIT: no seam is planned for this record; the takeover's song is the next
                    # record, the staged one follows it (its seam is planned once the takeover's song is the record)
                    t_ev = max(0, evp["in_bars"]) * bar_s
                    pm = rc._play_min_bars() * bar_s
                    after = [s for s in staged_any if s.track.id != ev_tid]
                    row(t_ev, "RECORD", f"{self.title(ev_tid)[:30]} plays (the takeover is this record's exit)",
                        f"from its drop · min play until {_mmss(t_ev + pm)} · " + (f"{after[0].track.title[:24]} follows it, its seam planned then" if after else "the next record is chosen then"),
                        key=("record-ev", ev_tid), bar=evp.get("bar"))
                elif staged_any:
                    row(None, "NEXT", f"{staged_any[0].track.title[:30]} is staged", "its seam is being placed (a bar or two)", key=("next-staged", staged_any[0].track.id))
                else:
                    left = (rec.get("settle_left") or 0) * bar_s
                    row(None, "NEXT", "no next record staged yet",
                        ("the DJ stages its pick while this record plays toward its minimum" if left > 0 else "the DJ is picking now") + " · put a song UP NEXT to decide it",
                        key=("next-none",))
            on_decks = {s.track.id for s in rc.songs.values()}
            after = [self.title(t)[:24] + (" (via)" if t in self.via_ids else "") for t in self.up_next
                     if t not in on_decks or (rc.seam is None or rc.songs.get(rc.seam["b"]) is None or rc.songs[rc.seam["b"]].track.id != t)]
            if after:
                row(None, "UP NEXT", ", ".join(after[:4]) + (" …" if len(after) > 4 else ""),
                    "your queue, in order" + (" · (via) = a record on the route to the one after it" if any(t in self.via_ids for t in self.up_next) else ""),
                    key=("upnext", tuple(after[:4])))
            # THE SET (docs/DJ_THE_SET.md): the chain after the queue, then where the clock goes
            for h in self.horizon[:5]:
                row(h.get("t"), "THEN", h["title"][:30], " · ".join(h.get("why") or []), key=("then", h["id"]), tid=h["id"])
            if not self.horizon:
                note = getattr(self, "_hz_note", "") or ""
                rec_bars = (rec or {}).get("since_bars") or 0
                if note:
                    row(None, "THEN", note[:80], "put a song UP NEXT to go somewhere else", key=("then-none", note[:30]))
                elif rec_bars > 4:
                    row(None, "THEN", "the set is being planned", "a few seconds", key=("then-planning",))
            if getattr(rc, "follow_clock", True) and (self.horizon or self._chain_pinned()):
                try:
                    step_s = rc.play_min_s + 30.0
                    t0 = next((r["t"] for r in rows if r["kind"] == "SEAM" and r.get("t") is not None), rc.play_min_s)
                    seq, clk = [], float(rc.master_bpm)
                    for i, pid in enumerate(self._chain_pinned()):
                        t = self.track(pid)
                        if t is not None and t.bpm:
                            seq.append((t0 + i * step_s, float(t.bpm)))
                    for h in self.horizon[:5]:
                        seq.append((h.get("t"), float(h["bpm"])))
                    words, last = [f"{clk:.0f} now"], clk
                    for t_at, b in seq:
                        if abs(b / last - 1.0) >= 0.01:
                            words.append(f"{b:.0f}" + (f" by {_mmss(t_at)}" if t_at is not None else ""))
                            last = b
                    if len(words) > 1:
                        row(None, "CLOCK", " → ".join(words), "the clock travels to each record's own tempo, half a percent a bar", key=("clock", tuple(words)))
                except Exception:
                    pass
        elif sysm is not None and sysm.current is not None:
            st = sysm.status()
            cur = st.get("current") or {}
            pos, dur = float(cur.get("pos_s") or 0.0), float(cur.get("duration_s") or 0.0)
            exit_in = None
            try:
                ep = getattr(sysm, "_exit_played", None)
                if ep:
                    exit_in = max(0.0, float(ep) - self._played_s())
            except Exception:
                exit_in = None
            row(0.0, "NOW", cur.get("title", "")[:34], f"{_mmss(pos)} of {_mmss(dur)}" + (f" · its exit in {_mmss(exit_in)}" if exit_in is not None else ""), key=("now1", cur.get("title")))
            nxt, plan = st.get("next") or {}, st.get("plan") or {}
            if nxt:
                eta = st.get("blend_in_s")
                row(eta, "SEAM", f"{nxt.get('title', '')[:30]} comes in" + (f" · {plan.get('style')} {plan.get('beats', 0):.0f} beats" if plan else ""),
                    ("begins " + (f"in {_mmss(eta)}" if eta is not None else "when the exit comes")) + (" · the seam is planned when the exit comes into range" if not plan else ""),
                    key=("seam1", nxt.get("title")))
            else:
                row(None, "NEXT", "not chosen yet", "the autoDJ picks as the exit comes into range · put a song UP NEXT to decide it", key=("next1",))
            if self.up_next:
                row(None, "UP NEXT", ", ".join(self.title(t)[:24] for t in self.up_next[:4]), "your queue, in order", key=("upnext1", tuple(self.up_next[:4])))
        else:
            row(None, "PLAN", "nothing playing yet", "press START", key=("idle",))
        return rows

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
                plan = (st.get("arrangement") or {}).get("plan")
                if staged:
                    bar_s = 4 * 60.0 / max(st.get("master_bpm") or 120, 1)
                    # the ETA counts to the PLAN's bar (decided once), not to the next phrase tick
                    if plan and plan.get("what") in ("voice_in", "seam"):
                        eta = max(0, plan["in_bars"]) * bar_s
                        how = (f"arrives as a voice: {plan['why']}" if plan["what"] == "voice_in" else f"the seam: {plan['words']} - {plan['why']}")
                    else:
                        eta = None
                        how = f"waits: the plan is {plan['words']} first" if plan else "in through one lane"
                    out["next"] = {"title": (plan["title"] if plan and plan.get("what") in ("voice_in", "seam") else staged[0]["title"]),
                                   "artist": staged[0].get("artist"), "eta_s": eta, "how": how}
                live = [s for s in songs.values() if s.get("entered")]
                arr = st.get("arrangement") or {}
                rec, sm = arr.get("record"), arr.get("seam")
                if rec is not None:
                    bar_s = 4 * 60.0 / max(st.get("master_bpm") or 120, 1)
                    played = rec.get("since_bars") or 0
                    bits = [f"the record: {rec['title'][:28]} ({int(played * bar_s) // 60}:{int(played * bar_s) % 60:02d} played)"]
                    if sm:
                        if sm.get("state") == "running":
                            bits.append(f"seam: {sm['words']}")
                        else:
                            left_s = max(0, sm["in_bars"]) * bar_s
                            bits.append(f"next: {sm['words']} in {int(left_s) // 60}:{int(left_s) % 60:02d} - {sm['why']}"
                                        + (f" (its drop {sm['drop_after_bars']} bars after)" if not sm.get("on_landmark") else "")
                                        + (f", rehearsed {sm['rehearsed']}" if sm.get("rehearsed") else "") + (" (held)" if st.get("hold") else ""))
                    else:
                        left = max(0, rec.get("settle_left") or 0)
                        bits.append("plays on" + (f" ({int(left * bar_s)} s to its minimum)" if left else "") + ": no next record staged yet" + (" (held)" if st.get("hold") else ""))
                    evp = rec.get("event")
                    if evp:
                        bits.append(f"event in {max(0, evp['in_bars'])} bars: {evp['why']}")
                else:
                    bits = [f"{len(live)} song{'s' if len(live) != 1 else ''} layered", f"a move every {st.get('change_bars')} bars",
                            "vocal lane free" if self.rc.vocal_freedom > 0.5 else "vocals cautious", f"energy {self.dials['energy']}"]
                # THE PLAN in words: the one next event and its bar (decided once), then the bed and the voices
                plan = arr.get("plan")
                if rec is not None:
                    pass
                elif arr.get("landing_in_bars") is not None:
                    bits.append(f"a new bed lands in {arr['landing_in_bars']} bars")
                elif plan:
                    bits.append(f"next: {plan['words']} ({max(0, plan['in_bars'])} bars) - {plan['why']}" + (" (held)" if st.get("hold") else ""))
                else:
                    bits.append("nothing planned: the bed plays on" + (" (held)" if st.get("hold") else ""))
                bed = arr.get("bed")
                if bed and rec is None:
                    b = f"bed: {bed['title'][:26]}"
                    if bed.get("since_bars") is not None:
                        b += f" for {bed['since_bars']} bars"
                    if bed.get("kind") == "build":
                        b += " (building to its drop)"
                    bits.append(b)
                for v in arr.get("voices") or []:
                    bits.append(f"voice {v['title'][:26]}: heard {v['heard_bars']} bars" + (" (on its way out)" if v.get("leaving") else ""))
                mo = st.get("moment")
                if mo:
                    bits.insert(1, f"{mo['label']} {'running' if mo['started'] else 'armed'} ({mo['who']}): the one in {mo['in_bars']} bars on {mo['what']}"
                                + (f" of {mo['other'][:20]}" if mo.get("other") and mo["kind"] in ("takeover", "bridge") else ""))
                if arr.get("moves_10min") is not None:
                    bits.append(f"{arr['moves_10min']} lane moves in the last 10 min")
                out["intent"] = ", ".join(bits)
                out["wait_why"] = arr.get("wait_why")
                out["moment"] = mo
                out["moment_stats"] = st.get("moment_stats") or {}
                out["moment_tune"] = st.get("moment_tune") or {}
                out["lanes"] = {ln: (songs.get(d) or {}).get("title") if d else None for ln, d in lanes.items()}
                out["recent"] = [f"{t}  {m}" for t, m in (st.get("moves") or [])[-8:]]
                out["error"] = out["error"] or st.get("error")
        except Exception as e:  # noqa: BLE001
            out["error"] = f"status: {type(e).__name__}: {e}"
        try:
            out["why"], out["in_effect"] = self._explain()
        except Exception as e:  # noqa: BLE001
            out["why"], out["in_effect"] = [f"(explain: {type(e).__name__}: {e})"], []
        try:
            out["plan_rows"] = self._plan_rows()
        except Exception as e:  # noqa: BLE001
            out["plan_rows"] = [{"t": None, "kind": "PLAN", "text": f"(the plan could not be read: {type(e).__name__}: {e})", "sub": "", "key": "err"}]
        out["horizon"] = [{"id": h["id"], "title": h["title"], "bpm": h["bpm"], "t": h.get("t"), "why": list(h.get("why") or [])} for h in self.horizon]
        out["set_why"] = self._hz_why
        out["requests"] = {k: dict(v) for k, v in self.requests.items()}
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
        try:
            out["ready"] = self.ready()
        except Exception as e:  # noqa: BLE001
            out["ready"] = []
            out["error"] = out.get("error") or f"ready: {type(e).__name__}: {e}"
        out["programs"] = {k: (v[0], v[1]) for k, v in PROGRAMS.items()}
        try:
            from lib.dj.session import SESSION_ID
            out["session"] = SESSION_ID
        except Exception:
            out["session"] = None
        out["last_dropout"] = self.last_dropout
        out["flags"] = list(self.flags[-5:])
        out["n_flags"] = len(self.flags)
        out["flag_kinds"] = dict(FLAG_KINDS)
        try:
            out["rate_target"] = self.rate_target()
        except Exception:
            out["rate_target"] = None
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
                     "voice_out": "the old bed song, now only a voice, fades out; one song's journey is complete",
                     "moment_arm": "a MOMENT is armed: it lands on the target song's drop or hook (else its next phrase); nothing else moves until it is done",
                     "moment": "a step of the armed MOMENT (take away, build, the hole, the one)",
                     "moment_cancel": "the moment was cancelled; everything came back on the bar",
                     "plan": "the plan moved: a rule forced the next event to another bar (the text says why); plans are decided once and otherwise kept",
                     "duck": "a voice's chords clash with the bed in the sections both are in now: it is held at half until they agree again",
                     "unduck": "the sections agree again: the voice comes back up"}
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
        try:
            from lib.dj.session import log_row
            log_row({"event": "director", "text": str(msg)[:200]})
        except Exception:
            pass
