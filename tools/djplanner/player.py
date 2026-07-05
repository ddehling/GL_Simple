"""Audio playback for the planner.

TrackPlayer   - one decoded track, instant seek/scrub (Analysis tab).
PlanPreview   - plays the COMPILED plan exactly as drawn (Mix tab): the
                Set tab's plan is compiled into a deterministic event
                script driven straight through the submix, with a
                background decode cache for instant click-around.
"""
import queue
import threading
import time

import numpy as np

RATE = 44100


class _Device:
    """Shared miniaudio pull device around a ring-ish queue of blocks."""

    def __init__(self, fetch):
        """fetch(n) -> float32 (n,2) called on the audio thread."""
        import miniaudio
        self._fetch = fetch

        def gen():
            required = yield b""
            while True:
                try:
                    chunk = self._fetch(required)
                except Exception:
                    chunk = np.zeros((required, 2), dtype=np.float32)
                required = yield chunk.astype(np.float32).tobytes()
        g = gen()
        next(g)
        self.dev = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2,
            sample_rate=RATE)
        self.dev.start(g)

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


class TrackPlayer:
    """Play a (n,2)/(n,) float32 array with sample-accurate seek."""

    def __init__(self):
        self.samples = None
        self.pos = 0
        self.playing = False
        self._dev = None

    def load(self, samples):
        s = np.asarray(samples, dtype=np.float32)
        if s.ndim == 1:
            s = np.stack([s, s], axis=1)
        self.samples = s
        self.pos = 0

    def _fetch(self, n):
        if not self.playing or self.samples is None:
            return np.zeros((n, 2), dtype=np.float32)
        a = self.samples[self.pos:self.pos + n]
        self.pos += len(a)
        if len(a) < n:
            self.playing = False
            a = np.concatenate([a, np.zeros((n - len(a), 2),
                                            dtype=np.float32)])
        return a

    def play(self):
        if self.samples is None:
            return
        if self._dev is None:
            self._dev = _Device(self._fetch)
        self.playing = True

    def pause(self):
        self.playing = False

    def seek(self, seconds):
        if self.samples is not None:
            self.pos = int(np.clip(seconds * RATE, 0,
                                   len(self.samples) - 1))

    def time_s(self):
        return self.pos / RATE

    def close(self):
        self.playing = False
        if self._dev is not None:
            self._dev.close()
            self._dev = None


class PlanPreview:
    """Plays the COMPILED plan, exactly as drawn.

    Unlike a live show (where the DJ system re-plans on the fly), the
    preview compiles the Set tab's plan into a deterministic event script
    and drives the submix directly - what you hear IS the timeline you see:
    same exit points, same styles, same EQ automation.

    Instant click-around: every track in the set is pre-decoded into an
    int16 cache in the background, so seeking anywhere (playing or stopped)
    rebuilds the script and is audible in ~0.1-0.2s. The playhead maps the
    submix sample clock back to DRAWN timeline seconds through per-seam
    anchors, so the marker and the audio can't drift apart.
    """

    CACHE_TRACKS = 8               # int16 stereo, ~53 MB per 5-min track
    LEAD_S = 20.0                  # load the incoming deck this early

    def __init__(self, db):
        self.db = db
        self.compiled = None
        self.brain = None
        self._cache = {}           # track_id -> int16 (n,2)
        self._order = []
        self._cache_lock = threading.Lock()
        self._predecoder = None
        # ~1.5s ahead-buffer: the decode cache means the producer never
        # stalls long, and a short buffer keeps LIVE monitor displays
        # honest (audible lag is exposed via audible_lag_s()).
        self._buf = queue.Queue(maxsize=30)
        self._leftover = np.zeros((0, 2), dtype=np.float32)
        self._dev = None
        self._producer = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._sub = None
        self._deck_slot = {}        # deck name -> slot index it is playing
        self.playing = False
        self.error = ""
        self.decoding = ""

    # -- plan + cache -------------------------------------------------------
    def set_plan(self, compiled, brain):
        self.compiled = compiled
        self.brain = brain
        # Pre-decode the whole set in the background for instant seeks.
        if compiled and compiled["slots"]:
            tracks = [s["track"] for s in compiled["slots"]]
            self._predecoder = threading.Thread(
                target=self._predecode, args=(tracks,), daemon=True)
            self._predecoder.start()

    def _predecode(self, tracks):
        for t in tracks:
            with self._cache_lock:
                if t.id in self._cache:
                    continue
            try:
                self.decoding = t.title
                self._decode(t)
            except Exception:
                pass
        self.decoding = ""

    def _decode(self, track):
        from lib.dj.features import decode_file_stereo
        s = decode_file_stereo(self.db.abs(track.path))
        s16 = (np.clip(s, -1.0, 1.0) * 32767).astype(np.int16)
        with self._cache_lock:
            self._cache[track.id] = s16
            if track.id in self._order:
                self._order.remove(track.id)
            self._order.append(track.id)
            while len(self._order) > self.CACHE_TRACKS:
                self._cache.pop(self._order.pop(0), None)
        return s16

    def _samples(self, track):
        with self._cache_lock:
            s16 = self._cache.get(track.id)
            if s16 is not None and track.id in self._order:
                self._order.remove(track.id)
                self._order.append(track.id)
        if s16 is None:
            s16 = self._decode(track)
        return s16.astype(np.float32) / 32767.0

    # -- script compilation -----------------------------------------------------
    def _build_script(self, slot0, cue0):
        """Compile the plan from (slot0, cue0) into producer actions
        [(clock, 'load'|'post', payload)] on absolute submix clocks.
        Returns (actions, anchors)."""
        slots = self.compiled["slots"]
        actions = []
        anchors = []
        active, incoming = "a", "b"
        first = slots[slot0]["track"]
        actions.append((0, "load", {"deck": active, "track": first,
                                    "cue": cue0, "start": True,
                                    "slot": slot0}))
        c_ref, s_ref = 0, cue0     # active deck: source s_ref at clock c_ref
        for k in range(slot0, len(slots) - 1):
            plan = slots[k]["transition"]
            if plan is None:
                break
            cur, nxt = slots[k]["track"], slots[k + 1]["track"]
            blend_clock = c_ref + int((plan["out_s"] - s_ref) * RATE)
            snapshot = {"clock": blend_clock,
                        "decks": {active: {"time_s": plan["out_s"],
                                           "rate": 1.0}}}
            ev, swap_at, blend_at = self.brain.build_events(
                plan, snapshot, active, incoming, cur, nxt)
            # If we seeked PAST a technique's setup (loop-build, bassline
            # layer, etc. begin well before the exit), its events land at
            # negative clocks and would fire immediately - a mistimed mess.
            # Fall back to a clean bass_swap that starts at the exit point.
            if ev and min(e["at"] for e in ev) < c_ref:
                simple = dict(plan)
                simple["style"] = "bass_swap"
                simple["beats"] = 16
                ev, swap_at, blend_at = self.brain.build_events(
                    simple, snapshot, active, incoming, cur, nxt)
                plan = simple
            # Keep the full slow glide-home (~20s for a 3% match) so the new
            # track eases back to its natural tempo imperceptibly - the old
            # 1.5s snap lurched audibly after every mix. The tiny seam-timing
            # drift is self-corrected by the telemetry playhead.
            rate_b = plan["rate"]
            if plan["style"] == "cut_at_drop":
                start_cue = max(0.0, plan["in_s"] - 16 * nxt.period_s)
            else:
                start_cue = plan["in_s"]
            load_clock = max(0, blend_at - int(self.LEAD_S * RATE))
            actions.append((load_clock, "load",
                            {"deck": incoming, "track": nxt,
                             "cue": start_cue, "start": False,
                             "slot": k + 1}))
            actions.append((load_clock + 1, "post", ev))
            anchors.append((blend_at, slots[k + 1]["start_offset_s"]))
            # New dominant deck's reference after the swap (rate snapped).
            if plan["style"] == "long_fade":
                s_swap = start_cue + (swap_at - blend_at) / RATE
            else:
                s_swap = start_cue + rate_b * (swap_at - blend_at) / RATE
            c_ref, s_ref = swap_at, s_swap
            active, incoming = incoming, active
        return actions, anchors

    # -- transport -----------------------------------------------------------
    def play_at(self, drawn_t):
        """Start (or jump) playback at a drawn-timeline time. Works while
        playing or stopped - standard DJ-software click-around."""
        if not self.compiled or not self.compiled["slots"]:
            return
        slots = self.compiled["slots"]
        slot = 0
        for i, s in enumerate(slots):
            if drawn_t >= s["start_offset_s"]:
                slot = i
        s = slots[slot]
        track = s["track"]
        in_s = s.get("in_s")
        if in_s is None:
            in_s = track.mix_ins[0]["time_s"] if track.mix_ins else 0.0
        cue = in_s + max(drawn_t - s["start_offset_s"], 0.0)
        # Keep clear of the seam so the scripted blend still runs whole.
        if s["transition"] is not None:
            cue = min(cue, s["transition"]["out_s"] - 2.0)
        cue = track.nearest_downbeat(max(cue, 0.0))
        self._restart(slot, cue, s["start_offset_s"] + (cue - in_s))

    def _restart(self, slot, cue, drawn_start):
        self._halt_producer()
        try:
            actions, anchors = self._build_script(slot, cue)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
            return
        self._deck_slot = {}
        self._stop.clear()
        self._pause.clear()
        self.playing = True
        self._producer = threading.Thread(
            target=self._produce, args=(actions,), daemon=True)
        self._producer.start()
        if self._dev is None:
            self._dev = _Device(self._fetch)

    def pause(self, on):
        (self._pause.set if on else self._pause.clear)()

    def stop(self):
        self._halt_producer()
        self.playing = False

    def close(self):
        self.stop()
        if self._dev is not None:
            self._dev.close()
            self._dev = None

    def _halt_producer(self):
        self._stop.set()
        if self._producer is not None:
            self._producer.join(timeout=2.0)
            self._producer = None
        self._sub = None
        try:
            while True:
                self._buf.get_nowait()
        except queue.Empty:
            pass
        self._leftover = np.zeros((0, 2), dtype=np.float32)

    # -- navigation ---------------------------------------------------------
    def _active_deck_slot(self):
        """(deck_dict, slot_index) for the LOUDEST playing deck, or (None,None).
        Derived from real telemetry - the ground truth of what's audible."""
        sub = self._sub
        if sub is None or not self.playing:
            return None, None
        decks = (sub.telemetry or {}).get("decks") or {}
        cands = [(d.get("gain", 0.0), self._deck_slot[name], d)
                 for name, d in decks.items()
                 if d.get("playing") and name in self._deck_slot]
        if not cands:
            return None, None
        # Loudest deck wins; when two are within 0.2 gain (crossfade
        # midpoint) prefer the LATER slot so position advances into the
        # incoming track instead of flickering back to the outgoing one.
        top = max(c[0] for c in cands)
        near = [c for c in cands if top - c[0] <= 0.2]
        gain, slot, d = max(near, key=lambda c: c[1])
        return d, slot

    def playhead(self):
        """Position in DRAWN timeline seconds from the loudest deck's real
        source position - monotonic within a track, steps at handover, never
        interpolation-jumps."""
        d, slot = self._active_deck_slot()
        if d is None or not self.compiled:
            return None
        s = self.compiled["slots"][slot]
        in_s = s.get("in_s")
        if in_s is None:
            in_s = s["track"].mix_ins[0]["time_s"] if s["track"].mix_ins else 0.0
        return s["start_offset_s"] + max(d.get("time_s", 0.0) - in_s, 0.0)

    def slot_at_playhead(self):
        _, slot = self._active_deck_slot()
        return slot if slot is not None else 0

    def jump_slot(self, delta):
        if not self.compiled:
            return
        slots = self.compiled["slots"]
        i = max(0, min(self.slot_at_playhead() + delta, len(slots) - 1))
        self.play_at(slots[i]["start_offset_s"] + 0.01)

    def next_seam(self, pre_roll=8.0):
        """Deterministic: jump to just before the next drawn seam."""
        t = self.playhead()
        if t is None or not self.compiled:
            return
        for s in self.compiled["slots"][1:]:
            if s["start_offset_s"] > t + pre_roll:
                self.play_at(max(s["start_offset_s"] - pre_roll, 0.0))
                return

    def telemetry(self):
        sub = self._sub
        return sub.telemetry if sub is not None else {}

    def audible_lag_s(self):
        """Seconds the producer is ahead of the speakers (ring fill)."""
        return (self._buf.qsize() * 2205 + len(self._leftover)) / RATE

    def cached_samples(self, track_id):
        """int16 stereo array for a set track, or None (live monitor)."""
        with self._cache_lock:
            return self._cache.get(track_id)

    # -- producer -------------------------------------------------------------
    def _fetch(self, n):
        if self._pause.is_set() or not self.playing:
            return np.zeros((n, 2), dtype=np.float32)
        out = np.zeros((n, 2), dtype=np.float32)
        got = 0
        if len(self._leftover):
            take = min(n, len(self._leftover))
            out[:take] = self._leftover[:take]
            self._leftover = self._leftover[take:]
            got = take
        while got < n:
            try:
                blk = self._buf.get_nowait()
            except queue.Empty:
                break
            take = min(n - got, len(blk))
            out[got:got + take] = blk[:take]
            self._leftover = blk[take:]
            got += take
        return out

    def _produce(self, actions):
        try:
            from lib.dj.submix import DJSubmix
            sub = DJSubmix()
            self._sub = sub
            actions = sorted(actions, key=lambda a: a[0])
            block = 2205
            while not self._stop.is_set():
                if self._pause.is_set():
                    time.sleep(0.05)
                    continue
                while actions and actions[0][0] <= sub.clock:
                    _, kind, payload = actions.pop(0)
                    if kind == "load":
                        t = payload["track"]
                        samples = self._samples(t)
                        self._deck_slot[payload["deck"]] = payload["slot"]
                        sub.post({"cmd": "load", "deck": payload["deck"],
                                  "samples": samples, "track_id": t.id,
                                  "grid": t.grid, "gain_db": t.gain_db,
                                  "kick_offset_s": t.kick_offset_s,
                                  "cue_s": payload["cue"]})
                        if payload.get("start"):
                            sub.post({"cmd": "gain", "deck": payload["deck"],
                                      "value": 1.0, "ramp_s": 0.05})
                            sub.post({"cmd": "start",
                                      "deck": payload["deck"]})
                    elif kind == "post":
                        sub.post_many(payload)
                blk = np.clip(sub.read(block), -1.0, 1.0)
                while not self._stop.is_set():
                    try:
                        self._buf.put(blk, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
