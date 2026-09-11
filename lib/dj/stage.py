"""The stem stage: several stems from several tracks live at once, on one clock.

A LANE is one stem of one track playing a bar-aligned loop (or a whole
section) on its own deck, time-stretched to the stage's master tempo and
key-shifted toward the master key. Lanes come in and go out on the next bar
of the master lane; the master is the first lane that came in, and when it
leaves the next live lane takes over the clock. The submix's beat PLL holds
every other lane on the master's kicks (one PLL session per slave).

The stage does the technical work - quantisation, stretch, key shift, the
harmonic guard, gain staging - and exposes only musical decisions: which
stem of which track, which section, how many bars, in or out, how loud.
Nothing here touches a device: the caller mounts `stage.submix` on an
AudioEngine (the planner's Stage tab) or drives it through the engine's
mixer generator (the gate, tools/tests/_dj_stage_test.py).
"""
import math

import numpy as np

from lib.dj.submix import DJSubmix, RATE

STEMS = ("drums", "bass", "other", "vocals")
LEAD_S = 0.25                    # an action never lands closer than this to "now" (the event must be ahead of the clock)
RATE_MIN, RATE_MAX = 0.90, 1.10  # the deck's own wall (deck.set_rate)
KEY_SHIFT_MAX = 3                # semitones a melodic stem may be moved toward the master key
CLASH_BELOW = 0.55               # Camelot compatibility under this (after the best shift) is a clash
IN_BEATS = 1.0                   # a lane's gain rises over this many beats when it comes in
OUT_BEATS = 1.0                  # ...and falls over this many when it goes out


def camelot_compat(c1, c2):
    from lib.dj.brain import camelot_compat as _cc
    return _cc(c1, c2)


def shift_camelot(cam, semitones):
    from lib.dj.brain import _shift_camelot
    return _shift_camelot(cam, semitones)


class Lane:
    def __init__(self, name):
        self.deck = name
        self.track = None          # TrackInfo
        self.n_samples = 0
        self.stem = None
        self.section_i = None
        self.loop_bars = 8
        self.loop = None           # (start_s, end_s) in track time
        self.gain = 1.0
        self.state = "empty"       # empty | loaded | armed | live | leaving
        self.key_shift = 0
        self.compat = None         # Camelot compatibility with the master after the shift
        self.clash = False
        self.rate = 1.0
        self.pending_at = None     # submix clock of the scheduled action

    def as_dict(self):
        return {"deck": self.deck, "track_id": self.track.id if self.track else None,
                "title": self.track.title if self.track else None, "stem": self.stem,
                "section": self.section_i, "loop_bars": self.loop_bars, "loop": self.loop,
                "gain": self.gain, "state": self.state, "key_shift": self.key_shift,
                "compat": self.compat, "clash": self.clash, "rate": round(self.rate, 4)}


class Stage:
    def __init__(self, db, music_root, n_lanes=4):
        self.db = db
        self.music_root = music_root
        self.lanes = [Lane(f"lane{i}") for i in range(n_lanes)]
        self.submix = DJSubmix(deck_names=tuple(l.deck for l in self.lanes))
        self.master = None         # lane index of the clock
        self.master_bpm = None
        self.master_camelot = None
        self.log = []              # what the stage did, in order (the tab's status line, the gate's record)

    # -- material --------------------------------------------------------------
    def load(self, i, track):
        """Decode a track and its stems onto lane i's deck (call off the audio thread). The lane
        comes up silent with every stem closed; arm() opens one."""
        from lib.dj.features import decode_file_stereo
        from lib.dj.stems import load_stems
        lane = self.lanes[i]
        samples = decode_file_stereo(self.db.abs(track.path))
        stems = load_stems(self.music_root, track.id, expected_len=len(samples)) if getattr(track, "has_stems", False) else None
        if not stems:
            raise ValueError(f"{track.title}: no stems on disk")
        self.submix.post_many([
            {"cmd": "unload", "deck": lane.deck},
            {"cmd": "load", "deck": lane.deck, "samples": samples, "grid": track.grid, "track_id": track.id,
             "gain_db": track.gain_db, "stems": stems, "cue_s": 0.0},
            {"cmd": "gain", "deck": lane.deck, "value": 0.0, "ramp_s": 0.01},
            {"cmd": "stem_gains", "deck": lane.deck, "gains": {s: 0.0 for s in STEMS}, "ramp_s": 0.01},
        ])
        lane.track, lane.n_samples, lane.state = track, len(samples), "loaded"
        lane.stem, lane.section_i, lane.loop, lane.pending_at = None, None, None, None
        self._note(f"{lane.deck}: {track.title} loaded ({track.bpm:.1f} bpm, {track.camelot})")

    # -- the clock -------------------------------------------------------------
    def _tel(self):
        return self.submix.telemetry or {}

    def _grid_seg(self, track, t):
        for seg in track.grid or []:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return (track.grid or [None])[0]

    def bar_len_s(self, track, t):
        seg = self._grid_seg(track, t)
        return 4 * (seg["period_s"] if seg else track.period_s)

    def next_bar_clock(self):
        """The submix clock of the master lane's next downbeat at least LEAD_S ahead; 'now + LEAD_S'
        when there is no master yet."""
        tel = self._tel()
        now = int(tel.get("clock", self.submix.clock))
        if self.master is None:
            return now + int(LEAD_S * RATE)
        lane = self.lanes[self.master]
        d = (tel.get("decks") or {}).get(lane.deck) or {}
        if not d.get("playing"):
            return now + int(LEAD_S * RATE)
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        seg = self._grid_seg(lane.track, time_s)
        if seg is None:
            return now + int(LEAD_S * RATE)
        bar = 4 * seg["period_s"]
        first_down = seg["first_beat_s"] + lane.track.downbeat_offset * seg["period_s"]
        # the deck's time runs inside its loop: the next downbeat after now + lead, in track time
        t_min = time_s + LEAD_S * rate
        k = math.ceil((t_min - first_down) / bar - 1e-6)
        t_next = first_down + k * bar
        if lane.loop and t_next >= lane.loop[1] - 1e-3:
            # the loop wraps before that downbeat: the wrap IS the next downbeat (loops are bar-aligned)
            t_next = lane.loop[1]
        return now + int((t_next - time_s) / rate * RATE)

    # -- the harmonic guard ----------------------------------------------------
    def key_shift_for(self, track, stem):
        """(semitones, compatibility, clash) that put a melodic stem nearest the master key."""
        if stem == "drums" or not self.master_camelot or not track.camelot:
            return 0, None, False
        best = (0, camelot_compat(self.master_camelot, track.camelot))
        for s in (1, -1, 2, -2, 3, -3):
            if s > KEY_SHIFT_MAX or -s > KEY_SHIFT_MAX:
                continue
            c = camelot_compat(self.master_camelot, shift_camelot(track.camelot, s))
            if c > best[1] + 1e-9:
                best = (s, c)
            if best[1] >= 0.9:
                break
        return best[0], best[1], best[1] < CLASH_BELOW

    # -- actions ---------------------------------------------------------------
    def arm(self, i, stem, section_i, loop_bars=None, gain=None, allow_clash=False):
        """Bring lane i in on the next bar with `stem` looping `loop_bars` bars of section
        `section_i` (None = the whole section). The first lane in becomes the master clock.
        Returns the scheduled submix clock, or None when refused (a clash the caller did not allow,
        a tempo outside the deck's wall)."""
        lane = self.lanes[i]
        track = lane.track
        if track is None or stem not in STEMS:
            return None
        secs = track.sections or [{"start_s": 0.0, "end_s": track.duration_s, "kind": "all"}]
        sec = secs[max(0, min(section_i, len(secs) - 1))]
        # the loop: bar-aligned at the section's start, loop_bars long (or the whole section)
        start = track.nearest_downbeat(sec["start_s"])
        bar = self.bar_len_s(track, start)
        n_bars = int(loop_bars) if loop_bars else max(1, int(round((sec["end_s"] - start) / bar)))
        end = start + n_bars * bar
        if end > track.duration_s - 0.5:
            n_bars = max(1, int((track.duration_s - 0.5 - start) / bar))
            end = start + n_bars * bar
        # the clock: the first lane sets it, the rest follow it
        if self.master is None or self.lanes[self.master].state not in ("live", "armed"):
            rate = 1.0
            becomes_master = True
        else:
            rate = self.master_bpm / max(track.bpm, 1e-6)
            becomes_master = False
            if not (RATE_MIN <= rate <= RATE_MAX):
                self._note(f"{lane.deck}: {track.title} refused - {track.bpm:.1f} bpm needs rate {rate:.3f}")
                return None
        shift, compat, clash = self.key_shift_for(track, stem) if not becomes_master else (0, None, False)
        if clash and not allow_clash:
            self._note(f"{lane.deck}: {track.title} {stem} refused - key clash ({track.camelot} vs master {self.master_camelot}, best fit {compat:.2f})")
            lane.clash, lane.compat, lane.key_shift = clash, compat, shift
            return None
        at = self.next_bar_clock()
        beat = bar / 4.0 / rate
        g = float(lane.gain if gain is None else gain)
        ev = [
            {"at": at, "cmd": "cue", "deck": lane.deck, "time_s": start},
            {"at": at, "cmd": "rate", "deck": lane.deck, "value": rate},
            {"at": at, "cmd": "pitch", "deck": lane.deck, "semitones": float(shift)},
            {"at": at, "cmd": "loop", "deck": lane.deck, "start_s": start, "end_s": end},
            {"at": at, "cmd": "stem_gains", "deck": lane.deck, "gains": {s: (1.0 if s == stem else 0.0) for s in STEMS}, "ramp_s": 0.01},
            {"at": at, "cmd": "eq", "deck": lane.deck, "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
            {"at": at, "cmd": "start", "deck": lane.deck},
            {"at": at, "cmd": "gain", "deck": lane.deck, "value": g, "ramp_s": IN_BEATS * beat},
        ]
        if becomes_master:
            self.master, self.master_bpm, self.master_camelot = i, float(track.bpm), track.camelot
            # every other live lane now follows this clock (a promotion after the old master left)
            for j, other in enumerate(self.lanes):
                if j != i and other.state in ("live", "armed"):
                    ev.append({"at": at, "cmd": "sync", "slave": other.deck, "master": lane.deck, "bias_beats": 0.0, "audio_pll": True})
        else:
            ev.append({"at": at, "cmd": "sync", "slave": lane.deck, "master": self.lanes[self.master].deck,
                       "bias_beats": 0.0, "audio_pll": True})
        self.submix.post_many(ev)
        lane.stem, lane.section_i, lane.loop_bars, lane.loop = stem, section_i, n_bars, (start, end)
        lane.gain, lane.rate, lane.key_shift, lane.compat, lane.clash = g, rate, shift, compat, clash
        lane.state, lane.pending_at = "armed", at
        self._note(f"{lane.deck}: {track.title} {stem} in at bar clock {at} - {n_bars} bars from {start:.1f}s, rate {rate:.3f}, "
                   f"shift {shift:+d} st" + (" (MASTER)" if becomes_master else f" (compat {compat:.2f})" if compat is not None else ""))
        return at

    def release(self, i):
        """Take lane i out on the next bar. A leaving master hands the clock to the next live lane."""
        lane = self.lanes[i]
        if lane.state not in ("live", "armed"):
            return None
        at = self.next_bar_clock()
        beat = self.bar_len_s(lane.track, lane.loop[0] if lane.loop else 0.0) / 4.0 / max(lane.rate, 1e-6)
        stop_at = at + int(OUT_BEATS * beat * RATE) + 256
        ev = [{"at": at, "cmd": "gain", "deck": lane.deck, "value": 0.0, "ramp_s": OUT_BEATS * beat},
              {"at": stop_at, "cmd": "stop", "deck": lane.deck},
              {"at": stop_at, "cmd": "clear_loop", "deck": lane.deck},
              {"at": stop_at, "cmd": "end_sync", "slave": lane.deck}]
        if self.master == i:
            others = [j for j, o in enumerate(self.lanes) if j != i and o.state in ("live", "armed")]
            if others:
                new = others[0]
                ev.append({"at": at, "cmd": "end_sync", "slave": self.lanes[new].deck})
                for j in others[1:]:
                    ev.append({"at": at, "cmd": "sync", "slave": self.lanes[j].deck, "master": self.lanes[new].deck,
                               "bias_beats": 0.0, "audio_pll": True})
                self.master = new
                self._note(f"{self.lanes[new].deck} takes the clock from {lane.deck}")
            else:
                self.master, self.master_bpm, self.master_camelot = None, None, None
        self.submix.post_many(ev)
        lane.state, lane.pending_at = "leaving", stop_at
        self._note(f"{lane.deck}: out at bar clock {at}")
        return at

    def set_gain(self, i, g, ramp_s=0.1):
        lane = self.lanes[i]
        lane.gain = float(g)
        if lane.state in ("live", "armed"):
            self.submix.post({"cmd": "gain", "deck": lane.deck, "value": lane.gain, "ramp_s": ramp_s})

    def set_loop_bars(self, i, bars):
        """Shorten or lengthen a live lane's loop from the same start, on the next bar."""
        lane = self.lanes[i]
        if lane.loop is None or lane.state not in ("live", "armed"):
            lane.loop_bars = int(bars)
            return None
        start = lane.loop[0]
        bar = self.bar_len_s(lane.track, start)
        end = start + int(bars) * bar
        at = self.next_bar_clock()
        self.submix.post({"at": at, "cmd": "loop", "deck": lane.deck, "start_s": start, "end_s": end})
        lane.loop, lane.loop_bars = (start, end), int(bars)
        return at

    # -- state -----------------------------------------------------------------
    def refresh(self):
        """Advance lane states from the submix clock (armed -> live once the action fired, leaving -> loaded)."""
        now = int(self._tel().get("clock", self.submix.clock))
        for lane in self.lanes:
            if lane.pending_at is not None and now >= lane.pending_at:
                if lane.state == "armed":
                    lane.state = "live"
                elif lane.state == "leaving":
                    lane.state = "loaded"
                lane.pending_at = None

    def status(self):
        tel = self._tel()
        decks = tel.get("decks") or {}
        sync = tel.get("sync") or {}
        views = dict(sync.get("slaves") or ({sync["slave"]: sync} if sync else {}))
        out = {"master": self.lanes[self.master].deck if self.master is not None else None,
               "master_bpm": self.master_bpm, "master_camelot": self.master_camelot, "lanes": []}
        for lane in self.lanes:
            d = dict(lane.as_dict())
            t = decks.get(lane.deck) or {}
            d.update({"playing": bool(t.get("playing")), "time_s": t.get("time_s"), "beat_phase": t.get("beat_phase"),
                      "deck_gain": t.get("gain")})
            v = views.get(lane.deck)
            if v:
                d["audible_err_beats"] = v.get("audible_err_beats")
            out["lanes"].append(d)
        return out

    def _note(self, msg):
        self.log.append(msg)
        if len(self.log) > 200:
            self.log = self.log[-200:]
