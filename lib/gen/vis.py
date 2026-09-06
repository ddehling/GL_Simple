"""Generator ground truth -> visuals coupling (the sibling of lib/dj/vis.py).

While the generative system plays, the composer KNOWS the beat, the bar,
the phrase, the energy target and every upcoming drop - it decided them.
This maps that knowledge onto the reactive state keys every shader already
consumes (audio_energy, build_level, drop/drop_decay, beat pulses), the
same contract DJVisualCoupler fulfils for records. The DSP path stays as
the baseline; ground truth wins when it exists. No renderer imports: the
offline gate (tools/tests/_gen_vis_test.py) drives it with plain dicts."""
from __future__ import annotations

import numpy as np


class GenVisualCoupler:
    def __init__(self):
        self._energy_sm = None
        self._drop_seen = None
        self._drop_env = 0.0
        self._drop_tau = 0.6
        self._beat_prev = None
        self._beat_env = 0.0

    def apply(self, state, audio_dt, lb):
        """Mutate `state` in place. `lb` is GenSystem.live_beat() or None.
        Every branch is a no-op while gen_active is False."""
        dt = audio_dt or 0.025
        on = bool(state.get("gen_active"))
        e = state.get("gen_energy")
        if on and e is not None:
            k = 1.0 - float(np.exp(-dt / 0.8))
            sm = float(e) if self._energy_sm is None else self._energy_sm + (float(e) - self._energy_sm) * k
            self._energy_sm = sm
            state["audio_energy"] = sm
        else:
            self._energy_sm = None
        eta = state.get("gen_next_drop_eta")
        if on and eta is not None and eta < 8.0:
            state["build_level"] = max(state.get("build_level", 0.0), min(1.0, 1.0 - eta / 8.0))
        ddt = state.get("gen_drop_t")
        if on and ddt and ddt != self._drop_seen:
            self._drop_seen = ddt
            self._drop_env = 1.0
            state["drop"] = True
        if self._drop_env > 0.001:
            state["drop_decay"] = max(state.get("drop_decay", 0.0), self._drop_env)
            self._drop_env *= float(np.exp(-dt / self._drop_tau))
        if on and lb is not None:
            state["bpm"] = lb["bpm"]
            ph = lb["phase"]
            onset = self._beat_prev is not None and ph < self._beat_prev - 0.5
            self._beat_prev = ph
            drive = lb.get("drive", 1.0)
            if onset and drive >= 0.2:
                self._beat_env = drive
            benv = self._beat_env
            state["beat"] = bool(state.get("beat")) or (onset and drive >= 0.2)
            state["beat_decay"] = max(state.get("beat_decay", 0.0), benv)
            state["beat_phase"] = ph
            state["bar_phase"] = lb["bar_phase"]
            state["phrase_phase"] = lb["phrase_phase"]
            state["beat_confidence"] = max(state.get("beat_confidence", 0.0), 0.99)
            pulse = benv * min(1.0, lb.get("bass_share", 0.4) * 2.5)
            for key in ("beat_intensity", "bass_punch", "audio_punch"):
                state[key] = max(state.get(key, 0.0), pulse)
            self._beat_env = benv * float(np.exp(-dt / 0.18))
        else:
            self._beat_prev = None
            self._beat_env = 0.0
