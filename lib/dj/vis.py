"""DJ ground-truth -> visuals coupling.

While the autonomous DJ plays, the decks KNOW what the room is about to
hear - energy curves, drop times, the beat grid, a pending operator
MOMENT - and this class translates that knowledge into the reactive
state keys every shader already consumes (audio_energy, build_level,
drop/drop_decay, beat pulses). The DSP path stays as the mic-mode
fallback; ground truth wins when it exists.

Extracted from Stories_OGL's audio-state method so the OFFLINE trace
gate (tools/tests/_dj_moment_vis_test.py) exercises the exact code that
drives the room - the light choreography of a crowd moment can be
validated as curves without the rig. Keep this class free of renderer /
hardware imports for that reason.
"""
import numpy as np


class DJVisualCoupler:
    """Stateful per-frame mapper: DJ outstate keys (already merged into
    `state` by the scheduler) + the deck's live beat -> shader-facing
    reactive keys. Call apply() once per audio tick AFTER the DSP `sig`
    baseline has been written into `state`."""

    def __init__(self):
        self._energy_sm = None       # smoothed ground-truth energy
        self._drop_seen = None       # last consumed dj_drop_t stamp
        self._drop_env = 0.0         # drop flash envelope
        self._drop_tau = 0.35        # its decay: 0.35s natural, 1.1s hard
        self._beat_prev = None       # last grid beat phase (onset detect)
        self._beat_env = 0.0         # grid-pulse floor envelope

    def apply(self, state, audio_dt, lb):
        """Mutate `state` in place. `lb` is DJSystem.live_beat() or None.

        When the DJ is inactive every branch below is a no-op and the
        DSP values already in `state` pass through untouched."""
        dt = audio_dt or 0.025
        dj_on = bool(state.get("dj_active"))
        # GROUND-TRUTH ENERGY: the DSP estimate reads every steady track
        # as ~medium (AGC bands hover at 1.0), which left the club unable
        # to tell chill from peak. Smoothed ~0.8s - just enough to hide
        # the 2 Hz curve steps.
        dj_e = state.get("dj_energy")
        if dj_on and dj_e is not None:
            k = 1.0 - float(np.exp(-dt / 0.8))
            sm = dj_e if self._energy_sm is None \
                else self._energy_sm + (float(dj_e) - self._energy_sm) * k
            self._energy_sm = sm
            state["audio_energy"] = sm
        else:
            self._energy_sm = None
        # DJ FOREKNOWLEDGE -> deterministic build: the decks publish the
        # next drop/seam ETA; ramp build_level through the final 8s so
        # every pattern's coil-up (and the director's squeeze) lands
        # BEFORE every known drop - the DSP riser detector only catches
        # builds the mastering makes obvious.
        dj_eta = state.get("dj_next_drop_eta")
        if dj_on and dj_eta is not None and dj_eta < 8.0:
            state["build_level"] = max(state["build_level"],
                                       min(1.0, 1.0 - dj_eta / 8.0))
        # DJ ground-truth drops: the decks KNOW when a drop section lands
        # (published as a wall-time stamp). Fire the same drop/drop_decay
        # signals the DSP detector would - hard sets never give the DSP
        # path the quiet episode it needs to arm, so without this the
        # club sat still through the hardest-hitting moments.
        ddt = state.get("dj_drop_t")
        if dj_on and ddt and ddt != self._drop_seen:
            self._drop_seen = ddt
            self._drop_env = 1.0
            # An ENGINEERED landing (operator MOMENT) slams the room ~3x
            # longer than a passing musical drop - the whole build/hole
            # choreography exists for this instant, and a 0.35s flash
            # under it read as 'the lights amp up for a couple of
            # seconds' (user). Natural drops keep the quick flash.
            self._drop_tau = 1.1 if state.get("dj_drop_hard") else 0.35
            state["drop"] = True
        if self._drop_env > 0.001:
            state["drop_decay"] = max(state["drop_decay"], self._drop_env)
            self._drop_env *= float(np.exp(-dt / self._drop_tau))
        # GROUND-TRUTH BEAT: while the DJ plays, the audible deck's stored
        # grid IS the beat - sample-tight bpm/phase/bar/phrase for every
        # beat-synced shader, where the DSP detector on the mix lags and
        # quantizes ('doesn't respond to beats in a clear manner' - user).
        # Punches get a grid-pulse FLOOR scaled by the section's bass
        # share: relentless hard sets flatten the AGC punch envelopes
        # exactly when the floor hits hardest.
        if lb is not None:
            state["bpm"] = lb["bpm"]
            ph = lb["phase"]
            onset = self._beat_prev is not None and ph < self._beat_prev - 0.5
            self._beat_prev = ph
            drive = lb.get("drive", 1.0)
            if onset and drive >= 0.2:
                self._beat_env = drive
            benv = self._beat_env
            # Phases/bpm stay grid-true through breakdowns (motion should
            # keep gliding); PULSES follow the section's actual rhythm -
            # a resting kick must not flash the room.
            state["beat"] = bool(state["beat"]) or (onset and drive >= 0.2)
            state["beat_decay"] = max(state["beat_decay"], benv)
            state["beat_phase"] = ph
            state["bar_phase"] = lb["bar_phase"]
            state["phrase_phase"] = lb["phrase_phase"]
            state["beat_confidence"] = max(state["beat_confidence"], 0.95)
            pulse = benv * min(1.0, lb["bass_share"] * 2.5)
            state["beat_intensity"] = max(state["beat_intensity"], pulse)
            state["bass_punch"] = max(state["bass_punch"], pulse)
            state["audio_punch"] = max(state["audio_punch"], pulse)
            self._beat_env = benv * float(np.exp(-dt / 0.18))
        else:
            self._beat_prev = None
            self._beat_env = 0.0
        # OPERATOR MOMENT breath-hold: through the gesture's silent span
        # (the hole / the dying platter / the echo stall) the room holds
        # its breath WITH the music - build pinned at max, beat pulses
        # suppressed. Placed AFTER the ground-truth beat block above:
        # the deck grid keeps ticking through the silence, and its pulse
        # floor would otherwise flash the room on beats the ear can't
        # hear. Phases/bpm are left alone so motion glides through.
        if dj_on and state.get("dj_moment_hole"):
            state["build_level"] = 1.0
            state["beat"] = False
            state["beat_decay"] = min(state["beat_decay"], 0.15)
            for k in ("beat_intensity", "bass_punch", "audio_punch"):
                state[k] = min(state[k], 0.1)
