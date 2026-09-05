"""SuperCollider backend (Spike SC-0 of docs/GENERATIVE_MUSIC_PLAN.md §3.1).

The composer's NoteEvents become timestamped OSC bundles for scsynth.
SynthDefs are compiled here in Python with supriya - no sclang anywhere.

Two modes on the same synthdefs and the same event translation:
  * NRT   render_nrt(events, seconds, out.wav): supriya Score -> scsynth -N.
          Hermetic: no audio device, deterministic, ~faster than realtime.
  * LIVE  SCLive: boots scsynth (pw-jack / PortAudio), schedules bundles
          `latency` seconds ahead of the server clock. The show would
          capture its output back through a PipeWire null sink (plan §3.1
          routing R2); gen_player --backend sc-live just plays it.

Optional dependency: `pip install supriya`, plus the scsynth binary
(Debian: supercollider-server; sc3-plugins optional, not required here)."""
from __future__ import annotations

import os
import time

from lib.gen import RATE
from lib.gen.theory import midi_to_hz


def _synthdefs():
    """Build the rack's SynthDefs. Core UGens only (no sc3-plugins), so
    any stock scsynth renders them."""
    from supriya import Envelope, synthdef
    from supriya.ugens import (BPF, HPF, LPF, CombC, EnvGen, FreeVerb, Impulse,
                               In, Line, MoogFF, Out, Pan2, RLPF, Saw, SinOsc,
                               VarSaw, WhiteNoise, XLine, Lag, DelayC, Decay2)

    @synthdef()
    def gen_kick(out=0, amp=0.9, decay=0.38, pitch=46.0, pan=0.0):
        freq = XLine.kr(start=pitch * 4.5, stop=pitch, duration=0.045)
        env = EnvGen.kr(envelope=Envelope.percussive(attack_time=0.001, release_time=decay), done_action=2)
        click = WhiteNoise.ar() * EnvGen.kr(envelope=Envelope.percussive(0.0005, 0.004))
        sig = (SinOsc.ar(frequency=freq) * 1.6 * env + click * 0.4).tanh()
        Out.ar(bus=out, source=Pan2.ar(source=sig * amp, position=pan))

    @synthdef()
    def gen_clap(out=0, amp=0.5, pan=0.05):
        noise = BPF.ar(source=WhiteNoise.ar(), frequency=1500.0, reciprocal_of_q=1.2)
        env = EnvGen.kr(envelope=Envelope.percussive(attack_time=0.001, release_time=0.11), done_action=2)
        # three fast bursts then a tail
        bursts = Decay2.ar(source=Impulse.ar(frequency=90.0) * Line.kr(start=1.0, stop=0.0, duration=0.035), attack_time=0.001, decay_time=0.012)
        sig = noise * (bursts * 1.5 + env * 0.8)
        Out.ar(bus=out, source=Pan2.ar(source=sig * amp, position=pan))

    @synthdef()
    def gen_snare(out=0, amp=0.5, pan=0.05):
        tone = SinOsc.ar(frequency=185.0) * EnvGen.kr(envelope=Envelope.percussive(0.001, 0.045)) * 0.6
        noise = HPF.ar(source=WhiteNoise.ar(), frequency=1800.0) * EnvGen.kr(envelope=Envelope.percussive(0.001, 0.08), done_action=2)
        Out.ar(bus=out, source=Pan2.ar(source=(tone + noise * 0.8) * amp, position=pan))

    @synthdef()
    def gen_hat(out=0, amp=0.3, decay=0.05, pan=-0.25):
        env = EnvGen.kr(envelope=Envelope.percussive(attack_time=0.0005, release_time=decay), done_action=2)
        sig = HPF.ar(source=WhiteNoise.ar(), frequency=7500.0) * env
        Out.ar(bus=out, source=Pan2.ar(source=sig * amp, position=pan))

    @synthdef()
    def gen_perc(out=0, amp=0.3, hz=380.0, pan=0.4):
        freq = XLine.kr(start=hz * 1.8, stop=hz, duration=0.012)
        env = EnvGen.kr(envelope=Envelope.percussive(0.001, 0.05), done_action=2)
        tick = WhiteNoise.ar() * EnvGen.kr(envelope=Envelope.percussive(0.0005, 0.004)) * 0.4
        Out.ar(bus=out, source=Pan2.ar(source=(SinOsc.ar(frequency=freq) * env + tick) * amp, position=pan))

    @synthdef()
    def gen_bass(out=0, freq=55.0, amp=0.7, gate=1, cutoff=900.0, res=0.35, pan=0.0):
        env = EnvGen.kr(envelope=Envelope.adsr(0.002, 0.1, 0.65, 0.06), gate=gate, done_action=2)
        fenv = EnvGen.kr(envelope=Envelope.adsr(0.001, 0.09, 0.15, 0.05), gate=gate)
        osc = Saw.ar(frequency=freq) + SinOsc.ar(frequency=freq * 0.5) * 0.6
        sig = MoogFF.ar(source=osc, frequency=cutoff * (1.0 + 2.5 * fenv), gain=res * 3.5)
        Out.ar(bus=out, source=Pan2.ar(source=sig.tanh() * env * amp, position=pan))

    @synthdef()
    def gen_lead(out=0, freq=440.0, amp=0.3, gate=1, cutoff=2600.0, res=0.25, pan=-0.15):
        env = EnvGen.kr(envelope=Envelope.adsr(0.006, 0.18, 0.55, 0.14), gate=gate, done_action=2)
        fenv = EnvGen.kr(envelope=Envelope.adsr(0.003, 0.2, 0.35, 0.1), gate=gate)
        vib = 1.0 + SinOsc.kr(frequency=5.3) * 0.004
        osc = Saw.ar(frequency=freq * vib * 0.996) + Saw.ar(frequency=freq * vib * 1.004)
        sig = RLPF.ar(source=osc * 0.5, frequency=cutoff * (1.0 + 1.6 * fenv), reciprocal_of_q=1.0 - res * 0.8)
        Out.ar(bus=out, source=Pan2.ar(source=sig * env * amp, position=pan))

    @synthdef()
    def gen_pad(out=0, freq=220.0, amp=0.25, gate=1, cutoff=1400.0, pan=0.0):
        env = EnvGen.kr(envelope=Envelope.adsr(0.5, 0.6, 0.85, 0.9), gate=gate, done_action=2)
        fenv = EnvGen.kr(envelope=Envelope.adsr(1.2, 1.5, 0.6, 1.0), gate=gate)
        osc = (VarSaw.ar(frequency=freq * 0.993, width=0.4) + VarSaw.ar(frequency=freq * 0.997, width=0.5)
               + VarSaw.ar(frequency=freq * 1.003, width=0.5) + VarSaw.ar(frequency=freq * 1.007, width=0.6)) * 0.25
        sig = LPF.ar(source=osc, frequency=cutoff * (0.5 + 1.2 * fenv))
        Out.ar(bus=out, source=Pan2.ar(source=sig * env * amp, position=pan))

    @synthdef()
    def gen_pluck(out=0, freq=440.0, amp=0.28, gate=1, cutoff=3200.0, pan=0.2):
        env = EnvGen.kr(envelope=Envelope.adsr(0.001, 0.25, 0.0, 0.09), gate=gate, done_action=2)
        fenv = EnvGen.kr(envelope=Envelope.percussive(0.001, 0.12))
        osc = Saw.ar(frequency=freq * 0.9977) + VarSaw.ar(frequency=freq * 1.0023, width=0.3)
        sig = RLPF.ar(source=osc * 0.5, frequency=cutoff * (0.3 + 3.0 * fenv), reciprocal_of_q=0.7)
        Out.ar(bus=out, source=Pan2.ar(source=sig * env * amp, position=pan))

    @synthdef()
    def gen_fx(in_bus=16, out=0, delay_time=0.36, delay_fb=0.42, room=0.6, mix_delay=0.35, mix_reverb=0.3):
        """Master FX: reads the send bus (stereo), returns delay + reverb."""
        src = In.ar(bus=in_bus, channel_count=2)
        dl = CombC.ar(source=src, maximum_delay_time=2.0, delay_time=delay_time, decay_time=delay_time * 6.0)
        rv = FreeVerb.ar(source=src, mix=1.0, room_size=room, damping=0.4)
        Out.ar(bus=out, source=dl * mix_delay + rv * mix_reverb)

    @synthdef()
    def gen_master(bus=0, gain=0.8, ceiling=0.95):
        """Master limiter on the output bus (the numpy rack's soft clip)."""
        from supriya.ugens import Limiter, ReplaceOut
        src = In.ar(bus=bus, channel_count=2)
        ReplaceOut.ar(bus=bus, source=Limiter.ar(source=src * gain, level=ceiling, duration=0.008))

    return {
        "master": gen_master,
        "kick": gen_kick, "clap": gen_clap, "snare": gen_snare, "hat": gen_hat,
        "perc": gen_perc, "bass": gen_bass, "lead": gen_lead, "pad": gen_pad,
        "pluck": gen_pluck, "fx": gen_fx,
    }


SEND_BUS = 16       # private stereo audio bus for delay/reverb sends


def _translate(events, style, key_of_slot):
    """NoteEvents -> [(t_seconds, synthdef_name, kwargs, gate_end_seconds|None)]."""
    out = []
    for e in events:
        patch = style["slots"].get(e.slot)
        if patch is None:
            continue
        voice = patch.get("voice")
        name = key_of_slot.get(voice)
        if name is None:
            continue
        t = e.at / RATE
        g = float(patch.get("gain", 0.5))
        kw = {"amp": g * (0.4 + 0.6 * e.vel)}
        if voice in ("kick",):
            kw["decay"] = float(patch.get("decay", 0.38))
        elif voice == "hat":
            kw["decay"] = float(e.params.get("decay", patch.get("decay", 0.05)))
        elif voice in ("bass", "lead", "pad", "pluck"):
            kw["freq"] = midi_to_hz(e.pitch)
            kw["cutoff"] = float(e.params.get("cutoff", patch.get("cutoff", 1200.0))) * (0.35 + 0.65 * e.vel)
            if voice in ("bass", "lead"):
                kw["res"] = float(patch.get("res", 0.25))
        gate_end = (e.at + e.dur) / RATE if voice in ("bass", "lead", "pad", "pluck") else None
        send = float(patch.get("send_delay", 0.0)) + float(patch.get("send_reverb", 0.0))
        out.append((t, name, kw, gate_end, send))
    return out


class SCBackend:
    """Shared: synthdefs + translation. Subclasses schedule."""

    def __init__(self, style: dict):
        self.style = style
        self.defs = _synthdefs()
        self.names = {k: v for k, v in self.defs.items()}

    def plan(self, events):
        return _translate(events, self.style, {k: k for k in self.defs})


def render_nrt(events, seconds: float, out_path: str, style: dict, sample_rate: int = RATE):
    """Render events through scsynth in non-realtime. Returns out_path."""
    import supriya
    from supriya import Score

    be = SCBackend(style)
    score = Score(options=supriya.Options(output_bus_channel_count=2, input_bus_channel_count=0,
                                          sample_rate=sample_rate, audio_bus_channel_count=64))
    plan = be.plan(events)
    with score.at(0):
        score.add_synthdefs(*be.defs.values())
        fx_group = score.add_group()
        main = score.add_group(add_action="ADD_BEFORE", target_node=fx_group)
        fx_group.add_synth(be.defs["fx"], in_bus=SEND_BUS, out=0,
                           delay_time=60.0 / _bpm_guess(style) * 0.75)
        fx_group.add_synth(be.defs["master"], add_action="ADD_TO_TAIL", bus=0)
    # a note with sends plays twice: dry to 0 and a scaled copy to the send bus
    for t, name, kw, gate_end, send in plan:
        with score.at(t):
            synth = main.add_synth(be.defs[name], out=0, **kw)
            if send > 0.0:
                kw2 = dict(kw)
                kw2["amp"] = kw["amp"] * send
                s2 = main.add_synth(be.defs[name], out=SEND_BUS, **kw2)
            else:
                s2 = None
        if gate_end is not None:
            with score.at(gate_end):
                synth.set(gate=0)
                if s2 is not None:
                    s2.set(gate=0)
    with score.at(seconds):
        score.do_nothing()
    import asyncio
    import inspect
    result = score.render(out_path, duration=seconds, sample_rate=sample_rate,
                          header_format="WAV", sample_format="INT16")
    if inspect.isawaitable(result):          # supriya >= 26: Score.render is async
        result = asyncio.run(result)
    path, code = result
    if code != 0:
        raise RuntimeError(f"scsynth NRT failed with code {code}")
    return str(path)


def _bpm_guess(style):
    lo, hi = style["bpm"]
    return 0.5 * (lo + hi)


class SCLive:
    """Realtime scsynth driven `latency` seconds ahead. Feed it phrases as
    the composer produces them; it converts sample times to server time."""

    def __init__(self, style: dict, bpm: float, latency: float = 0.2, options=None):
        import supriya
        self.be = SCBackend(style)
        self.latency = float(latency)
        self.server = supriya.Server().boot(options=options)
        self.server.add_synthdefs(*self.be.defs.values())
        self.server.sync()
        self.fx_group = self.server.add_group()
        self.main = self.server.add_group(add_action="ADD_BEFORE", target_node=self.fx_group)
        self.fx_group.add_synth(self.be.defs["fx"], in_bus=SEND_BUS, out=0, delay_time=60.0 / bpm * 0.75)
        self.fx_group.add_synth(self.be.defs["master"], add_action="ADD_TO_TAIL", bus=0)
        self.t0 = time.time() + self.latency      # wall time of sample 0

    def schedule(self, events):
        for t, name, kw, gate_end, send in self.be.plan(events):
            at = self.t0 + t
            with self.server.at(at):
                synth = self.main.add_synth(self.be.defs[name], out=0, **kw)
                s2 = None
                if send > 0.0:
                    kw2 = dict(kw); kw2["amp"] = kw["amp"] * send
                    s2 = self.main.add_synth(self.be.defs[name], out=SEND_BUS, **kw2)
            if gate_end is not None:
                with self.server.at(self.t0 + gate_end):
                    synth.set(gate=0)
                    if s2 is not None:
                        s2.set(gate=0)

    def now_sample(self):
        return int((time.time() - self.t0) * RATE)

    def quit(self):
        try:
            self.server.quit()
        except Exception:
            pass
