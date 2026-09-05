"""Analog-style voices. Each renders ONE note fully at note-on (gate +
release tail) as a mono float32 buffer - the DJ's fx.py pattern of
'synthesise off the deadline, ship a buffer'. The rack mixes buffers.

Every voice: render(pitch_midi, vel, dur_samples, patch, params, rng)
-> np.ndarray float32. patch = style slot dict; params = per-note dict.
"""
from __future__ import annotations

import math

import numpy as np

from lib.gen import RATE
from lib.gen.synth import dsp
from lib.gen.theory import midi_to_hz


def _ones(n, v):
    return np.full(n, float(v), dtype=np.float32)


def _softclip(x, drive=1.0):
    return np.tanh(x * drive).astype(np.float32)


class Voice:
    release = 0.05

    def render(self, pitch, vel, dur, patch, params, rng):
        raise NotImplementedError


class KickVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        decay = float(patch.get("decay", 0.38))     # ~time to silence (like SC's percussive env)
        n = int(decay * 1.6 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        f_end = float(patch.get("pitch", 46.0))
        f_start = f_end * 4.5
        sweep = float(patch.get("sweep", 0.045))
        freq = f_end + (f_start - f_end) * np.exp(-t / sweep)
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        body = np.sin(phase).astype(np.float32)
        amp = dsp.exp_decay(n, decay / 3.5, RATE)
        click_n = int(0.004 * RATE)
        click = np.zeros(n, dtype=np.float32)
        click[:click_n] = (rng.standard_normal(click_n) * np.linspace(1, 0, click_n)).astype(np.float32) * 0.5
        out = _softclip(body * amp * 1.6 + click, 1.2) * vel
        return out


class SnareVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.22 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        tone = np.sin(2 * np.pi * 185.0 * t) * np.exp(-t / 0.045) * 0.6
        noise = rng.standard_normal(n).astype(np.float32)
        noise = dsp.svf(noise, _ones(n, 1800.0), 0.2, RATE, 2) * np.exp(-t / 0.08)
        return ((tone + noise * 0.8) * vel).astype(np.float32)


class ClapVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.28 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        noise = rng.standard_normal(n).astype(np.float32)
        noise = dsp.svf(noise, _ones(n, 1500.0), 0.55, RATE, 1)
        env = np.zeros(n, dtype=np.float32)
        for k, off in enumerate((0.0, 0.011, 0.021, 0.031)):
            i0 = int(off * RATE)
            seg = t[: n - i0]
            env[i0:] = np.maximum(env[i0:], np.exp(-seg / (0.008 if k < 3 else 0.09)))
        return (noise * env * 1.4 * vel).astype(np.float32)


class HatVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        decay = float(params.get("decay", patch.get("decay", 0.05)))
        n = int(decay * 5 * RATE) + 64
        t = np.arange(n, dtype=np.float32) / RATE
        # metallic core: six square waves at inharmonic ratios (808-style)
        core = np.zeros(n, dtype=np.float32)
        for r in (1.0, 1.342, 1.2312, 1.6532, 1.9523, 2.1523):
            core += np.sign(np.sin(2 * np.pi * 320.0 * r * t + 0.3 * r))
        noise = rng.standard_normal(n).astype(np.float32)
        x = (core * 0.15 + noise * 0.7).astype(np.float32)
        x = dsp.svf(x, _ones(n, 7500.0), 0.25, RATE, 2)
        env = np.exp(-t / decay).astype(np.float32)
        return (x * env * 0.9 * vel).astype(np.float32)


class PercVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.16 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        f0 = float(params.get("hz", 330.0 + 120.0 * rng.random()))
        freq = f0 * (1.0 + 0.8 * np.exp(-t / 0.012))
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        tone = np.sin(phase) * np.exp(-t / 0.05)
        tick = rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.004) * 0.4
        return ((tone + tick) * vel).astype(np.float32)


class _Subtractive(Voice):
    """Shared saw/pulse -> SVF -> ADSR engine. Subclasses set the shape."""
    detunes = (0.0,)
    sub = 0.0
    pulse = 0.0
    a, d, s, r = 0.004, 0.12, 0.7, 0.08
    fa, fd, fs, fr = 0.002, 0.15, 0.3, 0.1
    fenv_amt = 2.0
    vibrato = 0.0
    glide = 0.0

    def render(self, pitch, vel, dur, patch, params, rng):
        rel = float(patch.get("release", self.r))
        n = int(dur + rel * RATE * 4) + 32
        hz = midi_to_hz(pitch)
        t = np.arange(n, dtype=np.float32) / RATE
        freq = _ones(n, hz)
        if self.vibrato:
            freq *= (1.0 + self.vibrato * np.sin(2 * np.pi * 5.3 * t) * np.minimum(1.0, t / 0.4)).astype(np.float32)
        if params.get("glide_from"):
            g = int(self.glide * RATE)
            if g > 0:
                f0 = midi_to_hz(params["glide_from"])
                freq[:g] = np.geomspace(f0, hz, g, dtype=np.float32)
        osc = np.zeros(n, dtype=np.float32)
        for cents in self.detunes:
            f = freq * (2.0 ** (cents / 1200.0))
            if self.pulse > 0.0 and rng.random() < self.pulse:
                o, _ = dsp.polyblep_pulse(n, rng.random(), f.astype(np.float32), float(RATE), 0.5 - 0.2 * rng.random())
            else:
                o, _ = dsp.polyblep_saw(n, rng.random(), f.astype(np.float32), float(RATE))
            osc += o
        osc /= len(self.detunes)
        if self.sub > 0.0:
            osc += self.sub * np.sin(2 * np.pi * np.cumsum(freq * 0.5) / RATE).astype(np.float32)
        cutoff = float(params.get("cutoff", patch.get("cutoff", 1200.0)))
        res = float(patch.get("res", 0.25))
        fenv = dsp.adsr(n, int(dur), self.fa, self.fd, self.fs, self.fr, RATE)
        cut = cutoff * (0.35 + 0.65 * vel) * (1.0 + self.fenv_amt * fenv)
        y = dsp.svf(osc.astype(np.float32), cut.astype(np.float32), res, RATE, 0)
        amp = dsp.adsr(n, int(dur), self.a, self.d, self.s, rel, RATE)
        out = (y * amp * (0.4 + 0.6 * vel)).astype(np.float32)
        return _softclip(out, 1.0)


class BassVoice(_Subtractive):
    detunes = (0.0,)
    sub = 0.6
    a, d, s, r = 0.002, 0.1, 0.65, 0.06
    fa, fd, fs, fr = 0.001, 0.09, 0.15, 0.05
    fenv_amt = 2.5
    glide = 0.03


class LeadVoice(_Subtractive):
    detunes = (-7.0, 7.0)
    a, d, s, r = 0.006, 0.18, 0.55, 0.14
    fa, fd, fs, fr = 0.003, 0.2, 0.35, 0.1
    fenv_amt = 1.6
    vibrato = 0.004
    pulse = 0.4


class PadVoice(_Subtractive):
    detunes = (-12.0, -5.0, 5.0, 12.0)
    a, d, s, r = 0.5, 0.6, 0.85, 0.9
    fa, fd, fs, fr = 1.2, 1.5, 0.6, 1.0
    fenv_amt = 1.2


class PluckVoice(_Subtractive):
    detunes = (-4.0, 4.0)
    pulse = 0.5
    a, d, s, r = 0.001, 0.25, 0.0, 0.09
    fa, fd, fs, fr = 0.001, 0.12, 0.0, 0.05
    fenv_amt = 3.0


VOICES = {
    "kick": KickVoice, "snare": SnareVoice, "clap": ClapVoice, "hat": HatVoice,
    "perc": PercVoice, "bass": BassVoice, "lead": LeadVoice, "pad": PadVoice,
    "pluck": PluckVoice,
}
