"""Analog-style voices. Each renders ONE note fully at note-on (gate +
release tail) as a float32 buffer - the DJ's fx.py pattern of
'synthesise off the deadline, ship a buffer'. The rack mixes buffers.

Every voice: render(pitch_midi, vel, dur_samples, patch, params, rng)
-> np.ndarray float32, mono (n,) or stereo (n, 2). patch = style slot
dict; params = per-note dict.

Patches override voice parameters: any class attribute listed in
`_Subtractive.TUNABLE` (and the drum voices' named knobs) can be set in
the style's slot dict, so `ambient` and `groove` can share a voice class
and still sound like different instruments.

What keeps these from sounding like a preset bank: every pitched note
gets its own slow pitch drift per oscillator (analog VCOs never sit
still), its own filter wobble, key-tracked and velocity-tracked cutoff,
detuned pairs split left/right, vibrato that arrives late with a rate of
its own, and a filter that can be driven. All of it is drawn from the
rack's seeded rng, so a seed still renders bit-identically.
"""
from __future__ import annotations

import os

import numpy as np

from lib.gen import RATE
from lib.gen.synth import dsp
from lib.gen.theory import midi_to_hz


def _ones(n, v):
    return np.full(n, float(v), dtype=np.float32)


def _softclip(x, drive=1.0):
    return np.tanh(x * drive).astype(np.float32)


def _drift(n, rate_hz, depth, rng):
    """Slow random modulation (n,) in [-depth, depth]."""
    k = int(n * rate_hz / RATE) + 3
    seeds = rng.uniform(-1.0, 1.0, k).astype(np.float32)
    return dsp.smooth_noise(n, float(rate_hz), float(RATE), seeds) * np.float32(depth)


def _noise(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def _bp(x, hz, res, drive=1.0):
    return dsp.svf_tpt(x, _ones(x.shape[0], hz), res, RATE, 1, drive)


def _hp(x, hz, res=0.1):
    return dsp.svf_tpt(x, _ones(x.shape[0], hz), res, RATE, 2, 1.0)


def _lp(x, hz, res=0.1, drive=1.0):
    return dsp.svf_tpt(x, _ones(x.shape[0], hz), res, RATE, 0, drive)


def _stereo(mono_l, mono_r=None):
    return np.stack([mono_l, mono_l if mono_r is None else mono_r], axis=1)


class Voice:
    release = 0.05

    def p(self, patch, name, default):
        return patch.get(name, default)

    def render(self, pitch, vel, dur, patch, params, rng):
        raise NotImplementedError


# -- drums ---------------------------------------------------------------------

class KickVoice(Voice):
    """808-ish: pure sine sweep, soft click, tanh body."""

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


class Kick909Voice(Voice):
    """909-style: longer pitch sweep, body driven into tanh for harmonics,
    a band-passed click and a short noise punch, then a low-pass to keep
    the distortion round rather than fizzy."""

    def render(self, pitch, vel, dur, patch, params, rng):
        decay = float(patch.get("decay", 0.5))
        drive = float(patch.get("drive", 2.6))
        f_end = float(patch.get("pitch", 50.0))
        n = int(decay * 1.5 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        sweep = float(patch.get("sweep", 0.06))
        freq = f_end + (f_end * 3.2 - f_end) * np.exp(-t / sweep)
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        # drive the OSCILLATOR, then the envelope: saturation adds harmonics
        # without flattening the punch-to-tail dynamics (a 909 fill stays a fill)
        body = np.tanh(np.sin(phase) * drive * (0.7 + 0.5 * vel)).astype(np.float32)
        body *= dsp.exp_decay(n, decay / 3.4, RATE)
        click = _noise(n, rng) * np.exp(-t / 0.003).astype(np.float32)
        click = _bp(click, 2500.0 + 800.0 * vel, 0.5) * 1.2
        punch = _noise(n, rng) * np.exp(-t / 0.012).astype(np.float32)
        punch = _lp(punch, 900.0, 0.2) * 0.6
        out = _lp((body * 1.1 + click + punch).astype(np.float32), 5200.0, 0.05)
        return (out * (0.6 + 0.4 * vel)).astype(np.float32)


class SnareVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.22 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        tone = np.sin(2 * np.pi * (185.0 + 6.0 * rng.standard_normal()) * t) * np.exp(-t / 0.045) * 0.6
        noise = _hp(_noise(n, rng), 1800.0 + 200.0 * rng.standard_normal(), 0.2) * np.exp(-t / 0.08)
        return ((tone + noise * 0.8) * (0.5 + 0.5 * vel)).astype(np.float32)


class ClapVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.28 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        noise = _bp(_noise(n, rng), 1500.0 + 150.0 * rng.standard_normal(), 0.55)
        env = np.zeros(n, dtype=np.float32)
        # burst spacing wobbles a little: no two claps are the same hands
        for k, off in enumerate((0.0, 0.011, 0.021, 0.031)):
            i0 = int((off + (0.0015 * rng.random() if k else 0.0)) * RATE)
            seg = t[: n - i0]
            env[i0:] = np.maximum(env[i0:], np.exp(-seg / (0.008 if k < 3 else 0.09)))
        return (noise * env * 1.4 * (0.5 + 0.5 * vel)).astype(np.float32)


class Clap909Voice(Voice):
    """909 clap: three tight noise bursts through a resonant band-pass
    around 1 kHz, then a longer band-limited tail with its own noise so
    the tail does not just repeat the bursts."""

    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.42 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        hz = float(patch.get("hz", 1050.0)) * (1.0 + 0.04 * rng.standard_normal())
        bursts = np.zeros(n, dtype=np.float32)
        for k in range(3):
            i0 = int((k * 0.0105 + 0.001 * rng.random()) * RATE)
            seg = t[: n - i0]
            bursts[i0:] = np.maximum(bursts[i0:], np.exp(-seg / 0.0055))
        head = _bp(_noise(n, rng), hz, 0.7, 1.6) * bursts
        tail_env = np.exp(-np.maximum(t - 0.031, 0.0) / float(patch.get("tail", 0.11))) * (t >= 0.03)
        tail = _bp(_noise(n, rng), hz * 1.15, 0.45) * tail_env.astype(np.float32) * 0.9
        air = _hp(_noise(n, rng), 6000.0, 0.1) * np.exp(-t / 0.05).astype(np.float32) * 0.25
        out = (head * 1.6 + tail + air) * (0.45 + 0.55 * vel)
        return _softclip(out, 1.3)


class HatVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        decay = float(params.get("decay", patch.get("decay", 0.05)))
        decay *= 0.85 + 0.3 * vel                    # harder hits ring longer
        n = int(decay * 5 * RATE) + 64
        t = np.arange(n, dtype=np.float32) / RATE
        # metallic core: six square waves at inharmonic ratios (808-style)
        core = np.zeros(n, dtype=np.float32)
        base = 320.0 * (1.0 + 0.02 * rng.standard_normal())
        for r in (1.0, 1.342, 1.2312, 1.6532, 1.9523, 2.1523):
            core += np.sign(np.sin(2 * np.pi * base * r * t + 0.3 * r))
        x = (core * 0.15 + _noise(n, rng) * 0.7).astype(np.float32)
        x = _hp(x, 7000.0 + 1500.0 * vel, 0.25)
        env = np.exp(-t / decay).astype(np.float32)
        return (x * env * 0.9 * (0.35 + 0.65 * vel)).astype(np.float32)


class RideVoice(Voice):
    """Ride cymbal: inharmonic square partials + noise, high-passed, a
    long wash and a short bright ping on top."""

    def render(self, pitch, vel, dur, patch, params, rng):
        decay = float(patch.get("decay", 0.9)) * (0.8 + 0.4 * vel)
        n = int(decay * 3 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        base = 480.0 * (1.0 + 0.015 * rng.standard_normal())
        core = np.zeros(n, dtype=np.float32)
        for r in (1.0, 1.483, 2.112, 2.641, 3.17, 4.03):
            core += np.sign(np.sin(2 * np.pi * base * r * t + rng.random()))
        wash = _hp((core * 0.12 + _noise(n, rng) * 0.5).astype(np.float32), 4500.0, 0.2)
        wash *= np.exp(-t / decay).astype(np.float32)
        ping = np.sin(2 * np.pi * 3200.0 * t) * np.exp(-t / 0.04) * 0.5
        return ((wash + ping) * (0.35 + 0.65 * vel)).astype(np.float32)


class ShakerVoice(Voice):
    """Shaker: band-passed noise with a swelling attack (the beads move
    before they hit) and a quick decay; accents get a longer, brighter hit."""

    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.11 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        att = 0.006 + 0.008 * (1.0 - vel)
        env = (1.0 - np.exp(-t / att)) * np.exp(-t / (0.028 + 0.03 * vel))
        x = _bp(_noise(n, rng), float(patch.get("hz", 7200.0)) * (1.0 + 0.1 * vel), 0.35)
        return (x * env.astype(np.float32) * 1.3 * (0.35 + 0.65 * vel)).astype(np.float32)


class TomVoice(Voice):
    """Pitched tom: the note pitch is the tom, a short downward sweep
    into it, a little skin noise at the front."""

    def render(self, pitch, vel, dur, patch, params, rng):
        decay = float(patch.get("decay", 0.32))
        n = int(decay * 3 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        f = midi_to_hz(pitch)
        freq = f * (1.0 + 0.7 * np.exp(-t / 0.035))
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        body = np.sin(phase) * np.exp(-t / decay)
        skin = _bp(_noise(n, rng), f * 4.0, 0.4) * np.exp(-t / 0.015) * 0.5
        out = np.tanh((body * 1.3 + skin) * 1.4) * (0.5 + 0.5 * vel)
        return out.astype(np.float32)


class RimVoice(Voice):
    """Rimshot: a resonant click, very short."""

    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.07 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        click = _bp(_noise(n, rng), 1800.0 + 100.0 * rng.standard_normal(), 0.85) * np.exp(-t / 0.012)
        ring = np.sin(2 * np.pi * 820.0 * t) * np.exp(-t / 0.02) * 0.6
        return ((click * 1.8 + ring) * (0.45 + 0.55 * vel)).astype(np.float32)


class PercVoice(Voice):
    def render(self, pitch, vel, dur, patch, params, rng):
        n = int(0.16 * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        f0 = float(params.get("hz", 330.0 + 120.0 * rng.random()))
        freq = f0 * (1.0 + 0.8 * np.exp(-t / 0.012))
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        tone = np.sin(phase) * np.exp(-t / 0.05)
        tick = _noise(n, rng) * np.exp(-t / 0.004) * 0.4
        return ((tone + tick) * (0.4 + 0.6 * vel)).astype(np.float32)


# -- pitched -------------------------------------------------------------------

class _Subtractive(Voice):
    """Shared saw/pulse -> TPT SVF (with drive) -> ADSR engine. Subclasses
    set the shape; a style patch may override any TUNABLE attribute.

    Stereo: with two or more detuned oscillators the odd ones go left and
    the even ones right (with cross-bleed), so a pad or lead is WIDE
    without a chorus in the chain."""
    TUNABLE = ("detunes", "sub", "pulse", "a", "d", "s", "r", "fa", "fd", "fs", "fr", "fenv_amt",
               "vibrato", "vibrato_delay", "glide", "drift", "filter_lfo", "filter_lfo_hz",
               "keytrack", "stereo", "bleed", "tremolo", "drive", "hp")
    detunes = (0.0,)
    sub = 0.0
    pulse = 0.0
    a, d, s, r = 0.004, 0.12, 0.7, 0.08
    fa, fd, fs, fr = 0.002, 0.15, 0.3, 0.1
    fenv_amt = 2.0
    vibrato = 0.0           # depth (fraction of pitch) once it has arrived
    vibrato_delay = 0.25    # seconds before it arrives
    glide = 0.0
    drift = 3.0             # cents of slow random pitch wander per oscillator
    filter_lfo = 0.0        # fraction of cutoff wobbled by a slow random LFO
    filter_lfo_hz = 0.3
    keytrack = 0.3          # cutoff follows pitch: 0 none .. 1 full
    stereo = True
    bleed = 0.35            # how much of each side leaks to the other
    tremolo = 0.0
    drive = 1.0             # filter input drive (>1 saturates)
    hp = 0.0                # high-pass Hz before the filter (0 = off)

    def _cfg(self, patch):
        return {k: patch.get(k, getattr(self, k)) for k in self.TUNABLE}

    def _oscillators(self, n, freq, c, rng):
        detunes = tuple(c["detunes"])
        k = len(detunes)
        wide = c["stereo"] and k >= 2
        bleed = float(c["bleed"])
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)
        for i, cents in enumerate(detunes):
            if c["drift"]:
                cc = cents + _drift(n, 0.6 + 0.5 * rng.random(), float(c["drift"]), rng)
                f = (freq * np.exp2(cc / 1200.0)).astype(np.float32)
            else:
                f = (freq * np.float32(2.0 ** (cents / 1200.0))).astype(np.float32)
            if c["pulse"] > 0.0 and rng.random() < c["pulse"]:
                o, _ = dsp.polyblep_pulse(n, rng.random(), f, float(RATE), 0.5 - 0.2 * rng.random())
            else:
                o, _ = dsp.polyblep_saw(n, rng.random(), f, float(RATE))
            if wide:
                if i % 2 == 0:
                    left += o
                    right += o * bleed
                else:
                    right += o
                    left += o * bleed
            else:
                left += o
        norm = 1.0 / (k * (1.0 + bleed) * 0.5) if wide else 1.0 / k
        left *= norm
        right *= norm
        return left, right, wide

    def render(self, pitch, vel, dur, patch, params, rng):
        c = self._cfg(patch)
        rel = float(patch.get("release", c["r"]))
        n = int(dur + rel * RATE * 4) + 32
        hz = midi_to_hz(pitch)
        t = np.arange(n, dtype=np.float32) / RATE
        freq = _ones(n, hz)
        if c["vibrato"]:
            rate_hz = 4.6 + 1.8 * rng.random()
            onset = np.clip((t - float(c["vibrato_delay"])) / 0.35, 0.0, 1.0)
            freq = (freq * (1.0 + float(c["vibrato"]) * np.sin(2 * np.pi * rate_hz * t + 6.28 * rng.random()) * onset)).astype(np.float32)
        if params.get("glide_from"):
            g = min(int(float(params.get("glide", c["glide"])) * RATE), n)
            if g > 0:
                f0 = midi_to_hz(params["glide_from"])
                freq[:g] = np.geomspace(f0, hz, g, dtype=np.float32)
        left, right, wide = self._oscillators(n, freq, c, rng)
        if c["sub"] > 0.0:
            sub = (float(c["sub"]) * np.sin(2 * np.pi * np.cumsum(freq * 0.5) / RATE)).astype(np.float32)
            left += sub
            if wide:
                right += sub
        cutoff = float(params.get("cutoff", patch.get("cutoff", 1200.0)))
        res = float(patch.get("res", 0.25))
        fenv = dsp.adsr(n, int(dur), c["fa"], c["fd"], c["fs"], c["fr"], RATE)
        track = (hz / 261.6) ** float(c["keytrack"])
        cut = cutoff * track * (0.3 + 0.7 * vel) * (1.0 + float(c["fenv_amt"]) * fenv)
        if c["filter_lfo"]:
            cut = cut * (1.0 + _drift(n, float(c["filter_lfo_hz"]) * (0.7 + 0.6 * rng.random()), float(c["filter_lfo"]), rng))
        cut = cut.astype(np.float32)
        amp = dsp.adsr(n, int(dur), c["a"], c["d"], c["s"], rel, RATE)
        if c["tremolo"]:
            trem = 1.0 - float(c["tremolo"]) * 0.5 * (1.0 - np.cos(2 * np.pi * (3.5 + 2.0 * rng.random()) * t))
            amp = (amp * trem).astype(np.float32)
        gain = np.float32(0.3 + 0.7 * vel)
        drive = float(c["drive"])
        hp = float(c["hp"])
        if hp > 0.0:
            left = _hp(left, hp)
            if wide:
                right = _hp(right, hp)
        yl = dsp.svf_tpt(left, cut, res, RATE, 0, drive) * amp * gain
        if not wide:
            return _softclip(yl, 1.0)
        yr = dsp.svf_tpt(right, cut, res, RATE, 0, drive) * amp * gain
        return np.stack([_softclip(yl, 1.0), _softclip(yr, 1.0)], axis=1)


class BassVoice(_Subtractive):
    detunes = (0.0,)
    sub = 0.6
    a, d, s, r = 0.002, 0.1, 0.65, 0.06
    fa, fd, fs, fr = 0.001, 0.09, 0.15, 0.05
    fenv_amt = 2.5
    glide = 0.06
    drift = 1.5
    keytrack = 0.15
    stereo = False
    drive = 1.6


class LeadVoice(_Subtractive):
    detunes = (-7.0, 7.0)
    a, d, s, r = 0.006, 0.18, 0.55, 0.14
    fa, fd, fs, fr = 0.003, 0.2, 0.35, 0.1
    fenv_amt = 1.6
    vibrato = 0.006
    vibrato_delay = 0.22
    pulse = 0.4
    drift = 4.0
    filter_lfo = 0.12
    filter_lfo_hz = 0.8
    keytrack = 0.4
    bleed = 0.5
    drive = 1.4


class PadVoice(_Subtractive):
    detunes = (-12.0, -5.0, 5.0, 12.0)
    a, d, s, r = 0.5, 0.6, 0.85, 0.9
    fa, fd, fs, fr = 1.2, 1.5, 0.6, 1.0
    fenv_amt = 1.2
    drift = 5.0
    filter_lfo = 0.35
    filter_lfo_hz = 0.15
    keytrack = 0.25
    bleed = 0.25
    hp = 90.0


class SuperSawVoice(_Subtractive):
    """Seven detuned saws spread across the image, high-passed so the
    stack does not fog the bass: the trance / big-room pad and lead."""
    detunes = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0)
    a, d, s, r = 0.02, 0.4, 0.8, 0.5
    fa, fd, fs, fr = 0.01, 0.5, 0.5, 0.5
    fenv_amt = 1.0
    drift = 3.0
    filter_lfo = 0.2
    filter_lfo_hz = 0.2
    keytrack = 0.3
    bleed = 0.2
    hp = 160.0


class PluckVoice(_Subtractive):
    detunes = (-4.0, 4.0)
    pulse = 0.5
    a, d, s, r = 0.001, 0.25, 0.0, 0.09
    fa, fd, fs, fr = 0.001, 0.12, 0.0, 0.05
    fenv_amt = 3.0
    drift = 2.0
    keytrack = 0.5
    bleed = 0.6
    drive = 1.2


class KeysVoice(_Subtractive):
    """Electric-piano-ish stab: two detuned pulses + a soft bell partial,
    mid-length decay, slow tremolo. Distinct from the arp's pluck so chord
    stabs and arpeggios read as two instruments."""
    detunes = (-6.0, 6.0)
    pulse = 1.0
    a, d, s, r = 0.002, 0.45, 0.25, 0.18
    fa, fd, fs, fr = 0.001, 0.3, 0.2, 0.1
    fenv_amt = 1.8
    drift = 2.5
    keytrack = 0.5
    tremolo = 0.25
    bleed = 0.5

    def render(self, pitch, vel, dur, patch, params, rng):
        out = super().render(pitch, vel, dur, patch, params, rng)
        n = out.shape[0]
        t = np.arange(n, dtype=np.float32) / RATE
        bell = (np.sin(2 * np.pi * midi_to_hz(pitch) * 2.0 * t) * np.exp(-t / 0.12)
                * 0.18 * (0.3 + 0.7 * vel)).astype(np.float32)
        if out.ndim == 2:
            out[:, 0] += bell
            out[:, 1] += bell
        else:
            out += bell
        return out


class FMVoice(Voice):
    """2-operator FM. patch: fm_ratio (modulator/carrier, 3.5 = bell,
    2.0 = e-piano, 1.0 = brassy), fm_index (peak modulation index),
    fm_decay (index decay s), a/d/s/r. Velocity drives the index, so soft
    notes are pure and hard notes bite - the classic FM behaviour."""

    def render(self, pitch, vel, dur, patch, params, rng):
        a, d, s, r = (float(patch.get("a", 0.002)), float(patch.get("d", 0.35)),
                      float(patch.get("s", 0.2)), float(patch.get("r", 0.25)))
        n = int(dur + r * RATE * 4) + 32
        hz = midi_to_hz(pitch)
        t = np.arange(n, dtype=np.float32) / RATE
        ratio = float(patch.get("fm_ratio", 3.5)) * (1.0 + 0.002 * rng.standard_normal())
        index = float(patch.get("fm_index", 2.2)) * (0.4 + 0.6 * vel)
        idx_env = (index * (0.15 + 0.85 * np.exp(-t / float(patch.get("fm_decay", 0.3))))).astype(np.float32)
        freq = (_ones(n, hz) * (1.0 + _drift(n, 0.5, 2.0, rng) / 1200.0)).astype(np.float32)
        car = dsp.fm_sine(n, rng.random(), freq, ratio, idx_env, float(RATE))
        amp = dsp.adsr(n, int(dur), a, d, s, r, RATE)
        out = car * amp * (0.3 + 0.7 * vel)
        # a touch of the 2nd harmonic on the right for width
        car2 = dsp.fm_sine(n, rng.random(), freq, ratio * 1.003, idx_env * 0.9, float(RATE)) * amp * (0.3 + 0.7 * vel)
        return _stereo(_softclip(out), _softclip(car2)) * np.float32(0.8)


class SampleVoice(Voice):
    """One-shot sample player. patch: file (path, relative to the repo
    or absolute), base_midi (the pitch the file is at, default 60), and
    optional 'decay' (s) to fade a long file. Missing file -> silence
    (reported once) so a show never stops for an asset."""
    _cache = {}
    _warned = set()

    @classmethod
    def load(cls, path):
        if path in cls._cache:
            return cls._cache[path]
        data = None
        try:
            import soundfile as sf
            x, sr = sf.read(path, dtype="float32", always_2d=True)
            if sr != RATE:
                idx = np.arange(0, x.shape[0], sr / RATE)
                x = np.stack([np.interp(idx, np.arange(x.shape[0]), x[:, ch]) for ch in range(x.shape[1])], axis=1).astype(np.float32)
            data = x if x.shape[1] == 2 else np.repeat(x[:, :1], 2, axis=1)
        except Exception as e:  # noqa: BLE001
            if path not in cls._warned:
                cls._warned.add(path)
                print(f"[GEN] sample {path!r} unavailable ({e.__class__.__name__}); slot is silent")
        cls._cache[path] = data
        return data

    def render(self, pitch, vel, dur, patch, params, rng):
        ref = str(params.get("file", patch.get("file", "")))
        base = patch.get("base_midi")
        bank = patch.get("samples")
        if bank and "file" not in params:              # multisample: the tone nearest the note
            best = min(bank, key=lambda b: abs(int(b.get("base_midi", 60)) - int(round(pitch))))
            ref, base = str(best["file"]), int(best.get("base_midi", 60))
        if ref.startswith("oneshots:"):
            from lib.gen.synth import oneshots
            ref, man_base = oneshots.resolve(ref)
            if base is None:
                base = man_base
        x = self.load(ref or "")
        if x is None:
            return np.zeros((64, 2), dtype=np.float32)
        ratio = 2.0 ** ((float(pitch) - float(base if base is not None else 60)) / 12.0)
        if abs(ratio - 1.0) > 1e-4:
            idx = np.arange(0, x.shape[0] - 1, ratio)
            x = np.stack([np.interp(idx, np.arange(x.shape[0]), x[:, ch]) for ch in range(2)], axis=1).astype(np.float32)
        rate = float(params.get("rate", 1.0) or 1.0)
        if abs(rate - 1.0) > 0.01:                     # constant-pitch time-stretch (vocal phrases at a new tempo)
            try:
                import librosa
                x = np.stack([librosa.effects.time_stretch(np.ascontiguousarray(x[:, ch]), rate=rate) for ch in range(2)], axis=1).astype(np.float32)
            except Exception:  # noqa: BLE001 - no librosa: play it unstretched
                pass
        decay = patch.get("decay")
        if decay:
            t = np.arange(x.shape[0], dtype=np.float32) / RATE
            x = x * np.exp(-t / float(decay))[:, None]
        if dur and patch.get("loop") and x.shape[0] > int(0.5 * RATE) and x.shape[0] < dur + int(0.3 * RATE):
            # a sustained texture held longer than the file: loop its body with crossfades (a drone, a pad)
            xf = min(int(0.1 * RATE), x.shape[0] // 4)
            head = int(0.05 * RATE)
            body = x[head:]                              # skip the attack, loop the sustain
            need = int(dur + 0.3 * RATE)
            out = [x[:head]]
            total = head
            piece = body.copy()
            piece[:xf] *= np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
            piece[-xf:] *= np.linspace(1.0, 0.0, xf, dtype=np.float32)[:, None]
            pos = head
            buf = np.zeros((need + body.shape[0], 2), dtype=np.float32)
            buf[:head] = x[:head]
            while pos < need:
                buf[pos:pos + body.shape[0]] += piece
                pos += body.shape[0] - xf
            x = buf[:need]
        if dur and x.shape[0] > dur + int(0.3 * RATE) and patch.get("samples"):
            n = int(dur + 0.3 * RATE)                  # a bank tone follows the note length
            x = x[:n].copy()
            f = int(0.25 * RATE)
            x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
        return (x * (0.4 + 0.6 * vel)).astype(np.float32)


class KarplusVoice(Voice):
    """Karplus-Strong plucked string (physical model): a noise burst
    circulating in a fractional delay line with damping. patch: decay
    (loop gain 0.9..0.999, default 0.996 -> long ring), brightness
    (0 dull nylon .. 1 bright steel), pick (burst length s), body (a
    little low-pass resonance on the output). Sounds nothing like the
    subtractive pluck: guitar, harp, kalimba territory."""

    def render(self, pitch, vel, dur, patch, params, rng):
        rel = float(patch.get("release", 0.4))
        n = int(dur + rel * RATE) + 32
        hz = midi_to_hz(pitch)
        period = RATE / max(hz, 20.0)
        decay = float(patch.get("decay", 0.996)) ** (100.0 / max(period, 1.0))   # loudness-independent ring time
        decay = min(0.9995, max(0.5, decay * (0.9 + 0.1 * vel)))
        bright = float(patch.get("brightness", 0.5)) * (0.6 + 0.4 * vel)
        pick_n = int(max(0.001, float(patch.get("pick", 0.004))) * RATE)
        pick_n = min(pick_n, int(period) + 2)
        excite = rng.standard_normal(pick_n).astype(np.float32) * np.float32(1.0 - 0.5 * (1.0 - vel))
        excite = _lp(excite, hz * (2.0 + 6.0 * bright), 0.1)      # a softer pick: the fundamental leads
        y = dsp.karplus(n, float(period), decay, bright, excite)
        y = y / max(float(np.abs(y[: int(period) + pick_n + 16]).max()), 1e-6) * np.float32(0.7 + 0.3 * vel)
        gate_env = np.ones(n, dtype=np.float32)
        tail = n - int(dur)
        if tail > 0:
            gate_env[int(dur):] = np.exp(-np.arange(tail, dtype=np.float32) / (rel * RATE * 0.5))
        body = float(patch.get("body", 0.0))
        if body > 0.0:
            y = y * (1.0 - body) + _lp(y, hz * 4.0, 0.35) * body
        return (y * gate_env).astype(np.float32)


class FxVoice(Voice):
    """Transition material, dispatched on params['kind']:
      riser   - noise + rising saw, high-pass opening, swelling over dur
      revcym  - reversed cymbal wash that ends exactly at dur (the drop)
      impact  - sub boom + noise burst with a long tail
      sweep   - downlifter: noise low-pass falling over dur"""

    def render(self, pitch, vel, dur, patch, params, rng):
        kind = params.get("kind", "riser")
        n = max(int(dur), int(0.1 * RATE))
        t = np.arange(n, dtype=np.float32) / RATE
        u = t / max(t[-1], 1e-3)
        if kind == "riser":
            hz = (180.0 * (40.0 ** u)).astype(np.float32)             # 180 Hz -> 7.2 kHz
            nl = dsp.svf_tpt(_noise(n, rng), hz, 0.35, RATE, 2, 1.0)
            nr = dsp.svf_tpt(_noise(n, rng), hz, 0.35, RATE, 2, 1.0)
            f0 = midi_to_hz(pitch)
            f = (f0 * (2.0 ** u)).astype(np.float32)                  # an octave up over the riser
            saw, _ = dsp.polyblep_saw(n, rng.random(), f, float(RATE))
            saw = dsp.svf_tpt(saw, (hz * 2.0).astype(np.float32), 0.3, RATE, 0, 1.0)
            env = (u ** 2.2).astype(np.float32)
            tail = int(0.02 * RATE)
            env[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)
            out = _stereo((nl * 0.7 + saw * 0.35) * env, (nr * 0.7 + saw * 0.35) * env)
        elif kind == "revcym":
            base = 460.0
            core = np.zeros(n, dtype=np.float32)
            for r in (1.0, 1.483, 2.112, 2.641, 3.17):
                core += np.sign(np.sin(2 * np.pi * base * r * t + rng.random()))
            wash = _hp((core * 0.12 + _noise(n, rng) * 0.5).astype(np.float32), 4000.0, 0.2)
            env = (np.exp(-(1.0 - u) * 4.0)).astype(np.float32)
            tail = int(0.005 * RATE)
            env[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)
            out = _stereo(wash * env)
        elif kind == "impact":
            n = int(1.6 * RATE)
            t = np.arange(n, dtype=np.float32) / RATE
            freq = 40.0 + 110.0 * np.exp(-t / 0.08)
            boom = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t / 0.45)
            burst = _lp(_noise(n, rng), 1800.0, 0.2) * np.exp(-t / 0.35) * 0.6
            out = _stereo(np.tanh((boom * 1.4 + burst) * 1.5).astype(np.float32))
        else:  # sweep / downlifter
            hz = (7000.0 * (0.03 ** u)).astype(np.float32)            # 7 kHz -> 210 Hz
            nl = dsp.svf_tpt(_noise(n, rng), hz, 0.3, RATE, 0, 1.0)
            nr = dsp.svf_tpt(_noise(n, rng), hz, 0.3, RATE, 0, 1.0)
            env = (np.exp(-u * 2.5)).astype(np.float32)
            out = _stereo(nl * env, nr * env)
        return (out * np.float32(0.4 + 0.6 * vel)).astype(np.float32)


VOICES = {
    "kick": KickVoice, "kick909": Kick909Voice, "snare": SnareVoice, "clap": ClapVoice,
    "clap909": Clap909Voice, "hat": HatVoice, "ride": RideVoice, "shaker": ShakerVoice,
    "tom": TomVoice, "rim": RimVoice, "perc": PercVoice,
    "bass": BassVoice, "lead": LeadVoice, "pad": PadVoice, "supersaw": SuperSawVoice,
    "pluck": PluckVoice, "keys": KeysVoice, "fm": FMVoice, "sample": SampleVoice, "fx": FxVoice,
    "ks": KarplusVoice,
}
