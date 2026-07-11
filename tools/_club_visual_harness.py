"""Offline VISUAL harness for the club scenes - render, look, measure.

Drives every club pattern shader in a hidden GL context against the
synthetic 128-BPM track (same analyzer pipeline the app uses), emulates the
fan's brightness limiter, and writes per-scene contact sheets: 8 frames
spanning one beat cycle, upscaled, annotated with reactivity metrics.
This is how club visuals get judged and tuned WITHOUT a dance floor:
if the on-beat frame doesn't look dramatically different from the off-beat
frame here, it won't on the fan either.

Metrics per scene (printed as a table):
  cov    fraction of pixels lit (>0.04) on the beat frame
  lum+   mean luminance of the beat frame (post-limiter, 0..1)
  beat%  luminance ratio beat/off-beat (temporal brightness contrast)
  mot    mean |frame diff| across the beat cycle (structural motion)

Usage:
  python tools/_club_visual_harness.py [out_dir] [scene ...]
Defaults: out_dir = ./_club_frames, all ten scenes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image, ImageDraw

from lib.beat_detector import BeatDetector
from lib.audio_signals import AudioStructure
from tools._club_signals_test import (synth_track, band_powers_for_window,
                                      RATE, FPS, CHUNK, HOP, WIN_SHORT,
                                      WIN_LONG, BEAT_S, hann)

W, H = 128, 300
DT = 1.0 / FPS
WARMUP_S = 14.0            # tempo lock + scene fade-in
CAPTURE_BEATS = 2          # capture window after warmup
BRIGHTNESS_LIMIT = 0.1     # fan project.yaml
LIMIT_THRESHOLD = 0.8
LIMIT_SMOOTHING = 0.05

# scene -> preset params come straight from the project file
from projects.fan import weather_params as wp

# All club pattern events, matching the set's background_events (order is
# irrelevant here - the canvas sorts by z_centroid).
from projects.fan.shaders.club_horizon_glow import shader_club_horizon_glow
from projects.fan.shaders.club_milk_echo import shader_club_milk_echo
from projects.fan.shaders.club_mandala import shader_club_mandala
from projects.fan.shaders.club_flow_field import shader_club_flow_field
from projects.fan.shaders.club_wave_dance import shader_club_wave_dance
from projects.fan.shaders.club_pyramids import shader_club_pyramids
from projects.fan.shaders.club_fibers import shader_club_fibers
from projects.fan.shaders.club_airhandler import shader_club_airhandler
from projects.fan.shaders.club_soul import shader_club_soul
from projects.fan.shaders.club_tesla import shader_club_tesla
from projects.fan.shaders.club_fountain import shader_club_fountain
from projects.fan.shaders.club_interference import shader_club_interference
from projects.fan.shaders.club_spiral import shader_club_spiral
from projects.fan.shaders.club_starfall import shader_club_starfall
from projects.fan.shaders.club_fractal_dive import shader_club_fractal_dive
from projects.fan.shaders.club_rorschach import shader_club_rorschach
from projects.fan.shaders.club_spirograph import shader_club_spirograph
from projects.fan.shaders.club_membrane import shader_club_membrane
from projects.fan.shaders.club_turntable import shader_club_turntable
from projects.fan.shaders.club_spectrum_stars import shader_club_spectrum_stars
from projects.fan.shaders.club_ribbon import shader_club_ribbon
from projects.fan.shaders.club_tunnel import shader_club_tunnel
from projects.fan.shaders.club_sparkle import shader_club_sparkle
from projects.fan.shaders.club_eq_bars import shader_club_eq_bars
from projects.fan.shaders.club_radar_sweep import shader_club_radar_sweep
from projects.fan.shaders.club_lasers import shader_club_lasers
from projects.fan.shaders.club_chaser import shader_club_chaser
from projects.fan.shaders.club_kick_flare import shader_club_kick_flare
from projects.fan.shaders.club_shockwave import shader_club_shockwave
from projects.fan.shaders.club_strobe import shader_club_strobe
from projects.fan.shaders.audio_balls import shader_audio_balls

WRAPPERS = [
    ("horizon_glow", shader_club_horizon_glow, {}),
    ("milk_echo", shader_club_milk_echo, {}),
    ("mandala", shader_club_mandala, {}),
    ("flow_field", shader_club_flow_field, {}),
    ("wave_dance", shader_club_wave_dance, {}),
    ("pyramids", shader_club_pyramids, {}),
    ("fibers", shader_club_fibers, {}),
    ("airhandler", shader_club_airhandler, {}),
    ("soul", shader_club_soul, {}),
    ("tesla", shader_club_tesla, {}),
    ("fountain", shader_club_fountain, {}),
    ("interference", shader_club_interference, {}),
    ("spiral", shader_club_spiral, {}),
    ("starfall", shader_club_starfall, {}),
    ("fractal_dive", shader_club_fractal_dive, {}),
    ("rorschach", shader_club_rorschach, {}),
    ("spirograph", shader_club_spirograph, {}),
    ("membrane", shader_club_membrane, {}),
    ("turntable", shader_club_turntable, {}),
    ("stars", shader_club_spectrum_stars, {}),
    ("ribbon", shader_club_ribbon, {}),
    ("tunnel", shader_club_tunnel, {}),
    ("sparkle", shader_club_sparkle, {}),
    ("eq_bars", shader_club_eq_bars, {}),
    ("balls", shader_audio_balls, {"mixer_op_key": "orb_level",
                                   "beat_pulse": True,
                                   "lightning_threshold": 1.3,
                                   "lightning_probability": 0.35}),
    ("radar", shader_club_radar_sweep, {}),
    ("lasers", shader_club_lasers, {}),
    ("chaser", shader_club_chaser, {}),
    ("kick_flare", shader_club_kick_flare, {}),
    ("shockwave", shader_club_shockwave, {}),
    ("strobe", shader_club_strobe, {}),
]


class StubViewport:
    def __init__(self):
        self.width = W
        self.height = H
        self.effects = []

    def add_effect(self, effect_class, **params):
        eff = effect_class(self, **params)
        eff.init()
        self.effects.append(eff)
        return eff


class StubRenderer:
    def __init__(self, vp):
        self._vp = vp

    def get_viewport(self, frame_id):
        return self._vp


def build_signal_timeline():
    """Precompute per-frame sound dicts + beat/structure signals once."""
    samples = synth_track()
    bd = BeatDetector()
    st = AudioStructure()
    prev_bp = None
    recent = []
    frames = []
    wave_peak = 1e-6
    chroma_freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
    chroma_mask = (chroma_freqs >= 80.0) & (chroma_freqs <= 5000.0)
    chroma_pcs = (np.round(12.0 * np.log2(chroma_freqs[chroma_mask] / 440.0)) % 12).astype(np.int64)
    pos = 0
    while pos + CHUNK <= len(samples):
        win = samples[pos:pos + CHUNK]
        bp = band_powers_for_window(win)
        if prev_bp is None:
            prev_bp = bp.copy()
        else:
            bp = 0.95 * bp + 0.05 * prev_bp
            prev_bp = bp.copy()
        recent.insert(0, bp)
        if len(recent) > WIN_LONG:
            recent.pop()
        mean_short = np.mean(recent[:WIN_SHORT], axis=0)
        mean_long = np.mean(recent, axis=0)
        mean_short = np.where(mean_short < 1e-10, 1e-10, mean_short)
        mean_long = np.where(mean_long < 1e-10, 1e-10, mean_long)
        sound = {
            "raw_bands": np.array([bp]),
            "norm_short": np.array([bp / mean_short]),
            "norm_long": np.array([bp / mean_long]),
            "norm_long_relu": np.array([np.maximum(0.0, bp / mean_long - 1.0)]),
        }
        # 12-bin chroma fold of this window (mirrors analyzer.get_chroma).
        mags = np.abs(np.fft.rfft(win * hann))
        cw_power = mags[chroma_mask] ** 2
        chro = np.bincount(chroma_pcs, weights=cw_power, minlength=12)
        ctot = float(chro.sum())
        chro = (chro / ctot).astype(np.float32) if ctot > 1e-9 else np.zeros(12, np.float32)
        beat = bd.update(sound, DT)
        sig = st.update(sound, beat, DT)
        # AGC-normalized time-domain waveform slice (mirrors
        # MicrophoneAnalyzer.get_waveform).
        wtail = win[-2048:]
        wv = wtail[:16 * 128].reshape(128, 16).mean(axis=1)
        peak = max(abs(float(wv.max())), abs(float(wv.min())), 1e-6)
        wave_peak = max(wave_peak * 0.995, peak)
        wf = np.clip(wv / wave_peak, -1.0, 1.0).astype(np.float32) \
            if wave_peak > 1e-4 else np.zeros(128, dtype=np.float32)
        frames.append((sound, beat, sig, wf, chro))
        pos += HOP
    return frames


def scene_outstate(preset, renderer):
    """Static (non-signal) outstate keys for a scene."""
    o = {"shader_renderer": renderer}
    keys = ["club_energy", "club_palette", "eq_gain", "eq_color_mode",
            "laser_density", "beat_flash_amount", "strobe_rate",
            "strobe_brightness", "orb_level", "tunnel_level",
            "star_field_level", "shockwave_level", "kick_flare_level",
            "sparkle_level", "chaser_level", "ribbon_level",
            "horizon_glow_level", "radar_sweep_level", "milk_echo_level",
            "mandala_level", "flow_field_level", "wave_dance_level",
            "pyramid_level",
            "fiber_level",
            "airhandler_level",
            "soul_level",
            "tesla_level",
            "fountain_level",
            "interference_level",
            "spiral_level",
            "membrane_level",
            "turntable_level",
            "starfall_level",
            "fractal_level",
            "rorschach_level",
            "spirograph_level"]
    for k in keys:
        o[k] = preset.get(k, wp.DEFAULT_WEATHER_PARAMS.get(k, 0.0))
    return o


def apply_signals(o, sound, beat, sig, wf, chro):
    o["sound"] = sound
    o["waveform"] = wf
    o["chroma"] = chro
    o["beat"] = beat["onset"]
    o["beat_decay"] = beat["decay"]
    o["bpm"] = beat["bpm"]
    o["beat_phase"] = beat["phase"]
    o["beat_intensity"] = beat["strength"]
    o["beat_confidence"] = beat["confidence"]
    o["beat_count"] = beat["count"]
    o["bar_phase"] = beat["bar_phase"]
    o["phrase_phase"] = beat["phrase_phase"]
    o["audio_bass"] = sig["bass"]
    o["audio_mid"] = sig["mid"]
    o["audio_high"] = sig["high"]
    o["bass_punch"] = sig["bass_punch"]
    o["mid_punch"] = sig["mid_punch"]
    o["high_punch"] = sig["high_punch"]
    o["audio_punch"] = sig["punch"]
    o["audio_energy"] = sig["energy"]
    o["build_level"] = sig["build"]
    o["drop"] = sig["drop"]
    o["drop_decay"] = sig["drop_decay"]


def make_fbo():
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    color = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0)
    depth = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, W, H)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth)
    assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
    return fbo


def read_frame(divisor_state):
    """Read the FBO, apply the app's brightness limiter, return HxWx3 u8."""
    data = glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].astype(np.float64)
    img = np.flipud(img)   # v_uv.y=1 (outer ring) at the top
    normalized = img.sum() / (H * W * 255.0 * 3.0)
    if normalized <= BRIGHTNESS_LIMIT * LIMIT_THRESHOLD:
        target = 1.0
    else:
        target = max(1.0, normalized / BRIGHTNESS_LIMIT)
    divisor_state[0] = (LIMIT_SMOOTHING * target
                        + (1 - LIMIT_SMOOTHING) * divisor_state[0])
    if divisor_state[0] > 1.001:
        img = img / divisor_state[0]
    return np.clip(img, 0, 255).astype(np.uint8)


def render_scene(scene_name, preset, timeline, out_dir, canvas_fbo=0):
    vp = StubViewport()
    vp.fbo = canvas_fbo          # effects with private FBOs rebind this
    renderer = StubRenderer(vp)
    out = scene_outstate(preset, renderer)
    states = [{'count': 0, 'elapsed_time': 0.0, 'duration': 1e9} for _ in WRAPPERS]
    fbo = glGenFramebuffers(0) if False else None  # placeholder, fbo made by caller

    divisor = [1.0]
    frames_out = []       # (t, img) captured frames
    beat_frames = []      # frame indices where a beat fired (post-warmup)

    n_frames = min(len(timeline), int((WARMUP_S + CAPTURE_BEATS * BEAT_S + 1.0) / DT))
    t = 0.0
    for i in range(n_frames):
        sound, beat, sig, wf, chro = timeline[i]
        apply_signals(out, sound, beat, sig, wf, chro)
        # Wrapper pass (mirrors EventScheduler): sets fields, spawns things.
        for w, (name, fn, kw) in enumerate(WRAPPERS):
            st = states[w]
            st['elapsed_time'] = t
            fn(st, out, **kw)
            st['count'] += 1
        # Canvas pass: back-to-front by z_centroid.
        ordered = sorted(vp.effects, key=lambda e: -getattr(e, 'z_centroid', 0.5))
        glViewport(0, 0, W, H)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for eff in ordered:
            if eff.enabled:
                eff.update(DT, out)
        for eff in ordered:
            if eff.enabled:
                eff.render(out)
        if t >= WARMUP_S:
            img = read_frame(divisor)
            frames_out.append((t, img, bool(out["beat"])))
            if out["beat"]:
                beat_frames.append(len(frames_out) - 1)
        else:
            read_frame(divisor)   # keep the limiter state warm
        t += DT

    # cleanup GL objects so scenes don't leak into each other
    for eff in vp.effects:
        eff.cleanup()

    # ---- pick frames + metrics ----
    if len(beat_frames) < 2:
        print(f"  !! {scene_name}: no beats captured")
        return None
    b0 = beat_frames[0]
    period = beat_frames[1] - b0
    # 8 frames evenly across one beat cycle starting 1 frame after the beat
    idxs = [min(b0 + 1 + int(k * period / 8.0), len(frames_out) - 1) for k in range(8)]
    sel = [frames_out[i][1] for i in idxs]

    on_beat = frames_out[min(b0 + 1, len(frames_out) - 1)][1].astype(np.float64)
    lum_on = on_beat.mean() / 255.0
    cov = float((on_beat.max(axis=2) > 10).mean())
    # Beat contrast averaged over EVERY captured beat (a single on/off pair
    # is sampling noise for sparse stochastic patterns like meteor showers):
    # mean luminance of the 2 frames after each beat vs the 2 mid-beat frames.
    lums = np.array([f[1].astype(np.float64).mean() / 255.0 for f in frames_out])
    on_idx, off_idx = [], []
    for b in beat_frames:
        for k in (1, 2):
            if b + k < len(lums):
                on_idx.append(b + k)
        for k in (period // 2, period // 2 + 1):
            if b + k < len(lums):
                off_idx.append(b + k)
    beat_ratio = (float(lums[on_idx].mean()) / max(float(lums[off_idx].mean()), 1e-6)
                  if on_idx and off_idx else 1.0)
    diffs = [np.abs(sel[k].astype(np.float64) - sel[k - 1].astype(np.float64)).mean()
             for k in range(1, len(sel))]
    motion = float(np.mean(diffs)) / 255.0

    # ---- contact sheet: 8 frames across one beat, 2x upscale ----
    scale = 2
    sheet = Image.new("RGB", (8 * (W * scale + 4) + 4, H * scale + 30), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for k, img in enumerate(sel):
        im = Image.fromarray(img).resize((W * scale, H * scale), Image.NEAREST)
        sheet.paste(im, (4 + k * (W * scale + 4), 26))
    d.text((6, 6), f"{scene_name}   cov={cov:.2f} lum={lum_on:.3f} "
                   f"beat x{beat_ratio:.2f} motion={motion:.4f}   "
                   f"(8 frames over one beat, post-limiter)", fill=(255, 255, 100))
    path = os.path.join(out_dir, f"{scene_name}.png")
    sheet.save(path)
    return dict(scene=scene_name, cov=cov, lum=lum_on, ratio=beat_ratio,
                motion=motion, path=path)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "_club_frames"
    os.makedirs(out_dir, exist_ok=True)
    only = set(sys.argv[2:])

    if not glfw.init():
        print("FAIL: glfw.init"); sys.exit(1)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "club-visual-harness", None, None)
    if win is None:
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, "club-visual-harness", None, None)
    glfw.make_context_current(win)

    # Global GL state, mirroring create_window().
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    fbo = make_fbo()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    print("Precomputing audio timeline...")
    timeline = build_signal_timeline()

    scenes = wp.WEATHER_SETS['club']['states']
    results = []
    for s in scenes:
        if only and s not in only:
            continue
        preset = wp.WEATHER_PRESETS[wp.WeatherState(s)]
        print(f"Rendering {s}...")
        r = render_scene(s, preset, timeline, out_dir, canvas_fbo=fbo)
        if r:
            results.append(r)

    print(f"\n{'scene':22s} {'cov':>5s} {'lum':>6s} {'beatX':>6s} {'motion':>7s}")
    print("-" * 52)
    for r in results:
        print(f"{r['scene']:22s} {r['cov']:5.2f} {r['lum']:6.3f} "
              f"{r['ratio']:6.2f} {r['motion']:7.4f}")
    print(f"\nSheets in {os.path.abspath(out_dir)}")
    glfw.terminate()


if __name__ == "__main__":
    main()
