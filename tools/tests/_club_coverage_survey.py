"""Club COVERAGE survey - which rooms and patterns leave the fan mostly dark.

The user's complaint (2026-10-26): "some of the club patterns are basically
too sparse. It's ok to have stuff come in and out, but a few have a very
small number of pixels lit for extended periods" (spiral hunt named). This
tool measures exactly that, offline, with the same rig as
_club_visual_harness.py (hidden GL context, synthetic 128-BPM track with a
breakdown + drop, the fan's brightness limiter emulated): per frame the
fraction of pixels lit, then over a long window the TIME statistics of
that fraction - mean, the dark tail (10th percentile) and the longest
continuous stretch below the "mostly dark" line.

Two passes:
  rooms      every club state as the director plays it (its preset levels)
  patterns   every pattern ALONE at level 1.0 / club_energy 0.8, so a dark
             room can be blamed on a pattern

Measured on the LED side, the way the render pipeline sends it: the frame
goes through gamma 2.0 (the web panel's default), then the per-receiver
brightness limiter (mean duty capped at 0.1), and a pixel counts as lit by
its LED DUTY. Gamma matters: a shader value of 10/255 is 0.15 % duty (off),
40/255 is 2.5 % (a dim glow in a dark room), 80/255 is 10 %.

Columns:
  cov2    mean fraction of pixels above 2 % LED duty (lit at all)
  cov6    mean fraction above 6 % duty (clearly lit)
  p10     10th percentile of cov2 over time (how dark the dark stretches are)
  dark_s  longest continuous run with cov2 < DARK_COV, in seconds
  lum     mean LED duty (post-limiter, 0..1; the limiter compresses above 0.08)

Writes <out>/rooms.txt, <out>/patterns.txt and, for each entry, a 3-frame
strip (darkest 10% frame | median frame | brightest 10% frame) so a number
can be checked by eye.

Usage:
  python tools/tests/_club_coverage_survey.py [--out DIR] [--rooms] [--patterns]
                                              [--only name ...] [--capture-s N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image, ImageDraw

from tools.tests import _club_visual_harness as H
from tools.tests._club_signals_test import FPS
from projects.fan import weather_params as wp

W, H_ = H.W, H.H
DT = 1.0 / FPS
WARMUP_S = 14.0        # tempo lock + the wrappers' 3 s fade-in
DARK_COV = 0.03        # below this fraction of pixels lit the fan reads as "dark"
GAMMA = 2.0            # web_gamma default, applied before the limiter (render_pipeline)
LIT_DUTY = 0.02        # LED duty that reads as lit in a dark room
CLEAR_DUTY = 0.06


def read_frame_led(divisor_state):
    """Read the FBO and return the LED duty per pixel (HxWx3 float 0..1): gamma
    first, then the app's limiter on the gamma'd frame, like _send_to_displays."""
    data = glReadPixels(0, 0, W, H_, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape(H_, W, 4)[:, :, :3].astype(np.float64) / 255.0
    img = np.flipud(img) ** GAMMA
    normalized = img.mean()
    if normalized <= H.BRIGHTNESS_LIMIT * H.LIMIT_THRESHOLD:
        target = 1.0
    else:
        target = max(1.0, normalized / H.BRIGHTNESS_LIMIT)
    divisor_state[0] = (H.LIMIT_SMOOTHING * target
                        + (1 - H.LIMIT_SMOOTHING) * divisor_state[0])
    if divisor_state[0] > 1.001:
        img = img / divisor_state[0]
    return np.clip(img, 0.0, 1.0)

# wrapper name (as in the harness) -> the level param that turns it on
LEVEL_KEYS = {
    "horizon_glow": "horizon_glow_level", "milk_echo": "milk_echo_level",
    "mandala": "mandala_level", "flow_field": "flow_field_level",
    "wave_dance": "wave_dance_level", "pyramids": "pyramid_level",
    "fibers": "fiber_level", "soul": "soul_level", "tesla": "tesla_level",
    "fountain": "fountain_level", "interference": "interference_level",
    "spiral": "spiral_level", "starfall": "starfall_level",
    "fractal_dive": "fractal_level", "rorschach": "rorschach_level",
    "spirograph": "spirograph_level", "membrane": "membrane_level",
    "turntable": "turntable_level", "stars": "star_field_level",
    "ribbon": "ribbon_level", "tunnel": "tunnel_level",
    "sparkle": "sparkle_level", "eq_bars": "eq_gain", "balls": "orb_level",
    "radar": "radar_sweep_level", "lasers": "laser_density",
    "chaser": "chaser_level", "kick_flare": "kick_flare_level",
    "shockwave": "shockwave_level", "strobe": "beat_flash_amount",
    "urchin": "urchin_level", "predator": "predator_level",
    "crystal": "crystal_level", "digital_flame": "flame_level",
    "singularity": "singularity_level", "spots": "spots_level",
    "club_rain": "rain_level", "voronoi_sphere": "orb_level",   # it reads orb_level, like balls
    "audio_curve": "audiocurve_level",
}


def measure(name, preset, wrappers, timeline, capture_s, out_dir):
    vp = H.StubViewport()
    vp.fbo = _FBO[0]
    renderer = H.StubRenderer(vp)
    out = H.scene_outstate(preset, renderer)
    # a pattern alone: the key may not be in DEFAULT_WEATHER_PARAMS yet
    for k, v in preset.items():
        if k not in out:
            out[k] = v
    states = [{'count': 0, 'elapsed_time': 0.0, 'duration': 1e9} for _ in wrappers]
    divisor = [1.0]
    n_frames = min(len(timeline), int((WARMUP_S + capture_s) / DT))
    covs10, covs40, lums, keep = [], [], [], []
    t = 0.0
    for i in range(n_frames):
        sound, beat, sig, wf, chro = timeline[i]
        H.apply_signals(out, sound, beat, sig, wf, chro)
        for w, (_n, fn, kw) in enumerate(wrappers):
            st = states[w]
            st['elapsed_time'] = t
            fn(st, out, **kw)
            st['count'] += 1
        ordered = sorted(vp.effects, key=lambda e: -getattr(e, 'z_centroid', 0.5))
        glViewport(0, 0, W, H_)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for eff in ordered:
            if eff.enabled:
                eff.update(DT, out)
        for eff in ordered:
            if eff.enabled:
                eff.render(out)
        duty = read_frame_led(divisor)
        if t >= WARMUP_S:
            mx = duty.max(axis=2)
            covs10.append(float((mx > LIT_DUTY).mean()))
            covs40.append(float((mx > CLEAR_DUTY).mean()))
            lums.append(float(duty.mean()))
            # the strip shows the shader frame (duty ** 1/gamma), what the preview shows
            keep.append(np.clip(duty ** (1.0 / GAMMA) * 255.0, 0, 255).astype(np.uint8)
                        if (len(keep) < 4000) else None)
        t += DT
    for eff in vp.effects:
        eff.cleanup()

    c10 = np.array(covs10)
    c40 = np.array(covs40)
    # longest run below DARK_COV
    longest = run = 0
    for v in c10:
        run = run + 1 if v < DARK_COV else 0
        longest = max(longest, run)
    res = dict(name=name, cov10=float(c10.mean()), cov40=float(c40.mean()),
               p10=float(np.percentile(c10, 10)), dark_s=longest * DT,
               lum=float(np.mean(lums)), frames=len(c10))

    # strip: darkest-10% frame | median | brightest-10%
    order = np.argsort(c10)
    picks = [order[max(0, int(0.10 * len(order)) - 1)],
             order[len(order) // 2],
             order[min(len(order) - 1, int(0.90 * len(order)))]]
    scale = 2
    sheet = Image.new("RGB", (3 * (W * scale + 4) + 4, H_ * scale + 30), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for k, idx in enumerate(picks):
        im = Image.fromarray(keep[idx]).resize((W * scale, H_ * scale), Image.NEAREST)
        sheet.paste(im, (4 + k * (W * scale + 4), 26))
        d.text((6 + k * (W * scale + 4), H_ * scale + 28 - 12),
               f"cov {c10[idx]:.3f}", fill=(200, 200, 200))
    d.text((6, 6), f"{name}  cov2 {res["cov10"]:.3f}  p10 {res['p10']:.3f}  "
                   f"dark {res['dark_s']:.1f}s  lum {res['lum']:.3f}   "
                   f"[dark10% | median | bright10%]", fill=(255, 255, 100))
    sheet.save(os.path.join(out_dir, f"{name}.png"))
    return res


def table(rows, title):
    lines = [f"{title}",
             f"{'name':24s} {'cov2':>6s} {'cov6':>6s} {'p10':>6s} {'dark_s':>7s} {'lum':>6s}",
             "-" * 60]
    for r in sorted(rows, key=lambda r: r['cov10']):
        lines.append(f"{r['name']:24s} {r['cov10']:6.3f} {r['cov40']:6.3f} "
                     f"{r['p10']:6.3f} {r['dark_s']:7.1f} {r['lum']:6.3f}")
    return "\n".join(lines)


_FBO = [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_club_survey")
    ap.add_argument("--rooms", action="store_true")
    ap.add_argument("--patterns", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--capture-s", type=float, default=36.0,
                    help="seconds measured after the warm-up (the track is 58 s: "
                         "groove to 30, break to 36, drop, groove to 50)")
    a = ap.parse_args()
    if not a.rooms and not a.patterns:
        a.rooms = a.patterns = True
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only)

    if not glfw.init():
        print("FAIL: glfw.init"); sys.exit(1)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "club-coverage-survey", None, None)
    if win is None:
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, "club-coverage-survey", None, None)
    glfw.make_context_current(win)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    _FBO[0] = H.make_fbo()
    glBindFramebuffer(GL_FRAMEBUFFER, _FBO[0])

    print("Precomputing audio timeline...", flush=True)
    timeline = H.build_signal_timeline()

    if a.patterns:
        rows = []
        for name, fn, kw in H.WRAPPERS:
            if only and name not in only:
                continue
            key = LEVEL_KEYS[name]
            preset = {key: 1.0, "club_energy": 0.8, "club_palette": 0.5}
            if name == "strobe":
                preset.update(strobe_rate=1.0, strobe_brightness=1.0)
            print(f"pattern {name} ({key})...", flush=True)
            rows.append(measure(name, preset, [(name, fn, kw)], timeline,
                                a.capture_s, a.out))
        txt = table(rows, f"PATTERNS alone at level 1.0, club_energy 0.8 "
                          f"({a.capture_s:.0f} s after warm-up; dark = cov2 < {DARK_COV})")
        print("\n" + txt)
        with open(os.path.join(a.out, "patterns.txt"), "w") as f:
            f.write(txt + "\n")

    if a.rooms:
        rows = []
        for s in wp.WEATHER_SETS['club']['states']:
            if only and s not in only:
                continue
            preset = wp.WEATHER_PRESETS[wp.WeatherState(s)]
            print(f"room {s}...", flush=True)
            rows.append(measure(s, preset, H.WRAPPERS, timeline, a.capture_s, a.out))
        txt = table(rows, f"ROOMS as the director plays them "
                          f"({a.capture_s:.0f} s after warm-up; dark = cov2 < {DARK_COV})")
        print("\n" + txt)
        with open(os.path.join(a.out, "rooms.txt"), "w") as f:
            f.write(txt + "\n")

    print(f"\nStrips in {os.path.abspath(a.out)}")
    glfw.terminate()


if __name__ == "__main__":
    main()
