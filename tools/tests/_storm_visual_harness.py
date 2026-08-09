"""Offline VISUAL harness for the Storm Watch realm - render, look, measure.

Same idea as _wyne_visual_harness.py: drive every storm_* shader in a hidden
GL context against each state's real preset from projects/fan/weather_params.py,
emulate the fan's brightness limiter, and write one annotated contact sheet
per state plus a metrics table. If a state reads as a flat dim wash here, it
reads as a flat dim wash on the hardware.

Storm Watch asks two questions the wyne harness did not, so this one also
answers them directly:

  DISTINCTNESS - a pairwise difference matrix over the fan-view frames.
  Storm has 17 states but only 11 layers, and several states differ by
  nothing but rain_rate and tint. The matrix says which pairs are, in
  fact, the same picture.

  SIGNATURE VISIBILITY - `--drop` re-renders a state with one layer
  removed. The delta is how much that layer actually contributes AFTER
  the limiter has scaled the whole frame. A "signature" layer that moves
  lum by <5% is not a signature; it is drowned by the sky and the rain.

Metrics per frame (post-limiter), same definitions as the wyne harness:
  cov    fraction of pixels lit above 0.08 (features, not the backdrop)
  lum    mean luminance, 0..1 - the limiter's actual output level
  p95    95th-percentile luminance; HIGH p95 with LOW cov is the
         sparse-bright profile the contrast playbook asks for
  sat    mean saturation of the lit pixels
  hues   how many of 12 hue bins the lit pixels occupy

Usage:
  python tools/tests/_storm_visual_harness.py [out_dir] [storm_state ...]
  python tools/tests/_storm_visual_harness.py --drop
Defaults: out_dir = ./_storm_frames, all 17 states + the distinctness matrix.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import math
import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image, ImageDraw

from projects.fan import weather_params as wp

W, H = 128, 300
FPS = 30.0
DT = 1.0 / FPS
WARMUP_S = 9.0             # let fades, phases and eased params settle
N_FRAMES = 4               # frames per contact sheet
FRAME_GAP_S = 1.4          # wall-clock between captured frames
BRIGHTNESS_LIMIT = 0.1     # fan project.yaml
LIMIT_THRESHOLD = 0.8
LIMIT_SMOOTHING = 0.05
UPSCALE = 2

from projects.fan.shaders.storm_sky import shader_storm_sky
from projects.fan.shaders.storm_clouds import shader_storm_clouds
from projects.fan.shaders.storm_color_band import shader_storm_color_band
from projects.fan.shaders.storm_rain import shader_storm_rain
from projects.fan.shaders.storm_rain_glass import shader_storm_rain_glass
from projects.fan.shaders.storm_mist import shader_storm_mist
from projects.fan.shaders.storm_puddle_ripples import shader_storm_puddle_ripples
from projects.fan.shaders.storm_lightning import shader_storm_lightning
from projects.fan.shaders.storm_thunder import shader_storm_thunder
from projects.fan.shaders.storm_moonshaft import shader_storm_moonshaft
from projects.fan.shaders.storm_moonbow import shader_storm_moonbow
from functools import partial
from projects.fan.shaders.stars import shader_stars

# Mirrors the set's "storm_stars" event_map entry: star visibility comes
# from the state's starryness alone, with no day/night gate.
_storm_stars = partial(shader_stars, num_stars=3500, audio_sensitivity=0,
                       drift_x=0.3, ignore_season=True)

# Mirrors WEATHER_SETS["storm_world"]["background_events"]. Order is
# irrelevant - the canvas sorts by z_centroid - but any NEW storm layer must
# be added here or it will silently not render in this tool while working in
# the app (the trap the wyne harness documents).
WRAPPERS = [
    ("storm_sky", shader_storm_sky),
    ("storm_clouds", shader_storm_clouds),
    ("storm_color_band", shader_storm_color_band),
    ("storm_rain", shader_storm_rain),
    ("storm_rain_glass", shader_storm_rain_glass),
    ("storm_mist", shader_storm_mist),
    ("storm_puddle_ripples", shader_storm_puddle_ripples),
    ("storm_lightning", shader_storm_lightning),
    ("storm_thunder", shader_storm_thunder),
    ("storm_moonshaft", shader_storm_moonshaft),
    ("storm_moonbow", shader_storm_moonbow),
    ("stars", _storm_stars),
]

# The storm block of the project output map (projects/fan/weather_params.py).
# Any NEW storm param must be listed here or the harness renders with it at
# its default while the app shows it working.
STORM_PARAMS = ["rain_speed", "rain_angle", "rain_vortex", "rain_grain",
                "rain_color", "cloud_darkness", "cloud_type",
                "window_streaks", "thunder", "moonshaft", "sky_band", "storm_tint",
                "mist_density", "color_band", "puddle_density",
                "moonbow_intensity"]

# Core params lib/weather_state.py publishes under a different name, or
# derives. Handled explicitly in outstate_for_state().
CORE_PASSTHROUGH = ["starryness", "celestial_visibility", "lightning_probability"]

# The layers whose whole reason to exist is to make one state unmistakable.
# `--drop` measures whether they actually do.
SIGNATURES = [
    ("storm_moonbow", "storm_moonbow"),
    ("storm_moonshaft", "storm_easing"),
    ("storm_color_band", "storm_aurora_squall"),
    ("storm_rain_glass", "storm_steady_rain"),
    ("storm_thunder", "storm_rolling_thunder"),
    ("storm_lightning", "storm_distant_lightning"),
    ("storm_mist", "storm_petrichor_mist"),
    ("storm_puddle_ripples", "storm_afterdrip"),
]


class StubCanvas:
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
    def __init__(self, canvas):
        self._c = canvas

    def get_viewport(self, frame_id):
        return self._c


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
    img = np.flipud(img)          # v_uv.y=1 (outer ring) ends up at the top
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


_FAN_MAP = None


def fan_view(img, fw=520, fh=270):
    """Warp an FBO frame into the semicircle the audience actually sees.

    The 128x300 buffer is NOT what the fan looks like: x wraps around a
    semicircle (angle) and y is radial (inner 4ft -> outer 20.6ft). Judging
    shapes on the flat buffer is meaningless - a 'round' blob in buffer space
    is a smeared arc on the hardware.
    """
    global _FAN_MAP
    from renderer.fan_coords import INNER_R_FT, OUTER_R_FT
    if _FAN_MAP is None or _FAN_MAP[0] != (fw, fh):
        px = (np.arange(fw) - fw / 2.0) / (fw / 2.0) * OUTER_R_FT
        py = (fh - 0.5 - np.arange(fh)) / fh * OUTER_R_FT
        fx, fy = np.meshgrid(px, py)
        r = np.hypot(fx, fy)
        th = np.arctan2(fy, fx)
        ok = (r >= INNER_R_FT) & (r <= OUTER_R_FT) & (fy >= 0)
        u = 1.0 - th / np.pi
        v = (r - INNER_R_FT) / (OUTER_R_FT - INNER_R_FT)
        col = np.clip((u * W).astype(int), 0, W - 1)
        # img rows are already flipud'd, so row 0 is v=1 (the outer ring).
        row = np.clip(((1.0 - v) * H).astype(int), 0, H - 1)
        _FAN_MAP = ((fw, fh), row, col, ok)
    _, row, col, ok = _FAN_MAP
    out = np.zeros((fh, fw, 3), np.uint8)
    out[ok] = img[row[ok], col[ok]]
    return out


def metrics(img):
    """Perceptual audit of one post-limiter frame."""
    f = img.astype(np.float64) / 255.0
    lum = f.max(axis=2)                      # value channel
    lit = lum > 0.08                         # features, not the backdrop
    cov = float(lit.mean())
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = np.zeros_like(mx)
    nz = mx > 1e-6
    sat[nz] = (mx[nz] - mn[nz]) / mx[nz]
    if lit.sum() > 0:
        p95 = float(np.percentile(lum[lit], 95))
        msat = float(sat[lit].mean())
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        chroma = mx - mn
        hue_ok = lit & (chroma > 0.06)
        if hue_ok.sum() > 0:
            hue = np.zeros_like(mx)
            idx = (mx == r) & hue_ok
            hue[idx] = ((g[idx] - b[idx]) / chroma[idx]) % 6
            idx = (mx == g) & hue_ok
            hue[idx] = ((b[idx] - r[idx]) / chroma[idx]) + 2
            idx = (mx == b) & hue_ok
            hue[idx] = ((r[idx] - g[idx]) / chroma[idx]) + 4
            bins = np.clip((hue[hue_ok] / 6.0 * 12).astype(int), 0, 11)
            counts = np.bincount(bins, minlength=12)
            nhue = int((counts >= max(3, 0.01 * hue_ok.sum())).sum())
        else:
            nhue = 0
    else:
        p95, msat, nhue = 0.0, 0.0, 0
    return dict(cov=cov, lum=float(lum.mean()), p95=p95, sat=msat, hues=nhue)


def base_outstate(renderer):
    out = {'shader_renderer': renderer, 'frame_id': 0}
    # Seed from the project's own output map so a param whose "off" value
    # isn't zero (rain_grain, rain_speed, cloud_*) starts where the app
    # would start it, not at 0.
    for k in STORM_PARAMS:
        out[k] = wp.OUTSTATE_PUBLISH.get(k, 0.0)
    out['storm_tint'] = [0.26, 0.30, 0.44]
    for k in CORE_PASSTHROUGH:
        out[k] = 0.0
    out['rain'] = 0.0
    out['wind'] = 0.0
    out['fog_strength'] = 0.0
    out['season'] = 0.5
    return out


def outstate_for_state(state_name, renderer):
    """Publish a real preset the way lib/weather_state does for the set."""
    byval = {w.value: w for w in wp.WeatherState}
    preset = wp.WEATHER_PRESETS[byval[state_name]]
    out = base_outstate(renderer)

    def val(key, dflt):
        if key in preset:
            return preset[key]
        if key in wp.DEFAULT_WEATHER_PARAMS:
            return wp.DEFAULT_WEATHER_PARAMS[key]
        return dflt

    for k in STORM_PARAMS:
        out[k] = val(k, out[k])
    for k in CORE_PASSTHROUGH:
        out[k] = val(k, 0.0)

    # Renamed / derived core outputs (lib/weather_state.get_current_params).
    out['rain'] = float(val('rain_rate', 0.0))
    season = float(preset.get('season_preference', 0.5))
    out['season'] = season
    # storm_world declares no season_atmosphere_coupling -> coupling 1.0, so
    # wind is season-modulated and CAN GO NEGATIVE across the cycle.
    out['wind'] = float(val('wind_speed', 0.0)) * math.cos(2 * math.pi * (season - 0.125))
    out['fog_strength'] = max(0.0, float(val('fog', 0.0))
                              * (0.75 - 0.25 * math.cos(2 * math.pi * (season - 0.625))))
    return out


def render_case(label, out_mutator, out_dir, canvas_fbo, skip=None):
    """Render one case. `skip` drops layers by name - one name, or an
    iterable of them. Dropping the backdrop (storm_sky, storm_clouds) is how
    a foreground layer gets measured on its own: cov and p95 for a whole
    storm scene are set by the sky, so a change to the rain is nearly
    invisible in the composite numbers even when the rain changed a lot.
    """
    if skip is None:
        skip = ()
    elif isinstance(skip, str):
        skip = (skip,)
    wrappers = [w for w in WRAPPERS if w[0] not in skip]
    canvas = StubCanvas()
    canvas.fbo = canvas_fbo
    renderer = StubRenderer(canvas)
    out = base_outstate(renderer)
    out_mutator(out)

    states = [{'count': 0, 'elapsed_time': 0.0, 'duration': 1e9, 'frame_id': 0}
              for _ in wrappers]

    divisor = [1.0]
    frames, mets = [], []
    t = 0.0
    next_capture = WARMUP_S
    total = WARMUP_S + FRAME_GAP_S * N_FRAMES

    while len(frames) < N_FRAMES and t < total + 1.0:
        for i, (_, fn) in enumerate(wrappers):
            st = states[i]
            fn(st, out)
            st['count'] += 1
            st['elapsed_time'] = t

        glBindFramebuffer(GL_FRAMEBUFFER, canvas_fbo)
        glViewport(0, 0, W, H)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for eff in canvas.effects:
            eff.update(DT, out)
        # Canvas pass: back-to-front by z_centroid, same as GroupCanvas.
        for eff in sorted(canvas.effects,
                          key=lambda e: -getattr(e, 'z_centroid', 0.5)):
            eff.render(out)

        if t >= next_capture:
            img = read_frame(divisor)
            frames.append(img)
            mets.append(metrics(img))
            next_capture += FRAME_GAP_S
        else:
            read_frame(divisor)      # keep the limiter's smoothing running

        t += DT

    for eff in canvas.effects:
        eff.cleanup()

    if out_dir is not None:
        # ---- contact sheet: raw buffer strip on top, FAN VIEW below ----
        pad, top = 6, 16
        fans = [fan_view(f) for f in frames]
        fw, fh = fans[0].shape[1], fans[0].shape[0]
        strip_w = (W * UPSCALE + pad) * len(frames) + pad
        fan_row_w = (fw + pad) * len(fans) + pad
        sheet_w = max(strip_w, fan_row_w)
        sheet_h = H * UPSCALE + fh + top + pad * 2
        sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 18))
        for i, fr in enumerate(frames):
            im = Image.fromarray(fr).resize((W * UPSCALE, H * UPSCALE), Image.NEAREST)
            sheet.paste(im, (pad + i * (W * UPSCALE + pad), top))
        for i, fv in enumerate(fans):
            sheet.paste(Image.fromarray(fv),
                        (pad + i * (fw + pad), top + H * UPSCALE + pad))
        d = ImageDraw.Draw(sheet)
        m = mets[len(mets) // 2]
        d.text((pad, 3), f"{label}   cov {m['cov']:.3f}  lum {m['lum']:.3f}  "
                         f"p95 {m['p95']:.2f}  sat {m['sat']:.2f}  hues {m['hues']}",
               fill=(220, 220, 210))
        sheet.save(os.path.join(out_dir, f"{label}.png"))

    avg = {k: float(np.mean([mm[k] for mm in mets])) for k in mets[0]}
    # Median fan-view frame, kept for the pairwise distinctness matrix.
    avg['_frame'] = fan_view(frames[len(frames) // 2]).astype(np.float64) / 255.0
    return avg


def mutator_for(state_name):
    return lambda o: o.update(
        {k: v for k, v in outstate_for_state(state_name, o['shader_renderer']).items()
         if k != 'shader_renderer'})


def open_context(title):
    if not glfw.init():
        print("FAIL: glfw.init()")
        sys.exit(1)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, title, None, None)
    if win is None:
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, title, None, None)
    if win is None:
        print("FAIL: could not create a hidden GL context")
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(win)
    print(f"GL context: {glGetString(GL_VERSION).decode()}\n")


def run_drop_mode():
    """Signature-visibility audit: how much does each signature layer
    actually contribute to its own state, after the limiter?"""
    open_context("storm-drop")
    fbo = make_fbo()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    print(f"{'layer':<22} {'state':<24} {'lum with':>9} {'lum w/o':>9} "
          f"{'d lum %':>8} {'d pixels %':>11}")
    print("-" * 88)
    rows = []
    for layer, state in SIGNATURES:
        full = render_case(f"{state}", mutator_for(state), None, fbo)
        less = render_case(f"{state}", mutator_for(state), None, fbo, skip=layer)
        dl = (full['lum'] - less['lum']) / max(full['lum'], 1e-9) * 100.0
        diff = np.abs(full['_frame'] - less['_frame']).max(axis=2)
        dp = float((diff > 0.02).mean()) * 100.0
        rows.append((layer, state, dl, dp))
        print(f"{layer:<22} {state:<24} {full['lum']:>9.4f} {less['lum']:>9.4f} "
              f"{dl:>8.1f} {dp:>11.1f}")

    glfw.terminate()
    print()
    weak = [f"{l} ({s})" for l, s, dl, dp in rows if dp < 5.0]
    if weak:
        print(f"WARN barely present (changes < 5% of fan pixels): {', '.join(weak)}")
    else:
        print("every signature layer moves >5% of the fan's pixels in its own state")


def main():
    args = [a for a in sys.argv[1:]]
    if "--drop" in args:
        run_drop_mode()
        return

    out_dir = args[0] if args and not args[0].startswith("storm_") else "_storm_frames"
    picked = [a for a in args if a.startswith("storm_")]
    os.makedirs(out_dir, exist_ok=True)

    open_context("storm-harness")
    fbo = make_fbo()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    states = picked or list(wp.WEATHER_SETS["storm_world"]["states"])

    rows = []
    print(f"{'case':<26} {'cov':>6} {'lum':>6} {'p95':>6} {'sat':>6} {'hues':>5}")
    print("-" * 60)
    for s in states:
        avg = render_case(s, mutator_for(s), out_dir, fbo)
        rows.append((s, avg))
        print(f"{s:<26} {avg['cov']:>6.3f} {avg['lum']:>6.3f} "
              f"{avg['p95']:>6.2f} {avg['sat']:>6.2f} {avg['hues']:>5.0f}")

    glfw.terminate()

    # ---- distinctness: mean absolute difference between fan views ----
    print()
    print("closest state pairs (mean abs fan-view difference, 0 = identical picture)")
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = float(np.abs(rows[i][1]['_frame'] - rows[j][1]['_frame']).mean())
            pairs.append((d, rows[i][0], rows[j][0]))
    pairs.sort()
    for d, a, b in pairs[:12]:
        flag = "  <-- near-identical" if d < 0.010 else ""
        print(f"  {d:.4f}  {a:<24} vs {b}{flag}")

    print()
    scenes = [(n, a) for n, a in rows]
    dark = [n for n, a in scenes if a['lum'] < 0.006]
    washy = [n for n, a in scenes if a['cov'] > 0.55 and a['p95'] < 0.45]
    flat = [n for n, a in scenes if a['hues'] <= 1 and a['cov'] > 0.05]
    if dark:
        print(f"WARN near-black (lum < 0.006): {', '.join(dark)}")
    if washy:
        print(f"WARN wash-like (cov > 0.55 and p95 < 0.45): {', '.join(washy)}")
    if flat:
        print(f"WARN single-hue (hues <= 1): {', '.join(flat)}")
    if not (dark or washy or flat):
        print("no near-black / wash-like / single-hue states")
    print(f"\nwrote {len(states)} contact sheets to {out_dir}/")


if __name__ == "__main__":
    main()
