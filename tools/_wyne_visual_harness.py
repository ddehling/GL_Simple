"""Offline VISUAL harness for the FyneWynz (wine) realm - render, look, measure.

Drives every wyne_* shader in a hidden GL context against each state's real
preset from projects/fan/weather_params.py, emulates the fan's brightness
limiter, and writes one annotated contact sheet per state plus a metrics
table. This is how the realm gets judged WITHOUT the fan: if a state reads
as a flat dim wash here, it will read as a flat dim wash on the hardware.

It also renders a NARRATIVE block - each of the six story_* variables at
1.0 over a neutral base - so the three companions' light and dark faces can
be checked for distinctness, which is the whole point of authoring them as
opponent hues.

Metrics per frame (post-limiter):
  cov    fraction of pixels lit above 0.08 (features, not the backdrop)
  lum    mean luminance, 0..1 - the limiter's actual output level
  p95    95th-percentile luminance; HIGH p95 with LOW cov is the
         sparse-bright profile the contrast playbook asks for
  sat    mean saturation of the lit pixels (playbook lever 3)
  hues   how many of 12 hue bins the lit pixels occupy (playbook lever 1)

Usage:
  python tools/_wyne_visual_harness.py [out_dir] [state ...]
Defaults: out_dir = ./_wyne_frames, all 12 states + the narrative block.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from projects.fan.shaders.wyne_sky import shader_wyne_sky
from projects.fan.shaders.wyne_hillside import shader_wyne_hillside
from projects.fan.shaders.wyne_terroir import shader_wyne_terroir
from projects.fan.shaders.wyne_clusters import shader_wyne_clusters
from projects.fan.shaders.wyne_cellar import shader_wyne_cellar
from projects.fan.shaders.wyne_ferment import shader_wyne_ferment
from projects.fan.shaders.wyne_oak import shader_wyne_oak
from projects.fan.shaders.wyne_aroma import shader_wyne_aroma
from projects.fan.shaders.wyne_goblet import shader_wyne_goblet
from projects.fan.shaders.wyne_lees import shader_wyne_lees
from projects.fan.shaders.wyne_taint import shader_wyne_taint

# Mirrors the set's background_events (minus the engine `stars` event, which
# is not a project shader). Order is irrelevant - the canvas sorts by
# z_centroid - but any NEW wyne layer must be added here or it will silently
# not render in this tool while working in the app.
WRAPPERS = [
    ("sky", shader_wyne_sky),
    ("hillside", shader_wyne_hillside),
    ("terroir", shader_wyne_terroir),
    ("clusters", shader_wyne_clusters),
    ("cellar", shader_wyne_cellar),
    ("ferment", shader_wyne_ferment),
    ("oak", shader_wyne_oak),
    ("aroma", shader_wyne_aroma),
    ("goblet", shader_wyne_goblet),
    ("lees", shader_wyne_lees),
    ("taint", shader_wyne_taint),
]

# Any NEW wyne param must be listed here or the harness silently renders
# with it at zero while the app shows it working - the same trap as the
# WRAPPERS list above. bottle_rack was missing and the bottles simply
# never appeared in these contact sheets.
WYNE_PARAMS = ["wyne_tint", "must_hue", "row_density", "ripeness",
               "cluster_load", "rot_bloom", "cellar_depth",
               "ferment_activity", "aroma", "pour_flow", "lees_density",
               "taint_level", "bottle_rack"]

STORY_VARS = ["vine_light", "vine_dark", "cask_light", "cask_dark",
              "glass_light", "glass_dark"]


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
    object shapes on the flat buffer is meaningless - a 'tall' shape in
    buffer space is a radial wedge on the hardware. Inverse-maps each
    output pixel to (r, theta) -> uv and samples nearest.
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
    # 0.08, not 0.04: wyne_sky is now a full-arch backdrop that sits
    # deliberately dim across the whole annulus, and a 0.04 floor
    # counted that backdrop as 'lit', inflating cov everywhere and
    # tripping the wash heuristic on scenes that are actually fine.
    # This threshold measures FEATURES, not the presence of a sky.
    lit = lum > 0.08
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
        # Hue only where there is enough chroma to have one.
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
            # Count a bin only if it holds >=1% of the coloured pixels, so a
            # handful of stray pixels doesn't inflate the hue count.
            counts = np.bincount(bins, minlength=12)
            nhue = int((counts >= max(3, 0.01 * hue_ok.sum())).sum())
        else:
            nhue = 0
    else:
        p95, msat, nhue = 0.0, 0.0, 0
    return dict(cov=cov, lum=float(lum.mean()), p95=p95, sat=msat, hues=nhue)


def base_outstate(renderer):
    out = {'shader_renderer': renderer, 'frame_id': 0}
    for k in WYNE_PARAMS:
        out[k] = 0.0
    out['wyne_tint'] = [0.5, 0.4, 0.45]
    for v in STORY_VARS:
        out['story_' + v] = 0.0
    out['wind'] = 0.0
    # The engine clock every outdoor layer lights itself by.
    out['season'] = 0.5
    return out


def outstate_for_state(state_name, renderer):
    """Publish a real preset the way lib/weather_state does for the set."""
    byval = {w.value: w for w in wp.WeatherState}
    preset = wp.WEATHER_PRESETS[byval[state_name]]
    out = base_outstate(renderer)
    for k in WYNE_PARAMS:
        if k in preset:
            out[k] = preset[k]
        elif k in wp.DEFAULT_WEATHER_PARAMS:
            out[k] = wp.DEFAULT_WEATHER_PARAMS[k]
    # outstate publishes the derived signed 'wind', never 'wind_speed'.
    out['wind'] = float(preset.get('wind_speed', 0.0))
    # season_preference is the state's preferred point on the engine's
    # day/night clock; the engine eases toward it, so this is the
    # steady-state value the state will sit at.
    out['season'] = float(preset.get('season_preference', 0.5))
    return out


def render_case(label, out_mutator, out_dir, canvas_fbo):
    canvas = StubCanvas()
    canvas.fbo = canvas_fbo
    renderer = StubRenderer(canvas)
    out = base_outstate(renderer)
    out_mutator(out)

    states = [{'count': 0, 'elapsed_time': 0.0, 'duration': 1e9, 'frame_id': 0}
              for _ in WRAPPERS]

    divisor = [1.0]
    frames, mets = [], []
    t = 0.0
    next_capture = WARMUP_S
    total = WARMUP_S + FRAME_GAP_S * N_FRAMES

    while len(frames) < N_FRAMES and t < total + 1.0:
        for i, (_, fn) in enumerate(WRAPPERS):
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

    # ---- contact sheet: raw buffer strip on top, FAN VIEW below ----
    # The fan view is the one that matters - it is what the room sees.
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
    return avg


def main():
    args = [a for a in sys.argv[1:]]
    out_dir = args[0] if args and not args[0].startswith("wyne_") else "_wyne_frames"
    picked = [a for a in args if a.startswith("wyne_")]
    os.makedirs(out_dir, exist_ok=True)

    if not glfw.init():
        print("FAIL: glfw.init()")
        sys.exit(1)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "wyne-harness", None, None)
    if win is None:
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, "wyne-harness", None, None)
    if win is None:
        print("FAIL: could not create a hidden GL context")
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(win)
    print(f"GL context: {glGetString(GL_VERSION).decode()}\n")

    fbo = make_fbo()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    tod = [a.split(":", 1)[1] for a in args if a.startswith("tod:")]
    if tod:
        base = tod[0]
        print(f"{'case':<26} {'cov':>6} {'lum':>6} {'p95':>6} {'sat':>6} {'hues':>5}")
        print("-" * 60)
        for label, sv in (("midnight", 0.0), ("dawn", 0.25),
                          ("noon", 0.5), ("dusk", 0.75)):
            def mut(o, sv=sv, base=base):
                o.update({k: v for k, v in
                          outstate_for_state(base, o['shader_renderer']).items()
                          if k != 'shader_renderer'})
                o['season'] = sv
            avg = render_case(f"tod_{base}_{label}", mut, out_dir, fbo)
            print(f"{'tod_' + base + '_' + label:<26} {avg['cov']:>6.3f} "
                  f"{avg['lum']:>6.3f} {avg['p95']:>6.2f} {avg['sat']:>6.2f} "
                  f"{avg['hues']:>5.0f}")
        glfw.terminate()
        print(f"\nwrote 4 contact sheets to {out_dir}/")
        return

    states = picked or list(wp.WEATHER_SETS["fynewynz"]["states"])

    rows = []
    print(f"{'case':<26} {'cov':>6} {'lum':>6} {'p95':>6} {'sat':>6} {'hues':>5}")
    print("-" * 60)
    for s in states:
        avg = render_case(s, lambda o, s=s: o.update(
            {k: v for k, v in outstate_for_state(s, o['shader_renderer']).items()
             if k not in ('shader_renderer',)}), out_dir, fbo)
        rows.append((s, avg))
        print(f"{s:<26} {avg['cov']:>6.3f} {avg['lum']:>6.3f} "
              f"{avg['p95']:>6.2f} {avg['sat']:>6.2f} {avg['hues']:>5.0f}")

    if not picked:
        print()
        # Narrative block: each companion face alone, over a neutral base, so
        # the six faces can be compared for visibility and hue separation.
        for v in STORY_VARS:
            def mut(o, v=v):
                o['must_hue'] = 0.68
                o['wyne_tint'] = [0.30, 0.26, 0.42]
                o['story_' + v] = 1.0
            avg = render_case(f"narrative_{v}", mut, out_dir, fbo)
            rows.append((f"narrative_{v}", avg))
            print(f"{'narrative_' + v:<26} {avg['cov']:>6.3f} {avg['lum']:>6.3f} "
                  f"{avg['p95']:>6.2f} {avg['sat']:>6.2f} {avg['hues']:>5.0f}")

    glfw.terminate()

    print()
    # The wash / hue heuristics describe FULL SCENES. The narrative rows are
    # isolation renders - one story_* variable over a bare base - so their
    # coverage is just wyne_sky's intended full-arch backdrop and their p95
    # is low because nothing else in the frame is lit. Judging them by the
    # same bar produced false alarms that I twice "fixed" by dimming the
    # narrative shaders, which made them worse. Report them, don't grade them.
    scenes = [(n, a) for n, a in rows if not n.startswith('narrative_')]
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
        print("No near-black, wash-like or single-hue scenes.")
    print(f"\nwrote {len(rows)} contact sheets to {out_dir}/")


if __name__ == "__main__":
    main()
