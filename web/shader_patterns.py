"""Shader Lab pattern bank — prebuilt parameterized GLSL templates.

Most requests aren't novel programs, they're configurations: "slow blue
waves" is the waves skeleton plus three numbers. Each template here is
hand-tuned once against the fan geometry (fanUV physical space where
motion matters), the alpha rules, and the contrast playbook, then
instantiated by substituting typed, clamped parameters — so a matched
request compiles in milliseconds with no LLM call and no repair rounds.

Color is a PALETTE, not a single hue: every template defines a color
coordinate ``ct`` (0..1 — band position, radius, noise, per-star
randomness...) and paints it through a palette expression chosen at
instantiation: 'custom' (one hue), 'duo' (two hues blended — parsed from
"blue and purple ..."), or named schemes (rainbow, sunset, neon, ice,
pastel, candy). Every template also declares three live knobs:
speed | brightness | color shift (iKnob2 slides the palette).

``instant_match(text)`` is deliberately conservative: it only claims a
request when one template's tags clearly win AND most meaningful words
are recognized (colors, palettes, speeds, audio words...). Anything
ambiguous falls through to full codegen.

No GL imports here — templates are plain strings validated by
web/shader_lab.py::validate_glsl and compiled by the normal live-slot
path (so saving, editing, knobs, and chat refinement all keep working).
"""
import random
import re
from string import Template
from typing import Optional, Tuple

# Audio-reactivity choices, substituted as a brightness/size multiplier
# expression (~1.0 at rest).
AUDIO_EXPRS = {
    'none': '1.0',
    'bass': '(0.7 + 0.45 * iBass + 0.5 * iBassPunch)',
    'beat': '(0.75 + 0.5 * iBeat)',
    'energy': '(0.65 + 0.7 * iEnergy)',
}

# ---------------------------------------------------------------------------
# Palettes: expressions over the template's color coordinate `ct`.
# iKnob2 (rest 0.5) slides the whole palette live ("color shift" knob).
# ---------------------------------------------------------------------------

_CT = '(ct + mix(-0.5, 0.5, iKnob2))'
_CT01 = 'clamp(' + _CT + ', 0.0, 1.0)'

PALETTE_NAMES = ('custom', 'duo', 'rainbow', 'sunset', 'neon', 'ice',
                 'pastel', 'candy')


def _pal_expr(palette: str, hue: float, hue2: float, sat: float) -> str:
    """Build the vec3 color expression for one instantiation."""
    h, h2, s = f'{hue:.4f}', f'{hue2:.4f}', f'{sat:.4f}'
    if palette == 'duo':
        return (f'mix(hsv2rgb(vec3({h}, {s}, 1.0)), '
                f'hsv2rgb(vec3({h2}, {s}, 1.0)), {_CT01})')
    if palette == 'rainbow':
        return f'hsv2rgb(vec3(fract({h} + {_CT}), 0.8, 1.0))'
    if palette == 'pastel':
        return f'hsv2rgb(vec3(fract({h} + {_CT}), 0.35, 1.0))'
    if palette == 'candy':
        return f'hsv2rgb(vec3(fract(0.85 + {_CT} * 0.35), 0.6, 1.0))'
    if palette == 'sunset':
        return ('clamp(palette(' + _CT01 + ', vec3(0.5, 0.36, 0.32), '
                'vec3(0.5, 0.42, 0.38), vec3(1.0), '
                'vec3(0.0, 0.08, 0.18)), 0.0, 1.0)')
    if palette == 'neon':
        return ('clamp(palette(' + _CT01 + ', vec3(0.5, 0.25, 0.6), '
                'vec3(0.5, 0.55, 0.4), vec3(1.0), '
                'vec3(0.75, 0.35, 0.6)), 0.0, 1.0)')
    if palette == 'ice':
        return (f'mix(vec3(0.7, 0.85, 1.0), vec3(0.1, 0.3, 0.9), {_CT01})')
    # 'custom' — one hue, subtly shaded along ct
    return f'hsv2rgb(vec3(fract({h} + {_CT} * 0.10), {s}, 1.0))'


_KNOBS = '// KNOBS: speed | brightness | color shift\n'

# Every template shares this contract: params hue/hue2/sat/speed/palette
# always exist; speed is scaled by iKnob0 in-code, brightness by iKnob1,
# and the palette rides iKnob2. The template defines `float ct` (its
# color coordinate) before using $pal; $audio is an AUDIO_EXPRS entry.
PATTERNS = {
    'waves': {
        'desc': 'Layered swells with foam sparkling on the crests',
        'tags': ['wave', 'waves', 'ocean', 'swell', 'bands', 'sea'],
        'params': {
            'hue': {'default': 0.62, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.5, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.85, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 0.6, 'min': 0.05, 'max': 3.0},
            'scale': {'default': 12.0, 'min': 3.0, 'max': 40.0},
            'dir': {'default': 1.0, 'min': -1.0, 'max': 1.0},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float w = sin(uv.y * $scale - iTime * speed * $dir
                  + sin(uv.x * 6.28318 + iTime * speed * 0.3) * 0.7);
    float w2 = sin(uv.y * $scale * 1.7 + iTime * speed * 0.6 * $dir
                   + sin(uv.x * 9.4 - iTime * speed * 0.2) * 0.9);
    float band = clamp(smoothstep(0.1, 0.95, w)
                       + 0.45 * smoothstep(0.3, 1.0, w2), 0.0, 1.2);
    float crest = smoothstep(0.72, 0.95, w);
    float foam = crest * step(0.78, hash21(
        floor(vec2(fragCoord.x * 0.8, fragCoord.y * 0.4))
        + floor(iTime * speed * 5.0)));
    float ct = uv.y * 0.8 + band * 0.2;
    vec3 col = $pal * (0.3 + 0.7 * band) + vec3(1.0) * foam * 0.55;
    float a = (band * 0.75 + foam * 0.6) * $audio * mix(0.2, 2.0, iKnob1);
    fragColor = vec4(col, clamp(a, 0.0, 1.0));
}
''',
    },

    'orb': {
        'desc': 'A glowing ball trailing light as it wanders the fan',
        'tags': ['orb', 'ball', 'sphere', 'blob', 'bubble'],
        'params': {
            'hue': {'default': 0.33, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.55, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.9, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 0.8, 'min': 0.05, 'max': 3.0},
            'size': {'default': 0.2, 'min': 0.06, 'max': 0.45},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord);
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float r = $size * (0.75 + 0.25 * $audio);
    float lum = 0.0;
    float dmin = 10.0;
    for (int i = 0; i < 4; i++) {          // head + fading motion trail
        float fi = float(i);
        float tt = iTime - fi * 0.14 / max(speed, 0.2);
        vec2 ci = vec2(0.55 * sin(tt * 0.37 * speed),
                       0.55 + 0.28 * sin(tt * 0.23 * speed + 1.7));
        float di = length(p - ci);
        float w = (fi < 0.5) ? 1.0 : 0.4 / fi;
        lum += (smoothstep(r, r * 0.35, di)
                + glow(di, r, r * 1.6) * 0.5) * w;
        dmin = min(dmin, di);
    }
    lum = clamp(lum, 0.0, 1.4);
    float ct = clamp(dmin / max(r * 1.8, 0.001), 0.0, 1.0);
    vec3 col = $pal * lum + vec3(1.0) * smoothstep(r * 0.4, 0.0, dmin) * 0.35;
    float a = clamp(lum * mix(0.2, 2.0, iKnob1), 0.0, 1.0);
    fragColor = vec4(col, a);
}
''',
    },

    'rain': {
        'desc': 'Falling streaks that splash into rings at the hub rim',
        'tags': ['rain', 'raining', 'drops', 'drizzle', 'downpour', 'snow',
                 'falling'],
        'params': {
            'hue': {'default': 0.58, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.7, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.7, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 1.0, 'min': 0.05, 'max': 3.0},
            'density': {'default': 1.0, 'min': 0.3, 'max': 2.5},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord);
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float acc = 0.0;
    float tint = 0.0;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 q = p * (7.0 + fi * 4.0) * $density;
        q.y += iTime * speed * (1.6 + fi * 0.7);
        vec2 cell = floor(q);
        float rnd = hash21(cell);
        vec2 f = fract(q) - 0.5;
        float streak = smoothstep(0.45, 0.0, abs(f.x + (rnd - 0.5) * 0.6))
                     * smoothstep(0.5, -0.2, f.y)
                     * step(rnd, 0.35);
        acc += streak * (0.4 + 0.6 * fract(rnd * 7.0));
        tint += streak * rnd;
    }
    // splash rings where drops land at the hub rim
    float sphase = fract(iTime * speed * 1.5);
    float scol = floor(p.x * 4.0);
    float srnd = hash21(vec2(scol, floor(iTime * speed * 1.5)));
    vec2 sc = vec2((scol + 0.2 + srnd * 0.6) / 4.0, 0.21);
    float splash = glow(abs(length(vec2(p.x - sc.x, (p.y - sc.y) * 2.5))
                            - sphase * 0.22), 0.005, 0.035)
                 * (1.0 - sphase) * step(srnd, 0.55)
                 * smoothstep(0.35, 0.24, p.y);
    acc += splash * 1.3;
    float ct = clamp(tint + p.y * 0.3, 0.0, 1.0);
    vec3 col = $pal;
    float a = clamp(acc * 0.85 * $audio * mix(0.2, 2.0, iKnob1), 0.0, 1.0);
    fragColor = vec4(col, a);
}
''',
    },

    'fire': {
        'desc': 'Heat-warped flames with embers rising off the top',
        'tags': ['fire', 'flame', 'flames', 'burning', 'lava', 'ember',
                 'embers', 'inferno'],
        'params': {
            'hue': {'default': 0.02, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.12, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.95, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'duo'},
            'speed': {'default': 0.9, 'min': 0.05, 'max': 3.0},
            'height': {'default': 1.4, 'min': 0.6, 'max': 2.5},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord);
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    // heat-shimmer warp feeding the flame body
    float warp = fbm(vec2(p.x * 6.0 + iTime * speed * 0.4,
                          p.y * 3.0 - iTime * speed * 1.7)) - 0.5;
    float n = fbm(vec2(p.x * 4.0 + warp * 0.8, p.y * 2.5 - iTime * speed));
    float flame = clamp(n * 1.7 - p.y * $height, 0.0, 1.0);
    flame = pow(flame, 1.8) * $audio;
    // rising embers above the flame body
    vec2 eg = vec2(p.x * 7.0 + warp, p.y * 3.5 - iTime * speed * 1.4);
    float er = hash21(floor(eg));
    float ember = glow(length(fract(eg) - vec2(0.5 + (er - 0.5) * 0.6)),
                       0.015, 0.09)
                * step(er, 0.22) * smoothstep(1.4, 0.15, p.y)
                * (0.5 + 0.5 * sin(iTime * (3.0 + er * 5.0) + er * 30.0));
    float ct = clamp(flame, 0.0, 1.0);
    vec3 col = $pal * flame + vec3(1.0, 0.62, 0.25) * ember * 0.85;
    float a = clamp((flame + ember * 0.8) * mix(0.2, 2.0, iKnob1),
                    0.0, 1.0) * 0.95;
    fragColor = vec4(col, a);
}
''',
    },

    'plasma': {
        'desc': 'Domain-warped nebula curls with bright filaments',
        'tags': ['plasma', 'psychedelic', 'swirl', 'swirling', 'nebula',
                 'trippy', 'clouds'],
        'params': {
            'hue': {'default': 0.7, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.95, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.85, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'duo'},
            'speed': {'default': 0.5, 'min': 0.05, 'max': 3.0},
            'scale': {'default': 1.0, 'min': 0.4, 'max': 3.0},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord) * $scale * 3.0;
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float t = iTime * speed * 0.4;
    // domain-warped fbm: organic curls instead of flat sine interference
    vec2 q = vec2(fbm(p * 0.8 + t * 0.3),
                  fbm(p * 0.8 + vec2(5.2, 1.3) - t * 0.2));
    float m = fbm(p * 0.9 + 2.2 * q + t * 0.15);
    float hi = smoothstep(0.55, 0.85, m);          // bright filaments
    float ct = clamp(m * 1.5 - 0.2 + q.x * 0.3, 0.0, 1.0);
    vec3 col = $pal * (0.25 + 0.95 * m) + vec3(1.0, 0.95, 1.0) * hi * 0.3;
    float a = clamp((0.2 + 0.85 * m + hi * 0.3) * $audio
                    * mix(0.2, 2.0, iKnob1), 0.0, 1.0) * 0.85;
    fragColor = vec4(col, a);
}
''',
    },

    'sparkles': {
        'desc': 'Twinkles plus rare big stars with cross flares',
        'tags': ['sparkle', 'sparkles', 'glitter', 'twinkle', 'twinkling',
                 'star', 'stars', 'starfield', 'fireflies', 'firefly'],
        'params': {
            'hue': {'default': 0.14, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.6, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.4, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 1.0, 'min': 0.05, 'max': 3.0},
            'density': {'default': 1.0, 'min': 0.3, 'max': 2.5},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord);
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    vec2 g = p * 22.0 * $density;
    vec2 cell = floor(g);
    float rnd = hash21(cell);
    float tw = 0.5 + 0.5 * sin(iTime * speed * (2.0 + rnd * 6.0)
                               + rnd * 41.0);
    float d = length(fract(g) - 0.5);
    float star = glow(d, 0.03, 0.22) * step(rnd, 0.14) * tw * tw;
    // rare big stars with 4-point cross flares
    vec2 g2 = p * 9.0 * $density;
    vec2 f2 = fract(g2) - 0.5;
    float rnd2 = hash21(floor(g2) + 31.0);
    float tw2 = pow(0.5 + 0.5 * sin(iTime * speed * (1.0 + rnd2 * 3.0)
                                    + rnd2 * 17.0), 3.0);
    float big = (glow(length(f2), 0.02, 0.16)
                 + glow(abs(f2.x) + abs(f2.y), 0.0, 0.13) * 0.8)
              * step(rnd2, 0.05) * tw2;
    float ct = rnd;
    vec3 col = $pal * star + vec3(1.0) * big * 0.8;
    float a = clamp((star + big) * $audio * mix(0.2, 2.0, iKnob1),
                    0.0, 1.0);
    fragColor = vec4(col, a);
}
''',
    },

    'rays': {
        'desc': 'Rotating beams with dust shimmering outward through them',
        'tags': ['rays', 'ray', 'beams', 'beam', 'sunburst', 'searchlight',
                 'spokes'],
        'params': {
            'hue': {'default': 0.09, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.55, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.85, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 0.5, 'min': 0.05, 'max': 3.0},
            'count': {'default': 6.0, 'min': 2.0, 'max': 16.0},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float spin = iTime * speed * 1.5;
    float ang = uv.x * $count * 6.28318 + spin;
    float beam = pow(0.5 + 0.5 * sin(ang), 8.0);
    // godray shimmer: dust drifting outward through the beams
    float shimmer = 0.7 + 0.3 * vnoise(vec2(uv.x * 30.0,
                                            uv.y * 5.0 - iTime * speed * 2.0));
    beam *= shimmer;
    float fadeR = smoothstep(0.0, 0.25, uv.y) * (1.0 - uv.y * 0.35);
    float ct = fract(uv.x + spin * 0.05);
    vec3 col = $pal * beam;
    float a = clamp(beam * fadeR * 0.85 * $audio * mix(0.2, 2.0, iKnob1),
                    0.0, 1.0);
    fragColor = vec4(col, a);
}
''',
    },

    'rings': {
        'desc': 'Expanding arcs that waver and shimmer organically',
        'tags': ['rings', 'ring', 'ripple', 'ripples', 'circles', 'sonar',
                 'radar'],
        'params': {
            'hue': {'default': 0.5, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.85, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.85, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 0.6, 'min': 0.05, 'max': 3.0},
            'count': {'default': 6.0, 'min': 2.0, 'max': 18.0},
            'dir': {'default': 1.0, 'min': -1.0, 'max': 1.0},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float wob = vnoise(vec2(uv.x * 6.0, iTime * speed * 0.5)) - 0.5;
    float r = (uv.y + wob * 0.05) * $count - iTime * speed * $dir;
    float ring = pow(0.5 + 0.5 * sin(r * 6.28318), 6.0);
    ring *= 0.8 + 0.2 * sin(uv.x * 40.0 + iTime * speed * 3.0);
    float ct = fract(r * 0.25);
    vec3 col = $pal * ring;
    float a = clamp(ring * 0.85 * $audio * mix(0.2, 2.0, iKnob1), 0.0, 1.0);
    fragColor = vec4(col, a);
}
''',
    },

    'aurora': {
        'desc': 'Striated aurora curtains over a twinkling star backdrop',
        'tags': ['aurora', 'borealis', 'northern', 'curtains', 'curtain'],
        'params': {
            'hue': {'default': 0.38, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.75, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.85, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'duo'},
            'speed': {'default': 0.5, 'min': 0.05, 'max': 3.0},
            'scale': {'default': 1.0, 'min': 0.4, 'max': 2.5},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord);
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float n = fbm(vec2(p.x * 2.5 * $scale + iTime * speed * 0.15,
                       p.y * 1.2));
    float curtain = smoothstep(0.35, 0.75, n) * smoothstep(0.05, 0.35, p.y);
    float shimmer = 0.7 + 0.3 * vnoise(vec2(p.x * 20.0, iTime * speed));
    float stria = 0.75 + 0.25 * sin(p.x * 55.0 + n * 22.0);
    // faint twinkling stars behind the curtains
    vec2 sg = p * 16.0;
    float srnd = hash21(floor(sg));
    float stars = glow(length(fract(sg) - 0.5), 0.02, 0.1)
                * step(srnd, 0.07)
                * (0.4 + 0.6 * sin(iTime * (1.0 + srnd * 4.0) + srnd * 20.0));
    float ct = n;
    vec3 col = $pal * curtain * shimmer * stria
             + vec3(0.7, 0.8, 1.0) * stars * 0.35;
    float a = clamp((curtain * 0.8 + stars * 0.3) * $audio
                    * mix(0.2, 2.0, iKnob1), 0.0, 1.0);
    fragColor = vec4(col, a);
}
''',
    },

    'chase': {
        'desc': 'Comets shedding sparkling dust as they sweep around',
        'tags': ['chase', 'comet', 'comets', 'meteor', 'meteors', 'sweep',
                 'spinner', 'orbit', 'shooting'],
        'params': {
            'hue': {'default': 0.55, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.9, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.8, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 1.0, 'min': 0.05, 'max': 3.0},
            'count': {'default': 3.0, 'min': 1.0, 'max': 6.0},
            'tail': {'default': 0.35, 'min': 0.1, 'max': 0.8},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float acc = 0.0;
    vec3 col = vec3(0.0);
    for (int i = 0; i < $counti; i++) {
        float fi = float(i);
        float head = fract(iTime * speed * 0.25 + fi / $count);
        float d = fract(uv.x - head);
        float tail = pow(smoothstep($tail, 0.0, d), 1.5);
        float ct = fract(fi / $count + d * 0.3);
        // sparkling dust shed along the tail
        float dust = step(0.82, hash21(vec2(floor(uv.x * 80.0),
                                            floor(iTime * 16.0) + fi)))
                   * smoothstep($tail, 0.0, d) * 0.6;
        col += $pal * tail + vec3(1.0) * dust;
        acc += tail + dust;
    }
    float radial = smoothstep(0.0, 0.15, uv.y);
    float a = clamp(acc, 0.0, 1.0) * radial * 0.9 * $audio
            * mix(0.2, 2.0, iKnob1);
    fragColor = vec4(col, clamp(a, 0.0, 1.0));
}
''',
    },

    'breathe': {
        'desc': 'A breathing wash of color over slowly drifting clouds',
        'tags': ['breathe', 'breathing', 'wash', 'ambient', 'mood',
                 'glow', 'calm'],
        'params': {
            'hue': {'default': 0.76, 'min': 0.0, 'max': 1.0},
            'hue2': {'default': 0.6, 'min': 0.0, 'max': 1.0},
            'sat': {'default': 0.8, 'min': 0.0, 'max': 1.0},
            'palette': {'palette': True, 'default': 'custom'},
            'speed': {'default': 0.4, 'min': 0.05, 'max': 3.0},
            'audio': {'choice': AUDIO_EXPRS, 'default': 'none'},
        },
        'glsl': _KNOBS + '''void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fanUV(fragCoord);
    float speed = $speed * mix(0.25, 4.0, iKnob0);
    float breath = 0.5 + 0.5 * sin(iTime * speed * 1.2);
    float m = mix(0.35, 1.0, breath) * $audio;
    float center = 1.0 - smoothstep(0.45, 1.05, length(p - vec2(0.0, 0.4)));
    // slow drifting cloud texture so the wash is alive, not flat
    float tex = 0.7 + 0.3 * fbm(p * 2.2 + vec2(iTime * speed * 0.12,
                                               iTime * speed * 0.05));
    float ct = clamp(center * 0.6 + tex * 0.4, 0.0, 1.0);
    vec3 col = $pal * m * (0.35 + 0.65 * center) * tex;
    float a = clamp(m * (0.25 + 0.5 * center) * mix(0.2, 2.0, iKnob1),
                    0.0, 1.0) * 0.85;
    fragColor = vec4(col, a);
}
''',
    },
}


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def instantiate(pattern_id: str, overrides: Optional[dict] = None
                ) -> Tuple[str, str]:
    """Substitute (clamped) params into a template. Returns (glsl, desc).

    Unknown override keys are ignored; out-of-range floats clamp; bad
    choice/palette values fall back to defaults. Raises KeyError on
    unknown id.
    """
    spec = PATTERNS[pattern_id]
    overrides = overrides or {}
    vals = {}
    floats = {}
    palette = spec['params']['palette']['default']
    for name, p in spec['params'].items():
        v = overrides.get(name, p.get('default'))
        if p.get('palette'):
            palette = v if v in PALETTE_NAMES else p['default']
        elif 'choice' in p:
            if v not in p['choice']:
                v = p['default']
            vals[name] = p['choice'][v]
        else:
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = p['default']
            v = max(p['min'], min(p['max'], v))
            floats[name] = v
            vals[name] = f'{v:.4f}'
    vals['pal'] = _pal_expr(palette, floats.get('hue', 0.5),
                            floats.get('hue2', 0.8), floats.get('sat', 0.85))
    if 'count' in spec['params']:        # loop bounds need an int literal
        vals['counti'] = str(int(round(floats['count'])))
    glsl = Template(spec['glsl']).substitute(vals)
    return glsl, spec['desc']


# ---------------------------------------------------------------------------
# Instant matching (no LLM)
# ---------------------------------------------------------------------------

COLOR_HUES = {
    'red': 0.0, 'orange': 0.055, 'amber': 0.09, 'gold': 0.11,
    'yellow': 0.14, 'lime': 0.24, 'green': 0.33, 'emerald': 0.38,
    'teal': 0.45, 'turquoise': 0.47, 'cyan': 0.5, 'aqua': 0.5,
    'blue': 0.62, 'indigo': 0.7, 'purple': 0.76, 'violet': 0.79,
    'lavender': 0.72, 'magenta': 0.85, 'pink': 0.9, 'rose': 0.93,
    'crimson': 0.97, 'white': None, 'silver': None,
}
PALETTE_WORDS = {
    'rainbow': 'rainbow', 'colorful': 'rainbow', 'multicolor': 'rainbow',
    'multicolored': 'rainbow', 'sunset': 'sunset', 'neon': 'neon',
    'pastel': 'pastel', 'pastels': 'pastel', 'ice': 'ice', 'icy': 'ice',
    'candy': 'candy',
}
_SLOW_WORDS = {'slow', 'slowly', 'gentle', 'gently', 'calm', 'lazy',
               'soft', 'drifting', 'peaceful'}
_FAST_WORDS = {'fast', 'quick', 'quickly', 'rapid', 'energetic', 'frantic',
               'racing', 'intense'}
_AUDIO_WORDS = {'bass': 'bass', 'beat': 'beat', 'beats': 'beat',
                'music': 'beat', 'rhythm': 'beat', 'pulse': 'beat',
                'pulsing': 'beat', 'pulses': 'beat', 'energy': 'energy',
                'sound': 'energy', 'audio': 'energy', 'reactive': 'energy'}
_DENSE_WORDS = {'dense', 'many', 'lots', 'heavy', 'thick', 'busy'}
_SPARSE_WORDS = {'sparse', 'few', 'light', 'scattered', 'occasional'}
# Filler that neither votes for a template nor counts against coverage.
_STOP_WORDS = {
    'the', 'and', 'with', 'that', 'like', 'make', 'give', 'want', 'please',
    'some', 'more', 'very', 'kind', 'sort', 'them', 'they', 'over', 'across',
    'through', 'around', 'down', 'from', 'into', 'onto', 'pattern',
    'patterns', 'effect', 'colors', 'color', 'colored', 'colour', 'coloured',
    'display', 'leds', 'led', 'lights', 'fan', 'screen', 'show', 'showing',
    'bright', 'dark', 'deep', 'dim', 'subtle', 'nice', 'pretty', 'cool',
    'moving', 'would',
}


def instant_match(text: str):
    """Match a request against the bank with NO model call.

    Returns (pattern_id, glsl, human_desc) on a confident match, else
    None. Conservative on purpose: an unrecognized meaningful word means
    the person asked for something the bank can't express — codegen it.
    """
    words = re.findall(r"[a-z]+", (text or '').lower())
    if not words or len(words) > 24:
        return None

    scores = {pid: sum(1 for w in words if w in spec['tags'])
              for pid, spec in PATTERNS.items()}
    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    if scores[best] == 0 or (len(ranked) > 1 and ranked[1] == ranked[0]):
        return None

    # Coverage: every meaningful word must be something we understand.
    recognized = 0
    meaningful = 0
    for w in words:
        if len(w) <= 2 or w in _STOP_WORDS:
            continue
        meaningful += 1
        if (w in PATTERNS[best]['tags'] or w in COLOR_HUES
                or w in PALETTE_WORDS
                or w in _SLOW_WORDS or w in _FAST_WORDS
                or w in _AUDIO_WORDS or w in _DENSE_WORDS
                or w in _SPARSE_WORDS):
            recognized += 1
    if meaningful == 0 or recognized / meaningful < 0.75:
        return None

    spec = PATTERNS[best]
    overrides = {}
    pal_hits = [PALETTE_WORDS[w] for w in words if w in PALETTE_WORDS]
    colors = [w for w in words if w in COLOR_HUES]
    if pal_hits:
        overrides['palette'] = pal_hits[0]
        if colors and COLOR_HUES[colors[0]] is not None:
            overrides['hue'] = COLOR_HUES[colors[0]]   # anchors rainbow etc.
    elif len(colors) >= 2:
        h1, h2 = COLOR_HUES[colors[0]], COLOR_HUES[colors[1]]
        if h1 is not None and h2 is not None:
            overrides.update(palette='duo', hue=h1, hue2=h2)
        elif h1 is not None or h2 is not None:
            overrides.update(palette='custom',
                             hue=h1 if h1 is not None else h2, sat=0.5)
    elif colors:
        hue = COLOR_HUES[colors[0]]
        if hue is None:                     # white/silver
            overrides.update(palette='custom', sat=0.05)
        else:
            overrides.update(palette='custom', hue=hue)
    factor = (0.45 if any(w in _SLOW_WORDS for w in words) else
              1.9 if any(w in _FAST_WORDS for w in words) else None)
    if factor is not None:
        overrides['speed'] = spec['params']['speed']['default'] * factor
    hits = [_AUDIO_WORDS[w] for w in words if w in _AUDIO_WORDS]
    if hits:
        # A specific band beats the generic beat words: "pulse with the
        # bass" means bass reactivity, not beat flashes.
        overrides['audio'] = ('bass' if 'bass' in hits else
                              'energy' if 'energy' in hits else hits[0])
    if 'density' in spec['params']:
        if any(w in _DENSE_WORDS for w in words):
            overrides['density'] = spec['params']['density']['default'] * 1.7
        elif any(w in _SPARSE_WORDS for w in words):
            overrides['density'] = spec['params']['density']['default'] * 0.5

    glsl, desc = instantiate(best, overrides)
    return best, glsl, desc, overrides


def shuffle_params(pattern_id: str) -> dict:
    """A fresh random character for a layer: new colors, palette, pace.
    Powers the stage tiles' dice button — instant, no LLM."""
    spec = PATTERNS[pattern_id]['params']
    out = {'hue': round(random.random(), 3),
           'hue2': round(random.random(), 3),
           'palette': random.choice(PALETTE_NAMES)}
    if 'speed' in spec:
        lo, hi = spec['speed']['min'], spec['speed']['max']
        out['speed'] = round(lo + (hi - lo) * (0.1 + 0.5 * random.random()), 3)
    return out
