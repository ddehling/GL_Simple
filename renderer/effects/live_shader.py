"""Live Shader Lab — runtime hot-swappable fragment shader slot.

Engine infrastructure (no visual content baked in): the actual pattern is a
user/LLM-supplied Shadertoy-style ``mainImage`` body pushed in at runtime
through ``outstate['live_shader_source']`` (see web/shader_lab.py and the
/shaderlab page). The effect always keeps its last-good program: a failed
compile never blanks the LEDs and never disables the effect.

Contract for user code (everything else — version, precision, uniforms,
``main()`` — is owned by the wrapper here):

    void mainImage(out vec4 fragColor, in vec2 fragCoord)

``fragCoord`` arrives in pixels (Shadertoy convention), iResolution is the
canvas size (fan: 128 x 300, tall portrait). The header also provides the
fan-layout helpers ``fanPos``/``fanUV``/``fanAngle``/``fanRadius`` so user
code can draw in the installation's true physical space (the pixel grid is
a polar fan folded into a rectangle). Output is straight alpha (HARD RULE
4); the wrapper multiplies alpha by the spawn-fade uniform.
"""
import time as _time
import numpy as np
from OpenGL.GL import *
from typing import Dict, Optional, Tuple
import ctypes

from renderer.effects.base import ShaderEffect


# ---------------------------------------------------------------------------
# GLSL scaffolding
# ---------------------------------------------------------------------------

VERTEX_SHADER = """#version 310 es
precision highp float;

layout(location = 0) in vec2 position;

uniform float depth;

out vec2 fragCoord;

void main() {
    float mappedDepth = clamp(depth / 100.0, 0.0, 1.0);
    gl_Position = vec4(position, mappedDepth, 1.0);
    fragCoord = (position + 1.0) * 0.5;
}
"""

# The uniform contract the Shader Lab promises to user/LLM code. Inactive
# uniforms are optimized out by the driver; base.uniform() caches -1 for
# those and glUniform* calls with -1 are silent no-ops, so render() can set
# the whole block unconditionally.
FRAGMENT_HEADER = """#version 310 es
precision highp float;

uniform vec2  iResolution;
uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iHigh;
uniform float iBassPunch;
uniform float iHighPunch;
uniform float iBeat;
uniform float iEnergy;
uniform float iDrop;
uniform float iBuild;
uniform float iBPM;
uniform float iPhrase;
uniform float iSeason;
uniform float iWind;
uniform float iRain;
uniform float iFade;
uniform float iIntensity;   // web "Intensity" slider; applied in main()
uniform float iKnob0;       // live performance sliders, 0..1, rest at 0.5
uniform float iKnob1;       // (labels come from the pattern's optional
uniform float iKnob2;       //  `// KNOBS: a | b` first-line comment —
uniform float iKnob3;       //  see web/shader_lab.py::extract_knobs)

in vec2 fragCoord;
out vec4 outColor;

// ---- Fan layout (engine-provided; keep in sync with fan_geometry.py) ----
// The canvas is NOT physically rectangular: column x is one of the LED
// strips fanned across a 180-degree semicircle (left edge points left,
// center column straight up, right edge right); row y runs outward along
// the strip from the hub (4.0 ft) to the rim (20.6 ft). These helpers map
// a fragCoord to the installation's real geometry so patterns can draw in
// true physical space instead of the warped strip/pixel grid.

const float FAN_R_INNER = 4.0;    // feet, first LED of every strip
const float FAN_R_OUTER = 20.6;   // feet, last LED

// Angle of this pixel's strip in radians: PI at the left edge, 0 at the
// right (matches FanGeometry.thetas).
float fanAngle(vec2 fc) {
    return 3.141592653589793 * (1.0 - fc.x / iResolution.x);
}

// Distance of this pixel from the hub center, in feet (4.0 .. 20.6).
float fanRadius(vec2 fc) {
    return mix(FAN_R_INNER, FAN_R_OUTER, fc.y / iResolution.y);
}

// Physical position in feet: x right, y up, origin at the hub.
// x spans [-20.6, 20.6]; y spans [0, 20.6] (upper semicircle).
vec2 fanPos(vec2 fc) {
    float a = fanAngle(fc);
    return fanRadius(fc) * vec2(cos(a), sin(a));
}

// fanPos normalized by the outer radius: x in [-1, 1], y in [0, 1].
// Distances, circles and speeds are TRUE in this space.
vec2 fanUV(vec2 fc) {
    return fanPos(fc) / FAN_R_OUTER;
}

// ---- Standard library (engine-provided; user code must never redefine
// these — web/shader_lab.py rejects redefinitions by name). Generated
// patterns lean on these instead of re-deriving them every time, which
// keeps replies short and the implementations correct. -------------------

float hash11(float n) { return fract(sin(n * 127.1) * 43758.5453123); }
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Value noise, 0..1, feature size ~1 unit.
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i),                  hash21(i + vec2(1.0, 0.0)), f.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
               f.y);
}

// Fractal noise, 4 octaves, ~0..1.
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * vnoise(p);
        p = p * 2.03 + 19.7;
        a *= 0.5;
    }
    return v;
}

mat2 rot2(float a) { float s = sin(a), c = cos(a); return mat2(c, -s, s, c); }

// h,s,v each 0..1.
vec3 hsv2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0)
                     - 1.0, 0.0, 1.0);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

// IQ cosine palette: a + b*cos(2pi*(c*t + d)).
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.2831853 * (c * t + d));
}

// Signed distances (negative inside).
float sdCircle(vec2 p, vec2 c, float r) { return length(p - c) - r; }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
float sdBox(vec2 p, vec2 c, vec2 halfSize) {
    vec2 d = abs(p - c) - halfSize;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

// 1.0 inside distance `radius`, fading to 0.0 over `soft`.
float glow(float d, float radius, float soft) {
    return smoothstep(radius + soft, radius, d);
}

// Fake-3D ball: unit surface normal at position p for a sphere at center
// c radius r, or vec3(0.0) outside it. n.z faces the viewer — light with
// e.g. max(dot(n, normalize(vec3(-0.4, 0.6, 0.7))), 0.0).
vec3 sphereNormal(vec2 p, vec2 c, float r) {
    vec2 q = (p - c) / r;
    float d2 = dot(q, q);
    if (d2 >= 1.0) return vec3(0.0);
    return vec3(q, sqrt(1.0 - d2));
}

#line 1
"""

FRAGMENT_FOOTER = """
void main() {
    vec2 fc = fragCoord * iResolution;
    vec4 col = vec4(0.0);
    mainImage(col, fc);
    col.a = clamp(col.a, 0.0, 1.0) * iFade * iIntensity;
    outColor = col;   // straight alpha (HARD RULE 4)
}
"""

# Loaded when no user source is set (and as the compile target in init())
# so the effect is always in a valid, invisible state.
DEFAULT_USER_SOURCE = """void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(0.0);
}
"""


class ShaderCompileError(Exception):
    """Carries the raw driver info log (line numbers already match the
    user's code thanks to the ``#line 1`` directive in FRAGMENT_HEADER)."""


def _decode_log(log) -> str:
    if isinstance(log, bytes):
        log = log.decode('utf-8', 'replace')
    return str(log).strip()


def _compile_stage(source: str, stage) -> int:
    """Compile one shader stage with the info log captured directly —
    PyOpenGL's compileShader wraps the log in an exception repr that is
    painful to un-mangle; asking the driver ourselves keeps it clean for
    both humans and the LLM repair loop."""
    sh = glCreateShader(stage)
    glShaderSource(sh, source)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        log = _decode_log(glGetShaderInfoLog(sh))
        glDeleteShader(sh)
        raise ShaderCompileError(log or 'shader compile failed (no log)')
    return sh


# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------

class LiveShaderEffect(ShaderEffect):
    """Fullscreen-quad shader whose fragment source is swappable at runtime.

    ``set_source()`` is the hot-swap core and must only ever be called from
    the render thread (the only thread that owns the GL context). It
    side-compiles into a fresh program and swaps only on success, so the
    previous pattern keeps rendering through a failed compile.
    """

    def __init__(self, viewport, depth: float = 20.0):
        super().__init__(viewport)
        self.depth = depth
        self.z_centroid = depth / 100.0
        self.time = 0.0
        self.fade = 0.0
        self.user_source: Optional[str] = None
        self.compile_error: Optional[str] = None
        self.has_content = False

        # Fed by the wrapper each frame from outstate; smoothed in update().
        self.audio_bass = 0.0
        self.audio_mid = 0.0
        self.audio_high = 0.0
        self.bass_smooth = 0.0
        self.mid_smooth = 0.0
        self.high_smooth = 0.0
        # Transient envelopes / scalars passed straight through.
        self.bass_punch = 0.0
        self.high_punch = 0.0
        self.beat = 0.0
        self.energy = 0.0
        self.drop = 0.0
        self.build = 0.0
        self.bpm = 0.0
        self.phrase = 0.0
        self.season = 0.0
        self.wind = 0.0
        self.rain = 0.0
        # Live performance controls (web sliders via outstate).
        self.intensity = 1.0
        self.knobs = [0.5, 0.5, 0.5, 0.5]

        self.quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)

    # -- compilation ------------------------------------------------------

    def _build_fragment_source(self, user_glsl: str) -> str:
        return FRAGMENT_HEADER + user_glsl + FRAGMENT_FOOTER

    def _compile_program(self, user_glsl: str):
        vert = _compile_stage(VERTEX_SHADER, GL_VERTEX_SHADER)
        try:
            frag = _compile_stage(self._build_fragment_source(user_glsl),
                                  GL_FRAGMENT_SHADER)
        except ShaderCompileError:
            glDeleteShader(vert)
            raise
        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glLinkProgram(prog)
        glDeleteShader(vert)
        glDeleteShader(frag)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            log = _decode_log(glGetProgramInfoLog(prog))
            glDeleteProgram(prog)
            raise ShaderCompileError('link failed: '
                                     + (log or '(no log)'))
        return prog

    def compile_shader(self):
        # Called once by init(); the default source always compiles, so the
        # effect can never hit base.init()'s disable-on-error path in
        # normal operation.
        return self._compile_program(DEFAULT_USER_SOURCE)

    def set_source(self, user_glsl: Optional[str]) -> Tuple[bool, str]:
        """Hot-swap the fragment shader. Render thread only.

        Returns (ok, error_text). On failure the old program keeps running
        and ``error_text`` carries the cleaned driver log with line numbers
        that match the user's code.
        """
        target = user_glsl if user_glsl else DEFAULT_USER_SOURCE
        try:
            new_prog = self._compile_program(target)
        except ShaderCompileError as e:
            self.compile_error = str(e)
            return False, self.compile_error
        except Exception as e:  # unexpected GL/driver failure
            self.compile_error = f'GL error during compile: {e}'
            return False, self.compile_error

        old = self.shader
        self.shader = new_prog
        # Mandatory after a swap: GLuint program handles get recycled by the
        # driver, so stale cached uniform locations would silently bind to
        # the wrong slots (base.py uniform() docstring).
        self._uniform_cache.clear()
        if old:
            try:
                glDeleteProgram(old)
            except Exception:
                pass
        self.user_source = user_glsl if user_glsl else None
        self.has_content = bool(user_glsl)
        self.compile_error = None
        return True, ''

    # -- buffers ----------------------------------------------------------

    def setup_buffers(self):
        # Created exactly once; recompiles never touch buffers, so repeated
        # failed compiles cannot leak GL objects. Stored in the base-class
        # slots so base.cleanup() handles teardown.
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        self.VBOs = [vbo]
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes,
                     self.quad_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)

    # -- per-frame --------------------------------------------------------

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self.time += dt

        # Band means get attack/decay smoothing (same shape as fractal_fog);
        # punch/beat envelopes are already transient-shaped, no smoothing.
        attack = 1.0 - np.exp(-dt / 0.12)
        decay = 1.0 - np.exp(-dt / 0.45)
        for raw_name, smooth_name in (('audio_bass', 'bass_smooth'),
                                      ('audio_mid', 'mid_smooth'),
                                      ('audio_high', 'high_smooth')):
            raw = getattr(self, raw_name)
            smooth = getattr(self, smooth_name)
            k = attack if raw > smooth else decay
            setattr(self, smooth_name, smooth + (raw - smooth) * k)

    def render(self, state: Dict):
        if not self.enabled or not self.shader or not self.has_content:
            return

        prog = self.shader
        glUseProgram(prog)
        glUniform1f(self.uniform(prog, 'depth'), float(self.depth))
        glUniform2f(self.uniform(prog, 'iResolution'),
                    float(self.viewport.width), float(self.viewport.height))
        glUniform1f(self.uniform(prog, 'iTime'), self.time)
        glUniform1f(self.uniform(prog, 'iBass'), self.bass_smooth)
        glUniform1f(self.uniform(prog, 'iMid'), self.mid_smooth)
        glUniform1f(self.uniform(prog, 'iHigh'), self.high_smooth)
        glUniform1f(self.uniform(prog, 'iBassPunch'), self.bass_punch)
        glUniform1f(self.uniform(prog, 'iHighPunch'), self.high_punch)
        glUniform1f(self.uniform(prog, 'iBeat'), self.beat)
        glUniform1f(self.uniform(prog, 'iEnergy'), self.energy)
        glUniform1f(self.uniform(prog, 'iDrop'), self.drop)
        glUniform1f(self.uniform(prog, 'iBuild'), self.build)
        glUniform1f(self.uniform(prog, 'iBPM'), self.bpm)
        glUniform1f(self.uniform(prog, 'iPhrase'), self.phrase)
        glUniform1f(self.uniform(prog, 'iSeason'), self.season)
        glUniform1f(self.uniform(prog, 'iWind'), self.wind)
        glUniform1f(self.uniform(prog, 'iRain'), self.rain)
        glUniform1f(self.uniform(prog, 'iFade'), self.fade)
        glUniform1f(self.uniform(prog, 'iIntensity'), float(self.intensity))
        glUniform1f(self.uniform(prog, 'iKnob0'), float(self.knobs[0]))
        glUniform1f(self.uniform(prog, 'iKnob1'), float(self.knobs[1]))
        glUniform1f(self.uniform(prog, 'iKnob2'), float(self.knobs[2]))
        glUniform1f(self.uniform(prog, 'iKnob3'), float(self.knobs[3]))

        # Translucent layer: depth-TEST stays on, depth-WRITE off
        # (HARD RULE 2 / 6).
        glDepthMask(GL_FALSE)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glUseProgram(0)


# ---------------------------------------------------------------------------
# Event wrapper (auto-exported by renderer.effects.__init__)
# ---------------------------------------------------------------------------

# Number of independent live slots. Slot 0 draws first (bottom layer),
# higher slots composite on top via straight-alpha blending in draw order.
# web/shader_lab.py declares the same constant for the web layer (no GL
# import there); the gate test asserts they match.
NUM_SLOTS = 16


def shader_live_shader(state, outstate, depth=20.0):
    """Universal Shader Lab slots, scheduled as one implicit background
    event managing NUM_SLOTS independent LiveShaderEffect layers.

    outstate is the single source of truth (same pattern as
    shader_narrative_player): the web layer publishes per-slot
    ``live_shader_source_<i>`` + ``live_shader_seq_<i>`` and this wrapper
    applies them on the render thread, writing each compile outcome to
    ``live_shader_status_<i>``. Because the sources live in outstate, the
    patterns survive weather-set changes and project swaps — the respawned
    wrapper simply recompiles them.
    """
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            # Added in slot order so blending order (= list order) puts
            # higher slots on top.
            state['effects'] = [
                viewport.add_effect(LiveShaderEffect, depth=depth - i * 0.2)
                for i in range(NUM_SLOTS)]
            state['applied_seqs'] = [None] * NUM_SLOTS
            print(f'[shader_live_shader] Started ({NUM_SLOTS} live slots ready)')
        except Exception as e:
            print(f'[shader_live_shader] Failed to initialize: {e}')
            return

    if state['count'] == -1:
        for effect in state.get('effects', []):
            if effect in viewport.effects:
                viewport.effects.remove(effect)
                effect.cleanup()
        print('[shader_live_shader] Stopped')
        return

    effects = state.get('effects')
    if not effects:
        return

    # Shared per-frame inputs, computed once.
    audio = outstate.get('sound')
    bass = mid = high = 0.0
    if audio is not None:
        try:
            bands = audio['norm_short'][0]
            bass = float(np.mean(bands[0:8]))
            mid = float(np.mean(bands[8:20]))
            high = float(np.mean(bands[20:32]))
        except Exception:
            pass

    def _f(key, default=0.0):
        try:
            return float(outstate.get(key, default) or 0.0)
        except (TypeError, ValueError):
            return default

    shared = {
        'bass_punch': _f('bass_punch'), 'high_punch': _f('high_punch'),
        'beat': _f('beat_decay'), 'energy': _f('audio_energy'),
        'drop': _f('drop'), 'build': _f('build_level'),
        'bpm': _f('bpm'), 'phrase': _f('phrase_phase'),
        'season': _f('season'), 'wind': _f('wind'), 'rain': _f('rain'),
    }
    # Gentle fade-in after (re)spawn so set changes don't pop.
    fade = float(np.clip(state.get('elapsed_time', 0.0) / 1.0, 0.0, 1.0))

    for i, effect in enumerate(effects):
        # Apply a newly published source exactly once per seq. On respawn
        # (weather-set change) applied_seqs resets, so current patterns
        # recompile automatically.
        seq = outstate.get(f'live_shader_seq_{i}')
        if seq is not None and seq != state['applied_seqs'][i]:
            ok, err = effect.set_source(
                outstate.get(f'live_shader_source_{i}'))
            state['applied_seqs'][i] = seq
            outstate[f'live_shader_status_{i}'] = {
                'slot': i, 'seq': seq, 'ok': ok, 'error': err,
                'ts': _time.time(),
            }
            if not ok:
                print(f'[shader_live_shader] Slot {i} compile failed '
                      f'(seq {seq}):\n{err}')
            else:
                print(f'[shader_live_shader] Slot {i} compiled + swapped '
                      f'(seq {seq})')

        effect.audio_bass, effect.audio_mid, effect.audio_high = \
            bass, mid, high
        for attr, val in shared.items():
            setattr(effect, attr, val)
        effect.fade = fade

        # Per-slot performance controls published by the web layer.
        effect.intensity = max(0.0, min(1.0,
                                        _f(f'live_shader_intensity_{i}', 1.0)))
        knobs = outstate.get(f'live_shader_knobs_{i}')
        if isinstance(knobs, (list, tuple)):
            vals = []
            for k in range(4):
                try:
                    vals.append(max(0.0, min(1.0, float(knobs[k]))))
                except (IndexError, TypeError, ValueError):
                    vals.append(0.5)
            effect.knobs = vals
        else:
            effect.knobs = [0.5, 0.5, 0.5, 0.5]