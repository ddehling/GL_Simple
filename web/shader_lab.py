"""Shader Lab — natural-language shader generation for the live slot.

Runs entirely on web/worker threads (never the render thread): turns a
plain-language description ("slow purple waves that pulse with the bass")
into a Shadertoy-style ``mainImage`` body for renderer/effects/
live_shader.py, keeps a conversation so follow-ups revise instead of
restart, and repairs its own compile errors when the render thread
reports them.

AUTH — same two transports as the DJ planner copilot and the narrative
editor (tools/dj/planner/copilot.py, tools/narrative_editor_v2_qt.py),
tried in this order:
  1. CLI: the ``claude`` binary (Claude Code) via subprocess — uses the
     existing Claude Code session, NO API key. Preferred when installed.
  2. SDK: the ``anthropic`` package with an explicitly RESOLVED credential
     (the planner's saved key, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
     the `ant` CLI's OAuth token). Never a bare Anthropic() — the SDK
     validates auth at request time, so that constructs fine and then
     throws mid-turn ("Could not resolve authentication method").
Neither resolving -> editor-only mode.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional, Tuple

MODELS = {"opus": "claude-opus-5", "sonnet": "claude-sonnet-5",
          "haiku": "claude-haiku-4-5-20251001"}
DEFAULT_MODEL = "opus"
# Independent layered live slots (slot 0 = bottom). MUST match
# renderer/effects/live_shader.py::NUM_SLOTS (no import — this module
# stays free of GL dependencies); the gate asserts they agree.
NUM_SLOTS = 16
MAX_SOURCE_CHARS = 20000
MAX_HISTORY_MESSAGES = 12       # user+assistant entries kept per session
MIN_CALL_INTERVAL_S = 2.0
MAX_REJECT_REPAIRS = 2          # model-output rejections re-asked before failing
CLI_TIMEOUT_S = 300
MAX_TRANSCRIPT_CHARS = 60000    # CLI transport: cap the rendered history


# --------------------------------------------------------------------------
# Transport resolution — mirrors tools/dj/planner/copilot.py (not imported:
# that module drags in the whole DJ brain, which the web app must not).
# --------------------------------------------------------------------------

def _find_claude_exe():
    """Locate the `claude` CLI (Claude Code). None when not installed."""
    exe = shutil.which("claude")
    if exe:
        return exe
    import glob
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, "AppData", "Roaming", "npm", "claude.cmd"),
        os.path.join(home, "AppData", "Roaming", "npm", "claude"),
        os.path.join(home, ".local", "bin", "claude"),
        "/usr/local/bin/claude",
    ]
    cands += glob.glob(os.path.join(
        home, ".vscode", "extensions",
        "anthropic.claude-code-*", "resources", "native-binary", "claude.exe"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _planner_saved_key():
    """The API key saved from the DJ planner's copilot panel, if any."""
    try:
        path = os.path.join(os.path.expanduser("~"), ".gl_simple_copilot.json")
        with open(path, encoding="utf-8") as f:
            return (json.load(f).get("api_key") or "").strip() or None
    except (OSError, ValueError):
        return None


def _ant_oauth_token():
    """A short-lived OAuth access token from the `ant` CLI, if logged in."""
    exe = shutil.which("ant")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "auth", "print-credentials", "--access-token"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    tok = (r.stdout or "").strip()
    if r.returncode == 0 and tok and " " not in tok and len(tok) > 20:
        return tok
    return None


def _resolve_anthropic_client():
    """Return (client, None) or (None, reason). Never raises."""
    try:
        import anthropic
    except ImportError:
        return None, ("the `anthropic` package isn't installed "
                      "(pip install anthropic)")
    key = _planner_saved_key() or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return anthropic.Anthropic(api_key=key), None
    tok = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if tok:
        return anthropic.Anthropic(auth_token=tok), None
    tok = _ant_oauth_token()
    if tok:
        return anthropic.Anthropic(
            auth_token=tok,
            default_headers={"anthropic-beta": "oauth-2025-04-20"}), None
    return None, ("no Claude access — install the Claude Code CLI, set "
                  "ANTHROPIC_API_KEY, or run `ant auth login`, then restart")


_transport = None       # unresolved; else ('cli', exe) / ('sdk', client) / (None, reason)
_transport_lock = threading.Lock()


def _resolve_transport(refresh: bool = False):
    """Resolve once and cache: ('cli', exe) | ('sdk', client) | (None, reason)."""
    global _transport
    with _transport_lock:
        if _transport is None or refresh:
            exe = _find_claude_exe()
            if exe:
                _transport = ('cli', exe)
            else:
                client, why = _resolve_anthropic_client()
                _transport = ('sdk', client) if client else (None, why)
        return _transport


def _call_claude_cli(exe: str, system: str, transcript: str,
                     model_id: str) -> str:
    """One blocking `claude -p` turn (system prompt via temp file — the
    inline flag hits command-line length limits). Raises on failure."""
    sf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sys.txt", delete=False, encoding="utf-8")
    sf.write(system)
    sf.close()
    try:
        cmd = [exe, "--no-session-persistence", "--model", model_id,
               "--system-prompt-file", sf.name, "--output-format", "text", "-p"]
        r = subprocess.run(cmd, input=transcript, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=CLI_TIMEOUT_S)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "claude failed")
                               .strip()[:300])
        return (r.stdout or "").strip()
    finally:
        try:
            os.unlink(sf.name)
        except OSError:
            pass

# Tokens that must not appear in user shader code. ``uniform``/``layout``
# because the wrapper owns all declarations; ``while``/``do`` as the cheap
# GPU-hang guard; the rest are capabilities the 310-es fullscreen slot
# doesn't provide (textures, depth writes, compute-ish features).
_BANNED_TOKENS = (
    'while', 'do', 'uniform', 'layout', 'sampler', 'texture',
    'gl_FragDepth', 'buffer', 'imageStore', 'atomic', 'barrier',
)
_BANNED_DIRECTIVES = ('#version', '#include', '#extension', '#pragma')

# Functions the engine header already defines (fan geometry + the GLSL
# stdlib in renderer/effects/live_shader.py). A user redefinition would be
# a compile error on the device; rejecting by name here turns that into
# instant feedback the repair loop can act on without a GL round-trip.
STDLIB_NAMES = (
    'fanPos', 'fanUV', 'fanAngle', 'fanRadius',
    'hash11', 'hash21', 'vnoise', 'fbm', 'rot2', 'hsv2rgb', 'palette',
    'sdCircle', 'sdSegment', 'sdBox', 'glow', 'sphereNormal',
)

_MAINIMAGE_RE = re.compile(
    r'void\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*(?:in\s+)?vec2\s+\w+\s*\)')

_COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/', re.DOTALL)


def strip_comments(code: str) -> str:
    """GLSL with comments blanked out. The validator scans REAL code only —
    a harmless '// textured look' or '// do a slow spin' must not trip the
    banned-token scan (it did, and killed valid shaders)."""
    return _COMMENT_RE.sub(' ', code)


def validate_glsl(code: str) -> Tuple[bool, str]:
    """Cheap web-thread screening before code is queued for the GL thread.

    Returns (ok, reason). This is a safety/contract check, not a compiler —
    the real verdict comes from the driver on the render thread.
    """
    if not isinstance(code, str) or not code.strip():
        return False, "Empty shader source."
    if len(code) > MAX_SOURCE_CHARS:
        return False, f"Shader source too long (>{MAX_SOURCE_CHARS} chars)."
    bare = strip_comments(code)
    if not _MAINIMAGE_RE.search(bare):
        return False, ("Missing entry point: define "
                       "'void mainImage(out vec4 fragColor, in vec2 fragCoord)'.")
    for d in _BANNED_DIRECTIVES:
        if d in bare:
            return False, (f"'{d}' is not allowed — the engine owns the "
                           "preprocessor header.")
    for tok in _BANNED_TOKENS:
        if re.search(r'\b' + re.escape(tok) + r'\b', bare):
            return False, (f"'{tok}' is not allowed in Shader Lab code. "
                           "Use for-loops with small constant bounds; the "
                           "engine provides all uniforms.")
    for name in STDLIB_NAMES:
        if re.search(r'\b(?:float|vec[234]|mat[234])\s+' + name + r'\s*\(',
                     bare):
            return False, (f"'{name}' is already provided by the engine's "
                           "built-in library — call it directly, do not "
                           "redefine it.")
    return True, ""


_KNOBS_RE = re.compile(r'^\s*//\s*KNOBS:\s*(.+)$', re.MULTILINE)


def extract_knobs(code) -> list:
    """Knob labels a pattern declares via a `// KNOBS: a | b` comment line.

    Up to four names, mapping to the iKnob0..iKnob3 uniforms in order; []
    when the pattern declares none. The comment travels with the code, so
    saved patterns keep their knobs for free.
    """
    m = _KNOBS_RE.search(code or '')
    if not m:
        return []
    return [n.strip() for n in m.group(1).split('|') if n.strip()][:4]


_EDIT_FENCE_RE = re.compile(r'```edit\s*\n(.*?)```', re.DOTALL)
_EDIT_BODY_RE = re.compile(
    r'<{7} SEARCH\n(.*?)\n?={7}\n(.*?)\n?>{7} REPLACE', re.DOTALL)


def extract_edits(text: str) -> list:
    """SEARCH/REPLACE pairs from ```edit fences in a model reply.

    Refinements reply with these instead of regenerating the full shader —
    the dominant cost of a generation is output tokens, and "make it
    slower" only needs a few lines. Returns [(search, replace), ...].
    """
    pairs = []
    for body in _EDIT_FENCE_RE.findall(text or ''):
        pairs.extend(_EDIT_BODY_RE.findall(body))
    return pairs


def apply_edits(code: str, edits: list) -> Tuple[Optional[str], str]:
    """Apply SEARCH/REPLACE pairs in order. Returns (new_code, '') or
    (None, reason). Each SEARCH must match the current code exactly once —
    same contract as a code editor's find-and-replace."""
    for i, (search, replace) in enumerate(edits, 1):
        if not search.strip():
            return None, f'edit {i} has an empty SEARCH section'
        n = code.count(search)
        if n == 0:
            return None, (f'edit {i}: its SEARCH text does not appear in '
                          'the current shader (must match exactly, '
                          'including whitespace)')
        if n > 1:
            return None, (f'edit {i}: its SEARCH text appears {n} times — '
                          'include more surrounding lines to make it unique')
        code = code.replace(search, replace, 1)
    return code, ''


def extract_glsl(text: str) -> Optional[str]:
    """Pull the single fenced GLSL block out of a model reply."""
    blocks = re.findall(r'```(?:glsl|c)?\s*\n(.*?)```', text, re.DOTALL)
    if len(blocks) == 1:
        return blocks[0].strip()
    # Multiple blocks: take the one containing mainImage if unambiguous.
    hits = [b for b in blocks if 'mainImage' in b]
    if len(hits) == 1:
        return hits[0].strip()
    return None


def _load_contrast_hint() -> str:
    """One-paragraph digest of docs/shader_contrast_playbook.md intent."""
    return (
        "The physical output is globally dimmed to roughly 10% brightness "
        "(power-supply limit). NEVER rely on raw brightness for impact: "
        "build contrast from HUE differences, SPATIAL structure (edges, "
        "bands, moving shapes against dark background), and saturation. "
        "Avoid full-screen near-white fields — they clip into a uniform "
        "dim wash. Dark background + saturated moving features reads best."
    )


SYSTEM_PROMPT = f"""You write GLSL fragment-shader patterns for an LED art \
installation. The person describing the pattern is NOT a programmer — you \
translate their words into working code.

## Output format
Reply with ONE short sentence describing what you made or changed, then \
code in ONE of two forms (never both in the same reply, no other blocks, \
no explanations after):

1. NEW pattern, or a big rewrite → EXACTLY ONE fenced block tagged glsl \
holding the complete shader.
2. SMALL refinement of the current pattern (speed, color, size, one \
feature tweaked) → fenced block(s) tagged edit. Each contains:
```edit
<<<<<<< SEARCH
exact contiguous lines copied from the current shader
=======
the replacement lines
>>>>>>> REPLACE
```
SEARCH must match the current shader EXACTLY (whitespace included) and \
appear exactly once — copy enough surrounding lines to be unique. STRONGLY \
prefer edits for small changes: they are several times faster for the \
person waiting.

## Entry point (write ONLY this function and its helpers)
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord)
```
`fragCoord` is in pixels. Do NOT write `main()`, `#version`, `precision`, \
or any `uniform`/`in`/`out` declarations — the engine owns all of those.

## Canvas & PHYSICAL LAYOUT — important
iResolution is (128, 300), but the canvas is NOT a flat rectangle. The \
installation is a SEMICIRCULAR FAN standing on edge: each of the 128 \
columns is one LED strip, and the strips fan across a 180° arc (column 0 \
points left, the middle column straight up, column 127 right). Each \
strip's 300 rows run outward from the hub (4 ft) to the rim (20.6 ft). \
Drawing in raw pixel/uv space therefore WARPS shapes on the real thing — \
a uv-space circle becomes a bent wedge.

Engine-provided layout helpers (already defined — never redeclare them):
- `vec2 fanUV(fragCoord)` — TRUE physical position: x -1..1 (left..right), \
y 0..1 (ground..top). Distances, circles, and speeds are real here.
- `vec2 fanPos(fragCoord)` — the same in feet (rim = 20.6 ft).
- `float fanAngle(fragCoord)` — this strip's angle: PI (left) .. 0 (right).
- `float fanRadius(fragCoord)` — feet from the hub, 4.0 .. 20.6.

Choosing a space:
- Objects and motion through space — balls, creatures, rain, stars, \
ripples spreading from a point, anything "moving across" or "falling" — \
MUST use `vec2 p = fanUV(fragCoord);`. Physical down = decreasing p.y; \
the hub (bottom center) is at p = vec2(0.0, ~0.19).
- Strip-native looks — rays from the hub, expanding rings/arcs, \
per-strip chases, radial gradients — use raw \
`vec2 uv = fragCoord / iResolution;`: uv.x picks the ray/angle, uv.y is \
distance outward. (An arc on the fan is a horizontal line in uv.)

## Uniforms already provided (use freely, never declare)
- float iTime — seconds, for motion
- vec2 iResolution — (128, 300)
- float iBass, iMid, iHigh — smoothed band energy ~0..2 (1.0 = recent average)
- float iBassPunch, iHighPunch — transient punch envelopes, 0 at rest
- float iBeat — 1.0 on each beat, decaying to 0 ("pulse with the beat")
- float iEnergy — overall music energy 0..~1
- float iDrop — spikes at a drop; float iBuild — rises during a build-up
- float iBPM — tempo; float iPhrase — 0..1 position within the musical phrase
- float iSeason, iWind, iRain — environment 0..1
- float iKnob0..iKnob3 — live performance sliders, see below
(iFade and iIntensity are applied by the engine; ignore them.)

## Performance knobs (use them when the pattern has natural parameters)
iKnob0..iKnob3 are sliders the person can drag WHILE the pattern runs \
(each 0.0..1.0, resting at 0.5). If the pattern has obvious tweakable \
quantities — speed, density, size, hue shift — wire 1 to 4 of them to \
knobs and declare their labels in the FIRST line of the code block:
// KNOBS: speed | sparkle amount
iKnob0 gets the first label, iKnob1 the second, and so on. Only declare \
knobs the code actually uses, and rescale them sensibly in code, e.g. \
`float speed = mix(0.05, 1.5, iKnob0);` so 0.5 lands on a good default.

## Built-in helper library (already defined — NEVER redefine any of these)
- Noise: `hash11(f)`, `hash21(v2)` → 0..1; `vnoise(v2)` value noise 0..1; \
`fbm(v2)` 4-octave fractal noise ~0..1
- Color: `hsv2rgb(vec3(h,s,v))`; `palette(t,a,b,c,d)` = a+b*cos(2π(ct+d)) \
(vec3 args, IQ cosine palette)
- Shapes (signed distance, negative inside): `sdCircle(p,c,r)`, \
`sdSegment(p,a,b)`, `sdBox(p,c,halfSize)`; `glow(d,radius,soft)` → 1 \
inside `radius`, fading to 0 over `soft`
- 3D: `sphereNormal(p,c,r)` → unit normal of a fake-3D ball (vec3(0.0) \
outside; .z faces the viewer) — light it with a dot product
- `rot2(angle)` → mat2 rotation
USE these instead of writing your own — shorter replies reach the LEDs \
several times faster, and redefining a built-in is a rejected reply.

## Hard rules
- GLSL ES 3.10 syntax. Every float literal needs a decimal point (1.0 not 1).
- NO `while` or `do` loops. `for` loops only with SMALL CONSTANT bounds \
(<= 64 iterations).
- No textures, samplers, or derivative tricks; pure procedural math only.
- Output STRAIGHT (non-premultiplied) alpha: `fragColor = vec4(rgb, alpha);` \
Alpha is your layer's transparency over the scene behind it — use alpha < 1.0 \
or 0.0 in empty regions unless the person asks for a full background.
- {_load_contrast_hint()}

## Interpreting requests
- "pulse"/"react to music" → modulate size, brightness, or hue with \
iBassPunch, iBeat, or iEnergy.
- "calm"/"slow" → small iTime multipliers (0.05–0.3); "energetic" → larger.
- "3D" objects (ball, cube, tunnel...) → fake them with pure math: \
`sphereNormal()` + diffuse/specular dot products for balls, or a raymarched \
SDF with a for-loop of <= 64 constant steps. There are NO textures or \
samplers — NEVER call texture(); build all surface detail from the built-in \
noise (`vnoise`/`fbm`). "Spinning" → rotate the shading coordinates (rot2) \
or the light around the shape over iTime.
- Follow-ups revise the PREVIOUS shader — keep everything they didn't ask \
to change.

## Examples
Request: "gentle blue waves rising up the fan"  (strip-native → raw uv)
Reply:
Slow blue sine-waves drift upward, brightening softly on each beat.
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = fragCoord / iResolution;
    float wave = sin(uv.y * 12.0 - iTime * 0.6 + sin(uv.x * 6.0) * 0.8);
    float band = smoothstep(0.2, 0.9, wave);
    vec3 col = mix(vec3(0.0, 0.05, 0.2), vec3(0.1, 0.5, 1.0), band);
    col *= 0.8 + 0.4 * iBeat;
    float alpha = band * 0.85;
    fragColor = vec4(col, alpha);
}}
```

Request: "a green ball drifting around"  (an object → physical fanUV space)
Reply:
A soft green ball wanders the fan, staying truly round, flaring on the beat.
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 p = fanUV(fragCoord);                    // true physical space
    vec2 c = vec2(0.6 * sin(iTime * 0.31),        // wanders left-right
                  0.55 + 0.3 * sin(iTime * 0.23));
    float d = length(p - c);                      // real distance -> round ball
    float ball = smoothstep(0.22 + 0.04 * iBeat, 0.05, d);
    vec3 col = mix(vec3(0.0, 0.1, 0.02), vec3(0.2, 1.0, 0.35), ball);
    fragColor = vec4(col, ball * 0.9);
}}
```

Follow-up request: "make it drift faster"  (small tweak → edit block)
Reply:
The ball now wanders about twice as fast.
```edit
<<<<<<< SEARCH
    vec2 c = vec2(0.6 * sin(iTime * 0.31),        // wanders left-right
                  0.55 + 0.3 * sin(iTime * 0.23));
=======
    vec2 c = vec2(0.6 * sin(iTime * 0.65),        // wanders left-right
                  0.55 + 0.3 * sin(iTime * 0.48));
>>>>>>> REPLACE
```
"""


# ---------------------------------------------------------------------------
# Wish routing — decide what a free-form wish means for the stage, so the
# person NEVER has to pick a slot. Pure function: the gate drives it with
# scripted stage states.
# ---------------------------------------------------------------------------

_WISH_REMOVE = {'remove', 'clear', 'delete', 'stop', 'kill'}
_WISH_ALL = {'everything', 'all', 'stage'}
_WISH_ADDITIVE = {'add', 'also', 'another'}
# Adjustment/filler words that must not count as evidence a wish is about
# a particular layer.
_WISH_STOP = {
    'the', 'a', 'an', 'it', 'and', 'with', 'to', 'of', 'in', 'on', 'that',
    'make', 'more', 'less', 'bit', 'little', 'please', 'much', 'way',
    'slow', 'slower', 'fast', 'faster', 'bright', 'brighter', 'dim',
    'dimmer', 'big', 'bigger', 'small', 'smaller', 'color', 'colors',
    'pattern', 'layer', 'light', 'lights',
}


def _wish_slot_match(words, occupied):
    """The occupied slot the wish most plausibly refers to, or None."""
    best, best_hits = None, 0
    for s, info in occupied.items():
        sw = set()
        for key in ('name', 'desc', 'prompt'):
            sw |= set(re.findall(r'[a-z]+', str(info.get(key) or '').lower()))
        name = info.get('name')
        try:
            from web.shader_patterns import PATTERNS
            if name in PATTERNS:
                sw |= set(PATTERNS[name]['tags'])
        except ImportError:
            pass
        hits = len((words & sw) - _WISH_STOP)
        newer = (best is None or (info.get('since') or 0)
                 > (occupied[best].get('since') or 0))
        if hits > best_hits or (hits == best_hits and hits > 0 and newer):
            best, best_hits = s, hits
    return best if best_hits > 0 else None


def route_wish(prompt, occupied, last_slot=None, num_slots=None):
    """Decide what a wish means. Returns (action, slot, note).

    action: 'clear_all' | 'remove' | 'refine' | 'new'. For 'new', note is
    the display name of a replaced layer when the stage was full (else
    None). ``occupied`` maps slot -> its now-playing metadata dict.
    """
    if num_slots is None:
        num_slots = NUM_SLOTS
    words = set(re.findall(r'[a-z]+', (prompt or '').lower()))
    target = _wish_slot_match(words, occupied)

    if (words & _WISH_REMOVE) and not (words & _WISH_ADDITIVE):
        if words & _WISH_ALL:
            return 'clear_all', None, None
        if target is not None:
            return 'remove', target, None

    additive = bool(words & _WISH_ADDITIVE)
    if target is not None and not additive:
        return 'refine', target, None

    try:
        from web.shader_patterns import PATTERNS
        all_tags = set().union(*(set(p['tags']) for p in PATTERNS.values()))
    except ImportError:
        all_tags = set()
    looks_new = additive or bool(words & all_tags)
    if not looks_new and last_slot is not None and last_slot in occupied:
        return 'refine', last_slot, None   # "a bit slower" → what we just made

    for s in range(num_slots):
        if not occupied.get(s):
            return 'new', s, None
    oldest = min(occupied, key=lambda s: (occupied[s] or {}).get('since') or 0)
    info = occupied[oldest] or {}
    return 'new', oldest, (info.get('name') or info.get('desc')
                           or 'the oldest layer')


class ShaderLabSession:
    """One conversation with the model. v1: a single global session."""

    def __init__(self):
        self._history = []          # [{'role': ..., 'content': ...}]
        self._lock = threading.Lock()   # single in-flight call
        self._last_call = 0.0
        self.model = DEFAULT_MODEL      # key into MODELS
        # The code this conversation currently refers to — the target that
        # ```edit blocks are applied against. Kept canonical in history.
        self._current_glsl = None

    def set_model(self, key):
        """Pick the generation model ('opus' = best, 'sonnet' = fast)."""
        if key in MODELS:
            self.model = key

    # -- availability -----------------------------------------------------

    @staticmethod
    def available() -> bool:
        """Whether LLM generation can work (CLI found or credential resolved)."""
        return _resolve_transport()[0] is not None

    # -- conversation -----------------------------------------------------

    def _render_transcript(self) -> str:
        """History as plain text for the CLI transport (stateless `claude -p`
        calls — the whole conversation rides in the prompt each turn)."""
        parts = [("USER: " if m["role"] == "user" else "ASSISTANT: ")
                 + m["content"] for m in self._history]
        text = "\n\n".join(parts)
        while len(text) > MAX_TRANSCRIPT_CHARS and len(parts) > 1:
            parts.pop(0)
            text = "...\n\n" + "\n\n".join(parts)
        return text

    def reset(self):
        with self._lock:
            self._history = []
            self._current_glsl = None

    def seed(self, code: str, label: Optional[str] = None,
             prompt: str = '') -> bool:
        """Register externally-compiled code (a library load or a hand edit)
        as the conversation's current pattern, so follow-up chat refines IT
        instead of whatever the model last generated. Returns False (and
        does nothing) if a generation is mid-flight."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            src = (f'my saved pattern “{label}”' if label
                   else 'hand-edited shader code')
            note = (f' Its original description was: "{prompt}".'
                    if prompt else '')
            self._history.append({
                'role': 'user',
                'content': (f'I just loaded {src} into the live slot.{note} '
                            'Treat it as the current pattern and revise it '
                            'when I describe changes.')})
            self._history.append({
                'role': 'assistant',
                'content': ('Got it — the current pattern is now:\n'
                            '```glsl\n' + code + '\n```')})
            if len(self._history) > MAX_HISTORY_MESSAGES:
                self._history = self._history[-MAX_HISTORY_MESSAGES:]
                while self._history and self._history[0]['role'] != 'user':
                    self._history.pop(0)
            self._current_glsl = code
            return True
        finally:
            self._lock.release()

    def _turn(self, user_content: str,
              on_text=None) -> Tuple[Optional[str], Optional[str]]:
        """One transport round-trip. Caller holds self._lock. Returns
        (reply_text, error) — exactly one is non-None. History gains the
        user+assistant pair on success and nothing on failure.

        ``on_text(accumulated_text)`` streams the reply as it is written
        (SDK transport only — the CLI returns in one piece)."""
        wait = MIN_CALL_INTERVAL_S - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        mode, handle = _resolve_transport()
        if mode is None:
            return None, f"Claude unavailable: {handle}"
        model_id = MODELS.get(self.model, MODELS[DEFAULT_MODEL])

        self._history.append({"role": "user", "content": user_content})
        # Trim from the front, keeping an even user/assistant boundary.
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]
            while self._history and self._history[0]["role"] != "user":
                self._history.pop(0)

        if mode == 'cli':
            try:
                text = _call_claude_cli(
                    handle, SYSTEM_PROMPT, self._render_transcript(),
                    model_id)
            except Exception as e:
                self._history.pop()     # failed turn leaves no residue
                return None, f"Claude CLI error: {e}"
            finally:
                self._last_call = time.time()
        else:
            try:
                with handle.messages.stream(
                    model=model_id,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    messages=list(self._history),
                ) as stream:
                    if on_text is not None:
                        acc = []
                        for delta in stream.text_stream:
                            acc.append(delta)
                            try:
                                on_text(''.join(acc))
                            except Exception:
                                pass    # progress display must never abort
                    response = stream.get_final_message()
            except Exception as e:
                self._history.pop()     # failed turn leaves no residue
                return None, f"Claude API error: {e}"
            finally:
                self._last_call = time.time()

            if getattr(response, 'stop_reason', None) == 'refusal':
                self._history.pop()
                return None, ("The model declined this request — "
                              "try rephrasing the description.")
            text = "".join(b.text for b in response.content
                           if getattr(b, 'type', '') == 'text')

        self._history.append({"role": "assistant", "content": text})
        return text, None

    def _converse(self, user_content: str, on_text=None):
        """One user-visible turn: call the model, screen its code, and feed
        rejections straight back for up to MAX_REJECT_REPAIRS automatic
        rewrites (mirror of web_controller's driver-error repair loop, but
        for validator/format failures — those used to surface raw as
        'Generated code rejected: ...'). Returns (glsl, description, error);
        exactly one of glsl / error is non-None."""
        if not self._lock.acquire(blocking=False):
            return None, None, "A generation is already running — wait for it."
        try:
            msg, problem = user_content, ""
            for _ in range(1 + MAX_REJECT_REPAIRS):
                text, err = self._turn(msg, on_text=on_text)
                if err:
                    return None, None, err      # transport/refusal: not repairable
                glsl = extract_glsl(text)
                via_edits = False
                if glsl is None and self._current_glsl:
                    edits = extract_edits(text)
                    if edits:
                        glsl, apply_err = apply_edits(self._current_glsl,
                                                      edits)
                        if glsl is None:
                            problem = f"an edit failed to apply ({apply_err})"
                            msg = ("That edit could not be applied: "
                                   + apply_err + ". Reply instead with the "
                                   "FULL corrected shader: one sentence, "
                                   "then exactly ONE glsl block.")
                            continue
                        via_edits = True
                if glsl is not None:
                    ok, reason = validate_glsl(glsl)
                    if ok:
                        description = (text.split('```', 1)[0].strip()
                                       or "New pattern.")
                        if via_edits:
                            # Canonicalize history: store the merged shader
                            # in place of the edit reply, so the next
                            # refinement's SEARCH targets real code.
                            self._history[-1]["content"] = (
                                description + "\n```glsl\n" + glsl + "\n```")
                        self._current_glsl = glsl
                        return glsl, description, ""
                    problem = f"code rejected: {reason}"
                    msg = ("That code was rejected before compiling: "
                           + reason + " Rewrite the FULL shader within the "
                           "hard rules (pure procedural math — no banned "
                           "keywords anywhere) and reply in the same format: "
                           "one sentence, then exactly ONE glsl block.")
                else:
                    problem = "the reply had no usable code"
                    msg = ("Your reply had neither a ```glsl block nor "
                           "applicable ```edit blocks. Reply again: one "
                           "sentence, then exactly ONE glsl block holding "
                           "the complete mainImage shader.")
            return None, None, (f"The model couldn't produce acceptable code "
                                f"after {MAX_REJECT_REPAIRS + 1} attempts "
                                f"({problem}). Try rephrasing the request.")
        finally:
            self._lock.release()

    def generate(self, prompt: str, on_text=None):
        """First request or a refinement — the conversation disambiguates."""
        return self._converse(prompt.strip()[:2000], on_text=on_text)

    def repair(self, compiler_error: str, on_text=None):
        """Feed a driver compile error back for an automatic fix."""
        msg = ("That shader failed to compile on the device. Fix it and "
               "reply in the same format (one sentence + one glsl block). "
               "Compiler output:\n" + compiler_error[:3000])
        return self._converse(msg, on_text=on_text)
