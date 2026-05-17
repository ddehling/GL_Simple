"""
Forest birds — dark silhouettes flying across the upper sky of the
fan, above the canopy. Pattern B (fullscreen quad, no depth writes).
Same CPU-managed creature-list pattern as projects/fan/shaders/
desert_creatures.py, but birds-only and tuned for forest.

Renders IN FRONT of forest_canopy (depth 0.08 vs canopy 0.15) so
birds visibly pass across the leaves rather than getting obscured.
Reasons in fan-cartesian space so a "V-shape" silhouette is a real
horizontal V on the physical fan, not a radially-stretched smear.

Birds spawn from off-fan in feet, drift across, and despawn on the
opposite side. Each spawn has a 30% chance of bringing a small flock
of 2-4 wingmates trailing the leader. Day spawn rate is much higher
than night; complete silence at deep night when forest birds rest.

Dark silhouette color so the budget cost is near-zero (under
brightness_limit=0.1, dark-against-sky silhouettes are free contrast —
the sky pays the energy, the birds cost nothing). See
docs/shader_contrast_playbook.md "Silhouette against a band".
"""
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords


# Hard cap on birds alive at once (matches the shader array size).
# Bumped iteratively: 24 -> 96 -> 160 to accommodate migration state's
# increasing density. 160 vec4s of uniform storage is comfortable for
# ES3.10 (well under the 256-vector minimum required).
MAX_BIRDS = 160

# Bird altitude in physical feet. Full visible vertical range — from
# the inner ring (r = 4 ft, the inner edge of the LED area) to the
# outer ring (r = 20.6 ft). Below 4 ft is the empty inner zone with
# no LEDs, so spawning birds there would just waste slots.
ALT_MIN = 4.0
ALT_MAX = 20.6

# Spawn just off the fan edges so birds enter visibly.
SPAWN_X = 21.0

VERTEX_SHADER = """#version 310 es
precision highp float;

in vec2 position;
out vec2 v_uv;

void main() {
    v_uv = position * 0.5 + 0.5;
    // Depth 0.08 — in front of forest_canopy (0.15) so birds visually
    // pass across the leaves, but behind the global post-process layer.
    gl_Position = vec4(position, 0.08, 1.0);
}
"""

FRAGMENT_SHADER = f"""#version 310 es
precision highp float;

#define MAX_BIRDS {MAX_BIRDS}

in vec2 v_uv;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

// Each bird is one vec4: (x_ft, y_ft, scale, brightness).
//   scale is a 0.7..1.3 size multiplier so flocks read as varied sizes.
//   brightness handles spawn/despawn fade plus master fade.
uniform vec4 u_birds[MAX_BIRDS];
uniform int  u_active_count;
uniform float u_time;

// V-shape silhouette in fan-cartesian space at ``center``, scaled.
// Wingspan ~1.7 ft scaled (about 1.5x the previous version) with
// thicker wings and an explicit body so birds read clearly as birds
// rather than thin V lines.
vec4 draw_bird(vec2 phys, vec4 bird) {{
    vec2 center = bird.xy;
    float scale = bird.z;
    float bright = bird.w;
    if (bright < 0.005) return vec4(0.0);

    vec2 d = phys - center;
    // Slight time-based wing-flap modulating the M-slope so birds
    // don't read as static cutouts. Frequency varies per bird via
    // its x position so they're not all flapping in sync.
    float flap = sin(u_time * 4.5 + center.x * 1.3) * 0.18;
    float slope = (0.40 + flap) / scale;

    // M-shape: wings extend DOWN-and-out from the body, like a soaring
    // bird seen from below. The two line equations y = ±slope·x each
    // pass through two quadrants; clipping to ``d.y <= 0`` keeps only
    // the lower halves so we get an M not an X (the previous version
    // drew both full lines, producing an X cutout instead of a bird).
    //
    //   left wing  — line y = slope·x   (going through lower-left)
    //   right wing — line y = -slope·x  (going through lower-right)
    float wing_l = abs(d.y - d.x * slope);
    float wing_r = abs(d.y + d.x * slope);
    float span = 0.85 * scale;
    float wing_extent = step(abs(d.x), span);
    float thickness_lo = 0.14 * scale;
    float thickness_hi = 0.04 * scale;
    // ``step(d.y, 0.0)`` returns 1 when d.y <= 0 (lower half).
    float lower_half = step(d.y, 0.0);
    float wing = smoothstep(thickness_lo, thickness_hi,
                            min(wing_l, wing_r))
               * wing_extent
               * lower_half;

    // Body: small ellipse at center — wider in x than y for a bird
    // torso seen from below. Renders in both halves so the wing apex
    // has something to meet at.
    float body_x = d.x / (0.18 * scale);
    float body_y = d.y / (0.10 * scale);
    float body_r2 = body_x * body_x + body_y * body_y;
    float body = smoothstep(1.0, 0.5, body_r2);

    float silh = max(wing, body);
    if (silh < 0.005) return vec4(0.0);

    // Pure black silhouette — maximum contrast against the bright sky
    // backdrop. (Previously (0.10, 0.07, 0.05) warm brown read as
    // very subtle on the actual hardware.)
    vec3 col = vec3(0.0, 0.0, 0.0);
    return vec4(col, silh * bright);
}}

void main() {{
    vec2 phys = fan_uv_to_physical(v_uv);
    vec4 acc = vec4(0.0);

    for (int i = 0; i < MAX_BIRDS; i++) {{
        if (i >= u_active_count) break;
        vec4 contrib = draw_bird(phys, u_birds[i]);
        // Straight-alpha "over" compositing.
        float a_out = contrib.a + acc.a * (1.0 - contrib.a);
        if (a_out > 0.0) {{
            acc.rgb = (contrib.rgb * contrib.a + acc.rgb * acc.a * (1.0 - contrib.a)) / a_out;
            acc.a   = a_out;
        }}
    }}

    if (acc.a < 0.005) discard;
    fragColor = acc;
}}
"""


# ===========================================================================
# Event wrapper
# ===========================================================================

def shader_forest_birds(state, outstate, fade_duration=4.0):
    """Background event: dark bird silhouettes flying above the canopy."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(ForestBirdsEffect)
            state['effect'] = effect
        except Exception as e:
            import traceback
            print(f"ERROR initializing forest_birds: {e}")
            traceback.print_exc()
            return

    if 'effect' in state:
        eff = state['effect']
        # Daytime / nighttime modulation derived from starryness (rises
        # toward 1 at night). Day rate = max, night rate = 0.
        eff.day_factor = 1.0 - float(outstate.get('starryness', 0.0))
        # Per-state density multiplier (default 1.0). Migration state
        # sets this to 5.0 for a dense bird sky.
        eff.density_mult = max(0.0, float(outstate.get('bird_density', 1.0)))

        elapsed = state['elapsed_time']
        duration = state.get('duration')
        if duration is None or duration <= 0:
            eff.fade = 1.0
        else:
            if elapsed < fade_duration:
                f = elapsed / fade_duration
            elif elapsed > duration - fade_duration:
                f = (duration - elapsed) / fade_duration
            else:
                f = 1.0
            eff.fade = float(np.clip(f, 0.0, 1.0))

    if state['count'] == -1:
        if 'effect' in state:
            eff = state['effect']
            if eff in viewport.effects:
                viewport.effects.remove(eff)
            eff.cleanup()
            del state['effect']


# ===========================================================================
# Effect
# ===========================================================================

class _Bird:
    __slots__ = ('x', 'y', 'vx', 'vy', 'scale', 'life', 'max_life')

    def __init__(self, x, y, vx, vy, scale, max_life):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.scale = float(scale)
        self.life = 0.0
        self.max_life = float(max_life)


class ForestBirdsEffect(ShaderEffect):
    """Fullscreen-quad painter for the bird flock."""

    def __init__(self, viewport):
        super().__init__(viewport)
        # Higher than forest_canopy's render_priority (7.0) so birds
        # paint ON TOP of the leaves rather than under them. The
        # earlier gl_Position.z = 0.08 isn't enough because the group
        # canvas sorts by ``render_priority``, not depth, when depth
        # writes are disabled.
        self.render_priority = 8.0
        self.day_factor = 1.0
        self.density_mult = 1.0
        self.fade = 0.0
        self._time = 0.0
        self._spawn_acc = 0.0
        self._birds: list = []
        self._fan = FanCoords(viewport.width, viewport.height)
        self._upload = np.zeros((MAX_BIRDS, 4), dtype=np.float32)

    def compile_shader(self):
        v = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        f = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(v, f)

    def setup_buffers(self):
        verts = np.array([-1, -1,  1, -1,  -1, 1,  1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        self.VBOs.append(vbo)
        loc = glGetAttribLocation(self.shader, "position")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        glUseProgram(self.shader)
        self._fan.set_uniforms(self.shader)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # Bird lifecycle
    # ------------------------------------------------------------------

    def _spawn_one(self, leader_dir=None, leader_y=None, leader_speed=None):
        """Spawn a single bird; if leader_* args are given, this is a
        wingmate joining a flock."""
        if leader_dir is None:
            rtl = np.random.random() < 0.5
        else:
            rtl = leader_dir < 0
        sign = -1.0 if rtl else 1.0
        x0 = SPAWN_X if rtl else -SPAWN_X

        if leader_y is None:
            y = float(np.random.uniform(ALT_MIN, ALT_MAX))
        else:
            # Wingmate stays close to leader's altitude. Wider jitter
            # (was 0.6, now 1.2) so flocks spread vertically across the
            # wider altitude band rather than tight clumps.
            y = float(np.clip(leader_y + np.random.uniform(-1.2, 1.2),
                              ALT_MIN, ALT_MAX))

        if leader_speed is None:
            speed = float(np.random.uniform(1.0, 2.0))
        else:
            speed = leader_speed * float(np.random.uniform(0.95, 1.08))

        # Wingmates lag the leader a few feet in -sign x (entering
        # behind). Leaders enter at full x0.
        if leader_dir is not None:
            x0 = x0 - sign * float(np.random.uniform(1.0, 3.0))

        scale = float(np.random.uniform(0.75, 1.20))
        vx = sign * speed
        vy = float(np.random.uniform(-0.05, 0.05))
        life = 45.0
        self._birds.append(_Bird(x0, y, vx, vy, scale, life))

    def _try_spawn(self, dt: float):
        if len(self._birds) >= MAX_BIRDS:
            return
        # 0.35 birds/sec at full daytime, multiplied by per-state
        # density (1.0 default, 5.0 in migration). With ~45 s lifetime
        # and flock spawns, midday holds ~8-12 active and migration
        # pushes toward the MAX_BIRDS cap.
        rate = 0.35 * max(0.0, self.day_factor) * self.density_mult
        self._spawn_acc += dt * rate
        if self._spawn_acc >= 1.0 and len(self._birds) < MAX_BIRDS:
            self._spawn_acc -= 1.0
            if np.random.random() < 0.30:
                # Flock spawn: leader + 2-4 wingmates.
                leader_sign = -1.0 if np.random.random() < 0.5 else 1.0
                lead_y = float(np.random.uniform(ALT_MIN, ALT_MAX))
                lead_speed = float(np.random.uniform(1.0, 2.0))
                # Leader
                self._spawn_one()  # plain leader (random direction & altitude)
                # Match the just-spawned leader's actual values
                lead = self._birds[-1]
                n_followers = int(np.random.randint(2, 5))
                for _ in range(n_followers):
                    if len(self._birds) >= MAX_BIRDS:
                        break
                    self._spawn_one(leader_dir=lead.vx,
                                    leader_y=lead.y,
                                    leader_speed=abs(lead.vx))
            else:
                self._spawn_one()

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt
        # Advance live birds, prune expired or out-of-fan.
        alive = []
        for b in self._birds:
            b.life += dt
            b.x += b.vx * dt
            b.y += b.vy * dt
            if b.life >= b.max_life:
                continue
            if abs(b.x) > SPAWN_X + 2.0:
                continue
            alive.append(b)
        self._birds = alive
        self._try_spawn(dt)

    def render(self, state: Dict):
        if not self.enabled or self.fade < 0.01:
            return
        if not self._birds:
            return

        n = min(len(self._birds), MAX_BIRDS)
        for i in range(n):
            b = self._birds[i]
            # Spawn/despawn fade — quick ramp at lifetime endpoints.
            ramp = 1.0
            if b.life < 1.0:
                ramp = b.life
            elif b.life > b.max_life - 2.0:
                ramp = max(0.0, (b.max_life - b.life) * 0.5)
            self._upload[i, 0] = b.x
            self._upload[i, 1] = b.y
            self._upload[i, 2] = b.scale
            self._upload[i, 3] = float(np.clip(ramp, 0.0, 1.0)) * self.fade

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        self._fan.set_uniforms(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1i(glGetUniformLocation(self.shader, "u_active_count"), n)
        glUniform4fv(glGetUniformLocation(self.shader, "u_birds"),
                     MAX_BIRDS, self._upload.flatten())

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
