"""
BART Map effect - Bay Area Rapid Transit system visualization.

Draws the BART rail network as a static map with animated trains.
Train motion is physically based:
  - Inter-station travel time is proportional to real haversine distance
    at BART's average speed (BART_SPEED_KMH).
  - Trains dwell at each station for DWELL_TIME_SEC real seconds.
  - train_speed is a wall-clock multiplier (1.0 = true real time,
    40.0 = 40× speedup — recommended for visual use).

No network calls are made; everything is self-contained.
"""

import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List, Tuple
from .base import ShaderEffect

# ---------------------------------------------------------------------------
# Physics constants  (real-world seconds / km)
# ---------------------------------------------------------------------------
BART_SPEED_KMH = 60.0    # average inter-station speed (km/h)
DWELL_TIME_SEC = 20.0    # station stop time (real seconds)

# ===========================================================================
# BART system data (hardcoded — no network calls)
# ===========================================================================

# (name, latitude, longitude)
_STATIONS_RAW: List[Tuple[str, float, float]] = [
    # 0–8  Richmond corridor
    ("Richmond",             37.9367, -122.3532),
    ("El Cerrito del Norte", 37.9252, -122.3169),
    ("El Cerrito Plaza",     37.9024, -122.2995),
    ("North Berkeley",       37.8742, -122.2835),
    ("Berkeley",             37.8698, -122.2683),
    ("Ashby",                37.8527, -122.2700),
    ("MacArthur",            37.8286, -122.2673),
    ("19th St Oakland",      37.8078, -122.2688),
    ("12th St Oakland",      37.8034, -122.2717),
    # 9    West Oakland
    ("West Oakland",         37.8049, -122.2946),
    # 10–18  SF trunk + Daly City
    ("Embarcadero",          37.7929, -122.3969),
    ("Montgomery",           37.7894, -122.4010),
    ("Powell",               37.7844, -122.4079),
    ("Civic Center",         37.7796, -122.4137),
    ("16th St Mission",      37.7651, -122.4199),
    ("24th St Mission",      37.7523, -122.4183),
    ("Glen Park",            37.7329, -122.4342),
    ("Balboa Park",          37.7220, -122.4477),
    ("Daly City",            37.7063, -122.4687),
    # 19–23  Red line peninsula south
    ("Colma",                37.6896, -122.4660),
    ("South SF",             37.6643, -122.4440),
    ("San Bruno",            37.6305, -122.4165),
    ("SFO",                  37.6159, -122.3923),
    ("Millbrae",             37.5994, -122.3862),
    # 24–36  East Bay south
    ("Lake Merritt",         37.7977, -122.2655),
    ("Fruitvale",            37.7747, -122.2243),
    ("Coliseum",             37.7540, -122.1975),
    ("San Leandro",          37.7023, -122.1609),
    ("Bay Fair",             37.6897, -122.1302),
    ("Hayward",              37.6703, -122.0883),
    ("South Hayward",        37.6344, -122.0572),
    ("Union City",           37.5912, -122.0172),
    ("Fremont",              37.5573, -121.9760),
    ("Warm Springs",         37.5018, -121.9393),
    ("Milpitas",             37.4281, -121.9003),
    ("Berryessa",            37.3848, -121.8741),
    ("North San José",       37.3590, -121.8681),
    # 37–46  Yellow / Antioch line
    ("Bay Point",            37.9907, -121.9402),
    ("Pittsburg Center",     37.9990, -121.8839),
    ("Antioch",              37.9950, -121.7830),
    ("North Concord",        37.9763, -122.0296),
    ("Concord",              37.9728, -122.0319),
    ("Pleasant Hill",        37.9280, -122.0572),
    ("Walnut Creek",         37.9056, -122.0674),
    ("Lafayette",            37.8937, -122.1237),
    ("Orinda",               37.8782, -122.1827),
    ("Rockridge",            37.8440, -122.2518),
    # 47–49  Blue line east branch
    ("Castro Valley",        37.6921, -122.0750),
    ("West Dublin/Pleasanton", 37.6990, -121.9278),
    ("Dublin/Pleasanton",    37.7016, -121.8996),
]

# Geographic bounding box (slightly padded beyond the network extent)
_LAT_MIN, _LAT_MAX = 37.32, 38.02
_LON_MIN, _LON_MAX = -122.52, -121.72

# Lines: (name, RGB color tuple, ordered station indices)
_LINES: List[Tuple[str, Tuple[float, float, float], List[int]]] = [
    # Red  Richmond ↔ SFO/Millbrae
    ("Red",    (0.93, 0.11, 0.14),
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23]),
    # Orange  Richmond ↔ Berryessa/North San José  (East Bay only)
    ("Orange", (0.98, 0.52, 0.00),
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
      34, 35, 36]),
    # Yellow  Antioch ↔ Daly City  (via Concord / Walnut Creek)
    ("Yellow", (1.00, 0.85, 0.00),
     [39, 38, 37, 40, 41, 42, 43, 44, 45, 46, 6, 7, 8, 9, 10, 11, 12, 13,
      14, 15, 16, 17, 18]),
    # Green  Daly City ↔ Berryessa/North San José  (cross-bay)
    ("Green",  (0.36, 0.65, 0.24),
     [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33, 34, 35, 36]),
    # Blue  Dublin/Pleasanton ↔ Daly City
    ("Blue",   (0.09, 0.69, 0.90),
     [49, 48, 47, 28, 27, 26, 25, 24, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      17, 18]),
]

# Number of trains on each line (matches _LINES order)
_TRAIN_COUNTS = [5, 5, 5, 5, 4]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2.0 * np.arcsin(np.sqrt(a))


def _geo_to_norm(lat: float, lon: float, pad: float = 0.05) -> Tuple[float, float]:
    """Normalize lat/lon to [pad, 1-pad] in both axes (y=0 south, y=1 north)."""
    x = pad + (1 - 2 * pad) * (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)
    y = pad + (1 - 2 * pad) * (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
    return x, y


# ===========================================================================
# Event wrapper
# ===========================================================================

def shader_bart_map(state, outstate, train_speed=40.0, train_density=1.0):
    """
    BART Map background effect — compatible with EventScheduler.

    Reads from outstate:
        train_speed   (float) real-time multiplier (1.0 = wall-clock real,
                              40.0 = 40× speedup, recommended default)
        train_density (float) not yet used; reserved for future headway scaling
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(
                BARTMapEffect,
                train_speed=train_speed,
                train_density=train_density,
            )
            state['effect'] = effect
            print(f"✓ Initialized BART map for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize BART map: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        state['effect'].train_speed = outstate.get('train_speed', train_speed)
        state['effect'].set_train_density(outstate.get('train_density', train_density))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ===========================================================================
# ShaderEffect implementation
# ===========================================================================

class BARTMapEffect(ShaderEffect):
    """
    Renders the BART rail map with physically-timed animated trains.

    Train simulation:
      - pos  (float): fractional station index along the line's sequence.
                      Integer value = train is at that station.
      - dirs (±1):    direction of travel along the sequence.
      - dwell (float): real-seconds remaining at current station stop.
                       > 0 → train is stopped; 0 → train is moving.

    Each frame:  scaled_dt = dt * train_speed
      Dwelling trains count down their dwell.
      Moving trains advance by  scaled_dt / seg_travel_time[segment],
      where seg_travel_time is derived from haversine distance ÷ BART_SPEED_KMH.
      On reaching any integer position the train enters DWELL_TIME_SEC dwell.
      At termini the direction reverses.

    Visual:  dwelling trains render larger and brighter.
    """

    # ------------------------------------------------------------------
    # GLSL sources
    # ------------------------------------------------------------------

    _LINE_VERT = """
    #version 310 es
    precision highp float;
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec3 aColor;
    out vec3 vColor;
    uniform vec2 uResolution;
    void main() {
        vec2 clip = (aPos / uResolution) * 2.0 - 1.0;
        gl_Position = vec4(clip, 0.0, 1.0);
        vColor = aColor;
    }
    """

    _LINE_FRAG = """
    #version 310 es
    precision highp float;
    in vec3 vColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(vColor, 0.80);
    }
    """

    # Instance layout (8 floats): px py radius r g b alpha glow
    _CIRCLE_VERT = """
    #version 310 es
    precision highp float;
    layout(location = 0) in vec2  aQuad;
    layout(location = 1) in vec2  iPos;
    layout(location = 2) in float iRadius;
    layout(location = 3) in vec3  iColor;
    layout(location = 4) in float iAlpha;
    layout(location = 5) in float iGlow;
    out vec2  vQuad;
    out vec3  vColor;
    out float vAlpha;
    out float vGlow;
    uniform vec2 uResolution;
    void main() {
        vec2 world = iPos + aQuad * iRadius;
        vec2 clip  = (world / uResolution) * 2.0 - 1.0;
        gl_Position = vec4(clip, 0.0, 1.0);
        vQuad  = aQuad;
        vColor = iColor;
        vAlpha = iAlpha;
        vGlow  = iGlow;
    }
    """

    _CIRCLE_FRAG = """
    #version 310 es
    precision highp float;
    in vec2  vQuad;
    in vec3  vColor;
    in float vAlpha;
    in float vGlow;
    out vec4 FragColor;
    void main() {
        float d    = length(vQuad);
        if (d > 1.0) discard;
        float core = smoothstep(1.0, 0.5, d);
        float halo = vGlow * smoothstep(1.0, 0.0, d) * 0.6;
        float a    = vAlpha * clamp(core + halo, 0.0, 1.0);
        FragColor  = vec4(vColor, a);
    }
    """

    _INST_STRIDE = 8 * 4  # bytes per instance

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, viewport, train_speed: float = 40.0,
                 train_density: float = 1.0):
        super().__init__(viewport)
        self.train_speed   = train_speed   # real-time multiplier
        self.train_density = train_density
        self.render_priority = 5

        w, h = float(viewport.width), float(viewport.height)

        self._station_px = np.array(
            [(_geo_to_norm(lat, lon)[0] * w,
              _geo_to_norm(lat, lon)[1] * h)
             for _, lat, lon in _STATIONS_RAW],
            dtype=np.float32,
        )

        self._line_width = max(2.0, w * 0.009)
        self._station_r  = max(2.5, w * 0.007)
        self._train_r    = max(4.0, w * 0.014)

        self._bake_line_geometry()
        self._init_trains()

        # GPU handles (filled by compile_shader / setup_buffers)
        self.circle_shader    = None
        self._line_VAO        = None
        self._line_VBO        = None
        self._circle_VAO      = None
        self._circle_quad_VBO = None
        self._circle_inst_VBO = None
        self._n_line_verts    = 0
        self._max_instances   = 0
        self._station_inst    = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _bake_line_geometry(self):
        """Pre-compute thick-line quad vertices for every BART line."""
        self._line_chunks: List[np.ndarray] = []
        lw_half = self._line_width * 0.5

        for _name, color, seq in _LINES:
            verts: List[float] = []
            cr, cg, cb = color
            for i in range(len(seq) - 1):
                a = self._station_px[seq[i]]
                b = self._station_px[seq[i + 1]]
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = np.hypot(dx, dy)
                if L < 0.5:
                    continue
                nx, ny = -dy / L * lw_half, dx / L * lw_half
                verts += [
                    a[0]-nx, a[1]-ny, cr, cg, cb,
                    a[0]+nx, a[1]+ny, cr, cg, cb,
                    b[0]+nx, b[1]+ny, cr, cg, cb,
                    a[0]-nx, a[1]-ny, cr, cg, cb,
                    b[0]+nx, b[1]+ny, cr, cg, cb,
                    b[0]-nx, b[1]-ny, cr, cg, cb,
                ]
            self._line_chunks.append(
                np.array(verts, dtype=np.float32) if verts else np.zeros(0, np.float32)
            )

    def _init_trains(self):
        """
        Initialise train state for every line.

        seg_times[i] = real seconds to travel segment i→i+1,
                       based on haversine distance / BART_SPEED_KMH.
        """
        self._trains: List[dict] = []
        for line_idx, (_name, color, seq) in enumerate(_LINES):
            n     = _TRAIN_COUNTS[line_idx]
            n_seg = len(seq) - 1

            # Per-segment travel time (real seconds)
            seg_times = []
            for i in range(n_seg):
                s0 = _STATIONS_RAW[seq[i]]
                s1 = _STATIONS_RAW[seq[i + 1]]
                km = _haversine_km(s0[1], s0[2], s1[1], s1[2])
                seg_times.append(km / BART_SPEED_KMH * 3600.0)

            # Distribute trains: mid-segment start to avoid instant terminus
            pos  = np.linspace(0.5, n_seg - 0.5, n, dtype=np.float64)
            dirs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
            # No initial dwell — trains start moving
            dwell = np.zeros(n, dtype=np.float64)

            self._trains.append({
                'pos':       pos,
                'dirs':      dirs,
                'dwell':     dwell,
                'n_seg':     n_seg,
                'seq':       seq,
                'seg_times': seg_times,   # real seconds per segment
                'color':     np.array(color, dtype=np.float32),
            })

    # ------------------------------------------------------------------
    # ShaderEffect — compile_shader
    # ------------------------------------------------------------------

    def compile_shader(self):
        line_shader = shaders.compileProgram(
            shaders.compileShader(self._LINE_VERT,   GL_VERTEX_SHADER),
            shaders.compileShader(self._LINE_FRAG,   GL_FRAGMENT_SHADER),
        )
        self.circle_shader = shaders.compileProgram(
            shaders.compileShader(self._CIRCLE_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(self._CIRCLE_FRAG, GL_FRAGMENT_SHADER),
        )
        return line_shader   # stored as self.shader by base init()

    # ------------------------------------------------------------------
    # ShaderEffect — setup_buffers
    # ------------------------------------------------------------------

    def setup_buffers(self):
        # Static line quad geometry
        all_verts = (np.concatenate(self._line_chunks)
                     if self._line_chunks else np.zeros(0, np.float32))
        self._n_line_verts = len(all_verts) // 5

        self._line_VAO = glGenVertexArrays(1)
        glBindVertexArray(self._line_VAO)
        self._line_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._line_VBO)
        glBufferData(GL_ARRAY_BUFFER, all_verts.nbytes, all_verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))
        glBindVertexArray(0)
        self.VBOs.append(self._line_VBO)

        # Instanced circle geometry
        quad = np.array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1],
                        dtype=np.float32).reshape(-1, 2)

        self._circle_VAO = glGenVertexArrays(1)
        glBindVertexArray(self._circle_VAO)

        self._circle_quad_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._circle_quad_VBO)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(self._circle_quad_VBO)

        S = self._INST_STRIDE
        n_stations   = len(_STATIONS_RAW)
        n_trains_max = sum(_TRAIN_COUNTS) + 16
        self._max_instances = n_stations + n_trains_max

        self._circle_inst_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._circle_inst_VBO)
        glBufferData(GL_ARRAY_BUFFER, self._max_instances * S, None, GL_DYNAMIC_DRAW)

        # iPos (loc 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        # iRadius (loc 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(8))
        glVertexAttribDivisor(2, 1)
        # iColor (loc 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(12))
        glVertexAttribDivisor(3, 1)
        # iAlpha (loc 4)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(24))
        glVertexAttribDivisor(4, 1)
        # iGlow (loc 5)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(28))
        glVertexAttribDivisor(5, 1)

        glBindVertexArray(0)
        self.VBOs.append(self._circle_inst_VBO)

        # Pre-build static station instance rows
        n  = len(_STATIONS_RAW)
        sd = np.zeros((n, 8), dtype=np.float32)
        sd[:, 0] = self._station_px[:, 0]
        sd[:, 1] = self._station_px[:, 1]
        sd[:, 2] = self._station_r
        sd[:, 3:6] = [0.95, 0.95, 0.95]
        sd[:, 6] = 0.65
        sd[:, 7] = 0.10
        self._station_inst = sd

    # ------------------------------------------------------------------
    # ShaderEffect — update  (train simulation)
    # ------------------------------------------------------------------

    def set_train_density(self, density: float):
        self.train_density = float(density)

    def update(self, dt: float, state: Dict):
        """
        Advance every train by dt seconds of animation time.

        scaled_dt = dt * train_speed  converts animation time to
        equivalent real-world seconds for physics calculations.
        """
        scaled_dt = dt * self.train_speed

        for td in self._trains:
            seg_times = td['seg_times']
            n_seg     = td['n_seg']

            for i in range(len(td['pos'])):
                # ---- Dwell phase ----
                if td['dwell'][i] > 0.0:
                    td['dwell'][i] = max(0.0, td['dwell'][i] - scaled_dt)
                    continue

                pos = td['pos'][i]
                d   = td['dirs'][i]

                # Which segment's travel time applies?
                # Nudge pos slightly in the reverse direction before floor()
                # so a train sitting exactly at station k going backward
                # uses the segment k-1 → k (not k → k+1).
                seg_idx = int(np.clip(
                    pos - (1e-4 if d < 0 else 0.0),
                    0, n_seg - 1
                ))
                step = d * scaled_dt / seg_times[seg_idx]
                new_pos = pos + step

                # ---- Terminus check (bounce + dwell) ----
                if new_pos >= n_seg:
                    td['pos'][i]   = float(n_seg)
                    td['dirs'][i]  = -1.0
                    td['dwell'][i] = DWELL_TIME_SEC
                    continue
                if new_pos <= 0.0:
                    td['pos'][i]   = 0.0
                    td['dirs'][i]  = 1.0
                    td['dwell'][i] = DWELL_TIME_SEC
                    continue

                # ---- Intermediate station check ----
                # A train that is exactly at an integer has *just* left a
                # station; it should not immediately re-trigger a dwell.
                at_integer = abs(pos - round(pos)) < 1e-9

                if not at_integer:
                    if d > 0 and int(new_pos) > int(pos):
                        # Crossed a station travelling forward
                        td['pos'][i]   = float(int(new_pos))
                        td['dwell'][i] = DWELL_TIME_SEC
                        continue
                    if d < 0 and int(new_pos) < int(pos):
                        # Crossed a station travelling backward
                        # The station crossed is floor(old pos)
                        td['pos'][i]   = float(int(pos))
                        td['dwell'][i] = DWELL_TIME_SEC
                        continue

                td['pos'][i] = new_pos

    # ------------------------------------------------------------------
    # ShaderEffect — render
    # ------------------------------------------------------------------

    def _train_world_positions(self, line_idx: int) -> List[Tuple[np.ndarray, bool]]:
        """
        Return (pixel_pos, is_dwelling) for each train on the line.
        When dwelling the train sits exactly at its station pixel.
        """
        td  = self._trains[line_idx]
        seq = td['seq']
        out = []
        for i, p in enumerate(td['pos']):
            seg  = int(np.clip(p, 0, len(seq) - 2))
            frac = p - seg
            a    = self._station_px[seq[seg]]
            b    = self._station_px[seq[min(seg + 1, len(seq) - 1)]]
            out.append((a + frac * (b - a), td['dwell'][i] > 0.0))
        return out

    def render(self, state: Dict):
        super().render(state)
        if self.circle_shader is None:
            return

        w = float(self.viewport.width)
        h = float(self.viewport.height)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)

        # ---- 1. Draw BART line quads ----
        glUseProgram(self.shader)
        loc = glGetUniformLocation(self.shader, "uResolution")
        if loc != -1:
            glUniform2f(loc, w, h)
        glBindVertexArray(self._line_VAO)
        glDrawArrays(GL_TRIANGLES, 0, self._n_line_verts)
        glBindVertexArray(0)

        # ---- 2. Build per-frame instance data ----
        train_rows: List[List[float]] = []
        for line_idx, td in enumerate(self._trains):
            c = td['color']
            for wp, dwelling in self._train_world_positions(line_idx):
                if dwelling:
                    # Larger, brighter glow while stopped at a station
                    r, alpha, glow = self._train_r * 1.35, 1.0, 1.2
                else:
                    r, alpha, glow = self._train_r, 0.95, 0.80
                train_rows.append([wp[0], wp[1], r,
                                   c[0], c[1], c[2], alpha, glow])

        if train_rows:
            train_arr = np.array(train_rows, dtype=np.float32)
            instances = np.vstack([self._station_inst, train_arr])
        else:
            instances = self._station_inst

        n_inst = min(len(instances), self._max_instances)
        data   = instances[:n_inst].astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self._circle_inst_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)

        # ---- 3. Draw circles ----
        glUseProgram(self.circle_shader)
        loc = glGetUniformLocation(self.circle_shader, "uResolution")
        if loc != -1:
            glUniform2f(loc, w, h)
        glBindVertexArray(self._circle_VAO)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, n_inst)
        glBindVertexArray(0)
        glUseProgram(0)

        glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------
    # ShaderEffect — cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        try:
            if self._line_VAO:
                glDeleteVertexArrays(1, [self._line_VAO])
            if self._circle_VAO:
                glDeleteVertexArrays(1, [self._circle_VAO])
            if self.circle_shader:
                glDeleteProgram(self.circle_shader)
        except Exception:
            pass
        super().cleanup()   # handles self.VBOs and self.shader
