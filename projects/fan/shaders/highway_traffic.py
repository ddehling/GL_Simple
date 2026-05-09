"""
Bay Area Highway Traffic effect.

Draws major Bay Area freeways and bridges as thick polylines colored
green / yellow / red based on simulated traffic congestion.

Congestion is fully self-contained — no network calls:
  - Each segment has a base congestion level derived from its proximity to
    known Bay Area bottlenecks (Bay Bridge, Maze interchange, etc.).
  - Slow sinusoidal noise (different phase/frequency per segment) creates
    organic ebb-and-flow.
  - traffic_density (0–1 from weather state) scales the overall level:
      0.1  = late-night (nearly all green)
      0.45 = off-peak   (green/yellow mix)
      0.85 = peak hour  (yellow/red, bridge approaches solid red)
      0.95 = disrupted  (gridlock — drivers avoiding broken BART)

Color mapping:
    0.0 → green  (0, 1, 0)
    0.5 → yellow (1, 1, 0)
    1.0 → red    (1, 0, 0)

Render priority 3 — draws under BART lines (priority 5).
"""

import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List, Tuple
from renderer.effects.base import ShaderEffect
from .bart_map import _geo_to_fan_px, _haversine_km

# ===========================================================================
# Highway geometry data
# ===========================================================================

# (name, width_multiplier, base_congestion, [(lat, lon), ...])
#  width_multiplier: relative to _ROAD_BASE_WIDTH
#  base_congestion:  0–1 floor before hotspot and density scaling
_HIGHWAYS: List[Tuple[str, float, float, List[Tuple[float, float]]]] = [

    # I-80  Bay Bridge + East Bay spine → Richmond
    ("I-80", 1.00, 0.60, [
        (37.773, -122.410),  # SF/SoMa (US-101 merge)
        (37.783, -122.399),  # Bay Bridge SF ramp
        (37.791, -122.386),  # Bay Bridge SF toll plaza
        (37.812, -122.373),  # Bay Bridge center span
        (37.824, -122.357),  # Bay Bridge Oakland touchdown
        (37.826, -122.272),  # Maze interchange
        (37.846, -122.274),  # Emeryville
        (37.870, -122.281),  # Berkeley
        (37.894, -122.296),  # Albany
        (37.916, -122.318),  # El Cerrito
        (37.937, -122.352),  # Richmond
    ]),

    # I-880  Nimitz Freeway, Oakland → Fremont
    ("I-880", 0.90, 0.48, [
        (37.826, -122.272),  # Maze
        (37.807, -122.270),  # Downtown Oakland
        (37.769, -122.227),  # Oakland/Coliseum corridor
        (37.703, -122.162),  # San Leandro
        (37.665, -122.091),  # Hayward
        (37.634, -122.058),  # South Hayward
        (37.591, -122.017),  # Union City
        (37.557, -121.977),  # Fremont
    ]),

    # I-580  MacArthur Freeway, Oakland → Livermore
    ("I-580", 0.90, 0.48, [
        (37.826, -122.272),  # Maze
        (37.805, -122.228),  # East Oakland
        (37.780, -122.172),  # Oakland Hills
        (37.694, -122.079),  # Castro Valley
        (37.683, -121.935),  # Dublin
        (37.680, -121.780),  # Livermore
    ]),

    # I-680  Benicia → Concord → Walnut Creek → San Jose
    ("I-680", 0.85, 0.38, [
        (38.003, -122.074),  # Martinez / N Concord
        (37.978, -122.029),  # Concord
        (37.906, -122.065),  # Walnut Creek
        (37.820, -121.997),  # Danville
        (37.780, -121.977),  # San Ramon
        (37.702, -121.921),  # Dublin/I-580 junction
        (37.600, -121.887),  # Sunol grade
        (37.534, -121.913),  # Mission Fremont
        (37.428, -121.905),  # Milpitas
        (37.336, -121.890),  # San Jose
    ]),

    # CA-24  Oakland → Caldecott Tunnel → Walnut Creek
    ("CA-24", 0.80, 0.48, [
        (37.826, -122.270),  # Oakland / I-580 junction
        (37.838, -122.234),  # Caldecott Tunnel
        (37.878, -122.183),  # Orinda
        (37.906, -122.068),  # Walnut Creek / I-680 junction
    ]),

    # US-101  San Francisco → San Jose
    ("US-101-S", 0.90, 0.48, [
        (37.773, -122.410),  # SF / SoMa
        (37.720, -122.422),  # SF / Geneva Ave
        (37.680, -122.428),  # South SF
        (37.631, -122.412),  # San Bruno
        (37.566, -122.311),  # San Mateo / Burlingame
        (37.494, -122.217),  # Redwood City
        (37.413, -122.107),  # Palo Alto
        (37.372, -122.027),  # Sunnyvale
        (37.337, -121.889),  # San Jose
    ]),

    # I-280  SF → San Jose (coastal hills)
    ("I-280", 0.80, 0.33, [
        (37.719, -122.458),  # SF / Junipero Serra
        (37.679, -122.465),  # Daly City
        (37.621, -122.455),  # San Bruno Mountain
        (37.548, -122.311),  # Hillsborough
        (37.451, -122.175),  # Woodside
        (37.337, -122.023),  # Saratoga
        (37.325, -121.951),  # San Jose
    ]),

    # US-101 North  Marin → Golden Gate Bridge → SF
    ("US-101-N", 0.80, 0.38, [
        (37.920, -122.515),  # Marin / Corte Madera
        (37.863, -122.479),  # Sausalito
        (37.832, -122.478),  # Vista Point (N end GG Bridge)
        (37.810, -122.478),  # Golden Gate Bridge S end
        (37.800, -122.461),  # Presidio
        (37.773, -122.441),  # SF / Marina
    ]),

    # San Mateo Bridge (CA-92)
    ("CA-92",        0.65, 0.28, [
        (37.558, -122.258),
        (37.578, -122.188),
        (37.597, -122.120),
    ]),

    # Dumbarton Bridge (CA-84)
    ("CA-84",        0.65, 0.22, [
        (37.508, -122.213),
        (37.508, -122.163),
        (37.508, -122.118),
    ]),

    # Richmond – San Rafael Bridge (I-580 West)
    ("Richmond-SR",  0.65, 0.22, [
        (37.919, -122.381),
        (37.930, -122.424),
        (37.941, -122.481),
    ]),

    # I-238  Castro Valley connector (I-580 ↔ I-880)
    ("I-238",        0.65, 0.42, [
        (37.692, -122.079),
        (37.696, -122.111),
        (37.703, -122.142),
    ]),
]

# Known congestion hotspots — boost base congestion for nearby segments.
# (lat, lon, peak_congestion, influence_radius_km)
_HOTSPOTS: List[Tuple[float, float, float, float]] = [
    (37.810, -122.377, 0.88, 5.0),   # Bay Bridge (whole span)
    (37.826, -122.272, 0.85, 4.0),   # Maze interchange
    (37.773, -122.410, 0.70, 3.0),   # SF approaches (I-80/US-101 merge)
    (37.702, -121.921, 0.60, 3.5),   # Dublin interchange (I-580/I-680)
    (37.906, -122.065, 0.58, 3.0),   # Walnut Creek (I-680/CA-24)
    (37.337, -121.889, 0.60, 5.0),   # San Jose convergence
    (37.694, -122.079, 0.55, 2.5),   # Castro Valley I-580/I-238/I-880
]


def _hotspot_base(mid_lat: float, mid_lon: float, hw_base: float) -> float:
    """Return segment base congestion incorporating nearby hotspot influence."""
    result = hw_base
    for h_lat, h_lon, h_cong, h_r in _HOTSPOTS:
        d = _haversine_km(mid_lat, mid_lon, h_lat, h_lon)
        if d < h_r:
            result = max(result, h_cong * (1.0 - d / h_r))
    return min(result, 1.0)


# ===========================================================================
# Event wrapper
# ===========================================================================

def shader_highway_traffic(state, outstate, traffic_density=0.50):
    """
    Highway traffic background effect — compatible with EventScheduler.

    Reads from outstate:
        traffic_density (float 0–1)  overall congestion level
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
                HighwayTrafficEffect,
                traffic_density=traffic_density,
            )
            state['effect'] = effect
            print(f"✓ Initialized highway traffic for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize highway traffic: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        state['effect'].traffic_density = outstate.get('traffic_density', traffic_density)

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ===========================================================================
# ShaderEffect implementation
# ===========================================================================

class HighwayTrafficEffect(ShaderEffect):
    """
    Renders Bay Area highway traffic as dynamically colored polyline quads.

    Congestion per segment:
        c[i] = clip( base[i] * density * 1.5  +  0.22 * sin(t*freq[i] + phase[i]), 0, 1 )

    Color:
        R = clip(2*c, 0, 1)      →  0 at green, full at yellow
        G = clip(2*(1-c), 0, 1)  →  full at green, 0 at red
        B = 0
    """

    _ROAD_BASE_WIDTH_FRAC = 0.006   # base road width as fraction of viewport width

    _VERT = """
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

    _FRAG = """
    #version 310 es
    precision highp float;
    in vec3 vColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(vColor, 0.88);
    }
    """

    def __init__(self, viewport, traffic_density: float = 0.50):
        super().__init__(viewport)
        self.traffic_density = traffic_density
        self.render_priority = 3   # under BART lines (priority 5)

        self._road_base_width = max(1.5, viewport.width * self._ROAD_BASE_WIDTH_FRAC)

        self._bake_geometry()
        self._init_congestion()

        self._vbo      = None
        self._vao      = None
        self._n_verts  = 0
        self._sim_time = 0.0

    # ------------------------------------------------------------------
    # Geometry / simulation init
    # ------------------------------------------------------------------

    def _geo_px(self, lat: float, lon: float) -> np.ndarray:
        px, py = _geo_to_fan_px(lat, lon,
                                float(self.viewport.width),
                                float(self.viewport.height))
        return np.array([px, py], dtype=np.float32)

    def _bake_geometry(self):
        """Pre-compute per-segment quad vertex positions (pixel space)."""
        quads: List[np.ndarray] = []
        mids:  List[Tuple[float, float]] = []
        bases: List[float] = []

        for hw_name, width_mult, base_cong, waypoints in _HIGHWAYS:
            hw = self._road_base_width * width_mult
            hw_half = hw * 0.5
            for i in range(len(waypoints) - 1):
                lat0, lon0 = waypoints[i]
                lat1, lon1 = waypoints[i + 1]
                a = self._geo_px(lat0, lon0)
                b = self._geo_px(lat1, lon1)
                dx, dy = b - a
                L = np.hypot(dx, dy)
                if L < 0.1:
                    continue
                nx, ny = -dy / L * hw_half, dx / L * hw_half
                quads.append(np.array([
                    [a[0]-nx, a[1]-ny],
                    [a[0]+nx, a[1]+ny],
                    [b[0]+nx, b[1]+ny],
                    [a[0]-nx, a[1]-ny],
                    [b[0]+nx, b[1]+ny],
                    [b[0]-nx, b[1]-ny],
                ], dtype=np.float32))
                mids.append(((lat0 + lat1) * 0.5, (lon0 + lon1) * 0.5))
                bases.append(_hotspot_base((lat0+lat1)*0.5, (lon0+lon1)*0.5, base_cong))

        if quads:
            self._seg_positions = np.array(quads, dtype=np.float32)  # (S, 6, 2)
            self._seg_base_cong = np.array(bases, dtype=np.float32)  # (S,)
        else:
            self._seg_positions = np.zeros((0, 6, 2), dtype=np.float32)
            self._seg_base_cong = np.zeros(0, dtype=np.float32)

        self._n_segs = len(self._seg_positions)

    def _init_congestion(self):
        """Randomise per-segment noise parameters."""
        rng = np.random.default_rng(seed=42)
        base_freq = 2.0 * np.pi / 90.0          # one cycle per 90 s animation time
        self._cong_freq  = rng.uniform(0.5, 1.8, self._n_segs) * base_freq
        self._cong_phase = rng.uniform(0.0, 2.0 * np.pi, self._n_segs)
        self._congestion = self._seg_base_cong.copy()

    # ------------------------------------------------------------------
    # ShaderEffect — compile_shader
    # ------------------------------------------------------------------

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(self._VERT, GL_VERTEX_SHADER),
            shaders.compileShader(self._FRAG, GL_FRAGMENT_SHADER),
        )

    # ------------------------------------------------------------------
    # ShaderEffect — setup_buffers
    # ------------------------------------------------------------------

    def setup_buffers(self):
        self._n_verts = self._n_segs * 6

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        # Allocate: 5 floats per vertex (x, y, r, g, b)
        glBufferData(GL_ARRAY_BUFFER, self._n_verts * 5 * 4, None, GL_DYNAMIC_DRAW)

        S = 5 * 4   # stride = 5 floats
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, S, ctypes.c_void_p(2 * 4))

        glBindVertexArray(0)
        self.VBOs.append(self._vbo)

    # ------------------------------------------------------------------
    # ShaderEffect — update
    # ------------------------------------------------------------------

    def update(self, dt: float, state: Dict):
        """Advance congestion simulation by dt animation seconds."""
        if self._n_segs == 0:
            return
        self._sim_time += dt
        noise = np.sin(self._sim_time * self._cong_freq + self._cong_phase)
        density = float(self.traffic_density)
        self._congestion = np.clip(
            self._seg_base_cong * density * 1.5  +  0.22 * noise,
            0.0, 1.0,
        )

    # ------------------------------------------------------------------
    # ShaderEffect — render
    # ------------------------------------------------------------------

    def render(self, state: Dict):
        super().render(state)
        if self._n_segs == 0 or self._vao is None:
            return

        # Vectorised colour computation ──────────────────────────────
        c = self._congestion                           # (S,)
        r = np.clip(2.0 * c,        0.0, 1.0)         # green→yellow→red
        g = np.clip(2.0 * (1.0-c), 0.0, 1.0)
        b = np.zeros_like(c)
        seg_colors = np.stack([r, g, b], axis=1)       # (S, 3)

        # Expand to per-vertex  (S*6, 3)
        colors_pv = np.repeat(seg_colors, 6, axis=0)

        # Combine with positions (S*6, 5)
        pos_flat = self._seg_positions.reshape(-1, 2)  # (S*6, 2)
        vbo_data = np.hstack([pos_flat, colors_pv]).astype(np.float32)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)

        glUseProgram(self.shader)
        loc = glGetUniformLocation(self.shader, "uResolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vbo_data.nbytes, vbo_data)
        glDrawArrays(GL_TRIANGLES, 0, self._n_verts)
        glBindVertexArray(0)
        glUseProgram(0)

        glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------
    # ShaderEffect — cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        try:
            if self._vao:
                glDeleteVertexArrays(1, [self._vao])
        except Exception:
            pass
        super().cleanup()
