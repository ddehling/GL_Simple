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

import math
import numpy as np
import ctypes
import time as _time
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List, Tuple
from renderer.effects.base import ShaderEffect
from renderer.fan_coords import FanCoords

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

# Geographic bounding box — expanded to fill the full fan semicircle.
# Lat range matches BART extent; lon range widened for the 2:1 fan aspect.
_LAT_MIN, _LAT_MAX = 37.32, 38.02
_LON_MIN, _LON_MAX = -123.01, -121.23

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

# ===========================================================================
# Bay Area geography — simplified coastline polylines (lat, lon)
# Each list is a connected polyline drawn as a dim outline.
# ===========================================================================
_BAY_COASTLINES: List[List[Tuple[float, float]]] = [
    # SF Peninsula west coast (Pacific side, north to south)
    [
        (37.81, -122.48), (37.79, -122.51), (37.77, -122.51),
        (37.75, -122.51), (37.73, -122.50), (37.71, -122.49),
        (37.69, -122.49), (37.67, -122.47), (37.65, -122.46),
        (37.63, -122.44), (37.61, -122.43), (37.59, -122.41),
        (37.57, -122.39), (37.55, -122.38), (37.52, -122.37),
        (37.48, -122.35), (37.44, -122.33), (37.40, -122.32),
        (37.36, -122.31),
    ],
    # SF Peninsula east coast (Bay side, north to south)
    [
        (37.81, -122.39), (37.80, -122.39), (37.79, -122.39),
        (37.78, -122.39), (37.77, -122.40), (37.75, -122.40),
        (37.73, -122.40), (37.71, -122.40), (37.69, -122.41),
        (37.67, -122.39), (37.65, -122.38), (37.63, -122.37),
        (37.61, -122.36), (37.59, -122.34), (37.57, -122.32),
        (37.55, -122.30), (37.52, -122.26), (37.49, -122.21),
        (37.46, -122.17), (37.43, -122.12), (37.40, -122.08),
        (37.37, -122.06), (37.34, -121.99),
    ],
    # East Bay shoreline (Richmond to Fremont)
    [
        (37.97, -122.35), (37.94, -122.34), (37.91, -122.33),
        (37.88, -122.32), (37.86, -122.31), (37.84, -122.30),
        (37.82, -122.30), (37.80, -122.30), (37.79, -122.29),
        (37.77, -122.27), (37.75, -122.25), (37.73, -122.22),
        (37.71, -122.20), (37.69, -122.19), (37.67, -122.16),
        (37.65, -122.14), (37.62, -122.12), (37.60, -122.10),
        (37.58, -122.08), (37.55, -122.06), (37.52, -122.05),
    ],
    # SF northern waterfront (Golden Gate to Bay Bridge)
    [
        (37.81, -122.48), (37.81, -122.46), (37.81, -122.44),
        (37.81, -122.42), (37.81, -122.40), (37.81, -122.39),
        (37.80, -122.38), (37.80, -122.37), (37.80, -122.36),
        (37.80, -122.35),
    ],
    # Bay Bridge (SF to Oakland)
    [
        (37.80, -122.35), (37.81, -122.33), (37.82, -122.30),
    ],
    # Marin County south shore
    [
        (37.84, -122.48), (37.86, -122.47), (37.88, -122.45),
        (37.90, -122.44), (37.92, -122.42), (37.94, -122.40),
        (37.95, -122.39), (37.96, -122.38), (37.97, -122.35),
    ],
    # South Bay shoreline (Fremont to San Jose)
    [
        (37.52, -122.05), (37.49, -122.03), (37.47, -122.01),
        (37.44, -121.98), (37.41, -121.96), (37.38, -121.94),
        (37.36, -121.92), (37.34, -121.91),
    ],
    # South Bay west shore (Palo Alto area)
    [
        (37.34, -121.91), (37.36, -121.93), (37.39, -121.97),
        (37.42, -122.01), (37.44, -122.04), (37.46, -122.08),
        (37.48, -122.12), (37.49, -122.16), (37.49, -122.21),
    ],
]

_BAY_GEO_COLOR = (0.15, 0.22, 0.30)  # dim blue-grey for coastline outlines

# Colors for the background map regions (dim, muted)
_COLOR_OCEAN    = np.array([12, 22, 48], dtype=np.uint8)
_COLOR_BAY      = np.array([20, 38, 72], dtype=np.uint8)
_COLOR_LAND     = np.array([38, 50, 30], dtype=np.uint8)
_COLOR_URBAN    = np.array([55, 52, 45], dtype=np.uint8)   # warm grey
_COLOR_PARK     = np.array([25, 55, 25], dtype=np.uint8)   # green
_COLOR_MOUNTAIN = np.array([30, 40, 25], dtype=np.uint8)   # darker green-brown

# ===========================================================================
# Bay Area geography — polygons for region classification
# Format: list of (lat, lon) tuples forming closed regions.
# The wider lon range (-123.01 to -121.23) means we show more area.
# ===========================================================================

# Main coastline polygon — everything inside is "not ocean"
_LAND_POLYGON: List[Tuple[float, float]] = [
    # Marin / Sonoma coast (north, tracing south)
    (38.05, -123.01),  # NW corner of map
    (38.05, -122.70),  # Bodega Bay area
    (38.00, -122.65), (37.97, -122.60), (37.95, -122.55),
    (37.92, -122.52), (37.88, -122.52), (37.85, -122.50),
    (37.84, -122.48),  # Marin headlands
    (37.81, -122.48),  # Golden Gate
    # SF ocean coast south
    (37.79, -122.51), (37.77, -122.51), (37.75, -122.51),
    (37.73, -122.50), (37.71, -122.49), (37.69, -122.49),
    (37.67, -122.47), (37.65, -122.46), (37.63, -122.44),
    (37.61, -122.43), (37.59, -122.41), (37.57, -122.39),
    (37.55, -122.38),
    # Half Moon Bay / Santa Cruz coast
    (37.52, -122.44), (37.50, -122.45), (37.48, -122.44),
    (37.45, -122.43), (37.42, -122.41), (37.38, -122.39),
    (37.35, -122.38), (37.32, -122.36),
    # South edge of map, east across to close
    (37.32, -121.20),
    (38.05, -121.20),  # NE corner
    (38.05, -123.01),  # close
]

# SF Bay (central/north bay) — water polygon
_BAY_POLYGON: List[Tuple[float, float]] = [
    # SF waterfront east side
    (37.81, -122.39), (37.80, -122.38), (37.80, -122.37),
    (37.80, -122.35),
    # Bay Bridge to Oakland
    (37.82, -122.32), (37.82, -122.30),
    # East Bay shoreline north to Richmond
    (37.84, -122.30), (37.86, -122.31), (37.88, -122.32),
    (37.90, -122.33), (37.92, -122.34), (37.94, -122.34),
    (37.96, -122.35), (37.97, -122.35),
    # San Pablo Bay
    (37.98, -122.37), (37.98, -122.40), (37.97, -122.43),
    (37.96, -122.45),
    # Marin east shore
    (37.95, -122.47), (37.93, -122.47), (37.91, -122.47),
    (37.89, -122.48), (37.87, -122.48), (37.85, -122.48),
    (37.84, -122.48),
    # Back across Golden Gate
    (37.83, -122.47), (37.82, -122.44), (37.82, -122.42),
    (37.81, -122.41), (37.81, -122.39),
]

# South Bay — water polygon
_SOUTH_BAY_POLYGON: List[Tuple[float, float]] = [
    # East Bay shoreline going south
    (37.80, -122.29), (37.78, -122.27), (37.76, -122.25),
    (37.74, -122.23), (37.72, -122.21), (37.70, -122.19),
    (37.68, -122.17), (37.66, -122.15), (37.64, -122.13),
    (37.62, -122.11), (37.60, -122.09), (37.58, -122.07),
    (37.55, -122.05), (37.52, -122.04),
    # South tip of bay
    (37.49, -122.02), (37.47, -122.00), (37.44, -121.97),
    (37.42, -121.95), (37.39, -121.93), (37.37, -121.92),
    (37.35, -121.91),
    # West shore going back north
    (37.36, -121.93), (37.38, -121.95), (37.40, -121.98),
    (37.42, -122.01), (37.44, -122.04), (37.46, -122.08),
    (37.48, -122.12), (37.49, -122.16), (37.50, -122.20),
    # Peninsula east coast north
    (37.52, -122.25), (37.54, -122.28), (37.56, -122.30),
    (37.58, -122.32), (37.60, -122.34), (37.62, -122.35),
    (37.64, -122.37), (37.66, -122.38), (37.68, -122.39),
    (37.70, -122.40), (37.72, -122.40), (37.74, -122.40),
    (37.76, -122.40), (37.78, -122.39), (37.80, -122.39),
    (37.80, -122.29),
]

# San Pablo Bay (north of Richmond Bridge)
_SAN_PABLO_BAY_POLYGON: List[Tuple[float, float]] = [
    (37.97, -122.35), (37.98, -122.37), (37.99, -122.40),
    (38.00, -122.42), (38.01, -122.44), (38.02, -122.46),
    (38.03, -122.47),
    (38.04, -122.44), (38.04, -122.40), (38.03, -122.37),
    (38.02, -122.34), (38.00, -122.32), (37.98, -122.32),
    (37.97, -122.33), (37.97, -122.35),
]

# Urban areas — warm grey, brighter than land
_URBAN_POLYGONS: List[List[Tuple[float, float]]] = [
    # San Francisco proper
    [(37.81, -122.48), (37.81, -122.39), (37.71, -122.39),
     (37.71, -122.48), (37.81, -122.48)],
    # Oakland / Berkeley / Emeryville
    [(37.90, -122.31), (37.90, -122.22), (37.78, -122.22),
     (37.78, -122.31), (37.90, -122.31)],
    # San Jose metro
    [(37.42, -122.00), (37.42, -121.82), (37.28, -121.82),
     (37.28, -122.00), (37.42, -122.00)],
    # Concord / Walnut Creek / Pleasant Hill
    [(38.00, -122.08), (38.00, -121.90), (37.89, -121.90),
     (37.89, -122.08), (38.00, -122.08)],
    # Fremont / Hayward / Union City
    [(37.62, -122.13), (37.62, -121.95), (37.50, -121.95),
     (37.50, -122.13), (37.62, -122.13)],
    # Daly City / South SF / San Bruno
    [(37.71, -122.47), (37.71, -122.39), (37.63, -122.39),
     (37.63, -122.47), (37.71, -122.47)],
    # San Mateo / Redwood City
    [(37.58, -122.30), (37.58, -122.20), (37.48, -122.20),
     (37.48, -122.30), (37.58, -122.30)],
    # Palo Alto / Mountain View / Sunnyvale
    [(37.48, -122.18), (37.48, -122.00), (37.36, -122.00),
     (37.36, -122.18), (37.48, -122.18)],
    # Richmond / El Cerrito
    [(37.94, -122.38), (37.94, -122.30), (37.90, -122.30),
     (37.90, -122.38), (37.94, -122.38)],
    # Antioch / Pittsburg / Bay Point
    [(38.02, -121.95), (38.02, -121.78), (37.96, -121.78),
     (37.96, -121.95), (38.02, -121.95)],
    # Milpitas / Santa Clara
    [(37.45, -121.92), (37.45, -121.82), (37.34, -121.82),
     (37.34, -121.92), (37.45, -121.92)],
    # Livermore / Dublin / Pleasanton
    [(37.72, -121.92), (37.72, -121.75), (37.66, -121.75),
     (37.66, -121.92), (37.72, -121.92)],
]

# Parks / open space / forests — green
_PARK_POLYGONS: List[List[Tuple[float, float]]] = [
    # Golden Gate Park
    [(37.77, -122.51), (37.77, -122.45), (37.76, -122.45),
     (37.76, -122.51), (37.77, -122.51)],
    # Presidio
    [(37.80, -122.48), (37.80, -122.44), (37.79, -122.44),
     (37.79, -122.48), (37.80, -122.48)],
    # Marin Headlands / Muir Woods
    [(37.86, -122.55), (37.86, -122.48), (37.83, -122.48),
     (37.83, -122.55), (37.86, -122.55)],
    # East Bay Regional Parks (Tilden / Wildcat Canyon / Redwood)
    [(37.92, -122.24), (37.92, -122.17), (37.82, -122.17),
     (37.82, -122.24), (37.92, -122.24)],
    # Coyote Hills / Don Edwards Wildlife Refuge
    [(37.57, -122.10), (37.57, -122.05), (37.53, -122.05),
     (37.53, -122.10), (37.57, -122.10)],
    # San Bruno Mountain
    [(37.70, -122.44), (37.70, -122.41), (37.69, -122.41),
     (37.69, -122.44), (37.70, -122.44)],
    # Sunol / Ohlone Wilderness
    [(37.54, -121.85), (37.54, -121.78), (37.48, -121.78),
     (37.48, -121.85), (37.54, -121.85)],
    # Crystal Springs Reservoir area
    [(37.55, -122.38), (37.55, -122.33), (37.48, -122.33),
     (37.48, -122.38), (37.55, -122.38)],
    # Point Reyes / Olema (north Marin)
    [(38.05, -122.85), (38.05, -122.70), (37.95, -122.70),
     (37.95, -122.85), (38.05, -122.85)],
]

# Mountain / hill ranges — darker terrain
_MOUNTAIN_POLYGONS: List[List[Tuple[float, float]]] = [
    # Mt Tamalpais
    [(37.93, -122.60), (37.93, -122.52), (37.88, -122.52),
     (37.88, -122.60), (37.93, -122.60)],
    # Santa Cruz Mountains (west ridge)
    [(37.50, -122.42), (37.50, -122.32), (37.32, -122.32),
     (37.32, -122.42), (37.50, -122.42)],
    # Mt Diablo / surrounding hills
    [(37.90, -121.98), (37.90, -121.88), (37.84, -121.88),
     (37.84, -121.98), (37.90, -121.98)],
    # East Bay Hills (upper ridgeline)
    [(37.92, -122.17), (37.92, -122.13), (37.82, -122.13),
     (37.82, -122.17), (37.92, -122.17)],
    # Hamilton Range (east of San Jose)
    [(37.48, -121.78), (37.48, -121.65), (37.32, -121.65),
     (37.32, -121.78), (37.48, -121.78)],
    # Sonoma Mountains (far north)
    [(38.05, -122.60), (38.05, -122.48), (37.98, -122.48),
     (37.98, -122.60), (38.05, -122.60)],
    # Diablo Range south (east of Fremont)
    [(37.62, -121.82), (37.62, -121.68), (37.48, -121.68),
     (37.48, -121.82), (37.62, -121.82)],
]


def _point_in_polygon(px: float, py: float,
                      polygon: List[Tuple[float, float]]) -> bool:
    """Scalar ray-casting point-in-polygon test (kept for external callers
    such as city_lights.py that test individual random points)."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_mask(px: np.ndarray, py: np.ndarray,
                  polygon: List[Tuple[float, float]]) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test over arrays of points.

    polygon is a list of (y, x) tuples (lat, lon order to match the rest of
    this module). px and py are arrays of any matching shape.
    """
    n = len(polygon)
    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        cond = (yi > py) != (yj > py)
        # Safe division: where cond is False we don't use xint, so a divide
        # by a zero edge (yj == yi) doesn't matter — suppress the warning.
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= cond & (px < xint)
        j = i
    return inside


def _pixel_lat_lon(tex_w: int, tex_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays of shape (tex_h, tex_w) for every texel."""
    u = np.linspace(0.0, 1.0, tex_w, dtype=np.float64)
    v = np.linspace(0.0, 1.0, tex_h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # both (tex_h, tex_w)
    phys_x, phys_y = _fan.uv_to_physical_np(uu, vv)
    nx = (phys_x - _PHYS_X_MIN) / (_PHYS_X_MAX - _PHYS_X_MIN)
    ny = (phys_y - _PHYS_Y_MIN) / (_PHYS_Y_MAX - _PHYS_Y_MIN)
    lon = _LON_MIN + nx * (_LON_MAX - _LON_MIN)
    lat = _LAT_MIN + ny * (_LAT_MAX - _LAT_MIN)
    return lat, lon


def _build_geo_texture(tex_w: int, tex_h: int) -> np.ndarray:
    """Build an RGB texture classifying each pixel by geographic region.

    The texture maps directly to the buffer UV space — each texel
    corresponds to one FBO pixel.  At init time only.
    """
    lat, lon = _pixel_lat_lon(tex_w, tex_h)

    # Start everyone as ocean, then overlay regions in priority order.
    tex = np.broadcast_to(_COLOR_OCEAN, (tex_h, tex_w, 3)).copy()

    is_land = _polygon_mask(lon, lat, _LAND_POLYGON)
    tex[is_land] = _COLOR_LAND

    is_bay = is_land & (
        _polygon_mask(lon, lat, _BAY_POLYGON)
        | _polygon_mask(lon, lat, _SOUTH_BAY_POLYGON)
        | _polygon_mask(lon, lat, _SAN_PABLO_BAY_POLYGON)
    )
    tex[is_bay] = _COLOR_BAY

    land_only = is_land & ~is_bay

    # Urban first (lowest priority among land overlays), then mountain, then
    # park — later layers win, matching the original priority order
    # (park > mountain > urban > land).
    urban_mask = np.zeros_like(land_only)
    for poly in _URBAN_POLYGONS:
        urban_mask |= _polygon_mask(lon, lat, poly)
    tex[land_only & urban_mask] = _COLOR_URBAN

    mountain_mask = np.zeros_like(land_only)
    for poly in _MOUNTAIN_POLYGONS:
        mountain_mask |= _polygon_mask(lon, lat, poly)
    tex[land_only & mountain_mask] = _COLOR_MOUNTAIN

    park_mask = np.zeros_like(land_only)
    for poly in _PARK_POLYGONS:
        park_mask |= _polygon_mask(lon, lat, poly)
    tex[land_only & park_mask] = _COLOR_PARK

    return tex


def _build_fog_density_texture(tex_w: int, tex_h: int) -> np.ndarray:
    """Build a single-channel fog density texture for Bay Area spatial fog.

    Red channel = fog density 0-255.  Fog is densest over the Pacific,
    funnels through the Golden Gate, and thins in the East Bay.
    Built at init time only, same UV mapping as the geo texture.
    """
    lat, lon = _pixel_lat_lon(tex_w, tex_h)

    # Ocean fog (lon < -122.5)
    ocean_dist = (-122.50 - lon) * 4.0
    density = np.clip(ocean_dist, 0.0, 0.6)

    # Golden Gate plume — original took sqrt then squared, equivalent to gx²+gy²
    gx = (lon - (-122.43)) * 5.0
    gy = (lat - 37.82) * 8.0
    density += 0.6 * np.exp(-(gx * gx + gy * gy) * 0.8)

    # SF city fog blanket
    sf_x = np.clip((-122.35 - lon) * 2.0, 0.0, 0.5)
    sf_lat = np.exp(-((lat - 37.76) * 5.0) ** 2)
    density += sf_x * sf_lat * 0.5

    # Berkeley / North Oakland spill
    bk_x = (lon - (-122.27)) * 6.0
    bk_y = (lat - 37.87) * 8.0
    density += 0.35 * np.exp(-(bk_x * bk_x + bk_y * bk_y) * 1.2)

    # Fog thins east of Berkeley hills
    east_clear = np.clip((lon - (-122.15)) * 2.0, 0.0, 1.0)
    density *= (1.0 - east_clear * 0.85)

    return (np.clip(density, 0.0, 1.0) * 255).astype(np.uint8)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2.0 * np.arcsin(np.sqrt(a))


_fan = FanCoords()

# Physical region on the fan surface — spans the full semicircle so every
# pixel participates in the map.
_PHYS_X_MIN, _PHYS_X_MAX = -20.6, 20.6
_PHYS_Y_MIN, _PHYS_Y_MAX =   0.0, 20.6


def _geo_to_physical(lat: float, lon: float) -> Tuple[float, float]:
    """Map lat/lon to physical (x, y) feet on the fan surface."""
    nx = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)
    ny = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
    return (_PHYS_X_MIN + nx * (_PHYS_X_MAX - _PHYS_X_MIN),
            _PHYS_Y_MIN + ny * (_PHYS_Y_MAX - _PHYS_Y_MIN))


def _geo_to_fan_px(lat: float, lon: float, w: float, h: float) -> Tuple[float, float]:
    """Map lat/lon to buffer pixel coords via physical fan space."""
    phys_x, phys_y = _geo_to_physical(lat, lon)
    return _fan.physical_to_px(phys_x, phys_y)


# ===========================================================================
# Event wrapper
# ===========================================================================

# ---------------------------------------------------------------------------
# BART train-arrival text announcement data
# ---------------------------------------------------------------------------
# Ambient BART sound scheduling used to live here, driven by a hardcoded
# `media/sounds/bart_sounds` directory. That behavior has been extracted
# into the general-purpose `sound_pool` effect (see
# renderer/effects/sound_pool.py). The BarTiki weather set now wires up
# its pool via WEATHER_SETS["bartiki"]["sound_pool_dir"].
_BART_DESTINATIONS = ["SFO", "RICH", "DALY", "FREM", "PITS", "DUBL", "ANTI", "WARM"]


def _diurnal_fog_scale(t: float) -> float:
    """Marine-layer fog cycle: peaks pre-dawn, burns off midday, returns at dusk.

    t is time-of-day (0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset).
    Returns a multiplier for spatial fog density, ~0.3 at noon and ~1.3 at dawn.
    """
    midday = max(-math.cos(2.0 * math.pi * t), 0.0)        # 1 at noon, 0 at night
    dawn_dist = abs(((t - 0.22) + 0.5) % 1.0 - 0.5)         # peaks at t=0.22
    dawn_bump = math.exp(-(dawn_dist * 8.0) ** 2)
    return (1.0 - 0.7 * midday) + 0.3 * dawn_bump


def shader_bart_map(state, outstate, train_speed=40.0, train_density=1.0):
    """
    BART Map background effect — compatible with EventScheduler.

    Also drives BART ambient sounds and train arrival text announcements
    so all bartiki-specific behavior lives in one place.

    Reads from outstate:
        train_speed   (float) real-time multiplier
        train_density (float) reserved for future headway scaling
        season        (float) time-of-day cycle
        soundengine   (AudioEngine) for playing sounds
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
            state['_last_bart_sound_end'] = 0
            state['_last_arrival_text'] = 0
            # Publish fog density texture so the fog effect can use it
            outstate['fog_density_texture'] = effect._fog_density_texture
            outstate['spatial_fog'] = True
            print(f"✓ Initialized BART map for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize BART map: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        state['effect'].train_speed = outstate.get('train_speed', train_speed)
        state['effect'].set_train_density(outstate.get('train_density', train_density))
        tod = outstate.get('season', 0.5)
        state['effect'].time_of_day = tod
        outstate['spatial_fog_scale'] = _diurnal_fog_scale(tod)

        now = _time.time()

        # --- Train arrival text announcements ---
        time_since_text = now - state.get('_last_arrival_text', 0)
        if time_since_text > 60.0 and np.random.random() < 1 / 1200:
            # text.py stayed in the shared library (renderer/effects/),
            # so the relative import that worked when bart_map lived
            # alongside it now needs the absolute path.
            from renderer.effects.text import shader_text
            dest = np.random.choice(_BART_DESTINATIONS)
            mins = np.random.choice([2, 3, 5, 8])
            cars = np.random.choice([6, 8, 10])

            # Each line as separate text with unique lambda to bypass duplicate check
            def _make_text_fn(txt, yp):
                def _fn(st, os, **kw):
                    shader_text(st, os, text=txt,
                                color=(1.0, 0.2, 0.1), font_size=48,
                                x_pos=0.5, y_pos=yp,
                                auto_scale=True, scale_factor=1.0,
                                height_scale=2.0)
                return _fn

            scheduler = outstate.get('event_scheduler')
            if scheduler:
                scheduler.schedule_event(0, 10, _make_text_fn(dest, 0.283), frame_id=frame_id)
                scheduler.schedule_event(0, 10, _make_text_fn(f"{mins} min", 0.533), frame_id=frame_id)
                scheduler.schedule_event(0, 10, _make_text_fn(f"{cars} car", 0.783), frame_id=frame_id)
            state['_last_arrival_text'] = now

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            # Clear spatial fog so other weather sets get default depth-based fog
            outstate.pop('fog_density_texture', None)
            outstate.pop('spatial_fog', None)
            outstate.pop('spatial_fog_scale', None)


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

    _BG_VERT = """
    #version 310 es
    precision highp float;
    in vec2 position;
    out vec2 vUV;
    void main() {
        vUV = position * 0.5 + 0.5;
        float depth = 98.0 / 100.0;  // behind stars (99.99) but in front of clear
        gl_Position = vec4(position, depth, 1.0);
    }
    """

    _BG_FRAG = """
    #version 310 es
    precision highp float;
    in vec2 vUV;
    out vec4 FragColor;
    uniform sampler2D uGeoTex;
    uniform float uTimeOfDay;  // 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset

    void main() {
        vec3 geo = texture(uGeoTex, vUV).rgb;

        // Day/night brightness: cosine curve, noon=bright, midnight=dark
        float dayNight = -cos(uTimeOfDay * 6.28318);  // -1 at midnight, +1 at noon
        float brightness = 0.35 + 0.65 * max(dayNight, 0.0);  // 0.35 at night, 1.0 at noon
        brightness = max(brightness, 0.06);  // never fully black

        // Golden hour tint near dawn (0.2) and dusk (0.8)
        float dawnDist = min(abs(uTimeOfDay - 0.20), abs(uTimeOfDay - 1.20));
        float duskDist = abs(uTimeOfDay - 0.80);
        float goldenHour = exp(-dawnDist * dawnDist * 80.0)
                         + exp(-duskDist * duskDist * 80.0);
        vec3 goldenTint = vec3(0.3, 0.15, 0.0) * goldenHour;

        // At night, let background become semi-transparent so stars show through
        float nightAlpha = 0.4 + 0.6 * brightness;  // 0.4 at full night, 1.0 at noon
        FragColor = vec4(geo * brightness + goldenTint * brightness, nightAlpha);
    }
    """

    _LINE_VERT = """
    #version 310 es
    precision highp float;
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec3 aColor;
    out vec3 vColor;
    uniform vec2 uResolution;
    void main() {
        vec2 clip = (aPos / uResolution) * 2.0 - 1.0;
        float depth = 96.0 / 100.0;  // behind clouds (85-95), in front of background (98)
        gl_Position = vec4(clip, depth, 1.0);
        vColor = aColor;
    }
    """

    _LINE_FRAG = """
    #version 310 es
    precision highp float;
    in vec3 vColor;
    out vec4 FragColor;
    uniform float uTimeOfDay;
    void main() {
        // At night, BART lines glow brighter (constellation effect)
        float dayNight = -cos(uTimeOfDay * 6.28318);
        float nightBoost = 1.0 + 0.5 * max(-dayNight, 0.0);  // 1.0 day, 1.5 night
        float alpha = mix(0.70, 0.95, max(-dayNight, 0.0));   // more opaque at night
        FragColor = vec4(vColor * nightBoost, alpha);
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
        // Scale quad so circles are round in clip space (FBO is non-square)
        vec2 pixelRadius = iRadius * uResolution / max(uResolution.x, uResolution.y);
        vec2 world = iPos + aQuad * pixelRadius;
        vec2 clip  = (world / uResolution) * 2.0 - 1.0;
        float depth = 95.5 / 100.0;  // just in front of lines (96), behind clouds (85-95)
        gl_Position = vec4(clip, depth, 1.0);
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
        self.time_of_day   = 0.5          # 0=midnight, 0.5=noon
        self.render_priority = 5

        w, h = float(viewport.width), float(viewport.height)

        self._station_px = np.array(
            [_geo_to_fan_px(lat, lon, w, h)
             for _, lat, lon in _STATIONS_RAW],
            dtype=np.float32,
        )

        # Precompute per-station pixel scale for radius compensation
        self._station_scale = np.array(
            [_fan.pixel_scale_at_uv(*_fan.physical_to_uv(*_geo_to_physical(lat, lon)))
             for _, lat, lon in _STATIONS_RAW],
            dtype=np.float32,
        )

        self._line_width = max(3.0, w * 0.020)
        self._station_r  = max(2.5, w * 0.007)
        self._train_r    = max(6.0, w * 0.025)

        self._bake_line_geometry()
        self._init_trains()

        # GPU handles (filled by compile_shader / setup_buffers)
        self._bg_shader       = None
        self._bg_VAO          = None
        self._geo_texture     = None
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
        """Pre-compute thick-line quad vertices for coastline + BART lines."""
        w, h = float(self.viewport.width), float(self.viewport.height)

        # --- Coastline geometry (rendered first, behind BART) ---
        geo_lw = self._line_width * 0.4  # thinner than BART lines
        geo_half = geo_lw * 0.5
        cr, cg, cb = _BAY_GEO_COLOR
        geo_verts: List[float] = []
        for polyline in _BAY_COASTLINES:
            for i in range(len(polyline) - 1):
                ax, ay = _geo_to_fan_px(polyline[i][0], polyline[i][1], w, h)
                bx, by = _geo_to_fan_px(polyline[i+1][0], polyline[i+1][1], w, h)
                dx, dy = bx - ax, by - ay
                L = np.hypot(dx, dy)
                if L < 0.3:
                    continue
                nx, ny = -dy / L * geo_half, dx / L * geo_half
                geo_verts += [
                    ax-nx, ay-ny, cr, cg, cb,
                    ax+nx, ay+ny, cr, cg, cb,
                    bx+nx, by+ny, cr, cg, cb,
                    ax-nx, ay-ny, cr, cg, cb,
                    bx+nx, by+ny, cr, cg, cb,
                    bx-nx, by-ny, cr, cg, cb,
                ]
        self._n_geo_verts = len(geo_verts) // 5

        # --- BART line geometry ---
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

        # Combine: coastline first, then BART lines
        geo_arr = np.array(geo_verts, dtype=np.float32) if geo_verts else np.zeros(0, np.float32)
        self._geo_and_lines = geo_arr  # store for VBO upload

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
        self._bg_shader = shaders.compileProgram(
            shaders.compileShader(self._BG_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(self._BG_FRAG, GL_FRAGMENT_SHADER),
        )
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
        # --- Background geo texture ---
        w_int, h_int = self.viewport.width, self.viewport.height
        geo_tex_data = _build_geo_texture(w_int, h_int)
        self._geo_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._geo_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w_int, h_int, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, geo_tex_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        # --- Fog density texture (for spatial fog effect) ---
        fog_data = _build_fog_density_texture(w_int, h_int)
        self._fog_density_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._fog_density_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w_int, h_int, 0,
                     GL_RED, GL_UNSIGNED_BYTE, fog_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        # --- Background fullscreen quad VAO ---
        bg_quad = np.array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1],
                           dtype=np.float32)
        self._bg_VAO = glGenVertexArrays(1)
        glBindVertexArray(self._bg_VAO)
        bg_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bg_vbo)
        glBufferData(GL_ARRAY_BUFFER, bg_quad.nbytes, bg_quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)
        self.VBOs.append(bg_vbo)

        # --- Static line quad geometry: coastline + BART lines in one VBO ---
        bart_verts = (np.concatenate(self._line_chunks)
                      if self._line_chunks else np.zeros(0, np.float32))
        all_verts = np.concatenate([self._geo_and_lines, bart_verts]) \
            if len(self._geo_and_lines) > 0 else bart_verts
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
        sd[:, 2] = self._station_r / self._station_scale  # compensate per station
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

    def _train_world_positions(self, line_idx: int) -> List[Tuple[np.ndarray, bool, float]]:
        """
        Return (pixel_pos, is_dwelling, pixel_scale) for each train on the line.
        pixel_scale is interpolated from per-station scales for radius compensation.
        """
        td  = self._trains[line_idx]
        seq = td['seq']
        out = []
        for i, p in enumerate(td['pos']):
            seg  = int(np.clip(p, 0, len(seq) - 2))
            frac = p - seg
            idx_a, idx_b = seq[seg], seq[min(seg + 1, len(seq) - 1)]
            a    = self._station_px[idx_a]
            b    = self._station_px[idx_b]
            scale = self._station_scale[idx_a] + frac * (self._station_scale[idx_b] - self._station_scale[idx_a])
            out.append((a + frac * (b - a), td['dwell'][i] > 0.0, scale))
        return out

    def render(self, state: Dict):
        # NO state toggling — global state handles depth test + blend
        super().render(state)
        if self.circle_shader is None:
            return

        w = float(self.viewport.width)
        h = float(self.viewport.height)

        # ---- 0. Draw background geo map (z=98, behind stars at 99.99) ----
        # At night alpha drops so stars show through via standard blending
        glUseProgram(self._bg_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._geo_texture)
        loc = glGetUniformLocation(self._bg_shader, "uGeoTex")
        if loc != -1:
            glUniform1i(loc, 0)
        loc = glGetUniformLocation(self._bg_shader, "uTimeOfDay")
        if loc != -1:
            glUniform1f(loc, self.time_of_day)
        glBindVertexArray(self._bg_VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        # ---- 1. Draw BART line quads (z=50, constellation glow at night) ----
        glUseProgram(self.shader)
        loc = glGetUniformLocation(self.shader, "uResolution")
        if loc != -1:
            glUniform2f(loc, w, h)
        loc = glGetUniformLocation(self.shader, "uTimeOfDay")
        if loc != -1:
            glUniform1f(loc, self.time_of_day)
        glBindVertexArray(self._line_VAO)
        glDrawArrays(GL_TRIANGLES, 0, self._n_line_verts)
        glBindVertexArray(0)

        # ---- 2. Build per-frame instance data ----
        train_rows: List[List[float]] = []
        for line_idx, td in enumerate(self._trains):
            c = td['color']
            for wp, dwelling, scale in self._train_world_positions(line_idx):
                if dwelling:
                    r, alpha, glow = self._train_r * 1.35 / scale, 1.0, 1.2
                else:
                    r, alpha, glow = self._train_r / scale, 0.95, 0.80
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

        # ---- 3. Draw circles (z=25, in front of lines) ----
        glUseProgram(self.circle_shader)
        loc = glGetUniformLocation(self.circle_shader, "uResolution")
        if loc != -1:
            glUniform2f(loc, w, h)
        glBindVertexArray(self._circle_VAO)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, n_inst)
        glBindVertexArray(0)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # ShaderEffect — cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        try:
            if self._bg_VAO:
                glDeleteVertexArrays(1, [self._bg_VAO])
            if hasattr(self, '_bg_shader') and self._bg_shader:
                glDeleteProgram(self._bg_shader)
            if hasattr(self, '_geo_texture') and self._geo_texture:
                glDeleteTextures(1, [self._geo_texture])
            if self._line_VAO:
                glDeleteVertexArrays(1, [self._line_VAO])
            if self._circle_VAO:
                glDeleteVertexArrays(1, [self._circle_VAO])
            if self.circle_shader:
                glDeleteProgram(self.circle_shader)
        except Exception:
            pass
        super().cleanup()   # handles self.VBOs and self.shader
