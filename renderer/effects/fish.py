"""
Fish swimming effect - marine life swimming across the scene
GPU-accelerated fish system with schooling behavior and depth.

Fish move in physical (theta, r_ft) space on the fan: each fish is an
"arc swimmer" that picks a radius and changes its angular position over
time. Sizes and speeds are stored in physical units (feet / feet-per-second)
and converted to buffer pixels per frame using FanCoords. This makes a
"tang" the same physical size whether it's near the inner radius or the
outer rim.
"""
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FanCoords, FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL
from renderer.fan_geometry import FanGeometry

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_fish(state, outstate, activity=1.0):
    """
    Shader-based fish swimming effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_fish, activity=1.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        activity: Marine life activity multiplier
    """
    # Get the viewport
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    squish_top_width = outstate.get('scale', 1.0)  # Get scale from state
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize fish effect on first call
    if state['count'] == 0:
        # Start with base number (will adjust dynamically)
        num_fish = 30
        print(f"Initializing fish effect for frame {frame_id} (base: {num_fish} fish)")
        
        try:
            fish_effect = viewport.add_effect(
                FishEffect,
                num_fish=num_fish,
                activity=activity,
                squish_top_width=squish_top_width
            )
            state['fish_effect'] = fish_effect
            print(f"✓ Initialized shader fish for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize fish: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update activity every frame
    if 'fish_effect' in state:
        marine_life_activity = outstate.get('marine_life_activity', activity)
        state['fish_effect'].set_target_activity(marine_life_activity)
        
        # Update squish width from scale
        state['fish_effect'].squish_top_width = squish_top_width
    
    # On close event, clean up
    if state['count'] == -1:
        if 'fish_effect' in state:
            print(f"Cleaning up fish effect for frame {frame_id}")
            viewport.effects.remove(state['fish_effect'])
            state['fish_effect'].cleanup()
            print(f"✓ Cleaned up shader fish for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

# ----------------------------------------------------------------------------
# Fan physical constants (mirrors FanCoords)
# ----------------------------------------------------------------------------
INNER_R_FT = FanGeometry.PHYSICAL_INNER_FT   # 4.0
OUTER_R_FT = FanGeometry.PHYSICAL_OUTER_FT   # 20.6

# Spawn band: r in [SPAWN_R_MIN, SPAWN_R_MAX] feet.
# Preserves the previous "bottom 65% of buffer" region in physical terms:
#   v in [0, 0.65]  ->  r in [4.0, 4 + 0.65 * 16.6] = [4.0, 14.79]
# Water / sand boundary. The ocean_waves shader renders its beach (sand) in
# the top ~8% of the buffer at normal tide, extending to ~23% at low tide.
# In fan radial terms that corresponds to roughly r in [4.0, 5.3] ft at normal
# tide and r in [4.0, 7.8] ft at low tide. Fish must never swim into the sand
# region, so keep them a safe margin below the worst-case (low-tide) beach.
WATER_SURFACE_R_FT = 8.0

SPAWN_R_MIN = WATER_SURFACE_R_FT
SPAWN_R_MAX = INNER_R_FT + 0.65 * (OUTER_R_FT - INNER_R_FT)

# Visual scale: species sizes are stored in real-world feet, but at the
# fan's physical scale a real 6-inch silverside is a few pixels tall and
# impossible to read. This multiplier blows them up for legibility while
# keeping the relative species-to-species ratios honest.
FISH_DISPLAY_SIZE_SCALE = 3.0

# Hard caps for fish geometry in physical feet. The biggest template vertex
# sits at about 1.1 * size_ft from the fish center, so a fish near the inner
# radius (r = 4 ft) can safely be at most ~3 ft long without wrapping around
# the apex singularity. Clamping here keeps the "big" species big without
# letting them stretch across the fan.
FISH_MAX_SIZE_FT = 2.0
# Minimum radial offset of the fish center from the inner edge of the fan.
# Anything closer risks having tail vertices wrap to the other side.
FISH_INNER_MARGIN_FT = 1.2

# Radial drift scale. Kept modest -- aggressive radial drift makes fish
# ping-pong between the water surface and the outer rim, which looks dumb.
# Most of the "diagonal motion" illusion now comes from gentle drift plus
# the natural arc swimming.
FISH_DRIFT_SCALE = 0.8


# ----------------------------------------------------------------------------
# Body plans — vertex-shader morph parameters applied to the unit fish template
# (body_length, body_height, body_taper, tail_spread)
# ----------------------------------------------------------------------------
BODY_PLANS = {
    'standard':  (1.00, 1.00, 1.00, 1.00),
    'elongated': (1.70, 0.55, 0.70, 0.60),
    'round':     (0.75, 1.05, 0.50, 0.80),
    'flat':      (0.90, 1.55, 0.40, 0.30),
    'eel':       (2.20, 0.28, 1.20, 0.15),
}


# ----------------------------------------------------------------------------
# Pattern IDs for the fragment shader
# ----------------------------------------------------------------------------
PATTERN_PLAIN     = 0
PATTERN_VBANDS    = 1
PATTERN_HSTRIPE   = 2
PATTERN_SPOTS     = 3
PATTERN_GRADIENT  = 4
PATTERN_EYESPOT   = 5


# ----------------------------------------------------------------------------
# Fish species definitions
#
# size_range and speed_range are in PHYSICAL UNITS:
#   size_range : feet (head-to-tail length, roughly)
#   speed_range: feet per second (along the arc)
#
# behavior values:
#   'cruise'           - default, gentle course-corrections
#   'school'           - alignment with same-species neighbors
#   'dart'             - long pauses interrupted by speed bursts
#   'hover'            - near-zero base omega, small random wobble
#   'bottom_dweller'   - spawn restricted to high-r band, no radial drift
#   'surface'          - spawn restricted to low-r band, no radial drift
#   'vertical_drifter' - large radial drift, small theta speed
# ----------------------------------------------------------------------------
FISH_SPECIES = {
    'silverside': {
        'rarity': 0.30,
        'body_plan': 'elongated',
        'pattern': PATTERN_VBANDS,
        'behavior': 'school',
        'size_range': (0.13, 0.20),
        'speed_range': (1.2, 1.7),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.75, 0.95, n),
            np.random.uniform(0.75, 0.95, n),
            np.random.uniform(0.85, 1.00, n),
        ]),
        'school_distance': 1.5,   # feet
        'school_strength': 0.5,
    },
    'tang': {
        'rarity': 0.10,
        'body_plan': 'round',
        'pattern': PATTERN_GRADIENT,
        'behavior': 'cruise',
        'size_range': (0.30, 0.50),
        'speed_range': (0.6, 1.0),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.20, 0.40, n),
            np.random.uniform(0.50, 0.70, n),
            np.random.uniform(0.85, 1.00, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'clownfish': {
        'rarity': 0.08,
        'body_plan': 'standard',
        'pattern': PATTERN_HSTRIPE,
        'behavior': 'hover',
        'size_range': (0.20, 0.30),
        'speed_range': (0.4, 0.7),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.90, 1.00, n),
            np.random.uniform(0.40, 0.55, n),
            np.random.uniform(0.10, 0.25, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'damselfish': {
        'rarity': 0.15,
        'body_plan': 'standard',
        'pattern': PATTERN_VBANDS,
        'behavior': 'school',
        'size_range': (0.15, 0.25),
        'speed_range': (0.8, 1.4),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.30, 0.50, n),
            np.random.uniform(0.60, 0.80, n),
            np.random.uniform(0.55, 0.75, n),
        ]),
        'school_distance': 1.3,
        'school_strength': 0.45,
    },
    'grouper': {
        'rarity': 0.02,
        'body_plan': 'standard',
        'pattern': PATTERN_SPOTS,
        'behavior': 'bottom_dweller',
        'size_range': (0.55, 1.00),
        'speed_range': (0.3, 0.5),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.40, 0.60, n),
            np.random.uniform(0.40, 0.55, n),
            np.random.uniform(0.30, 0.45, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'angelfish': {
        'rarity': 0.05,
        'body_plan': 'round',
        'pattern': PATTERN_EYESPOT,
        'behavior': 'cruise',
        'size_range': (0.40, 0.60),
        'speed_range': (0.4, 0.7),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.85, 1.00, n),
            np.random.uniform(0.70, 0.90, n),
            np.random.uniform(0.20, 0.40, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'moray_eel': {
        'rarity': 0.02,
        'body_plan': 'eel',
        'pattern': PATTERN_SPOTS,
        'behavior': 'bottom_dweller',
        'size_range': (0.90, 1.60),
        'speed_range': (0.4, 0.7),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.30, 0.50, n),
            np.random.uniform(0.35, 0.50, n),
            np.random.uniform(0.20, 0.35, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'ribbon_eel': {
        'rarity': 0.02,
        'body_plan': 'eel',
        'pattern': PATTERN_GRADIENT,
        'behavior': 'cruise',
        'size_range': (0.50, 0.90),
        'speed_range': (0.5, 0.9),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.10, 0.30, n),
            np.random.uniform(0.30, 0.50, n),
            np.random.uniform(0.70, 0.95, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'ray': {
        'rarity': 0.02,
        'body_plan': 'flat',
        'pattern': PATTERN_PLAIN,
        'behavior': 'bottom_dweller',
        'size_range': (0.85, 1.50),
        'speed_range': (0.5, 0.9),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.30, 0.45, n),
            np.random.uniform(0.30, 0.45, n),
            np.random.uniform(0.35, 0.50, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'barracuda': {
        'rarity': 0.02,
        'body_plan': 'elongated',
        'pattern': PATTERN_GRADIENT,
        'behavior': 'dart',
        'size_range': (0.60, 1.10),
        'speed_range': (1.5, 2.5),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.55, 0.75, n),
            np.random.uniform(0.60, 0.80, n),
            np.random.uniform(0.65, 0.85, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'pufferfish': {
        'rarity': 0.05,
        'body_plan': 'round',
        'pattern': PATTERN_SPOTS,
        'behavior': 'hover',
        'size_range': (0.25, 0.40),
        'speed_range': (0.3, 0.5),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.75, 0.95, n),
            np.random.uniform(0.65, 0.85, n),
            np.random.uniform(0.30, 0.50, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'butterflyfish': {
        'rarity': 0.07,
        'body_plan': 'round',
        'pattern': PATTERN_VBANDS,
        'behavior': 'cruise',
        'size_range': (0.20, 0.35),
        'speed_range': (0.5, 0.9),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.85, 1.00, n),
            np.random.uniform(0.80, 0.95, n),
            np.random.uniform(0.20, 0.40, n),
        ]),
        'school_distance': 0.0,
        'school_strength': 0.0,
    },
    'jack': {
        'rarity': 0.07,
        'body_plan': 'elongated',
        'pattern': PATTERN_PLAIN,
        'behavior': 'school',
        'size_range': (0.40, 0.65),
        'speed_range': (1.0, 1.6),
        'color': lambda n: np.column_stack([
            np.random.uniform(0.65, 0.85, n),
            np.random.uniform(0.70, 0.90, n),
            np.random.uniform(0.75, 0.95, n),
        ]),
        'school_distance': 1.7,
        'school_strength': 0.45,
    },
}

class FishEffect(ShaderEffect):
    """GPU-based fish swimming effect using instanced rendering"""
    
    # Behavior IDs (kept as ints for fast numpy masking).
    BEHAVIOR_IDS = {
        'cruise': 0,
        'school': 1,
        'dart': 2,
        'hover': 3,
        'bottom_dweller': 4,
        'surface': 5,
    }

    def __init__(self, viewport, num_fish: int = 30, activity: float = 1.0, squish_top_width: float = 1.0):
        super().__init__(viewport)
        self.fan = FanCoords(viewport.width, viewport.height)
        self.num_fish = num_fish
        self.base_num_fish = num_fish
        self.target_fish = num_fish
        self.activity = activity
        self.instance_VBO = None
        self.time = 0.0  # accumulated time for behavior phase

        # Width scaling based on vertical position (legacy hack for flat-mode display)
        self.squish_top_width = squish_top_width

        # Physical state — fish swim along arcs of constant radius.
        self.thetas = None        # rad in [0, pi]
        self.r_fts = None         # ft in [INNER_R_FT, OUTER_R_FT]
        self.zs = None            # depth 0..100 (kept from old model)
        self.omegas = None        # rad/s along theta
        self.r_drifts = None      # ft/s radial drift
        self.base_speeds_fps = None  # target physical speed in ft/s
        self.directions = None    # +1/-1 initial sign of omega
        self.sizes_ft = None      # physical size in feet (used directly by the vertex shader)
        self.swim_phases = None   # tail-wiggle animation
        self.swim_speeds = None   # tail-wiggle frequency
        self.alphas = None
        self.colors = None
        self.species = None       # species index per fish
        self.body_lengths = None  # body morph (per fish)
        self.body_heights = None
        self.body_tapers = None
        self.tail_spreads = None
        self.patterns = None      # int pattern id per fish
        self.behaviors = None     # int behavior id per fish
        self.dart_phases = None   # phase offset for dart bursts
        self.rotations = None     # per-fish sprite rotation (radians), computed each frame

        # Buffer-pixel positions, recomputed each frame from (theta, r). Kept as
        # an (n, 3) array because the existing render path samples positions[:, 2]
        # (depth) for back-to-front sorting.
        self.positions = None

        self.species_names = list(FISH_SPECIES.keys())

        self._initialize_fish()

    # ------------------------------------------------------------------
    # Species data generation
    # ------------------------------------------------------------------

    def _build_species_arrays(self, species_indices):
        """Generate per-fish arrays for a batch of fish given their species ids.

        Returns a dict of numpy arrays the caller can splice into self.* fields.
        Also handles the spawn-band restriction for bottom_dweller / surface
        behaviors so those species end up in the right radial band.
        """
        n = len(species_indices)
        out = {
            'thetas':           np.random.uniform(0.0, np.pi, n),
            'r_fts':            np.empty(n),
            'zs':               np.random.uniform(0, 100, n),
            'directions':       np.random.choice([-1.0, 1.0], n).astype(np.float64),
            'base_speeds_fps':  np.empty(n),
            'sizes_ft':         np.empty(n),
            'colors':           np.empty((n, 3)),
            'body_lengths':     np.empty(n),
            'body_heights':     np.empty(n),
            'body_tapers':      np.empty(n),
            'tail_spreads':     np.empty(n),
            'patterns':         np.empty(n),
            'behaviors':        np.empty(n),
            'r_drifts':         np.zeros(n),
            'dart_phases':      np.random.uniform(0, 2 * np.pi, n),
        }

        for sp_idx, sp_name in enumerate(self.species_names):
            mask = species_indices == sp_idx
            k = int(np.sum(mask))
            if k == 0:
                continue
            spec = FISH_SPECIES[sp_name]

            out['sizes_ft'][mask] = np.random.uniform(*spec['size_range'], k)
            out['base_speeds_fps'][mask] = np.random.uniform(*spec['speed_range'], k)
            out['colors'][mask] = spec['color'](k)

            bl, bh, bt, ts = BODY_PLANS[spec['body_plan']]
            out['body_lengths'][mask] = bl
            out['body_heights'][mask] = bh
            out['body_tapers'][mask] = bt
            out['tail_spreads'][mask] = ts
            out['patterns'][mask] = float(spec['pattern'])

            beh_id = self.BEHAVIOR_IDS[spec['behavior']]
            out['behaviors'][mask] = float(beh_id)

            # Per-behavior radial spawn band.
            if spec['behavior'] == 'bottom_dweller':
                # Leave several feet of rim margin so a big moray/ray/grouper
                # can wiggle radially without poking past the fan's outer rim.
                out['r_fts'][mask] = np.random.uniform(12.0, OUTER_R_FT - 3.0, k)
            elif spec['behavior'] == 'surface':
                out['r_fts'][mask] = np.random.uniform(SPAWN_R_MIN, SPAWN_R_MIN + 3.0, k)
            else:
                # Size-based banding: small fish get pushed toward the outer
                # rim (bottom of display), where there's more arc length / more
                # LEDs per foot so their features are readable; big fish get
                # pushed toward the apex where they have room to be seen.
                # Size range across all species is roughly 0.13..3.2 ft.
                sz = out['sizes_ft'][mask]
                size_norm = np.clip((sz - 0.13) / (3.2 - 0.13), 0.0, 1.0)
                # Soften the banding: instead of forcing big fish fully to
                # the apex, gently bias them inward. This keeps the size
                # sorting visible without cramming a dozen barracudas into
                # a 2 ft band near the inner rim.
                preferred_r = SPAWN_R_MAX - size_norm * (SPAWN_R_MAX - SPAWN_R_MIN) * 0.55
                # Wide jitter so same-species fish spread out across radii.
                jitter = np.random.uniform(-4.0, 4.0, k)
                out['r_fts'][mask] = np.clip(preferred_r + jitter, SPAWN_R_MIN, SPAWN_R_MAX)

            # Modest radial drift -- enough to give some diagonal motion but
            # not so much that fish slam into the water surface or rim.
            if spec['behavior'] in ('bottom_dweller', 'surface'):
                out['r_drifts'][mask] = 0.0
            else:
                out['r_drifts'][mask] = np.random.uniform(-0.25, 0.25, k) * FISH_DRIFT_SCALE

        # Visual size boost: see FISH_DISPLAY_SIZE_SCALE.
        # Also reintroduce the old depth-based size falloff so distant fish look smaller.
        out['sizes_ft'] *= FISH_DISPLAY_SIZE_SCALE
        depth_factors = 1.0 - (out['zs'] / 100.0)  # 1 = near, 0 = far
        out['sizes_ft'] *= 0.65 + 0.35 * depth_factors

        # Hard cap on final display size so a fish can never be so big it
        # wraps around the inner-radius singularity or stretches off the fan.
        np.minimum(out['sizes_ft'], FISH_MAX_SIZE_FT, out=out['sizes_ft'])

        # Initial omega from base speed: omega = (speed / r) * direction
        out['omegas'] = (out['base_speeds_fps'] / out['r_fts']) * out['directions']
        return out

    def _initialize_fish(self):
        """Initialize all fish data as numpy arrays with species variety."""
        n = self.num_fish

        # Assign species based on rarity weights
        rarities = [FISH_SPECIES[sp]['rarity'] for sp in self.species_names]
        # Normalize in case rarities don't sum to exactly 1.0
        rarities = np.array(rarities, dtype=np.float64)
        rarities /= rarities.sum()
        self.species = np.random.choice(len(self.species_names), size=n, p=rarities)

        data = self._build_species_arrays(self.species)
        self.thetas = data['thetas']
        self.r_fts = data['r_fts']
        self.zs = data['zs']
        self.omegas = data['omegas']
        self.r_drifts = data['r_drifts']
        self.base_speeds_fps = data['base_speeds_fps']
        self.directions = data['directions']
        self.sizes_ft = data['sizes_ft']
        self.colors = data['colors']
        self.body_lengths = data['body_lengths']
        self.body_heights = data['body_heights']
        self.body_tapers = data['body_tapers']
        self.tail_spreads = data['tail_spreads']
        self.patterns = data['patterns']
        self.behaviors = data['behaviors']
        self.dart_phases = data['dart_phases']

        # Depth-driven alpha (kept from old model — gives back-to-front variation).
        depth_factors = 1.0 - (self.zs / 100.0)
        self.alphas = np.ones_like(depth_factors)

        # Tail-wiggle animation. Old code did: 8 + base_speed/10 with px-speeds in
        # the 6..35 range. New base_speeds are in ft/s (0.15..2.5), so use a
        # different rescale to land in roughly the same hz range.
        self.swim_phases = np.random.uniform(0, 2 * np.pi, n)
        self.swim_speeds = 6.0 + self.base_speeds_fps * 5.0

        # instance_pos (physical x_ft, y_ft, z_depth) and rotations are filled by _refresh_buffer_state.
        self.positions = np.zeros((n, 3))
        self.rotations = np.zeros(n)
        self._refresh_buffer_state()

    def _refresh_buffer_state(self):
        """Recompute physical-space fish positions and sprite rotations from
        (theta, r, omega, r_drift) state. The vertex shader does the
        physical->buffer conversion per template vertex, so we only need to
        hand it (x_ft, y_ft) per fish here.
        """
        if self.thetas is None or len(self.thetas) == 0:
            return

        # Fish center in physical feet.
        cos_t = np.cos(self.thetas)
        sin_t = np.sin(self.thetas)
        x_phys = self.r_fts * cos_t
        y_phys = self.r_fts * sin_t

        # instance_pos is now (x_ft, y_ft, z_depth).
        self.positions[:, 0] = x_phys
        self.positions[:, 1] = y_phys
        self.positions[:, 2] = self.zs

        # Rotation in the LOCAL POLAR FRAME, not in cartesian.
        #
        # The vertex shader treats the template's +x axis as "along the arc
        # tangent" and the +y axis as "radially outward" at the fish's
        # position. The fish's local-frame velocity is:
        #     tangential component = r * omega   (ft/s along arc)
        #     radial component     = r_drift     (ft/s outward)
        # The angle of that velocity vector in the local frame is the amount
        # we need to rotate the template so its head points along the motion.
        #
        # Doing the rotation in the LOCAL frame (instead of cartesian) means
        # "top of fish" is always radially outward, regardless of where the
        # fish sits on the arc. A fish at theta=0 and a fish at theta=pi have
        # the same body-up direction relative to the fan, as you'd expect.
        local_tangential = self.r_fts * self.omegas       # ft/s along tangent
        local_radial     = self.r_drifts                  # ft/s radially outward
        self.rotations = np.arctan2(local_radial, local_tangential)


    def _apply_index_mask(self, keep_mask):
        """Filter every per-fish array down to the rows where keep_mask is True."""
        self.thetas = self.thetas[keep_mask]
        self.r_fts = self.r_fts[keep_mask]
        self.zs = self.zs[keep_mask]
        self.omegas = self.omegas[keep_mask]
        self.r_drifts = self.r_drifts[keep_mask]
        self.base_speeds_fps = self.base_speeds_fps[keep_mask]
        self.directions = self.directions[keep_mask]
        self.sizes_ft = self.sizes_ft[keep_mask]
        self.swim_phases = self.swim_phases[keep_mask]
        self.swim_speeds = self.swim_speeds[keep_mask]
        self.alphas = self.alphas[keep_mask]
        self.colors = self.colors[keep_mask]
        self.species = self.species[keep_mask]
        self.body_lengths = self.body_lengths[keep_mask]
        self.body_heights = self.body_heights[keep_mask]
        self.body_tapers = self.body_tapers[keep_mask]
        self.tail_spreads = self.tail_spreads[keep_mask]
        self.patterns = self.patterns[keep_mask]
        self.behaviors = self.behaviors[keep_mask]
        self.dart_phases = self.dart_phases[keep_mask]
        self.rotations = self.rotations[keep_mask]
        self.positions = self.positions[keep_mask]
        self.num_fish = len(self.thetas)

    def _reset_fish(self, off_mask):
        """Re-spawn fish that have swum past theta=0 / theta=pi.

        If we're currently over the activity target, drop the off-arc fish
        instead of recycling them.
        """
        n_reset = int(np.sum(off_mask))
        if n_reset == 0:
            return

        current_count = self.num_fish
        if current_count > self.target_fish:
            n_to_remove = min(n_reset, current_count - self.target_fish)
            reset_indices = np.where(off_mask)[0]
            remove_indices = reset_indices[:n_to_remove]
            keep_mask = np.ones(current_count, dtype=bool)
            keep_mask[remove_indices] = False
            self._apply_index_mask(keep_mask)

            if n_to_remove >= n_reset:
                return
            # Recompute mask in the (now smaller) array.
            new_off = np.zeros(self.num_fish, dtype=bool)
            for old_idx in reset_indices[n_to_remove:]:
                new_off[old_idx - int(np.sum(remove_indices < old_idx))] = True
            off_mask = new_off
            n_reset = int(np.sum(off_mask))
            if n_reset == 0:
                return

        # Re-spawn: flip species, restart from opposite theta edge.
        rarities = np.array([FISH_SPECIES[sp]['rarity'] for sp in self.species_names], dtype=np.float64)
        rarities /= rarities.sum()
        new_species = np.random.choice(len(self.species_names), size=n_reset, p=rarities)
        data = self._build_species_arrays(new_species)

        idx = np.where(off_mask)[0]
        thetas_at_exit = self.thetas[idx]
        # Respawn OUTSIDE the visible arc on the opposite side so fish visibly
        # swim INTO view rather than appearing mid-arc. A fish that exited
        # theta=0 (right) comes in just past theta=pi (left edge), and its
        # omega drives it back toward 0 (negative omega in our convention).
        exited_right = thetas_at_exit < np.pi * 0.5
        new_thetas = np.where(exited_right, np.pi + 0.20, -0.20)
        new_dirs = np.where(exited_right, -1.0, 1.0)

        self.species[idx] = new_species
        self.thetas[idx] = new_thetas
        self.r_fts[idx] = data['r_fts']
        self.zs[idx] = data['zs']
        self.directions[idx] = new_dirs
        self.base_speeds_fps[idx] = data['base_speeds_fps']
        self.omegas[idx] = (data['base_speeds_fps'] / data['r_fts']) * new_dirs
        self.r_drifts[idx] = data['r_drifts']
        self.sizes_ft[idx] = data['sizes_ft']
        self.colors[idx] = data['colors']
        self.body_lengths[idx] = data['body_lengths']
        self.body_heights[idx] = data['body_heights']
        self.body_tapers[idx] = data['body_tapers']
        self.tail_spreads[idx] = data['tail_spreads']
        self.patterns[idx] = data['patterns']
        self.behaviors[idx] = data['behaviors']
        self.dart_phases[idx] = data['dart_phases']
        self.swim_phases[idx] = np.random.uniform(0, 2 * np.pi, n_reset)
        self.swim_speeds[idx] = 6.0 + data['base_speeds_fps'] * 5.0
        # alpha stays consistent with depth (zs unchanged for reset).
        depth_factors = 1.0 - (self.zs[idx] / 100.0)
        self.alphas[idx] = np.ones_like(depth_factors)

    def set_target_activity(self, activity):
        """Set target marine life activity and adjust fish count."""
        self.activity = activity
        self.target_fish = int(self.base_num_fish * activity)

    def _add_fish(self, n_to_add):
        """Add new fish to the simulation."""
        if n_to_add <= 0:
            return

        rarities = np.array([FISH_SPECIES[sp]['rarity'] for sp in self.species_names], dtype=np.float64)
        rarities /= rarities.sum()
        new_species = np.random.choice(len(self.species_names), size=n_to_add, p=rarities)
        data = self._build_species_arrays(new_species)

        depth_factors = 1.0 - (data['zs'] / 100.0)
        new_alphas = np.ones_like(depth_factors)
        new_swim_phases = np.random.uniform(0, 2 * np.pi, n_to_add)
        new_swim_speeds = 6.0 + data['base_speeds_fps'] * 5.0

        self.thetas = np.concatenate([self.thetas, data['thetas']])
        self.r_fts = np.concatenate([self.r_fts, data['r_fts']])
        self.zs = np.concatenate([self.zs, data['zs']])
        self.omegas = np.concatenate([self.omegas, data['omegas']])
        self.r_drifts = np.concatenate([self.r_drifts, data['r_drifts']])
        self.base_speeds_fps = np.concatenate([self.base_speeds_fps, data['base_speeds_fps']])
        self.directions = np.concatenate([self.directions, data['directions']])
        self.sizes_ft = np.concatenate([self.sizes_ft, data['sizes_ft']])
        self.swim_phases = np.concatenate([self.swim_phases, new_swim_phases])
        self.swim_speeds = np.concatenate([self.swim_speeds, new_swim_speeds])
        self.alphas = np.concatenate([self.alphas, new_alphas])
        self.colors = np.vstack([self.colors, data['colors']])
        self.species = np.concatenate([self.species, new_species])
        self.body_lengths = np.concatenate([self.body_lengths, data['body_lengths']])
        self.body_heights = np.concatenate([self.body_heights, data['body_heights']])
        self.body_tapers = np.concatenate([self.body_tapers, data['body_tapers']])
        self.tail_spreads = np.concatenate([self.tail_spreads, data['tail_spreads']])
        self.patterns = np.concatenate([self.patterns, data['patterns']])
        self.behaviors = np.concatenate([self.behaviors, data['behaviors']])
        self.dart_phases = np.concatenate([self.dart_phases, data['dart_phases']])
        self.rotations = np.concatenate([self.rotations, np.zeros(n_to_add)])
        self.positions = np.vstack([self.positions, np.zeros((n_to_add, 3))])

        self.num_fish = len(self.thetas)


    def compile_shader(self):
        """Compile vertex and fragment shaders for fish rendering.

        The vertex shader morphs a unit fish template (1 center + 16 body
        outline + 7 tail vertices, total 24) per instance using a vec4 of
        body parameters: (body_length, body_height, body_taper, tail_spread).
        Tail vertices are detected via gl_VertexID >= 17.

        The fragment shader picks one of several pattern styles via
        instance_pattern (an int passed as float).
        """
        vertex_shader = """
        #version 310 es
        precision highp float;

        layout(location = 0) in vec2 position;            // template vertex (unit fish)
        layout(location = 1) in vec3 instance_pos;        // (x_ft, y_ft, z_depth)
        layout(location = 2) in float instance_size_ft;   // overall fish length in feet
        layout(location = 3) in float instance_alpha;
        layout(location = 4) in vec3 instance_color;
        layout(location = 5) in float instance_reserved;  // unused (was size_y)
        layout(location = 6) in float instance_rotation;  // radians in PHYSICAL space
        layout(location = 7) in float instance_swim_phase;
        layout(location = 8) in vec4 instance_body;       // length, height, taper, tail_spread
        layout(location = 9) in float instance_pattern;

        out float fragAlpha;
        out vec3 fragColor;
        out vec2 fragCoord;
        out float fragPattern;

        """ + FAN_COORDS_UNIFORMS + FAN_COORDS_GLSL + """

        // Non-clamped physical->UV for per-vertex fish rendering.
        // Replicates FAN_COORDS_GLSL without the [0,1] clamp so off-fan
        // vertices fall outside the viewport and get clipped naturally.
        vec2 fish_physical_to_uv(vec2 xy) {
            float r = length(xy);
            float theta = atan(xy.y, xy.x);
            float u = 1.0 - theta / FAN_PI;
            float v = (r - u_inner_r_ft) / (u_outer_r_ft - u_inner_r_ft);
            return vec2(u, v);
        }

        // Shift a vertex's u so that it falls within 0.5 of a reference u.
        // atan2 has a discontinuity at theta = pi, and any fish whose template
        // vertex crosses below y=0 (below the fan's diameter line) will have
        // u jump by ~1, creating a triangle that spans the whole fan width.
        // We anchor the vertex to the fish center's unwrapped u.
        float unwrap_u(float vu, float ref_u) {
            float du = vu - ref_u;
            du -= floor(du + 0.5);  // bring to [-0.5, 0.5]
            return ref_u + du;
        }

        const int TAIL_START = 17; // first tail vertex index in the 24-vertex template

        void main() {
            float bodyLen = instance_body.x;
            float bodyHt  = instance_body.y;
            float bodyTaper = instance_body.z;
            float tailSpread = instance_body.w;

            // Morph the unit template (positions roughly in [-1.1, +1] x [-0.6, +0.6]).
            vec2 morphed = position;
            if (gl_VertexID >= TAIL_START) {
                morphed.x *= bodyLen;
                morphed.y *= tailSpread;
            } else {
                morphed.x *= bodyLen;
                morphed.y *= bodyHt;
                if (position.x < 0.0) {
                    float t = clamp(-position.x, 0.0, 1.0);
                    morphed.y *= mix(1.0, bodyTaper, t);
                }
            }
            float tail_wiggle = sin(instance_swim_phase) * 0.15;
            if (position.x < 0.0) {
                morphed.y += tail_wiggle * abs(position.x);
            }

            // Fish center UV and clip position (safe fallback for unsafe verts).
            vec2 center_uv = fish_physical_to_uv(instance_pos.xy);
            vec4 center_clip = vec4(
                center_uv.x * 2.0 - 1.0,
                1.0 - center_uv.y * 2.0,
                0.0, 1.0
            );

            // ---- Rotate the template in the LOCAL POLAR FRAME ----
            //
            // instance_rotation is the fish's velocity angle in the local
            // (tangent, radial_outward) frame, NOT in cartesian. After this
            // rotation:
            //   rotated.x = displacement in feet ALONG THE TANGENT (arc len)
            //   rotated.y = displacement in feet OUTWARD in radius
            //
            // Doing the rotation in the local frame (instead of cartesian)
            // makes "top of fish" always radially outward regardless of where
            // the fish sits on the arc. The old cartesian-rotation code had
            // template +y pointing in different physical directions at
            // different positions, which made fish render correctly in
            // rectangular buffer space but incorrectly in fan space.
            float cs = cos(instance_rotation);
            float sn = sin(instance_rotation);
            vec2 rotated = vec2(
                morphed.x * cs - morphed.y * sn,
                morphed.x * sn + morphed.y * cs
            );

            // ---- Convert template offsets to polar displacements ----
            //
            // rotated.x is the template's along-tangent offset (unitless,
            // template spans roughly -1 .. +1 for a unit fish).
            // rotated.y is the template's radial offset (same unit basis).
            //
            // To keep ANGULAR width constant across the fan (same arc-degrees
            // at any radius) we convert the tangent offset directly to delta
            // theta with a radius-independent angular-size factor:
            //
            //   dtheta = rotated.x * (size_ft / mid_r)
            //
            // Equivalent to arc length = size_ft * (r / mid_r) at radius r,
            // but computed straight in angular terms so big fish can't blow
            // up to 45 deg of arc when size_ft gets large.
            //
            // The radial direction stays in honest feet -- fish thickness
            // should NOT depend on the position along the arc.
            float center_theta = atan(instance_pos.y, instance_pos.x);
            float center_r = length(instance_pos.xy);
            float mid_r = (u_inner_r_ft + u_outer_r_ft) * 0.5;

            // Template +y is the fish's BACK. We want the back pointing toward
            // the water surface (the apex, = smaller radius), so flip the sign.
            float dtheta = rotated.x * (instance_size_ft / mid_r);
            float perp_ft = -rotated.y * instance_size_ft;

            float vert_theta = center_theta + dtheta;
            float vert_r = center_r + perp_ft;
            vec2 world_ft = vec2(vert_r * cos(vert_theta),
                                 vert_r * sin(vert_theta));

            // Physical -> buffer UV for this vertex, unclamped and unwrapped
            // around the center so atan2 discontinuities don't fling a vertex
            // to the opposite side of the buffer.
            vec2 vertex_uv = fish_physical_to_uv(world_ft);
            vec2 uv = vec2(
                unwrap_u(vertex_uv.x, center_uv.x),
                vertex_uv.y
            );

            // ---- Per-vertex geometric safety ----
            //
            // Two cases require collapsing a vertex to the fish center (which
            // makes any triangle containing it degenerate and non-rasterizing):
            //
            //   (a) Vertex falls inside the fan's apex hole (r < inner_r).
            //       Here atan2 returns an angle from the wrong side of the
            //       origin and the triangle would wrap halfway around the fan.
            //   (b) The fish CENTER itself is geometrically unsafe.
            //       If the center is invalid, kill the whole fish.
            //
            // Everything else -- vertices slightly past the outer rim, past
            // theta = 0 or theta = pi -- is rendered normally. unwrap_u keeps
            // the vertex u within 0.5 of the center u, so these vertices land
            // just outside [0, 1] and the GPU clips the triangle against the
            // viewport. The clipped remainder is tiny (a few pixels wide), so
            // there is no visible stretching.
            //
            // IMPORTANT: do NOT push unsafe vertices to gl_Position = (2, 2, ...)
            // to "kill" them. That leaves them as valid vertices outside the
            // frustum; the GPU clips the triangle against the frustum and the
            // clipped remainder stretches across the buffer, giving the
            // "fish spans 128 pixels" bug. Collapsing to the center clip
            // position is the only safe way.
            float world_r = length(world_ft);
            // Only collapse the fish if the center is DEFINITELY off-fan.
            // Do NOT reject fish just because center_y < 0 -- fish swimming
            // off the theta=0 / theta=pi edges have y dipping below zero for
            // a moment while they transition out, and GPU viewport clipping
            // handles them naturally. Rejecting on y<0 makes fish "pop" out
            // of existence the instant they cross the edge.
            bool center_unsafe = (center_r < u_inner_r_ft + 0.1);
            bool vertex_unsafe = (world_r < u_inner_r_ft * 0.9);

            if (center_unsafe || vertex_unsafe) {
                gl_Position = center_clip;
            } else {
                gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
            }

            fragAlpha = instance_alpha;
            fragColor = instance_color;
            fragCoord = position;
            fragPattern = instance_pattern;
        }
        """

        fragment_shader = """
        #version 310 es
        precision highp float;

        in float fragAlpha;
        in vec3 fragColor;
        in vec2 fragCoord;
        in float fragPattern;

        out vec4 FragColor;

        // Cheap noise for spotted patterns.
        float hash(vec2 p) {
            p = fract(p * vec2(123.45, 678.90));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
        }

        void main() {
            // Body / tail bounds in template space (unchanged from old shader).
            bool inTail = fragCoord.x < -0.6;
            float dist = length(vec2(fragCoord.x * 1.5, fragCoord.y));
            if (!inTail && dist > 1.0) discard;
            if (inTail) {
                float tailDist = abs(fragCoord.y);
                if (tailDist > 0.6) discard;
            }

            vec3 color = fragColor;
            int pat = int(fragPattern + 0.5);

            // Dorsal stripe is shared across most patterns (skipped for plain).
            if (pat != 0) {
                float dorsalStripe = smoothstep(0.15, 0.2, fragCoord.y) * smoothstep(0.35, 0.3, fragCoord.y);
                color = mix(color, color * 0.6, dorsalStripe * 0.7);
            }

            if (pat == 1) {
                // Vertical bands: 3 clean, wide dark bars across the body.
                // Template x runs roughly -1.1 .. +1.0. Use explicit band
                // centers so at small fish sizes the stripes don't alias
                // into noise.
                float bar0 = smoothstep(0.15, 0.05, abs(fragCoord.x + 0.55));
                float bar1 = smoothstep(0.15, 0.05, abs(fragCoord.x - 0.00));
                float bar2 = smoothstep(0.15, 0.05, abs(fragCoord.x - 0.55));
                float bars = clamp(bar0 + bar1 + bar2, 0.0, 1.0);
                color = mix(color, color * 0.45, bars);
            } else if (pat == 2) {
                // Horizontal stripes (clownfish-style)
                float band1 = smoothstep(0.05, 0.10, abs(abs(fragCoord.y) - 0.20));
                float band2 = smoothstep(0.05, 0.10, abs(abs(fragCoord.y) - 0.45));
                float bands = (1.0 - band1) + (1.0 - band2);
                color = mix(color, vec3(1.0), clamp(bands, 0.0, 1.0) * 0.6);
            } else if (pat == 3) {
                // Spots
                float spots = step(0.65, hash(floor(fragCoord * 12.0)));
                color = mix(color, color * 0.5, spots * 0.7);
            } else if (pat == 4) {
                // Smooth dorsal->belly gradient (no bands)
                float grad = clamp(fragCoord.y * 2.0 + 0.5, 0.0, 1.0);
                color = mix(color * 0.6, color * 1.25, grad);
            } else if (pat == 5) {
                // False eyespot near the tail
                vec2 eyePos = vec2(-0.4, 0.0);
                float d = length((fragCoord - eyePos) * vec2(1.5, 1.0));
                float ring  = smoothstep(0.20, 0.16, d) * (1.0 - smoothstep(0.10, 0.12, d));
                float pupil = smoothstep(0.07, 0.05, d);
                color = mix(color, vec3(0.05, 0.05, 0.10), pupil);
                color = mix(color, vec3(0.95, 0.95, 0.30), ring * 0.7);
            }
            // pat == 0 (plain) gets no extra pattern beyond the dorsal skip.

            // Belly highlight: lighter underside.
            float bellyGradient = smoothstep(0.0, -0.4, fragCoord.y);
            color = mix(color, color * 1.3, bellyGradient * 0.4);

            // Centerline sheen.
            float centerHighlight = 1.0 - abs(fragCoord.y) * 2.0;
            centerHighlight = smoothstep(0.3, 0.8, centerHighlight);
            color += vec3(0.15, 0.15, 0.2) * centerHighlight * 0.25;

            // Eye on the front of the body.
            vec2 eyePos = vec2(0.5, 0.15);
            float eyeDist = length(fragCoord - eyePos);
            float eye = smoothstep(0.08, 0.05, eyeDist);
            color = mix(color, vec3(0.1, 0.1, 0.15), eye);
            float eyeGlint = smoothstep(0.03, 0.01, length(fragCoord - (eyePos + vec2(-0.015, 0.015))));
            color = mix(color, vec3(1.0, 1.0, 1.0), eyeGlint * 0.9);

            // Tail fin darkening (kept from old shader for visual continuity).
            if (fragCoord.x < -0.3) {
                float tailPattern = sin(fragCoord.y * 12.0) * 0.5 + 0.5;
                float tailMask = smoothstep(0.4, 0.6, tailPattern);
                float tailIntensity = smoothstep(-0.3, -0.8, fragCoord.x);
                color = mix(color, color * 0.7, tailMask * tailIntensity * 0.5);
            }

            float bodyGradient = 1.0 - dist * 0.3;
            color *= bodyGradient;

            // Fully opaque fish. fragAlpha is still piped through in case
            // future effects want to fade individual fish, but the old
            // edge-fade was turning every fish into a semi-transparent blob.
            FragColor = vec4(color, fragAlpha);
        }
        """

        return shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )
    
    # Instance VBO layout (16 floats per instance):
    #   offset 0..2   pos (x_ft, y_ft, z)     location 1
    #   offset 3      size_ft                 location 2
    #   offset 4      alpha                   location 3
    #   offset 5..7   color                   location 4
    #   offset 8      reserved                location 5
    #   offset 9      rotation (radians)      location 6
    #   offset 10     swim_phase              location 7
    #   offset 11..14 body (len, ht, taper, tail_spread)  location 8
    #   offset 15     pattern                 location 9
    INSTANCE_FLOATS = 16

    def setup_buffers(self):
        """Set up VAO and VBOs for instanced rendering."""
        # 1 center + 16 body outline + 7 tail = 24 vertices.
        center = np.array([[0.0, 0.0]])

        body_segments = 16
        angles = np.linspace(-np.pi * 0.7, np.pi * 0.7, body_segments)
        x_coords = np.cos(angles)
        y_coords = np.sin(angles) * 0.5
        x_coords = x_coords * (0.6 + 0.4 * (1 + x_coords) / 2)
        body_outline = np.column_stack([x_coords, y_coords])

        tail_vertices = np.array([
            [-0.6, 0.25],
            [-0.9, 0.5],
            [-1.1, 0.55],
            [-0.9, 0.0],
            [-1.1, -0.55],
            [-0.9, -0.5],
            [-0.6, -0.25],
        ])

        fish_vertices = np.vstack([center, body_outline, tail_vertices]).astype(np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # Static template VBO
        fish_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, fish_VBO)
        glBufferData(GL_ARRAY_BUFFER, fish_vertices.nbytes, fish_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(fish_VBO)

        # Dynamic instance VBO
        self.instance_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        stride = self.INSTANCE_FLOATS * 4
        # Allocate enough space for the maximum expected fish count. We grow on
        # demand in render() if it's not enough.
        initial_capacity = max(self.num_fish, 1)
        glBufferData(GL_ARRAY_BUFFER, initial_capacity * stride, None, GL_DYNAMIC_DRAW)
        self._instance_capacity = initial_capacity

        def _attr(loc, count, offset_floats):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, count, GL_FLOAT, GL_FALSE, stride,
                                  ctypes.c_void_p(offset_floats * 4))
            glVertexAttribDivisor(loc, 1)

        _attr(1, 3, 0)   # instance_pos (x_ft, y_ft, z)
        _attr(2, 1, 3)   # instance_size_ft
        _attr(3, 1, 4)   # instance_alpha
        _attr(4, 3, 5)   # instance_color
        _attr(5, 1, 8)   # instance_reserved (unused)
        _attr(6, 1, 9)   # instance_rotation (radians, physical-space)
        _attr(7, 1, 10)  # instance_swim_phase
        _attr(8, 4, 11)  # instance_body (length, height, taper, tail_spread)
        _attr(9, 1, 15)  # instance_pattern

        self.VBOs.append(self.instance_VBO)
        glBindVertexArray(0)

        # Upload static fan-coord uniforms once (used by fan_physical_to_uv in the vertex shader).
        glUseProgram(self.shader)
        self.fan.set_uniforms(self.shader)
        glUseProgram(0)
    
    # ------------------------------------------------------------------
    # Per-frame physics
    # ------------------------------------------------------------------

    def update(self, dt: float, state: Dict):
        """Advance fish along their arcs and apply behavior modulation."""
        if self.num_fish == 0:
            return

        # Top up to target population.
        if self.target_fish > self.num_fish:
            self._add_fish(self.target_fish - self.num_fish)

        self.time += dt

        # Tail-wiggle animation.
        self.swim_phases += self.swim_speeds * dt

        # ---- Behavior-driven omega/r_drift modulation ----
        # Start each frame from the species base omega = (speed / r) * direction.
        base_speeds = self.base_speeds_fps
        base_omegas = (base_speeds / np.maximum(self.r_fts, 0.01)) * self.directions

        # Default omega is the base rate; behaviors override.
        self.omegas = base_omegas.copy()
        # Default radial drift comes from per-fish r_drifts (set at spawn).
        r_drifts = self.r_drifts.copy()

        # Behavior 2 - dart: long pause + occasional bursts.
        dart_mask = self.behaviors == 2
        if np.any(dart_mask):
            phase = self.dart_phases[dart_mask] + self.time * 2.0
            burst = (np.sin(phase) > 0.55).astype(np.float64)
            speed_factor = 0.25 + 1.75 * burst  # 0.25x normal cruise, 2.0x during burst
            self.omegas[dart_mask] = base_omegas[dart_mask] * speed_factor

        # Behavior 3 - hover: slow, steady drift. The old version added a
        # random wobble much larger than base_omega*0.15, which flipped the
        # sign of omega every frame -- fish facing-direction bounced back
        # and forth without ever translating. Now it's just a constant
        # slow cruise at 40% of base speed.
        hover_mask = self.behaviors == 3
        if np.any(hover_mask):
            self.omegas[hover_mask] = base_omegas[hover_mask] * 0.4

        # Behavior 4 - bottom_dweller: half-speed, no radial drift.
        bottom_mask = self.behaviors == 4
        if np.any(bottom_mask):
            self.omegas[bottom_mask] = base_omegas[bottom_mask] * 0.5
            r_drifts[bottom_mask] = 0.0

        # Behavior 5 - surface: gentle but consistent.
        surface_mask = self.behaviors == 5
        if np.any(surface_mask):
            self.omegas[surface_mask] = base_omegas[surface_mask] * 0.8
            r_drifts[surface_mask] = 0.0

        # ---- Schooling: alignment with same-species neighbors ----
        # Done in physical (x, y) feet so distance thresholds are physical.
        x_phys = self.r_fts * np.cos(self.thetas)
        y_phys = self.r_fts * np.sin(self.thetas)

        for sp_idx, sp_name in enumerate(self.species_names):
            spec = FISH_SPECIES[sp_name]
            if spec.get('school_distance', 0.0) <= 0.0:
                continue
            mask = self.species == sp_idx
            if int(np.sum(mask)) < 2:
                continue

            ids = np.where(mask)[0]
            sd = spec['school_distance']
            ss = spec['school_strength']
            xs, ys = x_phys[ids], y_phys[ids]
            omegas = self.omegas[ids]
            drifts = r_drifts[ids]

            # O(k^2) but k is small (typically <10 of any one species).
            for i in range(len(ids)):
                dx = xs - xs[i]
                dy = ys - ys[i]
                d2 = dx * dx + dy * dy
                neigh = (d2 > 1e-6) & (d2 < sd * sd)
                if not np.any(neigh):
                    continue
                avg_omega = float(np.mean(omegas[neigh]))
                avg_drift = float(np.mean(drifts[neigh]))
                omegas[i] = omegas[i] * (1.0 - ss) + avg_omega * ss
                drifts[i] = drifts[i] * (1.0 - ss) + avg_drift * ss

            self.omegas[ids] = omegas
            r_drifts[ids] = drifts
            # Maintain species speed magnitude (alignment shouldn't slow them).
            self.omegas[ids] = np.sign(self.omegas[ids]) * np.abs(base_omegas[ids])

        # ---- Random course wobble (excluded for hover/dart which already wobble) ----
        wobble_mask = ~((self.behaviors == 3) | (self.behaviors == 2))
        change_prob = dt * 0.25
        rng = np.random.random(self.num_fish)
        do_change = wobble_mask & (rng < change_prob)
        if np.any(do_change):
            n = int(np.sum(do_change))
            self.omegas[do_change] *= 1.0 + np.random.uniform(-0.15, 0.15, n)

        # ---- Integrate ----
        self.thetas += self.omegas * dt
        self.r_fts += r_drifts * dt

        # Reflect radial drift at the spawn band edges (per behavior).
        # Bottom dwellers: clamped to [12, OUTER_R_FT - 0.5].
        # Surface: clamped to [INNER_R_FT, 7].
        # Everything else: clamped to [SPAWN_R_MIN, SPAWN_R_MAX].
        r_lo = np.full(self.num_fish, SPAWN_R_MIN)
        r_hi = np.full(self.num_fish, SPAWN_R_MAX)
        r_lo[bottom_mask] = 12.0
        r_hi[bottom_mask] = OUTER_R_FT - 3.0
        r_lo[surface_mask] = SPAWN_R_MIN
        r_hi[surface_mask] = SPAWN_R_MIN + 3.0

        # Fish that hit a radial boundary are pushed back a bit and have
        # their radial drift ZEROED out (not reversed). Any future drift
        # comes from the random-wobble stage. This prevents the "slam the
        # wall, swing to the other wall, slam, repeat" ping-pong that looks
        # ridiculous on a narrow water column.
        too_low = self.r_fts < r_lo
        too_high = self.r_fts > r_hi
        if np.any(too_low):
            self.r_fts[too_low] = r_lo[too_low] + 0.2
            self.r_drifts[too_low] = 0.0
        if np.any(too_high):
            self.r_fts[too_high] = r_hi[too_high] - 0.2
            self.r_drifts[too_high] = 0.0

        # Recompute buffer-pixel positions and pixel sizes for render().
        self._refresh_buffer_state()

        # Off-arc reset (fish has swum well past theta=0 or theta=pi).
        # Use a generous margin so fish visibly swim OUT of the fan before
        # being respawned on the other side, rather than teleporting the
        # instant their center crosses the boundary.
        margin = 0.25  # radians -- about 14 degrees past the edge
        off_arc = (self.thetas < -margin) | (self.thetas > np.pi + margin)
        if np.any(off_arc):
            self._reset_fish(off_arc)
            # Refresh again because reset moved fish to opposite edges.
            self._refresh_buffer_state()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, state: Dict):
        """Render all fish using instanced rendering."""
        super().render(state)

        if self.num_fish == 0:
            return

        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))

        # Sort back-to-front by depth (z) so blending looks right.
        sort_idx = np.argsort(-self.positions[:, 2])

        # Assemble instance data: 16 floats per fish.
        n_fish = self.num_fish
        reserved = np.zeros(n_fish, dtype=np.float32)
        instance_data = np.column_stack([
            self.positions[sort_idx],          # 3  (x_ft, y_ft, z)           location 1
            self.sizes_ft[sort_idx],           # 1  instance_size_ft          location 2
            self.alphas[sort_idx],             # 1                             location 3
            self.colors[sort_idx],             # 3                             location 4
            reserved,                          # 1  instance_reserved          location 5
            self.rotations[sort_idx],          # 1  instance_rotation          location 6
            self.swim_phases[sort_idx],        # 1
            self.body_lengths[sort_idx],       # 1
            self.body_heights[sort_idx],       # 1
            self.body_tapers[sort_idx],        # 1
            self.tail_spreads[sort_idx],       # 1
            self.patterns[sort_idx],           # 1
        ]).astype(np.float32)

        stride = self.INSTANCE_FLOATS * 4
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        # Grow VBO if we now have more fish than the allocation can hold.
        if self.num_fish > self._instance_capacity:
            new_capacity = max(self.num_fish, self._instance_capacity * 2)
            glBufferData(GL_ARRAY_BUFFER, new_capacity * stride, None, GL_DYNAMIC_DRAW)
            self._instance_capacity = new_capacity
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)

        glDepthMask(GL_FALSE)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 24, self.num_fish)
        glDepthMask(GL_TRUE)

        glBindVertexArray(0)
        glUseProgram(0)
