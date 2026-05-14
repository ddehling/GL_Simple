"""Placeholder event map for the Weight of Light project.

Phase 5 only needs enough events for ``--project weight_of_light`` to boot
and render *something* to the preview composite. Three existing
fullscreen-quad effects are reused; each renders to whatever canvas size
WoL is using (Phase 5: 144x35 single canvas, Phase 6+: per-group canvases).

Real WoL effects (per-object pulse, sensor-reactive bloom, etc.) land in
Phase 8 alongside the project's weather sets.
"""
from renderer import effects as fx


EVENT_MAP = {
    # ---- Pattern theme (default) -------------------------------------
    "wol_voronoi_trunk":   (fx.shader_voronoi_sphere, {}, {"group": "trunk"}),
    "wol_wave_leaves":     (fx.shader_wave_terrain,   {}, {"group": "leaves"}),
    "wol_tunnel_ambient":  (fx.shader_tunnel,         {}, {"group": "ambient"}),

    # ---- Gentle theme ------------------------------------------------
    "wol_fog_trunk":       (fx.shader_fractal_fog,    {}, {"group": "trunk"}),
    "wol_spots_leaves":    (fx.shader_pixel_spots,    {}, {"group": "leaves"}),
    "wol_voronoi_ambient": (fx.shader_voronoi_sphere, {}, {"group": "ambient"}),

    # ---- Geometric theme --------------------------------------------
    "wol_isovalues_trunk":   (fx.shader_noise_isovalues, {}, {"group": "trunk"}),
    "wol_tentacle_leaves":   (fx.shader_tentacle,        {}, {"group": "leaves"}),
    "wol_isovalues_ambient": (fx.shader_noise_isovalues, {}, {"group": "ambient"}),

    # ---- Test theme — bouncing blue ball on the leaves canvas only.
    # ``shader_test_bouncing_ball`` lives in
    # projects/weight_of_light/shaders/ and is auto-imported into the
    # ``fx`` namespace by core.shader_loader on project boot/swap.
    "wol_test_bouncing_ball": (
        fx.shader_test_bouncing_ball,
        {"radius": 0.1, "speed_x": 0.25, "speed_y": 0.31,
         "color_rgb": (0.0, 0.3, 1.0)},
        {"group": "leaves"},
    ),

    # ---- Natural set: permanent sky/ground + weather overlays --------
    # All five run continuously as background events; intensity gates
    # (rain_rate / starryness / rainbow_intensity / celestial_visibility)
    # decide what's visible in any given weather state. Smooth
    # crossfades come for free from WeatherStateController's param
    # interpolation. Group ids must match projects/weight_of_light/
    # project.yaml.
    "wol_sky_daynight": (
        fx.shader_wol_sky_daynight,
        {"cycle_seconds": 600.0, "shimmer": 0.06,
         "shimmer_speed": 0.05, "dim_strength": 0.85},
        {"group": "Sky"},
    ),
    "wol_rain": (
        fx.shader_wol_rain,
        {"color_rgb": (0.85, 0.95, 1.0), "drops_per_strip": 3.0,
         "fall_speed": 0.8, "streak_length": 0.10},
        {"group": "Sky"},
    ),
    # Parallel rain instance on the Ground canvas. Same shader; the
    # u_rows uniform lets it autoadapt to the 9-row Ground canvas
    # without a separate fragment program. Slightly fewer drops per
    # strip and shorter streaks read better on the tighter ground
    # arcs.
    "wol_rain_ground": (
        fx.shader_wol_rain,
        {"color_rgb": (0.75, 0.85, 1.0), "drops_per_strip": 2.0,
         "fall_speed": 0.6, "streak_length": 0.08},
        {"group": "Ground"},
    ),
    "wol_stars": (
        fx.shader_wol_stars,
        {"stars_per_strip": 4.0, "twinkle_speed": 0.4,
         "star_radius": 0.012},
        {"group": "Sky"},
    ),
    "wol_ground_twinkle": (
        fx.shader_wol_ground_twinkle,
        {"spawn_chance": 0.12, "cycle_seconds": 2.0,
         "decay_rate": 1.6, "max_brightness": 1.0,
         "echo_chance": 0.30},
        {"group": "Ground"},
    ),
    "wol_rainbow": (
        fx.shader_wol_rainbow,
        {"max_alpha": 0.30, "walk_speed": 0.05,
         "glimmer_speed": 1.5, "glimmer_amp": 0.35},
        {"group": "Ground"},
    ),
    # Presence glow — extra brightness on a box's ground arc the
    # closer its LD2412 radar's strongest gate is, with hue and
    # shimmer driven by the static/motion energy split:
    #   * stationary person → warm settled glow (color_warm)
    #   * walking person    → cool active glow with shimmer (color_cool)
    # Reads ``state['radar']`` via the wrapper; quiet when no one's
    # near any box.
    "wol_presence_glow": (
        fx.shader_wol_presence_glow,
        {"color_warm":       (1.0, 0.55, 0.15),    # saturated amber
         "color_cool":       (0.20, 0.55, 1.0),    # saturated cool blue
         "max_alpha":        1.0,
         "shimmer_freq":     1.6,                  # subtler breathing,
         "shimmer_max_amp":  0.18,                 # not flicker
         "motion_norm":      150.0,
         # Smoothstep pivots remap real-world motion_ratio range
         # (~0.1 standing, ~0.55 walking) onto the full warm→cool
         # lerp so the hue actually flips end-to-end.
         "color_pivot_low":  0.10,
         "color_pivot_high": 0.55},
        {"group": "Ground"},
    ),

    # ---- Lightning (button-driven one-shot, scoped to one object) ----
    # Scheduled by projects.weight_of_light.button_router with
    # ``target_object_id`` baked in via functools.partial so concurrent
    # flashes on different boxes don't trip the scheduler's coarse
    # dedup-by-action.
    "wol_lightning_flash": (
        fx.shader_wol_lightning_flash,
        {"color_rgb": (1.0, 1.0, 1.0),
         "strip_delay": 0.04, "propagation_time": 0.06,
         "decay": 0.4, "dim_alpha": 0.4, "dim_recovery": 2.5},
        {"group": "Sky"},
    ),
}
