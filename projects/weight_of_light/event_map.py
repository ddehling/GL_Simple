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
}
