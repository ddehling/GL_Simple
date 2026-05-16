from renderer import effects as fx


EVENT_MAP = {
    "clouds": (fx.shader_drifting_clouds, {}),
    "firefly": (fx.shader_firefly, {}),
    "stars": (fx.shader_stars, {"num_stars": 2000, "audio_sensitivity": 0, "drift_x": 0.3}),
    "rain": (fx.shader_rain, {}),
    "fog": (fx.shader_fog, {
        "strength": 0.0,
        "color": (0.7, 0.7, 0.8),
        "fog_near": 0.0,
        "fog_far": 30.0
    }),
    "sandstorm": (fx.shader_sandstorm, {}),
    "desert_dunes": (fx.shader_desert_dunes, {}),
    "desert_sky": (fx.shader_desert_sky, {}),
    "desert_creatures": (fx.shader_desert_creatures, {}),
    "desert_rain": (fx.shader_desert_rain, {}),
    "fog_beings": (fx.shader_chromatic_fog_beings, {}),
    "falling_leaves": (fx.shader_falling_leaves, {}),
    "audio_balls": (fx.shader_audio_balls, {}),
    "audio_curve": (fx.shader_audio_curve, {}),
    "sunrise": (fx.shader_sunrise, {}),
    "game_of_life": (fx.shader_gameoflife, {}),
    "fractal_fog": (fx.shader_fractal_fog, {}),
    "noise_isovalues": (fx.shader_noise_isovalues, {}),
    "tentacle": (fx.shader_tentacle, {}),
    "tunnel_raymarch": (fx.shader_tunnel_raymarch, {}),
    "tunnel": (fx.shader_tunnel, {}),
    "voronoi_sphere": (fx.shader_voronoi_sphere, {}),
    "wave_terrain": (fx.shader_wave_terrain, {}),
    "wave_equation": (fx.shader_wave_equation, {}),
    "audio_scan_line": (fx.shader_audio_scan_line, {
        "scan_speed": 50.0,
        "trail_length": 75,
        "intensity_sensitivity": 2.0,
        "width_sensitivity": 0.5,
        "base_width": 2.0,
        "max_width": 20.0,
        "color_hue": 0.5
    }),
    "pixel_spots": (fx.shader_pixel_spots, {}),
    "vortex": (fx.shader_vortex, {}),
    "hurricane": (fx.shader_hurricane, {}),
    "lightning": (fx.shader_lightning, {}),
    "ocean_waves": (fx.shader_ocean_waves, {}),
    "kelp": (fx.shader_kelp, {}),
    "coral": (fx.shader_coral, {}),
    "tube_worms": (fx.shader_tube_worms, {}),
    "Bioluminescence": (fx.shader_bioluminescence, {}),
    "bubbles": (fx.shader_bubbles, {}),
    "fish": (fx.shader_fish, {}),
    "smoker": (fx.shader_smoker, {}),
    "test_pattern": (fx.shader_test_pattern, {"orientation": "vertical"}),
    "bart_map": (fx.shader_bart_map, {}),
    "highway_traffic": (fx.shader_highway_traffic, {}),
    "test_fan_coords": (fx.shader_test_fan_coords, {}),
    "city_lights": (fx.shader_city_lights, {}),
    "bay_shimmer": (fx.shader_bay_shimmer, {}),
    "pride_flag": (fx.shader_pride_flag, {}),
    "heart_pulse": (fx.shader_heart_pulse, {}),
    "thread_bonds": (fx.shader_thread_bonds, {}),
    "warm_bloom": (fx.shader_warm_bloom, {}),
    "distant_lights": (fx.shader_distant_lights, {}),
    # Depth 88 (gl_Position.z = 0.88) so aurora sits just in front
    # of canopy_godrays' sky backdrop at 0.95 — otherwise the sky
    # depth-writes occlude the aurora.
    "aurora": (fx.shader_aurora, {"depth": 88.0}),
    "canopy_godrays": (fx.shader_canopy_godrays, {}),
    "forest_canopy": (fx.shader_forest_canopy, {}),
    "dappled_shadows": (fx.shader_dappled_shadows, {}),
    "snowfall": (fx.shader_snowfall, {}),
    "rain_on_leaves": (fx.shader_rain_on_leaves, {}),
    "spore_drift": (fx.shader_spore_drift, {}),
    "stream_flow": (fx.shader_stream_flow, {}),
    "forest_eyes": (fx.shader_forest_eyes, {}),
    # ``narrative_player`` and ``sound_pool`` used to be declared here
    # but are now auto-inherited from ``core.default_events.DEFAULT_EVENT_MAP``.
    # Redeclare a key here if Fan ever needs different params for one
    # of them — project entries override the defaults.
}
