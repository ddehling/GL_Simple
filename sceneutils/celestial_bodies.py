from sceneutils.environmental import CelestialBody

# Configuration for all celestial bodies in the system
CELESTIAL_BODIES = [
            CelestialBody(  # noqa: F405
                size=8,
                roughness=0.3,
                orbital_speed=1,
                color_h=0.15,  # Yellowish
                color_s=0.25,
                color_v=0.6,
                tilt=-15,  # Vertical orbital plane
                shift=5,  # Rise in the east
                glow_factor=0.4,
                corona_size=2.0,
                name="moon",
                distance=1,
            ),
            CelestialBody(  # noqa: F405
                size=4.5,
                roughness=0.2,
                orbital_speed=0.7,
                color_h=0.0,  # Red
                color_s=0.9,
                color_v=0.7,
                tilt=30,  # 30° from vertical
                shift=5,  # Rise slightly north of east
                glow_factor=0.3,
                corona_size=1.5,
                name="red_planet",
                distance=1.5,
            ),
            CelestialBody(  # noqa: F405
                size=16,
                roughness=0.3,
                orbital_speed=0.30,
                color_h=0.65,  # Blueish
                color_s=0.9,
                color_v=0.7,
                tilt=35,  # 45° from vertical
                shift=-3,  # Rise slightly south of east
                glow_factor=1.3,
                corona_size=2.3,
                name="blue_planet",
                distance=5,
            ),
            CelestialBody(  # noqa: F405
                size=20,
                roughness=0.6,
                orbital_speed=-2,
                color_h=0.25,  # Blueish
                color_s=0.9,
                color_v=0.6,
                tilt=-40,  # 45° from vertical
                shift=3,  # Rise slightly south of east
                glow_factor=1.3,
                corona_size=1.3,
                name="yellow_planet",
                distance=6,
            ),
            CelestialBody(  # noqa: F405
                size=2.5,
                roughness=0.3,
                orbital_speed=-9,
                color_h=0.65,  # Blueish
                color_s=0.2,
                color_v=0.7,
                tilt=20,  # 45° from vertical
                shift=6,  # Rise slightly south of east
                glow_factor=0.5,
                corona_size=1.75,
                name="asteroid",
                distance=1.2,
            ),
            CelestialBody(  # noqa: F405
                size=6,
                roughness=0.5,
                orbital_speed=-0.50,
                color_h=0.35,  # Blueish
                color_s=1,
                color_v=0.5,
                tilt=-5,  # 45° from vertical
                shift=6,  # Rise slightly south of east
                glow_factor=1.3,
                corona_size=2.3,
                name="green_planet",
                distance=4,
            ),
            CelestialBody(  # noqa: F405
                size=12,
                roughness=0.25,
                orbital_speed=2.60,
                color_h=0.35,  # Blueish
                color_s=1,
                color_v=0.0,
                tilt=-10,  # 45° from vertical
                shift=-6,  # Rise slightly south of east
                glow_factor=1.3,
                corona_size=2.3,
                name="ghost_planet",
                distance=.4,
            )
        ]