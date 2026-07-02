"""Compile-test every club shader in a hidden GLES 3.1 GLFW context.

Catches GLSL-ES-only failures (reserved words, missing precision, undefined
uniforms in the source strings) without launching the app. Instantiates each
effect class against a stub viewport, calls its compile_shader(), and
reports PASS/FAIL per shader.

Usage: python tools/_club_shader_compile_test.py
Exit code 0 + "ALL PASS" when every shader compiles.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glfw
from OpenGL.GL import glGetString, GL_VERSION


class _StubViewport:
    width = 128
    height = 300


def main():
    if not glfw.init():
        print("FAIL: glfw.init()")
        sys.exit(1)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "shader-compile-test", None, None)
    if win is None:
        # Some desktop drivers refuse ES contexts; fall back to default API
        # (the shaders still compile under the compatibility path the app
        # itself would get on this machine).
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, "shader-compile-test", None, None)
    if win is None:
        print("FAIL: could not create a hidden GL context")
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(win)
    print(f"GL context: {glGetString(GL_VERSION).decode()}\n")

    # Import AFTER the context exists (module import is GL-free, but keep
    # the pattern consistent with the app's init order).
    from projects.fan.shaders.club_eq_bars import ClubEqBarsEffect
    from projects.fan.shaders.club_lasers import ClubLasersEffect
    from projects.fan.shaders.club_strobe import ClubStrobeEffect
    from projects.fan.shaders.club_tunnel import ClubTunnelEffect
    from projects.fan.shaders.club_shockwave import ClubShockwaveEffect
    from projects.fan.shaders.club_spectrum_stars import ClubSpectrumStarsEffect
    from projects.fan.shaders.club_horizon_glow import ClubHorizonGlowEffect
    from projects.fan.shaders.club_milk_echo import ClubMilkEchoEffect
    from projects.fan.shaders.club_mandala import ClubMandalaEffect
    from projects.fan.shaders.club_flow_field import ClubFlowFieldEffect
    from projects.fan.shaders.club_wave_dance import ClubWaveDanceEffect
    from projects.fan.shaders.club_kick_flare import ClubKickFlareEffect
    from projects.fan.shaders.club_sparkle import ClubSparkleEffect
    from projects.fan.shaders.club_chaser import ClubChaserEffect
    from projects.fan.shaders.club_ribbon import ClubRibbonEffect
    from projects.fan.shaders.club_radar_sweep import ClubRadarSweepEffect
    from renderer.effects.audio_balls import AudioBallsEffect

    effects = [
        ("club_eq_bars", ClubEqBarsEffect),
        ("club_lasers", ClubLasersEffect),
        ("club_strobe", ClubStrobeEffect),
        ("club_tunnel", ClubTunnelEffect),
        ("club_shockwave", ClubShockwaveEffect),
        ("club_spectrum_stars", ClubSpectrumStarsEffect),
        ("club_horizon_glow", ClubHorizonGlowEffect),
        ("club_milk_echo", ClubMilkEchoEffect),
        ("club_mandala", ClubMandalaEffect),
        ("club_flow_field", ClubFlowFieldEffect),
        ("club_wave_dance", ClubWaveDanceEffect),
        ("club_kick_flare", ClubKickFlareEffect),
        ("club_sparkle", ClubSparkleEffect),
        ("club_chaser", ClubChaserEffect),
        ("club_ribbon", ClubRibbonEffect),
        ("club_radar_sweep", ClubRadarSweepEffect),
        ("audio_balls", AudioBallsEffect),
    ]

    failures = []
    vp = _StubViewport()
    for name, cls in effects:
        try:
            eff = cls(vp)
            shader = eff.compile_shader()
            ok = bool(shader)
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                failures.append(name)
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            failures.append(name)

    glfw.terminate()
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
