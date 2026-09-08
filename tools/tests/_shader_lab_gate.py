"""Shader Lab gate — validation/extraction unit checks (no GL, no API).

Run directly:  python tools/tests/_shader_lab_gate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.shader_lab import (  # noqa: E402
    STDLIB_NAMES, apply_edits, extract_edits, extract_glsl, extract_knobs,
    validate_glsl)

GOOD = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(0.);}"

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(('PASS  ' if cond else 'FAIL  ') + name)


def fence(body, tag='glsl'):
    return f"```{tag}\n{body}\n```"


def main():
    # -- extract_glsl --
    check('extract: single tagged block',
          extract_glsl('desc\n' + fence(GOOD)) == GOOD)
    check('extract: no block -> None', extract_glsl('no code here') is None)
    check('extract: untagged block accepted',
          extract_glsl(fence(GOOD, tag='')) == GOOD)
    two = fence('float helper;') + '\n' + fence(GOOD)
    check('extract: prefers the mainImage block among several',
          extract_glsl(two) == GOOD)
    ambiguous = fence(GOOD) + '\n' + fence(GOOD)
    check('extract: two mainImage blocks -> None (ambiguous)',
          extract_glsl(ambiguous) is None)

    # -- validate_glsl --
    check('validate: good source passes', validate_glsl(GOOD)[0])
    check('validate: empty rejected', not validate_glsl('')[0])
    check('validate: missing mainImage rejected',
          not validate_glsl('void main(){}')[0])
    check('validate: while banned',
          not validate_glsl(GOOD + ' /* while */'.replace('/* while */', '')
                            + '\nvoid h(){while(true){}}')[0])
    check('validate: do banned',
          not validate_glsl(GOOD + '\nvoid h(){do{}while(false);}')[0])
    check('validate: uniform decl banned',
          not validate_glsl('uniform float x;\n' + GOOD)[0])
    check('validate: #version banned',
          not validate_glsl('#version 310 es\n' + GOOD)[0])
    check('validate: texture banned',
          not validate_glsl(GOOD.replace('vec4(0.)',
                                         'texture(s, fragCoord)'))[0])
    check('validate: #define allowed',
          validate_glsl('#define T (iTime*0.5)\n' + GOOD)[0])
    check('validate: length cap',
          not validate_glsl(GOOD + '\n' + '// x\n' * 20000)[0])
    check('validate: "dot" not caught by do-ban (word boundary)',
          validate_glsl(GOOD.replace('vec4(0.)',
                                     'vec4(dot(fragCoord, fragCoord))'))[0])

    # Banned words in COMMENTS must not reject real code (they did — a
    # '// textured look' comment killed a valid shader).
    check('validate: "texture" in line comment allowed',
          validate_glsl('// give it a textured look\n' + GOOD)[0])
    check('validate: "do"/"while" in comments allowed',
          validate_glsl('/* do a slow spin while it glows */\n' + GOOD)[0])
    check('validate: texture() in code still banned',
          not validate_glsl('// harmless\n' + GOOD.replace(
              'vec4(0.)', 'texture(s, fragCoord)'))[0])
    check('validate: mainImage only in a comment rejected',
          not validate_glsl('// void mainImage(out vec4 a, in vec2 b)\n'
                            'void main(){}')[0])

    # -- stdlib collision guard --
    check('validate: redefining a stdlib fn rejected',
          not validate_glsl('float hash11(float n) { return n; }\n'
                            + GOOD)[0])
    check('validate: redefining fanUV rejected',
          not validate_glsl('vec2 fanUV(vec2 fc) { return fc; }\n'
                            + GOOD)[0])
    check('validate: CALLING stdlib fns is fine',
          validate_glsl(GOOD.replace(
              'vec4(0.)', 'vec4(vnoise(fragCoord), fbm(fragCoord), '
              'hash11(1.0), 1.0)'))[0])

    # -- edit blocks (diff-based refinements) --
    def edit_block(search, replace):
        return ('```edit\n<<<<<<< SEARCH\n' + search + '\n=======\n'
                + replace + '\n>>>>>>> REPLACE\n```')

    pairs = extract_edits('tweak\n' + edit_block('a', 'b')
                          + '\n' + edit_block('c\nd', 'e'))
    check('edits: parses multiple blocks',
          pairs == [('a', 'b'), ('c\nd', 'e')])
    check('edits: none in plain text', extract_edits('no edits here') == [])
    merged, why = apply_edits('one two three', [('two', 'TWO')])
    check('edits: apply replaces uniquely', merged == 'one TWO three')
    merged, why = apply_edits('aa bb aa', [('aa', 'x')])
    check('edits: ambiguous SEARCH refused',
          merged is None and 'times' in why)
    merged, why = apply_edits('abc', [('zzz', 'x')])
    check('edits: missing SEARCH refused',
          merged is None and 'not appear' in why)

    # -- knob declarations --
    check('knobs: none declared -> []', extract_knobs(GOOD) == [])
    check('knobs: two names parsed',
          extract_knobs('// KNOBS: speed | sparkle amount\n' + GOOD)
          == ['speed', 'sparkle amount'])
    check('knobs: capped at four',
          len(extract_knobs('// KNOBS: a|b|c|d|e|f\n' + GOOD)) == 4)
    check('knobs: declaration survives validation',
          validate_glsl('// KNOBS: speed\n' + GOOD)[0])

    # -- reject-repair loop (scripted fake transport, no network) --
    import web.shader_lab as SL
    from web.shader_lab import ShaderLabSession

    def run_scripted(replies, session=None):
        """generate() against a fake CLI that pops canned replies.
        Returns (result_tuple, transcripts_sent, model_ids_used)."""
        calls, models = [], []
        real = (SL._resolve_transport, SL._call_claude_cli,
                SL.MIN_CALL_INTERVAL_S)
        SL._resolve_transport = lambda refresh=False: ('cli', 'fake-exe')

        def fake_cli(exe, system, transcript, model_id):
            calls.append(transcript)
            models.append(model_id)
            return replies[len(calls) - 1]

        SL._call_claude_cli = fake_cli
        SL.MIN_CALL_INTERVAL_S = 0.0
        try:
            sess = session or ShaderLabSession()
            return sess.generate('test pattern'), calls, models
        finally:
            (SL._resolve_transport, SL._call_claude_cli,
             SL.MIN_CALL_INTERVAL_S) = real

    bad = 'a textured ball\n' + fence(
        GOOD.replace('vec4(0.)', 'texture(s, fragCoord)'))
    good_reply = 'a plain ball\n' + fence(GOOD)

    (glsl, desc, err), calls, models = run_scripted([good_reply])
    check('repair: clean first reply needs one call',
          err == '' and glsl == GOOD and len(calls) == 1)
    check('model: defaults to opus', models[0] == 'claude-opus-5')

    (glsl, desc, err), calls, _ = run_scripted([bad, good_reply])
    check('repair: rejected code is re-asked and recovers',
          err == '' and glsl == GOOD and len(calls) == 2)
    check('repair: re-ask tells the model why',
          'rejected before compiling' in calls[1])

    (glsl, desc, err), calls, _ = run_scripted(['no code at all', good_reply])
    check('repair: missing code block is re-asked and recovers',
          err == '' and glsl == GOOD and len(calls) == 2)

    (glsl, desc, err), calls, _ = run_scripted([bad, bad, bad, bad])
    check('repair: gives up after 3 attempts with a clear error',
          glsl is None and len(calls) == 3 and 'rejected' in err)

    # -- model toggle --
    sess = ShaderLabSession()
    sess.set_model('sonnet')
    _, _, models = run_scripted([good_reply], session=sess)
    check('model: sonnet routes to claude-sonnet-5',
          models[0] == 'claude-sonnet-5')
    sess.set_model('nonsense')
    check('model: unknown key ignored', sess.model == 'sonnet')

    # -- seeding (library load / hand edit becomes the current pattern) --
    sess = ShaderLabSession()
    check('seed: accepted when idle',
          sess.seed(GOOD, 'myPat', 'orig prompt'))
    (glsl, desc, err), calls, _ = run_scripted([good_reply], session=sess)
    check('seed: follow-up turn sees the seeded pattern',
          err == '' and 'myPat' in calls[0] and GOOD in calls[0]
          and 'orig prompt' in calls[0])

    # -- diff refinements through the scripted transport --
    BRIGHT = GOOD.replace('vec4(0.)', 'vec4(1.)')
    edit_reply = 'Brighter.\n' + edit_block('fragColor=vec4(0.);',
                                            'fragColor=vec4(1.);')
    sess = ShaderLabSession()
    sess.seed(GOOD, 'base')
    (glsl, desc, err), calls, _ = run_scripted([edit_reply], session=sess)
    check('edits: refinement reply merges against current code',
          err == '' and glsl == BRIGHT)
    check('edits: history canonicalized to the merged shader',
          BRIGHT in sess._history[-1]['content']
          and 'SEARCH' not in sess._history[-1]['content'])

    bad_edit_reply = 'Tweak.\n' + edit_block('NOT IN THE CODE', 'x')
    sess = ShaderLabSession()
    sess.seed(GOOD, 'base')
    (glsl, desc, err), calls, _ = run_scripted(
        [bad_edit_reply, good_reply], session=sess)
    check('edits: failed apply re-asks for the full shader and recovers',
          err == '' and glsl == GOOD and len(calls) == 2
          and 'could not be applied' in calls[1])

    sess = ShaderLabSession()   # no current code: edit reply is unusable
    (glsl, desc, err), calls, _ = run_scripted(
        [edit_reply, good_reply], session=sess)
    check('edits: edit reply without current code re-asks for full shader',
          err == '' and glsl == GOOD and len(calls) == 2)

    # -- live-shader wrapper header (skipped if GL deps not importable) --
    try:
        from renderer.effects import live_shader as LS
        from renderer.fan_geometry import FanGeometry
    except ImportError as e:
        print(f'SKIP  header checks (import failed: {e})')
    else:
        hdr = LS.FRAGMENT_HEADER
        for fn in STDLIB_NAMES:     # validator ban-list <-> header in sync
            check(f'header: defines {fn}()', fn + '(' in hdr)
        check('header: ends with #line 1 (user error line numbers)',
              hdr.rstrip().endswith('#line 1'))
        check('header: radii match FanGeometry physical constants',
              f'FAN_R_INNER = {FanGeometry.PHYSICAL_INNER_FT:.1f}' in hdr
              and f'FAN_R_OUTER = {FanGeometry.PHYSICAL_OUTER_FT:.1f}' in hdr)
        from web.shader_lab import NUM_SLOTS as WEB_SLOTS
        check('slots: web NUM_SLOTS matches engine NUM_SLOTS',
              WEB_SLOTS == LS.NUM_SLOTS)

    # -- parameterized pattern bank --
    from web import shader_patterns as PB
    for pid in PB.PATTERNS:
        glsl_t, _ = PB.instantiate(pid)
        okp, whyp = validate_glsl(glsl_t)
        check(f'bank: {pid} instantiates + validates'
              + ('' if okp else f' ({whyp})'),
              okp and extract_knobs(glsl_t)
              == ['speed', 'brightness', 'color shift'])
    g, _ = PB.instantiate('waves', {'speed': 999, 'audio': 'nonsense',
                                    'hue': -5, 'palette': 'bogus'})
    check('bank: params clamp, bad choices fall back',
          validate_glsl(g)[0] and '3.0000' in g and ') * 1.0' in g)
    for pal in PB.PALETTE_NAMES:        # every palette compiles-by-validator
        gp, _ = PB.instantiate('plasma', {'palette': pal})
        check(f'bank: palette {pal} validates', validate_glsl(gp)[0])
    m_duo = PB.instant_match('blue and purple waves')
    check('bank: two colors become a duotone palette',
          m_duo is not None and 'mix(hsv2rgb' in m_duo[1]
          and '0.6200' in m_duo[1] and '0.7600' in m_duo[1])
    m_rb = PB.instant_match('rainbow sparkles')
    check('bank: palette words select named palettes',
          m_rb is not None and ', 0.8, 1.0))' in m_rb[1])
    m = PB.instant_match('slow blue waves')
    check('bank: clear request matches instantly',
          m is not None and m[0] == 'waves'
          and '0.6200' in m[1] and '0.2700' in m[1])
    check('bank: unknown subject falls through to codegen',
          PB.instant_match('a sea turtle swimming past the reef') is None)
    m3 = PB.instant_match('green sparkles that pulse with the bass')
    check('bank: audio words map to reactivity (bass wins over beat)',
          m3 is not None and m3[0] == 'sparkles' and 'iBass' in m3[1])
    check('bank: ambiguous two-template request declined',
          PB.instant_match('waves of rain') is None)

    # -- wish routing (slot-free stage management) --
    from web.shader_lab import route_wish
    occ = {0: {'name': 'waves', 'desc': 'Layered swells',
               'prompt': 'slow blue waves', 'since': 100}}
    check('wish: refine matches an active layer by name',
          route_wish('make the waves slower', occ) == ('refine', 0, None))
    check('wish: generic tweak follows the last touched layer',
          route_wish('a bit slower please', occ, last_slot=0)
          == ('refine', 0, None))
    check('wish: new subject takes an empty slot',
          route_wish('roaring fire', occ)[:2] == ('new', 1))
    check('wish: additive word forces a new layer',
          route_wish('add another waves layer', occ)[0] == 'new')
    check('wish: remove targets the named layer',
          route_wish('remove the waves', occ) == ('remove', 0, None))
    check('wish: clear everything',
          route_wish('clear everything', occ)[0] == 'clear_all')
    full = {s: {'name': f'p{s}', 'desc': '', 'prompt': '', 'since': s}
            for s in range(16)}
    act, s, note = route_wish('roaring fire', full)
    check('wish: full stage replaces the oldest layer',
          act == 'new' and s == 0 and note == 'p0')

    # -- blank stage (built-in empty weather set) --
    try:
        from lib.weather_set import BLANK_SET_NAME, WeatherSetManager
    except ImportError as e:
        print(f'SKIP  blank-stage checks (import failed: {e})')
    else:
        mgr = WeatherSetManager({})
        check('blank stage: set injected into weather sets',
              BLANK_SET_NAME in mgr.weather_sets)
        mgr.commit_set_change(BLANK_SET_NAME)
        check('blank stage: only the live_shader slots are scheduled',
              mgr.get_background_events() == ['live_shader'])
        check('blank stage: no random events',
              mgr.get_random_events_config()[0] == [])
        check('blank stage: state casts through the enum',
              len(mgr.get_set_states()) == 1)
        check('blank stage: suppresses per-state extras (events + ambient)',
              mgr.get_suppress_state_extras())
        other = next((n for n in mgr.weather_sets if n != BLANK_SET_NAME),
                     None)
        if other is None:   # engine repo ships no content sets of its own
            mgr.weather_sets['_gate_probe'] = {'states': []}
            other = '_gate_probe'
        mgr.commit_set_change(other)
        check('blank stage: normal sets keep their state extras',
              not mgr.get_suppress_state_extras())

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
