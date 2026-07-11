"""DEFAULT_EVENT_MAP — events every project automatically inherits.

These are universal building blocks tied to shared infrastructure
(narrative playback, randomized sound pools) rather than project-specific
visuals. New projects shouldn't have to remember to re-register them;
``core.project.Project.load_event_map`` merges this dict in front of the
project's own ``EVENT_MAP`` (project entries override the defaults if
they redeclare a key by name).

Entries here must reference engine-side infrastructure effects only —
``renderer/effects/`` deliberately contains NO visual content (every
content shader lives in a project repo under ``projects/<id>/shaders/``).
Anything that's only meaningful inside one project belongs in that
project's ``event_map.py``, not here.
"""
from renderer import effects as fx


DEFAULT_EVENT_MAP = {
    # Narrative playback. The active weather set declares its own narrative
    # via its ``narrative_script`` field, which is published to
    # ``outstate['narrative_script']`` and read by the event wrapper. Sets
    # without a script leave the effect silent.
    "narrative_player": (fx.shader_narrative_player, {
        "node_delay": 3.0,
        "restart_delay": 10.0,
    }),
    # Randomized ambient audio pool. The active set's ``sound_pool_dir``
    # picks the directory; clips are sampled at random. By default they play
    # with a gap between them; a set can set ``sound_pool_crossfade`` > 0 to
    # play them gaplessly, crossfading each clip into the next.
    "sound_pool": (fx.shader_sound_pool, {}),
}
