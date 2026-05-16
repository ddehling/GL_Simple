"""Background scene director — turns storms into scene transitions.

This is a passive event with no visual output of its own. It watches the
storm-intensity signal (derived from wind / rain_rate) and:

  * Publishes ``outstate['storm_obscuration']`` — a 0..1 value that
    background-silhouette shaders multiply into their alpha so they
    fade out during a storm and back in afterward.
  * Increments ``outstate['scene_id']`` on each storm's falling edge,
    gated by a minimum cooldown so scenes don't churn during rapid
    weather. Each background shader takes ``scene_id % N`` to pick
    among its own internal presets, so different shaders can carry
    different numbers of variations without coordination.

The director is meant to be added to background_events of any weather
set whose backgrounds should rotate. Sets that don't include it keep a
stable scene (scene_id stays at whatever it was last set to). Designed
to be cheap — no rendering, no allocations after init.

See ``docs/shader_contrast_playbook.md`` for why dark silhouettes
against bright sky are a high-value contrast tool, which is the design
this scene-rotation system supports.
"""
import random


# Scene-change triggering thresholds (operate on the SMOOTHED
# obscuration value, not raw intensity, so we change scenes at the
# point in time when backgrounds are most-hidden — not at storm
# start before obscuration has had time to ramp up, and not at storm
# end after silhouettes have re-emerged). Without this the swap was
# visible as a one-frame silhouette change.
#
# Increment scene_id once per storm cycle, when obscuration first
# crosses ``_OBSCURE_TO_SWAP``. Reset the "ready for next swap" flag
# when obscuration falls back below ``_OBSCURE_TO_RESET`` — at that
# point the storm has cleared, the new preset is fully visible, and
# we're armed for the next storm.
_OBSCURE_TO_SWAP  = 0.80
_OBSCURE_TO_RESET = 0.20

# Cooldown bounds (seconds). After incrementing scene_id, the next
# storm's falling edge can't increment again until at least this much
# time has passed. The jitter range prevents the operator from
# predicting when the world will change next.
_MIN_INTERVAL_S = 300.0      # 5 minutes
_JITTER_S       = 600.0      # +0..10 minutes

# Soft scene_id seed range. The first ``outstate['scene_id']`` value is
# picked uniformly from [0, _SEED_RANGE) at director start, so different
# boots start in different scenes. After that it monotonically
# increments — shaders pick presets via modulo their own preset count.
_SEED_RANGE = 256


def shader_background_director(state, outstate, fade_duration=0.0):
    """Director event. Add to a weather set's ``background_events`` to
    enable scene rotation on that set.

    No fade is meaningful here (the director has no visual output), so
    ``fade_duration`` is accepted but unused — kept in the signature so
    the event-map dispatcher can pass it like other background events.
    """
    if state['count'] == -1:
        # No teardown needed — outstate keys persist across event ends
        # so the next director instance picks up where the last left off.
        return

    elapsed = float(state.get('elapsed_time', 0.0))

    if state['count'] == 0:
        # First call: seed scene_id if no one else has, snapshot baseline.
        if 'scene_id' not in outstate:
            outstate['scene_id'] = random.randrange(_SEED_RANGE)
        # True once this storm cycle has triggered a scene change;
        # reset when obscuration falls back below the reset threshold.
        state['swapped_this_cycle'] = False
        # Elapsed time at the *last* successful increment. -inf means
        # "never" — the first storm encountered will increment without
        # being held off by the cooldown.
        state['last_change_time'] = -float('inf')
        state['next_min_interval'] = _MIN_INTERVAL_S + random.uniform(0, _JITTER_S)
        state['_prev_elapsed'] = elapsed
        outstate.setdefault('storm_obscuration', 0.0)
        # First frame: just seed state and return; dt is zero so smoothing
        # would no-op anyway.
        return

    # Derive storm intensity from actual storm signals — NOT wind,
    # which is always non-zero (ambient breeze exists in clear states
    # too) and would leave background silhouettes permanently
    # half-transparent. ``sand_density`` and ``rain_rate`` are 0 in
    # clear weather and ramp up only when a real storm is active.
    sand = float(outstate.get('sand_density', 0.0))
    rain = float(outstate.get('rain_rate', 0.0))
    intensity = max(sand, rain)

    # Smooth obscuration with a one-pole filter so it doesn't twitch on
    # noisy weather signals. Tau ~ 1.5 s — slow enough to feel like
    # atmosphere, fast enough that a 5-minute storm fully obscures.
    prev_elapsed = float(state.get('_prev_elapsed', elapsed))
    dt = max(elapsed - prev_elapsed, 0.0)
    state['_prev_elapsed'] = elapsed
    tau = 1.5
    alpha = 1.0 - pow(2.71828, -dt / tau) if dt > 0 else 0.0
    prev_obscuration = float(outstate.get('storm_obscuration', 0.0))
    obscuration = prev_obscuration + (intensity - prev_obscuration) * alpha
    outstate['storm_obscuration'] = max(0.0, min(1.0, obscuration))

    # Scene-change trigger fires on the SMOOTHED obscuration crossing
    # the high threshold — i.e. when silhouettes are actually obscured.
    # If we triggered on falling-edge instead, the swap would happen
    # AFTER backgrounds had re-emerged, producing a visible one-frame
    # silhouette flip (the bug this replaces). One swap per storm
    # cycle: ``swapped_this_cycle`` clamps until obscuration falls
    # back below the reset threshold.
    obs = outstate['storm_obscuration']
    if obs >= _OBSCURE_TO_SWAP and not state['swapped_this_cycle']:
        since_last = elapsed - state['last_change_time']
        if since_last >= state['next_min_interval']:
            outstate['scene_id'] = int(outstate.get('scene_id', 0)) + 1
            state['last_change_time'] = elapsed
            state['next_min_interval'] = _MIN_INTERVAL_S + random.uniform(0, _JITTER_S)
        # Mark cycle as handled even if we skipped due to cooldown,
        # so we don't try again on every frame this storm.
        state['swapped_this_cycle'] = True
    elif obs <= _OBSCURE_TO_RESET:
        state['swapped_this_cycle'] = False
