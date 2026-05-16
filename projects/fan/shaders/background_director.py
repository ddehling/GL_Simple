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


# Storm-intensity falling-edge detection. ``THRESHOLD_ENTER`` is the
# value the intensity must rise above to count as "in a storm" — below
# this, brief weather flickers don't count. ``THRESHOLD_EXIT`` is
# lower (hysteresis) so a noisy signal sitting near the boundary
# doesn't toggle the falling-edge detector every frame.
_THRESHOLD_ENTER = 0.45
_THRESHOLD_EXIT  = 0.25

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
        state['was_in_storm'] = False
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

    # Derive storm intensity. Wind is signed (~-2..2), rain_rate is 0..1
    # already. Sandstorm states use both, so taking max gives the right
    # "how obscured is the sky" reading either way.
    wind = abs(float(outstate.get('wind', 0.0)))
    rain = float(outstate.get('rain_rate', 0.0))
    # Wind ~0..2 in practice; halve so 1.0 reads as "full storm" parity
    # with rain. Capping at 1.0 keeps obscuration sane.
    intensity = max(min(wind * 0.5, 1.0), rain)

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

    # Falling-edge detector with hysteresis on the RAW intensity (not
    # the smoothed obscuration — using the smoothed signal would lag
    # the falling edge into the post-storm period, which is exactly
    # when shaders are visually returning).
    if intensity >= _THRESHOLD_ENTER:
        state['was_in_storm'] = True
    elif state['was_in_storm'] and intensity <= _THRESHOLD_EXIT:
        # Storm just cleared. Check cooldown.
        since_last = elapsed - state['last_change_time']
        if since_last >= state['next_min_interval']:
            outstate['scene_id'] = int(outstate.get('scene_id', 0)) + 1
            state['last_change_time'] = elapsed
            state['next_min_interval'] = _MIN_INTERVAL_S + random.uniform(0, _JITTER_S)
        state['was_in_storm'] = False
