"""Weight of Light — OSC button router.

Wired up by ``hooks: { button_router: projects.weight_of_light.button_router }``
in project.yaml; ``register(env_system)`` is called once at engine boot
(and again after a project hot-swap) by Stories_OGL._call_button_router_hook.

Incoming address shape (from each WoL box's onboard controller)::

    /wol/<host>/button/<idx>   <args...>

Where ``<host>`` is the box's mDNS hostname stem (e.g. ``wol-central``,
``wol-north``) — the same hostname the box advertises for DDP, minus
``.local``. ``<idx>`` is an integer button index local to that box.
``<args>`` is whatever payload the box sends (e.g. press state, analog
depth, counter); we pass it through to the per-button handler verbatim
for now — value semantics get nailed down later.

Closed-loop feedback: handlers receive the env_system, so they can call
``state['osc_send'](object_id, '/wol/feedback/...', value)`` to send
back to the same box (or any box, by id or display name).

To add behavior for a button, append to ``_BUTTON_HANDLERS`` below.
The default catch-all just logs.
"""
from __future__ import annotations

import functools
import random
from typing import Any, Callable, Dict


# Type: (env_system, object_id, button_idx, *args) -> None
ButtonHandler = Callable[..., None]


# Lightning event tuning. Kept at module scope so the operator can
# tweak without touching the handler logic. The duration covers the
# full effect: bright flash (~0.5 s decay tail) plus the sky dim
# recovery (2.5 s by default). Set generously so the shader cleans
# up after the dim has fully faded.
_LIGHTNING_EVENT = "wol_lightning_flash"
_LIGHTNING_DURATION_S = 3.5
# Groups the lightning fires on, and whether each group dims its
# non-target pixels. Sky gets the cinematic "flash blindness" dim;
# Ground doesn't (a closed arc going 40 % black around the lit
# target reads as a bug, not drama).
_LIGHTNING_GROUPS = (
    ("Sky", True),
    ("Ground", False),
)

# Multi-strike behaviour: after the primary strike, roll successive
# geometric "extra strike" probabilities so a typical press resolves
# to 1–3 visible bolts with the occasional 4-bolt cluster. Each
# follow-up has a chance of jumping to a different (usually adjacent)
# box. Numbers tuned by feel; tweak in place rather than building a
# config surface for them.
_FOLLOWUP_PROBS = (0.55, 0.35, 0.20)   # P(2nd | 1st), P(3rd | 2nd), P(4th | 3rd)
# Delay after the *primary* strike before the 2nd. Pulled longer
# than later follow-ups so the second bolt feels like a separate
# beat rather than a tight burst with the first — real lightning
# clusters often have a brief gap, then a flurry.
_SECOND_STRIKE_MIN_DELAY_S = 0.60
_SECOND_STRIKE_MAX_DELAY_S = 2.00
# Delay between any subsequent follow-ups (3rd onwards). Tighter
# range so later strikes register as the rumble of a multi-bolt
# storm cluster.
_FOLLOWUP_MIN_DELAY_S = 0.20
_FOLLOWUP_MAX_DELAY_S = 1.20
_FOLLOWUP_OTHER_OBJECT_PROB = 0.20     # P(follow-up jumps to a different box)


# ---------------------------------------------------------------------------
# Hook entry — Stories_OGL calls this once at boot (and again on swap)
# ---------------------------------------------------------------------------

def register(env_system) -> None:
    """Register the WoL OSC route on env_system.osc_listener.

    Builds a host-stem -> object_id map from this project's receivers
    (independent of mDNS resolution — the map is for incoming routing
    only, sender-side resolution happens lazily inside ProjectOscSender).
    """
    listener = getattr(env_system, "osc_listener", None)
    if listener is None:
        print("[wol_button_router] OSC listener not running; buttons inert")
        return

    receivers = (env_system.project.raw or {}).get("receivers") or []
    host_to_object_id: Dict[str, int] = {}
    for r in receivers:
        if not isinstance(r, dict):
            continue
        try:
            oid = int(r.get("object_id", -1))
        except (TypeError, ValueError):
            continue
        if oid < 0:
            continue
        host = r.get("host") or ""
        stem = _host_stem(host)
        if stem:
            host_to_object_id[stem] = oid

    # Closure captures the map + env_system so the handler stays
    # one-arg-from-listener-side ``handler(address, *args)``.
    def _on_wol(address: str, *args: Any) -> None:
        _dispatch(env_system, host_to_object_id, address, *args)

    listener.register_prefix("/wol/", _on_wol)
    print(f"[wol_button_router] registered /wol/* route; "
          f"{len(host_to_object_id)} hosts mapped: "
          f"{sorted(host_to_object_id.keys())}")


# ---------------------------------------------------------------------------
# Dispatch — parse address, look up device, invoke per-button handler
# ---------------------------------------------------------------------------

def _dispatch(env_system, host_map: Dict[str, int],
              address: str, *args: Any) -> None:
    """Parse ``/wol/<host>/<kind>[/...]``. ``kind=button`` dispatches to
    the per-button-index handler table; ``kind=radar`` delegates to
    ``radar.handle_osc``; anything else is silently ignored.
    """
    parts = address.strip("/").split("/")
    # Need at least /wol/<host>/<kind> (3 parts). Some kinds carry a
    # trailing segment (button index, radar sub-key); each branch
    # checks length as needed.
    if len(parts) < 3:
        return

    host = parts[1]
    kind = parts[2]

    object_id = host_map.get(host, -1)
    if object_id < 0:
        # Unknown device; log once-ish (real dedup belongs in the
        # listener if needed — for now this is rare enough that one
        # warning per stray message is OK).
        print(f"[wol_router] unknown host {host!r} on {address}; "
              f"args={args}")
        return

    if kind == "button":
        if len(parts) < 4:
            return
        try:
            button_idx = int(parts[3])
        except ValueError:
            return
        handler = _BUTTON_HANDLERS.get(button_idx, _default_button_handler)
        try:
            handler(env_system, object_id, button_idx, *args)
        except Exception as e:
            print(f"[wol_button] handler for button {button_idx} "
                  f"on object {object_id} raised: {e}")
        return

    if kind == "radar":
        # Lazy-import so a project that doesn't ship radar.py still
        # boots (we just won't process radar messages).
        try:
            from projects.weight_of_light import radar
        except Exception as e:
            print(f"[wol_radar] module load failed: {e}")
            return
        try:
            radar.handle_osc(env_system, object_id, parts[3:], *args)
        except Exception as e:
            print(f"[wol_radar] handle_osc on object {object_id} "
                  f"raised: {e}  (address={address})")
        return

    # Unknown kind — silently ignore. Add cases above as new kinds
    # come online.


# ---------------------------------------------------------------------------
# Per-button handlers — operator extends this dict to add behaviour
# ---------------------------------------------------------------------------

def _default_button_handler(env_system, object_id: int,
                            button_idx: int, *args: Any) -> None:
    """Catch-all: log the press and echo a feedback OSC back to the
    originating box so the operator can confirm the closed loop is
    working end-to-end. Replace per-button entries below as concrete
    behaviours land.
    """
    state = env_system.scheduler.state
    name = state.get("object_names", {}).get(object_id, f"id={object_id}")
    print(f"[wol_button] {name!r} button {button_idx}  args={args}")

    # Echo back to the same box: ``/wol/feedback/button/<idx> <args...>``.
    # Demonstrates that ``state['osc_send']`` is wired and the host
    # resolves end-to-end. Real handlers will probably send richer
    # feedback (LED brightness, status, etc.).
    osc_send = state.get("osc_send")
    if osc_send is not None:
        osc_send(object_id, f"/wol/feedback/button/{button_idx}", *args)


def _schedule_one_strike(env_system, target_object_id: int,
                         delay: float) -> list[str]:
    """Schedule one lightning strike on ``target_object_id``: one
    event per group in ``_LIGHTNING_GROUPS``, all sharing the same
    ``delay``. Sky's pass plays a thunder sample (Ground's doesn't,
    so a single visible bolt produces one boom rather than two).
    Returns the list of group names that successfully scheduled."""
    entry = env_system.weather_set.resolve_event(_LIGHTNING_EVENT)
    if entry is None:
        print(f"[wol_lightning] event {_LIGHTNING_EVENT!r} not in event_map; "
              f"strike skipped")
        return []
    effect_func, params, _ = entry
    g2f = env_system.project.group_to_frame_id()

    fired = []
    for group_name, dim_enable in _LIGHTNING_GROUPS:
        if group_name not in g2f:
            continue
        # Sky carries the thunder; Ground stays silent so the visible
        # multi-group strike registers as one boom.
        play_sound = (group_name == "Sky")
        action = functools.partial(
            effect_func,
            target_object_id=target_object_id,
            dim_enable=dim_enable,
            play_sound=play_sound,
            **params,
        )
        ev = env_system.scheduler.schedule_event(
            float(delay),
            _LIGHTNING_DURATION_S,
            action,
            frame_id=g2f[group_name],
            name=f"lightning_obj{target_object_id}_{group_name}_d{delay:.2f}",
        )
        if ev is not None:
            fired.append(group_name)
    return fired


def _pick_followup_target(originator: int, all_object_ids: list[int]) -> int:
    """Decide who the next strike hits. Most of the time it stays on
    the originating box; a small fraction of the time it jumps to a
    different box. Without an explicit adjacency map we just pick a
    uniform-random other object — adjacency-aware routing can layer
    in later if the visual benefit justifies it."""
    others = [oid for oid in all_object_ids if oid != originator]
    if others and random.random() < _FOLLOWUP_OTHER_OBJECT_PROB:
        return random.choice(others)
    return originator


def fire_lightning_chain(env_system, object_id: int,
                         tag: str = "lightning") -> int:
    """Schedule a complete lightning chain on ``object_id``: primary
    strike + a geometric tail of 0–3 follow-ups (mostly on the same
    box, occasionally jumping). Returns the total number of strikes
    that scheduled successfully.

    Used by both the OSC button-3 handler and WoL's per-frame
    random_events hook so any caller that wants "fire a dramatic
    lightning event on this box" gets the same behaviour without
    duplicating the chain math. ``tag`` is just a log label
    distinguishing the source ("lightning", "rand_lightning", etc.).
    """
    state = env_system.scheduler.state
    names = state.get("object_names", {})
    primary_name = names.get(object_id, f"id={object_id}")
    all_object_ids = sorted(names.keys()) if names else [object_id]

    fired_groups = _schedule_one_strike(env_system, object_id, 0.0)
    if not fired_groups:
        return 0

    strikes = [(0.0, object_id)]
    cumulative_delay = 0.0
    last_target = object_id
    for i, prob in enumerate(_FOLLOWUP_PROBS):
        if random.random() >= prob:
            break
        # The 2nd strike (i == 0, the first follow-up) gets a
        # longer gap range than subsequent ones so the chain has
        # a real "primary then aftershocks" beat rather than one
        # tight burst.
        if i == 0:
            gap = random.uniform(
                _SECOND_STRIKE_MIN_DELAY_S, _SECOND_STRIKE_MAX_DELAY_S)
        else:
            gap = random.uniform(
                _FOLLOWUP_MIN_DELAY_S, _FOLLOWUP_MAX_DELAY_S)
        cumulative_delay += gap
        next_target = _pick_followup_target(last_target, all_object_ids)
        groups = _schedule_one_strike(env_system, next_target, cumulative_delay)
        if not groups:
            break
        strikes.append((cumulative_delay, next_target))
        last_target = next_target

    if len(strikes) == 1:
        print(f"[wol_{tag}] FLASH on {primary_name!r} "
              f"(object_id={object_id})")
    else:
        chain = ", ".join(
            f"+{d*1000:.0f}ms→{names.get(t, t)!r}" for d, t in strikes
        )
        print(f"[wol_{tag}] FLASH x{len(strikes)} starting on "
              f"{primary_name!r}: {chain}")
    return len(strikes)


def _on_button_3(env_system, object_id: int,
                 button_idx: int, *args: Any) -> None:
    """Button 3: dramatic lightning flash, possibly with follow-ups.
    Thin wrapper over ``fire_lightning_chain`` so the per-frame
    random-events path and the OSC button path produce identical
    multi-strike behaviour.
    """
    fire_lightning_chain(env_system, object_id, tag="button_3")


# Per-button-index dispatch table. Keys are button indices, values are
# callables with the same signature as ``_default_button_handler``.
# Fill in concrete behaviours here as they're authored. Anything not
# listed falls through to the default (log + echo).
_BUTTON_HANDLERS: Dict[int, ButtonHandler] = {
    3: _on_button_3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _host_stem(host: str) -> str:
    """Drop ``.local`` (mDNS suffix) so the router map keys match the
    OSC-path host segment the boxes actually send. Empty string in →
    empty string out (caller filters)."""
    h = (host or "").strip()
    if h.endswith(".local"):
        h = h[:-len(".local")]
    return h
