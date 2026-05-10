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
    """Parse ``/wol/<host>/<kind>/<idx>``. Currently only ``kind=button``
    is wired; other kinds (sensor, status, etc.) silently fall through
    so the catch-all log doesn't fire on them either. Adjust if you
    want unknown kinds to log."""
    parts = address.strip("/").split("/")
    # Expect: ["wol", "<host>", "<kind>", "<idx>"], so 4 parts. Be
    # tolerant of trailing segments — first 4 are what we route on.
    if len(parts) < 4:
        return

    host = parts[1]
    kind = parts[2]
    if kind != "button":
        # Reserved for future kinds — silently ignore for now.
        return

    try:
        button_idx = int(parts[3])
    except ValueError:
        return

    object_id = host_map.get(host, -1)
    if object_id < 0:
        # Unknown device; log once-ish (real dedup belongs in the
        # listener if needed — for now this is rare enough that one
        # warning per stray message is OK).
        print(f"[wol_button] unknown host {host!r} on {address}; "
              f"args={args}")
        return

    handler = _BUTTON_HANDLERS.get(button_idx, _default_button_handler)
    try:
        handler(env_system, object_id, button_idx, *args)
    except Exception as e:
        print(f"[wol_button] handler for button {button_idx} "
              f"on object {object_id} raised: {e}")


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


def _on_button_3(env_system, object_id: int,
                 button_idx: int, *args: Any) -> None:
    """Button 3: schedule a dramatic lightning flash on the box that
    sent the press. Lights up the box's strips on BOTH the Sky and
    Ground groups (atlases gate per-pixel to that object only). The
    Sky pass also dims the surrounding non-target sky pixels for a
    couple of seconds — the "flash blindness" effect — recovering
    smoothly to the day/night gradient. Ground pass doesn't dim
    (a darkened ring around the lit ground arc reads as a bug
    rather than drama).

    Each press uses ``functools.partial`` to bake in target_object_id
    + the event_map-side params, producing a unique action object so
    EventScheduler's dedup-by-action doesn't drop concurrent flashes
    on different boxes.
    """
    state = env_system.scheduler.state
    name = state.get("object_names", {}).get(object_id, f"id={object_id}")

    entry = env_system.weather_set.resolve_event(_LIGHTNING_EVENT)
    if entry is None:
        print(f"[wol_button_3] event {_LIGHTNING_EVENT!r} not in event_map; "
              f"flash skipped")
        return
    effect_func, params, _ = entry

    g2f = env_system.project.group_to_frame_id()

    # Schedule one event per group the lightning should touch. Each
    # gets a fresh partial so the scheduler's coarse dedup-by-action
    # check (event.action == action) doesn't reject the second.
    fired_groups = []
    for group_name, dim_enable in _LIGHTNING_GROUPS:
        if group_name not in g2f:
            continue
        action = functools.partial(
            effect_func,
            target_object_id=object_id,
            dim_enable=dim_enable,
            **params,
        )
        ev = env_system.scheduler.schedule_event(
            0.0,
            _LIGHTNING_DURATION_S,
            action,
            frame_id=g2f[group_name],
            name=f"lightning_obj{object_id}_{group_name}",
        )
        if ev is not None:
            fired_groups.append(group_name)
        else:
            print(f"[wol_button_3] schedule rejected (dedup) for "
                  f"{name} on {group_name}")

    if fired_groups:
        print(f"[wol_button_3] FLASH on {name!r} "
              f"(object_id={object_id}, groups={fired_groups})")


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
