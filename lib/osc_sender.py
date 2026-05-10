"""Outbound OSC for closed-loop project communication.

Companion to ``lib/osc_listener.py``. Where the listener parks incoming
messages, this module *sends* messages back out — typically to the same
boxes that authored the incoming traffic, so a project hook can complete
a closed loop (e.g. button-press feedback, debug pings, per-object
control).

Targets are addressed by:
  * ``int``  — an object id from the project's receiver list
  * ``str``  — either an object name ("center") or a literal hostname
               ("wol-center.local"). Names are resolved against the
               project's name → object_id map; hostnames go straight to
               the resolver.
  * ``tuple[str, int]`` — explicit ``(host_or_ip, port)``, used when
               sending to something that isn't a project receiver.

Hostnames are resolved to IPs via ``lib.mdns_resolve.resolve`` (with
caching); resolution happens on the first send to each host and the
result is reused. Failures are logged once-per-host so an offline box
doesn't flood the terminal at every send.

Threading: ``send()`` is fire-and-forget UDP and is safe to call from
any thread (OSC handler thread, scheduler thread, render loop). The
underlying ``SimpleUDPClient`` is a UDP socket; concurrent sendto()
calls are kernel-serialized.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple, Union

try:
    from pythonosc.udp_client import SimpleUDPClient
    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _PYTHONOSC_AVAILABLE = False

from lib.mdns_resolve import resolve as _mdns_resolve


# Target shape — one of:
#   int                — object id (looked up in the project's id→target map)
#   str                — object name OR raw hostname
#   tuple[str, int]    — explicit (host_or_ip, port)
Target = Union[int, str, Tuple[str, int]]


class ProjectOscSender:
    """OSC outbound helper bound to a project's name/host map.

    ``targets_by_id``: maps object_id → (host_or_ip, port). Built at
    project boot from ``project.yaml`` receivers. Hosts may be unresolved
    mDNS names (resolution is lazy — first send to a host triggers it).

    ``name_to_id``: maps the box's display name (from geometry.yaml's
    ``objects[].name``) to its object_id, so callers can write
    ``send("center", "/foo", 1)`` instead of memorising IDs.

    ``return_port``: default port to send to when the resolved target
    only has a host (no explicit port). Per-project, defaults to 9000.
    """

    def __init__(
        self,
        targets_by_id: Dict[int, Tuple[str, int]],
        name_to_id: Dict[str, int],
        return_port: int = 9000,
    ):
        self._targets_by_id = dict(targets_by_id)
        self._name_to_id = dict(name_to_id)
        self.return_port = int(return_port)

        # Resolved (ip, port) per host_or_ip key. Lazy-filled.
        self._resolved: Dict[str, Optional[Tuple[str, int]]] = {}
        self._clients: Dict[Tuple[str, int], "SimpleUDPClient"] = {}
        self._lock = threading.Lock()
        # Track hosts we've already warned about so the log doesn't
        # repeat on every send.
        self._warned_unknown: set = set()
        self._warned_unresolved: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, target: Target, address: str, *args: Any) -> bool:
        """Send an OSC message to ``target``. Returns True on success
        (queued to UDP), False on resolution / unknown-target failure.

        Failures are non-fatal: a missing target logs once and returns
        False; the caller can ignore the return value for fire-and-forget
        use, or branch on it for diagnostics."""
        if not _PYTHONOSC_AVAILABLE:
            return False

        endpoint = self._resolve_target(target)
        if endpoint is None:
            return False

        try:
            client = self._get_client(endpoint)
            # SimpleUDPClient.send_message signature: (address, value)
            # where value is either a single arg or a list. python-osc
            # accepts a list of mixed types; pass our args list as-is.
            if len(args) == 0:
                client.send_message(address, [])
            elif len(args) == 1:
                client.send_message(address, args[0])
            else:
                client.send_message(address, list(args))
            return True
        except Exception as e:
            print(f"[OSC out] send to {endpoint} {address}: {e}")
            return False

    def update_target(self, object_id: int, host: str,
                      port: Optional[int] = None) -> None:
        """Replace or add an object's outbound target. Use this if the
        runtime learns of a box's address late (e.g. dynamic DHCP) so
        future sends pick up the new endpoint without restarting."""
        if port is None:
            port = self.return_port
        with self._lock:
            self._targets_by_id[int(object_id)] = (host, int(port))
            # Drop any cached resolution / client tied to the old host
            # so the next send re-resolves.
            self._resolved.pop(host, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_target(self, target: Target) -> Optional[Tuple[str, int]]:
        """Walk target → (host_or_ip, port) → (resolved_ip, port)."""
        # Step 1: target → (host_or_ip, port)
        host_port: Optional[Tuple[str, int]] = None

        if isinstance(target, tuple) and len(target) == 2:
            host_port = (str(target[0]), int(target[1]))
        elif isinstance(target, int):
            host_port = self._targets_by_id.get(int(target))
            if host_port is None:
                if target not in self._warned_unknown:
                    print(f"[OSC out] no target for object_id {target}")
                    self._warned_unknown.add(target)
                return None
        elif isinstance(target, str):
            # Name first (matches the project's display names), then
            # treat as a literal hostname.
            oid = self._name_to_id.get(target)
            if oid is not None:
                host_port = self._targets_by_id.get(oid)
                if host_port is None and target not in self._warned_unknown:
                    print(f"[OSC out] name {target!r} maps to object_id "
                          f"{oid} but no target endpoint")
                    self._warned_unknown.add(target)
            else:
                host_port = (target, self.return_port)
        else:
            return None

        if host_port is None:
            return None

        # Step 2: resolve the host portion via mDNS. Look-alike IPs
        # (already dotted-quad) bypass resolution.
        host, port = host_port
        if _looks_like_ip(host):
            return (host, port)

        with self._lock:
            cached = self._resolved.get(host, "MISS")
        if cached != "MISS":
            return cached if cached is not None else None

        ip = _mdns_resolve(host, timeout=3.0)
        with self._lock:
            self._resolved[host] = (ip, port) if ip else None
        if ip is None:
            if host not in self._warned_unresolved:
                print(f"[OSC out] could not resolve {host!r}; sends will "
                      f"silently drop until the host comes online")
                self._warned_unresolved.add(host)
            return None
        return (ip, port)

    def _get_client(self, endpoint: Tuple[str, int]) -> "SimpleUDPClient":
        """Lazy per-endpoint SimpleUDPClient cache. Same endpoint reuses
        the same socket object across all sends."""
        with self._lock:
            client = self._clients.get(endpoint)
            if client is None:
                client = SimpleUDPClient(endpoint[0], endpoint[1])
                self._clients[endpoint] = client
            return client


def _looks_like_ip(s: str) -> bool:
    """Quick check: dotted-quad IPv4. Doesn't validate octets — just
    skips mDNS resolution for things that look already-resolved."""
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit():
            return False
    return True


def make_send_callable(sender: ProjectOscSender) -> Callable[..., bool]:
    """Return a bound callable suitable for parking on
    ``state['osc_send']``. Effects can call it without holding a
    reference to the ProjectOscSender instance."""
    def _send(target: Target, address: str, *args: Any) -> bool:
        return sender.send(target, address, *args)
    return _send
