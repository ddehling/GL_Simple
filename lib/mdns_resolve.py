"""Resolve a EtherNode mDNS hostname to an IPv4 address.

Lets receivers be addressed by stable hostname (e.g. ``ethernode-7f928c``)
rather than DHCP-fragile IPs. Resolution strategy, in order:

1. **OS resolver** via ``socket.gethostbyname()``. Works on Linux with
   avahi-daemon, on macOS natively, and on Windows if Bonjour is
   installed (most operators have it because iTunes / Adobe / etc.
   bundle it). When this works it's the fastest path.
2. **Zeroconf browse** of ``_ethernode._tcp.local.`` — same service the
   firmware advertises and the EtherNode configurator's discovery uses.
   Catches the case where the OS doesn't speak mDNS but Python's
   zeroconf library does.

The function blocks up to ``timeout`` seconds total. Returns the
dotted-quad string on success, or ``None`` if both strategies miss.
"""

from __future__ import annotations

import socket
import time
from typing import Optional

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
    _ZC_AVAILABLE = True
except ImportError:
    _ZC_AVAILABLE = False


# Process-level cache of successful resolutions. Failed lookups do NOT cache
# — a host that's offline now might come up later, and we'd rather pay the
# timeout again than permanently exclude it. Hit-cache means project-swap
# back to a previously-resolved hostname is instant.
_RESOLVED_CACHE: dict[str, str] = {}


def _strip_local(name: str) -> str:
    """Normalise a hostname by removing trailing ``.local`` / ``.``"""
    n = name.rstrip('.')
    if n.endswith('.local'):
        n = n[:-len('.local')]
    return n


def _try_os_resolve(name: str) -> Optional[str]:
    base = _strip_local(name)
    # gethostbyname accepts both bare names (if the system resolver knows
    # them) and ``foo.local``. Try both forms.
    for candidate in (base, f'{base}.local'):
        try:
            return socket.gethostbyname(candidate)
        except socket.gaierror:
            continue
    return None


def _try_zeroconf(name: str, timeout: float) -> Optional[str]:
    if not _ZC_AVAILABLE:
        return None

    target = _strip_local(name)
    found = {}

    class L(ServiceListener):
        def add_service(self, zc, type_, svc_name):
            info = zc.get_service_info(type_, svc_name)
            if info and info.addresses:
                # Service names look like "ethernode-7f928c._ethernode._tcp.local."
                # — first segment matches the firmware's hostname.
                short = svc_name.split('.')[0]
                found[short] = socket.inet_ntoa(info.addresses[0])

        def remove_service(self, zc, type_, name): pass
        def update_service(self, zc, type_, name): pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, '_ethernode._tcp.local.', L())
        deadline = time.monotonic() + timeout
        # Poll instead of sleeping the full timeout so a fast-arriving
        # match returns immediately. Browser updates the dict from a
        # background thread.
        while time.monotonic() < deadline:
            if target in found:
                return found[target]
            time.sleep(0.1)
        return found.get(target)
    finally:
        zc.close()


def resolve(name: str, timeout: float = 3.0) -> Optional[str]:
    """Resolve `name` (e.g. "ethernode-7f928c" or "ethernode-7f928c.local") to an
    IPv4 string, or return None on failure. See module docstring for
    the resolution order."""
    if not name:
        return None
    # If the caller already passed an IP, just hand it back. Cheap and
    # avoids unnecessary mDNS chatter when configs mix IP and hostname
    # entries.
    try:
        socket.inet_aton(name)
        return name
    except OSError:
        pass

    cached = _RESOLVED_CACHE.get(name)
    if cached is not None:
        return cached

    ip = _try_os_resolve(name)
    if ip is None:
        ip = _try_zeroconf(name, timeout)
    if ip is not None:
        _RESOLVED_CACHE[name] = ip
    return ip
