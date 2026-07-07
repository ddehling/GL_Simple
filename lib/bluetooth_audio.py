"""Bluetooth A2DP audio-sink controller for GL_Simple (Linux only).

Turns this machine into a Bluetooth speaker advertised as **lucifera**: a
phone (or any A2DP source) connects and streams audio in. The incoming
stream shows up as a PipeWire / PulseAudio source, which the audio analyzer
captures as the ``"bluetooth"`` input source (see lib/audio_analyzer.py).

Design constraints (set by the feature request):
  * LINUX ONLY. Windows cannot act as an A2DP *sink* with its built-in
    Bluetooth stack, so on Windows/macOS — or any host missing BlueZ,
    D-Bus, or PyGObject — this module degrades to an inert stub: ``available``
    is False and every method is a safe no-op. Importing and instantiating it
    NEVER breaks the app. The web UI hides the control when ``available`` is
    False.
  * PER-DEVICE APPROVAL. While enabled, an incoming pairing / connection
    request is NOT auto-accepted. It is queued (``pending_pairings()``) for an
    admin to approve or deny from the web panel. The BlueZ agent reply is
    deferred until ``approve(mac)`` / ``deny(mac)`` is called.

Threading model: BlueZ is driven over the D-Bus system bus with a GLib main
loop running on a dedicated daemon thread. Public methods called from other
threads (enable/disable/approve/deny) marshal their work onto the GLib loop
via ``GLib.idle_add`` so all D-Bus interaction happens on one thread.

This requires, on the Linux host:
  * a running ``bluetoothd`` (BlueZ) with an A2DP-sink-capable adapter,
  * PipeWire (Pop OS, current Raspberry Pi OS) or PulseAudio with the
    Bluetooth modules loaded, so the connected device becomes a capture source,
  * the app's user in the ``bluetooth`` group (to talk to BlueZ over D-Bus).
See docs/BLUETOOTH_AUDIO.md for setup and the on-device test plan.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Dict, List, Optional


# Stable identity, consistent with the mDNS service name (lucifera.local) and
# the web panel. This is the name a phone sees when it scans for devices.
ADAPTER_ALIAS = "lucifera"

# BlueZ D-Bus constants.
_BLUEZ_SERVICE = "org.bluez"
_ADAPTER_IFACE = "org.bluez.Adapter1"
_DEVICE_IFACE = "org.bluez.Device1"
_AGENT_IFACE = "org.bluez.Agent1"
_AGENT_MANAGER_IFACE = "org.bluez.AgentManager1"
_PROPS_IFACE = "org.freedesktop.DBus.Properties"
_OBJMGR_IFACE = "org.freedesktop.DBus.ObjectManager"
_AGENT_PATH = "/gl_simple/lucifera/agent"
# A2DP Audio Source UUID (what a phone offers a speaker). Used to label the
# kind of authorization being requested.
_A2DP_SOURCE_UUID = "0000110a-0000-1000-8000-00805f9b34fb"


def _deps_available() -> bool:
    """True only on Linux with dbus-python + PyGObject importable."""
    if platform.system() != "Linux":
        return False
    try:
        import dbus  # noqa: F401
        import dbus.mainloop.glib  # noqa: F401
        from gi.repository import GLib  # noqa: F401
        return True
    except Exception:
        return False


def create_bluetooth_receiver(enabled_default: bool = False):
    """Factory: a real BlueZ receiver on a capable Linux host, else a stub.

    ``enabled_default`` is accepted for symmetry but intentionally ignored —
    the feature defaults OFF and is only ever turned on from the web UI.
    """
    if _deps_available():
        try:
            return _BlueZReceiver()
        except Exception as e:  # pragma: no cover - host-specific
            print(f"[BT] BlueZ init failed ({e}); Bluetooth audio disabled")
            return _StubReceiver()
    return _StubReceiver()


class _StubReceiver:
    """Inert no-op receiver for non-Linux / missing-deps hosts.

    Mirrors the real interface so callers never need a platform check.
    """

    available = False
    adapter_alias = ADAPTER_ALIAS

    def __init__(self, reason: str = ""):
        self._reason = reason or (
            "Bluetooth audio sink requires Linux with BlueZ + PipeWire/PulseAudio"
        )

    # --- lifecycle ---
    def enable(self) -> bool:
        return False

    def disable(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    # --- state ---
    def is_enabled(self) -> bool:
        return False

    def pending_pairings(self) -> List[dict]:
        return []

    def connected_devices(self) -> List[dict]:
        return []

    def source_hint(self) -> Optional[str]:
        return None

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    # --- approval ---
    def approve(self, mac: str) -> None:
        pass

    def deny(self, mac: str) -> None:
        pass


class _BlueZReceiver:
    """Real A2DP-sink controller over BlueZ / D-Bus (Linux).

    Lifecycle: constructed once at startup (cheap; only opens the system bus
    and finds an adapter). ``enable()`` powers the adapter, sets the alias to
    ``lucifera``, registers the pairing agent, and makes the adapter
    discoverable + pairable. ``disable()`` stops discoverability and pairing
    but leaves the alias intact. The GLib loop thread starts on first
    ``enable()`` and runs until ``shutdown()``.
    """

    available = True
    adapter_alias = ADAPTER_ALIAS

    def __init__(self):
        import dbus
        import dbus.mainloop.glib
        from gi.repository import GLib

        self._dbus = dbus
        self._GLib = GLib

        # The agent's deferred BlueZ replies, keyed by device object path.
        # Each entry: {"mac", "name", "kind", "ok", "err", "path"}.
        self._pending: Dict[str, dict] = {}
        # Connected A2DP devices, keyed by mac: {"mac", "name"}.
        self._connected: Dict[str, dict] = {}
        self._lock = threading.Lock()

        self._enabled = False
        self._loop = None
        self._loop_thread: Optional[threading.Thread] = None
        self._agent_registered = False

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self._bus = dbus.SystemBus()

        self._adapter_path = self._find_adapter_path()
        if self._adapter_path is None:
            raise RuntimeError("no BlueZ adapter found")
        print(f"[BT] adapter {self._adapter_path} (alias will be '{ADAPTER_ALIAS}')")

    # ------------------------------------------------------------------
    # D-Bus helpers
    # ------------------------------------------------------------------

    def _find_adapter_path(self) -> Optional[str]:
        try:
            mgr = self._dbus.Interface(
                self._bus.get_object(_BLUEZ_SERVICE, "/"), _OBJMGR_IFACE)
            for path, ifaces in mgr.GetManagedObjects().items():
                if _ADAPTER_IFACE in ifaces:
                    return str(path)
        except Exception as e:
            print(f"[BT] adapter discovery failed: {e}")
        return None

    def _adapter_props(self):
        return self._dbus.Interface(
            self._bus.get_object(_BLUEZ_SERVICE, self._adapter_path), _PROPS_IFACE)

    def _set_adapter_prop(self, name: str, value) -> None:
        self._adapter_props().Set(_ADAPTER_IFACE, name, value)

    def _device_name(self, path: str) -> str:
        try:
            props = self._dbus.Interface(
                self._bus.get_object(_BLUEZ_SERVICE, path), _PROPS_IFACE)
            return str(props.Get(_DEVICE_IFACE, "Alias"))
        except Exception:
            return "Unknown device"

    @staticmethod
    def _mac_from_path(path: str) -> str:
        # .../dev_AA_BB_CC_DD_EE_FF -> AA:BB:CC:DD:EE:FF
        tail = path.rsplit("/", 1)[-1]
        if tail.startswith("dev_"):
            return tail[4:].replace("_", ":")
        return tail

    # ------------------------------------------------------------------
    # GLib loop thread + cross-thread marshalling
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop = self._GLib.MainLoop()

        def _run():
            try:
                self._loop.run()
            except Exception as e:  # pragma: no cover
                print(f"[BT] GLib loop stopped: {e}")

        self._loop_thread = threading.Thread(
            target=_run, name="bt-glib", daemon=True)
        self._loop_thread.start()

    def _on_loop(self, func) -> None:
        """Schedule ``func`` to run once on the GLib loop thread."""
        self._GLib.idle_add(lambda: (func(), False)[1])

    # ------------------------------------------------------------------
    # Pairing agent (org.bluez.Agent1)
    # ------------------------------------------------------------------

    def _build_agent(self):
        import dbus.service

        receiver = self

        class _Agent(dbus.service.Object):
            # Each authorization-style method uses async_callbacks so the reply
            # to BlueZ is deferred until the web admin approves/denies. Until
            # then BlueZ waits; the phone shows "connecting".

            @dbus.service.method(_AGENT_IFACE, in_signature="", out_signature="")
            def Release(self):
                pass

            @dbus.service.method(_AGENT_IFACE, in_signature="os",
                                 out_signature="", async_callbacks=("ok", "err"))
            def AuthorizeService(self, device, uuid, ok, err):
                receiver._queue(str(device), "connect", ok, err)

            @dbus.service.method(_AGENT_IFACE, in_signature="o",
                                 out_signature="", async_callbacks=("ok", "err"))
            def RequestAuthorization(self, device, ok, err):
                receiver._queue(str(device), "pair", ok, err)

            @dbus.service.method(_AGENT_IFACE, in_signature="ou",
                                 out_signature="", async_callbacks=("ok", "err"))
            def RequestConfirmation(self, device, passkey, ok, err):
                receiver._queue(str(device), "pair", ok, err)

            @dbus.service.method(_AGENT_IFACE, in_signature="o",
                                 out_signature="s")
            def RequestPinCode(self, device):
                # Legacy PIN pairing — phones use SSP, so reject to keep the
                # approval path single (numeric/just-works only).
                raise _Rejected("PIN pairing not supported")

            @dbus.service.method(_AGENT_IFACE, in_signature="o",
                                 out_signature="u")
            def RequestPasskey(self, device):
                raise _Rejected("passkey entry not supported")

            @dbus.service.method(_AGENT_IFACE, in_signature="os",
                                 out_signature="")
            def DisplayPinCode(self, device, pincode):
                pass

            @dbus.service.method(_AGENT_IFACE, in_signature="ouq",
                                 out_signature="")
            def DisplayPasskey(self, device, passkey, entered):
                pass

            @dbus.service.method(_AGENT_IFACE, in_signature="",
                                 out_signature="")
            def Cancel(self):
                receiver._cancel_all()

        return _Agent(self._bus, _AGENT_PATH)

    def _queue(self, device_path: str, kind: str, ok, err) -> None:
        """Called from the GLib thread by the agent. Stash the deferred reply
        and surface the request to the web UI via pending_pairings()."""
        mac = self._mac_from_path(device_path)
        name = self._device_name(device_path)
        with self._lock:
            # A device re-requesting (e.g. pair then connect) supersedes the
            # earlier deferred reply; reject the stale one so BlueZ isn't left
            # waiting on two callbacks for one path.
            old = self._pending.get(device_path)
            if old is not None:
                try:
                    old["err"](_Rejected("superseded"))
                except Exception:
                    pass
            self._pending[device_path] = {
                "mac": mac, "name": name, "kind": kind,
                "ok": ok, "err": err, "path": device_path,
            }
        print(f"[BT] {kind} request from {name} [{mac}] — awaiting web approval")

    def _cancel_all(self) -> None:
        with self._lock:
            self._pending.clear()

    # ------------------------------------------------------------------
    # Connection tracking
    # ------------------------------------------------------------------

    def _watch_connections(self) -> None:
        # Fires whenever a Device1's properties change; we care about Connected.
        self._bus.add_signal_receiver(
            self._on_device_props,
            dbus_interface=_PROPS_IFACE,
            signal_name="PropertiesChanged",
            arg0=_DEVICE_IFACE,
            path_keyword="path",
        )
        # Seed with devices that are ALREADY connected: a phone paired and
        # attached before the app (re)started never fires PropertiesChanged,
        # so without this scan it stays invisible and the analyzer's
        # bluetooth auto-switch never triggers.
        try:
            mgr = self._dbus.Interface(
                self._bus.get_object(_BLUEZ_SERVICE, "/"), _OBJMGR_IFACE)
            for path, ifaces in mgr.GetManagedObjects().items():
                dev = ifaces.get(_DEVICE_IFACE)
                if dev and bool(dev.get("Connected", False)):
                    mac = self._mac_from_path(str(path))
                    name = str(dev.get("Name", mac))
                    with self._lock:
                        self._connected[mac] = {"mac": mac, "name": name}
                    print(f"[BT] already connected: {name} [{mac}]")
        except Exception as e:
            print(f"[BT] connected-device scan failed: {e}")

    def _on_device_props(self, interface, changed, invalidated, path=None):
        if "Connected" not in changed or path is None:
            return
        mac = self._mac_from_path(path)
        if bool(changed["Connected"]):
            name = self._device_name(path)
            with self._lock:
                self._connected[mac] = {"mac": mac, "name": name}
            print(f"[BT] connected: {name} [{mac}]")
        else:
            with self._lock:
                self._connected.pop(mac, None)
            print(f"[BT] disconnected: [{mac}]")

    # ------------------------------------------------------------------
    # Public API (thread-safe; D-Bus work marshalled onto the GLib loop)
    # ------------------------------------------------------------------

    def enable(self) -> bool:
        self._ensure_loop()

        def _do():
            try:
                if not self._agent_registered:
                    self._agent = self._build_agent()
                    mgr = self._dbus.Interface(
                        self._bus.get_object(_BLUEZ_SERVICE, "/org/bluez"),
                        _AGENT_MANAGER_IFACE)
                    mgr.RegisterAgent(_AGENT_PATH, "DisplayYesNo")
                    mgr.RequestDefaultAgent(_AGENT_PATH)
                    self._watch_connections()
                    self._agent_registered = True
                self._set_adapter_prop("Powered", self._dbus.Boolean(True))
                self._set_adapter_prop("Alias", self._dbus.String(ADAPTER_ALIAS))
                self._set_adapter_prop("Pairable", self._dbus.Boolean(True))
                # 0 = no timeout: stays discoverable until disabled.
                self._set_adapter_prop(
                    "DiscoverableTimeout", self._dbus.UInt32(0))
                self._set_adapter_prop("Discoverable", self._dbus.Boolean(True))
                self._enabled = True
                print(f"[BT] enabled — discoverable as '{ADAPTER_ALIAS}'")
            except Exception as e:
                print(f"[BT] enable failed: {e}")
                self._enabled = False

        self._on_loop(_do)
        return True

    def disable(self) -> None:
        def _do():
            try:
                self._set_adapter_prop("Discoverable", self._dbus.Boolean(False))
                self._set_adapter_prop("Pairable", self._dbus.Boolean(False))
            except Exception as e:
                print(f"[BT] disable warning: {e}")
            # Reject anything still awaiting approval.
            with self._lock:
                pend = list(self._pending.values())
                self._pending.clear()
            for p in pend:
                try:
                    p["err"](_Rejected("bluetooth disabled"))
                except Exception:
                    pass
            self._enabled = False
            print("[BT] disabled — no longer discoverable")

        self._on_loop(_do)

    def shutdown(self) -> None:
        try:
            self.disable()
        except Exception:
            pass
        if self._loop is not None:
            self._on_loop(self._loop.quit)

    def is_enabled(self) -> bool:
        return self._enabled

    def pending_pairings(self) -> List[dict]:
        with self._lock:
            return [{"mac": p["mac"], "name": p["name"], "kind": p["kind"]}
                    for p in self._pending.values()]

    def connected_devices(self) -> List[dict]:
        with self._lock:
            return list(self._connected.values())

    def source_hint(self) -> Optional[str]:
        """Name fragment the analyzer should match for the live BT capture
        source, or None if nothing is connected. PipeWire/Pulse name the node
        ``bluez_input.AA_BB_...`` / ``bluez_source.AA_BB_...``; matching the
        connected MAC (underscored) is the most specific. Falls back to the
        generic ``bluez`` token."""
        with self._lock:
            if not self._connected:
                return None
            mac = next(iter(self._connected))
        return mac.replace(":", "_")

    def approve(self, mac: str) -> None:
        self._resolve(mac, approve=True)

    def deny(self, mac: str) -> None:
        self._resolve(mac, approve=False)

    def _resolve(self, mac: str, approve: bool) -> None:
        with self._lock:
            match = next((p for p in self._pending.values()
                          if p["mac"].lower() == mac.lower()), None)
            if match is not None:
                self._pending.pop(match["path"], None)
        if match is None:
            return

        def _do():
            try:
                if approve:
                    # Mark the device trusted FIRST so BlueZ auto-authorizes the
                    # follow-up service connections (A2DP/AVRCP) without firing
                    # the agent again — operator approves once, not per profile.
                    self._set_trusted(match["path"])
                    match["ok"]()
                    print(f"[BT] approved {match['name']} [{mac}]")
                else:
                    match["err"](_Rejected("denied by operator"))
                    print(f"[BT] denied {match['name']} [{mac}]")
            except Exception as e:
                print(f"[BT] resolve failed for {mac}: {e}")

        self._on_loop(_do)

    def _set_trusted(self, path: str) -> None:
        """Set org.bluez.Device1.Trusted=true so subsequent service
        authorizations for this device are granted automatically by BlueZ."""
        try:
            import dbus
            props = self._dbus.Interface(
                self._bus.get_object(_BLUEZ_SERVICE, path), _PROPS_IFACE)
            props.Set(_DEVICE_IFACE, "Trusted", dbus.Boolean(True))
        except Exception as e:
            print(f"[BT] could not set Trusted on {path}: {e}")


# org.bluez.Error.Rejected — the DBusException BlueZ expects when an agent
# refuses a request. Defined lazily so the module imports cleanly without dbus.
try:  # pragma: no cover - import-time, host dependent
    import dbus

    class _Rejected(dbus.DBusException):
        _dbus_error_name = "org.bluez.Error.Rejected"
except Exception:  # dbus not present (e.g. Windows): never used here.
    class _Rejected(Exception):  # type: ignore
        pass
