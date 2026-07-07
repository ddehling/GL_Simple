# Bluetooth Audio Input ("lucifera")

Lets a phone (or any Bluetooth A2DP source) connect to the show machine and
stream audio in, which then drives the audio-reactive visuals. The machine
advertises itself as **lucifera**; an operator enables it and approves each
device from the web control panel.

> **Linux only.** Windows cannot act as a Bluetooth audio *sink* (receiver)
> with its built-in stack — there is no supported API to make the PC appear as
> a Bluetooth speaker. On Windows/macOS the feature self-reports as
> unavailable and the control is hidden. Primary targets: **Pop!_OS** and
> **Raspberry Pi OS**. (If you need Windows, the workaround is a USB Bluetooth
> *audio-receiver dongle* that presents as a normal USB line-in — pair the
> phone to the dongle and select the `Line-in` source; no code involved.)

## How it works

```
phone ──A2DP──▶ BlueZ (bluetoothd) ──▶ PipeWire/PulseAudio capture source
                     ▲                          │
        org.bluez D-Bus agent            "bluetooth" input source
        (per-device approval)            (lib/audio_analyzer.py)
                     ▲                          │
        lib/bluetooth_audio.py  ◀── control ──▶ FFT / bands / beat → shaders
                     ▲
        web panel toggle + approve/deny (Socket.IO)
```

- **[lib/bluetooth_audio.py](../lib/bluetooth_audio.py)** — `BluetoothAudioReceiver`
  drives BlueZ over the D-Bus system bus on a dedicated GLib loop thread. It
  sets the adapter alias to `lucifera`, makes it discoverable/pairable on
  demand, and registers a pairing **agent** that defers every pairing/connection
  request for operator approval (it never auto-accepts). On a non-Linux host or
  one missing BlueZ/D-Bus/PyGObject it is an inert stub — importing/using it
  never breaks the app.
- **[lib/audio_analyzer.py](../lib/audio_analyzer.py)** — gains a `"bluetooth"`
  input source that captures from the connected device's PipeWire/Pulse node
  (`bluez_input.<MAC>` / `bluez_source.<MAC>`), matched by the live MAC hint.
- **[Stories_OGL.py](../Stories_OGL.py)** — owns the receiver, drains web
  actions (enable/disable/approve/deny) on the existing ~5 Hz web-control tick,
  mirrors state to the UI, and **auto-switches** the analyzer to the
  `bluetooth` source when a device connects (restoring the prior source when it
  disconnects).
- **Web** — a Bluetooth block in the Audio section of the control panel: an
  On/Off toggle, an approval card per pending device, and the connected list.
  Open to anyone with the control panel — per-device pairing approval is the
  access control.

## Setup

`bin/linux-install.sh` does all of this automatically (non-fatal if any part
fails — the feature just shows as unavailable). To do it by hand:

```bash
# Runtime: BlueZ + PipeWire Bluetooth plugin (or the PulseAudio module)
sudo apt install bluez libspa-0.2-bluetooth        # PipeWire (Pop!_OS, current Pi OS)
# sudo apt install bluez pulseaudio-module-bluetooth  # older PulseAudio hosts

# Build deps for the D-Bus / GLib Python bindings, then install into the venv.
# python3-dev is required — dbus-python's build fails with "Python dependency
# not found" without it.
sudo apt install python3-dev libdbus-1-dev libglib2.0-dev libgirepository1.0-dev \
                 libcairo2-dev gobject-introspection
source venv/bin/activate
# PyGObject is pinned <3.52: 3.52+ needs girepository-2.0, but apt ships
# libgirepository1.0-dev (the 1.0 series) on Ubuntu/Pop 24.04.
pip install dbus-python "PyGObject<3.52"

# Let the app talk to BlueZ without root (re-login afterward)
sudo usermod -aG bluetooth "$USER"
```

> **venv note:** the bindings must be importable *inside the venv*. The commands
> above pip-install them there. If you'd rather use the system packages
> (`python3-dbus`, `python3-gi`), recreate the venv with
> `python3 -m venv --system-site-packages venv` — but that exposes all system
> packages and can shadow pinned versions, so the pip route is preferred.

Confirm the bindings import in the venv:

```bash
source venv/bin/activate
python -c "import dbus, dbus.mainloop.glib; from gi.repository import GLib; print('BT deps OK')"
```

## On-device test plan (Pi / Pop!_OS)

This feature **cannot be tested on Windows** (the dev box) — verify it here.

1. **Availability.** Start the app and open the web panel → Audio section. The
   "Bluetooth Input (lucifera)" block should appear with an **Off** toggle. If
   it instead shows "Bluetooth input unavailable: …", the bindings or BlueZ
   aren't ready — fix per the message, then re-check the import command above.
2. **Enable.** Click the toggle → it turns **On**, status reads *"Discoverable
   as lucifera"*. Verify from a phone: scan for Bluetooth devices — **lucifera**
   should appear. (CLI cross-check: `bluetoothctl show` → `Discoverable: yes`,
   `Alias: lucifera`.)
3. **Approval.** Tap `lucifera` on the phone to connect. An approval card should
   appear in the panel ("… wants to pair/connect"). The phone shows
   "connecting" and waits. Click **Approve** → pairing completes. (Click
   **Deny** to confirm the phone is rejected instead.)
4. **Audio routes.** Play music on the phone. The panel should show
   *"Connected: <phone>"*, the input source auto-switches to **Bluetooth**, and
   the audio meter + audio-reactive visuals respond. CLI cross-check that the
   capture node exists: `pactl list short sources | grep bluez` (works for both
   PipeWire and PulseAudio).
5. **Disconnect.** Disconnect from the phone → the source falls back to the
   prior input; "Connected" clears.
6. **Disable.** Toggle **Off** → `lucifera` disappears from phone scans;
   `bluetoothctl show` reports `Discoverable: no`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Block shows "unavailable" | `import dbus`/`gi` fails in the venv, or no BlueZ adapter. Run the import check; ensure `bluetoothd` is running (`systemctl status bluetooth`). |
| `lucifera` not discoverable | Toggle is off, or adapter is soft-blocked: `rfkill unblock bluetooth`. Check `bluetoothctl show`. |
| Pairs but no audio source | PipeWire BT plugin missing — install `libspa-0.2-bluetooth` (or `pulseaudio-module-bluetooth`) and restart the audio service / re-login. Confirm with `pactl list short sources | grep bluez`. |
| Connected but visuals barely react / react to the room | The analyzer must capture the bluez node natively — the log should say `[Audio] source=bluetooth via pw-record 'bluez_input...'` (or `via parec` on genuine PulseAudio). If it says `via 'default'` or `no capture node found`, install `pipewire-utils`/`pulseaudio-utils` (`pw-record`, `pactl`). Without them, PortAudio can't see bluez nodes and silently captures the default input (the mic) instead. Note: on PipeWire, `parec` can also serve hw-sink monitors broken/silent — `pw-record` is the reliable path. |
| Approval card never appears | App not in the `bluetooth` group (re-login after `usermod`), or another default agent is registered (e.g. a desktop's). Stop competing agents or run on a headless box. |
| Audio is choppy | A2DP codec/quality — usually fine; check `pw-top` / CPU on a Pi. |

## Security notes

- Nothing connects without an operator **Approve** while the toggle is on; the
  toggle off-state stops all discoverability and pairing.
- The adapter alias is set to `lucifera` system-wide while enabled; it is left
  as-is on disable (cosmetic).
