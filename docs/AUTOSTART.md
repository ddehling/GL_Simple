# GL_Simple Autostart (Pop!_OS / systemd user service)

Stories_OGL.py is configured to start on boot via a systemd **user service** with linger enabled. This works because `config.yaml` has `headless: true` — no desktop session is required.

## Files

Source-of-truth copies are version-controlled in this repo:

- `bin/gl-simple.service` — the systemd user unit
- `bin/wait_for_audio.sh` — ExecStartPre helper that waits for a real audio sink

At runtime the service unit must be installed to `~/.config/systemd/user/`:

```bash
mkdir -p ~/.config/systemd/user
cp bin/gl-simple.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable gl-simple.service
```

The `wait_for_audio.sh` script is executed from its repo path (`/home/led/Desktop/devel/GL_Simple/bin/wait_for_audio.sh`), so it doesn't need to be copied — just keep it executable (`chmod +x bin/wait_for_audio.sh`).

If you edit `bin/gl-simple.service` in the repo, re-run `cp` + `daemon-reload` + `systemctl --user restart gl-simple` to apply.

## Remaining setup (run once)

Enable linger so the user service starts at boot *before* login:

```bash
sudo loginctl enable-linger led
```

Verify:

```bash
loginctl show-user led | grep Linger   # should show Linger=yes
```

## Starting / testing

Do **not** start the service while `Stories_OGL.py` is already running manually — they will conflict on the web port (5000), sACN, and the audio device.

- **Reboot test (cleanest):** `sudo reboot`, then after boot run `systemctl --user status gl-simple`.
- **Test now:** kill the running instance first, then `systemctl --user start gl-simple`.

## Managing the service

```bash
systemctl --user status gl-simple        # check state
systemctl --user start gl-simple         # start
systemctl --user stop gl-simple          # stop
systemctl --user restart gl-simple       # restart
systemctl --user disable gl-simple       # disable autostart
systemctl --user enable gl-simple        # re-enable autostart
journalctl --user -u gl-simple -f        # live logs
journalctl --user -u gl-simple -n 200    # last 200 log lines
```

After editing the unit file:

```bash
systemctl --user daemon-reload
systemctl --user restart gl-simple
```

## If you switch to `headless: false`

A user service with linger runs without a graphical session, so the GL window won't appear. Switch to a desktop autostart entry instead:

1. Disable the service: `systemctl --user disable --now gl-simple`
2. Create `~/.config/autostart/gl-simple.desktop`:

   ```ini
   [Desktop Entry]
   Type=Application
   Name=GL_Simple
   Exec=/home/led/Desktop/devel/GL_Simple/venv/bin/python /home/led/Desktop/devel/GL_Simple/Stories_OGL.py
   Path=/home/led/Desktop/devel/GL_Simple
   X-GNOME-Autostart-enabled=true
   ```

   This runs after login instead of at boot.

## Auto-stop on login (dev workflow)

The service has `Conflicts=graphical-session.target` — when you log into the desktop, Cosmic activates `graphical-session.target` and systemd stops gl-simple automatically. This frees the web port, sACN, audio device, and DMX for dev work.

- At boot without login: service runs.
- On login: service stops.
- On logout (without reboot): service does NOT auto-restart. Manually: `systemctl --user start gl-simple`, or just reboot.
- Temporarily disable autostart for a dev session: `systemctl --user disable --now gl-simple` (re-enable with `enable --now`).

## Required group membership

`led` must be in `audio`, `render`, and `video` groups. Without these, PipeWire can't open `/dev/snd/*` and EGL can't open `/dev/dri/*` pre-login (logind only grants device ACLs to users with an active seat login).

```bash
sudo usermod -a -G audio,render,video led
# then reboot for the user systemd manager to pick up the new groups
```

Verify: `groups led` should include `audio render video`.

## Why the service waits for the audio graph

Even with linger enabled, `pipewire.service`, `pipewire-pulse.service`, and `wireplumber.service` are socket-activated — they don't actually start at boot until a client connects. And once they do start, WirePlumber needs time to enumerate ALSA cards and assign a default sink before any playback will produce sound. The unit handles both:

1. `After=` / `Wants=` list all three PipeWire services so systemd orders gl-simple after them.
2. `ExecStartPre` explicitly starts pipewire/pipewire-pulse/wireplumber so they don't sit idle waiting for a trigger.
3. `ExecStartPre` polls `pactl info` for up to 30 seconds until a real `Default Sink:` is reported — this is the key to audio output working before a user logs in.
4. A final 3-second settle delay before Python launches.

You can see the readiness log in the journal: look for `audio ready: default sink=...`.

## Troubleshooting

- **Sound doesn't play at boot (before login):** check `journalctl --user -u gl-simple -b | grep -E "(audio ready|WARN|sink)"`. If you see `WARN: no default sink after 30s`, WirePlumber isn't auto-assigning a default without a session — manual fix is to force a sink via `wpctl set-default <ID>` in a `ExecStartPre` line using the known device name.
- **TONOR mic not detected:** USB enumeration can lag. Increase the final `ExecStartPre=/bin/sleep` value to 10-15s.
- **Logs for current boot:** `journalctl --user -u gl-simple -b`
- **Logs for a previous boot:** `journalctl --user -u gl-simple -b -1` (2 boots ago = `-b -2`, etc.)
- **Inspect service state without logging in:** SSH into the machine, then `journalctl --user -u gl-simple -b`. Or from a local root session: `sudo machinectl shell led@` then run the user systemctl commands.
