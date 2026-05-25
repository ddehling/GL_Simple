# GL_Simple Autostart (Pop!_OS / Cosmic session)

Stories_OGL.py is configured to start automatically on boot by enabling **auto-login** to the Cosmic desktop, then firing a standard XDG **autostart** entry inside the session. The app runs in a visible `xterm` window so you can `Ctrl+C` to stop it like any other dev run.

A legacy systemd user service also exists (disabled) — see [Alternative: systemd user service](#alternative-systemd-user-service) at the bottom.

## One-script install

```bash
./bin/install_autostart.sh
```

Installs xterm, adds you to the `audio render video` groups, configures greetd auto-login to Cosmic, and writes `~/.config/autostart/gl-simple.desktop` pointing at `bin/run.sh`. Idempotent. Reboot after. The manual recipe below is what the script automates — read it if you need to deviate (different distro, different display manager).

## Files

- `~/.config/autostart/gl-simple.desktop` — XDG autostart entry launched by the Cosmic session on login.
- `/etc/greetd/cosmic-greeter.toml` — greetd config; an `[initial_session]` block here enables auto-login.

## One-time setup

### 1. Enable auto-login

Append an `[initial_session]` block to the greetd config:

```bash
sudo tee -a /etc/greetd/cosmic-greeter.toml > /dev/null <<'EOF'

[initial_session]
command = "/usr/bin/start-cosmic"
user = "led"
EOF
```

Verify: `cat /etc/greetd/cosmic-greeter.toml` should show the new block at the end.

To disable auto-login later, delete that block (use your editor or copy/paste the file without it).

### 2. Install xterm (if not already)

```bash
sudo apt install xterm
```

### 3. Create the autostart entry

Write `~/.config/autostart/gl-simple.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=GL_Simple
Comment=OpenGL lighting controller — starts with the graphical session
Exec=xterm -T GL_Simple -geometry 120x30+50+50 -hold -e /home/led/Desktop/devel/GL_Simple/venv/bin/python /home/led/Desktop/devel/GL_Simple/Stories_OGL.py
Path=/home/led/Desktop/devel/GL_Simple
Terminal=false
X-GNOME-Autostart-enabled=true
```

Notes on the Exec line:
- `-T GL_Simple` — window title.
- `-geometry 120x30+50+50` — 120 cols × 30 rows, positioned at (50, 50). Adjust to taste.
- `-hold` — keeps the window open after the script exits so you can read any traceback.
- `-e <cmd>` — must be the last flag; everything after is the command to run.
- Tweak fonts by adding `-fa Monospace -fs 11` before `-e`.

## What happens on boot

1. greetd sees `[initial_session]` → runs `/usr/bin/start-cosmic` as user `led`.
2. Cosmic session starts (no login screen).
3. The autostart entry fires → xterm opens with Stories_OGL.py running inside.

## Day-to-day use

- **Stop the app:** `Ctrl+C` in the xterm window (or close the window).
- **Re-run it manually:** any terminal → `cd ~/Desktop/devel/GL_Simple && venv/bin/python Stories_OGL.py`.
- **Restart the running xterm instance:** kill the xterm, then re-launch with the same command the `.desktop` file uses. `setsid -f` detaches it so it survives the spawning shell:
  ```bash
  pkill -f 'xterm.*GL_Simple'
  setsid -f xterm -T GL_Simple -geometry 120x30+50+50 -hold \
    -e /home/led/Desktop/devel/GL_Simple/venv/bin/python \
       /home/led/Desktop/devel/GL_Simple/Stories_OGL.py
  ```
  **Note:** `systemctl --user restart gl-simple.service` does **not** affect this instance — the xterm path is launched by XDG autostart, not systemd. The systemd unit is a separate (disabled) code path; see [Alternative: systemd user service](#alternative-systemd-user-service).
- **Skip the autostart for one session:** before reboot, rename the file:
  ```bash
  mv ~/.config/autostart/gl-simple.desktop ~/.config/autostart/gl-simple.desktop.disabled
  ```
  Reboot, do dev work, then rename it back. Or set `Hidden=true` inline.
- **Disable auto-login temporarily:** remove the `[initial_session]` block from `/etc/greetd/cosmic-greeter.toml` and reboot. Re-add to restore.

## Required group membership

`led` must be in `audio`, `render`, and `video` groups — these grant access to the audio and GPU device nodes. The auto-login path doesn't strictly need `audio` (logind grants it via ACL on session start), but these groups are still recommended for headless/remote scenarios and were required for the old systemd-service approach.

```bash
sudo usermod -a -G audio,render,video led
# then reboot for the user systemd manager to pick up the new groups
```

Verify: `groups led` should include `audio render video`.

## Alternative: systemd user service

A user service unit (`bin/gl-simple.service`) and a helper (`bin/wait_for_audio.sh`) are checked into the repo. This path starts Stories_OGL.py **before login**, useful for truly headless installations with `headless: true` in `config.yaml`.

This approach had a subtle performance issue with the web preview: when running pre-login, Python's GC falls behind allocations under scheduler/CPU conditions that differ from an active session, and the preview pipeline leaks objects. The auto-login path above sidesteps this entirely by running the app inside a normal user session.

To install the service anyway:

```bash
mkdir -p ~/.config/systemd/user
cp bin/gl-simple.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable gl-simple.service
sudo loginctl enable-linger led    # so it starts at boot before login
```

Then reboot. Check with `systemctl --user status gl-simple` and `journalctl --user -u gl-simple -b`.

The unit currently has `Conflicts=graphical-session.target` commented out — re-enable it if you want the service to auto-stop when you log in.

### Why the service waits for the audio graph

Even with linger enabled, `pipewire.service`, `pipewire-pulse.service`, and `wireplumber.service` are socket-activated and take time to enumerate ALSA cards. The helper `bin/wait_for_audio.sh` explicitly starts them and polls `pactl info` (up to 25 s) for a real (non-`auto_null`) default sink, force-setting it via `pactl set-default-sink` if needed, unmuting hardware mixers, and setting volume to 100%.

## Troubleshooting

- **App doesn't appear on boot:** check Cosmic's session log: `journalctl --user -b | grep -iE "autostart|gl-simple|xterm"`. Confirm the `.desktop` file has `chmod +x` isn't required for `.desktop` files, but the Exec binary (`xterm`) must be on PATH.
- **Auto-login doesn't happen:** `sudo journalctl -u greetd -b` — look for errors parsing the `[initial_session]` block.
- **Web preview triggers progressive FPS drop:** avoid leaving the preview tab open for long sessions; the control panel (`/`) doesn't have this issue.
