# GL_Simple Autostart (Pop!_OS / Cosmic session)

Stories_OGL.py is configured to start automatically on boot by enabling **auto-login** to the Cosmic desktop, then firing a standard XDG **autostart** entry inside the session. The app runs in a visible `xterm` window so you can `Ctrl+C` to stop it like any other dev run.

## One-script install

```bash
./bin/linux-autostart.sh           # enable (default)
./bin/linux-autostart.sh disable   # turn it back off
```

Installs xterm, adds you to the `audio render video` groups, configures greetd auto-login to Cosmic, and writes `~/.config/autostart/gl-simple.desktop` pointing at `bin/linux-run.sh`. Idempotent. Reboot after. Run `./bin/linux-autostart.sh disable` to remove the autostart entry again (auto-login is left as-is). The manual recipe below is what the script automates — read it if you need to deviate (different distro, different display manager).

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
- **Skip the autostart for one session:** before reboot, rename the file:
  ```bash
  mv ~/.config/autostart/gl-simple.desktop ~/.config/autostart/gl-simple.desktop.disabled
  ```
  Reboot, do dev work, then rename it back. Or set `Hidden=true` inline.
- **Disable auto-login temporarily:** remove the `[initial_session]` block from `/etc/greetd/cosmic-greeter.toml` and reboot. Re-add to restore.

## Required group membership

`led` must be in `audio`, `render`, and `video` groups — these grant access to the audio and GPU device nodes. The auto-login path doesn't strictly need `audio` (logind grants it via ACL on session start), but the `render` and `video` groups are still recommended to guarantee GPU and device access.

```bash
sudo usermod -a -G audio,render,video led
# then reboot for the user systemd manager to pick up the new groups
```

Verify: `groups led` should include `audio render video`.

## Troubleshooting

- **App doesn't appear on boot:** check Cosmic's session log: `journalctl --user -b | grep -iE "autostart|gl-simple|xterm"`. Confirm the `.desktop` file has `chmod +x` isn't required for `.desktop` files, but the Exec binary (`xterm`) must be on PATH.
- **Auto-login doesn't happen:** `sudo journalctl -u greetd -b` — look for errors parsing the `[initial_session]` block.
- **Web preview triggers progressive FPS drop:** avoid leaving the preview tab open for long sessions; the control panel (`/`) doesn't have this issue.
