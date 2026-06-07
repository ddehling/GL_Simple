#!/usr/bin/env bash
# Manage GL_Simple's boot-time autostart on Pop!_OS and Raspberry Pi OS.
# Run it with NO arguments - it asks whether to enable or disable.
#
#   enable  - configure auto-login + write the XDG autostart entry so the
#             app launches on boot.
#   disable - remove that entry so it no longer launches (auto-login is
#             left as-is).
#
# What `enable` does:
#   1. Auto-detects the display manager (greetd/Cosmic on Pop!_OS,
#      gdm3 on Pop!_OS LTS/Ubuntu, or LightDM on Raspberry Pi OS) and
#      exits only if none of them is found.
#   2. Installs xterm (the autostart launches the app inside an xterm
#      window so Ctrl+C / tracebacks behave normally).
#   3. Adds the current user to audio, render, video groups.
#   4. Enables passwordless auto-login for the current user via that
#      display manager (greetd, GDM, or LightDM / raspi-config on a Pi).
#   5. Writes ~/.config/autostart/gl-simple.desktop pointing at
#      bin/linux-run.sh, which pulls latest engine + projects then launches.
#
# Requires sudo for steps 2-4. Idempotent: re-running is safe and
# skips already-applied steps. Reboot after to see the full effect.
#
# Reference: docs/AUTOSTART.md

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_DIR="$(pwd)"
USER_NAME="$(whoami)"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/gl-simple.desktop"

# Refuse to run as root - we use 'sudo' explicitly for the few steps
# that need it. Running the whole script as root would write the XDG
# autostart entry to /root/.config/ and set AutomaticLogin=root,
# neither of which is what you want. SUDO_USER is set when invoked
# via sudo (it holds the original user's name); use that as the hint.
if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: do not run this script with sudo / as root." >&2
    if [ -n "${SUDO_USER:-}" ]; then
        echo "       Re-run as $SUDO_USER (no sudo prefix):" >&2
        echo "           ./bin/linux-autostart.sh" >&2
    else
        echo "       Re-run as the regular user who will use this machine:" >&2
        echo "           ./bin/linux-autostart.sh" >&2
    fi
    echo "       The script calls sudo internally for the few steps that need it." >&2
    exit 1
fi

# Ask what to do - no flags or arguments, just run the script and pick.
echo "====================================="
echo "  GL_Simple Autostart"
echo "====================================="
echo "  1) enable   - launch the app automatically on boot"
echo "  2) disable  - stop launching it on boot (auto-login left as-is)"
echo ""
MODE=""
while [ -z "$MODE" ]; do
    read -rp "Choose 1 or 2: " choice
    case "$choice" in
        1|enable)  MODE="enable" ;;
        2|disable) MODE="disable" ;;
        *)         echo "  Please enter 1 or 2." ;;
    esac
done
echo ""

# disable: remove the user's XDG autostart entry (no sudo / display manager
# needed). Auto-login is intentionally left untouched.
if [ "$MODE" = "disable" ]; then
    echo "Disabling autostart..."
    removed=false
    for f in "$DESKTOP_FILE" "$DESKTOP_FILE.disabled"; do
        if [ -e "$f" ]; then
            rm -f "$f"
            echo "  removed $f"
            removed=true
        fi
    done
    $removed || echo "  nothing to remove - no autostart entry at $DESKTOP_FILE"
    echo ""
    echo "The app will no longer launch on boot. Desktop auto-login is left"
    echo "unchanged. Run this script again and choose 'enable' to turn it back on."
    exit 0
fi

# ----- enable: configure auto-login + write the autostart entry -----
GREETD_CONF=/etc/greetd/cosmic-greeter.toml
GDM_CONF=/etc/gdm3/custom.conf
GDM_CONF_ALT=/etc/gdm/custom.conf
LIGHTDM_CONF=/etc/lightdm/lightdm.conf

# Detect which display manager owns auto-login on this machine.
# Preference order: greetd (Pop!_OS Cosmic) -> gdm3/gdm (Pop!_OS LTS,
# Ubuntu, Fedora workstation) -> lightdm (Raspberry Pi OS, Lubuntu,
# Mint MATE/XFCE).
DM=""
if [ -f "$GREETD_CONF" ]; then
    DM="greetd"
elif [ -f "$GDM_CONF" ]; then
    DM="gdm3"
elif [ -f "$GDM_CONF_ALT" ]; then
    DM="gdm"
    GDM_CONF="$GDM_CONF_ALT"
elif [ -f "$LIGHTDM_CONF" ] || command -v lightdm >/dev/null 2>&1; then
    DM="lightdm"
fi

echo "Enabling autostart..."
echo "  user:    $USER_NAME"
echo "  repo:    $REPO_DIR"
echo ""

# -----------------------------------------------------------------------
# 1. Display-manager detection
# -----------------------------------------------------------------------
case "$DM" in
    greetd)
        echo "  display-manager: greetd (Cosmic)"
        if ! command -v start-cosmic >/dev/null 2>&1; then
            echo "WARNING: /usr/bin/start-cosmic not found. Auto-login will fail" >&2
            echo "         unless Cosmic is installed. Continuing anyway." >&2
        fi
        ;;
    gdm3|gdm)
        echo "  display-manager: $DM ($GDM_CONF)"
        ;;
    lightdm)
        echo "  display-manager: lightdm ($LIGHTDM_CONF)"
        ;;
    *)
        echo "ERROR: no supported display manager detected." >&2
        echo "       Looked for: $GREETD_CONF, $GDM_CONF, $GDM_CONF_ALT, $LIGHTDM_CONF" >&2
        echo "       Supported: greetd (Pop!_OS Cosmic), gdm3 (Pop!_OS LTS, Ubuntu)," >&2
        echo "                  lightdm (Raspberry Pi OS, Lubuntu, Mint XFCE)." >&2
        echo "       For other display managers (e.g. sddm/KDE), configure" >&2
        echo "       auto-login manually - see docs/AUTOSTART.md." >&2
        exit 1
        ;;
esac
echo ""

# -----------------------------------------------------------------------
# 2. Install xterm
# -----------------------------------------------------------------------
if ! command -v xterm >/dev/null 2>&1; then
    echo "[1/5] Installing xterm..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends xterm
else
    echo "[1/5] xterm OK"
fi

# -----------------------------------------------------------------------
# 3. Group membership
# -----------------------------------------------------------------------
NEED_GROUPS=()
for grp in audio render video; do
    if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$grp"; then
        NEED_GROUPS+=("$grp")
    fi
done
if [ ${#NEED_GROUPS[@]} -gt 0 ]; then
    echo "[2/5] Adding $USER_NAME to groups: ${NEED_GROUPS[*]}"
    sudo usermod -a -G "$(IFS=,; echo "${NEED_GROUPS[*]}")" "$USER_NAME"
    echo "      (group changes take effect on next login)"
else
    echo "[2/5] groups OK (already in audio, render, video)"
fi

# -----------------------------------------------------------------------
# 4. Auto-login (per display manager)
# -----------------------------------------------------------------------
case "$DM" in
    greetd)
        if grep -q "^\[initial_session\]" "$GREETD_CONF"; then
            echo "[3/5] greetd auto-login already configured in $GREETD_CONF"
        else
            echo "[3/5] Enabling auto-login for $USER_NAME via greetd..."
            sudo tee -a "$GREETD_CONF" > /dev/null <<EOF

[initial_session]
command = "/usr/bin/start-cosmic"
user = "$USER_NAME"
EOF
        fi
        ;;
    gdm3|gdm)
        # GDM ini parser merges multiple [daemon] sections, but to be tidy
        # we use a python helper to insert AutomaticLogin* keys into the
        # existing [daemon] section if present, or create one if not.
        if grep -qE "^AutomaticLogin\s*=\s*$USER_NAME\s*$" "$GDM_CONF" 2>/dev/null \
           && grep -qE "^AutomaticLoginEnable\s*=\s*[tT]rue\s*$" "$GDM_CONF" 2>/dev/null; then
            echo "[3/5] GDM auto-login already configured for $USER_NAME in $GDM_CONF"
        else
            echo "[3/5] Enabling GDM auto-login for $USER_NAME..."
            sudo python3 - "$GDM_CONF" "$USER_NAME" <<'PY'
import sys, re, pathlib
path = pathlib.Path(sys.argv[1])
user = sys.argv[2]
text = path.read_text() if path.exists() else ""
# Strip any existing AutomaticLogin* lines so we don't end up with duplicates.
text = re.sub(r'(?m)^\s*AutomaticLogin(?:Enable)?\s*=.*\n?', '', text)
# If [daemon] section exists, insert the two keys right after its header.
if re.search(r'(?m)^\[daemon\]\s*$', text):
    text = re.sub(
        r'(?m)^(\[daemon\]\s*)$',
        f'\\1\nAutomaticLoginEnable=true\nAutomaticLogin={user}',
        text, count=1,
    )
else:
    if text and not text.endswith('\n'):
        text += '\n'
    text += f'\n[daemon]\nAutomaticLoginEnable=true\nAutomaticLogin={user}\n'
path.write_text(text)
PY
        fi
        ;;
    lightdm)
        # On Raspberry Pi OS the cleanest way is to call raspi-config
        # if present (sets up autologin AND any related TTY config).
        # On other lightdm distros (Lubuntu, Mint XFCE) we edit
        # /etc/lightdm/lightdm.conf directly.
        if command -v raspi-config >/dev/null 2>&1; then
            # B4 = "Desktop Autologin" in raspi-config's nonint scheme.
            # Idempotent: re-running is a no-op if already set.
            echo "[3/5] Enabling LightDM auto-login via raspi-config..."
            sudo raspi-config nonint do_boot_behaviour B4
        elif grep -qE "^autologin-user\s*=\s*$USER_NAME\s*$" "$LIGHTDM_CONF" 2>/dev/null; then
            echo "[3/5] LightDM auto-login already configured for $USER_NAME"
        else
            echo "[3/5] Enabling LightDM auto-login for $USER_NAME..."
            sudo python3 - "$LIGHTDM_CONF" "$USER_NAME" <<'PY'
import sys, re, pathlib
path = pathlib.Path(sys.argv[1])
user = sys.argv[2]
text = path.read_text() if path.exists() else ""
# Strip any existing autologin-user* lines (avoid duplicates).
text = re.sub(r'(?m)^\s*autologin-user(?:-timeout)?\s*=.*\n?', '', text)
# Insert the two keys inside the [Seat:*] section. Create the
# section if it doesn't exist.
if re.search(r'(?m)^\[Seat:\*\]\s*$', text):
    text = re.sub(
        r'(?m)^(\[Seat:\*\]\s*)$',
        f'\\1\nautologin-user={user}\nautologin-user-timeout=0',
        text, count=1,
    )
else:
    if text and not text.endswith('\n'):
        text += '\n'
    text += f'\n[Seat:*]\nautologin-user={user}\nautologin-user-timeout=0\n'
path.write_text(text)
PY
            # LightDM requires the user to be in the 'autologin' or
            # 'nopasswdlogin' group on most distros, otherwise PAM
            # blocks the no-password session. Create the group if
            # missing and add the user.
            if ! getent group autologin >/dev/null 2>&1; then
                sudo groupadd -r autologin || true
            fi
            sudo usermod -aG autologin "$USER_NAME" || true
        fi
        ;;
esac

# -----------------------------------------------------------------------
# 5. XDG autostart entry pointing at bin/linux-run.sh
# -----------------------------------------------------------------------
echo "[4/5] Writing $DESKTOP_FILE ..."
mkdir -p "$AUTOSTART_DIR"
# Exec points at linux-autostart-launcher.sh (plain bash, no X dependency to
# start) rather than xterm directly. On a Wayland desktop the autostart can
# fire before XWayland is ready, and a direct xterm would fail to open a window
# and exit silently — the app would never launch. The launcher retries the
# xterm until the display is ready, then falls back to a headless launch so a
# kiosk always comes up. See bin/linux-autostart-launcher.sh.
chmod +x "$REPO_DIR/bin/linux-autostart-launcher.sh" 2>/dev/null || true
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=GL_Simple
Comment=OpenGL lighting controller - starts with the graphical session
Exec=/bin/bash $REPO_DIR/bin/linux-autostart-launcher.sh
Path=$REPO_DIR
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

# -----------------------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------------------
echo "[5/5] Done."
echo ""
echo "Next steps:"
echo "  - Reboot.  ('sudo reboot')"
echo "  - On boot: $DM auto-logs in as $USER_NAME, the .desktop autostart"
echo "    opens an xterm running bin/linux-run.sh, which pulls latest project"
echo "    code (offline-tolerant) and launches Stories_OGL.py."
echo ""
echo "To turn the autostart back off: run this script again and choose 'disable'."
echo ""
case "$DM" in
    greetd)   echo "To disable auto-login: remove [initial_session] from $GREETD_CONF and reboot." ;;
    gdm3|gdm) echo "To disable auto-login: set AutomaticLoginEnable=false in $GDM_CONF and reboot." ;;
    lightdm)  echo "To disable auto-login: 'sudo raspi-config' -> System Options -> Boot/Auto Login -> 'Desktop' (no autologin), then reboot. Or remove the autologin-user lines from $LIGHTDM_CONF." ;;
esac
echo ""
echo "Reference: docs/AUTOSTART.md"
