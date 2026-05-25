#!/usr/bin/env bash
# Configure GL_Simple to boot straight to the Cosmic desktop and run
# automatically on Pop!_OS.
#
# What it does:
#   1. Verifies we're on Pop!_OS with greetd + Cosmic (warns + exits
#      otherwise; other distros need different display-manager config).
#   2. Installs xterm (the autostart launches the app inside an xterm
#      window so Ctrl+C / tracebacks behave normally).
#   3. Adds the current user to audio, render, video groups.
#   4. Appends an [initial_session] block to greetd's config to enable
#      auto-login to Cosmic (no greeter screen).
#   5. Writes ~/.config/autostart/gl-simple.desktop pointing at
#      bin/run.sh, which pulls latest engine + projects then launches.
#
# Requires sudo for steps 2-4. Idempotent: re-running is safe and
# skips already-applied steps. Reboot after to see the full effect.
#
# Reference: docs/AUTOSTART.md

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_DIR="$(pwd)"
USER_NAME="$(whoami)"
GREETD_CONF=/etc/greetd/cosmic-greeter.toml
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/gl-simple.desktop"

echo "====================================="
echo "  GL_Simple Autostart Installer"
echo "====================================="
echo "  user:    $USER_NAME"
echo "  repo:    $REPO_DIR"
echo ""

# -----------------------------------------------------------------------
# 1. Sanity check: Pop!_OS with greetd + Cosmic
# -----------------------------------------------------------------------
if [ ! -f /etc/os-release ] || ! grep -q -i "pop" /etc/os-release; then
    echo "WARNING: this machine doesn't look like Pop!_OS." >&2
    echo "         /etc/os-release:" >&2
    grep -E '^(NAME|VERSION|ID)=' /etc/os-release 2>/dev/null | sed 's/^/           /' >&2
    echo ""
    echo "         The autostart recipe in docs/AUTOSTART.md is specific to" >&2
    echo "         Pop!_OS's greetd + Cosmic stack. Other distros need" >&2
    echo "         different display-manager config (gdm3, lightdm, sddm)." >&2
    echo "         Continuing anyway - you'll need to verify each step." >&2
    echo ""
fi

if [ ! -f "$GREETD_CONF" ]; then
    echo "ERROR: $GREETD_CONF not found. Expected greetd as the display" >&2
    echo "       manager. Abort - see docs/AUTOSTART.md for alternatives." >&2
    exit 1
fi

if ! command -v start-cosmic >/dev/null 2>&1; then
    echo "WARNING: /usr/bin/start-cosmic not found. Auto-login will fail" >&2
    echo "         unless Cosmic is installed. Continuing anyway." >&2
fi

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
# 4. Auto-login via greetd [initial_session]
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# 5. XDG autostart entry pointing at bin/run.sh
# -----------------------------------------------------------------------
echo "[4/5] Writing $DESKTOP_FILE ..."
mkdir -p "$AUTOSTART_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=GL_Simple
Comment=OpenGL lighting controller - starts with the graphical session
Exec=xterm -T GL_Simple -geometry 120x30+50+50 -hold -e /bin/bash $REPO_DIR/bin/run.sh
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
echo "  - On boot: greetd auto-logs into Cosmic, the .desktop autostart"
echo "    opens an xterm running bin/run.sh, which pulls latest project"
echo "    code (offline-tolerant) and launches Stories_OGL.py."
echo ""
echo "To skip the autostart for one boot (dev mode):"
echo "  mv $DESKTOP_FILE $DESKTOP_FILE.disabled && sudo reboot"
echo "  (rename back when you're done)"
echo ""
echo "To disable auto-login: remove the [initial_session] block from"
echo "  $GREETD_CONF and reboot."
echo ""
echo "Reference: docs/AUTOSTART.md"
