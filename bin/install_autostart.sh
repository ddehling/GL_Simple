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
GDM_CONF=/etc/gdm3/custom.conf
GDM_CONF_ALT=/etc/gdm/custom.conf
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/gl-simple.desktop"

# Detect which display manager owns auto-login on this machine.
DM=""
if [ -f "$GREETD_CONF" ]; then
    DM="greetd"
elif [ -f "$GDM_CONF" ]; then
    DM="gdm3"
elif [ -f "$GDM_CONF_ALT" ]; then
    DM="gdm"
    GDM_CONF="$GDM_CONF_ALT"
fi

echo "====================================="
echo "  GL_Simple Autostart Installer"
echo "====================================="
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
    *)
        echo "ERROR: no supported display manager detected." >&2
        echo "       Looked for: $GREETD_CONF, $GDM_CONF, $GDM_CONF_ALT" >&2
        echo "       Supported: greetd (Pop!_OS Cosmic), gdm3 (Pop!_OS LTS, Ubuntu)." >&2
        echo "       For others (lightdm, sddm), see docs/AUTOSTART.md and configure" >&2
        echo "       auto-login manually, then re-run this script with --skip-login." >&2
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
esac

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
echo "  - On boot: $DM auto-logs in as $USER_NAME, the .desktop autostart"
echo "    opens an xterm running bin/run.sh, which pulls latest project"
echo "    code (offline-tolerant) and launches Stories_OGL.py."
echo ""
echo "To skip the autostart for one boot (dev mode):"
echo "  mv $DESKTOP_FILE $DESKTOP_FILE.disabled && sudo reboot"
echo "  (rename back when you're done)"
echo ""
case "$DM" in
    greetd) echo "To disable auto-login: remove [initial_session] from $GREETD_CONF and reboot." ;;
    gdm3|gdm) echo "To disable auto-login: set AutomaticLoginEnable=false in $GDM_CONF and reboot." ;;
esac
echo ""
echo "Reference: docs/AUTOSTART.md"
