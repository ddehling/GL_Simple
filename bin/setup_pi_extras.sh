#!/usr/bin/env bash
# Raspberry Pi specific one-shot tuning for GL_Simple.
#
# What this does (idempotent - safe to re-run):
#   1. Verifies we're on a Pi (bails politely if not).
#   2. Bumps GPU memory split to 256 MB if currently lower. The
#      OpenGL ES 3.1 shaders + FBO + DMA from FBO readback need
#      headroom; the Pi OS default (76 MB) is too tight for the
#      effect stack and you get frame stalls or shader compile
#      failures.
#   3. Switches to the full KMS GL driver if currently legacy. The
#      stack uses GLES 3.1 features that the legacy driver doesn't
#      support reliably.
#   4. Installs xterm + libcap2-bin + raspi-config (typically all
#      present on Pi OS desktop, missing on lite images).
#
# After running this:
#   - sudo reboot (boot config changes take effect on reboot)
#   - then ./bin/setup.sh (same script everyone else uses)
#   - then ./bin/install_autostart.sh (now LightDM-aware)
#
# Doesn't touch GitHub, doesn't install Python deps, doesn't clone
# projects - those happen in bin/setup.sh. This file is just for
# the things that are Pi-specific AND best done before the rest.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: do not run as root. The script calls sudo internally." >&2
    exit 1
fi

echo "====================================="
echo "  GL_Simple Pi extras"
echo "====================================="
echo ""

# -----------------------------------------------------------------------
# 1. Confirm we're on a Pi
# -----------------------------------------------------------------------
MODEL=""
if [ -r /proc/device-tree/model ]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model)"
fi
if [ -z "$MODEL" ] || ! echo "$MODEL" | grep -qi "raspberry pi"; then
    echo "WARNING: this machine doesn't look like a Raspberry Pi." >&2
    echo "         /proc/device-tree/model: ${MODEL:-<not present>}" >&2
    echo "" >&2
    echo "         Pi-specific tweaks (GPU memory split, GL driver mode)" >&2
    echo "         only make sense on a Pi. If you're on a generic Linux" >&2
    echo "         machine, just run ./bin/setup.sh directly." >&2
    exit 1
fi
echo "[1/4] Detected: $MODEL"

# -----------------------------------------------------------------------
# 2. Pi tools (xterm, libcap2-bin for setcap, raspi-config)
# -----------------------------------------------------------------------
NEED_PKGS=()
for pkg in xterm libcap2-bin raspi-config; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        NEED_PKGS+=("$pkg")
    fi
done
if [ ${#NEED_PKGS[@]} -gt 0 ]; then
    echo "[2/4] Installing: ${NEED_PKGS[*]}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${NEED_PKGS[@]}"
else
    echo "[2/4] xterm, libcap2-bin, raspi-config all present"
fi

# -----------------------------------------------------------------------
# 3. GPU memory split
# -----------------------------------------------------------------------
# raspi-config's nonint API for get_config_var reads /boot/config.txt.
# Default on Pi OS is 76 MB which is tight for our shader stack. Bump
# to 256 MB. Only changes if currently lower so re-runs are no-ops.
CURRENT_GPU_MEM="$(sudo raspi-config nonint get_config_var gpu_mem /boot/firmware/config.txt 2>/dev/null \
                  || sudo raspi-config nonint get_config_var gpu_mem /boot/config.txt 2>/dev/null \
                  || echo 76)"
TARGET_GPU_MEM=256
if [ "${CURRENT_GPU_MEM:-76}" -ge "$TARGET_GPU_MEM" ]; then
    echo "[3/4] gpu_mem already at ${CURRENT_GPU_MEM} MB (>= ${TARGET_GPU_MEM})"
else
    echo "[3/4] Bumping gpu_mem ${CURRENT_GPU_MEM} -> ${TARGET_GPU_MEM} MB..."
    sudo raspi-config nonint do_memory_split "$TARGET_GPU_MEM"
fi

# -----------------------------------------------------------------------
# 4. Full KMS GL driver
# -----------------------------------------------------------------------
# raspi-config "Advanced > GL Driver" options:
#   G1 = legacy non-GL desktop driver
#   G2 = GL (Fake KMS)        [deprecated on Bookworm]
#   G3 = GL (Full KMS)        [what we want; default on Pi OS Bookworm]
# Pi OS Bookworm uses KMS by default but older releases (Bullseye)
# may still be on legacy. do_gldriver G3 is idempotent.
echo "[4/4] Ensuring Full KMS GL driver is selected..."
# Older raspi-config exposes do_gldriver; newer Pi OS (Bookworm)
# dropped the menu but still ships KMS by default - the nonint call
# fails harmlessly on those. We don't gate on success.
sudo raspi-config nonint do_gldriver G3 2>/dev/null \
    || echo "      (do_gldriver not available - assuming KMS is already default)"

echo ""
echo "====================================="
echo "  Pi extras complete."
echo "====================================="
echo ""
echo "Next steps:"
echo "  1. sudo reboot           # apply boot/config.txt changes"
echo "  2. ./bin/setup.sh        # install rest of the system + projects"
echo "  3. ./bin/install_autostart.sh   # boot-to-app autostart"
