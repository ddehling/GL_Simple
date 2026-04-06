#!/usr/bin/env bash
# Install NixOS on a Beelink Mini S via nixos-anywhere.
#
# Prerequisites:
#   - Nix installed on this (dev) machine
#   - Target booted from NixOS minimal ISO (USB stick)
#   - Root password set on the installer (run: passwd)
#   - Target connected via ethernet
#
# Usage:
#   ./bin/nixos-install.sh root@192.168.124.123

set -euo pipefail

TARGET="${1:?Usage: $0 <user@host>}"
DISK="/dev/sda"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FLAKE_DIR="$REPO_DIR/nixos"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
HOST="${TARGET#*@}"

echo "==> Target:  $TARGET"
echo "==> Disk:    $DISK"
echo "==> Flake:   $FLAKE_DIR#lucifera"
echo ""

# Verify the target is a NixOS installer
echo "==> Verifying target is a NixOS installer..."
if ! ssh $SSH_OPTS "$TARGET" "grep -q 'VARIANT_ID=installer' /etc/os-release 2>/dev/null"; then
  echo "ERROR: Target does not appear to be a NixOS installer."
  echo "       Boot the machine from a NixOS minimal ISO USB first."
  exit 1
fi
echo "==> NixOS installer detected."

# Wipe the disk so disko can partition cleanly
echo "==> Wiping disk..."
ssh $SSH_OPTS "$TARGET" bash -s <<WIPE
  set -euo pipefail
  umount /mnt/boot 2>/dev/null || true
  umount /mnt 2>/dev/null || true
  swapoff -a 2>/dev/null || true
  for part in \$(lsblk -ln -o NAME "$DISK" | tail -n +2); do
    wipefs -af "/dev/\$part" 2>/dev/null || true
  done
  wipefs -af "$DISK"
  sgdisk --zap-all "$DISK"
  echo "Disk wiped."
WIPE

# Install NixOS
# nixos-anywhere detects the USB installer and skips kexec automatically.
# --build-on local: build on dev machine, send only the final closure
# --no-disko-deps: installer already has partitioning tools
echo "==> Installing NixOS..."
nix run github:nix-community/nixos-anywhere -- \
  --build-on local \
  --no-disko-deps \
  --generate-hardware-config nixos-facter "$FLAKE_DIR/hosts/facter.json" \
  --flake "$FLAKE_DIR#lucifera" \
  --target-host "$TARGET"

# Wait for the machine to reboot into NixOS.
# After reboot, ethernet has static IP 192.168.68.144 (production LAN).
# If the dev machine isn't on that network, connect WiFi on the target.
echo ""
echo "==> Install complete! The machine is rebooting."
echo ""
echo "    On the target console:"
echo "      1. Wait for boot (enp1s0 timeout is normal if not on production LAN)"
echo "      2. Log in as lucifera / lucifera"
echo "      3. Connect WiFi:  nmcli device wifi connect \"SSID\" password \"PASS\""
echo "      4. Get the IP:    ip addr"
echo ""
read -rp "==> Enter the target's reachable IP address: " FINAL_IP

ssh-keygen -R "$FINAL_IP" 2>/dev/null || true

echo "==> Waiting for NixOS SSH at $FINAL_IP..."
for i in $(seq 1 60); do
  if ssh $SSH_OPTS -o ConnectTimeout=5 "root@$FINAL_IP" true 2>/dev/null; then
    echo "==> NixOS is up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: Timed out waiting for NixOS."
    exit 1
  fi
  sleep 5
done

# Copy the repo to the target
echo "==> Copying repo to /home/lucifera/GL_Simple..."
REMOTE_URL=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || echo "https://github.com/ddehling/GL_Simple.git")
ssh $SSH_OPTS "root@$FINAL_IP" "mkdir -p /home/lucifera/GL_Simple"
rsync -az --progress --exclude='venv' --exclude='__pycache__' \
  -e "ssh $SSH_OPTS" \
  "$REPO_DIR/" "root@$FINAL_IP:/home/lucifera/GL_Simple/"
ssh $SSH_OPTS "root@$FINAL_IP" bash -s <<SETUP
  cd /home/lucifera/GL_Simple
  git remote set-url origin "$REMOTE_URL"
  chown -R lucifera:users /home/lucifera/GL_Simple
SETUP

echo ""
echo "==> Done!"
echo "    The repo is at /home/lucifera/GL_Simple on the target."
echo ""
echo "    Commit the generated facter.json:"
echo "      git add nixos/hosts/facter.json && git commit -m 'Add hardware facter report'"
echo ""
echo "    Connect via:"
echo "      ssh lucifera@$FINAL_IP"
echo "    (IP will be 192.168.68.144 on the production network)"
