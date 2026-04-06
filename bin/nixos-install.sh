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
# After reboot it will have the static IP 192.168.68.144 on ethernet.
# The dev machine must be on the same network (or use WiFi — see below).
STATIC_IP="192.168.68.144"
echo ""
echo "==> Install complete. Waiting for reboot at $STATIC_IP..."
echo "    (If not on the 192.168.68.0/24 network, connect a monitor and"
echo "     use nmcli to connect WiFi, then Ctrl+C and SCP the repo manually.)"
sleep 10

ssh-keygen -R "$STATIC_IP" 2>/dev/null || true
FINAL_IP="$STATIC_IP"

for i in $(seq 1 60); do
  if ssh $SSH_OPTS -o ConnectTimeout=5 "root@$FINAL_IP" true 2>/dev/null; then
    echo "==> NixOS is up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "==> Machine not reachable at $STATIC_IP."
    echo "    Check the console for the WiFi IP."
    read -rp "==> Enter the IP address: " FINAL_IP
    ssh-keygen -R "$FINAL_IP" 2>/dev/null || true
  fi
  sleep 5
done

# Copy the repo to the target
echo "==> Copying repo to /home/lucifera/GL_Simple..."
REMOTE_URL=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || echo "https://github.com/ddehling/GL_Simple.git")
ssh $SSH_OPTS "root@$FINAL_IP" "mkdir -p /home/lucifera"
scp -r $SSH_OPTS "$REPO_DIR" "root@$FINAL_IP:/home/lucifera/GL_Simple"
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
