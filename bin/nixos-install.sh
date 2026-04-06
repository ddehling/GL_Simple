#!/usr/bin/env bash
# Install NixOS on a Beelink Mini S via nixos-anywhere.
#
# Assumes:
#   - Nix is installed on this (dev) machine
#   - The target is running any Linux with SSH enabled for root
#   - The target has an ethernet connection
#
# Usage:
#   ./bin/nixos-install.sh root@192.168.124.123
#   ./bin/nixos-install.sh root@192.168.124.123 /dev/nvme0n1

set -euo pipefail

TARGET="${1:?Usage: $0 <user@host> [disk-device]}"
DISK="${2:-/dev/sda}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FLAKE_DIR="$REPO_DIR/nixos"

echo "==> Target:  $TARGET"
echo "==> Disk:    $DISK"
echo "==> Flake:   $FLAKE_DIR#lucifera"
echo ""

# Phase 1: kexec into the NixOS installer
echo "==> Phase 1: kexec into NixOS installer..."
nix run github:nix-community/nixos-anywhere -- \
  --phases kexec \
  --target-host "$TARGET"

# Wait for the installer to come up
echo "==> Waiting for installer to boot..."
HOST="${TARGET#*@}"
for i in $(seq 1 30); do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     "root@$HOST" true 2>/dev/null; then
    echo "==> Installer is up."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Timed out waiting for installer. Check the target machine."
    exit 1
  fi
  sleep 5
done

# Wipe the disk so disko can partition cleanly
echo "==> Wiping disk $DISK..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$HOST" bash -s <<WIPE
  set -euo pipefail
  # Unmount anything left over
  umount /mnt/boot 2>/dev/null || true
  umount /mnt 2>/dev/null || true
  swapoff -a 2>/dev/null || true
  # Wipe partition signatures then the disk itself
  for part in \$(lsblk -ln -o NAME "$DISK" | tail -n +2); do
    wipefs -af "/dev/\$part" 2>/dev/null || true
  done
  wipefs -af "$DISK"
  sgdisk --zap-all "$DISK"
  echo "Disk wiped."
WIPE

# Phase 2: install NixOS
echo "==> Phase 2: Installing NixOS..."
nix run github:nix-community/nixos-anywhere -- \
  --phases install \
  --generate-hardware-config nixos-facter "$FLAKE_DIR/hosts/facter.json" \
  --flake "$FLAKE_DIR#lucifera" \
  --target-host "root@$HOST"

echo ""
echo "==> Done! The machine should reboot into NixOS."
echo "    Commit the generated facter.json:"
echo "      git add nixos/hosts/facter.json && git commit -m 'Add hardware facter report'"
echo ""
echo "    After reboot, connect via:"
echo "      ssh lucifera@$HOST"
echo "    (IP will change to 192.168.68.144 once on the production network)"
