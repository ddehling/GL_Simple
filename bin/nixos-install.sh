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

set -euo pipefail

TARGET="${1:?Usage: $0 <user@host>}"
DISK="/dev/sda"
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
  --flake "$FLAKE_DIR#lucifera" \
  --target-host "$TARGET"

# After kexec, the installer boots with a new DHCP IP.
# Ask the user to find it from the target's console (ip addr).
echo ""
echo "==> The installer has booted, but it likely has a new IP (DHCP)."
echo "    Check the target's console and run: ip addr"
echo ""
read -rp "==> Enter the installer's IP address: " INSTALLER_IP

# Clear any stale host keys for the new IP
ssh-keygen -R "$INSTALLER_IP" 2>/dev/null || true

# Wait for SSH on the installer
echo "==> Waiting for installer SSH at $INSTALLER_IP..."
for i in $(seq 1 30); do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     "root@$INSTALLER_IP" true 2>/dev/null; then
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
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$INSTALLER_IP" bash -s <<WIPE
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

# Bundle the repo to send to the target via --extra-files
echo "==> Bundling repo for deployment..."
EXTRA_DIR=$(mktemp -d)
mkdir -p "$EXTRA_DIR/home/lucifera"
git -C "$REPO_DIR" clone --local "$REPO_DIR" "$EXTRA_DIR/home/lucifera/GL_Simple"
# Point the remote at GitHub so git pull works immediately
REMOTE_URL=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || echo "https://github.com/ddehling/GL_Simple.git")
git -C "$EXTRA_DIR/home/lucifera/GL_Simple" remote set-url origin "$REMOTE_URL"
# Set ownership (uid/gid 1000 = first normal user, i.e. lucifera)
chown -R 1000:1000 "$EXTRA_DIR/home/lucifera"

# Phase 2: install NixOS
echo "==> Phase 2: Installing NixOS..."
nix run github:nix-community/nixos-anywhere -- \
  --phases install \
  --extra-files "$EXTRA_DIR" \
  --generate-hardware-config nixos-facter "$FLAKE_DIR/hosts/facter.json" \
  --flake "$FLAKE_DIR#lucifera" \
  --target-host "root@$INSTALLER_IP"

# Clean up
rm -rf "$EXTRA_DIR"

echo ""
echo "==> Done! The machine should reboot into NixOS."
echo "    The repo has been copied to /home/lucifera/GL_Simple on the target."
echo ""
echo "    Commit the generated facter.json:"
echo "      git add nixos/hosts/facter.json && git commit -m 'Add hardware facter report'"
echo ""
echo "    After reboot, connect via:"
echo "      ssh lucifera@<ip>"
echo "    (IP will be 192.168.68.144 on the production network)"
