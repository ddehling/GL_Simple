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
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

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
echo ""
echo "==> The installer has booted, but it likely has a new IP (DHCP)."
echo "    Check the target's console and run: ip addr"
echo ""
read -rp "==> Enter the installer's IP address: " INSTALLER_IP

ssh-keygen -R "$INSTALLER_IP" 2>/dev/null || true

# Wait for SSH on the installer
echo "==> Waiting for installer SSH at $INSTALLER_IP..."
for i in $(seq 1 30); do
  if ssh $SSH_OPTS -o ConnectTimeout=5 "root@$INSTALLER_IP" true 2>/dev/null; then
    echo "==> Installer is up."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Timed out waiting for installer. Check the target machine."
    exit 1
  fi
  sleep 5
done

# Wipe the disk and set up swap so the installer has enough memory
# (8GB RAM isn't enough for the full nix store cache in tmpfs)
echo "==> Wiping disk and creating temp swap..."
ssh $SSH_OPTS "root@$INSTALLER_IP" bash -s <<WIPE
  set -euo pipefail
  umount /mnt/boot 2>/dev/null || true
  umount /mnt 2>/dev/null || true
  swapoff -a 2>/dev/null || true
  for part in \$(lsblk -ln -o NAME "$DISK" | tail -n +2); do
    wipefs -af "/dev/\$part" 2>/dev/null || true
  done
  wipefs -af "$DISK"
  sgdisk --zap-all "$DISK"

  # Create temp swap + nix cache partitions so the installer has enough space.
  # Disko will zap and repartition the disk during install — these are temporary.
  sgdisk -n 1:0:+4G -t 1:8200 "$DISK"
  sgdisk -n 2:0:+2G "$DISK"
  partprobe "$DISK"
  sleep 2

  mkswap "${DISK}1"
  swapon "${DISK}1"

  # Expand tmpfs to use RAM + swap
  mount -o remount,size=12G /

  # Move nix binary cache to real disk so it doesn't fill tmpfs
  mkfs.ext4 -F "${DISK}2"
  mkdir -p /tmp/nixcache
  mount "${DISK}2" /tmp/nixcache
  rm -rf /root/.cache/nix
  mkdir -p /root/.cache
  ln -s /tmp/nixcache /root/.cache/nix

  echo "Disk wiped. Temp swap + cache active."
  free -h
  df -h /tmp/nixcache
WIPE

# Clean up temp partitions before nixos-anywhere runs disko
echo "==> Cleaning up temp partitions..."
ssh $SSH_OPTS "root@$INSTALLER_IP" bash -s <<CLEANUP
  set -euo pipefail
  umount /tmp/nixcache 2>/dev/null || true
  swapoff "${DISK}1" 2>/dev/null || true
  wipefs -af "${DISK}2" 2>/dev/null || true
  wipefs -af "${DISK}1" 2>/dev/null || true
  wipefs -af "$DISK" 2>/dev/null || true
  sgdisk --zap-all "$DISK" 2>/dev/null || true
  partprobe "$DISK"
  sleep 1
  echo "Temp partitions cleaned up."
CLEANUP

# Phase 2: install NixOS
# --build-on local: build on dev machine, send only the final closure (saves RAM)
# --no-disko-deps: don't store disko deps in RAM (8GB Beelink is tight)
echo "==> Phase 2: Installing NixOS..."
nix run github:nix-community/nixos-anywhere -- \
  --phases install \
  --build-on local \
  --no-disko-deps \
  --generate-hardware-config nixos-facter "$FLAKE_DIR/hosts/facter.json" \
  --flake "$FLAKE_DIR#lucifera" \
  --target-host "root@$INSTALLER_IP"

# Wait for the machine to reboot into NixOS
echo ""
echo "==> Install complete. The machine is rebooting into NixOS."
echo "    It will come up with a new IP (DHCP on the current network)."
echo ""
read -rp "==> Enter the rebooted machine's IP address: " FINAL_IP

ssh-keygen -R "$FINAL_IP" 2>/dev/null || true

echo "==> Waiting for NixOS to come up at $FINAL_IP..."
for i in $(seq 1 60); do
  if ssh $SSH_OPTS -o ConnectTimeout=5 "root@$FINAL_IP" true 2>/dev/null; then
    echo "==> NixOS is up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: Timed out waiting for NixOS. Check the target machine."
    exit 1
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
