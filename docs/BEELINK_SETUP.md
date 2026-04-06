# Beelink Mini S — Dedicated GL_Simple Machine

## Hardware Specs

| Component | Details |
|-----------|---------|
| **Model** | Beelink Mini S |
| **CPU** | Intel Celeron N5095 (Jasper Lake), 4C/4T, 2.0-2.9 GHz, 10nm |
| **GPU** | Intel UHD Graphics (integrated) |
| **RAM** | 8GB DDR4 (max 16GB @ 2933 MHz) |
| **Storage** | M.2 SATA 2280 SSD (128GB or 256GB) |
| **Display** | Dual HDMI, 4K UHD |
| **Network** | Gigabit Ethernet, WiFi 5, Bluetooth 4.0 |
| **OS** | NixOS 25.05 |
| **BIOS** | AMI Aptio Setup Utility |

## BIOS Configuration

### Auto Power-On (Restore on AC Power Loss)

This makes the machine boot automatically whenever AC power is applied — no power button press needed.

1. Press **Delete** repeatedly during boot to enter the Aptio Setup Utility
2. Navigate to **Chipset** > **PCH-IO Configuration**
3. Set **"State After G3"** to **"S0 State"**
4. Press **F4** to save and exit

**What the values mean:**
- **G3** = Total AC power loss (wall power removed)
- **S0** = Full power on — machine boots immediately when power is restored
- **S5** = Soft off (default) — machine stays off, waits for power button

**To test:** Shut down, unplug the power adapter, plug it back in. The machine should boot on its own.

## Operating Modes

The machine operates in two modes:

### Production Mode (default, headless)

The normal operating mode. No monitor connected. GL_Simple runs automatically on boot.

- Auto-login to `lucifera` on TTY
- GL_Simple starts as a systemd service (headless)
- Auto-restarts on crash (systemd `Restart=on-failure`)
- Static IP on the local network
- COSMIC desktop is installed but not started
- No internet required for normal operation

### Development Mode

Occasionally needed for debugging, writing code, or pulling updates. Connect a monitor and keyboard.

```bash
sudo systemctl start display-manager   # start COSMIC desktop
sudo systemctl stop gl-simple.service   # stop auto-running instance
# ... work interactively ...
sudo systemctl start gl-simple.service  # resume production mode
sudo systemctl stop display-manager     # back to headless (optional)
```

## Boot Sequence (Production Mode)

```
AC Power On
  → BIOS auto-boot (State After G3 = S0)
    → NixOS boots
      → Auto-login on TTY (lucifera)
        → systemd starts gl-simple.service
          → GL_Simple runs headless
```

## NixOS Configuration

All system configuration lives in `nixos/` within this repo:

```
nixos/
├── flake.nix                      # Entry point — NixOS 25.05 + disko + COSMIC + nixos-facter
├── disk-config.nix                # Declarative disk layout (GPT, ESP + ext4 root)
├── hosts/
│   ├── lucifera.nix               # Full machine config (network, users, services)
│   └── facter.json                # Auto-generated hardware report (from nixos-facter)
└── modules/
    └── gl-simple.nix              # GL_Simple systemd service + Python deps
```

Everything — disk partitioning, hardware detection, static IP, SSH, auto-login, firewall, the GL_Simple service, COSMIC desktop — is declarative in the Nix config. Rebuilding or replacing the machine is a single command.

### Tools

- **[disko](https://github.com/nix-community/disko)** — declarative disk partitioning (no manual fdisk/mkfs)
- **[nixos-facter](https://github.com/nix-community/nixos-facter)** — auto-detects hardware, replaces hand-written hardware-configuration.nix
- **[nixos-anywhere](https://github.com/nix-community/nixos-anywhere)** — remote NixOS install over SSH (works on any running Linux)

### Initial Install (from your dev machine)

The Beelink is currently running Pop!_OS with SSH enabled. nixos-anywhere will SSH in, kexec into a NixOS installer, partition the disk with disko, detect hardware with nixos-facter, and apply the full config — all remotely.

**Prerequisites:**
- Nix installed on your dev machine
- SSH access to `root@192.168.68.144` on the Beelink (the Pop!_OS install)
- Beelink connected via ethernet (not WiFi)

**Prepare the Beelink (one-time, on the Pop!_OS install):**

1. Install and start SSH if not already running:
   ```bash
   sudo apt install openssh-server
   sudo systemctl enable --now ssh
   ```

2. Set a root password (needed for nixos-anywhere to SSH in as root):
   ```bash
   sudo passwd root
   ```

3. Enable root SSH login in `/etc/ssh/sshd_config`:
   ```bash
   sudo sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
   sudo systemctl restart ssh
   ```

4. Verify from your dev machine:
   ```bash
   ssh root@192.168.68.144
   ```

5. Check the disk device path (needed for `disk-config.nix`):
   ```bash
   lsblk
   ```
   If the disk is `/dev/sda`, no changes needed. If it's `/dev/nvme0n1`, update `disk-config.nix` accordingly.

**Deploy:**
```bash
nix run github:nix-community/nixos-anywhere -- \
  --generate-hardware-config nixos-facter ./nixos/hosts/facter.json \
  --flake ./nixos#lucifera \
  --target-host root@192.168.68.144
```

After install, commit the generated `facter.json` to the repo so future rebuilds have the hardware report.

### Applying Config Changes

After editing the Nix config:

```bash
cd /home/lucifera/GL_Simple/nixos
sudo nixos-rebuild switch --flake .#lucifera
```

### Updating GL_Simple Code

Since the machine is usually offline, pull manually when internet is available:

```bash
cd /home/lucifera/GL_Simple
git pull origin main
sudo systemctl restart gl-simple.service
```

## Network Details

| Setting | Value |
|---------|-------|
| **Hostname** | lucifera |
| **Static IP** | 192.168.68.144 |
| **Subnet** | 192.168.68.0/24 |
| **Gateway** | 192.168.68.1 |
| **SSH** | `ssh lucifera@192.168.68.144` |
| **Web Control Panel** | `http://192.168.68.144:80` |
| **Web Preview** | `http://192.168.68.144/preview` |

## TODO

- [ ] Enable root SSH on Pop!_OS so nixos-anywhere can connect
- [ ] Verify disk device path (`/dev/sda` vs `/dev/nvme0n1`) on the Beelink
- [ ] Run nixos-anywhere to install NixOS
- [ ] Commit generated `facter.json` to the repo
- [ ] Verify all Python packages resolve in nixpkgs (some may need overrides)
- [ ] Test headless OpenGL under NixOS (Mesa/EGL driver config)
- [ ] Consider a watchdog or health-check endpoint on the web server
