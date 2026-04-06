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
| **Network** | Gigabit Ethernet (`enp1s0`), WiFi 5, Bluetooth 4.0 |
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
- Auto-restarts on crash (systemd `Restart=on-failure`, 5s delay)
- Static IP `192.168.68.144` on ethernet (offline production LAN)
- WiFi available for internet access (configured per-machine via NetworkManager)
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
AC Power On (power strip plugged in)
  → BIOS auto-boot (State After G3 = S0)
    → NixOS boots
      → Auto-login on TTY (lucifera)
        → systemd starts gl-simple.service
          → GL_Simple renders headless immediately
            → DMX/sACN packets flow once network gear boots (~10-20s)
              → Lights come on
```

All devices (computer, router, switch, DMX controllers) can be on the same power strip. The computer boots fastest and starts rendering immediately. DMX output is UDP fire-and-forget — packets silently go nowhere until the network gear is ready, then lights come on automatically. No race conditions.

## Networking

### Dual-network setup

| Interface | Network | Config | Purpose |
|-----------|---------|--------|---------|
| **Ethernet** (`enp1s0`) | `192.168.68.0/24` | Static IP `192.168.68.144`, no gateway | Offline production LAN (DMX controllers, web panel) |
| **WiFi** | Home/dev network | DHCP via NetworkManager | Internet access for updates, git pull |

The ethernet interface has no default gateway — the offline production network doesn't route to the internet. When WiFi is connected, NetworkManager provides the default route, so `git pull` and internet access work alongside the static ethernet IP.

**Connect WiFi (per-machine, persists across reboots):**
```bash
nmcli device wifi connect "YourSSID" password "YourPassword"
```

### Network details

| Setting | Value |
|---------|-------|
| **Hostname** | lucifera |
| **Static IP** | 192.168.68.144 |
| **Subnet** | 192.168.68.0/24 |
| **SSH** | `ssh lucifera@192.168.68.144` (or WiFi IP) |
| **Web Control Panel** | `http://192.168.68.144:80` |
| **Web Preview** | `http://192.168.68.144/preview` |

### Default credentials

| User | Password |
|------|----------|
| `lucifera` | `lucifera` |
| `root` | `lucifera` |

Change with `passwd` after first login. These are offline lighting controllers — security is not a concern.

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

### Installing / Reimaging

The install script handles everything: kexec into NixOS installer, wipe disk, partition, install, copy the repo to the target.

**Prerequisites:**
- Nix installed on your dev machine
- SSH access to `root@<target-ip>` (root password is `lucifera` on already-imaged machines)
- Target connected via ethernet to your network

**On a fresh machine (Pop!_OS or similar), enable SSH first:**

1. Install and start SSH if not already running:
   ```bash
   sudo apt install openssh-server
   sudo systemctl enable --now ssh
   ```

2. Set a root password and enable root login:
   ```bash
   sudo passwd root
   sudo sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
   sudo systemctl restart ssh
   ```

**Deploy:**
```bash
./bin/nixos-install.sh root@<ip>
```

The script:
1. Boots the target into a NixOS installer via kexec
2. Wipes the disk (clears stale partition tables)
3. Partitions with disko, generates hardware report, installs NixOS
4. Copies the GL_Simple repo to `/home/lucifera/GL_Simple` with the correct GitHub remote

After install, commit the generated `facter.json` to the repo.

**Reimaging an already-installed machine** is the same command — root SSH is enabled and the password is `lucifera`.

<details>
<summary>Manual deploy (if you prefer)</summary>

```bash
# Phase 1: kexec into installer
nix run github:nix-community/nixos-anywhere -- \
  --phases kexec \
  --target-host root@<ip>

# Wipe the disk (SSH into the installer)
ssh root@<installer-ip> "wipefs -af /dev/sda && sgdisk --zap-all /dev/sda"

# Phase 2: install
nix run github:nix-community/nixos-anywhere -- \
  --phases install \
  --generate-hardware-config nixos-facter ./nixos/hosts/facter.json \
  --flake ./nixos#lucifera \
  --target-host root@<installer-ip>
```

</details>

### Applying Config Changes

After editing the Nix config:

```bash
cd /home/lucifera/GL_Simple/nixos
sudo nixos-rebuild switch --flake .#lucifera
```

### Updating GL_Simple Code

Connect to WiFi if not already, then:

```bash
cd /home/lucifera/GL_Simple
git pull origin main
sudo systemctl restart gl-simple.service
```

## TODO

- [x] Enable root SSH on Pop!_OS so nixos-anywhere can connect
- [x] Verify disk device path (`/dev/sda`) on the Beelink
- [ ] Run nixos-anywhere to install NixOS
- [ ] Commit generated `facter.json` to the repo
- [ ] Verify all Python packages resolve in nixpkgs (some may need overrides)
- [ ] Test headless OpenGL under NixOS (Mesa/EGL driver config)
- [ ] Consider a watchdog or health-check endpoint on the web server
