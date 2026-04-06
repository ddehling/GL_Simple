{ config, pkgs, lib, ... }:

{
  imports = [
    ../modules/gl-simple.nix
  ];

  # ---------- Nix settings ----------
  nixpkgs.config.allowUnfree = true;
  nix.settings = {
    experimental-features = [ "nix-command" "flakes" ];
    trusted-users = [ "root" "lucifera" ];
  };

  # ---------- Boot ----------
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # ---------- CPU / GPU Performance ----------
  powerManagement.cpuFreqGovernor = "performance";

  # Never sleep/suspend — this machine runs a light show all night
  systemd.targets.sleep.enable = false;
  systemd.targets.suspend.enable = false;
  systemd.targets.hibernate.enable = false;
  systemd.targets.hybrid-sleep.enable = false;

  # Force Intel GPU to max frequency — default power management keeps it at
  # 300/1000 MHz which tanks shader FPS from ~40 to ~12
  systemd.services.gpu-max-freq = {
    description = "Set Intel GPU to max frequency";
    after = [ "multi-user.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
      ExecStart = "${pkgs.bash}/bin/bash -c 'echo 1000 > /sys/class/drm/card1/gt/gt0/rps_min_freq_mhz'";
    };
  };

  # ---------- Hostname ----------
  networking.hostName = "lucifera";

  # ---------- Networking ----------
  # NetworkManager handles WiFi (credentials configured per-machine at runtime).
  # Ethernet is unmanaged — static IP for the offline production network.
  networking.networkmanager = {
    enable = true;
    unmanaged = [ "enp1s0" ];
  };
  networking.interfaces.enp1s0 = {
    useDHCP = false;
    ipv4.addresses = [{
      address = "192.168.68.144";
      prefixLength = 24;
    }];
  };
  # No default gateway on ethernet — the offline production network doesn't
  # route to the internet. When WiFi is connected, NetworkManager provides
  # the default route automatically.
  networking.nameservers = [ "1.1.1.1" "8.8.8.8" ];

  # ---------- Firewall ----------
  networking.firewall = {
    enable = true;
    allowedUDPPorts = [
      5353 # mDNS (Avahi)
    ];
    allowedTCPPorts = [
      22   # SSH
      80   # GL_Simple web control panel
      5000 # GL_Simple web (fallback port)
    ];
  };

  # ---------- mDNS (lucifera.local) ----------
  services.avahi = {
    enable = true;
    nssmdns4 = true;
    reflector = true;
    publish = {
      enable = true;
      addresses = true;
    };
  };

  # ---------- Locale / timezone ----------
  time.timeZone = "America/New_York";
  i18n.defaultLocale = "en_US.UTF-8";

  # ---------- SSH ----------
  services.openssh = {
    enable = true;
    settings = {
      PermitRootLogin = "yes";
      PasswordAuthentication = true;
    };
  };

  # ---------- Desktop (COSMIC, for dev mode) ----------
  # Not started by default. Plug in a monitor and run:
  #   sudo systemctl start display-manager
  services.desktopManager.cosmic.enable = true;
  services.displayManager.cosmic-greeter.enable = true;

  # Greeter is enabled but not started on boot (headless by default).
  # To enter dev mode: sudo systemctl start cosmic-greeter
  systemd.services.cosmic-greeter.wantedBy = lib.mkForce [];
  systemd.services.display-manager.wantedBy = lib.mkForce [];

  # ---------- Auto-login TTY (headless production) ----------
  services.getty.autologinUser = "lucifera";

  # ---------- Graphics / OpenGL ----------
  hardware.graphics = {
    enable = true;
    extraPackages = with pkgs; [
      mesa
      intel-media-driver  # N5095 integrated GPU
    ];
  };

  # ---------- GL_Simple service ----------
  gl-simple = {
    enable = true;
    workingDirectory = "/home/lucifera/GL_Simple";
  };

  # ---------- Users ----------
  users.users.root.initialPassword = "lucifera";
  users.users.lucifera = {
    isNormalUser = true;
    description = "lucifera";
    initialPassword = "lucifera";
    extraGroups = [ "wheel" "video" "audio" "networkmanager" ];
  };

  security.sudo.wheelNeedsPassword = false;

  # ---------- Packages ----------
  environment.systemPackages = with pkgs; [
    vim
    curl
    htop
    git
    python3

    # Dev mode tools (used when COSMIC desktop is running)
    vscode
    google-chrome
    alacritty
  ];

  # ---------- Misc ----------
  system.stateVersion = "25.05";
}
