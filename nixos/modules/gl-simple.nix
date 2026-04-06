{ config, pkgs, lib, ... }:

let
  cfg = config.gl-simple;

  # Python environment with all GL_Simple dependencies
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    numpy
    scipy
    numba
    matplotlib
    opencv4
    pyopengl
    pygame
    psutil
    flask
    flask-socketio
    pyyaml
    scikit-image
    sounddevice
    soundfile
    miniaudio
    sacn
    pyglm
    zeroconf
    glfw
  ]);
in
{
  options.gl-simple = {
    enable = lib.mkEnableOption "GL_Simple DMX lighting controller";

    workingDirectory = lib.mkOption {
      type = lib.types.path;
      default = "/home/lucifera/GL_Simple";
      description = "Path to the GL_Simple repository";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.gl-simple = {
      description = "GL_Simple DMX Lighting Controller";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = "lucifera";
        WorkingDirectory = cfg.workingDirectory;
        ExecStart = "${pythonEnv}/bin/python Stories_OGL.py";
        Restart = "on-failure";
        RestartSec = 5;

        # Headless OpenGL needs access to the GPU
        SupplementaryGroups = [ "video" ];

        # Allow binding to port 80 without running as root
        AmbientCapabilities = [ "CAP_NET_BIND_SERVICE" ];
      };

      environment = {
        # Let Mesa find the GPU for headless EGL
        MESA_LOADER_DRIVER_OVERRIDE = "iris";
      };
    };
  };
}
