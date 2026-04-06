{
  description = "GL_Simple — Beelink Mini S (Celeron N5095) lighting controller";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixos-cosmic.url = "github:lilyinstarlight/nixos-cosmic";
    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixos-facter-modules.url = "github:numtide/nixos-facter-modules";
  };

  outputs = { self, nixpkgs, nixos-cosmic, disko, nixos-facter-modules, ... }:
    let
      system = "x86_64-linux";
    in
    {
      # Deploy from your dev machine with:
      #
      #   nix run github:nix-community/nixos-anywhere -- \
      #     --generate-hardware-config nixos-facter ./nixos/hosts/facter.json \
      #     --flake ./nixos#lucifera \
      #     --target-host root@192.168.68.144
      #
      nixosConfigurations.lucifera = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          disko.nixosModules.disko
          nixos-facter-modules.nixosModules.facter
          nixos-cosmic.nixosModules.default
          ./hosts/lucifera.nix
          ./disk-config.nix
          { hardware.facter.reportPath = ./hosts/facter.json; }
        ];
      };
    };
}
