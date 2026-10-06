{
  description = "SQL Transforms dev shell: Python 3.14 + uv, Rust, JDK 21 for the Spark gate";

  # nixpkgs comes from the channel tarball, not `github:NixOS/nixpkgs`. Cloud
  # sessions reach GitHub only for the repos the session is scoped to, so a
  # github: input cannot be fetched there; channels.nixos.org is reachable.
  inputs.nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            python314
            uv
            # Confit's PyO3 extension: maturin builds it during `uv sync`.
            rustc
            cargo
            clippy
            rustfmt
            # The dialect cross-engine gate runs a local Spark session; its pins
            # were measured under JDK 21 (same as CI).
            jdk21
            mise
            git
          ];

          # uv uses the Nix Python instead of downloading its own build, so the
          # venv links against the same interpreter the shell provides.
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pkgs.python314}/bin/python3.14";
          JAVA_HOME = "${pkgs.jdk21}";
          # PyPI manylinux wheels (pyarrow, duckdb, ...) dlopen libstdc++ and
          # libz by soname; the Nix Python does not search the host's /usr/lib.
          LD_LIBRARY_PATH = pkgs.lib.optionalString pkgs.stdenv.isLinux (
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
          );
        };
      });
    };
}
