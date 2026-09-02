{
  description = "FreeToken development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forEachSystem (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          python = pkgs.python313.withPackages (ps: [
            ps.debugpy
            ps.pytest
          ]);
          cuda = pkgs.cudaPackages_13_0;
        in {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.pyright
              pkgs.llvmPackages_22.clang-tools
              pkgs.gcc
              pkgs.ninja
              cuda.cudatoolkit
            ];

            shellHook = ''
              export CUDA_HOME="${cuda.cudatoolkit}"
              export CUDA_PATH="$CUDA_HOME"
              export CUDACXX="$CUDA_HOME/bin/nvcc"
              if [ -d "$PWD/.venv/bin" ]; then
                export PATH="$PWD/.venv/bin:$PATH"
              fi
              if [ -d "$PWD/.venv/lib/python3.13/site-packages" ]; then
                export PYTHONPATH="$PWD:$PWD/.venv/lib/python3.13/site-packages:$PWD/python''${PYTHONPATH:+:$PYTHONPATH}"
              else
                export PYTHONPATH="$PWD:$PWD/python''${PYTHONPATH:+:$PYTHONPATH}"
              fi
              export LD_LIBRARY_PATH="$CUDA_HOME/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };
        });
    };
}
