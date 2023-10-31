{
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfreePredicate = pkg: builtins.elem (nixpkgs.lib.getName pkg) [
      "cudatoolkit"
      "nvidia-x11"
    ]; };
  in {
    devShells.x86_64-linux.default = let
    fhs =
      pkgs.buildFHSUserEnv {
        name = "gaussian-splatting";
        targetPkgs = pkgs: with pkgs; [
          gcc11Stdenv
          gcc11Stdenv.cc
          micromamba
          cudaPackages_11_6.cudatoolkit
          libxcrypt-legacy
          eudev
          libxml2
          glm
          libGL
          embree
          ffmpeg.lib
          xorg.libX11
          which
        ];

        profile = ''
          set -eh
          export MAMBA_ROOT_PREFIX="$HOME/.mamba"
          export CUDA_HOME="${nixpkgs.lib.getBin pkgs.cudaPackages_11_6.cudatoolkit}"
          # export TORCH_CUDA_ARCH_LIST="7.5+PTX"
          eval "$(micromamba shell hook -s posix)"

          if [[ ! -f $MAMBA_ROOT_PREFIX/envs/gaussian_splatting/good ]]; then
            micromamba create -q -n gaussian_splatting || true
            micromamba activate gaussian_splatting
            micromamba install --yes -f environment.yml
            touch $MAMBA_ROOT_PREFIX/envs/gaussian_splatting/good
          else
            micromamba activate gaussian_splatting
          fi
          set +e
          '';
      };
    in fhs.env;
  };
}
