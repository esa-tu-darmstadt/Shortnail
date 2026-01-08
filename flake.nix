{
  description = "env flake";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.self.submodules = true;
  inputs.circt_src = {
    url = "path:./circt";
    flake = false;
  };

  outputs = { self, nixpkgs, flake-utils, circt_src }:
    flake-utils.lib.eachDefaultSystem (system: let
      nixpkgs_cfg = {
        allowUnfree = true;
      };
      pkgs = (import nixpkgs {
        inherit system;
        config = nixpkgs_cfg;
      });

      my-ortools = pkgs.callPackage ./nix/ortools.nix {};
      my_circt = pkgs.callPackage ./nix/circt.nix { inherit circt_src; ortools = my-ortools; };

      ln_build_deps = with pkgs; [
        cmake
      ];

      ln_runtime_deps = with pkgs; [
        my_circt
        my_circt.llvm

        my-ortools
      ];

      shortnail = with pkgs; stdenv.mkDerivation {
        pname = "shortnail";
        version = "version-not-found";
        src = ./.;

        enableParallelBuilding = true;

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Debug"
          "-DLLVM_ENABLE_ASSERTIONS=ON"
          "-DMLIR_DIR=${my_circt.llvm.dev}/lib/cmake/mlir"
          "-DCIRCT_DIR=${my_circt.dev}/lib/cmake/circt"
        ];

        # Avoid: libMLIRIR.a(Verifier.cpp.o): undefined reference to symbol '_ZSt15__once_callable@@GLIBCXX_3.4.11'
        NIX_CFLAGS_LINK = "-lstdc++"; # explictily link in the C++ stdlib
        nativeBuildInputs = ln_build_deps;
        buildInputs = ln_runtime_deps;

        meta.mainProgram = "shortnail-opt";

        outputs = [
          "out"
          "lib"
          "dev"
        ];

        postInstall = ''
          moveToOutput lib "$lib"
          moveToOutput lib/cmake "$dev"

          substituteInPlace $dev/lib/cmake/shortnail/SHORTNAILConfig.cmake \
            --replace-fail "\''${SHORTNAIL_INSTALL_PREFIX}/lib/cmake/shortnail" "$dev/lib/cmake/shortnail" \
            --replace-fail "\''${SHORTNAIL_INSTALL_PREFIX}/include" "$dev/include" \
            --replace-fail "\''${SHORTNAIL_INSTALL_PREFIX}/lib" "$lib/lib" \
            --replace-fail "\''${SHORTNAIL_INSTALL_PREFIX}/bin" "$out/bin" \
            --replace-fail "\''${SHORTNAIL_INSTALL_PREFIX}" "$out"
          substituteInPlace $dev/lib/cmake/shortnail/SHORTNAILTargets-debug.cmake \
            --replace-fail "\''${_IMPORT_PREFIX}/lib" "$lib/lib"
        '';

        passthru = {
          circt = my_circt;
          ortools = my-ortools;
        };
      };

    in rec {
      packages = rec {
        inherit shortnail;
        default = shortnail;
      };
      devShell = pkgs.mkShell {
        buildInputs = [
          # shortnail
        ];
        nativeBuildInputs = ln_runtime_deps ++ ln_build_deps;
      };
    }
  );
}
