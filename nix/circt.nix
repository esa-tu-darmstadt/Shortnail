{
  stdenv,
  lib,
  cmake,
  coreutils,
  git,
  fetchFromGitHub,
  ninja,
  lit,
  z3,
  gitUpdater,
  callPackage,
  ortools,
  circt_src
}:

let
  version = "LN-pinned";
  circt-llvm = callPackage ./circt-llvm.nix { inherit circt_src version;};
in
stdenv.mkDerivation rec {
  pname = "circt";
  inherit version;
  src = circt_src;

  requiredSystemFeatures = [ "big-parallel" ];

  nativeBuildInputs = [
    cmake
    ninja
    git
    z3
  ];
  buildInputs = [
    circt-llvm
    ortools
  ];

  patches = [
    ./circt_ortools_required.patch
    ./circt_export_verilog.patch
  ];

  cmakeFlags = [
    "-DMLIR_DIR=${circt-llvm.dev}/lib/cmake/mlir"
    "-DCMAKE_BUILD_TYPE=Debug"

    "-DLLVM_EXTERNAL_LIT=${lit}/bin/.lit-wrapped"
    "-DCIRCT_LLHD_SIM_ENABLED=OFF"
  ];

  doCheck = false;

  outputs = [
    "out"
    "lib"
    "dev"
  ];

  postInstall = ''
    moveToOutput lib "$lib"
    moveToOutput lib/cmake "$dev"

    substituteInPlace $dev/lib/cmake/circt/CIRCTConfig.cmake \
      --replace-fail "\''${CIRCT_INSTALL_PREFIX}/lib/cmake/mlir" "${circt-llvm.dev}/lib/cmake/mlir" \
      --replace-fail "\''${CIRCT_INSTALL_PREFIX}/lib/cmake/circt" "$dev/lib/cmake/circt" \
      --replace-fail "\''${CIRCT_INSTALL_PREFIX}/include" "$dev/include" \
      --replace-fail "\''${CIRCT_INSTALL_PREFIX}/lib" "$lib/lib" \
      --replace-fail "\''${CIRCT_INSTALL_PREFIX}/bin" "$out/bin" \
      --replace-fail "\''${CIRCT_INSTALL_PREFIX}" "$out"
    substituteInPlace $dev/lib/cmake/circt/CIRCTTargets-debug.cmake \
      --replace-fail "\''${_IMPORT_PREFIX}/lib" "$lib/lib"
  '';

  passthru = {
    llvm = circt-llvm;
  };
}
