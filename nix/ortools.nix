{
  abseil-cpp_202508,
  bzip2,
  cbc,
  cmake,
  eigen,
  ensureNewerSourcesForZipFilesHook,
  fetchFromGitHub,
  fetchpatch,
  glpk,
  highs,
  clp,
  scipopt-scip,
  lib,
  pkg-config,
  protobuf_32,
  re2,
  stdenv,
  swig,
  unzip,
  zlib,
}:

let
  # OR-Tools strictly requires specific versions of abseil-cpp and
  # protobuf. Do not un-pin these, even if you're upgrading them to
  # what might happen to be the latest version at the current moment;
  # future upgrades *will* break the build. (9.15: abseil 20250814,
  # protobuf v32 — mirrors the nixpkgs or-tools 9.15 recipe.)
  abseil-cpp = abseil-cpp_202508;
  protobuf = protobuf_32.override { inherit abseil-cpp; };
  patched_re2 = re2.override { inherit abseil-cpp; };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "or-tools";
  version = "9.15";

  src = fetchFromGitHub {
    owner = "google";
    repo = "or-tools";
    tag = "v${finalAttrs.version}";
    hash = "sha256-9+tvgP/+/VY6wu7lzTdP4xfiJIgPSLVR9lEdZjQCZkE=";
  };

  cmakeFlags =
    [
      (lib.cmakeBool "BUILD_DEPS" false)
      (lib.cmakeBool "BUILD_PYTHON" false)
      (lib.cmakeBool "BUILD_SAMPLES" false)
      (lib.cmakeBool "BUILD_EXAMPLES" false)
      (lib.cmakeBool "BUILD_pybind11" false)
      (lib.cmakeFeature "CMAKE_INSTALL_BINDIR" "bin")
      (lib.cmakeFeature "CMAKE_INSTALL_INCLUDEDIR" "include")
      (lib.cmakeFeature "CMAKE_INSTALL_LIBDIR" "lib")
      (lib.cmakeBool "FETCH_PYTHON_DEPS" false)
      (lib.cmakeBool "USE_GLPK" true)
      (lib.cmakeBool "USE_SCIP" true)
      (lib.cmakeBool "USE_HIGHS" true)
      (lib.cmakeBool "USE_COINOR" true)
    ]
    ++ lib.optionals stdenv.hostPlatform.isDarwin [
      (lib.cmakeBool "CMAKE_MACOSX_RPATH" false)
    ];

  strictDeps = true;

  nativeBuildInputs = [
    cmake
    ensureNewerSourcesForZipFilesHook
    pkg-config
    swig
    unzip
    protobuf
  ];
  buildInputs = [
    abseil-cpp
    bzip2
    cbc
    eigen
    glpk
    highs
    patched_re2
    zlib
  ];
  propagatedBuildInputs = [
    abseil-cpp
    protobuf
    highs
    scipopt-scip
    zlib
    bzip2
    patched_re2
    cbc
    glpk
    eigen

    pkg-config
    clp
  ];

  # hardening must be disabled to compile or-tools with scip
  hardeningDisable = [ "format" ];

  # some tests fail on linux and hang on darwin
  doCheck = false;
})