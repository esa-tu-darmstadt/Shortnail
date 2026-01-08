# Shortnail - MLIR dialect and tools for CoreDSL

## Initial setup
Shortnail has the same [dependencies as CIRCT](https://github.com/llvm/circt/tree/main/#setting-this-up), in a nutshell: C++17 compiler, CMake, Ninja, Python, Git.
OR-Tools are mandatory. The easiest way to get a suitable version is to use the script in CIRCT (see step 2 below).
Alternatively, CIRCT's Docker image for integration tests (`ghcr.io/circt/images/circt-integration-test`) is also a good starting point.

```
$ git submodule update --init --recursive --depth 1
$ OR_TOOLS_VER="9.11" ./build_deps.sh
$ ./build_circt.sh
$ ./build_shortnail.sh
```

## Running the tests
```
$ cd build ; ninja check-shortnail
```
Or
```
$ ./build_shortnail.sh
```

## Development
Shortnail adheres to the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html), with the execption that curly braces on simple `if`/`else` statements are acceptable. Please ensure that code is reformatted by `clang-format` prior to committing. The sample VS code configuration is set up to reformat the code on save.

Shortnail provides a language server for its MLIR dialects. To use it, run `ninja shortnail-lsp-server` after changes to the dialect definitions.

### Building the nix package
Building Shortnail using the provided `flake.nix` might fail in case the `$TMPDIR` does not provide enough space for the nix builder.
Note that compiling all dependencies in Debug mode requires approximately 200 GB of storage.
Hence, it is recommended to set `export TMPDIR=<somewhere>` to `<somewhere>` with enough storage capacity before attempting to build Shortnail using nix.

## License
Shortnail is available under the Apache License v2.0 and built on [CIRCT](https://circt.llvm.org), [MLIR](https://mlir.llvm.org) and [LLVM](https://llvm.org), which are available under the Apache License v2.0 with LLVM Exceptions.
