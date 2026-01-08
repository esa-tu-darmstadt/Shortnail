#!/usr/bin/env bash

# exit if any command fails
set -o errexit
set -o nounset
set -o pipefail

cmake -G Ninja \
  -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCIRCT_DIR=circt/build/lib/cmake/circt \
  -DMLIR_DIR=circt/build/lib/cmake/mlir \
  -DLLVM_DIR=circt/build/lib/cmake/llvm \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cd build
ninja shortnail-opt
ninja shortnail-lsp-server
ninja shortnail-doc
ninja check-shortnail
