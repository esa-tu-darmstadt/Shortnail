#!/usr/bin/env bash

# exit if any command fails
set -o errexit
set -o nounset
set -o pipefail

BUILD_WITH_CCACHE=""
which ccache && BUILD_WITH_CCACHE="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache" || true

cmake -G Ninja \
  -S circt/llvm/llvm -B circt/build \
  -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON \
  -DLLVM_EXTERNAL_PROJECTS="circt" -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=circt \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DVERILATOR_DISABLE=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  $BUILD_WITH_CCACHE

cd circt
git apply ../nix/circt_export_verilog.patch || true
cd build
ninja check-circt || true
ninja tblgen-lsp-server || true
ninja mlir-pdll-lsp-server || true
ninja circt-lsp-server || true
