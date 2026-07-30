#!/usr/bin/env bash

# exit if any command fails
set -o errexit
set -o nounset
set -o pipefail

# Patch CIRCT before configuring: circt_ortools_objlib_defs.patch edits
# lib/Scheduling/CMakeLists.txt, so it has to be in place before cmake runs.
# Re-running this script over an already-patched tree is expected and must stay
# a no-op, but a patch that genuinely fails to apply has to abort here: silently
# skipping the OR-Tools one leaves obj.CIRCTScheduling without -DOR_PROTO_DLL=,
# and the build then dies much later with a confusing missing-libMLIROptLib.a
# error from the shortnail step.
for patch in circt_ortools_objlib_defs circt_export_verilog; do
  if git -C circt apply --reverse --check "../nix/$patch.patch" 2>/dev/null; then
    echo "already applied: nix/$patch.patch"
  else
    git -C circt apply "../nix/$patch.patch"
  fi
done

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

cd circt/build
ninja check-circt || true
ninja tblgen-lsp-server || true
ninja mlir-pdll-lsp-server || true
ninja circt-lsp-server || true
