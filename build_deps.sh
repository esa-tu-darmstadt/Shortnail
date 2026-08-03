#!/usr/bin/env bash

# exit if any command fails
set -o errexit
set -o nounset
set -o pipefail

# Check that the required environment variables are set
test "${OR_TOOLS_VER+x}"

# Download circt & llvm first
git submodule update --init --recursive --depth 1
# Install ortools
chmod +x ./circt/utils/*.sh
sed -i "/^OR_TOOLS_VER=/c\OR_TOOLS_VER=${OR_TOOLS_VER}" ./circt/utils/get-or-tools.sh

# nix/ortools_gurobi_loader.patch is required for any build that can reach the
# MathOpt Gurobi backend (`-schedule-lil solver=GUROBI`); without it the first
# GRB* call throws bad_function_call, which is fatal under -fno-exceptions.
# get-or-tools.sh offers no hook between unpacking the tarball and configuring
# it, so inject the step ahead of its cmake line. Not version-guarded on
# purpose: on an OR_TOOLS_VER bump this fails loudly, which is the prompt to
# re-check whether the fix is still needed (see the patch header).
ORTOOLS_PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nix/ortools_gurobi_loader.patch"
sed -i "\|^cmake -S . -B build|i patch -p1 <\"${ORTOOLS_PATCH}\"" \
  ./circt/utils/get-or-tools.sh

. ./circt/utils/get-or-tools.sh
