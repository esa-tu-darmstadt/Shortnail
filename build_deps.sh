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
. ./circt/utils/get-or-tools.sh
