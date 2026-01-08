# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'SHORTNAIL'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.shortnail_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# TODO check which of the additional flags are ACTUALLY needed to make it compile in the isolated environment
llvm_config.with_system_environment([
  'HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP', 'GUROBI_HOME', 'GRB_LICENSE_FILE',
  'NIX_BINTOOLS', 'NIX_BUILD_CORES', 'NIX_CC', 'NIX_LDFLAGS', 'NIX_STORE',
  'NIX_BINTOOLS_WRAPPER_TARGET_HOST_x86_64_unknown_linux_gnu',
  'NIX_CC_WRAPPER_TARGET_HOST_x86_64_unknown_linux_gnu', 
  'NIX_CFLAGS_COMPILE', 'NIX_ENFORCE_NO_NATIVE', 'NIX_HARDENING_ENABLE'
])

llvm_config.use_default_substitutions()

# Set the timeout, if requested.
if config.timeout is not None and config.timeout != "":
  lit_config.maxIndividualTestTime = int(config.timeout)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.shortnail_obj_root, 'test')
config.shortnail_tools_dir = os.path.join(config.shortnail_obj_root, 'bin')

# there's probably a cleaner way to determine this
config.circt_tools_dir = os.path.join(config.shortnail_src_root,
                                      'circt/build/bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.shortnail_tools_dir, config.circt_tools_dir, config.llvm_tools_dir
]
tools = ['shortnail-opt']

llvm_config.add_tool_substitutions(tools, tool_dirs)

