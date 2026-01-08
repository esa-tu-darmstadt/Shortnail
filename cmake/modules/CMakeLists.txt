# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(SHORTNAIL_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/shortnail)
set(shortnail_cmake_builddir "${CMAKE_BINARY_DIR}/${SHORTNAIL_INSTALL_PACKAGE_DIR}")

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
get_property(SHORTNAIL_EXPORTS GLOBAL PROPERTY SHORTNAIL_EXPORTS)
export(TARGETS ${SHORTNAIL_EXPORTS} FILE ${shortnail_cmake_builddir}/SHORTNAILTargets.cmake)

get_property(SHORTNAIL_ALL_LIBS GLOBAL PROPERTY SHORTNAIL_ALL_LIBS)
get_property(SHORTNAIL_DIALECT_LIBS GLOBAL PROPERTY SHORTNAIL_DIALECT_LIBS)
get_property(SHORTNAIL_CONVERSION_LIBS GLOBAL PROPERTY SHORTNAIL_CONVERSION_LIBS)

# Generate SHORTNAILConfig.cmake for the build tree.
set(SHORTNAIL_CONFIG_CMAKE_DIR "${shortnail_cmake_builddir}")
set(SHORTNAIL_CONFIG_LIBRARY_DIRS "${SHORTNAIL_LIBRARY_DIR}")
set(SHORTNAIL_CONFIG_BINARY_DIR "${SHORTNAIL_BINARY_DIR}")
set(SHORTNAIL_CONFIG_TOOLS_DIR "${SHORTNAIL_TOOLS_DIR}")
set(SHORTNAIL_CONFIG_INCLUDE_EXPORTS "include(\"\${SHORTNAIL_CMAKE_DIR}/SHORTNAILTargets.cmake\")")
set(SHORTNAIL_CONFIG_INCLUDE_DIRS
  "${SHORTNAIL_SOURCE_DIR}/include"
  "${SHORTNAIL_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SHORTNAILConfig.cmake.in
  ${shortnail_cmake_builddir}/SHORTNAILConfig.cmake
  @ONLY)
set(SHORTNAIL_CONFIG_CMAKE_DIR)
set(SHORTNAIL_CONFIG_LIBRARY_DIRS)
set(SHORTNAIL_CONFIG_BINARY_DIR)
set(SHORTNAIL_CONFIG_TOOLS_DIR)
set(SHORTNAIL_CONFIG_INCLUDE_EXPORTS)
set(SHORTNAIL_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that add '<build>/lib/cmake/shortnail' to
# their CMAKE_MODULE_PATH, place API modules next to it.
# Copy without source permissions because the source could be read-only,
# but we need to write into the copied folder.
file(COPY .
  DESTINATION ${shortnail_cmake_builddir}
  NO_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
)

# Generate SHORTNAILConfig.cmake for the install tree.
set(SHORTNAIL_CONFIG_CODE "
# Compute the installation prefix from this SHORTNAILConfig.cmake file location.
get_filename_component(SHORTNAIL_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${SHORTNAIL_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(SHORTNAIL_CONFIG_CODE "${SHORTNAIL_CONFIG_CODE}
get_filename_component(SHORTNAIL_INSTALL_PREFIX \"\${SHORTNAIL_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(SHORTNAIL_CONFIG_CMAKE_DIR "\${SHORTNAIL_INSTALL_PREFIX}/${SHORTNAIL_INSTALL_PACKAGE_DIR}")
set(SHORTNAIL_CONFIG_LIBRARY_DIRS "\${SHORTNAIL_INSTALL_PREFIX}/lib")
set(SHORTNAIL_CONFIG_BINARY_DIR "\${SHORTNAIL_INSTALL_PREFIX}")
set(SHORTNAIL_CONFIG_TOOLS_DIR "\${SHORTNAIL_INSTALL_PREFIX}/bin")
set(SHORTNAIL_CONFIG_INCLUDE_EXPORTS "include(\"\${SHORTNAIL_CMAKE_DIR}/SHORTNAILTargets.cmake\")")
set(SHORTNAIL_CONFIG_INCLUDE_DIRS
  "\${SHORTNAIL_INSTALL_PREFIX}/include"
  )
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/SHORTNAILConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/SHORTNAILConfig.cmake
    @ONLY)
set(SHORTNAIL_CONFIG_CODE)
set(SHORTNAIL_CONFIG_CMAKE_DIR)
set(SHORTNAIL_CONFIG_LIBRARY_DIRS)
set(SHORTNAIL_CONFIG_BINARY_DIR)
set(SHORTNAIL_CONFIG_TOOLS_DIR)
set(SHORTNAIL_CONFIG_INCLUDE_EXPORTS)
set(SHORTNAIL_CONFIG_INCLUDE_DIRS)

# Include the cmake files so other tools can use shortnail-tblgen, etc.
install(EXPORT SHORTNAILTargets DESTINATION ${SHORTNAIL_INSTALL_PACKAGE_DIR}
        COMPONENT shortnail-cmake-exports)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/SHORTNAILConfig.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/AddSHORTNAIL.cmake
  DESTINATION ${SHORTNAIL_INSTALL_PACKAGE_DIR}
  COMPONENT shortnail-cmake-exports)

# if(NOT LLVM_ENABLE_IDE)
  # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
  add_custom_target(shortnail-cmake-exports)
  add_llvm_install_targets(install-shortnail-cmake-exports
                            COMPONENT shortnail-cmake-exports)
# endif()
