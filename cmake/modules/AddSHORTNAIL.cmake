include_guard()

function(add_shortnail_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(shortnail-headers MLIR${dialect}IncGen)
endfunction()

function(add_shortnail_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(shortnail-headers MLIR${interface}IncGen)
endfunction()

# Additional parameters are forwarded to tablegen.
function(add_shortnail_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${PROJECT_SOURCE_DIR}/docs/${output_path}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(shortnail-doc ${output_id}DocGen)
endfunction()

function(add_shortnail_dialect_doc dialect dialect_namespace)
  add_shortnail_doc(
    ${dialect} ${dialect}Dialect
    -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_shortnail_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL)
  add_shortnail_library_install(${name})
endfunction()

macro(add_shortnail_executable name)
  add_llvm_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "shortnail executables")
endmacro()

macro(add_shortnail_tool name)
  add_shortnail_executable(${name} ${ARGN})

  get_target_export_arg(${name} SHORTNAIL export_to_shortnailtargets)
  install(TARGETS ${name}
    ${export_to_shortnailtargets}
    RUNTIME DESTINATION "${SHORTNAIL_TOOLS_INSTALL_DIR}"
    COMPONENT ${name})

  if(NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-${name}
      DEPENDS ${name}
      COMPONENT ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY SHORTNAIL_EXPORTS ${name})
endmacro()

# Adds a SHORTNAIL library target for installation.  This should normally only be
# called from add_shortnail_library().
function(add_shortnail_library_install name)
  get_target_export_arg(${name} SHORTNAIL export_to_shortnailtargets UMBRELLA shortnail-libraries)
  install(TARGETS ${name}
    COMPONENT ${name}
    ${export_to_shortnailtargets}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    # Note that CMake will create a directory like:
    #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
    # and put object files there.
    OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )

  # if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-${name}
                            DEPENDS ${name}
                            COMPONENT ${name})
  # endif()
  set_property(GLOBAL APPEND PROPERTY SHORTNAIL_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY SHORTNAIL_EXPORTS ${name})
endfunction()

function(add_shortnail_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY SHORTNAIL_DIALECT_LIBS ${name})
  add_shortnail_library(${ARGV} DEPENDS shortnail-headers)
endfunction()

function(add_shortnail_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY SHORTNAIL_CONVERSION_LIBS ${name})
  add_shortnail_library(${ARGV} DEPENDS shortnail-headers)
endfunction()
