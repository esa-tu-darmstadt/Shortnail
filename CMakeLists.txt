cmake_minimum_required(VERSION 3.13.4)
project(shortnail LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(ortools CONFIG)
if (ortools_FOUND)
else()
  # Look for OR-Tools (and its deps) via -DOR_TOOLS_PATH or in circt/ext
  if(DEFINED OR_TOOLS_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${OR_TOOLS_PATH})
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/circt/ext)
  find_package(ortools REQUIRED CONFIG)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ortools.cmake in: ${ortools_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

get_filename_component(ortools_CMAKEDIR ${ortools_DIR} DIRECTORY)
get_filename_component(ortools_LIBDIR ${ortools_CMAKEDIR} DIRECTORY)
list(APPEND CMAKE_INSTALL_RPATH ${ortools_LIBDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)



#-------------------------------------------------------------------------------
# SHORTNAIL configuration
#-------------------------------------------------------------------------------

# SHORTNAIL project.
set(SHORTNAIL_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(SHORTNAIL_MAIN_INCLUDE_DIR ${SHORTNAIL_MAIN_SRC_DIR}/include)

set(SHORTNAIL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHORTNAIL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SHORTNAIL_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(SHORTNAIL_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(SHORTNAIL_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddSHORTNAIL)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(shortnail-headers)
set_target_properties(shortnail-headers PROPERTIES FOLDER "Misc")
add_custom_target(shortnail-doc)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})

# Add SHORTNAIL files to the include path
include_directories(${SHORTNAIL_MAIN_INCLUDE_DIR})
include_directories(${SHORTNAIL_INCLUDE_DIR})

add_subdirectory(include/shortnail)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

# Custom target to install all CIRCT libraries
add_custom_target(shortnail-libraries)
set_target_properties(shortnail-libraries PROPERTIES FOLDER "Misc")
get_property(SHORTNAIL_LIBS GLOBAL PROPERTY SHORTNAIL_ALL_LIBS)
if(SHORTNAIL_LIBS)
  list(REMOVE_DUPLICATES SHORTNAIL_LIBS)
  foreach(lib ${SHORTNAIL_LIBS})
    add_dependencies(shortnail-libraries ${lib})
  endforeach()
endif()

install(DIRECTORY include/shortnail
  DESTINATION include
  COMPONENT shortnail-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${SHORTNAIL_INCLUDE_DIR}/shortnail
  DESTINATION include
  COMPONENT shortnail-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

add_subdirectory(cmake/modules)
