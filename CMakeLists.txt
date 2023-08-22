# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

cmake_minimum_required(VERSION 3.20)

# set the default CMAKE_INSTALL_PREFIX to the current directory/install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# print where MKLShim layer will be installed
message(STATUS "H4I-MKLShim will be installed to: ${CMAKE_INSTALL_PREFIX}")

include (${CMAKE_CURRENT_SOURCE_DIR}/CMake/MKLShimVersion.cmake)
project(MKLShim
    VERSION ${MKLShim_VERSION}
    LANGUAGES CXX)

# check if INTEL_COMPILER_PATH was passed to CMake invocation, if not, look for icpx in PATH
if(NOT DEFINED INTEL_COMPILER_PATH)
    find_program(INTEL_COMPILER_PATH icpx REQUIRED)
    message(STATUS "Found Intel C++ Compiler: ${INTEL_COMPILER_PATH}")
endif()

set(CMAKE_CXX_COMPILER ${INTEL_COMPILER_PATH})
set(CMAKE_CXX_COMPILER_ID "IntelLLVM")

# Define a target capturing common configuration settings.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
# Unfortunately, it doesn't seem to be possible to set all
# of the properties we want on this target and have them
# be inherited by targets that "link" it.
# In particular, we can't set a few C++ standards properties
# and so either have to set them globally or on every target.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(MKLShimCommonConfig INTERFACE)
target_compile_features(MKLShimCommonConfig
    INTERFACE
        cxx_std_17
    )

# We use the SYCL version of MKL, so ensure that our compiler
# knows how to compile SYCL.
# TODO this *only* supports the Intel C++ compiler.
find_package(IntelSYCL REQUIRED)
if(IntelSYCL_FOUND)
    # Ensure that when we look for MKL, we import the
    # support for using its SYCL version.
    # Despite the name of the variable, we are only
    # after the SYCL version, not DPC++ support.
    set(DPCPP_COMPILER ON)
endif()
set(MKL_THREADING sequential)
find_package(MKL CONFIG REQUIRED)
message(STATUS "Known MKL Targets: ${MKL_IMPORTED_TARGETS}")

add_subdirectory(src)
#add_subdirectory(test)

