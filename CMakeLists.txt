# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

cmake_minimum_required(VERSION 3.24)
include (${CMAKE_SOURCE_DIR}/CMake/MKLShimVersion.cmake)
project(MKLShim
    VERSION ${MKLShim_VERSION}
    LANGUAGES CXX)

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
find_package(IntelDPCPP REQUIRED)
# Note that the current Intel CMake support sets DPC++-related
# compiler flags that cause the compiler to issue deprecation
# warnings.  So - strip out the DPC++ flags.
string(REPLACE "--dpcpp " "" NEW_SYCL_FLAGS "${SYCL_FLAGS}")
set(SYCL_FLAGS ${NEW_SYCL_FLAGS} CACHE STRING "SYCL flags for the compiler" FORCE)
string(REPLACE "--dpcpp " "" NEW_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS ${NEW_CMAKE_CXX_FLAGS})

# We use MKL for implementations of functions we expose.
if(IntelDPCPP_FOUND)
    # Ensure that when we look for MKL, we import the
    # support for using its SYCL version.
    # Despite the name of the variable, we are only
    # after the SYCL version, not DPC++ support.
    set(DPCPP_COMPILER ON)
endif()
find_package(MKL CONFIG REQUIRED)
message(STATUS "Known MKL Targets: ${MKL_IMPORTED_TARGETS}")

add_subdirectory(src)
#add_subdirectory(test)

