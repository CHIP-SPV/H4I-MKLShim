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

# We use MKL for implementations of functions we expose.
find_package(MKL CONFIG REQUIRED)
#message(STATUS "Known MKL Targets: ${MKL_IMPORTED_TARGETS}")

add_subdirectory(src)
#add_subdirectory(test)

