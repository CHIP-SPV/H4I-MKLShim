cmake_minimum_required(VERSION 3.24)
include (${CMAKE_SOURCE_DIR}/CMake/MKLShimVersion.cmake)
project(H4I-MKLShim
    VERSION ${MKLShim_VERSION}
    LANGUAGES CXX)

# Define a target capturing common compiler configuration.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
# Unfortunately, it doesn't seem to be possible to set some
# of the C++ standards-related properties on this target
# and have it inherited by targets that "link" from it.
# We have to set the global variables that those target
# properties default to, or set them explicitly on the
# actual targets themselves with set_target_properties.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(H4I-MKLShimCommonConfig INTERFACE)
target_compile_features(H4I-MKLShimCommonConfig
    INTERFACE
        cxx_std_17
    )

add_subdirectory(src)
#add_subdirectory(test)

