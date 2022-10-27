cmake_minimum_required(VERSION 3.24)
project(H4I-MKLShim
    VERSION 0.1.0
    LANGUAGES CXX)

# Define a target capturing common compiler configuration.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
add_library(MKLShimCommonConfig INTERFACE)
target_compile_features(MKLShimCommonConfig
    INTERFACE
        cxx_std_17
    )
set_target_properties(MKLShimCommonConfig
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)

add_subdirectory(src)
#add_subdirectory(test)

