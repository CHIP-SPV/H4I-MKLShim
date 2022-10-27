
# We use MKL for implementations of functions we expose.
find_package(MKL CONFIG REQUIRED)
message(STATUS "Known MKL Targets: ${MKL_IMPORTED_TARGETS}")

# Specify how to build the library.
add_library(MKLShim SHARED
    sgemm.cpp)
target_include_directories(MKLShim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:/include>
    )
target_link_libraries(MKLShim
    PRIVATE
        MKLShimCommonConfig
    )


