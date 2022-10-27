

# We use MKL for implementations of functions we expose.
find_package(MKL CONFIG REQUIRED)
message(STATUS "Known MKL Targets: ${MKL_IMPORTED_TARGETS}")

# Specify how to build the library.
add_library(MKLShim SHARED
    sgemm.cpp)
target_include_directories(MKLShim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
target_link_libraries(MKLShim
    PRIVATE
        H4I-MKLShimCommonConfig
    )

# Specify how to install the library.
include (GNUInstallDirs)
install(TARGETS MKLShim
        EXPORT MKLShim
)
install(FILES 
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/mklshim.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/Context.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/types.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/sgemm.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/h4i/mklshim/
    )

install(EXPORT MKLShim
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/H4I-MKLShim
    NAMESPACE H4I::
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/CMake/H4I-MKLShimConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/H4I-MKLShimConfig.cmake
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/H4I-MKLShim
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/H4I-MKLShimConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/H4I-MKLShim
)

