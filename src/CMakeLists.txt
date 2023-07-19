# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# Specify how to build the library.
add_library(MKLShim SHARED
    Context.cpp
    Common.cpp
    onemklblas.cpp
    onemklsolver.cpp
    onemklfft.cpp)
target_include_directories(MKLShim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
# When defining the library's link interface, ideally we would
# use MKL::MKL_DPCPP. But then our "clients" will see this 
# specific target as a dependency.  If those clients are not
# built with a DPC++ compiler, the MKL CMake import logic won't
# define the MKL::MKL_DPCPP target making the CMake configuration
# fail for any of our clients.  To avoid this, we extract
# the relevant info from the MKL::MKL_DPCPP target and 
# use that instead of the target itself.  This also reflects
# our avoidance of using DPC++ in the compiler (see top level
# CMakeLists.txt for more details).
get_target_property(MKL_DPCPP_INTERFACE_LINK_LIBS MKL::MKL_DPCPP INTERFACE_LINK_LIBRARIES)
target_link_libraries(MKLShim
    PRIVATE
        MKLShimCommonConfig
        MKL::MKL_DPCPP
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
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/common.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/onemklblas.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/onemklsolver.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/onemklfft.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/h4i/mklshim/
    )

install(EXPORT MKLShim
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MKLShim
    NAMESPACE H4I::
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/CMake/MKLShimConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MKLShimConfig.cmake
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MKLShim
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MKLShimConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MKLShim
)

