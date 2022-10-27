

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

# Specify how to install the library.
include (GNUInstallDirs)
install(TARGETS MKLShim)
install(FILES 
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/mklshim.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/Context.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/types.h
            ${CMAKE_SOURCE_DIR}/include/h4i/mklshim/sgemm.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/h4i/mklshim/
    )

