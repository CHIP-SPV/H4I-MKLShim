# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# Add the basic context tests
add_executable(mklshim_context_test context_test.cpp)
target_link_libraries(mklshim_context_test PRIVATE MKLShim hip::host)
target_include_directories(mklshim_context_test PRIVATE 
                          ${CMAKE_BINARY_DIR}/include 
                          ${CMAKE_SOURCE_DIR}/H4I-MKLShim/include)

# Add tests to CTest
add_test(NAME h4i-mklshim_context_test COMMAND mklshim_context_test) 