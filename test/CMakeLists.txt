# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# Add the basic context tests
add_executable(mklshim_context_test context_test.cpp)
target_link_libraries(mklshim_context_test PRIVATE MKLShim hip::host)
target_include_directories(mklshim_context_test PRIVATE 
                          ${CMAKE_BINARY_DIR}/include 
                          ${CMAKE_SOURCE_DIR}/include)

# Add the batch functions API smoke test (no correctness checking)
add_executable(mklshim_batch_smoke_test batch_smoke_test.cpp)
target_link_libraries(mklshim_batch_smoke_test PRIVATE MKLShim hip::host)
target_include_directories(mklshim_batch_smoke_test PRIVATE 
                          ${CMAKE_BINARY_DIR}/include 
                          ${CMAKE_SOURCE_DIR}/include)

# Add the batch functions correctness test (compares batch vs non-batch + CPU validation)
add_executable(mklshim_batch_correctness_test batch_correctness_test.cpp)
target_link_libraries(mklshim_batch_correctness_test PRIVATE MKLShim hip::host)
target_include_directories(mklshim_batch_correctness_test PRIVATE 
                          ${CMAKE_BINARY_DIR}/include 
                          ${CMAKE_SOURCE_DIR}/include)

# Add the handle management test (GitHub issue #44 reproducer)
add_executable(mklshim_handle_management_test handle_management_test.cpp)
target_link_libraries(mklshim_handle_management_test PRIVATE MKLShim hip::host)
target_include_directories(mklshim_handle_management_test PRIVATE
                          ${CMAKE_BINARY_DIR}/include
                          ${CMAKE_SOURCE_DIR}/include)
target_compile_options(mklshim_handle_management_test PRIVATE -fsycl)

# Reproducer: 3D R2C must produce cuFFT-compatible interleaved-complex output
add_executable(mklshim_fft_3d_cufft_layout_test fft_3d_cufft_layout_test.cpp)
target_link_libraries(mklshim_fft_3d_cufft_layout_test PRIVATE MKLShim hip::host)
target_include_directories(mklshim_fft_3d_cufft_layout_test PRIVATE
                          ${CMAKE_BINARY_DIR}/include
                          ${CMAKE_SOURCE_DIR}/include)

# Add tests to CTest
add_test(NAME h4i-mklshim_context_test COMMAND mklshim_context_test) 
set_tests_properties(h4i-mklshim_context_test PROPERTIES 
    PASS_REGULAR_EXPRESSION "All tests passed successfully!")

add_test(NAME h4i-mklshim_batch_smoke_test COMMAND mklshim_batch_smoke_test)
set_tests_properties(h4i-mklshim_batch_smoke_test PROPERTIES 
    PASS_REGULAR_EXPRESSION "All batch function API smoke tests passed!")

add_test(NAME h4i-mklshim_batch_correctness_test COMMAND mklshim_batch_correctness_test)
set_tests_properties(h4i-mklshim_batch_correctness_test PROPERTIES 
    PASS_REGULAR_EXPRESSION "All batch function correctness tests passed!")

add_test(NAME h4i-mklshim_handle_management_test COMMAND mklshim_handle_management_test)
set_tests_properties(h4i-mklshim_handle_management_test PROPERTIES
    PASS_REGULAR_EXPRESSION "All handle management tests passed successfully!")

add_test(NAME h4i-mklshim_fft_3d_cufft_layout_test COMMAND mklshim_fft_3d_cufft_layout_test)
set_tests_properties(h4i-mklshim_fft_3d_cufft_layout_test PROPERTIES
    PASS_REGULAR_EXPRESSION "All cuFFT-layout 3D R2C tests passed!")
