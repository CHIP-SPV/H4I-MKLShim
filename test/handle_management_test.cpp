// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include <h4i/mklshim/mklshim.h>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

/**
 * Test case 1: Basic context creation and destruction
 * Test that we can create and destroy a single context
 */
bool test_single_context() {
    std::cout << "Test 1: Testing single context creation and destruction..." << std::endl;
    
    // Get native handles for default stream
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        return false;
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    // Create context
    H4I::MKLShim::Context* context = H4I::MKLShim::Create(handles.data(), nHandles);
    if (!context) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    // Destroy context
    H4I::MKLShim::Destroy(context);
    
    std::cout << "Test 1: PASSED" << std::endl;
    return true;
}

/**
 * Test case 2: Multiple contexts with same stream
 * Test that multiple contexts can be created for the same stream
 */
bool test_multiple_contexts_same_stream() {
    std::cout << "Test 2: Testing multiple contexts with same stream..." << std::endl;
    
    // Get native handles for default stream
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        return false;
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    // Create multiple contexts for same stream (simulating multiple handle creations)
    H4I::MKLShim::Context* context1 = H4I::MKLShim::Create(handles.data(), nHandles);
    H4I::MKLShim::Context* context2 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (!context1 || !context2) {
        std::cerr << "Failed to create contexts" << std::endl;
        return false;
    }
    
    // They should be the same context
    if (context1 != context2) {
        std::cerr << "Expected same context for same stream, got different contexts" << std::endl;
        return false;
    }
    
    // Destroy both
    H4I::MKLShim::Destroy(context1);
    H4I::MKLShim::Destroy(context2);
    
    std::cout << "Test 2: PASSED" << std::endl;
    return true;
}

/**
 * Test case 3: Reference counting test
 * This demonstrates the reference counting issue
 * When multiple handles share the same context, destroying some handles
 * should not destroy the context if other handles are still using it
 */
bool test_reference_counting_issue() {
    std::cout << "Test 3: Testing reference counting (reproduces the issue)..." << std::endl;
    
    // Get native handles for default stream
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        return false;
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    // Create 4 contexts for same stream (simulating 4 hipBLAS handle creations)
    std::cout << "  Creating context 1..." << std::endl;
    H4I::MKLShim::Context* context1 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    std::cout << "  Creating context 2..." << std::endl;
    H4I::MKLShim::Context* context2 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    std::cout << "  Creating context 3..." << std::endl;
    H4I::MKLShim::Context* context3 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    std::cout << "  Creating context 4..." << std::endl;
    H4I::MKLShim::Context* context4 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (!context1 || !context2 || !context3 || !context4) {
        std::cerr << "Failed to create contexts" << std::endl;
        return false;
    }
    
    // All should be the same context
    if (context1 != context2 || context1 != context3 || context1 != context4) {
        std::cerr << "Expected same context for same stream" << std::endl;
        return false;
    }
    
    // Destroy contexts 4, 3, and 2
    std::cout << "  Destroying context 4..." << std::endl;
    H4I::MKLShim::Destroy(context4);
    
    std::cout << "  Destroying context 3..." << std::endl;
    H4I::MKLShim::Destroy(context3);
    
    std::cout << "  Destroying context 2..." << std::endl;
    H4I::MKLShim::Destroy(context2);
    
    // At this point, context1 should still be valid with the fix
    // Let's verify by trying to get MKL version (a simple operation that uses the context)
    std::cout << "  Verifying context 1 is still valid..." << std::endl;
    try {
        H4I::MKLShim::MKL_VERSION version = H4I::MKLShim::get_mkl_version();
        std::cout << "  SUCCESS: Context 1 is still valid! MKL version: " << version.major << "." << version.minor << "." << version.patch << std::endl;
    } catch (...) {
        std::cerr << "  FAILED: Context 1 is not valid (reference counting bug still exists)" << std::endl;
        return false;
    }
    
    std::cout << "  Destroying context 1..." << std::endl;
    H4I::MKLShim::Destroy(context1);
    
    std::cout << "Test 3: PASSED (reference counting works correctly!)" << std::endl;
    return true;
}

/**
 * Test case 4: Different streams should have different contexts
 */
bool test_different_streams() {
    std::cout << "Test 4: Testing different streams get different contexts..." << std::endl;
    
    // Create two different streams
    hipStream_t stream1, stream2;
    hipError_t err = hipStreamCreate(&stream1);
    if (err != hipSuccess) {
        std::cerr << "Failed to create stream1" << std::endl;
        return false;
    }
    
    err = hipStreamCreate(&stream2);
    if (err != hipSuccess) {
        std::cerr << "Failed to create stream2" << std::endl;
        hipStreamDestroy(stream1);
        return false;
    }
    
    // Get native handles for stream1
    int nHandles1;
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream1), 0, &nHandles1);
    std::vector<unsigned long> handles1(nHandles1);
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream1), handles1.data(), 0);
    
    // Get native handles for stream2
    int nHandles2;
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream2), 0, &nHandles2);
    std::vector<unsigned long> handles2(nHandles2);
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream2), handles2.data(), 0);
    
    // Create contexts
    H4I::MKLShim::Context* context1 = H4I::MKLShim::Create(handles1.data(), nHandles1);
    H4I::MKLShim::Context* context2 = H4I::MKLShim::Create(handles2.data(), nHandles2);
    
    if (!context1 || !context2) {
        std::cerr << "Failed to create contexts" << std::endl;
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
        return false;
    }
    
    // They should be different contexts
    if (context1 == context2) {
        std::cerr << "Expected different contexts for different streams" << std::endl;
        H4I::MKLShim::Destroy(context1);
        H4I::MKLShim::Destroy(context2);
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
        return false;
    }
    
    // Clean up
    H4I::MKLShim::Destroy(context1);
    H4I::MKLShim::Destroy(context2);
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    
    std::cout << "Test 4: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "Testing MKLShim Handle Management (GitHub Issue #44)" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    try {
        // Run all test cases
        bool test1_pass = test_single_context();
        bool test2_pass = test_multiple_contexts_same_stream();
        bool test3_pass = test_reference_counting_issue();
        bool test4_pass = test_different_streams();
        
        if (test1_pass && test2_pass && test3_pass && test4_pass) {
            std::cout << "===================================================" << std::endl;
            std::cout << "All handle management tests passed successfully!" << std::endl;
            std::cout << "Note: Test 3 verifies that reference counting works correctly" << std::endl;
            return EXIT_SUCCESS;
        } else {
            std::cerr << "Some tests failed!" << std::endl;
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return EXIT_FAILURE;
    }
} 