#include <h4i/mklshim/mklshim.h>
#include <h4i/mklshim/impl/Context.h>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <thread>
#include <atomic>
#include <chrono>

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
    
    if (!context1) {
        std::cerr << "Failed to create context1" << std::endl;
        return false;
    }
    
    // Check initial reference count
    int refCount1 = context1->getRefCount();
    std::cout << "  After creating context 1, ref count: " << refCount1 << " (expected: 1)" << std::endl;
    if (refCount1 != 1) {
        std::cerr << "  ERROR: Expected ref count 1, got " << refCount1 << std::endl;
        return false;
    }
    
    std::cout << "  Creating context 2..." << std::endl;
    H4I::MKLShim::Context* context2 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (!context2) {
        std::cerr << "Failed to create context2" << std::endl;
        return false;
    }
    
    // Check reference count after second context
    int refCount2 = context1->getRefCount();
    std::cout << "  After creating context 2, ref count: " << refCount2 << " (expected: 2)" << std::endl;
    if (refCount2 != 2) {
        std::cerr << "  ERROR: Expected ref count 2, got " << refCount2 << std::endl;
        return false;
    }
    
    std::cout << "  Creating context 3..." << std::endl;
    H4I::MKLShim::Context* context3 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    std::cout << "  Creating context 4..." << std::endl;
    H4I::MKLShim::Context* context4 = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (!context3 || !context4) {
        std::cerr << "Failed to create contexts" << std::endl;
        return false;
    }
    
    // All should be the same context
    if (context1 != context2 || context1 != context3 || context1 != context4) {
        std::cerr << "Expected same context for same stream" << std::endl;
        return false;
    }
    
    // Check reference count after all contexts created
    int refCount4 = context1->getRefCount();
    std::cout << "  After creating all 4 contexts, ref count: " << refCount4 << " (expected: 4)" << std::endl;
    if (refCount4 != 4) {
        std::cerr << "  ERROR: Expected ref count 4, got " << refCount4 << std::endl;
        return false;
    }
    
    // Destroy contexts 4, 3, and 2
    std::cout << "  Destroying context 4..." << std::endl;
    H4I::MKLShim::Destroy(context4);
    
    int refCountAfter4 = context1->getRefCount();
    std::cout << "  After destroying context 4, ref count: " << refCountAfter4 << " (expected: 3)" << std::endl;
    if (refCountAfter4 != 3) {
        std::cerr << "  ERROR: Expected ref count 3, got " << refCountAfter4 << std::endl;
        return false;
    }
    
    std::cout << "  Destroying context 3..." << std::endl;
    H4I::MKLShim::Destroy(context3);
    
    int refCountAfter3 = context1->getRefCount();
    std::cout << "  After destroying context 3, ref count: " << refCountAfter3 << " (expected: 2)" << std::endl;
    if (refCountAfter3 != 2) {
        std::cerr << "  ERROR: Expected ref count 2, got " << refCountAfter3 << std::endl;
        return false;
    }
    
    std::cout << "  Destroying context 2..." << std::endl;
    H4I::MKLShim::Destroy(context2);
    
    int refCountAfter2 = context1->getRefCount();
    std::cout << "  After destroying context 2, ref count: " << refCountAfter2 << " (expected: 1)" << std::endl;
    if (refCountAfter2 != 1) {
        std::cerr << "  ERROR: Expected ref count 1, got " << refCountAfter2 << std::endl;
        return false;
    }
    
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

/**
 * Test case 5: Thread safety test
 * Test that multiple threads can create and destroy contexts simultaneously
 * without causing race conditions in the reference counting
 */
bool test_thread_safety() {
    std::cout << "Test 5: Testing thread safety of context reference counting..." << std::endl;
    
    // Get native handles for default stream
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        return false;
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    const int num_threads = 8;
    const int operations_per_thread = 50;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    
    // Function that each thread will execute
    auto thread_worker = [&](int thread_id) {
        try {
            for (int i = 0; i < operations_per_thread; ++i) {
                // Create context
                H4I::MKLShim::Context* context = H4I::MKLShim::Create(handles.data(), nHandles);
                if (!context) {
                    failure_count.fetch_add(1);
                    continue;
                }
                
                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                
                // Destroy context
                H4I::MKLShim::Destroy(context);
                success_count.fetch_add(1);
            }
        } catch (...) {
            failure_count.fetch_add(operations_per_thread);
        }
    };
    
    // Launch threads
    std::cout << "  Launching " << num_threads << " threads, each doing " 
              << operations_per_thread << " create/destroy operations..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(thread_worker, i);
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "  Completed in " << duration.count() << "ms" << std::endl;
    std::cout << "  Successful operations: " << success_count.load() << std::endl;
    std::cout << "  Failed operations: " << failure_count.load() << std::endl;
    
    // Check results
    int expected_total = num_threads * operations_per_thread;
    if (success_count.load() != expected_total || failure_count.load() != 0) {
        std::cerr << "Expected " << expected_total << " successful operations, got " 
                  << success_count.load() << " successful and " << failure_count.load() << " failed" << std::endl;
        return false;
    }
    
    std::cout << "Test 5: PASSED (thread safety verified!)" << std::endl;
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
        bool test5_pass = test_thread_safety();
        
        if (test1_pass && test2_pass && test3_pass && test4_pass && test5_pass) {
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