// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include "h4i/mklshim/mklshim.h"

int main() {
    std::cout << "Testing MKLShim Context Creation and Basic Functions" << std::endl;
    
    // Get native handles for default stream
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    // Test context creation
    H4I::MKLShim::Context* context = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (context == nullptr) {
        std::cerr << "Failed to create MKLShim context" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Successfully created MKLShim context" << std::endl;
    
    // Test context update with a new stream
    hipStream_t stream;
    hipError_t hipStatus = hipStreamCreate(&stream);
    if (hipStatus != hipSuccess) {
        std::cerr << "Failed to create HIP stream" << std::endl;
        H4I::MKLShim::Destroy(context);
        return EXIT_FAILURE;
    }
    
    // Get native handles for the new stream
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), 0, &nHandles);
    std::vector<unsigned long> streamHandles(nHandles);
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), streamHandles.data(), 0);
    
    // Update the context with the new stream
    H4I::MKLShim::Context* updatedContext = H4I::MKLShim::Update(context, streamHandles.data(), nHandles);
    
    if (updatedContext == nullptr) {
        std::cerr << "Failed to update MKLShim context with new stream" << std::endl;
        hipStreamDestroy(stream);
        H4I::MKLShim::Destroy(context);
        return EXIT_FAILURE;
    }
    
    std::cout << "Successfully updated MKLShim context with new stream" << std::endl;
    
    // Check MKL version
    H4I::MKLShim::MKL_VERSION version = H4I::MKLShim::get_mkl_version();
    std::cout << "MKL Version: " << version.major << "." << version.minor << "." << version.patch << std::endl;

    // Check if MKL version is >= 2023.0.2
    bool is_recent_mkl = H4I::MKLShim::is_mkl_eq_higher_2023_0_2();
    std::cout << "MKL version is_mkl_eq_higher_2023_0_2: " << (is_recent_mkl ? "true" : "false") << std::endl;
    
    // Clean up
    hipStreamDestroy(stream);
    
    // Test SetStream with a new stream
    hipStream_t stream2;
    hipStatus = hipStreamCreate(&stream2);
    if (hipStatus != hipSuccess) {
        std::cerr << "Failed to create HIP stream for SetStream test" << std::endl;
        H4I::MKLShim::Destroy(updatedContext);
        return EXIT_FAILURE;
    }

    // Get native handles for the new stream (stream2)
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream2), 0, &nHandles);
    std::vector<unsigned long> streamHandles2(nHandles);
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream2), streamHandles2.data(), 0);

    // Set the stream for the context
    H4I::MKLShim::SetStream(updatedContext, streamHandles2.data(), nHandles);
    std::cout << "Successfully called SetStream for MKLShim context with a new stream" << std::endl;

    // Clean up the second stream
    hipStreamDestroy(stream2);

    // Test destruction
    H4I::MKLShim::Destroy(updatedContext);

    std::cout << "All tests passed successfully!" << std::endl;
    return EXIT_SUCCESS;
} 