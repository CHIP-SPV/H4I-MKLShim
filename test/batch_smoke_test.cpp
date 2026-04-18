// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include <iostream>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include "h4i/mklshim/mklshim.h"

// Helper function for creating basic test matrices
template<typename T>
void createTestMatrix(T* matrix, int64_t n, int64_t lda, T diag_val) {
    // Zero-initialize
    for (int64_t i = 0; i < n * lda; ++i) {
        matrix[i] = T(0.0);
    }
    // Set diagonal elements to ensure well-conditioned matrix
    for (int64_t i = 0; i < n; ++i) {
        matrix[i * lda + i] = diag_val;
    }
}

// Complex type specializations
template<>
void createTestMatrix<float _Complex>(float _Complex* matrix, int64_t n, int64_t lda, float _Complex diag_val) {
    for (int64_t i = 0; i < n * lda; ++i) {
        matrix[i] = 0.0f + 0.0fi;
    }
    for (int64_t i = 0; i < n; ++i) {
        matrix[i * lda + i] = diag_val;
    }
}

template<>
void createTestMatrix<double _Complex>(double _Complex* matrix, int64_t n, int64_t lda, double _Complex diag_val) {
    for (int64_t i = 0; i < n * lda; ++i) {
        matrix[i] = 0.0 + 0.0i;
    }
    for (int64_t i = 0; i < n; ++i) {
        matrix[i * lda + i] = diag_val;
    }
}

bool testSinglePrecisionBatchSmoke(H4I::MKLShim::Context* context) {
    std::cout << "API Smoke Test: Single precision batch functions..." << std::endl;
    
    const int64_t group_count = 1;
    const int64_t n = 3;
    const int64_t m_values[1] = {n};
    const int64_t n_values[1] = {n};
    const int64_t lda_values[1] = {n};
    const int64_t nrhs_values[1] = {1};
    const int64_t ldb_values[1] = {n};
    const int64_t group_sizes[1] = {1};
    H4I::MKLShim::onemklTranspose trans_values[1] = {H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS};
    
    // Get scratchpad sizes (this also tests the scratchpad size functions)
    auto getrf_scratch = H4I::MKLShim::Sgetrf_batch_ScPadSz(context, group_count,
                                                             const_cast<int64_t*>(m_values),
                                                             const_cast<int64_t*>(n_values),
                                                             const_cast<int64_t*>(lda_values),
                                                             const_cast<int64_t*>(group_sizes));
    
    auto getri_scratch = H4I::MKLShim::Sgetri_batch_ScPadSz(context, group_count,
                                                             const_cast<int64_t*>(n_values),
                                                             const_cast<int64_t*>(lda_values),
                                                             const_cast<int64_t*>(group_sizes));
    
    auto getrs_scratch = H4I::MKLShim::Sgetrs_batch_ScPadSz(context, group_count,
                                                             trans_values,
                                                             const_cast<int64_t*>(n_values),
                                                             const_cast<int64_t*>(nrhs_values),
                                                             const_cast<int64_t*>(lda_values),
                                                             const_cast<int64_t*>(ldb_values),
                                                             const_cast<int64_t*>(group_sizes));
    
    if (getrf_scratch < 0 || getri_scratch < 0 || getrs_scratch < 0) {
        std::cerr << "Failed to get valid scratchpad sizes" << std::endl;
        return false;
    }
    
    // Allocate device memory
    float* A_device;
    float* B_device;
    int64_t* ipiv_device;
    float* scratch_device;
    float** A_ptrs_device;
    float** B_ptrs_device;
    int64_t** ipiv_ptrs_device;
    
    size_t max_scratch = std::max({getrf_scratch, getri_scratch, getrs_scratch});
    
    hipMalloc(&A_device, n * n * sizeof(float));
    hipMalloc(&B_device, n * nrhs_values[0] * sizeof(float));
    hipMalloc(&ipiv_device, n * sizeof(int64_t));
    hipMalloc(&scratch_device, max_scratch * sizeof(float));
    hipMalloc(&A_ptrs_device, sizeof(float*));
    hipMalloc(&B_ptrs_device, sizeof(float*));
    hipMalloc(&ipiv_ptrs_device, sizeof(int64_t*));
    
    // Create host data
    std::vector<float> A_host(n * n);
    std::vector<float> B_host(n * nrhs_values[0], 1.0f);
    createTestMatrix(A_host.data(), n, n, 2.0f);
    
    // Copy to device
    hipMemcpy(A_device, A_host.data(), A_host.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(B_device, B_host.data(), B_host.size() * sizeof(float), hipMemcpyHostToDevice);
    
    // Setup pointer arrays
    float* A_ptrs_host[1] = {A_device};
    float* B_ptrs_host[1] = {B_device};
    int64_t* ipiv_ptrs_host[1] = {ipiv_device};
    
    hipMemcpy(A_ptrs_device, A_ptrs_host, sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(B_ptrs_device, B_ptrs_host, sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_ptrs_device, ipiv_ptrs_host, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Test Sgetrf_batch - should not crash
    H4I::MKLShim::Sgetrf_batch(context, const_cast<int64_t*>(m_values), const_cast<int64_t*>(n_values), 
                                A_ptrs_device, const_cast<int64_t*>(lda_values), ipiv_ptrs_device, 
                                group_count, const_cast<int64_t*>(group_sizes), 
                                scratch_device, getrf_scratch);
    
    // Reset matrix for next test
    hipMemcpy(A_device, A_host.data(), A_host.size() * sizeof(float), hipMemcpyHostToDevice);
    
    // Test Sgetri_batch - should not crash
    H4I::MKLShim::Sgetri_batch(context, const_cast<int64_t*>(n_values), A_ptrs_device,
                                const_cast<int64_t*>(lda_values), ipiv_ptrs_device, group_count,
                                const_cast<int64_t*>(group_sizes), scratch_device, getri_scratch);
    
    // Reset matrix for next test
    hipMemcpy(A_device, A_host.data(), A_host.size() * sizeof(float), hipMemcpyHostToDevice);
    
    // Test Sgetrs_batch - should not crash
    H4I::MKLShim::Sgetrs_batch(context, trans_values, const_cast<int64_t*>(n_values),
                                const_cast<int64_t*>(nrhs_values), A_ptrs_device,
                                const_cast<int64_t*>(lda_values), ipiv_ptrs_device,
                                B_ptrs_device, const_cast<int64_t*>(ldb_values),
                                group_count, const_cast<int64_t*>(group_sizes),
                                scratch_device, getrs_scratch);
    
    // Cleanup
    hipFree(A_device);
    hipFree(B_device);
    hipFree(ipiv_device);
    hipFree(scratch_device);
    hipFree(A_ptrs_device);
    hipFree(B_ptrs_device);
    hipFree(ipiv_ptrs_device);
    
    std::cout << "Single precision batch functions smoke test passed" << std::endl;
    return true;
}

bool testDoublePrecisionBatchSmoke(H4I::MKLShim::Context* context) {
    std::cout << "API Smoke Test: Double precision batch functions..." << std::endl;
    
    const int64_t group_count = 1;
    const int64_t n = 3;
    const int64_t m_values[1] = {n};
    const int64_t n_values[1] = {n};
    const int64_t lda_values[1] = {n};
    const int64_t nrhs_values[1] = {1};
    const int64_t ldb_values[1] = {n};
    const int64_t group_sizes[1] = {1};
    H4I::MKLShim::onemklTranspose trans_values[1] = {H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS};
    
    // Get scratchpad sizes
    auto getrf_scratch = H4I::MKLShim::Dgetrf_batch_ScPadSz(context, group_count,
                                                             const_cast<int64_t*>(m_values),
                                                             const_cast<int64_t*>(n_values),
                                                             const_cast<int64_t*>(lda_values),
                                                             const_cast<int64_t*>(group_sizes));
    
    auto getri_scratch = H4I::MKLShim::Dgetri_batch_ScPadSz(context, group_count,
                                                             const_cast<int64_t*>(n_values),
                                                             const_cast<int64_t*>(lda_values),
                                                             const_cast<int64_t*>(group_sizes));
    
    auto getrs_scratch = H4I::MKLShim::Dgetrs_batch_ScPadSz(context, group_count,
                                                             trans_values,
                                                             const_cast<int64_t*>(n_values),
                                                             const_cast<int64_t*>(nrhs_values),
                                                             const_cast<int64_t*>(lda_values),
                                                             const_cast<int64_t*>(ldb_values),
                                                             const_cast<int64_t*>(group_sizes));
    
    if (getrf_scratch < 0 || getri_scratch < 0 || getrs_scratch < 0) {
        std::cerr << "Failed to get valid scratchpad sizes" << std::endl;
        return false;
    }
    
    // Allocate device memory
    double* A_device;
    double* B_device;
    int64_t* ipiv_device;
    double* scratch_device;
    double** A_ptrs_device;
    double** B_ptrs_device;
    int64_t** ipiv_ptrs_device;
    
    size_t max_scratch = std::max({getrf_scratch, getri_scratch, getrs_scratch});
    
    hipMalloc(&A_device, n * n * sizeof(double));
    hipMalloc(&B_device, n * nrhs_values[0] * sizeof(double));
    hipMalloc(&ipiv_device, n * sizeof(int64_t));
    hipMalloc(&scratch_device, max_scratch * sizeof(double));
    hipMalloc(&A_ptrs_device, sizeof(double*));
    hipMalloc(&B_ptrs_device, sizeof(double*));
    hipMalloc(&ipiv_ptrs_device, sizeof(int64_t*));
    
    // Create host data
    std::vector<double> A_host(n * n);
    std::vector<double> B_host(n * nrhs_values[0], 1.0);
    createTestMatrix(A_host.data(), n, n, 3.0);
    
    // Copy to device
    hipMemcpy(A_device, A_host.data(), A_host.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(B_device, B_host.data(), B_host.size() * sizeof(double), hipMemcpyHostToDevice);
    
    // Setup pointer arrays
    double* A_ptrs_host[1] = {A_device};
    double* B_ptrs_host[1] = {B_device};
    int64_t* ipiv_ptrs_host[1] = {ipiv_device};
    
    hipMemcpy(A_ptrs_device, A_ptrs_host, sizeof(double*), hipMemcpyHostToDevice);
    hipMemcpy(B_ptrs_device, B_ptrs_host, sizeof(double*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_ptrs_device, ipiv_ptrs_host, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Test all batch functions - should not crash
    H4I::MKLShim::Dgetrf_batch(context, const_cast<int64_t*>(m_values), const_cast<int64_t*>(n_values), 
                                A_ptrs_device, const_cast<int64_t*>(lda_values), ipiv_ptrs_device, 
                                group_count, const_cast<int64_t*>(group_sizes), 
                                scratch_device, getrf_scratch);
    
    hipMemcpy(A_device, A_host.data(), A_host.size() * sizeof(double), hipMemcpyHostToDevice);
    
    H4I::MKLShim::Dgetri_batch(context, const_cast<int64_t*>(n_values), A_ptrs_device,
                                const_cast<int64_t*>(lda_values), ipiv_ptrs_device, group_count,
                                const_cast<int64_t*>(group_sizes), scratch_device, getri_scratch);
    
    hipMemcpy(A_device, A_host.data(), A_host.size() * sizeof(double), hipMemcpyHostToDevice);
    
    H4I::MKLShim::Dgetrs_batch(context, trans_values, const_cast<int64_t*>(n_values),
                                const_cast<int64_t*>(nrhs_values), A_ptrs_device,
                                const_cast<int64_t*>(lda_values), ipiv_ptrs_device,
                                B_ptrs_device, const_cast<int64_t*>(ldb_values),
                                group_count, const_cast<int64_t*>(group_sizes),
                                scratch_device, getrs_scratch);
    
    // Cleanup
    hipFree(A_device);
    hipFree(B_device);
    hipFree(ipiv_device);
    hipFree(scratch_device);
    hipFree(A_ptrs_device);
    hipFree(B_ptrs_device);
    hipFree(ipiv_ptrs_device);
    
    std::cout << "Double precision batch functions smoke test passed" << std::endl;
    return true;
}

bool testComplexBatchSmoke(H4I::MKLShim::Context* context) {
    std::cout << "API Smoke Test: Complex batch functions..." << std::endl;
    
    const int64_t group_count = 1;
    const int64_t n = 2; // Smaller size for complex
    const int64_t m_values[1] = {n};
    const int64_t n_values[1] = {n};
    const int64_t lda_values[1] = {n};
    const int64_t nrhs_values[1] = {1};
    const int64_t ldb_values[1] = {n};
    const int64_t group_sizes[1] = {1};
    H4I::MKLShim::onemklTranspose trans_values[1] = {H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS};
    
    // Test complex float
    auto getrf_scratch_c = H4I::MKLShim::Cgetrf_batch_ScPadSz(context, group_count,
                                                               const_cast<int64_t*>(m_values),
                                                               const_cast<int64_t*>(n_values),
                                                               const_cast<int64_t*>(lda_values),
                                                               const_cast<int64_t*>(group_sizes));
    
    auto getri_scratch_c = H4I::MKLShim::Cgetri_batch_ScPadSz(context, group_count,
                                                               const_cast<int64_t*>(n_values),
                                                               const_cast<int64_t*>(lda_values),
                                                               const_cast<int64_t*>(group_sizes));
    
    auto getrs_scratch_c = H4I::MKLShim::Cgetrs_batch_ScPadSz(context, group_count,
                                                               trans_values,
                                                               const_cast<int64_t*>(n_values),
                                                               const_cast<int64_t*>(nrhs_values),
                                                               const_cast<int64_t*>(lda_values),
                                                               const_cast<int64_t*>(ldb_values),
                                                               const_cast<int64_t*>(group_sizes));
    
    // Test complex double
    auto getrf_scratch_z = H4I::MKLShim::Zgetrf_batch_ScPadSz(context, group_count,
                                                               const_cast<int64_t*>(m_values),
                                                               const_cast<int64_t*>(n_values),
                                                               const_cast<int64_t*>(lda_values),
                                                               const_cast<int64_t*>(group_sizes));
    
    if (getrf_scratch_c < 0 || getri_scratch_c < 0 || getrs_scratch_c < 0 || getrf_scratch_z < 0) {
        std::cerr << "Failed to get valid complex scratchpad sizes" << std::endl;
        return false;
    }
    
    // Quick test with minimal allocation for complex float
    float _Complex* A_c_device;
    float _Complex* B_c_device;
    int64_t* ipiv_c_device;
    float _Complex* scratch_c_device;
    float _Complex** A_c_ptrs_device;
    float _Complex** B_c_ptrs_device;
    int64_t** ipiv_c_ptrs_device;
    
    hipMalloc(&A_c_device, n * n * sizeof(float _Complex));
    hipMalloc(&B_c_device, n * sizeof(float _Complex));
    hipMalloc(&ipiv_c_device, n * sizeof(int64_t));
    hipMalloc(&scratch_c_device, getrf_scratch_c * sizeof(float _Complex));
    hipMalloc(&A_c_ptrs_device, sizeof(float _Complex*));
    hipMalloc(&B_c_ptrs_device, sizeof(float _Complex*));
    hipMalloc(&ipiv_c_ptrs_device, sizeof(int64_t*));
    
    std::vector<float _Complex> A_c_host(n * n);
    createTestMatrix(A_c_host.data(), n, n, 2.0f + 1.0fi);
    hipMemcpy(A_c_device, A_c_host.data(), A_c_host.size() * sizeof(float _Complex), hipMemcpyHostToDevice);
    
    float _Complex* A_c_ptrs_host[1] = {A_c_device};
    hipMemcpy(A_c_ptrs_device, A_c_ptrs_host, sizeof(float _Complex*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_c_ptrs_device, &ipiv_c_device, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Test Cgetrf_batch - should not crash
    H4I::MKLShim::Cgetrf_batch(context, const_cast<int64_t*>(m_values), const_cast<int64_t*>(n_values), 
                                A_c_ptrs_device, const_cast<int64_t*>(lda_values), ipiv_c_ptrs_device, 
                                group_count, const_cast<int64_t*>(group_sizes), 
                                scratch_c_device, getrf_scratch_c);
    
    // Similar quick test for complex double  
    double _Complex* A_z_device;
    double _Complex** A_z_ptrs_device;
    int64_t** ipiv_z_ptrs_device;
    double _Complex* scratch_z_device;
    
    hipMalloc(&A_z_device, n * n * sizeof(double _Complex));
    hipMalloc(&A_z_ptrs_device, sizeof(double _Complex*));
    hipMalloc(&ipiv_z_ptrs_device, sizeof(int64_t*));
    hipMalloc(&scratch_z_device, getrf_scratch_z * sizeof(double _Complex));
    
    std::vector<double _Complex> A_z_host(n * n);
    createTestMatrix(A_z_host.data(), n, n, 3.0 + 2.0i);
    hipMemcpy(A_z_device, A_z_host.data(), A_z_host.size() * sizeof(double _Complex), hipMemcpyHostToDevice);
    
    double _Complex* A_z_ptrs_host[1] = {A_z_device};
    hipMemcpy(A_z_ptrs_device, A_z_ptrs_host, sizeof(double _Complex*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_z_ptrs_device, &ipiv_c_device, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Test Zgetrf_batch - should not crash
    H4I::MKLShim::Zgetrf_batch(context, const_cast<int64_t*>(m_values), const_cast<int64_t*>(n_values), 
                                A_z_ptrs_device, const_cast<int64_t*>(lda_values), ipiv_z_ptrs_device, 
                                group_count, const_cast<int64_t*>(group_sizes), 
                                scratch_z_device, getrf_scratch_z);
    
    // Cleanup
    hipFree(A_c_device);
    hipFree(B_c_device);
    hipFree(ipiv_c_device);
    hipFree(scratch_c_device);
    hipFree(A_c_ptrs_device);
    hipFree(B_c_ptrs_device);
    hipFree(ipiv_c_ptrs_device);
    
    hipFree(A_z_device);
    hipFree(A_z_ptrs_device);
    hipFree(ipiv_z_ptrs_device);
    hipFree(scratch_z_device);
    
    std::cout << "Complex batch functions smoke test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "MKLShim Batch Functions API Smoke Test" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Setup HIP context
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    // Create MKLShim context
    H4I::MKLShim::Context* context = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (context == nullptr) {
        std::cerr << "Failed to create MKLShim context" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "MKLShim context created successfully" << std::endl;
    
    bool allTestsPassed = true;
    
    // Run smoke tests - these only check that functions don't crash
    allTestsPassed &= testSinglePrecisionBatchSmoke(context);
    auto skipIfNoFp64 = [](const char* name, auto fn) -> bool {
        try {
            return fn();
        } catch (const std::exception& e) {
            std::string msg(e.what());
            if (msg.find("fp64 is not supported") != std::string::npos ||
                msg.find("unsupported device") != std::string::npos) {
                std::cout << name << " SKIPPED (device does not support fp64)" << std::endl;
                return true;
            }
            throw;
        }
    };
    allTestsPassed &= skipIfNoFp64("Double precision batch smoke",
        [&]{ return testDoublePrecisionBatchSmoke(context); });
    allTestsPassed &= skipIfNoFp64("Complex batch smoke",
        [&]{ return testComplexBatchSmoke(context); });
    
    // Clean up
    H4I::MKLShim::Destroy(context);
    
    if (allTestsPassed) {
        std::cout << "\n===============================================" << std::endl;
        std::cout << "All batch function API smoke tests passed!" << std::endl;
        std::cout << "Note: These are API tests only - see correctness test for validation" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cerr << "\n===============================================" << std::endl;
        std::cerr << "Some batch function API smoke tests failed!" << std::endl;
        return EXIT_FAILURE;
    }
} 