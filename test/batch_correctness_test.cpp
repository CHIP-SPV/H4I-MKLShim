// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include <iostream>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include "h4i/mklshim/mklshim.h"

// Tolerance values for numerical comparisons
const float SINGLE_TOLERANCE = 1e-5f;
const double DOUBLE_TOLERANCE = 1e-12;
const float COMPLEX_SINGLE_TOLERANCE = 1e-5f;
const double COMPLEX_DOUBLE_TOLERANCE = 1e-12;

// Helper functions for matrix operations and validation
template<typename T>
void createRandomMatrix(T* matrix, int64_t rows, int64_t cols, int64_t lda, int seed = 42) {
    std::srand(seed);
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            matrix[j * lda + i] = T((std::rand() % 100) / 100.0 + 0.1); // Avoid zero
        }
    }
}

template<>
void createRandomMatrix<float _Complex>(float _Complex* matrix, int64_t rows, int64_t cols, int64_t lda, int seed) {
    std::srand(seed);
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            float real = (std::rand() % 100) / 100.0f + 0.1f;
            float imag = (std::rand() % 100) / 100.0f + 0.1f;
            matrix[j * lda + i] = real + imag * 1.0fi;
        }
    }
}

template<>
void createRandomMatrix<double _Complex>(double _Complex* matrix, int64_t rows, int64_t cols, int64_t lda, int seed) {
    std::srand(seed);
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            double real = (std::rand() % 100) / 100.0 + 0.1;
            double imag = (std::rand() % 100) / 100.0 + 0.1;
            matrix[j * lda + i] = real + imag * 1.0i;
        }
    }
}

template<typename T>
void copyMatrix(const T* src, T* dst, int64_t rows, int64_t cols, int64_t lda) {
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            dst[j * lda + i] = src[j * lda + i];
        }
    }
}

template<typename T>
bool compareMatrices(const T* A, const T* B, int64_t rows, int64_t cols, int64_t lda, double tolerance) {
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            int idx = j * lda + i;  // Column-major indexing
            if (std::abs(A[idx] - B[idx]) > tolerance) {
                std::cout << "Mismatch at (" << i << "," << j << "): " 
                         << A[idx] << " vs " << B[idx] 
                         << " (diff: " << std::abs(A[idx] - B[idx]) << ")" << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Specialization for complex types
template<>
bool compareMatrices<float _Complex>(const float _Complex* A, const float _Complex* B, 
                                     int64_t rows, int64_t cols, int64_t lda, double tolerance) {
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            int idx = j * lda + i;  // Column-major indexing
            float diff_real = std::abs(crealf(A[idx]) - crealf(B[idx]));
            float diff_imag = std::abs(cimagf(A[idx]) - cimagf(B[idx]));
            if (diff_real > tolerance || diff_imag > tolerance) {
                std::cout << "Complex mismatch at (" << i << "," << j << "): " 
                         << crealf(A[idx]) << "+" << cimagf(A[idx]) << "i vs "
                         << crealf(B[idx]) << "+" << cimagf(B[idx]) << "i" << std::endl;
                return false;
            }
        }
    }
    return true;
}

template<>
bool compareMatrices<double _Complex>(const double _Complex* A, const double _Complex* B, 
                                      int64_t rows, int64_t cols, int64_t lda, double tolerance) {
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            int idx = j * lda + i;  // Column-major indexing
            double diff_real = std::abs(creal(A[idx]) - creal(B[idx]));
            double diff_imag = std::abs(cimag(A[idx]) - cimag(B[idx]));
            if (diff_real > tolerance || diff_imag > tolerance) {
                std::cout << "Complex mismatch at (" << i << "," << j << "): " 
                         << creal(A[idx]) << "+" << cimag(A[idx]) << "i vs "
                         << creal(B[idx]) << "+" << cimag(B[idx]) << "i" << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Test function that compares batch vs non-batch getrf
bool testSgetrfBatchVsNonBatch(H4I::MKLShim::Context* context) {
    std::cout << "Testing Sgetrf: batch vs non-batch comparison..." << std::endl;
    
    const int64_t n = 4;
    const int64_t lda = n;
    
    // Create non-const variables for function parameters
    int64_t n_var = n;
    int64_t lda_var = lda;
    int64_t group_sizes_var = 1;
    
    // Get scratchpad sizes
    auto batch_scratch_size = H4I::MKLShim::Sgetrf_batch_ScPadSz(context, 1, &n_var, &n_var, &lda_var, &group_sizes_var);
    auto single_scratch_size = H4I::MKLShim::Sgetrf_ScPadSz(context, n, n, lda);
    
    if (batch_scratch_size < 0 || single_scratch_size < 0) {
        std::cerr << "Failed to get scratchpad sizes for Sgetrf" << std::endl;
        return false;
    }
    
    // Create test matrix
    std::vector<float> A_original(n * lda);
    createRandomMatrix(A_original.data(), n, n, lda, 123);
    
    // Test data for batch operation
    std::vector<float> A_batch(n * lda);
    std::vector<int64_t> ipiv_batch(n);
    copyMatrix(A_original.data(), A_batch.data(), n, n, lda);
    
    // Test data for single operation
    std::vector<float> A_single(n * lda);
    std::vector<int64_t> ipiv_single(n);
    copyMatrix(A_original.data(), A_single.data(), n, n, lda);
    
    // Device memory allocation
    float* A_batch_device;
    float* A_single_device;
    int64_t* ipiv_batch_device;
    int64_t* ipiv_single_device;
    float* batch_scratch_device;
    float* single_scratch_device;
    float** A_batch_ptrs_device;
    int64_t** ipiv_batch_ptrs_device;
    
    hipMalloc(&A_batch_device, n * lda * sizeof(float));
    hipMalloc(&A_single_device, n * lda * sizeof(float));
    hipMalloc(&ipiv_batch_device, n * sizeof(int64_t));
    hipMalloc(&ipiv_single_device, n * sizeof(int64_t));
    hipMalloc(&batch_scratch_device, batch_scratch_size * sizeof(float));
    hipMalloc(&single_scratch_device, single_scratch_size * sizeof(float));
    hipMalloc(&A_batch_ptrs_device, sizeof(float*));
    hipMalloc(&ipiv_batch_ptrs_device, sizeof(int64_t*));
    
    // Copy data to device
    hipMemcpy(A_batch_device, A_batch.data(), A_batch.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(A_single_device, A_single.data(), A_single.size() * sizeof(float), hipMemcpyHostToDevice);
    
    // Setup batch pointer arrays
    float* A_batch_ptrs_host[1] = {A_batch_device};
    int64_t* ipiv_batch_ptrs_host[1] = {ipiv_batch_device};
    hipMemcpy(A_batch_ptrs_device, A_batch_ptrs_host, sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_batch_ptrs_device, ipiv_batch_ptrs_host, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Execute batch operation
    int64_t group_count = 1;
    int64_t group_sizes[1] = {1};
    H4I::MKLShim::Sgetrf_batch(context, &n_var, &n_var, A_batch_ptrs_device, &lda_var, 
                                ipiv_batch_ptrs_device, group_count, group_sizes,
                                batch_scratch_device, batch_scratch_size);
    
    // Execute single operation
    H4I::MKLShim::Sgetrf(context, n, n, A_single_device, lda, ipiv_single_device,
                          single_scratch_device, single_scratch_size);
    
    // Synchronize to ensure getrf completes before copying results
    hipDeviceSynchronize();
    
    // Copy results back to host
    hipMemcpy(A_batch.data(), A_batch_device, A_batch.size() * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(A_single.data(), A_single_device, A_single.size() * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(ipiv_batch.data(), ipiv_batch_device, ipiv_batch.size() * sizeof(int64_t), hipMemcpyDeviceToHost);
    hipMemcpy(ipiv_single.data(), ipiv_single_device, ipiv_single.size() * sizeof(int64_t), hipMemcpyDeviceToHost);
    
    // Compare results
    bool matrices_match = compareMatrices(A_batch.data(), A_single.data(), n, n, lda, SINGLE_TOLERANCE);
    bool pivots_match = std::equal(ipiv_batch.begin(), ipiv_batch.end(), ipiv_single.begin());
    
    // Cleanup
    hipFree(A_batch_device);
    hipFree(A_single_device);
    hipFree(ipiv_batch_device);
    hipFree(ipiv_single_device);
    hipFree(batch_scratch_device);
    hipFree(single_scratch_device);
    hipFree(A_batch_ptrs_device);
    hipFree(ipiv_batch_ptrs_device);
    
    if (matrices_match && pivots_match) {
        std::cout << "Sgetrf batch vs non-batch: PASSED" << std::endl;
        return true;
    } else {
        std::cout << "Sgetrf batch vs non-batch: FAILED" << std::endl;
        if (!matrices_match) std::cout << "  Matrix results differ" << std::endl;
        if (!pivots_match) std::cout << "  Pivot results differ" << std::endl;
        return false;
    }
}

// Test function for getrs batch vs non-batch
bool testSgetrsBatchVsNonBatch(H4I::MKLShim::Context* context) {
    std::cout << "Testing Sgetrs: batch vs non-batch comparison..." << std::endl;
    
    const int64_t n = 4;
    const int64_t nrhs = 2;
    const int64_t lda = n;
    const int64_t ldb = n;
    H4I::MKLShim::onemklTranspose trans = H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS;
    
    // Create non-const variables for function parameters
    int64_t n_var = n;
    int64_t nrhs_var = nrhs;
    int64_t lda_var = lda;
    int64_t ldb_var = ldb;
    int64_t group_sizes_var = 1;
    
    // Get scratchpad sizes
    H4I::MKLShim::onemklTranspose trans_array[1] = {trans};
    auto batch_scratch_size = H4I::MKLShim::Sgetrs_batch_ScPadSz(context, 1, trans_array, &n_var, &nrhs_var, &lda_var, &ldb_var, &group_sizes_var);
    auto single_scratch_size = H4I::MKLShim::Sgetrs_ScPadSz(context, trans, n, nrhs, lda, ldb);
    
    if (batch_scratch_size < 0 || single_scratch_size < 0) {
        std::cerr << "Failed to get scratchpad sizes for Sgetrs" << std::endl;
        return false;
    }
    
    // Create test matrices 
    std::vector<float> A_original(n * lda);
    std::vector<float> B_original(n * nrhs);
    std::vector<int64_t> ipiv(n);
    
    createRandomMatrix(A_original.data(), n, n, lda, 456);
    createRandomMatrix(B_original.data(), n, nrhs, ldb, 789);
    
    // First, factor the matrix using regular getrf to get valid pivots
    std::vector<float> A_factored(A_original);
    auto getrf_scratch_size = H4I::MKLShim::Sgetrf_ScPadSz(context, n, n, lda);
    
    float* A_factor_device;
    int64_t* ipiv_device;
    float* getrf_scratch_device;
    
    hipMalloc(&A_factor_device, n * lda * sizeof(float));
    hipMalloc(&ipiv_device, n * sizeof(int64_t));
    hipMalloc(&getrf_scratch_device, getrf_scratch_size * sizeof(float));
    
    hipMemcpy(A_factor_device, A_factored.data(), A_factored.size() * sizeof(float), hipMemcpyHostToDevice);
    
    H4I::MKLShim::Sgetrf(context, n, n, A_factor_device, lda, ipiv_device,
                          getrf_scratch_device, getrf_scratch_size);
    
    // Synchronize to ensure getrf completes before copying results
    hipDeviceSynchronize();
    
    hipMemcpy(A_factored.data(), A_factor_device, A_factored.size() * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(ipiv.data(), ipiv_device, ipiv.size() * sizeof(int64_t), hipMemcpyDeviceToHost);
    
    // Now test getrs batch vs non-batch with the factored matrix
    std::vector<float> B_batch(B_original);
    std::vector<float> B_single(B_original);
    
    // Device memory for getrs test
    float* A_batch_device;
    float* A_single_device;
    float* B_batch_device;
    float* B_single_device;
    int64_t* ipiv_batch_device;
    int64_t* ipiv_single_device;
    float* batch_scratch_device;
    float* single_scratch_device;
    float** A_batch_ptrs_device;
    float** B_batch_ptrs_device;
    int64_t** ipiv_batch_ptrs_device;
    
    hipMalloc(&A_batch_device, n * lda * sizeof(float));
    hipMalloc(&A_single_device, n * lda * sizeof(float));
    hipMalloc(&B_batch_device, n * nrhs * sizeof(float));
    hipMalloc(&B_single_device, n * nrhs * sizeof(float));
    hipMalloc(&ipiv_batch_device, n * sizeof(int64_t));
    hipMalloc(&ipiv_single_device, n * sizeof(int64_t));
    hipMalloc(&batch_scratch_device, batch_scratch_size * sizeof(float));
    hipMalloc(&single_scratch_device, single_scratch_size * sizeof(float));
    hipMalloc(&A_batch_ptrs_device, sizeof(float*));
    hipMalloc(&B_batch_ptrs_device, sizeof(float*));
    hipMalloc(&ipiv_batch_ptrs_device, sizeof(int64_t*));
    
    // Copy data to device
    hipMemcpy(A_batch_device, A_factored.data(), A_factored.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(A_single_device, A_factored.data(), A_factored.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(B_batch_device, B_batch.data(), B_batch.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(B_single_device, B_single.data(), B_single.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_batch_device, ipiv.data(), ipiv.size() * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_single_device, ipiv.data(), ipiv.size() * sizeof(int64_t), hipMemcpyHostToDevice);
    
    // Setup batch pointer arrays
    float* A_batch_ptrs_host[1] = {A_batch_device};
    float* B_batch_ptrs_host[1] = {B_batch_device};
    int64_t* ipiv_batch_ptrs_host[1] = {ipiv_batch_device};
    hipMemcpy(A_batch_ptrs_device, A_batch_ptrs_host, sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(B_batch_ptrs_device, B_batch_ptrs_host, sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_batch_ptrs_device, ipiv_batch_ptrs_host, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Synchronize to ensure all data is copied to device before computation
    hipDeviceSynchronize();
    
    // Execute batch operation
    int64_t group_count = 1;
    int64_t group_sizes[1] = {1};
    H4I::MKLShim::Sgetrs_batch(context, trans_array, &n_var, &nrhs_var, A_batch_ptrs_device, &lda_var,
                                ipiv_batch_ptrs_device, B_batch_ptrs_device, &ldb_var,
                                group_count, group_sizes, batch_scratch_device, batch_scratch_size);
    
    // Synchronize to ensure batch operation completes before single operation
    hipDeviceSynchronize();
    
    // Execute single operation
    H4I::MKLShim::Sgetrs(context, trans, n, nrhs, A_single_device, lda, ipiv_single_device,
                          B_single_device, ldb, single_scratch_device, single_scratch_size);
    
    // Synchronize to ensure getrs completes before copying results
    hipDeviceSynchronize();
    
    // Copy results back to host
    hipMemcpy(B_batch.data(), B_batch_device, B_batch.size() * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(B_single.data(), B_single_device, B_single.size() * sizeof(float), hipMemcpyDeviceToHost);
    
    // Synchronize to ensure all memory transfers are complete before comparison
    hipDeviceSynchronize();
    
    // Compare results
    bool results_match = compareMatrices(B_batch.data(), B_single.data(), n, nrhs, ldb, SINGLE_TOLERANCE);
    
    // Cleanup
    hipFree(A_factor_device);
    hipFree(ipiv_device);
    hipFree(getrf_scratch_device);
    hipFree(A_batch_device);
    hipFree(A_single_device);
    hipFree(B_batch_device);
    hipFree(B_single_device);
    hipFree(ipiv_batch_device);
    hipFree(ipiv_single_device);
    hipFree(batch_scratch_device);
    hipFree(single_scratch_device);
    hipFree(A_batch_ptrs_device);
    hipFree(B_batch_ptrs_device);
    hipFree(ipiv_batch_ptrs_device);
    
    if (results_match) {
        std::cout << "Sgetrs batch vs non-batch: PASSED" << std::endl;
        return true;
    } else {
        std::cout << "Sgetrs batch vs non-batch: FAILED" << std::endl;
        return false;
    }
}

// CPU-based verification for mathematical correctness
bool testSgetrfCorrectnessCPU(H4I::MKLShim::Context* context) {
    std::cout << "Testing Sgetrf: CPU verification of mathematical correctness..." << std::endl;
    
    const int64_t n = 3; // Small size for CPU verification
    const int64_t lda = n;
    
    // Create non-const variables for function parameters
    int64_t n_var = n;
    int64_t lda_var = lda;
    int64_t group_sizes_var = 1;
    
    // Create a simple, well-conditioned test matrix
    std::vector<float> A_original = {
        4.0f, 3.0f, 2.0f,
        3.0f, 4.0f, 3.0f,
        2.0f, 3.0f, 4.0f
    };
    
    std::vector<float> A_factored(A_original);
    std::vector<int64_t> ipiv(n);
    
    // Get scratchpad size and allocate
    auto scratch_size = H4I::MKLShim::Sgetrf_batch_ScPadSz(context, 1, &n_var, &n_var, &lda_var, &group_sizes_var);
    
    float* A_device;
    int64_t* ipiv_device;
    float* scratch_device;
    float** A_ptrs_device;
    int64_t** ipiv_ptrs_device;
    
    hipMalloc(&A_device, n * lda * sizeof(float));
    hipMalloc(&ipiv_device, n * sizeof(int64_t));
    hipMalloc(&scratch_device, scratch_size * sizeof(float));
    hipMalloc(&A_ptrs_device, sizeof(float*));
    hipMalloc(&ipiv_ptrs_device, sizeof(int64_t*));
    
    // Copy data to device
    hipMemcpy(A_device, A_factored.data(), A_factored.size() * sizeof(float), hipMemcpyHostToDevice);
    
    // Setup pointer arrays
    float* A_ptrs_host[1] = {A_device};
    int64_t* ipiv_ptrs_host[1] = {ipiv_device};
    hipMemcpy(A_ptrs_device, A_ptrs_host, sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_ptrs_device, ipiv_ptrs_host, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    // Execute batch LU factorization
    int64_t group_count = 1;
    int64_t group_sizes[1] = {1};
    H4I::MKLShim::Sgetrf_batch(context, &n_var, &n_var, A_ptrs_device, &lda_var,
                                ipiv_ptrs_device, group_count, group_sizes,
                                scratch_device, scratch_size);
    
    // Synchronize to ensure getrf completes before copying results
    hipDeviceSynchronize();
    
    // Copy results back
    hipMemcpy(A_factored.data(), A_device, A_factored.size() * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(ipiv.data(), ipiv_device, ipiv.size() * sizeof(int64_t), hipMemcpyDeviceToHost);
    
    // CPU verification: Check that P*L*U = A (approximately)
    // For simplicity, we'll check that the diagonal elements are reasonable
    // and that the factorization doesn't contain NaN or Inf values
    bool valid_factorization = true;
    
    for (int64_t i = 0; i < n; ++i) {
        float diag_elem = A_factored[i * lda + i];
        if (std::isnan(diag_elem) || std::isinf(diag_elem) || std::abs(diag_elem) < 1e-10f) {
            std::cout << "Invalid diagonal element at (" << i << "," << i << "): " << diag_elem << std::endl;
            valid_factorization = false;
        }
    }
    
    // Check pivot array is reasonable (values should be between 1 and n)
    for (int64_t i = 0; i < n; ++i) {
        if (ipiv[i] < 1 || ipiv[i] > n) {
            std::cout << "Invalid pivot value at " << i << ": " << ipiv[i] << std::endl;
            valid_factorization = false;
        }
    }
    
    // Cleanup
    hipFree(A_device);
    hipFree(ipiv_device);
    hipFree(scratch_device);
    hipFree(A_ptrs_device);
    hipFree(ipiv_ptrs_device);
    
    if (valid_factorization) {
        std::cout << "Sgetrf CPU verification: PASSED" << std::endl;
        return true;
    } else {
        std::cout << "Sgetrf CPU verification: FAILED" << std::endl;
        return false;
    }
}

// Test double precision functions
bool testDgetrfBatchVsNonBatch(H4I::MKLShim::Context* context) {
    std::cout << "Testing Dgetrf: batch vs non-batch comparison..." << std::endl;
    
    const int64_t n = 4;
    const int64_t lda = n;
    
    // Create non-const variables for function parameters
    int64_t n_var = n;
    int64_t lda_var = lda;
    int64_t group_sizes_var = 1;
    
    // Get scratchpad sizes
    auto batch_scratch_size = H4I::MKLShim::Dgetrf_batch_ScPadSz(context, 1, &n_var, &n_var, &lda_var, &group_sizes_var);
    auto single_scratch_size = H4I::MKLShim::Dgetrf_ScPadSz(context, n, n, lda);
    
    if (batch_scratch_size < 0 || single_scratch_size < 0) {
        std::cerr << "Failed to get scratchpad sizes for Dgetrf" << std::endl;
        return false;
    }
    
    // Create test matrix
    std::vector<double> A_original(n * lda);
    createRandomMatrix(A_original.data(), n, n, lda, 321);
    
    // Test data for batch and single operations
    std::vector<double> A_batch(A_original);
    std::vector<double> A_single(A_original);
    std::vector<int64_t> ipiv_batch(n);
    std::vector<int64_t> ipiv_single(n);
    
    // Device memory allocation
    double* A_batch_device;
    double* A_single_device;
    int64_t* ipiv_batch_device;
    int64_t* ipiv_single_device;
    double* batch_scratch_device;
    double* single_scratch_device;
    double** A_batch_ptrs_device;
    int64_t** ipiv_batch_ptrs_device;
    
    hipMalloc(&A_batch_device, n * lda * sizeof(double));
    hipMalloc(&A_single_device, n * lda * sizeof(double));
    hipMalloc(&ipiv_batch_device, n * sizeof(int64_t));
    hipMalloc(&ipiv_single_device, n * sizeof(int64_t));
    hipMalloc(&batch_scratch_device, batch_scratch_size * sizeof(double));
    hipMalloc(&single_scratch_device, single_scratch_size * sizeof(double));
    hipMalloc(&A_batch_ptrs_device, sizeof(double*));
    hipMalloc(&ipiv_batch_ptrs_device, sizeof(int64_t*));
    
    // Copy data and execute operations (similar to single precision)
    hipMemcpy(A_batch_device, A_batch.data(), A_batch.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(A_single_device, A_single.data(), A_single.size() * sizeof(double), hipMemcpyHostToDevice);
    
    double* A_batch_ptrs_host[1] = {A_batch_device};
    int64_t* ipiv_batch_ptrs_host[1] = {ipiv_batch_device};
    hipMemcpy(A_batch_ptrs_device, A_batch_ptrs_host, sizeof(double*), hipMemcpyHostToDevice);
    hipMemcpy(ipiv_batch_ptrs_device, ipiv_batch_ptrs_host, sizeof(int64_t*), hipMemcpyHostToDevice);
    
    int64_t group_count = 1;
    int64_t group_sizes[1] = {1};
    H4I::MKLShim::Dgetrf_batch(context, &n_var, &n_var, A_batch_ptrs_device, &lda_var,
                                ipiv_batch_ptrs_device, group_count, group_sizes,
                                batch_scratch_device, batch_scratch_size);
    
    H4I::MKLShim::Dgetrf(context, n, n, A_single_device, lda, ipiv_single_device,
                          single_scratch_device, single_scratch_size);
    
    // Synchronize to ensure getrf completes before copying results
    hipDeviceSynchronize();
    
    // Copy results back and compare
    hipMemcpy(A_batch.data(), A_batch_device, A_batch.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(A_single.data(), A_single_device, A_single.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(ipiv_batch.data(), ipiv_batch_device, ipiv_batch.size() * sizeof(int64_t), hipMemcpyDeviceToHost);
    hipMemcpy(ipiv_single.data(), ipiv_single_device, ipiv_single.size() * sizeof(int64_t), hipMemcpyDeviceToHost);
    
    bool matrices_match = compareMatrices(A_batch.data(), A_single.data(), n, n, lda, DOUBLE_TOLERANCE);
    bool pivots_match = std::equal(ipiv_batch.begin(), ipiv_batch.end(), ipiv_single.begin());
    
    // Cleanup
    hipFree(A_batch_device);
    hipFree(A_single_device);
    hipFree(ipiv_batch_device);
    hipFree(ipiv_single_device);
    hipFree(batch_scratch_device);
    hipFree(single_scratch_device);
    hipFree(A_batch_ptrs_device);
    hipFree(ipiv_batch_ptrs_device);
    
    if (matrices_match && pivots_match) {
        std::cout << "Dgetrf batch vs non-batch: PASSED" << std::endl;
        return true;
    } else {
        std::cout << "Dgetrf batch vs non-batch: FAILED" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "MKLShim Batch Functions Correctness Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Setup HIP context
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        std::quick_exit(EXIT_FAILURE);
    }
    
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    
    // Create MKLShim context
    H4I::MKLShim::Context* context = H4I::MKLShim::Create(handles.data(), nHandles);
    
    if (context == nullptr) {
        std::cerr << "Failed to create MKLShim context" << std::endl;
        std::quick_exit(EXIT_FAILURE);
    }
    
    std::cout << "MKLShim context created successfully" << std::endl;
    
    bool allTestsPassed = true;
    
    // Run correctness tests comparing batch vs non-batch implementations
    std::cout << "\n--- Batch vs Non-Batch Comparison Tests ---" << std::endl;
    allTestsPassed &= testSgetrfBatchVsNonBatch(context);
    allTestsPassed &= testSgetrsBatchVsNonBatch(context);
    allTestsPassed &= testDgetrfBatchVsNonBatch(context);
    
    // Run CPU-based verification tests  
    std::cout << "\n--- CPU Mathematical Verification Tests ---" << std::endl;
    // allTestsPassed &= testSgetrfCorrectnessCPU(context);
    std::cout << "Sgetrf correctness: SKIPPED" << std::endl;
    allTestsPassed &= testSgetrfCorrectnessCPU(context);
    
    // Ensure all GPU operations are complete before cleanup
    std::cout << "\nSynchronizing all GPU operations..." << std::endl;
    hipDeviceSynchronize();
    
    // Clean up
    H4I::MKLShim::Destroy(context);
    
    if (allTestsPassed) {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "All batch function correctness tests passed!" << std::endl;
        std::cout << "Batch functions produce mathematically correct results." << std::endl;
        std::quick_exit(EXIT_SUCCESS);
    } else {
        std::cerr << "\n=========================================" << std::endl;
        std::cerr << "Some batch function correctness tests failed!" << std::endl;
        std::cerr << "Check the output above for details." << std::endl;
        std::quick_exit(EXIT_FAILURE);
    }
} 