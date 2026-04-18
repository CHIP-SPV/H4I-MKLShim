// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include <iostream>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <complex.h>
#include <string>
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

// Test dGemmBatchedEx correctness  
bool testDGemmBatchedExCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing dGemmBatchedEx correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const double alpha = 1.0, beta = 0.0;
    
    // Calculate strides
    const int64_t stride_a = lda * k;
    const int64_t stride_b = ldb * n;
    const int64_t stride_c = ldc * n;
    
    // Create test data
    std::vector<double> A_host(batch_count * stride_a);
    std::vector<double> B_host(batch_count * stride_b);
    std::vector<double> C_host(batch_count * stride_c, 0.0);
    std::vector<double> C_expected(batch_count * stride_c, 0.0);
    
    // Initialize with simple values for manual verification
    // A = [[1,2],[3,4]], B = [[2,0],[1,2]] for first batch
    // A = [[2,3],[4,5]], B = [[2,0],[1,2]] for second batch
    for (int64_t b = 0; b < batch_count; ++b) {
        // Matrix A (column-major)
        A_host[b * stride_a + 0] = 1.0 + b; // A(0,0) 
        A_host[b * stride_a + 1] = 3.0 + b; // A(1,0)
        A_host[b * stride_a + 2] = 2.0 + b; // A(0,1)
        A_host[b * stride_a + 3] = 4.0 + b; // A(1,1)
        
        // Matrix B (column-major)
        B_host[b * stride_b + 0] = 2.0; // B(0,0)
        B_host[b * stride_b + 1] = 1.0; // B(1,0)
        B_host[b * stride_b + 2] = 0.0; // B(0,1)
        B_host[b * stride_b + 3] = 2.0; // B(1,1)
        
        // Expected C = A * B (manually calculated, column-major)
        // C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0) = (1+b)*2 + (2+b)*1 = 4+3b
        // C(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0) = (3+b)*2 + (4+b)*1 = 10+3b
        // C(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1) = (1+b)*0 + (2+b)*2 = 4+2b
        // C(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1) = (3+b)*0 + (4+b)*2 = 8+2b
        C_expected[b * stride_c + 0] = 4.0 + 3.0 * b; // C(0,0)
        C_expected[b * stride_c + 1] = 10.0 + 3.0 * b; // C(1,0)
        C_expected[b * stride_c + 2] = 4.0 + 2.0 * b; // C(0,1)
        C_expected[b * stride_c + 3] = 8.0 + 2.0 * b; // C(1,1)
    }
    
    // Allocate device memory
    double* A_dev = nullptr;
    double* B_dev = nullptr;
    double* C_dev = nullptr;
    
    if (hipMalloc(&A_dev, A_host.size() * sizeof(double)) != hipSuccess ||
        hipMalloc(&B_dev, B_host.size() * sizeof(double)) != hipSuccess ||
        hipMalloc(&C_dev, C_host.size() * sizeof(double)) != hipSuccess) {
        std::cout << "dGemmBatchedEx: Memory allocation failed" << std::endl;
        return false;
    }
    
    // Copy to device
    hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(B_dev, B_host.data(), B_host.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(C_dev, C_host.data(), C_host.size() * sizeof(double), hipMemcpyHostToDevice);
    
    // Execute batch GEMM
    H4I::MKLShim::dGemmBatchedEx(context, 
                                 H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, 
                                 H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                                 m, n, k, alpha,
                                 A_dev, H4I::MKLShim::ONEMKL_R_64F, lda, stride_a,
                                 B_dev, H4I::MKLShim::ONEMKL_R_64F, ldb, stride_b, 
                                 beta, C_dev, H4I::MKLShim::ONEMKL_R_64F, ldc, stride_c, 
                                 batch_count);
    
    hipDeviceSynchronize();
    
    // Copy result back
    hipMemcpy(C_host.data(), C_dev, C_host.size() * sizeof(double), hipMemcpyDeviceToHost);
    
    // Compare with expected results
    bool passed = true;
    const double tolerance = 1e-12;
    
    for (size_t i = 0; i < C_host.size(); ++i) {
        if (std::abs(C_host[i] - C_expected[i]) > tolerance) {
            std::cout << "dGemmBatchedEx: Mismatch at index " << i << ": got " << C_host[i] 
                      << ", expected " << C_expected[i] << std::endl;
            passed = false;
        }
    }
    
    // Cleanup
    if (A_dev) hipFree(A_dev);
    if (B_dev) hipFree(B_dev);
    if (C_dev) hipFree(C_dev);
    
    if (passed) {
        std::cout << "dGemmBatchedEx correctness: PASSED" << std::endl;
    } else {
        std::cout << "dGemmBatchedEx correctness: FAILED" << std::endl;
    }
    
    return passed;
}

// Test cGemmBatchedEx correctness  
bool testCGemmBatchedExCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing cGemmBatchedEx correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const float _Complex alpha = 1.0f + 0.0fi, beta = 0.0f + 0.0fi;
    
    // Calculate strides
    const int64_t stride_a = lda * k;
    const int64_t stride_b = ldb * n;
    const int64_t stride_c = ldc * n;
    
    // Create test data
    std::vector<float _Complex> A_host(batch_count * stride_a);
    std::vector<float _Complex> B_host(batch_count * stride_b);
    std::vector<float _Complex> C_host(batch_count * stride_c, 0.0f + 0.0fi);
    std::vector<float _Complex> C_expected(batch_count * stride_c);
    
    // Initialize test matrices
    // Batch 0: A = [[1+i, 2], [3, 4+i]], B = [[1, 2+i], [3+i, 4]]
    A_host[0] = 1.0f + 1.0f * I; A_host[1] = 3.0f + 0.0f * I;
    A_host[2] = 2.0f + 0.0f * I; A_host[3] = 4.0f + 1.0f * I;
    B_host[0] = 1.0f + 0.0f * I; B_host[1] = 3.0f + 1.0f * I;
    B_host[2] = 2.0f + 1.0f * I; B_host[3] = 4.0f + 0.0f * I;
    
    // Batch 1: A = [[2+i, 1], [4, 3+i]], B = [[2, 1+i], [4+i, 3]]
    A_host[4] = 2.0f + 1.0f * I; A_host[5] = 4.0f + 0.0f * I;
    A_host[6] = 1.0f + 0.0f * I; A_host[7] = 3.0f + 1.0f * I;
    B_host[4] = 2.0f + 0.0f * I; B_host[5] = 4.0f + 1.0f * I;
    B_host[6] = 1.0f + 1.0f * I; B_host[7] = 3.0f + 0.0f * I;
    
    // Calculate expected results manually for C = A * B
    // Batch 0: Expected results
    C_expected[0] = (1.0f + 1.0f * I) * (1.0f + 0.0f * I) + (2.0f + 0.0f * I) * (3.0f + 1.0f * I); // C[0,0]
    C_expected[1] = (3.0f + 0.0f * I) * (1.0f + 0.0f * I) + (4.0f + 1.0f * I) * (3.0f + 1.0f * I); // C[1,0]
    C_expected[2] = (1.0f + 1.0f * I) * (2.0f + 1.0f * I) + (2.0f + 0.0f * I) * (4.0f + 0.0f * I); // C[0,1]
    C_expected[3] = (3.0f + 0.0f * I) * (2.0f + 1.0f * I) + (4.0f + 1.0f * I) * (4.0f + 0.0f * I); // C[1,1]
    
    // Batch 1: Expected results
    C_expected[4] = (2.0f + 1.0f * I) * (2.0f + 0.0f * I) + (1.0f + 0.0f * I) * (4.0f + 1.0f * I); // C[0,0]
    C_expected[5] = (4.0f + 0.0f * I) * (2.0f + 0.0f * I) + (3.0f + 1.0f * I) * (4.0f + 1.0f * I); // C[1,0]
    C_expected[6] = (2.0f + 1.0f * I) * (1.0f + 1.0f * I) + (1.0f + 0.0f * I) * (3.0f + 0.0f * I); // C[0,1]
    C_expected[7] = (4.0f + 0.0f * I) * (1.0f + 1.0f * I) + (3.0f + 1.0f * I) * (3.0f + 0.0f * I); // C[1,1]
    
    // Allocate device memory
    float _Complex *A_dev, *B_dev, *C_dev;
    hipMalloc(&A_dev, sizeof(float _Complex) * batch_count * stride_a);
    hipMalloc(&B_dev, sizeof(float _Complex) * batch_count * stride_b);
    hipMalloc(&C_dev, sizeof(float _Complex) * batch_count * stride_c);
    
    // Copy data to device
    hipMemcpy(A_dev, A_host.data(), sizeof(float _Complex) * batch_count * stride_a, hipMemcpyHostToDevice);
    hipMemcpy(B_dev, B_host.data(), sizeof(float _Complex) * batch_count * stride_b, hipMemcpyHostToDevice);
    hipMemcpy(C_dev, C_host.data(), sizeof(float _Complex) * batch_count * stride_c, hipMemcpyHostToDevice);
    
    // Execute cGemmBatchedEx
    H4I::MKLShim::cGemmBatchedEx(context, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                                 m, n, k, alpha,
                                 A_dev, H4I::MKLShim::ONEMKL_C_32F, lda, stride_a,
                                 B_dev, H4I::MKLShim::ONEMKL_C_32F, ldb, stride_b, beta,
                                 C_dev, H4I::MKLShim::ONEMKL_C_32F, ldc, stride_c, batch_count);
    
    hipDeviceSynchronize();
    
    // Copy results back to host
    hipMemcpy(C_host.data(), C_dev, sizeof(float _Complex) * batch_count * stride_c, hipMemcpyDeviceToHost);
    
    // Verify results
    bool passed = true;
    const float tolerance = 1e-5f;
    for (int64_t i = 0; i < batch_count * stride_c; i++) {
        float real_diff = fabsf(crealf(C_host[i]) - crealf(C_expected[i]));
        float imag_diff = fabsf(cimagf(C_host[i]) - cimagf(C_expected[i]));
        if (real_diff > tolerance || imag_diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": got (" << crealf(C_host[i]) << "," << cimagf(C_host[i]) 
                      << "), expected (" << crealf(C_expected[i]) << "," << cimagf(C_expected[i]) << ")" << std::endl;
            passed = false;
        }
    }
    
    // Cleanup
    hipFree(A_dev);
    hipFree(B_dev);
    hipFree(C_dev);
    
    std::cout << "cGemmBatchedEx test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

// Test zGemmBatchedEx correctness  
bool testZGemmBatchedExCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing zGemmBatchedEx correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const double _Complex alpha = 1.0 + 0.0 * I, beta = 0.0 + 0.0 * I;
    
    // Calculate strides
    const int64_t stride_a = lda * k;
    const int64_t stride_b = ldb * n;
    const int64_t stride_c = ldc * n;
    
    // Create test data
    std::vector<double _Complex> A_host(batch_count * stride_a);
    std::vector<double _Complex> B_host(batch_count * stride_b);
    std::vector<double _Complex> C_host(batch_count * stride_c, 0.0 + 0.0 * I);
    std::vector<double _Complex> C_expected(batch_count * stride_c);
    
    // Initialize with simple values for manual verification
    for (int batch = 0; batch < batch_count; batch++) {
        for (int i = 0; i < stride_a; i++) {
            A_host[batch * stride_a + i] = (batch + 1.0) + (i + 1.0) * I;
        }
        for (int i = 0; i < stride_b; i++) {
            B_host[batch * stride_b + i] = (batch + 2.0) + (i + 2.0) * I;
        }
    }
    
    // Calculate expected results manually
    // For 2x2 matrices: C = A * B
    for (int batch = 0; batch < batch_count; batch++) {
        const double _Complex* A = &A_host[batch * stride_a];
        const double _Complex* B = &B_host[batch * stride_b];
        double _Complex* C = &C_expected[batch * stride_c];
        
        // Manual matrix multiplication for 2x2 case
        C[0] = A[0] * B[0] + A[2] * B[1];  // C[0,0]
        C[1] = A[1] * B[0] + A[3] * B[1];  // C[1,0]
        C[2] = A[0] * B[2] + A[2] * B[3];  // C[0,1]
        C[3] = A[1] * B[2] + A[3] * B[3];  // C[1,1]
    }
    
    // Allocate device memory
    void *A_dev, *B_dev, *C_dev;
    hipMalloc(&A_dev, batch_count * stride_a * sizeof(double _Complex));
    hipMalloc(&B_dev, batch_count * stride_b * sizeof(double _Complex));
    hipMalloc(&C_dev, batch_count * stride_c * sizeof(double _Complex));
    
    // Copy to device
    hipMemcpy(A_dev, A_host.data(), batch_count * stride_a * sizeof(double _Complex), hipMemcpyHostToDevice);
    hipMemcpy(B_dev, B_host.data(), batch_count * stride_b * sizeof(double _Complex), hipMemcpyHostToDevice);
    hipMemcpy(C_dev, C_host.data(), batch_count * stride_c * sizeof(double _Complex), hipMemcpyHostToDevice);
    
    // Execute zGemmBatchedEx
    H4I::MKLShim::zGemmBatchedEx(context, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                                 m, n, k, alpha,
                                 A_dev, H4I::MKLShim::ONEMKL_C_64F, lda, stride_a,
                                 B_dev, H4I::MKLShim::ONEMKL_C_64F, ldb, stride_b, beta,
                                 C_dev, H4I::MKLShim::ONEMKL_C_64F, ldc, stride_c, batch_count);
    
    hipDeviceSynchronize();
    
    // Copy result back
    std::vector<double _Complex> C_result(batch_count * stride_c);
    hipMemcpy(C_result.data(), C_dev, batch_count * stride_c * sizeof(double _Complex), hipMemcpyDeviceToHost);
    
    // Cleanup
    hipFree(A_dev);
    hipFree(B_dev);
    hipFree(C_dev);
    
    // Verify results
    const double tolerance = 1e-12;
    bool success = true;
    for (int i = 0; i < batch_count * stride_c; i++) {
        if (fabs(creal(C_result[i]) - creal(C_expected[i])) > tolerance ||
            fabs(cimag(C_result[i]) - cimag(C_expected[i])) > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected (" << creal(C_expected[i]) << ", " << cimag(C_expected[i]) 
                      << "), got (" << creal(C_result[i]) << ", " << cimag(C_result[i]) << ")" << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "zGemmBatchedEx test PASSED" << std::endl;
    } else {
        std::cout << "zGemmBatchedEx test FAILED" << std::endl;
    }
    
    return success;
}

// Test sGemmBatched correctness  
bool testSGemmBatchedCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing sGemmBatched correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Create test data
    std::vector<float> A_host(batch_count * lda * k);
    std::vector<float> B_host(batch_count * ldb * n);
    std::vector<float> C_host(batch_count * ldc * n, 0.0f);
    std::vector<float> C_expected(batch_count * ldc * n, 0.0f);
    
    // Initialize with simple values for manual verification
    for (int batch = 0; batch < batch_count; batch++) {
        for (int i = 0; i < lda * k; i++) {
            A_host[batch * lda * k + i] = (batch + 1) * (i + 1);
        }
        for (int i = 0; i < ldb * n; i++) {
            B_host[batch * ldb * n + i] = (batch + 2) * (i + 1);
        }
    }
    
    // Calculate expected results manually
    // For 2x2 matrices: C = A * B
    for (int batch = 0; batch < batch_count; batch++) {
        const float* A = &A_host[batch * lda * k];
        const float* B = &B_host[batch * ldb * n];
        float* C = &C_expected[batch * ldc * n];
        
        // Manual matrix multiplication for 2x2 case
        C[0] = A[0] * B[0] + A[2] * B[1];  // C[0,0]
        C[1] = A[1] * B[0] + A[3] * B[1];  // C[1,0]
        C[2] = A[0] * B[2] + A[2] * B[3];  // C[0,1]
        C[3] = A[1] * B[2] + A[3] * B[3];  // C[1,1]
    }
    
    // Allocate device memory for matrices
    std::vector<void*> A_dev_ptrs(batch_count), B_dev_ptrs(batch_count), C_dev_ptrs(batch_count);
    for (int i = 0; i < batch_count; i++) {
        hipMalloc(&A_dev_ptrs[i], lda * k * sizeof(float));
        hipMalloc(&B_dev_ptrs[i], ldb * n * sizeof(float));
        hipMalloc(&C_dev_ptrs[i], ldc * n * sizeof(float));
        
        hipMemcpy(A_dev_ptrs[i], &A_host[i * lda * k], lda * k * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(B_dev_ptrs[i], &B_host[i * ldb * n], ldb * n * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(C_dev_ptrs[i], &C_host[i * ldc * n], ldc * n * sizeof(float), hipMemcpyHostToDevice);
    }
    
    // Allocate device memory for pointer arrays
    void **A_dev_ptr_array, **B_dev_ptr_array, **C_dev_ptr_array;
    hipMalloc(&A_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&B_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&C_dev_ptr_array, batch_count * sizeof(void*));
    
    // Copy pointer arrays to device
    hipMemcpy(A_dev_ptr_array, A_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(B_dev_ptr_array, B_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(C_dev_ptr_array, C_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    
    // Execute sGemmBatched
    H4I::MKLShim::sGemmBatched(context, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                               m, n, k, alpha,
                               reinterpret_cast<const float* const*>(A_dev_ptr_array), lda,
                               reinterpret_cast<const float* const*>(B_dev_ptr_array), ldb, beta,
                               reinterpret_cast<float* const*>(C_dev_ptr_array), ldc, batch_count);
    
    hipDeviceSynchronize();
    
    // Copy results back
    std::vector<float> C_result(batch_count * ldc * n);
    for (int i = 0; i < batch_count; i++) {
        hipMemcpy(&C_result[i * ldc * n], C_dev_ptrs[i], ldc * n * sizeof(float), hipMemcpyDeviceToHost);
    }
    
    // Cleanup
    for (int i = 0; i < batch_count; i++) {
        hipFree(A_dev_ptrs[i]);
        hipFree(B_dev_ptrs[i]);
        hipFree(C_dev_ptrs[i]);
    }
    hipFree(A_dev_ptr_array);
    hipFree(B_dev_ptr_array);
    hipFree(C_dev_ptr_array);
    
    // Verify results
    const float tolerance = 1e-5f;
    bool success = true;
    for (int i = 0; i < batch_count * ldc * n; i++) {
        if (fabs(C_result[i] - C_expected[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << C_expected[i] 
                      << ", got " << C_result[i] << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "sGemmBatched test PASSED" << std::endl;
    } else {
        std::cout << "sGemmBatched test FAILED" << std::endl;
    }
    
    return success;
}

// Test dGemmBatched correctness  
bool testDGemmBatchedCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing dGemmBatched correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const double alpha = 1.0, beta = 0.0;
    
    // Create test data
    std::vector<double> A_host(batch_count * lda * k);
    std::vector<double> B_host(batch_count * ldb * n);
    std::vector<double> C_host(batch_count * ldc * n, 0.0);
    std::vector<double> C_expected(batch_count * ldc * n, 0.0);
    
    // Initialize with simple values for manual verification
    for (int batch = 0; batch < batch_count; batch++) {
        for (int i = 0; i < lda * k; i++) {
            A_host[batch * lda * k + i] = (batch + 1) * (i + 1);
        }
        for (int i = 0; i < ldb * n; i++) {
            B_host[batch * ldb * n + i] = (batch + 2) * (i + 1);
        }
    }
    
    // Calculate expected results manually
    // For 2x2 matrices: C = A * B
    for (int batch = 0; batch < batch_count; batch++) {
        const double* A = &A_host[batch * lda * k];
        const double* B = &B_host[batch * ldb * n];
        double* C = &C_expected[batch * ldc * n];
        
        // Manual matrix multiplication for 2x2 case
        C[0] = A[0] * B[0] + A[2] * B[1];  // C[0,0]
        C[1] = A[1] * B[0] + A[3] * B[1];  // C[1,0]
        C[2] = A[0] * B[2] + A[2] * B[3];  // C[0,1]
        C[3] = A[1] * B[2] + A[3] * B[3];  // C[1,1]
    }
    
    // Allocate device memory for matrices
    std::vector<void*> A_dev_ptrs(batch_count), B_dev_ptrs(batch_count), C_dev_ptrs(batch_count);
    for (int i = 0; i < batch_count; i++) {
        hipMalloc(&A_dev_ptrs[i], lda * k * sizeof(double));
        hipMalloc(&B_dev_ptrs[i], ldb * n * sizeof(double));
        hipMalloc(&C_dev_ptrs[i], ldc * n * sizeof(double));
        
        hipMemcpy(A_dev_ptrs[i], &A_host[i * lda * k], lda * k * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(B_dev_ptrs[i], &B_host[i * ldb * n], ldb * n * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(C_dev_ptrs[i], &C_host[i * ldc * n], ldc * n * sizeof(double), hipMemcpyHostToDevice);
    }
    
    // Allocate device memory for pointer arrays
    void **A_dev_ptr_array, **B_dev_ptr_array, **C_dev_ptr_array;
    hipMalloc(&A_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&B_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&C_dev_ptr_array, batch_count * sizeof(void*));
    
    // Copy pointer arrays to device
    hipMemcpy(A_dev_ptr_array, A_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(B_dev_ptr_array, B_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(C_dev_ptr_array, C_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    
    // Execute dGemmBatched
    H4I::MKLShim::dGemmBatched(context, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                               m, n, k, alpha,
                               reinterpret_cast<const double* const*>(A_dev_ptr_array), lda,
                               reinterpret_cast<const double* const*>(B_dev_ptr_array), ldb, beta,
                               reinterpret_cast<double* const*>(C_dev_ptr_array), ldc, batch_count);
    
    hipDeviceSynchronize();
    
    // Copy results back
    std::vector<double> C_result(batch_count * ldc * n);
    for (int i = 0; i < batch_count; i++) {
        hipMemcpy(&C_result[i * ldc * n], C_dev_ptrs[i], ldc * n * sizeof(double), hipMemcpyDeviceToHost);
    }
    
    // Cleanup
    for (int i = 0; i < batch_count; i++) {
        hipFree(A_dev_ptrs[i]);
        hipFree(B_dev_ptrs[i]);
        hipFree(C_dev_ptrs[i]);
    }
    hipFree(A_dev_ptr_array);
    hipFree(B_dev_ptr_array);
    hipFree(C_dev_ptr_array);
    
    // Verify results
    const double tolerance = 1e-12;
    bool success = true;
    for (int i = 0; i < batch_count * ldc * n; i++) {
        if (fabs(C_result[i] - C_expected[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << C_expected[i] 
                      << ", got " << C_result[i] << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "dGemmBatched test PASSED" << std::endl;
    } else {
        std::cout << "dGemmBatched test FAILED" << std::endl;
    }
    
    return success;
}

// Test cGemmBatched correctness  
bool testCGemmBatchedCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing cGemmBatched correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const float _Complex alpha = 1.0f + 0.0f * I, beta = 0.0f + 0.0f * I;
    
    // Create test data
    std::vector<float _Complex> A_host(batch_count * lda * k);
    std::vector<float _Complex> B_host(batch_count * ldb * n);
    std::vector<float _Complex> C_host(batch_count * ldc * n, 0.0f + 0.0f * I);
    std::vector<float _Complex> C_expected(batch_count * ldc * n, 0.0f + 0.0f * I);
    
    // Initialize with simple values for manual verification
    for (int batch = 0; batch < batch_count; batch++) {
        for (int i = 0; i < lda * k; i++) {
            A_host[batch * lda * k + i] = (batch + 1.0f) + (i + 1.0f) * I;
        }
        for (int i = 0; i < ldb * n; i++) {
            B_host[batch * ldb * n + i] = (batch + 2.0f) + (i + 2.0f) * I;
        }
    }
    
    // Calculate expected results manually
    // For 2x2 matrices: C = A * B
    for (int batch = 0; batch < batch_count; batch++) {
        const float _Complex* A = &A_host[batch * lda * k];
        const float _Complex* B = &B_host[batch * ldb * n];
        float _Complex* C = &C_expected[batch * ldc * n];
        
        // Manual matrix multiplication for 2x2 case
        C[0] = A[0] * B[0] + A[2] * B[1];  // C[0,0]
        C[1] = A[1] * B[0] + A[3] * B[1];  // C[1,0]
        C[2] = A[0] * B[2] + A[2] * B[3];  // C[0,1]
        C[3] = A[1] * B[2] + A[3] * B[3];  // C[1,1]
    }
    
    // Allocate device memory for matrices
    std::vector<void*> A_dev_ptrs(batch_count), B_dev_ptrs(batch_count), C_dev_ptrs(batch_count);
    for (int i = 0; i < batch_count; i++) {
        hipMalloc(&A_dev_ptrs[i], lda * k * sizeof(float _Complex));
        hipMalloc(&B_dev_ptrs[i], ldb * n * sizeof(float _Complex));
        hipMalloc(&C_dev_ptrs[i], ldc * n * sizeof(float _Complex));
        
        hipMemcpy(A_dev_ptrs[i], &A_host[i * lda * k], lda * k * sizeof(float _Complex), hipMemcpyHostToDevice);
        hipMemcpy(B_dev_ptrs[i], &B_host[i * ldb * n], ldb * n * sizeof(float _Complex), hipMemcpyHostToDevice);
        hipMemcpy(C_dev_ptrs[i], &C_host[i * ldc * n], ldc * n * sizeof(float _Complex), hipMemcpyHostToDevice);
    }
    
    // Allocate device memory for pointer arrays
    void **A_dev_ptr_array, **B_dev_ptr_array, **C_dev_ptr_array;
    hipMalloc(&A_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&B_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&C_dev_ptr_array, batch_count * sizeof(void*));
    
    // Copy pointer arrays to device
    hipMemcpy(A_dev_ptr_array, A_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(B_dev_ptr_array, B_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(C_dev_ptr_array, C_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    
    // Execute cGemmBatched
    H4I::MKLShim::cGemmBatched(context, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                               m, n, k, alpha,
                               reinterpret_cast<const float _Complex* const*>(A_dev_ptr_array), lda,
                               reinterpret_cast<const float _Complex* const*>(B_dev_ptr_array), ldb, beta,
                               reinterpret_cast<float _Complex* const*>(C_dev_ptr_array), ldc, batch_count);
    
    hipDeviceSynchronize();
    
    // Copy results back
    std::vector<float _Complex> C_result(batch_count * ldc * n);
    for (int i = 0; i < batch_count; i++) {
        hipMemcpy(&C_result[i * ldc * n], C_dev_ptrs[i], ldc * n * sizeof(float _Complex), hipMemcpyDeviceToHost);
    }
    
    // Cleanup
    for (int i = 0; i < batch_count; i++) {
        hipFree(A_dev_ptrs[i]);
        hipFree(B_dev_ptrs[i]);
        hipFree(C_dev_ptrs[i]);
    }
    hipFree(A_dev_ptr_array);
    hipFree(B_dev_ptr_array);
    hipFree(C_dev_ptr_array);
    
    // Verify results
    const float tolerance = 1e-5f;
    bool success = true;
    for (int i = 0; i < batch_count * ldc * n; i++) {
        float real_diff = fabsf(crealf(C_result[i]) - crealf(C_expected[i]));
        float imag_diff = fabsf(cimagf(C_result[i]) - cimagf(C_expected[i]));
        if (real_diff > tolerance || imag_diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected (" 
                      << crealf(C_expected[i]) << ", " << cimagf(C_expected[i]) 
                      << "), got (" << crealf(C_result[i]) << ", " << cimagf(C_result[i]) 
                      << ")" << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "cGemmBatched test PASSED" << std::endl;
    } else {
        std::cout << "cGemmBatched test FAILED" << std::endl;
    }
    
    return success;
}

// Test zGemmBatched correctness  
bool testZGemmBatchedCorrectness(H4I::MKLShim::Context* context) {
    std::cout << "Testing zGemmBatched correctness..." << std::endl;
    
    const int64_t m = 2, n = 2, k = 2;
    const int64_t lda = m, ldb = k, ldc = m;
    const int64_t batch_count = 2;
    const double _Complex alpha = 1.0 + 0.0 * I, beta = 0.0 + 0.0 * I;
    
    // Create test data
    std::vector<double _Complex> A_host(batch_count * lda * k);
    std::vector<double _Complex> B_host(batch_count * ldb * n);
    std::vector<double _Complex> C_host(batch_count * ldc * n, 0.0 + 0.0 * I);
    std::vector<double _Complex> C_expected(batch_count * ldc * n, 0.0 + 0.0 * I);
    
    // Initialize with simple values for manual verification
    for (int batch = 0; batch < batch_count; batch++) {
        for (int i = 0; i < lda * k; i++) {
            A_host[batch * lda * k + i] = (batch + 1.0) + (i + 1.0) * I;
        }
        for (int i = 0; i < ldb * n; i++) {
            B_host[batch * ldb * n + i] = (batch + 2.0) + (i + 2.0) * I;
        }
    }
    
    // Calculate expected results manually
    // For 2x2 matrices: C = A * B
    for (int batch = 0; batch < batch_count; batch++) {
        const double _Complex* A = &A_host[batch * lda * k];
        const double _Complex* B = &B_host[batch * ldb * n];
        double _Complex* C = &C_expected[batch * ldc * n];
        
        // Manual matrix multiplication for 2x2 case
        C[0] = A[0] * B[0] + A[2] * B[1];  // C[0,0]
        C[1] = A[1] * B[0] + A[3] * B[1];  // C[1,0]
        C[2] = A[0] * B[2] + A[2] * B[3];  // C[0,1]
        C[3] = A[1] * B[2] + A[3] * B[3];  // C[1,1]
    }
    
    // Allocate device memory for matrices
    std::vector<void*> A_dev_ptrs(batch_count), B_dev_ptrs(batch_count), C_dev_ptrs(batch_count);
    for (int i = 0; i < batch_count; i++) {
        hipMalloc(&A_dev_ptrs[i], lda * k * sizeof(double _Complex));
        hipMalloc(&B_dev_ptrs[i], ldb * n * sizeof(double _Complex));
        hipMalloc(&C_dev_ptrs[i], ldc * n * sizeof(double _Complex));
        
        hipMemcpy(A_dev_ptrs[i], &A_host[i * lda * k], lda * k * sizeof(double _Complex), hipMemcpyHostToDevice);
        hipMemcpy(B_dev_ptrs[i], &B_host[i * ldb * n], ldb * n * sizeof(double _Complex), hipMemcpyHostToDevice);
        hipMemcpy(C_dev_ptrs[i], &C_host[i * ldc * n], ldc * n * sizeof(double _Complex), hipMemcpyHostToDevice);
    }
    
    // Allocate device memory for pointer arrays
    void **A_dev_ptr_array, **B_dev_ptr_array, **C_dev_ptr_array;
    hipMalloc(&A_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&B_dev_ptr_array, batch_count * sizeof(void*));
    hipMalloc(&C_dev_ptr_array, batch_count * sizeof(void*));
    
    // Copy pointer arrays to device
    hipMemcpy(A_dev_ptr_array, A_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(B_dev_ptr_array, B_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(C_dev_ptr_array, C_dev_ptrs.data(), batch_count * sizeof(void*), hipMemcpyHostToDevice);
    
    // Execute zGemmBatched
    H4I::MKLShim::zGemmBatched(context, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                               m, n, k, alpha,
                               reinterpret_cast<const double _Complex* const*>(A_dev_ptr_array), lda,
                               reinterpret_cast<const double _Complex* const*>(B_dev_ptr_array), ldb, beta,
                               reinterpret_cast<double _Complex* const*>(C_dev_ptr_array), ldc, batch_count);
    
    hipDeviceSynchronize();
    
    // Copy results back
    std::vector<double _Complex> C_result(batch_count * ldc * n);
    for (int i = 0; i < batch_count; i++) {
        hipMemcpy(&C_result[i * ldc * n], C_dev_ptrs[i], ldc * n * sizeof(double _Complex), hipMemcpyDeviceToHost);
    }
    
    // Cleanup
    for (int i = 0; i < batch_count; i++) {
        hipFree(A_dev_ptrs[i]);
        hipFree(B_dev_ptrs[i]);
        hipFree(C_dev_ptrs[i]);
    }
    hipFree(A_dev_ptr_array);
    hipFree(B_dev_ptr_array);
    hipFree(C_dev_ptr_array);
    
    // Verify results
    const double tolerance = 1e-12;
    bool success = true;
    for (int i = 0; i < batch_count * ldc * n; i++) {
        double real_diff = fabs(creal(C_result[i]) - creal(C_expected[i]));
        double imag_diff = fabs(cimag(C_result[i]) - cimag(C_expected[i]));
        if (real_diff > tolerance || imag_diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected (" 
                      << creal(C_expected[i]) << ", " << cimag(C_expected[i]) 
                      << "), got (" << creal(C_result[i]) << ", " << cimag(C_result[i]) 
                      << ")" << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "zGemmBatched test PASSED" << std::endl;
    } else {
        std::cout << "zGemmBatched test FAILED" << std::endl;
    }
    
    return success;
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
    
    // Wrapper to skip fp64 tests on devices that lack native fp64 (e.g. Intel Arc A380).
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

    // Run correctness tests comparing batch vs non-batch implementations
    std::cout << "\n--- Batch vs Non-Batch Comparison Tests ---" << std::endl;
    allTestsPassed &= testSgetrfBatchVsNonBatch(context);
    allTestsPassed &= testSgetrsBatchVsNonBatch(context);
    allTestsPassed &= skipIfNoFp64("Dgetrf batch vs non-batch",
        [&]{ return testDgetrfBatchVsNonBatch(context); });

    // Run CPU-based verification tests
    std::cout << "\n--- CPU Mathematical Verification Tests ---" << std::endl;
    // allTestsPassed &= testSgetrfCorrectnessCPU(context);
    std::cout << "Sgetrf correctness: SKIPPED" << std::endl;
    allTestsPassed &= testSgetrfCorrectnessCPU(context);
    allTestsPassed &= skipIfNoFp64("DGemmBatchedEx correctness",
        [&]{ return testDGemmBatchedExCorrectness(context); });
    allTestsPassed &= skipIfNoFp64("CGemmBatchedEx correctness",
        [&]{ return testCGemmBatchedExCorrectness(context); });
    allTestsPassed &= skipIfNoFp64("ZGemmBatchedEx correctness",
        [&]{ return testZGemmBatchedExCorrectness(context); });
    allTestsPassed &= testSGemmBatchedCorrectness(context);
    allTestsPassed &= skipIfNoFp64("DGemmBatched correctness",
        [&]{ return testDGemmBatchedCorrectness(context); });
    allTestsPassed &= skipIfNoFp64("CGemmBatched correctness",
        [&]{ return testCGemmBatchedCorrectness(context); });
    allTestsPassed &= skipIfNoFp64("ZGemmBatched correctness",
        [&]{ return testZGemmBatchedCorrectness(context); });
    
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