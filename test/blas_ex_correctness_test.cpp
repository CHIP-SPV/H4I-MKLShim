// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
//
// Correctness tests for the mixed precision BLAS entry points:
//   hDot    - half precision dot product
//   sGemmEx - R_16F -> R_16F and R_8I -> R_32I
//   dGemmEx - R_64F -> R_64F
// plus the "unsupported datatype combination" reporting of both GemmEx
// entry points.
//
// Every GPU result is validated against a CPU reference computed with the
// classic host CBLAS interface of the same oneMKL that backs the GPU path.
//
// Choice of CPU reference per test:
//   * half dot / half gemm: the operands are promoted to float and the
//     reference is cblas_sdot / cblas_sgemm.  Host CBLAS has no binary16 dot,
//     and the GPU result is rounded to half on the way out anyway, so a float
//     reference with a half sized tolerance is the tightest meaningful check.
//     All inputs are chosen to be exactly representable in binary16 so the
//     only error left is the rounding of the accumulator.
//   * int8 gemm: deliberately NOT cblas_gemm_s8u8s32.  That kernel multiplies
//     signed by *unsigned* 8 bit operands (s8 x u8), whereas GemmEx here is
//     s8 x s8, so every negative entry of B would be reinterpreted as a large
//     positive value and the reference would be wrong.  The operands are small
//     (|v| <= 3 with k = 24, so |C| <= 216), which is exactly representable in
//     a double, making cblas_dgemm on promoted copies an exact integer
//     reference.
//   * double gemm: cblas_dgemm directly.
//
// Output buffers are poisoned with a sentinel value before every call, so a
// backend that silently does nothing cannot pass a test.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <mkl_cblas.h>
#include "h4i/mklshim/mklshim.h"

// Host side binary16.  Both compilers used to build this project are
// clang based and support _Float16 on x86-64, and its layout is the IEEE 754
// binary16 that sycl::half uses on the device side.
using half_t = _Float16;

namespace {

const float  HALF_DOT_TOLERANCE  = 2e-3f;   // one half ulp is ~4.9e-4
const float  HALF_GEMM_TOLERANCE = 1e-2f;
const double DOUBLE_TOLERANCE    = 1e-12;

// Sentinels written into the output buffers before each call.
const float   POISON_HALF   = -111.0f;
const int32_t POISON_INT32  = -987654;
const double  POISON_DOUBLE = -1.0e30;

bool closeRel(double got, double want, double tol) {
    return std::fabs(got - want) <= tol * (1.0 + std::fabs(want));
}

// ---------------------------------------------------------------- hDot ----
bool testHDot(H4I::MKLShim::Context* ctxt) {
    std::cout << "\n--- hDot (half dot product) ---" << std::endl;
    const int64_t n = 1024;

    std::vector<half_t> hx(n), hy(n);
    std::vector<float>  fx(n), fy(n);
    for (int64_t i = 0; i < n; ++i) {
        // Multiples of 1/8 and 1/4 are exact in binary16.
        float a = ((i * 37) % 13 - 6) * 0.125f;
        float b = ((i * 53) % 11 - 5) * 0.25f;
        hx[i] = (half_t)a; fx[i] = a;
        hy[i] = (half_t)b; fy[i] = b;
    }
    // CPU reference: classic single precision CBLAS dot from oneMKL.
    const float expected = cblas_sdot((MKL_INT)n, fx.data(), 1, fy.data(), 1);

    half_t *dx = nullptr, *dy = nullptr, *dr = nullptr;
    hipMalloc(&dx, n * sizeof(half_t));
    hipMalloc(&dy, n * sizeof(half_t));
    hipMalloc(&dr, sizeof(half_t));
    hipMemcpy(dx, hx.data(), n * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), n * sizeof(half_t), hipMemcpyHostToDevice);

    const half_t poison = (half_t)POISON_HALF;
    hipMemcpy(dr, &poison, sizeof(half_t), hipMemcpyHostToDevice);

    H4I::MKLShim::hDot(ctxt, n, dx, 1, dy, 1, dr);
    hipDeviceSynchronize();

    half_t result = (half_t)0.0f;
    hipMemcpy(&result, dr, sizeof(half_t), hipMemcpyDeviceToHost);
    hipFree(dx); hipFree(dy); hipFree(dr);

    const float got = (float)result;
    bool ok = true;
    if (got == POISON_HALF) {
        std::cerr << "  FAIL: result buffer still holds the poison value; "
                     "no dot product was performed" << std::endl;
        ok = false;
    } else if (!closeRel(got, expected, HALF_DOT_TOLERANCE)) {
        std::cerr << "  FAIL: got " << got << ", cblas_sdot reference "
                  << expected << std::endl;
        ok = false;
    }
    if (ok) std::cout << "  PASS: " << got << " vs cblas_sdot " << expected
                      << std::endl;
    return ok;
}

// ------------------------------------------------------ sGemmEx R_16F ----
bool testGemmExHalf(H4I::MKLShim::Context* ctxt) {
    std::cout << "\n--- sGemmEx R_16F x R_16F -> R_16F ---" << std::endl;
    const int64_t m = 32, n = 24, k = 16;

    std::vector<half_t> hA(m * k), hB(k * n), hC(m * n);
    std::vector<float>  fA(m * k), fB(k * n), fC(m * n, 0.0f);
    for (int64_t i = 0; i < m * k; ++i) {
        float v = (((i * 7) % 9) - 4) * 0.25f;      // exact in binary16
        hA[i] = (half_t)v; fA[i] = v;
    }
    for (int64_t i = 0; i < k * n; ++i) {
        float v = (((i * 11) % 7) - 3) * 0.5f;
        hB[i] = (half_t)v; fB[i] = v;
    }
    // CPU reference: single precision CBLAS gemm on the promoted operands.
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (MKL_INT)m,
                (MKL_INT)n, (MKL_INT)k, 1.0f, fA.data(), (MKL_INT)m,
                fB.data(), (MKL_INT)k, 0.0f, fC.data(), (MKL_INT)m);

    for (int64_t i = 0; i < m * n; ++i) hC[i] = (half_t)POISON_HALF;

    half_t *dA = nullptr, *dB = nullptr, *dC = nullptr;
    hipMalloc(&dA, m * k * sizeof(half_t));
    hipMalloc(&dB, k * n * sizeof(half_t));
    hipMalloc(&dC, m * n * sizeof(half_t));
    hipMemcpy(dA, hA.data(), m * k * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), k * n * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), m * n * sizeof(half_t), hipMemcpyHostToDevice);

    bool dispatched = H4I::MKLShim::sGemmEx(
        ctxt, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
        H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, m, n, k, 1.0f,
        dA, H4I::MKLShim::ONEMKL_R_16F, m,
        dB, H4I::MKLShim::ONEMKL_R_16F, k, 0.0f,
        dC, H4I::MKLShim::ONEMKL_R_16F, m);
    hipDeviceSynchronize();
    hipMemcpy(hC.data(), dC, m * n * sizeof(half_t), hipMemcpyDeviceToHost);
    hipFree(dA); hipFree(dB); hipFree(dC);

    if (!dispatched) {
        std::cerr << "  FAIL: sGemmEx reported the combination as unsupported"
                  << std::endl;
        return false;
    }
    int poisoned = 0, bad = 0;
    for (int64_t i = 0; i < m * n; ++i) {
        float got = (float)hC[i];
        if (got == POISON_HALF) { poisoned++; continue; }
        if (!closeRel(got, fC[i], HALF_GEMM_TOLERANCE)) {
            if (bad == 0) {
                std::cerr << "  first mismatch at " << i << ": got " << got
                          << ", cblas_sgemm reference " << fC[i] << std::endl;
            }
            bad++;
        }
        if (!std::isfinite(got)) bad++;
    }
    if (poisoned) {
        std::cerr << "  FAIL: " << poisoned << "/" << (m * n)
                  << " outputs still hold the poison value" << std::endl;
        return false;
    }
    if (bad) {
        std::cerr << "  FAIL: " << bad << "/" << (m * n)
                  << " outputs disagree with the CPU reference" << std::endl;
        return false;
    }
    std::cout << "  PASS: " << (m * n)
              << " elements match cblas_sgemm within " << HALF_GEMM_TOLERANCE
              << " relative" << std::endl;
    return true;
}

// ------------------------------------------------- sGemmEx R_8I -> R_32I --
bool testGemmExInt8(H4I::MKLShim::Context* ctxt) {
    std::cout << "\n--- sGemmEx R_8I x R_8I -> R_32I ---" << std::endl;
    const int64_t m = 32, n = 24, k = 24;

    std::vector<int8_t>  hA(m * k), hB(k * n);
    std::vector<int32_t> hC(m * n);
    std::vector<double>  dblA(m * k), dblB(k * n), dblC(m * n, 0.0);
    for (int64_t i = 0; i < m * k; ++i) {
        hA[i] = (int8_t)(((i * 7) % 7) - 3);        // [-3, 3]
        dblA[i] = (double)hA[i];
    }
    for (int64_t i = 0; i < k * n; ++i) {
        hB[i] = (int8_t)(((i * 5) % 7) - 3);
        dblB[i] = (double)hB[i];
    }
    // CPU reference: cblas_dgemm on promoted copies.  See the file header for
    // why cblas_gemm_s8u8s32 is not usable for an s8 x s8 product.
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (MKL_INT)m,
                (MKL_INT)n, (MKL_INT)k, 1.0, dblA.data(), (MKL_INT)m,
                dblB.data(), (MKL_INT)k, 0.0, dblC.data(), (MKL_INT)m);

    for (int64_t i = 0; i < m * n; ++i) hC[i] = POISON_INT32;

    int8_t  *dA = nullptr, *dB = nullptr;
    int32_t *dC = nullptr;
    hipMalloc(&dA, m * k * sizeof(int8_t));
    hipMalloc(&dB, k * n * sizeof(int8_t));
    hipMalloc(&dC, m * n * sizeof(int32_t));
    hipMemcpy(dA, hA.data(), m * k * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), k * n * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), m * n * sizeof(int32_t), hipMemcpyHostToDevice);

    bool dispatched = H4I::MKLShim::sGemmEx(
        ctxt, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
        H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, m, n, k, 1.0f,
        dA, H4I::MKLShim::ONEMKL_R_8I, m,
        dB, H4I::MKLShim::ONEMKL_R_8I, k, 0.0f,
        dC, H4I::MKLShim::ONEMKL_R_32I, m);
    hipDeviceSynchronize();
    hipMemcpy(hC.data(), dC, m * n * sizeof(int32_t), hipMemcpyDeviceToHost);
    hipFree(dA); hipFree(dB); hipFree(dC);

    if (!dispatched) {
        std::cerr << "  FAIL: sGemmEx reported the combination as unsupported"
                  << std::endl;
        return false;
    }
    int poisoned = 0, bad = 0;
    for (int64_t i = 0; i < m * n; ++i) {
        if (hC[i] == POISON_INT32) { poisoned++; continue; }
        int32_t want = (int32_t)llround(dblC[i]);
        if (hC[i] != want) {
            if (bad == 0) {
                std::cerr << "  first mismatch at " << i << ": got " << hC[i]
                          << ", cblas_dgemm reference " << want << std::endl;
            }
            bad++;
        }
    }
    if (poisoned) {
        std::cerr << "  FAIL: " << poisoned << "/" << (m * n)
                  << " outputs still hold the poison value" << std::endl;
        return false;
    }
    if (bad) {
        std::cerr << "  FAIL: " << bad << "/" << (m * n)
                  << " outputs disagree with the CPU reference" << std::endl;
        return false;
    }
    std::cout << "  PASS: " << (m * n)
              << " elements match cblas_dgemm exactly" << std::endl;
    return true;
}

// ------------------------------------------------------ dGemmEx R_64F ----
bool testGemmExDouble(H4I::MKLShim::Context* ctxt) {
    std::cout << "\n--- dGemmEx R_64F x R_64F -> R_64F ---" << std::endl;
    const int64_t m = 32, n = 24, k = 16;

    std::vector<double> hA(m * k), hB(k * n), hC(m * n), ref(m * n, 0.0);
    for (int64_t i = 0; i < m * k; ++i) hA[i] = ((i * 7) % 9 - 4) * 0.25;
    for (int64_t i = 0; i < k * n; ++i) hB[i] = ((i * 11) % 7 - 3) * 0.5;
    // CPU reference: double precision CBLAS gemm.
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (MKL_INT)m,
                (MKL_INT)n, (MKL_INT)k, 2.0, hA.data(), (MKL_INT)m,
                hB.data(), (MKL_INT)k, 0.0, ref.data(), (MKL_INT)m);

    for (int64_t i = 0; i < m * n; ++i) hC[i] = POISON_DOUBLE;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    hipMalloc(&dA, m * k * sizeof(double));
    hipMalloc(&dB, k * n * sizeof(double));
    hipMalloc(&dC, m * n * sizeof(double));
    hipMemcpy(dA, hA.data(), m * k * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), k * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), m * n * sizeof(double), hipMemcpyHostToDevice);

    // alpha = 2.0 exercises the scale factor as well as the product.
    bool dispatched = H4I::MKLShim::dGemmEx(
        ctxt, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
        H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, m, n, k, 2.0,
        dA, H4I::MKLShim::ONEMKL_R_64F, m,
        dB, H4I::MKLShim::ONEMKL_R_64F, k, 0.0,
        dC, H4I::MKLShim::ONEMKL_R_64F, m);
    hipDeviceSynchronize();
    hipMemcpy(hC.data(), dC, m * n * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(dA); hipFree(dB); hipFree(dC);

    if (!dispatched) {
        std::cerr << "  FAIL: dGemmEx reported the combination as unsupported"
                  << std::endl;
        return false;
    }
    int poisoned = 0, bad = 0;
    for (int64_t i = 0; i < m * n; ++i) {
        if (hC[i] == POISON_DOUBLE) { poisoned++; continue; }
        if (!closeRel(hC[i], ref[i], DOUBLE_TOLERANCE)) {
            if (bad == 0) {
                std::cerr << "  first mismatch at " << i << ": got " << hC[i]
                          << ", cblas_dgemm reference " << ref[i] << std::endl;
            }
            bad++;
        }
    }
    if (poisoned) {
        std::cerr << "  FAIL: " << poisoned << "/" << (m * n)
                  << " outputs still hold the poison value" << std::endl;
        return false;
    }
    if (bad) {
        std::cerr << "  FAIL: " << bad << "/" << (m * n)
                  << " outputs disagree with the CPU reference" << std::endl;
        return false;
    }
    std::cout << "  PASS: " << (m * n)
              << " elements match cblas_dgemm" << std::endl;
    return true;
}

// --------------------------------------- unsupported combination report ---
bool testUnsupportedCombinationsReported(H4I::MKLShim::Context* ctxt) {
    std::cout << "\n--- GemmEx unsupported datatype combinations ---"
              << std::endl;
    const int64_t m = 8, n = 8, k = 8;
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    hipMalloc(&dA, m * k * sizeof(double));
    hipMalloc(&dB, k * n * sizeof(double));
    hipMalloc(&dC, m * n * sizeof(double));

    bool ok = true;
    // oneMKL has no float x float -> double gemm.
    if (H4I::MKLShim::sGemmEx(ctxt, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                              H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, m, n, k,
                              1.0f, dA, H4I::MKLShim::ONEMKL_R_32F, m,
                              dB, H4I::MKLShim::ONEMKL_R_32F, k, 0.0f,
                              dC, H4I::MKLShim::ONEMKL_R_64F, m)) {
        std::cerr << "  FAIL: sGemmEx claimed R_32F x R_32F -> R_64F is supported"
                  << std::endl;
        ok = false;
    }
    // dGemmEx only handles the all-double combination.
    if (H4I::MKLShim::dGemmEx(ctxt, H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS,
                              H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS, m, n, k,
                              1.0, dA, H4I::MKLShim::ONEMKL_R_32F, m,
                              dB, H4I::MKLShim::ONEMKL_R_32F, k, 0.0,
                              dC, H4I::MKLShim::ONEMKL_R_32F, m)) {
        std::cerr << "  FAIL: dGemmEx claimed R_32F x R_32F -> R_32F is supported"
                  << std::endl;
        ok = false;
    }
    hipFree(dA); hipFree(dB); hipFree(dC);
    if (ok) std::cout << "  PASS: both unsupported combinations reported"
                      << std::endl;
    return ok;
}

} // namespace

int main() {
    std::cout << "MKLShim BLAS-Ex Correctness Test" << std::endl;
    std::cout << "=================================" << std::endl;

    int nHandles = 0;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    if (nHandles <= 0) {
        std::cerr << "Failed to get number of native handles" << std::endl;
        std::quick_exit(EXIT_FAILURE);
    }
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);

    H4I::MKLShim::Context* context =
        H4I::MKLShim::Create(handles.data(), nHandles);
    if (context == nullptr) {
        std::cerr << "Failed to create MKLShim context" << std::endl;
        std::quick_exit(EXIT_FAILURE);
    }

    bool allTestsPassed = true;
    allTestsPassed &= testHDot(context);
    allTestsPassed &= testGemmExHalf(context);
    allTestsPassed &= testGemmExInt8(context);
    allTestsPassed &= testGemmExDouble(context);
    allTestsPassed &= testUnsupportedCombinationsReported(context);

    hipDeviceSynchronize();
    H4I::MKLShim::Destroy(context);

    if (allTestsPassed) {
        std::cout << "\n=================================" << std::endl;
        std::cout << "All BLAS-Ex correctness tests passed!" << std::endl;
        std::quick_exit(EXIT_SUCCESS);
    }
    std::cerr << "\n=================================" << std::endl;
    std::cerr << "Some BLAS-Ex correctness tests failed!" << std::endl;
    std::quick_exit(EXIT_FAILURE);
}
