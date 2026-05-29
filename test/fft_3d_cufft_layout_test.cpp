// Reproduces: 3D R2C output in default packed CCE storage doesn't match
// cuFFT's interleaved-complex layout that consumers (OpenMM, etc.) expect.
// With input all-ones, DC bin should equal N0*N1*N2 (sum of input).  Before
// the fix it returns a wrong value because OUTPUT_STRIDES sit in real units
// for COMPLEX_REAL storage rather than complex units for the cuFFT layout.
#include <cstdlib>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklfft.h"

int main() {
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    H4I::MKLShim::Context* ctxt = H4I::MKLShim::Create(handles.data(), nHandles);
    if (!ctxt) { std::cerr << "FAILED: Context creation" << std::endl; return EXIT_FAILURE; }

    const int N = 4;
    const size_t in_count  = (size_t)N * N * N;
    const size_t out_count = (size_t)N * N * (N/2 + 1);

    auto* desc = H4I::MKLShim::createFFTDescriptorSR(ctxt,
                    std::vector<std::int64_t>{N, N, N});
    if (!desc) { std::cerr << "FAILED: descriptor" << std::endl; return EXIT_FAILURE; }

    float* d_in = nullptr;
    float _Complex* d_out = nullptr;
    if (hipMalloc(&d_in, in_count * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_out, out_count * sizeof(float _Complex)) != hipSuccess) {
        std::cerr << "FAILED: hipMalloc" << std::endl; return EXIT_FAILURE;
    }
    std::vector<float> host_in(in_count, 1.0f);
    hipMemcpy(d_in, host_in.data(), in_count * sizeof(float), hipMemcpyHostToDevice);

    H4I::MKLShim::fftExecR2C(ctxt, desc, d_in, d_out);

    std::vector<float _Complex> host_out(out_count);
    hipMemcpy(host_out.data(), d_out, out_count * sizeof(float _Complex), hipMemcpyDeviceToHost);

    const float dc_re = __real__ host_out[0];
    const float dc_im = __imag__ host_out[0];
    const float expected_dc = (float)in_count;  // N^3 = 64
    if (dc_re != expected_dc || dc_im != 0.0f) {
        std::cerr << "FAILED: 3D R2C DC expected (" << expected_dc << ", 0), got ("
                  << dc_re << ", " << dc_im << ")" << std::endl;
        return EXIT_FAILURE;
    }
    // All other bins must be exactly 0 for all-ones input.
    for (size_t i = 1; i < out_count; i++) {
        if (__real__ host_out[i] != 0.0f || __imag__ host_out[i] != 0.0f) {
            std::cerr << "FAILED: bin " << i << " expected (0,0) got ("
                      << __real__ host_out[i] << ", " << __imag__ host_out[i] << ")"
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    H4I::MKLShim::destroyFFTDescriptorSR(ctxt, desc);
    hipFree(d_in); hipFree(d_out);
    H4I::MKLShim::Destroy(ctxt);

    std::cout << "All cuFFT-layout 3D R2C tests passed!" << std::endl;
    return EXIT_SUCCESS;
}
