#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklsolver.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"

namespace H4I::MKLShim
{

  int64_t Sgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<float>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sgebrd_ScPadSz")
  }

  int64_t Dgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<double>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dgebrd_scratchpad")
  }

  int64_t onemkl_Cgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cgebrd_scratchpad")
  }
  int64_t onemkl_Zgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zgebrd_scratchpad")
  }

  void Sgebrd(Context* ctxt, int64_t m, int64_t n, float* a, int64_t lda,
                    float* d, float* e, float* tauq, float* taup, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gebrd(ctxt->queue, m, n, a, lda, d, e, tauq, taup,
                  scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgebrd")
  }
  void Dgebrd(Context* ctxt, int64_t m, int64_t n, double* a, int64_t lda,
                    double* d, double* e, double* tauq, double* taup, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gebrd(ctxt->queue, m, n, a, lda, d, e, tauq, taup,
                  scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgebrd")
  }
  void Cgebrd(Context* ctxt, int64_t m, int64_t n, float _Complex* a, int64_t lda,
                    float * d, float* e, float _Complex* tauq, float _Complex* taup, float _Complex* scratchpad,
                    int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gebrd(ctxt->queue, m, n, reinterpret_cast<std::complex<float>*>(a), lda,
                                          d, e, reinterpret_cast<std::complex<float>*>(tauq),
                                          reinterpret_cast<std::complex<float>*>(taup),
                                          reinterpret_cast<std::complex<float>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Cgebrd")
  }
  void Zgebrd(Context* ctxt, int64_t m, int64_t n, double _Complex* a, int64_t lda,
                    double * d, double* e, double _Complex* tauq, double _Complex* taup, double _Complex* scratchpad,
                    int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gebrd(ctxt->queue, m, n, reinterpret_cast<std::complex<double>*>(a), lda,
                                          d, e, reinterpret_cast<std::complex<double>*>(tauq),
                                          reinterpret_cast<std::complex<double>*>(taup),
                                          reinterpret_cast<std::complex<double>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Zgebrd")
  }
}//H4I::MKLShim