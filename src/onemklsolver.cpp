#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklsolver.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"
#include "h4i/mklshim/types.h"
#include "h4i/mklshim/common.h"
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

  int64_t Cgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cgebrd_scratchpad")
  }
  int64_t Zgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
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

  //orgbr/ungbr
  int64_t Sorgbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k,
                              int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::orgbr_scratchpad_size<float>(ctxt->queue, convert(gen), m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sorgbr_ScPadSz")
  }

  int64_t Dorgbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k,
                              int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::orgbr_scratchpad_size<double>(ctxt->queue, convert(gen), m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dorgbr_ScPadSz")
  }
  void Sorgbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, float* A, int64_t lda,
                   float* tua, float* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::orgbr(ctxt->queue, convert(gen), m, n, k, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sorgbr")
  }

  void Dorgbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                   double* tua, double* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::orgbr(ctxt->queue, convert(gen), m, n, k, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dorgbr")
  }

  int64_t Cungbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k,
                              int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<float>>(ctxt->queue, convert(gen), m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cungbr_ScPadSz")
  }

  int64_t Zungbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k,
                              int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<double>>(ctxt->queue, convert(gen), m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zungbr_ScPadSz")
  }

  void Cungbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, float _Complex* A, int64_t lda,
                    float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ungbr(ctxt->queue, convert(gen), m, n, k, reinterpret_cast<std::complex<float>*>(A), lda,
                                          reinterpret_cast<std::complex<float>*>(tua), reinterpret_cast<std::complex<float>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Cungbr")
  }
  void Zungbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, double _Complex* A, int64_t lda,
                    double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ungbr(ctxt->queue, convert(gen), m, n, k, reinterpret_cast<std::complex<double>*>(A), lda,
                                          reinterpret_cast<std::complex<double>*>(tua), reinterpret_cast<std::complex<double>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Zungbr")
  }

  // orgqr/ungqr
  int64_t Sorgqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::orgqr_scratchpad_size<float>(ctxt->queue, m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sorgqr_ScPadSz")
  }

  int64_t Dorgqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::orgqr_scratchpad_size<double>(ctxt->queue, m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dorgqr_ScPadSz")
  }

  void Sorgqr(Context* ctxt, int64_t m, int64_t n, int64_t k, float* A, int64_t lda,
                    float* tua, float* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::orgqr(ctxt->queue, m, n, k, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sorgqr")
  }

  void Dorgqr(Context* ctxt, int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                    double* tua, double* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::orgqr(ctxt->queue, m, n, k, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dorgqr")
  }

  int64_t Cungqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<float>>(ctxt->queue, m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cungqr_ScPadSz")
  }

  int64_t Zungqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<double>>(ctxt->queue, m, n, k, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zungqr_ScPadSz")
  }

  void Cungqr(Context* ctxt, int64_t m, int64_t n, int64_t k, float _Complex* A, int64_t lda,
                    float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ungqr(ctxt->queue, m, n, k, reinterpret_cast<std::complex<float>*>(A), lda,
                                          reinterpret_cast<std::complex<float>*>(tua), reinterpret_cast<std::complex<float>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Cungqr")
  }
  void Zungqr(Context* ctxt, int64_t m, int64_t n, int64_t k, double _Complex* A, int64_t lda,
                    double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ungqr(ctxt->queue, m, n, k, reinterpret_cast<std::complex<double>*>(A), lda,
                                          reinterpret_cast<std::complex<double>*>(tua), reinterpret_cast<std::complex<double>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Zungqr")
  }

  // orgtr/ungtr
  int64_t Sorgtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::orgtr_scratchpad_size<float>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sorgtr_ScPadSz")
  }

  int64_t Dorgtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::orgtr_scratchpad_size<double>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dorgtr_ScPadSz")
  }

  void Sorgtr(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda,
                    float* tua, float* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::orgtr(ctxt->queue, convert(uplo), n, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sorgtr")
  }

  void Dorgtr(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda,
                    double* tua, double* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::orgtr(ctxt->queue, convert(uplo), n, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dorgtr")
  }

  int64_t Cungtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cungtr_ScPadSz")
  }

  int64_t Zungtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zungtr_ScPadSz")
  }

  void Cungtr(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda,
                    float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ungtr(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                          reinterpret_cast<std::complex<float>*>(tua), reinterpret_cast<std::complex<float>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Cungtr")
  }
  void Zungtr(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda,
                    double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ungtr(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                          reinterpret_cast<std::complex<double>*>(tua), reinterpret_cast<std::complex<double>*>(scratchpad),
                                          scratchpad_size);
    ONEMKL_CATCH("Zungtr")
  }

  //gesvd
  int64_t Sgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(ctxt->queue, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sgesvd_scratchpad")
  }
  int64_t Dgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(ctxt->queue, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dgesvd_scratchpad")
  }
  int64_t Cgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(ctxt->queue, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cgesvd_scratchpad")
  }
  int64_t Zgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(ctxt->queue, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zgesvd_scratchpad")
  }
  void Sgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, float* A, int64_t lda,
                      float* S, float* U, int64_t ldu, float* V, int64_t ldv, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gesvd(ctxt->queue, convert(jobu), convert(jobvt), m, n, A, lda, S, U, ldu, V, ldv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgesvd")
  }
  void Dgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, double* A, int64_t lda,
                      double* S, double* U, int64_t ldu, double* V, int64_t ldv, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gesvd(ctxt->queue, convert(jobu), convert(jobvt), m, n, A, lda, S, U, ldu, V, ldv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgesvd")
  }
  void Cgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, float _Complex* A, int64_t lda,
                      float* S, float _Complex* U, int64_t ldu, float _Complex* V, int64_t ldv, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gesvd(ctxt->queue, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<float>*>(A), lda,
                                          S, reinterpret_cast<std::complex<float>*>(U), ldu, reinterpret_cast<std::complex<float>*>(V), ldv,
                                          reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cgesvd")
  }
  void Zgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, double _Complex* A, int64_t lda,
                      double* S, double _Complex* U, int64_t ldu, double _Complex* V, int64_t ldv, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::gesvd(ctxt->queue, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<double>*>(A), lda,
                                          S, reinterpret_cast<std::complex<double>*>(U), ldu, reinterpret_cast<std::complex<double>*>(V), ldv,
                                          reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zgesvd")
  }

  //syevd/heevd
  int64_t Ssyevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::syevd_scratchpad_size<float>(ctxt->queue, convert(job), convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Ssyevd_scratchpad")
  }

  int64_t Dsyevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(ctxt->queue, convert(job), convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dsyevd_scratchpad")
  }

  void Ssyevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* w,
                    float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::syevd(ctxt->queue, convert(job), convert(uplo), n, A, lda, w, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Ssyevd")
  }

  void Dsyevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* w,
                    double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::syevd(ctxt->queue, convert(job), convert(uplo), n, A, lda, w, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dsyevd")
  }

  int64_t Cheevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<float>>(ctxt->queue, convert(job), convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cheevd_scratchpad")
  }

  int64_t Zheevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<double>>(ctxt->queue, convert(job), convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zheevd_scratchpad")
  }

  void Cheevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* w,
                    float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::heevd(ctxt->queue, convert(job), convert(uplo), n,
                    reinterpret_cast<std::complex<float>*>(A), lda, w, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cheevd")
  }

  void Zheevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* w,
                    double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::heevd(ctxt->queue, convert(job), convert(uplo), n,
                    reinterpret_cast<std::complex<double>*>(A), lda, w, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zheevd")
  }
}//H4I::MKLShim