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
  //gebrd
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

  // ormqr/unmqr
  int64_t Sormqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ormqr_scratchpad_size<float>(ctxt->queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sormqr_ScPadSz")
  }
  int64_t Dormqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ormqr_scratchpad_size<double>(ctxt->queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dormqr_ScPadSz")
  }
  int64_t Cunmqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>>(ctxt->queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cunmqr_ScPadSz")
  }
  int64_t Zunmqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>>(ctxt->queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zunmqr_ScPadSz")
  }
  void Sormqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
              float* A, int64_t lda, float* tua, float* C, int64_t ldc, float* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ormqr(ctxt->queue, convert(side), convert(trans), m, n, k, A, lda, tua, C, ldc, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sormqr")
  }
  void Dormqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    double* A, int64_t lda, double* tua, double* C, int64_t ldc, double* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ormqr(ctxt->queue, convert(side), convert(trans), m, n, k, A, lda, tua, C, ldc, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dormqr")
  }
  void Cunmqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    float _Complex* A, int64_t lda, float _Complex* tua, float _Complex* C, int64_t ldc,
                    float _Complex* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::unmqr(ctxt->queue, convert(side), convert(trans), m, n, k,
                            reinterpret_cast<std::complex<float>*>(A), lda, reinterpret_cast<std::complex<float>*>(tua),
                            reinterpret_cast<std::complex<float>*>(C), ldc, reinterpret_cast<std::complex<float>*>(scratchpad),
                            scratchpad_size);
    ONEMKL_CATCH("Cunmqr")
  }
  void Zunmqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    double _Complex* A, int64_t lda, double _Complex* tua, double _Complex* C, int64_t ldc,
                    double _Complex* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::unmqr(ctxt->queue, convert(side), convert(trans), m, n, k,
                            reinterpret_cast<std::complex<double>*>(A), lda, reinterpret_cast<std::complex<double>*>(tua),
                            reinterpret_cast<std::complex<double>*>(C), ldc, reinterpret_cast<std::complex<double>*>(scratchpad),
                            scratchpad_size);
    ONEMKL_CATCH("Zunmqr")
  }

  // ormtr/unmtr
  int64_t Sormtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ormtr_scratchpad_size<float>(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sormtr_ScPadSz")
  }
  int64_t Dormtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::ormtr_scratchpad_size<double>(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dormtr_ScPadSz")
  }
  void Sormtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float* A, int64_t lda,
              float* tua, float* C, int64_t ldc, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ormtr(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n, A, lda, tua, C, ldc,
                                             scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sormtr")
  }
  void Dormtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double* A, int64_t lda,
              double* tua, double* C, int64_t ldc, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::ormtr(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n, A, lda, tua, C, ldc,
                                             scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dormtr")
  }
  int64_t Cunmtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cunmtr_ScPadSz")
  }
  int64_t Zunmtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zunmtr_ScPadSz")
  }
  void Cunmtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float _Complex* A, int64_t lda,
              float _Complex* tua, float _Complex* C, int64_t ldc, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::unmtr(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n,
                                            reinterpret_cast<std::complex<float>*>(A), lda, reinterpret_cast<std::complex<float>*>(tua),
                                            reinterpret_cast<std::complex<float>*>(C), ldc, reinterpret_cast<std::complex<float>*>(scratchpad),
                                            scratchpad_size);
    ONEMKL_CATCH("Cunmtr")
  }
  void Zunmtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double _Complex* A, int64_t lda,
              double _Complex* tua, double _Complex* C, int64_t ldc, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::unmtr(ctxt->queue, convert(side), convert(uplo), convert(trans), m, n,
                                            reinterpret_cast<std::complex<double>*>(A), lda, reinterpret_cast<std::complex<double>*>(tua),
                                            reinterpret_cast<std::complex<double>*>(C), ldc, reinterpret_cast<std::complex<double>*>(scratchpad),
                                            scratchpad_size);
    ONEMKL_CATCH("Zunmtr")
  }

  //geqrf
  int64_t Sgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<float>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sgeqrf_scratchpad")
  }
  int64_t Dgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dgeqrf_scratchpad")
  }
  int64_t Cgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cgeqrf_scratchpad")
  }
  int64_t Zgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zgeqrf_scratchpad")
  }
  void Sgeqrf(Context* ctxt, int64_t m, int64_t n, float* A, int64_t lda, float* tua, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::geqrf(ctxt->queue, m, n, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgeqrf")
  }
  void Dgeqrf(Context* ctxt, int64_t m, int64_t n, double* A, int64_t lda, double* tua, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::geqrf(ctxt->queue, m, n, A, lda, tua, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgeqrf")
  }
  void Cgeqrf(Context* ctxt, int64_t m, int64_t n, float _Complex* A, int64_t lda, float _Complex* tua, float _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::geqrf(ctxt->queue, m, n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(tua),
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cgeqrf")
  }
  void Zgeqrf(Context* ctxt, int64_t m, int64_t n, double _Complex* A, int64_t lda, double _Complex* tua, double _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::geqrf(ctxt->queue, m, n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(tua),
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zgeqrf")
  }

  //getrf
  int64_t Sgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sgetrf_scratchpad")
  }
  int64_t Dgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dgetrf_scratchpad")
  }
  int64_t Cgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cgetrf_scratchpad")
  }
  int64_t Zgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>(ctxt->queue, m, n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zgetrf_scratchpad")
  }
  void Sgetrf(Context* ctxt, int64_t m, int64_t n, float* A, int64_t lda, int64_t *ipiv, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrf(ctxt->queue, m, n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgetrf")
  }
  void Dgetrf(Context* ctxt, int64_t m, int64_t n, double* A, int64_t lda, int64_t *ipiv, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrf(ctxt->queue, m, n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgetrf")
  }
  void Cgetrf(Context* ctxt, int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t *ipiv, float _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrf(ctxt->queue, m, n, reinterpret_cast<std::complex<float>*>(A), lda, ipiv,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cgetrf")
  }
  void Zgetrf(Context* ctxt, int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t *ipiv, double _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrf(ctxt->queue, m, n, reinterpret_cast<std::complex<double>*>(A), lda, ipiv,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zgetrf")
  }

  //getrs
  int64_t Sgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(ctxt->queue, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Sgetrs_scratchpad")
  }
  int64_t Dgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<double>(ctxt->queue, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dgetrs_scratchpad")
  }
  int64_t Cgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>(ctxt->queue, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cgetrs_scratchpad")
  }
  int64_t Zgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>(ctxt->queue, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zgetrs_scratchpad")
  }
  void Sgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, float* A, int64_t lda, std::int64_t *ipiv,
              float* B, int64_t ldb, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrs(ctxt->queue, convert(trans), n, nrhs, A, lda, ipiv, B, ldb, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgetrs")
  }
  void Dgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, double* A, int64_t lda, std::int64_t *ipiv,
              double* B, int64_t ldb, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrs(ctxt->queue, convert(trans), n, nrhs, A, lda, ipiv, B, ldb, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgetrs")
  }
  void Cgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex* A, int64_t lda, std::int64_t *ipiv,
              float _Complex* B, int64_t ldb, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrs(ctxt->queue, convert(trans), n, nrhs, reinterpret_cast<std::complex<float>*>(A), lda, ipiv,
                  reinterpret_cast<std::complex<float>*>(B), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cgetrs")
  }
  void Zgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex* A, int64_t lda, std::int64_t *ipiv,
              double _Complex* B, int64_t ldb, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::getrs(ctxt->queue, convert(trans), n, nrhs, reinterpret_cast<std::complex<double>*>(A), lda, ipiv,
                  reinterpret_cast<std::complex<double>*>(B), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zgetrs")
  }

//potrf
  int64_t Spotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<float>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Spotrf_ScPadSz")
  }
  int64_t Dpotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<double>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dpotrf_ScPadSz")
  }
  int64_t Cpotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cpotrf_ScPadSz")
  }
  int64_t Zpotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zpotrf_ScPadSz")
  }
  void Spotrf(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrf(ctxt->queue, convert(uplo), n, A, lda, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Spotrf")
  }
  void Dpotrf(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrf(ctxt->queue, convert(uplo), n, A, lda, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dpotrf")
  }
  void Cpotrf(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrf(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cpotrf")
  }
  void Zpotrf(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrf(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zpotrf")
  }

//potrf
  int64_t Spotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<float>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Spotri_ScPadSz")
  }
  int64_t Dpotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<double>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dpotri_ScPadSz")
  }
  int64_t Cpotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cpotri_ScPadSz")
  }
  int64_t Zpotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zpotri_ScPadSz")
  }
  void Spotri(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potri(ctxt->queue, convert(uplo), n, A, lda, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Spotri")
  }
  void Dpotri(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potri(ctxt->queue, convert(uplo), n, A, lda, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dpotri")
  }
  void Cpotri(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potri(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cpotri")
  }
  void Zpotri(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potri(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zpotri")
  }

  //potrs
  int64_t Spotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<float>(ctxt->queue, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Spotrs_scratchpad")
  }
  int64_t Dpotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<double>(ctxt->queue, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dpotrs_scratchpad")
  }
  int64_t Cpotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>(ctxt->queue, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Cpotrs_scratchpad")
  }
  int64_t Zpotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>(ctxt->queue, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zpotrs_scratchpad")
  }
  void Spotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, float* A, int64_t lda,
              float* B, int64_t ldb, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrs(ctxt->queue, convert(uplo), n, nrhs, A, lda, B, ldb, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Spotrs")
  }
  void Dpotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, double* A, int64_t lda,
              double* B, int64_t ldb, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrs(ctxt->queue, convert(uplo), n, nrhs, A, lda, B, ldb, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dpotrs")
  }
  void Cpotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex* A, int64_t lda,
              float _Complex* B, int64_t ldb, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrs(ctxt->queue, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float>*>(A), lda,
                  reinterpret_cast<std::complex<float>*>(B), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cpotrs")
  }
  void Zpotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex* A, int64_t lda,
              double _Complex* B, int64_t ldb, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::potrs(ctxt->queue, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double>*>(A), lda,
                  reinterpret_cast<std::complex<double>*>(B), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zpotrs")
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

  //sytrf
  int64_t Ssytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<float>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Ssytrf_ScPadSz")
  }
  int64_t Dsytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<double>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dsytrf_ScPadSz")
  }
  int64_t Csytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Csytrf_ScPadSz")
  }
  int64_t Zsytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Zsytrf_ScPadSz")
  }
  void Ssytrf(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, int64_t* ipiv, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::sytrf(ctxt->queue, convert(uplo), n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Ssytrf")
  }
  void Dsytrf(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, int64_t* ipiv, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::sytrf(ctxt->queue, convert(uplo), n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dsytrf")
  }
  void Csytrf(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t* ipiv, float _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::sytrf(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda, ipiv,
                                            reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Csytrf")
  }
  void Zsytrf(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t* ipiv, double _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::sytrf(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda, ipiv,
                                            reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zsytrf")
  }

  //sytrd/hetrd
  int64_t Ssytrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::sytrd_scratchpad_size<float>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Ssytrd_ScPadSz")
  }
  int64_t Dsytrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::sytrd_scratchpad_size<double>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dsytrd_ScPadSz")
  }
  int64_t Chetrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Chetrd_ScPadSz")
  }
  int64_t Zhetrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY
    auto size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(ctxt->queue, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH_NO_FLUSH("Dhetrd_ScPadSz")
  }
  void Ssytrd(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* d, float* e, float* tau,
             float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::sytrd(ctxt->queue, convert(uplo), n, A, lda, d, e, tau, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Ssytrf")
  }
  void Dsytrd(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* d, double* e, double* tau,
             double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::sytrd(ctxt->queue, convert(uplo), n, A, lda, d, e, tau, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dsytrf")
  }
  void Chetrd(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* d, float* e, float _Complex* tau,
             float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::hetrd(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda, d, e,
                                             reinterpret_cast<std::complex<float>*>(tau),
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Chetrd")
  }
  void Zhetrd(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* d, double* e, double _Complex* tau,
             double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY
    auto status = oneapi::mkl::lapack::hetrd(ctxt->queue, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda, d, e,
                                             reinterpret_cast<std::complex<double>*>(tau),
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zhetrd")
  }
}//H4I::MKLShim