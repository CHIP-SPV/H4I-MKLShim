// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklblas.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"


namespace H4I::MKLShim
{
  //dot
  void sDot(Context* ctxt, int64_t n, const float *x, int64_t incx, const float *y,
            int64_t incy, float *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::dot(ctxt->queue, n, x,
                                                      incx, y, incy, result);
    ONEMKL_CATCH("DOT")
  }

  void dDot(Context* ctxt, int64_t n, const double *x, int64_t incx, const double *y,
            int64_t incy, double *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::dot(ctxt->queue, n, x,
                                                      incx, y, incy, result);
    ONEMKL_CATCH("DOT")
  }

  void cDotc(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
             const float _Complex *y, int64_t incy, float _Complex *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::dotc(ctxt->queue, n,
                                                reinterpret_cast<const std::complex<float> *>(x), incx,
                                                reinterpret_cast<const std::complex<float> *>(y), incy,
                                                reinterpret_cast<std::complex<float> *>(result));
    ONEMKL_CATCH("DOT")
  }

  void zDotc(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
             const double _Complex *y, int64_t incy, double _Complex *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::dotc(ctxt->queue, n,
                                                reinterpret_cast<const std::complex<double> *>(x), incx,
                                                reinterpret_cast<const std::complex<double> *>(y), incy,
                                                reinterpret_cast<std::complex<double> *>(result));
    ONEMKL_CATCH("DOT")
  }

  void cDotu(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
             const float _Complex *y, int64_t incy, float _Complex *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::dotu(ctxt->queue, n,
                                                reinterpret_cast<const std::complex<float> *>(x), incx,
                                                reinterpret_cast<const std::complex<float> *>(y), incy,
                                                reinterpret_cast<std::complex<float> *>(result));
    ONEMKL_CATCH("DOT")
  }

  void zDotu(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
             const double _Complex *y, int64_t incy, double _Complex *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::dotu(ctxt->queue, n,
                                                reinterpret_cast<const std::complex<double> *>(x), incx,
                                                reinterpret_cast<const std::complex<double> *>(y), incy,
                                                reinterpret_cast<std::complex<double> *>(result));
    ONEMKL_CATCH("DOT")
  }

  //asum
  void sAsum(Context* ctxt, int64_t n, const float *x, int64_t incx,
            float *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::asum(ctxt->queue, n, x,
                                                        incx, result);
    ONEMKL_CATCH("ASUM")
  }

  void dAsum(Context* ctxt, int64_t n, const double *x, int64_t incx,
            double *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::asum(ctxt->queue, n, x,
                                                        incx, result);
    ONEMKL_CATCH("ASUM")
  }

  void cAsum(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
            float *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::asum(ctxt->queue, n, 
                                        reinterpret_cast<const std::complex<float> *>(x),
                                        incx, result);
    ONEMKL_CATCH("ASUM")
  }

  void zAsum(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
            double *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::asum(ctxt->queue, n, 
                                        reinterpret_cast<const std::complex<double> *>(x),
                                        incx, result);
    ONEMKL_CATCH("ASUM")
  }

  //axpy
  void sAxpy(Context* ctxt, int64_t n, float alpha, const float *x, std::int64_t incx,
                  float *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::axpy(ctxt->queue, n, alpha, x,
                                                incx, y, incy);
    ONEMKL_CATCH("AXPY")
  }

  void dAxpy(Context* ctxt, int64_t n, double alpha, const double *x, std::int64_t incx, 
                  double *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::axpy(ctxt->queue, n, alpha, x,
                                                  incx, y, incy);
    ONEMKL_CATCH("AXPY")
  }

  void cAxpy(Context* ctxt, int64_t n, float _Complex alpha, const float _Complex *x,
                  std::int64_t incx, float _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::axpy(ctxt->queue, n, alpha,
                              reinterpret_cast<const std::complex<float> *>(x), incx,
                              reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("AXPY")
  }

  void zAxpy(Context* ctxt, int64_t n, double _Complex alpha, const double _Complex *x,
                  std::int64_t incx, double _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::axpy(ctxt->queue, n, alpha,
                            reinterpret_cast<const std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("AXPY")
  }

  //scal
  void dScal(Context* ctxt, int64_t n, double alpha,
                    double *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::scal(ctxt->queue, n, alpha,
                                                    x, incx);
    ONEMKL_CATCH("SCAL")

  }

  void sScal(Context* ctxt, int64_t n, float alpha,
                              float *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::scal(ctxt->queue, n, alpha,
                                                        x, incx);
    ONEMKL_CATCH("SCAL")
  }

  void cScal(Context* ctxt, int64_t n,
                              float _Complex alpha, float _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::scal(ctxt->queue, n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<std::complex<float> *>(x),incx);
    ONEMKL_CATCH("SCAL")
  }

  void csScal(Context* ctxt, int64_t n,
                              float alpha, float _Complex *x,
                              int64_t incx) {
      ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::scal(ctxt->queue, n, alpha,
                                          reinterpret_cast<std::complex<float> *>(x),incx);
      ONEMKL_CATCH("SCAL")
  }

  void zsScal(Context* ctxt, int64_t n,
                              double _Complex alpha, double _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::scal(ctxt->queue, n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<std::complex<double> *>(x),incx);
    ONEMKL_CATCH("SCAL")
  }

  void zdScal(Context* ctxt, int64_t n,
                              double alpha, double _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::scal(ctxt->queue, n, alpha,
                                        reinterpret_cast<std::complex<double> *>(x),incx);
    ONEMKL_CATCH("SCAL")
  }

  //nrm2
  void dNrm2(Context* ctxt, int64_t n, const double *x,
                              int64_t incx, double *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::nrm2(ctxt->queue, n, x, incx, result);
    ONEMKL_CATCH("NRM2")
  }

  void sNrm2(Context* ctxt, int64_t n, const float *x,
                              int64_t incx, float *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::nrm2(ctxt->queue, n, x, incx, result);
    ONEMKL_CATCH("NRM2")
  }

  void cNrm2(Context* ctxt, int64_t n, const float _Complex *x,
                              int64_t incx, float *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::nrm2(ctxt->queue, n,
                    reinterpret_cast<const std::complex<float> *>(x), incx, result);
    ONEMKL_CATCH("NRM2")
  }

  void zNrm2(Context* ctxt, int64_t n, const double _Complex *x,
                              int64_t incx, double *result) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::nrm2(ctxt->queue, n,
                    reinterpret_cast<const std::complex<double> *>(x), incx, result);
    ONEMKL_CATCH("NRM2")
  }

  //copy
  void dCopy(Context* ctxt, int64_t n, const double *x,
                              int64_t incx, double *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::copy(ctxt->queue, n, x, incx, y, incy);
    ONEMKL_CATCH("COPY")
  }

  void sCopy(Context* ctxt, int64_t n, const float *x,
                              int64_t incx, float *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::copy(ctxt->queue, n, x, incx, y, incy);
    ONEMKL_CATCH("COPY")
  }

  void zCopy(Context* ctxt, int64_t n, const double _Complex *x,
                              int64_t incx, double _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::copy(ctxt->queue, n,
          reinterpret_cast<const std::complex<double> *>(x), incx,
          reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("COPY")
  }

  void cCopy(Context* ctxt, int64_t n, const float _Complex *x,
                              int64_t incx, float _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::copy(ctxt->queue, n,
          reinterpret_cast<const std::complex<float> *>(x), incx,
          reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("COPY")
  }

  //amax
  void dAmax(Context* ctxt, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n, x, incx, result);
    ONEMKL_CATCH("MAX")
  }
  void sAmax(Context* ctxt, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n, x, incx, result);
    ONEMKL_CATCH("MAX")
  }
  void zAmax(Context* ctxt, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n,
                              reinterpret_cast<const std::complex<double> *>(x), incx, result);
    ONEMKL_CATCH("MAX")
  }
  void cAmax(Context* ctxt, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n,
                              reinterpret_cast<const std::complex<float> *>(x), incx, result);
    ONEMKL_CATCH("MAX")
  }

  // amin
  void dAmin(Context* ctxt, int64_t n, const double *x, int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamin(ctxt->queue, n, x, incx, result);
    ONEMKL_CATCH("AMIN")
  }
  void sAmin(Context* ctxt, int64_t n, const float  *x, int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamin(ctxt->queue, n, x, incx, result);
    ONEMKL_CATCH("AMIN")
  }
  void zAmin(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamin(ctxt->queue, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    ONEMKL_CATCH("AMIN")
  }
  void cAmin(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx, int64_t *result){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::iamin(ctxt->queue, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    ONEMKL_CATCH("AMIN")
  }
  //swap
  void sSwap(Context* ctxt, int64_t n, float *x, int64_t incx, float *y, int64_t incy){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::swap(ctxt->queue, n, x, incx, y, incy);
    ONEMKL_CATCH("SWAP")
  }

  void dSwap(Context* ctxt, int64_t n, double *x, int64_t incx, double *y, int64_t incy){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::swap(ctxt->queue, n, x, incx, y, incy);
    ONEMKL_CATCH("SWAP")
  }

  void cSwap(Context* ctxt, int64_t n, float _Complex *x, int64_t incx,
              float _Complex *y, int64_t incy){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::swap(ctxt->queue, n,
                            reinterpret_cast<std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("SWAP")
  }

  void zSwap(Context* ctxt, int64_t n, double _Complex *x, int64_t incx,
              double _Complex *y, int64_t incy){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::swap(ctxt->queue, n,
                            reinterpret_cast<std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("SWAP")
  }
  //rot
  void sRot(Context* ctxt, int n, float* x, int incx, float* y, int incy,
              const float c, const float s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rot(ctxt->queue, n, x, incx, y, incy, c, s);
    ONEMKL_CATCH("ROT")
  }
  void dRot(Context* ctxt, int n, double* x, int incx, double* y, int incy,
              const double c, const double s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rot(ctxt->queue, n, x, incx, y, incy, c, s);
    ONEMKL_CATCH("ROT")
  }
  void cRot(Context* ctxt, int n, float _Complex* x, int incx, float _Complex* y, int incy,
              const float c, const float _Complex s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rot(ctxt->queue, n,
                    reinterpret_cast<std::complex<float> *>(x), incx,
                    reinterpret_cast<std::complex<float> *>(y), incy,
                    c, static_cast<std::complex<float> >(s));
    ONEMKL_CATCH("ROT")
  }
  void csRot(Context* ctxt, int n, float _Complex* x, int incx, float _Complex* y, int incy,
              const float c, const float s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rot(ctxt->queue, n,
                    reinterpret_cast<std::complex<float> *>(x), incx,
                    reinterpret_cast<std::complex<float> *>(y), incy, c, s);
    ONEMKL_CATCH("ROT")
  }
  void zRot(Context* ctxt, int n, double _Complex* x, int incx, double _Complex* y, int incy,
              const double c, const double _Complex s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rot(ctxt->queue, n,
                    reinterpret_cast<std::complex<double> *>(x), incx,
                    reinterpret_cast<std::complex<double> *>(y), incy,
                    c, static_cast<std::complex<double> >(s));
    ONEMKL_CATCH("ROT")
  }
  void zdRot(Context* ctxt, int n, double _Complex* x, int incx, double _Complex* y, int incy,
              const double c, const double s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rot(ctxt->queue, n,
                    reinterpret_cast<std::complex<double> *>(x), incx,
                    reinterpret_cast<std::complex<double> *>(y), incy, c, s);
    ONEMKL_CATCH("ROT")
  }
  //rotg
  void sRotg(Context* ctxt, float* a, float* b, float* c, float* s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotg(ctxt->queue, a, b, c, s);
    ONEMKL_CATCH("ROTG")
  }
  void dRotg(Context* ctxt, double* a, double* b, double* c, double* s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotg(ctxt->queue, a, b, c, s);
    ONEMKL_CATCH("ROTG")
  }
  void cRotg(Context* ctxt, float _Complex* a, float _Complex* b, float* c, float _Complex* s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotg(ctxt->queue,
                                                        reinterpret_cast<std::complex<float> *>(a),
                                                        reinterpret_cast<std::complex<float> *>(b), c,
                                                        reinterpret_cast<std::complex<float> *>(s));
    ONEMKL_CATCH("ROTG")
  }
  void zRotg(Context* ctxt, double _Complex* a, double _Complex* b, double* c, double _Complex* s){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotg(ctxt->queue,
                                                        reinterpret_cast<std::complex<double> *>(a),
                                                        reinterpret_cast<std::complex<double> *>(b), c,
                                                        reinterpret_cast<std::complex<double> *>(s));
    ONEMKL_CATCH("ROTG")
  }
  //rotm
  void sRotm(Context* ctxt, int64_t n, float *x, int64_t incx,
              float *y, int64_t incy, float* param) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotm(ctxt->queue, n, x, incx, y, incy, param);
    ONEMKL_CATCH("ROTM")
  }
  void dRotm(Context* ctxt, int64_t n, double *x, int64_t incx,
              double *y, int64_t incy, double* param){
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotm(ctxt->queue, n, x, incx, y, incy, param);
    ONEMKL_CATCH("ROTM")
  }

  //rotmg
  void sRotmg(Context* ctxt, float *d1, float *d2, float *x1, float y1, float* param) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotmg(ctxt->queue, d1, d2, x1, y1, param);
    ONEMKL_CATCH("ROTMG")
  }
  void dRotmg(Context* ctxt, double *d1, double *d2, double *x1, double y1, double* param) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::rotmg(ctxt->queue, d1, d2, x1, y1, param);
    ONEMKL_CATCH("ROTMG")
  }

  //--------------------------------------------------------
  // ===================== Level-2 =========================
  //--------------------------------------------------------
  oneapi::mkl::transpose convert(onemklTranspose val) {
    switch (val) {
      case ONEMKL_TRANSPOSE_NONTRANS:
          return oneapi::mkl::transpose::nontrans;
      case ONEMKL_TRANSPOSE_TRANS:
          return oneapi::mkl::transpose::trans;
      case ONEMLK_TRANSPOSE_CONJTRANS:
          return oneapi::mkl::transpose::conjtrans;
    }
  }

  oneapi::mkl::uplo convert(onemklUplo val) {
    switch(val) {
      case ONEMKL_UPLO_UPPER:
        return oneapi::mkl::uplo::upper;
      case ONEMKL_UPLO_LOWER:
        return oneapi::mkl::uplo::lower;
    }
  }

  oneapi::mkl::side convert(onemklSideMode val) {
    switch(val) {
      case ONEMKL_SIDE_LEFT:
        return oneapi::mkl::side::left;
      case ONEMKL_SIDE_RIGHT:
        return oneapi::mkl::side::right;
    }
  }

  oneapi::mkl::diag convert(onemklDiag val) {
    switch(val) {
      case ONEMKL_DIAG_NONUNIT:
        return oneapi::mkl::diag::nonunit;
      case ONEMKL_DIAG_UNIT:
        return oneapi::mkl::diag::unit;
    }
  }
  void sGbmv(Context* ctxt, onemklTranspose trans,
                    int64_t m, int64_t n, int64_t kl, int64_t ku,
                    float alpha, const float *a, int64_t lda,
                    const float *x, int64_t incx, float beta, float *y,
                    int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::gbmv(ctxt->queue,
                                  convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                  incx, beta, y, incy);
    ONEMKL_CATCH("")
  }

  void dGbmv(Context* ctxt, onemklTranspose trans,
                    int64_t m, int64_t n, int64_t kl, int64_t ku,
                    double alpha, const double *a, int64_t lda,
                    const double *x, int64_t incx, double beta, double *y,
                    int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::gbmv(ctxt->queue, convert(trans),
                                    m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }

  void cGbmv(Context* ctxt, onemklTranspose trans,
                    int64_t m, int64_t n, int64_t kl, int64_t ku,
                    float _Complex alpha, const float _Complex *a, int64_t lda,
                    const float _Complex *x, int64_t incx, float _Complex beta,
                    float _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::gbmv(ctxt->queue, convert(trans),
                                    m, n, kl, ku, static_cast<std::complex<float> >(alpha),
                                    reinterpret_cast<const std::complex<float> *>(a),
                                    lda, reinterpret_cast<const std::complex<float> *>(x),
                                    incx, static_cast<std::complex<float> >(beta),
                                    reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("")
  }

  void zGbmv(Context* ctxt, onemklTranspose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                    double _Complex alpha, const double _Complex *a, int64_t lda,
                    const double _Complex *x, int64_t incx, double _Complex beta,
                    double _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::gbmv(ctxt->queue, convert(trans), m,
                                        n, kl, ku, static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<const std::complex<double> *>(x), incx,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("GBMV")
  }

  void sGemv(Context* ctxt, onemklTranspose trans,
                            int64_t m, int64_t n, float alpha, const float *a,
                            int64_t lda, const float *x, int64_t incx, float beta,
                            float *y, int64_t incy) {
    ONEMKL_TRY
	  auto status = oneapi::mkl::blas::column_major::gemv(ctxt->queue, convert(trans),
                                            m, n, alpha, a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("GEMV")
  }

  void dGemv(Context* ctxt, onemklTranspose trans,
                            int64_t m, int64_t n, double alpha, const double *a,
                            int64_t lda, const double *x, int64_t incx, double beta,
                            double *y, int64_t incy) {
    ONEMKL_TRY
	  auto status = oneapi::mkl::blas::column_major::gemv(ctxt->queue, convert(trans),
                                            m, n, alpha, a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("GEMV")
  }

  void cGemv(Context* ctxt, onemklTranspose trans,
                            int64_t m, int64_t n, float _Complex alpha,
                            const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx,
                            float _Complex beta, float _Complex *y,
                            int64_t incy) {
    ONEMKL_TRY
	  auto status = oneapi::mkl::blas::column_major::gemv(ctxt->queue, convert(trans), m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(a), lda,
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            static_cast<std::complex<float> >(beta),
                                            reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("GEMV")
  }

  void zGemv(Context* ctxt, onemklTranspose trans,
                            int64_t m, int64_t n, double _Complex alpha,
                            const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx,
                            double _Complex beta, double _Complex *y,
                            int64_t incy) {
    ONEMKL_TRY
	  auto status = oneapi::mkl::blas::column_major::gemv(ctxt->queue, convert(trans), m, n,
                                            static_cast<std::complex<double> >(alpha),
                                            reinterpret_cast<const std::complex<double> *>(a), lda,
                                            reinterpret_cast<const std::complex<double> *>(x), incx,
                                            static_cast<std::complex<double> >(beta),
                                            reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("GEMV")
  }

  void sGer(Context* ctxt, int64_t m, int64_t n, float alpha,
                            const float *x, int64_t incx, const float *y, int64_t incy,
                            float *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::ger(ctxt->queue, m, n, alpha, x,
                                                    incx, y, incy, a, lda);
    ONEMKL_CATCH("")
  }

  void dGer(Context* ctxt, int64_t m, int64_t n, double alpha,
                            const double *x, int64_t incx, const double *y, int64_t incy,
                            double *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::ger(ctxt->queue, m, n, alpha, x,
                                                    incx, y, incy, a, lda);
    ONEMKL_CATCH("")
  }

  void cGerc(Context* ctxt, int64_t m, int64_t n, float _Complex alpha,
                            const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                            float _Complex *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::gerc(ctxt->queue, m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            reinterpret_cast<const std::complex<float> *>(y), incy,
                                            reinterpret_cast<std::complex<float> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void cGeru(Context* ctxt, int64_t m, int64_t n, float _Complex alpha,
                            const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                            float _Complex *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::geru(ctxt->queue, m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            reinterpret_cast<const std::complex<float> *>(y), incy,
                                            reinterpret_cast<std::complex<float> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void zGerc(Context* ctxt, int64_t m, int64_t n, double _Complex alpha,
                            const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                            double _Complex *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::gerc(ctxt->queue, m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void zGeru(Context* ctxt, int64_t m, int64_t n, double _Complex alpha,
                            const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                            double _Complex *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::geru(ctxt->queue, m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void cHbmv(Context* ctxt, onemklUplo uplo, int64_t n,
                            int64_t k, float _Complex alpha, const float _Complex *a,
                            int64_t lda, const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hbmv(ctxt->queue, convert(uplo), n,
                                          k, static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(x),
                                          incx, static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("")
  }

  void zHbmv(Context* ctxt, onemklUplo uplo, int64_t n,
                            int64_t k, double _Complex alpha, const double _Complex *a,
                            int64_t lda, const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hbmv(ctxt->queue, convert(uplo), n,
                                          k, static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(x),
                                          incx, static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("")
  }

  void cHemv(Context* ctxt, onemklUplo uplo, int64_t n,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hemv(ctxt->queue, convert(uplo), n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(x), incx,
                                          static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("")
  }

  void zHemv(Context* ctxt, onemklUplo uplo, int64_t n,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hemv(ctxt->queue, convert(uplo), n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(x), incx,
                                          static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("")
  }

  void cHer(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                            const float _Complex *x, int64_t incx, float _Complex *a,
                            int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::her(ctxt->queue, convert(uplo), n, alpha,
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<std::complex<float> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void zHer(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                            const double _Complex *x, int64_t incx, double _Complex *a,
                            int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::her(ctxt->queue, convert(uplo), n, alpha,
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<std::complex<double> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void cHer2(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex alpha,
                            const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                            float _Complex *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::her2(ctxt->queue, convert(uplo), n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(x), incx,
                                          reinterpret_cast<const std::complex<float> *>(y), incy,
                                          reinterpret_cast<std::complex<float> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void zHer2(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex alpha,
                            const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                            double _Complex *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::her2(ctxt->queue, convert(uplo), n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    ONEMKL_CATCH("")
  }

  void cHpmv(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *a, const float _Complex *x, int64_t incx,
                float _Complex beta, float _Complex *y, int64_t incy)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hpmv(ctxt->queue, convert(uplo), n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(a),
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        static_cast<std::complex<float> >(beta),
                                        reinterpret_cast<std::complex<float> *>(y), incy);
    ONEMKL_CATCH("")
  }
  void zHpmv(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *a, const double _Complex *x, int64_t incx,
                double _Complex beta, double _Complex *y, int64_t incy)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hpmv(ctxt->queue, convert(uplo), n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a),
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(y), incy);
    ONEMKL_CATCH("")
  }

  void cHpr(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                const float _Complex *x, int64_t incx, float _Complex *a)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hpr(ctxt->queue, convert(uplo), n,
                                        alpha, reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<std::complex<float> *>(a));
    ONEMKL_CATCH("")
  }
  void zHpr(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                const double _Complex *x, int64_t incx, double _Complex *a)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hpr(ctxt->queue, convert(uplo), n,
                                        alpha, reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<std::complex<double> *>(a));
    ONEMKL_CATCH("")
  }

  void cHpr2(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy, float _Complex *a)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hpr2(ctxt->queue, convert(uplo), n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<const std::complex<float> *>(y), incy,
                                        reinterpret_cast<std::complex<float> *>(a));
    ONEMKL_CATCH("")
  }
  void zHpr2(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy, double _Complex *a)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::hpr2(ctxt->queue, convert(uplo), n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<const std::complex<double> *>(y), incy,
                                        reinterpret_cast<std::complex<double> *>(a));
    ONEMKL_CATCH("")
  }

  void sSbmv(Context* ctxt, onemklUplo uplo, int64_t n, int64_t k,
                            float alpha, const float *a, int64_t lda, const float *x,
                            int64_t incx, float beta, float *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::sbmv(ctxt->queue, convert(uplo), n, k,
                                                    alpha, a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }
  void dSbmv(Context* ctxt, onemklUplo uplo, int64_t n, int64_t k,
                            double alpha, const double *a, int64_t lda, const double *x,
                            int64_t incx, double beta, double *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::sbmv(ctxt->queue, convert(uplo), n, k,
                                                    alpha, a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }

  void sSpmv(Context* ctxt, onemklUplo uplo, int64_t n,
                            float alpha, const float *a, const float *x,
                            int64_t incx, float beta, float *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::spmv(ctxt->queue, convert(uplo), n,
                                                    alpha, a, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }
  void dSpmv(Context* ctxt, onemklUplo uplo, int64_t n,
                            double alpha, const double *a, const double *x,
                            int64_t incx, double beta, double *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::spmv(ctxt->queue, convert(uplo), n,
                                                    alpha, a, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }

  void sSpr(Context* ctxt, onemklUplo uplo, int64_t n,
                  float alpha, const float *x, int64_t incx, float *a) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::spr(ctxt->queue, convert(uplo), n, alpha, x, incx, a);
    ONEMKL_CATCH("")
  }
  void dSpr(Context* ctxt, onemklUplo uplo, int64_t n,
                  double alpha, const double *x, int64_t incx, double *a) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::spr(ctxt->queue, convert(uplo), n, alpha, x, incx, a);
    ONEMKL_CATCH("")
  }

  void sSpr2(Context* ctxt, onemklUplo uplo, int64_t n,
                  float alpha, const float *x, int64_t incx,
                  const float *y, int64_t incy, float *a) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::spr2(ctxt->queue, convert(uplo), n, alpha,
                  x, incx, y, incy, a);
    ONEMKL_CATCH("")
  }

  void dSpr2(Context* ctxt, onemklUplo uplo, int64_t n,
                  double alpha, const double *x, int64_t incx,
                  const double *y, int64_t incy, double *a) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::spr2(ctxt->queue, convert(uplo), n, alpha,
                  x, incx, y, incy, a);
    ONEMKL_CATCH("")
  }

  void sSymv(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                            const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                            float *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::symv(ctxt->queue, convert(uplo), n, alpha,
                                                    a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }

  void dSymv(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                            const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                            double *y, int64_t incy) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::symv(ctxt->queue, convert(uplo), n, alpha,
                                                    a, lda, x, incx, beta, y, incy);
    ONEMKL_CATCH("")
  }

  void sSyr(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                            const float *x, int64_t incx, float *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::syr(ctxt->queue, convert(uplo), n, alpha,
                                                    x, incx, a, lda);
    ONEMKL_CATCH("")
  }

  void dSyr(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                            const double *x, int64_t incx, double *a, int64_t lda) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::syr(ctxt->queue, convert(uplo), n, alpha,
                                                    x, incx, a, lda);
    ONEMKL_CATCH("")
  }

  void sSyr2(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                            const float *x, int64_t incx, const float *y, int64_t incy, float *a, int64_t lda)
  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::syr2(ctxt->queue, convert(uplo), n, alpha, x, incx, y, incy, a, lda);
    ONEMKL_CATCH("")
  }
  void dSyr2(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                            const double *x, int64_t incx, const double *y, int64_t incy, double *a, int64_t lda)

  {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::syr2(ctxt->queue, convert(uplo), n, alpha, x, incx, y, incy, a, lda);
    ONEMKL_CATCH("")
  }

  void sTbmv(Context* ctxt, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float *a, int64_t lda, float *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbmv(ctxt->queue, convert(uplo), convert(trans),
                                                        convert(diag), n, k, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void dTbmv(Context* ctxt, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double *a, int64_t lda, double *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbmv(ctxt->queue, convert(uplo), convert(trans),
                                                    convert(diag), n, k, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void cTbmv(Context* ctxt, onemklUplo uplo,
                              onemklTranspose trans, onemklDiag diag, int64_t n,
                              int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbmv(ctxt->queue, convert(uplo), convert(trans),
                                            convert(diag), n, k, reinterpret_cast<const std::complex<float> *>(a),
                                            lda, reinterpret_cast<std::complex<float> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void zTbmv(Context* ctxt, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                            int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbmv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, k, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void sTbsv(Context* ctxt, onemklUplo uplo,
                              onemklTranspose trans, onemklDiag diag, int64_t n,
                              int64_t k, const float *a, int64_t lda, float *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbsv(ctxt->queue, convert(uplo), convert(trans),
                                                        convert(diag), n, k, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void dTbsv(Context* ctxt, onemklUplo uplo,
                              onemklTranspose trans, onemklDiag diag, int64_t n,
                              int64_t k, const double *a, int64_t lda, double *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbsv(ctxt->queue, convert(uplo), convert(trans),
                                                    convert(diag), n, k, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void cTbsv(Context* ctxt, onemklUplo uplo,
                              onemklTranspose trans, onemklDiag diag, int64_t n,
                              int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbsv(ctxt->queue, convert(uplo), convert(trans),
                                            convert(diag), n, k, reinterpret_cast<const std::complex<float> *>(a),
                                            lda, reinterpret_cast<std::complex<float> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void zTbsv(Context* ctxt, onemklUplo uplo,
                              onemklTranspose trans, onemklDiag diag, int64_t n,
                              int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tbsv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, k, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void sTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const float *a, float *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpmv(ctxt->queue, convert(uplo), convert(trans),
                                                    convert(diag), n, a, x, incx);
    ONEMKL_CATCH("")
  }

  void dTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const double *a, double *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpmv(ctxt->queue, convert(uplo), convert(trans),
                                                    convert(diag), n, a, x, incx);
    ONEMKL_CATCH("")
  }

  void cTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const float _Complex *a, float _Complex *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpmv(ctxt->queue, convert(uplo), convert(trans),
                                            convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                            reinterpret_cast<std::complex<float> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void zTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const double _Complex *a, double _Complex *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpmv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                        reinterpret_cast<std::complex<double> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void sTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const float *a, float *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpsv(ctxt->queue, convert(uplo), convert(trans),
                                                    convert(diag), m, a, x, incx);
    ONEMKL_CATCH("")
  }

  void dTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                const double *a, double *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpsv(ctxt->queue, convert(uplo), convert(trans),
                                                    convert(diag), m, a, x, incx);
    ONEMKL_CATCH("")
  }

  void cTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const float _Complex *a, float _Complex *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpsv(ctxt->queue, convert(uplo), convert(trans),
                                            convert(diag), m, reinterpret_cast<const std::complex<float> *>(a),
                                            reinterpret_cast<std::complex<float> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void zTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const double _Complex *a, double _Complex *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::tpsv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), m, reinterpret_cast<const std::complex<double> *>(a),
                                        reinterpret_cast<std::complex<double> *>(x), incx);
    ONEMKL_CATCH("")
  }

// trmv
  void sTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trmv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void dTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trmv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void cTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trmv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                        lda, reinterpret_cast<std::complex<float> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void zTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trmv(ctxt->queue, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    ONEMKL_CATCH("")
  }

// trsv
  void sTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trsv(ctxt->queue, convert(uplo), convert(trans),
                                          convert(diag), n, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void dTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                              int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trsv(ctxt->queue, convert(uplo), convert(trans),
                                          convert(diag), n, a, lda, x, incx);
    ONEMKL_CATCH("")
  }

  void cTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const float  _Complex *a, int64_t lda,
                              float _Complex *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trsv(ctxt->queue, convert(uplo), convert(trans),
                                          convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<std::complex<float> *>(x), incx);
    ONEMKL_CATCH("")
  }

  void zTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                              onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda,
                              double _Complex *x, int64_t incx) {
    ONEMKL_TRY
    auto status = oneapi::mkl::blas::column_major::trsv(ctxt->queue, convert(uplo), convert(trans),
                                          convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<std::complex<double> *>(x), incx);
    ONEMKL_CATCH("")
  }
}// end of namespace