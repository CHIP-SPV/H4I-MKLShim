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
}// end of namespace