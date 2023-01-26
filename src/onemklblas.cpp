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
}// end of namespace