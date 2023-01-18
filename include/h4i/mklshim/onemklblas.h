// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{
  void sAsum(Context* ctxt, int64_t n, const float *x, int64_t incx,
             float *result);
  void dAsum(Context* ctxt, int64_t n, const double *x, int64_t incx,
             double *result);
  void cAsum(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
             float *result);
  void zAsum(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
             double *result);

  void sAxpy(Context* ctxt, int64_t n, float alpha, const float *x, int64_t incx,
             float *y, int64_t incy);
  void dAxpy(Context* ctxt, int64_t n, double alpha, const double *x, int64_t incx,
             double *y, int64_t incy);
  void cAxpy(Context* ctxt, int64_t n, float _Complex alpha, const float _Complex *x,
             int64_t incx, float _Complex *y, int64_t incy);
  void zAxpy(Context* ctxt, int64_t n, double _Complex alpha, const double _Complex *x,
             int64_t incx, double _Complex *y, int64_t incy);

  void sAmax(Context* ctxt, int64_t n, const float  *x, int64_t incx,
              int64_t *result);
  void dAmax(Context* ctxt, int64_t n, const double *x, int64_t incx,
              int64_t *result);
  void cAmax(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
              int64_t *result);
  void zAmax(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
              int64_t *result);
} // namespace