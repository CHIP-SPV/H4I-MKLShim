// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{
  void sDot(Context* ctxt, int64_t n, const float *x, int64_t incx, const float *y,
            int64_t incy, float *result);
  void dDot(Context* ctxt, int64_t n, const double *x, int64_t incx, const double *y,
            int64_t incy, double *result);
  void cDotc(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
             const float _Complex *y, int64_t incy, float _Complex *result);
  void zDotc(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
             const double _Complex *y, int64_t incy, double _Complex *result);
  void cDotu(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
             const float _Complex *y, int64_t incy, float _Complex *result);
  void zDotu(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
             const double _Complex *y, int64_t incy, double _Complex *result);

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

  void dScal(Context* ctxt, int64_t n, double alpha, double *x, int64_t incx);
  void sScal(Context* ctxt, int64_t n, float alpha, float *x, int64_t incx);
  void cScal(Context* ctxt, int64_t n, float _Complex alpha, float _Complex *x,
              int64_t incx);
  void csScal(Context* ctxt, int64_t n, float alpha, float _Complex *x, int64_t incx);
  void zsScal(Context* ctxt, int64_t n, double _Complex alpha, double _Complex *x,
              int64_t incx);
  void zdScal(Context* ctxt, int64_t n, double alpha, double _Complex *x, int64_t incx);

  void dNrm2(Context* ctxt, int64_t n, const double *x, int64_t incx,
              double *result);
  void sNrm2(Context* ctxt, int64_t n, const float *x,int64_t incx,
              float *result);
  void cNrm2(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
              float *result);
  void zNrm2(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
              double *result);

  void dCopy(Context* ctxt, int64_t n, const double *x,int64_t incx,
              double *y, int64_t incy);
  void sCopy(Context* ctxt, int64_t n, const float *x,int64_t incx,
        float *y, int64_t incy);
  void zCopy(Context* ctxt, int64_t n, const double _Complex *x,
              int64_t incx, double _Complex *y, int64_t incy);
  void cCopy(Context* ctxt, int64_t n, const float _Complex *x,
              int64_t incx, float _Complex *y, int64_t incy);

  void sAmax(Context* ctxt, int64_t n, const float  *x, int64_t incx,
              int64_t *result);
  void dAmax(Context* ctxt, int64_t n, const double *x, int64_t incx,
              int64_t *result);
  void cAmax(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
              int64_t *result);
  void zAmax(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
              int64_t *result);

  void dAmin(Context* ctxt, int64_t n, const double *x, int64_t incx, int64_t *result);
  void sAmin(Context* ctxt, int64_t n, const float  *x, int64_t incx, int64_t *result);
  void zAmin(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx, int64_t *result);
  void cAmin(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx, int64_t *result);

  void sSwap(Context* ctxt, int64_t n, float *x, int64_t incx, float *y, int64_t incy);
  void dSwap(Context* ctxt, int64_t n, double *x, int64_t incx, double *y, int64_t incy);
  void cSwap(Context* ctxt, int64_t n, float _Complex *x, int64_t incx,
              float _Complex *y, int64_t incy);
  void zSwap(Context* ctxt, int64_t n, double _Complex *x, int64_t incx,
              double _Complex *y, int64_t incy);

  void sRot(Context* ctxt, int n, float* x, int incx, float* y, int incy,
              const float c, const float s);
  void dRot(Context* ctxt, int n, double* x, int incx, double* y, int incy,
              const double c, const double s);
  void cRot(Context* ctxt, int n, float _Complex* x, int incx, float _Complex* y, int incy,
              const float c, const float _Complex s);
  void csRot(Context* ctxt, int n, float _Complex* x, int incx, float _Complex* y, int incy,
              const float c, const float s);
  void zRot(Context* ctxt, int n, double _Complex* x, int incx, double _Complex* y, int incy,
              const double c, const double _Complex s);
  void zdRot(Context* ctxt, int n, double _Complex* x, int incx, double _Complex* y, int incy,
              const double c, const double s);

  void sRotg(Context* ctxt, float* a, float* b, float* c, float* s);
  void dRotg(Context* ctxt, double* a, double* b, double* c, double* s);
  void cRotg(Context* ctxt, float _Complex* a, float _Complex* b, float* c, float _Complex* s);
  void zRotg(Context* ctxt, double _Complex* a, double _Complex* b, double* c, double _Complex* s);

  void sRotm(Context* ctxt, int64_t n, float *x, int64_t incx, float *y, int64_t incy, float* param);
  void dRotm(Context* ctxt, int64_t n, double *x, int64_t incx, double *y, int64_t incy, double* param);

  void sRotmg(Context* ctxt, float *d1, float *d2, float *x1, float y1, float* param);
  void dRotmg(Context* ctxt, double *d1, double *d2, double *x1, double y1, double* param);
} // namespace