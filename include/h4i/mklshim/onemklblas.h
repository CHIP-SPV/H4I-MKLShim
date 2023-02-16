// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{
  typedef enum {
    ONEMKL_TRANSPOSE_NONTRANS,
    ONEMKL_TRANSPOSE_TRANS,
    ONEMLK_TRANSPOSE_CONJTRANS
  } onemklTranspose;

  typedef enum {
    ONEMKL_UPLO_UPPER,
    ONEMKL_UPLO_LOWER
  } onemklUplo;

  typedef enum {
    ONEMKL_SIDE_LEFT,
    ONEMKL_SIDE_RIGHT,
    ONEMKL_SIDE_BOTH
  } onemklSideMode;

  typedef enum {
    ONEMKL_DIAG_NONUNIT,
    ONEMKL_DIAG_UNIT
  } onemklDiag;

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

  void sGbmv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, int64_t kl, int64_t ku,
                  float alpha, const float *a, int64_t lda,
                  const float *x, int64_t incx, float beta, float *y,
                  int64_t incy);

  void dGbmv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, int64_t kl, int64_t ku,
                  double alpha, const double *a, int64_t lda,
                  const double *x, int64_t incx, double beta, double *y,
                  int64_t incy);

  void cGbmv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, int64_t kl, int64_t ku,
                  float _Complex alpha, const float _Complex *a, int64_t lda,
                  const float _Complex *x, int64_t incx, float _Complex beta,
                  float _Complex *y, int64_t incy);

  void zGbmv(Context* ctxt, onemklTranspose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                  double _Complex alpha, const double _Complex *a, int64_t lda,
                  const double _Complex *x, int64_t incx, double _Complex beta,
                  double _Complex *y, int64_t incy);

  void sGemv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, float alpha, const float *a,
                  int64_t lda, const float *x, int64_t incx, float beta,
                  float *y, int64_t incy);
  void dGemv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, double alpha, const double *a,
                  int64_t lda, const double *x, int64_t incx, double beta,
                  double *y, int64_t incy);
   void cGemv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, float _Complex alpha,
                  const float _Complex *a, int64_t lda,
                  const float _Complex *x, int64_t incx,
                  float _Complex beta, float _Complex *y,
                  int64_t incy);
  void zGemv(Context* ctxt, onemklTranspose trans,
                  int64_t m, int64_t n, double _Complex alpha,
                  const double _Complex *a, int64_t lda,
                  const double _Complex *x, int64_t incx,
                  double _Complex beta, double _Complex *y,
                  int64_t incy);
  void sGer(Context* ctxt, int64_t m, int64_t n, float alpha,
                  const float *x, int64_t incx, const float *y, int64_t incy,
                  float *a, int64_t lda);
  void dGer(Context* ctxt, int64_t m, int64_t n, double alpha,
                  const double *x, int64_t incx, const double *y, int64_t incy,
                  double *a, int64_t lda);
  void cGerc(Context* ctxt, int64_t m, int64_t n, float _Complex alpha,
                  const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                  float _Complex *a, int64_t lda);
  void cGeru(Context* ctxt, int64_t m, int64_t n, float _Complex alpha,
                  const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                  float _Complex *a, int64_t lda);
  void zGerc(Context* ctxt, int64_t m, int64_t n, double _Complex alpha,
                  const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                  double _Complex *a, int64_t lda);
  void zGeru(Context* ctxt, int64_t m, int64_t n, double _Complex alpha,
                  const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                  double _Complex *a, int64_t lda);
  void cHbmv(Context* ctxt, onemklUplo uplo, int64_t n,
                  int64_t k, float _Complex alpha, const float _Complex *a,
                  int64_t lda, const float _Complex *x, int64_t incx, float _Complex beta,
                  float _Complex *y, int64_t incy);
  void zHbmv(Context* ctxt, onemklUplo uplo, int64_t n,
                  int64_t k, double _Complex alpha, const double _Complex *a,
                  int64_t lda, const double _Complex *x, int64_t incx, double _Complex beta,
                  double _Complex *y, int64_t incy);
  void cHemv(Context* ctxt, onemklUplo uplo, int64_t n,
                  float _Complex alpha, const float _Complex *a, int64_t lda,
                  const float _Complex *x, int64_t incx, float _Complex beta,
                  float _Complex *y, int64_t incy);
  void zHemv(Context* ctxt, onemklUplo uplo, int64_t n,
                  double _Complex alpha, const double _Complex *a, int64_t lda,
                  const double _Complex *x, int64_t incx, double _Complex beta,
                  double _Complex *y, int64_t incy);
  void cHer(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                  const float _Complex *x, int64_t incx, float _Complex *a,
                  int64_t lda);
  void zHer(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                  const double _Complex *x, int64_t incx, double _Complex *a,
                  int64_t lda);
  void cHer2(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex alpha,
                  const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                  float _Complex *a, int64_t lda);
  void zHer2(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex alpha,
                  const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                  double _Complex *a, int64_t lda);
  void cHpmv(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex alpha,
                  const float _Complex *a, const float _Complex *x, int64_t incx,
                  float _Complex beta, float _Complex *y, int64_t incy);
  void zHpmv(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex alpha,
                  const double _Complex *a, const double _Complex *x, int64_t incx,
                  double _Complex beta, double _Complex *y, int64_t incy);
  void cHpr(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                  const float _Complex *x, int64_t incx, float _Complex *a);
  void zHpr(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                  const double _Complex *x, int64_t incx, double _Complex *a);
  void cHpr2(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex alpha,
                  const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy, float _Complex *a);
  void zHpr2(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex alpha,
                  const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy, double _Complex *a);
  void sSbmv(Context* ctxt, onemklUplo uplo, int64_t n, int64_t k,
                  float alpha, const float *a, int64_t lda, const float *x,
                  int64_t incx, float beta, float *y, int64_t incy);
  void dSbmv(Context* ctxt, onemklUplo uplo, int64_t n, int64_t k,
                  double alpha, const double *a, int64_t lda, const double *x,
                  int64_t incx, double beta, double *y, int64_t incy);
  void sSpmv(Context* ctxt, onemklUplo uplo, int64_t n,
                  float alpha, const float *a, const float *x,
                  int64_t incx, float beta, float *y, int64_t incy);
  void dSpmv(Context* ctxt, onemklUplo uplo, int64_t n,
                  double alpha, const double *a, const double *x,
                  int64_t incx, double beta, double *y, int64_t incy);
  void sSpr(Context* ctxt, onemklUplo uplo, int64_t n,
                  float alpha, const float *x, int64_t incx, float *a);
  void dSpr(Context* ctxt, onemklUplo uplo, int64_t n,
                  double alpha, const double *x, int64_t incx, double *a);
  void sSpr2(Context* ctxt, onemklUplo uplo, int64_t n,
                  float alpha, const float *x, int64_t incx,
                  const float *y, int64_t incy, float *a);
  void dSpr2(Context* ctxt, onemklUplo uplo, int64_t n,
                  double alpha, const double *x, int64_t incx,
                  const double *y, int64_t incy, double *a);
  void sSymv(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                  const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                  float *y, int64_t incy);
  void dSymv(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                  const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                  double *y, int64_t incy);
  void sSyr(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                  const float *x, int64_t incx, float *a, int64_t lda);
  void dSyr(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                  const double *x, int64_t incx, double *a, int64_t lda);
  void sSyr2(Context* ctxt, onemklUplo uplo, int64_t n, float alpha,
                  const float *x, int64_t incx, const float *y, int64_t incy, float *a, int64_t lda);
  void dSyr2(Context* ctxt, onemklUplo uplo, int64_t n, double alpha,
                  const double *x, int64_t incx, const double *y, int64_t incy, double *a, int64_t lda);
  void sTbmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const float *a, int64_t lda, float *x, int64_t incx);
  void dTbmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const double *a, int64_t lda, double *x, int64_t incx);
  void cTbmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                  int64_t incx);
  void zTbmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                  int64_t incx);
  void sTbsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const float *a, int64_t lda, float *x, int64_t incx);
  void dTbsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const double *a, int64_t lda, double *x, int64_t incx);
  void cTbsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                  int64_t incx);
  void zTbsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                  int64_t incx);
  void sTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const float *a, float *x, int64_t incx);
  void dTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const double *a, double *x, int64_t incx);
  void cTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const float _Complex *a, float _Complex *x, int64_t incx);
  void zTpmv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t n,
                  const double _Complex *a, double _Complex *x, int64_t incx);
  void sTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const float *a, float *x, int64_t incx);
  void dTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const double *a, double *x, int64_t incx);
  void cTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const float _Complex *a, float _Complex *x, int64_t incx);
  void zTpsv(Context* ctxt, onemklUplo uplo,
                  onemklTranspose trans, onemklDiag diag, int64_t m,
                  const double _Complex *a, double _Complex *x, int64_t incx);
  void sTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                  int64_t incx);
  void dTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                  int64_t incx);
  void cTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                        int64_t incx);
  void zTrmv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                  int64_t incx);
  void sTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                  int64_t incx);
  void dTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                  int64_t incx);
  void cTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const float  _Complex *a, int64_t lda,
                  float _Complex *x, int64_t incx);
  void zTrsv(Context* ctxt, onemklUplo uplo, onemklTranspose trans,
                  onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda,
                  double _Complex *x, int64_t incx);

  /*void hGemm(Context* ctxt, onemklTranspose transA,
                            onemklTranspose transB, int64_t m, int64_t n,
                            int64_t k, sycl::half alpha, const sycl::half *A, int64_t lda,
                            const sycl::half *B, int64_t ldb, sycl::half beta, sycl::half *C,
                            int64_t ldc);*/
  void sGemm(Context* ctxt, onemklTranspose transA,
                            onemklTranspose transB, int64_t m, int64_t n,
                            int64_t k, float alpha, const float *A, int64_t lda,
                            const float *B, int64_t ldb, float beta, float *C,
                            int64_t ldc);

  void dGemm(Context* ctxt, onemklTranspose transA,
                            onemklTranspose transB, int64_t m, int64_t n,
                            int64_t k, double alpha, const double *A,
                            int64_t lda, const double *B, int64_t ldb,
                            double beta, double *C, int64_t ldc);

  void cGemm(Context* ctxt, onemklTranspose transA,
                            onemklTranspose transB, int64_t m, int64_t n,
                            int64_t k, float _Complex alpha,
                            const float _Complex *A, int64_t lda,
                            const float _Complex *B, int64_t ldb,
                            float _Complex beta, float _Complex *C,
                            int64_t ldc);

  void zGemm(Context* ctxt, onemklTranspose transA,
                            onemklTranspose transB, int64_t m, int64_t n,
                            int64_t k, double _Complex alpha,
                            const double _Complex *A, int64_t lda,
                            const double _Complex *B, int64_t ldb,
                            double _Complex beta, double _Complex *C,
                            int64_t ldc);

  void cHerk(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  float alpha, const float _Complex* a, int64_t lda, float beta, float _Complex* c, int64_t ldc);
  void zHerk(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  double alpha, const double _Complex* a, int64_t lda, double beta, double _Complex* c, int64_t ldc);

  void cHer2k(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                  float beta, float _Complex* c, int64_t ldc);
  void zHer2k(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  double _Complex alpha, const double _Complex* a, int64_t lda,  const double _Complex* b, int64_t ldb,
                  double beta, double _Complex* c, int64_t ldc);

  void sSymm(Context* ctxt, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                  float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
                  float beta, float* c, int64_t ldc);
  void dSymm(Context* ctxt, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                  double alpha, const double* a, int64_t lda, const double* b, int64_t ldb,
                  double beta, double* c, int64_t ldc);
  void cSymm(Context* ctxt, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                  float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                  float _Complex beta, float _Complex* c, int64_t ldc);
  void zSymm(Context* ctxt, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                  double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                  double _Complex beta, double _Complex* c, int64_t ldc);

  void sSyrk(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  float alpha, const float* a, int64_t lda, float beta, float* c, int64_t ldc);
  void dSyrk(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  double alpha, const double* a, int64_t lda, double beta, double* c, int64_t ldc);
  void cSyrk(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  float _Complex alpha, const float _Complex* a, int64_t lda, float _Complex beta, float _Complex* c, int64_t ldc);
  void zSyrk(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  double _Complex alpha, const double _Complex* a, int64_t lda, double _Complex beta, double _Complex* c, int64_t ldc);

  void sSyr2k(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  float alpha, const float* a, int64_t lda, const float* b, int64_t ldb, float beta, float* c, int64_t ldc);
  void dSyr2k(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc);
  void cSyr2k(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                  float _Complex beta, float _Complex* c, int64_t ldc);
  void zSyr2k(Context* ctxt, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                  double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                  double _Complex beta, double _Complex* c, int64_t ldc);

  void cHemm(Context* ctxt, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                  float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                  float _Complex beta, float _Complex* c, int64_t ldc);
  void zHemm(Context* ctxt, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                  double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                  double _Complex beta, double _Complex* c, int64_t ldc);

  void sTrmm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t ldb);
  void dTrmm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, double alpha, const double *a, int64_t lda, double *b, int64_t ldb);
  void cTrmm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, float _Complex alpha, const float _Complex*a, int64_t lda, float _Complex*b, int64_t ldb);
  void zTrmm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb);

  void sTrsm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t ldb);
  void dTrsm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, double alpha, const double *a, int64_t lda, double *b, int64_t ldb);
  void cTrsm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, float _Complex alpha, const float _Complex*a, int64_t lda, float _Complex*b, int64_t ldb);
  void zTrsm(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                  int64_t m, int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb);

} // namespace