# pragma once

namespace H4I::MKLShim
{
  //gebrd
  int64_t Sgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Dgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Cgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Zgebrd_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  void Sgebrd(Context* ctxt, int64_t m, int64_t n, float* a, int64_t lda,
                    float* d, float* e, float* tauq, float* taup, float* scratchpad, int64_t scratchpad_size);
  void Dgebrd(Context* ctxt, int64_t m, int64_t n, double* a, int64_t lda,
                    double* d, double* e, double* tauq, double* taup, double* scratchpad, int64_t scratchpad_size);
  void Cgebrd(Context* ctxt, int64_t m, int64_t n, float _Complex* a, int64_t lda,
                    float* d, float* e, float _Complex* tauq, float _Complex* taup, float _Complex* scratchpad,
                    int64_t scratchpad_size);
  void Zgebrd(Context* ctxt, int64_t m, int64_t n, double _Complex* a, int64_t lda,
                    double* d, double* e, double _Complex* tauq, double _Complex* taup, double _Complex* scratchpad,
                    int64_t scratchpad_size);

  //orgbr/ungbr
  int64_t Sorgbr_ScPadSz(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, int64_t lda);
  int64_t Dorgbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k, int64_t lda);
  int64_t Cungbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k, int64_t lda);
  int64_t Zungbr_ScPadSz(Context* ctxt,onemklGen gen, int64_t m, int64_t n, int64_t k, int64_t lda);
  void Sorgbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, float* A, int64_t lda,
                    float* tua, float* scratchpad, int64_t scratchpad_size);
  void Dorgbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                    double* tua, double* scratchpad, int64_t scratchpad_size);
  void Cungbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, float _Complex* A, int64_t lda,
                    float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zungbr(Context* ctxt, onemklGen gen, int64_t m, int64_t n, int64_t k, double _Complex* A, int64_t lda,
                    double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size);

  // orgqr/ungqr
  int64_t Sorgqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda);
  int64_t Dorgqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda);
  int64_t Cungqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda);
  int64_t Zungqr_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t k, int64_t lda);
  void Sorgqr(Context* ctxt, int64_t m, int64_t n, int64_t k, float* A, int64_t lda,
                    float* tua, float* scratchpad, int64_t scratchpad_size);
  void Dorgqr(Context* ctxt, int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                    double* tua, double* scratchpad, int64_t scratchpad_size);
  void Cungqr(Context* ctxt, int64_t m, int64_t n, int64_t k, float _Complex* A, int64_t lda,
                    float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zungqr(Context* ctxt, int64_t m, int64_t n, int64_t k, double _Complex* A, int64_t lda,
                    double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size);

  // orgtr/ungtr
  int64_t Sorgtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Dorgtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Cungtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Zungtr_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  void Sorgtr(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda,
                    float* tua, float* scratchpad, int64_t scratchpad_size);
  void Dorgtr(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda,
                    double* tua, double* scratchpad, int64_t scratchpad_size);
  void Cungtr(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda,
                    float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zungtr(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda,
                    double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size);

  // ormqr/unmqr
  int64_t Sormqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc);
  int64_t Dormqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc);
  int64_t Cunmqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc);
  int64_t Zunmqr_ScPadSz(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldc);
  void Sormqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    float* A, int64_t lda, float* tua, float* C, int64_t ldc, float* scratchpad, int64_t scratchpad_size);
  void Dormqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    double* A, int64_t lda, double* tua, double* C, int64_t ldc, double* scratchpad, int64_t scratchpad_size);
  void Cunmqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    float _Complex* A, int64_t lda, float _Complex* tua, float _Complex* C, int64_t ldc,
                    float _Complex* scratchpad, int64_t scratchpad_size);
  void Zunmqr(Context* ctxt, onemklSideMode side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    double _Complex* A, int64_t lda, double _Complex* tua, double _Complex* C, int64_t ldc,
                    double _Complex* scratchpad, int64_t scratchpad_size);

  // ormtr/unmtr
  int64_t Sormtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc);
  int64_t Dormtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc);
  void Sormtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float* A, int64_t lda,
              float* tua, float* C, int64_t ldc, float* scratchpad, int64_t scratchpad_size);
  void Dormtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double* A, int64_t lda,
              double* tua, double* C, int64_t ldc, double* scratchpad, int64_t scratchpad_size);
  int64_t Cunmtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc);
  int64_t Zunmtr_ScPadSz(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc);
  void Cunmtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float _Complex* A, int64_t lda,
              float _Complex* tua, float _Complex* C, int64_t ldc, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zunmtr(Context* ctxt, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double _Complex* A, int64_t lda,
              double _Complex* tua, double _Complex* C, int64_t ldc, double _Complex* scratchpad, int64_t scratchpad_size);

  //geqrf
  int64_t Sgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Dgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Cgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Zgeqrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  void Sgeqrf(Context* ctxt, int64_t m, int64_t n, float* A, int64_t lda, float* tua, float* scratchpad, int64_t scratchpad_size);
  void Dgeqrf(Context* ctxt, int64_t m, int64_t n, double* A, int64_t lda, double* tua, double* scratchpad, int64_t scratchpad_size);
  void Cgeqrf(Context* ctxt, int64_t m, int64_t n, float _Complex* A, int64_t lda, float _Complex* tua, float _Complex* scratchpad,
             int64_t scratchpad_size);
  void Zgeqrf(Context* ctxt, int64_t m, int64_t n, double _Complex* A, int64_t lda, double _Complex* tua, double _Complex* scratchpad,
             int64_t scratchpad_size);

  //getrf
  int64_t Sgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Dgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Cgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  int64_t Zgetrf_ScPadSz(Context* ctxt, int64_t m, int64_t n, int64_t lda);
  void Sgetrf(Context* ctxt, int64_t m, int64_t n, float* A, int64_t lda, int64_t *ipiv, float* scratchpad, int64_t scratchpad_size);
  void Dgetrf(Context* ctxt, int64_t m, int64_t n, double* A, int64_t lda, int64_t *ipiv, double* scratchpad, int64_t scratchpad_size);
  void Cgetrf(Context* ctxt, int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t *ipiv, float _Complex* scratchpad,
             int64_t scratchpad_size);
  void Zgetrf(Context* ctxt, int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t *ipiv, double _Complex* scratchpad,
             int64_t scratchpad_size);

  //getrs
  int64_t Sgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  int64_t Dgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  int64_t Cgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  int64_t Zgetrs_ScPadSz(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  void Sgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, float* A, int64_t lda, std::int64_t *ipiv,
              float* B, int64_t ldb, float* scratchpad, int64_t scratchpad_size);
  void Dgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, double* A, int64_t lda, std::int64_t *ipiv,
              double* B, int64_t ldb, double* scratchpad, int64_t scratchpad_size);
  void Cgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex* A, int64_t lda, std::int64_t *ipiv,
              float _Complex* B, int64_t ldb, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zgetrs(Context* ctxt, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex* A, int64_t lda, std::int64_t *ipiv,
              double _Complex* B, int64_t ldb, double _Complex* scratchpad, int64_t scratchpad_size);

  //potrf
  int64_t Spotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Dpotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Cpotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Zpotrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  void Spotrf(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* scratchpad, int64_t scratchpad_size);
  void Dpotrf(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* scratchpad, int64_t scratchpad_size);
  void Cpotrf(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zpotrf(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* scratchpad, int64_t scratchpad_size);

  //potri
  int64_t Spotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Dpotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Cpotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Zpotri_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  void Spotri(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* scratchpad, int64_t scratchpad_size);
  void Dpotri(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* scratchpad, int64_t scratchpad_size);
  void Cpotri(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zpotri(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* scratchpad, int64_t scratchpad_size);

  //potrs
  int64_t Spotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  int64_t Dpotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  int64_t Cpotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  int64_t Zpotrs_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb);
  void Spotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, float* A, int64_t lda, std::int64_t *ipiv,
              float* B, int64_t ldb, float* scratchpad, int64_t scratchpad_size);
  void Dpotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, double* A, int64_t lda, std::int64_t *ipiv,
              double* B, int64_t ldb, double* scratchpad, int64_t scratchpad_size);
  void Cpotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex* A, int64_t lda, std::int64_t *ipiv,
              float _Complex* B, int64_t ldb, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zpotrs(Context* ctxt, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex* A, int64_t lda, std::int64_t *ipiv,
              double _Complex* B, int64_t ldb, double _Complex* scratchpad, int64_t scratchpad_size);

  //gesvd
  int64_t Sgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt);
  int64_t Dgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt);
  int64_t Cgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt);
  int64_t Zgesvd_ScPadSz(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                              int64_t lda, int64_t ldu, int64_t ldvt);
  void Sgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, float* A, int64_t lda,
                      float* S, float* U, int64_t ldu, float* V, int64_t ldv, float* scratchpad, int64_t scratchpad_size);
  void Dgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, double* A, int64_t lda,
                      double* S, double* U, int64_t ldu, double* V, int64_t ldv, double* scratchpad, int64_t scratchpad_size);
  void Cgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, float _Complex* A, int64_t lda,
                      float* S, float _Complex* U, int64_t ldu, float _Complex* V, int64_t ldv, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zgesvd(Context* ctxt, signed char jobu, signed char jobvt, int64_t m, int64_t n, double _Complex* A, int64_t lda,
                      double* S, double _Complex* U, int64_t ldu, double _Complex* V, int64_t ldv, double _Complex* scratchpad, int64_t scratchpad_size);

  //syevd/heevd
  int64_t Ssyevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Dsyevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
  void Ssyevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* w, float* scratchpad, int64_t scratchpad_size);
  void Dsyevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* w, double* scratchpad, int64_t scratchpad_size);

  int64_t Cheevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Zheevd_ScPadSz(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
  void Cheevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* w,
                    float _Complex* scratchpad, int64_t scratchpad_size);
  void Zheevd(Context* ctxt, onemklJob job, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* w,
                    double _Complex* scratchpad, int64_t scratchpad_size);

  //sytrf
  int64_t Ssytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Dsytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Csytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Zsytrf_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  void Ssytrf(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, int64_t* ipiv, float* scratchpad, int64_t scratchpad_size);
  void Dsytrf(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, int64_t* ipiv, double* scratchpad, int64_t scratchpad_size);
  void Csytrf(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t* ipiv, float _Complex* scratchpad, int64_t scratchpad_size);
  void Zsytrf(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t* ipiv, double _Complex* scratchpad, int64_t scratchpad_size);

  //sytrd/hetrd
  int64_t Ssytrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Dsytrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Chetrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  int64_t Zhetrd_ScPadSz(Context* ctxt, onemklUplo uplo, int64_t n, int64_t lda);
  void Ssytrd(Context* ctxt, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* d, float* e, float* tau, float* scratchpad, int64_t scratchpad_size);
  void Dsytrd(Context* ctxt, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* d, double* e, double* tau, double* scratchpad, int64_t scratchpad_size);
  void Chetrd(Context* ctxt, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* d, float* e, float _Complex* tau, float* scratchpad, int64_t scratchpad_size);
  void Zhetrd(Context* ctxt, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* d, double* e, double _Complex* tau, double* scratchpad, int64_t scratchpad_size);

}// H4I::MKLShim