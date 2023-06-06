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
}// H4I::MKLShim