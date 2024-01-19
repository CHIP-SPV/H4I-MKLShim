// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once
#include <stdint.h>

namespace H4I::MKLShim
{

struct MKL_VERSION {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
};

enum Operation
{
    N = 0,
    T = 1,
    C = 2
};

enum Datatype
{
    Real8I = 0,
    Real32I = 1,
    Real16F = 2,
    Real32F = 3
};

struct Context;

  typedef enum {
    ONEMKL_SIDE_LEFT,
    ONEMKL_SIDE_RIGHT,
    ONEMKL_SIDE_BOTH
  } onemklSideMode;

  typedef enum {
    ONEMKL_DIAG_NONUNIT,
    ONEMKL_DIAG_UNIT
  } onemklDiag;

    typedef enum {
    ONEMKL_GEN_Q,
    ONEMKL_GEN_P
  } onemklGen;

  typedef enum {
    ONEMKL_UPLO_UPPER,
    ONEMKL_UPLO_LOWER
  } onemklUplo;

  typedef enum {
    ONEMKL_JOB_NOVEC,
    ONEMKL_JOB_VEC
  } onemklJob;

  typedef enum {
    ONEMKL_TRANSPOSE_NONTRANS,
    ONEMKL_TRANSPOSE_TRANS,
    ONEMLK_TRANSPOSE_CONJTRANS
  } onemklTranspose;

// keep track of the four combinations of
// mkl fft precision (Single or Double) and
// starting domain (Real or Complex)
struct fftDescriptorSR;
struct fftDescriptorSC;
struct fftDescriptorDR;
struct fftDescriptorDC;

  typedef enum
  {
      ONEMKL_R_16F,
      ONEMKL_R_32F,
      ONEMKL_R_64F,
      ONEMKL_C_16F,
      ONEMKL_C_32F,
      ONEMKL_C_64F,
      ONEMKL_R_8I,
      ONEMKL_R_8U,
      ONEMKL_R_32I,
      ONEMKL_R_32U,
      ONEMKL_C_8I,
      ONEMKL_C_8U,
      ONEMKL_C_32I,
      ONEMKL_C_32U,
      ONEMKL_R_16B,
      ONEMKL_C_16B,
      ONEMKL_DATATYPE_INVALID
  } onemklDatatype_t;

} // namespace
