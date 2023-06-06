// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

namespace H4I::MKLShim
{

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

} // namespace
