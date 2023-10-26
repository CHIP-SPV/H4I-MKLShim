#include "h4i/mklshim/common.h"

namespace H4I::MKLShim
{
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

  oneapi::mkl::jobsvd convert(signed char j) {
    switch(j) {
      case 'N': return oneapi::mkl::jobsvd::N;
      case 'A': return oneapi::mkl::jobsvd::A;
      case 'S': return oneapi::mkl::jobsvd::S;
      case 'O': return oneapi::mkl::jobsvd::O;
      default : return oneapi::mkl::jobsvd::N; // need to test
    }
  }

  oneapi::mkl::job convert(onemklJob j) {
    switch(j) {
      case ONEMKL_JOB_NOVEC: return oneapi::mkl::job::novec;
      case ONEMKL_JOB_VEC: return oneapi::mkl::job::vec;
    }
  }

  oneapi::mkl::generate convert(onemklGen g) {
    switch(g) {
      case ONEMKL_GEN_Q: return oneapi::mkl::generate::q;
      case ONEMKL_GEN_P: return oneapi::mkl::generate::p;
    }
  }

  MKL_VERSION get_mkl_version() {
    MKL_VERSION mkl_version;
    MKLVersion version;
    mkl_get_version(&version);

    mkl_version.major = version.MajorVersion;
    mkl_version.minor = version.MinorVersion;
    mkl_version.patch = version.UpdateVersion;
    return mkl_version;
  }
} // namespace