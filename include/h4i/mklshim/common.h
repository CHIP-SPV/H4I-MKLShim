
#pragma once
#include "h4i/mklshim/types.h"
#include "oneapi/mkl.hpp"

namespace H4I::MKLShim
{
  oneapi::mkl::transpose convert(onemklTranspose val);
  oneapi::mkl::uplo convert(onemklUplo val);
  oneapi::mkl::side convert(onemklSideMode val);
  oneapi::mkl::diag convert(onemklDiag val);
  oneapi::mkl::jobsvd convert(signed char j);
  oneapi::mkl::job convert(onemklJob j);
  oneapi::mkl::generate convert(onemklGen g);
} // namespace