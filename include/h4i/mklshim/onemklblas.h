// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{
  void onemklDamax(Context* ctxt, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
  void onemklSamax(Context* ctxt, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
  void onemklZamax(Context* ctxt, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
  void onemklCamax(Context* ctxt, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);
} // namespace