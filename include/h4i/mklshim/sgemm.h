// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

#include "h4i/mklshim/types.h"
#include "h4i/mklshim/Context.h"

namespace H4I::MKLShim
{

void SGEMM(Context* ctxt,
        Operation transa,
        Operation transb,
        int m,
        int n,
        int k,
        const float* alpha,
        const float* A,
        int ldA,
        const float* B,
        int ldB,
        const float* beta,
        float* C,
        int ldC);

} // namespace

