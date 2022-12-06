// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

Context*
Create(void)
{
    return new Context();
}

void
Destroy(Context* ctxt)
{
    delete ctxt;
}

} // namespace

