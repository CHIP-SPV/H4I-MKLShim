// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

void
SetStream(Context* ctxt, unsigned long const* backendHandles, int numOfHandles, const char* backendName)
{
    if(ctxt != nullptr)
    {
        if (context_tbl.find(backendHandles[3]) != context_tbl.end()) {
            ctxt = context_tbl[backendHandles[3]];
        } else {
            // new context hence update corresponding sycl queue and other structures .....
            Update(ctxt, backendHandles, numOfHandles, backendName);
        }
    }
}

} // namespace

