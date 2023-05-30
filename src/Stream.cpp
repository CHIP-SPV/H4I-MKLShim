// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

void
SetStream(Context* ctxt, const std::array<uintptr_t, nHandles>& nativeHandles)
{
    if(ctxt != nullptr)
    {
        if (context_tbl.find(nativeHandles[3]) != context_tbl.end()) {
            ctxt = context_tbl[nativeHandles[3]];
        } else {
            // new context hence update corresponding sycl queue and other structures .....
            std::string backendName = (currentBackend == level0) ? "level0" : "opencl";
            Update(ctxt, nativeHandles.data(), nativeHandles.size(), backendName.c_str());
        }
    }
}

} // namespace

