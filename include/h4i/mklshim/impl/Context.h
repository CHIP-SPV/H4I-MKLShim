// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

namespace H4I::MKLShim
{
    struct Context
    {
        sycl::platform platform;
        sycl::device device;
        sycl::context context;
        sycl::queue queue;

        // Create default SYCL context and queue.
        // TODO Does this create a context for running SYCL on host CPU cores?
        Context(void)
          : platform(),
            device(),
            context(),
            queue()
        {
            // Nothing else to do.
        }
    };

inline
Context*
Create(void)
{
    return new Context();
}

inline
void
Destroy(Context* ctxt)
{
    delete ctxt;
}

} // namespace

