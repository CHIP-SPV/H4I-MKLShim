// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

namespace H4I::MKLShim
{
    struct Context
    {
        sycl::queue queue;
        sycl::device device;
        sycl::context context;
        sycl::platform platform;

        // Create default SYCL context and queue.
        // TODO if we have multiple GPUs in a node, how do we
        // select which one to use?
        Context(void)
        {
            // Nothing else to do.
        }
    };

} // namespace

