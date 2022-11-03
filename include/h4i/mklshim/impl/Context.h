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

} // namespace

