// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once
#include <sycl/sycl.hpp>
#include <atomic>

namespace H4I::MKLShim
{
    struct Context
    {
        sycl::queue queue;
        sycl::device device;
        sycl::context context;
        sycl::platform platform;
        
        // Reference count for manual reference counting
        std::atomic<int> ref_count;

        // Create default SYCL context and queue.
        // TODO if we have multiple GPUs in a node, how do we
        // select which one to use?
        Context(void) : ref_count(1)
        {
            // Nothing else to do.
        }
        
        void addRef() {
            ref_count.fetch_add(1);
        }
        
        int release() {
            return ref_count.fetch_sub(1) - 1;
        }
        
        int getRefCount() const {
            return ref_count.load();
        }
    };

} // namespace

