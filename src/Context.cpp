// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

Context* Update(Context* ctxt, unsigned long const* lzHandles, int numOfHandles) {
    // Obtain the handles to the LZ constructs.
    assert(nHandles == 4);
    auto hDriver  = (ze_driver_handle_t)lzHandles[0];
    auto hDevice  = (ze_device_handle_t)lzHandles[1];
    auto hContext = (ze_context_handle_t)lzHandles[2];
    auto hQueue   = (ze_command_queue_handle_t)lzHandles[3];

    // Build SYCL platform/device/queue from the LZ handles.
    ctxt->platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
    ctxt->device = sycl::ext::oneapi::level_zero::make_device(ctxt->platform, (pi_native_handle)hDevice);

    // FIX ME: only 1 device is returned from CHIP-SPV's lzHandles
    std::vector<sycl::device> sycl_devices(1);
    sycl_devices[0] = ctxt->device;
    ctxt->context = sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle)hContext, 1);
    ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hQueue, 1);
    return ctxt;
}

Context*
Create(unsigned long const* lzHandles, int numOfHandles)
{
    auto ctxt = new Context();
    return Update(ctxt, lzHandles, numOfHandles);
}

void
Destroy(Context* ctxt)
{
    // Fix Me: Since not all resources are owned by Sycl,
    // do we need to deleted Sycl pointers?
    //delete ctxt;
}

} // namespace

