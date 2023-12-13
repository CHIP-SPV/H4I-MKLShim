// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/common.h"
namespace H4I::MKLShim
{

// Indicates current backend used
Backend currentBackend;
std::unordered_map<uintptr_t, Context*> context_tbl;

Context* Update(Context* ctxt, unsigned long const* backendHandles, int numOfHandles, const char* backendName) {
    // Obtain the handles to the LZ constructs.
    assert(nHandles == 4);
    std::string strBackend(backendName);
    if (strBackend == "opencl") {
        currentBackend = opencl;
        cl_platform_id hPlatformId = (cl_platform_id)backendHandles[0];
        cl_device_id hDeviceId = (cl_device_id)backendHandles[1];
        cl_context hContext = (cl_context)backendHandles[2];
        cl_command_queue hQueue = (cl_command_queue)backendHandles[3];

        // Build SYCL platform/device/queue from the opencl handles.
        ctxt->platform = sycl::opencl::make_platform((pi_native_handle)hPlatformId);
        ctxt->device = sycl::opencl::make_device((pi_native_handle)hDeviceId);
        ctxt->context = sycl::opencl::make_context((pi_native_handle)hContext);
        ctxt->queue = sycl::opencl::make_queue(ctxt->context, (pi_native_handle)hQueue);
    } else {
        currentBackend = level0;
        auto hDriver  = (ze_driver_handle_t)backendHandles[0];
        auto hDevice  = (ze_device_handle_t)backendHandles[1];
        auto hContext = (ze_context_handle_t)backendHandles[2];
        auto hQueue   = (ze_command_queue_handle_t)backendHandles[3];

        // Build SYCL platform/device/queue from the LZ handles.
        ctxt->platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
        ctxt->device = sycl::ext::oneapi::level_zero::make_device(ctxt->platform, (pi_native_handle)hDevice);

        // FIX ME: only 1 device is returned from CHIP-SPV's lzHandles
        std::vector<sycl::device> sycl_devices(1);
        sycl_devices[0] = ctxt->device;
        ctxt->context = sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle)hContext, 1);
        ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hQueue, 1);
    }
    // add context to the table
    context_tbl[backendHandles[3]] = ctxt;
    return ctxt;
}

Context*
Create(unsigned long const* nativeHandles, int numOfHandles, const char* backendName)
{
    Context *ctxt;
    if (numOfHandles == 4 && (context_tbl.find(nativeHandles[3]) != context_tbl.end())) {
        ctxt = context_tbl[nativeHandles[3]];
    } else {
        ctxt = Update(new Context(), nativeHandles, numOfHandles, backendName);
    }

    // Get MKL version
    updateMKLVersion();
    return ctxt;
}

void
Destroy(Context* ctxt)
{
    // Fix Me: Since not all resources are owned by Sycl,
    // do we need to deleted Sycl pointers?
    //delete ctxt;
}

} // namespace

