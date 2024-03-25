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
    std::string strBackend(backendName);
    int idxOffset = numOfHandles == 5 ? 1 : 0;
    if (strBackend == "opencl") {
        currentBackend = opencl;
        cl_platform_id hPlatformId = (cl_platform_id)backendHandles[idxOffset + 0];
        cl_device_id hDeviceId = (cl_device_id)backendHandles[idxOffset + 1];
        cl_context hContext = (cl_context)backendHandles[idxOffset + 2];
        cl_command_queue hQueue = (cl_command_queue)backendHandles[idxOffset + 3];

        // Build SYCL platform/device/queue from the opencl handles.
        ctxt->platform = sycl::opencl::make_platform((pi_native_handle)hPlatformId);
        ctxt->device = sycl::opencl::make_device((pi_native_handle)hDeviceId);
        ctxt->context = sycl::opencl::make_context((pi_native_handle)hContext);
        ctxt->queue = sycl::opencl::make_queue(ctxt->context, (pi_native_handle)hQueue);
    } else if(strBackend == "level0") {
        currentBackend = level0;
        auto hDriver  = (ze_driver_handle_t)backendHandles[idxOffset + 0];
        auto hDevice  = (ze_device_handle_t)backendHandles[idxOffset + 1];
        auto hContext = (ze_context_handle_t)backendHandles[idxOffset + 2];
        auto hQueue   = (ze_command_queue_handle_t)backendHandles[idxOffset + 3];

        ze_command_list_handle_t hCommandList = 0;
        if (numOfHandles > 4)
            hCommandList = (ze_command_list_handle_t)backendHandles[idxOffset + 4];

        bool isImmCmdList = (hCommandList != nullptr);

        // Build SYCL platform/device/queue from the LZ handles.
        ctxt->platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
        ctxt->device = sycl::ext::oneapi::level_zero::make_device(ctxt->platform, (pi_native_handle)hDevice);

        // FIX ME: only 1 device is returned from CHIP-SPV's lzHandles
        std::vector<sycl::device> sycl_devices(1);
        sycl_devices[0] = ctxt->device;
        ctxt->context = sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle)hContext, 1);
        
        #if __INTEL_LLVM_COMPILER >= 20240000
            if (isImmCmdList) {
                ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hCommandList, true, 1, sycl::property::queue::in_order());
            } else {
                ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hQueue, false, 1, sycl::property::queue::in_order());
            }
        #else
            ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hQueue, 1);
        #endif    
    } else {
        std::cerr << "Unsupported backend: " << backendName << std::endl;
        std::abort();
    }
    // add context to the table
    context_tbl[backendHandles[3]] = ctxt;
    return ctxt;
}

Context*
Create(unsigned long const* nativeHandles, int numOfHandles, const char* backendName)
{
    Context *ctxt;
    int ctx_index = (numOfHandles == 5) ? 3 : 2;  // In old native handle call context handle used to be at '2' but on new it is '3'
    if (numOfHandles > 3 && (context_tbl.find(nativeHandles[ctx_index]) != context_tbl.end())) {
        ctxt = context_tbl[nativeHandles[ctx_index]];
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

