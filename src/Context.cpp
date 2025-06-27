// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/common.h"
#include <mutex>

// Check if UR API is available
#if INTEL_MKL_VERSION >= 20250000
  #include <sycl/backend.hpp>
  // Check if ur_native_handle_t is defined
  #ifdef UR_API_H_INCLUDED
    #define HAS_UR_API 1
  #else
    #define HAS_UR_API 0
  #endif
#elif INTEL_MKL_VERSION >= 20230000
  #include <sycl/ext/oneapi/backend/level_zero.hpp>
  #define HAS_UR_API 0
#else
  #include <CL/sycl/backend/level_zero.hpp>
  #define HAS_UR_API 0
#endif

namespace H4I::MKLShim
{

// Indicates current backend used
Backend currentBackend;
std::unordered_map<uintptr_t, Context*> context_tbl;
// Mutex to protect access to context_tbl
std::mutex context_tbl_mutex;

Context* Update(Context* ctxt, unsigned long const* handles, int numOfHandles) {
    // Obtain the handles to the LZ constructs.
    const char* backendName = (const char*)handles[BACKEND_NAME];
    
    std::string strBackend(backendName);
    
    if (strBackend == "opencl") {
        currentBackend = opencl;
        cl_platform_id hPlatformId = (cl_platform_id)handles[PLATFORM_DRIVER];
        cl_device_id hDeviceId = (cl_device_id)handles[DEVICE];
        cl_context hContext = (cl_context)handles[CONTEXT];
        cl_command_queue hQueue = (cl_command_queue)handles[QUEUE];

        // Build SYCL platform/device/queue from the opencl handles.
#if HAS_UR_API
        // MKL 2025 uses UR API
        ctxt->platform = sycl::detail::make_platform((ur_native_handle_t)hPlatformId, sycl::backend::opencl);
        ctxt->device = sycl::detail::make_device((ur_native_handle_t)hDeviceId, sycl::backend::opencl);
        ctxt->context = sycl::detail::make_context((ur_native_handle_t)hContext, {}, sycl::backend::opencl, false);
        ctxt->queue = sycl::detail::make_queue((ur_native_handle_t)hQueue, false, ctxt->context, &ctxt->device, false, {}, {}, sycl::backend::opencl);
#else
        // MKL 2024 and earlier use PI API
        ctxt->platform = sycl::opencl::make_platform((pi_native_handle)hPlatformId);
        ctxt->device = sycl::opencl::make_device((pi_native_handle)hDeviceId);
        ctxt->context = sycl::opencl::make_context((pi_native_handle)hContext);
        ctxt->queue = sycl::opencl::make_queue(ctxt->context, (pi_native_handle)hQueue);
#endif
    } else if(strBackend == "level0") {
        currentBackend = level0;
        auto hDriver  = (ze_driver_handle_t)handles[PLATFORM_DRIVER];
        auto hDevice  = (ze_device_handle_t)handles[DEVICE];
        auto hContext = (ze_context_handle_t)handles[CONTEXT];
        auto hQueue   = (ze_command_queue_handle_t)handles[QUEUE];

        ze_command_list_handle_t hCommandList = 0;
        if (numOfHandles == 6)
            hCommandList = (ze_command_list_handle_t)handles[COMMAND_LIST];

        bool isImmCmdList = (hCommandList != nullptr);

        // Build SYCL platform/device/queue from the LZ handles.
#if HAS_UR_API
        // MKL 2025 uses UR API
        ctxt->platform = sycl::detail::make_platform((ur_native_handle_t)hDriver, sycl::backend::ext_oneapi_level_zero);
        ctxt->device = sycl::detail::make_device((ur_native_handle_t)hDevice, sycl::backend::ext_oneapi_level_zero);

        std::vector<sycl::device> sycl_devices;
        sycl_devices = ctxt->platform.get_devices();
	// Use the specific device from CHIP-SPV, not all system devices
        ctxt->context = sycl::detail::make_context((ur_native_handle_t)hContext, {}, sycl::backend::ext_oneapi_level_zero, false,
						   sycl_devices);
        
        if (isImmCmdList) {
            ctxt->queue = sycl::detail::make_queue((ur_native_handle_t)hCommandList, true, ctxt->context, &ctxt->device, true, 
                                                  {sycl::property::queue::in_order()}, {}, sycl::backend::ext_oneapi_level_zero);
        } else {
            ctxt->queue = sycl::detail::make_queue((ur_native_handle_t)hQueue, false, ctxt->context, &ctxt->device, true, 
                                                  {sycl::property::queue::in_order()}, {}, sycl::backend::ext_oneapi_level_zero);
        }
#elif __INTEL_LLVM_COMPILER >= 20240000
        // MKL 2024 uses PI API with updated make_queue signature
        ctxt->platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
        ctxt->device = sycl::ext::oneapi::level_zero::make_device(ctxt->platform, (pi_native_handle)hDevice);

        std::vector<sycl::device> sycl_devices;
        sycl_devices = ctxt->platform.get_devices();
        ctxt->context = sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle)hContext, 1);
        
        if (isImmCmdList) {
            ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hCommandList, true, 1, sycl::property::queue::in_order());
        } else {
            ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hQueue, false, 1, sycl::property::queue::in_order());
        }
#else
        // MKL 2023 and earlier use PI API with older make_queue signature
        ctxt->platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
        ctxt->device = sycl::ext::oneapi::level_zero::make_device(ctxt->platform, (pi_native_handle)hDevice);

        std::vector<sycl::device> sycl_devices;
        sycl_devices = ctxt->platform.get_devices();
        ctxt->context = sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle)hContext, 1);
        
        ctxt->queue = sycl::ext::oneapi::level_zero::make_queue(ctxt->context, ctxt->device, (pi_native_handle)hQueue, 1);
#endif
    } else {
        std::cerr << "Unsupported backend: " << backendName << std::endl;
        std::abort();
    }
    // add context to the table
    context_tbl[handles[QUEUE]] = ctxt;
    return ctxt;
}

Context*
Create(unsigned long const* handles, int numOfHandles)
{
    // Error check
    // Number of handles must be at least 5 (backendName + 4 handles)
    if (numOfHandles < 5 || numOfHandles > 6) {
        std::cerr << "Error: Invalid handles\n";
        return nullptr;
    }
    
    Context *ctxt;
    
    // Lock the mutex to protect access to context_tbl
    std::lock_guard<std::mutex> lock(context_tbl_mutex);
    
    // check if context already exists for this queue
    if (context_tbl.find(handles[QUEUE]) != context_tbl.end()) {
        ctxt = context_tbl[handles[QUEUE]];
        // since context already exists, increment reference count
        ctxt->addRef();
    } else {
        // create new context (starts with ref count of 1)
        ctxt = Update(new Context(), handles, numOfHandles);
    }

    // Get MKL version
    updateMKLVersion();
    return ctxt;
}

void
Destroy(Context* ctxt)
{
    if (!ctxt) return;
   
    int newRefCount = ctxt->release();
    
    if (newRefCount == 0) {
        // Reference count reached zero, actually delete the context
        // Lock the mutex to protect access to context_tbl
        std::lock_guard<std::mutex> lock(context_tbl_mutex);
        
        // Remove context from the table
        context_tbl.erase(std::find_if(context_tbl.begin(), context_tbl.end(),
                                       [ctxt](const auto& pair) {
                                           return pair.second == ctxt;
                                       }));
        delete ctxt;
    }
    // If newRefCount > 0, other references still exist, don't delete
}

} // namespace
