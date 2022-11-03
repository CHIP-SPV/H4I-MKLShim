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
        // Obtain the native handles.
        auto hPlatform = (ze_driver_handle_t)nativeHandles[0];
        auto hDevice = (ze_device_handle_t)nativeHandles[1];
        auto hContext = (ze_context_handle_t)nativeHandles[2];
        auto hQueue = (ze_command_queue_handle_t)nativeHandles[3];

        // Build SYCL objects from native handles.
        ctxt->platform = sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(hPlatform);
        ctxt->device = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(hDevice);
        std::vector<sycl::device> devs;
        devs.push_back(ctxt->device);
        ctxt->context = sycl::level_zero::make<sycl::context>(devs, hContext);

#if READY
        auto asyncExceptionhandler = [](sycl::exception_list exceptions) {
            // Report all asynchronous exceptions that occurred.
            for(std::exception_ptr const& e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                } 
                catch(std::exception& e)
                {
                    std::cerr << "Async exception: " << e.what() << std::endl;
                }
            }

            // Rethrow the first asynchronous exception.
            for(std::exception_ptr const& e : exceptions)
            {
                std::rethrow_exception(e);
            }
        };
#endif // READY

        ctxt->queue = sycl::level_zero::make<sycl::queue>(ctxt->context, hQueue);
                                // , asyncExceptionhandler);
    }
}

} // namespace

