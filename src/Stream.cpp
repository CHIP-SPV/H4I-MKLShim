// Copyright 2021-2022 UT-Battelle
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
        // Obtain the native handles.
        auto hPlatform = (ze_driver_handle_t)nativeHandles[0];
        auto hDevice = (ze_device_handle_t)nativeHandles[1];
        auto hContext = (ze_context_handle_t)nativeHandles[2];
        auto hQueue = (ze_command_queue_handle_t)nativeHandles[3];

        // Build SYCL objects from native handles.
        constexpr const auto Backend = sycl::backend::ext_oneapi_level_zero;
        sycl::backend_input_t<Backend, sycl::platform> mpinput {hPlatform};
        ctxt->platform = sycl::make_platform<Backend>(mpinput);

        sycl::backend_input_t<Backend, sycl::device> mdinput {hDevice};
        ctxt->device = sycl::make_device<Backend>(mdinput);

        std::vector<sycl::device> devs;
        devs.push_back(ctxt->device);
        sycl::backend_input_t<Backend, sycl::context> mcinput {hContext, devs};
        ctxt->context = sycl::make_context<Backend>(mcinput);

        auto asyncExceptionHandler = [](sycl::exception_list exceptions) {
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

        sycl::backend_input_t<Backend, sycl::queue> mqinput(hQueue, ctxt->device);
        ctxt->queue = sycl::make_queue<Backend>(mqinput,
                                ctxt->context,
                                asyncExceptionHandler);
    }
}

} // namespace

