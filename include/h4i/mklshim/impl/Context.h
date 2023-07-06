// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <memory>
#include <array>
#include <unordered_map>
#include <sycl/queue.hpp>
#include <sycl/device.hpp>
#include <sycl/context.hpp>
#include <sycl/platform.hpp>
#include "h4i/mklshim/Context.h"


namespace H4I::MKLShim
{

class ContextImpl : public Context
{
public:
    // TODO if we have multiple GPUs in a node, how do we
    // select which one to use?
    struct SyclBackend
    {
        Context::Backend backend;
        Context::NativeHandleType mapKey;

        sycl::platform platform;
        sycl::device device;
        sycl::context context;
        sycl::queue queue;

        SyclBackend(void) = delete;
        SyclBackend(Context::Backend _backend, Context::NativeHandleType _mapKey)
          : backend(_backend),
            mapKey(_mapKey)
        {
            std::cerr << "In SyclBackend::SyclBackend" << std::endl;
            // Nothing else to do.
        }

        ~SyclBackend(void)
        {
            std::cerr << "In SyclBackend::~SyclBackend" << std::endl;
        }


        Backend GetBackend(void) const  { return backend; }
    };

    struct LevelZeroSyclBackend : public SyclBackend
    {
        LevelZeroSyclBackend(const NativeHandleArray& nativeHandles)
          : SyclBackend(Context::Backend::level0, nativeHandles.key())
        {
            auto hDriver = reinterpret_cast<ze_driver_handle_t>(nativeHandles[0]);
            auto hDevice = reinterpret_cast<ze_device_handle_t>(nativeHandles[1]);
            auto hContext = reinterpret_cast<ze_context_handle_t>(nativeHandles[2]);
            auto hQueue = reinterpret_cast<ze_command_queue_handle_t>(nativeHandles[3]);

            // Create SYCL objects around native Level Zero backend handles.
            // Note: we explicitly retain ownership of the context and device handles, 
            // rather than using the default behavior of transferring ownership to the SYCL runtime.
            // We do this because chipStar expects to retain ownership.  (TODO - verify this.)
            platform = sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(hDriver);
            device = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(hDevice);

            // TODO how to handle more than one GPU?
            // chipStar only returns one device with hipGetNativeDeviceHandles().
            std::vector<sycl::device> sycl_devices(1, device);
            context = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
                { hContext, sycl_devices, sycl::ext::oneapi::level_zero::ownership::keep } );

            queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(
                { hQueue, device, sycl::ext::oneapi::level_zero::ownership::keep },
                context);
        }
    };

    struct OpenCLSyclBackend : public SyclBackend
    {
        OpenCLSyclBackend(const NativeHandleArray& nativeHandles)
          : SyclBackend(Context::Backend::opencl, nativeHandles.key())
        {
            auto hPlatform = reinterpret_cast<pi_native_handle>(nativeHandles[0]);
            auto hDevice = reinterpret_cast<pi_native_handle>(nativeHandles[1]);
            auto hContext = reinterpret_cast<pi_native_handle>(nativeHandles[2]);
            auto hQueue = reinterpret_cast<pi_native_handle>(nativeHandles[3]);

            platform = sycl::opencl::make_platform(hPlatform);
            device = sycl::opencl::make_device(hDevice);
            context = sycl::opencl::make_context(hContext);
            queue = sycl::opencl::make_queue(context, hQueue);
        }
    };

    using KnownBackendMapType = std::array<std::unordered_map<NativeHandleType, std::shared_ptr<SyclBackend>>, Backend::last+1>;
    static KnownBackendMapType knownBackends;

    // Current SYCL backend.
    std::shared_ptr<SyclBackend> bedata;

    static std::shared_ptr<SyclBackend> MakeBackend(const NativeHandleArray& handles, Backend backend)
    {
        std::shared_ptr<SyclBackend> ret;
        switch(backend)
        {
            case level0:
                ret = std::make_shared<LevelZeroSyclBackend>(handles);
                break;
            case opencl:
                ret = std::make_shared<OpenCLSyclBackend>(handles);
                break;
            default:
                throw std::runtime_error("Unexpected SYCL backend type seen");
                break;
        }
        return ret;
    }

public:
    // Create a Context associated with the given backend handles.
    static Context* Create(const NativeHandleArray& handles, Backend backend);

#if READY
    ContextImpl(const NativeHandleArray& handles, Backend backend)
      : bedata(MakeBackend(handles, backend))
    { }
#endif // READY
    ContextImpl(std::shared_ptr<SyclBackend> _bedata)
      : bedata(_bedata)
    { }
    virtual ~ContextImpl(void);

    // Determine which backend we're associated with.
    Backend GetCurrentBackend(void) override
    {
        return bedata->GetBackend();
    }

    // Associate ourself with a different set of backend handles.
    void SetStream(const NativeHandleArray& handles) override;
};

} // namespace

