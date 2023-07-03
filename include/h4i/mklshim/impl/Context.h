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

        sycl::platform platform;
        sycl::device device;
        sycl::context context;
        sycl::queue queue;

        SyclBackend(Context::Backend _backend,
                    const sycl::platform& _platform,
                    const sycl::device& _device,
                    const sycl::context& _context,
                    const sycl::queue& _queue)
          : backend(_backend),
            platform(_platform),
            device(_device),
            context(_context),
            queue(_queue)
        {
            // Nothing else to do.
        }

        SyclBackend(void) = delete;

        Backend GetBackend(void) const  { return backend; }
    };

    struct LevelZeroSyclBackend : public SyclBackend
    {
    private:
        static SyclBackend MakeBaseBackend(const NativeHandleArray& nativeHandles)
        {
            auto hDriver = reinterpret_cast<pi_native_handle>(nativeHandles[0]);
            auto hDevice = reinterpret_cast<pi_native_handle>(nativeHandles[1]);
            auto hContext = reinterpret_cast<pi_native_handle>(nativeHandles[2]);
            auto hQueue = reinterpret_cast<pi_native_handle>(nativeHandles[3]);

            auto platform = sycl::ext::oneapi::level_zero::make_platform(hDriver);
            auto device = sycl::ext::oneapi::level_zero::make_device(platform, hDevice);

            // TODO how to handle more than one GPU?
            // chipStar only returns one device.
            std::vector<sycl::device> sycl_devices(1, device);
            auto context = sycl::ext::oneapi::level_zero::make_context(sycl_devices, hContext, 1);
            auto queue = sycl::ext::oneapi::level_zero::make_queue(context, device, hQueue);

            return SyclBackend(Context::Backend::level0, platform, device, context, queue);
        }

    public:
        LevelZeroSyclBackend(const NativeHandleArray& nativeHandles)
          : SyclBackend(MakeBaseBackend(nativeHandles))
        {
            // Nothing else to do.
        }
    };

    struct OpenCLSyclBackend : public SyclBackend
    {
    private:
        static SyclBackend MakeBaseBackend(const NativeHandleArray& nativeHandles)
        {
            auto hPlatform = reinterpret_cast<pi_native_handle>(nativeHandles[0]);
            auto hDevice = reinterpret_cast<pi_native_handle>(nativeHandles[1]);
            auto hContext = reinterpret_cast<pi_native_handle>(nativeHandles[2]);
            auto hQueue = reinterpret_cast<pi_native_handle>(nativeHandles[3]);

            auto platform = sycl::opencl::make_platform(hPlatform);
            auto device = sycl::opencl::make_device(hDevice);
            auto context = sycl::opencl::make_context(hContext);
            auto queue = sycl::opencl::make_queue(context, hQueue);

            return SyclBackend(Context::Backend::opencl, platform, device, context, queue);
        }

    public:
        OpenCLSyclBackend(const NativeHandleArray& nativeHandles)
          : SyclBackend(MakeBaseBackend(nativeHandles))
        {
            // Nothing else to do.
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

