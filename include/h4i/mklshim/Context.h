// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once
#include <string>
#include <array>
#include <unordered_map>

namespace H4I::MKLShim
{

// Context encapsulating native objects for current backend.
class Context
{
public:
    // Supported backends.
    enum Backend
    {
        level0 = 0, // default
        opencl,
        last = opencl,
    };

    // Nice name for collection of native handles used as context for a given backend.
    static constexpr int nNativeHandles = 4;
    using NativeHandleType = uintptr_t;
    struct NativeHandleArray : public std::array<NativeHandleType, nNativeHandles>
    {
        NativeHandleType key(void) const { return (*this)[3]; }
    };

    // Convert a backend name to the Enum.
    // May throw an exception if given an unrecognized backend name.
    static Backend ToBackend(const std::string& name)
    {
        static const std::unordered_map<std::string, Backend> map{
            {"default", Backend::level0},
            {"level0", Backend::level0},
            {"opencl", Backend::opencl}
        };

        return map.at(name);
    }

    // Create a Context associated with the given backend handles.
    static Context* Create(const NativeHandleArray& handles, Backend backend);

    Context(void) = default;
    virtual ~Context(void) { }

    // Determine which backend we're associated with.
    virtual Backend GetCurrentBackend(void) = 0;

    // Associate ourself with a different set of backend handles.
    virtual void SetStream(const NativeHandleArray& handles) = 0;
};

} // namespace

