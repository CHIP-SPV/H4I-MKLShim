// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

ContextImpl::KnownBackendMapType ContextImpl::knownBackends;

std::shared_ptr<ContextImpl::SyclBackend>
ContextImpl::FindOrCreateBackend(const NativeHandleArray& nativeHandles, Backend backend)
{
    std::shared_ptr<SyclBackend> ret;

    auto& backendMap = knownBackends[backend];

    // Check if we know about this backend context already.
    auto iter = backendMap.find(nativeHandles.key());
    if(iter != backendMap.end())
    {
        // We know about this context already.
        // Use the existing Backend rather than creating a new one.
        ret = iter->second;
    }
    else
    {
        // We don't know about this Backend context yet.
        // Create one and save it in case we are asked to recreate it later.
        ret = MakeBackend(nativeHandles, backend);
        backendMap[nativeHandles.key()] = ret;
    }
    assert(backendMap.find(nativeHandles.key()) != backendMap.end());

    return ret;
}

Context*
ContextImpl::Create(const NativeHandleArray& nativeHandles, Backend backend)
{
    std::shared_ptr<SyclBackend> sbe = FindOrCreateBackend(nativeHandles, backend);
    return new ContextImpl(sbe);
}

Context*
Context::Create(const NativeHandleArray& nativeHandles, Backend backend)
{
    return ContextImpl::Create(nativeHandles, backend);
}

ContextImpl::~ContextImpl(void)
{
    // Remove our backend from the known backend map.
    // We had better know about this backend context already.
    auto& backendMap = knownBackends[bedata->backend];
    auto iter = backendMap.find(bedata->mapKey);
    assert(iter != backendMap.end());
    backendMap.erase(iter);

    // Release the backend.
    bedata.reset();
}


void
ContextImpl::SetStream(const NativeHandleArray& nativeHandles)
{
    // Save access to our current backend before we (potentially) replace it.
    auto origMapKey = bedata->mapKey;

    // Find or create a backend for the given set of native handles.
    bedata = FindOrCreateBackend(nativeHandles, bedata->backend);

    // If the new backend is not the same as our original one, 
    // release the original backend.
    if(bedata->mapKey != origMapKey)
    {
        auto origBackendIter = knownBackends[bedata->backend].find(origMapKey);
        assert(origBackendIter != knownBackends[bedata->backend].end());
        knownBackends[bedata->backend].erase(origBackendIter);
    }
}

} // namespace

