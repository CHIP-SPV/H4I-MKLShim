// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

ContextImpl::KnownBackendMapType ContextImpl::knownBackends;

Context*
ContextImpl::Create(const NativeHandleArray& nativeHandles, Backend backend)
{
    std::shared_ptr<SyclBackend> sbe;

    auto& backendMap = knownBackends[backend];

    // Check if we know about this backend context already.
    auto iter = backendMap.find(nativeHandles.key());
    if(iter != backendMap.end())
    {
        // We know about this context already.
        // Just return it rather than creating a new one.
        sbe = iter->second;
    }
    else
    {
        // We don't know about this context yet.
        // Create one and save it in case we are asked to recreate it later.
        sbe = MakeBackend(nativeHandles, backend);
        backendMap[nativeHandles.key()] = sbe;
    }
    assert(backendMap.find(nativeHandles.key()) != backendMap.end());

    return new ContextImpl(sbe);
}

Context*
Context::Create(const NativeHandleArray& nativeHandles, Backend backend)
{
    return ContextImpl::Create(nativeHandles, backend);
}

ContextImpl::~ContextImpl(void)
{
    // Fix Me: Since not all resources are owned by Sycl,
    // do we need to deleted Sycl pointers?
    //delete ctxt;
}


void
ContextImpl::SetStream(const NativeHandleArray& nativeHandles)
{
    auto myBackend = bedata->GetBackend();
    auto& myBackendMap = knownBackends[myBackend];

    auto iter = myBackendMap.find(nativeHandles.key());
    if( iter != myBackendMap.end())
    {
        // We've seen this set of handles before.
        // Release our existing backend and associate with the one we found.
        bedata = iter->second;
    }
    else
    {
        // We haven't seen this set of handles before.
        bedata = MakeBackend(nativeHandles, myBackend);
        myBackendMap[nativeHandles.key()] = bedata;
    }
}

} // namespace

