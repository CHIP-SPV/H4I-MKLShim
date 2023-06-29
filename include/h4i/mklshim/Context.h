// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once
#include <string>
#include <array>

namespace H4I::MKLShim
{

constexpr const int nHandles = 4;
using NativeHandleArray = std::array<uintptr_t, nHandles>;

enum Backend{
  level0, // default
  opencl
};


// Convert a backend name to the Enum.
// May throw an exception if given an unrecognized backend name.
Backend ToBackend(const std::string&);

Backend GetCurrentBackend(void);

// Provide an interface for creating and manipulating
// Contexts that avoids the caller from seeing the backend-specific
// implementation.
struct Context;

Context* Create(const NativeHandleArray& handles, Backend backend);
void Destroy(Context* context);
void SetStream(Context* context, const NativeHandleArray& handles);

} // namespace

