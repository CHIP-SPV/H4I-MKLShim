// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once
#include <unordered_map>
namespace H4I::MKLShim
{

enum Backend{
  level0, // default
  opencl
};

struct Context;

// Since shim supports multiple backends hence this indicates current backend is in use
extern Backend currentBackend;

// Maintains a global table to avoid duplicate sycl queue creation for same native queue
// This helps synchronization between two libraries e.g. hipSolver and hipBlas
extern std::unordered_map<uintptr_t, Context*> context_tbl;

Context* Create(unsigned long const* lzHandles, int numOfHandles, const char* backendName);
Context* Update(Context* ctxt, unsigned long const* backendHandles, int numOfHandles, const char* backendName);
void Destroy(Context* context);

MKL_VERSION get_mkl_version();
bool is_mkl_eq_higher_2023_0_2();
} // namespace

