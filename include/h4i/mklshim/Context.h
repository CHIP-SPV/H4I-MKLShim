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

// Handle indices in the handles array
enum HandleIndex {
  BACKEND_NAME = 0,
  PLATFORM_DRIVER = 1,
  DEVICE = 2,
  CONTEXT = 3,
  QUEUE = 4,
  COMMAND_LIST = 5  // Optional, only for level0 with immediate command lists
};

struct Context;

// Since shim supports multiple backends hence this indicates current backend is in use
extern Backend currentBackend;

// Maintains a global table to avoid duplicate sycl queue creation for same native queue
// This helps synchronization between two libraries e.g. hipSolver and hipBlas
extern std::unordered_map<uintptr_t, Context*> context_tbl;

/**
 * Create (or retrieve) a MKLShim execution context bound to a given native backend queue.
 *
 * @param handles        Array of native backend handles. The first entry is a char* backend name
 *                       ("level0" or "opencl"), followed by PLATFORM_DRIVER, DEVICE, CONTEXT, QUEUE
 *                       and optionally COMMAND_LIST (for immediate L0).
 * @param numOfHandles   Number of elements in the handles array (must be 5 or 6).
 * @return               Raw pointer to the Context instance. Do NOT delete directly; call Destroy().
 */
Context* Create(unsigned long const* handles, int numOfHandles);

/**
 * Update an existing Context with a new queue / stream, preserving previously initialized state.
 *
 * @param ctxt           Context returned from Create() that should be associated with the new queue.
 * @param handles        Native backend handles describing the new queue (same format as Create).
 * @param numOfHandles   Number of elements in the handles array (5 or 6).
 * @return               The same raw Context pointer (for convenience).
 */
Context* Update(Context* ctxt, unsigned long const* handles, int numOfHandles);

/**
 * Decrease the reference count of a Context and delete it once no caller holds it.
 *
 * @param context   Context previously obtained from Create/Update.
 */
void Destroy(Context* context);

/**
 * Retrieve the Intel oneMKL version that the shim was built against/linked to.
 */
MKL_VERSION get_mkl_version();

/**
 * Helper to check whether the loaded oneMKL is at least 2023.0.2 (needed for some APIs).
 */
bool is_mkl_eq_higher_2023_0_2();
} // namespace

