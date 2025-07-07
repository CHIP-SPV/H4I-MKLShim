// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <array>
#include "h4i/mklshim/Context.h"

namespace H4I::MKLShim
{

/**
 * Associate an existing Context with a different backend stream / queue.
 * Internally this calls Update() but keeps the same Context pointer so external
 * callers do not need to track a new handle.
 *
 * @param ctxt           Context whose active SYCL queue should be replaced.
 * @param handles        Array of native backend handles describing the new stream.
 * @param numOfHandles   Number of elements in the handles array (5 or 6).
 */
void SetStream(Context* ctxt, unsigned long const* handles, int numOfHandles);
} // namespace

