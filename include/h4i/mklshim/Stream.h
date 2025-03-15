// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <array>
#include "h4i/mklshim/Context.h"

namespace H4I::MKLShim
{

constexpr const int nHandles = 4;
void SetStream(Context* ctxt, unsigned long const* backendHandles, int numOfHandles, const char* backendName);
} // namespace

