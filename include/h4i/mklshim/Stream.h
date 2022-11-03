#pragma once

#include <array>
#include "h4i/mklshim/Context.h"

namespace H4I::MKLShim
{

constexpr const int nHandles = 4;
void SetStream(Context* context, const std::array<uintptr_t, nHandles>& handles);

} // namespace

