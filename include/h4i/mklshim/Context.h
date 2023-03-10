// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

namespace H4I::MKLShim
{

struct Context;

Context* Create(unsigned long const* lzHandles, int numOfHandles);
void Destroy(Context* context);

} // namespace

