// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

namespace H4I::MKLShim
{

struct Context;

Context* Create(void);
void Destroy(Context* context);

} // namespace

