#pragma once

namespace H4I::MKLShim
{

struct Context;

Context* Create(void);
void Destroy(Context* context);
void SetStream(Context* context, uintptr_t* handles, int nHandles);

} // namespace

