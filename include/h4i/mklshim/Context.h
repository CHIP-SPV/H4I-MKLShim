#pragma once

namespace H4I::MKLShim
{

struct Context;

Context* Create(void);
void Destroy(Context* context);

} // namespace

