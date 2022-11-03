#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"

namespace H4I::MKLShim
{

Context*
Create(void)
{
    return new Context();
}

void
Destroy(Context* ctxt)
{
    delete ctxt;
}

} // namespace

