// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"

namespace H4I::MKLShim
{

void
SGEMM(Context* ctxt,
        Operation transa,
        Operation transb,
        int m,
        int n,
        int k,
        const float* alpha,
        const float* A,
        int ldA,
        const float* B,
        int ldB,
        const float* beta,
        float* C,
        int ldC)
{
    if(ctxt != nullptr)
    {
        // Do the SGEMM via MKL.
        try
        {
            oneapi::mkl::blas::gemm(ctxt->queue,
                                    ToMKLOp(transa),
                                    ToMKLOp(transb),
                                    m,
                                    n,
                                    k,
                                    *alpha,
                                    A,
                                    ldA,
                                    B,
                                    ldB,
                                    *beta,
                                    C,
                                    ldC);
        }
        catch(sycl::exception const& e)
        {
            std::cerr << "SGEMM SYCL exception: " << e.what() << std::endl;
            throw;
        }
        catch(std::exception const& e)
        {
            std::cerr << "SGEMM exception: " << e.what() << std::endl;
            throw;
        }

        // Catch any asynchronous exceptions before continuing.
        ctxt->queue.wait_and_throw();
    }
}

} // namespace H4I::MKLShim

