#pragma once

#include <unordered_map>

namespace H4I::MKLShim
{

inline
oneapi::mkl::transpose
ToMKLOp(Operation op)
{
    std::unordered_map<Operation, oneapi::mkl::transpose> map = 
    {
        {Operation::N, oneapi::mkl::transpose::N},
        {Operation::T, oneapi::mkl::transpose::T},
        {Operation::C, oneapi::mkl::transpose::C}
    };

    return map[op];
}

}

