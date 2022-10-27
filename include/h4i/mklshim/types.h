#pragma once

namespace H4I::MKLShim
{

enum Operation
{
    N = 0,
    T = 1,
    C = 2
};

enum Datatype
{
    Real8I = 0,
    Real32I = 1,
    Real16F = 2,
    Real32F = 3
};

struct Context;

} // namespace
