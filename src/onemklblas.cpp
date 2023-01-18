// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklblas.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"

namespace H4I::MKLShim
{
  void onemklDamax(Context* ctxt, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    if(ctxt == nullptr) {
      std::cerr << "Error context is null"<<std::endl;
      return;
    }
    try
    {
      auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n, x, incx, result);
      __FORCE_MKL_FLUSH__(status);
    }
    catch(sycl::exception const& e)
    {
      std::cerr << "MAX SYCL exception: " << e.what() << std::endl;
      throw;
    }
    catch(std::exception const& e)
    {
      std::cerr << "MAX exception: " << e.what() << std::endl;
      throw;
    }
  }
  void onemklSamax(Context* ctxt, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    if(ctxt == nullptr) {
      std::cerr << "Error context is null"<<std::endl;
      return;
    }
    try
    {
      auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n, x, incx, result);
      __FORCE_MKL_FLUSH__(status);
    }
    catch(sycl::exception const& e)
    {
      std::cerr << "MAX SYCL exception: " << e.what() << std::endl;
      throw;
    }
    catch(std::exception const& e)
    {
      std::cerr << "MAX exception: " << e.what() << std::endl;
      throw;
    }
  }
  void onemklZamax(Context* ctxt, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    if(ctxt == nullptr) {
      std::cerr << "Error context is null"<<std::endl;
      return;
    }
    try
    {
      auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n,
                              reinterpret_cast<const std::complex<double> *>(x), incx, result);
      __FORCE_MKL_FLUSH__(status);
    }
    catch(sycl::exception const& e)
    {
      std::cerr << "MAX SYCL exception: " << e.what() << std::endl;
      throw;
    }
    catch(std::exception const& e)
    {
      std::cerr << "MAX exception: " << e.what() << std::endl;
      throw;
    }
  }
  void onemklCamax(Context* ctxt, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    if(ctxt == nullptr) {
      std::cerr << "Error context is null"<<std::endl;
      return;
    }
    try
    {
      auto status = oneapi::mkl::blas::column_major::iamax(ctxt->queue, n,
                              reinterpret_cast<const std::complex<float> *>(x), incx, result);
      __FORCE_MKL_FLUSH__(status);
    }
    catch(sycl::exception const& e)
    {
      std::cerr << "MAX SYCL exception: " << e.what() << std::endl;
      throw;
    }
    catch(std::exception const& e)
    {
      std::cerr << "MAX exception: " << e.what() << std::endl;
      throw;
    }
  }
}// end of namespacecd