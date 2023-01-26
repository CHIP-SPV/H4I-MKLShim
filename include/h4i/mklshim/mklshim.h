// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "h4i/mklshim/types.h"
#include "h4i/mklshim/Context.h"
#include "h4i/mklshim/Stream.h"
#include "h4i/mklshim/sgemm.h"

// This is a workaround to flush MKL submissions into Level-zero queue,
// using unspecified but guaranteed behavior of intel-sycl runtime.
// Once SYCL standard committee approves sycl::queue::flush() we will change the macro to use the same
#define __FORCE_MKL_FLUSH__(cmd) \
    get_native<sycl::backend::ext_oneapi_level_zero>(cmd)

#define ONEMKL_TRY \
    if(ctxt == nullptr) { \
      std::cerr << "Error context is null"<<std::endl; \
      return;\
    }\
    try\
    {

#define ONEMKL_CATCH(msg) \
      __FORCE_MKL_FLUSH__(status);\
    }\
    catch(sycl::exception const& e)\
    {\
      std::cerr << msg<<" SYCL exception: " << e.what() << std::endl;\
      throw;\
    }\
    catch(std::exception const& e)\
    {\
      std::cerr << msg<<" exception: " << e.what() << std::endl;\
      throw;\
    }