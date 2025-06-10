// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "h4i/mklshim/types.h"
#include "h4i/mklshim/Context.h"
#include "h4i/mklshim/Stream.h"
#include "h4i/mklshim/onemklsolver.h"
#include "h4i/mklshim/onemklblas.h"

// This is a workaround to flush MKL submissions into Level-zero queue,
// using unspecified but guaranteed behavior of intel-sycl runtime.
// Once SYCL standard committee approves sycl::queue::flush() we will change the macro to use the same
// FIXED: Handle OneAPI 2024.2.2 backend mismatch issue where OneMKL returns events with wrong backend
#define __FORCE_MKL_FLUSH__(cmd)                                \
    try {                                                       \
        if (currentBackend == opencl) {                         \
            try {                                               \
                get_native<sycl::backend::opencl>(cmd);         \
            } catch (const sycl::exception& e) {                \
                /* OneMKL 2024.2.2 bug: status has wrong backend */ \
                get_native<sycl::backend::ext_oneapi_level_zero>(cmd); \
            }                                                   \
        } else {                                                \
            try {                                               \
                get_native<sycl::backend::ext_oneapi_level_zero>(cmd); \
            } catch (const sycl::exception& e) {                \
                /* Fallback to OpenCL if Level Zero fails */   \
                get_native<sycl::backend::opencl>(cmd);         \
            }                                                   \
        }                                                       \
    } catch (const sycl::exception& e) {                        \
        /* If both fail, skip the flush (best effort) */       \
    }

#define ONEMKL_TRY \
    if(ctxt == nullptr) { \
      std::cerr << "Error context is null"<<std::endl; \
      return;\
    }\
    try\
    {

#define ONEMKL_TRY_RETURN(retval) \
    if(ctxt == nullptr) { \
      std::cerr << "Error context is null"<<std::endl; \
      return retval;\
    }\
    try\
    {

#define __CATCH__(msg) \
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

#define ONEMKL_CATCH(msg) \
    status.wait_and_throw(); \
    __FORCE_MKL_FLUSH__(status); \
    __CATCH__(msg)

#define ONEMKL_CATCH_NO_FLUSH(msg) \
    return size;                   \
    __CATCH__(msg)