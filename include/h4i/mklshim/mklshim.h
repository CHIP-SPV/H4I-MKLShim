// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include "h4i/mklshim/types.h"
#include "h4i/mklshim/Context.h"
#include "h4i/mklshim/Stream.h"
#include "h4i/mklshim/onemklsolver.h"
#include "h4i/mklshim/onemklblas.h"


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
    __CATCH__(msg)

#define ONEMKL_CATCH_NO_FLUSH(msg) \
    return size;                   \
    __CATCH__(msg)