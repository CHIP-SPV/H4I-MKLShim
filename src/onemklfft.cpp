// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklfft.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"


namespace H4I::MKLShim
{

  // the fft descriptors are declared in include/h4i/mklshim/onemklfft.h,
  // but since the onemklfft.h header is needed from the hipfft layer, these
  // fft descriptors are defined here since intel fft descriptors are 
  // templated with values available only in the oneapi namespace and NOT
  // from the hipfft layer
  //
  // also, structs are defined for the four possible combinations of 
  // precision (Single or Double) and starting domain (Real or Complex)

  // struct for the Single precision and Real starting domain descriptor
  struct fftDescriptorSR
  {
      // descriptor for 1d transforms
      fftDescriptorSR(Context* ctxt, std::int64_t length,
                      int placement) : fft_plan(length)
      {
          // set placement
          if (placement == 1)
          {
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          
          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions,
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions,
                      std::int64_t in_strides[], std::int64_t out_strides[],
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // set these before the initial commit or get a FFT_INVALID_DESCRIPTOR exception at runtime
          fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          fft_plan.commit(ctxt->queue);
      }

      // execute R2C ... is it better to do this in the struct?
      void fftR2C(Context *ctxt, float *idata)
      {
          // ctxt->queue.wait();
          return;
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;
      
      ~fftDescriptorSR() {}
  };

  // struct for the Single precision and Complex starting domain descriptor
  struct fftDescriptorSC
  {
      // descriptor for 1d transforms
      fftDescriptorSC(Context* ctxt, std::int64_t length,
                      int placement) : fft_plan(length)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions,
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      fftDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions,
                      std::int64_t in_strides[], std::int64_t out_strides[],
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // set these before the initial commit or get a FFT_INVALID_DESCRIPTOR exception at runtime
          fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          fft_plan.commit(ctxt->queue);
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::COMPLEX> fft_plan;

      ~fftDescriptorSC() {}
  };

  // struct for the Double precision and Real starting domain descriptor
  struct fftDescriptorDR
  {
      // descriptor for 1d transforms
      fftDescriptorDR(Context* ctxt, std::int64_t length,
                      int placement) : fft_plan(length)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions,
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      fftDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions,
                      std::int64_t in_strides[], std::int64_t out_strides[],
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // set these before the initial commit or get a FFT_INVALID_DESCRIPTOR exception at runtime
          fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          fft_plan.commit(ctxt->queue);
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;

      ~fftDescriptorDR() {}
  };

  // struct for the Double precision and Complex starting domain descriptor
  struct fftDescriptorDC
  {
      // descriptor for 1d transforms
      fftDescriptorDC(Context* ctxt, std::int64_t length,
                      int placement) : fft_plan(length)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions,
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          // commit the plan
          fft_plan.commit(ctxt->queue);
      }

      fftDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions,
                      std::int64_t in_strides[], std::int64_t out_strides[],
                      int placement) : fft_plan(dimensions)
      {
          // set placement
          if (placement == 1)
          {   
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          {
             fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          }

          // set these before the initial commit or get a FFT_INVALID_DESCRIPTOR exception at runtime
          fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

          fft_plan.commit(ctxt->queue);
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::COMPLEX> fft_plan;

      ~fftDescriptorDC() {}
  };


  // simple queue check ... will be removed
  void checkFFTQueue(Context* ctxt) {
     std::cout << "in checkFFTQueue : device info =  " << 
	     ctxt->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
     return;
  }

  // simple plan check for the SR plan ... will be removed
  void checkFFTPlan(Context *ctxt, fftDescriptorSR *desc)
  {
     int64_t value = 0;
     float rvalue = 0.0;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::FORWARD_DOMAIN, &value);
     std::cout << "in checkFFTPlan : FORWARD_DOMAIN = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::PRECISION, &value);
     std::cout << "in checkFFTPlan : PRECISION = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::DIMENSION, &value);
     std::cout << "in checkFFTPlan : DIMENSION = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::LENGTHS, &value);
     std::cout << "in checkFFTPlan : LENGTHS = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, &value);
     std::cout << "in checkFFTPlan : CONJUGATE_EVEN_STORAGE = " << value << std::endl;
#if 1
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, &value);
     std::cout << "in checkFFTPlan : FWD_DISTANCE = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, &value);
     std::cout << "in checkFFTPlan : BWD_DISTANCE = " << value << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, &value);
     std::cout << "in checkFFTPlan : NUMBER_OF_TRANSFORMS = " << value << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, &rvalue);
     std::cout << "in checkFFTPlan : FORWARD_SCALE = " << rvalue << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, &rvalue);
     std::cout << "in checkFFTPlan : BACKWARD_SCALE = " << rvalue << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
     std::cout << "in checkFFTPlan : PLACEMENT = " << value << std::endl;

     // desc->fft_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 2);
     // desc->fft_plan.commit(ctxt->queue);

     // desc->fft_plan.get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, &value);
     // std::cout << "in checkFFTPlan : NUMBER_OF_TRANSFORMS = " << value << std::endl;
#endif

     return;
  }

  // simple plan check for the SR plan ... will be removed
  void checkFFTPlan(Context *ctxt, fftDescriptorSC *desc)
  {
     int64_t value = 0;
     float rvalue = 0.0;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::FORWARD_DOMAIN, &value);
     std::cout << "in checkFFTPlan : FORWARD_DOMAIN = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::PRECISION, &value);
     std::cout << "in checkFFTPlan : PRECISION = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::DIMENSION, &value);
     std::cout << "in checkFFTPlan : DIMENSION = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::LENGTHS, &value);
     std::cout << "in checkFFTPlan : LENGTHS = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, &value);
     std::cout << "in checkFFTPlan : CONJUGATE_EVEN_STORAGE = " << value << std::endl;
#if 1
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, &value);
     std::cout << "in checkFFTPlan : FWD_DISTANCE = " << value << std::endl;
     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, &value);
     std::cout << "in checkFFTPlan : BWD_DISTANCE = " << value << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, &value);
     std::cout << "in checkFFTPlan : NUMBER_OF_TRANSFORMS = " << value << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, &rvalue);
     std::cout << "in checkFFTPlan : FORWARD_SCALE = " << rvalue << std::endl;

     desc->fft_plan.get_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, &rvalue);
     std::cout << "in checkFFTPlan : BACKWARD_SCALE = " << rvalue << std::endl;

     // desc->fft_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 2);
     // desc->fft_plan.commit(ctxt->queue);

     // desc->fft_plan.get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, &value);
     // std::cout << "in checkFFTPlan : NUMBER_OF_TRANSFORMS = " << value << std::endl;
#endif

     return;
  }

  // void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata, float _Complex *odata)
  void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata)
  {
      std::cout << "in testFFTPlan : Nbytes = " << Nbytes << "   device info =  " <<
             ctxt->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
      // ctxt->queue.memcpy(odata, idata, Nbytes);
      // ctxt->queue.wait();

      oneapi::mkl::dft::compute_forward(descSR->fft_plan, idata).wait();

      // ctxt->queue.memcpy(odata, idata, Nbytes);
      // ctxt->queue.wait();

      return;
  }


  // create the 1d fft descriptors
  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt, int64_t length,
                                         int placement) {
     auto d = new fftDescriptorSR(ctxt, length, placement);
     return d;
  }

  fftDescriptorSC* createFFTDescriptorSC(Context* ctxt, int64_t length,
                                         int placement) {
     auto d = new fftDescriptorSC(ctxt, length, placement);
     return d;
  }

  fftDescriptorDR* createFFTDescriptorDR(Context* ctxt, int64_t length,
                                         int placement) {
     auto d = new fftDescriptorDR(ctxt, length, placement);
     return d;
  }

  fftDescriptorDC* createFFTDescriptorDC(Context* ctxt, int64_t length,
                                         int placement) {
     auto d = new fftDescriptorDC(ctxt, length, placement);
     return d;
  }

  // create the multi-dimensional fft descriptors
  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int placement) {
     auto d = new fftDescriptorSR(ctxt, dimensions, placement);
     return d;
  }

  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int64_t in_strides[], int64_t out_strides[],
                                         int placement) {
     auto d = new fftDescriptorSR(ctxt, dimensions,
                                  in_strides, out_strides,
                                  placement);
     return d;
  }

  fftDescriptorSC* createFFTDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int placement) {
     auto d = new fftDescriptorSC(ctxt, dimensions, placement);
     return d;
  }

  fftDescriptorSC* createFFTDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int64_t in_strides[], int64_t out_strides[],
                                         int placement) {
     auto d = new fftDescriptorSC(ctxt, dimensions,
                                  in_strides, out_strides,
                                  placement);
     return d;
  }

  fftDescriptorDR* createFFTDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int placement) {
     auto d = new fftDescriptorDR(ctxt, dimensions, placement);
     return d;
  }

  fftDescriptorDR* createFFTDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int64_t in_strides[], int64_t out_strides[],
                                         int placement) {
     auto d = new fftDescriptorDR(ctxt, dimensions,
                                  in_strides, out_strides,
                                  placement);
     return d;
  }

  fftDescriptorDC* createFFTDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int placement) {
     auto d = new fftDescriptorDC(ctxt, dimensions, placement);
     return d;
  }

  fftDescriptorDC* createFFTDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions,
                                         int64_t in_strides[], int64_t out_strides[],
                                         int placement) {
     auto d = new fftDescriptorDC(ctxt, dimensions,
                                  in_strides, out_strides,
                                  placement);
     return d;
  }

  // destroy the plans
  void destroyFFTDescriptorSR(Context* ctxt, fftDescriptorSR *descSR) {
     delete descSR;
     descSR = nullptr;
     return;
  }

  void destroyFFTDescriptorSC(Context* ctxt, fftDescriptorSC *descSC) {
     delete descSC;
     descSC = nullptr;
     return;
  }

  void destroyFFTDescriptorDR(Context* ctxt, fftDescriptorDR *descDR) {
     delete descDR;
     descDR = nullptr;
     return;
  }

  void destroyFFTDescriptorDC(Context* ctxt, fftDescriptorDC *descDC) {
     delete descDC;
     descDC = nullptr;
     return;
  }

  // set values for fft plans
  void setFFTPlanValuesSR(Context *ctxt, fftDescriptorSR *descSR,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms)
  {
      // this is set for 1D batched transforms INITIALLY until I understand 
      // how to use the inembed and onembed arrays from hipfft

      // set layout
      int64_t in_strides[2] = {(int64_t)0, input_stride};
      int64_t out_strides[2] = {(int64_t)0, output_stride};

      descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_distance);
      descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_distance);

      descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
      descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);

      descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, number_of_transforms);

      // commit the changes
      descSR->fft_plan.commit(ctxt->queue);

      return;
  }

  void setFFTPlanValuesSC(Context *ctxt, fftDescriptorSC *descSC,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms)
  {
      // this is set for 1D batched transforms INITIALLY until I understand 
      // how to use the inembed and onembed arrays from hipfft
     
      // set layout
      int64_t in_strides[2] = {(int64_t)0, input_stride};
      int64_t out_strides[2] = {(int64_t)0, output_stride}; 
     
      descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_distance);
      descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_distance);
     
      descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
      descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);
     
      descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, number_of_transforms);

      // commit the changes
      descSC->fft_plan.commit(ctxt->queue);

      return;
  }

  void setFFTPlanValuesDR(Context *ctxt, fftDescriptorDR *descDR,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms)
  {
      // this is set for 1D batched transforms INITIALLY until I understand 
      // how to use the inembed and onembed arrays from hipfft

      // set layout
      int64_t in_strides[2] = {(int64_t)0, input_stride};
      int64_t out_strides[2] = {(int64_t)0, output_stride};

      descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_distance);
      descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_distance);

      descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
      descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);

      descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, number_of_transforms);

      // commit the changes
      descDR->fft_plan.commit(ctxt->queue);

      return;
  }

  void setFFTPlanValuesDC(Context *ctxt, fftDescriptorDC *descDC,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms)
  {
      // this is set for 1D batched transforms INITIALLY until I understand 
      // how to use the inembed and onembed arrays from hipfft

      // set layout
      int64_t in_strides[2] = {(int64_t)0, input_stride};
      int64_t out_strides[2] = {(int64_t)0, output_stride};

      descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_distance);
      descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_distance);

      descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
      descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);

      descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, number_of_transforms);

      // commit the changes
      descDC->fft_plan.commit(ctxt->queue);

      return;
  }

  // execute the plans
  void fftExecR2C(Context *ctxt, fftDescriptorSR *descSR, float *idata, float _Complex *odata,
                  int64_t reset_placement, int64_t reset_r_strides, int64_t r_strides[])
  {
      // get the fft placement
      int64_t value = 0;
      descSR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (reset_placement == 1) // need to swap placements
      {
          if (value != DFTI_INPLACE)
          {
              std::cout << "         R2C setting PLACEMENT as in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          // if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         R2C setting PLACEMENT as not-in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          // }
          }

          if (reset_r_strides == 1)
          {
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, r_strides);
          }

          // now the placement and input strides have been reset, commit the changes
          descSR->fft_plan.commit(ctxt->queue);
      }

      if (idata == (float*)odata)
      {
	  // std::cout << "in fftExecR2C : in-place transform " << std::endl;
          // auto event = oneapi::mkl::dft::compute_forward(descSR->fft_plan,
          //                                                idata);
          // event.wait();

          oneapi::mkl::dft::compute_forward(descSR->fft_plan,
                                            idata).wait();
      }
      else
      {
          // std::cout << "in fftExecR2C : not-in-place transform " << std::endl;
          // auto event = oneapi::mkl::dft::compute_forward(descSR->fft_plan,
          //                                                idata, 
          //                                                reinterpret_cast<std::complex<float> *>(odata));
          // event.wait();

          oneapi::mkl::dft::compute_forward(descSR->fft_plan,
                                            idata,
                                            reinterpret_cast<std::complex<float> *>(odata)).wait();
      }

      return;
  }

  void fftExecC2R(Context *ctxt, fftDescriptorSR *descSR, float _Complex *idata, float *odata,
                  int64_t reset_placement, int64_t reset_r_strides, int64_t r_strides[])
  {
      int64_t value = 0;
      descSR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (reset_placement == 1) // need to swap placements
      {
          if (value != DFTI_INPLACE)
          {
              std::cout << "         C2R setting PLACEMENT as in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          // if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         C2R setting PLACEMENT as not-in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          // }
          }

          if (reset_r_strides == 1)
          {
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, r_strides);
          }

          // now the placement and input strides have been reset, commit the changes
          descSR->fft_plan.commit(ctxt->queue);
      }

      if ((float*)idata == odata)
      {
          // std::cout << "in fftExecC2R : in-place transform " << std::endl;
	  oneapi::mkl::dft::compute_backward(descSR->fft_plan,
			                     reinterpret_cast<std::complex<float> *>(idata)).wait();
      }
      else
      {
          // std::cout << "in fftExecC2R : not-in-place transform " << std::endl;
	  oneapi::mkl::dft::compute_backward(descSR->fft_plan, 
			                     reinterpret_cast<std::complex<float> *>(idata),
					     odata).wait();
      }

      return;
  }


  void fftExecC2Cforward(Context *ctxt,
                         fftDescriptorSC *descSC,
                         float _Complex *idata,
                         float _Complex *odata)
  {
      int64_t value = 0;
      descSC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (idata == odata)
      {
          // std::cout << "in fftExecC2C_fwd : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
          }

          oneapi::mkl::dft::compute_forward(descSC->fft_plan,
                                            reinterpret_cast<std::complex<float> *>(idata)).wait();
      }
      else
      {
          // std::cout << "in fftExecC2C_fwd : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
          }

          oneapi::mkl::dft::compute_forward(descSC->fft_plan,
                                            reinterpret_cast<std::complex<float> *>(idata),
                                            reinterpret_cast<std::complex<float> *>(odata)).wait();
      }

      return;
  }

  void fftExecC2Cbackward(Context *ctxt,
                          fftDescriptorSC *descSC,
                          float _Complex *idata,
                          float _Complex *odata)
  {   
      int64_t value = 0;
      descSC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (idata == odata)
      {
          // std::cout << "in fftExecC2C_bwd : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
          }

          oneapi::mkl::dft::compute_backward(descSC->fft_plan,
                                             reinterpret_cast<std::complex<float> *>(idata)).wait();
      }
      else
      {
          // std::cout << "in fftExecC2C_bwd : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
          }

              oneapi::mkl::dft::compute_backward(descSC->fft_plan,
                                                 reinterpret_cast<std::complex<float> *>(idata),
                                                 reinterpret_cast<std::complex<float> *>(odata)).wait();
      }

      return;
  }

  // execute the plans
  void fftExecD2Z(Context *ctxt, fftDescriptorDR *descDR, double *idata, double _Complex *odata,
                  int64_t reset_placement, int64_t reset_r_strides, int64_t r_strides[])
  {
      int64_t value = 0;
      descDR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (reset_placement == 1) // need to swap placements
      {
          if (value != DFTI_INPLACE)
          {
              std::cout << "         D2Z setting PLACEMENT as in-place \n";
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          // if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         D2Z setting PLACEMENT as not-in-place \n";
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          // }
          }

          if (reset_r_strides == 1)
          {
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, r_strides);
          }

          // now the placement and input strides have been reset, commit the changes
          descDR->fft_plan.commit(ctxt->queue);
      }

      if (idata == (double*)odata)
      {
          // std::cout << "in fftExecD2Z : in-place transform " << std::endl;
          oneapi::mkl::dft::compute_forward(descDR->fft_plan,
                                            idata).wait();
      }
      else
      {
          // std::cout << "in fftExecD2Z : not-in-place transform " << std::endl;
          oneapi::mkl::dft::compute_forward(descDR->fft_plan,
                                            idata,
                                            reinterpret_cast<std::complex<double> *>(odata)).wait();
      }

      return;
  }

  void fftExecZ2D(Context *ctxt, fftDescriptorDR *descDR, double _Complex *idata, double *odata,
                  int64_t reset_placement, int64_t reset_r_strides, int64_t r_strides[])
  {
      int64_t value = 0;
      descDR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (reset_placement == 1) // need to swap placements
      {
          if (value != DFTI_INPLACE)
          {
              std::cout << "         Z2D setting PLACEMENT as in-place \n";
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
          }
          else
          // if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         Z2D setting PLACEMENT as not-in-place \n";
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
          // }
          }

          if (reset_r_strides == 1)
          {
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, r_strides);
          }

          // now the placement and input strides have been reset, commit the changes
          descDR->fft_plan.commit(ctxt->queue);
      }

      if ((double*)idata == odata)
      {
          // std::cout << "in fftExecZ2D : in-place transform " << std::endl;
          oneapi::mkl::dft::compute_backward(descDR->fft_plan,
                                             reinterpret_cast<std::complex<double> *>(idata)).wait();
      }
      else
      {
          // std::cout << "in fftExecZ2D : not-in-place transform " << std::endl;
          oneapi::mkl::dft::compute_backward(descDR->fft_plan,
                                             reinterpret_cast<std::complex<double> *>(idata),
                                             odata).wait();
      }

      return;
  }

  void fftExecZ2Zforward(Context *ctxt,
                         fftDescriptorDC *descDC,
                         double _Complex *idata,
                         double _Complex *odata)
  {
      int64_t value = 0;
      descDC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (idata == odata)
      {
          // std::cout << "in fftExecZ2Z_fwd : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descDC->fft_plan.commit(ctxt->queue);
          }

          oneapi::mkl::dft::compute_forward(descDC->fft_plan,
                                            reinterpret_cast<std::complex<double> *>(idata)).wait();
      }
      else
      {
          // std::cout << "in fftExecZ2Z_fwd : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descDC->fft_plan.commit(ctxt->queue);
          }

          oneapi::mkl::dft::compute_forward(descDC->fft_plan,
                                            reinterpret_cast<std::complex<double> *>(idata),
                                            reinterpret_cast<std::complex<double> *>(odata)).wait();
      }

      return;
  }

  void fftExecZ2Zbackward(Context *ctxt,
                          fftDescriptorDC *descDC,
                          double _Complex *idata,
                          double _Complex *odata)
  {
      int64_t value = 0;
      descDC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (idata == odata)
      {
          // std::cout << "in fftExecZ2Z_bwd : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descDC->fft_plan.commit(ctxt->queue);
          }

          oneapi::mkl::dft::compute_backward(descDC->fft_plan,
                                             reinterpret_cast<std::complex<double> *>(idata)).wait();
      }
      else
      {
          // std::cout << "in fftExecZ2Z_bwd : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descDC->fft_plan.commit(ctxt->queue);
          }

              oneapi::mkl::dft::compute_backward(descDC->fft_plan,
                                                 reinterpret_cast<std::complex<double> *>(idata),
                                                 reinterpret_cast<std::complex<double> *>(odata)).wait();
      }

      return; 
  }


}// end of namespace
