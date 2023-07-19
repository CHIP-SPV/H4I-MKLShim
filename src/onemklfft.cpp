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
      fftDescriptorSR(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
	  // commit the plan
	  fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
	  // wait for everything to complete before continuing
	  reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
	  // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	  // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      // execute R2C ... is it better to do this in the struct?
      void fftR2C(Context *ctxt, float *idata)
      {
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
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
      fftDescriptorSC(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::COMPLEX> fft_plan;

      ~fftDescriptorSC() {}
  };

  // struct for the Double precision and Real starting domain descriptor
  struct fftDescriptorDR
  {
      // descriptor for 1d transforms
      fftDescriptorDR(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;

      ~fftDescriptorDR() {}
  };

  // struct for the Double precision and Complex starting domain descriptor
  struct fftDescriptorDC
  {
      // descriptor for 1d transforms
      fftDescriptorDC(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
          // wait for everything to complete before continuing
          // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::COMPLEX> fft_plan;

      ~fftDescriptorDC() {}
  };


  // simple queue check ... will be removed
  void checkFFTQueue(Context* ctxt) {
     std::cout << "in checkFFTQueue : device info =  " << 
	     reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
     return;
  }

  // simple plan check for the SR plan ... will be removed
  void checkFFTPlan(Context *ctxt, fftDescriptorSR *desc)
  {
     // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

     int64_t value = 0;
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

     reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

     return;
  }

  // void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata, float _Complex *odata)
  void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata)
  {
      std::cout << "in checkFFTPlan : Nbytes = " << Nbytes << "   device info =  " <<
             reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
      // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.memcpy(odata, idata, Nbytes);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      oneapi::mkl::dft::compute_forward(descSR->fft_plan, idata);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      // reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.memcpy(odata, idata, Nbytes);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      return;
  }

  // create the 1d fft descriptors
  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorSR(ctxt, length);
     return d;
  }

  fftDescriptorSC* createFFTDescriptorSC(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorSC(ctxt, length);
     return d;
  }

  fftDescriptorDR* createFFTDescriptorDR(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorDR(ctxt, length);
     return d;
  }

  fftDescriptorDC* createFFTDescriptorDC(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorDC(ctxt, length);
     return d;
  }

  // create the multi-dimensional fft descriptors
  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions) {
     auto d = new fftDescriptorSR(ctxt, dimensions);
     return d;
  }

  fftDescriptorSC* createFFTDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions) {
     auto d = new fftDescriptorSC(ctxt, dimensions);
     return d;
  }

  fftDescriptorDR* createFFTDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions) {
     auto d = new fftDescriptorDR(ctxt, dimensions);
     return d;
  }

  fftDescriptorDC* createFFTDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions) {
     auto d = new fftDescriptorDC(ctxt, dimensions);
     return d;
  }

  // execute the plans
  void fftExecR2C(Context *ctxt, fftDescriptorSR *descSR, float *idata, float _Complex *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      int64_t value = 0;
      descSR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      
      if (idata == (float*)odata)
      {
	  // std::cout << "in fftExecR2C : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
	      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSR->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSR->fft_plan,
                                            idata);
	  reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }
      else
      {
          // std::cout << "in fftExecR2C : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSR->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSR->fft_plan,
                                            idata, 
			                    reinterpret_cast<std::complex<float> *>(odata));
	  reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      return;
  }

  void fftExecC2R(Context *ctxt, fftDescriptorSR *descSR, float _Complex *idata, float *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      int64_t value = 0;
      descSR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      if ((float*)idata == odata)
      {
          // std::cout << "in fftExecC2R : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSR->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

	  oneapi::mkl::dft::compute_backward(descSR->fft_plan,
			                     reinterpret_cast<std::complex<float> *>(idata));
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }
      else
      {
          // std::cout << "in fftExecC2R : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSR->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

	  oneapi::mkl::dft::compute_backward(descSR->fft_plan, 
			                     reinterpret_cast<std::complex<float> *>(idata),
					     odata);
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      return;
  }


  void fftExecC2Cforward(Context *ctxt,
                         fftDescriptorSC *descSC,
                         float _Complex *idata,
                         float _Complex *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      int64_t value = 0;
      descSC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      if (idata == odata)
      {
          // std::cout << "in fftExecC2C_fwd : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSC->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSC->fft_plan,
                                            reinterpret_cast<std::complex<float> *>(idata));

          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }
      else
      {
          // std::cout << "in fftExecC2C_fwd : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSC->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSC->fft_plan,
                                            reinterpret_cast<std::complex<float> *>(idata),
                                            reinterpret_cast<std::complex<float> *>(odata));
          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      return;
  }

  void fftExecC2Cbackward(Context *ctxt,
                          fftDescriptorSC *descSC,
                          float _Complex *idata,
                          float _Complex *odata)
  {   
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
  
      int64_t value = 0;
      descSC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      if (idata == odata)
      {
          // std::cout << "in fftExecC2C_bwd : in-place transform " << std::endl;

          if (value != DFTI_INPLACE)
          {
              std::cout << "         setting PLACEMENT as in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSC->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descSC->fft_plan,
                                             reinterpret_cast<std::complex<float> *>(idata));

          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }
      else
      {
          // std::cout << "in fftExecC2C_bwd : not-in-place transform " << std::endl;

          if (value != DFTI_NOT_INPLACE)
          {
              std::cout << "         setting PLACEMENT as not-in-place \n";
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
              descSC->fft_plan.commit(reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue);
              reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
          }

              oneapi::mkl::dft::compute_backward(descSC->fft_plan,
                                                 reinterpret_cast<std::complex<float> *>(idata),
                                                 reinterpret_cast<std::complex<float> *>(odata));

          reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      }

      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();

      return;
  }

  // execute the plans
  void fftExecD2Z(Context *ctxt, fftDescriptorDR *descDR, double *idata, double _Complex *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      return;
  }

  void fftExecZ2D(Context *ctxt, fftDescriptorDR *descDR, double _Complex *idata, double *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      return;
  }

  void fftExecZ2Zforward(Context *ctxt,
                         fftDescriptorDC *descDC,
                         double _Complex *idata,
                         double _Complex *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      return;
  }

  void fftExecZ2Zbackward(Context *ctxt,
                          fftDescriptorDC *descDC,
                          double _Complex *idata,
                          double _Complex *odata)
  {
      reinterpret_cast<ContextImpl*>(ctxt)->bedata->queue.wait();
      return; 
  }


}// end of namespace
