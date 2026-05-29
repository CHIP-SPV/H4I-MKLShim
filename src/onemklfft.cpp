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
	  fft_plan.commit(ctxt->queue);
	  // wait for everything to complete before continuing
	  ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
	  // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	  // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // cuFFT-compatible layout: store output as N/2+1 interleaved complex
          // values along the last dim (not oneMKL's default packed CCE format),
          // with NOT_INPLACE placement and contiguous row-major strides.
          if (dimensions.size() >= 2) {
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                                 DFTI_COMPLEX_COMPLEX);
              std::vector<std::int64_t> fwd_strides(dimensions.size() + 1, 0);
              std::vector<std::int64_t> bwd_strides(dimensions.size() + 1, 0);
              std::int64_t s = 1, cs = 1;
              for (int i = dimensions.size() - 1; i >= 0; i--) {
                  fwd_strides[i + 1] = s;
                  bwd_strides[i + 1] = cs;
                  s *= dimensions[i];
                  cs *= (i == (int)dimensions.size() - 1)
                            ? (dimensions[i] / 2 + 1) : dimensions[i];
              }
              fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
              fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
          }
          fft_plan.commit(ctxt->queue);
          ctxt->queue.wait();
      }

      // descriptor with explicit FWD/BWD strides (for padded PlanMany).
      // oneMKL REAL domain default is COMPLEX_REAL conjugate-even storage,
      // where BWD_STRIDES must be expressed in real-element units. To pass
      // BWD_STRIDES in complex-element units (cuFFT convention), we must
      // switch to COMPLEX_COMPLEX storage.
      fftDescriptorSR(Context* ctxt,
                      std::vector<std::int64_t> dimensions,
                      std::vector<std::int64_t> fwd_strides,
                      std::vector<std::int64_t> bwd_strides)
          : fft_plan(dimensions),
            stored_fwd_strides(fwd_strides),
            stored_bwd_strides(bwd_strides)
      {
          fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::NOT_INPLACE);
          fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             DFTI_COMPLEX_COMPLEX);
          fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
          fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
          fft_plan.commit(ctxt->queue);
          ctxt->queue.wait();
      }

      void recommit(Context* ctxt)
      {
          if (!stored_fwd_strides.empty() || !stored_bwd_strides.empty()) {
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                                 DFTI_COMPLEX_COMPLEX);
          }
          if (!stored_fwd_strides.empty())
              fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, stored_fwd_strides);
          if (!stored_bwd_strides.empty())
              fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, stored_bwd_strides);
          fft_plan.commit(ctxt->queue);
          ctxt->queue.wait();
      }

      // execute R2C ... is it better to do this in the struct?
      void fftR2C(Context *ctxt, float *idata)
      {
          ctxt->queue.wait();
          return;
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;
      std::vector<std::int64_t> stored_fwd_strides;
      std::vector<std::int64_t> stored_bwd_strides;

      ~fftDescriptorSR() {}
  };

  // struct for the Single precision and Complex starting domain descriptor
  struct fftDescriptorSC
  {
      // descriptor for 1d transforms
      fftDescriptorSC(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
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
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // cuFFT-compatible layout: see fftDescriptorSR multi-dim ctor for rationale.
          if (dimensions.size() >= 2) {
              fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                                 DFTI_COMPLEX_COMPLEX);
              std::vector<std::int64_t> fwd_strides(dimensions.size() + 1, 0);
              std::vector<std::int64_t> bwd_strides(dimensions.size() + 1, 0);
              std::int64_t s = 1, cs = 1;
              for (int i = dimensions.size() - 1; i >= 0; i--) {
                  fwd_strides[i + 1] = s;
                  bwd_strides[i + 1] = cs;
                  s *= dimensions[i];
                  cs *= (i == (int)dimensions.size() - 1)
                            ? (dimensions[i] / 2 + 1) : dimensions[i];
              }
              fft_plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
              fft_plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
          }
          fft_plan.commit(ctxt->queue);
          ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
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
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();

	  // use the soon-to-be default (can be removed in the future) for storing complex numbers
          // fft_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
          // commit the plan
          // fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          // ctxt->queue.wait();
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
     // ctxt->queue.wait();

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

     ctxt->queue.wait();

     return;
  }

  // void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata, float _Complex *odata)
  void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata)
  {
      std::cout << "in checkFFTPlan : Nbytes = " << Nbytes << "   device info =  " <<
             ctxt->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
      // ctxt->queue.memcpy(odata, idata, Nbytes);
      ctxt->queue.wait();

      oneapi::mkl::dft::compute_forward(descSR->fft_plan, idata);
      ctxt->queue.wait();

      // ctxt->queue.memcpy(odata, idata, Nbytes);
      ctxt->queue.wait();

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

  // strided multi-dimensional R2C/C2R descriptor (cuFFT planMany compatibility)
  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt,
                                          std::vector<std::int64_t> dimensions,
                                          std::vector<std::int64_t> fwd_strides,
                                          std::vector<std::int64_t> bwd_strides) {
     return new fftDescriptorSR(ctxt, dimensions, fwd_strides, bwd_strides);
  }

  void recommitFFTDescriptorSR(Context* ctxt, fftDescriptorSR* desc) {
     desc->recommit(ctxt);
  }

  // execute the plans
  //
  // The Context's SYCL queue wraps chipStar's L0 immediate command list (see
  // Context.cpp:Update). All chipStar HIP kernels and MKL submissions land on
  // the same in-order command list, so ordering between them is enforced at
  // the L0 level. Host-side queue.wait() is only needed when MKL state must be
  // observable on the host (e.g. after commit() before set_value/get_value
  // reads it back); on the hot exec path it is pure overhead.
  void fftExecR2C(Context *ctxt, fftDescriptorSR *descSR, float *idata, float _Complex *odata)
  {
      int64_t value = 0;
      descSR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (idata == (float*)odata)
      {
          if (value != DFTI_INPLACE)
          {
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descSR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();   // commit may JIT; ensure plan ready before compute_*.
          }

          oneapi::mkl::dft::compute_forward(descSR->fft_plan, idata);
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descSR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSR->fft_plan,
                                            idata,
                                            reinterpret_cast<std::complex<float> *>(odata));
      }
  }

  void fftExecC2R(Context *ctxt, fftDescriptorSR *descSR, float _Complex *idata, float *odata)
  {
      int64_t value = 0;
      descSR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if ((float*)idata == odata)
      {
          if (value != DFTI_INPLACE)
          {
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descSR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descSR->fft_plan,
                                             reinterpret_cast<std::complex<float> *>(idata));
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descSR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descSR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descSR->fft_plan,
                                             reinterpret_cast<std::complex<float> *>(idata),
                                             odata);
      }
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
          if (value != DFTI_INPLACE)
          {
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSC->fft_plan,
                                            reinterpret_cast<std::complex<float> *>(idata));
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descSC->fft_plan,
                                            reinterpret_cast<std::complex<float> *>(idata),
                                            reinterpret_cast<std::complex<float> *>(odata));
      }
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
          if (value != DFTI_INPLACE)
          {
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descSC->fft_plan,
                                             reinterpret_cast<std::complex<float> *>(idata));
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descSC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descSC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descSC->fft_plan,
                                             reinterpret_cast<std::complex<float> *>(idata),
                                             reinterpret_cast<std::complex<float> *>(odata));
      }
  }

  void fftExecD2Z(Context *ctxt, fftDescriptorDR *descDR, double *idata, double _Complex *odata)
  {
      int64_t value = 0;
      descDR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if (idata == (double*)odata)
      {
          if (value != DFTI_INPLACE)
          {
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descDR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descDR->fft_plan, idata);
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descDR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descDR->fft_plan, idata,
                                            reinterpret_cast<std::complex<double> *>(odata));
      }
  }

  void fftExecZ2D(Context *ctxt, fftDescriptorDR *descDR, double _Complex *idata, double *odata)
  {
      int64_t value = 0;
      descDR->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);

      if ((double*)idata == odata)
      {
          if (value != DFTI_INPLACE)
          {
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descDR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descDR->fft_plan, odata);
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descDR->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descDR->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_backward(descDR->fft_plan,
                                             reinterpret_cast<std::complex<double> *>(idata),
                                             odata);
      }
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
          if (value != DFTI_INPLACE)
          {
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              descDC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descDC->fft_plan,
                                            reinterpret_cast<std::complex<double> *>(idata));
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              descDC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          oneapi::mkl::dft::compute_forward(descDC->fft_plan,
                                            reinterpret_cast<std::complex<double> *>(idata),
                                            reinterpret_cast<std::complex<double> *>(odata));
      }
  }

  void fftExecZ2Zbackward(Context *ctxt,
                          fftDescriptorDC *descDC,
                          double _Complex *idata,
                          double _Complex *odata)
  {
      ctxt->queue.wait();

      int64_t value = 0;
      descDC->fft_plan.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
      ctxt->queue.wait();

      if (idata == odata)
      {
          if (value != DFTI_INPLACE)
          {
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
              ctxt->queue.wait();
              descDC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          try {
              oneapi::mkl::dft::compute_backward(descDC->fft_plan, 
                                               reinterpret_cast<std::complex<double> *>(idata));
              ctxt->queue.wait();
          } catch (sycl::exception const& e) {
              std::cerr << "fftExecZ2Zbackward compute failed: " << e.what() << std::endl;
          }
      }
      else
      {
          if (value != DFTI_NOT_INPLACE)
          {
              descDC->fft_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
              ctxt->queue.wait();
              descDC->fft_plan.commit(ctxt->queue);
              ctxt->queue.wait();
          }

          try {
              oneapi::mkl::dft::compute_backward(descDC->fft_plan, 
                                               reinterpret_cast<std::complex<double> *>(idata),
                                               reinterpret_cast<std::complex<double> *>(odata));
              ctxt->queue.wait();
          } catch (sycl::exception const& e) {
              std::cerr << "fftExecZ2Zbackward compute failed: " << e.what() << std::endl;
          }
      }

      ctxt->queue.wait();
      return;
  }

  // Implementations of destroy functions
  void destroyFFTDescriptorSR(Context *ctxt, fftDescriptorSR *descSR) {
    if (descSR != nullptr) {
      delete descSR;
    }
  }

  void destroyFFTDescriptorSC(Context *ctxt, fftDescriptorSC *descSC) {
    if (descSC != nullptr) {
      delete descSC;
    }
  }

  void destroyFFTDescriptorDR(Context *ctxt, fftDescriptorDR *descDR) {
    if (descDR != nullptr) {
      delete descDR;
    }
  }

  void destroyFFTDescriptorDC(Context *ctxt, fftDescriptorDC *descDC) {
    if (descDC != nullptr) {
      delete descDC;
    }
  }

}// end of namespace
