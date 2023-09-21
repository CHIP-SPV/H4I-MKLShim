// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{

  // check functions
  void checkFFTQueue(Context *ctxt);
  void checkFFTPlan(Context *ctxt, fftDescriptorSR *descSR);
  void checkFFTPlan(Context *ctxt, fftDescriptorSC *descSC);
  // void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata, float *odata);
  void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata);


  // create the fft descriptor
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, int64_t nx);
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, int64_t nx,
                                         int64_t istride, int64_t idist,
                                         int64_t ostride, int64_t odist,
                                         int64_t batch);
  fftDescriptorSC* createFFTDescriptorSC(Context *ctxt, int64_t nx);
  fftDescriptorDR* createFFTDescriptorDR(Context *ctxt, int64_t nx);
  fftDescriptorDC* createFFTDescriptorDC(Context *ctxt, int64_t nx);

  // create the multi-dimensional fft descriptor
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, std::vector<std::int64_t> dimensions);
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, std::vector<std::int64_t> dimensions,
                                         int64_t istride, int64_t idist,
                                         int64_t ostride, int64_t odist,
                                         int64_t batch);
  fftDescriptorSC* createFFTDescriptorSC(Context *ctxt, std::vector<std::int64_t> dimensions);
  fftDescriptorDR* createFFTDescriptorDR(Context *ctxt, std::vector<std::int64_t> dimensions);
  fftDescriptorDC* createFFTDescriptorDC(Context *ctxt, std::vector<std::int64_t> dimensions);

  // destroy descriptors
  void destroyFFTDescriptorSR(Context *ctxt, fftDescriptorSR *descSR);
  void destroyFFTDescriptorSC(Context *ctxt, fftDescriptorSC *descSC);
  void destroyFFTDescriptorDR(Context *ctxt, fftDescriptorDR *descDR);
  void destroyFFTDescriptorDC(Context *ctxt, fftDescriptorDC *descDC);

  // set values for fft plans
  void setFFTPlanValuesSR(Context *ctxt, fftDescriptorSR *descSR,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms);

  void setFFTPlanValuesSC(Context *ctxt, fftDescriptorSC *descSC,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms);
  void setFFTPlanValuesDR(Context *ctxt, fftDescriptorDR *descDR,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms);

  void setFFTPlanValuesDC(Context *ctxt, fftDescriptorDC *descDC,
                          int64_t input_stride, int64_t fwd_distance,
                          int64_t output_stride, int64_t bwd_distance,
                          int64_t number_of_transforms);
  

  // Single precision, Real starting domain
  void fftExecR2C(Context *ctxt, fftDescriptorSR *descSR, float *idata, float _Complex *odata);
  void fftExecC2R(Context *ctxt, fftDescriptorSR *descSR, float _Complex *idata, float *odata);

  // Single precision, Complex starting domain
  void fftExecC2Cforward(Context *ctxt, 
		         fftDescriptorSC *descSC, 
		         float _Complex *idata, 
		         float _Complex *odata);
  void fftExecC2Cbackward(Context *ctxt, 
                          fftDescriptorSC *descSC,
                          float _Complex *idata,
                          float _Complex *odata);

  // Double precision, Real starting domain
  void fftExecD2Z(Context *ctxt, fftDescriptorDR *descDR, double *idata, double _Complex *odata);
  void fftExecZ2D(Context *ctxt, fftDescriptorDR *descDR, double _Complex *idata, double *odata);

  // Double precision, Complex starting domain
  void fftExecZ2Zforward(Context *ctxt,
                         fftDescriptorDC *descDC,
                         double _Complex *idata,
                         double _Complex *odata);
  void fftExecZ2Zbackward(Context *ctxt,
                          fftDescriptorDC *descDC,
                          double _Complex *idata,
                          double _Complex *odata);


} // namespace
