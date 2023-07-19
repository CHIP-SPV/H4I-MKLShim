// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{

  // check functions
  void checkFFTQueue(Context *ctxt);
  void checkFFTPlan(Context *ctxt, fftDescriptorSR *descSR);
  // void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata, float *odata);
  void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata);


  // create the fft descriptor
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, int64_t nx);
  fftDescriptorSC* createFFTDescriptorSC(Context *ctxt, int64_t nx);
  fftDescriptorDR* createFFTDescriptorDR(Context *ctxt, int64_t nx);
  fftDescriptorDC* createFFTDescriptorDC(Context *ctxt, int64_t nx);

  // create the multi-dimensional fft descriptor
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, std::vector<std::int64_t> dimensions);
  fftDescriptorSC* createFFTDescriptorSC(Context *ctxt, std::vector<std::int64_t> dimensions);
  fftDescriptorDR* createFFTDescriptorDR(Context *ctxt, std::vector<std::int64_t> dimensions);
  fftDescriptorDC* createFFTDescriptorDC(Context *ctxt, std::vector<std::int64_t> dimensions);

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
