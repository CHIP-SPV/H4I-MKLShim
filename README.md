# Overview

This library provides a shim API over functions provided by the SYCL-based
Intel MKL library version.  It is designed to be used as a backend allowing
other libraries that provide a well-defined HIP-based interface (e.g. HipBLAS)
to run on Intel GPUs.  It is needed because:
* the available HIP compilers cannot also make calls to the SYCL-based MKL functions; and
* the Intel DPC++ compiler that can compile code that calls the SYCL-based MKL
functions cannot compile the HIP headers.
This library's headers are implemented so that they are accepted by both compilers.

