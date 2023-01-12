<!---
Copyright 2021-2023 UT-Battelle
See LICENSE.txt in the root of the source distribution for license info.
-->

# Overview

This library provides a shim over functions provided by the SYCL-based Intel
MKL library implementation.  It is designed to be used as a backend allowing
other libraries that provide a well-defined HIP-based interface (e.g. HipBLAS)
to run on Intel GPUs.  It is needed because:
* the available HIP compilers cannot also make calls to the SYCL-based MKL functions; and
* the Intel DPC++ compiler that can compile code that calls the SYCL-based MKL
functions cannot compile the HIP headers.
This library's headers are implemented so that they are accepted by both compilers.

It is implemented as a distinct CMake project and maintained in a distinct 
source code repository because it is intended to service multiple 
higher-level libraries.

# Configure, Build, and Install

The project uses CMake for configuration, build, and install.

