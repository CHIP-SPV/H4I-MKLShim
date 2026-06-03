// Regression test for the context_tbl leak in MKLShim::Destroy.
//
// Create() inserts an entry keyed by the QUEUE native handle.
// Update() inserts a *second* entry under the new queue handle while
// keeping the same Context pointer, without bumping the refcount.
// Destroy() then erases only the FIRST table entry it finds matching
// the Context pointer (std::find_if), so any additional entries
// pointing to the now-deleted Context remain in context_tbl as
// dangling pointers.
//
// When a future Create() is called and the QUEUE handle happens to
// collide with one of those orphaned keys (which happens routinely
// when the chipStar runtime recycles its L0 cmd queue handles for
// freshly created hipStreams), the stale Context pointer is returned
// and oneMKL eventually segfaults on its destroyed SYCL queue.
//
// The fix: Destroy() must erase *every* table entry matching the
// Context pointer, not just the first.
//
// Before fix: this test fails with "table leaked N entries".
// After fix: this test prints "PASSED".
#include <cstdlib>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include "h4i/mklshim/mklshim.h"

namespace {

std::vector<unsigned long> handlesFor(uintptr_t stream) {
  int n = 0;
  hipGetBackendNativeHandles(stream, nullptr, &n);
  std::vector<unsigned long> h(n);
  hipGetBackendNativeHandles(stream, h.data(), 0);
  return h;
}

} // namespace

int main() {
  using H4I::MKLShim::Context;
  using H4I::MKLShim::Create;
  using H4I::MKLShim::Update;
  using H4I::MKLShim::Destroy;
  using H4I::MKLShim::context_tbl;

  const size_t startSize = context_tbl.size();
  std::cout << "context_tbl initial size = " << startSize << std::endl;

  // Grab two distinct queue handle sets: the default stream + a
  // freshly created stream. The two sets MUST differ in handles[QUEUE]
  // for the bug to be triggerable.
  auto hDefault = handlesFor(0);

  hipStream_t s = nullptr;
  if (hipStreamCreate(&s) != hipSuccess) {
    std::cerr << "FAILED: hipStreamCreate" << std::endl;
    return EXIT_FAILURE;
  }
  auto hStream = handlesFor(reinterpret_cast<uintptr_t>(s));

  // Create on default-stream handles, then Update to point at the
  // freshly created stream's handles. This is the same call sequence
  // H4I-HipFFT uses (hipfftCreate -> hipfftSetStream).
  Context *ctxt = Create(hDefault.data(), hDefault.size());
  if (ctxt == nullptr) {
    std::cerr << "FAILED: Create returned null" << std::endl;
    return EXIT_FAILURE;
  }
  Update(ctxt, hStream.data(), hStream.size());

  const size_t afterUpdateSize = context_tbl.size();
  std::cout << "context_tbl size after Create+Update = "
            << afterUpdateSize << std::endl;

  // Destroy the (only) live Context. The table must now drop every
  // entry that referenced it.
  Destroy(ctxt);

  const size_t finalSize = context_tbl.size();
  std::cout << "context_tbl size after Destroy = " << finalSize << std::endl;

  hipStreamDestroy(s);

  if (finalSize != startSize) {
    std::cerr << "FAILED: context_tbl leaked "
              << (finalSize - startSize) << " entries past Destroy()"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "PASSED" << std::endl;
  return EXIT_SUCCESS;
}
