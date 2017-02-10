#include "caffe/half/half_util.hpp"
#include "caffe/common.hpp"
#include <omp.h>

namespace caffe {
  void floats2halves_cpu(const float* f, Half* h, int n) {
    CHECK(f);
    CHECK(h);
    #pragma omp for
    for (int i = 0; i < n; i++) {
      h[i] = Half(f[i]);
    }
  }
  
  void halves2floats_cpu(const Half* h, float* f, int n) {
    CHECK(f);
    CHECK(h);
    #pragma omp for
    for (int i = 0; i < n; i++) {
      f[i] = float(h[i]);
    }
  }
} // namespace caffe
