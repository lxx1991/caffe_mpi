#ifndef CAFFEE_HALF_UTIL_HPP
#define CAFFEE_HALF_UTIL_HPP
#include "caffe/half/half.hpp"
namespace caffe {
  void floats2halves_cpu(const float* f, Half* h, int n);
  void halves2floats_cpu(const Half* h, float* f, int n);
  void floats2halves_gpu(const float* f, Half* h, int n);
  void halves2floats_gpu(const Half* h, float* f, int n);
}
#endif
