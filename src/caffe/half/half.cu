#include "caffe/half/half.hpp"
#include "caffe/common.hpp"
#include <omp.h>
namespace caffe {

  __global__ void copy_float2half_kernel(const int N, const float* idata, Half *odata)
  {
    for (int idx = threadIdx.x + blockDim.x*blockIdx.x; idx < N; idx += blockDim.x*gridDim.x) {
      odata[idx] = Half(idata[idx]);
    }
  }

  void floats2halves_gpu(const float* f, Half* h, int n) {
    CHECK(f);
    CHECK(h);
    copy_float2half_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, f, h);
    CUDA_POST_KERNEL_CHECK;
    
  }
  
  __global__ void copy_half2float_kernel(const int N, const Half* idata, float *odata)
  {
    for (int idx = threadIdx.x + blockDim.x*blockIdx.x; idx < N; idx += blockDim.x*gridDim.x) {
      odata[idx] = float(idata[idx]);
    }
  }
  void halves2floats_gpu(const Half* h, float* f, int n) {
    CHECK(f);
    CHECK(h);
    copy_half2float_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, h, f);
    CUDA_POST_KERNEL_CHECK;
    
  }


}
