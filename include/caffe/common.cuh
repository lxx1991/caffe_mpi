// Copyright 2014 George Papandreou

#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

#include "caffe/half/half.hpp"
#include "caffe/half/half_util.hpp"

static __inline__ __device__ caffe::Half atomicAdd(caffe::Half* address, caffe::Half val) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  union U {
      unsigned int i;
      __half2 h;
   };
   union Up {
      unsigned int * i;
      __half2 * h;
   };
   Up up;
   __shared__ __half2 temp_address;
   temp_address.x = ((__half)(*address)).x;
   up.h = &temp_address;
   U old;
   (old.h).x = ((__half)(*address)).x;
   U assumed;
   do {
      assumed.i = old.i;
      U temp;
      __half2 temp_val;
      temp_val.x = ((__half)val).x;
      temp.h = __hadd2(temp_val, assumed.h);
      old.i = atomicCAS( up.i, assumed.i, temp.i);
   } while (assumed.i != old.i );
   __half temp_val;
   temp_val.x = (old.h).x;
   return (caffe::Half)temp_val;
#else
   assert(false);
   return caffe::Half(.0f);
#endif
}


// CUDA: atomicAdd is not defined for doubles
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}


#endif
#endif

