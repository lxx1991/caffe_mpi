#ifndef CAFFE_HALF_CUDA_H
#define CAFFE_HALF_CUDA_H
#include <iostream>

// macro retrieved from eigen
#ifdef __NVCC__
  // Disable the "statement is unreachable" message
  #pragma diag_suppress code_is_unreachable
  // Disable the "dynamic initialization in unreachable code" message
  #pragma diag_suppress initialization_not_reachable
  // Disable the "calling a __host__ function from a __host__ __device__ function is not allowed" messages (yes, there are 4 of them)
  #pragma diag_suppress 2651
  #pragma diag_suppress 2653
  #pragma diag_suppress 2668
  #pragma diag_suppress 2669
  #pragma diag_suppress 2670
  #pragma diag_suppress 2671
#endif


#ifdef __CUDACC__
#define HOST_DEVICE_FUNC __host__ __device__
#else
#define HOST_DEVICE_FUNC
#endif


#if defined __CUDACC__
  #include <vector_types.h>
  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
    #define CAFFE_HAS_CUDA_FP16
  #endif
#endif



#if __cplusplus > 199711L
#define CAFFE_EXPLICIT_CAST(tgt_type) explicit operator tgt_type()
#else
#define CAFFE_EXPLICIT_CAST(tgt_type) operator tgt_type()
#endif


#if !defined(CAFFE_HAS_CUDA_FP16)
// Make our own __half definition that is similar to CUDA's.
#include <cmath>
// this definition will conflicts with struct in cublas_api.h imported by common.hpp
#ifndef CUDA_FP16_H_JNESTUG4
#define CUDA_FP16_H_JNESTUG4

 struct __half {
   HOST_DEVICE_FUNC __half() {}
   explicit HOST_DEVICE_FUNC __half(unsigned short raw) : x(raw) {}
   unsigned short x;
 };
#endif

#else
#include <cuda_fp16.h>
#include <host_defines.h>
#endif

namespace caffe {

namespace internal {

inline HOST_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x);
inline HOST_DEVICE_FUNC __half float_to_half_rtne(float ff);
inline HOST_DEVICE_FUNC float half_to_float(__half h);

} // end namespace internal

// Class definition.
// Half is also a POD.
struct Half : public __half {
  HOST_DEVICE_FUNC Half() {}

  HOST_DEVICE_FUNC Half(const __half& h) : __half(h) {}
  HOST_DEVICE_FUNC Half(const Half& h) : __half(h) {}

  explicit HOST_DEVICE_FUNC Half(bool b)
      : __half(internal::raw_uint16_to_half(b ? 0x3c00 : 0)) {}
  template<class T>
  explicit HOST_DEVICE_FUNC Half(const T& val)
      : __half(internal::float_to_half_rtne(static_cast<float>(val))) {}
  // use implicit Half cast, thus use explict float cast to avoid ambigous operator overloading for *, /
  HOST_DEVICE_FUNC Half(float f)
      : __half(internal::float_to_half_rtne(f)) {}

  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(bool) const {
    // +0.0 and -0.0 become false, everything else becomes true.
    return (x & 0x7fff) != 0;
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(signed char) const {
    return static_cast<signed char>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(unsigned char) const {
    return static_cast<unsigned char>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(short) const {
    return static_cast<short>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(unsigned short) const {
    return static_cast<unsigned short>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(int) const {
    return static_cast<int>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(unsigned int) const {
    return static_cast<unsigned int>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(long) const {
    return static_cast<long>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(unsigned long) const {
    return static_cast<unsigned long>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(long long) const {
    return static_cast<long long>(internal::half_to_float(*this));
  }

  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(unsigned long long) const {
    return static_cast<unsigned long long>(internal::half_to_float(*this));
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(float) const {
    return internal::half_to_float(*this);
  }
  HOST_DEVICE_FUNC CAFFE_EXPLICIT_CAST(double) const {
    return static_cast<double>(internal::half_to_float(*this));
  }

  HOST_DEVICE_FUNC Half& operator=(const Half& other) {
    x = other.x;
    return *this;
  }
};

#if defined(CAFFE_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530

// Intrinsics for native fp16 support. Note that on current hardware,
// these are no faster than fp32 arithmetic (you need to use the half2
// versions to get the ALU speed increased), but you do save the
// conversion steps back and forth.

__device__ Half operator + (const Half& a, const Half& b) {
  return __hadd(a, b);
}
__device__ Half operator * (const Half& a, const Half& b) {
  return __hmul(a, b);
}
__device__ Half operator - (const Half& a, const Half& b) {
  return __hsub(a, b);
}
__device__ Half operator / (const Half& a, const Half& b) {
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
}
__device__ Half operator - (const Half& a) {
  return __hneg(a);
}
__device__ Half& operator += (Half& a, const Half& b) {
  a = a + b;
  return a;
}
__device__ Half& operator *= (Half& a, const Half& b) {
  a = a * b;
  return a;
}
__device__ Half& operator -= (Half& a, const Half& b) {
  a = a - b;
  return a;
}
__device__ Half& operator /= (Half& a, const Half& b) {
  a = a / b;
  return a;
}
__device__ bool operator == (const Half& a, const Half& b) {
  return __heq(a, b);
}
__device__ bool operator != (const Half& a, const Half& b) {
  return __hne(a, b);
}
__device__ bool operator < (const Half& a, const Half& b) {
  return __hlt(a, b);
}
__device__ bool operator <= (const Half& a, const Half& b) {
  return __hle(a, b);
}
__device__ bool operator > (const Half& a, const Half& b) {
  return __hgt(a, b);
}
__device__ bool operator >= (const Half& a, const Half& b) {
  return __hge(a, b);
}

#else  // Emulate support for Half floats

// Definitions for CPUs and older CUDA, mostly working through conversion
// to/from fp32.

inline HOST_DEVICE_FUNC Half operator + (const Half& a, const Half& b) {
  return Half(float(a) + float(b));
}
inline HOST_DEVICE_FUNC Half operator * (const Half& a, const Half& b) {
  return Half(float(a) * float(b));
}
inline HOST_DEVICE_FUNC Half operator - (const Half& a, const Half& b) {
  return Half(float(a) - float(b));
}
inline HOST_DEVICE_FUNC Half operator / (const Half& a, const Half& b) {
  return Half(float(a) / float(b));
}
inline HOST_DEVICE_FUNC Half operator - (const Half& a) {
  Half result;
  result.x = a.x ^ 0x8000;
  return result;
}
inline HOST_DEVICE_FUNC Half& operator += (Half& a, const Half& b) {
  a = Half(float(a) + float(b));
  return a;
}
inline HOST_DEVICE_FUNC Half& operator *= (Half& a, const Half& b) {
  a = Half(float(a) * float(b));
  return a;
}
inline HOST_DEVICE_FUNC Half& operator -= (Half& a, const Half& b) {
  a = Half(float(a) - float(b));
  return a;
}
inline HOST_DEVICE_FUNC Half& operator /= (Half& a, const Half& b) {
  a = Half(float(a) / float(b));
  return a;
}
inline HOST_DEVICE_FUNC bool operator == (const Half& a, const Half& b) {
  return float(a) == float(b);
}
inline HOST_DEVICE_FUNC bool operator != (const Half& a, const Half& b) {
  return float(a) != float(b);
}
inline HOST_DEVICE_FUNC bool operator < (const Half& a, const Half& b) {
  return float(a) < float(b);
}
inline HOST_DEVICE_FUNC bool operator <= (const Half& a, const Half& b) {
  return float(a) <= float(b);
}
inline HOST_DEVICE_FUNC bool operator > (const Half& a, const Half& b) {
  return float(a) > float(b);
}
inline HOST_DEVICE_FUNC bool operator >= (const Half& a, const Half& b) {
  return float(a) >= float(b);
}

#endif  // Emulate support for half floats

// Conversion routines, including fallbacks for the host or older CUDA.
// Note that newer Intel CPUs (Haswell or newer) have vectorized versions of
// these in hardware. If we need more performance on older/other CPUs, they are
// also possible to vectorize directly.

namespace internal {

inline HOST_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x) {
  __half h;
  h.x = x;
  return h;
}

union FP32 {
  unsigned int u;
  float f;
};

inline HOST_DEVICE_FUNC __half float_to_half_rtne(float ff) {
#if defined(CAFFE_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __float2half(ff);
#else
  FP32 f; f.f = ff;

  const FP32 f32infty = { 255 << 23 };
  const FP32 f16max = { (127 + 16) << 23 };
  const FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
  unsigned int sign_mask = 0x80000000u;
  __half o;
  o.x = static_cast<unsigned short>(0x0u);

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) {  // result is Inf or NaN (all exponent bits set)
    o.x = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
  } else {  // (De)normalized number or zero
    if (f.u < (113 << 23)) {  // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      o.x = static_cast<unsigned short>(f.u - denorm_magic.u);
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      o.x = static_cast<unsigned short>(f.u >> 13);
    }
  }

  o.x |= static_cast<unsigned short>(sign >> 16);
  return o;
#endif
}

inline HOST_DEVICE_FUNC float half_to_float(__half h) {
#if defined(CAFFE_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __half2float(h);
#else
  const FP32 magic = { 113 << 23 };
  const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
  FP32 o;

  o.u = (h.x & 0x7fff) << 13;             // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u;   // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {     // Inf/NaN?
    o.u += (128 - 16) << 23;    // extra exp adjust
  } else if (exp == 0) {        // Zero/Denormal?
    o.u += 1 << 23;             // extra exp adjust
    o.f -= magic.f;             // renormalize
  }

  o.u |= (h.x & 0x8000) << 16;    // sign bit
  return o.f;
#endif
}

inline HOST_DEVICE_FUNC bool (isinf)(const caffe::Half& a) {
  return (a.x & 0x7fff) == 0x7c00;
}
inline HOST_DEVICE_FUNC bool (isnan)(const caffe::Half& a) {
#if defined(CAFFE_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(a);
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}

} // end namespace internal

// Traits.
namespace traits {
  HOST_DEVICE_FUNC static inline caffe::Half epsilon() {
    return internal::raw_uint16_to_half(0x0800);
  }
  HOST_DEVICE_FUNC static inline caffe::Half dummy_precision() { return Half(1e-2f); }
  HOST_DEVICE_FUNC static inline caffe::Half highest() {
    return internal::raw_uint16_to_half(0x7bff);
  }
  HOST_DEVICE_FUNC static inline caffe::Half lowest() {
    return internal::raw_uint16_to_half(0xfbff);
  }
  HOST_DEVICE_FUNC static inline caffe::Half infinity() {
    return internal::raw_uint16_to_half(0x7c00);
  }
  HOST_DEVICE_FUNC static inline caffe::Half quiet_NaN() {
    return internal::raw_uint16_to_half(0x7c01);
  }
  
}
// Infinity/NaN checks.


} // end namespace caffe

namespace std {

__attribute__((always_inline)) inline ostream& operator << (ostream& os, const caffe::Half& v) {
  os << static_cast<float>(v);
  return os;
}

#if __cplusplus > 199711L
template <>
struct hash<caffe::Half> {
  HOST_DEVICE_FUNC inline std::size_t operator()(const caffe::Half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};
#endif

} // namespace std


inline HOST_DEVICE_FUNC caffe::Half abs(const caffe::Half& a) {
  caffe::Half result;
  result.x = a.x & 0x7FFF;
  return result;
}
inline HOST_DEVICE_FUNC caffe::Half exp(const caffe::Half& a) {
  return caffe::Half(::expf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half log(const caffe::Half& a) {
  return caffe::Half(::logf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half sqrt(const caffe::Half& a) {
  return caffe::Half(::sqrtf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half pow(const caffe::Half& a, const caffe::Half& b) {
  return caffe::Half(::powf(float(a), float(b)));
}
inline HOST_DEVICE_FUNC caffe::Half sin(const caffe::Half& a) {
  return caffe::Half(::sinf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half cos(const caffe::Half& a) {
  return caffe::Half(::cosf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half tan(const caffe::Half& a) {
  return caffe::Half(::tanf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half tanh(const caffe::Half& a) {
  return caffe::Half(::tanhf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half floor(const caffe::Half& a) {
  return caffe::Half(::floorf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half round(const caffe::Half& a) {
  return floor(a + caffe::Half(0.5f));
}
inline HOST_DEVICE_FUNC caffe::Half ceil(const caffe::Half& a) {
  return caffe::Half(::ceilf(float(a)));
}

inline HOST_DEVICE_FUNC caffe::Half min(const caffe::Half& a, const caffe::Half& b) {
#if defined(CAFFE_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(b, a) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f2 < f1 ? b : a;
#endif
}
inline HOST_DEVICE_FUNC caffe::Half max(const caffe::Half& a, const caffe::Half& b) {
#if defined(CAFFE_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f1 < f2 ? b : a;
#endif
}



// Standard mathematical functions and trancendentals.
inline HOST_DEVICE_FUNC caffe::Half fabsh(const caffe::Half& a) {
  caffe::Half result;
  result.x = a.x & 0x7FFF;
  return result;
}
inline HOST_DEVICE_FUNC caffe::Half exph(const caffe::Half& a) {
  return caffe::Half(::expf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half logh(const caffe::Half& a) {
  return caffe::Half(::logf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half sqrth(const caffe::Half& a) {
  return caffe::Half(::sqrtf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half powh(const caffe::Half& a, const caffe::Half& b) {
  return caffe::Half(::powf(float(a), float(b)));
}
inline HOST_DEVICE_FUNC caffe::Half floorh(const caffe::Half& a) {
  return caffe::Half(::floorf(float(a)));
}
inline HOST_DEVICE_FUNC caffe::Half ceilh(const caffe::Half& a) {
  return caffe::Half(::ceilf(float(a)));
}
inline HOST_DEVICE_FUNC int (isnan)(const caffe::Half& a) {
  return (isnan)(a);
}
inline HOST_DEVICE_FUNC int (isinf)(const caffe::Half& a) {
  return (caffe::internal::isinf)(a);
}
inline HOST_DEVICE_FUNC int (isfinite)(const caffe::Half& a) {
  return !(caffe::internal::isinf)(a) && !(caffe::internal::isnan)(a);
}




#endif // CAFFE_HALF_CUDA_H
