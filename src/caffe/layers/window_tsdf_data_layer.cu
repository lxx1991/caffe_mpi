#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void depth2tsdf_depth(const int nthreads,
  const Dtype* depth_data, const Dtype* K_data,
  const Dtype* R_data, const Dtype* window_data, const int tsdf_size,
  const int im_h, const int im_w, Dtype* tsdf_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype x = Dtype(index % tsdf_size);
    Dtype y = Dtype((index / tsdf_size) % tsdf_size);   
    Dtype z = Dtype((index / tsdf_size / tsdf_size) % tsdf_size);
    Dtype delta_x = (window_data[3] - window_data[0]) / Dtype(tsdf_size);  
    Dtype delta_y = (window_data[4] - window_data[1]) / Dtype(tsdf_size);  
    Dtype delta_z = (window_data[5] - window_data[2]) / Dtype(tsdf_size);  
    Dtype mu = 2.0 * max(max(delta_x, delta_y), delta_z);
    tsdf_data[index] = -mu;
    
    x = window_data[0] + (x + 0.5) * delta_x;
    y = window_data[1] + (y + 0.5) * delta_y;
    z = window_data[2] + (z + 0.5) * delta_z;
    // project to image plane
    // swap y, z and -y
    Dtype xx = R_data[0] * x + R_data[3] * y + R_data[6] * z;
    Dtype zz = R_data[1] * x + R_data[4] * y + R_data[7] * z;
    Dtype yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = round(xx * K_data[0] / zz + K_data[2]) - 1;
    int iy = round(yy * K_data[4] / zz + K_data[5]) - 1;
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
      tsdf_data[index] = -1.0;
      return;
    }
    Dtype depth = depth_data[iy * im_w + ix];
    if (depth < 0.0001) {
      tsdf_data[index] = -1.0;
      return;
    }
    tsdf_data[index] = mu;
    // project the depth point to 3d
    Dtype tdx = (Dtype(ix) - K_data[2]) * depth / K_data[0];
    Dtype tdz =  - (Dtype(iy) - K_data[5]) * depth / K_data[4];
    Dtype tdy = depth;
    Dtype dx = R_data[0] * tdx + R_data[1] * tdy + R_data[2] * tdz;
    Dtype dy = R_data[3] * tdx + R_data[4] * tdy + R_data[5] * tdz;
    Dtype dz = R_data[6] * tdx + R_data[7] * tdy + R_data[8] * tdz;
    
    // distance
    Dtype dist = (x - dx) * (x - dx) + (y - dy) * (y - dy) +
        (z - dz) * (z - dz);
    dist = sqrt(dist);
    if (zz > depth) {
     dist = - dist;
    }
    dist = dist / mu;
    dist = max(dist, -1.0);
    dist = min(dist, 1.0);
    tsdf_data[index] = dist;
  }
}

template <typename Dtype>
void WindowTSDFDataLayer<Dtype>::depth2tsdf_GPU(
    const int tsdf_size, const int im_h, const int im_w) {
  const int count = tsdf_.count();
  const Dtype* depth_data = depth_.gpu_data();
  const Dtype* K_data = the_K_.gpu_data();
  const Dtype* R_data = the_R_.gpu_data();
  const Dtype* window_data = the_window_.gpu_data();
  Dtype* tsdf_data = tsdf_.mutable_gpu_data();
  depth2tsdf_depth<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, depth_data, K_data, R_data, window_data,
      tsdf_size, im_h, im_w, tsdf_data);
  CUDA_POST_KERNEL_CHECK;
}

template void WindowTSDFDataLayer<float>::depth2tsdf_GPU(
    const int tsdf_size, const int im_h, const int im_w);
template void WindowTSDFDataLayer<double>::depth2tsdf_GPU(
    const int tsdf_size, const int im_h, const int im_w);

} // namespace caffe
