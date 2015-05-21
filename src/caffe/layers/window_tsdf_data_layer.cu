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
  const Dtype* R_data, const Dtype* bb3d_data, const int tsdf_size,
  const int im_h, const int im_w, Dtype* tsdf_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int volume_size = tsdf_size * tsdf_size * tsdf_size;
    Dtype x = Dtype(index % tsdf_size);
    Dtype y = Dtype((index / tsdf_size) % tsdf_size);   
    Dtype z = Dtype((index / tsdf_size / tsdf_size) % tsdf_size);
    Dtype delta_x = 2 * bb3d_data[12] / Dtype(tsdf_size);  
    Dtype delta_y = 2 * bb3d_data[13] / Dtype(tsdf_size);  
    Dtype delta_z = 2 * bb3d_data[14] / Dtype(tsdf_size);  
    tsdf_data[index] = - 2.0 * delta_x;
    tsdf_data[index + volume_size] = - 2.0 * delta_y;
    tsdf_data[index + 2 * volume_size] = - 2.0 * delta_z;
    
    Dtype temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    Dtype temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    Dtype temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;
    // project to world coordinate
    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11];
    // project to image plane
    // swap y, z and -y
    Dtype xx = R_data[0] * x + R_data[3] * y + R_data[6] * z;
    Dtype zz = R_data[1] * x + R_data[4] * y + R_data[7] * z;
    Dtype yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = round(xx * K_data[0] / zz + K_data[2]) - 1;
    int iy = round(yy * K_data[4] / zz + K_data[5]) - 1;
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
      tsdf_data[index] = - 2.0 * delta_x;
      tsdf_data[index + volume_size] = - 2.0 * delta_y;
      tsdf_data[index + 2 * volume_size] = - 2.0 * delta_z;
      return;
    }
    Dtype depth = depth_data[iy * im_w + ix];
    if (depth < 0.0001) {
      tsdf_data[index] = - 2.0 * delta_x;
      tsdf_data[index + volume_size] = - 2.0 * delta_y;
      tsdf_data[index + 2 * volume_size] = - 2.0 * delta_z;
      return;
    }
    tsdf_data[index] = 2.0 * delta_x;
    tsdf_data[index + volume_size] = 2.0 * delta_y;
    tsdf_data[index + 2 * volume_size] = 2.0 * delta_z;
    // project the depth point to 3d
    Dtype tdx = (Dtype(ix + 1) - K_data[2]) * depth / K_data[0];
    Dtype tdz =  - (Dtype(iy + 1) - K_data[5]) * depth / K_data[4];
    Dtype tdy = depth;
    Dtype dx = R_data[0] * tdx + R_data[1] * tdy + R_data[2] * tdz;
    Dtype dy = R_data[3] * tdx + R_data[4] * tdy + R_data[5] * tdz;
    Dtype dz = R_data[6] * tdx + R_data[7] * tdy + R_data[8] * tdz;
    
    // distance
    Dtype tsdf_x = abs(x - dx);
    Dtype tsdf_y = abs(y - dy);
    Dtype tsdf_z = abs(z - dz);
    if (zz > depth) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
    }
    tsdf_x = max(tsdf_x, - 2.0 * delta_x);
    tsdf_y = max(tsdf_y, - 2.0 * delta_y);
    tsdf_z = max(tsdf_z, - 2.0 * delta_z);
    tsdf_x = min(tsdf_x, 2.0 * delta_x);
    tsdf_y = min(tsdf_y, 2.0 * delta_y);
    tsdf_z = min(tsdf_z, 2.0 * delta_z);

    tsdf_data[index] = tsdf_x;
    tsdf_data[index + volume_size] = tsdf_y;
    tsdf_data[index + 2 * volume_size] = tsdf_z;
  }
}

template <typename Dtype>
void WindowTSDFDataLayer<Dtype>::depth2tsdf_GPU(
    const int tsdf_size, const int im_h, const int im_w) {
  const int count = tsdf_size * tsdf_size * tsdf_size;
  const Dtype* depth_data = depth_.gpu_data();
  const Dtype* K_data = the_K_.gpu_data();
  const Dtype* R_data = the_R_.gpu_data();
  const Dtype* bb3d_data = the_bb3d_.gpu_data();
  Dtype* tsdf_data = tsdf_.mutable_gpu_data();
  depth2tsdf_depth<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, depth_data, K_data, R_data, bb3d_data,
      tsdf_size, im_h, im_w, tsdf_data);
  CUDA_POST_KERNEL_CHECK;
}

template void WindowTSDFDataLayer<float>::depth2tsdf_GPU(
    const int tsdf_size, const int im_h, const int im_w);
template void WindowTSDFDataLayer<double>::depth2tsdf_GPU(
    const int tsdf_size, const int im_h, const int im_w);

} // namespace caffe
