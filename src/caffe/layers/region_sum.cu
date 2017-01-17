#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void forward_kernel(const int n, const Dtype* data1, const Dtype* data2, const Dtype* data_mask,
    const int spatial_dim, const Dtype op1_, const Dtype op2_, const int mask_cnt, Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int temp = static_cast<int>(data_mask[index % spatial_dim]);
    data[index] = data1[index] * op1_ + ((temp == -1) ? 0 : data2[(index / spatial_dim) * mask_cnt + temp] * op2_);
  }
}

template <typename Dtype>
void RegionSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data0 = bottom[0]->gpu_data();
  const Dtype* bottom_data1 = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* mask_data = bottom[2]->gpu_data();
  const int spatial_dim = top[0]->height() * top[0]->width();
  const int count = spatial_dim * top[0]->channels();
  

  for (int i=0; i<bottom[0]->num(); i++)
  {
    int mask_cnt = static_cast<int>(bottom[3]->cpu_data()[i]);
    forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data0, bottom_data1, mask_data, spatial_dim, op1_, op2_, mask_cnt,
        top_data);

    bottom_data0 += bottom[0]->offset(1);
    bottom_data1 += bottom[1]->offset(1);
    mask_data += bottom[2]->offset(1);
    top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
__global__ void backward_kernel(const int n, Dtype* data1, Dtype* data2, const Dtype* data_mask,
    const int spatial_dim, const Dtype op1_, const Dtype op2_, const int mask_cnt, const Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) {
    data1[index] = data[index] * op1_;
    const int temp = static_cast<int>(data_mask[index % spatial_dim]);
    if (temp!=-1)
      data2[(index / spatial_dim) * mask_cnt + temp] = data[index] * op2_;
  }
}



template <typename Dtype>
void RegionSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_data0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_data1 = bottom[1]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_diff();
  const Dtype* mask_data = bottom[2]->gpu_data();
  const int spatial_dim = top[0]->height() * top[0]->width();
  const int count = spatial_dim * top[0]->channels();


  for (int i=0; i<bottom[0]->num(); i++)
  {
    int mask_cnt = static_cast<int>(bottom[3]->cpu_data()[i]);
    backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data0, bottom_data1, mask_data, spatial_dim, op1_, op2_, mask_cnt,
        top_data);

    bottom_data0 += bottom[0]->offset(1);
    bottom_data1 += bottom[1]->offset(1);
    mask_data += bottom[2]->offset(1);
    top_data += top[0]->offset(1);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionSumLayer);

}  // namespace caffe
