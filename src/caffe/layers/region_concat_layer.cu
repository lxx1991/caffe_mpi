#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void region_concat(const int nthreads, const Dtype* in_data, Dtype* out_data) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    out_data[index] = in_data[index];
  }
}

template <typename Dtype>
void RegionConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();

  for (int i=0; i<bottom[0]->num(); i++)
  {
    int mask_cnt = static_cast<int>(bottom[2]->cpu_data()[i]);
    int nthreads = bottom[0]->channels() * mask_cnt;
    region_concat<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[0]->gpu_data() + bottom[0]->offset(i), top_data);
    nthreads = bottom[1]->channels() * mask_cnt;
    region_concat<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[1]->gpu_data() + bottom[1]->offset(i), top_data + bottom[0]->channels() * mask_cnt);
    top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
void RegionConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();

  for (int i=0; i<bottom[0]->num(); i++)
  {
    int mask_cnt = static_cast<int>(bottom[2]->cpu_data()[i]);
    int nthreads = bottom[0]->channels() * mask_cnt;
    if (propagate_down[0])
      region_concat<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, bottom[0]->mutable_gpu_diff() + bottom[0]->offset(i));
    nthreads = bottom[1]->channels() * mask_cnt;
    if (propagate_down[1])
      region_concat<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff + bottom[0]->channels() * mask_cnt, bottom[1]->mutable_gpu_diff() + bottom[1]->offset(i));
    top_diff += top[0]->offset(1);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionConcatLayer);

}  // namespace caffe
