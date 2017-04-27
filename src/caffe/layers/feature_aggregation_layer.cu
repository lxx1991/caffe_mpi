#include <cfloat>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FeatureAggregationForward(const int n, const Dtype* in,
    const Dtype* scale, const int spatial_dim, const int chw, const int f_index, const int feature_num_,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int num = index / chw;
    const int scale_index = ((num * feature_num_) + f_index) * spatial_dim + (index % spatial_dim);
    out[index] += in[index] * scale[scale_index];
  }
}

template <typename Dtype>
void FeatureAggregationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* scale_data = bottom[0]->gpu_data();

  const int count = top[0]->count();
  const int spatial_dim = top[0]->count(2);
  const int chw = top[0]->count(3);

  caffe_gpu_set(count, Dtype(0), top_data);

  for (int i=1; i<=feature_num_; i++)
  {
    FeatureAggregationForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[i]->gpu_data(), scale_data, spatial_dim, chw, i - 1, feature_num_,
        top_data);
  }
}

template <typename Dtype>
__global__ void FeatureAggregationBackward(const int n, const Dtype* in,
    const Dtype* scale, const int spatial_dim, const int chw, const int f_index, const int feature_num_,
    const Dtype* out_diff, Dtype* in_diff, Dtype* scale_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int num = index / chw;
    const int scale_index = ((num * feature_num_) + f_index) * spatial_dim + (index % spatial_dim);

    in_diff[index] += out_diff[index] * scale[scale_index];
    scale_diff[scale_index] += out_diff[index] * in[index];
  }
}


template <typename Dtype>
void FeatureAggregationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* scale_data = bottom[0]->gpu_data();
  Dtype* scale_diff = bottom[0]->mutable_gpu_diff();

  const int count = top[0]->count();
  const int spatial_dim = top[0]->count(2);
  const int chw = top[0]->count(3);

  caffe_gpu_set(bottom[0]->count(), Dtype(0), scale_diff);

  for (int i=1; i<=feature_num_; i++)
  {
    caffe_gpu_set(count, Dtype(0), bottom[i]->mutable_gpu_diff());
    FeatureAggregationBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[i]->gpu_data(), scale_data, spatial_dim, chw, i - 1, feature_num_,
        top_diff, bottom[i]->mutable_gpu_diff(), scale_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FeatureAggregationLayer);

}  // namespace caffe
