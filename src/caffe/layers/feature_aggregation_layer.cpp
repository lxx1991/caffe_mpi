#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureAggregationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  feature_num_ = bottom.size() - 1;
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void FeatureAggregationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i=1; i<bottom.size(); i++)
  {
    CHECK_EQ(bottom[i]->count(2), bottom[0]->count(2));
    CHECK_EQ(bottom[i]->num(), bottom[0]->num());
    CHECK_EQ(bottom[i]->channels(), bottom[1]->channels());
  }
  CHECK_EQ(feature_num_, bottom[0]->channels());
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void FeatureAggregationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FeatureAggregationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FeatureAggregationLayer);
#endif

INSTANTIATE_CLASS(FeatureAggregationLayer);
REGISTER_LAYER_CLASS(FeatureAggregation);

}  // namespace caffe
