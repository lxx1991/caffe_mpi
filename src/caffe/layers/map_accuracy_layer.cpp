#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void MapAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  threshold_ = 0.5;
}

template <typename Dtype>
void MapAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MapAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  int count = bottom[0]->count();

  for(int i = 0; i < count; ++i){
    if (( bottom_data[i] - threshold_) * (bottom_label[i] - threshold_) > 0){
      accuracy += 1;
    }
  }

  top[0]->mutable_cpu_data()[0] = accuracy / count;
}

INSTANTIATE_CLASS(MapAccuracyLayer);
REGISTER_LAYER_CLASS(MapAccuracy);

}  // namespace caffe
