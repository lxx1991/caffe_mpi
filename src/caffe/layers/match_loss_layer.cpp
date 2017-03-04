#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MatchLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  
  lambda_ = this->layer_param_.match_loss_param().lambda();
  eps_ = this->layer_param_.match_loss_param().eps();
  penalize_ = this->layer_param_.match_loss_param().penalize();
}

template <typename Dtype>
void MatchLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  CHECK_EQ(bottom[0]->count(), bottom[2]->count());

}

template <typename Dtype>
void MatchLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* label0 = bottom[0]->cpu_data(), *label1 = bottom[1]->cpu_data();
  const Dtype* weight = bottom[2]->cpu_data();
  int count = 0;
  Dtype loss = 0;
  for (int i=0; i<bottom[0]->count(); i++)
  {
    const int label_value = static_cast<int>(label0[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      continue;
    }
    if (static_cast<int>(label0[i]) == static_cast<int>(label1[i]))
      loss -= weight[i];
    else
      loss += weight[i] * penalize_;
    loss -= lambda_ * std::sqrt(weight[i] * weight[i] + eps_ * eps_);
    ++count;
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void MatchLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {

    const Dtype* label0 = bottom[0]->cpu_data(), *label1 = bottom[1]->cpu_data();
    const Dtype* weight = bottom[2]->cpu_data();
    Dtype* bottom_diff = bottom[2]->mutable_cpu_diff();
    int count = 0;
    for (int i=0; i<bottom[0]->count(); i++)
    {
      const int label_value = static_cast<int>(label0[i]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }

      if (static_cast<int>(label0[i]) == static_cast<int>(label1[i]))
        bottom_diff[i] = -1;
      else
        bottom_diff[i] = penalize_;

      bottom_diff[i] += -lambda_ * weight[i] / std::sqrt(weight[i] * weight[i] + eps_ * eps_);
      ++count;
    }

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / count, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MatchLossLayer);
#endif

INSTANTIATE_CLASS(MatchLossLayer);
REGISTER_LAYER_CLASS(MatchLoss);

}  // namespace caffe
