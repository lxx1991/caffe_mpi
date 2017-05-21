#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SegAccLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void SegAccLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  outer_num_ = bottom[0]->count(0, 1);
  inner_num_ = bottom[0]->count(2);

  top[0]->Reshape(1, 1, 1, 1);
  if (top.size() > 1)
    top[1]->Reshape(outer_num_, 1, 1, 1);
}

template <typename Dtype>
void SegAccLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->channels();

  vector <int> confusion_matrix(num_labels * num_labels, 0);

  top[0]->mutable_cpu_data()[0] = 0;
  for (int i = 0; i < outer_num_; ++i) {
    std::fill(confusion_matrix.begin(), confusion_matrix.end(), 0);

    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
  

      int predict = 0;
      for (int k = 1; k < num_labels; ++k) {
        if (bottom_data[i * dim + k * inner_num_ + j] > bottom_data[i * dim + predict * inner_num_ + j])
          predict = k;
      }
      confusion_matrix[label_value * num_labels + predict]++;
    }

    Dtype mIoU = 0;
    for (int j = 0; j < num_labels; ++j)
    {
      int temp = 0;
      for (int k=0; k < num_labels; k++)
        if (k!=j)
          temp+= confusion_matrix[j * num_labels + k] + confusion_matrix[k * num_labels + j];
      if (temp == 0)
        mIoU += Dtype(1);
      else
        mIoU += Dtype(confusion_matrix[j * num_labels + j]) / Dtype(temp + confusion_matrix[j * num_labels + j]);
    }
    if (top.size() > 1)
      top[1]->mutable_cpu_data()[i] = mIoU / num_labels;
    top[0]->mutable_cpu_data()[0] += mIoU / num_labels / outer_num_;
  }
}

INSTANTIATE_CLASS(SegAccLayer);
REGISTER_LAYER_CLASS(SegAcc);

}  // namespace caffe
