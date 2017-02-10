#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI
  this->x_norm_gpu_data_ = NULL;
  this->x_norm_cpu_data_ = NULL;

  // extract param
  this->var_eps_ = this->layer_param_.bn_param().eps();
  this->decay_ = 1 - this->layer_param_.bn_param().momentum();
  this->moving_average_ = this->layer_param_.bn_param().moving_average();
  this->axis_ = 1;
  CHECK(this->axis_ == 1 || this->axis_ == 2) << "axis_ should be 1 or 2";
  this->channels_ = bottom[0]->LegacyShape(this->axis_);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    vector<int> shape(4, 1);
    shape[this->axis_] = this->channels_;

    // fill scale with scale_filler
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().slope_filler()));
    scale_filler->Fill(this->blobs_[0].get());

    // fill shift with shift_filler
    this->blobs_[1].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().bias_filler()));
    shift_filler->Fill(this->blobs_[1].get());

    // history mean
    this->blobs_[2].reset(new Blob<Dtype>(shape));
    caffe_set(this->channels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());

    // history variance
    this->blobs_[3].reset(new Blob<Dtype>(shape));
    caffe_set(this->channels_, Dtype(0), this->blobs_[3]->mutable_cpu_data());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  blob_mean_ = new Blob<Dtype>(1, this->channels_, 1, 1);
  blob_var_ = new Blob<Dtype>(1, this->channels_, 1, 1);
  blob_scale_ = new Blob<Dtype>(1, this->channels_, 1, 1);
  blob_shift_ = new Blob<Dtype>(1, this->channels_, 1, 1);
  // runing average stats does not use weight decay and learning rate
  if (this->layer_param_.param_size() < 4){
    while (this->layer_param_.param_size() < 4){
      this->layer_param_.mutable_param()->Add();
    }
    this->layer_param_.mutable_param(2)->set_lr_mult(float(0));
    this->layer_param_.mutable_param(2)->set_decay_mult(float(0));

    this->layer_param_.mutable_param(3)->set_lr_mult(float(0));
    this->layer_param_.mutable_param(3)->set_decay_mult(float(0));
  }
#endif
}

template <typename Dtype>
void BNDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI
  // Figure out the dimensions
  this->num_ = bottom[0]->num();
  if (this->axis_ == 2)
    this->num_ *= bottom[0]->channels();
  CHECK_EQ(this->channels_, bottom[0]->LegacyShape(this->axis_));
  this->x_norm_gpu_data_ = NULL;
  this->x_norm_cpu_data_ = NULL;
  this->count_ = bottom[0]->count();
  // reshape blob
  int h = bottom[0]->LegacyShape(this->axis_+1), w = 1;
  if (this->axis_ != 2) w = bottom[0]->shape(3);
  this->hw_ = h * w;
  top[0]->ReshapeLike(*bottom[0]);
  this->x_norm_.ReshapeLike(*bottom[0]);
  this->x_std_.Reshape(1, this->channels_, 1, 1);
#endif
}

template <typename Dtype>
void BNDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void BNDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(BNDataLayer);
#endif

INSTANTIATE_CLASS(BNDataLayer);
REGISTER_LAYER_CLASS(BNData);
}  // namespace caffe
