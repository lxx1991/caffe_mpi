#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class WarpingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  WarpingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_flow_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  
  void SetUp() {
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    blob_bottom_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_flow_);
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_);

    blob_flow_->Reshape(num, 2, 3, 5);
    for (int i = 0; i < blob_flow_->count(); i ++) {
      blob_flow_->mutable_cpu_data()[i] = (Dtype)rand()/RAND_MAX*0.9 + 0.05 + (int)rand()%3;
    }
    LayerParameter layer_param;
    WarpingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    GradientChecker<Dtype> checker(1e-3, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);

  }

  virtual ~WarpingLayerTest() { delete blob_bottom_; delete blob_flow_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_flow_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WarpingLayerTest, TestDtypesAndGPU);


TYPED_TEST(WarpingLayerTest, Test) {
  this->SetUp();
}

}  // namespace caffe
