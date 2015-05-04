#include <cmath>
#include <cstdlib>
#include <cstring>
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
class MapRegressionLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
    MapRegressionLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_max(1);
    filler_param.set_min(0);
    UniformFiller<Dtype> target_filler(filler_param);
    filler_param.set_max(2);
    filler_param.set_min(-2);
    UniformFiller<Dtype> pred_filler(filler_param);

    pred_filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    target_filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~MapRegressionLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestEuclideanForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    // First test the euclidean loss type, the default one
    LayerParameter layer_param;
    layer_param.mutable_map_regression_param()->set_loss_mode(MapRegressionParameter_LossMode_EUCLIDEAN);
    MapRegressionLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    MapRegressionLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }


  void TestSoftmaxForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    // First test the euclidean loss type, the default one
    LayerParameter layer_param;
    layer_param.mutable_map_regression_param()->set_loss_mode(MapRegressionParameter_LossMode_SOFTMAX);
    MapRegressionLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
            layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    MapRegressionLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
            layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  void TestHingeForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    // First test the euclidean loss type, the default one
    LayerParameter layer_param;
    layer_param.mutable_map_regression_param()->set_loss_mode(MapRegressionParameter_LossMode_HINGE);
    layer_param.mutable_map_regression_param()->set_beta(0.3);
    layer_param.mutable_map_regression_param()->set_alpha(10);
    MapRegressionLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
            layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    MapRegressionLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
            layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(loss_weight_1, kNonTrivialAbsThresh);
  }


  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MapRegressionLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MapRegressionLossLayerTest, TestEuclideanForward) {
  this->TestEuclideanForward();
}

TYPED_TEST(MapRegressionLossLayerTest, TestSoftmaxForward) {
    this->TestSoftmaxForward();
}

TYPED_TEST(MapRegressionLossLayerTest, TestHingeForward) {
    this->TestHingeForward();
}

TYPED_TEST(MapRegressionLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  MapRegressionLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MapRegressionLossLayerTest, TestSoftmaxGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_map_regression_param()->set_loss_mode(MapRegressionParameter_LossMode_SOFTMAX);
    MapRegressionLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_, 0);
}

TYPED_TEST(MapRegressionLossLayerTest, TestHingeGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_map_regression_param()->set_loss_mode(MapRegressionParameter_LossMode_HINGE);
    layer_param.mutable_map_regression_param()->set_beta(0.3);
    layer_param.mutable_map_regression_param()->set_alpha(4);
    layer_param.mutable_map_regression_param()->set_tau_plus(0.5);
    layer_param.mutable_map_regression_param()->set_tau_minus(0.4);
    MapRegressionLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_, 0);
}
}  // namespace caffe
