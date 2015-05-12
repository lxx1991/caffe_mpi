#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "../../../include/caffe/blob.hpp"

#define ind(ch, y, x) \
  (ch * map_size + y) * map_size + x

namespace caffe {

using boost::scoped_ptr;

template<typename TypeParam>
class MapAccuracyLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  MapAccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>), blob_bottom_label_(new Blob<Dtype>),
        blob_top_accuracy_(new Blob<Dtype>){ };

  virtual void SetUp() {


    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_accuracy_);


  }

  virtual ~MapAccuracyLayerTest() {};

  /**
  * Test AREA mode map generation
  */
  void TestMapAccuracy() {
    int batch_size = 10;
    int ch = 3;
    int w = 3;
    int h = 3;
    blob_bottom_data_->Reshape(batch_size, ch, h, w);
    blob_bottom_data_->Reshape(batch_size, ch, h, w);

  }
  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_accuracy_;
  std::vector<Blob<Dtype> *> blob_top_vec_;
  std::vector<Blob<Dtype> *> blob_bottom_vec_;
  int seed_;
};

TYPED_TEST_CASE(MapAccuracyLayerTest, TestDtypesAndDevices);

TYPED_TEST(MapAccuracyLayerTest, TestMapAccuracy)
{
}

} //namespace caffe