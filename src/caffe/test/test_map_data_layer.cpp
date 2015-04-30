#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"


namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class MapDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MapDataLayerTest()
      : blob_top_data_(new Blob <Dtype>) { };

  virtual void SetUp() {
    source_file.reset(new string("/media/ssd/test.txt"));
    blob_top_vec_.push_back(blob_top_data_);
  }

  virtual ~MapDataLayerTest() {delete  blob_top_data_;};

  /**
   * Put the data into the source file
   */
  void Fill(){
    std::ofstream outfile(source_file->c_str(), std::ofstream::out);
    outfile<<"0 256 256 1 0 0 128 128 256"<<std::endl;
    outfile<<"0 256 256 1 0 0 128 128 256"<<std::endl;
    outfile.close();
  }

  /**
   * Test reading data
   */
  void TestRead(){
    const int batch_size = 1;
    const int map_size = 10;
    const int map_ch = 4;
    LayerParameter param;
    param.set_phase(TRAIN);
    MapDataParameter* map_data_param = param.mutable_map_data_param();
    map_data_param->set_batch_size(batch_size);
    map_data_param->set_map_size(map_size);
    map_data_param->set_map_channels(map_ch);
    map_data_param->set_source_file(source_file->c_str());

    MapDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(batch_size, blob_top_data_->num());
    EXPECT_EQ(map_ch, blob_top_data_->channels());
    EXPECT_EQ(map_size, blob_top_data_->width());
    EXPECT_EQ(map_size, blob_top_data_->height());

    for (int iter = 0; iter < 10; ++iter){
      layer.Forward(blob_bottom_vec_, blob_top_vec_);

      for (int y = map_size/2; y < map_size; ++y){
        for (int x = 0; x < map_size/2+1; ++x){
          EXPECT_EQ(1, blob_top_data_->cpu_data()[(2 * map_size + y) * map_size + x]);
        }
      }
    }

  }

  shared_ptr<string> source_file;
  Blob<Dtype>* const blob_top_data_;
  std::vector<Blob<Dtype>* > blob_top_vec_;
  std::vector<Blob<Dtype>* > blob_bottom_vec_;
  int seed_;
};

TYPED_TEST_CASE(MapDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MapDataLayerTest, TestRead){
  this->Fill();
  this->TestRead();
}

}