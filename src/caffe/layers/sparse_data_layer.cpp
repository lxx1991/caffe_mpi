#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "../../../include/caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
SparseDataLayer<Dtype>::~SparseDataLayer() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void SparseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  //load parameters
  CHECK_GE(this->layer_param_.sparse_data_param().batch_size(), 0) << "batch size must be larger than 0";
  CHECK_GE(this->layer_param_.sparse_data_param().mat_channels(), 0) << "channel number must be larger than 0";
  CHECK_GE(this->layer_param_.sparse_data_param().mat_height(), 0) << "height must be larger than 0";
  CHECK_GE(this->layer_param_.sparse_data_param().mat_width(), 0) << "width must be larger than 0";

  ch_ = this->layer_param_.sparse_data_param().batch_size();
  width_ = this->layer_param_.sparse_data_param().mat_width();
  height_ = this->layer_param_.sparse_data_param().mat_height();
  batch_size_ = this->layer_param_.sparse_data_param().batch_size();

  if (this->layer_param_.sparse_data_param().has_source_file()){
    mfs_.open(this->layer_param_.sparse_data_param().source_file().c_str());
    if (mfs_.fail()){
      LOG(FATAL)<<"Cannot open file "<<this->layer_param_.sparse_data_param().source_file();
    }
  }else {
    LOG(ERROR)<<"No source file found, set the field `source_file` in net proto";
  }

  //load the input file

  int batch_size = this->layer_param_.map_data_param().batch_size();
  top[0]->Reshape(batch_size_, ch_, height_, width_);
  this->prefetch_data_.Reshape(batch_size_, ch_, height_, width_);
}

template <typename Dtype>
unsigned int SparseDataLayer<Dtype>::PrefetchRand() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}


template <typename Dtype>
void decodePOSEncoding(string& content, Dtype* data, int ch, int width, int height){
  const int pos_len = 4 - (ch == 1) - (width == 1) - (height == 1);

  boost::tokenizer<> tok(content);
  vector<Dtype> numbers;

  //ignore the first item, which is the image id
  for (boost::tokenizer<>::iterator it = ++tok.begin();
       it != tok.end(); ++it){
    Dtype num = atof(it->c_str());
    numbers.push_back(num);
  }
  CHECK_EQ(numbers.size() % (pos_len + 1), 0)<<"Incorrect number of integers in line: "<<content;
  const int n_pos = numbers.size() / (pos_len + 1);
  int idx = 0;
  for (int i = 0; i < n_pos; ++i){
    //check for non-singular dimensions
    int c  = (ch == 1)?0:numbers[idx++];
    int h  = (height == 1)?0:numbers[idx++];
    int w  = (width == 1)?0:numbers[idx++];

    //get value
    Dtype value = numbers[idx++];

    //set value to the matrix
    data[(c * height + h) * width + w] = value;
  }
}

template <typename Dtype>
void SparseDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();

  const int batch_size = this->layer_param_.map_data_param().batch_size();

  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  string buffer;
  for (int item_id = 0; item_id < batch_size; ++item_id){
    std::getline(mfs_, buffer);
    //Do decoding
    switch (this->layer_param().sparse_data_param().encode_type()){
      case SparseDataParameter_EncodingType_POS:{
        decodePOSEncoding(buffer, top_data + this->prefetch_data_.offset(item_id), ch_, width_, height_);
        break;
      };
      case SparseDataParameter_EncodingType_RLE:
        NOT_IMPLEMENTED;
        break;
    }
    //rewind if necessary
    if (mfs_.eof()){
      mfs_.seekg(0);
    }
  }

  //log the time consumption
  batch_timer.Stop();
  DLOG(INFO) <<"Prefetch batch: "<<batch_timer.MilliSeconds() <<" ms.";

}

INSTANTIATE_CLASS(SparseDataLayer);
REGISTER_LAYER_CLASS(SparseData);

}