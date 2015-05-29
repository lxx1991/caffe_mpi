#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/loss_layers.hpp>

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

/**
 * Load bbox data into memory for future use
 */
void readTextBBoxDB(vector<vector<int> > &db, std::ifstream &infile){
  string line;
  vector<string> tokens;
  while (std::getline(infile, line)){
    boost::tokenizer<> tok(line);
    vector<int> line_numbers;
    for (boost::tokenizer<>::iterator it = tok.begin();
         it != tok.end(); ++it){
      int num = atoi(it->c_str());
      line_numbers.push_back(num);
    }

    CHECK_EQ(0, (line_numbers.size() - 3) % 6)<<"line length must be 8*n + 1";
    //make sure that every line has a same number of integers
    db.push_back(line_numbers);
  }
}

/**build the map using AREA mode, meaning all pixel surrounded by the bounxing box will be put to 1*/
template <typename Dtype>
inline void buildAreaMap(int map_ch, int map_width, int map_height,
                         vector<int>& bbox_info, Dtype* data,
                         bool bg_channel, bool prob){
  int n_box = (bbox_info.size() - 3) / 6;
  int image_width = bbox_info[1];
  int image_height = bbox_info[2];
  caffe_set(map_width*map_height*map_ch, Dtype(0), data);
  for (int b = 0; b < n_box; ++b) {
    int box_head = 3 + 6 * b;
    int map_start_x = std::min((int) (Dtype(bbox_info[box_head + 2] - 1) / image_width * map_width), map_width);
    int map_start_y = std::min((int) (Dtype(bbox_info[box_head + 3] - 1) / image_height * map_height), map_height);
    int map_end_x = std::min((int) (Dtype(bbox_info[box_head + 4] - 1) / image_width * map_width), map_width);
    int map_end_y = std::min((int) (Dtype(bbox_info[box_head + 5] - 1) / image_width * map_height), map_height);
    int map_working_channel = bbox_info[box_head] * (int) sqrt(map_ch) + bbox_info[box_head + 1];

    CHECK_LT(map_working_channel, map_ch) << "channel index must be less than number of channel in the map";

    Dtype *start_ptr = data + map_working_channel * (map_width * map_height);

    for (int y = map_start_y; y < map_end_y; ++y){
      for (int x = map_start_x; x < map_end_x; ++x){
        start_ptr[y * map_width + x] =1;
      }
    }

  }

  //assing background label to bg pixels
  if (bg_channel) {
    Dtype *bg_channel_ptr = data + (map_ch - 1) * (map_width * map_height);
    int channel_offset = map_width * map_height;
    for (int y = 0; y < map_height; ++y) {
      for (int x = 0; x < map_width; ++x) {
        int offset = y * map_width + x;
        Dtype sum = 0;
        for (int c = 0; c < map_ch-1; ++c) sum += data[offset + channel_offset * c];
        if (sum == 0) bg_channel_ptr[offset] += 1; //modified to reflect multiple bounding boxes
      }
    }
  }

  if(prob){
    int channel_offset = map_width * map_height;
    for (int y = 0; y < map_height; ++y){
      for (int x = 0; x < map_width; ++x){
        int offset = y * map_width + x;
        Dtype sum = 0;
        for (int c = 0; c < map_ch; ++c) sum += data[offset + channel_offset * c];
        sum = std::max(Dtype(kLOG_THRESHOLD), sum);
        for (int c = 0; c <map_ch; ++c) data[offset + channel_offset * c]/= sum;
      }
    }
  }


//  //debug dump
//  std::ofstream of("/media/ssd/dump.txt", std::ofstream::out);
//
//  for(int i = 0; i < bbox_info.size(); ++i){
//    of << bbox_info[i] <<" ";
//  }
//  of << std::endl;
//  of << map_start_x <<" "<<map_start_y<<" "<<map_end_x<<" "<<map_end_y<<std::endl;
//
//  for (int ch = 0; ch < map_ch; ++ch){
//    for ( int y = 0; y < map_height; ++y){
//      for(int x = 0; x < map_width; ++x){
//        of<<data[(ch * map_height + y)*map_width + x]<<" ";
//      }
//      of<<std::endl;
//    }
//    of <<std::endl;
//  }
}

/**build the map using CENTER mode, meaning only the center pixel of the bounding box will be set to 1*/
template <typename Dtype>
inline void buildCenterMap(int map_ch, int map_width, int map_height, vector<int>& bbox_info, Dtype* data){
  CHECK_EQ(9, bbox_info.size()) << "Input must have 9 number per line";
  int bbox_center_x = Dtype((bbox_info[5] + bbox_info[7])/2)/bbox_info[1] * map_width;
  int bbox_center_y = Dtype((bbox_info[6] + bbox_info[8])/2)/bbox_info[2] * map_height;
  int map_working_channel = bbox_info[3] * (int) sqrt(map_ch) + bbox_info[4];

  CHECK_LT(map_working_channel, map_ch)<<"channel index must lest than number of channel in the map";

  data[(map_working_channel*map_height + bbox_center_y)*map_width + bbox_center_x] = ((bbox_center_x > 0) &&(bbox_center_y > 0))?1:0;
}

/*!
* Build bounding box map and write to the blob
* For current implementation every vector should be in follwing schema
* \code{.unparsed}
* [0]image_index [1]image_width [2]image_height  [3]bbox_channel_h [4]bbox_channel_w [5]bbox_start_x [6]bbox_start_y [7]bbox_end_x [8]bbox_end_y
* \endcode
*/
template <typename Dtype>
inline void buildBBoxMap(int map_ch, int map_width, int map_height,
                  vector<int>& bbox_info, Dtype* data, MapDataParameter_MapMode mode,
                         bool bg_channel, bool prob){
  switch (mode) {
    case MapDataParameter_MapMode_AREA: {
      buildAreaMap(map_ch, map_width, map_height, bbox_info, data,
                   bg_channel, prob);
      break;
    }
    case MapDataParameter_MapMode_CENTER: {
      buildCenterMap(map_ch, map_width, map_height, bbox_info, data);
      break;
    };
  }

//  //debug dump
//  std::ofstream of("/media/ssd/dump.txt", std::ofstream::out);
//  for (int ch = 0; ch < map_ch; ++ch){
//    for ( int y = 0; y < map_height; ++y){
//      for(int x = 0; x < map_width; ++x){
//        of<<data[(ch * map_height + y)*map_width + x]<<" ";
//      }
//      of<<std::endl;
//    }
//    of <<std::endl;
//  }

}

template <typename Dtype>
void smoothImage(int map_ch, int map_width, int map_height,
                 int kernel_radius, Dtype sigma,
                 Dtype* data);

template<>
void smoothImage<float>(int map_ch,int map_width, int map_height,
                        int kernel_radius, float sigma,
                        float* data){
  int channel_step = map_height * map_width;
  int ksize = 2 * kernel_radius + 1;
  int range = kernel_radius * kernel_radius;
  //build filter kernel
  cv::Mat smoothing_kernel(ksize, ksize, CV_32F);
  for (int x = 0; x <= kernel_radius; ++x){
    for (int y = 0; y <= kernel_radius; ++y){
      float distsqr = std::pow(x - kernel_radius, 2) + std::pow(y - kernel_radius, 2);

      float v = (distsqr <= range)?std::max(0., std::exp(-0.5 * distsqr / std::pow(sigma, 2))):float(0.);
      smoothing_kernel.at<float>(y, x) = v;
      smoothing_kernel.at<float>(ksize - 1 - y, ksize - 1 - x) = v;
      smoothing_kernel.at<float>(ksize - 1 - y, x) = v;
      smoothing_kernel.at<float>(y, ksize - 1 - x) = v;
    }
  }

  DLOG(INFO)<<" kernel "<<smoothing_kernel<<std::endl;

  //do channel-wise convolution
  for (int ch = 0; ch < map_ch; ch++){
    cv::Mat m(map_height, map_width, CV_32F, data + ch * channel_step);
    cv::filter2D(m, m, CV_32F, smoothing_kernel, cvPoint(-1, -1), 0, cv::BORDER_CONSTANT);
  }
}

template<>
void smoothImage<double>(int map_ch, int map_width, int map_height,
                         int kernel_radius, double sigma,
                         double* data){
  int channel_step = map_height * map_width;
  int ksize = 2 * kernel_radius + 1;
  int range = kernel_radius * kernel_radius;

  //build filter kernel
  cv::Mat smoothing_kernel(ksize, ksize, CV_64F);
  for (int x = 0; x <= kernel_radius; ++x){
    for (int y = 0; y <= kernel_radius; ++y){
      double distsqr = std::pow(x - kernel_radius, 2) + std::pow(y - kernel_radius, 2);

      double v = (distsqr <= range)?std::max(0., std::exp(-0.5 * distsqr / std::pow(sigma, 2))):float(0.);
      smoothing_kernel.at<double>(y, x) = v;
      smoothing_kernel.at<double>(ksize - 1 - y, ksize - 1 - x) = v;
      smoothing_kernel.at<double>(ksize - 1 - y, x) = v;
      smoothing_kernel.at<double>(y, ksize - 1 - x) = v;
    }
  }

  //do channel-wise convolution
  for (int ch = 0; ch < map_ch; ch++){
    cv::Mat m(map_height, map_width, CV_64F, data + ch * channel_step);
    cv::filter2D(m, m, CV_64F, smoothing_kernel, cvPoint(-1, -1), 0, cv::BORDER_CONSTANT);
  }
}

template <typename Dtype>
void smoothTensor(int map_ch, int map_width, int map_height,
                 vector<int>& bbox_info, Dtype sigma,
                 Dtype* data){
  int channel_step = map_height * map_width;
  int pos_channel = bbox_info[4];
  int pos_x = Dtype((bbox_info[5] + bbox_info[7])/2)/bbox_info[1] * map_width;
  int pos_y = Dtype((bbox_info[6] + bbox_info[8])/2)/bbox_info[2] * map_height;

  if ((pos_x == 0) && (pos_y == 0)){
    return;
  }
  for (int ch = 0; ch < map_ch; ++ch){
    for (int y = 0; y < map_height; ++y){
      for (int x = 0; x < map_width; ++x){
        Dtype distance = std::pow(2*(ch - pos_channel), 2) + std::pow(y - pos_y, 2) + std::pow(x - pos_x, 2);
        Dtype v = std::exp(Dtype(-0.5) * distance / std::pow(sigma, 2));
        data[ch * channel_step + y * map_width + x] = (v>1e-5)?v: Dtype(0);
      }
    }
  }

  //debug dump
//    std::ofstream of("/media/ssd/dump.txt", std::ofstream::out);
//  for (int ch = 0; ch < map_ch; ++ch){
//    for ( int y = 0; y < map_height; ++y){
//      for(int x = 0; x < map_width; ++x){
//        of<<data[(ch * map_height + y)*map_width + x]<<" ";
//      }
//      of<<std::endl;
//    }
//    of <<std::endl;
//  }
}


/*!
* Smooth the bounding box map according to given smoothing operation.
* For current implementation we support following smoothing types:
* -# No Smooth.
* -# Gaussian smoothing accross channels.
*    this will treat the input map as a tensor and do ball-like gaussian smoothing.
* -# Guassian smoothing inside the map channel.
*    this will smooth the map only inside each channel, just like general image smoothing.
*/
template <typename Dtype>
inline void smoothBBoxMap(int map_ch, int map_width, int map_height, Dtype sigma,
                         vector<int>& bbox_info, Dtype* data, MapDataParameter_SmoothType mode) {
  switch (mode) {
    case MapDataParameter_SmoothType_NO_SMOOTH: {
      //This optional does nothing on the map
      break;
    }
    case MapDataParameter_SmoothType_GAUSSIAN: {
      smoothTensor<Dtype>(map_ch, map_width, map_height, bbox_info, sigma, data);
      break;
    };
    case MapDataParameter_SmoothType_GAUSSIAN_IN_CHANNEL:{
      smoothImage<Dtype>(map_ch, map_width, map_height, (map_width + map_height)/4, sigma, data);

      break;
    };
  }

  //debug dump
//  std::ofstream of("/media/ssd/dump.txt", std::ofstream::out);
//  for (int ch = 0; ch < map_ch; ++ch){
//    for ( int y = 0; y < map_height; ++y){
//      for(int x = 0; x < map_width; ++x){
//        of<<data[(ch * map_height + y)*map_width + x]<<" ";
//      }
//      of<<std::endl;
//    }
//    of <<std::endl;
//  }
}


template <typename Dtype>
MapDataLayer<Dtype>::~MapDataLayer() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  //load parameters
  CHECK_GE(this->layer_param_.map_data_param().batch_size(), 0) << "batch size for maps must be larger than 0";
  if (this->layer_param_.map_data_param().has_map_size_h()
      && this->layer_param_.map_data_param().has_map_size_w()){
    //requiring a rectangular map
    map_width_ = this->layer_param_.map_data_param().map_size_w();
    map_height_ = this->layer_param_.map_data_param().map_size_h();
  }else{
    //Using a square map
    map_width_ = map_height_ = this->layer_param_.map_data_param().map_size();
  }

  bg_channel_ = this->layer_param_.map_data_param().background_channel();

  map_ch_ = this->layer_param_.map_data_param().map_channels();

  if (bg_channel_) ++map_ch_;

  sigma_ = this->layer_param_.map_data_param().sigma();

  if (this->layer_param_.map_data_param().has_source_file()){
    db_ = TEXT_FILE;
  }else if (this->layer_param_.map_data_param().has_db()){
    db_ = HDD_DB;
  }else{
    LOG(ERROR)<<"No database found, use either `source file` or `db` to specify a "
        <<"data source";
  }

  //load the input file
  switch (db_){
    case TEXT_FILE:{
      std::ifstream infile(this->layer_param_.map_data_param().source_file().c_str());
      CHECK(infile.good()) << "Failed to open"
        << this->layer_param_.map_data_param().source_file().c_str() << std::endl;
      readTextBBoxDB(text_file_database_, infile);
      LOG(INFO)<<"Number of bbox "<<text_file_database_.size();

      //set cursor to first
      text_file_cursor_ = text_file_database_.begin();
      break;
    };
    case HDD_DB:{
      NOT_IMPLEMENTED;
      break;
    };
  }
  int batch_size = this->layer_param_.map_data_param().batch_size();
  top[0]->Reshape(batch_size, map_ch_, map_height_, map_width_);
  this->prefetch_data_.Reshape(batch_size, map_ch_, map_height_, map_width_);
}

template <typename Dtype>
unsigned int MapDataLayer<Dtype>::PrefetchRand() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void MapDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();

  const int batch_size = this->layer_param_.map_data_param().batch_size();

  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  for (int item_id = 0; item_id < batch_size; ++item_id){
    switch (db_) {
      case (TEXT_FILE): {
        Dtype *item_data = top_data + this->prefetch_data_.offset(item_id);
        buildBBoxMap<Dtype>(map_ch_, map_width_, map_height_, *text_file_cursor_, item_data,
                            this->layer_param_.map_data_param().map_mode(),
                            bg_channel_, this->layer_param_.map_data_param().probability());
        //perform smoothing operation
        smoothBBoxMap<Dtype>(map_ch_, map_width_, map_height_, sigma_, *text_file_cursor_, item_data,
                      this->layer_param_.map_data_param().smooth_type());
        //go to the next item and rewind if reaches the end
        text_file_cursor_++;
        if(text_file_cursor_ == text_file_database_.end()){
          text_file_cursor_ = text_file_database_.begin();
        }
        break;
      }
      case (HDD_DB):{
        NOT_IMPLEMENTED;
        break;
      }
    }
  }

  //log the time consumption
  batch_timer.Stop();
  DLOG(INFO) <<"Prefetch batch: "<<batch_timer.MilliSeconds() <<" ms.";

}

INSTANTIATE_CLASS(MapDataLayer);
REGISTER_LAYER_CLASS(MapData);

}
