#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > WindowTSDFDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
WindowTSDFDataLayer<Dtype>::~WindowTSDFDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void WindowTSDFDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 z1 x2 y2 z2

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_tsdf_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_tsdf_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_tsdf_data_param().fg_fraction() << std::endl
      << "  cache_images: "
      << this->layer_param_.window_tsdf_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.window_tsdf_data_param().root_folder();

  cache_images_ = this->layer_param_.window_tsdf_data_param().cache_images();
  string root_folder = this->layer_param_.window_tsdf_data_param().root_folder();

  const bool prefetch_needs_rand =
      this->layer_param_.window_tsdf_data_param().mirror() ||
      this->layer_param_.window_tsdf_data_param().crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  std::ifstream infile(this->layer_param_.window_tsdf_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_tsdf_data_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(2);
    infile >> image_size[0] >> image_size[1];
    vector<Dtype> R(9);
    infile >> R[0] >> R[1] >> R[2] >> R[3] >> R[4]
           >> R[5] >> R[6] >> R[7] >> R[8];
    vector<Dtype> K(9);
    infile >> K[0] >> K[1] >> K[2] >> K[3] >> K[4]
           >> K[5] >> K[6] >> K[7] >> K[8];
    image_database_.push_back(std::make_pair(image_path, image_size));
    R_.push_back(R);
    K_.push_back(K);

    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.window_tsdf_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.window_tsdf_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label;
      float overlap, b1, b2, b3, b4, b5, b6, b7, b8, b9;
      float c1, c2, c3, o1, o2, o3;
      infile >> label >> overlap >> b1 >> b2 >> b3 >> b4 >> b5 >> b6
          >> b7 >> b8 >> b9 >> c1 >> c2 >> c3 >> o1 >> o2 >> o3;

      vector<float> window(WindowTSDFDataLayer::NUM);
      window[WindowTSDFDataLayer::IMAGE_INDEX] = image_index;
      window[WindowTSDFDataLayer::LABEL] = label;
      window[WindowTSDFDataLayer::OVERLAP] = overlap;
      window[WindowTSDFDataLayer::B1] = b1;
      window[WindowTSDFDataLayer::B2] = b2;
      window[WindowTSDFDataLayer::B3] = b3;
      window[WindowTSDFDataLayer::B4] = b4;
      window[WindowTSDFDataLayer::B5] = b5;
      window[WindowTSDFDataLayer::B6] = b6;
      window[WindowTSDFDataLayer::B7] = b7;
      window[WindowTSDFDataLayer::B8] = b8;
      window[WindowTSDFDataLayer::B9] = b9;
      window[WindowTSDFDataLayer::C1] = c1;
      window[WindowTSDFDataLayer::C2] = c2;
      window[WindowTSDFDataLayer::C3] = c3;
      window[WindowTSDFDataLayer::O1] = o1;
      window[WindowTSDFDataLayer::O2] = o2;
      window[WindowTSDFDataLayer::O3] = o3;

      // add window to foreground list or background list
      if (overlap >= fg_threshold) {
        int label = window[WindowTSDFDataLayer::LABEL];
        //CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      } else if (overlap < bg_threshold) {
        // background window, force label and overlap to 0
        window[WindowTSDFDataLayer::LABEL] = 0;
        window[WindowTSDFDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        label_hist[0]++;
      }
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_tsdf_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_tsdf_data_param().crop_mode();

  // image
  const int tsdf_size = this->layer_param_.window_tsdf_data_param().crop_size();
  CHECK_GT(tsdf_size, 0);
  const int batch_size = this->layer_param_.window_tsdf_data_param().batch_size();
  vector<int> data_shape(5);
  data_shape[0] = batch_size; data_shape[1] = 3;
  data_shape[2] = tsdf_size; data_shape[3] = tsdf_size; data_shape[4] = tsdf_size; 
  top[0]->Reshape(data_shape);
  this->prefetch_data_.Reshape(data_shape);
  LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
      << top[0]->shape(1) << "," << top[0]->shape(2) << ","
      << top[0]->shape(3) << "," << top[0]->shape(4);
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);

  vector<int> tsdf_shape(4);
  tsdf_shape[0] = 3; tsdf_shape[1] = tsdf_size; tsdf_shape[2] = tsdf_size; tsdf_shape[3] = tsdf_size;
  tsdf_.Reshape(tsdf_shape);
  vector<int> matrix_shape(1, 9);
  the_R_.Reshape(matrix_shape);
  the_K_.Reshape(matrix_shape);
  vector<int> bb3d_shape(1, 15);
  the_bb3d_.Reshape(bb3d_shape);
}

template <typename Dtype>
unsigned int WindowTSDFDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void WindowTSDFDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.window_tsdf_data_param().scale();
  const int batch_size = this->layer_param_.window_tsdf_data_param().batch_size();
  const int context_pad = this->layer_param_.window_tsdf_data_param().context_pad();
  const int tsdf_size = this->layer_param_.window_tsdf_data_param().crop_size();
  const bool mirror = this->layer_param_.window_tsdf_data_param().mirror();
  const float fg_fraction =
      this->layer_param_.window_tsdf_data_param().fg_fraction();

  const string& crop_mode = this->layer_param_.window_tsdf_data_param().crop_mode();
  const bool classification = this->layer_param_.window_tsdf_data_param().classification();
  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      timer.Start();
      const unsigned int rand_index = PrefetchRand();
      vector<float> window;
      if (classification) {
        window = fg_windows_[rand_index % fg_windows_.size()];
      } else {
        window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];
      }

      bool do_mirror = mirror && PrefetchRand() % 2;

      // load the image containing the window
      int image_id = window[WindowTSDFDataLayer<Dtype>::IMAGE_INDEX];
      pair<std::string, vector<int> > image = image_database_[image_id];
      vector<Dtype> R = R_[image_id];
      vector<Dtype> K = K_[image_id];

      cv::Mat cv_img;
      if (this->cache_images_) {
        pair<std::string, Datum> image_cached =
          image_database_cache_[image_id];
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_ANYDEPTH);
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
        }
      }
      read_time += timer.MicroSeconds();
      timer.Start();

      // crop the bb3d to camera view
      Dtype* bb3d_data = the_bb3d_.mutable_cpu_data();
      bb3d_data[9] = window[WindowTSDFDataLayer<Dtype>::C1];
      bb3d_data[10] = window[WindowTSDFDataLayer<Dtype>::C2];
      bb3d_data[11] = window[WindowTSDFDataLayer<Dtype>::C3];
      int orientation = PrefetchRand() % 4 + 1;
      switch (orientation) {
        case 1:
            bb3d_data[0] = window[WindowTSDFDataLayer<Dtype>::B1];
            bb3d_data[1] = window[WindowTSDFDataLayer<Dtype>::B2];
            bb3d_data[2] = window[WindowTSDFDataLayer<Dtype>::B3];
            bb3d_data[3] = window[WindowTSDFDataLayer<Dtype>::B4];
            bb3d_data[4] = window[WindowTSDFDataLayer<Dtype>::B5];
            bb3d_data[5] = window[WindowTSDFDataLayer<Dtype>::B6];
            bb3d_data[6] = window[WindowTSDFDataLayer<Dtype>::B7];
            bb3d_data[7] = window[WindowTSDFDataLayer<Dtype>::B8];
            bb3d_data[8] = window[WindowTSDFDataLayer<Dtype>::B9];
            bb3d_data[12] = window[WindowTSDFDataLayer<Dtype>::O1];
            bb3d_data[13] = window[WindowTSDFDataLayer<Dtype>::O2];
            bb3d_data[14] = window[WindowTSDFDataLayer<Dtype>::O3];
            break;
        case 2:
            bb3d_data[0] = - window[WindowTSDFDataLayer<Dtype>::B4];
            bb3d_data[1] = - window[WindowTSDFDataLayer<Dtype>::B5];
            bb3d_data[2] = - window[WindowTSDFDataLayer<Dtype>::B6];
            bb3d_data[3] = window[WindowTSDFDataLayer<Dtype>::B1];
            bb3d_data[4] = window[WindowTSDFDataLayer<Dtype>::B2];
            bb3d_data[5] = window[WindowTSDFDataLayer<Dtype>::B3];
            bb3d_data[6] = window[WindowTSDFDataLayer<Dtype>::B7];
            bb3d_data[7] = window[WindowTSDFDataLayer<Dtype>::B8];
            bb3d_data[8] = window[WindowTSDFDataLayer<Dtype>::B9];
            bb3d_data[12] = window[WindowTSDFDataLayer<Dtype>::O2];
            bb3d_data[13] = window[WindowTSDFDataLayer<Dtype>::O1];
            bb3d_data[14] = window[WindowTSDFDataLayer<Dtype>::O3];
            break;
        case 3:
            bb3d_data[0] = - window[WindowTSDFDataLayer<Dtype>::B1];
            bb3d_data[1] = - window[WindowTSDFDataLayer<Dtype>::B2];
            bb3d_data[2] = - window[WindowTSDFDataLayer<Dtype>::B3];
            bb3d_data[3] = - window[WindowTSDFDataLayer<Dtype>::B4];
            bb3d_data[4] = - window[WindowTSDFDataLayer<Dtype>::B5];
            bb3d_data[5] = - window[WindowTSDFDataLayer<Dtype>::B6];
            bb3d_data[6] = window[WindowTSDFDataLayer<Dtype>::B7];
            bb3d_data[7] = window[WindowTSDFDataLayer<Dtype>::B8];
            bb3d_data[8] = window[WindowTSDFDataLayer<Dtype>::B9];
            bb3d_data[12] = window[WindowTSDFDataLayer<Dtype>::O1];
            bb3d_data[13] = window[WindowTSDFDataLayer<Dtype>::O2];
            bb3d_data[14] = window[WindowTSDFDataLayer<Dtype>::O3];
            break;
        case 4:
            bb3d_data[0] = window[WindowTSDFDataLayer<Dtype>::B4];
            bb3d_data[1] = window[WindowTSDFDataLayer<Dtype>::B5];
            bb3d_data[2] = window[WindowTSDFDataLayer<Dtype>::B6];
            bb3d_data[3] = - window[WindowTSDFDataLayer<Dtype>::B1];
            bb3d_data[4] = - window[WindowTSDFDataLayer<Dtype>::B2];
            bb3d_data[5] = - window[WindowTSDFDataLayer<Dtype>::B3];
            bb3d_data[6] = window[WindowTSDFDataLayer<Dtype>::B7];
            bb3d_data[7] = window[WindowTSDFDataLayer<Dtype>::B8];
            bb3d_data[8] = window[WindowTSDFDataLayer<Dtype>::B9];
            bb3d_data[12] = window[WindowTSDFDataLayer<Dtype>::O2];
            bb3d_data[13] = window[WindowTSDFDataLayer<Dtype>::O1];
            bb3d_data[14] = window[WindowTSDFDataLayer<Dtype>::O3];
            break;
        default:
            LOG(FATAL) << "TSDF can only embed in 4 possible directions "
                       << orientation << " should be 1,2,3,4";
      }

      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(tsdf_size) /
            static_cast<Dtype>(tsdf_size - 2*context_pad);

        // compute the expanded region
        Dtype half_width = bb3d_data[12];
        Dtype half_height = bb3d_data[13];
        Dtype half_depth = bb3d_data[14];
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
          if (half_depth > half_height) {
            half_height = half_depth;
            half_width = half_depth;
          } else {
            half_depth = half_height;
          }
        }
        bb3d_data[12] = half_width * context_scale;
        bb3d_data[13] = half_height * context_scale;
        bb3d_data[14] = half_depth * context_scale;
      }
      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_img, cv_img, 1);
      }
      // write to depth, R, K, window blob
      depth_.Reshape(1, 1, cv_img.rows, cv_img.cols);
      Dtype* depth_data = depth_.mutable_cpu_data();
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          int depth_index =  h * cv_img.cols + w;
          uint16_t depth_value =
              cv_img.at<ushort>(h,w) << 13 | cv_img.at<ushort>(h,w) >> 3;
          depth_data[depth_index] = Dtype(depth_value) / 1000.0;
        }
      }
      Dtype* R_data = the_R_.mutable_cpu_data();
      for (int i = 0; i < 9; ++i) {
        R_data[i] = R[i];
      } 
      Dtype* K_data = the_K_.mutable_cpu_data();
      for (int i = 0; i < 9; ++i) {
        K_data[i] = K[i];
      } 
      // depth to tsdf
      depth2tsdf_GPU(tsdf_size, cv_img.rows, cv_img.cols);
      // copy to top
      const Dtype* tsdf_data = tsdf_.cpu_data();
      const int datum_size = tsdf_.count();
      caffe_axpy(datum_size, scale, tsdf_data, top_data + item_id * datum_size);

      trans_time += timer.MicroSeconds();
      // get window label
      top_label[item_id] = window[WindowTSDFDataLayer<Dtype>::LABEL];

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowTSDFDataLayer<Dtype>::X1]+1 << std::endl
          << window[WindowTSDFDataLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowTSDFDataLayer<Dtype>::X2]+1 << std::endl
          << window[WindowTSDFDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[item_id] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((item_id * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      item_id++;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void WindowTSDFDataLayer<Dtype>::depth2tsdf_CPU(
    const int tsdf_size, const int im_h, const int im_w) {
  const int count = tsdf_size * tsdf_size * tsdf_size; 
  const Dtype* depth_data = depth_.cpu_data();
  const Dtype* K_data = the_K_.cpu_data();
  const Dtype* R_data = the_R_.cpu_data();
  const Dtype* bb3d_data = the_bb3d_.cpu_data();
  Dtype* tsdf_data = tsdf_.mutable_cpu_data();
  for (int index = 0; index < count; ++index) {
    Dtype x = Dtype(index % tsdf_size);
    Dtype y = Dtype((index / tsdf_size) % tsdf_size);
    Dtype z = Dtype((index / tsdf_size / tsdf_size) % tsdf_size);
    Dtype delta_x = 2 * bb3d_data[12] / Dtype(tsdf_size);
    Dtype delta_y = 2 * bb3d_data[13] / Dtype(tsdf_size);
    Dtype delta_z = 2 * bb3d_data[14] / Dtype(tsdf_size);
    tsdf_data[index] = - 2.0 * delta_x;
    tsdf_data[index + count] = - 2.0 * delta_y;
    tsdf_data[index + 2 * count] = - 2.0 * delta_z;

    Dtype temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    Dtype temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    Dtype temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;
    // project to world coordinate
    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11];
    // project to image plane
    // swap y, z and -y
    Dtype xx = R_data[0] * x + R_data[3] * y + R_data[6] * z;
    Dtype zz = R_data[1] * x + R_data[4] * y + R_data[7] * z;
    Dtype yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = round(xx * K_data[0] / zz + K_data[2]) - 1;
    int iy = round(yy * K_data[4] / zz + K_data[5]) - 1;
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.00001) {
      tsdf_data[index] = - 2.0 * delta_x;
      tsdf_data[index + count] = - 2.0 * delta_y;
      tsdf_data[index + 2 * count] = - 2.0 * delta_z;
      continue;
    }
    Dtype depth = depth_data[iy * im_w + ix];
    if (depth < 0.00001) {
      tsdf_data[index] = - 2.0 * delta_x;
      tsdf_data[index + count] = - 2.0 * delta_y;
      tsdf_data[index + 2 * count] = - 2.0 * delta_z;
      continue;
    }
    tsdf_data[index] = 2.0 * delta_x;
    tsdf_data[index + count] = 2.0 * delta_y;
    tsdf_data[index + 2 * count] = 2.0 * delta_z;
    Dtype tdx = (Dtype(ix + 1) - K_data[2]) * depth / K_data[0];
    Dtype tdz =  - (Dtype(iy + 1) - K_data[5]) * depth / K_data[4];
    Dtype tdy = depth;
    Dtype dx = R_data[0] * tdx + R_data[1] * tdy + R_data[2] * tdz;
    Dtype dy = R_data[3] * tdx + R_data[4] * tdy + R_data[5] * tdz;
    Dtype dz = R_data[6] * tdx + R_data[7] * tdy + R_data[8] * tdz;
    // distance
    tsdf_data[index] = std::fabs(x - dx);
    tsdf_data[index + count] = std::fabs(y - dy);
    tsdf_data[index + 2 * count] = std::fabs(z - dz);
    if (zz > depth) {
      tsdf_data[index] = - tsdf_data[index];
      tsdf_data[index + count] = - tsdf_data[index + count];
      tsdf_data[index + 2 * count] = - tsdf_data[index + 2 * count];
    }
    tsdf_data[index] = std::max(tsdf_data[index], - 2 * delta_x);
    tsdf_data[index + count] = std::max(tsdf_data[index + count], - 2 * delta_y);
    tsdf_data[index + 2 * count] = std::max(tsdf_data[index + 2 * count], - 2 * delta_z);
    tsdf_data[index] = std::min(tsdf_data[index], 2 * delta_x);
    tsdf_data[index + count] = std::min(tsdf_data[index + count], 2 * delta_y);
    tsdf_data[index + 2 * count] = std::min(tsdf_data[index + 2 * count], 2 * delta_z);
  }
}

INSTANTIATE_CLASS(WindowTSDFDataLayer);
REGISTER_LAYER_CLASS(WindowTSDFData);

}  // namespace caffe
