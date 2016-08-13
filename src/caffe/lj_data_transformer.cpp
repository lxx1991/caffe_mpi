#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include "caffe/lj_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
LJDataTransformer<Dtype>::LJDataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  CHECK(param_.short_edge_size() == 0 || param_.short_edge_size() == 2) <<
      "If short_edge_size specified, min and max are needed.";
  CHECK(param_.short_edge_size() == 0 ||
        (param_.short_edge_size() == 2 && param_.crop_size() > 0
         && param_.short_edge(0) >= param_.crop_size())) <<
      "If short_edge_size specified, crop_size must be specified.";
  if (param_.eigs_size() > 0) {
      const int channels = param_.eigs_channels();
      CHECK(param_.eigs_size() == (channels * channels));
      gaussian_rng_.Reshape(channels, 1, 1, 1);
      intensity_alter_.Reshape(channels, 1, 1, 1);
  }
}

template<typename Dtype>
void LJDataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data,
                     const bool is_sib, const int short_edge,
                     const bool mirror, const int sib_num, const int sib_ind) {
  // Process short edge
  cv::Mat cv_img = DatumToCVMat(datum);
  cv::Mat cv_img_resized = cv_img;
  if (param_.short_edge_size() == 2 || (is_sib && short_edge > 0)) {
      int local_short_edge = short_edge;
      if (!short_edge) {
        const int short_edge_min = param_.short_edge(0);
        const int short_edge_range = param_.short_edge(1) - param_.short_edge(0);
        CHECK_GE(short_edge_range, 0) << "Short edge range must be greater than or equal to 0";
        local_short_edge = short_edge_range == 0 ? short_edge_min : short_edge_min + Rand(short_edge_range + 1);
      }
      float scale = (1.0 * local_short_edge) / std::min(cv_img.rows, cv_img.cols);
      int height = static_cast<int>(scale * cv_img.rows);
      int width = static_cast<int>(scale * cv_img.cols);
      cv::resize(cv_img, cv_img_resized, cv::Size(width, height));
  }

  const int crop_size = param_.crop_size();
  const int img_channels = cv_img_resized.channels();
  const int img_height = cv_img_resized.rows;
  const int img_width = cv_img_resized.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  const Dtype scale = param_.scale();
  const bool do_mirror = is_sib ? mirror : param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Dtype* mean = NULL;
  if (has_mean_file) {
      CHECK_EQ(img_channels, data_mean_.channels());
      CHECK_EQ(img_height, data_mean_.height());
      CHECK_EQ(img_width, data_mean_.width());
      mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
      CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
          "Specify either 1 mean_value or as many as channels: " << img_channels;
      if (img_channels > 1 && mean_values_.size() == 1) {
          // Replicate the mean_value for simplicity
          for (int c = 1; c < img_channels; ++c) {
              mean_values_.push_back(mean_values_[0]);
          }
      }
  }

    // Process intensity altering
  const bool has_eigs = TRAIN == phase_ && param_.eigs_size() > 0;
  if (has_eigs) {
      CHECK(param_.eigs_channels() == img_channels);
      caffe_rng_gaussian<Dtype>(gaussian_rng_.count(), Dtype(0), Dtype(0.1), gaussian_rng_.mutable_cpu_data());
      Dtype* intensity_alter = intensity_alter_.mutable_cpu_data();
      const Dtype* gaussian_rng = gaussian_rng_.cpu_data();
      int eig_index = 0;
      for (int c = 0; c < intensity_alter_.num(); ++c) {
          intensity_alter[c] = 0;
          for (int cc = 0; cc < intensity_alter_.num(); ++cc)
            intensity_alter[c] += gaussian_rng[cc] * param_.eigs(eig_index++);
      }
  }

  int height = cv_img_resized.rows;
  int width = cv_img_resized.cols;

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_resized;
  if (is_sib) {
      bool is_short_height = (height == std::min(height, width));
      int crop_range = is_short_height ? img_width : img_height;
      CHECK_GT(crop_size, 0) << "crop_size should be set when use sib.";
      CHECK_GE(sib_num, 1);
      crop_range = static_cast<int>(1.0 * (crop_range - crop_size + 1) / sib_num);
      if (phase_ == TRAIN || param_.rand_crop_test()) {
          h_off = is_short_height ? Rand(img_height - crop_size + 1) : Rand(crop_range) + crop_range * sib_ind;
          w_off = (!is_short_height) ? Rand(img_width - crop_size + 1) : Rand(crop_range) + crop_range * sib_ind;
      } else {
          h_off = is_short_height ? (img_height - crop_size) / 2 : crop_range / 2 + crop_range * sib_ind;
          w_off = (!is_short_height) ? (img_width - crop_size) / 2 : crop_range / 2 + crop_range * sib_ind;
      }
      height = crop_size;
      width = crop_size;
  } else if (crop_size) {
      // We only do random crop when we do training
      // or rand_crop_test is true
      if (phase_ == TRAIN || param_.rand_crop_test()) {
          h_off = Rand(img_height - crop_size + 1);
          w_off = Rand(img_width - crop_size + 1);
      } else {
          h_off = (img_height - crop_size) / 2;
          w_off = (img_width - crop_size) / 2;
      }
      cv::Rect roi(w_off, h_off, crop_size, crop_size);
      cv_cropped_img = cv_img_resized(roi);
      height = crop_size;
      width = crop_size;
  }

  CHECK(cv_cropped_img.data);

  const Dtype* intensity_alter = has_eigs ? intensity_alter_.cpu_data() : NULL;
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_eigs) {
            pixel += intensity_alter[c];
        }
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
            if (has_mean_values) {
              transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
            } else {
              transformed_data[top_index] = pixel * scale;
            }
          }
        }
      }
    }
}


template<typename Dtype>
void LJDataTransformer<Dtype>::Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                     const bool is_sib, const int short_edge,
                     const bool mirror, const int sib_num, const int sib_ind) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob, is_sib, short_edge, mirror, sib_num, sib_ind);
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_GE(num, 1);

  if (is_sib) {
    CHECK_GT(crop_size, 0);
    CHECK_EQ(crop_size, short_edge);
  } else if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data, is_sib, short_edge, mirror, sib_num, sib_ind);
}

template<typename Dtype>
void LJDataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector, Blob<Dtype>* transformed_blob,
                     const bool is_sib, const int short_edge,
                     const bool mirror, const int sib_num, const int sib_ind) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob, is_sib, short_edge, mirror, sib_num, sib_ind);
  }
}

template<typename Dtype>
void LJDataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector, Blob<Dtype>* transformed_blob,
                     const bool is_sib, const int short_edge,
                     const bool mirror, const int sib_num, const int sib_ind) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob, is_sib, short_edge, mirror, sib_num, sib_ind);
  }
}

template<typename Dtype>
void LJDataTransformer<Dtype>::Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
                     const bool is_sib, const int short_edge,
                     const bool mirror, const int sib_num, const int sib_ind) {
  // Process short edge
  cv::Mat cv_img_resized = cv_img;
  if (param_.short_edge_size() == 2 || (is_sib && short_edge > 0)) {
      int local_short_edge = short_edge;
      if (!short_edge) {
        const int short_edge_min = param_.short_edge(0);
        const int short_edge_range = param_.short_edge(1) - param_.short_edge(0);
        CHECK_GE(short_edge_range, 0) << "Short edge range must be greater than or equal to 0";
        local_short_edge = short_edge_range == 0 ? short_edge_min : short_edge_min + Rand(short_edge_range + 1);
      }
      float scale = (1.0 * local_short_edge) / std::min(cv_img.rows, cv_img.cols);
      int height = static_cast<int>(scale * cv_img.rows);
      int width = static_cast<int>(scale * cv_img.cols);
      cv::resize(cv_img, cv_img_resized, cv::Size(width, height));
  }

  const int crop_size = param_.crop_size();
  const int img_channels = cv_img_resized.channels();
  const int img_height = cv_img_resized.rows;
  const int img_width = cv_img_resized.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = is_sib ? mirror : param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  // Process intensity altering
  const bool has_eigs = TRAIN == phase_ && param_.eigs_size() > 0;
  if (has_eigs) {
      CHECK(param_.eigs_channels() == img_channels);
      caffe_rng_gaussian<Dtype>(gaussian_rng_.count(), Dtype(0), Dtype(0.1), gaussian_rng_.mutable_cpu_data());
      Dtype* intensity_alter = intensity_alter_.mutable_cpu_data();
      const Dtype* gaussian_rng = gaussian_rng_.cpu_data();
      int eig_index = 0;
      for (int c = 0; c < intensity_alter_.num(); ++c) {
          intensity_alter[c] = 0;
          for (int cc = 0; cc < intensity_alter_.num(); ++cc)
            intensity_alter[c] += gaussian_rng[cc] * param_.eigs(eig_index++);
      }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_resized;
  if (is_sib) {
    bool is_short_height = (height == std::min(height, width));
    int crop_range = is_short_height ? img_width : img_height;
    CHECK_GT(crop_size, 0) << "crop_size should be set when use sib.";
    CHECK_GE(sib_num, 1);
    crop_range = static_cast<int>(1.0 * (crop_range - crop_size + 1) / sib_num);
    if (phase_ == TRAIN || param_.rand_crop_test()) {
      h_off = is_short_height ? Rand(img_height - crop_size + 1) : Rand(crop_range) + crop_range * sib_ind;
      w_off = (!is_short_height) ? Rand(img_width - crop_size + 1) : Rand(crop_range) + crop_range * sib_ind;
    } else {
      h_off = is_short_height ? (img_height - crop_size) / 2 : crop_range / 2 + crop_range * sib_ind;
      w_off = (!is_short_height) ? (img_width - crop_size) / 2 : crop_range / 2 + crop_range * sib_ind;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img_resized(roi);
  }else if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training
    // or rand_crop_test is true
    if (phase_ == TRAIN || param_.rand_crop_test()) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img_resized(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  const Dtype* intensity_alter = has_eigs ? intensity_alter_.cpu_data() : NULL;
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_eigs) {
            pixel += intensity_alter[c];
        }
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void LJDataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob,
                     const bool is_sib, const int short_edge,
                     const bool mirror, const int sib_num, const int sib_ind) {
  if (param_.short_edge_size() > 0)
    LOG(FATAL) << "Short edge crop is not supported with Blob input.";
  if (is_sib)
    LOG(FATAL) << "Sib is not supported with Blob input.";
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  // Process intensity altering
  const bool has_eigs = TRAIN == phase_ && param_.eigs_size() > 0;
  if (has_eigs) {
      CHECK(param_.eigs_channels() == input_channels);
      caffe_rng_gaussian<Dtype>(gaussian_rng_.count(), Dtype(0), Dtype(0.1), gaussian_rng_.mutable_cpu_data());
      Dtype* intensity_alter = intensity_alter_.mutable_cpu_data();
      const Dtype* gaussian_rng = gaussian_rng_.cpu_data();
      int eig_index = 0;
      for (int c = 0; c < intensity_alter_.num(); ++c) {
          intensity_alter[c] = 0;
          for (int cc = 0; cc < intensity_alter_.num(); ++cc)
            intensity_alter[c] += gaussian_rng[cc] * param_.eigs(eig_index++);
      }
      for (int n = 0; n < input_num; ++n) {
          for (int c = 0; c < input_channels; ++c) {
              int offset = input_blob->offset(n, c);
              caffe_add_scalar(input_height * input_width, intensity_alter[c],
                  input_data + offset);
          }
      }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> LJDataTransformer<Dtype>::InferBlobShape(const Datum& datum, const bool is_sib) {
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img, is_sib);
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.short_edge_size() == 0 && !is_sib) {
    CHECK_GE(datum_height, crop_size);
    CHECK_GE(datum_width, crop_size);
  }

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> LJDataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector, const bool is_sib) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0], is_sib);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

template<typename Dtype>
vector<int> LJDataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img, const bool is_sib) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  if (param_.short_edge_size() == 0 && !is_sib) {
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);
  }
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> LJDataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector, const bool is_sib) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0], is_sib);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

template <typename Dtype>
void LJDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size()) ||
      param_.rand_crop_test();
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int LJDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(LJDataTransformer);

}  // namespace caffe
