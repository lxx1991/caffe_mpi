#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride and inputs.
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    CHECK(!(pool_param.kernel_size_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  }
  // Setup input dimensions (input_shape_)
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  channels_ = bottom[0]->shape(channel_axis_);
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  input_shape_.Reshape(bottom_dim_blob_shape);
  int* input_shape_data = input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }
  // Setup pooling kernel dimensions (kernel_shape_)
  vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (global_pooling_) {
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] = input_shape_data[i+1];
    }
  } else {
    if (pool_param.has_kernel_h() || pool_param.has_kernel_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D pooling.";
      CHECK_EQ(0, pool_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
      kernel_shape_data[0] = pool_param.kernel_h();
      kernel_shape_data[1] = pool_param.kernel_w();
    } else {
      const int num_kernel_dims = pool_param.kernel_size_size();
      CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
          << "kernel_size must be specified once, or once per spatial dimension "
          << "(kernel_size specified " << num_kernel_dims << " times; "
          << num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i]
            = pool_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
    } 
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Pooling dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (global_pooling_) {
    for (int i = 0; i < num_spatial_axes_; ++i) { // global pooling: 1 stride
      stride_data[i] = 1;
    }
  } else {
    if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "stride_h & stride_w can only be used for 2D pooling.";
      CHECK_EQ(0, pool_param.stride_size())
          << "Either stride or stride_h/w should be specified; not both.";
      stride_data[0] = pool_param.stride_h();
      stride_data[1] = pool_param.stride_w();
    } else {
      const int num_stride_dims = pool_param.stride_size();
      CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
            num_stride_dims == num_spatial_axes_)
          << "stride must be specified once, or once per spatial dimension "
          << "(stride specified " << num_stride_dims << " times; "
          << num_spatial_axes_ << " spatial dims);";
      const int kDefaultStride = 1;
      for (int i = 0; i < num_spatial_axes_; ++i) {
        stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
            pool_param.stride((num_stride_dims == 1) ? 0 : i);
      }
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (global_pooling_) {
    for (int i = 0; i < num_spatial_axes_; ++i) { // global pooling: 0 pad
      pad_data[i] = 0;
    }
  } else {
    if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "pad_h & pad_w can only be used for 2D pooling.";
      CHECK_EQ(0, pool_param.pad_size())
          << "Either pad or pad_h/w should be specified; not both.";
      pad_data[0] = pool_param.pad_h();
      pad_data[1] = pool_param.pad_w();
    } else {
      const int num_pad_dims = pool_param.pad_size();
      CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
            num_pad_dims == num_spatial_axes_)
          << "pad must be specified once, or once per spatial dimension "
          << "(pad specified " << num_pad_dims << " times; "
          << num_spatial_axes_ << " spatial dims);";
      const int kDefaultPad = 0;
      for (int i = 0; i < num_spatial_axes_; ++i) {
        pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
            pool_param.pad((num_pad_dims == 1) ? 0 : i);
      }
    }
  }
  // check padding
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (pad_data[i] != 0) {
      CHECK(pool_param.pool()
          == PoolingParameter_PoolMethod_AVE
          || pool_param.pool()
          == PoolingParameter_PoolMethod_MAX)
          << "Padding implemented only for average and max pooling.";
      CHECK_LT(pad_data[i], kernel_shape_data[i]);
    }
  }
  // Setup output dimension (output_shape_).
  output_shape_.Reshape(spatial_dim_blob_shape);
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  num_ = bottom[0]->count(0, channel_axis_);
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // shape the tops
  compute_output_shape();
  const int* output_shape_data = this->output_shape_.cpu_data();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(first_spatial_axis); // Discard input spatial axes.
  for (int i = 0 ; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_data[i]);
  }
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (pool_param.pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(top_shape);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (pool_param.pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(top_shape);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::compute_output_shape() {
  // input_shape_ +1 to skip channel axis
  const int* input_shape_data = this->input_shape_.cpu_data() + 1;
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  int* output_shape_data = this->output_shape_.mutable_cpu_data();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    const int input_dim = input_shape_data[i];
    int output_dim = static_cast<int>(ceil(static_cast<float>
        (input_dim + 2 * pad_data[i] - kernel_shape_data[i]) / stride_data[i])) + 1;
    if (pad_data[i]) {
      // If we have padding, ensure that the last pooling starts strictly
      // inside the image (instead of at the padding); otherwise clip the last.
      if ((output_dim - 1) * stride_data[i] >= input_dim + pad_data[i]) {
        --output_dim;
      }
      CHECK_LT((output_dim - 1) * stride_data[i], input_dim + pad_data[i]);
    }
    output_shape_data[i] = output_dim; 
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  int kernel_dim = 1;
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    kernel_dim *= kernel_shape_data[i];
  }
  int output_dim = 1;
  const int* output_shape_data = this->output_shape_.cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    output_dim *= output_shape_data[i];
  }
  const int* input_shape_data = this->input_shape_.cpu_data() + 1;
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();

  vector<int> start_pos(num_spatial_axes_, 0);
  vector<int> end_pos(num_spatial_axes_, 0);
  vector<int> pooled_pos(num_spatial_axes_, 0);
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int p = 0; p < output_dim; ++p) {
          const int pool_index = p;
          // idx2sub for output.
          int offset = p;
          for (int i = num_spatial_axes_ - 1; i >=0; --i) {
            if (i < num_spatial_axes_ - 1) {
              offset /= output_shape_data[i + 1];
            }
            pooled_pos[i] = offset % output_shape_data[i];
          }
          // start and end for input.
          int region_dim = 1;
          for (int i = 0; i < num_spatial_axes_; ++i) {
            start_pos[i] = pooled_pos[i] * stride_data[i] - pad_data[i];
            end_pos[i] = min(start_pos[i] + kernel_shape_data[i],
                input_shape_data[i]);
            start_pos[i] = max(start_pos[i], 0);
            CHECK_GE(end_pos[i], start_pos[i])
                << "pooling region should be positive";
            region_dim *= end_pos[i] - start_pos[i];
          }
          for (int d = 0; d < region_dim; ++d) {
            // idx2sub for local region
            vector<int> local_pos(num_spatial_axes_, 0);
            int local_offset = d;
            for (int i = num_spatial_axes_ - 1; i >= 0; --i) {
              if (i < num_spatial_axes_ - 1) {
                local_offset /= (end_pos[i + 1] - start_pos[i + 1]);
              }
              local_pos[i] = local_offset % (end_pos[i] - start_pos[i]);
            }
            // sub2indx for input
            int index = start_pos[0] + local_pos[0];
            for (int i = 1; i < num_spatial_axes_; ++i) {
              index *= input_shape_data[i];
              index += start_pos[i] + local_pos[i];
            }
            if (bottom_data[index] > top_data[pool_index]) {
              top_data[pool_index] = bottom_data[index];
              if (use_top_mask) {
                top_mask[pool_index] = static_cast<Dtype>(index);
              } else {
                mask[pool_index] = index;
              }
            }
          } 
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int p = 0; p < output_dim; ++p) {
          const int pool_index = p;
          // idx2sub for output.
          int offset = p;
          for (int i = num_spatial_axes_ - 1; i >=0; --i) {
            if (i < num_spatial_axes_ - 1) {
              offset /= output_shape_data[i + 1];
            }
            pooled_pos[i] = offset % output_shape_data[i];
          }
          // start and end for input.
          int region_dim = 1;
          for (int i = 0; i < num_spatial_axes_; ++i) {
            start_pos[i] = pooled_pos[i] * stride_data[i] - pad_data[i];
            end_pos[i] = min(start_pos[i] + kernel_shape_data[i],
                input_shape_data[i]);
            start_pos[i] = max(start_pos[i], 0);
            CHECK_GE(end_pos[i], start_pos[i])
                << "pooling region should be positive";
            region_dim *= end_pos[i] - start_pos[i];
          }
          for (int d = 0; d < region_dim; ++d) {
            // idx2sub for local region
            vector<int> local_pos(num_spatial_axes_, 0);
            int local_offset = d;
            for (int i = num_spatial_axes_ - 1; i >= 0; --i) {
              if (i < num_spatial_axes_ - 1) {
                local_offset /= (end_pos[i + 1] - start_pos[i + 1]);
              }
              local_pos[i] = local_offset % (end_pos[i] - start_pos[i]);
            }
            // sub2indx for input
            int index = start_pos[0] + local_pos[0];
            for (int i = 1; i < num_spatial_axes_; ++i) {
              index *= input_shape_data[i];
              index += start_pos[i] + local_pos[i];
            }
            top_data[pool_index] += bottom_data[index];
          } 
          top_data[pool_index] /= region_dim;
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  int kernel_dim = 1;
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    kernel_dim *= kernel_shape_data[i];
  }
  int output_dim = 1;
  const int* output_shape_data = this->output_shape_.cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    output_dim *= output_shape_data[i];
  }
  const int* input_shape_data = this->input_shape_.cpu_data() + 1;
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();

  vector<int> start_pos(num_spatial_axes_, 0);
  vector<int> end_pos(num_spatial_axes_, 0);
  vector<int> pooled_pos(num_spatial_axes_, 0);
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int p = 0; p < output_dim; ++p) {
          const int index = p;
          const int bottom_index = use_top_mask ? top_mask[index] : mask[index];
          bottom_diff[bottom_index] += top_diff[index];
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int p = 0; p < output_dim; ++p) {
          const int index = p;
          // idx2sub for output
          int offset = p;
          for (int i = num_spatial_axes_ - 1; i >= 0; --i) {
            if (i < num_spatial_axes_ - 1) {
              offset /= output_shape_data[i + 1];
            }
            pooled_pos[i] = offset % output_shape_data[i];
          }
          // start and end for input.
          int region_dim = 1;
          for (int i = 0; i < num_spatial_axes_; ++i) {
            start_pos[i] = pooled_pos[i] * stride_data[i] - pad_data[i];
            end_pos[i] = min(start_pos[i] + kernel_shape_data[i],
                input_shape_data[i]);
            start_pos[i] = max(start_pos[i], 0);
            CHECK_GE(end_pos[i], start_pos[i])
                << "pooling region should be positive";
            region_dim *= end_pos[i] - start_pos[i];
          }
          for (int d = 0; d < region_dim; ++d) {
            // idx2sub for local region
            vector<int> local_pos(num_spatial_axes_, 0);
            int local_offset = d;
            for (int i = num_spatial_axes_ - 1; i >= 0; --i) {
              if (i < num_spatial_axes_ - 1) {
                local_offset /= (end_pos[i+1] - start_pos[i+1]);
              }
              local_pos[i] = local_offset % (end_pos[i] - start_pos[i]);
            }
            // sub2idx for input
            int bottom_index = start_pos[0] + local_pos[0];
            for (int i = 1; i < num_spatial_axes_; ++i) {
              bottom_index *= input_shape_data[i];
              bottom_index += start_pos[i] + local_pos[i];
            }
            bottom_diff[bottom_index] += top_diff[index] / region_dim;
          }
        } 
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
