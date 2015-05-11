#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, int num_spatial_axes>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels,
    const int* input_shape, const int* output_shape, const int* kernel_shape,
    const int* stride_shape, const int* pad_shape,
    Dtype* top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pooled_pos[num_spatial_axes];
    int start_pos[num_spatial_axes];
    int end_pos[num_spatial_axes];
    // idx2sub for output.
    int offset = index;
    for (int i = num_spatial_axes - 1; i >= 0; --i) {
      if (i < num_spatial_axes - 1) {
        offset /= output_shape[i + 1];
      }
      pooled_pos[i] = offset % output_shape[i];
    }
    offset /= output_shape[0];
    int c = offset % channels;
    offset /= channels;
    int n = offset % num;
    // start and end for input
    int region_dim = 1;
    int bottom_spatial_dim = 1;
    for (int i = 0; i < num_spatial_axes; ++i) {
      start_pos[i] = pooled_pos[i] * stride_shape[i] - pad_shape[i];
      end_pos[i] = min(start_pos[i] + kernel_shape[i], input_shape[i]);
      start_pos[i] = max(start_pos[i], 0);
      region_dim *= end_pos[i] - start_pos[i];
      bottom_spatial_dim *= input_shape[i];
    }
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    int local_pos[num_spatial_axes];
    bottom_data += (n * channels + c) * bottom_spatial_dim;
    for (int d = 0; d < region_dim; ++d) {
      // idx2sub for local region
      int local_offset = d;
      for (int i = num_spatial_axes - 1; i >= 0; --i) {
        if (i < num_spatial_axes - 1) {
          local_offset /= (end_pos[i + 1] - start_pos[i + 1]);
        }
        local_pos[i] = local_offset % (end_pos[i] - start_pos[i]);
      }
      // sub2idx for input
      int bottom_index = start_pos[0] + local_pos[0];
      for (int i = 1; i < num_spatial_axes; ++i) {
        bottom_index *= input_shape[i];
        bottom_index += start_pos[i] + local_pos[i];
      } 
      if (bottom_data[bottom_index] > maxval) {
        maxidx = bottom_index;
        maxval = bottom_data[maxidx];
      } 
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype, int num_spatial_axes>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, 
    const int* input_shape, const int* output_shape, const int* kernel_shape,
    const int* stride_shape, const int* pad_shape, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pooled_pos[num_spatial_axes];
    int start_pos[num_spatial_axes];
    int end_pos[num_spatial_axes];
    // idx2sub for output.
    int offset = index;
    for (int i = num_spatial_axes - 1; i >= 0; --i) {
      if( i < num_spatial_axes - 1) {
        offset /= output_shape[i + 1];
      }
      pooled_pos[i] = offset % output_shape[i];
    }
    offset /= output_shape[0];
    int c = offset % channels;
    offset /= channels;
    int n = offset % num;
    // start and end for input.
    int region_dim = 1;
    int bottom_spatial_dim = 1;
    for (int i = 0; i < num_spatial_axes; ++i) {
      start_pos[i] = pooled_pos[i] * stride_shape[i] - pad_shape[i];
      end_pos[i] = min(start_pos[i] + kernel_shape[i], input_shape[i]);
      start_pos[i] = max(start_pos[i], 0);
      region_dim *= end_pos[i] - start_pos[i];
      bottom_spatial_dim *= input_shape[i];
    }
    Dtype aveval = 0;
    int local_pos[num_spatial_axes];
    bottom_data += (n * channels + c) * bottom_spatial_dim;
    for (int d = 0; d < region_dim; ++d) {
      // idx2sub for local region
      int local_offset = d;
      for (int i = num_spatial_axes - 1; i >= 0; --i) {
        if (i < num_spatial_axes - 1) {
          local_offset /= (end_pos[i + 1] - start_pos[i + 1]);
        }
        local_pos[i] = local_offset % (end_pos[i] - start_pos[i]);
      }
      // sub2idx for input
      int bottom_index = start_pos[0] + local_pos[0];
      for (int i = 1; i < num_spatial_axes; ++i) {
        bottom_index *= input_shape[i];
        bottom_index += start_pos[i] + local_pos[i];
      }
      aveval += bottom_data[bottom_index];
    }
    top_data[index] = aveval / region_dim;
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  size_t count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  const int* input_shape_data = input_shape_.gpu_data() + 1;
  const int* output_shape_data = output_shape_.gpu_data();
  const int* kernel_shape_data = kernel_shape_.gpu_data();
  const int* stride_data = stride_.gpu_data();
  const int* pad_data = pad_.gpu_data();

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    switch (num_spatial_axes_) {
      case 1:
        MaxPoolForward<Dtype, 1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 2:
        MaxPoolForward<Dtype, 2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 3:
        MaxPoolForward<Dtype, 3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 4:
        MaxPoolForward<Dtype, 4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 5:
        MaxPoolForward<Dtype, 5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 6:
        MaxPoolForward<Dtype, 6><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 7:
        MaxPoolForward<Dtype, 7><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 8:
        MaxPoolForward<Dtype, 8><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 9:
        MaxPoolForward<Dtype, 9><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      case 10:
        MaxPoolForward<Dtype, 10><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data, 
            kernel_shape_data, stride_data, pad_data,
            top_data, mask, top_mask);
      break;
      default:
        LOG(FATAL) << "MaxPoolForward does not support computation with "
                   << num_spatial_axes_ << " spatial axes";
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    switch (num_spatial_axes_) {
      case 1:
        AvePoolForward<Dtype, 1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 2:
        AvePoolForward<Dtype, 2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 3:
        AvePoolForward<Dtype, 3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 4:
        AvePoolForward<Dtype, 4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 5:
        AvePoolForward<Dtype, 5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 6:
        AvePoolForward<Dtype, 6><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 7:
        AvePoolForward<Dtype, 7><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 8:
        AvePoolForward<Dtype, 8><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 9:
        AvePoolForward<Dtype, 9><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      case 10:
        AvePoolForward<Dtype, 10><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, top_data);
      break;
      default:
        LOG(FATAL) << "AvePoolForward does not support computation with "
                   << num_spatial_axes_ << " spatial axes";
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype, int num_spatial_axes>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int* input_shape, const int* output_shape, const int* kernel_shape,
    const int* stride_shape, const int* pad_shape, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int bottom_pos[num_spatial_axes];
    int top_start_pos[num_spatial_axes];
    int top_end_pos[num_spatial_axes];
    int offset = index;
    for (int i = num_spatial_axes - 1; i >= 0; --i) {
      if (i < num_spatial_axes - 1) {
        offset /= input_shape[i + 1];
      }
      bottom_pos[i] = offset % input_shape[i];
    }
    offset /= input_shape[0];
    int c = offset % channels;
    offset /= channels;
    int n = offset % num;
    int bottom_index = bottom_pos[0];
    for (int i = 1; i < num_spatial_axes; ++i) {
      bottom_index *= input_shape[i];
      bottom_index += bottom_pos[i];
    }
    int region_dim = 1;
    int top_spatial_dim = 1;
    for (int i = 0; i < num_spatial_axes; ++i) {
      top_start_pos[i] = (bottom_pos[i] + pad_shape[i] < kernel_shape[i]) ?
          0 : (bottom_pos[i] + pad_shape[i]- kernel_shape[i]) / stride_shape[i] + 1;
      top_end_pos[i] = min((bottom_pos[i] + pad_shape[i]) / stride_shape[i] + 1, output_shape[i]);
      region_dim *= top_end_pos[i] - top_start_pos[i];
      top_spatial_dim *= output_shape[i];
    }
    offset = (n * channels + c) * top_spatial_dim;
    Dtype gradient = 0;
    int local_pos[num_spatial_axes];
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int d = 0; d < region_dim; ++d) {
        // idx2sub for local region
        int local_offset = d;
        for (int i = num_spatial_axes - 1; i >= 0; --i) {
          if (i < num_spatial_axes - 1) {
            local_offset /= (top_end_pos[i+1] - top_start_pos[i+1]);
          }
          local_pos[i] = local_offset % (top_end_pos[i] - top_start_pos[i]);
        }
        // sub2idx for top
        int top_index = top_start_pos[0] + local_pos[0];
        for (int i = 1; i < num_spatial_axes; ++i) {
          top_index *= output_shape[i];
          top_index += top_start_pos[i] + local_pos[i];
        }
        if (mask[top_index] == bottom_index) {
          gradient += top_diff[top_index];
        }
      }
    } else {
      top_mask += offset;
      for (int d = 0; d < region_dim; ++d) {
        // idx2sub for local region
        int local_offset = d;
        for (int i = num_spatial_axes - 1; i >= 0; --i) {
          if (i < num_spatial_axes - 1) {
            local_offset /= (top_end_pos[i+1] - top_start_pos[i+1]);
          }
          local_pos[i] = local_offset % (top_end_pos[i] - top_start_pos[i]);
        }
        // sub2idx for top
        int top_index = top_start_pos[0] + local_pos[0];
        for (int i = 1; i < num_spatial_axes; ++i) {
          top_index *= output_shape[i];
          top_index += top_start_pos[i] + local_pos[i];
        }
        if (top_mask[top_index] == bottom_index) {
          gradient += top_diff[top_index];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype, int num_spatial_axes>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels,
    const int* input_shape, const int* output_shape, const int* kernel_shape,
    const int* stride_shape, const int* pad_shape, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int bottom_pos[num_spatial_axes];
    int top_start_pos[num_spatial_axes];
    int top_end_pos[num_spatial_axes];
    int offset = index;
    for (int i = num_spatial_axes - 1; i >= 0; --i) {
      if (i < num_spatial_axes - 1) {
        offset /= input_shape[i + 1];
      }
      bottom_pos[i] = offset % input_shape[i] + pad_shape[i];
    }
    offset /= input_shape[0];
    int c = offset % channels;
    offset /= channels;
    int n = offset % num;
    int region_dim = 1;
    int top_spatial_dim = 1;
    for (int i = 0; i < num_spatial_axes; ++i) {
      top_start_pos[i] = (bottom_pos[i] < kernel_shape[i]) ?
          0 : (bottom_pos[i] - kernel_shape[i]) / stride_shape[i] + 1;
      top_end_pos[i] = min(bottom_pos[i] / stride_shape[i] + 1, output_shape[i]);
      region_dim *= top_end_pos[i] - top_start_pos[i];
      top_spatial_dim *= output_shape[i];
    }
    Dtype gradient = 0;
    top_diff += (n * channels + c) * top_spatial_dim;
    int local_pos[num_spatial_axes];
    for (int d = 0; d < region_dim; ++d) {
      // idx2sub for local region
      int local_offset = d;
      for (int i = num_spatial_axes - 1; i >= 0; --i) {
        if (i < num_spatial_axes - 1) {
          local_offset /= (top_end_pos[i + 1] - top_start_pos[i + 1]);
        }
        local_pos[i] = local_offset % (top_end_pos[i] - top_start_pos[i]);
      }
      // sub2idx for top and figure out the pooling size
      int top_index = top_start_pos[0] + local_pos[0];
      int pool_size = 1;
      int temp_start, temp_end;
      for (int i = 1; i < num_spatial_axes; ++i) {
        top_index *= output_shape[i];
        top_index += top_start_pos[i] + local_pos[i];
        temp_start = (top_start_pos[i] + local_pos[i]) * stride_shape[i] - pad_shape[i];
        temp_end = min(temp_start + kernel_shape[i], input_shape[i] + pad_shape[i]);
        pool_size *= temp_end - temp_start;
      }
      gradient += top_diff[top_index] / pool_size;
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const size_t count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  const int* input_shape_data = input_shape_.gpu_data() + 1;
  const int* output_shape_data = output_shape_.gpu_data();
  const int* kernel_shape_data = kernel_shape_.gpu_data();
  const int* stride_data = stride_.gpu_data();
  const int* pad_data = pad_.gpu_data();

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    switch (num_spatial_axes_) {
      case 1:
        MaxPoolBackward<Dtype, 1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 2:
        MaxPoolBackward<Dtype, 2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 3:
        MaxPoolBackward<Dtype, 3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 4:
        MaxPoolBackward<Dtype, 4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 5:
        MaxPoolBackward<Dtype, 5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 6:
        MaxPoolBackward<Dtype, 6><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 7:
        MaxPoolBackward<Dtype, 7><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 8:
        MaxPoolBackward<Dtype, 8><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 9:
        MaxPoolBackward<Dtype, 10><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 10:
        MaxPoolBackward<Dtype, 10><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      default:
         LOG(FATAL) << "MaxPoolBackward does not support computation with "
                   << num_spatial_axes_ << " spatial axes";
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    switch (num_spatial_axes_) {
      case 1:
        AvePoolBackward<Dtype, 1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 2:
        AvePoolBackward<Dtype, 2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 3:
        AvePoolBackward<Dtype, 3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 4:
        AvePoolBackward<Dtype, 4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 5:
        AvePoolBackward<Dtype, 5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 6:
        AvePoolBackward<Dtype, 6><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 7:
        AvePoolBackward<Dtype, 7><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 8:
        AvePoolBackward<Dtype, 8><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 9:
        AvePoolBackward<Dtype, 9><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      case 10:
        AvePoolBackward<Dtype, 10><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            input_shape_data, output_shape_data,
            kernel_shape_data, stride_data, pad_data, bottom_diff);
      break;
      default:
         LOG(FATAL) << "MaxPoolBackward does not support computation with "
                   << num_spatial_axes_ << " spatial axes";
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
