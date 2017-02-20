#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mpi_functions.hpp"

namespace caffe {

#define THREAD_BLOCK_SIZE 256

template <typename Dtype>
__global__ void mean_statistic_before_allreduce(const int num, const int map_size, const int channels,
    Dtype stat_ratio, bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* history_mean, Dtype* out, Dtype* local_mean) {
#ifdef USE_MPI
  __shared__ Dtype buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  if(!moving_mean) {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
      if(i < num * map_size)
        buffer[threadIdx.x] += in[location];
    }
    __syncthreads();
    for(int i = blockDim.x / 2; i > 0; i >>= 1) {
      if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      buffer[0] = buffer[0] * stat_ratio;
      if(save_mean) history_mean[blockIdx.x] = decay * buffer[0] + com_decay * history_mean[blockIdx.x];
    }
  }
  else if(threadIdx.x == 0)
    buffer[0] = history_mean[blockIdx.x];

  __syncthreads();

  local_mean[blockIdx.x] = buffer[0];
#endif
}

template <typename Dtype>
__global__ void mean_statistic_after_allreduce(const int num, const int map_size, const int channels,
    Dtype stat_ratio, bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* history_mean, Dtype* out, Dtype* local_mean) {
#ifdef USE_MPI
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size)
      out[location] = in[location] - local_mean[blockIdx.x];
  }
#endif
}

template <typename Dtype>
__global__ void var_statistic_before_allreduce(const int num, const int map_size, const int channels,
    Dtype in_pow, Dtype stat_ratio, Dtype stat_eps, Dtype stat_pow,
    bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* history_mean, Dtype* out,
    Dtype* x_norm,Dtype* x_std, const Dtype* scale,const Dtype* shift, Dtype* local_var) {
#ifdef USE_MPI
  __shared__ Dtype buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  if(!moving_mean) {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
      if(i < num * map_size)
        buffer[threadIdx.x] += pow(in[location],in_pow);
    }
    __syncthreads();
    for(int i = blockDim.x/2; i > 0; i >>= 1) {
      if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      buffer[0] = buffer[0] * stat_ratio;
      if(save_mean) history_mean[blockIdx.x] = decay * buffer[0] + com_decay * history_mean[blockIdx.x];
    }
  }
  else if(threadIdx.x == 0)
    buffer[0] = history_mean[blockIdx.x];

  __syncthreads();
   
  local_var[blockIdx.x] = buffer[0];
#endif
}

template <typename Dtype>
__global__ void var_statistic_after_allreduce(const int num, const int map_size, const int channels,
    Dtype in_pow, Dtype stat_ratio, Dtype stat_eps, Dtype stat_pow,
    bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* history_mean, Dtype* out,
    Dtype* x_norm,Dtype* x_std, const Dtype* scale,const Dtype* shift, Dtype* local_var)
{
#ifdef USE_MPI
  Dtype temp = pow(local_var[blockIdx.x] + stat_eps, stat_pow);
  Dtype scale_value = scale[blockIdx.x], shift_value = shift[blockIdx.x];
  if(threadIdx.x == 0) x_std[blockIdx.x] = temp;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      x_norm[location] = in[location] / temp;
      out[location] = in[location] / temp * scale_value + shift_value;
    }
  }
#endif
}


template <typename Dtype>
__global__ void var_statistic_after_allreduce(const int num, const int map_size, const int channels,
    Dtype in_pow, Dtype stat_ratio, Dtype stat_eps, Dtype stat_pow,
    bool save_mean, bool moving_mean, Dtype decay, Dtype com_decay,
    const Dtype* in, Dtype* history_mean, Dtype* out,
    Dtype* x_norm,Dtype* x_std, const Dtype* r,const Dtype* d, const Dtype* scale,const Dtype* shift, Dtype* local_var)
{
#ifdef USE_MPI
  Dtype temp = pow(local_var[blockIdx.x] + stat_eps, stat_pow);
  Dtype scale_value = scale[blockIdx.x], shift_value = shift[blockIdx.x];
  Dtype r_value = r[blockIdx.x], d_value = d[blockIdx.x];
  if(threadIdx.x == 0) x_std[blockIdx.x] = temp;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      x_norm[location] = in[location] / temp * r_value + d_value;
      out[location] = (in[location] / temp * r_value + d_value) * scale_value + shift_value;
    }
  }
#endif
}


template <typename Dtype>
void BNDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI
  int num_ = bottom[0]->num();
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();

  this->update_max_rd();

  local_mean_ = blob_mean_->mutable_gpu_data();
  local_var_ = blob_var_->mutable_gpu_data();
  
  if (this->phase_ == TEST){
    BNLayer<Dtype>::Forward_gpu(bottom, top);
  } else {
    const Dtype* const_bottom_data = bottom[0]->gpu_data();
    const Dtype* const_top_data = top[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
  
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    const Dtype* shift_data = this->blobs_[1]->gpu_data();
    bool save_mean = this->phase_ == TRAIN && this->param_propagate_down_[0];


    if (this->rebn_)
    {
      caffe_copy(channels_, this->blobs_[2]->gpu_data(), this->d_.mutable_gpu_data()); // temp buffer
      caffe_copy(channels_, this->blobs_[3]->gpu_data(), this->r_.mutable_gpu_data()); // temp buffer
    }
  
    mean_statistic_before_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_,
                     Dtype(1. / (height_ * width_ * num_)),save_mean,
                     (this->phase_ == TEST || !this->param_propagate_down_[0]) && this->moving_average_, this->decay_, Dtype(1) - this->decay_,
  		   //(!this->param_propagate_down_[0]) && this->moving_average_, this->decay_, Dtype(1) - this->decay_,
                     const_bottom_data, this->blobs_[2]->mutable_gpu_data(), top_data, local_mean_);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_scal(this->channels_, Dtype(1) / Caffe::MPI_all_rank(), local_mean_);
    CUDA_CHECK(cudaMemcpy(blob_mean_->mutable_cpu_data(), blob_mean_->mutable_gpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    mpi_force_synchronize();
    caffe_iallreduce(blob_mean_->mutable_cpu_data(), this->channels_);
    mpi_force_synchronize();

    CUDA_CHECK(cudaMemcpy(blob_mean_->mutable_gpu_data(), blob_mean_->mutable_cpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();



    if (this->rebn_)
    {
      //Dtype temp_scale = Dtype(1) / Caffe::MPI_all_rank();
      for (int i=0; i<channels_; i++)
      {
        //LOG(ERROR) << this->max_d_;
        //LOG(ERROR) << this->r_.cpu_data()[i] + this->bn_eps_;
        //LOG(ERROR) << temp_scale * blob_mean_->cpu_data()[i] - this->d_.cpu_data()[i];
        Dtype s1 = blob_mean_->cpu_data()[i] - this->d_.cpu_data()[i] , s2 = sqrt(this->r_.cpu_data()[i] + this->bn_eps_);

        if (s2 * this->max_d_ <= s1)
          this->d_.mutable_cpu_data()[i] = this->max_d_;
        else if (-s2 * this->max_d_ >= s1)
          this->d_.mutable_cpu_data()[i] = -this->max_d_;
        else
          this->d_.mutable_cpu_data()[i] = s1 / s2;
        //this->d_.mutable_cpu_data()[i] = 0;
        //LOG(ERROR) << this->d_.cpu_data()[i];

        if (Caffe::MPI_my_rank() == 0 && i == 0)
          LOG(ERROR)  << "ddddddddddddd" << ' '<< s1/s2 << ' ' << -this->max_d_ << ' ' << this->max_d_;
      }
    }

    
    mean_statistic_after_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_,
                     Dtype(1. / (height_ * width_ * num_)),save_mean,
                     (this->phase_ == TEST || !this->param_propagate_down_[0]) && this->moving_average_, this->decay_, Dtype(1) - this->decay_,
                     //(!this->param_propagate_down_[0]) && this->moving_average_, this->decay_, Dtype(1) - this->decay_,
                     const_bottom_data, this->blobs_[2]->mutable_gpu_data(), top_data, local_mean_);
    int m = num_ * height_ * width_*Caffe::MPI_all_rank();
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    var_statistic_before_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_, Dtype(2),
                     Dtype(1. / (height_ * width_ * num_)), this->bn_eps_, Dtype(0.5),
                     save_mean, (this->phase_ == TEST || !this->param_propagate_down_[0]) && this->moving_average_,
  		   //save_mean, (!this->param_propagate_down_[0]) && this->moving_average_,
                     this->decay_* bias_correction_factor, Dtype(1)-this->decay_* bias_correction_factor,
                     const_top_data, this->blobs_[3]->mutable_gpu_data(),
                     top_data,this->x_norm_.mutable_gpu_data(),this->x_std_.mutable_gpu_data(),scale_data,shift_data, local_var_);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_scal(this->channels_, Dtype(1) / Caffe::MPI_all_rank(), local_var_);
    CUDA_CHECK(cudaMemcpy(blob_var_->mutable_cpu_data(), blob_var_->mutable_gpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    mpi_force_synchronize();
    caffe_iallreduce(blob_var_->mutable_cpu_data(), this->channels_);
    mpi_force_synchronize();
    CUDA_CHECK(cudaMemcpy(blob_var_->mutable_gpu_data(), blob_var_->mutable_cpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();


    if (this->rebn_)
    {
      //Dtype temp_scale = Dtype(1) / Caffe::MPI_all_rank();
      for (int i=0; i<channels_; i++)
      {
        
        Dtype s1 = sqrt(blob_var_->cpu_data()[i] + this->bn_eps_) , s2 = sqrt(this->r_.cpu_data()[i] + this->bn_eps_);

        if (s2 * this->max_r_ <= s1)
          this->r_.mutable_cpu_data()[i] = this->max_r_;
        else if (s2 / this->max_r_ >= s1)
          this->r_.mutable_cpu_data()[i] = Dtype(1)/this->max_r_;
        else
        {
          //LOG(ERROR) << s1 << ' ' << s2 << ' ' << this->max_r_;
          this->r_.mutable_cpu_data()[i] = s1 / s2;
        }
        if (Caffe::MPI_my_rank() == 0 && i == 0)
          LOG(ERROR)  << "rrrrrrrrrrrrr" << ' '<< s1/s2 << ' ' <<  Dtype(1) / this->max_r_<< ' ' << this->max_r_;
      }
      var_statistic_after_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_, Dtype(2),
                 Dtype(1. / (height_ * width_ * num_)), this->bn_eps_, Dtype(0.5),
                 save_mean, (this->phase_ == TEST || !this->param_propagate_down_[0]) && this->moving_average_,
                 //save_mean, (!this->param_propagate_down_[0]) && this->moving_average_,
                 this->decay_* bias_correction_factor, Dtype(1)-this->decay_*bias_correction_factor,
                 const_top_data, this->blobs_[3]->mutable_gpu_data(),
                 top_data,this->x_norm_.mutable_gpu_data(),this->x_std_.mutable_gpu_data(), this->r_.gpu_data(), this->d_.gpu_data(), scale_data,shift_data, local_var_);
    }
    else
    {
      var_statistic_after_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_, Dtype(2),
                       Dtype(1. / (height_ * width_ * num_)), this->bn_eps_, Dtype(0.5),
                       save_mean, (this->phase_ == TEST || !this->param_propagate_down_[0]) && this->moving_average_,
                       //save_mean, (!this->param_propagate_down_[0]) && this->moving_average_,
                       this->decay_* bias_correction_factor, Dtype(1)-this->decay_*bias_correction_factor,
                       const_top_data, this->blobs_[3]->mutable_gpu_data(),
                       top_data,this->x_norm_.mutable_gpu_data(),this->x_std_.mutable_gpu_data(), scale_data,shift_data, local_var_);
    }
  }
#endif
}

template <typename Dtype>
__global__ void scale_shift_bottom_gradient(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, Dtype* scale_diff, Dtype* shift_diff, const Dtype* scale_data,
    const Dtype* x_std, Dtype* out, const int num_thread) {
#ifdef USE_MPI
  __shared__ Dtype buffer_scale_diff[THREAD_BLOCK_SIZE];
  __shared__ Dtype buffer_shift_diff[THREAD_BLOCK_SIZE];
  buffer_scale_diff[threadIdx.x] = 0;
  buffer_shift_diff[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size){
      buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
      buffer_shift_diff[threadIdx.x] += in[location];
    }
  }
  __syncthreads();
  for(int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(threadIdx.x < i) buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i];
    if(threadIdx.x < i) buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i];
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    scale_diff[blockIdx.x] = buffer_scale_diff[0];
    shift_diff[blockIdx.x] = buffer_shift_diff[0];
  }
  __syncthreads();
  Dtype s_data_v = scale_data[blockIdx.x], x_std_v = x_std[blockIdx.x];
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      out[location] = s_data_v * (in[location] - (x_norm[location] *
          buffer_scale_diff[0] + buffer_shift_diff[0]) / (num * map_size * num_thread)) / x_std_v;
    }
  }
#endif
}

template <typename Dtype>
__global__ void scale_shift_bottom_gradient_before_allreduce(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, Dtype* scale_diff, Dtype* shift_diff, const Dtype* scale_data,
    const Dtype* x_std, Dtype* out, const int num_thread, Dtype* local_scale, Dtype* local_shift) {
#ifdef USE_MPI
  __shared__ Dtype buffer_scale_diff[THREAD_BLOCK_SIZE];
  __shared__ Dtype buffer_shift_diff[THREAD_BLOCK_SIZE];
  buffer_scale_diff[threadIdx.x] = 0;
  buffer_shift_diff[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size){
      buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
      buffer_shift_diff[threadIdx.x] += in[location];
    }
  }
  __syncthreads();
  for(int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(threadIdx.x < i) buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i];
    if(threadIdx.x < i) buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i];
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    scale_diff[blockIdx.x] = buffer_scale_diff[0];
    shift_diff[blockIdx.x] = buffer_shift_diff[0];
    local_scale[blockIdx.x] = buffer_scale_diff[0];
    local_shift[blockIdx.x] = buffer_shift_diff[0];
  }
  __syncthreads();
#endif
}

template <typename Dtype>
__global__ void scale_shift_bottom_gradient_after_allreduce(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, Dtype* scale_diff, Dtype* shift_diff, const Dtype* scale_data,
    const Dtype* x_std, Dtype* out, const int num_thread, Dtype* local_scale, Dtype* local_shift) {
#ifdef USE_MPI
  Dtype s_data_v = scale_data[blockIdx.x], x_std_v = x_std[blockIdx.x];
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      out[location] = s_data_v * (in[location] - (x_norm[location] *
          local_scale[blockIdx.x] + local_shift[blockIdx.x]) / (num * map_size * num_thread)) / x_std_v;
    }
  }
#endif
}


template <typename Dtype>
__global__ void scale_shift_bottom_gradient_after_allreduce(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, Dtype* scale_diff, Dtype* shift_diff, const Dtype* scale_data,
    const Dtype* x_std, const Dtype* r, Dtype* out, const int num_thread, Dtype* local_scale, Dtype* local_shift) {
#ifdef USE_MPI
  Dtype s_data_v = scale_data[blockIdx.x], x_std_v = x_std[blockIdx.x], r_v = r[blockIdx.x];
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if(i < num * map_size) {
      out[location] = s_data_v * (in[location] - (x_norm[location] *
          local_scale[blockIdx.x] + local_shift[blockIdx.x]) / (num * map_size * num_thread)) / x_std_v * r_v;
    }
  }
#endif
}

template <typename Dtype>
void BNDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#ifdef USE_MPI
  int num_ = bottom[0]->num();
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();

  local_scale_ = blob_scale_->mutable_gpu_data();
  local_shift_ = blob_shift_->mutable_gpu_data();
  
  const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* const_top_diff = top[0]->gpu_diff();

  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();

  if (this->param_propagate_down_[0] && propagate_down[0]) {
    scale_shift_bottom_gradient_before_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_,
        const_top_diff, this->x_norm_.gpu_data(), scale_diff, shift_diff, scale_data, this->x_std_.gpu_data(), bottom_diff, Caffe::MPI_all_rank(), local_scale_,local_shift_);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaMemcpy(blob_scale_->mutable_cpu_data(), blob_scale_->mutable_gpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blob_shift_->mutable_cpu_data(), blob_shift_->mutable_gpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    mpi_force_synchronize();
    caffe_iallreduce(blob_scale_->mutable_cpu_data(), this->channels_);
    caffe_iallreduce(blob_shift_->mutable_cpu_data(), this->channels_);
    mpi_force_synchronize();
   
    CUDA_CHECK(cudaMemcpy(blob_scale_->mutable_gpu_data(), blob_scale_->mutable_cpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(blob_shift_->mutable_gpu_data(), blob_shift_->mutable_cpu_data(), this->channels_*sizeof(Dtype), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize(); 

    if (this->rebn_)
      scale_shift_bottom_gradient_after_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_,
          const_top_diff, this->x_norm_.gpu_data(), scale_diff, shift_diff, scale_data, this->x_std_.gpu_data(), this->r_.gpu_data(), bottom_diff, Caffe::MPI_all_rank(), local_scale_,local_shift_);
    else
      scale_shift_bottom_gradient_after_allreduce<Dtype><<<this->channels_, THREAD_BLOCK_SIZE>>>(num_, height_ * width_, channels_,
        const_top_diff, this->x_norm_.gpu_data(), scale_diff, shift_diff, scale_data, this->x_std_.gpu_data(), bottom_diff, Caffe::MPI_all_rank(), local_scale_,local_shift_);
    CUDA_POST_KERNEL_CHECK;
    // scale mean and variance
    caffe_gpu_scal(this->channels_, Dtype(1) / Caffe::MPI_all_rank(), this->blobs_[2]->mutable_gpu_data());
    caffe_gpu_scal(this->channels_, Dtype(1) / Caffe::MPI_all_rank(), this->blobs_[3]->mutable_gpu_data());

    // synchronize
    cudaDeviceSynchronize();
    mpi_force_synchronize();
    caffe_iallreduce(this->blobs_[0]->mutable_cpu_diff(), this->channels_);
    caffe_iallreduce(this->blobs_[1]->mutable_cpu_diff(), this->channels_);
    caffe_iallreduce(this->blobs_[2]->mutable_cpu_data(), this->channels_);
    caffe_iallreduce(this->blobs_[3]->mutable_cpu_data(), this->channels_);
    mpi_force_synchronize();
  }
#endif // USE_MPI
}

INSTANTIATE_LAYER_GPU_FUNCS(BNDataLayer);
}  // namespace caffe
