#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void warping_forward(const int nthreads, const Dtype* in_data, const Dtype* flow_data, const int channels, const int height, const int width, Dtype* out_data) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const Dtype dh = flow_data[((n * 2 + 0) * height + h) * width + w];
    const Dtype dw = flow_data[((n * 2 + 1) * height + h) * width + w];
    out_data[index] = 0;

    int th = int(Dtype(h) + dh), tw = int(Dtype(w)+dw);
    Dtype weight = (1 - (Dtype(h-th) + dh)) * (1 - (Dtype(w-tw) + dw));
  	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;

  	th = int(Dtype(h) + dh + 1), tw = int(Dtype(w)+dw), weight =  (1 - (Dtype(th-h) - dh)) * (1 - (Dtype(w-tw) + dw));
  	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;

  	th = int(Dtype(h) + dh), tw = int(Dtype(w)+dw + 1), weight =  (1 - (Dtype(h-th) + dh)) * (1 - (Dtype(tw-w) - dw));
  	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;

  	th = int(Dtype(h) + dh + 1), tw = int(Dtype(w)+dw + 1), weight = (1 - (Dtype(th-h) - dh)) * (1 - (Dtype(tw-w) - dw));
  	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;
  }
}






template <typename Dtype>
void WarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data =  bottom[0]->gpu_data();
  const Dtype* flow_data = bottom[1]->gpu_data();
  int nthreads = top[0]->count();

  warping_forward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, bottom_data, flow_data, channels_, height_, width_, top_data);
}



template <typename Dtype>
__global__ void warping_backward_data(const int nthreads, const Dtype* in_data, const Dtype* head, const Dtype* edge, const int channels, const int height, const int width, Dtype* out_data) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    out_data[index] = 0;
    for (int i = head[(n * height + h) * width + w]; i!= -1; i = edge[i])
    	out_data[index] += edge[i+3] * in_data[((n *channels + c)*height + static_cast<int>(edge[i+1])) * width + static_cast<int>(edge[i+2])];
  }
}

template <typename Dtype>
__global__ void warping_backward_flow(const int nthreads, const Dtype* in_data, const Dtype* bottom_data, const Dtype* flow_data, const int channels, const int height, const int width, const int spatial_dim, Dtype* out_data) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;

    out_data[index] = 0;
    out_data[index + spatial_dim] = 0;

    const Dtype dh = flow_data[index];
    const Dtype dw = flow_data[index + spatial_dim];

   const Dtype* top_diff = in_data + n * channels * spatial_dim + index;

    for (int c=0; c<channels; c++)
    {
    	int th = int(Dtype(h) + dh), tw = int(Dtype(w)+dw);
    	if (th>=0 && th<height && tw>=0 && tw < width)
    	{
    		out_data[index] -= (1 - (Dtype(w-tw) + dw)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    		out_data[index + spatial_dim] -= (1 - (Dtype(h-th) + dh)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    	}
  		th = int(Dtype(h) + dh + 1), tw = int(Dtype(w)+dw);
  		if (th>=0 && th<height && tw>=0 && tw < width)
    	{
    		out_data[index] += (1 - (Dtype(w-tw) + dw)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    		out_data[index + spatial_dim] -= (1 - (Dtype(th-h) - dh)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    	}
  		th = int(Dtype(h) + dh), tw = int(Dtype(w)+dw + 1);
  		if (th>=0 && th<height && tw>=0 && tw < width)
    	{
    		out_data[index] -= (1 - (Dtype(tw-w) - dw)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    		out_data[index + spatial_dim] += (1 - (Dtype(h-th) + dh)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    	}
  		th = int(Dtype(h) + dh + 1), tw = int(Dtype(w)+dw + 1);
  		if (th>=0 && th<height && tw>=0 && tw < width)
    	{
    		out_data[index] += (1 - (Dtype(tw-w) - dw)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    		out_data[index + spatial_dim] += (1 - (Dtype(th-h) - dh)) * bottom_data[((n * channels + c) * height + th) * width + tw] * (*top_diff);
    	}
    	top_diff += spatial_dim;
    }
  }
}

template <typename Dtype>
void  WarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* flow_data = bottom[1]->gpu_data();
  const Dtype* flow_data_cpu = bottom[1]->cpu_data();
  LOG(ERROR) << "start";

  Dtype* edge = edge_.mutable_cpu_data();
  edge_cnt_ = 0;
  Dtype* head = head_.mutable_cpu_data();
  caffe_set(head_.count(), Dtype(-1), head);

  //build the edge map
  for (int n=0; n<bottom[1]->num(); n++)
  	for (int h=0; h<bottom[1]->height(); h++)
  		for (int w=0; w<bottom[1]->width(); w++)
  		{
  			Dtype dh = flow_data_cpu[((n * 2 + 0) * height_ + h) * width_ + w];
  			Dtype dw = flow_data_cpu[((n * 2 + 1) * height_ + h) * width_ + w];
  			for (int i=0; i<2; i++)
  				for (int j=0; j<2; j++)
  				{
  					int th = int(Dtype(h) + dh + i), tw = int(Dtype(w) + dw + j);
  					Dtype weight = (1 - abs(Dtype(h + dh - th))) * (1 - abs(Dtype(w+ dw - tw)));

  					if (th>=0 && th<height_ && tw>=0 && tw < width_)
  					{
  						int offset = (n * height_ + th) * width_ + tw;
  						edge[edge_cnt_] = head[offset];
  						head[offset] = static_cast<Dtype>(edge_cnt_);
  						edge[edge_cnt_ + 1] = static_cast<Dtype>(h);
  						edge[edge_cnt_ + 2] = static_cast<Dtype>(w);
  						edge[edge_cnt_ + 3] = weight;
  						edge_cnt_ += 4;
  					}
  				}
  		}

  LOG(ERROR) << "edge";
  if (propagate_down[0])
  {
	  int nthreads = bottom[0]->count();
	  warping_backward_data<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	    nthreads, top_diff, head_.gpu_data(), edge_.gpu_data(), channels_, height_, width_, bottom[0]->mutable_gpu_diff());
  }
  LOG(ERROR) << "bp1";
  if (propagate_down[1])
  {
	  int nthreads = bottom[1]->num() * spatial_dim_;
	  warping_backward_flow<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	    nthreads, top_diff, bottom[0]->gpu_data(), bottom[1]->gpu_data(), channels_, height_, width_, spatial_dim_, bottom[1]->mutable_gpu_diff());	
  }
  LOG(ERROR) << "bp2";
}


INSTANTIATE_LAYER_GPU_FUNCS(WarpingLayer);

}  // namespace caffe
