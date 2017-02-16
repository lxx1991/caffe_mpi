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

    int th = int(Dtype(h) + dh), tw = int(Dtype(w)+dw), weight = (Dtype(h-th) + dh) * (Dtype(w-tw) + dw);
	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;

	th = int(Dtype(h) + dh + 1), tw = int(Dtype(w)+dw), weight =  (Dtype(th-h) - dh) * (Dtype(w-tw) + dw);
	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;

	th = int(Dtype(h) + dh), tw = int(Dtype(w)+dw + 1), weight =  (Dtype(h-th) + dh) * (Dtype(tw-w) - dw);
	out_data[index] += (th>=0 && th<height && tw>=0 && tw < width) ? in_data[((n * channels + c) * height + th) * width + tw] * weight : 0;

	th = int(Dtype(h) + dh + 1), tw = int(Dtype(w)+dw + 1), weight = (Dtype(th-h) - dh) * (Dtype(tw-w) - dw);
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
void  WarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


INSTANTIATE_LAYER_GPU_FUNCS(WarpingLayer);

}  // namespace caffe
