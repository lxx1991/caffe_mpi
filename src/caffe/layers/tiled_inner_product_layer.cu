// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col_old.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

}

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

}

INSTANTIATE_CLASS(TiledInnerProductLayer);

}  // namespace caffe
