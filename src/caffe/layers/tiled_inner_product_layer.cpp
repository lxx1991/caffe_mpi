// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col_old.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	split_h_ = this->layer_param_.tiled_inner_product_param().split_h();
	split_w_ = this->layer_param_.tiled_inner_product_param().split_w();

	//calculate number of divided regions
	region_num_ = split_h_ * split_w_;

	num_output_ = this->layer_param_.inner_product_param().num_output() * region_num_;

	//get overlap
	overlap_h_ = this->layer_param_.tiled_inner_product_param().overlap_h();
	overlap_w_ = this->layer_param_.tiled_inner_product_param().overlap_w();

	//get bottom info
	int bottom_width, bottom_height, bottom_channels;
	bottom_channels = bottom[0]->channels();
	bottom_height = bottom[0]->height();
	bottom_width = bottom[0]->width();

	num_ = bottom[0]->num();
	height_ = bottom_height;
	width_ = bottom_width;

	//setup inner internal product layers
	int h_step = bottom_height / split_h_;
	int w_step = bottom_width / split_w_;

	internal_top_blobs_.clear();
	internal_bottom_blobs_.clear();

	vector<Blob<Dtype>* > temp_top_vec;
	vector<Blob<Dtype>* > temp_bottom_vec;
	for (int row = 0; row < split_h_; ++row){
		for (int col = 0; col < split_w_; ++col){
			//Create layer and push the parameter to global blob list
			inner_product_layers_.push_back(shared_ptr<InnerProductLayer<Dtype> >(new InnerProductLayer<Dtype>(this->layer_param_)));
			//Setup Bottom and Top blobs
			internal_top_blobs_.push_back(new Blob<Dtype>());
			internal_bottom_blobs_.push_back(new Blob<Dtype>());

			int sub_input_size_h, sub_input_size_w;

			//region in the middle will get double overlap
			sub_input_size_h = ((row > 0) && (row < split_h_ - 1))?(h_step + 2 * overlap_h_):(h_step + 2 * overlap_h_);
			CHECK_LT(sub_input_size_h, bottom_height) << "Region height "<< sub_input_size_h
					<<" should not be larger than input map height "<<bottom_height;

			//region in the middle will get double overlap
			sub_input_size_w = ((col > 0) && (col < split_w_ - 1))?(w_step + 2 * overlap_w_):(w_step + overlap_w_);
			CHECK_LT(sub_input_size_w, bottom_width) << "Region width "<< sub_input_size_w
					<<" should not be larger than input map width "<<bottom_width;

			internal_bottom_blobs_.back()->Reshape(num_, bottom_channels,
					sub_input_size_h, sub_input_size_w);

			//Do internal layer setup
			temp_top_vec.clear(); temp_top_vec.push_back(internal_top_blobs_.back());
			temp_bottom_vec.clear(); temp_bottom_vec.push_back(internal_bottom_blobs_.back());
			(this->inner_product_layers_.back())->SetUp(temp_bottom_vec, &temp_top_vec);

			//Link paramters
			this->blobs_.insert(this->blobs_.end(), (this->inner_product_layers_.back())->blobs().begin(), (this->inner_product_layers_.back())->blobs().end());
		}
	}

	//setup the concat layer
	this->concat_layer_.reset(new ConcatLayer<Dtype>(this->layer_param_));
	this->concat_layer_->SetUp(this->internal_bottom_blobs_, &this->internal_top_blobs_);

	//setup data transform





}

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Figure out the dimensions

  (*top)[0]->Reshape(bottom[0]->num(), num_output_, 1, 1);
}

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {


}

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

}

INSTANTIATE_CLASS(TiledInnerProductLayer);

}  // namespace caffe
