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

/*
 * Split the feature map
 */
template <typename Dtype>
void get_sub_map(const Dtype* full_, int full_width_, int full_height_,
				int channel_, int num_,
				int h_start_, int w_start_,
				int h_size_, int w_size_,
				Dtype* sub_){
	const Dtype* full_ptr_;
	Dtype* sub_ptr_;
	for (int n = 0; n < num_; ++n){
		for (int c = 0; c < channel_; ++c){
			full_ptr_ = full_ + (full_width_ * full_height_) * channel_ * num_ + h_start_ * full_width_ + w_start_;
			sub_ptr_ = sub_ + h_size_ * w_size_ * channel_ * num_;
			for (int h = 0; h < h_size_; ++h ){
				for (int w = 0; w < w_size_; ++w){
					sub_ptr_[w] = full_ptr_[w];
				}
				full_ptr_ += full_width_;
				sub_ptr_ += w_size_;
			}
		}
	}
}

template <typename Dtype>
void put_sub_map(Dtype* full_, int full_width_, int full_height_,
		int channel_, int num_,
		int h_start_, int w_start_,
		int h_size_, int w_size_,
		const Dtype* sub_){

	const Dtype* sub_ptr_;
	Dtype* full_ptr_;

	for (int n = 0; n < num_; ++n){
		for (int c = 0; c < channel_; ++c){
			full_ptr_ = full_ + (full_width_ * full_height_) * channel_ * num_ + h_start_ * full_width_ + w_start_;
			sub_ptr_ = sub_ + h_size_ * w_size_ * channel_ * num_;
			for (int h = 0; h < h_size_; ++h ){
				for (int w = 0; w < w_size_; ++w){
					full_ptr_[w] = sub_ptr_[w];
				}
				full_ptr_ += full_width_;
				sub_ptr_ += w_size_;
			}
		}
	}

}

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

	//Transform data

	//forward internal innerproduct layers
	vector<Blob<Dtype>* > temp_top_vec;
	vector<Blob<Dtype>* > temp_bottom_vec;

	int bottom_width_ = bottom[0]->width();
	int bottom_height_ = bottom[0]->height();
	int bottom_channel_ = bottom[0]->channels();

	//inner internal product layers info
	int h_step_ = bottom_height_ / split_h_;
	int w_step_ = bottom_width_ / split_w_;

	const Dtype* bottom_data_ = bottom[0]->cpu_data();

	for (int i = 0; i < inner_product_layers_.size(); ++i){

		//find region info
		int row = i / this->split_w_;
		int col = i % this->split_w_;

		int w_start_, h_start_, w_size_, h_size_;

		w_start_ = w_step_ * col - (col > 0)?overlap_w_:0;
		h_start_ = h_step_ * row - (row > 0)?overlap_h_:0;

		w_size_ = w_step_ + overlap_w_ + ((col > 0)&&(col < split_w_))?(overlap_w_):0;
		h_size_ = h_step_ + overlap_h_ + ((row > 0)&&(row < split_h_))?(overlap_h_):0;

		Dtype* sub_data_ = internal_bottom_blobs_[i]->mutable_cpu_data();
		get_sub_map<Dtype>(bottom_data_, bottom_width_, bottom_height_,
							bottom_channel_, num_,
							h_start_, w_start_,
							h_size_, w_size_,
							sub_data_);
		//Forward
		temp_top_vec.clear();temp_bottom_vec.clear();
		temp_top_vec.push_back(internal_top_blobs_[i]);
		temp_bottom_vec.push_back(internal_bottom_blobs_[i]);

		this->inner_product_layers_[i]->Forward(temp_bottom_vec, &temp_top_vec);
	}

	//concate internal output
	this->concat_layer_->Forward(internal_top_blobs_, top);

}

template <typename Dtype>
void TiledInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

	//backward concate layer
	this->concat_layer_->Backward(top, propagate_down, &this->internal_top_blobs_);

	//forward internal innerproduct layers
	vector<Blob<Dtype>* > temp_top_vec;
	vector<Blob<Dtype>* > temp_bottom_vec;

	int bottom_width_ = (*bottom)[0]->width();
	int bottom_height_ = (*bottom)[0]->height();
	int bottom_channel_ = (*bottom)[0]->channels();

	//inner internal product layers info
	int h_step_ = bottom_height_ / split_h_;
	int w_step_ = bottom_width_ / split_w_;

	Dtype* bottom_data_ = (*bottom)[0]->mutable_cpu_data();


	for (int i = 0; i < inner_product_layers_.size(); ++ i){

		//Forward
		temp_top_vec.clear(); temp_bottom_vec.clear();
		temp_top_vec.push_back(internal_top_blobs_[i]);
		temp_bottom_vec.push_back(internal_bottom_blobs_[i]);

		this->inner_product_layers_[i]->Backward(temp_top_vec, propagate_down, &temp_bottom_vec);

		//find region info
		int row = i / this->split_w_;
		int col = i % this->split_w_;

		int w_start_, h_start_, w_size_, h_size_;

		w_start_ = w_step_ * col - (col > 0)?overlap_w_:0;
		h_start_ = h_step_ * row - (row > 0)?overlap_h_:0;

		w_size_ = w_step_ + overlap_w_ + ((col > 0)&&(col < split_w_))?(overlap_w_):0;
		h_size_ = h_step_ + overlap_h_ + ((row > 0)&&(row < split_h_))?(overlap_h_):0;

		const Dtype* sub_data_ = internal_bottom_blobs_[i]->cpu_data();
		put_sub_map<Dtype>(bottom_data_, bottom_width_, bottom_height_,
							bottom_channel_, num_,
							h_start_, w_start_,
							h_size_, w_size_,
							sub_data_);
	}

}

INSTANTIATE_CLASS(TiledInnerProductLayer);

}  // namespace caffe
