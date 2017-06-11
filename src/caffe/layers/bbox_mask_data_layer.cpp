#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <cmath>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif


namespace caffe{
template <typename Dtype>
BBoxMaskDataLayer<Dtype>:: ~BBoxMaskDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void BBoxMaskDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	const string& source = this->layer_param_.seg_refine_param().source();
	const string& root_dir = this->layer_param_.seg_refine_param().root_dir();
	batch_size_ = this->layer_param_.seg_refine_param().batch_size();

	LOG(INFO) << "Opening file: " << source;
	std::ifstream infile(source.c_str());
	string img_filename;
	string label_filename;
	string instance_filename;

	while (infile >> img_filename >> label_filename >> instance_filename){
		lines_.push_back(root_dir + img_filename);
		labels_.push_back(std::make_pair(root_dir + label_filename, root_dir + instance_filename));
	}

	if (this->layer_param_.seg_refine_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;

	Datum datum_data;

	CHECK(ReadSegDataToDatum(lines_[lines_id_].c_str(), &datum_data, true));

	int crop_height = datum_data.height();
	int crop_width = datum_data.width();

	if (this->layer_param_.transform_param().has_crop_size())
	{
		crop_height = this->layer_param_.transform_param().crop_size();
		crop_width = this->layer_param_.transform_param().crop_size();
	}
	else if (this->layer_param_.transform_param().has_crop_height() && this->layer_param_.transform_param().has_crop_width())
	{
		crop_height = this->layer_param_.transform_param().crop_height();
		crop_width = this->layer_param_.transform_param().crop_width();
	}

	if (batch_size_ != 1)
		CHECK(this->layer_param_.transform_param().has_crop_size() || (this->layer_param_.transform_param().has_crop_height() && this->layer_param_.transform_param().has_crop_width()));

	bbox_height_ = crop_height;
	bbox_width_ = crop_width;


	//image
	top[0]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);
	this->prefetch_data_.Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);

	//label
	top[1]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_label_.Reshape(batch_size_, 1, crop_height, crop_width);

	this->prefetch_others_.clear();
	for (int i=0; i<top.size() - 2; i++)
		this->prefetch_others_.push_back(new Blob<Dtype>());

	//instance label
	top[2]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_others_[0]->Reshape(batch_size_, 1, crop_height, crop_width);

	//warp mask
	top[3]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_others_[1]->Reshape(batch_size_, 1, crop_height, crop_width);

	//hint image
	top[4]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);
	this->prefetch_others_[2]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);

	//hint label
	top[5]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_others_[3]->Reshape(batch_size_, 1, crop_height, crop_width);


	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
	LOG(INFO) << "output instance label size: " << top[2]->num() << "," << top[2]->channels() << "," << top[2]->height() << "," << top[2]->width();
	LOG(INFO) << "output warp mask size: " << top[3]->num() << "," << top[3]->channels() << "," << top[3]->height() << "," << top[3]->width();
	LOG(INFO) << "output hint image size: " << top[4]->num() << "," << top[4]->channels() << "," << top[4]->height() << "," << top[4]->width();
	LOG(INFO) << "output hint label size: " << top[5]->num() << "," << top[5]->channels() << "," << top[5]->height() << "," << top[5]->width();
}

template <typename Dtype>
void BBoxMaskDataLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), labels_.begin(), labels_.end(), prefetch_rng);
}


template <typename Dtype>
void BBoxMaskDataLayer<Dtype>::gen_bbox_mask(vector<cv::Mat> &mat_data, vector<cv::Mat> &mat_label, vector<Datum> &datum_data, vector<Datum> &datum_label, bool bbox_aug)
{

  	const int datum_height = mat_data[0].rows;
	const int datum_width = mat_data[0].cols;
	int bbox[4];

	bbox[0] = datum_width - 1;			//start w
	bbox[1] = datum_height - 1;		//start h
	bbox[2] = 0;								//end w
	bbox[3] = 0;								//end h

	for (int i = 0; i<datum_height; i++)
		for (int j=0; j<datum_width; j++)
		{
			int ptr_v = (int)mat_label[1].at<uchar>(i, j);
			if (ptr_v == 1)
			{
				bbox[0] = std::min((j), bbox[0]);			//start w
				bbox[1] = std::min((i), bbox[1]);			//start h
				bbox[2] = std::max((j), bbox[2]);			//end w
				bbox[3] = std::max((i), bbox[3]);			//end h
			}

		}

	int bbox_h = std::max(0, int(bbox[3] - bbox[1]));
	int bbox_w = std::max(0, int(bbox[2] - bbox[0]));

	if (bbox_aug)
	{
		

		bbox[0] += this->data_transformer_->Rand(-bbox_w * 0.20, bbox_w * 0.20);								//start w
		bbox[1] += this->data_transformer_->Rand(-bbox_h * 0.20, bbox_h * 0.20);								//start h
		bbox[2] += this->data_transformer_->Rand(-bbox_w * 0.20, bbox_w * 0.20);								//end w
		bbox[3] += this->data_transformer_->Rand(-bbox_h * 0.20, bbox_h * 0.20);								//end h
		
		bbox_h = std::max(0, int(bbox[3] - bbox[1]));
		bbox_w = std::max(0, int(bbox[2] - bbox[0]));
	}

	bbox[0] -= bbox_w * 0.30;								//start w
	bbox[1] -= bbox_h * 0.30;								//start h
	bbox[2] += bbox_w * 0.30;								//end w
	bbox[3] += bbox_h * 0.30;								//end h


	bbox[0] = std::min(std::max(0, bbox[0]), datum_width - 1);								//start w
	bbox[1] = std::min(std::max(0, bbox[1]), datum_height - 1);								//start h
	bbox[2] = std::min(std::max(bbox[0], bbox[2]), datum_width - 1);								//end w
	bbox[3] = std::min(std::max(bbox[1], bbox[3]), datum_height - 1);								//end h


	for (int c=0; c<mat_label.size(); c++)
	{
		cv::Mat M(mat_label[c], cv::Rect(bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1));
		cv::resize(M, M, cv::Size(bbox_width_, bbox_height_), 0, 0, CV_INTER_NN);
		CVMatToDatum(M, &datum_label[c]);
	}

	cv::Mat M(mat_data[0], cv::Rect(bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1));
	cv::resize(M, M, cv::Size(bbox_width_, bbox_height_), 0, 0);
	CVMatToDatum(M, &datum_data[0]);
}




template <typename Dtype>
void BBoxMaskDataLayer<Dtype>::InternalThreadEntry(){

	const int lines_size = lines_.size();
	const int ignore_label = this->layer_param_.transform_param().ignore_label();


	vector< Blob<Dtype>* > prefetch_data, prefetch_hint_data, prefetch_label, prefetch_hint_label;

	prefetch_data.push_back(&this->prefetch_data_);

	prefetch_label.push_back(&this->prefetch_label_);
	prefetch_label.push_back(this->prefetch_others_[0]);
	prefetch_label.push_back(this->prefetch_others_[1]);

	prefetch_hint_data.push_back(this->prefetch_others_[2]);
	prefetch_hint_label.push_back(this->prefetch_others_[3]);


	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{

		vector<cv::Mat> mat_label(2), mat_data(1);
		vector<Datum> datum_label(2), datum_data(1);

		CHECK_GT(lines_size, lines_id_);

		CHECK(ReadSegDataToCVMat(lines_[lines_id_].c_str(), mat_data[0], true));
		CHECK(ReadSegDataToCVMat(labels_[lines_id_].first.c_str(), mat_label[0], false));
		CHECK(ReadSegDataToCVMat(labels_[lines_id_].second.c_str(), mat_label[1], false));

		// random pick instance and re-id

	  	const int datum_height = mat_data[0].rows;
  		const int datum_width = mat_data[0].cols;

		int max_idx = 0;
		for (int i = 0; i<datum_height; i++)
			for (int j=0; j<datum_width; j++)
			{
				int label_value = (int)mat_label[1].at<uchar>(i, j);
				if (label_value != ignore_label)
				  max_idx = std::max(max_idx, label_value);
			}

		if (max_idx == 0)
		{
			LOG(ERROR) << "no instance!!!!!!!!!!!!!!!!!!!!";
			continue;
		}

		int idx = this->data_transformer_->Rand(max_idx) + 1;

		for (int i = 0; i<datum_height; i++)
			for (int j=0; j<datum_width; j++)
			{
				int label_value = (int)mat_label[1].at<uchar>(i, j);
				if (label_value != ignore_label)
				{
					if (label_value == idx)
						mat_label[1].at<uchar>(i, j) = 1;
					else
					{
						mat_label[0].at<uchar>(i, j) = 0;
						mat_label[1].at<uchar>(i, j) = 0;
					}
				}
				else
					mat_label[0].at<uchar>(i, j)= ignore_label;
			}


		// crop bbox and resize
		gen_bbox_mask(mat_data, mat_label, datum_data, datum_label, true);


		// transfor_aug warp
		datum_label.push_back(Datum());
		this->data_transformer_->Transform_aug(datum_label[1], datum_label[2], 1);

		// Transfor to 0-3 top
		cv::Mat temp;
		this->data_transformer_->Transform(datum_data, datum_label, temp, prefetch_data, prefetch_label, NULL, batch_iter);

		// // crop bbox and resize
		// datum_label.pop_back();
		// gen_bbox_mask(mat_data, mat_label, datum_data, datum_label, false);
		// std::swap(datum_label[0], datum_label[1]);
		// datum_label.pop_back();

		// // transfor_aug image && label
		// this->data_transformer_->Transform_aug2(datum_data[0], datum_label[0], 1);


		// // transfor to 4-5 top
		// this->data_transformer_->Transform(datum_data, datum_label, temp, prefetch_hint_data, prefetch_hint_label, NULL, batch_iter);

		// random drop 3 or (4,5)
		if (this->data_transformer_->Rand(10) < 4)
		{
			caffe_set(this->prefetch_others_[1]->count(1), Dtype(0), this->prefetch_others_[1]->mutable_cpu_data() + this->prefetch_others_[1]->offset(batch_iter));
		}
		// else if (this->data_transformer_->Rand(10) < 2)
		// {
		// 	caffe_set(this->prefetch_others_[2]->count(1), Dtype(0), this->prefetch_others_[2]->mutable_cpu_data() + this->prefetch_others_[2]->offset(batch_iter));
		// 	caffe_set(this->prefetch_others_[3]->count(1), Dtype(0), this->prefetch_others_[3]->mutable_cpu_data() + this->prefetch_others_[3]->offset(batch_iter));
		// } 


		if (false)
		{
		  	cv::Mat im_data(this->prefetch_data_.height(), this->prefetch_data_.width(), CV_8UC3);

			int color_map[256][3];

			color_map[0][0] = 0, color_map[0][1] = 0, color_map[0][2] = 0;
			color_map[255][0] = 255, color_map[255][1] = 255, color_map[255][2] = 255;

			for (int i=1; i<255; i++)
				for (int j=0; j <3; j++)
					color_map[i][j] = rand() % 256;

		  	int tot = rand() * 10000 + rand() + lines_id_;
		  	char temp_path[200];

		  	mkdir("temp", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		  	sprintf(temp_path, "temp/%d",tot);
		  	mkdir(temp_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

		  	for (int p1 = 0; p1 < this->prefetch_data_.height(); p1 ++)
		  		for (int p2 = 0; p2 < this->prefetch_data_.width(); p2 ++)
		  		{
		  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_data_.data_at(batch_iter, 0 , p1, p2)+104);
		  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_data_.data_at(batch_iter, 1 , p1, p2)+117);
		  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_data_.data_at(batch_iter, 2 , p1, p2)+123);	
		  		}
		  	sprintf(temp_path, "temp/%d/image.png", tot);
  			imwrite(temp_path, im_data);


	  		for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
				{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_label_.data_at(batch_iter, 0, p1, p2)][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_label_.data_at(batch_iter, 0, p1, p2)][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_label_.data_at(batch_iter, 0, p1, p2)][2];
				}
			sprintf(temp_path, "temp/%d/label.png", tot);
			imwrite(temp_path, im_data);


	  		for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
				{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, 0, p1, p2)][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, 0, p1, p2)][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, 0, p1, p2)][2];
				}
			sprintf(temp_path, "temp/%d/instance.png", tot);
			imwrite(temp_path, im_data);


	  		for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
				{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[1]->data_at(batch_iter, 0, p1, p2)][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[1]->data_at(batch_iter, 0, p1, p2)][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[1]->data_at(batch_iter, 0, p1, p2)][2];
				}
			sprintf(temp_path, "temp/%d/mask.png", tot);
			imwrite(temp_path, im_data);

	  		for (int p1 = 0; p1 < this->prefetch_others_[2]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[2]->width(); p2 ++)
	  	  		{
  	  				im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_others_[2]->data_at(batch_iter, 0 , p1, p2)+104);
  	  				im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_others_[2]->data_at(batch_iter, 1 , p1, p2)+117);
  	  				im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_others_[2]->data_at(batch_iter, 2 , p1, p2)+123);
  	  			}
		  	sprintf(temp_path, "temp/%d/hint_image.png", tot);
		  	imwrite(temp_path, im_data);
			

	  		for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
				{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[3]->data_at(batch_iter, 0, p1, p2)][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[3]->data_at(batch_iter, 0, p1, p2)][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[3]->data_at(batch_iter, 0, p1, p2)][2];
				}
			sprintf(temp_path, "temp/%d/hint_mask.png", tot);
			imwrite(temp_path, im_data);

		}

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.seg_refine_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
}

INSTANTIATE_CLASS(BBoxMaskDataLayer);
REGISTER_LAYER_CLASS(BBoxMaskData);
}
