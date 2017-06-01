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
BBoxVideoDataLayer<Dtype>:: ~BBoxVideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void BBoxVideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	batch_size_ = this->layer_param_.bbox_video_data_param().batch_size();
	bbox_height_ = this->layer_param_.bbox_video_data_param().bbox_height();
	bbox_width_ = this->layer_param_.bbox_video_data_param().bbox_width();
	stride_ = this->layer_param_.transform_param().stride();
	ignore_label_ = this->layer_param_.transform_param().ignore_label();

	image_pattern_ = this->layer_param_.bbox_video_data_param().image_pattern();
	label_pattern_ = this->layer_param_.bbox_video_data_param().label_pattern();

	const string& source = this->layer_param_.bbox_video_data_param().source();
	const string& root_dir = this->layer_param_.bbox_video_data_param().root_dir();

	LOG(INFO) << "Opening file: " << source;

	std::ifstream infile(source.c_str());

	string img_dir;
	string label_dir;
	string warp_dir;
	int frame_num;

	while (infile >> img_dir >> label_dir >> warp_dir >> frame_num){
		lines_.push_back(std::make_pair(root_dir + img_dir, root_dir + label_dir));
		labels_.push_back(std::make_pair(root_dir + warp_dir, frame_num));
	}

	if (this->layer_param_.bbox_video_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;	

	Datum datum_data, datum_label;
	vector<string> img_vec, label_vec;
	img_vec.clear(), label_vec.clear();

	sprintf(string_buf, image_pattern_.c_str(), 0);
	img_vec.push_back(lines_[lines_id_].first + string(string_buf));

	
	CHECK(ReadSegVideoToDatum(img_vec, label_vec, &datum_data, &datum_label, true));

	int crop_height = ((datum_data.height() - 1) / stride_ + 1) * stride_;
	int crop_width = ((datum_data.width() - 1) / stride_ + 1) * stride_;

	if (this->layer_param_.transform_param().has_crop_size())
	{
		NOT_IMPLEMENTED;
	}
	else if (this->layer_param_.transform_param().has_crop_height() && this->layer_param_.transform_param().has_crop_width())
	{
		NOT_IMPLEMENTED;
	}

	if (batch_size_ != 1)
		NOT_IMPLEMENTED;

	//image
	top[0]->Reshape(batch_size_ , datum_data.channels(), crop_height, crop_width);
	this->prefetch_data_.Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);

	//bbox
	top[1]->Reshape(5, 1, 1, 5);
	this->prefetch_label_.Reshape(5, 1, 1, 5);


	this->prefetch_others_.clear();
	for (int i=0; i<top.size() - 2; i++)
		this->prefetch_others_.push_back(new Blob<Dtype>());

	//bbox label
	top[2]->Reshape(5 , 1, bbox_height_, bbox_width_);
	this->prefetch_others_[0]->Reshape(5 , 1, bbox_height_, bbox_width_);


	//hint image
	top[3]->Reshape(batch_size_ , datum_data.channels(), crop_height, crop_width);
	this->prefetch_others_[1]->Reshape(batch_size_ , datum_data.channels(), crop_height, crop_width);

	//hint bbox
	top[4]->Reshape(5, 1, 1, 5);
	this->prefetch_others_[2]->Reshape(5, 1, 1, 5);

	//hint bbox label
	top[5]->Reshape(5 , 1, bbox_height_, bbox_width_);
	this->prefetch_others_[3]->Reshape(5 , 1, bbox_height_, bbox_width_);

	//warp bbox label
	top[6]->Reshape(5 , 1, bbox_height_, bbox_width_);
	this->prefetch_others_[4]->Reshape(5 , 1, bbox_height_, bbox_width_);


	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output bbox size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
	LOG(INFO) << "output bbox label size: " << top[2]->num() << "," << top[2]->channels() << "," << top[2]->height() << "," << top[2]->width();
	LOG(INFO) << "output hint image size: " << top[3]->num() << "," << top[3]->channels() << "," << top[3]->height() << "," << top[3]->width();
	LOG(INFO) << "output hint bbox size: " << top[4]->num() << "," << top[4]->channels() << "," << top[4]->height() << "," << top[4]->width();
	LOG(INFO) << "output hint bbox label size: " << top[5]->num() << "," << top[5]->channels() << "," << top[5]->height() << "," << top[5]->width();
	LOG(INFO) << "output warp bbox label size: " << top[6]->num() << "," << top[6]->channels() << "," << top[6]->height() << "," << top[6]->width();
}

template <typename Dtype>
void BBoxVideoDataLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), labels_.begin(), labels_.end(), prefetch_rng);
}


template <typename Dtype>
void BBoxVideoDataLayer<Dtype>::gen_bbox_and_mask(Blob<Dtype> &label_data, int channel, Blob<Dtype> &bbox, Blob<Dtype> &mask, vector<int>& instance_ids, bool bbox_aug)
{
	int tot_instance_num = instance_ids.size(), vis[256];

	bbox.Reshape(tot_instance_num, 1, 1, 5);
	mask.Reshape(tot_instance_num, 1, bbox_height_, bbox_width_);

	Dtype *ptr_bbox = bbox.mutable_cpu_data();
	Dtype *ptr_mask = mask.mutable_cpu_data();

	memset(vis, 255, sizeof(vis));

	for (int i = 0; i < tot_instance_num; i++)
	{
		ptr_bbox[i*5 + 1] = label_data.width() - 1;			//start w
		ptr_bbox[i*5 + 2] = label_data.height() - 1;		//start h
		ptr_bbox[i*5 + 3] = 0;								//end w
		ptr_bbox[i*5 + 4] = 0;								//end h
		vis[instance_ids[i]] = i;
	}

	vector <cv::Mat> temp;
	for (int i=0; i<tot_instance_num; i++)
		temp.push_back(cv::Mat::zeros(label_data.height(), label_data.width(), CV_8U));

	const Dtype *ptr = label_data.cpu_data() + label_data.offset(0, channel);

	for (int i = 0; i<label_data.height(); i++)
		for (int j=0; j<label_data.width(); j++)
		{
			int ptr_v = (int)(*ptr + 0.5);
			if (vis[ptr_v] != -1)
			{
				ptr_bbox[vis[ptr_v]*5 + 1] = std::min(Dtype(j), ptr_bbox[vis[ptr_v]*5 + 1]);			//start w
				ptr_bbox[vis[ptr_v]*5 + 2] = std::min(Dtype(i), ptr_bbox[vis[ptr_v]*5 + 2]);			//start h
				ptr_bbox[vis[ptr_v]*5 + 3] = std::max(Dtype(j), ptr_bbox[vis[ptr_v]*5 + 3]);			//end w
				ptr_bbox[vis[ptr_v]*5 + 4] = std::max(Dtype(i), ptr_bbox[vis[ptr_v]*5 + 4]);			//end h
				temp[vis[ptr_v]].at<uchar>(i, j) = 1;
			}
			ptr++;
		}

	for (int i = 0; i < tot_instance_num; i++)
	{
		if (bbox_aug)
		{
			int bbox_h = std::max(0, int(ptr_bbox[4] - ptr_bbox[2]));
			int bbox_w = std::max(0, int(ptr_bbox[3] - ptr_bbox[1]));

			ptr_bbox[1] += this->data_transformer_->Rand(-bbox_w * 0.40, bbox_w * 0.40);								//start w
			ptr_bbox[2] += this->data_transformer_->Rand(-bbox_h * 0.40, bbox_h * 0.40);								//start h
			ptr_bbox[3] += this->data_transformer_->Rand(-bbox_w * 0.40, bbox_w * 0.40);								//end w
			ptr_bbox[4] += this->data_transformer_->Rand(-bbox_h * 0.40, bbox_h * 0.40);								//end h
			
			bbox_h = std::max(0, int(ptr_bbox[4] - ptr_bbox[2]));
			bbox_w = std::max(0, int(ptr_bbox[3] - ptr_bbox[1]));

			ptr_bbox[1] -= bbox_w * 0.40;								//start w
			ptr_bbox[2] -= bbox_h * 0.40;								//start h
			ptr_bbox[3] += bbox_w * 0.40;								//end w
			ptr_bbox[4] += bbox_h * 0.40;								//end h
		}

		ptr_bbox[0] = 0;
		ptr_bbox[1] = std::min(std::max(0.0, round(ptr_bbox[1] / stride_) * stride_), label_data.width() - 1.0);								//start w
		ptr_bbox[2] = std::min(std::max(0.0, round(ptr_bbox[2] / stride_) * stride_), label_data.height() - 1.0);								//start h
		ptr_bbox[3] = std::min(std::max(0.0, round(ptr_bbox[3] / stride_) * stride_), label_data.width() - 1.0);								//end w
		ptr_bbox[4] = std::min(std::max(0.0, round(ptr_bbox[4] / stride_) * stride_), label_data.height() - 1.0);								//end h

		ptr_bbox[3] = std::max(ptr_bbox[3], ptr_bbox[1]);
		ptr_bbox[4] = std::max(ptr_bbox[4], ptr_bbox[2]);

		cv::Mat M(temp[i], cv::Rect(ptr_bbox[1], ptr_bbox[2], ptr_bbox[3] - ptr_bbox[1] + 1, ptr_bbox[4] - ptr_bbox[2] + 1));
		
		cv::resize(M, M, cv::Size(bbox_width_, bbox_height_), 0, 0, CV_INTER_NN);

		for (int j=0; j<bbox_width_ * bbox_height_; j++)
			ptr_mask[j] = (Dtype)*(M.data + j);
		
		ptr_bbox += 5;
		ptr_mask += bbox_width_ * bbox_height_;
	}
}



template <typename Dtype>
void BBoxVideoDataLayer<Dtype>::gen_mask(Blob<Dtype> &label_data, int channel, Blob<Dtype> &bbox, Blob<Dtype> &mask, vector<int>& instance_ids)
{
	int tot_instance_num = instance_ids.size(), vis[256];

	mask.Reshape(tot_instance_num, 1, bbox_height_, bbox_width_);

	const Dtype *ptr_bbox = bbox.cpu_data();
	Dtype *ptr_mask = mask.mutable_cpu_data();

	memset(vis, 255, sizeof(vis));

	for (int i = 0; i < tot_instance_num; i++)
		vis[instance_ids[i]] = i;
	

	vector <cv::Mat> temp;
	for (int i=0; i<tot_instance_num; i++)
		temp.push_back(cv::Mat::zeros(label_data.height(), label_data.width(), CV_8U));

	const Dtype *ptr = label_data.cpu_data() + label_data.offset(0, channel);

	for (int i = 0; i<label_data.height(); i++)
		for (int j=0; j<label_data.width(); j++)
		{
			int ptr_v = (int)(*ptr + 0.5);
			if (vis[ptr_v] != -1)
				temp[vis[ptr_v]].at<uchar>(i, j) = 1;
			ptr++;
		}

	for (int i = 0; i < tot_instance_num; i++)
	{
		cv::Mat M(temp[i], cv::Rect(ptr_bbox[1], ptr_bbox[2], ptr_bbox[3] - ptr_bbox[1] + 1, ptr_bbox[4] - ptr_bbox[2] + 1));
		
		cv::resize(M, M, cv::Size(bbox_width_, bbox_height_), 0, 0, CV_INTER_NN);

		for (int j=0; j<bbox_width_ * bbox_height_; j++)
			ptr_mask[j] = (Dtype)*(M.data + j);
		
		ptr_bbox += 5;
		ptr_mask += bbox_width_ * bbox_height_;
	}
}



template <typename Dtype>
void BBoxVideoDataLayer<Dtype>::InternalThreadEntry(){

	const bool bbox_aug = this->layer_param_.bbox_video_data_param().bbox_aug();

	const int lines_size = lines_.size();

	vector<int> instance_ids;
	int vis[256], tot_instance_num = 0;
	Blob<Dtype> data_buff, label_buff;

	//data_transfor
	CHECK_GT(lines_size, lines_id_);

	Datum datum_data, datum_label;
	vector<string> img_vec, label_vec;



	int current_frame = this->data_transformer_->Rand(labels_[lines_id_].second), dist = labels_[lines_id_].second  / 2;
	while(current_frame == 0)
		current_frame = this->data_transformer_->Rand(labels_[lines_id_].second);

	while (true)
	{
		int hint_frame = this->data_transformer_->Rand(labels_[lines_id_].second);
		while(std::abs(hint_frame - current_frame) < dist)
			hint_frame = this->data_transformer_->Rand(labels_[lines_id_].second);

		img_vec.clear(), label_vec.clear();

		sprintf(string_buf, image_pattern_.c_str(), current_frame);
		img_vec.push_back(lines_[lines_id_].first + string(string_buf));

		sprintf(string_buf, image_pattern_.c_str(), hint_frame);
		img_vec.push_back(lines_[lines_id_].first + string(string_buf));


		sprintf(string_buf, label_pattern_.c_str(), current_frame);
		label_vec.push_back(lines_[lines_id_].second + string(string_buf));

		sprintf(string_buf, label_pattern_.c_str(), hint_frame);
		label_vec.push_back(lines_[lines_id_].second + string(string_buf));

		sprintf(string_buf, label_pattern_.c_str(), current_frame);
		label_vec.push_back(labels_[lines_id_].first + string(string_buf));

		CHECK(ReadSegVideoToDatum(img_vec, label_vec, &datum_data, &datum_label, true));


		// check label
		bool all_label = true;
		memset(vis, 255, sizeof(vis));
		instance_ids.clear();
		const string& label = datum_label.data();

		// hint label
		int datum_height = datum_label.height(), datum_width = datum_label.width();
		for (int i = datum_height * datum_width; i < datum_height * datum_width * 2; ++i)
		{
			int label_v = static_cast<int>(static_cast<uint8_t>(label[i]) + 0.5);
			if (ignore_label_ != label_v && label_v != 0)
				vis[label_v] = 1;
		}

		for (int i = 0; i < datum_height * datum_width; ++i)
		{
			int label_v = static_cast<int>(static_cast<uint8_t>(label[i]) + 0.5);
			if (ignore_label_ != label_v && label_v != 0)
			{
				if (vis[label_v] != 2)
					instance_ids.push_back(label_v);

				if (vis[label_v] == -1)
				{
					all_label = false;
					break;
				}
				else
					vis[label_v] = 2;
			}
		}

		if (all_label && instance_ids.size() != 0)
			break;

		// LOG(ERROR) << "Fail" << ' ' << label_vec[0] << ' ' << instance_ids.size() << ' ' << current_frame << ' ' <<hint_frame;
		if (instance_ids.size() == 0)
		{
			current_frame = this->data_transformer_->Rand(labels_[lines_id_].second);
			while(current_frame == 0)
				current_frame = this->data_transformer_->Rand(labels_[lines_id_].second);
		}
		dist = dist * 2 / 3;
	}

	this->data_transformer_->Transform(datum_data, datum_label, &data_buff, &label_buff, 0);

	//top 0
	this->prefetch_data_.Reshape(1, data_buff.channels() / 2, data_buff.height(), data_buff.width());
	for (int i=0; i<this->prefetch_data_.count(); i++)
		this->prefetch_data_.mutable_cpu_data()[i] = data_buff.cpu_data()[i];

	//top 3
	this->prefetch_others_[1]->Reshape(1, data_buff.channels() / 2, data_buff.height(), data_buff.width());
	for (int i=0, j=this->prefetch_data_.count(); i<this->prefetch_others_[1]->count(); i++, j++)
		this->prefetch_others_[1]->mutable_cpu_data()[i] = data_buff.cpu_data()[j];


	tot_instance_num = instance_ids.size();

	if (tot_instance_num == 0)
	{
		NOT_IMPLEMENTED;
		// LOG(FATE) << "no instance !!!!!!!!!!!!!!!!!!!";
		// this->prefetch_label_.Reshape(1, 1, bbox_height_, bbox_width_);
		// this->prefetch_others_[1]->Reshape(1, 1, 1, 5);
		// this->prefetch_others_[2]->Reshape(1, 1, bbox_height_, bbox_width_);
		// this->prefetch_others_[3]->Reshape(1, 1, bbox_height_, bbox_width_);
		// caffe_set(this->prefetch_label_.count(), Dtype(0), this->prefetch_label_.mutable_cpu_data());	
		// caffe_set(this->prefetch_others_[1]->count(), Dtype(0), this->prefetch_others_[1]->mutable_cpu_data());
		// caffe_set(this->prefetch_others_[0]->count(), Dtype(0), this->prefetch_others_[0]->mutable_cpu_data());	
		// caffe_set(this->prefetch_others_[1]->count(), Dtype(0), this->prefetch_others_[1]->mutable_cpu_data());
	}
	else
	{
		gen_bbox_and_mask(label_buff, 0, this->prefetch_label_, *this->prefetch_others_[0], instance_ids, bbox_aug);

		gen_bbox_and_mask(label_buff, 1, *this->prefetch_others_[2], *this->prefetch_others_[3], instance_ids, false);

		gen_mask(label_buff, 2, this->prefetch_label_, *this->prefetch_others_[4], instance_ids);


		for (int i=0; i<tot_instance_num; i++)
			if (this->data_transformer_->Rand(10) != 0)
			{
				caffe_set(bbox_height_ * bbox_width_, Dtype(0.5), this->prefetch_others_[4]->mutable_cpu_data() + this->prefetch_others_[4]->offset(i));
			}


		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.bbox_video_data_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
	

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
	  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_data_.data_at(0, 0 , p1, p2)+104);
	  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_data_.data_at(0, 1 , p1, p2)+117);
	  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_data_.data_at(0, 2 , p1, p2)+123);	
	  		}
		sprintf(temp_path, "temp/%d/image.png", tot);
		imwrite(temp_path, im_data);

		for (int p1 = 0; p1 < this->prefetch_others_[1]->height(); p1 ++)
			for (int p2 = 0; p2 < this->prefetch_others_[1]->width(); p2 ++)
			{
	  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_others_[1]->data_at(0, 0 , p1, p2)+104);
	  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_others_[1]->data_at(0, 1 , p1, p2)+117);
	  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_others_[1]->data_at(0, 2 , p1, p2)+123);	
			}
		sprintf(temp_path, "temp/%d/hint_image.png", tot);
		imwrite(temp_path, im_data);

		im_data = cv::Mat::zeros(bbox_height_, bbox_width_, CV_8UC3);
		for (int i=0; i<this->prefetch_others_[0]->num(); i++)
		{
  			for (int p1 = 0; p1 < this->prefetch_others_[0]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[0]->width(); p2 ++)
	  	  		{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[0]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[0]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[0]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][2];
	  	  		}
		  	sprintf(temp_path, "temp/%d/instance_%d.png", tot, i);
		  	imwrite(temp_path, im_data);
	  	}


		for (int i=0; i<this->prefetch_others_[3]->num(); i++)
		{
  			for (int p1 = 0; p1 < this->prefetch_others_[3]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[3]->width(); p2 ++)
	  	  		{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[3]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[3]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[3]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][2];
	  	  		}
		  	sprintf(temp_path, "temp/%d/hint_instance_%d.png", tot, i);
		  	imwrite(temp_path, im_data);
	  	}

		for (int i=0; i<this->prefetch_others_[4]->num(); i++)
		{
  			for (int p1 = 0; p1 < this->prefetch_others_[4]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[4]->width(); p2 ++)
	  	  		{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[4]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[4]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[4]->data_at(i, 0, p1, p2) == 0 ? 0 : i+1][2];
	  	  		}
		  	sprintf(temp_path, "temp/%d/warp_instance_%d.png", tot, i);
		  	imwrite(temp_path, im_data);
	  	}

	}

}

INSTANTIATE_CLASS(BBoxVideoDataLayer);
REGISTER_LAYER_CLASS(BBoxVideoData);
}
