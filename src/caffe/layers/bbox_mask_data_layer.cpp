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

	batch_size_ = this->layer_param_.bbox_mask_data_param().batch_size();
	const string& source = this->layer_param_.bbox_mask_data_param().source();
	const string& root_dir = this->layer_param_.bbox_mask_data_param().root_dir();
	const bool bbox_height = this->layer_param_.bbox_mask_data_param().bbox_height();
	const bool bbox_width = this->layer_param_.bbox_mask_data_param().bbox_width();

	LOG(INFO) << "Opening file: " << source;

	std::ifstream infile(source.c_str());
	string img_filename;
	string label_filename;
	string instance_filename;
	while (infile >> img_filename >> label_filename >> instance_filename){
		lines_.push_back(root_dir + img_filename);
		labels_.push_back(std::make_pair(root_dir + label_filename, root_dir + instance_filename));
	}

	if (this->layer_param_.bbox_mask_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;	

	Datum datum_data, datum_label;
	vector<string> img_vec, label_vec;
	img_vec.clear(), label_vec.clear();
	img_vec.push_back(lines_[lines_id_]);
	label_vec.push_back(labels_[lines_id_].first);
	label_vec.push_back(labels_[lines_id_].second);

	CHECK(ReadSegVideoToDatum(img_vec, label_vec, &datum_data, &datum_label, true));

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

	//image
	top[0]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);
	this->prefetch_data_.Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);

	//label
	top[1]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_label_.Reshape(batch_size_, 1, crop_height, crop_width);



	this->prefetch_others_.clear();

	for (int i=0; i<2; i++)
		this->prefetch_others_.push_back(new Blob<Dtype>());

	//bbox
	top[2]->Reshape(1, 1, 1, 5);
	this->prefetch_others_[0]->Reshape(1, 1, 1, 5);

	//bbox label
	top[3]->Reshape(1, 1, bbox_height, bbox_width);
	this->prefetch_others_[1]->Reshape(1, 1, bbox_height, bbox_width);


	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
	LOG(INFO) << "output bbox size: " << top[2]->num() << "," << top[2]->channels() << "," << top[2]->height() << "," << top[2]->width();
	LOG(INFO) << "output bbox label size: " << top[3]->num() << "," << top[3]->channels() << "," << top[3]->height() << "," << top[2]->width();
}

template <typename Dtype>
void BBoxMaskDataLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), labels_.begin(), labels_.end(), prefetch_rng);
}

template <typename Dtype>
void BBoxMaskDataLayer<Dtype>::InternalThreadEntry(){

	const bool bbox_aug = this->layer_param_.bbox_mask_data_param().bbox_aug();
	const bool bbox_height = this->layer_param_.bbox_mask_data_param().bbox_height();
	const bool bbox_width = this->layer_param_.bbox_mask_data_param().bbox_width();
	const int stride = this->layer_param_.transform_param().stride();

	const int lines_size = lines_.size();

	vector<int> instance_num(batch_size_, 0);
	vector< vector<int> > instance_ids(batch_size_, vector<int>());
	int vis[256], tot_instance_num = 0;
	Blob<Dtype> label_buff;


	//data_transfor
	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{
		CHECK_GT(lines_size, lines_id_);

		Datum datum_data, datum_label;
		vector<string> img_vec, label_vec;
		img_vec.clear(), label_vec.clear();
		img_vec.push_back(lines_[lines_id_]);
		label_vec.push_back(labels_[lines_id_].first);
		label_vec.push_back(labels_[lines_id_].second);

		CHECK(ReadSegVideoToDatum(img_vec, label_vec, &datum_data, &datum_label, true));

		this->data_transformer_->Transform(datum_data, datum_label, &this->prefetch_data_, &label_buff, batch_iter);

		memset(vis, 255, sizeof(vis));
		const Dtype *ptr = label_buff.cpu_data() + label_buff.offset(batch_iter, 1);
		for (int i = 0; i<label_buff.height(); i++)
			for (int j=0; j<label_buff.width(); j++)
			{
				int ptr_v = (int)(*ptr + 0.5);
				if (ptr_v != 0 && ptr_v != this->layer_param_.transform_param().ignore_label() && vis[ptr_v] == -1)
				{
					vis[ptr_v] =  instance_num[batch_iter];
					instance_num[batch_iter]++;
					instance_ids[batch_iter].push_back(ptr_v);
				}
				ptr++;
			}
		tot_instance_num += instance_num[batch_iter];
	}



	this->prefetch_label_.Reshape(batch_size_, 1, label_buff.height(), label_buff.width());
	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{
		const Dtype *ptr1 = label_buff.cpu_data() + label_buff.offset(batch_iter);
		Dtype *ptr2 = this->prefetch_label_.mutable_cpu_data() + this->prefetch_label_.offset(batch_iter);

		for (int i = 0; i<label_buff.height(); i++)
			for (int j=0; j<label_buff.width(); j++)
				*(ptr2++) = *(ptr1++);
	}

	this->prefetch_others_[0]->Reshape(tot_instance_num, 1, 1, 5);
	this->prefetch_others_[1]->Reshape(tot_instance_num, 1, bbox_height, bbox_width);

	//label
	Dtype* ptr_bbox = this->prefetch_others_[0]->mutable_cpu_data();
	Dtype* ptr_mask = this->prefetch_others_[1]->mutable_cpu_data();

	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{
		memset(vis, 255, sizeof(vis));
		for (int i = 0; i < instance_num[batch_iter]; i++)
		{
			ptr_bbox[i*5 + 0] = batch_iter;
			ptr_bbox[i*5 + 1] = label_buff.width() - 1;			//start w
			ptr_bbox[i*5 + 2] = label_buff.height() - 1;		//start h
			ptr_bbox[i*5 + 3] = 0;								//end w
			ptr_bbox[i*5 + 4] = 0;								//end h
			vis[instance_ids[batch_iter][i]] = i;
		}

		vector <cv::Mat> temp(instance_num[batch_iter], cv::Mat::zeros(label_buff.height(), label_buff.width(), CV_32F));

		const Dtype *ptr = label_buff.cpu_data() + label_buff.offset(batch_iter, 1);

		for (int i = 0; i<label_buff.height(); i++)
			for (int j=0; j<label_buff.width(); j++)
			{
				int ptr_v = (int)(*ptr + 0.5);
				if (vis[ptr_v] != -1)
				{
					ptr_bbox[vis[ptr_v]*5 + 1] = std::min(Dtype(j), ptr_bbox[vis[ptr_v]*5 + 1]);			//start w
					ptr_bbox[vis[ptr_v]*5 + 2] = std::min(Dtype(i), ptr_bbox[vis[ptr_v]*5 + 2]);			//start h
					ptr_bbox[vis[ptr_v]*5 + 3] = std::max(Dtype(j), ptr_bbox[vis[ptr_v]*5 + 3]);			//end w
					ptr_bbox[vis[ptr_v]*5 + 4] = std::max(Dtype(i), ptr_bbox[vis[ptr_v]*5 + 4]);			//end h
					temp[vis[ptr_v]].at<float>(i, j) = 1.0f;
				}
				ptr++;
			}

		for (int i = 0; i < instance_num[batch_iter]; i++)
		{
			if (bbox_aug)
			{
				int bbox_h = ptr_bbox[4] - ptr_bbox[2];
				int bbox_w = ptr_bbox[3] - ptr_bbox[1];

				ptr_bbox[1] += this->data_transformer_->Rand(-bbox_w * 0.05, bbox_w * 0.05);								//start w
				ptr_bbox[2] += this->data_transformer_->Rand(-bbox_h * 0.05, bbox_h * 0.05);								//start h
				ptr_bbox[3] += this->data_transformer_->Rand(-bbox_w * 0.05, bbox_w * 0.05);								//end w
				ptr_bbox[4] += this->data_transformer_->Rand(-bbox_h * 0.05, bbox_h * 0.05);								//end h

				ptr_bbox[1] = std::min(std::max(0.0, round(ptr_bbox[1] / stride) * stride), label_buff.width() - 1.0);								//start w
				ptr_bbox[2] = std::min(std::max(0.0, round(ptr_bbox[1] / stride) * stride), label_buff.height() - 1.0);								//start h
				ptr_bbox[3] = std::min(std::max(0.0, round(ptr_bbox[1] / stride) * stride), label_buff.width() - 1.0);								//end w
				ptr_bbox[4] = std::min(std::max(0.0, round(ptr_bbox[1] / stride) * stride), label_buff.height() - 1.0);								//end h
			}
			LOG(ERROR) <<ptr_bbox[1] << ' ' <<ptr_bbox[2]<< ' '<< ptr_bbox[3] << ' ' << ptr_bbox[4];
			cv::Mat M(temp[i], cv::Rect(ptr_bbox[1], ptr_bbox[2], ptr_bbox[3] - ptr_bbox[1] + 1, ptr_bbox[4] - ptr_bbox[2] + 1));
			cv::resize(M, M, cv::Size(bbox_width, bbox_height), 0, 0, CV_INTER_NN);

			for (int j=0; j<bbox_width * bbox_height; j++)
				ptr_mask[j] = *(M.data + j);
			
			ptr_bbox += 5;
			ptr_mask += bbox_width * bbox_height;
		}
		

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.bbox_mask_data_param().shuffle()) {
				ShuffleImages();
			}
		}
	}



	if (true)
	{
	  	cv::Mat im_data(this->prefetch_data_.height(), this->prefetch_data_.width(), CV_8UC3);

		int color_map[256][3];

		color_map[0][0] = 0, color_map[0][1] = 0, color_map[0][2] = 0;
		color_map[255][0] = 0, color_map[255][1] = 0, color_map[255][2] = 0;

		for (int i=1; i<255; i++)
			for (int j=0; j <3; j++)
				color_map[i][j] = rand() % 256;

	  	int tot = rand() * 10000 + rand() + lines_id_;
	  	char temp_path[200];

	  	mkdir("temp", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	  	sprintf(temp_path, "temp/%d",tot);
	  	mkdir(temp_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
		{
		  	for (int p1 = 0; p1 < this->prefetch_data_.height(); p1 ++)
		  		for (int p2 = 0; p2 < this->prefetch_data_.width(); p2 ++)
		  		{
		  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_data_.data_at(batch_iter, 0 , p1, p2)+104);
		  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_data_.data_at(batch_iter, 1 , p1, p2)+117);
		  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_data_.data_at(batch_iter, 2 , p1, p2)+123);	
		  		}
			sprintf(temp_path, "temp/%d/image_%d.png", tot, batch_iter);
			imwrite(temp_path, im_data);

			for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
				for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
				{
					im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_label_.data_at(batch_iter, 0, p1, p2)][0];
					im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_label_.data_at(batch_iter, 0, p1, p2)][1];
					im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_label_.data_at(batch_iter, 0, p1, p2)][2];
					}
			sprintf(temp_path, "temp/%d/label_%d.png", tot, batch_iter);
			imwrite(temp_path, im_data);
		}


		im_data = cv::Mat::zeros(bbox_height, bbox_width, CV_8UC3);
		for (int i=0; i<this->prefetch_others_[1]->num(); i++)
		{
  			for (int p1 = 0; p1 < this->prefetch_others_[1]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[1]->width(); p2 ++)
	  	  		{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[1]->data_at(i, 0, p1, p2)][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[1]->data_at(i, 0, p1, p2)][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[1]->data_at(i, 0, p1, p2)][2];
	  	  		}
		  	sprintf(temp_path, "temp/%d/instance_%d.png", tot, i);
		  	imwrite(temp_path, im_data);
	  	}
	}
}

INSTANTIATE_CLASS(BBoxMaskDataLayer);
REGISTER_LAYER_CLASS(BBoxMaskData);
}
