#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

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
SegRefineLayer<Dtype>:: ~SegRefineLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SegRefineLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	const int stride = this->layer_param_.transform_param().stride();
	const string& source = this->layer_param_.seg_refine_param().source();
	const string& root_dir = this->layer_param_.seg_refine_param().root_dir();
	const int instance_num = this->layer_param_.seg_refine_param().instance_num();

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


	int crop_height = datum_data.height() / stride * stride;
	int crop_width = datum_data.width() / stride * stride;

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

	batch_size_ = this->layer_param_.seg_refine_param().batch_size();
	if (batch_size_ != 1)
		CHECK(this->layer_param_.transform_param().has_crop_size() || (this->layer_param_.transform_param().has_crop_height() && this->layer_param_.transform_param().has_crop_width()));

	//image
	top[0]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);
	this->prefetch_data_.Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);

	//class label
	top[1]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_label_.Reshape(batch_size_, 1, crop_height, crop_width);

	this->prefetch_others_.clear();
	for (int i=0; i<top.size() - 2; i++)
		this->prefetch_others_.push_back(new Blob<Dtype>());

	//instance label
	top[2]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_others_[0]->Reshape(batch_size_, 1, crop_height, crop_width);

	//mask
	top[3]->Reshape(batch_size_, instance_num, crop_height, crop_width);
	this->prefetch_others_[1]->Reshape(batch_size_, instance_num, crop_height, crop_width);

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output class label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
	LOG(INFO) << "output instance label size: " << top[2]->num() << "," << top[2]->channels() << "," << top[2]->height() << "," << top[2]->width();
	LOG(INFO) << "output mask size: " << top[3]->num() << "," << top[3]->channels() << "," << top[3]->height() << "," << top[3]->width();
}

template <typename Dtype>
void SegRefineLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), labels_.begin(), labels_.end(), prefetch_rng);
}

template <typename Dtype>
void SegRefineLayer<Dtype>::InternalThreadEntry(){

	const int instance_num = this->layer_param_.seg_refine_param().instance_num();
	const int ignore_label = this->layer_param_.transform_param().ignore_label();

	Datum datum_data;
	vector<Datum> datum_label(3);	
	const int lines_size = lines_.size();

	Blob<Dtype> label_buff;

	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{
		CHECK_GT(lines_size, lines_id_);


		CHECK(ReadSegDataToDatum(lines_[lines_id_].c_str(), &datum_data, true));
		CHECK(ReadSegDataToDatum(labels_[lines_id_].first.c_str(), &datum_label[0], false));
		CHECK(ReadSegDataToDatum(labels_[lines_id_].second.c_str(), &datum_label[1], false));

		//find max_idx & re-rank

	  	const int datum_height = datum_label[1].height();
  		const int datum_width = datum_label[1].width();
	  	string* ptr = datum_label[1].mutable_data();
		
		int max_idx = instance_num;

		for (int data_index = 0; data_index < datum_height * datum_width; ++data_index)
		{
			int label_value = (int)(*ptr)[data_index];
			if (label_value != ignore_label)
			  max_idx = std::max(max_idx, label_value);
		}
		int vis[256]; vis[0] = 0; vis[ignore_label] = ignore_label;
		for (int i=1; i < max_idx + 1; i++)
		{
			vis[i] = (i <= instance_num) ? i : 0;
			std::swap(vis[i], vis[this->data_transformer_->Rand(i) + 1]);
		}

		for (int data_index = 0; data_index < datum_height * datum_width; ++data_index)
			(*ptr)[data_index] = (uchar)vis[(int)(*ptr)[data_index]];


		//augment
		this->data_transformer_->Transform_aug(datum_label[1], datum_label[2], instance_num);

		//concat label
		Datum concat_label;
		concat_label.set_channels(1 + 1 + instance_num);
		concat_label.set_height(datum_height);
		concat_label.set_width(datum_width);
		concat_label.clear_data();
		concat_label.clear_float_data();
		ptr = concat_label.mutable_data();

		for (int i=0; i<datum_label.size(); i++)
			for (int data_index=0; data_index < datum_label[i].channels() * datum_height * datum_width; data_index++)
				ptr->push_back(datum_label[i].data()[data_index]);

		this->data_transformer_->Transform(datum_data, concat_label, &this->prefetch_data_, &label_buff, batch_iter);


		if (this->layer_param_.seg_refine_param().balance())
		{
			for (int t = 0; t < 20; t++)
			{
				std::vector<int> cnt(256, 0); int max_label_cnt = 0;
				for (int p1 = 0; p1 < label_buff.height(); p1 ++)
		  	  		for (int p2 = 0; p2 < label_buff.width(); p2 ++)
		  	  		{
		  	  			int label_value = (int)label_buff.data_at(batch_iter, 1, p1, p2);
		  	  			cnt[label_value]++;
		  	  		}
		  	  	for (int i = 0; i<cnt.size(); i++)
		  			max_label_cnt = std::max(max_label_cnt, cnt[i]);

		  		if (max_label_cnt > 0.99 * label_buff.height() * label_buff.width())
		  			this->data_transformer_->Transform(datum_data, concat_label, &this->prefetch_data_, &label_buff, batch_iter);
	  			else
	  				break;
	  			// if (t == 19)
	  			// 	LOG(INFO) << "Balance Fail";
			}
		}



		//push back

		if (batch_iter == 0)
		{
			this->prefetch_label_.Reshape(batch_size_, 1, label_buff.height(), label_buff.width());
			this->prefetch_others_[0]->Reshape(batch_size_, 1, label_buff.height(), label_buff.width());
			this->prefetch_others_[1]->Reshape(batch_size_, instance_num, label_buff.height(), label_buff.width());
		}

		int tot_offset = label_buff.offset(batch_iter);

		caffe_copy(this->prefetch_label_.count(1), label_buff.cpu_data() + tot_offset, this->prefetch_label_.mutable_cpu_data() + this->prefetch_label_.offset(batch_iter));
		tot_offset += this->prefetch_label_.count(1);

		caffe_copy(this->prefetch_others_[0]->count(1), label_buff.cpu_data() + tot_offset, this->prefetch_others_[0]->mutable_cpu_data() + this->prefetch_others_[0]->offset(batch_iter));
		tot_offset += this->prefetch_others_[0]->count(1);

		caffe_copy(this->prefetch_others_[1]->count(1), label_buff.cpu_data() + tot_offset, this->prefetch_others_[1]->mutable_cpu_data() + this->prefetch_others_[1]->offset(batch_iter));
		
#ifdef USE_MPI
		if  (Caffe::MPI_my_rank() == 0)
#endif
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
			sprintf(temp_path, "temp/%d/class_label.png", tot);
			imwrite(temp_path, im_data);



	  		for (int p1 = 0; p1 < this->prefetch_others_[0]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[0]->width(); p2 ++)
				{
	  	  			im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, 0, p1, p2)][0];
	  	  			im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, 0, p1, p2)][1];
	  	  			im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, 0, p1, p2)][2];
				}
			sprintf(temp_path, "temp/%d/instance_label.png", tot);
			imwrite(temp_path, im_data);


		  	for (int i=0; i < this->prefetch_others_[1]->channels(); i++)
		  	{
		  		for (int p1 = 0; p1 < this->prefetch_others_[1]->height(); p1 ++)
		  	  		for (int p2 = 0; p2 < this->prefetch_others_[1]->width(); p2 ++)
		  	  		{
	  	  				im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[1]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][0];
	  	  				im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[1]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][1];
	  	  				im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[1]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][2];
	  	  			}
			  	sprintf(temp_path, "temp/%d/mask_%02d.png", tot, i);
			  	imwrite(temp_path, im_data);
			}
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

INSTANTIATE_CLASS(SegRefineLayer);
REGISTER_LAYER_CLASS(SegRefine);
}
