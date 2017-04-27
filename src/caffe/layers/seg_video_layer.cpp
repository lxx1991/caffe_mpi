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
SegVideoLayer<Dtype>:: ~SegVideoLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SegVideoLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	const int stride = this->layer_param_.transform_param().stride();
	const string& source = this->layer_param_.seg_video_param().source();
	const string& root_dir = this->layer_param_.seg_video_param().root_dir();
	image_pattern_ = this->layer_param_.seg_video_param().image_pattern();
	label_pattern_ = this->layer_param_.seg_video_param().label_pattern();
	has_video_label_ = this->layer_param_.seg_video_param().has_video_label();
	
	frame_idx.clear();

	if (this->layer_param_.seg_video_param().fr_index_size() != 0)
	{
		fr_num_ = this->layer_param_.seg_video_param().fr_index_size();
		for (int i=0; i<fr_num_; i++)
			frame_idx.push_back(this->layer_param_.seg_video_param().fr_index(i));
	}
	else
	{
		CHECK(this->layer_param_.seg_video_param().has_fr_st());
		CHECK(this->layer_param_.seg_video_param().has_fr_en());
		CHECK(this->layer_param_.seg_video_param().has_fr_num());

		fr_num_ = this->layer_param_.seg_video_param().fr_num();
		frame_idx.push_back(0);
		for (int i=this->layer_param_.seg_video_param().fr_st(); i<=this->layer_param_.seg_video_param().fr_en(); i++)
		{
			if (i!=0)
				frame_idx.push_back(i);
		}
		CHECK(frame_idx.size() >= fr_num_);
	}

	LOG(INFO) << "Opening file: " << source;
	std::ifstream infile(source.c_str());
	string img_filename;
	string label_filename;
	int label_framenum;
	while (infile >> img_filename >> label_filename >> label_framenum){
		lines_.push_back(root_dir + img_filename);
		labels_.push_back(std::make_pair(root_dir + label_filename, label_framenum));
	}

	if (this->layer_param_.seg_video_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;


	Datum datum_data, datum_label;

	sprintf(image_buf, image_pattern_.c_str(), labels_[lines_id_].second);
	sprintf(label_buf, label_pattern_.c_str(), labels_[lines_id_].second);
	
	CHECK(ReadSegDataToDatum(lines_[lines_id_] + string(image_buf) , labels_[lines_id_].first + string(label_buf), &datum_data, &datum_label, true));


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
  	else if (this->layer_param_.transform_param().has_upper_size())
	{
		crop_height = std::min(crop_height, this->layer_param_.transform_param().upper_size());
		crop_width = std::min(crop_width, this->layer_param_.transform_param().upper_size());
	}
	else if (this->layer_param_.transform_param().has_upper_height() && this->layer_param_.transform_param().has_upper_width())
	{
		crop_height = std::min(crop_height, this->layer_param_.transform_param().upper_height());
		crop_width = std::min(crop_width, this->layer_param_.transform_param().upper_width());
	}

	batch_size_ = this->layer_param_.seg_video_param().batch_size();
	if (batch_size_ != 1)
		CHECK(this->layer_param_.transform_param().has_crop_size() || (this->layer_param_.transform_param().has_crop_height() && this->layer_param_.transform_param().has_crop_width()));


	top[0]->Reshape(batch_size_, datum_data.channels() * fr_num_, crop_height, crop_width);
	this->prefetch_data_.Reshape(batch_size_, datum_data.channels() * fr_num_, crop_height, crop_width);

	top[1]->Reshape(batch_size_, datum_label.channels() * (has_video_label_?fr_num_:1), crop_height, crop_width);
	this->prefetch_label_.Reshape(batch_size_, datum_label.channels() * (has_video_label_?fr_num_:1), crop_height, crop_width);

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
}

template <typename Dtype>
void SegVideoLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), labels_.begin(), labels_.end(), prefetch_rng);
}

template <typename Dtype>
void SegVideoLayer<Dtype>::InternalThreadEntry(){

	Datum datum_data, datum_label;
	CHECK(this->prefetch_data_.count());
	
	const int lines_size = lines_.size();

	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{
		CHECK_GT(lines_size, lines_id_);

		//sample frame index
		vector <std::string> frame_paths, label_paths;
		frame_paths.clear(); label_paths.clear();

		for (int i=0; i<fr_num_; i++)
		{
			if (i!=0 && this->layer_param_.seg_video_param().fr_index_size() == 0)
				std::swap(frame_idx[i], frame_idx[i + (caffe_rng_rand() % (fr_num_ - i))]);

			sprintf(image_buf, image_pattern_.c_str(), labels_[lines_id_].second + frame_idx[i]);
			frame_paths.push_back(lines_[lines_id_] + string(image_buf));

			if (frame_idx[i]==0 || has_video_label_)
			{
				sprintf(label_buf, label_pattern_.c_str(), labels_[lines_id_].second + frame_idx[i]);
				label_paths.push_back(labels_[lines_id_].first + string(label_buf));
			}
		}
		
		CHECK(ReadSegVideoToDatum(frame_paths, label_paths, &datum_data, &datum_label, true));

		this->data_transformer_->Transform(datum_data, datum_label, &this->prefetch_data_, &this->prefetch_label_, batch_iter);

		/*if (this->layer_param_.seg_video_param().balance())
		{
			for (int t = 0; t < 10; t++)
			{
				std::vector<int> cnt(256, 0); int max_label_cnt = 0;
				for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
		  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
		  	  		{
		  	  			int label_value = (int)this->prefetch_label_.data_at(0, 0, p1, p2);
		  	  			cnt[label_value]++;
		  	  		}
		  	  	for (int i = 0; i<cnt.size(); i++)
		  			max_label_cnt = std::max(max_label_cnt, cnt[i]);

		  		if (max_label_cnt > 0.8 * this->prefetch_label_.count())
		  			this->data_transformer_->Transform(datum_data, datum_label, &this->prefetch_data_, &this->prefetch_label_);
	  			else
	  				break;
			}
		}*/

		
		if (true)
		{
		  	cv::Mat im_data(this->prefetch_data_.height(), this->prefetch_data_.width(), CV_8UC3);
		  	cv::Mat im_label(this->prefetch_label_.height(), this->prefetch_label_.width(), CV_8UC1);

		  	int tot = rand() * 10000 + rand() + lines_id_;
		  	char temp_path[200];

		  	mkdir("temp", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		  	sprintf(temp_path, "temp/%d",tot);
		  	mkdir(temp_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

		  	for (int i=0; i < fr_num_; i++)
		  	{
			  	for (int p1 = 0; p1 < this->prefetch_data_.height(); p1 ++)
			  		for (int p2 = 0; p2 < this->prefetch_data_.width(); p2 ++)
			  		{
			  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_data_.data_at(batch_iter, 0 + 3*i, p1, p2)+104);
			  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_data_.data_at(batch_iter, 1 + 3*i, p1, p2)+117);
			  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_data_.data_at(batch_iter, 2 + 3*i, p1, p2)+123);	
			  		}
			  	sprintf(temp_path, "temp/%d/image_%02d.png", tot, i);
	  			imwrite(temp_path, im_data);
	  		}

		  	for (int i=0; i < this->prefetch_label_.channels(); i++)
		  	{
		  		for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
		  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
		  	  			im_label.at<uchar>(p1, p2) = this->prefetch_label_.data_at(batch_iter, i, p1, p2);
			  	sprintf(temp_path, "temp/%d/label_%02d.png", tot, i);
			  	imwrite(temp_path, im_label);
			 }
		}

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.seg_video_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
}

INSTANTIATE_CLASS(SegVideoLayer);
REGISTER_LAYER_CLASS(SegVideo);
}
