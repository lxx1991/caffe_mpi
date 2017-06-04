#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <set>
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
SegRefineVideoLayer<Dtype>:: ~SegRefineVideoLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SegRefineVideoLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	const int stride = this->layer_param_.transform_param().stride();
	const string& source = this->layer_param_.seg_refine_param().source();
	const string& root_dir = this->layer_param_.seg_refine_param().root_dir();
	const int instance_num = this->layer_param_.seg_refine_param().instance_num();

	batch_size_ = this->layer_param_.seg_refine_param().batch_size();
	image_pattern_ = this->layer_param_.seg_refine_param().image_pattern();
	instance_pattern_ = this->layer_param_.seg_refine_param().instance_pattern();
	flow_pattern_ = this->layer_param_.seg_refine_param().flow_pattern();
	use_warp_ = this->layer_param_.seg_refine_param().has_warp_pattern();
	if (use_warp_)
		warp_pattern_ = this->layer_param_.seg_refine_param().warp_pattern();


	LOG(INFO) << "Opening file: " << source;
	std::ifstream infile(source.c_str());
	
	string video_name;
	int frame_num;

	while (infile >> video_name >> frame_num){
		lines_.push_back(std::make_pair(video_name, frame_num));
	}

	if (this->layer_param_.seg_refine_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;

	Datum datum_data;

	sprintf(string_buf, (root_dir + image_pattern_).c_str(), lines_[lines_id_].first.c_str(), 0);
	CHECK(ReadSegDataToDatum(string_buf, &datum_data, true));


	int crop_height = ((datum_data.height() - 1) / stride + 1) * stride + 1;
	int crop_width = ((datum_data.width() - 1) / stride + 1) * stride + 1;

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

	//instance label
	top[1]->Reshape(batch_size_, 1, crop_height, crop_width);
	this->prefetch_label_.Reshape(batch_size_, 1, crop_height, crop_width);

	this->prefetch_others_.clear();
	for (int i=0; i<top.size() - 2; i++)
		this->prefetch_others_.push_back(new Blob<Dtype>());

	//mask
	top[2]->Reshape(batch_size_, instance_num, crop_height, crop_width);
	this->prefetch_others_[0]->Reshape(batch_size_, instance_num, crop_height, crop_width);

	//flow
	top[3]->Reshape(batch_size_, 2, crop_height, crop_width);
	this->prefetch_others_[1]->Reshape(batch_size_, 2, crop_height, crop_width);

	//hint image
	top[4]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);
	this->prefetch_others_[2]->Reshape(batch_size_, datum_data.channels(), crop_height, crop_width);

	//hint label
	top[5]->Reshape(batch_size_, instance_num, crop_height, crop_width);
	this->prefetch_others_[3]->Reshape(batch_size_, instance_num, crop_height, crop_width);

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output instance label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
	LOG(INFO) << "output mask size: " << top[2]->num() << "," << top[2]->channels() << "," << top[2]->height() << "," << top[2]->width();
	LOG(INFO) << "output flow size: " << top[3]->num() << "," << top[3]->channels() << "," << top[3]->height() << "," << top[3]->width();
	LOG(INFO) << "output hint image size: " << top[4]->num() << "," << top[4]->channels() << "," << top[4]->height() << "," << top[4]->width();
	LOG(INFO) << "output hint label size: " << top[5]->num() << "," << top[5]->channels() << "," << top[5]->height() << "," << top[5]->width();
}

template <typename Dtype>
void SegRefineVideoLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void SegRefineVideoLayer<Dtype>::IndexToProb(Datum &datum_data, int instance_num)
{
  	const int datum_height = datum_data.height();
	const int datum_width = datum_data.width();
	CHECK(datum_data.channels() == 1);

	string* datum_data_string = datum_data.mutable_data();
	datum_data.set_channels(instance_num);

	for (int c=1; c<instance_num; c++)
		for (int h=0; h<datum_height; h++)
			for (int w=0; w<datum_width; w++)
				datum_data_string->push_back(0);


	for (int h=0; h<datum_height; h++)
		for (int w=0; w<datum_width; w++)
		{
			int idx = (*datum_data_string)[h * datum_width + w] - 1;
			CHECK_GT(instance_num, idx);
			for (int c=0; c<instance_num; c++)
				(*datum_data_string)[(c * datum_height + h) * datum_width + w] = (c == idx) ? 1 : 0;
		}
}

template <typename Dtype>
void SegRefineVideoLayer<Dtype>::InternalThreadEntry(){

	const int instance_num = this->layer_param_.seg_refine_param().instance_num();
	const int ignore_label = this->layer_param_.transform_param().ignore_label();
	const string& root_dir = this->layer_param_.seg_refine_param().root_dir();



	vector<Datum> datum_label(3), datum_data(2);
	cv::Mat flow_data;

	const int lines_size = lines_.size();

	vector< Blob<Dtype>* > prefetch_data, prefetch_label;

	prefetch_data.push_back(&this->prefetch_data_);
	prefetch_data.push_back(this->prefetch_others_[2]);

	prefetch_label.push_back(&this->prefetch_label_);
	prefetch_label.push_back(this->prefetch_others_[0]);
	prefetch_label.push_back(this->prefetch_others_[3]);


	for (int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{

		CHECK_GT(lines_size, lines_id_);

		int current_frame = this->data_transformer_->Rand(lines_[lines_id_].second), dist = lines_[lines_id_].second  / 3;
		while(current_frame == 0)
			current_frame = this->data_transformer_->Rand(lines_[lines_id_].second);

		//image
		sprintf(string_buf, (root_dir + image_pattern_).c_str(), lines_[lines_id_].first.c_str(), current_frame);
		CHECK(ReadSegDataToDatum(string_buf, &datum_data[0], true));

		//instance label
		sprintf(string_buf, (root_dir + instance_pattern_).c_str(), lines_[lines_id_].first.c_str(), current_frame);
		CHECK(ReadSegDataToDatum(string_buf, &datum_label[0], false));

		//flow data
		sprintf(string_buf, (root_dir + flow_pattern_).c_str(), lines_[lines_id_].first.c_str(), current_frame);
		CHECK(ReadFlowToCVMat(string_buf, flow_data));

		//find max_idx & re-rank
	  	const int datum_height = datum_label[0].height();
  		const int datum_width = datum_label[0].width();
	  	string* ptr = datum_label[0].mutable_data();
		
		int max_idx = instance_num;
		for (int data_index = 0; data_index < datum_height * datum_width; ++data_index)
		{
			int label_value = (int)(*ptr)[data_index];
			if (label_value != ignore_label)
			  max_idx = std::max(max_idx, label_value);
		}

		int vis[256]; 
		memset(vis, 0, sizeof(vis));
		vis[ignore_label] = ignore_label;
		for (int i=1; i < max_idx + 1; i++)
		{
			vis[i] = (i <= instance_num) ? i : 0;
			std::swap(vis[i], vis[this->data_transformer_->Rand(i) + 1]);
		}
		
		std::set<int> hash_table;
		for (int data_index = 0; data_index < datum_height * datum_width; ++data_index)
		{
			int label_value = (int)(*ptr)[data_index];
			if (label_value != ignore_label)
			{
				(*ptr)[data_index] = (uchar)vis[label_value];
				hash_table.insert((*ptr)[data_index]);
			}
			else
				(*ptr)[data_index] = 0;
		}

		//mask label
		if (use_warp_)
		{
			sprintf(string_buf, (root_dir + warp_pattern_).c_str(), lines_[lines_id_].first.c_str(), current_frame);
			CHECK(ReadSegDataToDatum(string_buf, &datum_label[1], false));
			ptr = datum_label[1].mutable_data();
			for (int data_index = 0; data_index < datum_height * datum_width; ++data_index)
			{
				int label_value = (int)(*ptr)[data_index];
				if (label_value != ignore_label)
					(*ptr)[data_index] = (uchar)vis[label_value];
				else
					(*ptr)[data_index] = 0;
			}
			IndexToProb(datum_label[1], instance_num);
		}
		else
			this->data_transformer_->Transform_aug(datum_label[0], datum_label[1], instance_num);

		//hint image and label
		while (true)
		{
			vector<int> count_table(instance_num + 1, 0);

			int hint_frame = this->data_transformer_->Rand(lines_[lines_id_].second);
			while(std::abs(hint_frame - current_frame) < dist)
				hint_frame = this->data_transformer_->Rand(lines_[lines_id_].second);

			sprintf(string_buf, (root_dir + instance_pattern_).c_str(), lines_[lines_id_].first.c_str(), hint_frame);
			CHECK(ReadSegDataToDatum(string_buf, &datum_label[2], false));

			string* ptr = datum_label[2].mutable_data();
			for (int data_index = 0; data_index < datum_height * datum_width; ++data_index)
			{
				int label_value = (int)(*ptr)[data_index];
				if (label_value != ignore_label)
				{
					(*ptr)[data_index] = (uchar)vis[label_value];
					count_table[(*ptr)[data_index]]++;
				}
				else
					(*ptr)[data_index] = 0;
			}

			bool flag = true;
			for (std::set<int>::iterator it = hash_table.begin(); it!=hash_table.end(); it++)
			{
				if (count_table[*it] == 0)
					flag = false;
			}
			if (flag)
			{
				sprintf(string_buf, (root_dir + image_pattern_).c_str(), lines_[lines_id_].first.c_str(), hint_frame);
				CHECK(ReadSegDataToDatum(string_buf, &datum_data[1], true));
				IndexToProb(datum_label[2], instance_num);
				break;
			}
			else
				dist = dist / 2;
		}

		this->data_transformer_->Transform(datum_data, datum_label, flow_data, prefetch_data, prefetch_label, this->prefetch_others_[1], batch_iter);
		
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
			sprintf(temp_path, "temp/%d/instance_label.png", tot);
			imwrite(temp_path, im_data);




		  	for (int i=0; i < this->prefetch_others_[0]->channels(); i++)
		  	{
		  		for (int p1 = 0; p1 < this->prefetch_others_[0]->height(); p1 ++)
		  	  		for (int p2 = 0; p2 < this->prefetch_others_[0]->width(); p2 ++)
		  	  		{
	  	  				im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][0];
	  	  				im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][1];
	  	  				im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[0]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][2];
	  	  			}
			  	sprintf(temp_path, "temp/%d/mask_%02d.png", tot, i);
			  	imwrite(temp_path, im_data);
			}


			double max_flow, min_flow;
			cv::minMaxLoc(flow_data, &min_flow, &max_flow);
	  		for (int p1 = 0; p1 < this->prefetch_others_[1]->height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_others_[1]->width(); p2 ++)
				{
	  	  			im_data.at<uchar>(p1, p2*3+0) = (int)( 255.0 * (this->prefetch_others_[1]->data_at(batch_iter, 0, p1, p2) -  min_flow) / (max_flow - min_flow) );
	  	  			im_data.at<uchar>(p1, p2*3+1) = (int)( 255.0 * (this->prefetch_others_[1]->data_at(batch_iter, 1, p1, p2) -  min_flow) / (max_flow - min_flow) );
	  	  			im_data.at<uchar>(p1, p2*3+2) = 0;
				}
			sprintf(temp_path, "temp/%d/flow.png", tot);
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
			

		  	for (int i=0; i < this->prefetch_others_[3]->channels(); i++)
		  	{
		  		for (int p1 = 0; p1 < this->prefetch_others_[3]->height(); p1 ++)
		  	  		for (int p2 = 0; p2 < this->prefetch_others_[3]->width(); p2 ++)
		  	  		{
	  	  				im_data.at<uchar>(p1, p2*3+0) = color_map[(int)this->prefetch_others_[3]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][0];
	  	  				im_data.at<uchar>(p1, p2*3+1) = color_map[(int)this->prefetch_others_[3]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][1];
	  	  				im_data.at<uchar>(p1, p2*3+2) = color_map[(int)this->prefetch_others_[3]->data_at(batch_iter, i, p1, p2) < 0.5 ? 0 : i+1][2];
	  	  			}
			  	sprintf(temp_path, "temp/%d/hint_%02d.png", tot, i);
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

INSTANTIATE_CLASS(SegRefineVideoLayer);
REGISTER_LAYER_CLASS(SegRefineVideo);
}
