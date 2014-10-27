#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MapRegressionLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), 1)<<"Regressioned map must be a one-channel intensity map";
  CHECK_EQ(bottom[1]->channels(), 6)<<"Groundtruth map must be represented by a 6-digit tuple";

  CHECK_GT(bottom[0]->height(), 1)<<"Regressed map must be larger than 1-pixel.";
  CHECK_GT(bottom[0]->width(), 1);

  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
//  CHECK_EQ(bottom[2]->channels(), 1);
//  CHECK_EQ(bottom[2]->height(), 1);
//  CHECK_EQ(bottom[2]->width(), 1);
  diff_map_.Reshape(bottom[0]->num(), 1, bottom[0]->width(), bottom[0]->height());
//  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
//  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
//  // vector of ones used to sum along channels
//  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
//  for (int i = 0; i < bottom[0]->channels(); ++i)
//    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void MapRegressionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

	Dtype loss(0.);
	const Dtype* bbox_data = bottom[1]->cpu_data();
	const Dtype* map_data = bottom[0]->cpu_data();
	Dtype* diff_map_data = diff_map_.mutable_cpu_data();
	for (int i = 0; i < bottom[0]->num(); ++i){
		int x_min = (int) bbox_data[i * 6];
		int y_min = (int) bbox_data[i * 6 + 1];
		int x_max = (int) bbox_data[i * 6 + 2];
		int y_max = (int) bbox_data[i * 6 + 3];
		const Dtype* map_data_row = map_data + bottom[0]->offset(i);
		Dtype* diff_map_data_row = diff_map_data + diff_map_.offset(i);
		for (int h = 0; h < bottom[0]->height(); ++h){
			for (int w = 0; w < bottom[0]->width(); ++w){
				// Just use 0/1 loss, maybe change it to better
				// Loss metric is Euclidean distance
				Dtype ref = (
							(w >= x_min)
							&& (w <= x_max)
							&& (h >= y_min)
							&& (h <= y_max)
							)?Dtype(1.):Dtype(0.);

				//Cache difference map for gradients
				diff_map_data_row[w] = ref - map_data_row[w];

				// Loss relates to a square sum over the diff map
				loss += (diff_map_data_row[w]) * (diff_map_data_row[w]);
			}
			diff_map_data_row += diff_map_.width();
			map_data_row += bottom[0]->width();
		}
	}
	// Loss formulation \frac{1}{m}\sum_i\frac{1}{2}(ref-data)^2
	(*top)[0]->mutable_cpu_data()[0] = loss / Dtype(bottom[0]->num()) / Dtype(2.);
}

template <typename Dtype>
void MapRegressionLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {

	const Dtype alpha = Dtype(-1.) * top[0]->cpu_diff()[0] / (*bottom)[0]->num();
	if (propagate_down[0]) {
		caffe_cpu_axpby(
				(*bottom)[0]->count(),
				alpha,
				this->diff_map_.cpu_data(),
				Dtype(0.),
				(*bottom)[0]->mutable_cpu_data());
	}
}

#ifdef CPU_ONLY
STUB_GPU(MapRegressionLossLayer);
#endif

INSTANTIATE_CLASS(MapRegressionLossLayer);

}  // namespace caffe
