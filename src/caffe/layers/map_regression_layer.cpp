//
// Created by alex on 4/21/15.
//

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

  template <typename Dtype>
  inline Dtype sigmoid(Dtype x) {
      return 1. / (1. + exp(-x));
  }

  template<typename Dtype>
  inline int8_t caffe_partial_sign(Dtype val, Dtype pos_val, Dtype neg_val) {
    int flag = (Dtype(0) < val) - (val < Dtype(0));
    Dtype rst;
    rst = (flag == 1)?pos_val:flag;
    rst = (rst==-1)?neg_val:rst;
    return rst;
  }

  template <typename Dtype>
  void caffe_cpu_partial_sign(const int count, Dtype* x, Dtype* y, Dtype pos_val, Dtype neg_val){
    for (int i = 0; i < count; ++i){
        y[i] = caffe_partial_sign(x[i], pos_val, neg_val);
    }
  }
  template void caffe_cpu_partial_sign<float>(const int, float*, float*, float, float);
  template void caffe_cpu_partial_sign<double>(const int, double*, double*, double, double);

  template<typename Dtype>
  void MapRegressionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    beta_ = this->layer_param_.map_regression_param().beta();
    alpha_ = this->layer_param_.map_regression_param().alpha();

    tau_plus_ = this->layer_param_.map_regression_param().tau_plus();
    tau_minus_ = this->layer_param_.map_regression_param().tau_minus();

    epsilon_ = this->layer_param_.map_regression_param().epsilon();

    switch (this->layer_param_.map_regression_param().loss_mode()){
        case MapRegressionParameter_LossMode_SOFTMAX:
            loss_mode_ = SOFTMAX;
            break;
        case MapRegressionParameter_LossMode_EUCLIDEAN:
            loss_mode_ = EUCLIDEAN;
            break;
        case MapRegressionParameter_LossMode_HINGE:
            loss_mode_ = HINGE;
            break;
        case MapRegressionParameter_LossMode_INFOGAIN:
            loss_mode_ = INFOGAIN;
            break;
    }
  }

  template<typename Dtype>
  void MapRegressionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
    LossLayer<Dtype>::Reshape(bottom, top);

    //bottom blobs shape check
    CHECK_EQ(bottom.size(), 2)
        <<"bottom must contain two maps";
    CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
        <<"two map must have same number of elements";

    switch (loss_mode_){
        case EUCLIDEAN:
            buffer_.ReshapeLike(*bottom[0]);
            break;
        case SOFTMAX:
            buffer_.ReshapeLike(*bottom[0]);
            break;
        case HINGE:
            buffer_.ReshapeLike(*bottom[0]);
            break;
        case INFOGAIN:
            buffer_.ReshapeLike(*bottom[0]);
            break;

    }
  }

  /**
   * Forward computations
   */
  template <typename Dtype>
  Dtype computeEuclideanLoss(const Dtype* pred_data, Dtype* buffer_data, const Dtype* obj_data, int num, int dim){
    int count = num * dim;
    caffe_sub(
            count,
            pred_data,
            obj_data,
            buffer_data);
    Dtype dot = caffe_cpu_dot(count, buffer_data, buffer_data);
    Dtype loss = dot / num / Dtype(2);
    return loss;
  }

  template <typename Dtype>
  Dtype computeSoftmaxLoss(const Dtype* pred_data, Dtype* buffer_data, const Dtype* obj_data, int num, int dim){
    const int count = num * dim;

    //Save sigmoid output for gradient computation
    for (int i = 0; i < count; ++i) {
        buffer_data[i] = sigmoid(pred_data[i]);
    }

    //Compute sigmoid cross entropy loss in one run
    Dtype loss = 0;
    for (int i = 0; i < count; ++i) {
        loss -= (obj_data[i] - (pred_data[i] >= 0))*pred_data[i] -
                log(1 + exp(pred_data[i] - 2 * pred_data[i] * (pred_data[i] >= 0) ));
    }
    return loss / count;
  }

  template <typename Dtype>
  Dtype computeHingeLoss(const Dtype* pred_data, Dtype* buffer_data, Dtype* diff_data, const Dtype* obj_data,
                         int num, int dim,
                         Dtype alpha, Dtype beta, Dtype tau_plus, Dtype tau_minus, Dtype epsilon){
    const int count = num * dim;

    for (int i = 0; i < count; ++i){
      buffer_data[i] = obj_data[i] * alpha * (obj_data[i]>tau_plus) - (obj_data[i]<tau_minus);
      Dtype ls = (buffer_data[i] > 0) * alpha * obj_data[i] * (Dtype(1) - pred_data[i]) + (buffer_data[i] < 0) * pred_data[i];
      diff_data[i] = std::max(Dtype(0), ls);
    }

    Dtype loss = caffe_cpu_asum(count, diff_data) / count;
    return loss;
  }

  template <typename Dtype>
  Dtype computeInfoGainLoss(const Dtype* pred_data, Dtype* mutable_buffer_data, Dtype* mutable_diff_data, const Dtype* obj_data,
                            int num, int dim, int sp_dim=1){
    const int count = num * dim;

    Dtype loss = 0;
    for (int i = 0; i <count; ++i){
      Dtype prob = std::max(pred_data[i], Dtype(kLOG_THRESHOLD));
      Dtype gt = obj_data[i];
      loss -= gt * log(prob);
      mutable_buffer_data[i] = 1 / prob;
      mutable_diff_data[i] = gt;
    }

    return loss / (num * sp_dim);
  }

  /**
   * Backward computations
   */
  template <typename Dtype>
  void computeEuclideanDiff(const Dtype* buffer_data, Dtype* diff, int num, int dim, Dtype loss_weight, Dtype sign){
    const Dtype alpha = sign * loss_weight / num;
    const int count = num * dim;
    caffe_cpu_axpby(
            count, // count
            alpha, // alpha
            buffer_data, // a
            Dtype(0), // beta
            diff);  // b
  }

  template <typename Dtype>
  void computeSoftmaxDiff(const Dtype* buffer_data, const Dtype* obj_data, Dtype* diff, int num, int dim, Dtype loss_weight){
    const int count = num * dim;
    caffe_sub(count, buffer_data, obj_data, diff);
    caffe_scal(count, loss_weight / count, diff);
  }

  template <typename Dtype>
  void computeHingeDiff(const Dtype* buffer_data, Dtype* diff_data,
                        int num, int dim, Dtype loss_weight,
                        Dtype alpha, Dtype beta, Dtype tau_plus, Dtype tau_minus, Dtype epsilon){
    const int count = num * dim;
    caffe_cpu_sign(count, diff_data, diff_data);
    caffe_mul(count, buffer_data, diff_data, diff_data);
    caffe_scal(count, Dtype(-1) * loss_weight / count, diff_data);
  }

  template <typename Dtype>
  void computeInfoGainDiff(const Dtype* buffer_data, Dtype* diff, int num, int dim, int sp_size, Dtype loss_weight){
    const int count = num * dim;
    Dtype scale = -loss_weight / (num * sp_size);
    for (int i = 0; i < count; ++i){
      diff[i] *= (scale * buffer_data[i]);
    }
  }



  template <typename Dtype>
  void MapRegressionLossLayer<Dtype>::Forward_cpu(vector<Blob<Dtype> *> const &bottom,
                                                  vector<Blob<Dtype> *> const &top) {
    const Dtype* pred_data = bottom[0]->cpu_data();
    const Dtype* obj_data = bottom[1]->cpu_data();
    Dtype* mutable_buffer_data = buffer_.mutable_cpu_data();
    Dtype* mutable_diff_data = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->shape(0);
    int dim = bottom[0]->count(1);
    int sp_size = bottom[0]->count(2);

    switch (loss_mode_){
      case EUCLIDEAN:
        top[0]->mutable_cpu_data()[0] = computeEuclideanLoss(pred_data, mutable_buffer_data, obj_data, num, dim);
        break;
      case SOFTMAX:
        top[0]->mutable_cpu_data()[0] = computeSoftmaxLoss(pred_data, mutable_buffer_data, obj_data, num, dim);
        break;
      case HINGE:
        top[0]->mutable_cpu_data()[0] = computeHingeLoss(pred_data, mutable_buffer_data, mutable_diff_data, obj_data,
                                                         num, dim,
                                                         alpha_, beta_, tau_plus_, tau_minus_, epsilon_);
        break;
      case INFOGAIN:
        top[0]->mutable_cpu_data()[0] = computeInfoGainLoss(pred_data, mutable_buffer_data, mutable_diff_data, obj_data,
                                                            num, dim, sp_size);
        break;
    }
  }

  template <typename Dtype>
  void MapRegressionLossLayer<Dtype>::Backward_cpu(vector<Blob<Dtype> *> const &top,
                                                   const vector<bool> &propagate_down,
                                                   vector<Blob<Dtype> *> const &bottom) {

    Dtype loss_weight = top[0]->cpu_diff()[0];
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        const Dtype* buffer_data = buffer_.cpu_data();
        Dtype* diff_data = bottom[i]->mutable_cpu_diff();

        int num = bottom[i]->shape(0);
        int dim = bottom[i]->count(1);
        int sp_size = bottom[i]->count(2);

        switch (loss_mode_){
          case EUCLIDEAN: {
            Dtype sign = (i == 0) ? 1 : -1;
            computeEuclideanDiff(buffer_data, diff_data, num, dim, loss_weight, sign);
            break;
          }
          case SOFTMAX: {
            if (i == 0) {
                const Dtype* obj_data = bottom[1]->cpu_data();
                computeSoftmaxDiff(buffer_data, obj_data, diff_data, num, dim, loss_weight);
            }else{
                LOG(INFO) << this->type()
                << " Softmax loss cannot backpropagate to label inputs.";
            }
            break;
          }
          case HINGE: {
            computeHingeDiff(buffer_data, diff_data,
                             num, dim, loss_weight,
                             alpha_, beta_, tau_plus_, tau_minus_, epsilon_);
            break;
          }
          case INFOGAIN:{
            computeInfoGainDiff(buffer_data, diff_data, num, dim, sp_size, loss_weight);
            std::cout<<"Diff"<<bottom[i]->asum_diff()/bottom[i]->count();
            break;
          };
        }
      }
    }
  }

INSTANTIATE_CLASS(MapRegressionLossLayer);
REGISTER_LAYER_CLASS(MapRegressionLoss);
} // namespace caffe
