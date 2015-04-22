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
    void MapRegressionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        beta_ = this->layer_param_.map_regression_param().beta();

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
            loss -= (obj_data[i] - 1)*pred_data[i] -
                    log(1 + exp(-pred_data[i]));
        }
        return loss;
    }

    template <typename Dtype>
    Dtype computeHingeLoss(const Dtype* pred_data, Dtype* buffer_data, const Dtype* obj_data, int num, int dim){
        const int count = num * dim;

//        caffe_copy(count, pred_data, pred_data);
//        for (int i = 0; i < num; ++i) {
//            bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
//        }
//        for (int i = 0; i < num; ++i) {
//            for (int j = 0; j < dim; ++j) {
//                bottom_diff[i * dim + j] = std::max(
//                        Dtype(0), 1 + bottom_diff[i * dim + j]);
//            }
//        }
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
        caffe_scal(count, loss_weight / num, diff);
    }

    template <typename Dtype>
    void computeHingeDiff(const Dtype* buffer_data, Dtype* diff, int num, int dim, Dtype loss_weight, Dtype sign){

    }

    template <typename Dtype>
    void MapRegressionLossLayer<Dtype>::Forward_cpu(vector<Blob<Dtype> *> const &bottom,
                                                    vector<Blob<Dtype> *> const &top) {
        const Dtype* pred_data = bottom[0]->cpu_data();
        const Dtype* obj_data = bottom[1]->cpu_data();
        Dtype* mutable_buffer_data = buffer_.mutable_cpu_data();
        int num = bottom[0]->shape(0);
        int dim = bottom[0]->count(1);

        switch (loss_mode_){
            case EUCLIDEAN:
                top[0]->mutable_cpu_data()[0] = computeEuclideanLoss(pred_data, mutable_buffer_data, obj_data, num, dim);
                break;

            //TODO: add loss forward implementations.
            case SOFTMAX:
                top[0]->mutable_cpu_data()[0] = computeSoftmaxLoss(pred_data, mutable_buffer_data, obj_data, num, dim);
                break;
            case HINGE:
                NOT_IMPLEMENTED;
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
                            LOG(FATAL) << this->type()
                            << " Softmax loss cannot backpropagate to label inputs.";
                        }
                        break;
                    }
                    case HINGE: {
                        NOT_IMPLEMENTED;
                        break;
                    }
                }
            }
        }
    }

    INSTANTIATE_CLASS(MapRegressionLossLayer);
    REGISTER_LAYER_CLASS(MapRegressionLoss);
} // namespace caffe
