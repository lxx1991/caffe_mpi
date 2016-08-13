#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void SibDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  for (int ix = 0; ix < sib_num_; ++ix) {
    // Reshape to loaded data.
    top[ix]->ReshapeLike(*prefetch_data_[ix]);
    // Copy the data
    caffe_copy(prefetch_data_[ix]->count(), prefetch_data_[ix]->cpu_data(),
               top[ix]->mutable_gpu_data());
  }
  if (has_label_) {
    // Reshape to loaded labels.
    top[sib_num_]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[sib_num_]->mutable_gpu_data());
  }
#ifdef USE_MPI
  //advance (all_rank - (my_rank+1)) mini-batches to be ready for next run
  OffsetCursor(top[0]->num() * (Caffe::MPI_all_rank() - 1));
#endif
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(SibDataLayer);

}  // namespace caffe
