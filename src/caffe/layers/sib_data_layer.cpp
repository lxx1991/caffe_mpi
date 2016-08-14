#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
SibDataLayer<Dtype>::~SibDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void SibDataLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sib_num_ = sib_data_param_.sib_num();
  CHECK_GE(sib_num_, 1);
  has_label_ = sib_data_param_.has_label();
  CHECK(sib_data_param_.short_edge_size() == 0 || sib_data_param_.short_edge_size() == 2) <<
      "If short_edge_size specified, min and max are needed.";
  short_edge_range_ = 0;
  use_short_edge_ = sib_data_param_.short_edge_size() == 2;
  if (use_short_edge_) {
    short_edge_range_ = sib_data_param_.short_edge(1) - sib_data_param_.short_edge(0);
    CHECK_GE(short_edge_range_, 0) << "Short edge range must be greater than or equal to 0";
  }
  InitRand();
  data_transformer_.reset(
      new LJDataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();

  // Set initial input mode to sequence
  cur_input_mode_ = SEQUENCE;
  if (sib_data_param_.has_mining_hard_param()) {
    cur_input_mode_ = MINING;
    md_.reset(new MiningHard(sib_data_param_.mining_hard_param()));
  }

  // Initialize DB
  db_.reset(db::GetDB(sib_data_param_.backend()));
  db_->Open(sib_data_param_.source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (sib_data_param_.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % sib_data_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  CHECK_EQ(top.size(), sib_num_ + has_label_)
      << "Number of tops is smaller than needed: " << sib_num_ + has_label_;
  // Read a data point, to initialize the prefetch and top blobs.
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = data_transformer_->InferBlobShape(datum, true);
  transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = sib_data_param_.batch_size();
  prefetch_data_.resize(sib_num_);
  for (int ix = 0; ix < sib_num_; ++ix) {
    prefetch_data_[ix].reset(new Blob<Dtype>());
    prefetch_data_[ix]->Reshape(top_shape);
    top[ix]->ReshapeLike(*prefetch_data_[ix]);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (has_label_) {
    vector<int> label_shape(1, sib_data_param_.batch_size());
    top[sib_num_]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  for (int ix = 0; ix < sib_num_; ++ix) {
      prefetch_data_[ix]->mutable_cpu_data();
  }
  if (has_label_) {
    this->prefetch_label_.mutable_cpu_data();
  }
#ifdef USE_MPI
  //advance (my_rank) mini-batches to be ready for first run
  OffsetCursor(top[0]->num() * Caffe::MPI_my_rank());
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void SibDataLayer<Dtype>::CreatePrefetchThread() {
  InitRand();
  this->data_transformer_->InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void SibDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void SibDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  for (int ix = 0; ix < sib_num_; ++ix) {
    CHECK(this->prefetch_data_[ix]->count());
  }
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = sib_data_param_.batch_size();
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum, true);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  for (int ix = 0; ix < sib_num_; ++ix) {
      prefetch_data_[ix]->Reshape(top_shape);
  }

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (has_label_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a datum
    Datum datum;
    if (cur_input_mode_ == SEQUENCE) {
      datum.ParseFromString(cursor_->value());
      // put the key into shuffle pool
      shuffle_key_pool_.push_back(cursor_->key());
    }else if (cur_input_mode_ == SHUFFLE){
      datum.ParseFromString(cursor_->Lookup(*shuffle_cursor_));
    }else if(cur_input_mode_ == MINING) {
      datum.ParseFromString(cursor_->Lookup(md_->peek()));
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int short_edge = 0;
    const bool do_mirror = sib_data_param_.mirror() && Rand(2);
    if (use_short_edge_ > 0) {
      short_edge = short_edge_range_ == 0 ? sib_data_param_.short_edge(0) : sib_data_param_.short_edge(0) + Rand(short_edge_range_ + 1);
    }
    for (int ix = 0; ix < sib_num_; ++ix) {
      Dtype* top_data = prefetch_data_[ix]->mutable_cpu_data();
      int offset = prefetch_data_[ix]->offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_), true, short_edge, do_mirror, sib_num_, ix);
    }
    // Copy label.
    if (has_label_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    timer.Start();
    // go to the next item.
    if (cur_input_mode_ == SEQUENCE) {
      cursor_->Next();
      if (!cursor_->valid()) {
        DLOG(INFO) << "Restarting data prefetching from start.";
        cursor_->SeekToFirst();

        if (sib_data_param_.shuffle() == true){
          LOG(INFO)<<"Entering shuffling mode after first epoch";
          cur_input_mode_ = SHUFFLE;
          shuffle(shuffle_key_pool_.begin(), shuffle_key_pool_.end());
          shuffle_cursor_ = shuffle_key_pool_.begin();
        }
      }
    } else if (cur_input_mode_ == SHUFFLE){
      shuffle_cursor_++;
      if (shuffle_cursor_ == shuffle_key_pool_.end()){
        LOG(INFO)<<"Restarting stream and shuffle again";
        shuffle(shuffle_key_pool_.begin(), shuffle_key_pool_.end());
        shuffle_cursor_ = shuffle_key_pool_.begin();
      }
    } else if (cur_input_mode_ == MINING) {
      md_->next();
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void SibDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  for (int ix = 0; ix < sib_num_; ++ix) {
    // Reshape to loaded data.
    top[ix]->ReshapeLike(*prefetch_data_[ix]);
    // Copy the data
    caffe_copy(prefetch_data_[ix]->count(), prefetch_data_[ix]->cpu_data(),
               top[ix]->mutable_cpu_data());
  }
    DLOG(INFO) << "Prefetch copied";
  if (has_label_) {
    // Reshape to loaded labels.
    top[sib_num_]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
                       top[sib_num_]->mutable_cpu_data());
  }

#ifdef USE_MPI
  //advance (all_rank - (my_rank+1)) mini-batches to be ready for next run
  OffsetCursor(top[0]->num() * (Caffe::MPI_all_rank() - 1));
#endif
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

template <typename Dtype>
void SibDataLayer<Dtype>::InitRand() {
  const bool needs_rand = sib_data_param_.mirror() ||
        use_short_edge_;
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int SibDataLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(SibDataLayer, Forward);
#endif

INSTANTIATE_CLASS(SibDataLayer);
REGISTER_LAYER_CLASS(SibData);

}  // namespace caffe
