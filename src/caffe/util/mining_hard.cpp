#include <cmath>

#include "caffe/util/mining_hard.hpp"

namespace caffe {
  MiningHard::MiningHard(const MiningHardParameter& param)
      : param_(param) {
    // Load [key label notation] database
    cls_num_ = param_.cls_num();
    db_.reserve(cls_num_);
    db_.resize(cls_num_, std::vector<std::pair<std::string, double> >());
    load_db();
    // Preprocess output keys
    batch_size_ = param_.batch_size();
    prefetch_ = param_.prefetch();
    cache_size_ = batch_size_ * prefetch_;
    output_keys_.reserve(cache_size_);
    output_keys_.resize(cache_size_, "");
   // Shuffle entries of every class
    classes_.clear();
    for (size_t ix = 0; ix < cls_num_; ++ix) {
      if (db_[ix].size() > 0) {
        classes_.push_back(ix);
        shuffle(db_[ix].begin(), db_[ix].end());
      }
    }
    CHECK(classes_.size()) << "No valid class";
    // Preprocess class array
    int valid_class_num = classes_.size();
    int copy_num = static_cast<int>(ceil(1.0 * batch_size_ / valid_class_num));
    CHECK_GE(copy_num, 1);
    classes_.reserve(copy_num * valid_class_num);
    for (int ix = 0; ix < copy_num; ++ix) {
      for (int jx = 0; jx < valid_class_num; ++jx) {
        classes_.push_back(classes_[jx]);
      }
    }
    // Preprocess iters for every class
    iters_.reserve(db_.size());
    iters_.resize(db_.size(), 0);
   // Load batch
    load_batch();
    // Done initialization
    output_iter_ = 0;
    valid_ = true;
  }

  void MiningHard::load_db() {
    const string& source = param_.source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string line;
    std::string key;
    size_t label;
    double notation;
    int entry_num = 0;
    while (infile >> key >> label >> notation) {
      db_[label].push_back(std::make_pair(key, notation));
      ++entry_num;
    }
    CHECK(entry_num) << "File is empty";
    LOG(INFO) << "Load " << entry_num << " entries.";
  }

  std::string MiningHard::peek() {
    return output_keys_[output_iter_];
  }

  void MiningHard::next() {
    ++output_iter_;
    if (output_iter_ >= cache_size_) {
      valid_ = false;
      load_batch();
      output_iter_ = 0;
      valid_ = true;
    }
  }

  void MiningHard::load_batch() {
    LOG(INFO) << "MiningHard: Load another batch.";
    switch (param_.strategy()) {
    case MiningHardParameter_Strategy_BALANCE:
      strategy_balance();
      break;
    default:
      LOG(FATAL) << "Unknown mining strategy.";
    }
  }

  void MiningHard::strategy_balance() {
    for (size_t ix = 0; ix < prefetch_; ++ix) {
      shuffle(classes_.begin(), classes_.end());
      for (size_t jx = 0; jx < batch_size_; ++jx) {
        const size_t index = ix * batch_size_ + jx;
        const size_t class_id = classes_[jx];
        if (iters_[class_id] >= db_[class_id].size()) {
          shuffle(db_[class_id].begin(), db_[class_id].end());
          iters_[class_id] = 0;
        }
        output_keys_[index] = db_[class_id][iters_[class_id]].first;
        ++iters_[class_id];
      }
    }
  }
} // namespace caffe
