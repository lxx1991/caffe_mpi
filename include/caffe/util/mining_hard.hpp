#ifndef MINING_HARD_HPP_
#define MINING_HARD_HPP_

#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  class MiningHard {
   public:
      explicit MiningHard(const MiningHardParameter& param_);
      virtual ~MiningHard() {}
      std::string peek();
      void next();
      bool valid() {return valid_;}
  protected:
    void load_db();
    void load_batch();
    void strategy_balance();

    std::vector<std::vector<std::pair<std::string, double> > > db_;
    std::vector<size_t> iters_;
    std::vector<std::string> output_keys_;
    std::vector<size_t> classes_;
    size_t output_iter_;
    size_t batch_size_;
    size_t prefetch_;
    size_t cache_size_;
    size_t cls_num_;
    bool valid_;
    const MiningHardParameter param_;

  DISABLE_COPY_AND_ASSIGN(MiningHard);
  };
} // namespace caffe

#endif // MINING_HARD_HPP_
