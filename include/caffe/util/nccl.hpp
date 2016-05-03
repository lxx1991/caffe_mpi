//
// Created by yjxiong on 5/3/16.
//

#ifndef CAFFE_NCCL_H
#define CAFFE_NCCL_H

#ifdef USE_NCCL

#include "caffe/util/channel.hpp"
#include <caffe/util/device_alternate.hpp>
#include <nccl.h>
#include <caffe/blob.hpp>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    LOG(FATAL)<<"NCCL failure: " \
              <<ncclGetErrorString(r);   \
  }                                                 \
} while(0)

namespace caffe {
/**
* @brief a light-weight wrapper for NVIDIA NCCL package
*/
    class NCCLCommunicator {
    public:
        NCCLCommunicator(int gpu_id);

        void AllReduceSum(const void *src_ptr, void *dst_ptr,
                       int count, int dtype_size);

    private:
        ncclUniqueId comm_id_;
        ncclComm_t nccl_comm_;
        cudaStream_t stream_;

        shared_ptr<Blob<float> > buffer_;
    };
}
#endif

#endif //CAFFE_NCCL_H
