#ifdef USE_NCCL

#include "caffe/util/nccl.hpp"
#include "mpi.h"
namespace caffe {
    NCCLCommunicator::NCCLCommunicator(int gpu_id) {

        Caffe::MPI_build_rank();
        int rank = Caffe::MPI_my_rank();
        int world_size = Caffe::MPI_all_rank();

        CUDA_CHECK(cudaSetDevice(gpu_id));
        NCCLCHECK(ncclGetUniqueId(&comm_id_));
        MPI_Barrier(MPI_COMM_WORLD);

        // init NCCL comm
        MPI_Bcast(&comm_id_, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
        NCCLCHECK(ncclCommInitRank(&nccl_comm_, world_size, comm_id_, rank));

        //create stream
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

        buffer_.reset(new Blob<float>(1,1,1,1));

    };

    void NCCLCommunicator::AllReduceSum(const void *src_ptr, void *dst_ptr,
                                     int count, int dtype_size) {

        if (src_ptr != dst_ptr) {
            NCCLCHECK(ncclAllReduce(src_ptr, dst_ptr, count, ncclFloat, ncclSum,
                          nccl_comm_, stream_));
            cudaStreamSynchronize(stream_);
        }
        else {
            //NCCL does not support inplace all reduce
            if (count > buffer_->count()){
                buffer_.reset(new Blob<float>(count, 1,1,1));
            }
            void* buffer_data = buffer_->mutable_gpu_data();
            NCCLCHECK(ncclAllReduce(src_ptr, buffer_data, count, ncclFloat, ncclSum,
                          nccl_comm_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            cudaMemcpyAsync(dst_ptr, buffer_data, count*sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }
}

#endif