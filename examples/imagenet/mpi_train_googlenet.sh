#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/googlenet/log \
mpirun -np 2 ./build/tools/caffe train \
    --solver=models/hierarchical10/solver.prototxt \
    --snapshot=models/hierarchical10/googlenet_train_iter_255000.solverstate
#    --gpu=3
