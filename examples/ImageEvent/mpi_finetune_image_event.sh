#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/ImageEvent/log \
mpirun -np 3 \
./build/tools/caffe train \
    --solver=models/ImageEvent/solver.prototxt \
    --weights=models/ImageEvent/googlenet_train_iter_270000.caffemodel
    #--snapshot=models/ImageEvent/snapshots/ImageEvent_finetune_Overfeat_iter_3000.solverstate
#    --gpu=3
