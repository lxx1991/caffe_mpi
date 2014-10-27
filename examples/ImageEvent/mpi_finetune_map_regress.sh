#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/ImageEvent/map_log \
mpirun -np 1 \
./build/tools/caffe train \
    --solver=models/ImageEvent/map_regress_solver.prototxt \
    --weights=models/ImageEvent/imagenet-overfeat_iter_860000
    #--snapshot=models/ImageEvent/snapshots/ImageEvent_finetune_Overfeat_iter_3000.solverstate
#    --gpu=3
