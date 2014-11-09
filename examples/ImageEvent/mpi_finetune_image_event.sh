#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/ImageEvent/log \
mpirun -np 2 \
./build/tools/caffe train \
    --solver=models/ImageEvent/solver_alexnet.prototxt \
    --weights=models/ImageEvent/imagenet-alex_train_iter_410000 \

#    --weights=models/ImageEvent/imagenet-alex_train_iter_410000 \
#    --weights=models/ImageEvent/imagenet-overfeat_iter_860000 \



    #--snapshot=models/ImageEvent/snapshots/ImageEvent_finetune_Overfeat_org_iter_4000.solverstate \
    #--weights=models/ImageEvent/imagenet-overfeat_iter_860000 \
#    --gpu=3
