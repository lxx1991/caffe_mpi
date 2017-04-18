#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/semantic_segmentation/fcn_res_50_cityscapes/log_aug_10 \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/semantic_segmentation/fcn_res_50_cityscapes/fcn_res_50_video_solver.prototxt \
    --weights=models/pretrained/initmodel/resnet50.caffemodel
