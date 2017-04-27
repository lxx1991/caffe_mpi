#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/video_seg/fcn_res_50_cityscapes/log \
    /usr/local/openmpi/bin/mpirun -np 2 \
    build/install/bin/caffe train \
    --solver=models/video_seg/fcn_res_50_cityscapes/fcn_res_50_solver.prototxt \
    --weights=models/pretrained/initmodel/resnet50.caffemodel,models/pretrained/flownets_init.caffemodel
