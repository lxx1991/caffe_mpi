#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/layer_cascade/fcn_res_101/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/layer_cascade/fcn_res_101/fcn_res_101_bndata_solver.prototxt \
    --weights=models/pretrained/initmodel/resnet101.caffemodel
