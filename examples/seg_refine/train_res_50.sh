#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/seg_refine/fcn_res_50/log \
    /usr/local/openmpi/bin/mpirun -np 2 \
    build/install/bin/caffe train \
    --solver=models/seg_refine/fcn_res_50/fcn_res_50_solver.prototxt \
    --weights=models/pretrained/initmodel/resnet50.caffemodel