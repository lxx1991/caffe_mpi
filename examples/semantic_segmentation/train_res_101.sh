#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/semantic_segmentation/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/semantic_segmentation/fcn_res_101_solver.prototxt \
    --weights=models/pretrained/res_101.caffemodel