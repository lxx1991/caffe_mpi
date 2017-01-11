#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/layer_cascade/fcn_ires_v2/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/layer_cascade/fcn_ires_v2/fcn_ires_v2_voc_solver.prototxt \
    --weights=models/layer_cascade/fcn_ires_v2/pretrained/ires_v2.caffemodel