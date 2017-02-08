#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/mult_scale/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/mult_scale/fcn_ires_v2/fcn_ires_v2_voc_solver.prototxt \
    --weights=models/semantic_segmentation/fcn_ires_v2/snapshot/fcn_ires_v2_iter_30000.caffemodel