#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/seg_refine/fcn_res_50/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/seg_refine/fcn_res_50/fcn_res_50_voc_solver.prototxt \
    --weights=models/seg_refine/fcn_res_50/snapshot/fcn_res_50_iter_60000.caffemodel