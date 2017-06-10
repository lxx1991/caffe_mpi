#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/seg_refine/fcn_res_101_val/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/seg_refine/fcn_res_101_val/fcn_res_101_davis_solver.prototxt \
    --snapshot=models/seg_refine/fcn_res_101_val/snapshot/fcn_res_101_davis_iter_15000.solverstate
    # --weights=models/seg_refine/fcn_res_101_val/snapshot/fcn_res_101_finetune_iter_10000.caffemodel