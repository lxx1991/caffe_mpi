#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/semantic_segmentation/fcn_res_101/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/semantic_segmentation/fcn_res_101/fcn_res_101_solver.prototxt \
    --weights=models/semantic_segmentation/fcn_res_101/snapshot/fcn_res_101_bn_iter_10000.caffemodel
    #--weights=models/pretrained/initmodel/resnet101.caffemodel