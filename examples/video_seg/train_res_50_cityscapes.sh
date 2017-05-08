#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/video_seg/fcn_res_50_cityscapes/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/video_seg/fcn_res_50_cityscapes/fcn_res_50_solver.prototxt \
    --weights=models/semantic_segmentation/fcn_res_50_cityscapes/snapshot_orig/fcn_res_50_iter_60000.caffemodel,models/pretrained/flownets_finetune.caffemodel
 #   --weights=models/pretrained/initmodel/resnet50.caffemodel,models/pretrained/flownets_init.caffemodel