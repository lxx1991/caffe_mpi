#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/semantic_segmentation/fcn_res_50_cityscapes/log_aug_10 \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/semantic_segmentation/fcn_res_50_cityscapes/fcn_res_50_video_solver.prototxt \
    --weights=models/pretrained/initmodel/resnet50.caffemodel
    #--snapshot=models/semantic_segmentation/fcn_res_50_cityscapes/snapshot_aug_10/fcn_res_50_iter_100000.solverstate
    #--weights=models/pretrained/initmodel/resnet50.caffemodel