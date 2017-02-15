#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/semantic_segmentation/log \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=models/semantic_segmentation/fcn_ires_v2/fcn_ires_v2_voc_solver.prototxt \
    --weights=models/pretrained/ires_v2.caffemodel
	#--weights=models/semantic_segmentation/fcn_ires_v2/snapshot/fcn_ires_v2_coco_iter_80000.caffemodel