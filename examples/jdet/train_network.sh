#!/usr/bin/env sh
NET=clarify
DEVICE_ID=${1:-0}  
TESTING=$2

LOG_DIR=`~/anaconda/bin/python examples/jdet/build_solver.py $NET $DEVICE_ID $TESTING`

echo Log path $LOG_DIR
GOOGLE_LOG_DIR=$LOG_DIR ./build/tools/caffe train \
    --solver=models/jdet/solver_$NET.prototxt  \
    --weights=models/jdet/pretrained/clarify/origin_clarify.caffemodel
