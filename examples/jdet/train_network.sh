#!/usr/bin/env sh
NET=clarify
DEVICE_ID=${1:-0}  
WEIGHTS=${2:-"models/jdet/pretrained/clarify/origin_clarify.caffemodel"}
SNAPSHOT_PATH='/media/datadisk_c/snapshots/jdet/'

LOG_DIR=`~/anaconda/bin/python examples/jdet/build_solver.py $NET $DEVICE_ID $SNAPSHOT_PATH`

echo Log path $LOG_DIR
GOOGLE_LOG_DIR=$LOG_DIR \
    mpirun -np 3 ./build/install/bin/caffe train \
    --solver=models/jdet/solver_$NET.prototxt  \
    --weights=$WEIGHTS
