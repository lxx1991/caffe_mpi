rec_path=$1
iter=$2
device_id=$3

proto_name=`ls $1/*proto*.prototxt`
solver_name=`ls $1/*solver*.prototxt`
solver_state=`ls $1/*_$iter.solverstate`

LOG_DIR=$1/log

#change net to point to the real net proto
sed -i "/net:/c\\net: \"${proto_name}\"" $solver_name

GOOGLE_LOG_DIR=$LOG_DIR ./build/tools/caffe train \
	    --solver=$solver_name \
	    --snapshot=$solver_state --gpu=$3
